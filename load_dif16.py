import os
import sys
import json
import argparse
import torch
import types
import re


# Ensure local package imports work whether run from repo root or from this folder
SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import the module that defines the model class used in training
# This is necessary so torch.load can resolve the class during unpickling
# Make optional so we can still read state_dict-only checkpoints without the package
_GYM_MP_AVAILABLE = False
try:
    import gym_mp.models2  # noqa: F401
    _GYM_MP_AVAILABLE = True
except Exception:
    # Fallback: user might have provided gym_mp/model2.py instead of models2.py
    try:
        import gym_mp.model2 as _gm2  # type: ignore
        import types as _types  # local import to avoid polluting top-level
        # Alias it so "gym_mp.models2" resolves
        sys.modules["gym_mp.models2"] = _gm2
        # And attach attribute on parent package for completeness
        if hasattr(sys.modules.get("gym_mp", None), "model2") and not hasattr(sys.modules["gym_mp"], "models2"):
            setattr(sys.modules["gym_mp"], "models2", _gm2)
        _GYM_MP_AVAILABLE = True
    except Exception:
        _GYM_MP_AVAILABLE = False


DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "dif_pose_vec.json")


def resolve_pt_path(config_path: str) -> str:
    with open(config_path, "r") as f:
        cfg = json.load(f)

    pt_dir = cfg["path"]["pt_path"]
    pt_name = cfg["model"]["pt_name"] + ".pt"

    # Prefer path relative to this script directory (e.g., motion_prediction/work)
    candidate = os.path.join(SCRIPT_DIR, pt_dir, pt_name)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Load dif16 model checkpoint")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to dif_pose_vec.json (defaults to work/config/dif_pose_vec.json)",
    )
    parser.add_argument(
        "--pt",
        default=None,
        help="Direct path to .pt file (overrides config)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to map model to (default: cpu)",
    )
    parser.add_argument(
        "--unsafe",
        action="store_true",
        help=(
            "Allow full unpickling (weights_only=False). Use only if checkpoint is trusted."
        ),
    )
    args = parser.parse_args()

    if args.pt is not None:
        pt_path = args.pt
    else:
        pt_path = resolve_pt_path(args.config)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Checkpoint not found: {pt_path}")

    print(f"Loading model from: {pt_path}")

    def ensure_dummy_for(module_path: str, class_name: str) -> None:
        parts = module_path.split(".")
        # Create module chain if missing
        for i in range(len(parts)):
            full_name = ".".join(parts[: i + 1])
            if full_name not in sys.modules:
                new_mod = types.ModuleType(full_name)
                sys.modules[full_name] = new_mod
                if i > 0:
                    parent_name = ".".join(parts[:i])
                    setattr(sys.modules[parent_name], parts[i], new_mod)
        target_mod = sys.modules[".".join(parts)]
        if not hasattr(target_mod, class_name):
            dummy_cls = type(class_name, (), {})
            setattr(target_mod, class_name, dummy_cls)
        # Allowlist the class for safe weights loading
        try:
            from torch.serialization import add_safe_globals

            add_safe_globals([getattr(target_mod, class_name)])
        except Exception:
            pass

    loaded_obj = None
    # Prefer weights_only=True to avoid requiring class definitions when possible
    # If safe loader complains about unsupported globals, dynamically allowlist them with dummies and retry.
    attempts = 0
    max_attempts = 5
    while attempts < max_attempts and loaded_obj is None:
        try:
            loaded_obj = torch.load(pt_path, map_location=device, weights_only=False)
            #print("작동중")
        except TypeError:
            # Older torch versions do not support weights_only
            loaded_obj = torch.load(pt_path, map_location=device)
        except Exception as e:
            msg = str(e)
            m = re.search(r"Unsupported global: GLOBAL ([\w\.]+)\.([\w]+)", msg)
            if m:
                module_path, class_name = m.group(1), m.group(2)
                ensure_dummy_for(module_path, class_name)
                attempts += 1
                continue
            else:
                raise

    # Optional unsafe fallback if safe loader didn't yield a usable object
    if loaded_obj is None and args.unsafe:
        attempts = 0
        while attempts < max_attempts and loaded_obj is None:
            try:
                loaded_obj = torch.load(pt_path, map_location=device, weights_only=False)
            except ModuleNotFoundError as e:
                # e.g., No module named 'gym_mp'
                missing = None
                m = re.search(r"No module named '([^']+)'", str(e))
                if m:
                    missing = m.group(1)
                if missing:
                    # Create chain up to missing module
                    parts = missing.split(".")
                    for i in range(len(parts)):
                        full_name = ".".join(parts[: i + 1])
                        if full_name not in sys.modules:
                            new_mod = types.ModuleType(full_name)
                            sys.modules[full_name] = new_mod
                            if i > 0:
                                parent_name = ".".join(parts[:i])
                                setattr(sys.modules[parent_name], parts[i], new_mod)
                    attempts += 1
                    continue
                raise
            except AttributeError as e:
                # e.g., module 'gym_mp.models2' has no attribute 'ClassName'
                msg = str(e)
                m = re.search(r"module '([\w\.]+)' has no attribute '([\w]+)'", msg)
                if not m:
                    # e.g., Can't get attribute 'UNetTransformer' on <module 'gym_mp.models2'>
                    m2 = re.search(r"Can't get attribute '([\w]+)' on <module '([\w\.]+)'>", msg)
                    if m2:
                        # reorder to match (module_path, class_name)
                        module_path, class_name = m2.group(2), m2.group(1)
                        ensure_dummy_for(module_path, class_name)
                        attempts += 1
                        continue
                else:
                    module_path, class_name = m.group(1), m.group(2)
                    ensure_dummy_for(module_path, class_name)
                    attempts += 1
                    continue
                raise

    if isinstance(loaded_obj, torch.nn.Module):
        model = loaded_obj
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded: {type(model).__name__}")
        print(f"Device: {device}")
        print(f"Parameters: {num_params}")
    elif isinstance(loaded_obj, dict):
        # Treat as dict checkpoint and summarize
        keys = list(loaded_obj.keys())
        tensor_items = [v for v in loaded_obj.values() if isinstance(v, torch.Tensor)]
        total_params = sum(t.numel() for t in tensor_items)
        print("Loaded: dict checkpoint")
        if keys:
            print(f"Keys: {keys[:20]}{' ...' if len(keys) > 20 else ''}")
        if tensor_items:
            print(f"Detected flat state_dict: {len(tensor_items)} tensors | Total params: {total_params}")
        # Common nested patterns
        for k in ["state_dict", "model_state_dict", "model", "net", "module"]:
            v = loaded_obj.get(k, None)
            if isinstance(v, dict):
                t_items = [vv for vv in v.values() if isinstance(vv, torch.Tensor)]
                t_params = sum(t.numel() for t in t_items)
                print(f"Nested '{k}': {len(t_items)} tensors | Total params: {t_params}")
        if not _GYM_MP_AVAILABLE:
            print("Note: gym_mp not available; skipped reconstructing model class.")
    else:
        print(f"Loaded object of type: {type(loaded_obj).__name__}")
        print("Unable to summarize parameters for this object type.")


if __name__ == "__main__":
    main()


