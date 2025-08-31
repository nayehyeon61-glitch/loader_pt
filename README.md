## Loader usage

간단한 로더(`load_dif16.py`)로 PyTorch `.pt` 체크포인트를 로드하고 요약할 수 있습니다. CPU/GPU 선택과 안전 로딩(safe) / 전체 언피클(unsafe) 로딩을 지원합니다.

### Prerequisites
- Python 3.8+
- PyTorch 설치 (CUDA 사용 시 CUDA가 설치된 환경)
- 프로젝트 구조 예시:
  - `load_dif16.py`
  - `gym_mp/model2.py` 또는 `gym_mp/models2.py`
  - `gym_mp/models_vqvae.py` (VQ-VAE 모델 로드를 위해 필요)

참고: `gym_mp/model2.py`만 있는 경우에도 스크립트가 자동으로 `gym_mp.models2`로 별칭(alias)을 걸어줍니다.

### Quick start

- CPU에서 로드 (권장 기본):
```bash
python3 /home/nayaehyun/test_loader/load_dif16.py --pt /home/nayaehyun/test_loader/dif_multi_n_attn.pt --device cpu --unsafe
```

- GPU에서 로드 (CUDA 사용 가능 시):
```bash
python3 /home/nayaehyun/test_loader/load_dif16.py --pt /home/nayaehyun/test_loader/dif_multi_n_attn.pt --device cuda --unsafe
```

- VQ-VAE 체크포인트 로드:
```bash
python3 /home/nayaehyun/test_loader/load_dif16.py --pt /home/nayaehyun/test_loader/vqvae_category.pt --device cpu --unsafe
```

### Safe vs Unsafe 모드
- 기본적으로 안전 로딩(Weights-only)을 먼저 시도합니다. 그러나 커스텀 클래스에 의존하는 체크포인트는 안전 모드에서 구조 추론이 제한될 수 있습니다.
- 이때 `--unsafe` 플래그를 사용하면 전체 언피클을 수행합니다. 신뢰 가능한 파일에서만 사용하세요.

- 안전 모드 예시(필요시 시도):
```bash
python3 /home/nayaehyun/test_loader/load_dif16.py --pt /home/nayaehyun/test_loader/dif_multi_n_attn.pt --device cpu
```

### 예시 출력
- `dif_multi_n_attn.pt` 로드 시:
```text
Loaded: DenoiseDiffusion14_mul_vec_n_atten
Device: cpu
Parameters: 8037714
```

- `vqvae_category.pt` 로드 시:
```text
Loaded: Model3
Device: cpu
Parameters: 68899427
```

### Troubleshooting
- 모듈을 찾을 수 없음(ModuleNotFoundError: `gym_mp.models2` 등):
  - `gym_mp/model2.py` 또는 `gym_mp/models2.py`가 존재하는지 확인하세요.
  - VQ-VAE의 경우 `gym_mp/models_vqvae.py`가 필요합니다.
  - 스크립트는 `model2.py` → `models2` 별칭을 자동 생성합니다.

- "Can't get attribute ..." 또는 커스텀 클래스 관련 에러:
  - `--unsafe` 플래그로 다시 시도하세요.

- 안전 모드에서 최상위 객체가 `NoneType`으로 보이는 경우:
  - 체크포인트가 커스텀 방식으로 저장되었을 수 있습니다. 원 훈련 코드에서 `state_dict`로 재저장하여 사용하세요.

### Forward 테스트(선택)
모델 입력 텐서 형태가 알려져 있다면, 간단한 스크립트에서 로드 후 forward를 호출할 수 있습니다. 입력 형태가 다를 수 있으므로, 정확한 입력 사양을 아신 경우에만 사용하세요.

```python
import torch
from pathlib import Path

pt_path = Path('/home/nayaehyun/test_loader/dif_multi_n_attn.pt')  # 또는 vqvae_category.pt
device = torch.device('cpu')

# 로더 스크립트와 동일한 import 경로가 보장되어야 합니다
import sys
sys.path.insert(0, '/home/nayaehyun/test_loader')
import gym_mp.model2 as _gm2  # 필요 시 models_vqvae 등 함께 import
sys.modules['gym_mp.models2'] = _gm2

model = torch.load(str(pt_path), map_location=device, weights_only=False)
model.eval()

# TODO: 아래에 실제 입력 크기를 맞춰주세요
dummy_input = torch.randn(1, 3, 256, 256, device=device)
with torch.no_grad():
    _ = model(dummy_input)
print('forward OK')
```

입력 스펙을 모르신다면 알려주세요. 해당 모델에 맞는 더미 입력을 같이 구성해 드리겠습니다.


