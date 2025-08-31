import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class MoEDecoder(nn.Module):
    def __init__(self, input_dim,hidden_dim,  output_dim, dropout=0.4, num_experts=8):
        super(MoEDecoder, self).__init__()
        self.num_experts = num_experts

        # Experts: 각 전문가에 대해 별도의 네트워크를 생성
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim)
        ) for _ in range(num_experts)])

        # Gating network: 각 전문가의 가중치를 결정
        self.gating_network = nn.Linear(input_dim, num_experts)
        
        

    def forward(self, x):
        # Gating network를 통해 각 전문가의 가중치 계산
        
        gate_values = F.softmax(self.gating_network(x), dim=-1)
        #gate_values.unsqueeze(1)
        
        # 각 전문가의 출력을 계산하고, 가중치를 곱한 후 합산 (cnn )
        output = sum(gate_values[:, i].unsqueeze(1) * self.experts[i](x.unsqueeze(1)).squeeze(1) for i in range(self.num_experts))

        return output

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.24):
        super(Encoder, self).__init__()
        
        self.dropout = dropout
        #self.Conv1 = nn.Conv1d(1, 1, 9, 1, 4 )
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        #self.Conv2 = nn.Conv1d(1, 1, 9, 1, 4 )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = self.ln1((F.elu(self.fc1(x))))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.ln2((F.elu(self.fc2(x1))))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = F.elu(self.fc3(x2))
        x3 = F.dropout(x3, self.dropout, training=self.training)        
        z = self.fc4(x3)
        z = z.squeeze()
        return z
    
    
class gruEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.24):
        super(gruEncoder, self).__init__()
        
        self.dropout = dropout
        #self.Conv1 = nn.Conv1d(1, 1, 9, 1, 4 )
        self.fc1 = nn.GRU(input_dim, hidden_dim, 2, batch_first=True, dropout=dropout)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.Moe = MoEDecoder(hidden_dim, hidden_dim, latent_dim, 0, 8 )
        #self.Conv2 = nn.Conv1d(1, 1, 9, 1, 4 )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, hidden=None):
        x = x.unsqueeze(1)
        x = F.dropout(x, self.dropout, training=self.training)
        x1, newhidden = self.fc1(x, hidden)
        x1 = self.ln1((F.elu(x1)))
        
        x1 = x1.squeeze()
        
        z = self.Moe(x1)    
        
        return z, newhidden

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        z = F.relu(self.fc1(z))
        x_recon = self.fc2(z)
        return x_recon

class GumbelSoftmaxVQVAE(nn.Module):
    def __init__(self, input_dim,frames ,hidden_dim, latent_dim, num_embeddings, temperature=1.0):
        super(GumbelSoftmaxVQVAE, self).__init__()
        self.encoder = MoEDecoder(input_dim*frames, latent_dim, 4, hidden_dim)
        self.targetencoder = Encoder(input_dim, 64, latent_dim)
        self.decoder = MoEDecoder(latent_dim, input_dim, 8, hidden_dim)
        self.codebook = nn.Embedding(num_embeddings, latent_dim)
        self.num_embeddings = num_embeddings
        self.temperature = temperature
        self.latent_dim = latent_dim
        
        # Initialize codebook weights
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def set_normalization(self,std,avg):
        self.data_std=std
        self.data_avg=avg
        
    def normalize(self, t):
        return (t - self.data_avg) / self.data_std
    
    def denormalize(self, t):

        return t*self.data_std + self.data_avg
    
    def forward(self, x, result=None):
        
        
        if result != None:
            x = x.squeeze(-1)
            estimate = self.encoder(x)
            
            target = self.targetencoder(result)
            matching_loss = F.mse_loss(estimate, target)
            logits = torch.matmul(target, self.codebook.weight.T)
            one_hot = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
            
            x_recon = self.decoder(one_hot)
            return x_recon, matching_loss
        else:
            estimate = self.encoder(x)
            logits = torch.matmul(estimate, self.codebook.weight.T)
            one_hot = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
            x_recon = self.decoder(one_hot)
            return x_recon
        
            
            
            
        

    def compute_loss(self, x, x_recon, z_e, z_q):
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        
        # Commitment loss
        commitment_loss = F.mse_loss(z_e, z_q)
        
        # Total loss
        total_loss = recon_loss + 0.25 * commitment_loss
        return total_loss
    
    
class Model3(nn.Module):
    def __init__(self, input_dim,frames ,hidden_dim, latent_dim, num_embeddings, temperature=1.0):
        super(Model3, self).__init__()

        inputframe = frames
        self.Encoder = Encoder(input_dim, hidden_dim, num_embeddings*latent_dim)
        self.Estimator = MoEDecoder(input_dim*(1), hidden_dim, num_embeddings*latent_dim, 0.4, 8 )
        self.Decoder = gruEncoder(num_embeddings*latent_dim,hidden_dim, input_dim,0)
        self.input_size = input_dim
        self.input_frame = frames
        
        self.C = num_embeddings
        self.D = latent_dim

    def set_normalization(self,std,avg):
        self.data_std=std
        self.data_avg=avg
        
    def normalize(self, t):
        return (t - self.data_avg) / self.data_std
    
    def denormalize(self, t):

        return t*self.data_std + self.data_avg
    

    def sample_gumbel(self, tensor, scale, eps=1e-20):
        scale = scale.reshape(-1,1,1,1) #This is noise scale between 0 and 1
        noise = torch.rand_like(tensor) - 0.5 #This is random noise between -0.5 and 0.5
        samples = scale * noise + 0.5 #This is noise rescaled between 0 and 1 where 0.5 is default for 0 noise
        return -torch.log(-torch.log(samples + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature, scale):
        y = logits + self.sample_gumbel(logits, scale)
        return F.softmax(y / temperature, dim=-1)
    
    def gumbel_softmax(self, logits, temperature, scale):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature, scale)

        y_soft = y.view(logits.shape)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        y_hard = y_hard.view(logits.shape)

        return y_soft, y_hard

    def sample(self, z, knn):
        z = z.reshape(-1, self.C, self.D)
        z = z.unsqueeze(0).repeat(knn.size(0), 1, 1, 1)
        z_soft, z_hard = self.gumbel_softmax(z, 1.0, knn)
        z_soft = z_soft.reshape(-1, self.C*self.D)
        z_hard = z_hard.reshape(-1, self.C*self.D)
        return z_soft, z_hard
    
    def forward(self, x, knn, t=None, hidden=None): #x=input, knn=samples, t=output
        #training
        
        #x = torch.cat([x, x[:,:35] - x[:,315:]], dim=1)
        x_vec = x[:,:35] - x[:,315:]
        if t is not None:
            #Normalize
            t_vec = t[:,315:] - x[:,315:350]
            #t_vec = t - x
            #Encode Y
            
            #target_logits = self.Encoder(torch.cat((x,t_vec), dim=1))
            target_logits = self.Encoder(t_vec)
            target_probs, target = self.sample(target_logits, knn)

            #Encode X
            estimate_logits = self.Estimator(x_vec)
            estimate_probs, estimate = self.sample(estimate_logits, knn)
            criterion = nn.CrossEntropyLoss()
            #Decode
            y, hidden = self.Decoder(target, hidden)
            torch.mean(estimate)
            #Renormalize
            return y, F.mse_loss(target_probs, estimate_probs), F.mse_loss(t_vec, y), hidden
        #inference
        else:
            #Normalize
            knn = knn.unsqueeze(0).repeat(30,1)
            #Encode X
            estimate_logits = self.Estimator(x_vec)
            estimate_probs, estimate = self.sample(estimate_logits, knn)
            
            
            #Decode
            y , hidden = self.Decoder(estimate, hidden)
            if y.size(0) > 1:
                y = y.mean(dim=0, keepdim=True)

            #y = y.view(-1, 35).sum(dim=0, keepdim=True)
            
            #Renormalize
            #return y[:,729:]
            return x[:,315:350] + y, hidden