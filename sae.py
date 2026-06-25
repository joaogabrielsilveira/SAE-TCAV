import os

import torch
from torch import nn
import torch.nn.functional as F
from results import save_model_stats
from filepaths import get_env_path

SAE_MODEL_PATH = get_env_path('models')

def topk(x:torch.Tensor, k:int) -> torch.Tensor:
    top_vals, top_indices = torch.topk(x, k)
    new_t = torch.zeros_like(x, dtype=x.dtype, device=x.device)
    new_t.scatter_(dim=-1, index=top_indices, src=top_vals)

    return new_t

class SAE(nn.Module):
    """ Implementa o Sparse AutoEncoder.
        O objetivo do modelo é criar uma representação expandida e esparsa de embeddings de outros modelos,
        equilibrando acurácia na reconstrução dos dados originais e esparsidade da representação expandida. """

    def __init__(self, data_dimension:int=192, scaling_factor:float=1.5, use_decoder_bias: bool = False, type: str='ReLU', k: int | None = None, k_aux: int | None = None):
        super().__init__()
        self.num_latents = int(data_dimension*scaling_factor)

        self.encoder = nn.Linear(data_dimension, self.num_latents, bias=True)

        if type == 'ReLU':
            if use_decoder_bias:
                self.decoder_bias = nn.Parameter(torch.zeros(int(data_dimension)))
            else:
                self.decoder_bias = None
        
        elif type == 'TopK':
            self.decoder = nn.Linear(self.num_latents, data_dimension, bias=use_decoder_bias)
            self.k = k
            self.k_aux = k_aux
        
        else:
            raise RuntimeError("Invalid SAE type; Try \'ReLU\' or \'TopK\'")

        self.type = type

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """ Cria uma representação expandida e esparsa do vetor original """
        if self.type == 'ReLU':
            return F.relu(self.encoder(x))
        elif self.type == 'TopK':
            pre_selection = self.encoder(x)
            selected =  topk(pre_selection, k=self.k)
            return pre_selection, selected

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """ Reconstrói um vetor a partir de sua representação expandida. """
        if self.type == 'ReLU':
            return F.linear(h, self.encoder.weight.t(), self.decoder_bias)
        elif self.type == 'TopK':
            return self.decoder(h)

    def forward(self, x:torch.Tensor):
        if self.type == 'ReLU':
            h = self.encode(x)
            z_hat = self.decode(h)
            return h, z_hat
        
        elif self.type == 'TopK':
            pre_selection, h = self.encode(x)
            z_hat = self.decoder(h)

            e = x - z_hat

            dead_mask = (h == 0)
            dead_neurons = pre_selection.masked_fill(~dead_mask, float('-inf'))

            z = topk(dead_neurons, k=self.k_aux)
            e_hat = self.decode(z)
            # perda residual com tok_k_aux neuronios mortos
            l_aux = F.mse_loss(e_hat, e)
            return h, z_hat, l_aux
        
    def loss(self, x:torch.Tensor, alpha:float):
        if self.type == 'ReLU':
            h, reconstruction = self.forward(x)
            mse = F.mse_loss(reconstruction, x)
            l1 = alpha * h.abs().mean()
            return mse + l1
        
        elif self.type == 'TopK':
            h, z_hat, l_aux = self.forward(x)
            mse_loss = F.mse_loss(x, z_hat)

            return mse_loss + l_aux * alpha
            
def train_sae_model(inputs: torch.Tensor, epochs:int=1000, learning_rate:float=1e-3, weight_decay:float=0.0,
                    alpha:float=1e-1, save_data=True, data_source:str = 'training', use_decoder_bias: bool = False,
                    rng_seed: int=42, use_cache: bool=False, scaling_factor: float=1.5, type: str='ReLU',
                    k: int=12, k_aux:int=64) -> SAE:
    """" Treina o Sparse AutoEncoder usando a entrada e os hiperparâmetros passados.
         O parâmetro alpha é a constante que controla a penalização por dados densos. """
    torch.manual_seed(rng_seed)
    model = SAE(use_decoder_bias=use_decoder_bias, scaling_factor=scaling_factor, type=type, k=k, k_aux=k_aux)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    model.to(device)
    inputs.to(device)

    if use_cache and os.path.exists(SAE_MODEL_PATH + f'/{data_source}_sae.pth'):
        print(f'Carregando SAE do arquivo salvo')
        model.load_state_dict(torch.load(f=SAE_MODEL_PATH + f'/{data_source}_sae.pth'))
        return model

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)    # original_sparsity = float((inputs <= 1e-5).float().mean().detach().cpu().item())
    # print(f'Original sparsity: {original_sparsity*100}%')
    inputs = inputs.to(device, dtype=torch.float32)
    losses = []

    for epoch in range(1, epochs + 1):
        # Treinamento do modelo e aplicação aos inputs
        model.train()

        loss = model.loss(inputs, alpha)
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if model.type == 'TopK':
            model.decoder.weight.data = F.normalize(model.decoder.weight.data, p=2, dim=0)

        # Avaliação do modelo
        model.eval()
        losses.append(loss.item())
        if epoch % 200 == 0:
            with torch.no_grad():
                # cos_sim = torch.nn.functional.cosine_similarity(reconstruction, inputs, dim=1)
                # mean_cos_sim = torch.mean(cos_sim)
                # mean_perc_loss = (1 - mean_cos_sim.item()) * 100
                # sparsity = float((h <= 1e-5).float().mean().detach().cpu().item())
                # print(f"Epoch {epoch}: loss={loss.item():.6f}")
                pass

    if save_data:
        save_model_stats(inputs, model.encode(inputs), model.decode(model.encode(inputs)), {'epochs': epochs, 'learning_rate': learning_rate,
                                                                  'alpha': alpha, 'weight_decay': weight_decay}, data_source)
        torch.save(obj=model.state_dict(), f=SAE_MODEL_PATH + f'/{data_source}_sae.pth')
        
    model.to('cpu')
    inputs.to('cpu')
    return model