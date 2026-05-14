import os

import torch
from torch import nn
import torch.nn.functional as F
from results import save_model_stats
from filepaths import get_env_path

SAE_MODEL_PATH = get_env_path('models')

def print_tensor_data(tensor: torch.Tensor) -> None:
    print(f' >>>>>\n Value: {tensor}\n Shape: {tensor.shape}\n DataType: {tensor.dtype}\n Device: {tensor.device} \n <<<<<')

class SAE(nn.Module):
    """ Implementa o Sparse AutoEncoder.
        O objetivo do modelo é criar uma representação expandida e esparsa de embeddings de outros modelos,
        equilibrando acurácia na reconstrução dos dados originais e esparsidade da representação expandida. """

    def __init__(self, data_dimension:int=192, scaling_factor:float=1.5):
        super().__init__()

        self.data_dimension = data_dimension
        self.scaling_factor = scaling_factor

        self.encoder = nn.Linear(data_dimension, int(scaling_factor * data_dimension), bias=True)

        # self.decoder_bias = nn.Parameter(torch.zeros(int(self.data_dimension)))
        self.decoder_bias = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """ Cria uma representação expandida e esparsa do vetor original """
        return F.relu(self.encoder(x))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """ Reconstrói um vetor a partir de sua representação expandida. """
        return F.linear(h, self.encoder.weight.t(), self.decoder_bias)

    def forward(self, x:torch.Tensor):
        h = self.encode(x)
        z_hat = self.decode(h)
        return h, z_hat

def train_sae_model(inputs: torch.Tensor, epochs:int=2000, learning_rate:float=1e-3, weight_decay:float=0.0,
                    alpha:float=8e-4, save_data=True, data_source:str = 'training') -> SAE:
    """" Treina o Sparse AutoEncoder usando a entrada e os hiperparâmetros passados.
         O parâmetro alpha é a constante que controla a penalização por dados densos. """
    model = SAE()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    if os.path.exists(SAE_MODEL_PATH + f'/{data_source}_sae.pth'):
        print(f'Carregando SAE do arquivo salvo')
        model.load_state_dict(torch.load(f=SAE_MODEL_PATH + f'/{data_source}_sae.pth'))
        return model

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # original_sparsity = float((inputs <= 1e-5).float().mean().detach().cpu().item())
    # print(f'Original sparsity: {original_sparsity*100}%')
    inputs = inputs.to(device, dtype=torch.float32)
    losses = []

    for epoch in range(1, epochs + 1):
        # Treinamento do modelo e aplicação aos inputs
        model.train()
        h, reconstruction = model(inputs)

        # Construção da função de perda: Erro de reconstrução e penalidade de esparsidade
        mse = F.mse_loss(reconstruction, inputs)
        l1 = alpha * h.abs().sum(dim=-1).mean()
        loss = mse + l1

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Avaliação do modelo
        model.eval()
        losses.append(loss.item())
        if epoch % 100 == 0:
            with torch.no_grad():
                # cos_sim = torch.nn.functional.cosine_similarity(reconstruction, inputs, dim=1)
                # mean_cos_sim = torch.mean(cos_sim)
                # mean_perc_loss = (1 - mean_cos_sim.item()) * 100
                sparsity = float((h <= 1e-5).float().mean().detach().cpu().item())
                print(f"Epoch {epoch}: loss={loss.item()}, sparsity={sparsity*100:.2f}%")

    if save_data:
        save_model_stats(inputs, model.encode(inputs), model.decode(model.encode(inputs)), {'epochs': epochs, 'learning_rate': learning_rate,
                                                                  'alpha': alpha, 'weight_decay': weight_decay}, data_source)
    torch.save(obj=model.state_dict(), f=SAE_MODEL_PATH + f'/{data_source}_sae.pth')
    return model