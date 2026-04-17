import torch
from torch import nn
import torch.nn.functional as F
from results import save_model_stats

def print_tensor_data(tensor: torch.Tensor) -> None:
    print(f' >>>>>\n Value: {tensor}\n Shape: {tensor.shape}\n DataType: {tensor.dtype}\n Device: {tensor.device} \n <<<<<')

class SAE(nn.Module):
    """ Implementa o Sparse AutoEncoder.
        O objetivo do modelo é criar uma representação expandida e esparsa de embeddings de outros modelos,
        equilibrando acurácia na reconstrução dos dados originais e esparsidade da representação expandida. """

    def __init__(self, data_dimension:int=192, scaling_factor:float=1.5):
        super(SAE, self).__init__()

        self.data_dimension = data_dimension
        self.scaling_factor = scaling_factor

        self.encoder_matrix = nn.Parameter(torch.zeros(int(self.data_dimension * self.scaling_factor), self.data_dimension))
        nn.init.kaiming_uniform_(self.encoder_matrix)

        self.encoder_bias = nn.Parameter(torch.zeros(int(self.data_dimension * self.scaling_factor)))

        self.decoder_bias = nn.Parameter(torch.zeros(int(self.data_dimension)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """ Cria uma representação expandida e esparsa do vetor original """
        return F.relu(F.linear(x, self.encoder_matrix, self.encoder_bias))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """ Reconstrói um vetor a partir de sua representação expandida. """
        return F.linear(h, self.encoder_matrix.t(), self.decoder_bias)

    def forward(self, x:torch.Tensor):
        h = self.encode(x)
        z_hat = self.decode(h)
        return h, z_hat

def train_sae_model(inputs: torch.Tensor, epochs:int=10000, learning_rate:float=1e-3, weight_decay:float=0.0,
                    alpha:float=5e-4, save_data=True) -> SAE:
    """" Treina o Sparse AutoEncoder usando a entrada e os hiperparâmetros passados.
         O parâmetro alpha é a constante que controla a penalização por dados densos. """
    model = SAE()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    inputs = inputs.to(device)
    losses = []

    for epoch in range(1, epochs + 1):
        # Treinamento do modelo e aplicação aos inputs
        model.train()
        h, reconstruction = model(inputs)

        # Construção da função de perda: Erro de reconstrução e penalidade de esparsidade
        reconstruction_loss = ((torch.norm(reconstruction - inputs, p=2)) ** 2) / inputs.shape[1]
        l1_loss = alpha * torch.norm(h, p=1)
        loss = reconstruction_loss + l1_loss

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Avaliação do modelo
        model.eval()
        losses.append(loss.item())
        if epoch % 100 == 0:
            cos_sim = torch.nn.functional.cosine_similarity(reconstruction, inputs, dim=1)
            mean_cos_sim = torch.mean(cos_sim)
            mean_perc_loss = (1 - mean_cos_sim.item()) * 100
            print(f"Epoch {epoch}: loss={loss.item()}, cosine_diff={mean_perc_loss}%")

    if save_data:
        save_model_stats(inputs, model.encode(inputs), model.decode(model.encode(inputs)), {'epochs': epochs, 'learning_rate': learning_rate,
                                                                  'alpha': alpha, 'weight_decay': weight_decay})
    return model