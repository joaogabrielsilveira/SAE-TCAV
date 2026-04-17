import torch
from torch import nn
from results import save_model_stats

def print_tensor_data(tensor: torch.Tensor):
  print(f' >>>>>\n Value: {tensor}\n Shape: {tensor.shape}\n DataType: {tensor.dtype}\n Device: {tensor.device} \n <<<<<')

class SAE(nn.Module):
    def __init__(self, data_dimension=192, scaling_factor=1.5):
        self.data_dimension = data_dimension
        self.scaling_factor = scaling_factor
        super(SAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(self.data_dimension, int(self.data_dimension * self.scaling_factor)),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(self.data_dimension * self.scaling_factor), self.data_dimension)
        )

    def forward(self, x):
        h = self.encoder(x)
        z_hat = self.decoder(h)
        return h, z_hat

def train_sae_model(inputs: torch.Tensor, epochs=10000, learning_rate=1e-3, weight_decay=0.0, alpha=5e-4):
    model = SAE()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    inputs = inputs.to(device)
    losses = []

    for epoch in range(1, epochs + 1):
        # treinamento do modelo e aplicação aos inputs
        model.train()
        h, reconstruction = model(inputs)

        # construção da função de perda: erro de reconstrução e penalidade de esparsidade
        reconstruction_loss = ((torch.norm(reconstruction - inputs, p=2)) ** 2) / inputs.shape[1]
        l1_loss = alpha * torch.norm(h, p=1)
        loss = reconstruction_loss + l1_loss

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # avaliação do modelo
        model.eval()
        losses.append(loss.item())
        if epoch % 100 == 0:
            cos_sim = torch.nn.functional.cosine_similarity(reconstruction, inputs, dim=1)
            mean_cos_sim = torch.mean(cos_sim)
            mean_perc_loss = (1 - mean_cos_sim.item()) * 100
            print(f"Epoch {epoch}: loss={loss.item()}, cosine_diff={mean_perc_loss}%")

    save_model_stats(inputs, model(inputs)[0], model(inputs)[1], {'epochs': epochs, 'learning_rate': learning_rate,
                                                                  'alpha': alpha, 'weight_decay': weight_decay})