import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
from filepaths import get_env_path

MODEL_RESULTS_PATH = get_env_path('stats/SAE.txt')
MODEL_RESULTS_CSV_PATH = get_env_path('stats/SAE.csv')

def save_model_stats(original_input: torch.Tensor, encoded: torch.Tensor, decoded: torch.Tensor,
                     stats: dict[str, float]) -> None:
    """" Salva os hiperparâmetros e métricas de desempenho do modelo treinado.
         As métricas são: similaridade cosseno em % e número de colunas nulas. """
    with torch.no_grad():
        mean_mod_zeroes = encoded.shape[1] - torch.mean(torch.count_nonzero(encoded, dim=1).float())
        cos_sim = F.cosine_similarity(original_input, decoded, dim=1)
        pairwise_cos_sim = 0.0
        for feat1 in range(encoded.shape[1]):
            for feat2 in range(encoded.shape[1]):
                if feat1 != feat2:
                    pairwise_cos_sim += F.cosine_similarity(encoded[:, feat1], encoded[:, feat2], dim=0)

        pairwise_cos_sim /= (encoded.shape[1] * (encoded.shape[1] - 1))

        mean_cos_sim = torch.mean(cos_sim)
        mean_perc_loss = (1 - mean_cos_sim.item()) * 100

        output = f'##### RESULTADOS #####\n'\
                    f'Hiperparâmetros: epochs={stats["epochs"]}, lr={stats["learning_rate"]}, alpha={stats["alpha"]}, weight_decay={stats["weight_decay"]}\n'\
                    f'Média de nulos nos embeddings modificados: {mean_mod_zeroes} / {encoded[0].shape[0]}\n'\
                    f'Semelhança cosseno entre pares média: {pairwise_cos_sim}\n'\
                    f'Diferença cosseno média(%): {mean_perc_loss}\n\n'

        with open(MODEL_RESULTS_PATH, 'a+') as out_file:
          out_file.write(output)

        with open(MODEL_RESULTS_CSV_PATH, 'a+') as csv_out:
          csv_out.write(f'{stats["epochs"]},{stats["learning_rate"]},{stats["alpha"]},{stats["weight_decay"]},{mean_mod_zeroes},{mean_perc_loss}\n')

        print(output)

def plot_losses(losses: list | np.ndarray | torch.Tensor) -> None:
    """" Traça um gráfico com a progressão da perda do modelo ao longo das epochs. """
    x = range(1, len(losses) + 1)
    y = losses
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_yscale('log')
    
    plt.plot(x, y)
    plt.grid()
    plt.title("Sparse AutoEncoder Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()