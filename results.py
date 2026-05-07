import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
from filepaths import get_env_path

MODEL_RESULTS_PATH = get_env_path('stats/SAE.txt')
MODEL_RESULTS_CSV_PATH = get_env_path('stats/SAE.csv')

def save_model_stats(original_input: torch.Tensor, encoded: torch.Tensor, decoded: torch.Tensor,
                     stats: dict[str, float], data_source: str) -> None:
    """" Salva os hiperparâmetros e métricas de desempenho do modelo treinado.
         As métricas são: similaridade cosseno em % e número de colunas nulas. """
    with (torch.no_grad()):
        mean_mod_zeroes = encoded.shape[1] - torch.mean(torch.count_nonzero(encoded, dim=1).float())
        cos_sim = F.cosine_similarity(original_input, decoded, dim=1)
        pairwise_cos_sim = 0.0
        for feat1 in range(encoded.shape[1]):
            for feat2 in range(encoded.shape[1]):
                if feat1 != feat2:
                    if encoded[:, feat1].sum() != 0 and encoded[:, feat2].sum() != 0:
                        pairwise_cos_sim += F.cosine_similarity(encoded[:, feat1], encoded[:, feat2], dim=0)
                        # print(encoded[:, feat1].nonzero(), encoded[:, feat2].nonzero(), pairwise_cos_sim)
                        # input()

        pairwise_cos_sim /= (encoded.shape[1] * (encoded.shape[1] - 1))
        zero = 0
        encoded = encoded.detach().cpu().numpy()

        real_sparsity = 0.0
        for feat in range(encoded.shape[1]):
            if np.count_nonzero(encoded[:, feat] > 1e-5) == 0:
                zero += 1
            else:
                real_sparsity += (encoded[:, feat] <= 1e-5).mean()
                # print(np.count_nonzero(encoded[:, feat] <= 1e-5))

        real_sparsity /= (encoded.shape[1])
        print(f'Final sparsity: {real_sparsity}')
        print(f'Nulos: {zero}/{encoded.shape[1]}')
        mean_cos_sim = torch.mean(cos_sim)
        # mean_perc_loss = (1 - mean_cos_sim.item()) * 100

        output = f'##### RESULTADOS ({data_source}) #####\n'\
                    f'Hiperparâmetros: epochs={stats["epochs"]}, lr={stats["learning_rate"]}, alpha={stats["alpha"]}\n'\
                    f'Média de sparsity nos embeddings modificados (não nulos): {real_sparsity*100:.3f}%\n'\
                    f'Semelhança cosseno média entre pares: {pairwise_cos_sim}\n'\
                    f'Semelhança cosseno média entre orginal e encodado: {mean_cos_sim}\n'\
                    f'Vetores nulos: {zero} / {encoded.shape[1]}\n\n'

        with open(MODEL_RESULTS_PATH, 'a+') as out_file:
          out_file.write(output)

        # with open(MODEL_RESULTS_CSV_PATH, 'a+') as csv_out:
          # csv_out.write(f'{stats["epochs"]},{stats["learning_rate"]},{stats["alpha"]},{stats["weight_decay"]},{mean_mod_zeroes},{mean_perc_loss}\n')

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