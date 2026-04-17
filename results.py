import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F

MODEL_RESULTS_PATH = 'stats/SAE.txt'
MODEL_RESULTS_CSV_PATH = 'stats/SAE.csv'



def save_model_stats(original_input: torch.Tensor, encoded: torch.Tensor, decoded: torch.Tensor,
                     stats: dict[str, float]):
    mean_mod_zeroes = encoded.shape[1] - torch.mean(torch.count_nonzero(encoded, dim=1).float())
    cos_sim = F.cosine_similarity(original_input, decoded, dim=1)
    mean_cos_sim = torch.mean(cos_sim)
    mean_perc_loss = (1 - mean_cos_sim.item()) * 100
    
    output = f'##### RESULTADOS #####\n'\
                f'Hiperparâmetros: epochs={stats['epochs']}, lr={stats['learning_rate']}, alpha={stats['alpha']}, weight_decay={stats['weight_decay']}\n'\
                f'Média de nulos nos embeddings modificados: {mean_mod_zeroes} / {encoded[0].shape[0]}\n'\
                f'Diferença cosseno média(%): {mean_perc_loss}\n\n'
    
    with open(MODEL_RESULTS_PATH, 'a+') as out_file:
      out_file.write(output)
    
    with open(MODEL_RESULTS_CSV_PATH, 'a+') as csv_out:
      csv_out.write(f'{stats['epochs']},{stats['learning_rate']},{stats['alpha']},{stats['weight_decay']},{mean_mod_zeroes},{mean_perc_loss}\n')
    
    print(output)

def plot_losses(losses: list | np.ndarray | torch.Tensor):
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