import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
from filepaths import get_env_path
import csv
from typing import Any
import pandas as pd

MODEL_RESULTS_PATH = get_env_path('stats/SAE.txt')
MODEL_RESULTS_CSV_PATH = get_env_path('stats/SAE.csv')
CID10_PATH = get_env_path('data/CID-10-SUBCATEGORIAS.CSV')

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

def cid10_dict():
    with open(CID10_PATH, 'r', encoding='latin1') as in_file:
        out = {}
        reader = csv.reader(in_file, delimiter=';')
        next(reader, None)
        for row in reader:
            code = row[0].lower()
            name = row[4].lower()

            while len(code) < 4:
                code += '0'
            
            if row[7]:
                for alt_code in row[7].split(sep=','):
                    while len(alt_code) < 4:
                        alt_code += '0'
                    out[alt_code.lower()] = name
            
            out[code] = name
    
    return out

events_dict = {
    # Categoria 1: Tratamentos Filtrantes (Diálise)
    "event_c1dialise_hd": "Sessões de Hemodiálise",
    "event_c1dialise_dp": "Sessões de Diálise Peritoneal",

    # Categoria 2: Medicamentos
    "event_c2med_aza": "Uso de Azatioprina (Imunossupressor)",
    "event_c2med_csa": "Uso de Ciclosporina (Imunossupressor)",
    "event_c2med_tacro": "Uso de Tacrolimo (Imunossupressor)",
    "event_c2med_micof": "Uso de Micofenolato (Imunossupressor)",
    "event_c2med_sevel": "Uso de Sevelamer (Controle de Fósforo)",
    "event_c2med_calci": "Uso de Calcitriol/Cálcio",
    "event_c2med_eritro": "Uso de Eritropoietina",
    "event_c2med_hidfe": "Uso de Ferro Endovenoso",
    "event_c2med_antiane": "Uso de Antianêmicos",

    # Categoria 3: Acessos Vasculares
    "event_c3acesso_ct": "Uso de Cateter Temporário (Acesso de Emergência)",
    "event_c3acesso_ctk": "Uso de Cateter Tunelizado (Permcath)",
    "event_c3acesso_fv": "Uso de Fístula Arteriovenosa",

    # Categoria 5: Cirurgias e Transplantes
    "event_c5tx_tx": "Transplante Renal (Ativo)",
    "event_c5tx_extx": "Perda do Transplante (Retorno à Diálise)",

    # Categoria 6: Ambiente / Hospitalização
    "event_c6interna_inter": "Internações"
}

def translate_event_name(event: str, cid_dict: dict[str, str]):
    event = event.lower().strip()
    
    if 'diagn' in event:
        event = event.replace("diagn_", "")
        while len(event) < 4:
            event += '0'
        name = cid_dict.get(event, None)
    
    else:
        name = events_dict.get(event, None)
    
    if name is not None:
        return name.capitalize()
    
    else:
        return 'EVENTO INVÁLIDO'

def translate_event_names(full_text: str, cid_dict: dict[str, str]) -> list[str]:
    words = full_text.split(' ')
    new_words = []
    for word in words:
        new_words.append(word)
        translated = translate_event_name(event=word, cid_dict=cid_dict)
        if translated != 'EVENTO INVÁLIDO':
            new_words.append(f'[{translated}]')
    
    return ' '.join(new_words)


def tcav_result_df_from_concepts(cavs: dict[str, Any]) -> pd.DataFrame:
    cols_to_keep = ['Factor', 'Rule', 'TCAV_score', 'p_value', 't_stat']
    df_dict = {
        key: {k: v for k, v in obj.items() if k in cols_to_keep} for key, obj in cavs.items()
    }
    
    return pd.DataFrame(df_dict).transpose().reset_index(drop=True)
