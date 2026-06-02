from sae import train_sae_model
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from matplotlib.ticker import PercentFormatter

def activation_threshold(concept: np.ndarray, perc: int=90):
    pos_act = concept[concept > 0]
    if len(pos_act) == 0:
        return np.inf
    return np.percentile(pos_act, perc)

def high_activation_matrix(concepts: np.ndarray, perc: int=90):
    matrix = []
    for k in range(concepts.shape[1]):
        concept_k = concepts[:, k]
        thresh = activation_threshold(concept_k, perc)
        mask = (concept_k > thresh)
        matrix.append(mask)
    
    return np.asarray(matrix, dtype=bool).transpose()

def max_activation_overlap(sae_i: dict[str], concept:int, sae_j: dict[str], perc: int = 90):
    overlaps = []
    matrix_i, matrix_j = sae_i['high_act_matrix'], sae_j['high_act_matrix']

    c_i = matrix_i[:, concept]
    for k in range(matrix_j.shape[1]):
        c_j = matrix_j[:, k]
        intersection = (c_i & c_j).sum()
        union = (c_i | c_j).sum()
        if union > 0:
            overlap = intersection / union
        else:
            overlap = 0.0

        overlaps.append(overlap)

    best_pair = np.argmax(overlaps)
    return (best_pair, overlaps[best_pair])

def max_cosine_similarity(sae_i: dict[str], concept:int, sae_j: dict[str]):
    cos_sims = []
    w_i, w_j = sae_i['encoder_weights'], sae_j['encoder_weights']

    c_i = w_i[concept, :]
    nrm_i = np.linalg.norm(c_i)
    if nrm_i == 0:
        return (-1, 0.0)
    
    for k in range(w_j.shape[0]):
        c_j = w_j[k, :]

        dot = np.dot(c_i, c_j)
        nrm_j = np.linalg.norm(c_j)
        if nrm_j == 0:
            cosine_similarity = 0.0
        else:
            cosine_similarity = dot / (nrm_i * nrm_j)
        cos_sims.append(cosine_similarity)

    best_pair = np.argmax(cos_sims)
    return (best_pair, cos_sims[best_pair])

def cosine_similarity_matrix(sae_i: dict[str], sae_j: dict[str]) -> np.ndarray:
    w_i, w_j = sae_i['encoder_weights'], sae_j['encoder_weights']
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(w_i, w_j)

def train_all_saes(num_models: int, embs: np.ndarray, alpha: float=1e-1, scaling_factor: float=1.5, model_type: str='ReLU') -> list[dict[str]]:
    sae_list = []
    for i in range(num_models):

        if i == 0:
            # primeira seed fixada igual à usada na análise do pipeline para ter uma base
            current_seed = 42
        else:
            current_seed = 10 * (i ** 2) + 50 * i + 75
    
        sae = train_sae_model(inputs=torch.tensor(embs), type=model_type, alpha=alpha, scaling_factor=scaling_factor, save_data=False, use_decoder_bias=True, use_cache=False, rng_seed=current_seed, epochs=1000)

        if model_type == 'ReLU':
            weights = sae.encoder.weight.cpu().detach().numpy()
            codes = sae.encode(x=torch.tensor(embs)).cpu().detach().numpy()
        elif model_type == 'TopK':
            weights = sae.decoder.weight.T.cpu().detach().numpy()
            codes = sae.encode(x=torch.tensor(embs))[1].cpu().detach().numpy()
            
        sparsity = (codes <= 1e-5).mean()
        dead_neurons = np.all(codes <= 1e-5, axis=0).sum().item()
        reconstruction = sae.decode(torch.tensor(codes))
        mse = torch.nn.functional.mse_loss(reconstruction, torch.tensor(embs))

        # matriz binária. cada (i, j) representa se o conceito j teve ativação
        # alta (acima do percentil perc de ativações positivas) na amostra i
        high_act_matrix = high_activation_matrix(codes, perc=90)

        sae_list.append({
            'idx': i,
            'model': sae,
            'mse': mse,
            'encoded_embs': codes,
            'sparsity_level': sparsity,
            'encoder_weights': weights,
            'dead_neurons': dead_neurons,
            'high_activation_matrix': high_act_matrix
        })
    return sae_list

def get_overlap(sae_i: dict[str], idx_i: int, sae_j: dict[str], idx_j: int) -> float:
    high_matrix_i, high_matrix_j = sae_i['high_activation_matrix'], sae_j['high_activation_matrix']
    high_mask_i, high_mask_j = high_matrix_i[:, idx_i], high_matrix_j[:, idx_j]

    intersection = (high_mask_i & high_mask_j).sum()
    union = (high_mask_i | high_mask_j).sum()
    if union == 0:
        return 0.0
    
    return float(intersection / union)
def get_concepts_matching(sae_i: dict[str], sae_j: dict[str]):
    if sae_i['idx'] == sae_j['idx']:
        return None
    
    cos_sim_matrix = cosine_similarity_matrix(sae_i, sae_j)
    rows_idx, cols_idx = linear_sum_assignment(cos_sim_matrix, maximize=True)

    results = []
    for i, j in zip(rows_idx, cols_idx):
        overlap = get_overlap(sae_i=sae_i, idx_i=i,
                              sae_j=sae_j, idx_j=j)
        results.append({
            'original_concept': i,
            'best_pair': j,
            'cos_sim': cos_sim_matrix[i][j],
            'overlap': overlap
        })
    
    return pd.DataFrame(results)

def get_all_pairwise_matchings(num_models: int, sae_list: list[dict[str]]) -> list[list[pd.DataFrame]]:
    matchings_matrix = [[None for _ in range(num_models)] for _ in range(num_models)]
    for i in range(num_models):
        for j in range(num_models):
            if i == j: continue
            matchings_matrix[i][j] = get_concepts_matching(sae_i=sae_list[i], sae_j=sae_list[j])
    
    return matchings_matrix

def get_matching_stats(match_df: pd.DataFrame, relevant_cos_sim_threshold: float=0.7):
    mean_cos_sim = match_df['cos_sim'].mean()
    match_df['is_relevant_pair'] = match_df['cos_sim'] > relevant_cos_sim_threshold    
    fraction_good_matches = match_df['is_relevant_pair'].mean()
    mean_cos_sim_good_matches = (match_df[match_df['is_relevant_pair'] == True])['cos_sim'].mean()

    if np.isnan(fraction_good_matches) or np.isnan(mean_cos_sim_good_matches):
        fraction_good_matches, mean_cos_sim_good_matches = 0.0, 0.0

    return {
        'mean_cos_sim': mean_cos_sim,
        'relevant_pairs_fraction': fraction_good_matches,
        'mean_cos_sim_relevant_pairs': mean_cos_sim_good_matches
    }

def get_model_stats(sae_list: list[dict[str]]):
    models_df = pd.DataFrame(sae_list)

    mean_dead_neurons = models_df['dead_neurons'].mean()
    mean_sparsity_level = models_df['sparsity_level'].mean()
    mean_mse = models_df['mse'].mean()

    return {
        'mean_dead_neurons': mean_dead_neurons,
        'mean_sparsity': mean_sparsity_level,
        'mean_mse': mean_mse
    }

def get_full_run_df(match_matrix: list[list], n_models: int) -> pd.DataFrame:
    flat_matrix = [match_matrix[i][j] for i in range(n_models) for j in range(n_models)]
    return pd.concat(flat_matrix, axis=0)

def run_sae_random_comparison(model_nums: list[int], alphas: list[int], embs: np.ndarray, scaling_factors: list[float], model_type: str='ReLU'):
    stats_dict = {}
    model_stats_dict = {}
    for m in scaling_factors:
        for a in alphas:
            print(f'Training {max(model_nums)} SAEs with Alpha = {a}, Scaling Factor = {m}')
            sae_list = train_all_saes(num_models=max(model_nums), embs=embs, alpha=a, scaling_factor=m, model_type=model_type)
            match_matrix = get_all_pairwise_matchings(num_models=max(model_nums), sae_list=sae_list)
            for n in model_nums:
                print(f'Calculating matching stats for the first {n} models')
                full_df = get_full_run_df(match_matrix=match_matrix, n_models=n)
                run_stats = get_matching_stats(full_df)
                model_stats = get_model_stats(sae_list)
                # print(f'Relevant pairs: {run_stats["relevant_pairs_fraction"]}')
                stats_dict[(m, n, a)] = run_stats
                model_stats_dict[(m, a)] = model_stats
    
    return stats_dict, model_stats_dict

def plot_run_results(full_results: dict[tuple[int, int, int]], model_results: dict[tuple[int, int]], alpha: float, scaling_factors: list[float], model_n: list[int]):
    if len(scaling_factors) == 1:
        rel = []
        cos_sims = []
        scaling_factor = scaling_factors[0]
        for n in model_n:
            result = full_results[(scaling_factor, n, alpha)]
            relevancy = result['relevant_pairs_fraction']
            rel.append(relevancy)

            cos_sim = result['mean_cos_sim']
            cos_sims.append(cos_sim)

        
        mean_mse = model_results[(scaling_factor, alpha)]['mean_mse']
        mean_sparsity = model_results[(scaling_factor, alpha)]['mean_sparsity']
        mean_dead_neurons = model_results[(scaling_factor, alpha)]['mean_dead_neurons']

        plt.figure(figsize=(7, 7))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.plot(model_n, rel, c='b', label='Fraction of concepts with relevant pairs')
        plt.title(f'Relevant pairs fraction x number of SAE models trained\nAlpha = {alpha}, Scaling factor = {scaling_factor}')
        
        num_latents = int(192*scaling_factor)
        perc_dead_neurons = (mean_dead_neurons / num_latents) * 100
        total_mean_cos_sim = np.asarray(cos_sims).mean()

        model_stats_text = (
            f'Mean model sparsity: {mean_sparsity*100:.4f}%\n'
            f'Mean model dead neurons: {perc_dead_neurons:.2f}% ({mean_dead_neurons:.2f}/{num_latents})\n'
            f'Total mean cosine similarity: {total_mean_cos_sim:.4f}\n'
            f'Mean MSE: {mean_mse:.4f}'
        )
        
        plt.subplots_adjust(bottom=0.25)

        plt.figtext(
            0.5, 0.20,
            model_stats_text,
            ha='center', va='top',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='lightgray')
        )
       
        plt.grid()
        plt.savefig(f'stats/sae/good_pairs[alpha={alpha},scale={scaling_factor}].png', bbox_inches='tight')

    else:
        rel = {s: [] for s in scaling_factors}
        cos_sims = {s: [] for s in scaling_factors}
        full_model_stats_text = ['Statistics per model scale:']

        for s in scaling_factors:
            for n in model_n:
                result = full_results[(s, n, alpha)]
                relevancy = result['relevant_pairs_fraction']
                rel[s].append(relevancy)

                cos_sim = result['mean_cos_sim']
                cos_sims[s].append(cos_sim)

            mean_mse = model_results[(s, alpha)]['mean_mse']
            mean_sparsity = model_results[(s, alpha)]['mean_sparsity']
            mean_dead_neurons = model_results[(s, alpha)]['mean_dead_neurons']

            num_latents = int(192*s)
            perc_dead_neurons = (mean_dead_neurons / num_latents) * 100
            total_mean_cos_sim = np.asarray(cos_sims[s]).mean()

            model_stats_text = f'Scale {s} |  Sparsity: {mean_sparsity*100:.2f}%  |  Dead Neurons: {perc_dead_neurons:.2f}%  |  MSE: {mean_mse:.4f}\n'
            full_model_stats_text.append(model_stats_text)
            
        final_text = '\n'.join(full_model_stats_text)
        
        plt.figure(figsize=(7, 7))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        
        plt.subplots_adjust(bottom=0.30)
        plt.figtext(
            0.5, 0.22,
            final_text,
            ha='center',
            va='top',            
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='lightgray')
        )

        colors = plt.cm.tab10.colors
        for idx, s in enumerate(scaling_factors):
            plt.scatter(model_n, rel[s], color=colors[idx%len(colors)], label=f'Scaling factor = {s}', s=10 + (25 * np.log2(s)), alpha=0.8)
        plt.title(f'Relevant pairs fraction x number of SAE models trained\nAlpha = {alpha}, Scaling factors = {[s for s in scaling_factors]}')
        plt.grid()
        plt.legend(loc='best')
        plt.savefig(f'stats/sae/good_pairs[alpha={alpha}].png', bbox_inches='tight')
