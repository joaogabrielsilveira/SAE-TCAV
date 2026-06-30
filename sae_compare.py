from sae import SAE, train_sae_model
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from matplotlib.ticker import PercentFormatter
from tabpfn_model import load_or_extract_embeddings, scale_embeddings, temporal_test_subsplits, fit_dr_tabpfn, walkforward_evaluate_tabpfn, TabPFNClassifier, TabPFNEvalConfig, EmbeddingExtractConfig

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

def overlap_matrix(sae_i: dict[str], sae_j: dict[str]) -> np.ndarray:
    high_i, high_j = sae_i['high_activation_matrix'], sae_j['high_activation_matrix']
    # nesta matriz de interseção, cada elemento (a, b) conta quantas vezes o conceito a de i ativou junto do conceito j de b
    intersection = np.dot(high_i.T, high_j)

    # essses vetores contam quantas vezes cada conceito ativou individualmente
    sum_i = high_i.sum(axis=0).reshape(-1, 1)
    sum_j = high_j.sum(axis=0).reshape(1, -1)

    union = sum_i + sum_j - intersection

    with np.errstate(divide='ignore', invalid='ignore'):
        matrix = np.where(union > 0, intersection / union, 0.0)
    
    # print(f'overlap_matrix: {matrix.shape}')
    return matrix

def train_all_saes(num_models: int, embs: np.ndarray, alpha: float=1e-1, scaling_factor: float=1.5, model_type: str='ReLU', k:int=16, k_aux:int=64) -> list[dict[str]]:
    sae_list = []
    for i in range(num_models):

        if i == 0:
            # primeira seed fixada igual à usada na análise do pipeline para ter uma base
            current_seed = 42
        else:
            current_seed = 10 * (i ** 2) + 50 * i + 75
    
        sae = train_sae_model(inputs=torch.tensor(embs), type=model_type, alpha=alpha, scaling_factor=scaling_factor, save_data=False, use_decoder_bias=True, use_cache=False, rng_seed=current_seed, epochs=1000, k=k, k_aux=k_aux)

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
def get_concepts_matching(sae_i: dict[str], sae_j: dict[str], pair_criteria: str='cos_sim'):
    if sae_i['idx'] == sae_j['idx']:
        return None
    
    cos_sim_matrix = cosine_similarity_matrix(sae_i, sae_j)

    if pair_criteria == 'cos_sim':
        pairing_matrix = cos_sim_matrix

    elif pair_criteria == 'overlap':
        ov_matrix = overlap_matrix(sae_i, sae_j)
        pairing_matrix = ov_matrix

    else:
        raise RuntimeError('Invalid pairing criteira, try \'cos_sim\' or \'overlap\'')

    rows_idx, cols_idx = linear_sum_assignment(pairing_matrix, maximize=True)

    results = []
    for i, j in zip(rows_idx, cols_idx):
        if pair_criteria == 'cos_sim':
            overlap = get_overlap(sae_i=sae_i, idx_i=i, sae_j=sae_j, idx_j=j)
        else:
            overlap = ov_matrix[i][j]
        results.append({
            'sae_i_idx': sae_i['idx'],
            'sae_j_idx': sae_j['idx'],
            'original_concept': i,
            'best_pair': j,
            'cos_sim': cos_sim_matrix[i][j],
            'overlap': overlap
        })
    
    return pd.DataFrame(results)

def get_all_pairwise_matchings(num_models: int, sae_list: list[dict[str]], pair_criteria: str='cos_sim') -> list[list[pd.DataFrame]]:
    matchings_matrix = [[None for _ in range(num_models)] for _ in range(num_models)]
    for i in range(num_models):
        for j in range(num_models):
            if i == j: continue
            matchings_matrix[i][j] = get_concepts_matching(sae_i=sae_list[i], sae_j=sae_list[j], pair_criteria=pair_criteria)
    
    return matchings_matrix

def get_matching_stats(match_df: pd.DataFrame, pair_criteria: str='cos_sim', relevant_pair_threshold: float=0.7):
    mean_cos_sim = match_df['cos_sim'].mean()
    mean_overlap = match_df['overlap'].mean()
    if pair_criteria == 'cos_sim':
        match_df['is_relevant_pair'] = match_df['cos_sim'] > relevant_pair_threshold
    elif pair_criteria == 'overlap':
        match_df['is_relevant_pair'] = match_df['overlap'] > relevant_pair_threshold
    else:
        raise RuntimeError('Invalid pairing criteira, try \'cos_sim\' or \'overlap\'')
    fraction_good_matches = match_df['is_relevant_pair'].mean()
    mean_cos_sim_good_matches = (match_df[match_df['is_relevant_pair'] == True])['cos_sim'].mean()
    mean_overlap_good_matches = (match_df[match_df['is_relevant_pair'] == True])['overlap'].mean()

    if np.isnan(fraction_good_matches) or np.isnan(mean_cos_sim_good_matches):
        fraction_good_matches, mean_cos_sim_good_matches = 0.0, 0.0

    return {
        'relevant_pairs_fraction': fraction_good_matches,
        'mean_cos_sim': mean_cos_sim,
        'mean_cos_sim_relevant_pairs': mean_cos_sim_good_matches,
        'mean_overlap': mean_overlap,
        'mean_overlap_relevant_pairs': mean_overlap_good_matches
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

def run_sae_random_comparison(model_nums: list[int], hyper_params: list[int | float], embs: np.ndarray, scaling_factors: list[float], pair_criteria: str='cos_sim', relevant_pair_threshold: float=0.7, model_type: str='ReLU'):
    stats_dict = {}
    model_stats_dict = {}
    for m in scaling_factors:
        for a in hyper_params:
            if model_type == 'ReLU':
                print(f'Training {max(model_nums)} {model_type} SAEs with Alpha = {a}, Scaling Factor = {m}')
                sae_list = train_all_saes(num_models=max(model_nums), embs=embs, alpha=a, scaling_factor=m, model_type=model_type)
            elif model_type == 'TopK':
                print(f'Training {max(model_nums)} {model_type} SAEs with K = {a}, Scaling Factor = {m}')
                sae_list = train_all_saes(num_models=max(model_nums), embs=embs, k=a, k_aux=4*a, scaling_factor=m, model_type=model_type)
            match_matrix = get_all_pairwise_matchings(num_models=max(model_nums), sae_list=sae_list, pair_criteria=pair_criteria)
            for n in model_nums:
                # print(f'Calculating matching stats for the first {n} models')
                full_df = get_full_run_df(match_matrix=match_matrix, n_models=n)
                run_stats = get_matching_stats(full_df, pair_criteria=pair_criteria, relevant_pair_threshold=relevant_pair_threshold)
                model_stats = get_model_stats(sae_list)
                stats_dict[(m, n, a)] = run_stats
                model_stats_dict[(m, a)] = model_stats
    
    return stats_dict, model_stats_dict, full_df, sae_list

def plot_run_results(full_results: dict[tuple[int, int, int]], model_results: dict[tuple[int, int]], hyper_param: int | float, scaling_factors: list[float], model_n: list[int], pair_criteria: str='cos_sim', concept_info: str='fraction', model_type: str='ReLU'):
    if model_type == 'ReLU':
        hyper = 'Alpha'
    elif model_type == 'TopK':
        hyper = 'K'

    if len(scaling_factors) == 1:
        rel = []
        cos_sims = []
        overlaps = []
        scaling_factor = scaling_factors[0]
        for n in model_n:
            result = full_results[(scaling_factor, n, hyper_param)]
            relevancy = result['relevant_pairs_fraction']
            if concept_info == 'fraction':
                rel.append(relevancy)
            else:
                rel.append(relevancy * (192*scaling_factor))

            cos_sim = result['mean_cos_sim']
            cos_sims.append(cos_sim)

            overlap = result['mean_overlap']
            overlaps.append(overlap)

        mean_mse = model_results[(scaling_factor, hyper_param)]['mean_mse']
        mean_sparsity = model_results[(scaling_factor, hyper_param)]['mean_sparsity']
        mean_dead_neurons = model_results[(scaling_factor, hyper_param)]['mean_dead_neurons']

        plt.figure(figsize=(7, 7))
        if concept_info == 'fraction':
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.plot(model_n, rel, c='b')
        plt.title(f'Relevant pairs (by {pair_criteria}) fraction x number of SAE models trained\n{hyper} = {hyper_param}, Scaling factor = {scaling_factor}')
        
        num_latents = int(192*scaling_factor)
        perc_dead_neurons = (mean_dead_neurons / num_latents) * 100
        total_mean_cos_sim = np.asarray(cos_sims).mean()
        perc_mean_overlap = np.asarray(overlaps).mean() * 100

        model_stats_text = (
            f'Mean model sparsity: {mean_sparsity*100:.4f}%\n'
            f'Mean model dead neurons: {perc_dead_neurons:.2f}% ({mean_dead_neurons:.2f}/{num_latents})\n'
            f'Total mean cosine similarity: {total_mean_cos_sim:.4f}\n'
            f'Total mean overlap: {perc_mean_overlap:.2f}%\n'
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
        print(f'Salvando grafico em stats/sae/{model_type}/good_pairs_{pair_criteria}[{hyper}={hyper_param},scale={scaling_factor}].png')
        plt.savefig(f'stats/sae/{model_type}/good_pairs_{pair_criteria}[{hyper}={hyper_param},scale={scaling_factor}].png', bbox_inches='tight')

    else:
        rel = {s: [] for s in scaling_factors}
        cos_sims = {s: [] for s in scaling_factors}
        overlaps = {s: [] for s in scaling_factors}

        full_model_stats_text = ['Statistics per model scale:']

        for s in scaling_factors:
            num_latents = int(192*s)
            for n in model_n:
                result = full_results[(s, n, hyper_param)]
                relevancy = result['relevant_pairs_fraction']
                if concept_info == 'fraction':
                    rel[s].append(relevancy)
                else:
                    rel[s].append(relevancy * num_latents)

                cos_sim = result['mean_cos_sim']
                cos_sims[s].append(cos_sim)

                overlap = result['mean_overlap']
                overlaps[s].append(overlap)

            mean_mse = model_results[(s, hyper_param)]['mean_mse']
            mean_sparsity = model_results[(s, hyper_param)]['mean_sparsity']
            mean_dead_neurons = model_results[(s, hyper_param)]['mean_dead_neurons']

            perc_dead_neurons = (mean_dead_neurons / num_latents) * 100
            total_mean_cos_sim = np.asarray(cos_sims[s]).mean()
            perc_mean_overlap = np.asarray(overlaps[s]).mean() * 100

            model_stats_text = f'Scale {s} |  Sparsity: {mean_sparsity*100:.2f}%  | Overlap: {perc_mean_overlap:.2f}%  | Cos sim: {total_mean_cos_sim:.4f}  |  Dead Neurons: {perc_dead_neurons:.2f}%  |  MSE: {mean_mse:.4f}\n'
            full_model_stats_text.append(model_stats_text)
            
        final_text = '\n'.join(full_model_stats_text).strip()
        
        plt.figure(figsize=(7, 7))
        if concept_info == 'fraction':
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

        plt.title(f'Relevant pairs (by {pair_criteria}) fraction  x number of SAE models trained\n{hyper} = {hyper_param}, Scaling factors = {[s for s in scaling_factors]}')
        plt.grid()
        plt.legend(loc='best')
        plt.savefig(f'stats/sae/{model_type}/good_pairs_{pair_criteria}[{hyper}={hyper_param}].png', bbox_inches='tight')
    

def run_random_comparison_year_differences(model_nums: list[int], hyper_param: int | float, scaling_factor: float, pair_criteria: str='cos_sim', relevant_pair_threshold: float=0.7, model_type: str='ReLU',
                                            embedding_info: dict[str] = None, years: np.ndarray | np.ndarray = None, feature_cols: list[str] = None):
    
    X_test_df = embedding_info['X_test_df']
    ytd_map = embedding_info['ytd_map']

    years_unique = np.unique(years)
    train_year = years_unique[0]

    X_train_df = X_test_df[X_test_df['year'] == train_year]

    pos_samples = X_train_df[X_train_df['DEATH'] > 0]
    neg_cases = X_train_df[X_train_df['DEATH'] == 0]

    neg_samples = neg_cases.sample(n=len(pos_samples), random_state=42)
    X_train_balanced = pd.concat([pos_samples, neg_samples], axis=0).sample(frac=1, random_state=42)

    y_train_balanced = np.asarray(X_train_balanced['DEATH'] > 0, dtype=np.int32)
    X_train_np = np.asarray(X_train_balanced[feature_cols])
    years_train = years[X_train_balanced.index]

    eval_cfg = TabPFNEvalConfig()
    fit_out = fit_dr_tabpfn(X_train_balanced[feature_cols], y_train_balanced, train_years=years_train, eval_cfg=eval_cfg)
    drift_model: TabPFNClassifier = fit_out["model"]
    model_add_x_device = fit_out["model_add_x_device"]
    example_add_shape = fit_out["example_add_shape"]

    concept_stats = []
    eval_stats = []
    baseline_comparison = []
    baseline_model: SAE = None
    for dist, year in enumerate(years_unique):
        test_year = train_year + dist
        print(f'Extracting stats for training: {train_year}, test: {test_year}')

        X_test_np = X_test_df[X_test_df['year'] == test_year][feature_cols]
        assert(len(X_test_np) == (X_test_df['year'] == test_year).sum())

        years_train = years[X_test_df['year'] == train_year]
        years_test = years[X_test_df['year'] == test_year]

        # wf = walkforward_evaluate_tabpfn(
        #     drift_model=drift_model,
        #     test_rows=X_test_df,
        #     test_years=[test_year],
        #     top_k_events=feature_cols,
        #     train_years=[train_year],
        #     model_add_x_device=model_add_x_device,
        #     batch_size_predict=512,
        #     example_add_shape=example_add_shape,
        #     use_cache=False
        # )

        # results_per_year = wf["results_per_year"]
        # results_df = pd.DataFrame(results_per_year).sort_values("year").drop('y_pred_bin', axis=1).reset_index(drop=True)
        # eval_stats.append(results_df)
        # print(results_df)

        emb_cfg = EmbeddingExtractConfig()
        emb_cfg.use_cache = True
        emb_out = load_or_extract_embeddings(
            model=drift_model,
            X_train_np=X_train_np,
            X_test_np=X_test_np,
            years_train=years_train,
            years_test=years_test,
            year_to_domain_map=ytd_map,
            embeddings_dir='test',
            cfg=emb_cfg,
            device=model_add_x_device,
            example_add_shape=example_add_shape,
        )

        train_emb = emb_out['train_emb_flat'].astype(np.float32)
        test_emb = emb_out['test_emb_flat'].astype(np.float32)

        train_emb_scaled, test_emb_scaled = scale_embeddings(train_emb, test_emb, fit_test=True)

        print(f'Embs: mean={test_emb_scaled.mean()}, Std={test_emb_scaled.std()}')
        res, model_res, full_pair_df, sae_list = run_sae_random_comparison(
            model_nums=model_nums,
            hyper_params=[hyper_param],
            embs=test_emb_scaled, 
            scaling_factors=[scaling_factor], 
            model_type=model_type,
            pair_criteria=pair_criteria,
        )

        if dist == 0:
            # modelo treinado com dados do primeiro ano do período
            baseline_model = sae_list[0]['model']
            model_device = next(baseline_model.parameters()).device

        embs_t = torch.tensor(test_emb_scaled, dtype=torch.float32, device=model_device)
        year_codes = baseline_model.encode(embs_t)
        year_mse = torch.nn.functional.mse_loss(baseline_model.decode(year_codes), embs_t).item()
        dead_neurons = ((year_codes <= 1e-5).all(dim=0).sum().item())
        dead_neurons_perc = dead_neurons / year_codes.shape[1]
        active_per_sample = ((year_codes > 1e-5).to(dtype=torch.float32)).sum(dim=1).mean().item()
        sparsity = ((year_codes <= 1e-5).to(dtype=torch.float32)).mean().item()
        total_latents = embs_t.shape[1] * scaling_factor
        # weights = sae_list[0]['encoder_weights']

        print(f'MSE (SAE: {train_year}, Embeddings: {year}): {year_mse:.4f}')
        baseline_comparison.append({
            'sae_train_year': train_year,
            'sae_codes_year': year,
            'mse': year_mse,
            'dead_neurons': dead_neurons_perc,
            'active_per_sample': active_per_sample,
            'sparsity': sparsity
        })

        full_pair_df['is_relevant_pair'] = full_pair_df[pair_criteria] >= 0.7
        num_surviving_concepts = full_pair_df['is_relevant_pair'].sum()
        mean_surviving_concepts = full_pair_df['is_relevant_pair'].mean()

        original_sae_df = full_pair_df[full_pair_df['sae_i_idx'] == 0].copy()

        stats_by_concept = original_sae_df.groupby('original_concept').agg(
            survival_rate=('is_relevant_pair', 'mean'),
            mean_cos_sim=('cos_sim', 'mean'),
            mean_overlap=('overlap', 'mean')
        ).reset_index().sort_values('survival_rate', ascending=False)

        surviving_concepts = stats_by_concept[stats_by_concept['survival_rate'] > 0.0]

        concept_stats.append(
            {
                'train_year': train_year,
                'test_year': test_year,
                'results': res,
                'model_results': model_res,
                'full_pair_df': full_pair_df,
                'sae_list': sae_list,
                'train_emb': train_emb_scaled,
                'test_emb': test_emb_scaled,
                'test_emb_code': embs_t,
                'stats_by_concept': stats_by_concept,
                'surviving_concepts_first_sae': surviving_concepts,
                'num_surviving_concepts_first_sae': len(surviving_concepts),
                'mean_fraction_surviving_concepts': mean_surviving_concepts,
                'true_retention_rate': mean_surviving_concepts * total_latents / (total_latents - dead_neurons),
            }
        )

        print(f'{(mean_surviving_concepts * 100):.2f}% Surviving concepts found ({num_surviving_concepts}/{len(full_pair_df)} pairs)')
    
    if eval_stats:
        return pd.DataFrame(concept_stats), pd.concat(eval_stats, axis=0), pd.DataFrame(baseline_comparison)
    else:
        return pd.DataFrame(concept_stats), None, pd.DataFrame(baseline_comparison)
        