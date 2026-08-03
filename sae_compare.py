from decision_tree import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tcav import *
from tabpfn_model import fit_dr_tabpfn, TabPFNEvalConfig, make_dist_tensor
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sae import SAE, train_sae_model
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from pickle import dump, load
from typing import Sequence
from tabpfn_model import load_or_extract_embeddings, scale_embeddings, temporal_test_subsplits, fit_dr_tabpfn, walkforward_evaluate_tabpfn, TabPFNClassifier, TabPFNEvalConfig, EmbeddingExtractConfig
from runtime_acceleration import resolve_torch_device
from progress_utils import progress_iter


def activation_threshold(concept: np.ndarray, perc: int = 90):
    pos_act = concept[concept > 0]
    if len(pos_act) == 0:
        return np.inf
    return np.percentile(pos_act, perc)


def high_activation_profiles(
    concepts: np.ndarray,
    percentiles: Sequence[int] = (70, 80, 90),
) -> dict[int, dict[str, np.ndarray]]:
    """Return positive-only thresholds and strict masks at each percentile."""

    values = np.asarray(concepts)
    if values.ndim != 2:
        raise ValueError("concepts must be a two-dimensional array")
    normalized = tuple(int(percentile) for percentile in percentiles)
    if not normalized:
        raise ValueError("percentiles cannot be empty")
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError("percentiles must be ordered and unique")
    if any(not 0 <= percentile <= 100 for percentile in normalized):
        raise ValueError("percentiles must lie in [0, 100]")

    thresholds = np.full((len(normalized), values.shape[1]), np.inf, dtype=float)
    for factor in range(values.shape[1]):
        positive = values[:, factor][values[:, factor] > 0]
        if len(positive):
            thresholds[:, factor] = np.percentile(positive, normalized)
    return {
        percentile: {
            "masks": np.asarray(values > thresholds[index], dtype=bool),
            "thresholds": thresholds[index].copy(),
        }
        for index, percentile in enumerate(normalized)
    }


def high_activation_matrix(concepts: np.ndarray, perc: int = 90):
    """Compatibility adapter for legacy single-percentile callers."""

    return high_activation_profiles(concepts, (int(perc),))[int(perc)]["masks"]


def encode_sae(
    sae_or_run,
    embeddings: np.ndarray,
    *,
    device: str = "auto",
    batch_size: int | None = None,
) -> np.ndarray:
    """Encode shared records in bounded batches on the requested device."""

    model = sae_or_run.get('model') if isinstance(
        sae_or_run, dict) else sae_or_run
    model_type = sae_or_run.get('model_type') if isinstance(
        sae_or_run, dict) else None
    if model_type not in (None, 'ReLU', 'TopK'):
        raise ValueError(f'Unsupported SAE model type: {model_type}')
    values = np.asarray(embeddings)
    if values.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if batch_size is None:
        batch_size = max(len(values), 1)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    target_device = resolve_torch_device(device)
    original_device = next(model.parameters()).device
    was_training = model.training
    encoded_batches: list[np.ndarray] = []
    model.to(target_device)
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                batch = torch.as_tensor(
                    values[start: start + batch_size],
                    dtype=torch.float32,
                    device=target_device,
                )
                encoded = model.encode(batch)
                if isinstance(encoded, tuple):
                    encoded = encoded[1]
                encoded_batches.append(encoded.cpu().numpy())
    finally:
        model.to(original_device)
        model.train(was_training)
    if not encoded_batches:
        return np.empty((0, model.num_latents), dtype=np.float32)
    return np.concatenate(encoded_batches, axis=0)


def encode_sae_runs(
    sae_runs: list[dict],
    embeddings: np.ndarray,
    *,
    device: str = "auto",
    batch_size: int | None = None,
    show_progress: bool = False,
) -> dict[int, np.ndarray]:
    """Encode identical records through every configured SAE run."""

    return {
        int(run['idx']): encode_sae(
            run,
            embeddings,
            device=device,
            batch_size=batch_size,
        )
        for run in progress_iter(
            sae_runs,
            enabled=show_progress,
            desc="Encoding SAE runs",
            total=len(sae_runs),
            unit="run",
        )
    }


def max_activation_overlap(sae_i: dict[str], concept: int, sae_j: dict[str], perc: int = 90):
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


def max_cosine_similarity(sae_i: dict[str], concept: int, sae_j: dict[str]):
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


def train_all_saes(num_models: int, embs: np.ndarray, alpha: float = 1e-1,
                   scaling_factor: float = 1.5, model_type: str = 'ReLU',
                   k: int = 16, k_aux: int = 64,
                   universal_embs: np.ndarray = None,
                   seeds: Sequence[int] | None = None,
                   epochs: int = 1000,
                   learning_rate: float = 1e-3,
                   weight_decay: float = 0.0,
                   device: str = "auto",
                   encoding_batch_size: int = 4096,
                   show_progress: bool = False) -> list[dict[str]]:
    if seeds is not None and len(seeds) != num_models:
        raise ValueError("seeds must contain exactly num_models entries")
    if encoding_batch_size < 1:
        raise ValueError("encoding_batch_size must be positive")

    sae_list = []
    for i in progress_iter(
        range(num_models),
        enabled=show_progress,
        desc="Training SAE runs",
        total=num_models,
        unit="run",
    ):

        if seeds is not None:
            current_seed = int(seeds[i])
        elif i == 0:
            # primeira seed fixada igual à usada na análise do pipeline para ter uma base
            current_seed = 42
        else:
            current_seed = 10 * (i ** 2) + 50 * i + 75

        sae = train_sae_model(
            inputs=torch.tensor(embs),
            type=model_type,
            alpha=alpha,
            scaling_factor=scaling_factor,
            save_data=False,
            use_decoder_bias=True,
            use_cache=False,
            rng_seed=current_seed,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            k=k,
            k_aux=k_aux,
            device=device,
            show_progress=show_progress,
            progress_desc=f"SAE {i + 1}/{num_models} epochs",
        )
        run_for_encoding = {"model": sae, "model_type": model_type}
        codes = encode_sae(
            run_for_encoding,
            embs,
            device=device,
            batch_size=encoding_batch_size,
        )
        codes_universal = (
            encode_sae(
                run_for_encoding,
                universal_embs,
                device=device,
                batch_size=encoding_batch_size,
            )
            if universal_embs is not None
            else None
        )

        if model_type == 'ReLU':
            weights = sae.encoder.weight.cpu().detach().numpy()

        elif model_type == 'TopK':
            weights = sae.decoder.weight.T.cpu().detach().numpy()

        sparsity = (codes <= 1e-5).mean()
        dead_neurons = np.all(codes <= 1e-5, axis=0).sum().item()
        reconstruction = sae.decode(torch.tensor(codes))
        mse = torch.nn.functional.mse_loss(reconstruction, torch.tensor(embs))

        # matriz binária. cada (i, j) representa se o conceito j teve ativação
        # alta (acima do percentil perc de ativações positivas) na amostra i
        high_act_matrix = high_activation_matrix(
            codes if codes_universal is None else codes_universal, perc=90)

        sae_list.append({
            'idx': i,
            'run_id': f'sae_{i}',
            'seed': current_seed,
            'model_type': model_type,
            'model': sae,
            'mse': mse,
            'encoded_embs': codes,
            'sparsity_level': sparsity,
            'encoder_weights': weights,
            # Explicit name for new code; legacy key remains unchanged.
            'decoder_directions': weights,
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


def get_concepts_matching(sae_i: dict[str], sae_j: dict[str], pair_criteria: str = 'cos_sim', allow_repeat: bool = False):
    if not allow_repeat and sae_i['idx'] == sae_j['idx']:
        return None

    cos_sim_matrix = cosine_similarity_matrix(sae_i, sae_j)

    if pair_criteria == 'cos_sim':
        pairing_matrix = cos_sim_matrix

    elif pair_criteria == 'overlap':
        ov_matrix = overlap_matrix(sae_i, sae_j)
        pairing_matrix = ov_matrix

    else:
        raise RuntimeError(
            'Invalid pairing criteira, try \'cos_sim\' or \'overlap\'')

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


def get_all_pairwise_matchings(num_models: int, sae_list: list[dict[str]], pair_criteria: str = 'cos_sim') -> list[list[pd.DataFrame]]:
    matchings_matrix = [[None for _ in range(
        num_models)] for _ in range(num_models)]
    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                continue
            matchings_matrix[i][j] = get_concepts_matching(
                sae_i=sae_list[i], sae_j=sae_list[j], pair_criteria=pair_criteria)

    return matchings_matrix


def get_matching_stats(match_df: pd.DataFrame, pair_criteria: str = 'cos_sim', relevant_pair_threshold: float = 0.7):
    mean_cos_sim = match_df['cos_sim'].mean()
    mean_overlap = match_df['overlap'].mean()
    if pair_criteria == 'cos_sim':
        match_df['is_relevant_pair'] = match_df['cos_sim'] > relevant_pair_threshold
    elif pair_criteria == 'overlap':
        match_df['is_relevant_pair'] = match_df['overlap'] > relevant_pair_threshold
    else:
        raise RuntimeError(
            'Invalid pairing criteira, try \'cos_sim\' or \'overlap\'')
    fraction_good_matches = match_df['is_relevant_pair'].mean()
    mean_cos_sim_good_matches = (match_df[match_df['is_relevant_pair'] == True])[
        'cos_sim'].mean()
    mean_overlap_good_matches = (match_df[match_df['is_relevant_pair'] == True])[
        'overlap'].mean()

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
    flat_matrix = [match_matrix[i][j]
                   for i in range(n_models) for j in range(n_models)]
    return pd.concat(flat_matrix, axis=0)


def run_sae_random_comparison(model_nums: list[int], hyper_params: list[int | float], embs: np.ndarray, scaling_factors: list[float], pair_criteria: str = 'cos_sim', relevant_pair_threshold: float = 0.7, model_type: str = 'ReLU'):
    stats_dict = {}
    model_stats_dict = {}
    for m in scaling_factors:
        for a in hyper_params:
            if model_type == 'ReLU':
                print(
                    f'Training {max(model_nums)} {model_type} SAEs with Alpha = {a}, Scaling Factor = {m}')
                sae_list = train_all_saes(num_models=max(
                    model_nums), embs=embs, alpha=a, scaling_factor=m, model_type=model_type)
            elif model_type == 'TopK':
                print(
                    f'Training {max(model_nums)} {model_type} SAEs with K = {a}, Scaling Factor = {m}')
                sae_list = train_all_saes(num_models=max(
                    model_nums), embs=embs, k=a, k_aux=4*a, scaling_factor=m, model_type=model_type)
            match_matrix = get_all_pairwise_matchings(num_models=max(
                model_nums), sae_list=sae_list, pair_criteria=pair_criteria)
            for n in model_nums:
                # print(f'Calculating matching stats for the first {n} models')
                full_df = get_full_run_df(
                    match_matrix=match_matrix, n_models=n)
                run_stats = get_matching_stats(
                    full_df, pair_criteria=pair_criteria, relevant_pair_threshold=relevant_pair_threshold)
                model_stats = get_model_stats(sae_list)
                stats_dict[(m, n, a)] = run_stats
                model_stats_dict[(m, a)] = model_stats

    return stats_dict, model_stats_dict, full_df, sae_list


def plot_run_results(full_results: dict[tuple[int, int, int]], model_results: dict[tuple[int, int]], hyper_param: int | float, scaling_factors: list[float], model_n: list[int], pair_criteria: str = 'cos_sim', concept_info: str = 'fraction', model_type: str = 'ReLU'):
    if model_type == 'ReLU':
        hyper = 'Alpha'
    elif model_type == 'TopK':
        hyper = 'K'

    plot_dir = Path('stats/sae') / model_type
    plot_dir.mkdir(parents=True, exist_ok=True)

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
        mean_sparsity = model_results[(
            scaling_factor, hyper_param)]['mean_sparsity']
        mean_dead_neurons = model_results[(
            scaling_factor, hyper_param)]['mean_dead_neurons']

        plt.figure(figsize=(7, 7))
        if concept_info == 'fraction':
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.plot(model_n, rel, c='b')
        plt.title(
            f'Relevant pairs (by {pair_criteria}) fraction x number of SAE models trained\n{hyper} = {hyper_param}, Scaling factor = {scaling_factor}')

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
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='#f8f9fa', edgecolor='lightgray')
        )

        plt.grid()
        plot_path = plot_dir / (
            f'good_pairs_{pair_criteria}'
            f'[{hyper}={hyper_param},scale={scaling_factor}].png'
        )
        print(f'Salvando grafico em {plot_path}')
        plt.savefig(plot_path, bbox_inches='tight')

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
            mean_dead_neurons = model_results[(
                s, hyper_param)]['mean_dead_neurons']

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
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='#f8f9fa', edgecolor='lightgray')
        )

        colors = plt.cm.tab10.colors
        for idx, s in enumerate(scaling_factors):
            plt.scatter(model_n, rel[s], color=colors[idx % len(
                colors)], label=f'Scaling factor = {s}', s=10 + (25 * np.log2(s)), alpha=0.8)

        plt.title(
            f'Relevant pairs (by {pair_criteria}) fraction  x number of SAE models trained\n{hyper} = {hyper_param}, Scaling factors = {[s for s in scaling_factors]}')
        plt.grid()
        plt.legend(loc='best')
        plot_path = plot_dir / (
            f'good_pairs_{pair_criteria}[{hyper}={hyper_param}].png'
        )
        print(f'Salvando grafico em {plot_path}')
        plt.savefig(plot_path, bbox_inches='tight')


def run_random_comparison_year_differences(model_nums: list[int], hyper_param: int | float, scaling_factor: float, pair_criteria: str = 'cos_sim', relevant_pair_threshold: float = 0.7, model_type: str = 'ReLU',
                                           embedding_info: dict[str] = None, years: np.ndarray | np.ndarray = None, feature_cols: list[str] = None):

    X_test_df = embedding_info['X_test_df']
    ytd_map = embedding_info['ytd_map']

    years_unique = np.unique(years)
    train_year = years_unique[0]

    X_train_df = X_test_df[X_test_df['year'] == train_year]

    pos_samples = X_train_df[X_train_df['DEATH'] > 0]
    neg_cases = X_train_df[X_train_df['DEATH'] == 0]

    neg_samples = neg_cases.sample(n=len(pos_samples), random_state=42)
    X_train_balanced = pd.concat(
        [pos_samples, neg_samples], axis=0).sample(frac=1, random_state=42)

    y_train_balanced = np.asarray(
        X_train_balanced['DEATH'] > 0, dtype=np.int32)
    X_train_np = np.asarray(X_train_balanced[feature_cols])
    years_train = years[X_train_balanced.index]

    eval_cfg = TabPFNEvalConfig()
    fit_out = fit_dr_tabpfn(
        X_train_balanced[feature_cols], y_train_balanced, train_years=years_train, eval_cfg=eval_cfg)
    drift_model: TabPFNClassifier = fit_out["model"]
    model_add_x_device = fit_out["model_add_x_device"]
    example_add_shape = fit_out["example_add_shape"]

    concept_stats = []
    eval_stats = []
    baseline_comparison = []
    baseline_model: SAE = None
    for dist, year in enumerate(years_unique):
        test_year = train_year + dist
        print(
            f'Extracting stats for training: {train_year}, test: {test_year}')

        X_test_np = X_test_df[X_test_df['year'] == test_year][feature_cols]
        assert (len(X_test_np) == (X_test_df['year'] == test_year).sum())

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

        train_emb_scaled, test_emb_scaled = scale_embeddings(
            train_emb, test_emb, fit_test=True)

        print(
            f'Embs: mean={test_emb_scaled.mean()}, Std={test_emb_scaled.std()}')
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

        embs_t = torch.tensor(
            test_emb_scaled, dtype=torch.float32, device=model_device)
        year_codes = baseline_model.encode(embs_t)
        year_mse = torch.nn.functional.mse_loss(
            baseline_model.decode(year_codes), embs_t).item()
        dead_neurons = ((year_codes <= 1e-5).all(dim=0).sum().item())
        dead_neurons_perc = dead_neurons / year_codes.shape[1]
        active_per_sample = (
            (year_codes > 1e-5).to(dtype=torch.float32)).sum(dim=1).mean().item()
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


def UNUSED_run_concept_drift_test(model_info: dict[str], X_test_df: pd.DataFrame, years_test: np.ndarray, features: list[str], scaler: StandardScaler, alpha: float = 1.0, num_models: int = 15, match_method: str = 'hungarian'):
    drift_model: TabPFNClassifier = model_info['model']
    ytd_map = model_info['ytd_map']
    cfg = model_info['cfg']
    cfg.use_cache = True
    example_add_shape = model_info['example_add_shape']
    device = model_info['device']

    years = np.unique(years_test)
    sae_lists = {y: [] for y in years}

    X_test_np = np.asarray(X_test_df[features], dtype=np.float32)
    years_test_np = years_test
    full_emb_out = load_or_extract_embeddings(
        model=drift_model,
        X_train_np=None,
        X_test_np=X_test_np,
        years_train=None,
        years_test=years_test_np,
        year_to_domain_map=ytd_map,
        cfg=cfg,
        example_add_shape=example_add_shape,
        device=device,
        embeddings_dir='test'
    )
    full_emb_np = np.asarray(full_emb_out['test_emb_flat'], dtype=np.float32)
    full_emb_np_scaled = scaler.transform(full_emb_np)

    for year in years:
        X_test_year = X_test_df[X_test_df['year'] == year][features]
        X_test_np = np.asarray(X_test_year, dtype=np.float32)
        years_test_np = np.asarray(
            years_test[X_test_df['year'] == year], dtype=np.int32)

        assert (len(X_test_np) == len(years_test_np))

        emb_out_year = load_or_extract_embeddings(
            model=drift_model,
            X_train_np=None,
            X_test_np=X_test_np,
            years_train=None,
            years_test=years_test_np,
            year_to_domain_map=ytd_map,
            cfg=cfg,
            example_add_shape=example_add_shape,
            device=device,
            embeddings_dir='test'
        )
        embs_year_np = np.asarray(
            emb_out_year['test_emb_flat'], dtype=np.float32)
        embs_year_scaled = scaler.transform(embs_year_np)

        sae_lists[year] = train_all_saes(num_models, embs_year_scaled, alpha=alpha,
                                         scaling_factor=1.5, model_type='ReLU', universal_embs=full_emb_np_scaled)
    rows = []
    for i, year_i in enumerate(years):
        for j, year_j in enumerate(years[i:]):
            sae_list_i = sae_lists[year_i]
            sae_list_j = sae_lists[year_j]

            assert (len(sae_list_i) == len(sae_list_j))

            results = []
            print(f'Comparing {year_i} and {year_j}')
            for s_i in range(len(sae_list_i)):
                for s_j in range(len(sae_list_j)):

                    if s_i == s_j:
                        continue

                    model_i = sae_list_i[s_i]
                    model_j = sae_list_j[s_j]

                    codes_i = model_i['model'].encode(torch.tensor(
                        full_emb_np_scaled, dtype=torch.float32))
                    codes_j = model_j['model'].encode(torch.tensor(
                        full_emb_np_scaled, dtype=torch.float32))

                    is_active_i = ((codes_i > 1e-5).sum(dim=0)
                                   > 0).cpu().numpy()
                    is_active_j = ((codes_j > 1e-5).sum(dim=0)
                                   > 0).cpu().numpy()

                    cos_sim_matrix = cosine_similarity_matrix(model_i, model_j)
                    masked_cos_sim_matrix = cos_sim_matrix.copy()
                    masked_cos_sim_matrix[:, ~is_active_j] = -np.inf

                    if match_method == 'hungarian':
                        rows_idx, cols_idx = linear_sum_assignment(
                            masked_cos_sim_matrix, maximize=True)
                    elif match_method == 'maxcos':
                        rows_idx = range(masked_cos_sim_matrix.shape[0])
                        cols_idx = np.argmax(masked_cos_sim_matrix, axis=1)

                    for a, b in zip(rows_idx, cols_idx):
                        overlap = get_overlap(
                            sae_i=model_i, idx_i=a, sae_j=model_j, idx_j=b)

                        results.append({
                            'seed_idx_i': model_i['idx'],
                            'seed_idx_j': model_j['idx'],
                            'year_sae_i': year_i,
                            'year_sae_j': year_j,
                            'original_concept': a,
                            'best_pair': b,
                            'cos_sim': cos_sim_matrix[a][b],
                            'overlap': overlap,
                            'is_active_base': is_active_i[a],
                            'is_active_pair': is_active_j[b]
                        })

            year_pair_stats = pd.DataFrame(results)

            both_active_mask = (year_pair_stats['is_active_base']) & (
                year_pair_stats['is_active_pair'])
            total_active_base = year_pair_stats['is_active_base'].sum()
            survivors = ((year_pair_stats['cos_sim'] > 0.7) &
                         (year_pair_stats['is_active_base'] == True) &
                         (year_pair_stats['is_active_pair'] == True)).sum()

            true_survival_rate = survivors / total_active_base if total_active_base > 0 else 0.0
            print(
                f'Sobreviventes: {survivors}, conceitos ativos: {total_active_base}')

            true_mean_cos_sim = year_pair_stats[both_active_mask]['cos_sim'].mean(
            )

            true_overlap = year_pair_stats[both_active_mask]['overlap'].mean()

            year_stats_acc = pd.DataFrame([{
                # 'mean_cos_sim': year_pair_stats['cos_sim'].mean(),
                # 'mean_overlap': year_pair_stats['overlap'].mean(),
                'mean_cos_sim': true_mean_cos_sim,
                'mean_overlap': true_overlap,
                'survival_rate': true_survival_rate,
                # 'survival_rate': ((year_pair_stats['cos_sim'] > 0.7).mean()),
                'year_a': year_i,
                'year_b': year_j
            }])

            rows.append(year_stats_acc)

    full_stats = pd.concat(rows, axis=0)
    full_stats['dist'] = full_stats['year_b'] - full_stats['year_a']
    dist_stats = full_stats.groupby(by='dist').agg(
        mean_cos_sim=('mean_cos_sim', 'mean'),
        std_cos_sim=('mean_cos_sim', 'std'),
        mean_overlap=('mean_overlap', 'mean'),
        std_overlap=('mean_overlap', 'std'),
        mean_survival_rate=('survival_rate', 'mean'),
        std_survival_rate=('survival_rate', 'std')
    )
    with open(f'stats/concepts/temporal_distance_stats[a={alpha}].pkl', 'wb') as out:
        from pickle import dump
        dump(dist_stats, out)
    with open(f'stats/concepts/full_temp_dist_df[a={alpha}].pkl', 'wb') as out:
        from pickle import dump
        dump(full_stats, out)

    return full_stats, dist_stats


def calculate_concept_correlation_drift(X_test_df: pd.DataFrame, test_embs: np.ndarray, test_years_np: np.ndarray, features: list[str], alpha: float = 1.0, undersample: bool = True):
    import warnings
    warnings.filterwarnings(
        'ignore', category=RuntimeWarning, module='numpy.lib.function_base')

    test_embs = torch.tensor(test_embs, dtype=torch.float32)
    unique_years = np.unique(test_years_np)
    year_to_domain_combined = {y: i for i, y in enumerate(unique_years)}
    y_test_np = np.asarray(X_test_df['DEATH'] > 0, dtype=bool)

    results = []
    for reference_year in unique_years:
        reference_year_mask = (
            X_test_df['year'] == reference_year).astype(bool)
        X_reference_df: pd.DataFrame = X_test_df[reference_year_mask][features]
        embs_reference = test_embs[reference_year_mask]
        y_ref = y_test_np[reference_year_mask]
        years_ref = test_years_np[reference_year_mask]

        emb_scaler = StandardScaler()
        data_scaler = StandardScaler()

        embs_reference = torch.tensor(emb_scaler.fit_transform(
            embs_reference.numpy()), dtype=torch.float32)
        X_reference_df = pd.DataFrame(data_scaler.fit_transform(
            X_reference_df), index=X_reference_df.index, columns=X_reference_df.columns)

        baseline_sae = train_sae_model(
            inputs=embs_reference, alpha=1.0, scaling_factor=1.5, type='ReLU', rng_seed=42, save_data=False)
        with torch.no_grad():
            codes_ref, reconstruction_reference = baseline_sae(embs_reference)

        nan_tensor = torch.tensor(
            float('nan'), dtype=codes_ref.dtype, device=codes_ref.device)
        codes_ref_pos = torch.where(codes_ref > 0, codes_ref, nan_tensor)
        medians = torch.nanquantile(
            input=codes_ref_pos, q=torch.tensor(0.5), dim=0)
        above_thresh_mask_reference = (codes_ref >= medians)
        high_activations_reference = torch.sum(
            above_thresh_mask_reference, dim=0)
        high_activations_sum_reference = torch.sum(
            above_thresh_mask_reference * codes_ref, dim=0)
        high_activations_mean_reference = high_activations_sum_reference / \
            torch.clamp(high_activations_reference, 1)

        active_features_mask_reference = (codes_ref > 1e-5).sum(dim=0) > 0
        active_indices_reference = torch.where(
            active_features_mask_reference)[0].cpu().numpy()

        if undersample:
            pos_idx = np.where(y_ref == 1)[0]
            neg_idx = np.where(y_ref == 0)[0]

            if len(pos_idx) > 0 and len(neg_idx) > 0:
                min_size = min(len(pos_idx), len(neg_idx))
                pos_sample = np.random.choice(
                    pos_idx, size=min_size, replace=False)
                neg_sample = np.random.choice(
                    neg_idx, size=min_size, replace=False)
                all_samples = np.concatenate([pos_sample, neg_sample])
                np.random.shuffle(all_samples)

                X_ref_bal = X_reference_df.iloc[all_samples]
                y_ref_bal = y_ref[all_samples]
                years_ref_bal = years_ref[all_samples]
            else:
                X_ref_bal, y_ref_bal, years_ref_bal = X_reference_df, y_ref, years_ref
        else:
            X_ref_bal, y_ref_bal, years_ref_bal = X_reference_df, y_ref, years_ref

        tabpfn_fit_out = fit_dr_tabpfn(
            X_train=X_ref_bal,
            y_train=y_ref_bal,
            train_years=years_ref_bal,
            eval_cfg=TabPFNEvalConfig()
        )

        baseline_tabpfn = tabpfn_fit_out['model']
        model_add_x_device = tabpfn_fit_out['model_add_x_device']
        example_add_shape = tabpfn_fit_out['example_add_shape']
        dist_dom = np.asarray([year_to_domain_combined[int(y)]
                              for y in years_ref], dtype=np.int64)

        significant_df = get_stable_concepts_year(
            year=reference_year, X_np_scaled=np.asarray(
                X_reference_df, dtype=np.float32),
            X_df_scaled=X_reference_df, tabpfn=baseline_tabpfn,
            embs=np.asarray(embs_reference, dtype=np.float32),
            codes=codes_ref, y_np=y_ref, dist_np=dist_dom,
            embs_scaler=emb_scaler, feature_cols=features
        )

        nan_tensor = torch.tensor(
            float('nan'), dtype=codes_ref.dtype, device=codes_ref.device)
        codes_ref_pos = torch.where(codes_ref > 0, codes_ref, nan_tensor)
        medians = torch.nanquantile(
            input=codes_ref_pos, q=torch.tensor(0.5), dim=0)
        above_thresh_mask_reference = (codes_ref >= medians)
        high_activations_reference = torch.sum(
            above_thresh_mask_reference, dim=0)
        high_activations_sum_reference = torch.sum(
            above_thresh_mask_reference * codes_ref, dim=0)
        high_activations_mean_reference = high_activations_sum_reference / \
            torch.clamp(high_activations_reference, 1)

        # active_features_mask_reference = (codes_ref > 1e-5).sum(dim=0) > 0
        # active_indices_reference = torch.where(active_features_mask_reference)[0].cpu().numpy()
        active_indices_reference = significant_df['Factor'].values.tolist()
        active_features_mask_reference = torch.isin(torch.tensor(
            range(codes_ref.shape[1])), torch.tensor(active_indices_reference))

        reference_associations = []
        for feat in range(codes_ref.shape[1]):
            if feat in active_indices_reference:
                corr_matrix = X_reference_df.corrwith(pd.Series(
                    codes_ref[:, feat].cpu().numpy(), index=X_reference_df.index)).fillna(0)
                reference_associations.append(corr_matrix.values)
            else:
                reference_associations.append(
                    np.zeros(X_reference_df.shape[1], dtype=np.float32))
        reference_associations = np.asarray(
            reference_associations, dtype=np.float32)

        for test_year in unique_years:
            if test_year < reference_year:
                continue

            test_year_mask = (X_test_df['year'] == test_year).astype(bool)
            if test_year == reference_year:
                embs_test = embs_reference
                X_tests_df = X_reference_df
                y_test = y_ref
                years_test = years_ref
            else:
                embs_test = test_embs[test_year_mask]
                X_tests_df = X_test_df[test_year_mask][features]
                y_test = y_test_np[test_year_mask]
                years_test = test_years_np[test_year_mask]
                embs_test = torch.tensor(emb_scaler.transform(
                    embs_test.numpy()), dtype=torch.float32)
                X_tests_df = pd.DataFrame(data_scaler.transform(
                    X_tests_df), index=X_tests_df.index, columns=X_tests_df.columns)

            with torch.no_grad():
                codes_test, reconstruction_test = baseline_sae(embs_test)

            above_thresh_mask_test = (codes_test >= medians)
            high_activations_test = torch.sum(above_thresh_mask_test, dim=0)
            high_activations_sum_test = torch.sum(
                above_thresh_mask_test * codes_test, dim=0)
            high_activations_mean_test = high_activations_sum_test / \
                torch.clamp(high_activations_test, 1)

            mean_act_ratio = high_activations_mean_test[active_features_mask_reference] / \
                high_activations_mean_reference[active_features_mask_reference]
            mean_magnitude_ratio = mean_act_ratio.mean().item()

            high_act_ratio = high_activations_test[active_features_mask_reference] / \
                high_activations_reference[active_features_mask_reference]
            num_active_concepts = active_features_mask_reference.sum().item()
            dead_concepts = (high_act_ratio < 0.1).sum(
            ).item() / num_active_concepts
            stable_concepts = ((high_act_ratio >= 0.75) & (
                high_act_ratio <= 1.5)).sum().item() / num_active_concepts
            overused_concepts = (high_act_ratio > 2).sum(
            ).item() / num_active_concepts

            year_mse = torch.nn.functional.mse_loss(
                reconstruction_test, embs_test).item()
            concepts_per_patient = (
                codes_test > 1e-5).float().sum(dim=1).mean().item()

            test_associations = []
            for feat in range(codes_ref.shape[1]):
                if feat in active_indices_reference:
                    corr_matrix = X_tests_df.corrwith(
                        pd.Series(codes_test[:, feat].cpu().numpy(), index=X_tests_df.index)).fillna(0)
                    test_associations.append(corr_matrix.values)
                else:
                    test_associations.append(
                        np.zeros(X_tests_df.shape[1], dtype=np.float32))

            test_associations = np.asarray(test_associations, dtype=np.float32)

            ref_tensor = torch.tensor(
                reference_associations, dtype=torch.float32)
            test_tensor = torch.tensor(test_associations, dtype=torch.float32)

            cos_sim_vector = torch.nn.functional.cosine_similarity(
                ref_tensor[active_indices_reference], test_tensor[active_indices_reference], dim=1)
            mean_cos_sim = cos_sim_vector.mean().item()

            active_features_mask_test = (codes_test > 1e-5).sum(dim=0) > 0
            active_indices_test = torch.where(active_features_mask_test)[
                0].cpu().numpy()

            reference_active = (active_features_mask_reference).sum().item()
            both_active = (active_features_mask_reference &
                           active_features_mask_test).sum().item()

            survival_rate = both_active / reference_active if reference_active > 0 else 0.0

            preds = []
            for start in range(0, len(X_tests_df), 512):
                end = min(start+512, len(X_tests_df))
                dist_dom = np.asarray([year_to_domain_combined[int(y)]
                                      for y in years_test[start:end]], dtype=np.int64)
                dist_dom_t = make_dist_tensor(
                    dist_dom_np=dist_dom,
                    model_add_x_device=model_add_x_device,
                    example_add_shape=example_add_shape,
                )

                y_pred = baseline_tabpfn.predict_proba(X=X_tests_df.iloc[start:end].values.astype(np.float32),
                                                       additional_x={"dist_shift_domain": dist_dom_t})

                preds.append(y_pred)

            y_pred_total = np.vstack(preds)
            y_pred_proba = y_pred_total[:, 1]
            y_pred_bin = np.argmax(y_pred_total, axis=1)

            acc = accuracy_score(y_true=y_test, y_pred=y_pred_bin)
            f1_macro = f1_score(
                y_true=y_test, y_pred=y_pred_bin, average='macro')
            f1_pos = f1_score(y_true=y_test, y_pred=y_pred_bin)
            roc_auc = roc_auc_score(y_true=y_test, y_score=y_pred_proba)

            results.append({
                'train_year': reference_year,
                'test_year': test_year,
                'mse': year_mse,
                'active_per_sample': concepts_per_patient,
                'mean_cos_sim': mean_cos_sim,
                'survival_rate': survival_rate,
                'accuracy': acc,
                'f1_macro': f1_macro,
                'f1_pos': f1_pos,
                'roc_auc_score': roc_auc,
                'dead_concepts': dead_concepts,
                'stable_concepts': stable_concepts,
                'overused_concepts': overused_concepts,
                'high_activation_ratio': mean_magnitude_ratio
            })

    results = pd.DataFrame(results)
    results['dist'] = results['test_year'] - results['train_year']

    results_agg = results.groupby(by='dist').agg(
        mean_cos_sim=('mean_cos_sim', 'mean'),
        std_cos_sim=('mean_cos_sim', 'std'),
        mean_active=('active_per_sample', 'mean'),
        std_active=('active_per_sample', 'std'),
        mean_mse=('mse', 'mean'),
        std_mse=('mse', 'std'),
        mean_survival_rate=('survival_rate', 'mean'),
        std_survival_rate=('survival_rate', 'std'),
        mean_f1_macro=('f1_macro', 'mean'),
        std_f1_macro=('f1_macro', 'std'),
        mean_f1_pos=('f1_pos', 'mean'),
        std_f1_pos=('f1_pos', 'std'),
        mean_accuracy=('accuracy', 'mean'),
        std_accuracy=('accuracy', 'std'),
        mean_rocauc=('roc_auc_score', 'mean'),
        std_rocauc=('roc_auc_score', 'std'),
        mean_stable_concepts=('stable_concepts', 'mean'),
        std_stable_concepts=('stable_concepts', 'std'),
        # mean_dead_concepts=('dead_concepts', 'mean'),
        # std_dead_concepts=('dead_concepts', 'std'),
        # mean_overused_concepts=('overused_concepts', 'mean'),
        # std_overused_concepts=('overused_concepts', 'std'),
        mean_activation_ratio=('high_activation_ratio', 'mean'),
        std_activation_ratio=('high_activation_ratio', 'std')
    )

    cols = results_agg.columns.tolist()
    with open(f'stats/concept_drift_full.pkl', 'wb') as out:
        dump(results, out)
    with open(f'stats/concept_drift.pkl', 'wb') as out:
        dump(results_agg, out)
    pd.set_option('display.max_columns', None)
    display_cols = [col for col in cols if 'std' not in col]
    print(results_agg[display_cols])
    return results, results_agg


def plotar_termometro_drift(df_agg: pd.DataFrame):
    """
    Plota as 4 métricas de Concept Drift num único gráfico normalizado.
    Todas as métricas são escaladas de forma que o seu valor máximo seja 100%.
    Assume que df_agg possui colunas com os prefixos 'mean_' e 'std_' para:
    cos_sim, survival_rate, f1_pos e mse.
    """

    # Garantir que 'dist' existe como coluna a partir do index
    if 'dist' not in df_agg.columns:
        df_agg['dist'] = df_agg.index

    dist = df_agg['dist']

    # Função auxiliar para normalizar as métricas e seus respectivos desvios padrão
    def normalizar(mean_col, std_col):
        max_val = df_agg[mean_col].max()
        if max_val == 0 or pd.isna(max_val):
            return df_agg[mean_col] * 0, df_agg[std_col].fillna(0) * 0

        norm_mean = (df_agg[mean_col] / max_val) * 100
        norm_std = (df_agg[std_col].fillna(0) / max_val) * 100
        return norm_mean, norm_std

    # Normalização das 4 métricas usando o prefixo exigido
    mean_cos, std_cos = normalizar('mean_cos_sim', 'std_cos_sim')
    mean_surv, std_surv = normalizar('mean_survival_rate', 'std_survival_rate')
    mean_f1, std_f1 = normalizar('mean_f1_pos', 'std_f1_pos')
    mean_mse, std_mse = normalizar('mean_mse', 'std_mse')

    # Criar a figura
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Eixo Único (Esquerda): Proporção do Máximo (0-100%)
    ax1.set_xlabel(r'Distância Temporal ($\Delta t$ em anos)',
                   fontsize=12, fontweight='bold')
    ax1.set_ylabel('Proporção do Valor Máximo (%)', fontsize=12,
                   color='black', fontweight='bold')

    # 1. Estabilidade Semântica (Cos Sim)
    p1, = ax1.plot(dist, mean_cos, color='tab:blue', linewidth=2.5,
                   marker='o', label='Estabilidade Semântica (Cos)')
    # ax1.fill_between(dist, mean_cos - std_cos, mean_cos + std_cos, color='tab:blue', alpha=0.15)

    # 2. Sobrevivência Funcional
    p2, = ax1.plot(dist, mean_surv, color='tab:green', linewidth=2.5,
                   marker='s', linestyle='--', label='Sobrevivência Funcional')
    # ax1.fill_between(dist, mean_surv - std_surv, mean_surv + std_surv, color='tab:green', alpha=0.15)

    # 3. F1-Score (Classe Positiva)
    p3, = ax1.plot(dist, mean_f1, color='tab:purple', linewidth=2.5,
                   marker='D', label='F1-Score (Classe Positiva)')
    # ax1.fill_between(dist, mean_f1 - std_f1, mean_f1 + std_f1, color='tab:purple', alpha=0.15)

    # 4. Erro OOD (MSE)
    p4, = ax1.plot(dist, mean_mse, color='tab:red',
                   linewidth=2.5, marker='^', label='Erro OOD (MSE)')
    # ax1.fill_between(dist, mean_mse - std_mse, mean_mse + std_mse, color='tab:red', alpha=0.15)

    # Colorir os ticks e labels do eixo
    ax1.tick_params(axis='y', labelcolor='black')

    # Ajustar limites do Eixo (0 a 105% para ter uma margem no topo)
    ax1.set_ylim(bottom=0, top=105)

    # Grid focado apenas no eixo X
    ax1.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax1.set_xticks(dist.unique())

    # Consolidar legendas dentro do gráfico (loc='best') para garantir que não sejam cortadas na visualização do seu IDE
    linhas = [p1, p2, p3, p4]
    ax1.legend(linhas, [l.get_label() for l in linhas],
               loc='best', frameon=True, framealpha=0.9, fontsize=10)

    plt.title('Erosão Estrutural vs Desempenho Preditivo (Valores Normalizados)',
              fontsize=14, pad=20, fontweight='bold')

    # Linha de base vertical (dt = 0)
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('graphs/sae_drift_normalizado.png')


def get_stable_concepts_year(year: int, X_df_scaled: pd.DataFrame, X_np_scaled: np.ndarray, tabpfn: TabPFNClassifier, embs: np.ndarray, codes: np.ndarray, y_np: np.ndarray, dist_np: np.ndarray, embs_scaler: StandardScaler, feature_cols: list[str]):
    tree_rules = train_binary_trees(embs, X_np_scaled, feature_cols)
    best_p = max(tree_rules.keys(), key=lambda p: len(tree_rules[p]))
    rules_df = pd.DataFrame(tree_rules[best_p])
    high_quantile = 1.0 - (best_p / 100.0)

    gradients_path = Path(GRADS_FILE)
    year_gradients_path = gradients_path.with_name(
        f'{gradients_path.stem}_{year}{gradients_path.suffix}'
    )
    grads = get_model_gradients(
        model=tabpfn,
        X=X_np_scaled,
        dist_vec=dist_np,
        cache_file=year_gradients_path,
    )
    cavs = train_cavs_from_rules(rules_per_percentile=tree_rules[best_p], X_cav_train_df=X_df_scaled, cav_train_emb=embs,
                                 cav_train_emb_encoded=codes, y_cav_train=y_np,
                                 feature_cols=feature_cols, emb_scaler=embs_scaler, high_quantile=high_quantile, min_pos_samples=50, random_state=42)
    tcav_scores = get_tcav_scores(cavs=cavs.values(), grads=grads)

    significant_concepts, significant_df = get_significant_concepts(
        cavs=cavs, tcav_scores=tcav_scores, best_rules=tree_rules[best_p],
        grads=grads, embs=embs, scaler=embs_scaler
    )

    return significant_df
