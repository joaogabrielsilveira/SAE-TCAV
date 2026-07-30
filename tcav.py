import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from typing_extensions import Any
from pickle import dump, load
import os
from filepaths import get_env_path
from tabpfn import TabPFNClassifier
from tabpfn_model import make_dist_tensor
from torch.func import grad, vmap
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy import stats
import warnings
from pathlib import Path
from runtime_acceleration import resolve_torch_device
from progress_utils import progress_iter

CAVS_FILE = get_env_path('models/tcav/cavs.pkl')
GRADS_FILE = get_env_path('models/tcav/grads.pkl')

def get_model_gradients(model: TabPFNClassifier, dist_vec: np.ndarray, X: np.ndarray,
                        cache_file: str | os.PathLike | None = None,
                        batch_size: int = 128,
                        device: str = "auto",
                        show_progress: bool = False,
                        use_cache: bool = True) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    gradient_device = resolve_torch_device(device)
    gradients_file = str(cache_file) if cache_file is not None else GRADS_FILE

    if use_cache and os.path.exists(gradients_file):
        with open(gradients_file, 'rb') as f:
            grads = load(f)
            print(f'Carregando grads de {gradients_file}')
            if cache_file is not None and np.asarray(grads).shape[0] != X.shape[0]:
                raise ValueError('Cached gradient count does not match requested records')
            return grads

    model_decode_layer = model.model_processed_.decoder_dict['standard']
    original_device = next(model_decode_layer.parameters()).device
    model_decode_layer.to(gradient_device)
    gradients = []
    try:
        batch_starts = range(0, X.shape[0], batch_size)
        for s in progress_iter(
            batch_starts,
            enabled=show_progress,
            desc="Computing TCAV gradients",
            total=len(batch_starts),
            unit="batch",
            leave=False,
        ):
            e = min(s + batch_size, X.shape[0])

            x_batch = X[s:e].astype(np.float32)
            dist_batch = dist_vec[s:e]

            dist_t = torch.tensor(
                dist_batch, dtype=torch.long, device='cpu'
            ).reshape(-1, 1, 1)

            with torch.enable_grad():
                emb = model.get_embeddings(
                    x_batch,
                    additional_x={"dist_shift_domain": dist_t},
                )
                if emb.ndim == 3 and emb.shape[0] == 1:
                    emb = emb[0]
                elif emb.ndim == 3 and emb.shape[1] == 1:
                    emb = emb.squeeze(1)

                emb_in = emb.clone().detach().to(
                    gradient_device, dtype=torch.float32
                ).requires_grad_(True)

                def single_pass(x_p):
                    out = model_decode_layer(x_p).unsqueeze(0)
                    if out.ndim == 3:
                        out = out[0]
                    return out[0, 1]

                batch_grad = vmap(grad(single_pass))(emb_in)
                gradients.append(batch_grad.detach().cpu().numpy())
    finally:
        model_decode_layer.to(original_device)

    if use_cache:
        Path(gradients_file).parent.mkdir(parents=True, exist_ok=True)
        with open(gradients_file, 'wb') as f:
            dump(np.vstack(gradients), f)
    return np.vstack(gradients)

def calculate_tcav_score(cav: np.ndarray, model_gradients: np.ndarray) -> float:
    prod = np.dot(model_gradients, cav)
    return np.mean(prod > 0)

def get_tcav_scores(cavs: list[dict], grads: np.ndarray) -> dict[int, float]:
    scores = {}
    for cav in cavs:
        idx, v = cav['Factor'], cav['CAV']
        scores[idx] = calculate_tcav_score(v, grads)

    return scores


def compare_tcav_pair(cav_i: np.ndarray, cav_j: np.ndarray,
                      model_gradients: np.ndarray | None = None,
                      neutral_band: float = 0.1) -> dict[str, Any]:
    """Summarize CAV direction and optional TCAV effects for one factor pair."""

    cav_i = np.asarray(cav_i, dtype=float)
    cav_j = np.asarray(cav_j, dtype=float)
    denominator = np.linalg.norm(cav_i) * np.linalg.norm(cav_j)
    cosine = float(np.dot(cav_i, cav_j) / denominator) if denominator else 0.0
    result: dict[str, Any] = {'cav_cosine': cosine}
    if model_gradients is None:
        return result

    score_i = float(calculate_tcav_score(cav_i, model_gradients))
    score_j = float(calculate_tcav_score(cav_j, model_gradients))

    def effect_sign(score: float) -> int:
        if score > 0.5 + neutral_band:
            return 1
        if score < 0.5 - neutral_band:
            return -1
        return 0

    sign_i, sign_j = effect_sign(score_i), effect_sign(score_j)
    result.update({
        'tcav_i': score_i,
        'tcav_j': score_j,
        'tcav_abs_difference': abs(score_i - score_j),
        'tcav_effect_sign_i': sign_i,
        'tcav_effect_sign_j': sign_j,
        'tcav_effect_sign_agreement': sign_i == sign_j,
    })
    return result

def extract_rule_conditions(path: str):
    # print(path)
    conds_list = []
    conditions = path.split(' AND ')
    for condition in conditions:
        condition = condition.strip()
        # print(condition)
        op = ''
        if '>=' in condition:
            feat, thresh = condition.split(' >= ')
            op = '>='
        elif '>' in condition:
            feat, thresh = condition.split(' > ')
            op = '>'
        elif '<=' in condition:
            feat, thresh = condition.split(' <= ')
            op = '<='
        elif '<' in condition:
            feat, thresh = condition.split(' < ')
            op = '<'
        else:
            print(condition)
            input()
            continue
        thresh = float(thresh.strip())
        conds_list.append((feat.strip(), op, thresh))

    # print(conds_list)

    return conds_list

def get_rule_mask(X_df: pd.DataFrame, conditions: list[Any], feature_cols: list[str]) -> np.ndarray:
    mask = np.ones(X_df.shape[0], dtype=bool)


    for cond in conditions:
        feat, op, thresh = cond
        # feat_idx = feature_cols.index(feat)
        if feat not in X_df.columns:
            for col in X_df.columns:
                if feat in col:
                    feat = col
        values = X_df[feat].values
        if op == '>':
            mask = mask & (values > thresh)
        elif op == '>=':
            mask = mask & (values >= thresh)
        elif op == '<':
            mask = mask & (values < thresh)
        elif op == '<=':
            mask = mask & (values <= thresh)

    return mask

def train_cavs_from_rules(rules_per_percentile: dict[str, Any], X_cav_train_df: pd.DataFrame,
                          cav_train_emb: np.ndarray, cav_train_emb_encoded: np.ndarray, y_cav_train: np.ndarray,
                          feature_cols: list[str], emb_scaler: StandardScaler,
                           high_quantile: float=0.5, min_pos_samples: int=50, random_state: int=42) -> dict[Any, Any]:
    rng = np.random.default_rng(random_state)
    cav_dict = {}
    cav_list = []

    for rule in rules_per_percentile:
        factor = rule['Factor']
        path = rule['Rule']

        encoded_activations = cav_train_emb_encoded[:, factor]
        n_pos_encoded_activations = (encoded_activations > 0).sum()
        if n_pos_encoded_activations == 0:
            continue

        conds = extract_rule_conditions(path)
        mask = get_rule_mask(X_cav_train_df, conds, feature_cols)

        # pacientes que seguem a regra
        true_idx = np.where(mask == True)[0]

        # se tiver muito poucos, é ruim de treinar uma regressão, desista
        if len(true_idx) < min_pos_samples:
            continue

        # / não seguem a regra
        false_idx = np.where(mask == False)[0]

        # ativações baixas devem ser consideravelmente abaixo do percentil que gerou a regra
        low_activation_thresh = np.quantile(encoded_activations, max(high_quantile - 0.1, 0.0))

        # idx_pos_low = [encoded_activations[true_idx] <= low_activation_thresh]
        # pacientes que nçao seguem a regra e têm ativação muito baixa
        # print(encoded_activations.shape, false_idx.shape)
        idx_neg_low = false_idx[encoded_activations[false_idx] <= low_activation_thresh]

        # o "alvo", quantidade ideal de negativos, é igual à de positivos (balanceamento de classes)
        n_neg_target = len(true_idx)


        # se possível, pegamops um número equivalente de amostras negativas
        if len(idx_neg_low) >= n_neg_target:
            idx_negative = rng.choice(idx_neg_low, size=n_neg_target, replace=False)

        # caso contrário, havendo o mínimo necessário, pegamos todas
        elif len(idx_neg_low) >= min_pos_samples:
            idx_negative = idx_neg_low

        # se nem o mínimo puder ser alcançado, desista
        else:
            continue

        idx_positive = true_idx

        # cria os vetores combinados de entrada e pred para a regressão
        X = np.vstack([cav_train_emb[idx_positive], cav_train_emb[idx_negative]])
        y = np.hstack([np.ones(len(idx_positive)), np.zeros(len(idx_negative))])

        # treina a regressão com os dados balanceados (simula fronteira que separa entradas que seguem e não seguem a regra)
        # essencialemente, ela aprende como as regras criadas a partir dos conceitos esparsos aparecem nos embeddings originais do modelo
        clf = LogisticRegression(C=0.1, solver='liblinear',penalty='l2', class_weight='balanced',
                                 max_iter=1000, random_state=random_state)
        clf.fit(X, y)

        # o vetor com coeficientes desta regressão, então, aponta para onde essa regra aumenta
        # essa direção é depois comparada com os gradientes do modelo
        cav = clf.coef_[0]
        nrm = np.linalg.norm(cav)
        if (nrm) > 0:
            cav /= nrm

        cav_list.append(cav)
        cav_dict[factor] = {
            'Factor': factor,
            'CAV': cav,
            'clf': clf,
            'Low_thresh': low_activation_thresh,
            'positive_idx': idx_positive,
            'negative_idx': idx_negative,
            'Rule': path
        }

    return cav_dict

def train_cav_from_subset(embs: np.ndarray, idx_pos: np.ndarray, idx_neg: np.ndarray,
                          scaler_emb: StandardScaler, sample_fraction: float=1.0, rng_seed: int=42) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)

    n_pos = len(idx_pos)
    n_sample_pos = max(2, int(n_pos * sample_fraction))
    idx_pos_sample = rng.choice(idx_pos, size=min(n_sample_pos, n_pos), replace=False)

    n_neg = len(idx_neg)
    n_target_neg = len(idx_pos_sample)
    if n_neg == 0:
        return None

    idx_neg_sample = rng.choice(idx_neg, size=min(n_target_neg, n_neg), replace=False)
    emb_pos = embs[idx_pos_sample]
    y_pos = np.ones(len(idx_pos_sample))
    emb_neg = embs[idx_neg_sample]
    y_neg = np.zeros(len(idx_neg_sample))

    emb_total = np.vstack([emb_pos, emb_neg])
    y_total = np.hstack([y_pos, y_neg])

    clf = LogisticRegression(C=0.1, solver='liblinear',penalty='l2', class_weight='balanced',
                                 max_iter=1000, random_state=rng_seed)
    clf.fit(emb_total, y_total)

    v = clf.coef_[0]
    nrm = np.linalg.norm(v)
    if nrm > 0:
        v /= nrm

    return v

def generate_random_cav(embs: np.ndarray, scaler_emb: StandardScaler, n_samples: int, rng_seed: int=42) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    idx_all = np.arange(embs.shape[0])
    rng.shuffle(idx_all)

    n = min(n_samples, embs.shape[0] // 2)
    if n < 2:
        return None

    idx_pos = idx_all[0:n]
    idx_neg = idx_all[n:2*n]

    emb_pos = embs[idx_pos]
    y_pos = np.ones(n)
    emb_neg = embs[idx_neg]
    y_neg = np.zeros(n)

    emb_total = np.vstack([emb_pos, emb_neg])
    y_total = np.hstack([y_pos, y_neg])

    clf = LogisticRegression(C=0.1, solver='liblinear',penalty='l2', class_weight='balanced',
                                 max_iter=1000, random_state=rng_seed)
    clf.fit(emb_total, y_total)

    v = clf.coef_[0]
    nrm = np.linalg.norm(v)
    if nrm > 0:
        v /= nrm

    return v

def robust_tcav_significance_test(concept_idx: int, embs: np.ndarray, idx_pos: np.ndarray,
                                  idx_neg: np.ndarray, model_grads: np.ndarray, scaler_emb: StandardScaler,
                                  n_runs: int=15, sample_fraction: float =1.0, rng_seed: int=42) -> dict[str, Any]:

    # testes com dados negativos (que não obedecem à regra) aleatórios,
    # em vez de selecionar os que mais se distanciam da outra classe

    concept_cavs = []
    concept_tcav_scores = []

    for i in range(n_runs):
        seed = rng_seed + i * 100
        new_cav = train_cav_from_subset(
            embs=embs, idx_pos=idx_pos, idx_neg=idx_neg,
            scaler_emb=scaler_emb, sample_fraction=sample_fraction,
            rng_seed=seed
        )
        if new_cav is None:
            continue

        concept_cavs.append(new_cav)

        score = calculate_tcav_score(cav=new_cav, model_gradients=model_grads)
        concept_tcav_scores.append(score)

    random_cavs = []
    random_scores = []

    n_random_samples = max(20, min(len(idx_pos), len(idx_neg)) // 2)
    for i in range(n_runs):
        seed = rng_seed + 10000 + i * 100
        new_cav = generate_random_cav(
            embs=embs, scaler_emb=scaler_emb,
            n_samples=n_random_samples,
            rng_seed=seed
        )

        if new_cav is None:
            continue

        random_cavs.append(new_cav)

        score = calculate_tcav_score(cav=new_cav, model_gradients=model_grads)
        random_scores.append(score)

    concept_tcav_scores = np.asarray(concept_tcav_scores)
    random_scores = np.asarray(random_scores)

    std_dev_concept = np.std(concept_tcav_scores)
    std_dev_random = np.std(random_scores)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t, p_value = stats.ttest_ind(concept_tcav_scores, random_scores, nan_policy='raise')
        # input()
        # t, p_value = stats.ttest_ind(concept_tcav_scores, random_scores, nan_policy='raise')

    pooled_std_dev = np.sqrt(
                    (
                    (len(concept_tcav_scores) - 1) * std_dev_concept ** 2\
                    +(len(random_scores) - 1) * std_dev_random ** 2
                    ) / (len(concept_tcav_scores) + len(random_scores) - 2)
                )
    d = (np.mean(concept_tcav_scores) - np.mean(random_scores)) / (pooled_std_dev + 1e-10)


    return {
        'Factor': concept_idx,
        'concept_cavs': concept_cavs,
        'random_cavs': random_cavs,
        'concept_scores': concept_tcav_scores,
        'random_scores': random_scores,
        'p_value': p_value,
        't_stat': t,
        'is_significant': (not np.isnan(p_value) or p_value < 0.05),
        'cohens_d': d
    }

def compute_feature_associations(concept_activations: np.ndarray, X_features: np.ndarray, feature_names: list[str], quantile: float=0.1):
    high_thresh = np.quantile(concept_activations, 1.0 - quantile)
    low_thresh = np.quantile(concept_activations, quantile)

    high_idx = (concept_activations >= high_thresh)
    low_idx = (concept_activations <= low_thresh)

    rows = []
    for i, feat in enumerate(feature_names):
        feat_high = X_features[high_idx, i]
        feat_low = X_features[low_idx, i]

        mean_high = np.nanmean(feat_high)
        mean_low = np.nanmean(feat_low)

        std_high = np.nanstd(feat_high)
        std_low = np.nanstd(feat_low)

        pooled_variance = ((std_high ** 2 + std_low ** 2) / 2)
        pooled_std = np.sqrt(pooled_variance)

        # cohen's d: mede a diferença entre as médias de dois grupos (quantificada em desvios-padrão)
        # neste caso, ele compara as médias dos valores de uma feature entre os grupos de alta e baixa ativação do conceito analisado
        d = (mean_high - mean_low) / pooled_std if pooled_std > 0 else 0.0

        try:
            # teste two-sided que também compara as médias de duas distribuições (e dá a chance de elas serem diferente por acaso, no p_value)
            _, pval = stats.mannwhitneyu(feat_high, feat_low)
        except Exception:
            pval = 1.0

        rows.append(
            {
                'feature': feat,
                'mean_high': mean_high,
                'mean_low': mean_low,
                'diff': mean_high - mean_low,
                'std_high': std_high,
                'std_low': std_low,
                'cohens_d': d,
                'p_value': pval,
                'n_high': len(high_idx),
                'n_low': len(low_idx)
            }
        )

    df = pd.DataFrame(rows)
    df['significant'] = df['p_value'] < 0.05
    df = df.sort_values('cohens_d', key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)
    return df, high_idx, low_idx

def run_feature_association_dual_split(significant_factors: pd.DataFrame, tcav_eval_concept_activations: np.ndarray,
                                       held_out_concept_activations: np.ndarray, X_tcav_eval: np.ndarray, X_held_out: np.ndarray,
                                       feature_cols: list[str], quantile: float=0.1):
    fa_results_eval = {}
    fa_results_held_out = {}
    consistency_rows = []

    for factor in significant_factors['Factor']:
        print(f'Factor {factor}')
        factor_activations_tcav = tcav_eval_concept_activations[:, factor]
        df_fa_eval, _, _ = compute_feature_associations(concept_activations=factor_activations_tcav, X_features=X_tcav_eval, feature_names=feature_cols, quantile=quantile)
        fa_results_eval[factor] = df_fa_eval

        factor_activations_held_out = held_out_concept_activations[:, factor]
        df_fa_held_out, _, _ = compute_feature_associations(concept_activations=factor_activations_held_out, X_features=X_held_out, feature_names=feature_cols, quantile=quantile)
        fa_results_held_out[factor] = df_fa_held_out

        # top 5 conceitos com maior diferenciação de valor entre as amostras de alta ativação e as de baixa ativação
        top_eval = set(df_fa_eval.head(5)['feature'].values.tolist())
        top_held_out = set(df_fa_held_out.head(5)['feature'].values.tolist())

        # conceitos que apareceram nos dois conjuntos
        overlap = len(top_eval & top_held_out)

        consistency_rows.append(
            {
                'concept': factor,
                'top5_overlap': overlap,
                'top5_overlap_ratio': float(overlap / 5.0)
            }
        )

    consistency_df = pd.DataFrame(consistency_rows).sort_values('concept').reset_index(drop=True)
    return fa_results_eval, fa_results_held_out, consistency_df

def sparse_readout(concept_activations: np.ndarray, X_features: np.ndarray, feature_names: list[str], cv: int=5):
    ### avalia com um modelo linear a importância das features de entrada para a ativação do conceito ###
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    # modelo linear com incentivo à esparsidade que mapeia a influência das features na ativação do conceitos
    # usa k-fold cv para encontrar o melhor alpha (hiperparametro de esparsidade)
    model = LassoCV(cv=cv, random_state=42, max_iter=10000)
    model.fit(X_scaled, concept_activations)

    coefs = model.coef_
    # r² mede o erro médio das predições do modelo e dá um score (máximo 1.0)
    r2_train = model.score(X_scaled, concept_activations)

    cv_scores = cross_val_score(estimator=Lasso(alpha=model.alpha_, max_iter=10000),
                                X=X_scaled, y=concept_activations, cv=cv, scoring='r2')

    r2_cv = float(cv_scores.mean())
    r2_cv_std = float(np.std(cv_scores))

    idx = (np.abs(coefs) > 1e-10)
    selected_features = [feature_names[i] for i in idx]
    selected_coefs = coefs[idx]

    sorted_idx = np.argsort(np.abs(selected_coefs))[::-1]
    selected_features = [feature_names[i] for i in sorted_idx]
    selected_coefs = selected_coefs[sorted_idx]

    return {
        'coefs': coefs,
        'alpha': float(model.alpha_),
        'r2_train': r2_train,
        'r2_cv': r2_cv,
        'r2_cv_std': r2_cv_std,
        'selected_features': selected_features,
        'selected_coefs': selected_coefs,
        'scaler': scaler,
        'model': model,
    }


def evaluate_sparse_readout(sparse_result: dict[str], X_features: np.ndarray, concept_activations: np.ndarray):
    scaler = sparse_result['scaler']
    model = sparse_result['model']

    X_scaled = scaler.transform(X_features)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    r2 = float (model.score(X_scaled, concept_activations))
    pred = model.predict(X_scaled)
    # corrcoef cria uma matriz 2x2 de correlação entre predições e ativações reais
    # a diagonal principal é a correlação de predições com predições (100%) e ativações com ativações (100%),
    # já os outros membros correalcionam ativações e predições (ou vice-versa, o que é equivalente)
    corr = float(np.corrcoef(pred, concept_activations)[0, 1]) if len(concept_activations) > 1 else np.nan

    return {'r2': r2, 'correlation': corr}

def run_sparse_readout_dual_split(significant_factors: pd.DataFrame, cav_train_concept_activations: np.ndarray,
                                  tcav_eval_concept_activations: np.ndarray, held_out_concept_activations: np.ndarray,
                                  X_cav_train: np.ndarray, X_tcav_eval: np.ndarray, X_held_out: np.ndarray,
                                  feature_cols: list[str], cv: int, overfit_drop_warn_threshold: float):
    sparse_readout_results = {}
    sparse_readout_validation = {}
    summary_rows = []

    for factor in significant_factors['Factor']:
        cav_train_sparse_readout = sparse_readout(
            concept_activations=cav_train_concept_activations[:, factor],
            X_features=X_cav_train, feature_names=feature_cols, cv=cv
        )

        sparse_readout_results[factor] = cav_train_sparse_readout

        tcav_eval_result = evaluate_sparse_readout(
            sparse_result=cav_train_sparse_readout, X_features=X_tcav_eval,
            concept_activations=tcav_eval_concept_activations[:, factor]
        )

        held_out_result = evaluate_sparse_readout(
            sparse_result=cav_train_sparse_readout, X_features=X_held_out,
            concept_activations=held_out_concept_activations[:, factor]
        )

        sparse_readout_validation[factor] = {
            'tcav_eval': tcav_eval_result,
            'test': held_out_result
        }

        r2_drop_tcav = cav_train_sparse_readout['r2_cv'] - tcav_eval_result['r2']
        r2_drop_test = cav_train_sparse_readout['r2_cv'] - held_out_result['r2']

        summary_rows.append(
            {
                'concept': factor,
                'alpha': cav_train_sparse_readout['alpha'],
                'r2_train': cav_train_sparse_readout['r2_train'],
                'r2_cv': cav_train_sparse_readout['r2_cv'],
                'r2_cv_std': cav_train_sparse_readout['r2_cv_std'],
                'r2_tcav_eval': tcav_eval_result['r2'],
                'corr_tcav_eval': tcav_eval_result['correlation'],
                'r2_test': held_out_result['r2'],
                'corr_test': held_out_result['correlation'],
                'r2_drop_tcav_eval': r2_drop_tcav,
                'r2_drop_test': r2_drop_test,
                'warn_overfit': bool(r2_drop_test > overfit_drop_warn_threshold),
                'n_selected_features': len(cav_train_sparse_readout['selected_features'])
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values('concept').reset_index()
    return sparse_readout_results, sparse_readout_validation, summary_df

def get_significant_concepts(cavs: dict[str], tcav_scores: list[float], best_rules: dict[str], grads: np.ndarray, embs: np.ndarray, scaler: StandardScaler):
    for cav in cavs.values():
        idx = cav['Factor']
        cav['TCAV_score'] = tcav_scores[idx]

    for rule in best_rules:
        idx = rule['Factor']
        if idx in cavs:
            prec, rec = rule['Precision'], rule['Recall']
            cavs[idx]['Precision'] = prec
            cavs[idx]['Recall'] = rec

    robust_tcav_results = {}
    for idx, info in cavs.items():
        robust_result = robust_tcav_significance_test(
            concept_idx=idx, embs=embs,
            idx_pos=info['positive_idx'],
            idx_neg=info['negative_idx'],
            model_grads=grads, scaler_emb=scaler,
            sample_fraction=1.0, rng_seed=42
        )
        robust_tcav_results[idx] = robust_result

    significant_concepts = {}
    for idx in cavs:
        if robust_tcav_results[idx]['is_significant'] and abs(cavs[idx]['TCAV_score'] - 0.5) > 0.1:
            significant_concepts[idx] = cavs[idx]
            significant_concepts[idx]['p_value'] = robust_tcav_results[idx]['p_value']
            significant_concepts[idx]['t_stat'] = robust_tcav_results[idx]['t_stat']
            significant_concepts[idx]['Precision'] = cavs[idx]['Precision']
            significant_concepts[idx]['Recall'] = cavs[idx]['Recall']

    significant_df = pd.DataFrame([val for _, val in significant_concepts.items()])

    return significant_concepts, significant_df