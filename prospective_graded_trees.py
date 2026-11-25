"""Graded regression-tree helpers for the prospective concept audit (Roadmap Work 7, Scope B).

Each SAE factor is a nonnegative activation, divided by a scale frozen on the reference SAE training rows. A shallow
regression tree on clinical features X describes that normalized activation on the reference rule-discovery rows. The
tree is then held fixed and scored in later years against the held-out reference evaluation rows: skill against the
frozen reference mean, calibration, hurdle AUROC, Kendall tau on active rows, leaf shares and means, and Spearman
between the tree output and the observed activation. Controls have the same candidate count: random directions matched
on sparsity, unbiased random directions, principal components with both signs, and a tree on record-duration features.
"""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

import concept_subgroup_drift_pilot as pilot
import flchain_probe as probe
import prospective_audit as pa
from concept_study.alternative_bases import pca_encoder
from semantic_rules import RuleSet

SAE_SEEDS = (42, 135, 215)
SAE_EPOCHS = 100
SAE_SETTINGS = {"model_type": "ReLU", "alpha": 0.1, "learning_rate": 0.001, "scaling_factor": 1.5,
                "weight_decay": 0.0}
ENCODING_BATCH_SIZE = 256
CANDIDATE_CAP_FRACTION = 0.10
TREE_MAX_DEPTH = 3
TREE_MIN_SAMPLES_LEAF = 10
CV_FOLDS = 5
CCP_TIE = 1e-6
RIDGE_ALPHAS = np.logspace(-3, 5, 33)
SMALL_LEAF_STATUS = "sem intervalo"
UNDEFINED_STATUS = "média indefinida"
WORKER_BLAS_THREADS = 1
LEAF_QUANTITIES = ("leaf_share", "active_share", "positive_mean", "mean")


# ---------------------------------------------------------------- roles and disjoint row sets

def role_masks(row_role, year, reference_year):
    """Reference-year row sets: SAE training, tree fit (rule_discovery), probe fit (cav_fit), family fit, evaluation."""
    role = np.asarray(row_role)
    at_reference = np.asarray(year) == int(reference_year)
    masks = {
        "sae_fit": at_reference & (role == "sae_discovery"),
        "tree_fit": at_reference & (role == "rule_discovery"),
        "probe_fit": at_reference & (role == "cav_fit"),
        "family_fit": at_reference & np.isin(role, ["rule_discovery", "cav_fit"]),
        "reference_eval": at_reference & (role == "evaluation"),
    }
    assert_disjoint(masks["tree_fit"], masks["probe_fit"])
    return masks


def assert_disjoint(first, second):
    """Refuse two row masks that share a row (the probe must never see the tree-fit rows)."""
    shared = np.asarray(first, dtype=bool) & np.asarray(second, dtype=bool)
    if shared.any():
        raise ValueError(f"Row sets overlap on {int(shared.sum())} rows")


def survivor_row_mask(patient_id, year, cohort_patient_id, cohort_year):
    """Bridge rows whose (patient_id, year) pair is a prospective cohort row, i.e. alive on 31 December of that year."""
    bridge = pd.MultiIndex.from_arrays([np.asarray(patient_id, dtype=str), np.asarray(year, dtype=np.int64)])
    cohort = pd.MultiIndex.from_arrays([np.asarray(cohort_patient_id, dtype=str),
                                        np.asarray(cohort_year, dtype=np.int64)])
    return np.asarray(bridge.isin(cohort), dtype=bool)


# ---------------------------------------------------------------- factor scales and dead factors

def live_factor_mask(values):
    """Factors with at least one positive activation on the given rows; the others have no defined mu_j^+."""
    return (np.asarray(values, dtype=float) > 0).any(axis=0)


def factor_scale_table(values):
    """Per factor: active share P(y>0), mu_j^+ = E[y | y>0], median of positives and median/mean ratio."""
    values = np.asarray(values, dtype=float)
    rows = []
    for factor in range(values.shape[1]):
        column = values[:, factor]
        positive = column[column > 0]
        mean = float(positive.mean()) if len(positive) else np.nan
        median = float(np.median(positive)) if len(positive) else np.nan
        rows.append({"factor": factor, "active_share": float((column > 0).mean()), "mu_plus": mean,
                     "median_plus": median, "median_to_mean": median / mean if len(positive) else np.nan})
    return pd.DataFrame(rows).set_index("factor")


def normalize_activations(values, scale):
    """Activations divided by a frozen scale; a dead factor (no finite positive scale) is refused, not divided."""
    scale = np.asarray(scale, dtype=float)
    if not (np.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError("Scales must be finite and positive; drop dead factors before normalizing")
    return np.asarray(values, dtype=float) / scale


def candidate_columns(codes_by_seed, candidates):
    """Medoid activation columns of the candidate families, one column per candidate, over the given rows."""
    return np.column_stack([np.asarray(codes_by_seed[seed], dtype=float)[:, factor]
                            for seed, factor in (candidate["medoid"] for candidate in candidates)])


# ---------------------------------------------------------------- controls

def matched_bias(scores, target_share):
    """ReLU bias giving exactly round(target_share * n) positive rows among the given scores (ties not split)."""
    ordered = np.sort(np.asarray(scores, dtype=float))[::-1]
    count = int(round(float(target_share) * len(ordered)))
    if count <= 0:
        return float(-ordered[0] - 1.0)
    if count >= len(ordered):
        return float(-ordered[-1] + 1.0)
    return float(-(ordered[count - 1] + ordered[count]) / 2.0)


def _unit_directions(rng, dimension, count):
    directions = rng.standard_normal((dimension, count))
    return directions / np.linalg.norm(directions, axis=0, keepdims=True)


def sparsity_matched_random_factors(Z_fit, Z_all, target_shares, rng):
    """One random unit direction per target share, ReLU with a bias chosen on the fit rows to match that share."""
    Z_fit, Z_all = np.asarray(Z_fit, dtype=float), np.asarray(Z_all, dtype=float)
    directions = _unit_directions(rng, Z_fit.shape[1], len(target_shares))
    fit_scores = Z_fit @ directions
    biases = np.array([matched_bias(fit_scores[:, j], share) for j, share in enumerate(target_shares)])
    return np.maximum(Z_all @ directions + biases, 0.0), biases


def unbiased_random_factors(Z_all, count, rng):
    """ReLU of random unit directions without a bias."""
    directions = _unit_directions(rng, np.asarray(Z_all).shape[1], count)
    return np.maximum(np.asarray(Z_all, dtype=float) @ directions, 0.0)


def pca_factors(Z_fit, Z_all, count):
    """ReLU(+score) and ReLU(-score) of leading principal components of the fit rows, count factors in all."""
    return np.asarray(pca_encoder(np.asarray(Z_fit, dtype=float), count)(np.asarray(Z_all, dtype=float)), dtype=float)


def control_factor_sets(Z_std, fit_rows, target_shares, seed=probe.DIRECTION_SEED):
    """Sparsity-matched biased random, unbiased random and PCA control factors, each with one factor per candidate."""
    rng = np.random.default_rng(seed)
    count = len(target_shares)
    biased, biases = sparsity_matched_random_factors(Z_std[fit_rows], Z_std, target_shares, rng)
    controls = {"random_biased": biased,
                "random_unbiased": unbiased_random_factors(Z_std, count, rng),
                "pca": pca_factors(Z_std[fit_rows], Z_std, count)}
    return controls, biases


def sparsity_match_table(biased_columns, fit_rows, target_shares):
    """Target and achieved active share of each biased-random control factor on the SAE fit rows (no pass mark)."""
    achieved = (np.asarray(biased_columns, dtype=float)[np.asarray(fit_rows, dtype=bool)] > 0).mean(axis=0)
    target = np.asarray(target_shares, dtype=float)
    return pd.DataFrame({"control_index": np.arange(len(target)), "target_active_share": target,
                         "achieved_active_share": achieved, "gap": achieved - target})


# ---------------------------------------------------------------- shallow tree and structural stability

def tree_factory(ccp_alpha=0.0):
    return DecisionTreeRegressor(criterion="squared_error", max_depth=TREE_MAX_DEPTH,
                                 min_samples_leaf=TREE_MIN_SAMPLES_LEAF, random_state=pa.SEED, ccp_alpha=ccp_alpha)


def patient_grouped_folds(patients):
    """Five folds in which no patient appears on both sides of a split."""
    patients = np.asarray(patients)
    return list(GroupKFold(n_splits=CV_FOLDS).split(np.zeros(len(patients)), groups=patients))


def choose_ccp_alpha(alphas, squared_error, tie=CCP_TIE):
    """Smallest cross-validated error; within `tie` of it the largest alpha (the simplest tree) wins."""
    alphas = np.asarray(alphas, dtype=float)
    error = np.asarray(squared_error, dtype=float)
    return float(alphas[error <= error.min() + tie].max())


def select_ccp_alpha(X, y, patients):
    """Patient-grouped cross-validated error for every cost-complexity alpha on the full-data pruning path."""
    alphas = np.unique(tree_factory().cost_complexity_pruning_path(X, y).ccp_alphas)
    squared = np.zeros(len(alphas))
    rows_scored = 0
    for fit_rows, test_rows in patient_grouped_folds(patients):
        rows_scored += len(test_rows)
        for index, alpha in enumerate(alphas):
            model = tree_factory(alpha).fit(X[fit_rows], y[fit_rows])
            squared[index] += ((model.predict(X[test_rows]) - y[test_rows]) ** 2).sum()
    error = squared / rows_scored
    return choose_ccp_alpha(alphas, error), pd.DataFrame({"ccp_alpha": alphas, "cv_mse": error})


def fit_graded_tree(X, y, patients):
    """Shallow tree on the reference tree-fit rows, pruned by patient-grouped cross-validation."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    alpha, cv_table = select_ccp_alpha(X, y, patients)
    tree = tree_factory(alpha).fit(X, y)
    status = "constant" if tree.get_n_leaves() == 1 else "ok"
    return {"tree": tree, "ccp_alpha": alpha, "status": status, "cv": cv_table}


def leaf_nodes(tree):
    return np.flatnonzero(tree.tree_.children_left == -1)


def leaf_table(tree, feature_names):
    """One row per leaf: path conditions, numeric leaf mean in s_j units and fit rows, ordered by that mean."""
    nodes = tree.tree_
    rows = []

    def walk(node, conditions):
        if nodes.children_left[node] == -1:
            rows.append({"leaf": int(node), "conditions": " and ".join(conditions) or "all rows",
                         "mean": float(nodes.value[node][0][0]), "n": int(nodes.n_node_samples[node])})
            return
        name, cut = feature_names[nodes.feature[node]], nodes.threshold[node]
        walk(nodes.children_left[node], conditions + [f"{name} <= {cut:.6g}"])
        walk(nodes.children_right[node], conditions + [f"{name} > {cut:.6g}"])

    walk(0, [])
    return pd.DataFrame(rows).sort_values("mean", kind="stable").reset_index(drop=True)


def root_feature_name(tree, feature_names):
    nodes = tree.tree_
    return "constant" if nodes.children_left[0] == -1 else feature_names[nodes.feature[0]]


def structural_stability(X, y, patients, ccp_alpha, X_eval, prediction_eval, feature_names, draws, seed):
    """Patient-bootstrap refits of the frozen-alpha tree: Spearman with the original prediction and the root feature."""
    X, y, X_eval = np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(X_eval, dtype=float)
    universe, counts = pa.patient_draw_counts(patients, draws=draws, seed=seed)
    codes = pa.row_patient_codes(patients, universe)
    rows = []
    for draw in range(draws):
        index = np.repeat(np.arange(len(y)), counts[draw][codes])
        tree = tree_factory(ccp_alpha).fit(X[index], y[index])
        rows.append({"draw": draw, "spearman_with_original": _spearman(prediction_eval, tree.predict(X_eval)),
                     "root_feature": root_feature_name(tree, feature_names)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- reference baseline and leaf summaries

def count_status(count):
    """'média indefinida' for no rows, 'sem intervalo' for exactly one row, 'ok' otherwise."""
    if count == 0:
        return UNDEFINED_STATUS
    return SMALL_LEAF_STATUS if count == 1 else "ok"


def leaf_summary(leaf_ids, y, leaves):
    """Per leaf: rows n, active rows, leaf share, P(y>0 | leaf), E[y | y>0, leaf] and E[y | leaf] (zeros included).

    `status` is the count_status of n (the row count behind P(y>0 | leaf) and E[y | leaf]); `positive_status` is the
    count_status of the active rows (the row count behind E[y | y>0, leaf]).
    """
    ids, y = np.asarray(leaf_ids), np.asarray(y, dtype=float)
    rows = []
    for leaf in leaves:
        inside = ids == leaf
        values = y[inside]
        count = int(inside.sum())
        active = values[values > 0]
        rows.append({"leaf": int(leaf), "n": count, "n_active": len(active),
                     "leaf_share": count / len(y) if len(y) else np.nan,
                     "active_share": float((values > 0).mean()) if count else np.nan,
                     "positive_mean": float(active.mean()) if len(active) else np.nan,
                     "mean": float(values.mean()) if count else np.nan,
                     "status": count_status(count),
                     "positive_status": count_status(len(active))})
    return pd.DataFrame(rows)


def reference_leaf_baseline(tree, X_eval, y_eval):
    """Frozen leaf baseline from the held-out reference evaluation rows only (never from the tree-fit rows)."""
    return leaf_summary(tree.apply(np.asarray(X_eval, dtype=float)), y_eval, leaf_nodes(tree))


def weighted_leaf_regression(observed, held, counts):
    """Leaf means at year t on the held-out reference leaf means, weighted by n_t; returns (slope, intercept)."""
    observed, held, counts = (np.asarray(values, dtype=float) for values in (observed, held, counts))
    keep = np.isfinite(observed) & np.isfinite(held) & (counts >= 1)
    if keep.sum() < 2 or np.ptp(held[keep]) == 0:
        return np.nan, np.nan
    root = np.sqrt(counts[keep])
    design = np.column_stack([np.ones(keep.sum()), held[keep]]) * root[:, None]
    intercept, slope = np.linalg.lstsq(design, observed[keep] * root, rcond=None)[0]
    return float(slope), float(intercept)


# ---------------------------------------------------------------- longitudinal metrics

def _spearman(first, second):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if len(first) < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return np.nan
    return float(spearmanr(first, second)[0])


def _skill(y, prediction, reference_mean):
    denominator = np.mean((y - reference_mean) ** 2)
    if not denominator > 0:
        return np.nan
    return float(1.0 - np.mean((y - prediction) ** 2) / denominator)


def _calibration(y, prediction):
    if len(y) < 2 or np.ptp(prediction) == 0:
        return np.nan, np.nan
    design = np.column_stack([np.ones(len(prediction)), prediction])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(intercept), float(slope)


def hurdle_auroc(y, prediction):
    """AUROC of the tree output for the indicator 1[y > 0] (the zero of the ReLU)."""
    active = (np.asarray(y) > 0).astype(int)
    if np.ptp(prediction) == 0 or active.min() == active.max():
        return np.nan
    return float(roc_auc_score(active, prediction))


def active_kendall_tau_b(y, prediction):
    """Kendall tau-b between tree output and observed activation over the active rows (y > 0)."""
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    active = y > 0
    if active.sum() < 2 or np.ptp(y[active]) == 0 or np.ptp(prediction[active]) == 0:
        return np.nan
    return float(kendalltau(y[active], prediction[active], variant="b")[0])


def year_factor_metrics(y, prediction, reference_mean):
    """Skill against the frozen reference mean, errors in s_j units, calibration line, hurdle AUROC, tau-b, Spearman."""
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    intercept, slope = _calibration(y, prediction)
    return {"n": len(y), "active_share": float((y > 0).mean()), "mean_actual": float(y.mean()),
            "mean_prediction": float(prediction.mean()),
            "skill": _skill(y, prediction, reference_mean),
            "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))), "mae": float(np.mean(np.abs(y - prediction))),
            "calibration_intercept": intercept, "calibration_slope": slope,
            "hurdle_auroc": hurdle_auroc(y, prediction),
            "kendall_tau_b_actives": active_kendall_tau_b(y, prediction),
            "spearman_patient": _spearman(prediction, y)}


# ---------------------------------------------------------------- paired patient bootstrap

def paired_draw_statistics(statistic, columns, codes, counts):
    """Statistic on the full rows and on each patient-bootstrap draw; a row takes its patient's draw count.

    `codes` maps rows to a shared patient universe and `counts` holds one row of patient counts per draw, so
    every year and every arm that shares the universe is resampled with the same patients.
    """
    codes, counts = np.asarray(codes), np.asarray(counts)
    point = pd.Series(statistic(**columns), dtype=float)
    rows = []
    for draw in counts:
        index = np.repeat(np.arange(len(codes)), draw[codes])
        rows.append(statistic(**{name: np.asarray(values)[index] for name, values in columns.items()}))
    return point, pd.DataFrame(rows)


def summarize_paired_draws(point, draws):
    """Estimate, 95% percentile interval, valid draws and the share of undefined draws per metric."""
    table = pa.summarize_draws(point, draws)
    table["undefined_share"] = draws.isna().mean()
    return table


# ---------------------------------------------------------------- probe and gradient geometry

def fit_cav_probe(Z, target):
    """RidgeCV on standardized embeddings (cav_fit rows only) with the fixed logarithmic penalty grid."""
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    return model.fit(np.asarray(Z, dtype=float), np.asarray(target, dtype=float))


def raw_probe_weights(probe_model):
    """Probe coefficients on raw embedding units (coefficient divided by the standardization scale)."""
    scaler = probe_model.named_steps["standardscaler"]
    ridge = probe_model.named_steps["ridgecv"]
    return np.asarray(ridge.coef_, dtype=float) / scaler.scale_


def sae_factor_raw_direction(dictionary, factor):
    """Encoder row of one SAE factor on raw embedding units, comparable with the probe weights."""
    encoder = dictionary["run"]["model"].encoder.weight.detach().cpu().numpy()
    return encoder[factor] / dictionary["scaler"].scale_


def cosine(first, second):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    return float(first @ second / (np.linalg.norm(first) * np.linalg.norm(second)))


def component_energy_curve(weights, Z_fit):
    """Cumulative share of the probe weight energy along the principal components of the fit embeddings."""
    Z = np.asarray(Z_fit, dtype=float)
    _, _, components = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=True)
    weights = np.asarray(weights, dtype=float)
    return np.cumsum((components @ weights) ** 2 / (weights @ weights))


def head_gradient_angle_degrees(weights, mean_gradient):
    return float(np.degrees(np.arccos(np.clip(cosine(weights, mean_gradient), -1.0, 1.0))))


def mean_directional_derivative(gradients, weights):
    """Mean over the evaluated patients of the death-probability gradient projected on the probe direction."""
    weights = np.asarray(weights, dtype=float)
    unit = weights / np.linalg.norm(weights)
    return float(np.mean(np.asarray(gradients, dtype=float) @ unit))


# ---------------------------------------------------------------- family candidates and binary-rule arm

def family_candidate_set(codes_by_seed, live_by_seed, id_prefix, device, fraction=CANDIDATE_CAP_FRACTION):
    """Seed-matched families over live factors, full-family medoids ranked by positive count and capped."""
    seeds = list(codes_by_seed)
    matrices = [np.asarray(codes_by_seed[seed], dtype=float)[:, live_by_seed[seed]] for seed in seeds]
    factor_ids = [np.flatnonzero(live_by_seed[seed]) for seed in seeds]
    result = pilot.build_factor_families(matrices, seeds, device=device, id_prefix=id_prefix, factor_ids=factor_ids)
    medoids = {family["id"]: np.asarray(codes_by_seed[family["medoid"][0]], dtype=float)[:, family["medoid"][1]]
               for family in result["families"] if family["kind"] == "full"}
    latents = np.asarray(codes_by_seed[seeds[0]]).shape[1]
    selection = pilot.family_candidates(result["families"], medoids, latents, fraction)
    return {"counts": result["counts"], "per_seed": result["per_seed"], "families": result["families"],
            "selection": selection}


def run_binary_rule_arm(X, patients, feature_names, candidates, activations, sweep, rule, *, reference_year,
                        split_seed, workers, rule_source, recurrence):
    """Secondary arm: the frozen binary-rule sweep on the rule-discovery rows, one activation column per candidate."""
    values = np.asarray(activations, dtype=float)
    if values.shape != (len(X), len(candidates)):
        raise ValueError("Binary-rule activations need one column per candidate over the rule rows")
    return pilot.run_threshold_sweep(X, patients, feature_names, candidates, values.T, sweep, rule,
                                     reference_year=reference_year, split_seed=split_seed, workers=workers,
                                     rule_source=rule_source, recurrence=recurrence)


# ---------------------------------------------------------------- same-year bridge caches (read-only)

def open_bridge_cache(directory):
    """Read-only view of one saved unit cache; its directory is named by the configuration hash."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    records = [json.loads(line) for line in (directory / "units.jsonl").read_text().splitlines() if line.strip()]
    sources = {record["source_hash"] for record in records if record.get("config_hash") == directory.name}
    if len(sources) != 1:
        raise ValueError(f"Cache must hold exactly one source hash, found {len(sources)}")
    runtime = {**config, "disk_guard_gib": 0, "device": "cpu"}
    return pilot.UnitCache(directory.parent, runtime, sources.pop(), read_only=True, expected_hash=directory.name)


def checked_bridge_embeddings(predictor, roles, year, reference_year):
    """Rows and float embeddings of a saved predictor output, after checking its reference year and roles."""
    reference_year = int(reference_year)
    if int(predictor["reference_year"]) != reference_year:
        raise ValueError("Predictor output belongs to a different reference year")
    if predictor["roles_identity"] != roles["identity"]:
        raise ValueError("Predictor output was produced for different roles")
    rows = np.asarray(predictor["row_index"])
    embeddings = np.asarray(predictor["embeddings"], dtype=float)
    if not np.array_equal(rows, np.flatnonzero(np.asarray(year) >= reference_year)):
        raise ValueError("Predictor output must cover every row of years r..last")
    if embeddings.ndim != 2 or len(embeddings) != len(rows):
        raise ValueError("Predictor embeddings must align with row_index")
    return rows, embeddings


# ---------------------------------------------------------------- reference SAEs

def reference_sae_config():
    return {"profile": "pilot", "epochs": SAE_EPOCHS, "sae_seeds": list(SAE_SEEDS), "sae": dict(SAE_SETTINGS),
            "encoding_batch_size": ENCODING_BATCH_SIZE}


def fit_reference_saes(Z, device, seeds=SAE_SEEDS):
    """One reference SAE per seed on the sae_discovery embeddings, each with its frozen scaler."""
    config = reference_sae_config()
    return {seed: pilot.fit_reference_dictionary(Z, config, device, seed=seed) for seed in seeds}


def encode_reference_saes(dictionaries, Z, device):
    return {seed: pilot.encode_reference_dictionary(dictionary, np.asarray(Z, dtype=np.float32), device,
                                                    batch_size=ENCODING_BATCH_SIZE)
            for seed, dictionary in dictionaries.items()}


# ---------------------------------------------------------------- row sets, features and seed members

DURATION_FEATURES = ("last_month", "active_months", "total_count")


def duration_features(timing, X):
    """Record-duration and intensity features only: last care month, active months and total clinical event count."""
    total = np.asarray(X, dtype=float).sum(axis=1)
    return np.column_stack([np.asarray(timing["last_month"], dtype=float),
                            np.asarray(timing["active_months"], dtype=float), total])


def evaluation_year_mask(row_role, year, reference_year, last_year=pa.LAST_OUTCOME_INPUT_YEAR):
    """Evaluation-role rows from the reference year to the last outcome-valid input year."""
    year = np.asarray(year)
    return ((year >= int(reference_year)) & (year <= int(last_year))
            & (np.asarray(row_role) == "evaluation"))


def _member_factor(family, seed):
    for member_seed, factor in family["members"]:
        if member_seed == seed:
            return factor
    raise ValueError(f"Family {family['id']} has no member in seed {seed}")


def seed_member_columns(codes_by_seed, families, candidates):
    """For each seed, one activation column per candidate: the candidate family's member factor in that seed."""
    family_by_id = {family["id"]: family for family in families}
    columns = {}
    for seed in codes_by_seed:
        values = np.asarray(codes_by_seed[seed], dtype=float)
        columns[seed] = np.column_stack([values[:, _member_factor(family_by_id[candidate["family_id"]], seed)]
                                         for candidate in candidates])
    return columns


# ---------------------------------------------------------------- frozen binary rule arm (secondary)

def frozen_rule_metrics(entry, X, activation):
    """Rule R and cut-off A = activation > cutoff on the given rows: shares, P(A | R) and AUROC(activation for R)."""
    values = np.asarray(activation, dtype=float)
    if not len(values):
        return {"rule_share": np.nan, "active_share": np.nan, "active_given_rule": np.nan,
                "auroc_activation_rule": np.nan}
    rule = RuleSet.from_dict(entry["rule_set"])
    in_rule = np.asarray(rule.mask(np.asarray(X, dtype=float)), dtype=bool)
    active = values > float(entry["cutoff"])
    has_both = bool(in_rule.any() and (~in_rule).any())
    return {"rule_share": float(in_rule.mean()), "active_share": float(active.mean()),
            "active_given_rule": float(active[in_rule].mean()) if in_rule.any() else np.nan,
            "auroc_activation_rule": float(roc_auc_score(in_rule.astype(int), values)) if has_both else np.nan}


def frozen_rule_year_rows(entry, X, activation, years, patients, universe, draws, seed):
    """Frozen-rule metrics for every input year, with paired patient-bootstrap intervals over one shared universe."""
    universe, counts = pa.patient_draw_counts(universe, draws=draws, seed=seed)
    codes = pa.row_patient_codes(patients, universe)
    X, activation, years = np.asarray(X, dtype=float), np.asarray(activation, dtype=float), np.asarray(years)
    rows = []
    for year in np.unique(years):
        inside = years == year
        columns = {"X": X[inside], "activation": activation[inside]}
        point, draw_table = paired_draw_statistics(
            lambda X, activation: frozen_rule_metrics(entry, X, activation), columns, codes[inside], counts)
        rows.append(summarize_paired_draws(point, draw_table).reset_index(names="metric").assign(input_year=int(year)))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- parallel candidate jobs and bridge dictionaries

def limit_worker_threads():
    """Pool initializer: each worker process uses WORKER_BLAS_THREADS BLAS and OpenMP threads."""
    threadpool_limits(limits=WORKER_BLAS_THREADS)


def map_in_processes(function, jobs, workers):
    """Run function over the jobs in a fork-based process pool; results come back in job order.

    The thread limit is applied inside each worker, not in this process: a limit set here and restored after the pool
    made a later scipy fit segfault in the same process.
    """
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=context,
                             initializer=limit_worker_threads) as pool:
        return list(pool.map(function, jobs))


def load_bridge_dictionaries(cache, reference_year, device):
    """Frozen same-year reference SAEs of one saved bridge cache, one per seed, with their scalers checked."""
    return pilot.load_reference_dictionaries(cache, int(reference_year), device)


# ---------------------------------------------------------------- one candidate: tree, years, probes

EVALUATION_KEYS = ("variant", "reference_year", "arm", "candidate")


def _leaf_regression(leaf, y, leaves, held_means):
    """Year-t leaf means E[y | leaf] (all leaf rows, n_t weights) on the held-out reference leaf means."""
    observed = leaf_summary(leaf, y, leaves)
    return weighted_leaf_regression(observed["mean"], held_means, observed["n"])


def _summary_rows(statistic, columns, patients, universe, counts, **keys):
    """Point estimate, interval and undefined share of one statistic over the paired patient draws."""
    codes = pa.row_patient_codes(patients, universe)
    point, draw_table = paired_draw_statistics(statistic, columns, codes, counts)
    return summarize_paired_draws(point, draw_table).reset_index(names="metric").assign(**keys)


def leaf_metric_name(quantity, leaf):
    return f"{quantity}_leaf{int(leaf)}"


def leaf_metric_values(y, leaf, leaves):
    """Every leaf quantity at one input year, as flat metrics named by quantity and leaf."""
    table = leaf_summary(leaf, y, leaves)
    metrics = {}
    for quantity in LEAF_QUANTITIES:
        values = table[quantity].to_numpy(dtype=float)
        for position, leaf_id in enumerate(leaves):
            metrics[leaf_metric_name(quantity, leaf_id)] = values[position]
    return metrics


def leaf_draws(columns, leaves, patients, universe, counts):
    """Leaf quantities on the full rows and on each patient-bootstrap draw (point Series, draws DataFrame)."""
    return paired_draw_statistics(lambda y, leaf: leaf_metric_values(y, leaf, leaves), columns,
                                  pa.row_patient_codes(patients, universe), counts)


def leaf_quantity_status(quantity, counts):
    """Status of one leaf quantity from its leaf_summary row; the leaf share is always defined."""
    if quantity == "leaf_share":
        return "ok"
    return counts["positive_status"] if quantity == "positive_mean" else counts["status"]


def comparison_status(*statuses):
    """Worst status among the sides of a comparison: 'média indefinida', then 'sem intervalo', else 'ok'."""
    for status in (UNDEFINED_STATUS, SMALL_LEAF_STATUS):
        if status in statuses:
            return status
    return "ok"


def leaf_interval_rows(columns, patients, leaves, reference, reference_draws, universe, counts, **keys):
    """One row per leaf and quantity at one input year: estimate, interval, reference value and paired delta.

    `reference` is the held-out reference leaf table; `reference_draws` is leaf_draws on the reference rows with the
    same universe and counts (computed once per job). The delta draw is the year-t draw minus the reference draw, so
    rows that are the reference rows give exactly zero.

    `status` describes the year-t value (count_status of its rows); `delta_status` is the worst status of the year-t
    and reference values. Intervals are blank unless their status is 'ok'. Estimates are blank when undefined.
    """
    point, draws = leaf_draws(columns, leaves, patients, universe, counts)
    delta_point, delta_draws = pa.difference_draws(point, draws, *reference_draws)
    level = summarize_paired_draws(point, draws)
    delta = summarize_paired_draws(delta_point, delta_draws)
    year_counts = leaf_summary(columns["leaf"], columns["y"], leaves)
    rows = []
    for position, leaf_id in enumerate(leaves):
        year_row, reference_row = year_counts.iloc[position], reference.iloc[position]
        for quantity in LEAF_QUANTITIES:
            name = leaf_metric_name(quantity, leaf_id)
            status = leaf_quantity_status(quantity, year_row)
            delta_status = comparison_status(status, leaf_quantity_status(quantity, reference_row))
            rows.append({**keys, "leaf": int(leaf_id), "quantity": quantity, "status": status,
                         "delta_status": delta_status,
                         "n": int(year_row["n"]), "n_active": int(year_row["n_active"]),
                         "reference_n": int(reference_row["n"]), "reference_n_active": int(reference_row["n_active"]),
                         "estimate": level.loc[name, "estimate"],
                         "ci_low": level.loc[name, "ci_low"] if status == "ok" else np.nan,
                         "ci_high": level.loc[name, "ci_high"] if status == "ok" else np.nan,
                         "valid_draws": level.loc[name, "valid_draws"],
                         "undefined_share": level.loc[name, "undefined_share"],
                         "reference_value": reference_row[quantity],
                         "delta_estimate": delta.loc[name, "estimate"],
                         "delta_valid_draws": delta.loc[name, "valid_draws"],
                         "delta_undefined_share": delta.loc[name, "undefined_share"],
                         "delta_ci_low": delta.loc[name, "ci_low"] if delta_status == "ok" else np.nan,
                         "delta_ci_high": delta.loc[name, "ci_high"] if delta_status == "ok" else np.nan})
    return pd.DataFrame(rows)


def _probe_rows(job, tree, reference_mean, reference_y, universe, counts, keys):
    probe_inputs = job["probe"]
    evaluation_y = np.asarray(job["evaluation_y"], dtype=float)
    patients = np.asarray(job["evaluation_patients"], dtype=str)
    years = np.asarray(job["evaluation_year"])
    target = tree.predict(np.asarray(probe_inputs["X_fit"], dtype=float))
    ridge = fit_cav_probe(probe_inputs["Z_fit"], target)
    weights = raw_probe_weights(ridge)
    prediction = ridge.predict(np.asarray(probe_inputs["Z_eval"], dtype=float))
    activation_probe = fit_cav_probe(probe_inputs["Z_fit"], probe_inputs["y_fit"])
    direction = np.asarray(probe_inputs["direction"], dtype=float)
    gradients = np.asarray(probe_inputs["gradients"], dtype=float)
    reference_year = keys["reference_year"]
    frames = []
    for year in np.unique(years):
        rows = years == year
        frames.append(_summary_rows(
            lambda y, prediction: year_factor_metrics(y, prediction, reference_mean),
            {"y": evaluation_y[rows], "prediction": prediction[rows]}, patients[rows], universe, counts,
            section="probe_metrics", input_year=int(year), **keys))
        frames.append(_summary_rows(
            lambda gradients: {"gradient_angle_degrees": head_gradient_angle_degrees(weights, gradients.mean(axis=0)),
                               "mean_directional_derivative": mean_directional_derivative(gradients, weights)},
            {"gradients": gradients[rows]}, patients[rows], universe, counts,
            section="geometry", input_year=int(year), **keys))
    sanity = pd.DataFrame([
        {"metric": "r2_reference_activation",
         "estimate": float(activation_probe.score(np.asarray(probe_inputs["Z_reference"], dtype=float), reference_y))},
        {"metric": "cosine_tree_direction", "estimate": cosine(weights, direction)},
        {"metric": "cosine_activation_direction", "estimate": cosine(raw_probe_weights(activation_probe), direction)}])
    frames.append(sanity.assign(section="sanity", input_year=reference_year, **keys))
    curve = component_energy_curve(weights, probe_inputs["Z_fit"])
    frames.append(pd.DataFrame({"metric": [f"cumulative_energy_pc{k}" for k in range(1, len(curve) + 1)],
                                "estimate": curve}).assign(section="energy", input_year=reference_year, **keys))
    return pd.concat(frames, ignore_index=True)


def evaluate_candidate(job):
    """Fit one candidate's shallow tree and score it in every later input year, actual and held at the reference year.

    Held is defined per evaluation row: the row keeps its own clinical features X, so the tree output and leaf are the
    actual ones, and only the activation y is the one captured with the year input set to the reference year
    (job["held_y"], aligned with evaluation rows; None means no held variant). Returns the candidate summary, the year
    metrics with paired patient-bootstrap intervals, the leaf table, and probe rows when the job carries probe inputs.
    """
    keys = {name: job[name] for name in EVALUATION_KEYS}
    feature_names = list(job["feature_names"])
    fitted = fit_graded_tree(job["fit_X"], job["fit_y"], job["fit_patients"])
    tree = fitted["tree"]
    leaves = leaf_nodes(tree)
    reference_X = np.asarray(job["reference_X"], dtype=float)
    reference_y = np.asarray(job["reference_y"], dtype=float)
    reference_mean = float(reference_y.mean())
    baseline = reference_leaf_baseline(tree, reference_X, reference_y)
    held_means = baseline["mean"].to_numpy(dtype=float)
    universe = np.asarray(job["universe"], dtype=str)
    _, counts = pa.patient_draw_counts(universe, draws=job["draws"], seed=job["bootstrap_seed"])
    reference_draws = leaf_draws({"y": reference_y, "leaf": tree.apply(reference_X)}, leaves,
                                 np.asarray(job["reference_patients"], dtype=str), universe, counts)
    stability = structural_stability(job["fit_X"], job["fit_y"], job["fit_patients"], fitted["ccp_alpha"],
                                     reference_X, tree.predict(reference_X), feature_names,
                                     draws=job["structural_draws"], seed=pa.SEED)
    root = root_feature_name(tree, feature_names)
    spearman = stability["spearman_with_original"].astype(float)
    summary = {**keys, "tree_status": fitted["status"], "ccp_alpha": fitted["ccp_alpha"],
               "leaves": tree.get_n_leaves(), "depth": tree.get_depth(), "root_feature": root,
               "root_share": float((stability["root_feature"] == root).mean()),
               "structural_spearman_median": float(spearman.median()),
               "structural_spearman_low": float(spearman.quantile(pa.LOW_QUANTILE)),
               "structural_spearman_high": float(spearman.quantile(pa.HIGH_QUANTILE)),
               "fit_rows": len(job["fit_y"]), "reference_rows": len(reference_y),
               "reference_mean": reference_mean, "reference_active_share": float((reference_y > 0).mean())}

    def year_statistic(y, prediction, leaf):
        metrics = year_factor_metrics(y, prediction, reference_mean)
        slope, intercept = _leaf_regression(leaf, y, leaves, held_means)
        return {**metrics, "leaf_regression_slope": slope, "leaf_regression_intercept": intercept}

    evaluation_X = np.asarray(job["evaluation_X"], dtype=float)
    evaluation_y = np.asarray(job["evaluation_y"], dtype=float)
    patients = np.asarray(job["evaluation_patients"], dtype=str)
    years = np.asarray(job["evaluation_year"])
    held_y = job["held_y"]
    if held_y is not None:
        held_y = np.asarray(held_y, dtype=float)
        if len(held_y) != len(evaluation_y):
            raise ValueError("Held activations must align with the evaluation rows")
    year_frames, leaf_frames = [], []
    for year in np.unique(years):
        rows = years == year
        actual = {"y": evaluation_y[rows], "prediction": tree.predict(evaluation_X[rows]),
                  "leaf": tree.apply(evaluation_X[rows])}
        year_frames.append(_summary_rows(year_statistic, actual, patients[rows], universe, counts,
                                         comparison="actual", input_year=int(year), **keys))
        leaf_frames.append(leaf_interval_rows({"y": actual["y"], "leaf": actual["leaf"]}, patients[rows], leaves,
                                              baseline, reference_draws, universe, counts,
                                              comparison="actual", input_year=int(year), **keys))
        if held_y is not None:
            held = {**actual, "y": held_y[rows]}
            year_frames.append(_summary_rows(year_statistic, held, patients[rows], universe, counts,
                                             comparison="held", input_year=int(year), **keys))
            leaf_frames.append(leaf_interval_rows({"y": held["y"], "leaf": held["leaf"]}, patients[rows], leaves,
                                                  baseline, reference_draws, universe, counts,
                                                  comparison="held", input_year=int(year), **keys))
    leaf_conditions = leaf_table(tree, feature_names)[["leaf", "conditions"]]
    leaf_rows = pd.concat(leaf_frames, ignore_index=True).merge(leaf_conditions, on="leaf", how="left")
    probe_rows = (_probe_rows(job, tree, reference_mean, reference_y, universe, counts, keys)
                  if job.get("probe") is not None else pd.DataFrame(columns=["section", "metric", "estimate"]))
    return {"summary": pd.DataFrame([summary]), "year_metrics": pd.concat(year_frames, ignore_index=True),
            "leaves": leaf_rows, "probe_rows": probe_rows}
