"""Calendar-landmarked prospective mortality target and DR-TabPFN last_3 audit helpers (Roadmap Work 7, Scope A).

Features X_{i,t} are calendar-year-t inputs of patients alive on 31 December of t. The outcome is death in
year t+1 from the full-history raw DEATH events. Patients who died on or before t are removed. Context for a reference year r
uses the last_3 window max(2007, r-2)..r. Patients are split into roles with the frozen reference split.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import concept_subgroup_drift_pilot as pilot
import flchain_probe as probe
from concept_study import error_analysis as ea

FIRST_INPUT_YEAR = 2007
LAST_OUTCOME_INPUT_YEAR = 2013
LAST_REPRESENTATION_INPUT_YEAR = 2014
CONTEXT_YEARS = 3
SELECTION_ROLES = ("rule_discovery", "cav_fit", "concept_selection")
EVALUATION_ROLE = "evaluation"
REFERENCE_CONTEXT_ROLE = "tabpfn_context"
BOOTSTRAP_DRAWS = 500
BOOTSTRAP_SEED = 20261106
SEED = 42
LOW_QUANTILE = 0.025
HIGH_QUANTILE = 0.975


# ---------------------------------------------------------------- population and cohort

def load_population(path, expected_sha256: str) -> dict:
    """Read the prepared population after its file SHA-256 is verified (refused before unpickling)."""
    return pilot.load_pilot_population(path, expected_sha256)


def first_death_years(events: pd.DataFrame) -> pd.Series:
    """First calendar year with a raw DEATH event per patient, from full-history events."""
    deaths = events.loc[events["event"] == "DEATH"]
    years = pd.to_datetime(deaths["date"]).dt.year
    return years.groupby(deaths["patient_id"].astype(str)).min().astype(np.int64)


def first_death_year_by_row(population: dict, events: pd.DataFrame) -> np.ndarray:
    """Population-row first DEATH year; NaN for patients with no DEATH event (never died)."""
    patients = np.asarray(population["patient_id"], dtype=str)
    event_patients = {str(value) for value in events["patient_id"].unique()}
    absent = sorted(set(patients.tolist()) - event_patients)
    if absent:
        raise ValueError(f"{len(absent)} population patients have no raw events, e.g. {absent[:3]}")
    return first_death_years(events).reindex(patients).to_numpy(dtype=float)


def landmark_cohort(population: dict, first_death_year_raw: np.ndarray) -> pd.DataFrame:
    """Rows alive at 31 December of their input year, with next-year death as the outcome.

    Keeps input years 2007-2014 where first_death_year_raw is missing (never died) or later than the input year.
    The outcome y_next = 1 when the first raw DEATH year is the input year + 1. It does not depend on a row in
    the next year. y_next is NaN for input year 2014, which is representation only (2015 deaths are censored).
    Input year 2015 is dropped.
    """
    year = np.asarray(population["year"], dtype=np.int64)
    first = np.asarray(first_death_year_raw, dtype=float)
    if first.shape != year.shape:
        raise ValueError("first_death_year_raw must align with population rows")
    inside = (year >= FIRST_INPUT_YEAR) & (year <= LAST_REPRESENTATION_INPUT_YEAR)
    alive_at_landmark = np.isnan(first) | (first > year)
    rows = np.flatnonzero(inside & alive_at_landmark)
    died_next_year = (first[rows] == year[rows] + 1).astype(float)
    outcome_valid = year[rows] <= LAST_OUTCOME_INPUT_YEAR
    return pd.DataFrame({
        "row": rows,
        "patient_id": np.asarray(population["patient_id"], dtype=str)[rows],
        "year": year[rows],
        "first_death_year_raw": first[rows],
        "y_next": np.where(outcome_valid, died_next_year, np.nan),
        "outcome_valid": outcome_valid,
        "representation_only": year[rows] == LAST_REPRESENTATION_INPUT_YEAR,
    })


def cohort_check_rows(cohort: pd.DataFrame) -> pd.DataFrame:
    """Integrity checks of the landmarked cohort: no pre-landmark deaths, 2015 dropped, y_next only when valid."""
    same_or_earlier_death = int((cohort["first_death_year_raw"] <= cohort["year"]).sum())
    latest_year = int(cohort["year"].max())
    valid = cohort["outcome_valid"].to_numpy(dtype=bool)
    y_defined_exactly_when_valid = bool(
        cohort.loc[valid, "y_next"].notna().all() and cohort.loc[~valid, "y_next"].isna().all())
    return pd.DataFrame([
        {"check": "rows with raw DEATH on or before the input year", "value": same_or_earlier_death,
         "rule": "0", "passed": same_or_earlier_death == 0},
        {"check": "latest input year in cohort", "value": latest_year,
         "rule": f"<= {LAST_REPRESENTATION_INPUT_YEAR}", "passed": latest_year <= LAST_REPRESENTATION_INPUT_YEAR},
        {"check": "y_next defined exactly on outcome-valid rows", "value": int(valid.sum()),
         "rule": "defined iff input year <= 2013", "passed": y_defined_exactly_when_valid},
    ])


def add_roles(cohort: pd.DataFrame, population: dict, reference_year: int, seed: int = SEED):
    """Attach the frozen reference-year roles to cohort rows (split on the full population, not the cohort)."""
    roles = pilot.reference_year_roles(population, int(reference_year), seed)
    pilot.check_role_coverage(population, roles)
    rows = cohort["row"].to_numpy()
    out = cohort.copy()
    out["role"] = roles["row_role"][rows]
    out["is_entrant"] = roles["is_entrant"][rows]
    return out, roles


# ---------------------------------------------------------------- context, roles and masks

def context_first_year(reference_year: int) -> int:
    """First year of the last_3 window: max(2007, r-2)."""
    return max(FIRST_INPUT_YEAR, int(reference_year) - (CONTEXT_YEARS - 1))


def context_mask(cohort: pd.DataFrame, reference_year: int) -> np.ndarray:
    """last_3 context: reference-year rows of tabpfn_context patients, plus earlier window rows of every cohort patient."""
    r = int(reference_year)
    year = cohort["year"].to_numpy()
    in_window = (year >= context_first_year(r)) & (year <= r)
    reference_context = (year == r) & (cohort["role"].to_numpy() == REFERENCE_CONTEXT_ROLE)
    earlier = year < r
    return in_window & (reference_context | earlier)


def selection_mask(cohort: pd.DataFrame, reference_year: int) -> np.ndarray:
    """Reference-year rows used to choose the frozen Death-F1 threshold."""
    year = cohort["year"].to_numpy()
    return ((year == int(reference_year)) & cohort["outcome_valid"].to_numpy(dtype=bool)
            & np.isin(cohort["role"].to_numpy(), SELECTION_ROLES))


def evaluation_mask(cohort: pd.DataFrame, reference_year: int) -> np.ndarray:
    """Evaluation-role rows from the reference year onward with an observed outcome."""
    year = cohort["year"].to_numpy()
    return ((year >= int(reference_year)) & cohort["outcome_valid"].to_numpy(dtype=bool)
            & (cohort["role"].to_numpy() == EVALUATION_ROLE))


def prior_context_counts(cohort: pd.DataFrame, context: np.ndarray) -> np.ndarray:
    """Per cohort row, the number of context rows its patient has (patient_rows_in_context)."""
    patients = cohort["patient_id"].to_numpy()
    flags = pd.Series(np.asarray(context, dtype=np.int64))
    return flags.groupby(patients).transform("sum").to_numpy(dtype=np.int64)


# ---------------------------------------------------------------- threshold and metrics

def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else np.nan


def _flag_counts(labels: np.ndarray, probabilities: np.ndarray, threshold: float, weights: np.ndarray) -> dict:
    flagged = probabilities >= threshold
    death = labels == 1
    return {
        "total": weights.sum(), "deaths": weights[death].sum(), "survivors": weights[~death].sum(),
        "tp": weights[death & flagged].sum(), "fp": weights[~death & flagged].sum(),
        "fn": weights[death & ~flagged].sum(),
    }


def death_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Probability cut-off with the highest Death-F1 over the given rows; ties go to the higher cut-off."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("labels and probabilities must share a 1-D shape")
    if not np.isin(y, [0, 1]).all() or y.sum() == 0:
        raise ValueError("threshold selection needs binary labels with at least one death")
    candidates = np.unique(p)
    scores = []
    for threshold in candidates:
        c = _flag_counts(y, p, threshold, np.ones(len(y)))
        scores.append(_ratio(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"]))
    scores = np.asarray(scores)
    return float(candidates[np.flatnonzero(scores == scores.max())[-1]])


def recalibration_slope_intercept(labels, probabilities, weights) -> tuple[float, float]:
    """Weighted logistic recalibration of the outcome on logit(p); NaN when only one class carries weight."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w[y == 1].sum() <= 0 or w[y == 0].sum() <= 0:
        return np.nan, np.nan
    eps = np.finfo(float).eps
    clipped = np.clip(p, eps, 1.0 - eps)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, max_iter=1000)
    model.fit(logit, y, sample_weight=w)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def longitudinal_metrics(labels, probabilities, weights, threshold: float, base_rate: float) -> dict:
    """Weighted performance at one year: discrimination, calibration, frozen-threshold flags and entropy (nats)."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=float)
    w = np.asarray(weights, dtype=float)
    c = _flag_counts(y, p, threshold, w)
    brier = _ratio((w * (p - y) ** 2).sum(), c["total"])
    brier_reference = _ratio((w * (base_rate - y) ** 2).sum(), c["total"])
    slope, intercept = recalibration_slope_intercept(y, p, w)
    has_both = c["deaths"] > 0 and c["survivors"] > 0
    return {
        "rows": c["total"],
        "deaths": c["deaths"],
        "auroc": float(roc_auc_score(y, p, sample_weight=w)) if has_both else np.nan,
        "pr_auc": float(average_precision_score(y, p, sample_weight=w)) if c["deaths"] > 0 else np.nan,
        "brier": brier,
        "brier_skill": 1.0 - brier / brier_reference if brier_reference > 0 else np.nan,
        "recalibration_slope": slope,
        "recalibration_intercept": intercept,
        "death_f1": _ratio(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"]),
        "recall": _ratio(c["tp"], c["deaths"]),
        "false_alarm_rate": _ratio(c["fp"], c["survivors"]),
        "share_flagged": _ratio(c["tp"] + c["fp"], c["total"]),
        "caught_deaths": c["tp"],
        "false_alarms": c["fp"],
        "mean_probability": _ratio((w * p).sum(), c["total"]),
        "mean_entropy_nats": _ratio((w * probe.binary_entropy_nats(p)).sum(), c["total"]),
    }


# ---------------------------------------------------------------- paired patient bootstrap

def patient_draw_counts(patient_ids, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED):
    """Patient bootstrap counts over the sorted unique population patient_id strings.

    Each draw resamples the universe with replacement. Every row inherits its patient's count, so one draw
    pairs the same patients across years, models and the actual/held comparison.
    """
    universe = np.unique(np.asarray(patient_ids, dtype=str))
    picks = np.random.default_rng(int(seed)).integers(0, len(universe), size=(int(draws), len(universe)))
    counts = np.stack([np.bincount(row, minlength=len(universe)) for row in picks])
    return universe, counts


def row_patient_codes(patient_ids, universe) -> np.ndarray:
    """Position of each row's patient_id in the bootstrap universe."""
    universe = np.asarray(universe, dtype=str)
    ids = np.asarray(patient_ids, dtype=str)
    codes = np.searchsorted(universe, ids)
    if np.any(codes >= len(universe)) or not np.array_equal(universe[np.minimum(codes, len(universe) - 1)], ids):
        raise ValueError("patient_id missing from the bootstrap universe")
    return codes


def metric_draws(labels, probabilities, codes, counts, threshold: float, base_rate: float):
    """Point estimates and one row of metrics per patient-bootstrap draw."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=float)
    point = pd.Series(longitudinal_metrics(y, p, np.ones(len(y)), threshold, base_rate))
    weights = counts[:, np.asarray(codes)].astype(float)
    rows = [longitudinal_metrics(y, p, row_weights, threshold, base_rate) for row_weights in weights]
    return point, pd.DataFrame(rows, columns=point.index)


def difference_draws(point_a, draws_a, point_b, draws_b):
    """Paired difference A - B, computed draw by draw."""
    return point_a - point_b, draws_a - draws_b


def summarize_draws(point: pd.Series, draws: pd.DataFrame) -> pd.DataFrame:
    """Estimate, 95% percentile interval and number of valid (finite) draws per metric."""
    return pd.DataFrame({
        "estimate": point,
        "ci_low": draws.quantile(LOW_QUANTILE),
        "ci_high": draws.quantile(HIGH_QUANTILE),
        "valid_draws": draws.notna().sum(),
    })


# ---------------------------------------------------------------- timing and CPU baselines

def timing_features(cohort: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """last_month and active_months of each cohort patient-year from non-DEATH events.

    Patient-years without any care event get 0 and no_care_event = True (counted, not dropped).
    """
    patients = cohort["patient_id"].astype(str).to_numpy()
    table = ea.record_timing(events, set(patients))
    index = pd.MultiIndex.from_arrays([patients, cohort["year"].astype(np.int64).to_numpy()],
                                      names=["patient_id", "year"])
    found = table.reindex(index)
    out = pd.DataFrame({
        "last_month": found["last_month"].to_numpy(dtype=float),
        "active_months": found["active_months"].to_numpy(dtype=float),
    })
    out["no_care_event"] = out["last_month"].isna().to_numpy()
    return out.fillna({"last_month": 0.0, "active_months": 0.0})


def baseline_scores(X_context, y_context, timing_context, X_eval, timing_eval) -> dict:
    """Death probabilities of the CPU baselines refitted on the same context rows."""
    models = ea.baseline_models()
    y = np.asarray(y_context, dtype=np.int64)
    X_c, X_e = np.asarray(X_context, dtype=float), np.asarray(X_eval, dtype=float)
    scores = {
        "logistic": models["logistic"]().fit(np.log1p(X_c), y).predict_proba(np.log1p(X_e))[:, 1],
        "xgboost": models["boosting"]().fit(np.log1p(X_c), y).predict_proba(np.log1p(X_e))[:, 1],
        "total_count": models["count"]().fit(np.log1p(X_c.sum(axis=1, keepdims=True)), y)
        .predict_proba(np.log1p(X_e.sum(axis=1, keepdims=True)))[:, 1],
    }
    for name in ("last_month", "active_months"):
        fit_x = np.asarray(timing_context[[name]], dtype=float)
        eval_x = np.asarray(timing_eval[[name]], dtype=float)
        scores[name] = models["count"]().fit(fit_x, y).predict_proba(eval_x)[:, 1]
    return scores


# ---------------------------------------------------------------- DR-TabPFN fit, capture and checks

def fit_dr_tabpfn_context(X_context, y_context, years_context, *, domain_origin: int, device: str,
                          seed: int = SEED, use_context_scaler: bool = True) -> dict:
    """Fit single_fast DR-TabPFN on last_3 context rows; the context scaler is primary, historical scaling is the sensitivity."""
    import temporal_input_attribution as tia
    from tabpfn.best_models import TabPFNModelPathsConfig, get_best_tabpfn
    import tabpfn

    X = np.ascontiguousarray(np.asarray(X_context, dtype=np.float32))
    y = np.ascontiguousarray(np.asarray(y_context, dtype=np.float32))
    years = np.asarray(years_context, dtype=np.int64)
    if X.ndim != 2 or len(X) == 0 or set(np.unique(y).tolist()) != {0.0, 1.0}:
        raise ValueError("Context requires a 2-D feature matrix and both mortality classes {0, 1}")
    if years.min() < int(domain_origin):
        raise ValueError("Context years precede the domain origin")
    domains = np.ascontiguousarray(years - int(domain_origin), dtype=np.int64)
    precision = pilot.configure_predictor_precision(device)

    checkpoint = Path(str(resources.files(tabpfn))) / "model_cache" / "tabpfn_dist_model_1.cpkt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Installed checkpoint absent: {checkpoint}")
    checkpoint_sha = probe.file_sha256(checkpoint)
    paths = TabPFNModelPathsConfig(paths=[str(checkpoint)], task_type="dist_shift_multiclass")
    classifier = get_best_tabpfn(task_type="dist_shift_multiclass", model_type="single_fast",
                                 paths_config=paths, debug=False, device=device)
    classifier.seed = int(seed)
    classifier.fit_at_predict_time = False
    classifier.sklearn_compatible_precision = False
    classifier.fp16_inference = False
    classifier.normalize_with_test = False
    classifier.pilot_precision_ = precision
    classifier.pilot_checkpoint_sha256_ = checkpoint_sha
    classifier.pilot_domain_policy_ = ("context_minmax_domain_minus_origin" if use_context_scaler
                                       else "historical_cached_domain_minus_origin")
    if hasattr(classifier, "show_progress"):
        classifier.show_progress = False
    classifier.init_model_and_get_model_config()
    classifier = tia.prepare_context_classifier(classifier, use_context_scaler=bool(use_context_scaler))
    with pilot.predictor_sdpa_context(device):
        classifier.fit(X, y, additional_x={"dist_shift_domain": domains})
    if len(classifier.ensemble_configurations_) != 1 or len(classifier.cached_models_) != 1:
        raise RuntimeError("Expected exactly one frozen configuration/model for single_fast")
    if list(classifier.classes_) != [0, 1]:
        raise RuntimeError(f"Unexpected class order: {classifier.classes_}")
    subsample = classifier.ensemble_configurations_[0].subsample_samples_configuration
    subsample_rows = len(X) if subsample is None else len(subsample)
    if subsample_rows != len(X):
        raise RuntimeError(f"Silent context subsampling detected: {subsample_rows} != {len(X)}")
    return {
        "classifier": classifier,
        "device": device,
        "domain_origin": int(domain_origin),
        "use_context_scaler": bool(use_context_scaler),
        "context_identity": pilot.predictor_context_identity(classifier),
        "checkpoint_sha256": checkpoint_sha,
        "context_rows": int(len(X)),
        "subsample_rows": int(subsample_rows),
        "precision": precision,
    }


def parity_check(fitted: dict, X_probe, years_probe) -> dict:
    """Cached context predictions against matched joint execution (max |dp|, |dh| <= 1e-4)."""
    return probe.check_matched_joint_parity(fitted, X_probe, years_probe)


def capture_scored_rows(fitted: dict, X, years, *, batch_size: int) -> dict:
    """Same-call death probabilities (1-D, class 1), layers[11] embeddings and the head continuation for the rows."""
    captured = probe.capture_flchain_batches(fitted, X, years, batch_size=batch_size)
    class_probabilities = captured["probabilities"]
    if class_probabilities.ndim != 2 or class_probabilities.shape[1] != 2:
        raise ValueError(f"Expected (rows, 2) class probabilities, got {class_probabilities.shape}")
    captured["probabilities"] = class_probabilities[:, 1]
    return captured


def gradients_with_fd(captured: dict, *, grad_batch_size: int, fd_rows: int):
    """Exact death-probability gradients on layers[11] and the finite-difference report on the largest-norm rows."""
    return probe.compute_gradients_and_fd_check(captured["continuation"], captured["embeddings"],
                                                grad_batch_size=grad_batch_size, fd_rows=fd_rows)


def geometry_summary(embeddings, gradients, probabilities, sae_discovery_embeddings) -> dict:
    """Work 2 last-layer geometry with the standard deviation chart from sae_discovery embeddings."""
    scale, zero_columns = probe.standardization_scale(sae_discovery_embeddings)
    rng = np.random.default_rng(probe.DIRECTION_SEED)
    result = probe.analyze_layer_geometry(embeddings, gradients, probabilities, scale, rng=rng,
                                          n_random=probe.RANDOM_DIRECTIONS, top_axes=probe.TOP_AXES,
                                          interior=probe.INTERIOR_CUTS)
    result["standardization_zero_columns"] = zero_columns
    return result


def flatten_geometry(result: dict) -> pd.DataFrame:
    """Long table of geometry results: section, key (chart/scale/family), metric, value."""
    rows = []
    for section, block in result.items():
        if not isinstance(block, dict):
            rows.append((section, "", "value", block))
            continue
        for key, values in block.items():
            key_text = "|".join(str(part) for part in (key if isinstance(key, tuple) else (key,)))
            if isinstance(values, dict):
                rows.extend((section, key_text, metric, number) for metric, number in values.items())
            else:
                rows.append((section, key_text, "value", values))
    return pd.DataFrame(rows, columns=["section", "key", "metric", "value"])


# ---------------------------------------------------------------- Work 8a export

def export_splits_metadata(path, population, first_death_year_raw, cohort, roles_by_reference,
                           contexts_by_reference, thresholds_by_reference) -> None:
    """Write row masks, y_next, roles, first raw DEATH year, last_3 context masks and thresholds per reference year."""
    n = len(population["year"])
    rows = cohort["row"].to_numpy()
    in_cohort = np.zeros(n, dtype=bool)
    in_cohort[rows] = True
    outcome_valid = np.zeros(n, dtype=bool)
    outcome_valid[rows] = cohort["outcome_valid"].to_numpy(dtype=bool)
    y_next = np.full(n, np.nan)
    y_next[rows] = cohort["y_next"].to_numpy(dtype=float)
    arrays = {
        "year": np.asarray(population["year"], dtype=np.int64),
        "patient_id": np.asarray(population["patient_id"], dtype=str),
        "first_death_year_raw": np.asarray(first_death_year_raw, dtype=float),
        "in_cohort": in_cohort,
        "outcome_valid": outcome_valid,
        "y_next": y_next,
    }
    for reference in sorted(roles_by_reference):
        context = np.zeros(n, dtype=bool)
        context[rows[np.asarray(contexts_by_reference[reference], dtype=bool)]] = True
        arrays[f"role_{reference}"] = np.asarray(roles_by_reference[reference]["row_role"], dtype=str)
        arrays[f"last_3_context_{reference}"] = context
        arrays[f"threshold_{reference}"] = np.float64(thresholds_by_reference[reference])
    np.savez_compressed(Path(path), **arrays)
