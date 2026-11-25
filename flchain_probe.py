"""FLChain last-layer compression, gradient geometry, and year-input sensitivity probe (Roadmap Work 6b).

Exposes modular FLChain dataset/target/split loaders reusable by Work 8b alongside
DR-TabPFN context fitting, same-call capture, head-gradient geometry, paired-bootstrap
fixed-patient ``sample.yr`` sensitivity, and CPU reference baselines.
"""

from __future__ import annotations

import hashlib
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import tcav_geometry as geometry

EXPECTED_CSV_SHA256 = "6b9198b58354c2f2d45edaf4b38c237b9b9fcc22703a73baab75dd6f3d72606c"
FLCHAIN_FEATURES = ("age", "sex_male", "kappa", "lambda", "flc.grp", "creatinine", "mgus")
FORBIDDEN_FEATURES = ("futime", "chapter", "death", "sample.yr", "sex", "Unnamed: 0")
_REQUIRED_RAW_COLUMNS = (
    "age", "sex", "sample.yr", "kappa", "lambda", "flc.grp", "creatinine", "mgus", "futime", "death"
)

DOMAIN_ORIGIN = 1995
REFERENCE_YEAR = 1997
PRIMARY_HORIZON_DAYS = 1095
SENSITIVITY_HORIZON_DAYS = 1825
SPLIT_SEED = 42

BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 20261106
RANDOM_DIRECTIONS = 2000
TOP_AXES = 5
INTERIOR_CUTS = (0.1, 0.9)
DIRECTION_SEED = 20261124

SUS_WORK2_BENCHMARKS = {
    "raw_embedding_pr": (2.50, "1.90-3.21 (primary 7 builds, 2007-2014)"),
    "std_embedding_pr": (4.08, "2.80-4.92 (primary 7 builds, 2007-2014)"),
    "top5_var_raw": (0.935, "median top-5 PC variance share (raw)"),
    "top5_var_std": (0.875, "median top-5 PC variance share (standardized)"),
    "grad_resultant_length": (0.975, "0.966-0.992 (35 primary build-years)"),
    "grad_pr_raw_prob": (1.08, "centered 1.08 / uncentered 1.11 (raw prob)"),
    "grad_pr_std_prob": (1.03, "centered 1.03 / uncentered 1.03 (std prob)"),
    "random_tcav_interior_iso": (0.186, "all rows mean, isotropic (SE ~0.004-0.010)"),
    "random_tcav_interior_top5": (0.066, "all rows mean, top-5 span"),
    "random_tcav_interior_comp": (0.255, "all rows mean, orthogonal complement"),
}


def file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hex digest for a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_flchain_dataframe(
    csv_path: Path | str,
    expected_sha256: str | None = EXPECTED_CSV_SHA256,
) -> pd.DataFrame:
    """Load FLChain CSV, verify SHA-256 when provided, and validate raw schema without imputing NaNs."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"FLChain CSV not found: {path}")
    if expected_sha256 is not None:
        observed_sha = file_sha256(path)
        if observed_sha != expected_sha256:
            raise ValueError(f"FLChain CSV SHA-256 mismatch: expected {expected_sha256}, got {observed_sha}")

    df = pd.read_csv(path).reset_index(drop=True)
    missing = [col for col in _REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required FLChain columns: {missing}")

    if df["sex"].isna().any() or not set(df["sex"].unique()) <= {"F", "M"}:
        raise ValueError("FLChain sex column must contain only 'F' and 'M' with no missing values")
    if df["sample.yr"].isna().any():
        raise ValueError("FLChain sample.yr contains missing values")
    years = pd.to_numeric(df["sample.yr"], errors="raise")
    if not np.all(np.isfinite(years)) or not np.all(years == years.astype(np.int64)):
        raise ValueError("FLChain sample.yr must contain finite integer years")
    if df["death"].isna().any() or not set(df["death"].unique()) <= {0, 1}:
        raise ValueError("FLChain death column must contain only 0 and 1")
    futime = df["futime"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(futime)) or np.any(futime < 0):
        raise ValueError("FLChain futime must be finite and non-negative")

    out = df.copy()
    if "Unnamed: 0" in out.columns and out["Unnamed: 0"].is_unique:
        out["subject_id"] = out["Unnamed: 0"].astype(np.int64)
    else:
        out["subject_id"] = np.arange(len(out), dtype=np.int64)
    if not out["subject_id"].is_unique:
        raise ValueError("subject_id must be unique across FLChain rows")

    out["sample.yr"] = years.astype(np.int64)
    out["death"] = out["death"].astype(np.int64)
    out["sex_male"] = (out["sex"] == "M").astype(np.float32)
    for col in FLCHAIN_FEATURES:
        out[col] = out[col].astype(np.float32)
    return out


def extract_feature_matrix(
    df: pd.DataFrame,
    feature_names: tuple[str, ...] = FLCHAIN_FEATURES,
) -> np.ndarray:
    """Return contiguous float32 matrix of the 7 allowed FLChain clinical features."""
    names = tuple(feature_names)
    leaked = [col for col in names if col in FORBIDDEN_FEATURES]
    if leaked:
        raise ValueError(f"Forbidden leakage columns in feature list: {leaked}")
    if names != FLCHAIN_FEATURES:
        raise ValueError(f"Expected exact FLCHAIN_FEATURES {FLCHAIN_FEATURES}, got {names}")
    missing = [col for col in names if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in DataFrame: {missing}")
    return np.ascontiguousarray(df[list(names)].to_numpy(dtype=np.float32))


def censoring_audit_table(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (1095, 1825, 2555),
) -> pd.DataFrame:
    """Summarize right-censoring and event counts across sample.yr, 2002-2003, and all rows."""
    years = df["sample.yr"].to_numpy(dtype=np.int64)
    strata = [(str(y), years == y) for y in sorted(np.unique(years))]
    strata.append(("2002-2003", np.isin(years, [2002, 2003])))
    strata.append(("all", np.ones(len(df), dtype=bool)))

    rows = []
    death = df["death"].to_numpy(dtype=np.int64)
    futime = df["futime"].to_numpy(dtype=np.float64)
    for horizon in list(horizons) + [None]:
        target_name = f"horizon_{horizon}d" if horizon is not None else "raw_death"
        for stratum, mask in strata:
            total = int(mask.sum())
            if horizon is None:
                censored = 0
                kept = total
                events = int(death[mask].sum())
            else:
                censored_mask = mask & (death == 0) & (futime < horizon)
                kept_mask = mask & ~censored_mask
                censored = int(censored_mask.sum())
                kept = int(kept_mask.sum())
                events = int(((death == 1) & (futime <= horizon) & kept_mask).sum())
            rows.append({
                "target": target_name,
                "horizon_days": horizon,
                "stratum": stratum,
                "total_rows": total,
                "censored_rows": censored,
                "censored_share": float(censored / total) if total else np.nan,
                "kept_rows": kept,
                "events": events,
                "event_rate": float(events / kept) if kept else np.nan,
            })
    return pd.DataFrame(rows)


def build_flchain_target(df: pd.DataFrame, horizon_days: int | None) -> tuple[pd.DataFrame, np.ndarray]:
    """Construct uncensored fixed-horizon binary mortality target or unpooled raw death target."""
    death = df["death"].to_numpy(dtype=np.int64)
    if horizon_days is None:
        return df.reset_index(drop=True), death.copy()
    if int(horizon_days) <= 0:
        raise ValueError(f"horizon_days must be positive or None, got {horizon_days}")
    futime = df["futime"].to_numpy(dtype=np.float64)
    censored = (death == 0) & (futime < int(horizon_days))
    kept = df.loc[~censored].reset_index(drop=True)
    y = ((kept["death"].to_numpy(dtype=np.int64) == 1)
         & (kept["futime"].to_numpy(dtype=np.float64) <= int(horizon_days))).astype(np.int64)
    return kept, y


def split_flchain_chronological(
    df: pd.DataFrame,
    y: np.ndarray,
    *,
    domain_origin: int = DOMAIN_ORIGIN,
    reference_year: int = REFERENCE_YEAR,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Build deterministic chronological splits: 1995-1997 context (1997 50/25/25) and 1998-2003 future."""
    if len(df) != len(y):
        raise ValueError("DataFrame and target y lengths differ")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("Target y must be binary {0, 1}")

    years = df["sample.yr"].to_numpy(dtype=np.int64)
    if np.any(years < domain_origin):
        raise ValueError("Found rows preceding domain_origin")

    prior_idx = np.flatnonzero((years >= domain_origin) & (years < reference_year))
    ref_idx = np.flatnonzero(years == reference_year)
    if len(ref_idx) < 4 or len(np.unique(y[ref_idx])) < 2:
        raise ValueError("Reference year requires both outcome classes and >= 4 rows for 50/25/25 split")

    ref_context_idx, ref_hold_idx = train_test_split(
        ref_idx, test_size=0.5, stratify=y[ref_idx], random_state=int(seed)
    )
    ref_fit_idx, ref_eval_idx = train_test_split(
        ref_hold_idx, test_size=0.5, stratify=y[ref_hold_idx], random_state=int(seed)
    )

    context_idx = np.sort(np.concatenate([prior_idx, ref_context_idx])).astype(np.int64)
    ref_fit_idx = np.sort(ref_fit_idx).astype(np.int64)
    ref_eval_idx = np.sort(ref_eval_idx).astype(np.int64)
    future_idx = np.sort(np.flatnonzero(years > reference_year)).astype(np.int64)

    partitions = (context_idx, ref_fit_idx, ref_eval_idx, future_idx)
    total_rows = sum(len(part) for part in partitions)
    all_idx = np.concatenate(partitions)
    if total_rows != len(df) or len(np.unique(all_idx)) != len(df):
        raise AssertionError("Chronological splits are not a disjoint cover of the dataset")
    if "subject_id" in df.columns:
        subjects = df["subject_id"].to_numpy()
        if len(np.unique(subjects[all_idx])) != total_rows:
            raise AssertionError("Subject overlap across chronological splits")

    strata = {"1997_eval": ref_eval_idx}
    for yr in (1998, 1999, 2000, 2001):
        strata[str(yr)] = future_idx[years[future_idx] == yr]
    strata["2002-2003"] = future_idx[np.isin(years[future_idx], [2002, 2003])]
    strata["1998-2001"] = future_idx[np.isin(years[future_idx], [1998, 1999, 2000, 2001])]
    strata["1998-2003"] = future_idx

    return {
        "context_idx": context_idx,
        "ref_fit_idx": ref_fit_idx,
        "ref_eval_idx": ref_eval_idx,
        "future_idx": future_idx,
        "domain_origin": int(domain_origin),
        "reference_year": int(reference_year),
        "seed": int(seed),
        "strata": strata,
    }


def _make_domain_array(years: np.ndarray, domain_origin: int) -> np.ndarray:
    yrs = np.asarray(years, dtype=np.int64)
    if np.any(yrs < int(domain_origin)):
        raise ValueError(f"Years cannot precede domain_origin={domain_origin}")
    return np.ascontiguousarray(yrs - int(domain_origin), dtype=np.int64)


def fit_dr_tabpfn_flchain(
    X_context: np.ndarray,
    y_context: np.ndarray,
    years_context: np.ndarray,
    *,
    device: str = "cpu",
    seed: int = SPLIT_SEED,
    domain_origin: int = DOMAIN_ORIGIN,
    use_context_scaler: bool = True,
) -> dict[str, Any]:
    """Fit single_fast DR-TabPFN on trailing FLChain context rows with explicit temporal scaling."""
    import tabpfn
    from tabpfn.best_models import TabPFNModelPathsConfig, get_best_tabpfn
    import concept_subgroup_drift_pilot as pilot
    import temporal_input_attribution as tia

    X_c = np.ascontiguousarray(np.asarray(X_context, dtype=np.float32))
    y_c = np.ascontiguousarray(np.asarray(y_context, dtype=np.float32))
    if X_c.ndim != 2 or X_c.shape[1] != len(FLCHAIN_FEATURES):
        raise ValueError(f"Expected X_context shape (*, {len(FLCHAIN_FEATURES)}), got {X_c.shape}")
    if len(X_c) == 0 or set(np.unique(y_c).tolist()) != {0.0, 1.0}:
        raise ValueError("Context requires both mortality classes {0, 1}")

    domains_c = _make_domain_array(years_context, domain_origin)
    precision = pilot.configure_predictor_precision(device)

    checkpoint = Path(str(resources.files(tabpfn))) / "model_cache" / "tabpfn_dist_model_1.cpkt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Installed checkpoint absent: {checkpoint}")
    checkpoint_sha = file_sha256(checkpoint)

    paths = TabPFNModelPathsConfig(paths=[str(checkpoint)], task_type="dist_shift_multiclass")
    classifier = get_best_tabpfn(
        task_type="dist_shift_multiclass",
        model_type="single_fast",
        paths_config=paths,
        debug=False,
        device=device,
    )
    classifier.seed = int(seed)
    classifier.fit_at_predict_time = False
    classifier.sklearn_compatible_precision = False
    classifier.fp16_inference = False
    classifier.normalize_with_test = False
    classifier.pilot_precision_ = precision
    classifier.pilot_checkpoint_sha256_ = checkpoint_sha
    classifier.pilot_domain_policy_ = (
        "context_minmax_domain_minus_origin" if use_context_scaler else "historical_cached_domain_minus_origin"
    )
    if hasattr(classifier, "show_progress"):
        classifier.show_progress = False

    classifier.init_model_and_get_model_config()
    classifier = tia.prepare_context_classifier(classifier, use_context_scaler=bool(use_context_scaler))

    with pilot.predictor_sdpa_context(device):
        classifier.fit(X_c, y_c, additional_x={"dist_shift_domain": domains_c})

    if len(classifier.ensemble_configurations_) != 1 or len(classifier.cached_models_) != 1:
        raise RuntimeError("Expected exactly one frozen configuration/model for single_fast")
    if list(classifier.classes_) != [0, 1]:
        raise RuntimeError(f"Unexpected class order: {classifier.classes_}")

    subsample = classifier.ensemble_configurations_[0].subsample_samples_configuration
    subsample_rows = len(X_c) if subsample is None else len(subsample)
    if subsample_rows != len(X_c):
        raise RuntimeError(f"Silent context subsampling detected: {subsample_rows} != {len(X_c)}")

    return {
        "classifier": classifier,
        "device": device,
        "domain_origin": int(domain_origin),
        "use_context_scaler": bool(use_context_scaler),
        "context_identity": pilot.predictor_context_identity(classifier),
        "checkpoint_sha256": checkpoint_sha,
        "context_rows": int(len(X_c)),
        "subsample_rows": int(subsample_rows),
        "precision": precision,
    }


def capture_flchain_batches(
    fitted: dict[str, Any],
    X_query: np.ndarray,
    query_years: np.ndarray,
    *,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Capture same-call probabilities and layers[11] embeddings via capture_same_call."""
    import concept_subgroup_drift_pilot as pilot
    import tabpfn_model
    import temporal_input_attribution as tia

    classifier = fitted["classifier"]
    device = fitted["device"]
    X_q = np.ascontiguousarray(np.asarray(X_query, dtype=np.float32))
    domains = _make_domain_array(query_years, fitted["domain_origin"])
    if len(X_q) != len(domains):
        raise ValueError("X_query and query_years lengths differ")
    if len(X_q) == 0:
        raise ValueError("Cannot capture predictions on empty query matrix")

    additional_device, fitted_domain_shape = tabpfn_model.infer_model_additional_x_info(classifier)
    probs_chunks, emb_chunks = [], []
    max_replay = 0.0
    t_mins, t_maxs = [], []
    continuation = None

    def identity_fn():
        return pilot.predictor_context_identity(classifier)

    for start in range(0, len(X_q), batch_size):
        end = min(start + batch_size, len(X_q))
        xb = X_q[start:end]
        dom_tensor = tabpfn_model.make_dist_tensor(domains[start:end], additional_device, fitted_domain_shape)

        def predict_fn(feats=xb, dom=dom_tensor):
            with pilot.predictor_sdpa_context(device):
                return classifier.predict_full(
                    feats, additional_x={"dist_shift_domain": dom}, get_additional_outputs=["test_embeddings"]
                )

        batch_out = tia.capture_same_call(
            classifier, classifier.cached_models_[0], predict_fn, identity_fn, pilot._frozen_continuation
        )
        p_b = np.asarray(batch_out["probabilities"], dtype=np.float32)
        h_raw = np.asarray(batch_out["embeddings"], dtype=np.float32)
        if h_raw.ndim != 3 or h_raw.shape[1] != 1:
            raise RuntimeError(f"Expected (rows, 1, dim) embeddings, got {h_raw.shape}")
        h_b = np.ascontiguousarray(h_raw[:, 0, :], dtype=np.float32)
        if not np.isfinite(p_b).all() or not np.isfinite(h_b).all():
            raise RuntimeError("Nonfinite probabilities or embeddings observed")

        probs_chunks.append(p_b)
        emb_chunks.append(h_b)
        max_replay = max(max_replay, float(batch_out["replay_max_absolute_error"]))
        t_vals = batch_out["time_values"].detach().cpu().numpy()
        t_mins.append(float(np.min(t_vals)))
        t_maxs.append(float(np.max(t_vals)))
        continuation = batch_out["continuation"]

    return {
        "probabilities": np.vstack(probs_chunks),
        "embeddings": np.vstack(emb_chunks),
        "continuation": continuation,
        "max_replay_error": float(max_replay),
        "time_values_summary": {"min": min(t_mins), "max": max(t_maxs)},
    }


def check_matched_joint_parity(
    fitted: dict[str, Any],
    X_probe: np.ndarray,
    years_probe: np.ndarray,
) -> dict[str, Any]:
    """Compare cached ContextMinMaxScaler predictions/embeddings against matched_joint_execution."""
    import concept_subgroup_drift_pilot as pilot
    import tabpfn_model
    import temporal_input_attribution as tia

    cached = capture_flchain_batches(fitted, X_probe, years_probe, batch_size=len(X_probe))
    classifier = fitted["classifier"]
    device = fitted["device"]
    X_p = np.ascontiguousarray(np.asarray(X_probe, dtype=np.float32))
    domains = _make_domain_array(years_probe, fitted["domain_origin"])
    additional_device, fitted_domain_shape = tabpfn_model.infer_model_additional_x_info(classifier)
    dom_tensor = tabpfn_model.make_dist_tensor(domains, additional_device, fitted_domain_shape)

    def identity_fn():
        return pilot.predictor_context_identity(classifier)

    def predict_fn():
        with pilot.predictor_sdpa_context(device):
            return classifier.predict_full(
                X_p, additional_x={"dist_shift_domain": dom_tensor}, get_additional_outputs=["test_embeddings"]
            )

    with tia.matched_joint_execution(classifier):
        joint = tia.capture_same_call(
            classifier, classifier.model_processed_, predict_fn, identity_fn, pilot._frozen_continuation
        )

    p_diff = float(np.max(np.abs(cached["probabilities"] - np.asarray(joint["probabilities"], dtype=np.float32))))
    h_joint = np.asarray(joint["embeddings"], dtype=np.float32)[:, 0, :]
    h_diff = float(np.max(np.abs(cached["embeddings"] - h_joint)))
    return {
        "rows": int(len(X_p)),
        "proba_max_abs_diff": p_diff,
        "emb_max_abs_diff": h_diff,
        "passed": bool(p_diff <= 1e-4 and h_diff <= 1e-4),
    }


def compute_gradients_and_fd_check(
    continuation: Any,
    embeddings: np.ndarray,
    *,
    grad_batch_size: int = 512,
    fd_rows: int = 16,
    seed: int = DIRECTION_SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute exact death-probability gradients on layers[11] and verify with finite differences."""
    import concept_subgroup_drift_pilot as pilot

    gradients = pilot.batched_death_gradients(continuation, embeddings, batch_rows=int(grad_batch_size))
    n_check = min(int(fd_rows), len(embeddings))
    head_device = next(continuation.head.parameters()).device
    grad_norms = np.linalg.norm(gradients, axis=1)
    top_idx = np.argsort(grad_norms)[::-1][:n_check]
    g_probe = gradients[top_idx].astype(np.float64)
    g_norms = np.maximum(np.linalg.norm(g_probe, axis=1, keepdims=True), 1e-12)
    g_unit = g_probe / g_norms
    rng = np.random.default_rng(int(seed))
    rand_raw = rng.standard_normal((n_check, embeddings.shape[1]))
    rand_unit = rand_raw / np.linalg.norm(rand_raw, axis=1, keepdims=True)
    dirs_np = (g_unit + rand_unit).astype(np.float64)
    dirs_np = (dirs_np / np.linalg.norm(dirs_np, axis=1, keepdims=True)).astype(np.float32)
    h_probe = torch.tensor(embeddings[top_idx], dtype=torch.float32, device=head_device)
    d_probe = torch.tensor(dirs_np, dtype=torch.float32, device=head_device)
    fd_report = pilot._finite_difference_check(continuation, h_probe, d_probe, purpose="flchain_last_layer_fd")
    return gradients, fd_report


def binary_entropy_nats(probabilities: np.ndarray) -> np.ndarray:
    """Compute pointwise binary predictive entropy H(p) = -p log p - (1-p) log(1-p) in nats."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probs).all() or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    out = np.zeros_like(probs, dtype=np.float64)
    interior = (probs > 0.0) & (probs < 1.0)
    p = probs[interior]
    out[interior] = -p * np.log(p) - (1.0 - p) * np.log1p(-p)
    return out


def standardization_scale(ref_fit_embeddings: np.ndarray) -> tuple[np.ndarray, int]:
    """Columnwise SD on ref_fit embeddings (ddof=0); zeros replaced by 1.0 with count reported."""
    sd = np.std(np.asarray(ref_fit_embeddings, dtype=np.float64), axis=0, ddof=0)
    zero_mask = (sd == 0.0) | ~np.isfinite(sd)
    zero_cols = int(zero_mask.sum())
    scale = np.where(zero_mask, 1.0, sd)
    return scale, zero_cols


def analyze_layer_geometry(
    embeddings: np.ndarray,
    gradients: np.ndarray,
    probabilities: np.ndarray,
    ref_scale: np.ndarray,
    *,
    rng: np.random.Generator,
    n_random: int = RANDOM_DIRECTIONS,
    top_axes: int = TOP_AXES,
    interior: tuple[float, float] = INTERIOR_CUTS,
) -> dict[str, Any]:
    """Run Work 2 geometry and random-direction TCAV diagnostics on last-layer embeddings/gradients."""
    emb = np.asarray(embeddings, dtype=np.float64)
    grad_prob = np.asarray(gradients, dtype=np.float64)
    prob = np.asarray(probabilities, dtype=np.float64)
    grad_log_odds = geometry.log_odds_gradients(grad_prob, prob)
    scale = np.asarray(ref_scale, dtype=np.float64)

    spectrum = {}
    coherence_out = {}
    dimension_out = {}
    energy_out = {}
    random_out = {}

    for chart, chart_scale in (("raw", np.ones_like(scale)), ("standardized", scale)):
        chart_emb = emb / chart_scale
        _, axes, variances = geometry.principal_axes(chart_emb)
        pos_var = np.maximum(variances, 0.0)
        tot_var = float(pos_var.sum())
        cov = np.cov(chart_emb - chart_emb.mean(axis=0), rowvar=False)
        spectrum[chart] = {
            "n_rows": int(len(chart_emb)),
            "embedding_pr": float(geometry.participation_ratio(cov)) if tot_var > 0 else np.nan,
            "top1_var_share": float(pos_var[:1].sum() / tot_var) if tot_var > 0 else np.nan,
            "top5_var_share": float(pos_var[:5].sum() / tot_var) if tot_var > 0 else np.nan,
            "top10_var_share": float(pos_var[:10].sum() / tot_var) if tot_var > 0 else np.nan,
        }

        for gscale, gvals in (("probability", grad_prob), ("log_odds", grad_log_odds)):
            chart_grad = gvals * chart_scale
            coh = geometry.coherence(chart_grad)
            cosines = coh["cosines"]
            coherence_out[(chart, gscale)] = {
                "rows_used": int(coh["rows_used"]),
                "rows_dropped": int(coh["rows_dropped"]),
                "resultant_length": float(coh["resultant_length"]),
                "leading_share": float(coh["leading_share"]),
                "median_cosine": float(np.nanmedian(cosines)) if len(cosines) else np.nan,
                "p05_cosine": float(np.nanpercentile(cosines, 5)) if len(cosines) and np.isfinite(cosines).any() else np.nan,
            }
            centred, uncentred = geometry.gradient_moments(chart_grad)
            dimension_out[(chart, gscale)] = {
                "centered_pr": float(geometry.participation_ratio(centred)) if float(np.sum(centred * centred.T)) > 0 else np.nan,
                "uncentered_pr": float(geometry.participation_ratio(uncentred)) if float(np.sum(uncentred * uncentred.T)) > 0 else np.nan,
            }
            for k in (1, 5, 10):
                k_eff = min(k, axes.shape[1])
                energy_out[(chart, gscale, k)] = float(geometry.energy_share(chart_grad, axes, k_eff))

        chart_grad_prob = grad_prob * chart_scale
        finite_mask = np.isfinite(chart_grad_prob).all(axis=1)
        g_fin = chart_grad_prob[finite_mask]
        all_rows = np.ones((len(g_fin), 1), dtype=bool)
        top_eff = min(int(top_axes), axes.shape[1])
        for family in geometry.DIRECTION_FAMILIES:
            dirs = geometry.random_directions(rng, int(n_random), axes, top_eff, family)
            tcav = geometry.positive_share(g_fin @ dirs.T, all_rows)[0]
            share, se = geometry.interior_share(tcav, low=interior[0], high=interior[1])
            fin_tcav = tcav[np.isfinite(tcav)]
            random_out[(chart, family)] = {
                "n_draws": int(len(fin_tcav)),
                "interior_share": float(share),
                "interior_se": float(se),
                "exact_binary_share": float(((fin_tcav == 0.0) | (fin_tcav == 1.0)).mean()) if len(fin_tcav) else np.nan,
                "mean_tcav": float(fin_tcav.mean()) if len(fin_tcav) else np.nan,
            }

    return {
        "spectrum": spectrum,
        "coherence": coherence_out,
        "dimension": dimension_out,
        "energy": energy_out,
        "random": random_out,
    }


def _safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def _bootstrap_summary(values: np.ndarray, point: float) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if not np.isfinite(point) or len(finite) == 0:
        return {"estimate": float(point), "ci_low": np.nan, "ci_high": np.nan, "valid_draws": 0}
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return {
        "estimate": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "valid_draws": int(len(finite)),
    }


def paired_bootstrap_sensitivity(
    p_actual: np.ndarray,
    p_held: np.ndarray,
    y: np.ndarray,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired subject-level bootstrap comparing actual vs held sample.yr predictions."""
    pa = np.asarray(p_actual, dtype=np.float64)
    ph = np.asarray(p_held, dtype=np.float64)
    yt = np.asarray(y, dtype=np.int64)
    if pa.shape != ph.shape or pa.shape != yt.shape or pa.ndim != 1:
        raise ValueError("p_actual, p_held, and y must share 1-D shape")

    n = len(yt)
    ea = binary_entropy_nats(pa)
    eh = binary_entropy_nats(ph)
    ba = (pa - yt) ** 2
    bh = (ph - yt) ** 2

    rng = np.random.default_rng(int(seed))
    idx_draws = rng.integers(0, n, size=(int(draws), n))

    def _boot_pair(va: np.ndarray, vh: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        diff = va - vh
        return (
            _bootstrap_summary(va[idx_draws].mean(axis=1), float(va.mean())),
            _bootstrap_summary(vh[idx_draws].mean(axis=1), float(vh.mean())),
            _bootstrap_summary(diff[idx_draws].mean(axis=1), float(diff.mean())),
        )

    ent_a, ent_h, ent_d = _boot_pair(ea, eh)
    prb_a, prb_h, prb_d = _boot_pair(pa, ph)
    bri_a, bri_h, bri_d = _boot_pair(ba, bh)

    auc_a_pt, auc_h_pt = _safe_auroc(yt, pa), _safe_auroc(yt, ph)
    auc_d_pt = auc_a_pt - auc_h_pt if np.isfinite(auc_a_pt) and np.isfinite(auc_h_pt) else np.nan
    auc_a_draws = np.array([_safe_auroc(yt[idx], pa[idx]) for idx in idx_draws], dtype=np.float64)
    auc_h_draws = np.array([_safe_auroc(yt[idx], ph[idx]) for idx in idx_draws], dtype=np.float64)

    return {
        "n_rows": int(n),
        "n_events": int(yt.sum()),
        "mean_entropy_actual": ent_a,
        "mean_entropy_held": ent_h,
        "mean_entropy_diff": ent_d,
        "mean_probability_actual": prb_a,
        "mean_probability_held": prb_h,
        "mean_probability_diff": prb_d,
        "brier_actual": bri_a,
        "brier_held": bri_h,
        "brier_diff": bri_d,
        "auroc_actual": _bootstrap_summary(auc_a_draws, auc_a_pt),
        "auroc_held": _bootstrap_summary(auc_h_draws, auc_h_pt),
        "auroc_diff": _bootstrap_summary(auc_a_draws - auc_h_draws, auc_d_pt),
    }


def evaluate_probabilities_bootstrap(
    p: np.ndarray,
    y: np.ndarray,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute AUROC, Brier, mean probability, and mean entropy with 95% subject-level bootstrap CIs."""
    probs = np.asarray(p, dtype=np.float64)
    yt = np.asarray(y, dtype=np.int64)
    if probs.shape != yt.shape or probs.ndim != 1:
        raise ValueError("Probabilities and labels must share 1-D shape")
    n = len(yt)
    ent = binary_entropy_nats(probs)
    brier_vec = (probs - yt) ** 2
    rng = np.random.default_rng(int(seed))
    idx_draws = rng.integers(0, n, size=(int(draws), n))

    auc_pt = _safe_auroc(yt, probs)
    auc_draws = np.array([_safe_auroc(yt[idx], probs[idx]) for idx in idx_draws], dtype=np.float64)
    auc_stat = _bootstrap_summary(auc_draws, auc_pt)
    brier_stat = _bootstrap_summary(brier_vec[idx_draws].mean(axis=1), float(brier_vec.mean()))
    prob_stat = _bootstrap_summary(probs[idx_draws].mean(axis=1), float(probs.mean()))
    ent_stat = _bootstrap_summary(ent[idx_draws].mean(axis=1), float(ent.mean()))

    return {
        "n_rows": int(n),
        "n_events": int(yt.sum()),
        "event_rate": float(yt.mean()) if n else np.nan,
        "auroc": auc_stat["estimate"],
        "auroc_ci_low": auc_stat["ci_low"],
        "auroc_ci_high": auc_stat["ci_high"],
        "auroc_valid_draws": auc_stat["valid_draws"],
        "brier": brier_stat["estimate"],
        "brier_ci_low": brier_stat["ci_low"],
        "brier_ci_high": brier_stat["ci_high"],
        "mean_probability": prob_stat["estimate"],
        "mean_probability_ci_low": prob_stat["ci_low"],
        "mean_probability_ci_high": prob_stat["ci_high"],
        "mean_entropy": ent_stat["estimate"],
        "mean_entropy_ci_low": ent_stat["ci_low"],
        "mean_entropy_ci_high": ent_stat["ci_high"],
    }


def fit_and_score_cpu_baselines(
    X_context: np.ndarray,
    y_context: np.ndarray,
    eval_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    seed: int = SPLIT_SEED,
    draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Fit LogisticRegression and HistGradientBoostingClassifier on context rows and evaluate strata."""
    models = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=int(seed))),
        ]),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=int(seed)),
    }
    rows = []
    for model_name, est in models.items():
        est.fit(X_context, y_context)
        for stratum, (X_eval, y_eval) in eval_splits.items():
            p_eval = est.predict_proba(X_eval)[:, 1]
            metrics = evaluate_probabilities_bootstrap(p_eval, y_eval, draws=draws, seed=bootstrap_seed)
            rows.append({"model": model_name, "stratum": stratum, **metrics})
    return pd.DataFrame(rows)


def build_sus_comparison_table(flchain_summary: dict[str, Any]) -> pd.DataFrame:
    """Tabulate FLChain last-layer metrics side-by-side with verified SUS CKD Work 2 benchmarks."""
    caveat = (
        "Cross-dataset differences are multi-factor: 7 dense labs vs 500 sparse event counts, "
        "fixed 3-yr horizon vs same-year mortality, Olmsted 1995-2003 vs SUS 1997-2015, "
        "single context (~5,390 rows) vs 7 trailing-window SUS builds."
    )
    metric_specs = [
        ("raw_embedding_pr", "Last-layer embedding PR (raw)"),
        ("std_embedding_pr", "Last-layer embedding PR (standardized)"),
        ("top5_var_raw", "Top-5 PC variance share (raw)"),
        ("top5_var_std", "Top-5 PC variance share (standardized)"),
        ("grad_resultant_length", "Head-gradient resultant length R (raw prob)"),
        ("grad_pr_raw_prob", "Head-gradient centered PR (raw prob)"),
        ("grad_pr_std_prob", "Head-gradient centered PR (std prob)"),
        ("random_tcav_interior_iso", "Random-direction TCAV interior share (raw, isotropic)"),
        ("random_tcav_interior_top5", "Random-direction TCAV interior share (raw, top-5)"),
        ("random_tcav_interior_comp", "Random-direction TCAV interior share (raw, complement)"),
    ]
    rows = []
    for key, label in metric_specs:
        sus_val, sus_note = SUS_WORK2_BENCHMARKS[key]
        rows.append({
            "metric_key": key,
            "metric": label,
            "flchain_1997_eval": flchain_summary.get("1997_eval", {}).get(key, np.nan),
            "flchain_1998_2003_pooled": flchain_summary.get("1998-2003", {}).get(key, np.nan),
            "sus_ckd_work2_reference": sus_val,
            "sus_ckd_work2_note": sus_note,
            "scope_caveat": caveat,
        })
    return pd.DataFrame(rows)
