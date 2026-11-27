"""Outcome-grounded calibration audit and Work 8 layer-geometry diagnostic helpers.

Implements frozen-cutoff and threshold-free calibration audits (paired patient bootstrap and Work 7
Scope A summary reconciliation), fixed-axis and aligned-refit partial PC1-by-year analyses
controlling for linear and natural-spline predicted logit, exact-RNG continuous TCAV saturation,
analytical isotropic gradient null floors, non-nested dispersion summaries, correlation-matrix
input-feature participation ratios with column-permutation nulls, and an aggregate-only SUS
notebook output guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

import flchain_probe as probe
import layer_compression as lc
import prospective_audit as pa
import tcav_geometry as geometry

BOOTSTRAP_DRAWS = 500
BOOTSTRAP_SEED = 20261106
DIRECTION_COUNT = 2000
DIRECTION_SEED = 20261124
PERM_NULL_DRAWS = 200
PERM_NULL_SEED = 20261126
SUBSAMPLE_N_CONTRAST = 168
SUBSAMPLE_N_SMALL = 46
SUBSAMPLE_REPEATS = 20
SUBSAMPLE_SEED = 20261127
LOGIT_SPLINE_DF = 4
AXIS_ALIGNMENT_MIN_COS = 0.9
NEWTON_MAX_ITER = 25
NEWTON_TOL = 1e-8
NEWTON_MAX_ABS_COEF = 50.0
PROB_CLIP_EPS = float(np.finfo(float).eps)
INTERIOR_LOW = 0.1
INTERIOR_HIGH = 0.9
TOP_K = 5
SUS_LAST_OUTCOME_YEAR = 2014
POPULATION_SHA256 = "c8804591c93087241b9533dce77dd54b2f1e77b23a36400527459cfa8f99add7"
FLCHAIN_CSV_SHA256 = "6b9198b58354c2f2d45edaf4b38c237b9b9fcc22703a73baab75dd6f3d72606c"
PARITY_ATOL = 1e-6
RNG_PARITY_ATOL = 0.0
R_IDENTITY_ATOL = 1e-12


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else np.nan


def newton_recalibration_slope_intercept(
    labels,
    probabilities,
    weights,
    *,
    max_iter: int = NEWTON_MAX_ITER,
    tol: float = NEWTON_TOL,
    max_abs_coef: float = NEWTON_MAX_ABS_COEF,
) -> tuple[float, float, str]:
    """Fast weighted 1-D Newton-Raphson logistic recalibration with separation/convergence guards."""
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = w > 0
    if not np.any(mask):
        return np.nan, np.nan, "single_class"
    ym, pm, wm = y[mask], p[mask], w[mask]
    if wm[ym == 1.0].sum() <= 0 or wm[ym == 0.0].sum() <= 0:
        return np.nan, np.nan, "single_class"

    clipped = np.clip(pm, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)
    x = np.log(clipped / (1.0 - clipped))
    if not np.isfinite(x).all() or np.ptp(x) == 0.0:
        return np.nan, np.nan, "non_convergent"

    auc = float(roc_auc_score(ym, x, sample_weight=wm))
    if auc <= 0.0 or auc >= 1.0:
        return np.nan, np.nan, "separated"

    b0 = 0.0
    b1 = 1.0
    converged = False
    for _ in range(int(max_iter)):
        eta = np.clip(b0 + b1 * x, -max_abs_coef, max_abs_coef)
        q = 1.0 / (1.0 + np.exp(-eta))
        res = q - ym
        g0 = float(np.sum(wm * res))
        g1 = float(np.sum(wm * res * x))
        v = wm * q * (1.0 - q)
        h00 = float(np.sum(v))
        h01 = float(np.sum(v * x))
        h11 = float(np.sum(v * x * x))
        det = h00 * h11 - h01 * h01
        trace = h00 + h11
        if not np.isfinite(det) or det <= 1e-14 * max(trace * trace, 1e-30):
            return np.nan, np.nan, "non_convergent"
        d0 = (h11 * g0 - h01 * g1) / det
        d1 = (h00 * g1 - h01 * g0) / det
        if not (np.isfinite(d0) and np.isfinite(d1)):
            return np.nan, np.nan, "non_convergent"

        cur_loss = float(np.sum(wm * (np.logaddexp(0.0, b0 + b1 * x) - ym * (b0 + b1 * x))))
        step = 1.0
        accepted = False
        while step >= 2.0**-12:
            nb0 = b0 - step * d0
            nb1 = b1 - step * d1
            if max(abs(nb0), abs(nb1)) <= max_abs_coef * 1.5:
                new_eta = nb0 + nb1 * x
                new_loss = float(np.sum(wm * (np.logaddexp(0.0, new_eta) - ym * new_eta)))
                if np.isfinite(new_loss) and new_loss <= cur_loss + 1e-12 * max(1.0, abs(cur_loss)):
                    b0, b1 = nb0, nb1
                    accepted = True
                    break
            step *= 0.5
        if not accepted:
            return np.nan, np.nan, "non_convergent"
        if max(abs(step * d0), abs(step * d1)) < tol:
            converged = True
            break

    if not converged or not (np.isfinite(b0) and np.isfinite(b1)) or max(abs(b0), abs(b1)) > max_abs_coef:
        return np.nan, np.nan, "non_convergent"
    return float(b1), float(b0), "ok"


def calibration_metrics(
    labels,
    probabilities,
    weights,
    *,
    base_rate: float,
    threshold: float | None = None,
) -> dict:
    """Weighted discrimination, outcome-grounded calibration, and optional threshold metrics."""
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    pos_w = w > 0
    total = float(w.sum())
    if total <= 0:
        raise ValueError("Total bootstrap weight must be positive")

    death = y == 1.0
    deaths = float(w[death].sum())
    survivors = float(w[~death].sum())
    obs_rate = _ratio(deaths, total)
    mean_prob = _ratio(float(np.sum(w * p)), total)
    citl = mean_prob - obs_rate

    brier = _ratio(float(np.sum(w * (p - y) ** 2)), total)
    brier_ref = _ratio(float(np.sum(w * (float(base_rate) - y) ** 2)), total)
    brier_skill = 1.0 - brier / brier_ref if brier_ref > 0 else np.nan

    clipped = np.clip(p, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)
    row_ll = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
    log_loss = _ratio(float(np.sum(w * row_ll)), total)
    clipped_rows = int(np.sum(pos_w & ((p < PROB_CLIP_EPS) | (p > 1.0 - PROB_CLIP_EPS))))
    max_row_log_loss = float(np.max(row_ll[pos_w])) if np.any(pos_w) else np.nan

    slope, intercept, recal_status = newton_recalibration_slope_intercept(y, p, w)
    has_both = deaths > 0 and survivors > 0
    out = {
        "rows": total,
        "deaths": deaths,
        "observed_rate": obs_rate,
        "mean_probability": mean_prob,
        "calibration_in_the_large": citl,
        "abs_calibration_in_the_large": abs(citl),
        "brier": brier,
        "brier_skill": brier_skill,
        "log_loss": log_loss,
        "recalibration_slope": slope,
        "recalibration_intercept": intercept,
        "recalibration_status": recal_status,
        "mean_entropy_nats": _ratio(float(np.sum(w * probe.binary_entropy_nats(p))), total),
        "auroc": float(roc_auc_score(y[pos_w], p[pos_w], sample_weight=w[pos_w])) if has_both else np.nan,
        "pr_auc": float(average_precision_score(y[pos_w], p[pos_w], sample_weight=w[pos_w])) if deaths > 0 else np.nan,
        "clipped_rows": clipped_rows,
        "max_row_log_loss": max_row_log_loss,
    }
    if threshold is not None:
        flagged = p >= float(threshold)
        tp = float(w[death & flagged].sum())
        fp = float(w[(~death) & flagged].sum())
        fn = float(w[death & (~flagged)].sum())
        out.update({
            "recall": _ratio(tp, deaths),
            "false_alarm_rate": _ratio(fp, survivors),
            "share_flagged": _ratio(tp + fp, total),
            "caught_deaths": tp,
            "false_alarms": fp,
            "death_f1": _ratio(2.0 * tp, 2.0 * tp + fp + fn),
        })
    return out


def paired_calibration_bootstrap(
    labels,
    p_actual,
    p_held,
    patient_ids,
    *,
    base_rate: float,
    threshold: float | None = None,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Paired patient-bootstrap intervals across actual, held, and actual-minus-held."""
    y = np.asarray(labels, dtype=np.float64)
    pa_arr = np.asarray(p_actual, dtype=np.float64)
    ph_arr = np.asarray(p_held, dtype=np.float64)
    pids = np.asarray(patient_ids, dtype=str)
    if not (len(y) == len(pa_arr) == len(ph_arr) == len(pids)):
        raise ValueError("labels, p_actual, p_held, and patient_ids must align")

    universe, counts = pa.patient_draw_counts(pids, draws=draws, seed=seed)
    codes = pa.row_patient_codes(pids, universe)
    ones = np.ones(len(y), dtype=np.float64)

    pt_act = calibration_metrics(y, pa_arr, ones, base_rate=base_rate, threshold=threshold)
    pt_held = calibration_metrics(y, ph_arr, ones, base_rate=base_rate, threshold=threshold)

    skip_keys = {"recalibration_status", "clipped_rows", "max_row_log_loss"}
    metric_names = [k for k in pt_act if k not in skip_keys]

    act_draws = {m: np.empty(draws, dtype=np.float64) for m in metric_names}
    held_draws = {m: np.empty(draws, dtype=np.float64) for m in metric_names}
    ok_act = np.zeros(draws, dtype=bool)
    ok_held = np.zeros(draws, dtype=bool)

    with threadpool_limits(limits=1):
        for i in range(int(draws)):
            w = counts[i, codes].astype(np.float64)
            ma = calibration_metrics(y, pa_arr, w, base_rate=base_rate, threshold=threshold)
            mh = calibration_metrics(y, ph_arr, w, base_rate=base_rate, threshold=threshold)
            ok_act[i] = ma["recalibration_status"] == "ok"
            ok_held[i] = mh["recalibration_status"] == "ok"
            for m in metric_names:
                act_draws[m][i] = ma[m]
                held_draws[m][i] = mh[m]

    recal_metrics = {"recalibration_slope", "recalibration_intercept"}
    rows = []
    for quantity, pt_map, draw_map, ok_mask, clip_cnt, max_ll in (
        ("actual", pt_act, act_draws, ok_act, pt_act["clipped_rows"], pt_act["max_row_log_loss"]),
        ("held", pt_held, held_draws, ok_held, pt_held["clipped_rows"], pt_held["max_row_log_loss"]),
        (
            "actual_minus_held",
            {m: pt_act[m] - pt_held[m] for m in metric_names},
            {m: act_draws[m] - held_draws[m] for m in metric_names},
            ok_act & ok_held,
            max(pt_act["clipped_rows"], pt_held["clipped_rows"]),
            max(pt_act["max_row_log_loss"], pt_held["max_row_log_loss"]),
        ),
    ):
        non_conv = int(draws - np.sum(ok_mask))
        for m in metric_names:
            arr = draw_map[m]
            finite = arr[np.isfinite(arr)]
            est = float(pt_map[m])
            if np.isfinite(est) and len(finite) > 0:
                lo, hi = np.percentile(finite, [2.5, 97.5])
            else:
                lo, hi = np.nan, np.nan
            rows.append({
                "quantity": quantity,
                "metric": m,
                "estimate": est,
                "ci_low": float(lo),
                "ci_high": float(hi),
                "valid_draws": int(len(finite)),
                "non_convergent_or_separated_draws": non_conv if m in recal_metrics else 0,
                "clipped_rows": int(clip_cnt),
                "max_row_log_loss": float(max_ll),
                "status": "ok",
            })
    return pd.DataFrame(rows)


def reconcile_prospective_scope_a_calibration(
    perf_df: pd.DataFrame,
    sens_df: pd.DataFrame,
) -> pd.DataFrame:
    """Reconcile Work 7 Scope A aggregate CSVs into actual, recovered-held, and diff rows."""
    perf = perf_df.copy()
    perf["input_year_str"] = perf["input_year"].astype(str)
    sens = sens_df.loc[sens_df["comparison"] == "actual_minus_held_at_reference"].copy()
    sens["input_year_str"] = sens["input_year"].astype(str)

    key_cols = ["reference_year", "scaling", "input_year_str", "stratum", "metric"]
    merged = perf.merge(
        sens[key_cols + ["estimate", "ci_low", "ci_high", "valid_draws"]],
        on=key_cols,
        how="left",
        suffixes=("_actual", "_diff"),
    )

    out_rows = []
    for _, row in merged.iterrows():
        ref_y = row["reference_year"]
        in_y = row["input_year"]
        in_str = row["input_year_str"]
        is_ref = in_str == str(ref_y)
        sc = row["scaling"]
        st = row["stratum"]
        m = row["metric"]
        est_act = float(row["estimate_actual"])
        out_rows.append({
            "reference_year": ref_y, "scaling": sc, "input_year": in_y, "stratum": st,
            "is_reference_year_contrast": is_ref, "quantity": "actual", "metric": m,
            "estimate": est_act, "ci_low": float(row["ci_low_actual"]),
            "ci_high": float(row["ci_high_actual"]),
            "valid_draws": int(row["valid_draws_actual"]), "status": "ok",
        })
        has_diff = pd.notna(row["estimate_diff"])
        est_diff = float(row["estimate_diff"]) if has_diff else np.nan
        out_rows.append({
            "reference_year": ref_y, "scaling": sc, "input_year": in_y, "stratum": st,
            "is_reference_year_contrast": is_ref, "quantity": "held", "metric": m,
            "estimate": (est_act - est_diff) if has_diff else np.nan,
            "ci_low": np.nan, "ci_high": np.nan, "valid_draws": 0,
            "status": "point_only_recovered_from_csv" if has_diff else "unavailable_in_source_csv",
        })
        out_rows.append({
            "reference_year": ref_y, "scaling": sc, "input_year": in_y, "stratum": st,
            "is_reference_year_contrast": is_ref, "quantity": "actual_minus_held", "metric": m,
            "estimate": est_diff,
            "ci_low": float(row["ci_low_diff"]) if has_diff else np.nan,
            "ci_high": float(row["ci_high_diff"]) if has_diff else np.nan,
            "valid_draws": int(row["valid_draws_diff"]) if pd.notna(row["valid_draws_diff"]) else 0,
            "status": "ok" if has_diff else "unavailable_in_source_csv",
        })

    # Derived per-group observed_rate, calibration_in_the_large, abs_calibration_in_the_large, log_loss
    base_df = pd.DataFrame(out_rows)
    group_cols = ["reference_year", "scaling", "input_year", "stratum", "is_reference_year_contrast"]
    derived_rows = []
    for gkey, grp in base_df.groupby(group_cols, sort=False, dropna=False):
        ref_y, sc, in_y, st, is_ref = gkey
        lookup = {(r["quantity"], r["metric"]): r for _, r in grp.iterrows()}
        rows_est = lookup.get(("actual", "rows"), {}).get("estimate", np.nan)
        deaths_est = lookup.get(("actual", "deaths"), {}).get("estimate", np.nan)
        obs_rate = _ratio(deaths_est, rows_est)
        mp_act = lookup.get(("actual", "mean_probability"), {}).get("estimate", np.nan)
        mp_held = lookup.get(("held", "mean_probability"), {}).get("estimate", np.nan)
        mp_diff_row = lookup.get(("actual_minus_held", "mean_probability"), None)

        derived_rows.extend([
            {
                "reference_year": ref_y, "scaling": sc, "input_year": in_y, "stratum": st,
                "is_reference_year_contrast": is_ref, "quantity": "actual", "metric": "observed_rate",
                "estimate": obs_rate, "ci_low": np.nan, "ci_high": np.nan, "valid_draws": 0,
                "status": "point_only_recovered_from_csv",
            },
            {
                "reference_year": ref_y, "scaling": sc, "input_year": in_y, "stratum": st,
                "is_reference_year_contrast": is_ref, "quantity": "held", "metric": "observed_rate",
                "estimate": obs_rate, "ci_low": np.nan, "ci_high": np.nan, "valid_draws": 0,
                "status": "point_only_recovered_from_csv",
            },
            {
                "reference_year": ref_y, "scaling": sc, "input_year": in_y, "stratum": st,
                "is_reference_year_contrast": is_ref, "quantity": "actual_minus_held", "metric": "observed_rate",
                "estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "valid_draws": int(mp_diff_row["valid_draws"]) if mp_diff_row is not None else 0, "status": "ok",
            },
        ])

        citl_act = mp_act - obs_rate
        citl_held = mp_held - obs_rate
        derived_rows.extend([
            {
                "reference_year": ref_y, "scaling": sc, "input_year": in_y, "stratum": st,
                "is_reference_year_contrast": is_ref, "quantity": "actual", "metric": "calibration_in_the_large",
                "estimate": citl_act, "ci_low": np.nan, "ci_high": np.nan, "valid_draws": 0,
                "status": "level_ci_unavailable_no_joint_draws",
            },
            {
                "reference_year": ref_y, "scaling": sc, "input_year": in_y, "stratum": st,
                "is_reference_year_contrast": is_ref, "quantity": "held", "metric": "calibration_in_the_large",
                "estimate": citl_held, "ci_low": np.nan, "ci_high": np.nan, "valid_draws": 0,
                "status": "level_ci_unavailable_no_joint_draws",
            },
            {
                "reference_year": ref_y, "scaling": sc, "input_year": in_y, "stratum": st,
                "is_reference_year_contrast": is_ref, "quantity": "actual_minus_held", "metric": "calibration_in_the_large",
                "estimate": float(mp_diff_row["estimate"]) if mp_diff_row is not None else np.nan,
                "ci_low": float(mp_diff_row["ci_low"]) if mp_diff_row is not None else np.nan,
                "ci_high": float(mp_diff_row["ci_high"]) if mp_diff_row is not None else np.nan,
                "valid_draws": int(mp_diff_row["valid_draws"]) if mp_diff_row is not None else 0,
                "status": str(mp_diff_row["status"]) if mp_diff_row is not None else "unavailable_in_source_csv",
            },
            {
                "reference_year": ref_y, "scaling": sc, "input_year": in_y, "stratum": st,
                "is_reference_year_contrast": is_ref, "quantity": "actual", "metric": "abs_calibration_in_the_large",
                "estimate": abs(citl_act), "ci_low": np.nan, "ci_high": np.nan, "valid_draws": 0,
                "status": "level_ci_unavailable_no_joint_draws",
            },
            {
                "reference_year": ref_y, "scaling": sc, "input_year": in_y, "stratum": st,
                "is_reference_year_contrast": is_ref, "quantity": "held", "metric": "abs_calibration_in_the_large",
                "estimate": abs(citl_held), "ci_low": np.nan, "ci_high": np.nan, "valid_draws": 0,
                "status": "level_ci_unavailable_no_joint_draws",
            },
            {
                "reference_year": ref_y, "scaling": sc, "input_year": in_y, "stratum": st,
                "is_reference_year_contrast": is_ref, "quantity": "actual_minus_held", "metric": "abs_calibration_in_the_large",
                "estimate": abs(citl_act) - abs(citl_held), "ci_low": np.nan, "ci_high": np.nan, "valid_draws": 0,
                "status": "level_ci_unavailable_no_joint_draws",
            },
        ])

        for q in ("actual", "held", "actual_minus_held"):
            derived_rows.append({
                "reference_year": ref_y, "scaling": sc, "input_year": in_y, "stratum": st,
                "is_reference_year_contrast": is_ref, "quantity": q, "metric": "log_loss",
                "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "valid_draws": 0,
                "status": "unavailable_no_per_row_probs",
            })

    return pd.concat([base_df, pd.DataFrame(derived_rows)], ignore_index=True)


def natural_cubic_spline_basis(x, df: int = LOGIT_SPLINE_DF) -> np.ndarray:
    """Pure NumPy natural cubic spline basis without intercept (linear beyond boundary knots)."""
    if int(df) < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    vals = np.asarray(x, dtype=np.float64)
    if vals.ndim != 1:
        raise ValueError("x must be 1-D")
    span = float(np.ptp(vals))
    if span == 0.0:
        return np.zeros((len(vals), 0), dtype=np.float64)
    if int(df) == 1:
        return vals[:, None]

    k_min, k_max = float(np.min(vals)), float(np.max(vals))
    raw_interior = np.quantile(vals, np.linspace(0.0, 1.0, int(df))[1:-1])
    interior = np.unique(raw_interior[(raw_interior > k_min) & (raw_interior < k_max)])
    knots = np.concatenate([[k_min], interior, [k_max]])
    n_knots = len(knots)
    if n_knots <= 2:
        return vals[:, None]

    def _d_term(knot_idx: int) -> np.ndarray:
        k_j = knots[knot_idx]
        return (np.maximum(0.0, vals - k_j) ** 3 - np.maximum(0.0, vals - k_max) ** 3) / (k_max - k_j)

    d_last = _d_term(n_knots - 2)
    cols = [vals] + [_d_term(j) - d_last for j in range(n_knots - 2)]
    return np.column_stack(cols)


def weighted_ols_r2(y, X, weights) -> float:
    """Weighted least-squares coefficient of determination R^2 with explicit intercept."""
    target = np.asarray(y, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = w > 0
    if not np.any(mask):
        return np.nan
    ym = target[mask]
    wm = w[mask]
    y_mean = float(np.sum(wm * ym) / np.sum(wm))
    sst = float(np.sum(wm * (ym - y_mean) ** 2))
    if sst <= 0.0:
        return np.nan

    design = np.asarray(X, dtype=np.float64)
    if design.ndim == 1:
        design = design[:, None]
    if design.shape[1] == 0:
        return 0.0
    dm = design[mask]
    sqrt_w = np.sqrt(wm)[:, None]
    A = np.column_stack([np.ones(len(ym), dtype=np.float64), dm]) * sqrt_w
    b = ym * sqrt_w[:, 0]
    coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    resid = ym - (coef[0] + dm @ coef[1:])
    sse = float(np.sum(wm * resid * resid))
    return float(np.clip(1.0 - sse / sst, 0.0, 1.0))


def _weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    mask = w > 0
    xm, ym, wm = x[mask], y[mask], w[mask]
    wsum = float(wm.sum())
    if wsum <= 0.0:
        return np.nan
    mx = float(np.sum(wm * xm) / wsum)
    my = float(np.sum(wm * ym) / wsum)
    vx = float(np.sum(wm * (xm - mx) ** 2))
    vy = float(np.sum(wm * (ym - my) ** 2))
    if vx <= 0.0 or vy <= 0.0:
        return np.nan
    return float(np.sum(wm * (xm - mx) * (ym - my)) / np.sqrt(vx * vy))


def _weighted_residuals(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> np.ndarray:
    mask = w > 0
    ym, dm, wm = y[mask], X[mask], w[mask]
    sqrt_w = np.sqrt(wm)[:, None]
    A = np.column_stack([np.ones(len(ym), dtype=np.float64), dm]) * sqrt_w
    coef, _, _, _ = np.linalg.lstsq(A, ym * sqrt_w[:, 0], rcond=None)
    return y - (coef[0] + np.asarray(X, dtype=np.float64) @ coef[1:])


def _partial_pc1_metrics_for_weights(
    pc1: np.ndarray,
    emb_var_total: float,
    pc1_var_w: float,
    X_year_cat: np.ndarray,
    y_lin: np.ndarray,
    logit: np.ndarray,
    spline_basis: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Compute fixed-axis or aligned-refit PC1-by-year partial association metrics."""
    r_lin = _weighted_pearson(pc1, y_lin, weights)
    eta2_cat = weighted_ols_r2(pc1, X_year_cat, weights)
    r2_l = weighted_ols_r2(pc1, logit[:, None], weights)
    res_l = 1.0 - r2_l if np.isfinite(r2_l) else np.nan
    r2_l_cat = weighted_ols_r2(pc1, np.column_stack([logit, X_year_cat]), weights)
    delta_cat_l = max(0.0, r2_l_cat - r2_l) if np.isfinite(r2_l) and np.isfinite(r2_l_cat) else np.nan
    p_eta2_l = delta_cat_l / res_l if np.isfinite(res_l) and res_l > 1e-12 else np.nan

    res_pc1 = _weighted_residuals(pc1, logit[:, None], weights)
    res_yr = _weighted_residuals(y_lin, logit[:, None], weights)
    partial_r_l = _weighted_pearson(res_pc1, res_yr, weights)

    w_int = np.rint(weights).astype(np.int64)
    if np.all(w_int == weights) and w_int.sum() > 1:
        exp_pc1 = np.repeat(res_pc1, w_int)
        exp_yr = np.repeat(res_yr, w_int)
        spearman_partial = float(spearmanr(exp_pc1, exp_yr).statistic) if (np.ptp(exp_pc1) > 0 and np.ptp(exp_yr) > 0) else np.nan
    else:
        spearman_partial = np.nan

    r2_s = weighted_ols_r2(pc1, spline_basis, weights) if spline_basis.shape[1] > 0 else 0.0
    res_s = 1.0 - r2_s if np.isfinite(r2_s) else np.nan
    X_s_cat = np.column_stack([spline_basis, X_year_cat]) if spline_basis.shape[1] > 0 else X_year_cat
    r2_s_cat = weighted_ols_r2(pc1, X_s_cat, weights)
    delta_cat_s = max(0.0, r2_s_cat - r2_s) if np.isfinite(r2_s) and np.isfinite(r2_s_cat) else np.nan
    p_eta2_s = delta_cat_s / res_s if np.isfinite(res_s) and res_s > 1e-12 else np.nan

    X_s_lin = np.column_stack([spline_basis, y_lin]) if spline_basis.shape[1] > 0 else y_lin[:, None]
    r2_s_lin = weighted_ols_r2(pc1, X_s_lin, weights)
    delta_lin_s = max(0.0, r2_s_lin - r2_s) if np.isfinite(r2_s) and np.isfinite(r2_s_lin) else np.nan

    return {
        "pc1_var_share": _ratio(pc1_var_w, emb_var_total),
        "eta_squared_year_cat": eta2_cat,
        "pearson_r_year_linear": r_lin,
        "abs_pearson_r_year_linear": abs(r_lin) if np.isfinite(r_lin) else np.nan,
        "r2_logit_linear": r2_l,
        "residual_var_after_logit_linear": res_l,
        "r2_logit_linear_plus_year_cat": r2_l_cat,
        "delta_r2_year_cat_given_logit_linear": delta_cat_l,
        "partial_eta2_year_cat_given_logit_linear": p_eta2_l,
        "partial_r_year_linear_given_logit_linear": partial_r_l,
        "spearman_partial_r_year_linear_given_logit": spearman_partial,
        "r2_logit_spline": r2_s,
        "residual_var_after_logit_spline": res_s,
        "r2_logit_spline_plus_year_cat": r2_s_cat,
        "delta_r2_year_cat_given_logit_spline": delta_cat_s,
        "partial_eta2_year_cat_given_logit_spline": p_eta2_s,
        "delta_r2_year_linear_given_logit_spline": delta_lin_s,
    }


def partial_pc1_year_analysis(
    embeddings,
    years,
    probabilities,
    patient_ids,
    *,
    spline_df: int = LOGIT_SPLINE_DF,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    run_refit_sensitivity: bool = True,
    min_axis_cos: float = AXIS_ALIGNMENT_MIN_COS,
) -> pd.DataFrame:
    """Partial association between embedding PC1 and calendar year controlling for predicted logit."""
    emb = np.asarray(embeddings, dtype=np.float64)
    yr = np.asarray(years, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    pids = np.asarray(patient_ids, dtype=str)
    if not (len(emb) == len(yr) == len(probs) == len(pids)):
        raise ValueError("embeddings, years, probabilities, and patient_ids must align")

    clipped = np.clip(probs, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)
    logit = np.log(clipped / (1.0 - clipped))
    spline_basis = natural_cubic_spline_basis(logit, df=spline_df)
    uniq_years = np.sort(np.unique(yr))
    if len(uniq_years) < 2:
        raise ValueError("partial_pc1_year_analysis requires at least 2 distinct years")
    X_year_cat = np.column_stack([(yr == u).astype(np.float64) for u in uniq_years[1:]])
    y_lin = yr.astype(np.float64)

    mean_emb, axes, _ = geometry.principal_axes(emb)
    v0 = axes[:, 0].copy()
    pc1_0 = (emb - mean_emb) @ v0
    ones = np.ones(len(emb), dtype=np.float64)
    if _weighted_pearson(pc1_0, y_lin, ones) < 0:
        v0 = -v0
        pc1_0 = -pc1_0

    cov0 = lc.weighted_covariance(emb, ones)
    pt_vals = _partial_pc1_metrics_for_weights(
        pc1_0, float(np.trace(cov0)), float(v0 @ cov0 @ v0), X_year_cat, y_lin, logit, spline_basis, ones
    )

    universe, counts = pa.patient_draw_counts(pids, draws=draws, seed=seed)
    codes = pa.row_patient_codes(pids, universe)
    metric_names = list(pt_vals.keys())
    fixed_draws = {m: np.empty(draws, dtype=np.float64) for m in metric_names}
    refit_draws = {m: np.empty(draws, dtype=np.float64) for m in metric_names} if run_refit_sensitivity else {}
    abs_cos_draws = np.empty(draws, dtype=np.float64)

    with threadpool_limits(limits=1):
        for i in range(int(draws)):
            w = counts[i, codes].astype(np.float64)
            cov_w = lc.weighted_covariance(emb, w)
            tr_w = float(np.trace(cov_w))
            res_f = _partial_pc1_metrics_for_weights(
                pc1_0, tr_w, float(v0 @ cov_w @ v0), X_year_cat, y_lin, logit, spline_basis, w
            )
            for m in metric_names:
                fixed_draws[m][i] = res_f[m]

            if run_refit_sensitivity:
                evals, evecs = np.linalg.eigh(cov_w)
                v_b = evecs[:, -1]
                cos_b = float(v_b @ v0)
                abs_cos_draws[i] = abs(cos_b)
                if cos_b < 0:
                    v_b = -v_b
                pc1_b = emb @ v_b
                res_r = _partial_pc1_metrics_for_weights(
                    pc1_b, tr_w, float(max(evals[-1], 0.0)), X_year_cat, y_lin, logit, spline_basis, w
                )
                for m in metric_names:
                    refit_draws[m][i] = res_r[m]

    rows = []
    for m in metric_names:
        arr = fixed_draws[m]
        fin = arr[np.isfinite(arr)]
        est = float(pt_vals[m])
        lo, hi = np.percentile(fin, [2.5, 97.5]) if np.isfinite(est) and len(fin) else (np.nan, np.nan)
        rows.append({
            "mode": "fixed_pc1", "metric": m, "estimate": est,
            "ci_low": float(lo), "ci_high": float(hi), "valid_draws": int(len(fin)),
            "rows": int(len(emb)),
        })

    if run_refit_sensitivity:
        for m in metric_names:
            arr = refit_draws[m]
            fin = arr[np.isfinite(arr)]
            est = float(pt_vals[m])
            lo, hi = np.percentile(fin, [2.5, 97.5]) if np.isfinite(est) and len(fin) else (np.nan, np.nan)
            rows.append({
                "mode": "refit_pc1_aligned", "metric": m, "estimate": est,
                "ci_low": float(lo), "ci_high": float(hi), "valid_draws": int(len(fin)),
                "rows": int(len(emb)),
            })
        rows.append({
            "mode": "refit_pc1_aligned", "metric": "axis_drift_share",
            "estimate": float(np.mean(abs_cos_draws < float(min_axis_cos))),
            "ci_low": np.nan, "ci_high": np.nan, "valid_draws": int(draws), "rows": int(len(emb)),
        })
        rows.append({
            "mode": "refit_pc1_aligned", "metric": "min_abs_axis_cos",
            "estimate": float(np.min(abs_cos_draws)),
            "ci_low": np.nan, "ci_high": np.nan, "valid_draws": int(draws), "rows": int(len(emb)),
        })
    return pd.DataFrame(rows)


def continuous_tcav_saturation_for_layer(
    embeddings,
    gradients,
    ref_scale,
    *,
    direction_count: int = DIRECTION_COUNT,
    direction_seed: int = DIRECTION_SEED,
    top_axes: int = TOP_K,
    low: float = INTERIOR_LOW,
    high: float = INTERIOR_HIGH,
) -> pd.DataFrame:
    """Threshold-free continuous TCAV saturation matching Work 8's exact per-(stratum, layer) RNG stream."""
    emb = np.asarray(embeddings, dtype=np.float64)
    grad_prob = np.asarray(gradients, dtype=np.float64)
    scale = np.asarray(ref_scale, dtype=np.float64)
    rng = np.random.default_rng(int(direction_seed))

    rows = []
    for chart, chart_scale in (("raw", np.ones_like(scale)), ("standardized", scale)):
        chart_emb = emb / chart_scale
        _, axes, _ = geometry.principal_axes(chart_emb)
        chart_grad_prob = grad_prob * chart_scale
        finite_mask = np.isfinite(chart_grad_prob).all(axis=1)
        g_fin = chart_grad_prob[finite_mask]
        all_rows = np.ones((len(g_fin), 1), dtype=bool)
        top_eff = min(int(top_axes), axes.shape[1])
        for family in geometry.DIRECTION_FAMILIES:
            dirs = geometry.random_directions(rng, int(direction_count), axes, top_eff, family)
            tcav = geometry.positive_share(g_fin @ dirs.T, all_rows)[0]
            fin = tcav[np.isfinite(tcav)]
            n_fin = len(fin)
            int_share, int_se = geometry.interior_share(fin, low=low, high=high)
            if n_fin > 0:
                ent = probe.binary_entropy_nats(fin)
                ent_mean = float(ent.mean())
                ent_se = float(ent.std(ddof=1) / np.sqrt(n_fin)) if n_fin > 1 else 0.0
                dev = np.abs(fin - 0.5)
                dev_mean = float(dev.mean())
                dev_se = float(dev.std(ddof=1) / np.sqrt(n_fin)) if n_fin > 1 else 0.0
                exact_mean = float(((fin == 0.0) | (fin == 1.0)).mean())
                exact_se = float(np.sqrt(exact_mean * (1.0 - exact_mean) / n_fin))
            else:
                ent_mean = ent_se = dev_mean = dev_se = exact_mean = exact_se = np.nan
            rows.append({
                "chart": chart,
                "family": family,
                "rows_used": int(len(g_fin)),
                "n_draws": int(n_fin),
                "mean_tcav_binary_entropy_nats": ent_mean,
                "mean_tcav_binary_entropy_se": ent_se,
                "mean_abs_dev_from_half": dev_mean,
                "mean_abs_dev_from_half_se": dev_se,
                "exact_endpoint_share": exact_mean,
                "exact_endpoint_se": exact_se,
                "interior_share": float(int_share),
                "interior_se": float(int_se),
            })
    return pd.DataFrame(rows)


def continuous_tcav_equal_n_subsamples(
    embeddings,
    gradients,
    ref_scale,
    *,
    subsample_n: int,
    repeats: int = SUBSAMPLE_REPEATS,
    subsample_seed: int = SUBSAMPLE_SEED,
    direction_count: int = DIRECTION_COUNT,
    direction_seed: int = DIRECTION_SEED,
    top_axes: int = TOP_K,
) -> pd.DataFrame:
    """Control finite-row k/n granularity by averaging continuous TCAV saturation across equal-n subsamples."""
    emb = np.asarray(embeddings, dtype=np.float64)
    grad = np.asarray(gradients, dtype=np.float64)
    n_sub = int(subsample_n)
    if len(emb) < n_sub:
        raise ValueError(f"Cannot draw subsample_n={n_sub} from {len(emb)} rows")
    rng = np.random.default_rng(int(subsample_seed))
    frames = []
    for rep in range(int(repeats)):
        idx = rng.choice(len(emb), size=n_sub, replace=False)
        res = continuous_tcav_saturation_for_layer(
            emb[idx], grad[idx], ref_scale,
            direction_count=direction_count,
            direction_seed=int(direction_seed) + rep,
            top_axes=top_axes,
        )
        res["repeat"] = rep
        frames.append(res)
    all_reps = pd.concat(frames, ignore_index=True)
    agg = (
        all_reps.groupby(["chart", "family"], sort=False, as_index=False)
        .agg(
            mean_tcav_binary_entropy_nats=("mean_tcav_binary_entropy_nats", "mean"),
            sd_tcav_binary_entropy_nats=("mean_tcav_binary_entropy_nats", "std"),
            mean_abs_dev_from_half=("mean_abs_dev_from_half", "mean"),
            sd_abs_dev_from_half=("mean_abs_dev_from_half", "std"),
            exact_endpoint_share=("exact_endpoint_share", "mean"),
            sd_exact_endpoint_share=("exact_endpoint_share", "std"),
            interior_share=("interior_share", "mean"),
            sd_interior_share=("interior_share", "std"),
        )
    )
    agg["subsample_n"] = n_sub
    agg["repeats"] = int(repeats)
    agg["source_rows"] = int(len(emb))
    return agg


def analytical_null_resultant(n_rows_used: int, dimension: int = 192) -> dict:
    """Analytical isotropic random-unit-gradient floor: exact RMS 1/sqrt(n) and CLT mean c_d/sqrt(n)."""
    n = int(n_rows_used)
    d = int(dimension)
    if n < 1 or d < 1:
        raise ValueError(f"n_rows_used and dimension must be positive, got n={n}, d={d}")
    log_cd = 0.5 * np.log(2.0 / float(d)) + float(gammaln(0.5 * (d + 1.0)) - gammaln(0.5 * d))
    c_d = float(np.exp(log_cd))
    rms_floor = float(1.0 / np.sqrt(n))
    return {
        "n_rows_used": n,
        "dimension": d,
        "rms_floor": rms_floor,
        "c_d": c_d,
        "clt_expected_floor": float(c_d / np.sqrt(n)),
        "asymptotic_error_order": "O(n^-1)",
    }


def non_nested_dispersion_summary(strata_df: pd.DataFrame) -> pd.DataFrame:
    """Within-dataset, per-layer descriptive Spearman association over non-nested single-year strata."""
    required = {
        "dataset", "scaling_scheme", "layer", "chart", "stratum",
        "probability_resultant_length", "prob_std", "prob_iqr", "logit_std", "logit_iqr", "prevalence", "status",
    }
    missing = sorted(required - set(strata_df.columns))
    if missing:
        raise ValueError(f"Missing required columns in strata_df: {missing}")

    covariates = (
        ("prob_std", "dispersion"),
        ("prob_iqr", "dispersion"),
        ("logit_std", "dispersion"),
        ("logit_iqr", "dispersion"),
        ("prevalence", "descriptive_covariate"),
    )
    rows = []
    for gkey, grp in strata_df.groupby(["dataset", "scaling_scheme", "layer", "chart"], sort=False):
        if grp["stratum"].duplicated().any():
            raise ValueError(f"Duplicate stratum inside non_nested_dispersion_summary group {gkey}")
        ok = grp.loc[grp["status"] == "ok"]
        dataset, scaling_scheme, layer, chart = gkey
        for cov, role in covariates:
            sub = ok[["probability_resultant_length", cov]].dropna()
            x = sub[cov].to_numpy(dtype=np.float64)
            r = sub["probability_resultant_length"].to_numpy(dtype=np.float64)
            if len(sub) >= 2 and np.ptp(x) > 0 and np.ptp(r) > 0:
                rho = float(spearmanr(x, r).statistic)
            else:
                rho = np.nan
            rows.append({
                "dataset": dataset,
                "scaling_scheme": scaling_scheme,
                "layer": layer,
                "chart": chart,
                "covariate": cov,
                "covariate_role": role,
                "n_strata": int(len(sub)),
                "spearman_rho": rho,
                "min_covariate": float(np.min(x)) if len(x) else np.nan,
                "max_covariate": float(np.max(x)) if len(x) else np.nan,
                "min_resultant": float(np.min(r)) if len(r) else np.nan,
                "max_resultant": float(np.max(r)) if len(r) else np.nan,
            })
    return pd.DataFrame(rows)


def _spectrum_stats_from_matrix(corr: np.ndarray, p_active: int) -> tuple[float, float, float, float]:
    """Compute participation ratio and top-1/5/10 variance shares from a PSD covariance/correlation matrix."""
    if p_active == 0:
        return np.nan, np.nan, np.nan, np.nan
    if p_active == 1:
        return 1.0, 1.0, 1.0, 1.0
    evals = np.maximum(np.linalg.eigvalsh(corr)[::-1], 0.0)
    tot = float(evals.sum())
    if tot <= 0.0:
        return np.nan, np.nan, np.nan, np.nan
    pr = float((tot * tot) / float(np.sum(evals * evals)))
    return (
        pr,
        float(evals[:1].sum() / tot),
        float(evals[:min(5, p_active)].sum() / tot),
        float(evals[:min(10, p_active)].sum() / tot),
    )


def input_feature_spectrum(
    X_eval,
    X_ref_scale=None,
    *,
    log1p: bool = False,
    perm_draws: int = PERM_NULL_DRAWS,
    perm_seed: int = PERM_NULL_SEED,
) -> dict:
    """Correlation-matrix PR on active columns, permutation null PR, and secondary ref-std sensitivity."""
    X_arr = np.asarray(X_eval, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError("X_eval must be 2-D")
    finite_rows = np.isfinite(X_arr).all(axis=1)
    X_clean = X_arr[finite_rows]
    rows_used = int(len(X_clean))
    rows_dropped = int(len(X_arr) - rows_used)
    total_cols = int(X_arr.shape[1])
    if rows_used < 2:
        raise ValueError(f"Need at least 2 finite rows for feature spectrum, got {rows_used}")

    if log1p:
        if np.any(X_clean < 0.0):
            raise ValueError("log1p transform requires non-negative features")
        X_clean = np.log1p(X_clean)

    eval_var = np.var(X_clean, axis=0, ddof=0)
    eval_sd = np.sqrt(eval_var)
    active = eval_sd > 0.0
    p_active = int(active.sum())
    if p_active == 0:
        raise ValueError("No active columns with positive variance in X_eval")

    Z = (X_clean[:, active] - X_clean[:, active].mean(axis=0)) / eval_sd[active]
    corr = (Z.T @ Z) / float(rows_used)
    pr, t1, t5, t10 = _spectrum_stats_from_matrix(corr, p_active)

    rng = np.random.default_rng(int(perm_seed))
    null_prs = np.empty(int(perm_draws), dtype=np.float64)
    for b in range(int(perm_draws)):
        Z_perm = rng.permuted(Z, axis=0)
        Zp = Z_perm - Z_perm.mean(axis=0)
        sd_p = np.sqrt(np.mean(Zp * Zp, axis=0))
        Zpn = Zp / np.where(sd_p > 0.0, sd_p, 1.0)
        null_cov = (Zpn.T @ Zpn) / float(rows_used)
        tr = float(np.trace(null_cov))
        denom = float(np.sum(null_cov * null_cov))
        null_prs[b] = (tr * tr) / denom if denom > 0.0 else np.nan
    fin_null = null_prs[np.isfinite(null_prs)]

    out = {
        "rows_used": rows_used,
        "rows_dropped": rows_dropped,
        "total_columns": total_cols,
        "active_columns": p_active,
        "corr_matrix_pr": pr,
        "corr_matrix_pr_over_p_active": float(pr / p_active) if np.isfinite(pr) else np.nan,
        "top1_var_share": t1,
        "top5_var_share": t5,
        "top10_var_share": t10,
        "perm_null_pr_mean": float(np.mean(fin_null)) if len(fin_null) else np.nan,
        "perm_null_pr_ci_low": float(np.percentile(fin_null, 2.5)) if len(fin_null) else np.nan,
        "perm_null_pr_ci_high": float(np.percentile(fin_null, 97.5)) if len(fin_null) else np.nan,
        "ref_active_columns": np.nan,
        "ref_std_pr": np.nan,
        "max_eval_to_ref_var_ratio": np.nan,
        "dropped_eval_var_share": np.nan,
    }

    if X_ref_scale is not None:
        X_ref = np.asarray(X_ref_scale, dtype=np.float64)
        X_ref_clean = X_ref[np.isfinite(X_ref).all(axis=1)]
        if log1p:
            if np.any(X_ref_clean < 0.0):
                raise ValueError("log1p transform requires non-negative reference features")
            X_ref_clean = np.log1p(X_ref_clean)
        ref_var = np.var(X_ref_clean, axis=0, ddof=0)
        ref_sd = np.sqrt(ref_var)
        ref_active = ref_sd > 0.0
        n_ref_act = int(ref_active.sum())
        tot_eval_var = float(eval_var.sum())
        out["ref_active_columns"] = n_ref_act
        out["dropped_eval_var_share"] = float(eval_var[~ref_active].sum() / tot_eval_var) if tot_eval_var > 0 else 0.0
        if n_ref_act > 0:
            Z_ref_std = (X_clean[:, ref_active] - X_clean[:, ref_active].mean(axis=0)) / ref_sd[ref_active]
            cov_ref_std = (Z_ref_std.T @ Z_ref_std) / float(rows_used)
            ref_pr, _, _, _ = _spectrum_stats_from_matrix(cov_ref_std, n_ref_act)
            out["ref_std_pr"] = ref_pr
            out["max_eval_to_ref_var_ratio"] = float(np.max(eval_var[ref_active] / ref_var[ref_active]))
    return out


def multiplicity_summary(
    tables_by_name: dict[str, pd.DataFrame],
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Count reported 95% intervals excluding zero (excluding zero-by-construction reference controls)."""
    rows = []
    tot_all = 0
    sig_all = 0
    for name, df in tables_by_name.items():
        sub = df.copy()
        if "is_reference_year_contrast" in sub.columns:
            sub = sub.loc[~sub["is_reference_year_contrast"].fillna(False)]
        if "stratum" in sub.columns and "quantity" in sub.columns:
            sub = sub.loc[~((sub["stratum"].astype(str) == "1997_eval") & (sub["quantity"] == "actual_minus_held"))]
        fin = np.isfinite(sub["ci_low"].to_numpy(dtype=np.float64)) & np.isfinite(sub["ci_high"].to_numpy(dtype=np.float64))
        sub = sub.loc[fin]
        n_tot = int(len(sub))
        excl = int(((sub["ci_low"] > 0.0) | (sub["ci_high"] < 0.0)).sum()) if n_tot else 0
        tot_all += n_tot
        sig_all += excl
        rows.append({
            "table": name,
            "total_intervals": n_tot,
            "excluding_zero": excl,
            "share_excluding_zero": float(excl / n_tot) if n_tot else np.nan,
            "chance_expected_at_alpha": float(alpha * n_tot),
        })
    rows.append({
        "table": "ALL",
        "total_intervals": int(tot_all),
        "excluding_zero": int(sig_all),
        "share_excluding_zero": float(sig_all / tot_all) if tot_all else np.nan,
        "chance_expected_at_alpha": float(alpha * tot_all),
    })
    return pd.DataFrame(rows)


_PROHIBITED_WORDING_TERMS = (
    "caus" + "al",
    "counter" + "factual",
    "contra" + "factual",
)


def scan_notebook_for_sus_row_leakage(
    nb_path,
    *,
    max_text_lines_per_output: int = 80,
) -> pd.DataFrame:
    """Scan notebook cells and outputs for prohibited wording and per-row SUS data leakage."""
    nb = nbformat.read(Path(nb_path), as_version=4)
    prohibited_re = re.compile("|".join(_PROHIBITED_WORDING_TERMS), re.IGNORECASE)
    id_re = re.compile(r"\b(patient_id|subject_id|P\d{4,})\b")

    wording_hits = 0
    id_output_hits = 0
    oversized_outputs = 0

    for cell in nb.cells:
        if prohibited_re.search(cell.get("source", "")):
            wording_hits += 1
        for out in cell.get("outputs", []):
            chunks = []
            otype = out.get("output_type", "")
            if otype == "stream":
                chunks.append(out.get("text", ""))
            elif otype in ("execute_result", "display_data"):
                data = out.get("data", {})
                if "text/plain" in data:
                    chunks.append(data["text/plain"])
                if "text/html" in data:
                    html = data["text/html"]
                    chunks.append(html)
                    if html.lower().count("<tr") > int(max_text_lines_per_output):
                        oversized_outputs += 1
            elif otype == "error":
                chunks.append(out.get("evalue", ""))
                chunks.extend(out.get("traceback", []))

            joined = "\n".join(chunks)
            if prohibited_re.search(joined):
                wording_hits += 1
            if id_re.search(joined):
                id_output_hits += 1
            if len(joined.splitlines()) > int(max_text_lines_per_output):
                oversized_outputs += 1

    all_clear = (wording_hits == 0) and (id_output_hits == 0) and (oversized_outputs == 0)
    return pd.DataFrame([
        {"check": "prohibited_wording_in_notebook", "value": wording_hits, "rule": "== 0", "passed": wording_hits == 0},
        {"check": "patient_id_tokens_in_outputs", "value": id_output_hits, "rule": "== 0", "passed": id_output_hits == 0},
        {"check": "oversized_table_or_text_outputs", "value": oversized_outputs, "rule": "== 0", "passed": oversized_outputs == 0},
        {"check": "sus_output_hygiene_all_clear", "value": int(all_clear), "rule": "== 1", "passed": all_clear},
    ])

