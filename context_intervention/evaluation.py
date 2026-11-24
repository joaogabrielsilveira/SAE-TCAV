"""Metrics, threshold selection and patient-level uncertainty for predictions."""

import numpy as np
import pandas as pd

from context_intervention.analysis import performance


METRICS = ("death_f1", "precision", "recall", "average_precision", "auroc", "brier", "brier_skill")


def prediction_transitions(labels, baseline, modified, threshold):
    labels = np.asarray(labels)
    before = (np.asarray(baseline) >= threshold) == labels
    after = (np.asarray(modified) >= threshold) == labels
    return {"corrected": int((~before & after).sum()), "damaged": int((before & ~after).sum()),
            "remained_correct": int((before & after).sum()), "remained_wrong": int((~before & ~after).sum())}


def choose_threshold(labels, probabilities, frozen_threshold, reference_mortality):
    """Choose validation F1; ties prefer the threshold closest to the frozen one."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if len(labels) == 0 or len(np.unique(labels)) < 2 or not np.isfinite(probabilities).all():
        return float(frozen_threshold)
    candidates = np.unique(np.concatenate(([frozen_threshold, 0.0], probabilities, [np.nextafter(1., 2.)])))
    ranked = []
    for threshold in candidates:
        score = performance(labels, probabilities, threshold, reference_mortality)["death_f1"]
        ranked.append((score if np.isfinite(score) else -np.inf, -abs(threshold - frozen_threshold),
                       -threshold, float(threshold)))
    return max(ranked)[-1]


def temporal_summary(rows, probability_column="probability"):
    """Average each metric over years within reference, then over references."""
    required = {"reference", "year", "label", "threshold", "reference_mortality", probability_column}
    if not required.issubset(rows.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(rows.columns))}")
    results = []
    for (reference, year), group in rows.groupby(["reference", "year"], sort=True):
        if group["threshold"].nunique() != 1 or group["reference_mortality"].nunique() != 1:
            raise ValueError("Threshold and reference mortality must be fixed within each cell")
        results.append({"reference": reference, "year": year,
                        **performance(group["label"], group[probability_column], group["threshold"].iloc[0],
                                      group["reference_mortality"].iloc[0])})
    cells = pd.DataFrame(results)
    if cells.empty:
        return cells, {metric: np.nan for metric in METRICS}
    by_reference = cells.groupby("reference")[list(METRICS)].mean()
    return cells, by_reference.mean().to_dict()


def paired_patient_bootstrap(rows, baseline="baseline_probability", modified="probability", repetitions=1000,
                             seed=20261110):
    """Percentile intervals on paired differences, resampling whole patients."""
    if rows.empty:
        raise ValueError("Bootstrap needs prediction rows")
    ids = rows["patient_id"].astype(str).to_numpy()
    patients, inverse = np.unique(ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    _, base = temporal_summary(rows, baseline)
    _, intervention = temporal_summary(rows, modified)
    estimates = {metric: intervention[metric] - base[metric] for metric in METRICS}
    replicates = {metric: [] for metric in METRICS}
    for _ in range(repetitions):
        counts = np.bincount(rng.integers(0, len(patients), size=len(patients)), minlength=len(patients))
        indices = np.repeat(np.arange(len(rows)), counts[inverse])
        sample = rows.iloc[indices]
        _, base = temporal_summary(sample, baseline)
        _, intervention = temporal_summary(sample, modified)
        for metric in METRICS:
            replicates[metric].append(intervention[metric] - base[metric])
    intervals = {}
    for metric, values in replicates.items():
        finite = np.asarray(values)[np.isfinite(values)]
        intervals[metric] = {"difference": estimates[metric],
                             "lower": float(np.percentile(finite, 2.5)) if len(finite) else np.nan,
                             "upper": float(np.percentile(finite, 97.5)) if len(finite) else np.nan,
                             "valid_replicates": len(finite)}
    return intervals
