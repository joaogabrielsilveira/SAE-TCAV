"""Rule-source-aware CAV fitting and ungated temporal TCAV."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def rule_cohort_mask(rule, X: np.ndarray) -> np.ndarray:
    """Accept one conjunction or complete semantic OR-of-ANDs RuleSet."""

    if not hasattr(rule, "mask"):
        raise TypeError("rule must expose mask(X)")
    mask = np.asarray(rule.mask(np.asarray(X)), dtype=bool)
    if mask.shape != (len(X),):
        raise ValueError("rule mask is not aligned with CAV records")
    return mask


def train_temporal_cav(
    *,
    embeddings: np.ndarray,
    features: np.ndarray,
    activations: Sequence[float],
    rule,
    activation_target: float,
    rule_source: str,
    patient_ids: Sequence[object],
    minimum_positive: int = 50,
    minimum_negative: int = 50,
    seed: int = 42,
) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    raw = np.asarray(embeddings, dtype=float)
    values = np.asarray(activations, dtype=float)
    patients = np.asarray(patient_ids).astype(str)
    if raw.ndim != 2 or len(raw) != len(values) or len(raw) != len(patients):
        raise ValueError("CAV records, activations, and patients must align")
    positives = rule_cohort_mask(rule, features)
    finite = values[np.isfinite(values)]
    low_quantile = max(float(activation_target) - 0.1, 0.0)
    low_cutoff = float(np.quantile(finite, low_quantile)) if len(finite) else float("nan")
    eligible_negative = (~positives) & np.isfinite(values) & (values <= low_cutoff)
    positive_indices = np.flatnonzero(positives)
    negative_pool = np.flatnonzero(eligible_negative)
    if len(positive_indices) < minimum_positive:
        return _invalid("insufficient_cav_positives", rule_source, activation_target, len(positive_indices), len(negative_pool))
    if len(negative_pool) < minimum_negative:
        return _invalid("insufficient_cav_negatives", rule_source, activation_target, len(positive_indices), len(negative_pool))
    rng = np.random.default_rng(seed)
    balanced_count = min(len(positive_indices), len(negative_pool))
    positive_training_indices = np.sort(
        rng.choice(positive_indices, size=balanced_count, replace=False)
    )
    negative_indices = np.sort(
        rng.choice(negative_pool, size=balanced_count, replace=False)
    )
    indices = np.concatenate([positive_training_indices, negative_indices])
    labels = np.concatenate([np.ones(balanced_count, dtype=int), np.zeros(balanced_count, dtype=int)])
    model = LogisticRegression(C=0.1, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=1000, random_state=seed)
    model.fit(raw[indices], labels)
    direction = model.coef_[0].astype(float)
    norm = np.linalg.norm(direction)
    if norm:
        direction /= norm

    unique_groups = np.unique(patients[indices])
    folds = min(5, len(unique_groups))
    accuracy = None
    auroc = None
    if folds >= 2:
        try:
            predicted_probability = cross_val_predict(
                model, raw[indices], labels, groups=patients[indices],
                cv=GroupKFold(folds), method="predict_proba",
            )[:, 1]
            predicted = predicted_probability >= 0.5
            accuracy = float(np.mean(predicted == labels))
            auroc = float(roc_auc_score(labels, predicted_probability))
        except ValueError:
            # Grouped diagnostics never gate an otherwise supported CAV.
            accuracy = None
            auroc = None
    return {
        "valid": True,
        "failure_reason": None,
        "rule_source": rule_source,
        "activation_target": float(activation_target),
        "low_activation_cutoff": low_cutoff,
        "positive_count": len(positive_indices),
        "eligible_negative_count": len(negative_pool),
        "negative_count": len(negative_indices),
        "positive_training_count": len(positive_training_indices),
        "positive_indices": positive_training_indices,
        "negative_indices": negative_indices,
        "cav": direction,
        "cv_accuracy": accuracy,
        "cv_auroc": auroc,
    }


def _invalid(reason, source, target, positives, negatives):
    return {
        "valid": False, "failure_reason": reason, "rule_source": source,
        "activation_target": float(target), "positive_count": int(positives),
        "eligible_negative_count": int(negatives), "cav": None,
    }


def temporal_tcav(cav: Sequence[float], death_gradients: np.ndarray) -> dict[str, object]:
    direction = np.asarray(cav, dtype=float)
    gradients = np.asarray(death_gradients, dtype=float)
    if gradients.ndim != 2 or gradients.shape[1] != len(direction):
        raise ValueError("CAV and death-output gradients do not align")
    score = float(np.mean(gradients @ direction > 0))
    label = "negative" if score < 0.40 else "positive" if score > 0.60 else "neutral"
    return {"tcav": score, "tcav_direction": label, "directional": label != "neutral"}


def compare_rule_source_cavs(left, right, left_tcav: float, right_tcav: float):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    cosine = float(left @ right / denominator) if denominator else 0.0
    direction = lambda value: -1 if value < 0.40 else 1 if value > 0.60 else 0
    return {
        "cav_cosine": cosine,
        "tcav_difference": float(left_tcav - right_tcav),
        "direction_agreement": direction(left_tcav) == direction(right_tcav),
    }
