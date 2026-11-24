"""Descriptive checks of frozen-factor transfer between query and context roles."""

import numpy as np

from context_intervention.analysis import correlation


def groups(years, labels):
    years, labels = np.asarray(years), np.asarray(labels)
    yield {"scope": "pooled", "year": None, "outcome": None}, np.ones(len(years), bool)
    for year in np.unique(years):
        yield {"scope": "year", "year": int(year), "outcome": None}, years == year
    for label in np.unique(labels):
        yield {"scope": "outcome", "year": None, "outcome": int(label)}, labels == label
    for year in np.unique(years):
        for label in np.unique(labels):
            mask = (years == year) & (labels == label)
            if mask.any():
                yield {"scope": "year_outcome", "year": int(year), "outcome": int(label)}, mask


def mean_or_missing(values):
    return float(np.mean(values)) if len(values) else np.nan


def factor_summary(native, query, rule, cutoff):
    native, query, rule = np.asarray(native), np.asarray(query), np.asarray(rule, bool)
    active, query_active = native > cutoff, query > cutoff
    union = active | query_active
    result = {"n": len(native), "n_rule": int(rule.sum()), "n_outside_rule": int((~rule).sum()),
              "activation_correlation": correlation(native, query),
              "correlation_supported": bool(len(native) >= 3 and np.std(native) > 0 and np.std(query) > 0),
              "activation_membership_agreement": float((active == query_active).mean()),
              "positive_set_intersection": int((active & query_active).sum()),
              "positive_set_union": int(union.sum()),
              "positive_set_jaccard": float((active & query_active).sum() / union.sum()) if union.any() else np.nan}
    for name, values, membership in (("native", native, active), ("query", query, query_active)):
        result.update({f"{name}_mean_activation": float(values.mean()), f"{name}_std_activation": float(values.std()),
                       f"{name}_activation_range": float(np.ptp(values)), f"{name}_constant": bool(np.ptp(values) <= 1e-8),
                       f"{name}_active_count": int(membership.sum()), f"{name}_activation_rate": float(membership.mean()),
                       f"{name}_activation_in_rule": mean_or_missing(membership[rule]),
                       f"{name}_activation_outside_rule": mean_or_missing(membership[~rule]),
                       f"{name}_strength_in_rule": mean_or_missing(values[rule]),
                       f"{name}_strength_outside_rule": mean_or_missing(values[~rule])})
    return result


def reconstruction_summary(errors, energy):
    """Both terms are per-dimension means in the same frozen standardized space."""
    errors, energy = np.asarray(errors), np.asarray(energy)
    supported = energy > 0
    return {"standardized_mse": float(errors.mean()), "standardized_input_energy": float(energy.mean()),
            "relative_reconstruction_error": float(errors.sum() / energy.sum()) if energy.sum() > 0 else np.nan,
            "mean_patient_relative_error": mean_or_missing(errors[supported] / energy[supported]),
            "zero_energy_rows": int((~supported).sum()), "n": len(errors)}
