"""Metric quality and outcome-free dimensionality diagnostics."""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence

import numpy as np

from .config import MetricSynthesisConfig
from .profiles import profile_metrics


def _finite(value: Any) -> bool:
    return value is not None and bool(np.isfinite(value))


def metric_quality(
    rows: Sequence[Mapping[str, Any]], profile: str
) -> list[dict[str, Any]]:
    """Audit every prespecified metric without silently changing a profile."""
    names = profile_metrics(profile)
    eligible = [row for row in rows if int(row.get("temporal_distance", 0)) != 0]
    output: list[dict[str, Any]] = []
    finite_columns: list[np.ndarray] = []
    for name in names:
        values = np.asarray([float(row[name]) for row in eligible if _finite(row.get(name))])
        variance = float(np.var(values)) if len(values) else None
        unique = int(len(np.unique(values))) if len(values) else 0
        reason = (
            "no_finite_values" if not len(values)
            else "zero_variance" if unique < 2 or variance == 0.
            else "eligible"
        )
        output.append({
            "profile": profile,
            "metric": name,
            "row_count": len(eligible),
            "finite_count": int(len(values)),
            "missing_fraction": 1. - len(values) / len(eligible) if eligible else 1.,
            "unique_value_count": unique,
            "variance": variance,
            "eligibility_reason": reason,
        })
        if reason == "eligible" and len(values) == len(eligible):
            finite_columns.append(values)
    complete = [row for row in eligible if all(_finite(row.get(name)) for name in names)]
    rank = 0
    if complete:
        matrix = np.asarray([[float(row[name]) for name in names] for row in complete])
        rank = int(np.linalg.matrix_rank(matrix - matrix.mean(axis=0)))
    for record in output:
        record["effective_rank"] = rank
    return output


def _orient_loadings(loadings: np.ndarray) -> np.ndarray:
    """Order factors by energy and orient each largest loading positive."""
    order = np.argsort(np.sum(loadings ** 2, axis=0))[::-1]
    result = loadings[:, order].copy()
    for column in range(result.shape[1]):
        anchor = int(np.argmax(np.abs(result[:, column])))
        if result[anchor, column] < 0:
            result[:, column] *= -1
    return result


def varimax(loadings: np.ndarray, *, tolerance: float = 1e-7, iterations: int = 500) -> np.ndarray:
    """Return a deterministic orthogonal varimax rotation."""
    matrix = np.asarray(loadings, dtype=float)
    rows, columns = matrix.shape
    rotation = np.eye(columns)
    previous = 0.
    for _ in range(iterations):
        projected = matrix @ rotation
        u, singular, vh = np.linalg.svd(
            matrix.T @ (projected ** 3 - projected @ np.diag(np.sum(projected ** 2, axis=0)) / rows)
        )
        rotation = u @ vh
        objective = float(singular.sum())
        if previous and objective <= previous * (1. + tolerance):
            break
        previous = objective
    return _orient_loadings(matrix @ rotation)


def dimensionality(
    features: Sequence[Mapping[str, Any]], profile: str, config: MetricSynthesisConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Estimate a profile only when every prespecified metric is estimable."""
    from sklearn.decomposition import FactorAnalysis, PCA
    from sklearn.preprocessing import StandardScaler

    names = profile_metrics(profile)
    quality = metric_quality(features, profile)
    ineligible = [row for row in quality if row["eligibility_reason"] != "eligible"]
    rows = [row for row in features if int(row["temporal_distance"]) != 0 and all(_finite(row.get(name)) for name in names)]
    if ineligible:
        reason = "not_estimable_zero_variance" if any(row["eligibility_reason"] == "zero_variance" for row in ineligible) else "not_estimable_metric_quality"
        return [], [], [], [{"profile": profile, "status": reason, "row_count": len(rows),
                             "excluded_metrics": [row["metric"] for row in ineligible]}]
    if len(rows) < max(3, len(names) + 1):
        return [], [], [], [{"profile": profile, "status": "insufficient_complete_vectors", "row_count": len(rows)}]
    raw = np.asarray([[row[name] for name in names] for row in rows], dtype=float)
    matrix = StandardScaler().fit_transform(raw)
    effective_rank = int(np.linalg.matrix_rank(matrix))
    pca = PCA().fit(matrix)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    spectrum = [{"profile": profile, "component": index + 1, "eigenvalue": float(value),
                 "explained_variance_ratio": float(pca.explained_variance_ratio_[index]),
                 "cumulative_explained_variance": float(cumulative[index]), "kaiser_retained": bool(value > 1)}
                for index, value in enumerate(pca.explained_variance_)]
    oriented_pca = _orient_loadings(pca.components_.T * np.sqrt(pca.explained_variance_))
    loadings = [{"profile": profile, "metric": name,
                 **{f"PC{index + 1}": float(oriented_pca[row_index, index]) for index in range(len(names))}}
                for row_index, name in enumerate(names)]
    rng = np.random.default_rng(config.seed)
    null = np.empty((config.parallel_repetitions, len(names)))
    for repetition in range(config.parallel_repetitions):
        null[repetition] = np.linalg.eigvalsh(np.corrcoef(rng.normal(size=matrix.shape), rowvar=False))[::-1]
    parallel = [{"profile": profile, "component": index + 1,
                 "observed_eigenvalue": float(pca.explained_variance_[index]),
                 "parallel_95th_eigenvalue": float(np.quantile(null[:, index], .95)),
                 "retained": bool(pca.explained_variance_[index] > np.quantile(null[:, index], .95))}
                for index in range(len(names))]
    diagnostics: list[dict[str, Any]] = []
    for dimensions in range(1, len(names) + 1):
        if dimensions >= effective_rank or dimensions >= len(names):
            diagnostics.append({"profile": profile, "dimensions": dimensions, "status": "excluded",
                                "exclusion_reason": "factor_count_not_below_effective_rank_and_metric_count",
                                "effective_rank": effective_rank})
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = FactorAnalysis(n_components=dimensions, random_state=config.seed).fit(matrix)
                transformed = model.transform(matrix)
                reconstruction = transformed @ model.components_ + model.mean_
                likelihood = float(model.score(matrix) * len(matrix))
                rotated = varimax(model.components_.T)
                converged = int(model.n_iter_) < int(model.max_iter) and np.isfinite(likelihood)
                diagnostics.append({"profile": profile, "dimensions": dimensions,
                                    "status": "valid" if converged else "excluded", "converged": bool(converged),
                                    "iterations": int(model.n_iter_), "finite_likelihood": bool(np.isfinite(likelihood)),
                                    "log_likelihood": likelihood if np.isfinite(likelihood) else None,
                                    "reconstruction_mse": float(np.mean((matrix - reconstruction) ** 2)),
                                    "communalities": np.sum(rotated ** 2, axis=1).tolist(),
                                    "loadings": rotated.tolist() if converged else [],
                                    "warnings": [str(item.message) for item in caught],
                                    "exclusion_reason": None if converged else "nonconverged_or_nonfinite"})
            except Exception as error:
                diagnostics.append({"profile": profile, "dimensions": dimensions, "status": "excluded",
                                    "converged": False, "warnings": [str(item.message) for item in caught],
                                    "exclusion_reason": f"fit_error:{type(error).__name__}:{error}"})
    return spectrum, loadings, parallel, diagnostics


def bootstrap_stability(features: Sequence[Mapping[str, Any]], profile: str,
                        config: MetricSynthesisConfig, dimensions: int = 2) -> list[dict[str, Any]]:
    """Reference-cluster bootstrap stability for PCA subspaces and FA loadings."""
    from scipy.linalg import subspace_angles
    from sklearn.decomposition import FactorAnalysis, PCA
    from sklearn.preprocessing import StandardScaler

    names = profile_metrics(profile)
    rows = [row for row in features if int(row["temporal_distance"]) != 0 and all(_finite(row.get(name)) for name in names)]
    if config.bootstrap_repetitions < 1 or dimensions >= len(names) or len({int(row["reference_year"]) for row in rows}) < 2:
        return []
    if any(record["eligibility_reason"] != "eligible" for record in metric_quality(features, profile)):
        return []
    raw = np.asarray([[float(row[name]) for name in names] for row in rows])
    reference = np.asarray([int(row["reference_year"]) for row in rows])
    standardized = StandardScaler().fit_transform(raw)
    base_pca = PCA(n_components=dimensions, random_state=config.seed).fit(standardized).components_.T
    base_fa = varimax(FactorAnalysis(n_components=dimensions, random_state=config.seed).fit(standardized).components_.T)
    references = np.unique(reference)
    rng = np.random.default_rng(config.seed)
    output = []
    for repetition in range(config.bootstrap_repetitions):
        sampled = rng.choice(references, size=len(references), replace=True)
        indices = np.concatenate([np.flatnonzero(reference == year) for year in sampled])
        sample = StandardScaler().fit_transform(raw[indices])
        try:
            boot_pca = PCA(n_components=dimensions, random_state=config.seed).fit(sample).components_.T
            angle = float(np.max(subspace_angles(base_pca, boot_pca)))
            boot_fa = varimax(FactorAnalysis(n_components=dimensions, random_state=config.seed).fit(sample).components_.T)
            correlations = [abs(float(np.corrcoef(base_fa[:, column], boot_fa[:, column])[0, 1])) for column in range(dimensions)]
            output.append({"profile": profile, "repetition": repetition + 1, "sampled_reference_count": len(sampled),
                           "pca_max_principal_angle_radians": angle,
                           "fa_mean_absolute_loading_correlation": float(np.nanmean(correlations)), "status": "valid"})
        except Exception as error:
            output.append({"profile": profile, "repetition": repetition + 1, "sampled_reference_count": len(sampled),
                           "status": "excluded", "exclusion_reason": f"fit_error:{type(error).__name__}:{error}"})
    return output
