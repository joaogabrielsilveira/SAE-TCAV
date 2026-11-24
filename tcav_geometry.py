"""Gradient geometry of the historical TCAV measurement (roadmap Work 2, tests 1-4).

TCAV of one CAV fit in a patient group = share of the group's rows whose death-probability
gradient has a strictly positive dot product with the fit's unit direction. These functions
work on saved arrays (gradients, embeddings, directions, masks) and never refit anything.
"""
import zlib

import numpy as np


def build_rng(seed: int, build: str, reference: int, stream: int) -> np.random.Generator:
    """Random generator of one build and one purpose (stream), so a run on a few builds draws exactly
    what a run on all builds draws for the same build."""
    return np.random.default_rng([seed, reference, zlib.crc32(build.encode()), stream])


# ---------------------------------------------------------------- reconciliation with the saved study

def valid_fits(sensitivity: np.ndarray) -> np.ndarray:
    """Columns of the fits that were measured (failed fits are NaN columns), in their saved order."""
    return sensitivity[:, np.isfinite(sensitivity).all(axis=0)]


def positive_share(sensitivity: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """(cells, fits) share of each cell's rows with sensitivity strictly above zero; NaN for empty cells.

    sensitivity: (rows, fits); masks: (rows, cells). Rows with exactly zero sensitivity count in the
    denominator, as in the saved study.
    """
    counts = masks.sum(axis=0).astype(np.float64)
    positives = masks.T.astype(np.float64) @ (sensitivity > 0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = positives / counts[:, None]
    share[counts == 0] = np.nan
    return share


def per_fit_tcav(sensitivity: np.ndarray, groups: np.ndarray, years: np.ndarray, year_list: tuple) -> np.ndarray:
    """(groups, years, valid fits) TCAV of every measured fit, laid out like the saved 'per_fit|...|tcav'."""
    values = valid_fits(sensitivity)
    year_masks = np.stack([years == year for year in year_list], axis=1)
    cells = (groups[:, :, None] & year_masks[:, None, :]).reshape(len(years), -1)
    return positive_share(values, cells).reshape(groups.shape[1], len(year_list), values.shape[1])


def compare_sensitivities(gradients: np.ndarray, directions: np.ndarray, stored: np.ndarray) -> dict:
    """Recompute gradient . unit CAV for the measured fits and compare with the stored sensitivities."""
    measured = np.isfinite(directions).all(axis=1)
    recomputed = gradients.astype(np.float64) @ directions[measured].T.astype(np.float64)
    saved = stored[:, measured].astype(np.float64)
    nonzero = saved != 0
    return {
        "valid_fits": int(measured.sum()),
        "max_abs_difference": float(np.max(np.abs(recomputed - saved))) if saved.size else np.nan,
        "sign_disagreements": int(((recomputed > 0) != (saved > 0))[nonzero].sum()),
        "nonzero_rows": int(nonzero.any(axis=1).sum()),
    }


# ---------------------------------------------------------------- head replay and log-odds scale

PROBABILITY_TOLERANCE = {"rtol": 1e-4, "atol": 1e-5}   # concept_subgroup_drift_pilot.CONTINUATION_PARITY
GRADIENT_TOLERANCE = {"rtol": 1e-3, "atol": 1e-5}      # concept_subgroup_drift_pilot.GRADIENT_REPLAY


def _errors(replayed: np.ndarray, stored: np.ndarray) -> tuple:
    replayed, stored = replayed.astype(np.float64), stored.astype(np.float64)
    difference = np.abs(replayed - stored)
    nonzero = stored != 0
    relative = difference[nonzero] / np.abs(stored[nonzero])
    return float(difference.max()), float(relative.max()) if relative.size else np.nan


def replay_summary(replayed_probability, stored_probability, replayed_gradients, stored_gradients) -> dict:
    """Largest absolute and relative replay errors, and whether they sit inside the pilot's own tolerances."""
    p_abs, p_rel = _errors(replayed_probability, stored_probability)
    g_abs, g_rel = _errors(replayed_gradients, stored_gradients)
    return {
        "rows": int(len(stored_probability)),
        "probability_max_abs_error": p_abs, "probability_max_relative_error": p_rel,
        "probability_within_tolerance": bool(np.allclose(replayed_probability, stored_probability, **PROBABILITY_TOLERANCE)),
        "gradient_max_abs_error": g_abs, "gradient_max_relative_error": g_rel,
        "gradient_within_tolerance": bool(np.allclose(replayed_gradients, stored_gradients, **GRADIENT_TOLERANCE)),
    }


def log_odds_gradients(gradients: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Gradient of the death log-odds, g / (p (1 - p)), in float64; NaN rows where p is exactly 0 or 1."""
    p = probability.astype(np.float64)
    inside = (p > 0) & (p < 1)
    result = np.full(gradients.shape, np.nan)
    result[inside] = gradients[inside].astype(np.float64) / (p[inside] * (1 - p[inside]))[:, None]
    return result


# ---------------------------------------------------------------- Test 1: one shared gradient direction?

def unit_rows(gradients: np.ndarray) -> np.ndarray:
    """Finite nonzero gradient rows scaled to unit length (float64)."""
    g = gradients.astype(np.float64)
    norms = np.linalg.norm(g, axis=1)
    keep = np.isfinite(g).all(axis=1) & (norms > 0)
    return g[keep] / norms[keep, None]


def coherence(gradients: np.ndarray) -> dict:
    """How much the gradient directions of a set of rows agree.

    resultant_length: length of the mean unit gradient (1 = all rows point the same way, 0 = they cancel).
    mean_direction: that mean scaled to unit length (zeros when it is exactly zero).
    cosines: cosine of each row's gradient with the mean direction.
    leading_share: largest eigenvalue share of the mean outer product of unit gradients
    (1 = all rows on one line, either sign).
    """
    unit = unit_rows(gradients)
    resultant = unit.mean(axis=0) if len(unit) else np.zeros(gradients.shape[1])
    length = float(np.linalg.norm(resultant))
    direction = resultant / length if length > 0 else np.zeros_like(resultant)
    second_moment = unit.T @ unit / max(len(unit), 1)
    return {
        "rows_used": int(len(unit)), "rows_dropped": int(len(gradients) - len(unit)),
        "resultant_length": length, "mean_direction": direction,
        "cosines": unit @ direction if length > 0 else np.full(len(unit), np.nan),
        "leading_share": float(np.linalg.eigvalsh(second_moment)[-1]) if len(unit) else np.nan,
    }


def predicted_tcav(mean_direction: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """TCAV each fit would have if every gradient pointed along the mean direction: 1 when the fit
    has a positive dot product with it, else 0; NaN when there is no mean direction."""
    if not np.any(mean_direction):
        return np.full(len(directions), np.nan)
    return (directions @ mean_direction > 0).astype(np.float64)


def boundary_angle(mean_direction: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Signed angle in degrees between each fit and the plane perpendicular to the mean gradient
    direction (positive on the side the mean gradient points to)."""
    cosine = directions @ mean_direction / np.linalg.norm(directions, axis=1)
    return np.degrees(np.arcsin(np.clip(cosine, -1, 1)))


# ---------------------------------------------------------------- Test 2: do random directions saturate too?

DIRECTION_FAMILIES = ("isotropic", "top", "complement")


def principal_axes(embeddings: np.ndarray) -> tuple:
    """Mean, principal axes (columns, largest variance first) and their variances of the embedding rows."""
    values = embeddings.astype(np.float64)
    mean = values.mean(axis=0)
    variances, axes = np.linalg.eigh(np.cov(values - mean, rowvar=False))
    order = np.argsort(variances)[::-1]
    return mean, axes[:, order], variances[order]


def random_directions(rng: np.random.Generator, count: int, axes: np.ndarray, top: int, family: str) -> np.ndarray:
    """(count, dimension) unit directions uniform on the sphere of the whole space ("isotropic"), of the span
    of the first `top` principal axes ("top") or of its orthogonal complement ("complement")."""
    if family == "isotropic":
        basis = np.eye(axes.shape[0])
    elif family == "top":
        basis = axes[:, :top]
    elif family == "complement":
        basis = axes[:, top:]
    else:
        raise ValueError(f"family must be one of {DIRECTION_FAMILIES}, got {family!r}")
    directions = rng.standard_normal((count, basis.shape[1])) @ basis.T
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def interior_share(tcav: np.ndarray, low: float = 0.1, high: float = 0.9) -> tuple:
    """Share of finite TCAV values strictly between low and high, with its Monte Carlo standard error."""
    values = tcav[np.isfinite(tcav)]
    if not len(values):
        return np.nan, np.nan
    share = float(((values > low) & (values < high)).mean())
    return share, float(np.sqrt(share * (1 - share) / len(values)))


# ---------------------------------------------------------------- Test 3: which data axes carry the sign?

def split_direction(directions: np.ndarray, axes: np.ndarray, top: int) -> tuple:
    """Each direction split into its part inside the span of the first `top` principal axes and the rest.
    The two parts are not rescaled, so their derivatives add up to the full derivative."""
    basis = axes[:, :top]
    inside = directions @ basis @ basis.T
    return inside, directions - inside


def participation_ratio(matrix: np.ndarray) -> float:
    """tr(M)^2 / tr(M^2): roughly how many directions share the matrix's spread evenly."""
    return float(np.trace(matrix) ** 2 / np.sum(matrix * matrix.T))


def gradient_moments(gradients: np.ndarray) -> tuple:
    """Centred covariance and uncentred second moment of the finite gradient rows."""
    g = gradients[np.isfinite(gradients).all(axis=1)].astype(np.float64)
    return np.cov(g, rowvar=False), g.T @ g / len(g)


def energy_share(gradients: np.ndarray, axes: np.ndarray, top: int) -> float:
    """Share of the summed squared gradient length that lies in the span of the first `top` principal axes
    (top / dimension when the gradients ignore the data axes)."""
    g = gradients[np.isfinite(gradients).all(axis=1)].astype(np.float64)
    return float(np.sum((g @ axes[:, :top]) ** 2) / np.sum(g ** 2))


def standardized_chart(gradients: np.ndarray, directions: np.ndarray, scale: np.ndarray) -> tuple:
    """Gradients and directions in the standardized chart z = (h - mean) / scale.

    A displacement dh becomes dh / scale and a gradient g becomes g * scale, so every
    gradient . direction product (and hence every TCAV sign) is unchanged.
    """
    return gradients * scale, directions / scale


# ---------------------------------------------------------------- Test 4: does the shared direction persist off the data?

PERTURBATION_FAMILIES = ("off_axis", "full_space", "dilation")


def top_axes_length(embeddings: np.ndarray, mean: np.ndarray, axes: np.ndarray, top: int) -> float:
    """Root mean squared length of the centred embeddings inside the span of the first `top` principal axes:
    the typical spread of the data, used as the unit of displacement."""
    centred = (embeddings.astype(np.float64) - mean) @ axes[:, :top]
    return float(np.sqrt(np.mean(np.sum(centred ** 2, axis=1))))


def noise_draws(rng: np.random.Generator, embeddings: np.ndarray, axes: np.ndarray, top: int) -> dict:
    """One random unit direction per row for each noise family, drawn once and reused at every scale."""
    rows = len(embeddings)
    return {"off_axis": random_directions(rng, rows, axes, top, "complement"),
            "full_space": random_directions(rng, rows, axes, top, "isotropic")}


def perturb(embeddings: np.ndarray, mean: np.ndarray, noise: dict, length: float, scale: float, family: str) -> np.ndarray:
    """Embeddings moved away from the data.

    off_axis / full_space: each row moved by scale * length along its own noise direction
    (outside the top principal axes / anywhere). dilation: every row pushed away from the mean
    by the factor scale (1 = unchanged, 0 = all rows at the mean).
    """
    values = embeddings.astype(np.float64)
    if family == "dilation":
        return mean + scale * (values - mean)
    if family not in noise:
        raise ValueError(f"family must be one of {PERTURBATION_FAMILIES}, got {family!r}")
    return values + scale * length * noise[family]
