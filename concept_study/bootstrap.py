"""Joint patient bootstrap shared by every concept, year and context variant.

One replicate draws patients (by patient id) with replacement from all evaluation
patients of the study. A patient's draw count is used as the weight of every one of
its rows, in every concept and year, so statistics that share patients move together.
Row 0 of every weight matrix is all ones: it gives the ordinary point estimate.

Pooled statistics over concepts (correlations across concept-years, medians over concepts)
also resample concepts: see concept_resamples.
"""
import numpy as np
import torch

from concept_study import paths


def patient_weights(patient_ids: np.ndarray, replicates: int = paths.BOOTSTRAP_REPLICATES,
                    seed: int = paths.BOOTSTRAP_SEED) -> tuple:
    """Return (sorted unique patient ids, weight matrix of shape (1 + replicates, patients))."""
    unique = np.unique(np.asarray(patient_ids).astype(str))
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=replicates)
    weights = np.vstack([np.ones((1, len(unique))), draws]).astype(np.float32)
    return unique, weights


def row_weights(row_patient_ids: np.ndarray, unique_ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weights of each row (1 + replicates, rows), looked up by the row's patient id."""
    positions = np.searchsorted(unique_ids, np.asarray(row_patient_ids).astype(str))
    if not np.array_equal(unique_ids[positions], np.asarray(row_patient_ids).astype(str)):
        raise ValueError("Row patient missing from the bootstrap patient list")
    return weights[:, positions]


DEVICE = paths.DEVICE


def weighted_sums(weights: np.ndarray, masks: np.ndarray, values: np.ndarray) -> torch.Tensor:
    """Sum of weight x value over the rows of each mask, for every replicate.

    weights: (replicates, rows); masks: (rows, cells) booleans; values: (rows, columns).
    Returns a float32 GPU tensor (replicates, cells, columns); counts stay exact below 2**24.
    NaN values must be replaced by 0 (and counted with a separate indicator column) beforehand.
    """
    w = torch.as_tensor(weights, dtype=torch.float32, device=DEVICE)
    m = torch.as_tensor(masks, dtype=torch.float32, device=DEVICE)
    v = torch.as_tensor(values, dtype=torch.float32, device=DEVICE)
    out = torch.empty((w.shape[0], m.shape[1], v.shape[1]), dtype=torch.float32, device=DEVICE)
    for cell in range(m.shape[1]):
        out[:, cell, :] = w @ (m[:, cell, None] * v)
    return out


def percentile_interval(replicates: np.ndarray, axis: int = 0) -> tuple:
    """2.5% and 97.5% percentiles over replicates, ignoring undefined (NaN) replicates."""
    with np.errstate(all="ignore"):
        return (np.nanpercentile(replicates, 2.5, axis=axis), np.nanpercentile(replicates, 97.5, axis=axis))


def gpu_percentiles(values: torch.Tensor, quantiles=(0.025, 0.975)) -> list:
    """Percentiles over axis 0 ignoring NaN, with numpy's linear interpolation; fast on large tensors."""
    ordered = torch.sort(values, dim=0).values                     # NaN sort last
    valid = torch.isfinite(values).sum(0)
    out = []
    for q in quantiles:
        position = q * (valid - 1).clamp_min(0).to(values.dtype)
        low = position.floor().long()
        high = position.ceil().long()
        lower = torch.gather(ordered, 0, low[None]).squeeze(0)
        upper = torch.gather(ordered, 0, high[None]).squeeze(0)
        value = lower + (upper - lower) * (position - low.to(values.dtype))
        out.append(torch.where(valid > 0, value, torch.full_like(value, float("nan"))).cpu().numpy())
    return out


def concept_resamples(concepts: np.ndarray, replicates: int, seed: int = paths.BOOTSTRAP_SEED) -> list:
    """Rows and concept labels of the point estimate and of each concept-level replicate.

    Entry 0 is every row with its own concept. Entry b >= 1 draws the distinct concepts with
    replacement and takes all rows of every drawn concept; the k-th draw is labelled k, so a concept
    drawn twice enters as two separate concepts (e.g. two sets of concept indicators). Used together
    with the patient bootstrap: replicate b takes patient replicate b of the values at these rows.
    Returns a list of (row positions, labels).
    """
    names, codes = np.unique(np.asarray(concepts).astype(str), return_inverse=True)
    rows_of = [np.flatnonzero(codes == k) for k in range(len(names))]
    rng = np.random.default_rng(seed)
    out = [(np.arange(len(codes)), codes)]
    for _ in range(replicates):
        draw = rng.integers(0, len(names), len(names))
        rows = np.concatenate([rows_of[k] for k in draw]) if len(draw) else np.zeros(0, int)
        out.append((rows, np.repeat(np.arange(len(draw)), [len(rows_of[k]) for k in draw])))
    return out
