"""Reproducible, year-and-outcome-stratified context adjustments."""

import hashlib

import numpy as np


def patient_partition(patient_ids, seed=20261110):
    """Stable patient-level split; the same person stays in the same partition."""
    def part(patient):
        digest = hashlib.sha256(f"{seed}:{patient}".encode()).digest()
        return "development" if int.from_bytes(digest[:8], "big") / 2**64 < 0.6 else "validation"
    return np.array([part(str(patient)) for patient in patient_ids])


def strata(years, labels):
    years, labels = np.asarray(years), np.asarray(labels)
    if len(years) != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Year and binary outcome arrays must be aligned")
    return [(int(year), int(label)) for year, label in zip(years, labels)]


def quotas(keys, size):
    """Largest-remainder allocation proportional to original stratum sizes."""
    if not 0 <= size <= len(keys):
        raise ValueError("Sample size exceeds original context")
    unique, counts = np.unique(np.asarray(keys), axis=0, return_counts=True)
    if not len(counts):
        return {}
    exact = counts * size / len(keys)
    allocated = np.floor(exact).astype(int)
    order = np.argsort(-(exact - allocated), kind="stable")
    allocated[order[:size - allocated.sum()]] += 1
    return {tuple(key): int(count) for key, count in zip(unique, allocated)}


def sample_context(years, labels, size, seed, eligible=None, weights=None):
    """Draw once without replacement, preserving original stratum quotas."""
    keys = strata(years, labels)
    n = len(keys)
    eligible = np.ones(n, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if eligible.shape != (n,) or weights.shape != (n,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid eligibility or weights")
    rng = np.random.default_rng(seed)
    chosen = []
    key_array = np.asarray(keys)
    for key, count in quotas(keys, size).items():
        candidates = np.flatnonzero(np.all(key_array == key, axis=1) & eligible & (weights > 0))
        if len(candidates) < count:
            raise ValueError(f"Insufficient eligible context rows in stratum {key}")
        if count:
            probabilities = weights[candidates] / weights[candidates].sum()
            chosen.extend(rng.choice(candidates, size=count, replace=False, p=probabilities))
    return np.sort(np.asarray(chosen, dtype=int))


def graded_weights(priorities, strength):
    priorities = np.asarray(priorities, dtype=float)
    if not np.isfinite(priorities).all() or strength < 0:
        raise ValueError("Priorities must be finite and strength nonnegative")
    maximum = np.max(np.abs(priorities), initial=0)
    return np.ones_like(priorities) if maximum == 0 else np.exp(-strength * priorities / maximum)


def select_removals(years, labels, priorities, fraction, sample_size):
    """Remove highest positive priorities within each stratum, preserving sample support."""
    keys = np.asarray(strata(years, labels))
    priorities = np.asarray(priorities, dtype=float)
    if priorities.shape != (len(keys),) or not np.isfinite(priorities).all() or not 0 <= fraction <= 1:
        raise ValueError("Invalid priorities or removal fraction")
    counts = quotas(keys, sample_size)
    eligible = np.ones(len(keys), dtype=bool)
    for key, quota in counts.items():
        rows = np.flatnonzero(np.all(keys == key, axis=1))
        positive = rows[priorities[rows] > 0]
        remove = min(int(np.floor(len(positive) * fraction)), max(0, len(rows) - max(quota, 1)))
        order = np.argsort(-priorities[positive], kind="stable")
        eligible[positive[order[:remove]]] = False
    if set(np.asarray(labels)[eligible]) != {0, 1}:
        raise ValueError("Retained context needs both outcome classes")
    return eligible


def random_matched_removal(years, labels, selected, seed):
    """Random control with the same removed count in each year/outcome stratum."""
    keys = np.asarray(strata(years, labels))
    selected = np.asarray(selected, dtype=bool)
    if selected.shape != (len(keys),):
        raise ValueError("Removal mask is not aligned")
    rng = np.random.default_rng(seed)
    removed = np.zeros(len(keys), dtype=bool)
    for key in np.unique(keys, axis=0):
        rows = np.flatnonzero(np.all(keys == key, axis=1))
        count = selected[rows].sum()
        if count:
            removed[rng.choice(rows, size=count, replace=False)] = True
    return removed


def replace_neighborhood(years, labels, original_draw, excluded, seed):
    """Replace only excluded drawn rows, in their original strata, without duplicates."""
    keys = np.asarray(strata(years, labels))
    original_draw = np.asarray(original_draw, dtype=int)
    excluded = np.asarray(excluded, dtype=bool)
    if excluded.shape != (len(keys),) or len(np.unique(original_draw)) != len(original_draw):
        raise ValueError("Invalid original draw or exclusion mask")
    if np.any((original_draw < 0) | (original_draw >= len(keys))):
        raise ValueError("Draw contains an invalid row")
    retained = original_draw[~excluded[original_draw]]
    removed = original_draw[excluded[original_draw]]
    result = list(retained)
    rng = np.random.default_rng(seed)
    for key in np.unique(keys, axis=0):
        key_mask = np.all(keys == key, axis=1)
        count = int(key_mask[removed].sum())
        candidates = np.flatnonzero(key_mask & ~excluded & ~np.isin(np.arange(len(keys)), retained))
        if len(candidates) < count:
            raise ValueError(f"Insufficient replacement rows in stratum {tuple(key)}")
        if count:
            result.extend(rng.choice(candidates, size=count, replace=False))
    return np.sort(np.asarray(result, dtype=int))


def exclusion_priorities(n_rows, neighborhoods, effects):
    """Average jointly measured neighborhood effects for rows they contain."""
    totals, counts = np.zeros(n_rows), np.zeros(n_rows, dtype=int)
    for rows, effect in zip(neighborhoods, effects):
        rows = np.unique(np.asarray(rows, dtype=int))
        if not np.isfinite(effect):
            continue
        if np.any((rows < 0) | (rows >= n_rows)):
            raise ValueError("Neighborhood contains an invalid row")
        totals[rows] += effect
        counts[rows] += 1
    return np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)


def representatives(vectors, maximum=3):
    """Nearest to the mean, then furthest from the closest chosen representative."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Representatives require a finite matrix")
    if not len(vectors) or maximum <= 0:
        return np.array([], dtype=int)
    distances = np.linalg.norm(vectors - vectors.mean(axis=0), axis=1)
    chosen = [int(np.argmin(distances))]
    while len(chosen) < min(maximum, len(vectors)):
        nearest = np.min(np.linalg.norm(vectors[:, None, :] - vectors[chosen][None, :, :], axis=2), axis=1)
        nearest[chosen] = -np.inf
        chosen.append(int(np.argmax(nearest)))
    return np.asarray(chosen)


def neighborhood(context_vectors, patient_vectors, years, labels, size=250):
    """Most similar context rows, allocated by context year and outcome."""
    context, patients = np.asarray(context_vectors, dtype=float), np.asarray(patient_vectors, dtype=float)
    if context.ndim != 2 or patients.ndim != 2 or context.shape[1] != patients.shape[1] or not len(patients):
        raise ValueError("Aligned context and patient matrices are required")
    if not np.isfinite(context).all() or not np.isfinite(patients).all():
        raise ValueError("Representations must be finite")
    cn = np.linalg.norm(context, axis=1, keepdims=True)
    pn = np.linalg.norm(patients, axis=1, keepdims=True)
    context = np.divide(context, cn, out=np.zeros_like(context), where=cn > 0)
    patients = np.divide(patients, pn, out=np.zeros_like(patients), where=pn > 0)
    similarity = (context @ patients.T).max(axis=1)
    keys = np.asarray(strata(years, labels))
    selected = []
    for key, count in quotas(keys, min(size, len(context))).items():
        rows = np.flatnonzero(np.all(keys == key, axis=1))
        selected.extend(rows[np.argsort(-similarity[rows], kind="stable")[:count]])
    return np.sort(np.asarray(selected, dtype=int))
