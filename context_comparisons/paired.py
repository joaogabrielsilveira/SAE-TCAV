"""Paired sampling, similarity-ordered removal and a stable patient split.

The original study draws each 500-row context with ``rng.choice`` per stratum. Changing
eligibility or weights redraws a stratum completely, so hard and graded sampling were not
paired with uniform sampling. Its removals also rank rows only by neighborhood priority,
which takes few distinct values, so ties fell back to population row order. The original outcome-independent
60/40 patient hash split is preserved. Sampling and removal order change only in
the added comparisons; ``context_intervention`` is not modified.

Sampling uses common random numbers: every context row gets one fixed uniform key per
(seed, draw). A draw keeps the rows with the largest ``log(key) / weight`` in each
year/outcome stratum (Efraimidis-Spirakis weighted sampling without replacement).
Uniform sampling is the special case of equal weights; excluding rows only lets the next
keys in their strata enter, so the remaining rows of the uniform draw are kept.
"""

import numpy as np

from context_intervention import data, experiment, sampling
from context_intervention.analysis import performance


def development_rows(reference, last_year, partition):
    """Reuse the original stable patient hash split and past-only row eligibility.

    Assignment depends only on patient identity, not future mortality or entrants.
    Death counts can be uneven; report support and retain existing fallbacks.
    """
    return data.development_rows(reference, last_year, partition)


def sample_context(years, labels, size, seed, eligible=None, weights=None):
    """Common-random-number draw that keeps original stratum quotas."""
    keys = np.asarray(sampling.strata(years, labels))
    n = len(keys)
    eligible = np.ones(n, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if eligible.shape != (n,) or weights.shape != (n,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid eligibility or weights")
    uniform = 1 - np.random.default_rng(seed).random(n)  # in (0, 1], so the logarithm is finite
    usable = eligible & (weights > 0)
    score = np.full(n, -np.inf)
    score[usable] = np.log(uniform[usable]) / weights[usable]
    chosen = []
    for key, count in sampling.quotas(keys, size).items():
        rows = np.flatnonzero(np.all(keys == key, axis=1) & usable)
        if len(rows) < count:
            raise ValueError(f"Insufficient eligible context rows in stratum {key}")
        chosen.extend(rows[np.argsort(-score[rows], kind="stable")[:count]])
    return np.sort(np.asarray(chosen, dtype=int))


def similarity(context_vectors, patient_vectors):
    """Largest cosine similarity of each context row to the given patients (as in retrieval)."""
    context = np.asarray(context_vectors, dtype=float)
    patients = np.asarray(patient_vectors, dtype=float)
    if not len(patients):
        return np.zeros(len(context))
    cn = np.linalg.norm(context, axis=1, keepdims=True)
    pn = np.linalg.norm(patients, axis=1, keepdims=True)
    context = np.divide(context, cn, out=np.zeros_like(context), where=cn > 0)
    patients = np.divide(patients, pn, out=np.zeros_like(patients), where=pn > 0)
    return (context @ patients.T).max(axis=1)


def select_removals(years, labels, priorities, closeness, fraction, sample_size):
    """Original removal counts per stratum; equal priorities are broken by similarity.

    Counts match ``sampling.select_removals`` exactly. Only which rows go changes: among rows
    with the same priority, those most similar to the representative patients go first.
    """
    keys = np.asarray(sampling.strata(years, labels))
    priorities = np.asarray(priorities, dtype=float)
    closeness = np.asarray(closeness, dtype=float)
    if (priorities.shape != (len(keys),) or closeness.shape != (len(keys),) or not np.isfinite(priorities).all()
            or not np.isfinite(closeness).all() or not 0 <= fraction <= 1):
        raise ValueError("Invalid priorities, similarities or removal fraction")
    eligible = np.ones(len(keys), dtype=bool)
    for key, quota in sampling.quotas(keys, sample_size).items():
        rows = np.flatnonzero(np.all(keys == key, axis=1))
        positive = rows[priorities[rows] > 0]
        remove = min(int(np.floor(len(positive) * fraction)), max(0, len(rows) - max(quota, 1)))
        order = np.lexsort((-closeness[positive], -priorities[positive]))
        eligible[positive[order[:remove]]] = False
    if set(np.asarray(labels)[eligible]) != {0, 1}:
        raise ValueError("Retained context needs both outcome classes")
    return eligible


def draw_seed(seed, draw):
    return experiment.stable_seed(seed, draw, "paired_keys")


def predictions(reference, priorities, closeness, queries, form, strength, store, size=500, draws=4,
                seeds=experiment.SEEDS):
    """One probability vector per sampling seed; sampled forms share keys with uniform."""
    context, population = reference["context_rows"], reference["population"]
    years, labels = population["year"][context], population["y"][context]
    if form == "original":
        probabilities, _ = data.cached_predictor_rows(reference, queries)
        return np.repeat(probabilities[None], len(seeds), axis=0)
    retained = (select_removals(years, labels, priorities, closeness, strength, size)
                if form in ("direct", "hard_sampling") else np.ones(len(context), dtype=bool))
    if form == "direct":
        if retained.all():
            return predictions(reference, priorities, closeness, queries, "original", 0, store, size, draws, seeds)
        return np.repeat(store(context[retained], queries)[None], len(seeds), axis=0)
    if form not in ("uniform", "hard_sampling", "graded_sampling"):
        raise ValueError("Unknown intervention form")
    weights = sampling.graded_weights(priorities, strength) if form == "graded_sampling" else None
    output = []
    for seed in seeds:
        results = [store(context[sample_context(years, labels, size, draw_seed(seed, draw), retained, weights)], queries)
                   for draw in range(draws)]
        output.append(np.mean(results, axis=0))
    return np.asarray(output)


def random_removal_predictions(reference, priorities, closeness, queries, fraction, store, size=500,
                               seeds=experiment.SEEDS):
    """Random direct removal with the selected count in every year/outcome stratum."""
    context, population = reference["context_rows"], reference["population"]
    years, labels = population["year"][context], population["y"][context]
    retained = select_removals(years, labels, priorities, closeness, fraction, size)
    if retained.all():
        return predictions(reference, priorities, closeness, queries, "original", 0, store, size, seeds=seeds)
    result = []
    for seed in seeds:
        removed = sampling.random_matched_removal(years, labels, ~retained,
                                                  experiment.stable_seed(seed, "paired_random_removal"))
        result.append(store(context[~removed], queries))
    return np.asarray(result)


def neighborhood_priorities(reference, neighborhoods, queries, threshold, mortality, store,
                            size=500, draws=4, seeds=experiment.SEEDS):
    """Death-F1 change from excluding each neighborhood, paired with the same uniform draw.

    The excluded draw keeps every uniform row outside the neighborhood; the freed slots take
    the next keys in their strata.
    """
    context, population = reference["context_rows"], reference["population"]
    years, labels = population["year"][context], population["y"][context]
    outcomes = population["y"][queries]
    if len(np.unique(outcomes)) < 2:
        return np.zeros(len(context)), []
    effects = [[] for _ in neighborhoods]
    for seed in seeds:
        print(f"    Neighborhood exclusions, seed {seed}", flush=True)
        original, changed = [], [[] for _ in neighborhoods]
        for draw in range(draws):
            original.append(store(context[sample_context(years, labels, size, draw_seed(seed, draw))], queries))
            for index, neighborhood in enumerate(neighborhoods):
                eligible = np.ones(len(context), dtype=bool)
                eligible[neighborhood] = False
                changed[index].append(store(context[sample_context(years, labels, size, draw_seed(seed, draw),
                                                                   eligible)], queries))
        score = performance(outcomes, np.mean(original, axis=0), threshold, mortality)["death_f1"]
        for index in range(len(neighborhoods)):
            new_score = performance(outcomes, np.mean(changed[index], axis=0), threshold, mortality)["death_f1"]
            effects[index].append(new_score - score)
    effects = [float(np.mean(values)) for values in effects]
    return sampling.exclusion_priorities(len(context), neighborhoods, effects), effects
