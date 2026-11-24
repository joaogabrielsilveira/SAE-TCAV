"""Reuse the existing intervention definitions with a shared prediction store."""

import json

import numpy as np

from context_intervention import data, experiment, sampling
from context_intervention.analysis import performance


def policy_metadata(reference, priorities, form, strength, size=500, requested_form=None, reason=""):
    """Describe what is actually supplied, including full-context fallbacks."""
    context, population = reference["context_rows"], reference["population"]
    years, labels = population["year"][context], population["y"][context]
    retained = np.ones(len(context), dtype=bool)
    removed = 0
    if form in ("direct", "hard_sampling"):
        retained = sampling.select_removals(population["year"][context], population["y"][context],
                                            priorities, strength, size)
        removed = int((~retained).sum())
    effective = "original" if form == "direct" and removed == 0 else form
    supplied = len(context) - removed if effective == "direct" else len(context) if effective == "original" else size
    composition_rows = np.flatnonzero(retained) if effective == "direct" else np.arange(len(context))
    composition_size = supplied if effective in ("uniform", "hard_sampling", "graded_sampling") else len(composition_rows)
    composition = sampling.quotas(sampling.strata(years[composition_rows], labels[composition_rows]), composition_size)
    return {"requested_form": requested_form or form, "effective_form": effective,
            "context_size": supplied, "removed_rows": removed, "fallback": bool(reason),
            "context_composition": json.dumps({f"{year}/{label}": count for (year, label), count in composition.items()}, sort_keys=True),
            "fallback_reason": reason, "common_size_eligible": effective in ("uniform", "hard_sampling", "graded_sampling"),
            "guided_adjustment": effective in ("direct", "hard_sampling", "graded_sampling") and strength > 0,
            "secondary_threshold_status": "unavailable: equivalent validation tuning not implemented for added comparisons"}


def predictions(reference, priorities, queries, form, strength, store, size=500, draws=4,
                seeds=experiment.SEEDS):
    context, population = reference["context_rows"], reference["population"]
    years, labels = population["year"][context], population["y"][context]
    if form == "original":
        probabilities, _ = data.cached_predictor_rows(reference, queries)
        return np.repeat(probabilities[None], len(seeds), axis=0)
    if form in ("direct", "hard_sampling"):
        retained = sampling.select_removals(years, labels, priorities, strength, size)
    else:
        retained = np.ones(len(context), dtype=bool)
    if form == "direct":
        if retained.all():
            return predictions(reference, priorities, queries, "original", 0, store, size, draws, seeds)
        probabilities = store(context[retained], queries)
        return np.repeat(probabilities[None], len(seeds), axis=0)
    if form not in ("uniform", "hard_sampling", "graded_sampling"):
        raise ValueError("Unknown intervention form")
    weights = sampling.graded_weights(priorities, strength) if form == "graded_sampling" else None
    output = []
    for seed in seeds:
        results = []
        for draw in range(draws):
            selected = sampling.sample_context(years, labels, size, experiment.stable_seed(seed, draw, "uniform"),
                                               eligible=retained, weights=weights)
            results.append(store(context[selected], queries))
        output.append(np.mean(results, axis=0))
    return np.asarray(output)


def random_removal_predictions(reference, priorities, queries, fraction, store, size=500,
                               seeds=experiment.SEEDS):
    context, population = reference["context_rows"], reference["population"]
    years, labels = population["year"][context], population["y"][context]
    retained = sampling.select_removals(years, labels, priorities, fraction, size)
    if retained.all():
        return predictions(reference, priorities, queries, "original", 0, store, size, seeds=seeds)
    result = []
    for seed in seeds:
        removed = sampling.random_matched_removal(years, labels, ~retained,
                                                  experiment.stable_seed(seed, "random_removal"))
        result.append(store(context[~removed], queries))
    return np.asarray(result)


def neighborhood_priorities(reference, neighborhoods, queries, threshold, mortality, store,
                             size=500, draws=4, seeds=experiment.SEEDS):
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
            selected = sampling.sample_context(years, labels, size, experiment.stable_seed(seed, draw, "uniform"))
            original.append(store(context[selected], queries))
            for index, neighborhood in enumerate(neighborhoods):
                excluded = np.zeros(len(context), dtype=bool)
                excluded[neighborhood] = True
                replacement = sampling.replace_neighborhood(years, labels, selected, excluded,
                                                             experiment.stable_seed(seed, draw, "replace", index))
                changed[index].append(store(context[replacement], queries))
        score = performance(outcomes, np.mean(original, axis=0), threshold, mortality)["death_f1"]
        for index in range(len(neighborhoods)):
            new_score = performance(outcomes, np.mean(changed[index], axis=0), threshold, mortality)["death_f1"]
            effects[index].append(new_score - score)
    effects = [float(np.mean(values)) for values in effects]
    return sampling.exclusion_priorities(len(context), neighborhoods, effects), effects
