"""Inference primitives for paired context comparisons.

These functions do not choose or report final intervention effects. The full
chronological study and explanatory re-analysis still need orchestration.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from context_intervention import predictor, sampling
from context_intervention.analysis import GROUPS, performance, prediction_groups


SEEDS = (42, 135, 215)


def stable_seed(*values):
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def score_context(reference, context_rows, query_rows, directory, device="cuda"):
    """Cache a modified-context inference only when its full identity matches."""
    context_rows = np.asarray(context_rows, dtype=np.int64)
    query_rows = np.asarray(query_rows, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(context_rows).tobytes())
    digest.update(np.ascontiguousarray(query_rows).tobytes())
    for value in (reference["predictor"]["context_identity"],
                  reference["predictor"]["checkpoint_sha256"],
                  reference["population"]["fingerprint"], reference["origin"], reference["seed"]):
        digest.update(str(value).encode())
    for path in (Path(__file__), Path(predictor.__file__),
                 Path("concept_subgroup_drift_pilot.py"), Path("tabpfn_model.py")):
        digest.update(path.read_bytes())
    identity = digest.hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{identity}.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as stored:
            if str(stored["identity"]) != identity:
                raise ValueError("Context prediction cache identity mismatch")
            return stored["probabilities"].copy(), str(stored["device"])
    probabilities, _, actual_device = predictor.context_predictions(reference, context_rows, query_rows, device)
    temporary = target.with_name(target.stem + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, probabilities=probabilities, device=actual_device, identity=identity)
    os.replace(temporary, target)
    return probabilities, actual_device


def make_neighborhoods(context_vectors, development_vectors, years, labels,
                       development_labels, development_probabilities, threshold, size=250):
    """One deduplicated neighborhood per observed prediction group."""
    groups = prediction_groups(development_labels, development_probabilities, threshold)
    neighborhoods, details, seen = [], [], set()
    for group in GROUPS:
        rows = np.flatnonzero(groups == group)
        if not len(rows):
            details.append({"group": group, "patients": 0, "status": "unavailable"})
            continue
        representative_rows = rows[sampling.representatives(development_vectors[rows])]
        neighbors = sampling.neighborhood(context_vectors, development_vectors[representative_rows], years, labels, size)
        signature = tuple(neighbors)
        if signature not in seen:
            neighborhoods.append(neighbors)
            seen.add(signature)
            status = "available"
        else:
            status = "duplicate"
        details.append({"group": group, "patients": len(rows), "representative_rows": representative_rows.tolist(),
                        "neighbors": len(neighbors), "status": status})
    return neighborhoods, details


def neighborhood_effects(reference, neighborhoods, query_rows, threshold, reference_mortality,
                         output, sample_size=500, draws=4, seeds=SEEDS):
    """Measure paired neighborhood exclusions and return row selection priorities."""
    population = reference["population"]
    context = reference["context_rows"]
    years, labels = population["year"][context], population["y"][context]
    query_labels = population["y"][query_rows]
    if len(np.unique(query_labels)) < 2:
        return np.zeros(len(context)), []
    effects = [[] for _ in neighborhoods]
    for seed in seeds:
        print(f"  Comparing neighborhood exclusions, sampling seed {seed}", flush=True)
        original_probabilities = []
        changed_probabilities = [[] for _ in neighborhoods]
        for draw in range(draws):
            draw_seed = stable_seed(seed, draw, "uniform")
            chosen = sampling.sample_context(years, labels, sample_size, draw_seed)
            original, _ = score_context(reference, context[chosen], query_rows, output)
            original_probabilities.append(original)
            for index, neighborhood in enumerate(neighborhoods):
                excluded = np.zeros(len(context), dtype=bool)
                excluded[neighborhood] = True
                changed = sampling.replace_neighborhood(years, labels, chosen, excluded,
                                                        stable_seed(seed, draw, "replace", index))
                probabilities, _ = score_context(reference, context[changed], query_rows, output)
                changed_probabilities[index].append(probabilities)
        baseline = performance(query_labels, np.mean(original_probabilities, axis=0), threshold, reference_mortality)["death_f1"]
        for index in range(len(neighborhoods)):
            score = performance(query_labels, np.mean(changed_probabilities[index], axis=0), threshold,
                                reference_mortality)["death_f1"]
            effects[index].append(score - baseline)
    mean_effects = [float(np.mean(values)) for values in effects]
    return sampling.exclusion_priorities(len(context), neighborhoods, mean_effects), mean_effects


def intervention_predictions(reference, priorities, query_rows, form, strength, output,
                             sample_size=500, draws=4, seeds=SEEDS):
    """Return one probability vector per sampling seed for one candidate intervention."""
    from context_intervention.data import cached_predictor_rows

    context = reference["context_rows"]
    population = reference["population"]
    years, labels = population["year"][context], population["y"][context]
    priorities = np.asarray(priorities, dtype=float)
    if priorities.shape != (len(context),):
        raise ValueError("Priorities must have one value per original context row")
    if form == "original":
        probabilities, _ = cached_predictor_rows(reference, query_rows)
        return np.repeat(probabilities[None, :], len(seeds), axis=0)
    if form == "direct":
        retained = sampling.select_removals(years, labels, priorities, strength, sample_size)
        if retained.all():
            probabilities, _ = cached_predictor_rows(reference, query_rows)
            return np.repeat(probabilities[None, :], len(seeds), axis=0)
        probabilities, _ = score_context(reference, context[retained], query_rows, output)
        return np.repeat(probabilities[None, :], len(seeds), axis=0)
    if form not in ("uniform", "hard_sampling", "graded_sampling"):
        raise ValueError(f"Unknown intervention form: {form}")
    eligible = (sampling.select_removals(years, labels, priorities, strength, sample_size)
                if form == "hard_sampling" else None)
    weights = sampling.graded_weights(priorities, strength) if form == "graded_sampling" else None
    result = []
    for seed in seeds:
        probabilities = []
        for draw in range(draws):
            selected = sampling.sample_context(years, labels, sample_size, stable_seed(seed, draw, "uniform"),
                                               eligible=eligible, weights=weights)
            prediction, _ = score_context(reference, context[selected], query_rows, output)
            probabilities.append(prediction)
        result.append(np.mean(probabilities, axis=0))
    return np.asarray(result)


def direct_random_control(reference, priorities, fraction, query_rows, output, sample_size=500, seeds=SEEDS):
    """Direct deletion control matched to the selected deletion in every stratum."""
    context = reference["context_rows"]
    population = reference["population"]
    years, labels = population["year"][context], population["y"][context]
    retained = sampling.select_removals(years, labels, priorities, fraction, sample_size)
    if retained.all():
        return intervention_predictions(reference, priorities, query_rows, "original", 0, output,
                                        sample_size=sample_size, seeds=seeds)
    result = []
    for seed in seeds:
        removed = sampling.random_matched_removal(years, labels, ~retained, stable_seed(seed, "random_removal"))
        probabilities, _ = score_context(reference, context[~removed], query_rows, output)
        result.append(probabilities)
    return np.asarray(result)


def choose_intervention(reference, priorities, validation_rows, form, threshold, reference_mortality,
                        output, sample_size=500, draws=4, seeds=SEEDS):
    """Choose intervention strength on validation patients, independently for each form."""
    labels = reference["population"]["y"][validation_rows]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {"form": "original", "strength": 0., "reason": "validation lacks both outcomes"}, []
    if form == "direct":
        options = [("original", 0.), ("direct", .25), ("direct", .5)]
    elif form == "hard_sampling":
        options = [("uniform", 0.), ("hard_sampling", .25), ("hard_sampling", .5)]
    elif form == "graded_sampling":
        options = [("uniform", 0.), ("graded_sampling", np.log(2)), ("graded_sampling", np.log(4))]
    else:
        raise ValueError("Unknown intervention form")
    records = []
    for candidate, strength in options:
        probabilities = intervention_predictions(reference, priorities, validation_rows, candidate, strength,
                                                  output, sample_size, draws, seeds)
        metrics = [performance(labels, p, threshold, reference_mortality) for p in probabilities]
        records.append({"form": candidate, "strength": float(strength),
                        "death_f1": float(np.mean([m["death_f1"] for m in metrics])),
                        "brier_skill": float(np.mean([m["brier_skill"] for m in metrics]))})
    winner = max(records, key=lambda item: (item["death_f1"] if np.isfinite(item["death_f1"]) else -np.inf,
                                             item["brier_skill"] if np.isfinite(item["brier_skill"]) else -np.inf,
                                             -item["strength"]))
    return dict(winner), records
