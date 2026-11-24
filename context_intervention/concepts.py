"""Compare frozen concept activations under selected context changes."""

import numpy as np
import pandas as pd

from concept_study import measures
from context_intervention import data, experiment, predictor, sampling
from context_intervention.analysis import GROUPS, prediction_groups


def selected_contexts(reference, priorities, form, strength, sample_size=500, draws=4, seeds=experiment.SEEDS):
    """Regenerate the exact context rows used by an intervention."""
    context, population = reference["context_rows"], reference["population"]
    years, labels = population["year"][context], population["y"][context]
    if form in ("original", "direct"):
        retained = (np.ones(len(context), dtype=bool) if form == "original" else
                    sampling.select_removals(years, labels, priorities, strength, sample_size))
        return [(None, None, context[retained])]
    if form not in ("uniform", "hard_sampling", "graded_sampling"):
        raise ValueError("Unknown intervention")
    eligible = (sampling.select_removals(years, labels, priorities, strength, sample_size)
                if form == "hard_sampling" else None)
    weights = sampling.graded_weights(priorities, strength) if form == "graded_sampling" else None
    result = []
    for seed in seeds:
        for draw in range(draws):
            chosen = sampling.sample_context(years, labels, sample_size,
                                             experiment.stable_seed(seed, draw, "uniform"), eligible, weights)
            result.append((seed, draw, context[chosen]))
    return result


def comparison(reference, query_rows, priorities, form, strength, threshold, records,
               sample_size=500, draws=4, seeds=experiment.SEEDS, device="cuda"):
    """Return per-draw factor and rule summaries; do not threshold averaged codes."""
    _, build = next((item, measures.load_build(item)) for item in measures.load_index()["builds"]
                    if item["build"] == reference["build"] and item["reference"] == reference["reference"])
    population = reference["population"]
    lookup = {(str(patient), int(year)): index for index, (patient, year) in
              enumerate(zip(build["patient_id"], build["year"]))}
    positions = np.array([lookup[(str(population["patient_id"][row]), int(population["year"][row]))]
                          for row in query_rows], dtype=int)
    groups = prediction_groups(build["label"][positions], build["probability"][positions], threshold)
    originals = [measures.load_concept(record) for record in records]
    output = []
    for seed, draw, rows in selected_contexts(reference, priorities, form, strength, sample_size, draws, seeds):
        if form == "original":
            activations = np.column_stack([unit["activation"][positions] for unit in originals])
            actual_device = "cached"
        else:
            _, embeddings, actual_device = predictor.context_predictions(reference, rows, query_rows, device)
            activations = data.factor_activations(reference, embeddings, records, device)
        for column, record in enumerate(records):
            rule = originals[column]["in_R"][positions]
            active = activations[:, column] > record["cutoff"]
            for group in GROUPS:
                selected = groups == group
                in_rule, outside_rule = selected & rule, selected & ~rule
                if not selected.any():
                    continue
                output.append({"concept": record["concept"], "group": group, "seed": seed, "draw": draw,
                               "device": actual_device, "n": int(selected.sum()), "n_rule": int(in_rule.sum()),
                               "mean_activation": float(activations[selected, column].mean()),
                               "mean_activation_in_rule": float(activations[in_rule, column].mean()) if in_rule.any() else np.nan,
                               "activation_rate": float(active[selected].mean()),
                               "activation_rate_in_rule": float(active[in_rule].mean()) if in_rule.any() else np.nan,
                               "activation_rate_outside_rule": float(active[outside_rule].mean()) if outside_rule.any() else np.nan})
    return pd.DataFrame(output)
