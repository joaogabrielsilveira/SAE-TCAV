"""Native-context comparison and explicit checks of frozen-SAE transfer."""

import numpy as np
import pandas as pd
import torch

from concept_study import loading
from context_comparisons import native
from context_comparisons.diagnostics import factor_summary, groups, reconstruction_summary
from context_intervention import data, representations as original


def native_concepts(reference, records, output, original_output, device="cuda"):
    """Change only the context role, using the same frozen factor coordinates.

    Native states contain the known context outcome. Applying a query-trained
    SAE to these states is an explicit transfer experiment, not a claim that
    its factors retain identical meanings in both roles.
    """
    import concept_subgroup_drift_pilot as pilot

    states = native.extract(reference, output / "native_states")
    query_states, _ = original.original_context_embeddings(reference, original_output / "embeddings")
    reference_rows = np.concatenate([data.development_rows(reference, reference["reference"], part)
                                     for part in ("development", "validation")])
    _, development_states = data.cached_predictor_rows(reference, reference_rows)
    combined = np.vstack((states, query_states, development_states))
    factors = data.factor_activations(reference, combined, records, device)
    count = len(states)
    native_factors, query_factors, development_factors = factors[:count], factors[count:2*count], factors[2*count:]
    standardized = data.standardized(native_factors, development_factors)
    dictionary = {item["family"]: item for item in reference["inputs"]["inventory"]
                  if item["build"] == reference["build"] and item["reference"] == reference["reference"]}
    context_features = reference["population"]["X"][reference["context_rows"]]
    years = reference["population"]["year"][reference["context_rows"]]
    outcomes = reference["population"]["y"][reference["context_rows"]]
    diagnostics = []
    for column, record in enumerate(records):
        rule = dictionary[record["family"]]["frozen"]["rule"].mask(context_features)
        for group, mask in groups(years, outcomes):
            diagnostics.append({"build": reference["build"], "reference": reference["reference"],
                "concept": record["concept"], **group,
                **factor_summary(native_factors[mask, column], query_factors[mask, column], rule[mask], record["cutoff"])})
    pd.DataFrame(diagnostics).to_csv(output / f"native_factor_diagnostics_{reference['build']}_{reference['reference']}.csv",
                                   index=False)
    dictionaries = loading.reference_dictionaries(reference["inputs"], reference["build"], reference["reference"], device)
    reconstruction = []
    for seed, model in dictionaries.items():
        values = torch.as_tensor(np.vstack((states, query_states)), dtype=torch.float32, device=device)
        with torch.no_grad():
            errors = pilot.reconstruction_mse(model, values).detach().cpu().numpy()
        scaled = model["scaler"].transform(np.vstack((states, query_states)))
        energy = np.square(scaled).mean(axis=1)
        for group, mask in groups(years, outcomes):
            for role, offset in (("native", 0), ("query", count)):
                reconstruction.append({"seed": seed, "role": role, **group,
                    **reconstruction_summary(errors[offset:offset + count][mask], energy[offset:offset + count][mask])})
    pd.DataFrame(reconstruction).to_csv(output / f"native_reconstruction_{reference['build']}_{reference['reference']}.csv",
                                       index=False)
    return standardized
