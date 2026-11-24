"""Build the cached patient-year table: one file per reference build, one file per concept.

Only held-out evaluation patients are kept. For each reference build the file holds,
per patient-year row: patient id, year, death label (missing after 2014), predicted
death probability, original input features, all SAE activations of the three frozen
dictionaries and their reconstruction errors. For each concept the file holds the
medoid factor activation, the strong-activation (A) and rule (R) memberships, the
sensitivity of every row to each of the 100 frozen CAV fits (50 all-negative, 50
legacy) and the change in predicted risk / Brier loss when the factor's decoder
contribution is removed from the embedding.

Every file is written under a temporary name and renamed, so readers never see a partial file.
Run once on Fiji (about 10 minutes on the GPU; add --force to rebuild):
    .venv/bin/python -m concept_study.patient_table
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

from concept_study import loading, measures, paths


def concept_file_name(key: str) -> str:
    """File-system safe name of a concept key such as 'last_3/2011/2011/42/42:49,...'."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key) + ".npz"


def frozen_thresholds() -> dict:
    """{(variant, reference): death probability cut that maximised Death-F1 on reference roles}."""
    thresholds = {}
    for variant in paths.VARIANTS:
        freeze = json.loads((paths.PREVIOUS_RUNS / f"develop-{variant}/freeze.json").read_text())
        for year, record in freeze["thresholds"].items():
            if record["threshold"] is not None:
                thresholds[(variant, int(year))] = float(record["threshold"])
    return thresholds


def rule_features(rule_dict: dict) -> list:
    """Names of the input features used by a frozen rule, in order of first use."""
    names = []
    for rule in rule_dict.get("rules", []):
        for condition in rule.get("conditions", []):
            if condition["feature_name"] not in names:
                names.append(condition["feature_name"])
    return names


def reconstruction_errors(dictionaries: dict, embeddings: np.ndarray, device: str = paths.DEVICE) -> dict:
    """Per-row mean squared SAE reconstruction error for each frozen dictionary."""
    import concept_subgroup_drift_pilot as pilot
    values = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    return {seed: pilot.reconstruction_mse(dictionary, values).cpu().numpy().astype(np.float32)
            for seed, dictionary in dictionaries.items()}


def sensitivities(gradients: np.ndarray, directions: np.ndarray, device: str = paths.DEVICE) -> np.ndarray:
    """Gradient of death probability projected on each unit CAV direction (rows x fits).

    Failed fits (NaN directions) give NaN columns.
    """
    g = torch.as_tensor(gradients, dtype=torch.float32, device=device)
    d = torch.as_tensor(np.nan_to_num(directions), dtype=torch.float32, device=device)
    values = (g @ d.T).cpu().numpy()
    values[:, ~np.isfinite(directions).all(axis=1)] = np.nan
    return values


def removal_changes(unit: dict, rows: np.ndarray) -> tuple:
    """Change in predicted death probability and Brier loss after removing the factor's contribution.

    Same operation as the November study: subtract activation x decoder row x scaler
    scale from the raw embedding (the SAE residual is kept) and rescore with the frozen head.
    Both sides go through the same head call (same batches), so a row with zero activation
    gives exactly zero change instead of replay noise against the cached probabilities.
    """
    import concept_subgroup_drift_pilot as pilot
    embeddings = unit["embeddings"][rows]
    decoder_row = unit["decoders"][unit["seed"]][unit["factor"]]
    component = unit["activation"][rows, None] * (decoder_row * unit["scale"])
    original = pilot.batched_death_probabilities(unit["continuation"], embeddings, 256)[:, 1]
    removed = pilot.batched_death_probabilities(unit["continuation"], embeddings - component, 256)[:, 1]
    probability_change = (removed - original).astype(np.float32)
    labels = unit["labels"][rows]
    brier_change = ((removed - labels) ** 2 - (original - labels) ** 2).astype(np.float32)
    return probability_change, brier_change


def check_cached_probabilities(base: dict, rows: np.ndarray) -> float:
    """Rescore a sample of rows with the frozen head; they must match the cached probabilities."""
    import concept_subgroup_drift_pilot as pilot
    sample = rows[:64]
    replayed = pilot.batched_death_probabilities(base["continuation"], base["embeddings"][sample], 64)
    np.testing.assert_allclose(replayed, base["probabilities"][sample], rtol=1e-4, atol=1e-5)
    return float(np.max(np.abs(replayed - base["probabilities"][sample])))


def save_build(inputs: dict, base: dict, build: str, reference: int, output: Path,
               device: str = paths.DEVICE) -> np.ndarray:
    """Write the per-build file and return the evaluation row positions inside the build."""
    rows = np.flatnonzero(base["roles"] == "evaluation")
    labels, years = base["labels"][rows].astype(np.float32), base["years"][rows].astype(int)
    if not (np.isnan(labels[years > paths.LAST_OUTCOME_YEAR]).all()
            and np.isfinite(labels[years <= paths.LAST_OUTCOME_YEAR]).all()):
        raise ValueError(f"{build} {reference}: labels must be missing after {paths.LAST_OUTCOME_YEAR} and present up to it")
    dictionaries = loading.reference_dictionaries(inputs, build, reference, device)
    errors = reconstruction_errors(dictionaries, base["embeddings"][rows], device)
    reference_rows = (base["years"] == reference) & (base["roles"] == "rule_discovery")
    arrays = {
        "patient_id": base["patients"][rows].astype(str),
        "year": years,
        "label": labels,
        "probability": base["probabilities"][rows, 1].astype(np.float32),
        "features": base["X"][rows].astype(np.float32),
        "feature_names": np.array(base["feature_names"]),
        "reference_mortality": np.float32(np.nanmean(base["labels"][reference_rows])),
        "replay_error": np.float32(check_cached_probabilities(base, rows)),
    }
    for seed in paths.SAE_SEEDS:
        arrays[f"codes_{seed}"] = base["codes"][seed][rows].astype(np.float16)
        arrays[f"reconstruction_error_{seed}"] = errors[seed]
    measures.save_atomic(output / f"build_{build}_{reference}.npz", **arrays)
    return rows


def save_concept(unit: dict, rows: np.ndarray, directions: dict, fit_counts: dict, output: Path,
                 device: str = paths.DEVICE) -> dict:
    """Write the per-concept file and return its index record.

    'cav_fits' records, per estimator, why a concept has no usable CAV fits: "no key" (the concept
    is absent from the November freeze) or "0 of n measured"; otherwise "k of n measured".
    """
    activation = unit["activation"][rows].astype(np.float32)
    arrays = {
        "activation": activation,
        "in_A": activation > unit["cutoff"],
        "in_R": unit["rule_mask"][rows].astype(bool),
    }
    fits, status = {}, {}
    for estimator in paths.ESTIMATORS:
        counts = fit_counts.get((estimator, unit["key"]))
        status[estimator] = "no key" if counts is None else f"{counts[1]} of {counts[0]} measured"
        matrix = directions.get((estimator, unit["key"]))
        if matrix is not None:
            arrays[f"sensitivity_{estimator}"] = sensitivities(unit["gradients"][rows], matrix, device)
            fits[estimator] = int(np.isfinite(matrix).all(axis=1).sum())
    arrays["removal_probability_change"], arrays["removal_brier_change"] = removal_changes(unit, rows)
    name = concept_file_name(unit["key"])
    measures.save_atomic(output / "concepts" / name, **arrays)
    return {"concept": unit["key"], "file": name, "build": unit["key"].split("/")[0],
            "reference": unit["reference"], "family": unit["family"], "variants": unit["variants"],
            "seed": unit["seed"], "factor": unit["factor"], "cutoff": float(unit["cutoff"]),
            "rule_features": rule_features(unit["rule_dict"]), "valid_fits": fits, "cav_fits": status}


def build_patient_table(output: Path = paths.PATIENT_TABLE, force: bool = False, device: str = paths.DEVICE) -> Path:
    """Build every per-build and per-concept file plus index.json; skip when already built (unless force).

    Each build record lists the variants it serves; their frozen Death-F1 thresholds must agree.
    """
    index_path = output / "index.json"
    if index_path.exists() and not force:
        return index_path
    (output / "concepts").mkdir(parents=True, exist_ok=True)
    loading.extract_cav_directions(force=force)
    directions, fit_counts = loading.load_cav_directions()
    inputs = loading.open_inputs(device)
    thresholds = frozen_thresholds()
    served = {}
    for item in inputs["inventory"]:
        served.setdefault((item["build"], item["reference"]), set()).update(item["variants"])
    builds, concepts = [], []
    current, base, rows = None, None, None
    for item in inputs["inventory"]:
        if current != (item["build"], item["reference"]):
            current = (item["build"], item["reference"])
            base = loading.load_build(inputs, item["build"], item["reference"], item["roles"])
            rows = save_build(inputs, base, item["build"], item["reference"], output, device)
            threshold = {v: thresholds[(v, item["reference"])] for v in paths.VARIANTS}
            builds.append({"build": item["build"], "reference": item["reference"],
                           "file": f"build_{item['build']}_{item['reference']}.npz",
                           "variants": [v for v in paths.VARIANTS if v in served[current]],
                           "death_threshold": threshold})
            measures.death_threshold(builds[-1])
            print(f"build {current}: {len(rows)} evaluation patient-years", flush=True)
        unit = loading.concept_unit(inputs, base, item)
        concepts.append(save_concept(unit, rows, directions, fit_counts, output, device))
    temporary = index_path.with_name("index.tmp.json")
    temporary.write_text(json.dumps({"builds": builds, "concepts": concepts}, indent=1))
    os.replace(temporary, index_path)
    return index_path


if __name__ == "__main__":
    print(build_patient_table(force="--force" in sys.argv))
