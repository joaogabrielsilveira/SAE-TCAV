"""Validate the live frozen identities before reusing original study artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from concept_study import measures, paths
from context_intervention import data, experiment, predictor, study
from context_intervention.analysis import primary_builds


def current(reference, sample_size, draws):
    record, concepts = study.reference_info(reference["build"], reference["reference"])
    sources = sorted(Path("context_intervention").glob("*.py"))
    sources += [Path("concept_subgroup_drift_pilot.py"), Path("tabpfn_model.py")]
    tables = [paths.PATIENT_TABLE / record["file"]]
    tables += [paths.PATIENT_TABLE / "concepts" / item["file"] for item in concepts]
    dictionaries = {}
    for seed in reference["cache"].config["sae_seeds"]:
        unit = reference["cache"].load(f"S/{seed}/{reference['reference']}")
        dictionaries[str(seed)] = {key: unit[key] for key in ("sae_identity", "scaler_identity")}
    digest = hashlib.sha256()
    for name in ("row_index", "probabilities", "embeddings"):
        digest.update(np.ascontiguousarray(reference["predictor"][name]).tobytes())
    return {"context_identity": reference["predictor"]["context_identity"],
            "checkpoint_sha256": reference["predictor"]["checkpoint_sha256"],
            "population_fingerprint": str(reference["population"]["fingerprint"]),
            "patient_table_index": measures.index_hash(), "sample_size": sample_size, "draws": draws,
            "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "patient_tables": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tables},
            "dictionaries": dictionaries, "cached_predictions": digest.hexdigest()}


def validate(reference, recorded):
    actual = current(reference, recorded["sample_size"], recorded["draws"])
    if actual != recorded:
        changed = sorted(key for key in set(actual) | set(recorded) if actual.get(key) != recorded.get(key))
        raise ValueError(f"Current frozen identities differ: {changed}")
    return actual


def validate_artifacts(reference, update, directory):
    name = f"{reference['build']}_{reference['reference']}_{update}"
    details = json.loads((directory / f"development_{name}.json").read_text())
    validate(reference, details["provenance"])
    for representation in ("features", "concepts"):
        values = np.load(directory / f"priorities_{name}_{representation}.npy", allow_pickle=False)
        if values.shape != (len(reference["context_rows"]),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid original priorities: {name}, {representation}")
    if pd.read_csv(directory / f"concepts_{name}.csv.gz").empty:
        raise ValueError(f"Missing original concept measurements: {name}")
    predictions = pd.read_csv(directory / f"predictions_{name}.csv.gz", dtype={"patient_id": str})
    queries = data.evaluation_rows(reference, update + 1)
    population = reference["population"]
    expected = sorted(zip(population["patient_id"][queries].astype(str), population["y"][queries].astype(int)))
    required = {"original", "uniform_sampling"}
    required |= {f"{representation}_{form}" for representation in ("features", "concepts")
                 for form in ("direct", "hard_sampling", "graded_sampling")}
    if not required.issubset(set(predictions["method"])):
        raise ValueError(f"Original controls missing: {name}")
    if not np.isfinite(predictions["probability"]).all() or not predictions["probability"].between(0, 1).all():
        raise ValueError(f"Invalid original probabilities: {name}")
    for method, group in predictions.groupby("method"):
        if set(group["sampling_seed"]) != set(experiment.SEEDS):
            raise ValueError(f"Missing original sampling seed: {name}, {method}")
    for (_, _), group in predictions.groupby(["method", "sampling_seed"]):
        actual = sorted(zip(group["patient_id"], group["label"].astype(int)))
        if actual != expected or not (group["year"] == update + 1).all():
            raise ValueError(f"Original evaluation rows differ: {name}")
    return details


def validate_completed_study(directory):
    """Require each of the 21 expected identities and its data, rather than a count."""
    expected = set()
    for record in primary_builds(measures.load_index()):
        reference = predictor.open_reference(record["build"], record["reference"], "cpu")
        for update in range(record["reference"], 2014):
            expected.add(f"development_{record['build']}_{record['reference']}_{update}.json")
            validate_artifacts(reference, update, directory)
    actual = {path.name for path in directory.glob("development_*.json")}
    if actual != expected or len(expected) != 21:
        raise ValueError(f"Original completion set differs: missing={expected - actual}, extra={actual - expected}")
    return len(expected)
