"""Thin adapters over the November study loaders (cmr_common.py).

The November notebooks wrap these loaders in run guards (budget, output namespace,
source snapshot). This study only reads the same frozen objects, so it builds the
small context dictionary those loaders expect and calls them directly.
"""
import hashlib
import os
import pickle
from pathlib import Path

import numpy as np

import cmr_common
from concept_study import paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loader_context(device: str = paths.DEVICE) -> dict:
    """Context dictionary with the fields read by the cmr_common loaders."""
    root = paths.CODE_ROOT
    pilot_sources = ("concept_subgroup_drift_pilot.py", "concept_subgroup_drift_pilot.ipynb")
    return {
        "mode": "evaluate",
        "branch": "measurement",
        "root": root,
        "device": device,
        "input_files": {},
        "coverage": [],
        "source_hashes": {name: _sha256(root / name) for name in pilot_sources},
        "CMR_PILOT_ROOT": paths.PILOT_ROOT,
        "CMR_BUILDS_ROOT": paths.BUILDS_ROOT,
        "CMR_POPULATION": paths.POPULATION,
        "CMR_PREVIOUS_RUNS": paths.PREVIOUS_RUNS,
    }


def open_inputs(device: str = paths.DEVICE) -> dict:
    """Open caches, population and the family inventory restricted to the study variants."""
    ctx = loader_context(device)
    caches = cmr_common.open_caches(ctx)
    population, first_year = cmr_common.load_study_population(ctx)
    inventory = cmr_common.study_inventory(ctx, "measurement", caches, population)
    kept = []
    for _, build, reference, aliases, family, frozen, family_record, roles in inventory:
        variants = [name for name in aliases if name in paths.VARIANTS]
        if variants and reference in paths.REFERENCES:
            kept.append({"build": build, "reference": reference, "variants": variants, "family": family,
                         "frozen": frozen, "family_record": family_record, "roles": roles})
    kept.sort(key=lambda item: (item["build"], item["reference"], item["family"]))
    return {"ctx": ctx, "caches": caches, "population": population, "first_year": first_year, "inventory": kept}


def load_build(inputs: dict, build: str, reference: int, roles: dict) -> dict:
    """Embeddings, gradients, probabilities, SAE codes and frozen head of one reference build."""
    return cmr_common.load_allowed_build(inputs["ctx"], "measurement", inputs["population"],
                                         inputs["caches"][build], reference, roles, inputs["first_year"])


def concept_unit(inputs: dict, base: dict, item: dict) -> dict:
    """One concept (factor family) with its cutoff, frozen rule mask and medoid factor."""
    unit = cmr_common.family_unit(inputs["ctx"], base, inputs["caches"][item["build"]], inputs["population"],
                                  item["build"], item["reference"], item["variants"], item["family"],
                                  item["frozen"], item["family_record"], item["roles"])
    unit["variants"] = item["variants"]
    return unit


def reference_dictionaries(inputs: dict, build: str, reference: int, device: str = paths.DEVICE) -> dict:
    """Frozen SAE dictionaries of one reference build (for reconstruction error)."""
    import concept_subgroup_drift_pilot as pilot
    return pilot.load_reference_dictionaries(inputs["caches"][build], reference, device)


def concept_key(build: str, reference: int, family: str) -> str:
    """Key used by the November study for one concept."""
    return f"{build}/{reference}/{family}"


def extract_cav_directions(output: Path = paths.CAV_DIRECTIONS, force: bool = False) -> Path:
    """Read the 50 frozen C=0.1 CAV fits of each estimator once and save their unit directions.

    The November freeze is 1.2 GB; after this runs, the study only reads the small file.
    Fits that failed are stored as NaN rows so fit positions stay aligned with seeds 1001-1050.
    'fits|<estimator>|<key>' holds (listed fits, measured fits) for every concept key in the freeze,
    so a concept without directions can be told apart: key absent from the freeze, or no measured fit.
    """
    if output.exists() and not force:
        return output
    expected = paths.CAV_FREEZE.with_suffix(".sha256").read_text().strip()
    if _sha256(paths.CAV_FREEZE) != expected:
        raise ValueError("CAV freeze file does not match its recorded sha256")
    with open(paths.CAV_FREEZE, "rb") as handle:
        units = pickle.load(handle)["payload"]["units"]
    arrays = {}
    for key, result in units.items():
        for estimator in paths.ESTIMATORS:
            fits = result.get("joint", {}).get((estimator, paths.CAV_C), [])
            measured = [f for f in fits if f["status"] == "measured"]
            arrays[f"fits|{estimator}|{key}"] = np.array([len(fits), len(measured)])
            if not measured:
                continue
            matrix = np.full((len(fits), len(measured[0]["direction"])), np.nan, dtype=np.float32)
            for position, fit in enumerate(fits):
                if fit["status"] == "measured":
                    matrix[position] = fit["direction"]
            arrays[f"{estimator}|{key}"] = matrix
            arrays[f"seeds|{estimator}|{key}"] = np.array([fit["seed"] for fit in fits])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, output)
    return output


def load_cav_directions(path: Path = paths.CAV_DIRECTIONS) -> tuple:
    """({(estimator, concept key): (fits x 192) unit directions, NaN rows for failed fits},
    {(estimator, concept key): (listed fits, measured fits)})."""
    directions, counts = {}, {}
    with np.load(path) as stored:
        for name in stored.files:
            kind, rest = name.split("|", 1)
            if kind == "fits":
                estimator, key = rest.split("|", 1)
                counts[(estimator, key)] = tuple(int(v) for v in stored[name])
            elif kind != "seeds":
                directions[(kind, rest)] = stored[name]
    return directions, counts
