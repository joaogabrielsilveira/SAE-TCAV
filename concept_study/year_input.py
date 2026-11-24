"""Year-input experiment: re-predict later patients with the reference year as the model's year input.

DR-TabPFN receives each row's calendar year as an extra input ("dist_shift_domain" =
year minus the first context year). Here every later evaluation patient-year is scored
again with its features unchanged but its year input set to the reference year. Comparing
with the cached (true-year) predictions isolates what the year input alone does to
predictions, embeddings, SAE activations and therefore to every concept's A group.

Run one reference build per process (GPU builds one after the other, CPU builds in parallel):
    .venv/bin/python -m concept_study.year_input <build> <reference> [cpu threads, default 8]

SAE codes are stored as float32 from now on. Files written before 10 Nov 2026 hold float16 codes;
cohort.year_input_table rounds the true-year activation to the stored precision before comparing.
"""
import sys
import time

import numpy as np
import torch

from concept_study import loading, measures, paths


def restore_frozen_context(population: dict, roles: dict, predictor: dict, device: str, split_seed: int) -> dict:
    """Refit the frozen DR-TabPFN context of one reference build and check it is the cached one.

    Ported from pilot_follow_ups.ipynb (cell 18): context rows are the reference-year
    'tabpfn_context' patients; window builds add earlier rows up to each patient's death,
    with year inputs counted from the first context year.
    """
    import concept_subgroup_drift_pilot as pilot
    reference, year = roles["reference_year"], population["year"]
    rows = (year == reference) & (roles["row_role"] == "tabpfn_context") & ~roles["is_entrant"]
    window = predictor.get("context_window", "reference_only")
    first = min(predictor.get("context_years", [reference]))
    if window != "reference_only":
        rows |= (year >= first) & (year < reference) & ~population["post_death"]
    rows = np.flatnonzero(rows)
    columns = pilot._feature_columns(population["X"].shape[1])
    frame = pilot._prediction_frame(population, rows, columns)
    frame["y"] = population["y"][rows]
    extra = {} if window == "reference_only" else {"domain_origin": first}
    system = pilot.fit_frozen_predictor(frame, columns, device, seed=split_seed, **extra)
    same = all(system[key] == predictor[key] for key in ("context_identity", "checkpoint_sha256"))
    if not same or system["context_rows"] != predictor["context_rows_identity"]:
        raise ValueError("Rebuilt predictor context differs from the cached build")
    return system


def predict(system: dict, frame, device: str, batch: int = 64) -> tuple:
    """Death probabilities and embeddings of a frame, in batches (halving the batch on GPU memory errors)."""
    import concept_subgroup_drift_pilot as pilot
    probabilities, embeddings, cursor = [], [], 0
    while cursor < len(frame):
        try:
            captured = pilot.capture_frozen_predictions(system, frame.iloc[cursor:cursor + batch], device)
        except torch.cuda.OutOfMemoryError:
            if batch == 1:
                raise
            batch //= 2
            torch.cuda.empty_cache()
            continue
        probabilities.append(captured["probabilities"][:, 1].detach().cpu().numpy())
        embeddings.append(captured["embeddings"].detach().cpu().numpy())
        cursor += batch
    return np.concatenate(probabilities), np.concatenate(embeddings)


def population_rows(population: dict, patient_ids: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Population row of each (patient id, year) pair."""
    lookup = {(str(p), int(y)): i for i, (p, y) in enumerate(zip(population["patient_id"], population["year"]))}
    return np.array([lookup[(str(p), int(y))] for p, y in zip(patient_ids, years)], dtype=int)


def run_build(build: str, reference: int, threads: int = 8, device: str = paths.DEVICE) -> None:
    """Score one build's later evaluation patient-years with the reference year as year input.

    Predictors frozen on the CPU are replayed on the CPU with `threads` threads; the SAE runs on `device`.
    """
    import concept_subgroup_drift_pilot as pilot
    target = paths.YEAR_INPUT / f"build_{build}_{reference}.npz"
    if target.exists():
        return
    started = time.monotonic()
    inputs = loading.open_inputs(device)
    population, cache = inputs["population"], inputs["caches"][build]
    predictor = cache.load(f"P/{reference}")
    sae_device = device
    if predictor.get("device") == "cpu":
        device = "cpu"
        torch.set_num_threads(threads)
    roles = pilot.reference_year_roles(population, reference, seed=cache.config["split_seed"])
    system = restore_frozen_context(population, roles, predictor, device, int(cache.config["split_seed"]))

    table = measures.load_build({"file": f"build_{build}_{reference}.npz"})
    later = np.flatnonzero(table["year"] > reference)
    rows = population_rows(population, table["patient_id"][later], table["year"][later])
    frame = pilot._prediction_frame(population, rows, pilot._feature_columns(population["X"].shape[1]))

    check = frame.iloc[:32]
    replayed, _ = predict(system, check, device)
    replay_error = float(np.max(np.abs(replayed - table["probability"][later[:32]])))
    if replay_error > 1e-4:
        raise ValueError(f"True-year re-prediction differs from cached probabilities ({replay_error})")

    swapped = frame.copy()
    swapped["year"] = reference
    probability, embeddings = predict(system, swapped, device)
    del system
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dictionaries = pilot.load_reference_dictionaries(cache, reference, sae_device)
    codes = {f"codes_{seed}": pilot.encode_with_reference_sae(d, embeddings, sae_device, 256).astype(np.float32)
             for seed, d in dictionaries.items()}
    paths.YEAR_INPUT.mkdir(parents=True, exist_ok=True)
    np.savez(target, positions=later, patient_id=table["patient_id"][later], year=table["year"][later],
             probability_swapped=probability.astype(np.float32), replay_error=replay_error,
             device=device, seconds=time.monotonic() - started, **codes)
    print(f"{build} {reference}: {len(later)} rows on {device}, replay error {replay_error:.2e}, "
          f"{time.monotonic() - started:.0f} s", flush=True)


def load_year_input(build: str, reference: int) -> dict:
    path = paths.YEAR_INPUT / f"build_{build}_{reference}.npz"
    if not path.exists():
        return None
    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}


if __name__ == "__main__":
    run_build(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 8)
