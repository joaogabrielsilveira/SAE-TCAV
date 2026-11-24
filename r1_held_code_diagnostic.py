"""Compare the first r1/2007 held codes across predictor and SAE settings."""

from contextlib import ExitStack
import argparse
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import time

import numpy as np


REFERENCE = 2007
SEEDS = (42, 135, 215)
PREFIX_ROWS = 64
BLOCK_ROWS = 256
PRECHECK_ROWS = 32
FULL_HELD_ROWS = 2476
RTOL = 1e-4
ATOL = 1e-5
LEDGER = Path("stats/concept_agreement_study/extensions/temporal_input_attribution/checks/coverage-ledger.csv")


def _code_comparison(fresh, saved):
    fresh = fresh.astype(saved.dtype)
    if fresh.shape != saved.shape or not np.isfinite(fresh).all() or not np.isfinite(saved).all():
        raise ValueError("Expected aligned finite saved-precision code matrices")
    count = int(np.count_nonzero(fresh != saved))
    return {"equal": count == 0, "mismatched_entries": count, "entries": saved.size,
            "max_abs_error": float(np.max(np.abs(fresh.astype(np.float32) - saved.astype(np.float32))))}


def _encode_prefix(encode, dictionaries, embeddings, device, block):
    if len(embeddings) < block:
        raise ValueError("Encoder block requires genuine companion rows")
    return {seed: encode(dictionary, embeddings[:block], device, batch_size=block)[:PREFIX_ROWS]
            for seed, dictionary in dictionaries.items()}


def _agreement(fresh, other):
    if fresh.shape != other.shape or not np.isfinite(fresh).all() or not np.isfinite(other).all():
        raise ValueError("Expected aligned finite values for tolerance comparison")
    return {"within_tolerance": bool(np.allclose(fresh, other, rtol=RTOL, atol=ATOL)),
            "max_abs_error": float(np.max(np.abs(fresh - other)))}


def _capture(pilot, predictor, frame, device):
    result = pilot.capture_frozen_predictions(predictor, frame, device)
    if result["row_identity"] != pilot._identity_hash(list(zip(frame.patient_id, frame.year))):
        raise ValueError("Predictor query row order changed")
    return (result["probabilities"].detach().cpu().numpy()[:, 1].copy(),
            result["embeddings"].detach().cpu().numpy().copy())


def _query_schedule(pilot, replay, attribution, system, population, columns, actual, held, archived,
                    full_table=False):
    prefix = replay.select_held_batch(held, actual, REFERENCE, PREFIX_ROWS, PRECHECK_ROWS)
    rows = FULL_HELD_ROWS if full_table else BLOCK_ROWS
    selection = replay.select_held_batch(held, actual, REFERENCE, rows, PRECHECK_ROWS)
    if full_table:
        expected = np.flatnonzero(np.asarray(actual["year"]) > REFERENCE)
        if (len(held["year"]) != FULL_HELD_ROWS or len(expected) != FULL_HELD_ROWS
                or not np.array_equal(selection["held_actual_rows"], expected)):
            raise ValueError("Held archive does not cover all eligible actual rows in order")
    probes = replay.select_held_probe_rows(held, prefix, range(REFERENCE + 1, 2015)).values()
    candidate_positions = [probe["batch_position"] for probe in probes]
    archive_positions = [probe["archive_row"] for probe in probes]
    actual_keys = attribution._row_keys(actual, "actual")
    population_lookup = {(str(patient), int(year)): i for i, (patient, year)
                         in enumerate(zip(population["patient_id"], population["year"]))}

    def frame_for_actual_rows(rows):
        return pilot._prediction_frame(population, [population_lookup[actual_keys[int(row)]] for row in rows],
                                       columns)

    # The preceding original-size actual call is part of the failing CPU replay's schedule.
    actual_batch = replay.select_query_batch(population, archived["row_index"], REFERENCE,
                                             batch_size=int(system["saved_batch_rows"]))
    actual_frame = pilot._prediction_frame(population, actual_batch, columns)
    precheck = frame_for_actual_rows(selection["precheck_rows"])
    held_frame = frame_for_actual_rows(selection["held_actual_rows"])
    held_frame["year"] = REFERENCE
    return actual_frame, precheck, held_frame, (
        actual["probability"][selection["precheck_rows"]], candidate_positions,
        held["probability_swapped"][archive_positions],
        held["probability_swapped"][selection["held_rows"]] if full_table else
        held["probability_swapped"][prefix["held_rows"]])


def _capture_held_batches(capture, pilot, predictor, frame, device):
    blocks = [capture(pilot, predictor, frame.iloc[start:start + PREFIX_ROWS], device)
              for start in range(0, len(frame), PREFIX_ROWS)]
    if not blocks:
        raise ValueError("Held predictor input is empty")
    probabilities, embeddings = zip(*blocks)
    if any(len(p) != len(e) or len(p) != min(PREFIX_ROWS, len(frame) - index * PREFIX_ROWS)
           for index, (p, e) in enumerate(blocks)):
        raise ValueError("Held predictor output coverage is incomplete")
    return np.concatenate(probabilities), np.concatenate(embeddings)


def _encode_full_table(encode, dictionaries, embeddings, device):
    return {seed: encode(dictionary, embeddings, device, batch_size=BLOCK_ROWS)
            for seed, dictionary in dictionaries.items()}


def _full_table_evidence(codes, saved, probabilities, saved_probabilities, expected_rows):
    if (set(codes) != set(SEEDS) or set(saved) != set(SEEDS)
            or any(values.dtype != np.float16 or values.ndim != 2 or len(values) != expected_rows
                   for values in saved.values())
            or any(np.asarray(codes[seed]).shape != saved[seed].shape for seed in SEEDS)
            or np.asarray(probabilities).shape != (expected_rows,)
            or np.asarray(saved_probabilities).shape != (expected_rows,)):
        raise ValueError("Incomplete full held-table archive evidence")
    probability = _agreement(probabilities, saved_probabilities)
    summary = _arm_summary(codes, saved)
    return {"complete": summary["equal"] and probability["within_tolerance"],
            "coverage": {"held_rows": expected_rows, "predictor_batch_rows": PREFIX_ROWS,
                         "encoder_block_rows": BLOCK_ROWS,
                         "final_encoder_block_rows": expected_rows % BLOCK_ROWS or BLOCK_ROWS},
            "probability": probability, "codes": summary}


def _capture_predictor(pilot, attribution, system, fit_args, schedule, device, full_table=False,
                       expected_bindings=None):
    import torch

    context, columns, seed, positional, domain_origin = fit_args
    predictor = attribution._fit_frozen_predictor_with_state(
        context, columns, system, device, seed, positional.to(device), domain_origin=domain_origin)
    identity = attribution._comparable_predictor_identity(
        pilot, predictor["classifier"], predictor["context_rows"], "r1/2007")
    if (predictor["context_rows"] != system["context_rows_identity"]
            or predictor["checkpoint_sha256"] != system["saved_checkpoint_sha256"]
            or torch.device(identity["device"]).type != device
            or not torch.equal(attribution._native_positional_embeddings(predictor).cpu(), positional)):
        raise ValueError("Fitted predictor failed its frozen context, checkpoint or positional checks")
    actual, precheck, held, (saved_precheck, probe_positions, saved_probes, saved_prefix) = schedule
    _capture(pilot, predictor, actual, device)
    precheck_p, _ = _capture(pilot, predictor, precheck, device)
    probabilities, embeddings = _capture_held_batches(_capture, pilot, predictor, held, device)
    predictor_summary = {
        "device": identity["device"],
        "precision": predictor["settings"]["precision"],
        "actual_precheck_probability": _agreement(precheck_p, saved_precheck),
        "held_probe_probability": _agreement(probabilities[probe_positions], saved_probes),
        "held_prefix_probability": _agreement(probabilities[:PREFIX_ROWS], saved_prefix[:PREFIX_ROWS]),
    }
    if full_table:
        execution = identity.get("execution_state", {})
        predictor_summary["fitted_state_verified"] = (
            isinstance(expected_bindings, dict)
            and all(expected_bindings.get(key) and identity.get(key) == expected_bindings[key]
                    for key in ("system_identity", "clinical_input_identity", "context_rows_identity",
                                "checkpoint_sha256", "positional_state"))
            and identity.get("precision") == {"dtype": "fp32", "amp": "off", "tf32": False}
            and execution.get("eval") is True and execution.get("depth") is not None
            and execution.get("dropout") is not None
            and all(identity.get(key) for key in ("opaque_identity", "learned_parameter_identity",
                    "configuration_identity", "clinical_transform_identity", "fitted_fields_identity")))
        return embeddings, identity, predictor_summary, probabilities
    return embeddings, identity, predictor_summary


def _arm_summary(codes, archived):
    seeds = {seed: _code_comparison(codes[seed], archived[seed]) for seed in SEEDS}
    return {"equal": all(item["equal"] for item in seeds.values()),
            "mismatched_entries": sum(item["mismatched_entries"] for item in seeds.values()),
            "entries": sum(item["entries"] for item in seeds.values()), "seeds": seeds}


def diagnose(report, full_table=False):
    import torch
    import concept_subgroup_drift_pilot as pilot
    import r1_2007_replay as replay
    import temporal_input_attribution as attribution

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Diagnostic requires one visible CUDA device")
    torch.cuda.set_device(0)
    report["runtime"] = {
        "packages": {package: version(package) for package in ("numpy", "torch", "scikit-learn")},
        "gpu": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda,
        "torch_cpu_threads": torch.get_num_threads(), "sae_precision": "float32, TF32 disabled",
        "predictor_held_batch_rows": PREFIX_ROWS,
    }

    report["stage"] = "frozen_input_setup"
    audit = attribution.audit_frozen_inputs(LEDGER)
    system, = (item for item in audit["systems"] if (item["build"], item["reference"]) == ("r1", REFERENCE))
    if not system["current_checkpoint_matches"]:
        raise ValueError("Current checkpoint differs from frozen checkpoint")
    population_file = audit["common"]["files"]["population"]
    population = pilot.load_pilot_population(population_file["path"], population_file["sha256"])
    state = attribution._generated_state_for_system(
        audit, system, population, "cpu", attribution._file_sha256(Path(attribution.__file__)))
    positional_state = state["provenance"]["positional_state"] if full_table else None
    context, columns, _, _, _, _, bindings, config = attribution._current_scoring_context(
        pilot, population, system, positional_state)
    origin = None if config.get("context_window", "reference_only") == "reference_only" else system["domain_origin"]
    fit_args = context, columns, int(config["split_seed"]), state["positional_tensor"], origin
    with ExitStack() as stack:
        actual, held, archived = (
            stack.enter_context(np.load(system["files"][name]["path"], allow_pickle=False))
            for name in ("actual", "held", "predictor_archive"))
        schedule = _query_schedule(pilot, replay, attribution, system, population, columns, actual, held, archived,
                                   full_table=full_table)
        saved = {seed: held[f"codes_{seed}"][:FULL_HELD_ROWS if full_table else PREFIX_ROWS].copy()
                 for seed in SEEDS}
    expected_rows = FULL_HELD_ROWS if full_table else PREFIX_ROWS
    if any(values.dtype != np.float16 or values.ndim != 2 or values.shape[0] != expected_rows
           for values in saved.values()):
        raise ValueError("Expected frozen held float16 codes for all three seeds")

    run = Path(system["files"]["config"]["path"]).parent.parent
    frozen = attribution._FrozenUnits(run / "unit-cache" / config["config_hash"] / "units", config)
    if full_table:
        report["stage"] = "cuda_predictor"
        embeddings, identity, summary, probabilities = _capture_predictor(
            pilot, attribution, system, fit_args, schedule, "cuda", full_table=True,
            expected_bindings=bindings)
        report["predictors"]["cuda"] = summary
        if not summary["fitted_state_verified"]:
            return "predictor_identity_mismatch"
        if not all(summary[key]["within_tolerance"] for key in
                   ("actual_precheck_probability", "held_probe_probability", "held_prefix_probability")):
            return "predictor_probability_control_failed"
        report["stage"] = "cuda_sae_encoding"
        gc.collect()
        torch.cuda.empty_cache()
        dictionaries = pilot.load_reference_dictionaries(frozen, REFERENCE, "cpu")
        if set(dictionaries) != set(SEEDS):
            raise ValueError("Unexpected reference dictionary seeds")
        codes = _encode_full_table(pilot.encode_with_reference_sae, dictionaries, embeddings, "cuda")
        report["full_table"] = _full_table_evidence(codes, saved, probabilities, schedule[3][3], expected_rows)
        return "complete" if report["full_table"]["complete"] else "archive_comparison_failed"
    captures, identities, encoded = {}, {}, {}
    for predictor_device in ("cpu", "cuda"):
        report["stage"] = f"{predictor_device}_predictor"
        embeddings, identity, summary = _capture_predictor(
            pilot, attribution, system, fit_args, schedule, predictor_device)
        captures[predictor_device] = embeddings
        identities[predictor_device] = identity
        report["predictors"][predictor_device] = summary
        gc.collect()
        if predictor_device == "cuda":
            torch.cuda.empty_cache()
            report["predictor_fit_identity_matches"] = attribution.comparable_predictor_identities_match(
                identities["cpu"], identities["cuda"])
            report["fresh_cpu_cuda_prefix_embeddings"] = _agreement(
                captures["cpu"][:PREFIX_ROWS], embeddings[:PREFIX_ROWS])
            if not report["predictor_fit_identity_matches"]:
                return "predictor_identity_mismatch"
        if not all(summary[key]["within_tolerance"] for key in
                   ("actual_precheck_probability", "held_probe_probability")):
            return "predictor_probability_control_failed"

        report["stage"] = f"{predictor_device}_predictor_sae_encoding"
        # Load once on CPU and use the frozen encoder's device argument for both execution paths.
        dictionaries = pilot.load_reference_dictionaries(frozen, REFERENCE, "cpu")
        if set(dictionaries) != set(SEEDS):
            raise ValueError("Unexpected reference dictionary seeds")
        for sae_device in ("cpu", "cuda"):
            for block in (PREFIX_ROWS, BLOCK_ROWS):
                codes = _encode_prefix(pilot.encode_with_reference_sae, dictionaries, embeddings, sae_device, block)
                key = predictor_device, sae_device, block
                encoded[key] = codes
                summary = _arm_summary(codes, saved)
                report["arms"].append({"predictor": predictor_device, "sae": sae_device,
                                       "encoder_rows": block, **summary})
                if key == ("cpu", "cpu", PREFIX_ROWS):
                    report["baseline_mismatch_reproduced"] = not summary["equal"]
                    if summary["equal"]:
                        return "baseline_matched"

    for cpu_key, left in encoded.items():
        for axis, changed in enumerate(("cuda", "cuda", BLOCK_ROWS)):
            if cpu_key[axis] == changed:
                continue
            right_key = cpu_key[:axis] + (changed,) + cpu_key[axis + 1:]
            right = encoded[right_key]
            baseline = {seed: left[seed].astype(saved[seed].dtype) for seed in SEEDS}
            report["contrasts"].append({"from": cpu_key, "to": right_key, **_arm_summary(right, baseline)})
    return "complete"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-table", action="store_true")
    args = parser.parse_args()
    name = "held_code_full_table.public.json" if args.full_table else "held_code_diagnostic.public.json"
    destination = Path(os.environ["RPI_OUT_DIR"]) / name
    report = {"system": "r1/2007", "stage": "setup", "predictors": {}, "arms": [], "contrasts": []}
    # Reserve the filename exclusively so prior outputs cannot be overwritten.
    with destination.open("x") as stream:
        started = time.monotonic()
        try:
            report["status"] = diagnose(report, full_table=args.full_table)
        except Exception:
            report["status"] = "error"
            raise
        finally:
            if args.full_table:
                report["resources"] = {"elapsed_seconds": time.monotonic() - started,
                                       "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                import torch
                if torch.cuda.is_initialized():
                    report["resources"]["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    report["resources"]["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
