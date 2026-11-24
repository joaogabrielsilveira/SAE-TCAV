"""Compare 2008 held-code encoder blocks on one frozen CPU predictor capture."""

import argparse
from contextlib import ExitStack
import gc
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

import r1_held_code_diagnostic as helpers


REFERENCE = 2008
PREFIX_ROWS = 64
BLOCK_ROWS = 256
PRECHECK_ROWS = 32
SEEDS = (42, 135, 215)


def _setup():
    import concept_subgroup_drift_pilot as pilot
    import temporal_input_attribution as attribution

    audit = attribution.audit_frozen_inputs(helpers.LEDGER)
    systems = [system for system in audit["systems"] if int(system["reference"]) == REFERENCE]
    if len(systems) != 1:
        raise ValueError("Expected exactly one frozen 2008 system")
    system = systems[0]
    if not system["current_checkpoint_matches"]:
        raise ValueError("Current checkpoint differs from frozen checkpoint")
    file = audit["common"]["files"]["population"]
    population = pilot.load_pilot_population(file["path"], file["sha256"])
    fingerprints = attribution._current_state_input_fingerprints(audit, system)
    return pilot, attribution, audit, system, population, fingerprints


def _dictionaries(pilot, attribution, system, config):
    run = Path(system["files"]["config"]["path"]).parent.parent
    frozen = attribution._FrozenUnits(run / "unit-cache" / config["config_hash"] / "units", config)
    dictionaries = pilot.load_reference_dictionaries(frozen, REFERENCE, "cpu")
    if set(dictionaries) != set(SEEDS):
        raise ValueError("Unexpected reference dictionary seeds")
    return dictionaries


def capture_cpu(destination, report):
    import torch
    import r1_2007_replay as replay

    pilot, attribution, audit, system, population, fingerprints = _setup()
    if pilot.resolve_device("cpu") != "cpu":
        raise RuntimeError("CPU predictor is required")
    report["system"] = f"{system['build']}/{REFERENCE}"
    report["stage"] = "generated_state"
    source_hash = attribution._file_sha256(Path(attribution.__file__))
    state = attribution._generated_state_for_system(audit, system, population, "cpu", source_hash)
    context, columns, _, _, _, _, bindings, config = attribution._current_scoring_context(
        pilot, population, system, state["provenance"]["positional_state"])
    if bindings["context_rows_identity"] != system["context_rows_identity"]:
        raise ValueError("Frozen context row identity changed")
    origin = None if config.get("context_window", "reference_only") == "reference_only" else system["domain_origin"]
    report["stage"] = "predictor_fit"
    predictor = attribution._fit_frozen_predictor_with_state(
        context, columns, system, "cpu", int(config["split_seed"]), state["positional_tensor"],
        domain_origin=origin)
    identity = attribution._comparable_predictor_identity(
        pilot, predictor["classifier"], predictor["context_rows"], report["system"])
    if (predictor["context_rows"] != system["context_rows_identity"]
            or predictor["checkpoint_sha256"] != system["saved_checkpoint_sha256"]
            or not torch.equal(attribution._native_positional_embeddings(predictor).cpu(), state["positional_tensor"].cpu())
            or torch.device(identity["device"]).type != "cpu"
            or any(identity.get(key) != bindings.get(key) for key in
                   ("system_identity", "clinical_input_identity", "context_rows_identity", "checkpoint_sha256", "positional_state"))):
        raise ValueError("Fitted predictor failed frozen identity checks")
    report["fitted_state_verified"] = True

    with ExitStack() as stack:
        actual, held, archived = (stack.enter_context(np.load(system["files"][name]["path"], allow_pickle=False))
                                  for name in ("actual", "held", "predictor_archive"))
        prefix = replay.select_held_batch(held, actual, REFERENCE, PREFIX_ROWS, PRECHECK_ROWS)
        selected = replay.select_held_batch(held, actual, REFERENCE, BLOCK_ROWS, PRECHECK_ROWS)
        probes = replay.select_held_probe_rows(held, prefix, range(REFERENCE + 1, 2015))
        keys = attribution._row_keys(actual, "actual")
        lookup = {(str(patient), int(year)): i for i, (patient, year) in
                  enumerate(zip(population["patient_id"], population["year"]))}

        def frame_for(rows):
            return pilot._prediction_frame(population, [lookup[keys[int(row)]] for row in rows], columns)

        actual_rows = replay.select_query_batch(population, archived["row_index"], REFERENCE,
                                                batch_size=int(system["saved_batch_rows"]))
        actual_frame = pilot._prediction_frame(population, actual_rows, columns)
        precheck_frame = frame_for(selected["precheck_rows"])
        held_frame = frame_for(selected["held_actual_rows"])
        held_frame["year"] = REFERENCE
        saved_precheck = actual["probability"][selected["precheck_rows"]].copy()
        saved_probability = held["probability_swapped"][selected["held_rows"]].copy()
        saved = {seed: held[f"codes_{seed}"][:PREFIX_ROWS].copy() for seed in SEEDS}
    if any(values.ndim != 2 or len(values) != PREFIX_ROWS or values.dtype != np.float16
           for values in saved.values()):
        raise ValueError("Expected all three frozen float16 held-code prefixes")

    report["stage"] = "predictor_capture"
    helpers._capture(pilot, predictor, actual_frame, "cpu")
    precheck_probability, _ = helpers._capture(pilot, predictor, precheck_frame, "cpu")
    probability, embeddings = helpers._capture_held_batches(helpers._capture, pilot, predictor, held_frame, "cpu")
    probe_positions = [item["batch_position"] for item in probes.values()]
    report["probability_controls"] = {
        "actual_precheck": helpers._agreement(precheck_probability, saved_precheck),
        "held_prefix": helpers._agreement(probability[:PREFIX_ROWS], saved_probability[:PREFIX_ROWS]),
        "held_probes": helpers._agreement(probability[probe_positions], saved_probability[probe_positions]),
        "held_256": helpers._agreement(probability, saved_probability),
    }
    if not all(report["probability_controls"][name]["within_tolerance"]
               for name in ("actual_precheck", "held_prefix", "held_probes")):
        return "predictor_probability_control_failed"
    del predictor
    gc.collect()
    report["stage"] = "cpu_sae_encoding"
    dictionaries = _dictionaries(pilot, attribution, system, config)
    codes = {block: helpers._encode_prefix(pilot.encode_with_reference_sae, dictionaries, embeddings, "cpu", block)
             for block in (PREFIX_ROWS, BLOCK_ROWS)}
    report["arms"] = [{"predictor": "cpu", "sae": "cpu", "encoder_rows": block,
                       **helpers._arm_summary(codes[block], saved)} for block in (PREFIX_ROWS, BLOCK_ROWS)]

    private = destination / "private"
    private.mkdir(mode=0o700)
    metadata = {"system": report["system"], "input_fingerprints": fingerprints,
                "source_sha256": source_hash,
                "diagnostic_sha256": attribution._file_sha256(Path(__file__)),
                "helper_sha256": attribution._file_sha256(Path(helpers.__file__))}
    arrays = {"embeddings": embeddings, "metadata": np.array(json.dumps(metadata, sort_keys=True))}
    for seed in SEEDS:
        arrays[f"saved_{seed}"] = saved[seed]
        for block in (PREFIX_ROWS, BLOCK_ROWS):
            arrays[f"cpu_{block}_{seed}"] = codes[block][seed]
    fd = os.open(private / "capture.npz", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    report["coverage"] = {"predictor_held_rows": len(embeddings), "predictor_batch_rows": PREFIX_ROWS,
                          "compared_held_rows": PREFIX_ROWS}
    return "complete"


def encode_cuda(capture_path, report):
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA encoder comparison requires one visible CUDA device")
    pilot, attribution, _, system, _, fingerprints = _setup()
    pilot.configure_predictor_precision("cuda")
    report["system"] = f"{system['build']}/{REFERENCE}"
    with np.load(capture_path, allow_pickle=False) as capture:
        metadata = json.loads(str(capture["metadata"].item()))
        expected = {"system": report["system"], "input_fingerprints": fingerprints,
                    "source_sha256": attribution._file_sha256(Path(attribution.__file__)),
                    "diagnostic_sha256": attribution._file_sha256(Path(__file__)),
                    "helper_sha256": attribution._file_sha256(Path(helpers.__file__))}
        if metadata != expected:
            raise ValueError("Private capture does not match current frozen inputs and source")
        embeddings = capture["embeddings"].copy()
        saved = {seed: capture[f"saved_{seed}"].copy() for seed in SEEDS}
        cpu = {block: {seed: capture[f"cpu_{block}_{seed}"].copy() for seed in SEEDS}
               for block in (PREFIX_ROWS, BLOCK_ROWS)}
    if embeddings.ndim != 2 or len(embeddings) != BLOCK_ROWS or not np.isfinite(embeddings).all():
        raise ValueError("Private capture lacks the complete finite held embeddings")
    config = json.loads(Path(system["files"]["config"]["path"]).read_text())
    dictionaries = _dictionaries(pilot, attribution, system, config)
    report["stage"] = "cuda_sae_encoding"
    report["arms"], report["cpu_contrasts"] = [], []
    for block in (PREFIX_ROWS, BLOCK_ROWS):
        codes = helpers._encode_prefix(pilot.encode_with_reference_sae, dictionaries, embeddings, "cuda", block)
        report["arms"].append({"predictor": "cpu", "sae": "cuda", "encoder_rows": block,
                               **helpers._arm_summary(codes, saved)})
        baseline = {seed: cpu[block][seed].astype(saved[seed].dtype) for seed in SEEDS}
        report["cpu_contrasts"].append({"encoder_rows": block, **helpers._arm_summary(codes, baseline)})
    report["coverage"] = {"predictor_held_rows": len(embeddings), "compared_held_rows": PREFIX_ROWS}
    return "complete"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("capture-cpu", "encode-cuda"))
    parser.add_argument("--private-input", type=Path)
    args = parser.parse_args()
    if (args.phase == "encode-cuda") != (args.private_input is not None):
        parser.error("--private-input is required only for encode-cuda")
    destination = Path(os.environ["RPI_OUT_DIR"])
    report = {"stage": "setup", "phase": args.phase}
    started = time.monotonic()
    with (destination / f"held_code_2008_{args.phase}.public.json").open("x") as stream:
        try:
            report["status"] = (capture_cpu(destination, report) if args.phase == "capture-cpu"
                                else encode_cuda(args.private_input, report))
        except Exception:
            report["status"] = "error"
            raise
        finally:
            import torch
            report["resources"] = {"elapsed_seconds": time.monotonic() - started,
                                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                   "torch_cpu_threads": torch.get_num_threads()}
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
