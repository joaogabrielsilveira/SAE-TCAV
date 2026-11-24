"""Input identity and batch preparation for the isolated r1 replay."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


RECOVERED_SOURCE_SHA256 = "fe94560ebbfd3413f63eca78500827d85b47e9667664dd927a31b120bf87ac63"
CURRENT_SOURCE_SHA256 = "a44dbc87b66927c7517d8509469cfc948ca5a2d3b3a50d55b5a50915a2da6a12"
POPULATION_SHA256 = "c8804591c93087241b9533dce77dd54b2f1e77b23a36400527459cfa8f99add7"
CHECKPOINT_SHA256 = "e653d21bd9b2713cf857696f98aabb2a59a4009212d1a63744feccbca94462db"
# Canonical full-model parameter identity in analysis_outputs/temporal_input_attribution/
# checks_cpu_primary/timing.json. That audit hashes initialized-model.named_parameters()
# before fit and compares cached_models_[0].named_parameters() after fit for this pinned
# source/checkpoint and the same `_identity_hash` normalization.
CANONICAL_MODEL_PARAMETER_SHA256 = "2139fa19c7eb9f305c49533a6cc03703359bebbd64625dd8eacdc36a7399afba"
PROVENANCE_SHA256 = "3a38b52adf8a2b28f80c42b9c8c1eeece93d98273576edce36f45a95c3c1c231"
TABPFN_SOURCE_IDENTITY = {
    "method": "tabpfn-python-tree-sha256-v1",
    "sha256": "6a6519e2014ba5f6832278191f774e34607945e4d7c16307c3853744979089de",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_provenance_hashes(paths, expected_files):
    """Require each consumed file to match an available provenance SHA-256."""
    observed = {}
    for name, path in paths.items():
        expected = expected_files.get(name)
        if (not isinstance(expected, dict) or expected.get("status") != "available"
                or not isinstance(expected.get("sha256"), str)
                or len(expected["sha256"]) != 64):
            raise ValueError(f"provenance identity is unavailable for consumed file {name}")
        digest = file_sha256(path)
        if digest != expected["sha256"]:
            raise ValueError(f"provenance SHA-256 mismatch for consumed file {name}")
        observed[name] = digest
    return observed


def verify_dependency_identity(current, expected):
    """Require the audited current dependency source identity to remain unchanged."""
    if current != expected or expected != TABPFN_SOURCE_IDENTITY:
        raise ValueError("current TabPFN source identity differs from audited provenance")


def load_isolated_source(path, expected_sha256, module_name="_r1_replay_recovered_pilot"):
    """Load a hash-verified source snapshot without replacing the normal pilot import."""
    path = Path(path)
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"source SHA-256 mismatch: expected {expected_sha256}, got {observed}")
    if module_name == "concept_subgroup_drift_pilot":
        raise ValueError("recovered source must use an isolated module name")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source snapshot: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def _identity_keys(table, name):
    patients = np.asarray(table["patient_id"])
    years = np.asarray(table["year"])
    if (patients.ndim != 1 or years.ndim != 1 or len(patients) != len(years)
            or not np.issubdtype(years.dtype, np.integer)):
        raise ValueError(f"invalid {name} patient/year columns")
    keys = [(str(patient), int(year)) for patient, year in zip(patients, years)]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate patient/year identity in {name}")
    return keys


def select_query_batch(population, archived_rows, reference_year, batch_size=4096):
    """Validate the archived query order and return its first complete batch."""
    keys = _identity_keys(population, "population")
    years = np.asarray(population["year"])
    rows = np.asarray(archived_rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("archive row identity must be a one-dimensional integer array")
    expected = np.flatnonzero(years >= reference_year)
    if not np.array_equal(rows, expected):
        raise ValueError("archive row identity or query order mismatch")
    if batch_size < 1 or len(rows) < batch_size:
        raise ValueError(f"query archive has fewer than {batch_size} rows")
    # Key validation above ensures row identity is unique before positions are reused.
    return rows[:batch_size].copy()


def select_probe_rows(population, batch_rows, actual, reference_year,
                      probe_years=range(2007, 2015)):
    """Map first saved actual row/year probes into the original query batch."""
    keys = _identity_keys(population, "population")
    actual_keys = _identity_keys(actual, "actual archive")
    rows = np.asarray(batch_rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("batch rows must be a one-dimensional integer array")
    if np.any(rows < 0) or np.any(rows >= len(keys)) or len(set(rows.tolist())) != len(rows):
        raise ValueError("invalid batch row identity")
    population_lookup = {key: row for row, key in enumerate(keys)}
    batch_lookup = {int(row): position for position, row in enumerate(rows)}
    selected = {}
    for year in probe_years:
        if year < reference_year:
            raise ValueError(f"probe year {year} precedes reference year {reference_year}")
        matches = [position for position, key in enumerate(actual_keys) if key[1] == int(year)]
        if not matches:
            raise ValueError(f"probe year {year} is absent from the actual archive")
        actual_row = matches[0]
        identity = actual_keys[actual_row]
        population_row = population_lookup.get(identity)
        if population_row is None:
            raise ValueError(f"probe patient/year identity mismatch for {year}")
        if population_row not in batch_lookup:
            raise ValueError(f"probe patient/year identity is absent from query batch for {year}")
        selected[int(year)] = {"identity": identity, "actual_table_row": actual_row,
                               "population_row": population_row,
                               "batch_position": batch_lookup[population_row]}
    return selected


def select_held_batch(held, actual, reference_year, batch_size=64, precheck_size=32):
    """Select held rows in saved order and the producer's leading actual precheck rows."""
    held_keys = _identity_keys(held, "held archive")
    actual_keys = _identity_keys(actual, "actual archive")
    held_years = np.asarray(held["year"])
    actual_years = np.asarray(actual["year"])
    if batch_size < 1 or len(held_keys) < batch_size:
        raise ValueError(f"held archive has fewer than {batch_size} rows")
    eligible = np.flatnonzero(actual_years > reference_year)
    if precheck_size < 1 or len(eligible) < precheck_size:
        raise ValueError(f"actual archive has fewer than {precheck_size} precheck rows")
    held_rows = np.arange(batch_size, dtype=np.int64)
    precheck_rows = eligible[:precheck_size]
    actual_lookup = {key: index for index, key in enumerate(actual_keys)}
    mapped_actual_rows = np.asarray([actual_lookup.get(key, -1) for key in held_keys], dtype=np.int64)
    if np.any(mapped_actual_rows < 0):
        raise ValueError("held patient/year identity is absent from actual archive")
    if "positions" in held:
        positions = np.asarray(held["positions"])
        if (positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer)
                or len(positions) != len(held_keys) or np.any(positions < 0)
                or np.any(positions >= len(actual_keys))
                or not np.array_equal(positions, mapped_actual_rows)):
            raise ValueError("held patient/year identity mismatch with actual archive")
    if any(held_years[row] <= reference_year for row in held_rows):
        raise ValueError("held candidate batch must contain years after the reference")
    return {"held_rows": held_rows, "held_identities": held_keys[:batch_size],
            "held_actual_rows": mapped_actual_rows[:batch_size].copy(),
            "precheck_rows": precheck_rows.copy(),
            "precheck_identities": [actual_keys[int(row)] for row in precheck_rows]}


def select_held_probe_rows(held, selection, probe_years=range(2008, 2015)):
    """Map first saved held row per year to the candidate batch by patient/year identity."""
    keys = _identity_keys(held, "held archive")
    batch_rows = np.asarray(selection["held_rows"], dtype=np.int64)
    batch_keys = [keys[int(row)] for row in batch_rows]
    selected = {}
    for year in probe_years:
        matches = [position for position, key in enumerate(keys) if key[1] == int(year)]
        if not matches:
            raise ValueError(f"held probe year {year} is absent from the candidate archive")
        archive_row = matches[0]
        identity = keys[archive_row]
        try:
            batch_position = batch_keys.index(identity)
        except ValueError as error:
            raise ValueError(f"held probe identity is absent from the candidate batch for {year}") from error
        selected[int(year)] = {"identity": identity, "archive_row": archive_row,
                               "batch_position": batch_position}
    return selected


def prepare_r1_input_metadata(pilot, population, roles, actual, held, archived_rows,
                              saved_meta, config, reference_year=2007, batch_size=4096,
                              held_batch_size=64, precheck_size=32,
                              probe_years=range(2007, 2015)):
    """Validate frozen identities and select the original actual/held batches without inference."""
    from temporal_input_attribution import audit_paired_rows

    if roles.get("reference_year") != reference_year:
        raise ValueError("roles belong to a different reference year")
    population_identity = pilot._identity_hash({
        "X": population["X"], "y": population["y"], "year": population["year"],
        "patient_id": population["patient_id"], "record_keys": population["record_keys"],
        "feature_names": list(population["feature_names"]),
    })
    if population_identity != saved_meta.get("population_identity"):
        raise ValueError("population identity mismatch")
    if roles.get("identity") != saved_meta.get("roles_identity"):
        raise ValueError("role identity mismatch")

    years = np.asarray(population["year"])
    window = config.get("context_window", "reference_only")
    if window != "reference_only":
        raise ValueError(f"replay supports only saved reference_only context, got {window!r}")
    context_rows_mask = ((years == reference_year)
                         & (roles["row_role"] == "tabpfn_context")
                         & ~roles["is_entrant"])
    context_rows = np.flatnonzero(context_rows_mask)
    columns = pilot._feature_columns(population["X"].shape[1])
    context = pilot._prediction_frame(population, context_rows, columns)
    context_identity = pilot._identity_hash(list(zip(context.patient_id, context.year)))
    if context_identity != saved_meta.get("context_rows_identity"):
        raise ValueError("context row identity mismatch")
    if len(context_rows) != saved_meta.get("context_rows"):
        raise ValueError("context row count mismatch")
    if population["X"].shape[1] != saved_meta.get("feature_count"):
        raise ValueError("clinical feature count mismatch")
    settings = saved_meta.get("settings", {})
    if settings.get("domain_policy") != "calendar_year_minus_reference_year":
        raise ValueError("saved predictor domain policy mismatch")

    clinical = {"patient_id": population["patient_id"], "year": years,
                "label": population["y"], "features": population["X"],
                "feature_names": np.asarray(population["feature_names"])}
    clinical_report = audit_paired_rows(actual, held, clinical)
    query_rows = select_query_batch(population, archived_rows, reference_year, batch_size)
    probes = select_probe_rows(population, query_rows, actual, reference_year, probe_years)
    if set(probes) != set(probe_years):
        raise ValueError("fixed actual probes do not cover the requested years")
    held_selection = select_held_batch(held, actual, reference_year,
                                      held_batch_size, precheck_size)

    expected_row_identity = pilot._identity_hash({"row_index": np.asarray(archived_rows)})
    if expected_row_identity != saved_meta.get("output_row_identity"):
        raise ValueError("predictor output row identity mismatch")
    if (saved_meta.get("batch_rows") != batch_size
            or saved_meta.get("start_batch_rows") != batch_size
            or saved_meta.get("oom_backoffs") != []):
        raise ValueError("saved predictor batch schedule differs from the original schedule")
    if saved_meta.get("_rows") != len(archived_rows):
        raise ValueError("predictor archive row count mismatch")
    expected_chunks = (len(archived_rows) + batch_size - 1) // batch_size
    if saved_meta.get("inference_chunks") != expected_chunks:
        raise ValueError("predictor chunk count mismatch")

    return {
        "population_identity": population_identity,
        "roles_identity": roles["identity"],
        "context_identity": context_identity,
        "context_rows": context_rows,
        "feature_columns": columns,
        "clinical_report": clinical_report,
        "query_rows": query_rows,
        "probes": probes,
        "actual_probe_domains": {year: year - reference_year for year in probes},
        "held_selection": held_selection,
        "held_probe_domain": 0,
        "archive_rows": np.asarray(archived_rows).copy(),
        "query_batch_year_counts": {
            str(int(year)): int(np.sum(years[query_rows] == year))
            for year in np.unique(years[query_rows])
        },
        "held_batch_year_counts": {
            str(int(year)): int(np.sum(np.asarray(held["year"])[held_selection["held_rows"]] == year))
            for year in np.unique(np.asarray(held["year"])[held_selection["held_rows"]])
        },
        "actual_precheck_year_counts": {
            str(int(year)): int(np.sum(np.asarray(actual["year"])[held_selection["precheck_rows"]] == year))
            for year in np.unique(np.asarray(actual["year"])[held_selection["precheck_rows"]])
        },
    }


def run_preflight(output, *, _return_inputs=False):
    """Write a metadata-only input preparation record; never builds or calls a predictor."""
    import importlib.resources
    import os
    import resource
    import shutil
    import tabpfn

    from temporal_input_attribution import inventory_inputs, tabpfn_source_identity
    import concept_subgroup_drift_pilot as current_pilot

    root = Path.cwd()
    study = root / "stats/concept_agreement_study"
    history = Path.home() / "rpi-runs/SAE-TCAV/r1-pilot-fe945-20261009"
    provenance_path = study / "extensions/temporal_input_attribution/checks_cpu_primary/provenance.json"
    if file_sha256(provenance_path) != PROVENANCE_SHA256:
        raise ValueError("authoritative replay provenance file identity mismatch")
    provenance = json.loads(provenance_path.read_text())
    prior_systems = [item for item in provenance["systems"]
                     if item.get("build") == "r1" and int(item.get("reference")) == 2007]
    if len(prior_systems) != 1:
        raise ValueError("authoritative provenance lacks one r1/2007 input record")
    common_expected = provenance["common"]["files"]
    system_expected = prior_systems[0]["files"]

    config_path = history / "out/resolved-config.json"
    manifest_path = history / "out/run-manifest.json"
    index_path = study / "patient_table/index.json"
    verify_provenance_hashes({"config": config_path, "index": index_path},
                             {"config": system_expected["config"],
                              "index": common_expected["index"]})
    config = json.loads(config_path.read_text())
    run_unit = history / "unit-cache" / config["config_hash"] / "units/P/2007"
    meta_path, archive_path = run_unit / "meta.json", run_unit / "arrays.npz"
    saved_meta = json.loads(meta_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    recovered_path = study / "extensions/r1_2007_replay/recovered/concept_subgroup_drift_pilot.py"
    recovered = load_isolated_source(recovered_path, RECOVERED_SOURCE_SHA256)

    index = json.loads(index_path.read_text())
    records = [record for record in index["builds"]
               if record["build"] == "r1" and int(record["reference"]) == 2007]
    if len(records) != 1:
        raise ValueError("expected one saved r1/2007 archive record")
    record = records[0]
    actual_path = study / "patient_table" / record["file"]
    held_path = study / "year_input" / record["file"]
    population_path = root / "stats/temporal_robustness/_population/prepared.pkl"
    population = recovered.load_pilot_population(population_path, POPULATION_SHA256)
    roles = recovered.reference_year_roles(population, 2007, seed=int(config["split_seed"]))
    with np.load(actual_path, allow_pickle=False) as arrays:
        actual = {name: arrays[name] for name in arrays.files}
    with np.load(held_path, allow_pickle=False) as arrays:
        held = {name: arrays[name] for name in arrays.files}
    with np.load(archive_path, allow_pickle=False) as arrays:
        archived_rows = arrays["row_index"].copy()
        saved_probabilities = arrays["probabilities"].copy()
        saved_embeddings = arrays["embeddings"].copy()
        replay = prepare_r1_input_metadata(recovered, population, roles, actual, held,
                                           archived_rows, saved_meta, config)
        if not np.array_equal(archived_rows, np.flatnonzero(population["year"] >= 2007)):
            raise ValueError("predictor archive row order mismatch")
        lookup = {int(row): index for index, row in enumerate(archived_rows)}
        actual_keys = _identity_keys(actual, "actual archive")
        population_lookup = {key: row for row, key in enumerate(
            _identity_keys({"patient_id": population["patient_id"], "year": population["year"]}, "population"))}
        actual_archive_positions = np.asarray([
            lookup[population_lookup[key]] for key in actual_keys
        ], dtype=np.int64)
        if not np.array_equal(arrays["probabilities"][actual_archive_positions, 1], actual["probability"]):
            raise ValueError("saved mortality probabilities mismatch")

    package_root = Path(str(importlib.resources.files(tabpfn)))
    checkpoint_path = package_root / "model_cache/tabpfn_dist_model_1.cpkt"
    checkpoint_sha = file_sha256(checkpoint_path)
    if checkpoint_sha != CHECKPOINT_SHA256 or checkpoint_sha != saved_meta["checkpoint_sha256"]:
        raise ValueError("checkpoint identity mismatch")
    current_source_sha = file_sha256(root / "concept_subgroup_drift_pilot.py")
    if current_source_sha != CURRENT_SOURCE_SHA256:
        raise ValueError("current pilot source identity mismatch")
    dependency = tabpfn_source_identity(package_root)
    audited_dependency = provenance["common"]["provenance"]["current_tabpfn_source_identity"]
    verify_dependency_identity(dependency, audited_dependency)
    paths = {
        "index": index_path, "population": population_path, "checkpoint": checkpoint_path,
        "pilot_source": root / "concept_subgroup_drift_pilot.py",
        "predictor_wrapper": root / "tabpfn_model.py", "config": config_path,
        "manifest": manifest_path, "predictor_meta": meta_path, "predictor_archive": archive_path,
        "actual": actual_path, "held": held_path,
    }
    expected_files = {**common_expected, **system_expected}
    verified_hashes = verify_provenance_hashes(paths, expected_files)
    verified_hashes["recovered_pilot_source"] = file_sha256(recovered_path)
    inventory = inventory_inputs(paths, {})["files"]
    address_limit = resource.getrlimit(resource.RLIMIT_AS)
    report = {
        "status": "prepared",
        "reference_year": 2007,
        "source_sha256": {"recovered": RECOVERED_SOURCE_SHA256, "current": current_source_sha},
        "checkpoint_sha256": checkpoint_sha,
        "current_dependency_source_identity": dependency,
        "audited_dependency_source_identity": audited_dependency,
        "historical_dependency_source_identity": (manifest.get("dependency_source_identities") or {}).get("tabpfn", "unknown"),
        "authoritative_provenance_sha256": PROVENANCE_SHA256,
        "input_files": {name: {"sha256": verified_hashes[name],
                               "size_bytes": inventory[name]["size_bytes"]}
                        for name in paths},
        "resource_state": {
            "cpu_count": os.cpu_count(),
            "free_disk_bytes": shutil.disk_usage(root).free,
            "address_space_limit_bytes": list(address_limit),
            "tools": {"timeout": shutil.which("timeout"),
                      "time": "/usr/bin/time" if Path("/usr/bin/time").is_file() else None},
        },
        "population_identity": replay["population_identity"],
        "roles_identity": replay["roles_identity"],
        "context_rows_identity": replay["context_identity"],
        "saved_context_identity": saved_meta.get("context_identity"),
        "context_rows": len(replay["context_rows"]),
        "clinical_archive_validation": replay["clinical_report"],
        "predictor_rows": len(replay["archive_rows"]),
        "predictor_batch_rows": saved_meta["batch_rows"],
        "predictor_chunks": saved_meta["inference_chunks"],
        "oom_backoffs": saved_meta["oom_backoffs"],
        "query_batch_year_counts": replay["query_batch_year_counts"],
        "held_batch_year_counts": replay["held_batch_year_counts"],
        "actual_precheck_year_counts": replay["actual_precheck_year_counts"],
        "actual_probe_years": sorted(replay["probes"]),
        "actual_probe_domains": replay["actual_probe_domains"],
        "held_candidate_rows": len(replay["held_selection"]["held_rows"]),
        "held_precheck_rows": len(replay["held_selection"]["precheck_rows"]),
        "held_probe_domain": replay["held_probe_domain"],
        "historical_held_writer_identity": "unknown; candidate held schedule only",
        "historical_held_raw_embeddings": "unavailable",
    }
    actual_keys = _identity_keys(actual, "actual archive")
    actual_lookup = {key: index for index, key in enumerate(actual_keys)}
    actual_probe_targets = {}
    for year, probe in replay["probes"].items():
        actual_row = probe["actual_table_row"]
        archive_row = lookup[probe["population_row"]]
        actual_probe_targets[year] = {
            "probability": float(actual["probability"][actual_row]),
            "embedding": saved_embeddings[archive_row].copy(),
        }
    held_probes = select_held_probe_rows(held, replay["held_selection"])
    held_probe_targets = {year: float(held["probability_swapped"][probe["archive_row"]])
                          for year, probe in held_probes.items()}
    identity_facts = {
        "input_files": verified_hashes,
        "source_hashes": {"recovered": RECOVERED_SOURCE_SHA256, "current": current_source_sha},
        "dependency_identity": dependency,
        "configuration": {"config_hash": config.get("config_hash"), "split_seed": config["split_seed"]},
        "schedule": {"actual_batch_rows": 4096, "held_batch_rows": 64, "precheck_rows": 32,
                     "actual_probe_years": sorted(replay["probes"]),
                     "held_probe_years": sorted(held_probes)},
    }
    if _return_inputs:
        return {
            "root": root, "config": config, "manifest": manifest, "saved_meta": saved_meta,
            "population": population, "roles": roles, "actual": actual, "held": held,
            "archived_rows": archived_rows, "saved_probabilities": saved_probabilities,
            "saved_embeddings": saved_embeddings, "prepared": replay,
            "held_probes": held_probes, "actual_probe_targets": actual_probe_targets,
            "held_probe_targets": held_probe_targets,
            "current_pilot": current_pilot, "recovered_pilot": recovered,
            "source_hashes": identity_facts["source_hashes"],
            "dependency_identity": dependency, "fingerprint_facts": identity_facts,
            "runtime": {"device": "cpu", "threads": 1, "precision": "fp32", "seed": 42},
            "gate_rows": np.flatnonzero(population["year"] == 2007)[:recovered.PREDICTOR_GATE_ROWS],
            "query_batch_size": 4096, "held_batch_size": 64, "precheck_size": 32,
            "targets": {"actual_probabilities": np.asarray(
                            [item["probability"] for _, item in sorted(actual_probe_targets.items())]),
                        "actual_embeddings": np.stack(
                            [item["embedding"] for _, item in sorted(actual_probe_targets.items())]),
                        "held_probabilities": np.asarray(
                            [value for _, value in sorted(held_probe_targets.items())])},
        }
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preflight output: {output}")
    output.mkdir(parents=True)
    (output / "preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _load_r1_inputs():
    """Reuse the metadata-only preflight loader for execution arms."""
    return run_preflight(None, _return_inputs=True)


def _require_replay_runtime(inputs):
    import os
    import resource
    import shutil
    import torch

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required before predictor fitting")
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_AS)
    if soft_limit != 60_000_000 * 1024:
        raise RuntimeError("virtual-memory limit must be 60000000 KiB before predictor fitting")
    if not Path("/usr/bin/time").is_file() or not shutil.which("timeout"):
        raise RuntimeError("/usr/bin/time and timeout are required for replay execution")
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError("PyTorch thread count is not one")
    for source in ("recovered", "current"):
        inputs[f"{source}_pilot"].configure_predictor_precision("cpu")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("precision helper failed to disable TF32")
    return {"device": "cpu", "threads": 1, "precision": "fp32", "seed": 42,
            "memory_limit_kib": 60_000_000,
            "thread_environment": {name: os.environ[name] for name in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "NUMEXPR_NUM_THREADS")},
            "python": sys.version, "numpy": np.__version__, "torch": torch.__version__}


def _seed_replay_runtime(seed):
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _execution_fingerprint(inputs, runtime):
    facts = {"module_sha256": file_sha256(__file__),
             "inputs": inputs["fingerprint_facts"],
             "sources": inputs["source_hashes"],
             "dependency": inputs["dependency_identity"],
             "runtime": runtime,
             "schedule": {"query_batch_rows": inputs["query_batch_size"],
                          "held_batch_rows": inputs["held_batch_size"],
                          "precheck_rows": inputs["precheck_size"],
                          "seed": 42, "device": "cpu", "precision": "fp32"}}
    encoded = json.dumps(facts, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_write(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _checked_capture(capture, expected_rows, stage):
    probabilities = _numpy(capture.get("probabilities"))
    embeddings = _numpy(capture.get("embeddings"))
    if probabilities.shape != (expected_rows, 2) or probabilities.dtype != np.float32:
        raise RuntimeError(f"{stage} probabilities must be float32 with shape (rows, 2)")
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows or embeddings.dtype != np.float32:
        raise RuntimeError(f"{stage} embeddings must be a float32 row matrix")
    if not np.isfinite(probabilities).all() or not np.isfinite(embeddings).all():
        raise RuntimeError(f"{stage} returned nonfinite predictor outputs")
    return probabilities, embeddings


def _frame_for_rows(pilot, inputs, rows, reference_year=None):
    population = inputs["population"]
    columns = pilot._feature_columns(population["X"].shape[1])
    frame = pilot._prediction_frame(population, np.asarray(rows, dtype=np.int64), columns)
    if reference_year is not None:
        frame = frame.copy()
        frame["year"] = reference_year
    return frame


def _population_rows_for_actual(inputs, actual_rows):
    actual_keys = _identity_keys(inputs["actual"], "actual archive")
    population_keys = _identity_keys(
        {"patient_id": inputs["population"]["patient_id"],
         "year": inputs["population"]["year"]}, "population")
    population_lookup = {key: row for row, key in enumerate(population_keys)}
    return np.asarray([population_lookup[actual_keys[int(row)]] for row in actual_rows], dtype=np.int64)


def _state_report(pilot, system):
    report = pilot.predictor_identity_report(system["classifier"])
    fields = report.get("field_digests")
    if not isinstance(fields, dict):
        raise RuntimeError("fitted predictor identity report lacks field digests")
    classifier = system["classifier"]
    model = getattr(classifier, "model", None)
    cached_models = getattr(classifier, "cached_models_", [])
    if not isinstance(cached_models, (list, tuple)) or len(cached_models) != 1:
        raise RuntimeError("fitted predictor requires exactly one cached learned model")
    cached = cached_models[0]
    if not callable(getattr(cached, "named_parameters", None)):
        raise RuntimeError("first cached predictor model lacks named parameters")
    cached_parameters = {name: value.detach().cpu() if hasattr(value, "detach") else value
                         for name, value in cached.named_parameters()}
    if not cached_parameters:
        raise RuntimeError("first cached predictor model has no named parameters")
    cached_identity = pilot._identity_hash(cached_parameters)
    if model is not None:
        if not callable(getattr(model, "named_parameters", None)):
            raise RuntimeError("fitted predictor model lacks named parameters")
        model_parameters = {name: value.detach().cpu() if hasattr(value, "detach") else value
                            for name, value in model.named_parameters()}
        model_identity = pilot._identity_hash(model_parameters)
        if model_identity != cached_identity:
            raise RuntimeError("learned model parameters differ from cached model parameters")
    else:
        if (type(cached).__module__ != "tabpfn.model.transformer"
                or type(cached).__name__ != "PerFeatureTransformer"):
            raise RuntimeError("unsupported fitted predictor cached model container")
        if getattr(classifier, "pilot_checkpoint_sha256_", None) != CHECKPOINT_SHA256:
            raise RuntimeError("cached predictor checkpoint differs from canonical model identity")
        if cached_identity != CANONICAL_MODEL_PARAMETER_SHA256:
            raise RuntimeError("cached predictor full parameters differ from canonical checkpoint identity")
        model_identity = CANONICAL_MODEL_PARAMETER_SHA256
        model_parameters = cached_parameters
    return {"identity": report["identity"], "field_digests": fields,
            "learned_parameter_identity": model_identity,
            "learned_parameter_count": len(model_parameters),
            "cached_parameter_identity": cached_identity}


def _capture_condition(inputs, source, mode):
    """Fit one fresh CPU model, execute original gates, then capture one batch condition."""
    from time import monotonic

    condition_started = monotonic()
    pilot = inputs[f"{source}_pilot"]
    population = inputs["population"]
    prepared = inputs["prepared"]
    columns = pilot._feature_columns(population["X"].shape[1])
    context_rows = prepared["context_rows"]
    context = _frame_for_rows(pilot, inputs, context_rows)
    context["y"] = population["y"][context_rows]
    context_features = np.ascontiguousarray(context[columns].to_numpy(dtype=np.float32))
    context_labels = np.ascontiguousarray(context["y"].to_numpy(dtype=np.float32))
    clinical_identity = hashlib.sha256(
        str(context_features.shape).encode() + context_features.tobytes()
        + str(context_labels.shape).encode() + context_labels.tobytes()).hexdigest()

    _seed_replay_runtime(42)
    precision = pilot.configure_predictor_precision("cpu")
    import torch
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("source precision helper did not disable TF32")
    system = pilot.fit_frozen_predictor(context, columns, "cpu", seed=42)
    if system.get("reference_year") != 2007 or system.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise RuntimeError("fitted source has unexpected reference or checkpoint identity")
    if system.get("settings", {}).get("seed") != 42:
        raise RuntimeError("fitted predictor did not retain seed 42")
    if system.get("settings", {}).get("domain_policy") != "calendar_year_minus_reference_year":
        raise RuntimeError("fitted predictor has an unexpected domain policy")
    if system.get("settings", {}).get("precision", precision).get("amp", "off") != "off":
        raise RuntimeError("fitted predictor did not retain AMP-off precision")
    if system.get("context_rows") != prepared["context_identity"]:
        raise RuntimeError("fitted context row identity differs from prepared inputs")
    state_before_capture = _state_report(pilot, system)

    capture = pilot.capture_frozen_predictions
    if mode != "held_candidate":
        gate_rows = np.asarray(inputs["gate_rows"], dtype=np.int64)
        gate = _frame_for_rows(pilot, inputs, gate_rows)
        base = capture(system, gate, "cpu")
        repeated = capture(system, gate, "cpu")
        reversed_batch = capture(system, gate.iloc[::-1], "cpu")
        singles = [capture(system, gate.iloc[index:index + 1], "cpu") for index in range(len(gate))]
        for key in ("probabilities", "embeddings"):
            base_pair = _checked_capture(base, len(gate), "reference gate")
            base_values = base_pair[0 if key == "probabilities" else 1]
            repeated_values = _checked_capture(repeated, len(gate), "repeat gate")[0 if key == "probabilities" else 1]
            reversed_values = _checked_capture(reversed_batch, len(gate), "order gate")[0 if key == "probabilities" else 1][::-1]
            singleton_values = np.concatenate([
                _checked_capture(single, 1, "singleton gate")[0 if key == "probabilities" else 1]
                for single in singles])
            checks = [(repeated_values, base_values), (reversed_values, base_values),
                      (singleton_values, base_values)]
            for observed, expected in checks:
                if not np.allclose(observed, expected, rtol=1e-4, atol=1e-5):
                    raise RuntimeError(f"reference gate failed for {key}")
        del base, repeated, reversed_batch, singles, gate

    if mode == "actual_batch":
        rows = np.asarray(prepared["query_rows"], dtype=np.int64)
        captured = capture(system, _frame_for_rows(pilot, inputs, rows), "cpu")
        full_probabilities, full_embeddings = _checked_capture(captured, len(rows), "actual full batch")
        positions = [prepared["probes"][year]["batch_position"] for year in range(2007, 2015)]
        probabilities = full_probabilities[positions]
        embeddings = full_embeddings[positions]
        years = np.arange(2007, 2015, dtype=np.int64)
    elif mode == "actual_singletons":
        years = np.arange(2007, 2015, dtype=np.int64)
        probabilities, embeddings = [], []
        for year in years:
            row = prepared["probes"][int(year)]["population_row"]
            captured = capture(system, _frame_for_rows(pilot, inputs, [row]), "cpu")
            captured_probabilities, captured_embeddings = _checked_capture(captured, 1, "actual singleton")
            probabilities.append(captured_probabilities[0])
            embeddings.append(captured_embeddings[0])
        probabilities, embeddings = np.stack(probabilities), np.stack(embeddings)
    elif mode == "held_candidate":
        selection = prepared["held_selection"]
        actual_rows = np.asarray(selection["precheck_rows"], dtype=np.int64)
        precheck_population_rows = _population_rows_for_actual(inputs, actual_rows)
        precheck = capture(system, _frame_for_rows(pilot, inputs, precheck_population_rows), "cpu")
        precheck_probabilities, _ = _checked_capture(
            precheck, len(precheck_population_rows), "held actual precheck")
        actual_targets = np.asarray(inputs["actual"]["probability"], dtype=np.float32)[actual_rows]
        if actual_targets.shape != (len(actual_rows),) or not np.isfinite(actual_targets).all():
            raise RuntimeError("invalid actual probabilities for held precheck")
        precheck_errors = np.abs(precheck_probabilities[:, 1] - actual_targets)
        precheck_diagnostic = {
            "status": ("matched" if np.allclose(precheck_probabilities[:, 1], actual_targets,
                                                 rtol=1e-4, atol=1e-5) else "mismatch"),
            "count": int(len(actual_rows)),
            "max_abs_class1_probability_error": float(np.max(precheck_errors)) if len(precheck_errors) else 0.0,
            "within_combined_tolerance": bool(np.allclose(precheck_probabilities[:, 1], actual_targets,
                                                            rtol=1e-4, atol=1e-5)),
        }
        held_actual_rows = np.asarray(selection["held_actual_rows"], dtype=np.int64)
        held_population_rows = _population_rows_for_actual(inputs, held_actual_rows)
        held_frame = _frame_for_rows(pilot, inputs, held_population_rows, reference_year=2007)
        captured = capture(system, held_frame, "cpu")
        full_probabilities, full_embeddings = _checked_capture(captured, len(held_population_rows), "held candidate batch")
        years = np.arange(2008, 2015, dtype=np.int64)
        positions = [inputs["held_probes"][int(year)]["batch_position"] for year in years]
        probabilities = full_probabilities[positions]
        embeddings = full_embeddings[positions]
    else:
        raise ValueError(f"unknown capture condition: {mode}")

    report = _state_report(pilot, system)
    if (report["learned_parameter_identity"] != state_before_capture["learned_parameter_identity"]
            or report["identity"] != state_before_capture["identity"]):
        raise RuntimeError("fitted predictor identity changed during capture")
    condition = {"source": source, "mode": mode, "years": years,
                 "probabilities": np.asarray(probabilities, dtype=np.float32),
                 "embeddings": np.asarray(embeddings, dtype=np.float32),
                 "clinical_input_identity": clinical_identity,
                 "context_identity": system["context_identity"],
                 "context_rows_identity": system["context_rows"],
                 "checkpoint_sha256": system["checkpoint_sha256"],
                 "settings": system["settings"], "precision": precision,
                 "predictor_identity": report,
                 "learned_parameter_identity": report["learned_parameter_identity"],
                 "elapsed_seconds": monotonic() - condition_started}
    if mode == "held_candidate":
        condition["precheck_diagnostic"] = precheck_diagnostic
    del system
    import gc
    gc.collect()
    return condition


def _save_private_probes(path, condition, targets):
    content = {"years": condition["years"],
               "probabilities": condition["probabilities"],
               "embeddings": condition["embeddings"],
               "target_probabilities": np.asarray(targets["probabilities"], dtype=np.float32),
               "target_embeddings_available": np.asarray(targets.get("embeddings") is not None)}
    if targets.get("embeddings") is not None:
        content["target_embeddings"] = np.asarray(targets["embeddings"], dtype=np.float32)
    np.savez_compressed(path, **content)


def _condition_record(condition):
    return {key: value for key, value in condition.items()
            if key not in ("probabilities", "embeddings", "years")}


def _run_record(output, status, fingerprint, started, runtime, inputs, conditions=None, error=None):
    import datetime
    import resource
    from time import monotonic

    record = {"status": status, "fingerprint": fingerprint,
              "started_utc": started,
              "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "elapsed_seconds": monotonic() - started["monotonic"],
              "runtime": runtime,
              "input_fingerprints": inputs["fingerprint_facts"],
              "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              "conditions": conditions or {}, "error": error}
    record["started_utc"] = started["utc"]
    launcher_path = Path(output) / "run" / "run.json"
    if launcher_path.is_file():
        record["launcher"] = json.loads(launcher_path.read_text())
    return record


def _create_analysis_output(output):
    output = Path(output)
    if output.exists():
        if ({path.name for path in output.iterdir()} != {"run"}
                or not (output / "run").is_dir()):
            raise FileExistsError(f"refusing to overwrite replay output: {output}")
        return output
    output.mkdir(parents=True)
    return output


def run_feasibility(output):
    import datetime
    from time import monotonic

    inputs = _load_r1_inputs()
    runtime = _require_replay_runtime(inputs)
    fingerprint = _execution_fingerprint(inputs, runtime)
    output = _create_analysis_output(output)
    started = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "monotonic": monotonic()}
    _json_write(output / "completion.json", {"status": "running", "fingerprint": fingerprint})
    conditions = {}
    try:
        condition = _capture_condition(inputs, "recovered", "actual_batch")
        targets = inputs["targets"]
        _save_private_probes(output / "recovered_actual_probes.npz", condition,
                             {"probabilities": targets["actual_probabilities"],
                              "embeddings": targets["actual_embeddings"]})
        conditions["recovered_actual_batch"] = _condition_record(condition)
        run_record = _run_record(output, "completed", fingerprint, started, runtime, inputs, conditions)
        output_hash = file_sha256(output / "recovered_actual_probes.npz")
        run_record["outputs"] = {"recovered_actual_probes.npz": output_hash}
        _json_write(output / "run.json", run_record)
        completion = {"status": "completed", "fingerprint": fingerprint,
                      "outputs": run_record["outputs"], "run_sha256": file_sha256(output / "run.json")}
        _json_write(output / "completion.json", completion)
        _finalize_output_report(output, None, output)
        return {"status": "completed", "fingerprint": fingerprint, "runtime": runtime,
                "output": str(output), "conditions": conditions["recovered_actual_batch"]}
    except BaseException as error:
        run_record = _run_record(output, "failed", fingerprint, started, runtime, inputs, conditions,
                                 {"type": type(error).__name__, "message": str(error)})
        _json_write(output / "run.json", run_record)
        _json_write(output / "completion.json", {"status": "failed", "fingerprint": fingerprint,
                                                   "partial_outputs": run_record.get("partial_outputs", {}),
                                                   "error": run_record["error"],
                                                   "run_sha256": file_sha256(output / "run.json")})
        try:
            _finalize_output_report(output, None, output)
        except (OSError, ValueError, KeyError):
            pass
        raise


def _load_feasibility(feasibility, fingerprint):
    feasibility = Path(feasibility)
    completion_path = feasibility / "completion.json"
    run_path = feasibility / "run.json"
    probes_path = feasibility / "recovered_actual_probes.npz"
    if not completion_path.is_file() or not run_path.is_file() or not probes_path.is_file():
        raise ValueError("feasibility evidence is incomplete")
    completion = json.loads(completion_path.read_text())
    if completion.get("status") != "completed":
        raise ValueError("feasibility run did not complete")
    if completion.get("fingerprint") != fingerprint:
        raise ValueError("feasibility fingerprint mismatch")
    expected = completion.get("outputs", {}).get("recovered_actual_probes.npz")
    if expected != file_sha256(probes_path):
        raise ValueError("feasibility probe output identity mismatch")
    if completion.get("run_sha256") != file_sha256(run_path):
        raise ValueError("feasibility run-record identity mismatch")
    run_record = json.loads(run_path.read_text())
    if run_record.get("status") != "completed" or run_record.get("fingerprint") != fingerprint:
        raise ValueError("feasibility run record identity mismatch")
    if run_record.get("outputs") != completion.get("outputs"):
        raise ValueError("feasibility output manifest mismatch")
    for name, expected_hash in run_record["outputs"].items():
        path = feasibility / name
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"feasibility output checksum mismatch: {name}")
    with np.load(probes_path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def _compare_source_inputs(left, right, require_weights=True):
    if left["clinical_input_identity"] != right["clinical_input_identity"]:
        raise ValueError("source arms received different transformed clinical inputs")
    if left["context_rows_identity"] != right["context_rows_identity"]:
        raise ValueError("source arms received different context rows")
    if left["checkpoint_sha256"] != right["checkpoint_sha256"]:
        raise ValueError("source arms used different checkpoints")
    for key in ("model_type", "seed", "fit_at_predict_time", "fp16_inference",
                "sklearn_compatible_precision", "domain_policy"):
        if left["settings"].get(key) != right["settings"].get(key):
            raise ValueError(f"source arms used different predictor setting: {key}")
    left_fields = left["predictor_identity"]["field_digests"]
    right_fields = right["predictor_identity"]["field_digests"]
    identity_fields = ("X_", "y_", "_categorical_features", "additional_x_", "categorical_features",
                       "classes_", "c_processed_", "cached_preprocessors_", "cached_shuffles_",
                       "data_mean_", "data_std_", "average_logits", "softmax_temperature",
                       "ensemble_configurations_", "transformer_predict_kwargs", "preprocess_transforms",
                       "feature_shift_decoder", "normalize_with_test", "normalize_to_ranking",
                       "normalize_to_clip", "remove_outliers", "device_", "fit_at_predict_time",
                       "fp16_inference", "sklearn_compatible_precision", "pilot_checkpoint_sha256_",
                       "pilot_precision_", "pilot_domain_policy_")
    missing = [key for key in identity_fields if key not in left_fields or key not in right_fields]
    if missing:
        raise ValueError(f"source arms lack comparable fitted input/config identities: {missing}")
    comparable = list(identity_fields)
    mismatched = [key for key in comparable if left_fields[key] != right_fields[key]]
    if mismatched:
        raise ValueError(f"source arms have different fitted input/config identities: {mismatched}")
    weights_match = left.get("learned_parameter_identity") == right.get("learned_parameter_identity")
    if not weights_match:
        raise ValueError("source arms fitted different learned parameters")
    return {"compared_field_identities": comparable,
            "learned_parameters_match": weights_match}


def _record_identity_comparison(conditions, name, left, right):
    try:
        proof = _compare_source_inputs(left, right)
    except ValueError:
        conditions["identity_comparisons"][name] = {
            "status": "mismatch",
            "qualification": "fitted input, predictor configuration, or learned parameter identity differed",
        }
        raise
    recorded = {"status": "matched", **proof}
    conditions["identity_comparisons"][name] = recorded
    return recorded


def run_comparison(feasibility, output):
    import datetime
    from time import monotonic

    inputs = _load_r1_inputs()
    runtime = _require_replay_runtime(inputs)
    fingerprint = _execution_fingerprint(inputs, runtime)
    reused = _load_feasibility(feasibility, fingerprint)
    recovered_state = json.loads((Path(feasibility) / "run.json").read_text())["conditions"]["recovered_actual_batch"]
    output = _create_analysis_output(output)
    started = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "monotonic": monotonic()}
    _json_write(output / "completion.json", {"status": "running", "fingerprint": fingerprint})
    conditions = {"identity_comparisons": {}}
    output_files = {}
    try:
        current_full = _capture_condition(inputs, "current", "actual_batch")
        conditions["current_actual_batch"] = current_full
        _save_private_probes(output / "current_actual_batch.npz", current_full,
                             {"probabilities": inputs["targets"]["actual_probabilities"],
                              "embeddings": inputs["targets"]["actual_embeddings"]})
        output_files["current_actual_batch.npz"] = file_sha256(output / "current_actual_batch.npz")
        actual_identity = _record_identity_comparison(
            conditions, "actual_source", recovered_state, current_full)
        current_singletons = _capture_condition(inputs, "current", "actual_singletons")
        conditions["current_actual_singletons"] = current_singletons
        _save_private_probes(output / "current_actual_singletons.npz", current_singletons,
                             {"probabilities": inputs["targets"]["actual_probabilities"],
                              "embeddings": inputs["targets"]["actual_embeddings"]})
        output_files["current_actual_singletons.npz"] = file_sha256(output / "current_actual_singletons.npz")
        schedule_identity = _record_identity_comparison(
            conditions, "current_schedule", current_full, current_singletons)
        recovered_held = _capture_condition(inputs, "recovered", "held_candidate")
        conditions["recovered_held_candidate"] = recovered_held
        _save_private_probes(output / "recovered_held_candidate.npz", recovered_held,
                             {"probabilities": inputs["targets"]["held_probabilities"],
                              "embeddings": None})
        output_files["recovered_held_candidate.npz"] = file_sha256(output / "recovered_held_candidate.npz")
        current_held = _capture_condition(inputs, "current", "held_candidate")
        conditions["current_held_candidate"] = current_held
        _save_private_probes(output / "current_held_candidate.npz", current_held,
                             {"probabilities": inputs["targets"]["held_probabilities"],
                              "embeddings": None})
        output_files["current_held_candidate.npz"] = file_sha256(output / "current_held_candidate.npz")
        held_identity = _record_identity_comparison(
            conditions, "held_source", recovered_held, current_held)
        # The recovered actual condition was fit in the feasibility run and is reused only
        # after its complete input/runtime fingerprint has matched above.
        run_record = _run_record(output, "completed", fingerprint, started, runtime, inputs,
                                 {name: _condition_record(value) for name, value in conditions.items()})
        run_record["feasibility_reused"] = True
        run_record["outputs"] = output_files
        _json_write(output / "run.json", run_record)
        completion = {"status": "completed", "fingerprint": fingerprint,
                      "feasibility_reused": True, "outputs": run_record["outputs"],
                      "run_sha256": file_sha256(output / "run.json")}
        _json_write(output / "completion.json", completion)
        _finalize_output_report(feasibility, output, output)
        return {"status": "completed", "fingerprint": fingerprint,
                "feasibility_reused": True, "runtime": runtime, "output": str(output)}
    except BaseException as error:
        run_record = _run_record(output, "failed", fingerprint, started, runtime, inputs,
                                 {name: _condition_record(value) for name, value in conditions.items()},
                                 {"type": type(error).__name__, "message": str(error)})
        run_record["feasibility_reused"] = False
        run_record["partial_outputs"] = output_files
        _json_write(output / "run.json", run_record)
        _json_write(output / "completion.json", {"status": "failed", "fingerprint": fingerprint,
                                                   "partial_outputs": run_record["partial_outputs"],
                                                   "error": run_record["error"],
                                                   "run_sha256": file_sha256(output / "run.json")})
        try:
            _finalize_output_report(feasibility, output, output)
        except (OSError, ValueError, KeyError):
            pass
        raise


def compare_probe_arrays(left, right=None):
    years = np.asarray(left["years"])
    probabilities = np.asarray(left["probabilities"])
    embeddings = np.asarray(left["embeddings"])
    if (years.ndim != 1 or not np.issubdtype(years.dtype, np.integer)
            or probabilities.shape != (len(years), 2) or probabilities.dtype != np.float32
            or embeddings.ndim != 2 or len(embeddings) != len(years) or embeddings.dtype != np.float32
            or not np.isfinite(probabilities).all() or not np.isfinite(embeddings).all()):
        raise ValueError("invalid saved probe output axes, dtype, or finite values")
    if right is None:
        target_probabilities = np.asarray(left.get("target_probabilities"))
        if (target_probabilities.shape not in ((len(years),), probabilities.shape)
                or target_probabilities.dtype != np.float32 or not np.isfinite(target_probabilities).all()):
            raise ValueError("invalid saved target probability axes or dtype")
        target_embeddings_available = bool(np.asarray(left.get("target_embeddings_available", False)).item())
        target_embeddings = np.asarray(left["target_embeddings"]) if target_embeddings_available else None
        if target_embeddings_available and (target_embeddings.shape != embeddings.shape
                                              or target_embeddings.dtype != np.float32):
            raise ValueError("invalid saved target embedding axes or dtype")
    else:
        right_years = np.asarray(right["years"])
        target_probabilities = np.asarray(right["probabilities"])
        target_embeddings = np.asarray(right["embeddings"])
        if not np.array_equal(years, right_years):
            raise ValueError("saved probe years differ between comparison arms")
        if (target_probabilities.shape != probabilities.shape or target_probabilities.dtype != np.float32
                or target_embeddings.shape != embeddings.shape or target_embeddings.dtype != np.float32
                or not np.isfinite(target_probabilities).all() or not np.isfinite(target_embeddings).all()):
            raise ValueError("invalid reference probe output axes, dtype, or finite values")

    output = []
    for index, year in enumerate(years):
        target_probability = (target_probabilities[index, 1] if target_probabilities.ndim == 2
                              else target_probabilities[index])
        probability_error = float(abs(float(probabilities[index, 1]) - float(target_probability)))
        probability_pass = bool(np.isclose(probabilities[index, 1], target_probability,
                                           rtol=1e-4, atol=1e-5))
        if target_embeddings is None:
            embedding_status, embedding_error, embedding_pass = "unavailable", None, None
        else:
            difference = np.abs(embeddings[index] - target_embeddings[index])
            embedding_status = "available"
            embedding_error = float(np.max(difference)) if difference.size else 0.0
            embedding_pass = bool(np.allclose(embeddings[index], target_embeddings[index],
                                              rtol=1e-4, atol=1e-5))
        output.append({"year": int(year),
                       "probability_status": "available",
                       "max_abs_class1_probability_error": probability_error,
                       "class1_probability_within_combined_tolerance": probability_pass,
                       "raw_embedding_status": embedding_status,
                       "max_abs_raw_embedding_error": embedding_error,
                       "raw_embedding_within_combined_tolerance": embedding_pass})
    return output


def build_saved_report(feasibility, comparison=None):
    feasibility = Path(feasibility)
    comparison = Path(comparison) if comparison is not None else None
    feasibility_attempt = _load_saved_attempt(feasibility)
    comparison_attempt = _load_saved_attempt(comparison) if comparison is not None else None
    feasibility_run = feasibility_attempt["run"]
    comparison_run = comparison_attempt["run"] if comparison_attempt else None
    rows = []
    feasibility_valid = feasibility_attempt["status"] in ("completed", "failed")
    comparison_valid = (comparison_attempt is not None
                        and comparison_attempt["status"] in ("completed", "failed"))
    fingerprints_match = (feasibility_valid and comparison_valid
                          and feasibility_run.get("fingerprint") == comparison_run.get("fingerprint"))
    allow_metrics = feasibility_valid and (comparison is None or fingerprints_match)
    comparison_conditions = (comparison_run or {}).get("conditions", {})
    identity_proofs = comparison_conditions.get("identity_comparisons", {})

    def qualified(input_name, value):
        if input_name != "held":
            return value
        note = ("historical held writer/schedule and dependency identity unknown; "
                "candidate uses 64 rows; CPU replay vs original CUDA")
        return f"{value}; {note}" if value else note

    def add_file_metrics(name, arm, input_name, reference="archive", qualification=None,
                         identity_requirement=None):
        if identity_requirement is not None:
            proof = identity_proofs.get(identity_requirement)
            if not isinstance(proof, dict) or proof.get("status") != "matched":
                status = "identity_mismatch" if isinstance(proof, dict) and proof.get("status") == "mismatch" else "unavailable"
                rows.append({"year": None, "input": input_name, "comparison": arm,
                             "probability_status": status, "raw_embedding_status": status,
                             "qualification": "fitted identity comparison did not pass"})
                return
        left = feasibility_attempt["artifacts"].get(name)
        if left is None and comparison_attempt is not None:
            left = comparison_attempt["artifacts"].get(name)
        if left is None:
            rows.append({"year": None, "input": input_name, "comparison": arm,
                         "probability_status": "unavailable", "raw_embedding_status": "unavailable",
                         "qualification": qualified(input_name, "missing verified output")})
            return
        if reference == "archive":
            metrics = compare_probe_arrays(left)
        elif reference is None:
            rows.append({"year": None, "input": input_name, "comparison": arm,
                         "probability_status": "unavailable", "raw_embedding_status": "unavailable",
                         "qualification": qualified(input_name, "missing verified comparison reference")})
            return
        else:
            right = feasibility_attempt["artifacts"].get(reference)
            if right is None and comparison_attempt is not None:
                right = comparison_attempt["artifacts"].get(reference)
            if right is None:
                rows.append({"year": None, "input": input_name, "comparison": arm,
                             "probability_status": "unavailable", "raw_embedding_status": "unavailable",
                             "qualification": qualified(input_name, "missing verified comparison reference")})
                return
            metrics = compare_probe_arrays(left, right)
        for metric in metrics:
            rows.append({"year": metric["year"], "input": input_name,
                         "comparison": arm, "probability_status": metric["probability_status"],
                         "max_abs_class1_probability_error": metric["max_abs_class1_probability_error"],
                         "class1_probability_within_combined_tolerance": metric["class1_probability_within_combined_tolerance"],
                         "raw_embedding_status": metric["raw_embedding_status"],
                         "max_abs_raw_embedding_error": metric["max_abs_raw_embedding_error"],
                         "raw_embedding_within_combined_tolerance": metric["raw_embedding_within_combined_tolerance"],
                         "qualification": qualified(input_name, qualification)})
        return True

    if allow_metrics:
        add_file_metrics("recovered_actual_probes.npz", "recovered_actual_batch_vs_archive", "actual")
        _duplicate_held_reference_row(rows, "recovered_actual_batch_vs_archive")

    if comparison_attempt is not None and fingerprints_match:
        add_file_metrics("current_actual_batch.npz", "current_actual_batch_vs_archive", "actual")
        _duplicate_held_reference_row(rows, "current_actual_batch_vs_archive")
        add_file_metrics("current_actual_singletons.npz", "current_actual_singletons_vs_archive", "actual")
        _duplicate_held_reference_row(rows, "current_actual_singletons_vs_archive")
        add_file_metrics("current_actual_batch.npz", "recovered_vs_current_actual_batch", "actual",
                         "recovered_actual_probes.npz", identity_requirement="actual_source")
        add_file_metrics("current_actual_batch.npz", "current_full_vs_singletons", "actual",
                         "current_actual_singletons.npz", identity_requirement="current_schedule")
        add_file_metrics("recovered_held_candidate.npz", "recovered_held_candidate_vs_archive", "held")
        add_file_metrics("current_held_candidate.npz", "current_held_candidate_vs_archive", "held")
        add_file_metrics("current_held_candidate.npz", "recovered_vs_current_held_candidate", "held",
                         "recovered_held_candidate.npz", identity_requirement="held_source")

    source_records = [record for record in (feasibility_run, comparison_run) if record]
    source_hashes = {}
    dependency_identity = None
    for record in source_records:
        fingerprints = record.get("input_fingerprints", {})
        source_hashes.update(fingerprints.get("source_hashes", {}))
        dependency_identity = fingerprints.get("dependency_identity", dependency_identity)
    run_status = feasibility_attempt["status"]
    comparison_status = comparison_attempt["status"] if comparison_attempt is not None else "not_run"
    if comparison is not None:
        if not feasibility_valid:
            run_status = "invalid_feasibility"
        elif feasibility_attempt["status"] != "completed":
            run_status = "feasibility_failed"
        elif not comparison_valid:
            run_status = comparison_status
        elif not fingerprints_match:
            run_status = "fingerprint_mismatch"
            comparison_status = "fingerprint_mismatch"
        else:
            run_status = comparison_status
    return {
        "schema": "r1-2007-replay-aggregate-v1",
        "status": run_status,
        "feasibility_status": feasibility_attempt["status"],
        "comparison_status": comparison_status,
        "fingerprints": {"feasibility": feasibility_run.get("fingerprint") if feasibility_run else None,
                         "comparison": comparison_run.get("fingerprint") if comparison_run else None},
        "source_sha256": source_hashes,
        "dependency_source_identity": dependency_identity,
        "tolerances": {"rtol": 1e-4, "atol": 1e-5, "criterion": "combined elementwise tolerance"},
        "runtime": {"feasibility": _safe_runtime(feasibility_run),
                    "comparison": _safe_runtime(comparison_run)},
        "timing": {"feasibility": _safe_timing(feasibility_run),
                   "comparison": _safe_timing(comparison_run),
                   "conditions": {"feasibility": _safe_condition_timing(feasibility_run),
                                  "comparison": _safe_condition_timing(comparison_run)}},
        "arm_settings": {"feasibility": _safe_condition_settings(feasibility_run),
                         "comparison": _safe_condition_settings(comparison_run)},
        "learned_parameter_hashes": {"feasibility": _safe_parameter_hashes(feasibility_run),
                                     "comparison": _safe_parameter_hashes(comparison_run)},
        "identity_validation": {name: {"status": proof.get("status"),
                                        "learned_parameters_match": proof.get("learned_parameters_match"),
                                        "compared_field_count": len(proof.get("compared_field_identities", [])),
                                        "qualification": proof.get("qualification")}
                                for name, proof in identity_proofs.items() if isinstance(proof, dict)},
        "held_precheck_diagnostics": {
            name: {key: condition["precheck_diagnostic"].get(key)
                   for key in ("status", "count", "max_abs_class1_probability_error",
                               "within_combined_tolerance")}
            for name, condition in comparison_conditions.items()
            if isinstance(condition, dict) and isinstance(condition.get("precheck_diagnostic"), dict)
        },
        "held_2007_control": {
            "status": "duplicated_actual_reference_control",
            "independent_held_archive_claim": False,
            "description": "The 2007 held control repeats the actual reference probe; no independent held archive row exists.",
        },
        "historical_held_raw_embeddings": "unavailable",
        "limitations": {
            "historical_dependency_identity": "unknown; only current dependency identity is verified",
            "historical_held_writer_and_schedule": "unknown; current candidate uses a 64-row held batch",
            "execution_device": "CPU replay; original source runtime was CUDA",
            "omitted_non_inference_continuation": [
                "finite-difference gradient checks", "autograd gradient continuation",
                "continuation embedding export/rebuild/replay",
            ],
            "inference_scope": "preserves the reference classifier capture sequence and batch probes; captured probe embeddings are reported",
        },
        "comparisons": rows,
    }


def write_saved_report(feasibility, comparison=None, output=None):
    import datetime

    report = build_saved_report(feasibility, comparison)
    source = Path(comparison) if comparison is not None else Path(feasibility)
    output_path = Path(output) if output is not None else source / (
        "regenerated-report-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite report output: {output_path}")
    output_path.mkdir(parents=True)
    _write_report_files(output_path, report)
    return report


def _finalize_output_report(feasibility, comparison, output):
    """Finalize a runner-owned report and bind its checksums into completion evidence."""
    output = Path(output)
    report = build_saved_report(feasibility, comparison)
    paths = _write_report_files(output, report)
    hashes = {path.name: file_sha256(path) for path in paths}
    run_path, completion_path = output / "run.json", output / "completion.json"
    run_record = json.loads(run_path.read_text())
    completion = json.loads(completion_path.read_text())
    run_record["summary_files"] = hashes
    _json_write(run_path, run_record)
    completion["summary_files"] = hashes
    completion["run_sha256"] = file_sha256(run_path)
    _json_write(completion_path, completion)
    return report


def _load_saved_attempt(directory):
    if directory is None:
        return {"status": "not_run", "run": None, "artifacts": {}}
    directory = Path(directory)
    try:
        completion = json.loads((directory / "completion.json").read_text())
        run_path = directory / "run.json"
        run = json.loads(run_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"status": "incomplete_or_invalid", "run": None, "artifacts": {}}
    if completion.get("run_sha256") != file_sha256(run_path):
        return {"status": "invalid_run_checksum", "run": None, "artifacts": {}}
    status = completion.get("status")
    if status not in ("completed", "failed") or run.get("status") != status:
        return {"status": "incomplete_or_invalid", "run": None, "artifacts": {}}
    if completion.get("fingerprint") != run.get("fingerprint"):
        return {"status": "invalid_fingerprint", "run": None, "artifacts": {}}
    manifest_key = "outputs" if status == "completed" else "partial_outputs"
    manifest = run.get(manifest_key, {})
    if completion.get("outputs" if status == "completed" else "partial_outputs", manifest) != manifest:
        return {"status": "invalid_output_manifest", "run": None, "artifacts": {}}
    if completion.get("summary_files", {}) != run.get("summary_files", {}):
        return {"status": "invalid_summary_manifest", "run": None, "artifacts": {}}
    for name, expected in run.get("summary_files", {}).items():
        path = directory / name
        if not path.is_file() or file_sha256(path) != expected:
            return {"status": "invalid_summary_checksum", "run": None, "artifacts": {}}
    artifacts = {}
    for name, expected in manifest.items():
        path = directory / name
        if not path.is_file() or file_sha256(path) != expected:
            return {"status": "invalid_output_checksum", "run": None, "artifacts": {}}
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                artifacts[name] = {key: archive[key].copy() for key in archive.files}
    return {"status": status, "run": run, "artifacts": artifacts}


def _safe_run_record(directory):
    if directory is None:
        return None
    path = Path(directory) / "run.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _attempt_status(directory, run_record):
    if run_record is not None:
        return run_record.get("status", "unknown")
    if directory is None:
        return "not_run"
    completion = Path(directory) / "completion.json"
    if completion.is_file():
        try:
            return json.loads(completion.read_text()).get("status", "unknown")
        except (OSError, json.JSONDecodeError):
            return "invalid_record"
    return "not_started"


def _safe_runtime(run_record):
    if not run_record:
        return None
    runtime = run_record.get("runtime", {})
    return {key: runtime.get(key) for key in ("device", "threads", "precision", "seed", "memory_limit_kib")}


def _safe_timing(run_record):
    if not run_record:
        return None
    return {"elapsed_seconds": run_record.get("elapsed_seconds"),
            "peak_rss_kib": run_record.get("peak_rss_kib")}


def _safe_condition_timing(run_record):
    if not run_record:
        return {}
    return {name: condition.get("elapsed_seconds")
            for name, condition in run_record.get("conditions", {}).items()
            if isinstance(condition, dict) and "elapsed_seconds" in condition}


def _safe_condition_settings(run_record):
    if not run_record:
        return {}
    allowed = ("model_type", "seed", "fit_at_predict_time", "fp16_inference",
               "sklearn_compatible_precision", "domain_policy")
    summary = {}
    for name, condition in run_record.get("conditions", {}).items():
        if isinstance(condition, dict) and isinstance(condition.get("settings"), dict):
            summary[name] = {key: condition["settings"].get(key) for key in allowed}
    return summary


def _safe_parameter_hashes(run_record):
    if not run_record:
        return {}
    return {name: condition.get("learned_parameter_identity")
            for name, condition in run_record.get("conditions", {}).items()
            if isinstance(condition, dict) and condition.get("learned_parameter_identity")}


def _duplicate_held_reference_row(rows, comparison):
    actual = next((row for row in rows if row["comparison"] == comparison
                   and row["input"] == "actual" and row["year"] == 2007), None)
    if actual is None:
        return
    rows.append({**actual, "input": "held_2007_control",
                 "qualification": ("duplicated actual reference control; no independent held archive; "
                                   "historical held writer/schedule and dependency identity unknown; "
                                   "candidate uses 64 rows; CPU replay vs original CUDA")})


def _write_report_files(output, report):
    import csv

    output = Path(output)
    json_path = output / "report.json"
    csv_path = output / "report.csv"
    markdown_path = output / "report.md"
    _json_write(json_path, report)
    columns = ("year", "input", "comparison", "probability_status",
               "max_abs_class1_probability_error", "class1_probability_within_combined_tolerance",
               "raw_embedding_status", "max_abs_raw_embedding_error",
               "raw_embedding_within_combined_tolerance", "qualification")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["comparisons"])
    summary = ["# r1/2007 replay aggregate report", "",
               f"Status: {report['status']}",
               f"Feasibility: {report['feasibility_status']}",
               f"Comparison: {report['comparison_status']}",
               f"Per-year comparison rows: {len(report['comparisons'])}",
               "Tolerance: abs(error) <= atol + rtol * abs(reference), with rtol=1e-4 and atol=1e-5.",
               "Aggregate rows: [report.csv](report.csv); machine-readable details: [report.json](report.json).", "",
               "| Year | Input | Comparison | Max class-1 error | Class-1 pass | Max embedding error | Embedding pass |",
               "| ---: | --- | --- | ---: | :---: | ---: | :---: |"]
    for row in report["comparisons"]:
        summary.append("| {year} | {input} | {comparison} | {prob} | {probpass} | {embed} | {embedpass} |".format(
            year=row.get("year") if row.get("year") is not None else "—",
            input=row.get("input", "—"), comparison=row.get("comparison", "—"),
            prob=row.get("max_abs_class1_probability_error", "—"),
            probpass=row.get("class1_probability_within_combined_tolerance", "—"),
            embed=row.get("max_abs_raw_embedding_error", "—"),
            embedpass=row.get("raw_embedding_within_combined_tolerance", "—")))
    summary += ["", "## Held actual precheck", "",
                "The leading actual archive rows are compared to their saved class-1 probabilities; a numerical mismatch is diagnostic and does not block the held candidate batch.",
                "", "| Arm | Status | Rows | Max class-1 error | Within tolerance |",
                "| --- | --- | ---: | ---: | :---: |"]
    for name, diagnostic in report.get("held_precheck_diagnostics", {}).items():
        summary.append("| {name} | {status} | {count} | {error} | {passed} |".format(
            name=name, status=diagnostic.get("status", "unavailable"),
            count=diagnostic.get("count", "—"),
            error=diagnostic.get("max_abs_class1_probability_error", "—"),
            passed=diagnostic.get("within_combined_tolerance", "—")))
    summary += ["", "## Replay limitations", "",
               f"Historical dependency identity: {report['limitations']['historical_dependency_identity']}.",
               f"Held writer and schedule: {report['limitations']['historical_held_writer_and_schedule']}.",
               f"Execution device: {report['limitations']['execution_device']}.",
               "Omitted non-inference continuation: "
               + ", ".join(report["limitations"]["omitted_non_inference_continuation"]) + ".",
               f"Inference scope: {report['limitations']['inference_scope']}.", "",
               "Held 2007 control duplicates the actual reference output; no independent held archive is claimed.",
               "Historical held raw embeddings are unavailable.",
               "Patient identities, features, probabilities, and embeddings are not included.", ""]
    markdown_path.write_text("\n".join(summary))
    return csv_path, json_path, markdown_path
