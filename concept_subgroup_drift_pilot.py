"""Focused notebook fixtures; scientific/configuration behavior awaits observed red."""

import ast
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time


def _configuration_fixtures(presets, notebook):
    """Check the approved visible presets and override contract without patient data."""
    headings = [cell["source"][0] for cell in notebook["cells"]
                if cell["cell_type"] == "markdown" and cell["source"]
                and cell["source"][0].startswith("## ")]
    assert len(headings) == 12, "Notebook must retain twelve experimental sections"
    assert [int(h.split(".", 1)[0].removeprefix("## ")) for h in headings] == list(range(1, 13))
    expected = {
        "v0": ("cpu", 0, 900, 0, 0, 0),
        "smoke": ("cuda", 20, 2700, 1, 1, 2),
        "fit": ("cuda", 100, 6300, 4, 5, 4),
        "pilot": ("cuda", 100, 5400, 4, 5, 4),
    }
    for profile, values in expected.items():
        config = resolve_config(presets, profile)
        assert (config["device"], config["epochs"], config["budget_seconds"],
                config["candidate_cap"], config["max_followup_years"],
                config["intervention_cap"]) == values, profile
        assert config["profile"] == profile
        assert config["sae_seed"] == 42
        assert config["sae_seeds"] == ([42] if profile in ("v0", "smoke") else [42, 135, 215])
        assert config["cav_seeds"] == [42, 43, 44]
        assert config["activation_target"] == 0.9
        assert config["tcav_cutoffs"] == [0.1, 0.9]
        assert config["encoding_batch_size"] == 256
        assert config["aggregate_budget_seconds"] == 28800
        assert config["split_seed"] == config["primary_cav_seed"] == 42
        assert config["amp"] == "off" and config["tf32"] is False
        assert config["classification_threshold"] == 0.5
        assert config["bootstrap_replicates"] == 100
        assert config["confidence_level"] == 0.95
        assert config["label_availability"] == "end_of_year"
        assert config["sae"] == {"model_type": "ReLU", "scaling_factor": 1.5,
                                  "alpha": 0.1, "learning_rate": 0.001, "weight_decay": 0.0}

    modified = resolve_config(presets, "fit", {"sae_seed": 135, "budget_seconds": 600,
                                                "data_path": "events.feather",
                                                "cache_dir": "cache", "output_dir": "results"})
    assert modified["sae_seed"] == 135 and modified["budget_seconds"] == 600
    assert modified["data_path"] == "events.feather"
    assert modified["cache_dir"] == "cache" and modified["output_dir"] == "results"
    assert resolve_config(presets, "v0", {"device": "auto"})["device"] == "auto"
    assert resolve_config(presets, "fit", {"sae_seed": 215})["sae_seed"] == 215
    modified["cav_seeds"].append(999)
    modified["sae"]["alpha"] = 9.0
    fresh = resolve_config(presets, "fit")
    assert fresh["cav_seeds"] == [42, 43, 44] and fresh["sae"]["alpha"] == 0.1
    assert presets["shared"]["cav_seeds"] == [42, 43, 44]
    for profile, overrides in [
        ("unknown", {}), ("v0", {"device": "tpu"}),
        ("fit", {"sae_seed": 7}), ("smoke", {"sae_seed": 135}),
        ("pilot", {"sae_seed": 135}), ("v0", {"budget_seconds": 0}),
        ("smoke", {"budget_seconds": 2701}), ("fit", {"budget_seconds": "600"}),
        ("fit", {"sae_seed": True}), ("v0", {"budget_seconds": True}),
        ("fit", {"epochs": 101}), ("pilot", {"candidate_cap": 5}),
        ("smoke", {"activation_target": 0.7}), ("fit", {"unknown": 1}),
    ]:
        try:
            resolve_config(presets, profile, overrides)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid profile/override accepted: {profile}, {overrides}")


def resolve_config(presets, profile="v0", overrides=None):
    """Resolve independent presets; allowed overrides are paths, device, seed and budget."""
    raise NotImplementedError("Configuration resolution awaits observed Fiji fixture red")


def _atomic_json(path, payload):
    # pragmatic: tiny fixture evidence uses CPU I/O; no scientific computation here.
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_configuration_checks(notebook_path="concept_subgroup_drift_pilot.ipynb"):
    """Run focused checks on Fiji; persist failure and re-raise its original exception."""
    output = Path(os.environ["RPI_OUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    path = Path(notebook_path)
    notebook_bytes = path.read_bytes()
    notebook = json.loads(notebook_bytes)
    presets = None
    for cell in notebook["cells"]:
        if cell.get("metadata", {}).get("pilot_role") == "visible_presets":
            assignment = ast.parse("".join(cell["source"])).body[0]
            assert isinstance(assignment, ast.Assign) and assignment.targets[0].id == "PRESET_SPEC"
            presets = ast.literal_eval(assignment.value)
    assert presets is not None, "Visible notebook preset cell missing"
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).splitlines()
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() or None
    launcher_path = Path(os.environ["RPI_RUN_DIR"]) / "run.json" if os.environ.get("RPI_RUN_DIR") else None
    launcher = json.loads(launcher_path.read_text()) if launcher_path and launcher_path.exists() else None
    packages = {}
    for name in ("torch", "numpy", "pandas", "scikit-learn", "nbconvert", "ipykernel"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    started = time.monotonic()
    manifest = {
        "schema_version": "1", "run_id": Path(os.environ["RPI_RUN_DIR"]).name if os.environ.get("RPI_RUN_DIR") else output.name, "status": "running",
        "work_item": "concept-subgroup-drift-pilot", "smoke": False,
        "purpose": "configuration_fixture", "command": sys.orig_argv,
        "host": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "ended_utc": None, "seconds": None,
        "git": {"sha": git_sha, "branch": branch, "dirty": bool(dirty), "dirty_files": dirty},
        "config": {"path": str(path), "hash": "sha256:" + hashlib.sha256(notebook_bytes).hexdigest(),
                   "hash_method": "notebook_source_bytes", "resolved": None,
                   "unavailable_reason": "configuration resolution under test"},
        "data": [], "seed": 42, "deterministic": True,
        "accelerator": {"requested_device": "cpu", "resolved_device": "cpu",
                        "cuda_available": None, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "unavailable_reason": "CPU stdlib fixtures do not import torch or probe CUDA"},
        "environment": {"python": sys.version, "executable": sys.executable, "packages": packages},
        "outputs": {"metrics": "metrics.json", "stages": "stages.json",
                    "artifacts": [], "checkpoints_on_host_only": []},
        "launcher": launcher, "error": None,
    }
    _atomic_json(output / "run-manifest.json", manifest)
    stage = {"name": "configuration_fixtures", "status": "running",
             "started_utc": manifest["started_utc"], "ended_utc": None,
             "seconds": None, "error": None}
    _atomic_json(output / "stages.json", {"stages": [stage]})
    try:
        _configuration_fixtures(presets, notebook)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    else:
        manifest["status"] = "completed"
        print("Configuration fixtures passed")
    finally:
        manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["seconds"] = time.monotonic() - started
        _atomic_json(output / "run-manifest.json", manifest)
        stage.update(status=manifest["status"], ended_utc=manifest["ended_utc"],
                     seconds=manifest["seconds"], error=manifest["error"])
        _atomic_json(output / "stages.json", {"stages": [stage]})
        _atomic_json(output / "metrics.json", {"configuration_checks": manifest["status"],
                                               "error": manifest["error"]})
