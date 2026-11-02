"""Notebook configuration and focused fixtures; scientific stages remain unavailable."""

import ast
from copy import deepcopy
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


def _profile_checks(presets, notebook):
    """Check the v0/pilot/analysis profile contract, configuration hash and removed surfaces."""
    import copy
    import re
    profiles = sorted(set(presets) - {"shared"})
    assert profiles == ["analysis", "pilot", "v0"], f"Profiles must be exactly v0, pilot, analysis: {profiles}"
    for profile in ("smoke", "fit", "unknown", "", "Pilot", None):
        try:
            resolve_config(presets, profile)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Profile accepted: {profile!r}")
    rule = {"objective": "f2", "min_precision": 0.5, "min_lift": 1.5, "max_rules": 5,
            "max_rule_length": 10, "min_marginal_recall": 0.02, "rule_source": "semantic",
            "recurrence": {"max_candidates_per_bootstrap": 20, "family_similarity_threshold": 0.70,
                           "min_family_recurrence": 0.50}}
    expected = {
        "reference_year_candidates": list(range(2007, 2016)), "t0_only_reference_years": [2015],
        "split_seed": 42, "sae_seeds": [42, 135, 215], "epochs": 100,
        "threshold_sweep": [0.1, 0.2, 0.3, 0.4, 0.5], "rule": rule, "candidate_cap_fraction": 0.10,
        "matching_roles": ["rule_discovery", "cav_fit"], "chance_permutations": 20,
        "cav_seeds": [42, 43, 44], "primary_cav_seed": 42, "bootstrap_replicates": 100,
        "disk_guard_gib": 15, "joblib_workers": 32, "tcav_cutoffs": [0.1, 0.9],
        "encoding_batch_size": 256, "amp": "off", "tf32": False, "classification_threshold": 0.5,
        "confidence_level": 0.95, "label_availability": "end_of_year",
        "sae": {"model_type": "ReLU", "scaling_factor": 1.5, "alpha": 0.1,
                "learning_rate": 0.001, "weight_decay": 0.0},
        "data_path": None, "cache_dir": None, "output_dir": None}
    obsolete_keys = ("budget_seconds", "aggregate_budget_seconds", "activation_target", "candidate_cap")
    for profile, device in (("pilot", "cuda"), ("analysis", "cpu")):
        config = resolve_config(presets, profile)
        assert config["profile"] == profile and config["device"] == device, profile
        for key, value in expected.items():
            assert key in config and config[key] == value, f"{profile}.{key}: {config.get(key, 'missing')!r} != {value!r}"
        assert not [key for key in config if "budget" in key or key in obsolete_keys], profile
    tiny = resolve_config(presets, "v0")
    assert tiny["device"] == "cpu" and 0 < tiny["epochs"] <= 5 and tiny["joblib_workers"] <= 4
    assert set(tiny) == set(config), "v0 must expose the same configuration keys"
    assert not [key for key in tiny if "budget" in key or key in obsolete_keys]
    assert tiny["rule"]["rule_source"] == "semantic" and tiny["matching_roles"] == expected["matching_roles"]

    for overrides in ({"budget_seconds": 600}, {"epochs": 101}, {"sae_seeds": [42]}, {"unknown": 1},
                      {"device": "tpu"}, {"cache_dir": ""}, {"cache_dir": 3}, {"joblib_workers": 1}):
        try:
            resolve_config(presets, "pilot", overrides)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid override accepted: {overrides}")
    routed = resolve_config(presets, "pilot", configuration_overrides({"PILOT_CACHE_DIR": "cache"}))
    assert routed["cache_dir"] == "cache" and resolve_config(presets, "analysis", {"device": "auto"})["device"] == "auto"
    for environment in ({"PILOT_BUDGET_SECONDS": "600"}, {"PILOT_EPOCHS": "100"}):
        try:
            configuration_overrides(environment)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsupported environment accepted: {environment}")

    assert candidate_cap(288, expected["candidate_cap_fraction"]) == 29
    assert candidate_cap(int(192 * expected["sae"]["scaling_factor"]), expected["candidate_cap_fraction"]) == 29
    assert [candidate_cap(n) for n in (1, 10, 30, 100, 289, 291)] == [1, 1, 3, 10, 29, 30]
    for latents, fraction in ((0, 0.1), (-5, 0.1), (True, 0.1), (28.8, 0.1), (288, 0), (288, 1.5), (288, True)):
        try:
            candidate_cap(latents, fraction)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid cap input accepted: {latents!r}, {fraction!r}")

    base = resolve_config(presets, "pilot")
    digest = config_hash(base)
    assert re.fullmatch("[0-9a-f]{64}", digest) and base["config_hash"] == digest
    assert resolve_config(presets, "pilot")["config_hash"] == config_hash(copy.deepcopy(base)) == digest
    assert config_hash(dict(reversed(list(base.items())))) == digest
    assert resolve_config(presets, "analysis")["config_hash"] == digest, "analysis must reuse the pilot cache"
    moved = resolve_config(presets, "pilot", {"data_path": "d", "cache_dir": "c", "output_dir": "o", "device": "cpu"})
    assert moved["config_hash"] == digest, "paths and device must not change the hash"
    assert resolve_config(presets, "v0")["config_hash"] != digest
    operational = copy.deepcopy(base)
    operational.update(disk_guard_gib=16, joblib_workers=16)
    assert config_hash(operational) == digest, "disk guard and worker count must not change the hash"
    assert resolve_config(presets, "pilot", {"disk_guard_gib": 20})["config_hash"] == digest
    mutations = [("epochs", 101), ("sae_seeds", [42, 215, 135]), ("sae.alpha", 0.2), ("rule.min_lift", 1.6),
                 ("rule.rule_source", "high_precision"), ("rule.recurrence.min_family_recurrence", 0.6),
                 ("rule.recurrence.family_similarity_threshold", 0.8), ("rule.recurrence.max_candidates_per_bootstrap", 21),
                 ("threshold_sweep", [0.1, 0.2, 0.3, 0.4]),
                 ("candidate_cap_fraction", 0.2), ("matching_roles", ["rule_discovery"]),
                 ("chance_permutations", 21), ("cav_seeds", [42, 43, 45]), ("bootstrap_replicates", 101),
                 ("split_seed", 43),
                 ("reference_year_candidates", list(range(2007, 2015))), ("t0_only_reference_years", []),
                 ("tcav_cutoffs", [0.1, 0.8]), ("label_availability", "next_year")]
    hashes = {digest}
    for path, value in mutations:
        changed = copy.deepcopy(base)
        target = changed
        *parents, leaf = path.split(".")
        for parent in parents:
            target = target[parent]
        target[leaf] = value
        hashes.add(config_hash(changed))
    assert len(hashes) == len(mutations) + 1, "Every configuration value change must change the hash"

    for name in ("fit_reference_target", "stable_diagnostic_entrant_role", "diagnostic_patient_roles"):
        assert name not in globals(), f"Obsolete helper still exported: {name}"
        assert not [cell for cell in notebook["cells"] if cell["cell_type"] == "code"
                    and name in "".join(cell["source"])], f"Notebook still references {name}"
    cells = notebook["cells"]
    position = next(i for i, cell in enumerate(cells) if cell.get("metadata", {}).get("pilot_role") == "visible_presets")
    display = "".join(cells[position + 1]["source"])
    assert "json.dumps(CONFIG" in display and "PILOT_PROFILE" in display
    assert not [word for word in ("budget", "max_followup_years", "smoke") if word in display.lower()]


def _configuration_fixtures(presets, notebook):
    """Check the approved visible presets and override contract without patient data."""
    headings = [cell["source"][0] for cell in notebook["cells"]
                if cell["cell_type"] == "markdown" and cell["source"]
                and cell["source"][0].startswith("## ")]
    assert len(headings) == 12, "Notebook must retain twelve experimental sections"
    assert [int(h.split(".", 1)[0].removeprefix("## ")) for h in headings] == list(range(1, 13))
    _profile_checks(presets, notebook)
    assert [cell for cell in notebook["cells"] if cell["cell_type"] == "code"
            and "run_unit_cache_checks" in "".join(cell["source"])], "Notebook must run the unit cache checks"
    assert [cell for cell in notebook["cells"] if cell["cell_type"] == "code"
            and "run_reference_sae_unit_checks" in "".join(cell["source"])], "Notebook must run the reference SAE unit checks"
    assert [cell for cell in notebook["cells"] if cell["cell_type"] == "code"
            and "run_family_checks" in "".join(cell["source"])], "Notebook must run the family checks"
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "if PROFILE == \"v0\":" in sources and "pilot.run_orchestration_checks()" in sources, "v0 must check orchestration"
    for name in ("gpu_embedding_gate", "prerequisite", "run_reference_sae_unit_checks", "run_family_checks",
                 "run_cav_selection_checks", "run_idea2_checks", "run_group_intervention_checks"):
        assert name in sources, f"Pilot prerequisite gate missing: {name}"
    assert any(isinstance(node, ast.If) and ast.unparse(node.test) == "PROFILE == 'pilot'"
               and any(isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                       and ast.unparse(statement.value.func) == "prerequisite"
                       and len(statement.value.args) == 2
                       and ast.unparse(statement.value.args[1]) == "pilot.run_predictor_sdpa_checks"
                       for statement in node.body)
               for node in ast.walk(ast.parse(sources))), "Pilot must run scoped predictor SDPA prerequisite"
    for name in ("report_tables", "export_report", "compute_downstream", "overlap_cpu_gpu"):
        assert name in sources, f"Notebook orchestration/report API missing: {name}"
    assert [cell for cell in notebook["cells"] if cell["cell_type"] == "code"
            and "run_cav_selection_checks" in "".join(cell["source"])], "Notebook must run the CAV selection checks"
    assert [cell for cell in notebook["cells"] if cell["cell_type"] == "code"
            and "run_cavs_selection_unit" in "".join(cell["source"])], "Notebook must run the CAV selection unit"
    assert [cell for cell in notebook["cells"] if cell["cell_type"] == "code"
            and "run_reference_performance_unit" in "".join(cell["source"])], "Notebook must run the reference performance unit"
    modified = resolve_config(presets, "pilot", {"data_path": "events.feather", "cache_dir": "cache",
                                                 "output_dir": "results", "device": "cpu"})
    assert modified["data_path"] == "events.feather" and modified["device"] == "cpu"
    assert modified["cache_dir"] == "cache" and modified["output_dir"] == "results"
    modified["cav_seeds"].append(999)
    modified["sae"]["alpha"] = 9.0
    fresh = resolve_config(presets, "pilot")
    assert fresh["cav_seeds"] == [42, 43, 44] and fresh["sae"]["alpha"] == 0.1
    assert presets["shared"]["cav_seeds"] == [42, 43, 44]
    environment = {"PILOT_DEVICE": "cpu", "PILOT_DATA_PATH": "events.feather",
                   "PILOT_CACHE_DIR": "cache", "RPI_OUT_DIR": "out", "PILOT_DISK_GUARD_GIB": "20"}
    routed = resolve_config(presets, "pilot", configuration_overrides(environment))
    assert (routed["device"], routed["data_path"], routed["cache_dir"], routed["output_dir"],
            routed["disk_guard_gib"]) == ("cpu", "events.feather", "cache", "out", 20)
    for environment in ({"PILOT_EPOCHS": "101"}, {"PILOT_SAE_SEED": "42"}, {"PILOT_BUDGET_SECONDS": "600"},
                        {"PILOT_DISK_GUARD_GIB": ""}, {"PILOT_DISK_GUARD_GIB": "0"},
                        {"PILOT_DISK_GUARD_GIB": "1.5"}):
        try:
            resolve_config(presets, "pilot", configuration_overrides(environment))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid environment override accepted: {environment}")
    fake_python = "/fixture/venv/bin/python"
    fake_spec = {"argv": [fake_python, "-m", "ipykernel_launcher", "-f", "{connection_file}"]}
    assert verify_notebook_environment(fake_python, fake_spec, fake_python)["executable"] == fake_python
    for expected_python, spec, actual in [(fake_python, fake_spec, "/wrong/python"),
                                         (fake_python, {"argv": ["python"]}, fake_python),
                                         (None, fake_spec, fake_python),
                                         (fake_python, {"argv": []}, fake_python)]:
        try:
            verify_notebook_environment(expected_python, spec, actual)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("Wrong or unspecified notebook executable accepted")


PROFILES = ("v0", "pilot", "analysis")
# Absent context_window means reference_only, so R1 configuration hashes stay valid.
CONTEXT_WINDOWS = ("reference_only", "last_3", "all_history")
FIRST_CONTEXT_YEAR = 2007
HASH_EXCLUDED_KEYS = ("config_hash", "profile", "device", "data_path", "cache_dir", "output_dir",
                      "disk_guard_gib", "joblib_workers", "inference_batch_rows")


def resolve_config(presets, profile="v0", overrides=None):
    """Resolve independent presets; only paths, device and disk guard may override them."""
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile!r}")
    overrides = {} if overrides is None else overrides
    allowed = {"data_path", "cache_dir", "output_dir", "device", "disk_guard_gib", "context_window", "reference_start",
               "predictor_cpu_from"}
    if not isinstance(overrides, dict) or set(overrides) - allowed:
        raise ValueError("Only paths, device, disk guard, context window and reference start may override presets")
    if overrides.get("context_window", "last_3") not in CONTEXT_WINDOWS[1:]:
        raise ValueError(f"context_window override must be one of {CONTEXT_WINDOWS[1:]}")
    overrides = deepcopy(overrides)
    start = overrides.pop("reference_start", None)
    config = deepcopy(presets["shared"])
    config.update(deepcopy(presets[profile]))
    config.update(deepcopy(overrides))
    if start is not None:
        if start not in config["reference_year_candidates"]:
            raise ValueError("reference_start must be one of the preset reference years")
        config["reference_year_candidates"] = [year for year in config["reference_year_candidates"] if year >= start]
    if "predictor_cpu_from" in config and config["predictor_cpu_from"] not in config["reference_year_candidates"]:
        raise ValueError("predictor_cpu_from must be one of the build's reference years")
    config["profile"] = profile
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    for key in ("data_path", "cache_dir", "output_dir"):
        if key in overrides and (not isinstance(config[key], str) or not config[key].strip()):
            raise ValueError(f"{key} override must be a nonempty path string")
    if type(config["disk_guard_gib"]) is not int or config["disk_guard_gib"] <= 0:
        raise ValueError("disk_guard_gib must be a positive integer")
    config["config_hash"] = config_hash(config)
    return config


def config_hash(config):
    """Hash scientific settings; profile, device and locations do not identify cached results."""
    payload = {key: value for key, value in config.items() if key not in HASH_EXCLUDED_KEYS}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def candidate_cap(num_latents, fraction=0.10):
    """Largest candidate count: ceil(fraction * num_latents), exact for decimal fractions."""
    import math
    from fractions import Fraction
    if type(num_latents) is not int or num_latents <= 0:
        raise ValueError("num_latents must be a positive integer")
    if type(fraction) not in (int, float) or not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    return math.ceil(Fraction(str(fraction)) * num_latents)


def configuration_overrides(environment):
    """Translate explicitly supported environment inputs; material knobs stay frozen."""
    mapping = {"PILOT_DATA_PATH": "data_path", "PILOT_CACHE_DIR": "cache_dir",
               "PILOT_DEVICE": "device", "PILOT_DISK_GUARD_GIB": "disk_guard_gib",
               "RPI_OUT_DIR": "output_dir", "PILOT_CONTEXT_WINDOW": "context_window",
               "PILOT_REFERENCE_START": "reference_start", "PILOT_PREDICTOR_CPU_FROM": "predictor_cpu_from"}
    unknown = {key for key in environment if key.startswith("PILOT_")} - set(mapping) - {
        "PILOT_PROFILE", "PILOT_EXPECTED_PYTHON"}
    if unknown:
        raise ValueError(f"Unsupported pilot environment settings: {sorted(unknown)}")
    overrides = {target: environment[source] for source, target in mapping.items() if source in environment}
    if "disk_guard_gib" in overrides:
        value = overrides["disk_guard_gib"]
        if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
            raise ValueError("disk_guard_gib environment input must contain decimal integer digits")
        overrides["disk_guard_gib"] = int(value)
    for key in ("reference_start", "predictor_cpu_from"):
        if key in overrides:
            value = overrides[key]
            if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
                raise ValueError(f"{key} environment input must contain decimal integer digits")
            overrides[key] = int(value)
    return overrides


def verify_notebook_environment(expected_python, kernel_spec, executable=None):
    """Require the activated venv executable and run-local kernelspec to agree."""
    if not expected_python or not Path(expected_python).is_absolute():
        raise ValueError("PILOT_EXPECTED_PYTHON must name the absolute activated Fiji executable")
    executable = sys.executable if executable is None else executable
    expected = os.path.abspath(expected_python)
    argv = kernel_spec.get("argv", [])
    if os.path.abspath(executable) != expected:
        raise RuntimeError("Notebook executable differs from activated Fiji Python")
    if not argv or not Path(argv[0]).is_absolute() or os.path.abspath(argv[0]) != expected:
        raise RuntimeError("Run-local kernelspec executable differs from activated Fiji Python")
    return {"executable": expected, "kernel_executable": argv[0]}


def _atomic_json(path, payload):
    # pragmatic: tiny fixture evidence uses CPU I/O; no scientific computation here.
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class DiskGuardStop(Exception):
    """Raised, after recording the stop, when the next unit would leave too little free disk."""


UNIT_STAGES = ("P", "S", "D.families", "D.rules", "D.cavs_selection", "D.reference_performance", "D.idea1",
               "D.idea2_frozen", "D.idea2_yearly_sae", "D.idea2_yearly_families", "D.groups_interventions")
STAGE_CODE_VERSIONS = {"P": 2}


def unit_stage(key):
    parts = key.split("/") if isinstance(key, str) else []
    stage = parts[0] if parts[:1] in (["P"], ["S"]) else "D." + parts[2] if parts[:1] == ["D"] and len(parts) > 2 else None
    if stage not in UNIT_STAGES:
        raise ValueError(f"Unknown unit key: {key!r}")
    return stage


def unit_graph(config):
    """Ordered unit key to dependency keys: P(r), then seed-major S(seed, r), then D(r) sub-units."""
    years, seeds = list(config["reference_year_candidates"]), list(config["sae_seeds"])
    standalone = set(config["t0_only_reference_years"])
    if years != sorted(set(years)) or not seeds or len(set(seeds)) != len(seeds) or not standalone <= set(years):
        raise ValueError("Reference years must be ascending and unique, seeds unique, T0-only years listed years")
    graph = {f"P/{r}": () for r in years}
    for seed in seeds:
        graph.update({f"S/{seed}/{r}": (f"P/{r}",) for r in years})
    for r in years:
        reference, prefix = f"P/{r}", f"D/{r}/"
        families, rules, cavs = prefix + "families", prefix + "rules", prefix + "cavs_selection"
        graph[families] = (reference, *(f"S/{seed}/{r}" for seed in seeds))
        graph[rules] = (reference, families)
        graph[cavs] = (reference, families, rules)
        graph[prefix + "reference_performance"] = (reference, cavs)
        for t in ([] if r in standalone else [year for year in years if year > r]):
            idea1, frozen, yearly = f"{prefix}idea1/{t}", f"{prefix}idea2_frozen/{t}", f"{prefix}idea2_yearly_families/{t}"
            trained = [f"{prefix}idea2_yearly_sae/{seed}/{t}" for seed in seeds]
            graph[idea1] = (reference, rules, cavs)
            graph[frozen] = (reference, families, cavs)
            graph.update({key: (reference, f"S/{seed}/{r}") for seed, key in zip(seeds, trained)})
            graph[yearly] = (reference, families, *trained)
            graph[f"{prefix}groups_interventions/{t}"] = (reference, cavs, idea1, frozen)
    return graph


def unit_order(config):
    return list(unit_graph(config))


class UnitCache:
    """Resumable unit outputs under <root>/<config hash>; the manifest record is the commit point."""

    def __init__(self, root, config, source_hash, stage_versions=None, free_bytes=None, expected_hash=None, read_only=False):
        import shutil
        self.config, self.source_hash, self.config_hash = config, source_hash, config_hash(config)
        self.read_only = bool(read_only)
        if expected_hash is not None and expected_hash != self.config_hash:
            raise ValueError("Cache resume requested for a different configuration hash")
        if config.get("config_hash", self.config_hash) != self.config_hash:
            raise ValueError("Configuration hash does not match configuration values")
        versions = dict(stage_versions or {})
        if set(versions) - set(UNIT_STAGES) or any(type(value) is not int or value < 1 for value in versions.values()):
            raise ValueError("Stage versions must map known stages to positive integers")
        self.stage_versions = {stage: versions.get(stage, STAGE_CODE_VERSIONS.get(stage, 1)) for stage in UNIT_STAGES}
        self.graph = unit_graph(config)
        self.guard_bytes = int(config["disk_guard_gib"]) * 2**30
        self.directory = Path(root) / self.config_hash
        self._free_bytes = free_bytes or (lambda: shutil.disk_usage(self.directory).free)
        if self.read_only:
            if not (self.directory / "config.json").is_file():
                raise FileNotFoundError(f"Read-only cache absent: {self.directory}")
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
        frozen = json.loads(json.dumps({k: v for k, v in config.items() if k not in HASH_EXCLUDED_KEYS},
                                       sort_keys=True))
        stored = self.directory / "config.json"
        if not stored.exists():
            _atomic_json(stored, frozen)
        elif json.loads(stored.read_text()) != frozen:
            raise ValueError("Stored configuration differs from the requested configuration")
        self._manifest, self._latest, self._memo = self.directory / "units.jsonl", {}, {}
        if self._manifest.exists():
            raw = self._manifest.read_bytes()
            if self.read_only and raw and not raw.endswith(b"\n"):
                raise ValueError("Read-only cache manifest truncated; refused without repair")
            for line in raw.decode(errors="replace").splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    if self.read_only:
                        raise ValueError("Read-only cache manifest malformed; refused without repair")
                    continue
                if isinstance(record, dict) and record.get("config_hash") == self.config_hash and "key" in record:
                    self._latest[record["key"]] = record
            if raw and not raw.endswith(b"\n"):
                with open(self._manifest, "ab") as stream:
                    stream.write(b"\n")

    def _unit_path(self, key):
        self._dependencies(key)
        return self.directory.joinpath("units", *key.split("/"))

    def _tmp_path(self, key):
        self._dependencies(key)
        return self.directory / ".tmp" / key.replace("/", "__")

    def _dependencies(self, key):
        if key not in self.graph:
            raise ValueError(f"Unknown unit: {key!r}")
        return self.graph[key]

    def is_complete(self, key):
        if key in self._memo:
            return self._memo[key]
        record, dependencies = self._latest.get(key), self._dependencies(key)
        done = (record is not None and record["status"] == "ok"
                and record.get("source_hash") == self.source_hash
                and record["stage_version"] == self.stage_versions[unit_stage(key)]
                and self._unit_path(key).is_dir()
                
                and all(self.is_complete(dep) and record["dependencies"].get(dep) == self._latest[dep]["id"]
                        for dep in dependencies))
        self._memo[key] = done
        return done

    def _append(self, key, status, reason=None, **fields):
        if self.read_only:
            raise PermissionError("Cannot append to read-only unit cache")
        import uuid
        stage = unit_stage(key)
        record = {"id": uuid.uuid4().hex, "key": key, "config_hash": self.config_hash,
                  "source_hash": self.source_hash, "stage": stage, "stage_version": self.stage_versions[stage],
                  "status": status, "reason": reason, "device": self.config.get("device"),
                  "start_utc": None, "end_utc": None, "seconds": None, "rows": None,
                  "peak_gpu_bytes": None, "dependencies": {}}
        record.update(fields)
        with open(self._manifest, "a") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._latest[key] = record
        self._memo.clear()
        return record

    def _blocked_by(self, key):
        for dep in self._dependencies(key):
            if self.is_complete(dep):
                continue
            record = self._latest.get(dep)
            if record and record["status"] == "unavailable":
                return record["reason"]
            return f"dep_failed:{dep}" if record and record["status"] == "failed" else f"dep_not_run:{dep}"
        return None

    def run(self, key, fn, projected_bytes=0):
        if self.read_only:
            raise PermissionError("Cannot run units in read-only unit cache")
        import shutil
        import numpy as np
        if self.is_complete(key):
            return self._latest[key]
        reason = self._blocked_by(key)
        if reason is not None:
            latest = self._latest.get(key)
            if latest and (latest["status"], latest["reason"]) == ("unavailable", reason):
                return latest
            return self._append(key, "unavailable", reason)
        if self._free_bytes() - projected_bytes < self.guard_bytes:
            self._append(key, "stopped", "disk_guard")
            raise DiskGuardStop(key)
        tmp, final = self._tmp_path(key), self._unit_path(key)
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        cuda = self.config.get("device") != "cpu" and _cuda_available()
        started, clock = datetime.now(timezone.utc), time.monotonic()
        try:
            if cuda:
                import torch
                torch.cuda.reset_peak_memory_stats()
            payload = fn()
            if not isinstance(payload, dict) or "file" in payload:
                raise TypeError("A unit must return a dict without a 'file' key")
            arrays = {name: value.detach().cpu().numpy() if hasattr(value, "detach") else value
                      for name, value in payload.items() if hasattr(value, "detach") or isinstance(value, np.ndarray)}
            if any(array.dtype == object for array in arrays.values()):
                raise TypeError("Object arrays cannot be cached")
            meta = {name: value for name, value in payload.items() if name not in arrays}
            rows = meta.get("_rows")
            if rows is not None and (type(rows) is not int or rows < 0):
                raise ValueError("_rows must be a nonnegative integer")
            if arrays:
                with open(tmp / "arrays.npz", "wb") as stream:
                    np.savez(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
            with open(tmp / "meta.json", "w") as stream:
                json.dump(meta, stream, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            peak = None
            if cuda:
                torch.cuda.synchronize()
                peak = int(torch.cuda.max_memory_allocated())
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(final, ignore_errors=True)
            os.replace(tmp, final)
        except Exception as error:
            shutil.rmtree(tmp, ignore_errors=True)
            return self._append(key, "failed", f"{type(error).__name__}: {error}",
                                start_utc=started.isoformat(), end_utc=datetime.now(timezone.utc).isoformat(),
                                seconds=time.monotonic() - clock)
        return self._append(key, "ok", None, start_utc=started.isoformat(),
                            end_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic() - clock,
                            rows=rows, peak_gpu_bytes=peak,
                            dependencies={dep: self._latest[dep]["id"] for dep in self._dependencies(key)})

    def load(self, key):
        import numpy as np
        if not self.is_complete(key):
            raise KeyError(f"Unit is not complete: {key}")
        path = self._unit_path(key)
        result = json.loads((path / "meta.json").read_text())
        if (path / "arrays.npz").exists():
            with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
                result.update({name: arrays[name] for name in arrays.files})
        return result

    def completeness_table(self, unit_keys=None):
        rows = []
        for key in (self.graph if unit_keys is None else unit_keys):
            record = self._latest.get(key)
            if self.is_complete(key):
                status = "complete"
            elif record and record["status"] in ("failed", "unavailable"):
                status = f"{record['status']}:{record['reason']}"
            else:
                status = "not_run (stopped)"
            rows.append({"unit": key, "status": status})
        return rows


def _cuda_available():
    import torch
    return torch.cuda.is_available()


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
    git_root = Path(os.environ.get("RPI_GIT_ROOT", Path.cwd()))
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=git_root, text=True).splitlines()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=git_root, text=True).strip() or None
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
        "git": {"sha": git_sha, "branch": branch, "dirty": bool(dirty), "dirty_files": dirty,
                "metadata_root": str(git_root), "source_identity": "source_file_hashes"},
        "source_file_hashes": {path.name: hashlib.sha256(notebook_bytes).hexdigest(),
                               Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "config": {"path": str(path), "hash": "sha256:" + hashlib.sha256(notebook_bytes).hexdigest(),
                   "hash_method": "notebook_source_bytes", "resolved": None,
                   "unavailable_reason": "configuration resolution under test"},
        "data": [], "seed": 42, "deterministic": True,
        "accelerator": {"requested_device": "cpu", "resolved_device": "cpu",
                        "cuda_available": None, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "unavailable_reason": "CPU stdlib fixtures do not import torch or probe CUDA"},
        "environment": {"python": sys.version, "executable": sys.executable, "packages": packages},
        "outputs": {"metrics": "metrics.json", "stages": "stages.json", "resolved_config": None,
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
        resolved = resolve_config(presets, "v0", configuration_overrides(os.environ))
        manifest["config"] = {"path": str(path), "resolved": resolved,
                              "hash": "sha256:" + hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest(),
                              "hash_method": "resolved_config_sorted_json"}
        _atomic_json(output / "resolved-config.json", resolved)
        manifest["outputs"]["resolved_config"] = "resolved-config.json"
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


def reference_patient_roles(patient_ids, outcomes, seed=42):
    """Reuse the fixed reference split, then separate CAV fitting from selection."""
    import numpy as np
    from temporal_splits import split_reference_patients
    patients = _canonical_patient_ids(patient_ids)
    y = np.asarray(outcomes)
    if y.ndim != 1 or len(y) != len(patients) or not np.isin(y, [0, 1]).all():
        raise ValueError("Reference outcomes must be aligned binary values")
    original = split_reference_patients(patients, y, seed=seed)
    roles = {name: sorted(set(patients[i] for i in original[name]))
             for name in ("tabpfn_context", "sae_discovery", "rule_discovery")}
    roles["evaluation"] = sorted(set(patients[i] for i in original["t0_evaluation"]))
    source = original["rule_selection_cav"]
    subdivided = _stratified_patient_partition([patients[i] for i in source],
                                              y[source], (0.5, 0.5), seed)
    roles["cav_fit"], roles["concept_selection"] = subdivided
    _assert_patient_separation(roles)
    return roles


def stable_entrant_role(patient_id, experiment_id, seed=42):
    """Assign unseen patients once, without labels or observation order."""
    patient = _canonical_patient_ids([patient_id])[0]
    payload = json.dumps([patient, str(experiment_id), int(seed)], separators=(",", ":"))
    bucket = int.from_bytes(hashlib.sha256(payload.encode()).digest(), "big") % 16
    return "fitting" if bucket < 13 else "concept_selection" if bucket == 13 else "evaluation"


def annual_patient_inputs(events, patient_ids, years, feature_events, include_labels=False):
    """Build scoped annual counts; label access is explicit and same-year only.

    Rows with no selected events are retained and audited against the old pivot.
    Call with protected patients only when final evaluation is authorized.
    """
    import pandas as pd
    from database import canonicalize_event_df, build_pivot_preserve_presence
    patients = _canonical_patient_ids(patient_ids)
    features = list(feature_events)
    if any(str(feature).upper() == "DEATH" for feature in features) or len(set(features)) != len(features):
        raise ValueError("Features must be unique and exclude DEATH")
    required = {"patient_id", "date", "event"}
    if not required <= set(events.columns):
        raise ValueError("Event data lacks patient_id/date/event")
    _canonical_patient_ids(events.patient_id)
    # Canonicalization uses identifiers/calendar only; restrict before event counts/labels.
    scoped = canonicalize_event_df(events)
    scoped = scoped[scoped.patient_id.isin(patients) & scoped.year.isin(list(years))].copy()
    presence = pd.MultiIndex.from_frame(scoped[["patient_id", "year"]].drop_duplicates().sort_values(["patient_id", "year"]))
    pivot = build_pivot_preserve_presence(scoped, patients, years, features)
    result = pivot.reindex(index=presence, columns=features, fill_value=0).fillna(0).astype("float32")
    if include_labels:
        death_presence = scoped.loc[scoped.event == "DEATH", ["patient_id", "year"]].drop_duplicates()
        death_index = pd.MultiIndex.from_frame(death_presence)
        result["y"] = presence.isin(death_index).astype(int)
        compatible = build_pivot_preserve_presence(scoped, patients, years, features + ["DEATH"])
        extra = presence.difference(compatible.index)
        result.attrs["observation_audit"] = {
            "presence_rows": len(presence), "legacy_selected_plus_death_rows": len(compatible),
            "additional_rows": len(extra), "additional_row_hash": _identity_hash(list(extra)),
            "status": "compatible" if not len(extra) else "eligibility_decision_required",
            "post_death_policy": "unmodified; outcome-dependent final audit deferred",
        }
    else:
        result.attrs["observation_audit"] = {"status": "label_audit_deferred", "presence_rows": len(presence)}
    return result.reset_index()


class MortalityContinuation:
    def __init__(self, head, parse, aggregate, configuration, temperature,
                 classes, context_identity, reweight=False, eval_ys=None):
        import torch
        if list(classes) != [0, 1]:
            raise ValueError("Mortality continuation requires explicit classes [0, 1]")
        if temperature is None:
            raise ValueError("Frozen finite log-temperature is required")
        self.head = deepcopy(head).eval()
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)
        self.parse, self.aggregate = parse, aggregate
        self.configuration = deepcopy(configuration)
        self.temperature = float(temperature)
        if not torch.isfinite(torch.tensor(self.temperature)):
            raise ValueError("Nonfinite temperature")
        self.classes, self.context_identity = tuple(classes), str(context_identity)
        self.reweight = bool(reweight)
        self.eval_ys = eval_ys.detach().clone() if torch.is_tensor(eval_ys) else deepcopy(eval_ys)

    def probabilities(self, raw_embeddings):
        import torch
        if raw_embeddings.ndim != 2 or not torch.isfinite(raw_embeddings).all():
            raise ValueError("Raw embeddings must be finite (rows, dimension)")
        logits = self.head(raw_embeddings[:, None, :])[..., :len(self.classes)].float()
        logits = logits / torch.exp(torch.tensor(self.temperature, device=logits.device))
        result = self.aggregate(self.parse(logits, self.configuration), self.reweight, self.eval_ys)
        if result.shape != (len(raw_embeddings), len(self.classes)) or not torch.isfinite(result).all():
            raise RuntimeError("Continuation returned unsupported/nonfinite probability shape")
        return result

    def death_gradient(self, raw_embeddings):
        import torch
        with torch.enable_grad():
            h = raw_embeddings.detach().clone().requires_grad_(True)
            gradient, = torch.autograd.grad(self.probabilities(h)[:, 1].sum(), h)
        if not torch.isfinite(gradient).all():
            raise RuntimeError("Nonfinite mortality probability gradient")
        return gradient.detach()


def _canonical_patient_ids(patient_ids):
    seen = {}
    result = []
    for value in patient_ids:
        canonical = str(value)
        signature = (type(value).__name__, repr(value))
        if not canonical.strip() or canonical.lower() in ("nan", "none", "nat"):
            raise ValueError("Missing/empty patient identifier")
        if canonical in seen and seen[canonical] != signature:
            raise ValueError("Patient identifiers collide after normalization")
        seen[canonical] = signature
        result.append(canonical)
    return result


def _stratified_patient_partition(patient_ids, outcomes, fractions, seed):
    import numpy as np
    patients = np.asarray(patient_ids, dtype=str)
    outcomes = np.asarray(outcomes)
    unique = np.unique(patients)
    labels = []
    for patient in unique:
        values = np.unique(outcomes[patients == patient])
        if len(values) != 1 or values[0] not in (0, 1):
            raise ValueError("Conflicting/nonbinary patient outcomes")
        labels.append(values[0])
    rng = np.random.default_rng(seed)
    partitions = [[] for _ in fractions]
    for label in (0, 1):
        members = unique[np.asarray(labels) == label].copy()
        rng.shuffle(members)
        exact = np.asarray(fractions) * len(members)
        counts = np.floor(exact).astype(int)
        order = np.argsort(-(exact - counts), kind="stable")
        counts[order[:len(members) - counts.sum()]] += 1
        cursor = 0
        for index, count in enumerate(counts):
            partitions[index].extend(members[cursor:cursor + count].tolist())
            cursor += count
    return [sorted(partition) for partition in partitions]


def _assert_patient_separation(roles):
    seen = set()
    for patients in roles.values():
        if len(set(patients)) != len(patients) or seen & set(patients):
            raise ValueError("Patient reuse across roles")
        seen.update(patients)


def _identity_hash(payload):
    value = {"schema": "pilot-typed-value-v1", "value": _canonical_identity(payload, "root")}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _canonical_identity(value, path):
    """Closed typed value encoding; no pickle, repr, or opaque object state."""
    import math
    import numpy as np
    import torch
    from collections import OrderedDict
    if value is None:
        return ["none"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        return ["float", "nan" if math.isnan(value) else value.hex()]
    if type(value) is complex:
        return ["complex", _canonical_identity(value.real, path), _canonical_identity(value.imag, path)]
    if type(value) is str:
        return ["str", value]
    if type(value) is bytes:
        return ["bytes", value.hex()]
    if isinstance(value, np.generic):
        return ["numpy_scalar", _canonical_identity(np.asarray(value), path)]
    if type(value) is np.ndarray:
        if value.dtype.kind == "O":
            scalar_types = (bool, int, float, complex, str, bytes, type(None), np.generic)
            if any(not isinstance(item, scalar_types) for item in value.flat):
                raise TypeError(f"unsupported nonscalar object array at {path}")
            content = [_canonical_identity(item, f"{path}[{i}]") for i, item in enumerate(value.flat)]
        elif value.dtype.kind in "biufcmMSU":
            normalized = np.array(value, copy=True, order="C")
            if value.dtype.kind == "f":
                normalized[np.isnan(normalized)] = np.nan
            elif value.dtype.kind == "c":
                normalized.real[np.isnan(normalized.real)] = np.nan
                normalized.imag[np.isnan(normalized.imag)] = np.nan
            content = hashlib.sha256(normalized.tobytes(order="C")).hexdigest()
        else:
            raise TypeError(f"unsupported array dtype {value.dtype} at {path}")
        return ["ndarray", value.dtype.str, list(value.shape), content]
    if type(value) in (torch.Tensor, torch.nn.Parameter):
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError(f"unsupported tensor layout at {path}")
        try:
            array = value.detach().cpu().numpy()
        except (TypeError, RuntimeError) as error:
            raise TypeError(f"unsupported tensor dtype at {path}: {value.dtype}") from error
        return ["tensor", str(value.dtype), _canonical_identity(array, path)]
    if type(value) in (list, tuple):
        return ["list" if type(value) is list else "tuple",
                [_canonical_identity(item, f"{path}[{index}]") for index, item in enumerate(value)]]
    if type(value) in (dict, OrderedDict):
        entries = [(_canonical_identity(key, f"{path}.key"), _canonical_identity(item, f"{path}[{key}]"))
                   for key, item in value.items()]
        entries.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, separators=(",", ":")))
        return ["mapping", entries]
    exported = _supported_identity_object(value, path)
    return ["registered_object", type(value).__module__ + "." + type(value).__qualname__,
            _canonical_identity(exported, path)]


def _identity_attributes(value, names, path, optional=()):
    result = {}
    for name in names:
        if not hasattr(value, name):
            raise TypeError(f"unsupported missing required attribute {path}.{name}")
        result[name] = getattr(value, name)
    for name in optional:
        result[name] = {"present": hasattr(value, name),
                        "value": getattr(value, name) if hasattr(value, name) else None}
    return result


def _supported_identity_object(value, path):
    """Explicit configuration and fitted-preprocessor exporters for this fork."""
    from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer
    from tabpfn.scripts.estimator.configs import EnsembleConfiguration, PreprocessorConfig
    from tabpfn.scripts.estimator.feature_transformers import SafePowerTransformer
    from tabpfn.scripts.estimator.preprocessing import (SequentialFeatureTransformer,
        RemoveConstantFeaturesStep, NanHandlingPolynomialFeaturesStep,
        ReshapeFeatureDistributionsStep, EncodeCategoricalFeaturesStep, ShuffleFeaturesStep)
    kind = type(value)
    if kind is EnsembleConfiguration:
        return _identity_attributes(value, ("class_shift_configuration", "feature_shift_configuration",
            "preprocess_transform_configuration", "styles_configuration", "subsample_samples_configuration"), path)
    if kind is PreprocessorConfig:
        return _identity_attributes(value, ("name", "categorical_name", "append_original",
                                           "subsample_features", "global_transformer_name"), path)
    if kind is SequentialFeatureTransformer:
        return {"ordered_steps": list(value), **_identity_attributes(value, ("categorical_features_",), path)}
    common = ("categorical_features_after_transform_",)
    if kind is RemoveConstantFeaturesStep:
        return _identity_attributes(value, common + ("sel_",), path)
    if kind is NanHandlingPolynomialFeaturesStep:
        return _identity_attributes(value, common + ("max_poly_features", "poly_factor_1_idx",
                                                     "poly_factor_2_idx", "standardizer"), path)
    if kind is ShuffleFeaturesStep:
        return _identity_attributes(value, common + ("shuffle_method", "shuffle_index", "index_permutation_"), path)
    if kind is ReshapeFeatureDistributionsStep:
        fields = _identity_attributes(value, common + ("transform_name", "apply_to_categorical",
            "append_to_original", "subsample_features", "global_transformer_name", "subsampled_features_",
            "column_types_", "global_transformer_"), path)
        if value.global_transformer_ is not None:
            raise TypeError(f"unsupported fitted global transformer at {path}")
        # The transform application iterates this mapping; its order is semantic.
        fields["ordered_applied_transforms"] = list(value.applied_transforms_.items())
        return fields
    if kind is EncodeCategoricalFeaturesStep:
        if value.categorical_transform_name not in ("numeric", "none") or value.categorical_transformer_ is not None:
            raise TypeError(f"unsupported fitted categorical transformer at {path}")
        return _identity_attributes(value, common + ("categorical_transform_name", "categorical_transformer_"), path)
    if kind is StandardScaler:
        return _identity_attributes(value, ("copy", "with_mean", "with_std", "mean_", "var_", "scale_",
            "n_samples_seen_", "n_features_in_"), path, optional=("feature_names_in_",))
    if kind in (PowerTransformer, SafePowerTransformer):
        names = ("method", "standardize", "copy", "lambdas_", "n_features_in_")
        if kind is SafePowerTransformer:
            names += ("variance_threshold", "large_value_threshold", "revert_indices_")
        return _identity_attributes(value, names, path, optional=("feature_names_in_", "_scaler"))
    if kind is FunctionTransformer and value is ReshapeFeatureDistributionsStep.none_transformer:
        from types import FunctionType
        expected_code = (lambda x: x).__code__
        code_fields = ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_freevars",
                       "co_cellvars", "co_names", "co_consts", "co_code")
        for name in ("func", "inverse_func"):
            callback = getattr(value, name)
            if (type(callback) is not FunctionType or callback.__closure__ is not None
                    or callback.__defaults__ is not None or callback.__kwdefaults__ is not None
                    or any(getattr(callback.__code__, field) != getattr(expected_code, field) for field in code_fields)):
                raise TypeError(f"unsupported changed identity transformer {name} callback at {path}")
        fields = _identity_attributes(value, ("validate", "accept_sparse", "check_inverse", "feature_names_out",
                                             "kw_args", "inv_kw_args"), path)
        if value.kw_args is not None or value.inv_kw_args is not None:
            raise TypeError(f"unsupported identity transformer callback parameters at {path}")
        return {"semantic": "fork_none_transformer_exact_singleton", "parameters": fields}
    raise TypeError(f"unsupported identity type {kind.__module__}.{kind.__qualname__} at {path}")


def run_role_and_predictor_checks(device="cpu"):
    """Deterministic synthetic checks; never read primary patient outcomes."""
    import numpy as np
    import pandas as pd
    import torch
    from temporal_splits import split_reference_patients

    ids = [f"p{i:03d}" for i in range(128)]
    y = np.asarray([i % 2 for i in range(128)])
    roles = reference_patient_roles(ids, y)
    expected = split_reference_patients(ids, y, seed=42)
    assert set(roles) == {"tabpfn_context", "sae_discovery", "rule_discovery",
                          "cav_fit", "concept_selection", "evaluation"}
    for name in ("tabpfn_context", "sae_discovery", "rule_discovery"):
        assert set(roles[name]) == set(np.asarray(ids)[expected[name]])
    assert set(roles["evaluation"]) == set(np.asarray(ids)[expected["t0_evaluation"]])
    assert set(roles["cav_fit"]) | set(roles["concept_selection"]) == set(np.asarray(ids)[expected["rule_selection_cav"]])
    assert len(roles["cav_fit"]) == len(roles["concept_selection"]) == 8
    assert len(set().union(*(set(v) for v in roles.values()))) == 128
    for left in roles:
        for right in roles:
            if left != right:
                assert not set(roles[left]) & set(roles[right])
    reordered = reference_patient_roles(ids[::-1], y[::-1])
    assert all(set(roles[k]) == set(reordered[k]) for k in roles)
    try:
        reference_patient_roles([1, "1"], [0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("Normalized identifier collision accepted")
    entrance = {str(i): stable_entrant_role(str(i), "ref2007") for i in range(4096)}
    assert set(entrance.values()) == {"fitting", "concept_selection", "evaluation"}
    assert entrance == {str(i): stable_entrant_role(str(i), "ref2007") for i in reversed(range(4096))}
    assert entrance != {str(i): stable_entrant_role(str(i), "ref2008") for i in range(4096)}

    events = pd.DataFrame({"patient_id": ["a", "a", "a", "a", "b", "held"],
                           "date": ["2007-01-01", "2007-03-01", "2008-01-01", "2008-02-01", "2007-02-01", "2007-01-01"],
                           "event": ["X", "X", "DEATH", "X", "DEATH", "FUTURE_FEATURE"]})
    annual = annual_patient_inputs(events, ["a", "b"], [2007, 2008], ["X"], include_labels=True)
    assert list(annual["X"].astype(int)) == [2, 1, 0]
    assert list(annual["y"].astype(int)) == [0, 1, 1]
    assert list(annual["year"]) == [2007, 2008, 2007]
    assert "held" not in set(annual["patient_id"])
    without_labels = annual_patient_inputs(events, ["a"], [2007], ["X"], include_labels=False)
    assert "y" not in without_labels and "DEATH" not in without_labels
    changed = events.copy()
    changed.loc[changed.date.str.startswith("2008"), "event"] = "OTHER"
    pd.testing.assert_frame_equal(without_labels, annual_patient_inputs(changed, ["a"], [2007], ["X"], include_labels=False))
    try:
        annual_patient_inputs(events, ["a"], [2007], ["DEATH"])
    except ValueError:
        pass
    else:
        raise AssertionError("Target allowed as feature")
    collided = pd.DataFrame({"patient_id": [1, "1"], "date": ["2007-01-01"] * 2,
                             "event": ["X", "X"]})
    try:
        annual_patient_inputs(collided, ["1"], [2007], ["X"])
    except ValueError:
        pass
    else:
        raise AssertionError("Raw event identifiers collapsed across distinct patient types")
    try:
        capture_frozen_predictions({"classifier": object()}, pd.DataFrame(), device)
    except ValueError as error:
        assert "empty frame" in str(error)
    else:
        raise AssertionError("Empty continuation population accepted")

    torch.manual_seed(42)
    head = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.GELU(), torch.nn.Linear(5, 4)).to(device).eval()
    # Fixed nonlinear fixture with an appreciable probability derivative. The
    # original random head produced ~2.7e-6 differences at step1e-3: subtraction
    # cancellation in fp32 exceeded the unchanged finite-difference tolerance.
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head[0].weight[0, 0] = 1.0
        head[2].weight[0, 0] = 2.0
        head[2].weight[1, 0] = -2.0
    config = {"class_permutation": [1, 0]}
    labels = torch.tensor([0, 0, 0, 1], device=device)
    def parse(logits, cfg):
        return logits[..., cfg["class_permutation"]]
    def aggregate(logits, reweight, eval_ys):
        probabilities = torch.softmax(logits, dim=-1).mean(dim=1)
        if reweight:
            prior = torch.stack([(eval_ys == i).float().mean() for i in range(2)])
            probabilities = probabilities / prior
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        return probabilities
    continuation = MortalityContinuation(head, parse, aggregate, config, 0.4,
                                         [0, 1], "context-fixture", True, labels)
    h = torch.tensor([[-0.3, 0.2, 0.7], [0.4, -0.6, 0.1]], device=device)
    expected_p = aggregate(parse(head(h[:, None, :])[..., :2].float() / np.exp(0.4), config), True, labels)
    actual = continuation.probabilities(h)
    torch.testing.assert_close(actual, expected_p, rtol=1e-4, atol=1e-5)
    assert not torch.allclose(actual[:, 1], head(h)[:, 1])
    torch.testing.assert_close(continuation.probabilities(h.flip(0)).flip(0), actual, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(torch.cat([continuation.probabilities(row[None]) for row in h]), actual, rtol=1e-4, atol=1e-5)
    direction = torch.tensor([0.3, -0.4, 0.5], device=device)
    finite_evidence = _finite_difference_check(continuation, h, direction, "synthetic")
    assert continuation.context_identity == "context-fixture"
    config["class_permutation"] = [0, 1]
    labels.fill_(0)
    torch.testing.assert_close(continuation.probabilities(h), actual, rtol=1e-4, atol=1e-5)
    print("Role, annual target and exact nonlinear mortality continuation fixtures passed")
    result = {"finite_difference": finite_evidence}
    print(json.dumps(result, sort_keys=True))
    return result


def _finite_difference_check(continuation, h, direction, purpose):
    """Fixed approved numerical gate; record conditioning before asserting."""
    import torch
    step, rtol, atol = 1e-2, 1e-3, 1e-5
    gradient = continuation.death_gradient(h)
    analytical = (gradient * direction).sum(dim=1)
    with torch.no_grad():
        p_plus = continuation.probabilities(h + step * direction)[:, 1]
        p_minus = continuation.probabilities(h - step * direction)[:, 1]
        signal = p_plus - p_minus
        central = signal / (2 * step)
        absolute_error = (analytical - central).abs()
        combined_tolerance = atol + rtol * central.abs()
        elementwise_pass = absolute_error <= combined_tolerance
    analytical_values, central_values = analytical.detach().cpu().tolist(), central.detach().cpu().tolist()
    errors = absolute_error.detach().cpu().tolist()
    report = {"purpose": purpose, "objective": "final_mortality_probability_class_1",
              "context_identity": continuation.context_identity, "dtype": str(h.dtype),
              "device": str(h.device), "step": step, "rtol": rtol, "atol": atol,
              "comparison": "abs(analytical-central) <= atol + rtol*abs(central)",
              "direction_identity": _identity_hash(direction.detach().cpu().numpy()),
              "analytical_derivatives": analytical_values, "central_derivatives": central_values,
              "p_plus": p_plus.detach().cpu().tolist(), "p_minus": p_minus.detach().cpu().tolist(),
              "probability_signal": signal.detach().cpu().tolist(), "absolute_error": errors,
              "relative_error": [error / abs(value) if value else None for error, value in zip(errors, central_values)],
              "relative_error_reason": [None if value else "zero_central_derivative" for value in central_values],
              "combined_tolerance": combined_tolerance.detach().cpu().tolist(),
              "elementwise_pass": elementwise_pass.detach().cpu().tolist(),
              "passed": bool(elementwise_pass.all().cpu()),
              "max_absolute_error": max(errors),
              "minimum_absolute_probability_signal": float(signal.abs().min().cpu())}
    if os.environ.get("RPI_OUT_DIR"):
        evidence_dir = Path(os.environ["RPI_OUT_DIR"]) / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(evidence_dir / f"finite-difference-{purpose}.json", report)
    print(json.dumps(report, sort_keys=True))
    torch.testing.assert_close(analytical, central, rtol=rtol, atol=atol)
    return report


def predictor_context_identity(classifier):
    """Hash fitted context/configuration and the actual cached decoder parameters."""
    return predictor_identity_report(classifier)["identity"]


def predictor_identity_report(classifier):
    """Schema and per-field hashes make fresh-process identity disagreements auditable."""
    import tabpfn.scripts.estimator.configs as configurations_module
    import tabpfn.scripts.estimator.preprocessing as preprocessing_module
    import tabpfn.scripts.estimator.feature_transformers as transformers_module
    fields = {}
    required = ("X_", "y_", "additional_x_", "classes_", "ensemble_configurations_",
                "cached_preprocessors_", "cached_shuffles_", "c_processed_",
                "pilot_checkpoint_sha256_", "pilot_precision_", "pilot_domain_policy_")
    for name in required:
        if not hasattr(classifier, name):
            raise TypeError(f"unsupported incomplete fitted predictor: missing {name}")
    for name in ("X_", "y_", "additional_x_", "classes_", "ensemble_configurations_",
                 "softmax_temperature", "average_logits", "fit_at_predict_time",
                 "sklearn_compatible_precision", "transformer_predict_kwargs",
                 "fp16_inference", "preprocess_transforms", "normalize_with_test",
                 "feature_shift_decoder", "categorical_features", "normalize_to_ranking",
                 "normalize_to_clip", "remove_outliers", "c_processed_",
                 "_categorical_features", "data_mean_", "data_std_", "device_",
                 "cached_preprocessors_", "cached_shuffles_",
                 "pilot_checkpoint_sha256_", "pilot_precision_", "pilot_domain_policy_"):
        fields[name] = {"present": hasattr(classifier, name),
                        "value": getattr(classifier, name) if hasattr(classifier, name) else None}
    heads = []
    for model in getattr(classifier, "cached_models_", []):
        heads.append({key: value.detach().cpu().numpy()
                      for key, value in model.decoder_dict["standard"].state_dict().items()})
    fields["actual_cached_decoder_states"] = heads
    fields["exporter"] = {"schema": "pilot-fitted-predictor-v1",
        "source_fingerprints": {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (configurations_module, preprocessing_module, transformers_module)}}
    return {"schema": "pilot-fitted-predictor-v1", "value_schema": "pilot-typed-value-v1",
            "identity": _identity_hash(fields),
            "field_digests": {name: _identity_hash(value) for name, value in fields.items()},
            "exporter": fields["exporter"]}


def fit_frozen_predictor(context_frame, feature_names, device, seed=42, domain_origin=None):
    """Fit the installed drift-resilient single-fast predictor; never fall back.
    Window contexts pass domain_origin (first context year); each row's domain is its year minus that origin."""
    import numpy as np
    import tabpfn
    from importlib import resources
    from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig
    precision = configure_predictor_precision(device)
    if len(context_frame) == 0 or set(context_frame.y.unique()) != {0, 1}:
        raise ValueError("Reference context requires both mortality classes")
    years = context_frame.year.to_numpy(dtype=np.int64)
    if domain_origin is None and len(np.unique(years)) != 1:
        raise ValueError("Frozen reference context must contain exactly one calendar year")
    origin = int(years[0]) if domain_origin is None else int(domain_origin)
    if years.min() < origin:
        raise ValueError("Context rows precede the domain origin")
    checkpoint = Path(str(resources.files(tabpfn))) / "model_cache" / "tabpfn_dist_model_1.cpkt"
    if not checkpoint.is_file():
        raise FileNotFoundError("Installed drift-resilient checkpoint absent; downloads disabled")
    paths = TabPFNModelPathsConfig(paths=[str(checkpoint)], task_type="dist_shift_multiclass")
    classifier = get_best_tabpfn(task_type="dist_shift_multiclass", model_type="single_fast",
                                paths_config=paths, debug=False, device=device)
    classifier.seed = int(seed)
    classifier.fit_at_predict_time = False
    classifier.sklearn_compatible_precision = False
    classifier.fp16_inference = False
    classifier.normalize_with_test = False
    classifier.pilot_precision_ = precision
    classifier.pilot_checkpoint_sha256_ = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    classifier.pilot_domain_policy_ = ("calendar_year_minus_reference_year" if domain_origin is None
                                       else "calendar_year_minus_first_context_year")
    if hasattr(classifier, "show_progress"):
        classifier.show_progress = False
    X = np.ascontiguousarray(context_frame[list(feature_names)].to_numpy(dtype=np.float32))
    y = np.ascontiguousarray(context_frame.y.to_numpy(dtype=np.float32))
    with predictor_sdpa_context(device):
        classifier.fit(X, y, additional_x={"dist_shift_domain": np.ascontiguousarray(years - origin, dtype=np.int64)})
    if len(classifier.ensemble_configurations_) != 1 or len(classifier.cached_models_) != 1:
        raise RuntimeError("Pilot requires exactly one frozen executed configuration/model")
    if list(classifier.classes_) != [0, 1]:
        raise RuntimeError("Death class mapping differs from explicit [0, 1]")
    return {"classifier": classifier, "context_identity": predictor_context_identity(classifier),
            "context_rows": _identity_hash(list(zip(context_frame.patient_id, context_frame.year))),
            "reference_year": int(years.max()), "domain_origin": origin, "feature_names": tuple(feature_names),
            "checkpoint_sha256": classifier.pilot_checkpoint_sha256_,
            "settings": {"model_type": "single_fast", "fit_at_predict_time": False,
                         "sklearn_compatible_precision": False, "fp16_inference": False,
                         "precision": precision, "domain_policy": classifier.pilot_domain_policy_,
                         "seed": int(seed)}}


def configure_predictor_precision(device="cpu"):
    """One explicit runtime precision setup: fp32, autocast off, both TF32 flags off."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"amp": "off", "fp16_inference": False, "matmul_tf32": False, "cudnn_tf32": False,
            "sdpa_compatibility": _predictor_sdpa_policy(device)}


def _predictor_sdpa_policy(device):
    import torch
    capability = list(torch.cuda.get_device_capability()) if device == "cuda" else None
    return {"enabled": capability is not None and capability[0] < 8,
            "compute_capability": capability,
            "policy": "enable_math_on_cuda_sm_below_8",
            "scope": ["classifier.fit", "classifier.predict_full"],
            "backend_api": "torch.backends.cuda.sdp_kernel"}


def predictor_sdpa_context(device):
    """Scoped backend compatibility for the installed predictor's attention calls."""
    from contextlib import contextmanager
    from functools import wraps
    from inspect import signature
    import torch
    @contextmanager
    def scoped():
        if not _predictor_sdpa_policy(device)["enabled"]:
            yield
            return
        original = torch.backends.cuda.sdp_kernel
        original_signature = signature(original)
        @wraps(original)
        def enable_math(*args, **kwargs):
            request = original_signature.bind_partial(*args, **kwargs)
            request.arguments["enable_math"] = True
            return original(*request.args, **request.kwargs)
        torch.backends.cuda.sdp_kernel = enable_math
        try:
            yield
        finally:
            torch.backends.cuda.sdp_kernel = original
    return scoped()


def run_predictor_sdpa_checks():
    """Low-capability CUDA nested backend requests retain math; CPU/SM8 requests and restoration stay intact."""
    from contextlib import nullcontext
    from unittest import mock
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    import torch
    calls = []
    reject_mathless = [True]
    def backend(enable_flash=True, enable_math=True, enable_mem_efficient=True, **kwargs):
        calls.append(dict(enable_flash=enable_flash, enable_math=enable_math, enable_mem_efficient=enable_mem_efficient, **kwargs))
        if reject_mathless[0] and not enable_math:
            raise RuntimeError("No available kernel")
        return nullcontext()
    with mock.patch.object(torch.backends.cuda, "sdp_kernel", backend), mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)):
        with predictor_sdpa_context("cuda"):
            with torch.backends.cuda.sdp_kernel(False, False, True, marker="preserved"):
                pass
        assert calls[-1] == dict(enable_flash=False, enable_math=True, enable_mem_efficient=True, marker="preserved")
        assert torch.backends.cuda.sdp_kernel is backend
        def fail_inside():
            with predictor_sdpa_context("cuda"):
                raise RuntimeError("scoped exception")
        _refuses(fail_inside, "scoped exception", RuntimeError)
        assert torch.backends.cuda.sdp_kernel is backend, "Backend override leaked after exception"
        reject_mathless[0] = False
        with mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)):
            with predictor_sdpa_context("cuda"):
                with torch.backends.cuda.sdp_kernel(enable_math=False):
                    pass
            assert calls[-1]["enable_math"] is False
        with mock.patch.object(torch.cuda, "get_device_capability", side_effect=AssertionError("CPU queried CUDA capability")):
            with predictor_sdpa_context("cpu"):
                with torch.backends.cuda.sdp_kernel(enable_math=False):
                    pass
            assert calls[-1]["enable_math"] is False
        reject_mathless[0] = True
        head = torch.nn.Linear(2, 2)
        class Classifier:
            ensemble_configurations_ = [SimpleNamespace()]
            cached_models_ = [SimpleNamespace(decoder_dict={"standard": head})]
            classes_ = [0, 1]
            def fit(self, *args, **kwargs):
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                    pass
            def _parse_predictions_per_configuration(self, *args, **kwargs):
                pass
            def _aggregate_predictions(self, *args, **kwargs):
                pass
            def predict_full(self, *args, **kwargs):
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                    raise RuntimeError("predict site reached")
        frame = pd.DataFrame({"f0": [0., 1., 2., 3.], "f1": [1., 0., 1., 0.], "patient_id": ["a", "b", "c", "d"],
                              "year": [2007] * 4, "y": [0, 1, 0, 1]})
        classifier = Classifier()
        with mock.patch("tabpfn.best_models.get_best_tabpfn", return_value=classifier), mock.patch.dict(globals(), predictor_context_identity=lambda _: "fixture-context"):
            system = fit_frozen_predictor(frame, ["f0", "f1"], "cuda")
            assert calls[-1]["enable_math"] is True and calls[-1]["enable_flash"] is True
            assert torch.backends.cuda.sdp_kernel is backend
            with mock.patch("tabpfn_model.infer_model_additional_x_info", return_value=("cpu", (1, 1, 1))), mock.patch("tabpfn_model.make_dist_tensor", return_value=torch.zeros(4, 1, 1)):
                _refuses(lambda: capture_frozen_predictions(system, frame, "cuda"), "predict site reached", RuntimeError)
            assert calls[-1]["enable_math"] is True and torch.backends.cuda.sdp_kernel is backend
            assert not head._forward_pre_hooks, "Prediction hooks leaked after failure"
            assert "_parse_predictions_per_configuration" not in classifier.__dict__ and "_aggregate_predictions" not in classifier.__dict__
            policy = system["settings"]["precision"]["sdpa_compatibility"]
            assert policy["enabled"] is True and policy["compute_capability"] == [7, 5]
            assert system["settings"]["precision"]["amp"] == "off" and not torch.backends.cuda.matmul.allow_tf32
    print("Predictor SDPA fixtures passed: low-capability math forwarding, unchanged CPU/SM8 policy, fit/predict scopes, restoration")


def capture_frozen_predictions(system, frame, device):
    """Capture same-call probabilities/coordinates and replay the exact executed head.

    Temporary hooks observe inputs only; parser and aggregate wrappers call original
    bound methods unchanged, then restore the instance even if prediction fails.
    """
    import numpy as np
    import torch
    classifier = system["classifier"]
    if len(frame) == 0:
        raise ValueError("Cannot capture a predictor continuation from an empty frame")
    identity = predictor_context_identity(classifier)
    if identity != system["context_identity"]:
        raise RuntimeError("Frozen predictor context/configuration changed")
    if classifier.fit_at_predict_time or classifier.sklearn_compatible_precision or classifier.fp16_inference:
        raise RuntimeError("Unsupported context mode or nondifferentiable precision processing")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or torch.is_autocast_enabled():
        raise RuntimeError("Frozen fp32/AMP-off/TF32-off precision changed")
    if len(classifier.cached_models_) != 1:
        raise RuntimeError("Multiple models cannot share one flattened continuation")
    executed_head = classifier.cached_models_[0].decoder_dict["standard"]
    captured = {"head_inputs": []}
    handle = executed_head.register_forward_pre_hook(
        lambda module, args: captured["head_inputs"].append(args[0].detach().clone()))
    parse_original = classifier._parse_predictions_per_configuration
    aggregate_original = classifier._aggregate_predictions
    names = ("_parse_predictions_per_configuration", "_aggregate_predictions")
    prior = {name: classifier.__dict__.get(name) for name in names}
    present = {name: name in classifier.__dict__ for name in names}
    def observe_parse(predictions, ensemble_configurations):
        captured["configurations"] = deepcopy(ensemble_configurations)
        return parse_original(predictions, ensemble_configurations)
    def observe_aggregate(ensemble_outputs, **kwargs):
        captured["aggregate_kwargs"] = {
            key: value.detach().clone() if torch.is_tensor(value) else deepcopy(value)
            for key, value in kwargs.items()}
        return aggregate_original(ensemble_outputs, **kwargs)
    classifier._parse_predictions_per_configuration = observe_parse
    classifier._aggregate_predictions = observe_aggregate
    try:
        features = np.ascontiguousarray(frame[list(system["feature_names"])].to_numpy(dtype=np.float32))
        from temporal_cohorts import relative_domains
        years = np.asarray(frame.year)
        if not np.isfinite(years).all() or not np.equal(years, years.astype(np.int64)).all():
            raise ValueError("Prediction calendar years must be finite integers")
        if (years < system["reference_year"]).any():
            raise ValueError("Prediction before the reference year is unsupported")
        origin = system.get("domain_origin", system["reference_year"])
        domains = np.ascontiguousarray(relative_domains(origin, years), dtype=np.int64)
        from tabpfn_model import infer_model_additional_x_info, make_dist_tensor
        additional_device, fitted_domain_shape = infer_model_additional_x_info(classifier)
        if fitted_domain_shape is None or tuple(fitted_domain_shape[1:]) != (1, 1):
            raise RuntimeError("Unsupported fitted domain tensor shape")
        domain_tensor = make_dist_tensor(domains, additional_device, fitted_domain_shape)
        with torch.inference_mode(), predictor_sdpa_context(device):
            result = classifier.predict_full(features,
                additional_x={"dist_shift_domain": domain_tensor},
                get_additional_outputs=["test_embeddings"])
    finally:
        handle.remove()
        for name in names:
            if present[name]:
                setattr(classifier, name, prior[name])
            else:
                delattr(classifier, name)
    if predictor_context_identity(classifier) != identity:
        raise RuntimeError("Prediction mutated frozen fitted context/configuration")
    embeddings = torch.as_tensor(result["test_embeddings"], device=device)
    if embeddings.ndim != 3 or embeddings.shape[:2] != (len(frame), 1):
        raise RuntimeError("Expected unflattened same-call (rows, one configuration, dimension)")
    if len(captured["head_inputs"]) != 1 or len(captured.get("configurations", [])) != 1:
        raise RuntimeError("Executed decoder/configuration could not be uniquely identified")
    torch.testing.assert_close(captured["head_inputs"][0].to(device), embeddings, rtol=1e-4, atol=1e-5)
    # Clone outside inference mode so downstream autograd can save these tensors.
    h = embeddings[:, 0, :].clone()
    continuation = _frozen_continuation(classifier, executed_head, captured["configurations"],
                                        captured["aggregate_kwargs"], identity)
    p = torch.as_tensor(result["proba"], device=device, dtype=torch.float32)
    with torch.no_grad():
        replay = continuation.probabilities(h)
    torch.testing.assert_close(replay, p, rtol=1e-4, atol=1e-5)
    if not torch.isfinite(h).all() or not torch.isfinite(p).all():
        raise RuntimeError("Nonfinite same-system embeddings/probabilities")
    return {"probabilities": p, "embeddings": h, "continuation": continuation,
            "context_identity": identity,
            "row_identity": _identity_hash(list(zip(frame.patient_id, frame.year))),
            "parity_max_absolute_error": float((replay - p).abs().max().cpu())}


def _frozen_continuation(classifier, executed_head, configurations, aggregate_kwargs, identity):
    """MortalityContinuation around the executed head with the fork's own parser and aggregator, plus the options that
    let export_continuation rebuild it without the predictor."""
    # Only these scalar attributes are read by the fork's classification parser
    # and aggregator. Retain the original methods without copying a transformer.
    frozen_classifier = object.__new__(type(classifier))
    for name in ("is_classification_", "num_classes_", "average_logits", "device_"):
        setattr(frozen_classifier, name, deepcopy(getattr(classifier, name)))
    frozen_parse = frozen_classifier._parse_predictions_per_configuration
    frozen_aggregate = frozen_classifier._aggregate_predictions
    def parse(logits, configurations):
        return frozen_parse(logits, configurations)
    def aggregate(outputs, reweight, eval_ys):
        kwargs = dict(aggregate_kwargs)
        kwargs["reweight_probs_based_on_train"] = reweight
        kwargs["eval_ys"] = eval_ys
        return frozen_aggregate(outputs, **kwargs).squeeze(0)
    continuation = MortalityContinuation(executed_head, parse, aggregate,
        configurations, classifier.softmax_temperature, classifier.classes_, identity,
        aggregate_kwargs["reweight_probs_based_on_train"], aggregate_kwargs["eval_ys"])
    continuation.portable_options = {
        "is_classification": bool(frozen_classifier.is_classification_),
        "average_logits": bool(frozen_classifier.average_logits),
        "num_classes": int(frozen_classifier.num_classes_),
        "class_shifts": [configuration.class_shift_configuration for configuration in configurations]}
    return continuation


def run_installed_predictor_checks(device="cpu"):
    """Tiny synthetic installed-fork gate; no patient file, fitting or GPU shortcut."""
    import numpy as np
    import pandas as pd
    import torch
    # pragmatic: cap CPU threads for a tiny attention fixture; restore on return.
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, prior_threads))
    try:
        generator = np.random.default_rng(42)
        context = pd.DataFrame(generator.normal(size=(32, 3)).astype(np.float32), columns=["f0", "f1", "f2"])
        context["patient_id"] = [f"synthetic-context-{i}" for i in range(32)]
        context["year"], context["y"] = 2007, [i % 2 for i in range(32)]
        test = pd.DataFrame(generator.normal(size=(4, 3)).astype(np.float32), columns=["f0", "f1", "f2"])
        test["patient_id"], test["year"] = [f"synthetic-eval-{i}" for i in range(4)], 2008
        system = fit_frozen_predictor(context, ["f0", "f1", "f2"], device)
        base = capture_frozen_predictions(system, test, device)
        repeated = capture_frozen_predictions(system, test, device)
        reversed_result = capture_frozen_predictions(system, test.iloc[::-1], device)
        single = [capture_frozen_predictions(system, test.iloc[i:i + 1], device) for i in range(4)]
        for key in ("probabilities", "embeddings"):
            torch.testing.assert_close(base[key], repeated[key], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(base[key], reversed_result[key].flip(0), rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(base[key], torch.cat([row[key] for row in single]), rtol=1e-4, atol=1e-5)
        h, continuation = base["embeddings"], base["continuation"]
        direction = torch.linspace(-1, 1, h.shape[1], device=device)
        direction = direction / torch.linalg.vector_norm(direction)
        finite_evidence = _finite_difference_check(continuation, h, direction, "installed")
        portable = _check_continuation_equivalence(continuation, h, base["probabilities"], device)
        print("Installed frozen predictor same-call parity, repeat/order/batch, finite difference, serialized head passed")
        return {"context_identity": system["context_identity"], "settings": system["settings"],
                "continuation_identity": portable["identity"], "continuation_layers": portable["layers"],
                "identity_report": predictor_identity_report(system["classifier"]),
                "checkpoint_sha256": system["checkpoint_sha256"],
                "parity_max_absolute_error": base["parity_max_absolute_error"],
                "finite_difference": finite_evidence,
                "nonlinear_decoder": any(isinstance(module, torch.nn.GELU) for module in continuation.head.modules())}
    finally:
        torch.set_num_threads(prior_threads)


FITTING_SIDE_ROLE_NAMES = ("tabpfn_context", "sae_discovery", "rule_discovery", "cav_fit")
REFERENCE_ROLE_NAMES = FITTING_SIDE_ROLE_NAMES + ("concept_selection", "evaluation")
ENTRANT_ROLE_NAMES = ("fitting", "concept_selection", "evaluation")
HELDOUT_ROLE_NAMES = ("concept_selection", "evaluation")
ROLE_SUPPORT_COLUMNS = ("reference_year", "year", "population", "role", "rows", "patients", "deaths", "survivors",
                        "post_death_rows", "outcomes")
PREDICTOR_GATE_ROWS = 8
REAL_POPULATION_FACTS = {
    "sha256": "c8804591c93087241b9533dce77dd54b2f1e77b23a36400527459cfa8f99add7",
    "rows": 21900, "patients": 5636, "features": 500, "years": list(range(2007, 2016)),
    "rows_per_year": {"2007": 1894, "2008": 2393, "2009": 2485, "2010": 2594, "2011": 2520,
                      "2012": 2529, "2013": 2577, "2014": 2605, "2015": 2303},
    "feature_selection_max_year": 2006, "source_x_dtype": "float32"}


class PopulationFingerprintError(ValueError):
    """Population file bytes differ from the approved sha256; nothing was unpickled."""


class PredictorGateFailure(RuntimeError):
    """A frozen-predictor prerequisite gate failed; the reference year becomes unavailable."""

    def __init__(self, gate, detail):
        super().__init__(f"{gate}: {detail}")
        self.gate = gate


def _synthetic_columns(years, patients_per_year, n_features, seed):
    """Deterministic membership: 20% attrition, even-numbered gap returners, fresh entrants to fill each year."""
    import numpy as np
    rng = np.random.default_rng(seed)
    years = [int(year) for year in years]
    members, created = [list(range(patients_per_year))], patients_per_year
    for k in range(1, len(years)):
        previous = set(members[k - 1])
        kept = [i for i in members[k - 1] if (i + k) % 5 != 0]
        returning = [i for i in (members[k - 2] if k >= 2 else []) if i not in previous and i % 2 == 0]
        fresh = list(range(created, created + max(patients_per_year - len(kept) - len(returning), 0)))
        created += len(fresh)
        members.append(sorted(kept + returning + fresh))
    patient, calendar, outcome = [], [], []
    for k, ids in enumerate(members):
        for i in ids:
            patient.append(f"syn-{i:04d}")
            calendar.append(years[k])
            outcome.append(int((i + 3 * k) % 6 == 0))
    X = rng.normal(size=(len(patient), n_features)).astype(np.float32)
    X[:, 0] += 2.0 * np.asarray(outcome, dtype=np.float32)
    return {"X": X, "y": np.asarray(outcome), "year": np.asarray(calendar), "patient_id": np.asarray(patient),
            "record_keys": np.asarray([f"{p}|{t}" for p, t in zip(patient, calendar)]),
            "feature_names": [f"syn_feature_{j}" for j in range(n_features)],
            "feature_selection_max_year": years[0] - 1}


def synthetic_population(years=(2007, 2008, 2009), patients_per_year=40, n_features=6, seed=7):
    """Tiny in-process population with attrition, entrants and gap returners; no file."""
    return _build_population(**_synthetic_columns(years, patients_per_year, n_features, seed))


def _population_with(population, rows=None, **columns):
    """Copy of a population with full-length column replacements, then an optional row selection."""
    import numpy as np
    values = {name: population[name] for name in ("X", "y", "year", "patient_id", "record_keys")}
    values.update(columns)
    index = np.arange(len(values["y"])) if rows is None else np.asarray(rows)
    return _build_population(feature_names=population["feature_names"],
                             feature_selection_max_year=population["feature_selection_max_year"],
                             **{name: array[index] for name, array in values.items()})


def _first_line(error):
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return (lines[0] if lines else type(error).__name__)[:300]


def _build_population(X, y, year, patient_id, record_keys, feature_names, feature_selection_max_year,
                      file_sha256=None, file_bytes=None, source_dtype=None):
    """Compact retained population in source row order; exactly one row per patient-year."""
    import numpy as np
    features = np.ascontiguousarray(X, dtype=np.float32)
    outcome, calendar = np.asarray(y), np.asarray(year)
    patients = np.asarray(_canonical_patient_ids(np.asarray(patient_id).tolist()))
    keys = np.asarray(record_keys).astype(str)
    names = [str(name) for name in feature_names]
    rows = len(features)
    if features.ndim != 2 or rows == 0 or outcome.ndim != 1 or calendar.ndim != 1 or not (
            len(outcome) == len(calendar) == len(patients) == len(keys) == rows):
        raise ValueError("Population arrays must be nonempty and row-aligned")
    if len(names) != features.shape[1] or len(set(names)) != len(names):
        raise ValueError("Population feature names must be unique and match the feature columns")
    if not np.isin(outcome, [0, 1]).all():
        raise ValueError("Population outcomes must be binary")
    years = calendar.astype(np.int64)
    if not np.array_equal(years, calendar):
        raise ValueError("Population years must be integers")
    outcomes = outcome.astype(np.int64)
    if len(set(zip(patients.tolist(), years.tolist()))) != rows:
        raise ValueError("Population must contain exactly one row per patient-year")
    unique_patients, inverse = np.unique(patients, return_inverse=True)
    first_death = np.full(len(unique_patients), np.iinfo(np.int64).max, dtype=np.int64)
    died = outcomes == 1
    np.minimum.at(first_death, inverse[died], years[died])
    unique_years, counts = np.unique(years, return_counts=True)
    fingerprint = {
        "file_sha256": file_sha256, "file_bytes": file_bytes,
        "source_x_dtype": source_dtype or str(features.dtype),
        "rows": int(rows), "patients": int(len(unique_patients)), "features": int(features.shape[1]),
        "years": [int(value) for value in unique_years],
        "rows_per_year": {str(int(value)): int(count) for value, count in zip(unique_years, counts)},
        "feature_selection_max_year": int(feature_selection_max_year),
        "row_identity": _identity_hash({"patient_id": patients, "year": years}),
        "population_identity": _identity_hash({"X": features, "y": outcomes, "year": years, "patient_id": patients,
                                               "record_keys": keys, "feature_names": names})}
    return {"X": features, "y": outcomes, "year": years, "patient_id": patients, "record_keys": keys,
            "feature_names": tuple(names), "feature_selection_max_year": int(feature_selection_max_year),
            "post_death": years > first_death[inverse], "fingerprint": fingerprint}


def population_from_prepared(prepared, file_sha256=None, file_bytes=None):
    """Compact population from a prepared test partition; training rows only set the feature cut-off year."""
    import numpy as np
    from comparison_runner import _validate_prepared
    _validate_prepared(prepared)
    return _build_population(prepared.X_test, prepared.y_test, prepared.years_test, prepared.patient_ids,
                             prepared.record_keys, prepared.feature_names, int(np.max(prepared.years_train)),
                             file_sha256, file_bytes, str(prepared.X_test.dtype))


def load_pilot_population(path, expected_sha256):
    """Verify the file bytes against the approved sha256 before anything is unpickled."""
    import pickle
    import re
    from comparison_runner import _PreparedData
    if not isinstance(expected_sha256, str) or not re.fullmatch("[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be 64 lowercase hexadecimal characters")
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise PopulationFingerprintError(
            f"Population file sha256 {digest} differs from expected {expected_sha256}; refused before unpickling")
    size = len(raw)
    prepared = pickle.loads(raw)
    del raw
    if not isinstance(prepared, _PreparedData):
        raise TypeError(f"Population file holds {type(prepared).__name__}, not a prepared population")
    return population_from_prepared(prepared, digest, size)


def reference_year_roles(population, r, seed=42):
    """Roles for every year-t row under reference year r; only year-r outcomes are read."""
    import numpy as np
    r = int(r)
    year, patient = population["year"], population["patient_id"]
    at_reference = year == r
    if not at_reference.any():
        raise ValueError(f"Population has no rows in reference year {r}")
    reference_roles = reference_patient_roles(patient[at_reference].tolist(), population["y"][at_reference], seed)
    role_of = {member: role for role, members in reference_roles.items() for member in members}
    eligible = year >= r
    unique, inverse = np.unique(patient[eligible], return_inverse=True)
    names = [role_of[member] if member in role_of else stable_entrant_role(member, r, seed)
             for member in unique.tolist()]
    entrant = [member not in role_of for member in unique.tolist()]
    row_role = np.full(len(year), "", dtype="<U20")
    row_role[eligible] = np.asarray(names)[inverse]
    is_entrant = np.zeros(len(year), dtype=bool)
    is_entrant[eligible] = np.asarray(entrant)[inverse]
    return {"reference_year": r, "experiment_id": r, "split_seed": int(seed), "reference_roles": reference_roles,
            "row_role": row_role, "is_entrant": is_entrant,
            "identity": _identity_hash({"reference_year": r, "split_seed": int(seed),
                                        "row_role": row_role, "is_entrant": is_entrant})}


def year_roles(population, roles, t):
    """Population row indices, roles and entrant flags of calendar year t >= the reference year."""
    import numpy as np
    if int(t) < roles["reference_year"]:
        raise ValueError(f"Year {t} precedes reference year {roles['reference_year']}")
    rows = np.flatnonzero(population["year"] == int(t))
    return rows, roles["row_role"][rows], roles["is_entrant"][rows]


def check_role_coverage(population, roles):
    """Raise ValueError unless roles are disjoint, cover year r, and every eligible row has one role."""
    import numpy as np
    r, year, patient = roles["reference_year"], population["year"], population["patient_id"]
    eligible, role = year >= r, roles["row_role"]
    if len(role) != len(year) or len(roles["is_entrant"]) != len(year):
        raise ValueError("Role arrays must align with population rows")
    if not np.array_equal(role != "", eligible):
        raise ValueError("Exactly the rows of years >= the reference year must carry a role")
    reference = roles["reference_roles"]
    if set(reference) != set(REFERENCE_ROLE_NAMES):
        raise ValueError("Reference roles are incomplete")
    _assert_patient_separation(reference)
    members = set(patient[year == r].tolist())
    if set().union(*(set(values) for values in reference.values())) != members:
        raise ValueError("Reference roles must cover exactly the year-r patients")
    entrant = eligible & ~np.isin(patient, np.asarray(sorted(members)))
    if not np.array_equal(entrant, roles["is_entrant"]):
        raise ValueError("Entrants must be exactly the patients absent from the reference year")
    role_of = {member: name for name, values in reference.items() for member in values}
    for index in np.flatnonzero(eligible):
        allowed = ENTRANT_ROLE_NAMES if entrant[index] else (role_of[patient[index]],)
        if role[index] not in allowed:
            raise ValueError("A row carries a role outside its reference or entrant role set")
    pairs = set(zip(patient[eligible].tolist(), role[eligible].tolist()))
    if len(pairs) != len({member for member, _ in pairs}):
        raise ValueError("A patient changed role between years")
    return True


def role_support_table(population, roles, holdout_outcomes=False):
    """Compact fitting-side support per year >= r; later-year held-out outcomes are withheld by default."""
    import numpy as np
    r = roles["reference_year"]
    year, y, patient, post = population["year"], population["y"], population["patient_id"], population["post_death"]
    role, entrant = roles["row_role"], roles["is_entrant"]
    table = []
    for t in sorted(set(year[year >= r].tolist())):
        at = year == t
        groups = [("reference", name, at & ~entrant & (role == name)) for name in REFERENCE_ROLE_NAMES]
        groups.append(("reference", "all", at & ~entrant))
        if t > r:
            groups += [("entrant", name, at & entrant & (role == name)) for name in ENTRANT_ROLE_NAMES]
            groups.append(("entrant", "all", at & entrant))
        groups.append(("all", "all", at))
        for kind, name, mask in groups:
            reported = holdout_outcomes or t == r or (name not in HELDOUT_ROLE_NAMES and name != "all")
            deaths = int(y[mask].sum())
            rows = int(mask.sum())
            table.append({"reference_year": int(r), "year": int(t), "population": kind, "role": name,
                          "rows": rows, "patients": int(len(np.unique(patient[mask]))),
                          "deaths": deaths if reported else None,
                          "survivors": rows - deaths if reported else None,
                          "post_death_rows": int(post[mask].sum()) if reported else None,
                          "outcomes": "reported" if reported else "withheld_holdout"})
    return table


def write_role_support_csv(rows, path):
    """Atomic CSV of compact support counts; no patient identifiers."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(descriptor, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROLE_SUPPORT_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: "" if row[name] is None else row[name] for name in ROLE_SUPPORT_COLUMNS})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def projected_predictor_bytes(population, r, embedding_bound=512):
    """Upper bound of one predictor unit's cached bytes: float32 probabilities, embeddings and death gradients, plus the serialized head."""
    import numpy as np
    return int(np.sum(population["year"] >= int(r))) * (2 * int(embedding_bound) + 2) * 4 + projected_continuation_bytes(embedding_bound)


def _feature_columns(count):
    return [f"f{index}" for index in range(count)]


def _prediction_frame(population, rows, columns):
    import pandas as pd
    frame = pd.DataFrame(population["X"][rows], columns=columns)
    frame["patient_id"] = population["patient_id"][rows]
    frame["year"] = population["year"][rows]
    return frame


def predictor_unit(population, r, roles, config, device, *, fit=None, capture=None):
    """P(r): one context fit, gates, then one batched pass over every row of years r..last.
    Builds may set predictor_cpu_from: those predictor units run on CPU with joblib_workers threads."""
    import torch
    threads = torch.get_num_threads()
    if config.get("predictor_cpu_from") is not None and int(r) >= config["predictor_cpu_from"]:
        device = "cpu"
        torch.set_num_threads(int(config["joblib_workers"]))
    try:
        return _predictor_unit(population, int(r), roles, config, device,
                               fit_frozen_predictor if fit is None else fit,
                               capture_frozen_predictions if capture is None else capture)
    finally:
        torch.set_num_threads(threads)
        if device == "cuda":
            import gc
            gc.collect()
            torch.cuda.empty_cache()


def _predictor_unit(population, r, roles, config, device, fit, capture):
    import numpy as np
    import torch
    if device not in ("cpu", "cuda"):
        raise ValueError("Predictor unit needs a resolved device: cpu or cuda")
    if roles["reference_year"] != r:
        raise ValueError("Roles belong to a different reference year")
    year = population["year"]
    columns = _feature_columns(population["X"].shape[1])
    window = config.get("context_window", "reference_only")
    if window not in CONTEXT_WINDOWS:
        raise ValueError(f"Unknown context window: {window!r}")
    current = (year == r) & (roles["row_role"] == "tabpfn_context") & ~roles["is_entrant"]
    if not current.any():
        raise ValueError(f"No tabpfn_context rows in reference year {r}")
    # Earlier window years add every row up to the patient's first observed death.
    first = {"reference_only": r, "last_3": max(FIRST_CONTEXT_YEAR, r - 2), "all_history": FIRST_CONTEXT_YEAR}[window]
    prior = (year >= first) & (year < r) & ~population["post_death"]
    context_rows = np.flatnonzero(current | prior)
    if (year[context_rows] > r).any() or (year[context_rows] < FIRST_CONTEXT_YEAR).any():
        raise AssertionError("Context window holds rows outside the allowed years")
    if ((year[context_rows] == r) & (roles["row_role"][context_rows] != "tabpfn_context")).any():
        raise AssertionError("Year-r rows outside tabpfn_context entered the predictor context")
    if (population["post_death"][context_rows] & (year[context_rows] < r)).any():
        raise AssertionError("Post-death earlier-year row entered the predictor context")
    context = _prediction_frame(population, context_rows, columns)
    context["y"] = population["y"][context_rows]
    try:
        system = fit(context, columns, device, seed=int(config["split_seed"]),
                     **({} if window == "reference_only" else {"domain_origin": first}))
    except torch.cuda.OutOfMemoryError:
        raise
    except RuntimeError as error:
        raise PredictorGateFailure("fit", _first_line(error)) from error
    rows = np.flatnonzero(year >= r)
    gate_rows = np.flatnonzero(year == r)[:PREDICTOR_GATE_ROWS]
    if len(gate_rows) < 2:
        raise ValueError(f"Reference year {r} has too few rows for the predictor gates")
    gate_frame = _prediction_frame(population, gate_rows, columns)

    def guarded(frame):
        try:
            return capture(system, frame, device)
        except torch.cuda.OutOfMemoryError:
            raise
        except (AssertionError, RuntimeError) as error:
            raise PredictorGateFailure("capture", _first_line(error)) from error

    def host(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    def close(gate, left, right):
        left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
        try:
            np.testing.assert_allclose(left, right, rtol=1e-4, atol=1e-5)
        except AssertionError as error:
            raise PredictorGateFailure(gate, _first_line(error)) from error
        return float(np.max(np.abs(left - right)))

    base, repeated, flipped = guarded(gate_frame), guarded(gate_frame), guarded(gate_frame.iloc[::-1])
    singles = [guarded(gate_frame.iloc[index:index + 1]) for index in range(len(gate_frame))]
    if (base["embeddings"].ndim != 2 or len(base["embeddings"]) != len(gate_frame)
            or tuple(base["probabilities"].shape) != (len(gate_frame), 2)):
        raise PredictorGateFailure("embedding_axes", "expected (rows, dimension) embeddings and (rows, 2) probabilities")
    gates = {}
    for key in ("probabilities", "embeddings"):
        reference = host(base[key])
        gates[f"repeat_identity.{key}"] = close(f"repeat_identity.{key}", reference, host(repeated[key]))
        gates[f"order_stability.{key}"] = close(f"order_stability.{key}", reference, host(flipped[key])[::-1])
        gates[f"batch_stability.{key}"] = close(f"batch_stability.{key}", reference,
                                                np.concatenate([host(single[key]) for single in singles]))
    h, continuation = base["embeddings"], base["continuation"]
    direction = torch.linspace(-1, 1, h.shape[1], device=device)
    direction = direction / torch.linalg.vector_norm(direction)
    try:
        evidence = _finite_difference_check(continuation, h, direction, f"reference-{r}")
    except (AssertionError, RuntimeError) as error:
        raise PredictorGateFailure("finite_difference", _first_line(error)) from error
    nonlinear = any(isinstance(module, torch.nn.GELU) for module in continuation.head.modules())
    gate_reference = {key: host(base[key]) for key in ("probabilities", "embeddings")}
    try:
        gate_gradients = batched_death_gradients(continuation, h)
        continuation_arrays, continuation_meta = export_continuation(continuation)
        rebuilt = rebuild_continuation(continuation_arrays, continuation_meta, device)
        replay = replay_continuation(rebuilt, gate_reference["embeddings"], gate_reference["probabilities"], gate_gradients)
    except torch.cuda.OutOfMemoryError:
        raise
    except (TypeError, ValueError, RuntimeError) as error:
        raise PredictorGateFailure("continuation_export", _first_line(error)) from error
    del base, repeated, flipped, singles, h, continuation, direction, rebuilt

    frame = _prediction_frame(population, rows, columns)
    configured = config["inference_batch_rows"]
    if type(configured) is not int or configured < 1:
        raise ValueError("inference_batch_rows must be a positive integer")
    probabilities, embeddings, gradients, backoffs, parity, chunks, batch, cursor = [], [], [], [], 0.0, 0, min(configured, len(rows)), 0
    while cursor < len(rows):
        chunk = frame.iloc[cursor:cursor + batch]
        try:
            captured = capture(system, chunk, device)
            gradient = batched_death_gradients(captured["continuation"], captured["embeddings"], len(chunk))
        except torch.cuda.OutOfMemoryError:
            if batch <= 1:
                raise
            if device == "cuda":
                torch.cuda.empty_cache()
            backoffs.append({"rows": int(batch), "next_rows": int(batch // 2)})
            batch //= 2
            continue
        except (AssertionError, RuntimeError) as error:
            raise PredictorGateFailure("capture", _first_line(error)) from error
        probabilities.append(host(captured["probabilities"]).astype(np.float32))
        embeddings.append(host(captured["embeddings"]).astype(np.float32))
        gradients.append(gradient)
        parity = max(parity, float(captured["parity_max_absolute_error"]))
        cursor, chunks = cursor + len(chunk), chunks + 1
        del captured, gradient
    probabilities, embeddings, gradients = np.concatenate(probabilities), np.concatenate(embeddings), np.concatenate(gradients)
    full = {"probabilities": probabilities, "embeddings": embeddings}
    position = np.searchsorted(rows, gate_rows)
    for key, values in full.items():
        gates[f"full_pass_stability.{key}"] = close(f"full_pass_stability.{key}", gate_reference[key], values[position])
    gradient_stability = close("full_pass_stability.gradients", gate_gradients, gradients[position])
    if not (np.isfinite(probabilities).all() and np.isfinite(embeddings).all() and np.isfinite(gradients).all()):
        raise PredictorGateFailure("finite_outputs", "nonfinite probabilities, embeddings or gradients")
    identity_report = predictor_identity_report(system["classifier"])
    if identity_report["identity"] != system["context_identity"]:
        raise PredictorGateFailure("context_identity", "frozen context changed during the unit")
    counts = {str(int(value)): int(np.sum(year[rows] == value)) for value in np.unique(year[rows])}
    return {"row_index": rows.astype(np.int64), "probabilities": probabilities, "embeddings": embeddings,
            "gradients": gradients, **continuation_arrays, "continuation": continuation_meta,
            "continuation_identity": continuation_meta["identity"], "gate_positions": position.astype(np.int64),
            "continuation_replay": replay, "gradient_stability_max_abs_error": gradient_stability,
            "gradient_objective": "final_mortality_probability_class_1_wrt_raw_embeddings",
            "_rows": int(len(rows)), "reference_year": int(r), "last_year": int(year.max()),
            "rows_per_year": counts, "context_rows": int(len(context_rows)), "context_window": window,
            "context_years": sorted(int(value) for value in np.unique(year[context_rows])),
            "context_patients": int(len(np.unique(population["patient_id"][context_rows]))),
            "context_deaths": int(population["y"][context_rows].sum()),
            "embedding_dim": int(embeddings.shape[1]), "feature_count": int(population["X"].shape[1]),
            "device": device, "start_batch_rows": int(min(configured, len(rows))), "batch_rows": int(batch), "oom_backoffs": backoffs, "inference_chunks": int(chunks),
            "gates": gates, "parity_max_absolute_error": parity,
            "finite_difference": {name: evidence[name] for name in (
                "passed", "step", "rtol", "atol", "max_absolute_error", "minimum_absolute_probability_signal",
                "direction_identity", "objective")},
            "nonlinear_decoder": bool(nonlinear), "context_identity": system["context_identity"],
            "context_rows_identity": system["context_rows"], "checkpoint_sha256": system["checkpoint_sha256"],
            "settings": system["settings"],
            "identity_report": {"identity": identity_report["identity"],
                                "field_digests": identity_report["field_digests"]},
            "roles_identity": roles["identity"],
            "population_identity": population["fingerprint"]["population_identity"],
            "output_row_identity": _identity_hash({"row_index": rows})}


def audit_real_population(path, expected_sha256, output_dir=None, seed=42):
    """Real-file audit: fingerprint, per-year row counts, roles for every reference year, support CSV; no fit."""
    population = load_pilot_population(path, expected_sha256)
    fingerprint, facts = population["fingerprint"], REAL_POPULATION_FACTS
    assert expected_sha256 == facts["sha256"] == fingerprint["file_sha256"], "Audit covers the retained file only"
    for key in ("rows", "patients", "features", "years", "rows_per_year", "feature_selection_max_year",
                "source_x_dtype"):
        assert fingerprint[key] == facts[key], f"{key}: {fingerprint[key]!r} != {facts[key]!r}"
    output = Path(output_dir if output_dir is not None else os.environ["RPI_OUT_DIR"])
    year = population["year"]
    table, summary = [], {}
    for r in fingerprint["years"]:
        roles = reference_year_roles(population, r, seed)
        check_role_coverage(population, roles)
        rows = role_support_table(population, roles)
        eligible = {row["year"]: row["rows"] for row in rows if (row["population"], row["role"]) == ("all", "all")}
        assert eligible == {t: facts["rows_per_year"][str(t)] for t in fingerprint["years"] if t >= r}, r
        assert sum(eligible.values()) == int((year >= r).sum())
        table.extend(rows)
        summary[str(r)] = {
            "reference_patients": {name: len(members) for name, members in roles["reference_roles"].items()},
            "reference_rows": {row["role"]: row["rows"] for row in rows
                               if row["year"] == r and row["population"] == "reference"},
            "entrant_rows": {str(row["year"]): row["rows"] for row in rows
                             if row["population"] == "entrant" and row["role"] == "all"},
            "eligible_rows": sum(eligible.values()),
            "fitting_side_post_death_rows": sum(
                row["post_death_rows"] for row in rows if row["post_death_rows"] is not None
                and (row["role"] in FITTING_SIDE_ROLE_NAMES or (row["population"], row["role"]) == ("entrant", "fitting")))}
    csv_path = write_role_support_csv(table, output / "tables" / "role-support.csv")
    result = {"fingerprint": fingerprint, "csv": str(csv_path), "csv_rows": len(table), "reference_years": summary}
    print(json.dumps(result, sort_keys=True))
    return result

def _notebook_presets(notebook):
    for cell in notebook["cells"]:
        if cell.get("metadata", {}).get("pilot_role") == "visible_presets":
            return ast.literal_eval(ast.parse("".join(cell["source"])).body[0].value)
    raise AssertionError("Visible notebook preset cell missing")


def _population_configuration_checks(presets, notebook):
    sha = REAL_POPULATION_FACTS["sha256"]
    for profile in ("pilot", "analysis"):
        assert resolve_config(presets, profile)["population_sha256"] == sha, profile
    tiny = resolve_config(presets, "v0")
    assert tiny["population_sha256"] is None and tiny["data_path"] is None
    pilot = resolve_config(presets, "pilot")
    assert resolve_config(presets, "analysis")["config_hash"] == pilot["config_hash"]
    assert config_hash(dict(pilot, population_sha256="0" * 64)) != pilot["config_hash"]
    assert pilot["inference_batch_rows"] == 4096 and 0 < tiny["inference_batch_rows"] < 120
    assert "inference_batch_rows" in HASH_EXCLUDED_KEYS
    assert config_hash(dict(pilot, inference_batch_rows=8)) == pilot["config_hash"], "Batch size is operational"
    assert resolve_config(presets, "analysis")["inference_batch_rows"] == 4096
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert [text for text in code if "run_population_role_checks" in text], "Notebook must run the population checks"
    assert [text for text in code if "reference_year_roles" in text and "predictor_unit" in text
            and "role_support_table" in text], "Notebook must show roles, support audit and the predictor unit"


def _population_load_checks(population, work):
    import pickle
    import numpy as np
    from comparison_runner import _PreparedData
    years = population["fingerprint"]["years"]
    assert years == [2007, 2008, 2009] and population["fingerprint"]["rows"] == 120
    assert population["fingerprint"]["patients"] == 53

    def prepared(**changes):
        values = {"train_rows": None, "test_rows": None, "feature_names": population["feature_names"],
                  "X_train": population["X"][:4].copy(), "y_train": population["y"][:4].copy(),
                  "years_train": np.asarray([years[0] - 2] * 2 + [years[0] - 1] * 2),
                  "X_test": population["X"].copy(), "y_test": population["y"].copy(),
                  "years_test": population["year"].copy(), "patient_ids": population["patient_id"].copy(),
                  "record_keys": population["record_keys"].copy()}
        values.update(changes)
        return _PreparedData(**values)

    path = work / "prepared.pkl"
    payload = pickle.dumps(prepared())
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    loaded = load_pilot_population(path, digest)
    assert loaded["X"].dtype == np.float32 and loaded["X"].shape == (120, 6)
    assert loaded["y"].dtype == np.int64 and set(loaded["y"].tolist()) == {0, 1}
    assert loaded["year"].dtype == np.int64 and loaded["patient_id"].dtype.kind == "U"
    for name in ("X", "y", "year", "patient_id", "record_keys"):
        assert np.array_equal(loaded[name], population[name]), name
    assert loaded["feature_names"] == tuple(population["feature_names"])
    assert loaded["feature_selection_max_year"] == 2006 and loaded["post_death"].dtype == bool
    fingerprint = loaded["fingerprint"]
    assert fingerprint["file_sha256"] == digest and fingerprint["file_bytes"] == len(payload)
    assert fingerprint["rows_per_year"] == {"2007": 40, "2008": 40, "2009": 40}
    assert fingerprint["rows"] == 120 and fingerprint["patients"] == 53 and fingerprint["features"] == 6
    assert fingerprint["years"] == years and fingerprint["source_x_dtype"] == "float32"
    assert fingerprint["population_identity"] == population["fingerprint"]["population_identity"]
    assert fingerprint["row_identity"] == population["fingerprint"]["row_identity"]

    sentinel = work / "unpickled.flag"

    class Probe:
        def __reduce__(self):
            return Path.touch, (sentinel,)

    evil = work / "evil.pkl"
    evil.write_bytes(pickle.dumps(Probe()))
    pickle.loads(evil.read_bytes())
    assert sentinel.exists(), "The probe must prove that unpickling would execute code"
    sentinel.unlink()
    for wrong in ("0" * 64, hashlib.sha256(b"other").hexdigest()):
        try:
            load_pilot_population(evil, wrong)
        except PopulationFingerprintError:
            pass
        else:
            raise AssertionError("Fingerprint mismatch accepted")
        assert not sentinel.exists(), "File content was unpickled before the fingerprint was verified"
    tampered = bytearray(payload)
    tampered[len(tampered) // 2] ^= 1
    (work / "tampered.pkl").write_bytes(bytes(tampered))
    try:
        load_pilot_population(work / "tampered.pkl", digest)
    except PopulationFingerprintError:
        pass
    else:
        raise AssertionError("Tampered population accepted")
    for malformed in ("", "ABC", digest.upper(), 7, None):
        try:
            load_pilot_population(path, malformed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Malformed expected sha256 accepted: {malformed!r}")
    try:
        load_pilot_population(work / "absent.pkl", digest)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing population file accepted")
    other = pickle.dumps({"not": "a population"})
    (work / "other.pkl").write_bytes(other)
    try:
        load_pilot_population(work / "other.pkl", hashlib.sha256(other).hexdigest())
    except TypeError:
        pass
    else:
        raise AssertionError("Unexpected pickled object accepted")

    def refused(**changes):
        try:
            population_from_prepared(prepared(**changes))
        except ValueError:
            return
        raise AssertionError(f"Invalid population accepted: {sorted(changes)}")

    duplicate_ids, duplicate_years = population["patient_id"].copy(), population["year"].copy()
    duplicate_ids[1], duplicate_years[1] = duplicate_ids[0], duplicate_years[0]
    refused(patient_ids=duplicate_ids, years_test=duplicate_years)
    nonbinary = population["y"].copy()
    nonbinary[0] = 2
    refused(y_test=nonbinary)
    missing = population["patient_id"].astype(object)
    missing[3] = "nan"
    refused(patient_ids=missing)
    unique, inverse = np.unique(population["patient_id"], return_inverse=True)
    numeric = population_from_prepared(prepared(patient_ids=inverse.astype(np.int64)))
    assert numeric["patient_id"].dtype.kind == "U" and numeric["patient_id"].tolist() == [str(i) for i in inverse]


def _role_checks(population):
    import numpy as np
    year, patient, y = population["year"], population["patient_id"], population["y"]
    years = population["fingerprint"]["years"]
    all_roles = {}
    for r in years:
        roles = reference_year_roles(population, r, 42)
        all_roles[r] = roles
        assert check_role_coverage(population, roles) is True
        at_r = year == r
        members = set(patient[at_r].tolist())
        direct = reference_patient_roles(patient[at_r].tolist(), y[at_r], 42)
        assert roles["reference_roles"] == direct and roles["experiment_id"] == r and roles["split_seed"] == 42
        assert set(roles["reference_roles"]) == {"tabpfn_context", "sae_discovery", "rule_discovery",
                                                 "cav_fit", "concept_selection", "evaluation"}
        role_of = {member: name for name, values in direct.items() for member in values}
        for index in range(len(year)):
            member, t = patient[index], int(year[index])
            if t < r:
                expected, entrant = "", False
            elif member in members:
                expected, entrant = role_of[member], False
            else:
                expected, entrant = stable_entrant_role(member, r, 42), True
            assert roles["row_role"][index] == expected and bool(roles["is_entrant"][index]) == entrant, (r, index)
        assert int(roles["is_entrant"].sum()) == {2007: 20, 2008: 9, 2009: 0}[r]
        assert roles["is_entrant"][year < r].sum() == 0 and (roles["row_role"][year < r] == "").all()

        changed = reference_year_roles(_population_with(population, y=np.where(year != r, 1 - y, y)), r, 42)
        assert np.array_equal(changed["row_role"], roles["row_role"])
        assert np.array_equal(changed["is_entrant"], roles["is_entrant"])
        assert changed["reference_roles"] == roles["reference_roles"] and changed["identity"] == roles["identity"]
        order = np.random.default_rng(5).permutation(len(year))
        shuffled = reference_year_roles(_population_with(population, rows=order), r, 42)
        assert np.array_equal(shuffled["row_role"], roles["row_role"][order])
        assert np.array_equal(shuffled["is_entrant"], roles["is_entrant"][order])

    returners = [f"syn-{i:04d}" for i in (4, 14, 24, 34)]
    for member in returners:
        present = {t: bool(((patient == member) & (year == t)).any()) for t in years}
        assert present == {2007: True, 2008: False, 2009: True}, member
        early = (patient == member) & (year == 2007)
        late = (patient == member) & (year == 2009)
        first = all_roles[2007]
        assert not first["is_entrant"][late].any() and first["row_role"][late][0] == first["row_role"][early][0]
        assert first["row_role"][late][0] in first["reference_roles"] and member in first["reference_roles"][
            first["row_role"][late][0]]
        second = all_roles[2008]
        assert second["is_entrant"][late].all() and second["row_role"][late][0] == stable_entrant_role(member, 2008, 42)
        assert not second["is_entrant"][early].any() and second["row_role"][early][0] == ""
    unique = np.unique(patient).tolist()
    assert any(stable_entrant_role(member, 2007, 42) != stable_entrant_role(member, 2008, 42) for member in unique)
    for bad in (1999, 2010):
        try:
            reference_year_roles(population, bad, 42)
        except ValueError:
            pass
        else:
            raise AssertionError("Reference year without rows accepted")
    rows, names, flags = year_roles(population, all_roles[2008], 2009)
    assert rows.tolist() == np.flatnonzero(year == 2009).tolist() and len(names) == len(flags) == len(rows)
    assert flags.sum() == 9 and set(names[~flags]) <= set(REFERENCE_ROLE_NAMES) and set(names[flags]) <= set(ENTRANT_ROLE_NAMES)
    try:
        year_roles(population, all_roles[2008], 2007)
    except ValueError:
        pass
    else:
        raise AssertionError("Roles of a year before the reference year were returned")
    broken = dict(all_roles[2008], row_role=all_roles[2008]["row_role"].copy())
    broken["row_role"][np.flatnonzero(year == 2007)[0]] = "evaluation"
    try:
        check_role_coverage(population, broken)
    except ValueError:
        pass
    else:
        raise AssertionError("A role on a pre-reference row passed the coverage check")
    return all_roles


def _role_support_checks(population, all_roles, work):
    import csv
    import numpy as np
    planted = _build_population(
        X=np.zeros((13, 1)), y=[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0], year=[2007] * 4 + [2008] * 4 + [2009] * 5,
        patient_id=list("abcd") + list("acde") + list("abefc"), record_keys=[f"k{i}" for i in range(13)],
        feature_names=["f"], feature_selection_max_year=2006)
    assert planted["post_death"].tolist() == [False] * 9 + [True, False, False, True]
    reference = {"tabpfn_context": ["a"], "sae_discovery": [], "rule_discovery": ["c"], "cav_fit": [],
                 "concept_selection": ["e"], "evaluation": ["d"]}
    assigned = {"a": "tabpfn_context", "c": "rule_discovery", "e": "concept_selection", "d": "evaluation",
                "b": "fitting", "f": "evaluation"}
    members = list(planted["patient_id"])
    stamps = list(planted["year"])
    planted_roles = {"reference_year": 2008, "experiment_id": 2008, "split_seed": 42, "reference_roles": reference,
                     "row_role": np.asarray([assigned[p] if t >= 2008 else "" for p, t in zip(members, stamps)]),
                     "is_entrant": np.asarray([t >= 2008 and p in ("b", "f") for p, t in zip(members, stamps)]),
                     "identity": "planted"}
    assert check_role_coverage(planted, planted_roles) is True
    empty = (0, 0, 0, 0, 0)
    withheld = (None, None, None)
    expected = {
        (2008, "reference", "tabpfn_context"): (1, 1, 0, 1, 0), (2008, "reference", "sae_discovery"): empty,
        (2008, "reference", "rule_discovery"): (1, 1, 1, 0, 0), (2008, "reference", "cav_fit"): empty,
        (2008, "reference", "concept_selection"): (1, 1, 0, 1, 0), (2008, "reference", "evaluation"): (1, 1, 0, 1, 0),
        (2008, "reference", "all"): (4, 4, 1, 3, 0), (2008, "all", "all"): (4, 4, 1, 3, 0),
        (2009, "reference", "tabpfn_context"): (1, 1, 0, 1, 0), (2009, "reference", "sae_discovery"): empty,
        (2009, "reference", "rule_discovery"): (1, 1, 0, 1, 1), (2009, "reference", "cav_fit"): empty,
        (2009, "reference", "concept_selection"): (1, 1) + withheld, (2009, "reference", "evaluation"): (0, 0) + withheld,
        (2009, "reference", "all"): (3, 3) + withheld, (2009, "entrant", "fitting"): (1, 1, 0, 1, 1),
        (2009, "entrant", "concept_selection"): (0, 0) + withheld, (2009, "entrant", "evaluation"): (1, 1) + withheld,
        (2009, "entrant", "all"): (2, 2) + withheld, (2009, "all", "all"): (5, 5) + withheld}
    opened = dict(expected)
    opened.update({
        (2009, "reference", "concept_selection"): (1, 1, 1, 0, 0), (2009, "reference", "evaluation"): empty,
        (2009, "reference", "all"): (3, 3, 1, 2, 1), (2009, "entrant", "concept_selection"): empty,
        (2009, "entrant", "evaluation"): (1, 1, 1, 0, 0), (2009, "entrant", "all"): (2, 2, 1, 1, 1),
        (2009, "all", "all"): (5, 5, 2, 3, 2)})

    def keyed(table):
        assert len({(row["year"], row["population"], row["role"]) for row in table}) == len(table) == 20
        assert {row["reference_year"] for row in table} == {2008}
        return {(row["year"], row["population"], row["role"]): (
            row["rows"], row["patients"], row["deaths"], row["survivors"], row["post_death_rows"]) for row in table}

    default = role_support_table(planted, planted_roles)
    assert keyed(default) == expected
    assert [row["outcomes"] for row in default if row["year"] == 2008] == ["reported"] * 8
    assert {row["outcomes"] for row in default if (row["year"], row["population"], row["role"]) in (
        (2009, "reference", "evaluation"), (2009, "all", "all"))} == {"withheld_holdout"}
    assert keyed(role_support_table(planted, planted_roles, holdout_outcomes=True)) == opened
    assert sum(row["rows"] for row in default if (row["population"], row["role"]) == ("all", "all")) == 9

    years = population["fingerprint"]["years"]
    year, patient, y = population["year"], population["patient_id"], population["y"]
    first_death = {}
    for member, t, died in zip(patient.tolist(), year.tolist(), y.tolist()):
        if died:
            first_death[member] = min(first_death.get(member, 10 ** 9), t)
    for r, roles in all_roles.items():
        counts = {}
        for member, t, died, role, entrant in zip(patient.tolist(), year.tolist(), y.tolist(),
                                                  roles["row_role"].tolist(), roles["is_entrant"].tolist()):
            if t < r:
                continue
            kind = "entrant" if entrant else "reference"
            for key in ((t, kind, role), (t, kind, "all"), (t, "all", "all")):
                cell = counts.setdefault(key, {"rows": 0, "patients": set(), "deaths": 0, "post": 0})
                cell["rows"] += 1
                cell["patients"].add(member)
                cell["deaths"] += died
                cell["post"] += t > first_death.get(member, 10 ** 9)
        follow = [t for t in years if t > r]
        full = role_support_table(population, roles, holdout_outcomes=True)
        assert len(full) == 8 + 12 * len(follow)
        for row in full:
            cell = counts.get((row["year"], row["population"], row["role"]),
                              {"rows": 0, "patients": set(), "deaths": 0, "post": 0})
            assert (row["rows"], row["patients"], row["deaths"], row["survivors"], row["post_death_rows"]) == (
                cell["rows"], len(cell["patients"]), cell["deaths"], cell["rows"] - cell["deaths"], cell["post"]), (r, row)
        sealed = role_support_table(population, roles)
        for row, open_row in zip(sealed, full):
            hidden = row["year"] > r and (row["role"] in HELDOUT_ROLE_NAMES or row["role"] == "all")
            assert row["rows"] == open_row["rows"] and row["patients"] == open_row["patients"]
            if hidden:
                assert (row["deaths"], row["survivors"], row["post_death_rows"]) == (None, None, None)
                assert row["outcomes"] == "withheld_holdout"
            else:
                assert row == open_row
        totals = {row["year"]: row["rows"] for row in sealed if (row["population"], row["role"]) == ("all", "all")}
        assert totals == {t: int((year == t).sum()) for t in years if t >= r}
    path = write_role_support_csv(role_support_table(population, all_roles[2007]), work / "tables" / "role-support.csv")
    text = path.read_text()
    assert "syn-" not in text
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == list(ROLE_SUPPORT_COLUMNS) and len(rows) == 8 + 12 * 2
    assert [row for row in rows if row["year"] == "2009" and row["role"] == "evaluation"][0]["deaths"] == ""


def _predictor_unit_checks(population, presets, all_roles, work):
    import numpy as np
    import torch
    config = resolve_config(presets, "v0")
    years = population["fingerprint"]["years"]
    assert config["reference_year_candidates"] == years and config["t0_only_reference_years"] == [2009]
    year, last = population["year"], max(years)
    systems, fits, sizes = {}, [], []

    def recording_fit(frame, names, device, seed=42):
        fits.append(len(frame))
        system = fit_frozen_predictor(frame, names, device, seed=seed)
        systems[system["reference_year"]] = system
        return system

    def counting_capture(system, frame, device):
        sizes.append(len(frame))
        return capture_frozen_predictions(system, frame, device)

    def reuse(r):
        return lambda frame, names, device, seed=42: systems[r]

    free = lambda: 100 * 2 ** 30
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, prior_threads))
    try:
        cache = UnitCache(work / "cache", config, "source-fixture", free_bytes=free)
        outputs, records = {}, {}
        for r in years:
            roles = all_roles[r]
            eligible = np.flatnonzero(year >= r)
            before = len(sizes)
            record = records[r] = cache.run(f"P/{r}", lambda r=r, roles=roles: predictor_unit(
                population, r, roles, config, "cpu", fit=recording_fit, capture=counting_capture),
                projected_bytes=projected_predictor_bytes(population, r))
            assert record["status"] == "ok" and record["rows"] == len(eligible) and record["dependencies"] == {}, record
            out = outputs[r] = cache.load(f"P/{r}")
            assert out["row_index"].dtype == np.int64 and np.array_equal(out["row_index"], eligible)
            assert year[out["row_index"]].min() == r and set(year[out["row_index"]].tolist()) == set(range(r, last + 1))
            assert out["probabilities"].dtype == np.float32 and out["probabilities"].shape == (len(eligible), 2)
            assert out["embeddings"].dtype == np.float32 and out["embeddings"].shape == (len(eligible), out["embedding_dim"])
            assert np.isfinite(out["probabilities"]).all() and np.isfinite(out["embeddings"]).all()
            assert ((out["probabilities"] >= 0) & (out["probabilities"] <= 1)).all()
            np.testing.assert_allclose(out["probabilities"].sum(axis=1), 1.0, atol=1e-4)
            assert out["reference_year"] == r and out["last_year"] == last and out["feature_count"] == 6
            assert out["rows_per_year"] == {str(t): 40 for t in range(r, last + 1)}
            context = roles["reference_roles"]["tabpfn_context"]
            at_context = (year == r) & np.isin(population["patient_id"], context)
            assert out["context_rows"] == len(context) == int(at_context.sum())
            assert out["context_deaths"] == int(population["y"][at_context].sum()) and out["context_deaths"] > 0
            assert out["context_rows_identity"] == _identity_hash(list(zip(
                population["patient_id"][at_context].tolist(), year[at_context].tolist())))
            start = min(config["inference_batch_rows"], len(eligible))
            assert out["start_batch_rows"] == out["batch_rows"] == start and out["oom_backoffs"] == []
            assert out["inference_chunks"] == -(-len(eligible) // start)
            batches = sizes[before + 3 + PREDICTOR_GATE_ROWS:]
            assert sum(batches) == len(eligible) and max(batches) <= start, "Eligible rows must be captured in configured batches"
            assert set(out["gates"]) == {f"{gate}.{key}" for gate in (
                "repeat_identity", "order_stability", "batch_stability", "full_pass_stability")
                for key in ("probabilities", "embeddings")}
            assert all(0 <= value <= 1e-2 for value in out["gates"].values()), out["gates"]
            assert out["finite_difference"]["passed"] is True and out["finite_difference"]["step"] == 1e-2
            assert 0 <= out["parity_max_absolute_error"] <= 2e-4 and isinstance(out["nonlinear_decoder"], bool)
            assert out["settings"]["model_type"] == "single_fast" and out["settings"]["fit_at_predict_time"] is False
            assert out["device"] == "cpu" and out["roles_identity"] == roles["identity"]
            assert out["population_identity"] == population["fingerprint"]["population_identity"]
            assert len(out["context_identity"]) == 64 and out["identity_report"]["identity"] == out["context_identity"]
            _check_predictor_gradients(out, cache, r)
        assert fits == [len(all_roles[r]["reference_roles"]["tabpfn_context"]) for r in years], "One context fit per year"
        fitted = len(fits)
        again = cache.run("P/2007", lambda: predictor_unit(population, 2007, all_roles[2007], config, "cpu",
                                                              fit=recording_fit, capture=counting_capture))
        assert again["id"] == records[2007]["id"] and len(fits) == fitted, "Predictor was recomputed"
        assert [row["status"] for row in cache.completeness_table(["P/2007", "P/2008", "P/2009"])] == ["complete"] * 3

        poisoned = population["X"].copy()
        poisoned[year < 2008] = np.nan
        twin_population = _population_with(population, X=poisoned, y=np.where(year > 2008, 1 - population["y"], population["y"]))
        twin_roles = reference_year_roles(twin_population, 2008, 42)
        assert twin_roles["identity"] == all_roles[2008]["identity"]
        twin = predictor_unit(twin_population, 2008, twin_roles, config, "cpu")
        base = outputs[2008]
        assert np.array_equal(twin["row_index"], base["row_index"]) and twin["context_rows_identity"] == base["context_rows_identity"]
        np.testing.assert_allclose(twin["probabilities"], base["probabilities"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(twin["embeddings"], base["embeddings"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(twin["gradients"], base["gradients"], rtol=1e-3, atol=1e-5)
        assert twin["continuation_identity"] == base["continuation_identity"]

        limit = 30
        wide = dict(config, inference_batch_rows=120)
        refused = []

        def limited_capture(system, frame, device):
            if len(frame) > limit:
                refused.append(len(frame))
                raise torch.cuda.OutOfMemoryError("fixture out of memory")
            return capture_frozen_predictions(system, frame, device)

        small = predictor_unit(population, 2007, all_roles[2007], wide, "cpu", fit=reuse(2007), capture=limited_capture)
        assert small["start_batch_rows"] == 120 and refused == [120, 60] and small["oom_backoffs"] == [{"rows": 120, "next_rows": 60}, {"rows": 60, "next_rows": 30}]
        assert small["batch_rows"] == 30 and small["inference_chunks"] == 4 and small["_rows"] == 120
        np.testing.assert_allclose(small["probabilities"], outputs[2007]["probabilities"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(small["embeddings"], outputs[2007]["embeddings"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(small["gradients"], outputs[2007]["gradients"], rtol=1e-3, atol=1e-5)

        class WindowStop(Exception):
            pass

        frames = []

        def window_fit(frame, names, device, seed=42, **options):
            assert options == {"domain_origin": 2007}, options
            frames.append(frame)
            raise WindowStop

        try:
            predictor_unit(population, 2009, all_roles[2009], dict(config, context_window="all_history"), "cpu", fit=window_fit)
        except WindowStop:
            pass
        frame, = frames
        current = (year == 2009) & (all_roles[2009]["row_role"] == "tabpfn_context") & ~all_roles[2009]["is_entrant"]
        expected = current | ((year < 2009) & ~population["post_death"])
        assert population["post_death"][year < 2009].any(), "Fixture needs an earlier post-death row"
        assert sorted(zip(frame.patient_id, frame.year.tolist())) == sorted(zip(population["patient_id"][expected], year[expected].tolist()))
        windowed = fit_frozen_predictor(frame, _feature_columns(population["X"].shape[1]), "cpu", seed=42, domain_origin=2007)
        assert (windowed["reference_year"], windowed["domain_origin"]) == (2009, 2007)
        assert windowed["settings"]["domain_policy"] == "calendar_year_minus_first_context_year"
        later = _prediction_frame(population, np.flatnonzero(year == 2009)[:4], _feature_columns(population["X"].shape[1]))
        predicted = capture_frozen_predictions(windowed, later, "cpu")["probabilities"]
        assert np.isfinite(predicted.detach().cpu().numpy() if hasattr(predicted, "detach") else np.asarray(predicted)).all()
        for overrides in ({"context_window": "reference_only"}, {"context_window": "last_5"}):
            try:
                resolve_config(presets, "pilot", overrides)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid context window accepted: {overrides}")
        assert resolve_config(presets, "pilot", {"context_window": "last_3"})["config_hash"] != resolve_config(presets, "pilot")["config_hash"]
        devices = []

        def device_fit(frame, names, device, seed=42, **options):
            devices.append(device)
            raise WindowStop

        try:
            predictor_unit(population, 2009, all_roles[2009], dict(config, predictor_cpu_from=2009), "cuda", fit=device_fit)
        except WindowStop:
            pass
        assert devices == ["cpu"], devices
        routed = resolve_config(presets, "pilot", configuration_overrides({"PILOT_PREDICTOR_CPU_FROM": "2012"}))
        assert routed["predictor_cpu_from"] == 2012 and routed["config_hash"] != resolve_config(presets, "pilot")["config_hash"]
        late = resolve_config(presets, "pilot", configuration_overrides({"PILOT_REFERENCE_START": "2014"}))
        assert late["reference_year_candidates"] == [2014, 2015] and late["t0_only_reference_years"] == [2015]
        for environment in ({"PILOT_REFERENCE_START": "2006"}, {"PILOT_REFERENCE_START": "x"}):
            try:
                resolve_config(presets, "pilot", configuration_overrides(environment))
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid reference start accepted: {environment}")
        entered = []

        def exhausted_capture(system, frame, device):
            if len(frame) == 120:
                entered.append(True)
            if entered:
                raise torch.cuda.OutOfMemoryError("fixture out of memory")
            return capture_frozen_predictions(system, frame, device)

        try:
            predictor_unit(population, 2007, all_roles[2007], wide, "cpu", fit=reuse(2007), capture=exhausted_capture)
        except torch.cuda.OutOfMemoryError:
            pass
        else:
            raise AssertionError("Persistent out-of-memory must fail the unit once the batch cannot shrink")

        calls = []

        def flaky_capture(system, frame, device):
            result = capture_frozen_predictions(system, frame, device)
            calls.append(len(frame))
            return dict(result, embeddings=result["embeddings"] + 1.0) if len(calls) == 2 else result

        failing = UnitCache(work / "cache-gate-failure", config, "source-fixture", free_bytes=free)
        record = failing.run("P/2008", lambda: predictor_unit(population, 2008, all_roles[2008], config, "cpu",
                                                              fit=reuse(2008), capture=flaky_capture))
        assert record["status"] == "failed" and record["reason"].startswith("PredictorGateFailure: repeat_identity."), record
        assert not failing.is_complete("P/2008")
        executed = []
        for key in ("S/42/2008", "S/215/2008", "D/2008/families", "D/2008/reference_performance"):
            dependent = failing.run(key, lambda key=key: executed.append(key))
            assert (dependent["status"], dependent["reason"]) == ("unavailable", "dep_failed:P/2008"), (key, dependent)
        assert not executed, "A dependent of a failed predictor unit was executed"
        assert failing.run("P/2007", lambda: predictor_unit(population, 2007, all_roles[2007], config, "cpu",
                                                            fit=reuse(2007)))["status"] == "ok"
        table = {row["unit"]: row["status"] for row in failing.completeness_table(
            ["P/2007", "P/2008", "S/42/2008", "D/2008/families", "P/2009"])}
        assert table["P/2007"] == "complete" and table["S/42/2008"] == "unavailable:dep_failed:P/2008"
        assert table["P/2008"].startswith("failed:PredictorGateFailure: repeat_identity.")
        assert table["D/2008/families"] == "unavailable:dep_failed:P/2008" and table["P/2009"] == "not_run (stopped)"

        def broken_fit(frame, names, device, seed=42):
            raise RuntimeError("Pilot requires exactly one frozen executed configuration/model")

        try:
            predictor_unit(population, 2008, all_roles[2008], config, "cpu", fit=broken_fit)
        except PredictorGateFailure as error:
            assert error.gate == "fit"
        else:
            raise AssertionError("A rejected predictor configuration was accepted")
        for bad_device in ("auto", "cuda:0"):
            try:
                predictor_unit(population, 2008, all_roles[2008], config, bad_device, fit=reuse(2008))
            except ValueError:
                pass
            else:
                raise AssertionError("Unresolved device accepted")
        try:
            predictor_unit(population, 2007, all_roles[2008], config, "cpu", fit=reuse(2007))
        except ValueError:
            pass
        else:
            raise AssertionError("Roles of another reference year accepted")
        assert projected_predictor_bytes(population, 2008) == 80 * 1026 * 4 + projected_continuation_bytes(512)
        for r in years:
            used = sum(path.stat().st_size for path in cache._unit_path(f"P/{r}").rglob("*") if path.is_file())
            assert 0 < used <= projected_predictor_bytes(population, r), (r, used, projected_predictor_bytes(population, r))
    finally:
        torch.set_num_threads(prior_threads)


def run_population_role_checks(notebook_path="concept_subgroup_drift_pilot.ipynb"):
    """Population fingerprint, per-year roles and entrants, support audit and the P(r) unit on tiny CPU fixtures."""
    notebook = json.loads(Path(notebook_path).read_text())
    presets = _notebook_presets(notebook)
    _population_configuration_checks(presets, notebook)
    base = Path(os.environ["RPI_OUT_DIR"])
    base.mkdir(parents=True, exist_ok=True)
    population = synthetic_population()
    with tempfile.TemporaryDirectory(dir=base) as directory:
        work = Path(directory)
        _population_load_checks(population, work)
        all_roles = _role_checks(population)
        _role_support_checks(population, all_roles, work)
        _predictor_unit_checks(population, presets, all_roles, work)
    print("Population fingerprint, per-year roles, entrants, support audit and P(r) predictor unit fixtures passed")


def strict_activation_mask(activations, target):
    import numpy as np
    values = np.asarray(activations, dtype=float)
    if values.ndim != 1:
        raise ValueError("Activation membership requires one factor aligned with rows")
    if not target["valid"]:
        return np.zeros(len(values), dtype=bool)
    return np.isfinite(values) & (values > target["cutoff"])


def ordered_reference_candidates(activations, cap):
    import numpy as np
    values = np.asarray(activations)
    if values.ndim != 2 or type(cap) is not int or cap < 0:
        raise ValueError("Candidate cap must be nonnegative and codes two-dimensional")
    support = np.sum(np.isfinite(values) & (values > 0), axis=0)
    return sorted(range(values.shape[1]), key=lambda factor: (-int(support[factor]), factor))[:cap]


def frozen_subgroup_selection(gradients, cavs, masks, factor_identity):
    import numpy as np
    raw = np.asarray(gradients, dtype=float)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("Selection requires finite raw mortality-probability gradients")
    result = {}
    for subgroup, mask in masks.items():
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != (len(raw),):
            raise ValueError("Subgroup selection mask is not aligned")
        diagnostics = {}
        for seed in (42, 43, 44):
            cav = cavs.get(seed)
            score, reason = None, None
            if not selected.any():
                reason = "empty_selection_subgroup"
            elif not cav or not cav["valid"]:
                reason = cav.get("failure_reason", "unavailable_cav") if cav else "missing_cav_seed"
            else:
                direction = np.asarray(cav["cav"], dtype=float)
                if direction.shape != (raw.shape[1],) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
                    reason = "invalid_cav_direction"
                else:
                    score = float(np.mean(raw[selected] @ direction > 0))
            diagnostics[seed] = {"tcav": score, "unavailable_reason": reason}
        primary = diagnostics[42]["tcav"]
        result[subgroup] = {"factor_identity": str(factor_identity), "subgroup": subgroup,
            "primary_seed": 42, "primary_tcav": primary,
            "eligible": primary is not None and (primary <= .1 or primary >= .9),
            "selection_records": int(selected.sum()), "diagnostics": diagnostics,
            "unavailable_reason": diagnostics[42]["unavailable_reason"]}
    return result


def fit_pilot_cavs(embeddings, features, activations, rule, patient_ids, seeds=(42, 43, 44)):
    import numpy as np
    from temporal_cav import train_temporal_cav
    raw, X = np.asarray(embeddings, dtype=float), np.asarray(features)
    patients = np.asarray(_canonical_patient_ids(patient_ids))
    if raw.ndim != 2 or len(raw) != len(X) or len(raw) != len(patients):
        raise ValueError("CAV features/embeddings/patients must align")
    if not np.isfinite(raw).all():
        return {seed: {"valid": False, "cav": None, "failure_reason": "nonfinite_cav_embeddings"} for seed in seeds}
    result = {}
    for seed in seeds:
        value = train_temporal_cav(embeddings=raw, features=X, activations=activations,
                rule=rule, activation_target=.10, rule_source="semantic", patient_ids=patients,
                minimum_positive=2, minimum_negative=2, seed=seed)
        if value["valid"]:
            positive = len(np.unique(patients[value["positive_indices"]]))
            negative = len(np.unique(patients[value["negative_indices"]]))
            value.update(positive_patients=positive, negative_patients=negative)
            direction = np.asarray(value["cav"])
            reason = "insufficient_distinct_cav_patients" if min(positive, negative) < 2 else None
            if not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
                reason = "invalid_cav_direction"
            if reason:
                value.update(valid=False, failure_reason=reason, cav=None)
        result[int(seed)] = value
    return result


def fit_constant_baseline(fitting_labels):
    import numpy as np
    labels = np.asarray(fitting_labels)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Baseline fitting labels must be binary and one-dimensional")
    return {"q": float(labels.mean()) if len(labels) else None, "fitting_records": len(labels),
            "unavailable_reason": None if len(labels) else "empty_fitting_subgroup",
            "identity": _identity_hash(labels)}


def subgroup_reliability(labels, probabilities, patient_ids, baseline, bootstrap_replicates=100, seed=42):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    # pragmatic: compact annual subgroup metrics/bootstrap use CPU; revisit for million-row cohorts.
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    patients = np.asarray(_canonical_patient_ids(patient_ids))
    if y.ndim != 1 or p.shape != y.shape or len(patients) != len(y):
        raise ValueError("Reliability labels/probabilities/patients must align")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Reliability requires binary labels and finite probabilities in [0,1]")
    q = baseline["q"]
    if q is not None and (not np.isfinite(q) or not 0 <= q <= 1):
        raise ValueError("Frozen baseline probability must lie in [0,1]")
    names = ("brier", "baseline_brier", "brier_skill", "mean_risk", "prevalence",
             "auroc", "f1", "sensitivity", "specificity", "false_positive_rate", "ece")
    result = {name: None for name in names}
    result.update(records=len(y), patients=len(np.unique(patients)), classification_threshold=.5,
                  baseline_q=q, baseline_identity=baseline["identity"], unavailable_reasons={}, calibration_bins=[])
    if not len(y):
        result["unavailable_reasons"] = dict.fromkeys(names, "empty_subgroup")
    else:
        loss = (p - y) ** 2
        result.update(brier=float(loss.mean()), mean_risk=float(p.mean()), prevalence=float(y.mean()))
        if q is None:
            result["unavailable_reasons"].update(baseline_brier="empty_fitting_subgroup", brier_skill="empty_fitting_subgroup")
        else:
            baseline_loss = (q - y) ** 2
            result["baseline_brier"] = float(baseline_loss.mean())
            if result["baseline_brier"]:
                result["brier_skill"] = 1 - result["brier"] / result["baseline_brier"]
            else:
                result["unavailable_reasons"]["brier_skill"] = "zero_baseline_loss"
        predicted = p >= .5
        tp, fn = int(np.sum(predicted & (y == 1))), int(np.sum(~predicted & (y == 1)))
        tn, fp = int(np.sum(~predicted & (y == 0))), int(np.sum(predicted & (y == 0)))
        result.update(true_positive=tp, false_negative=fn, true_negative=tn, false_positive=fp)
        for metric, numerator, denominator in (("f1", 2 * tp, 2 * tp + fp + fn),
                    ("sensitivity", tp, tp + fn), ("specificity", tn, tn + fp), ("false_positive_rate", fp, tn + fp)):
            if denominator:
                result[metric] = numerator / denominator
            else:
                result["unavailable_reasons"][metric] = "zero_metric_denominator"
        if len(np.unique(y)) == 2:
            result["auroc"] = float(roc_auc_score(y, p))
        else:
            result["unavailable_reasons"]["auroc"] = "single_outcome_class"
        bins = np.minimum((p * 5).astype(int), 4)
        ece = 0.0
        for index in range(5):
            mask = bins == index
            row = {"bin": index, "records": int(mask.sum()), "mean_risk": None, "prevalence": None}
            if mask.any():
                row.update(mean_risk=float(p[mask].mean()), prevalence=float(y[mask].mean()))
                ece += mask.mean() * abs(row["mean_risk"] - row["prevalence"])
            result["calibration_bins"].append(row)
        result["ece"] = float(ece)
    result["bootstrap"] = _paired_patient_bootstrap(y, p, patients, q, bootstrap_replicates, seed)
    return result


def freeze_reference_objects(target, rule, cavs):
    result = {"target": deepcopy(target), "rule": deepcopy(rule), "cavs": deepcopy(cavs)}
    result["identity"] = _reference_object_identity(result)
    return result


def assert_reference_identity(reference):
    if reference["identity"] != _reference_object_identity(reference):
        raise RuntimeError("Frozen reference target/rule/CAV identity changed")


def _reference_object_identity(reference):
    return _identity_hash((reference["target"], reference["rule"].to_dict(), reference["cavs"]))


def _paired_patient_bootstrap(y, p, patients, q, replicates, seed):
    import numpy as np
    if type(replicates) is not int or replicates < 1:
        raise ValueError("Bootstrap replicates must be a positive integer")
    unique, inverse = np.unique(patients, return_inverse=True)
    report = {"replicates": replicates, "confidence_level": .95, "sampling_unit": "patient",
              "paired_model_baseline": True, "baseline_refitted": False, "seed": seed,
              "conditional_on": "fixed reference objects and fitting-derived q", "intervals": {},
              "valid_replicates": {}, "unavailable_reasons": {}}
    if len(unique) < 2:
        report["unavailable_reasons"] = dict.fromkeys(("brier", "baseline_brier", "brier_skill"), "fewer_than_two_patients")
        return report
    losses = (p - y) ** 2
    baseline_losses = (q - y) ** 2 if q is not None else None
    samples = {"brier": [], "baseline_brier": [], "brier_skill": []}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        multiplicity = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))[inverse]
        model_loss = float(np.sum(losses * multiplicity) / multiplicity.sum())
        samples["brier"].append(model_loss)
        if baseline_losses is not None:
            baseline_loss = float(np.sum(baseline_losses * multiplicity) / multiplicity.sum())
            samples["baseline_brier"].append(baseline_loss)
            if baseline_loss:
                samples["brier_skill"].append(1 - model_loss / baseline_loss)
    for name, values in samples.items():
        report["valid_replicates"][name] = len(values)
        report["intervals"][name] = np.quantile(values, [.025, .975]).tolist() if values else None
        if not values:
            report["unavailable_reasons"][name] = "empty_fitting_subgroup" if q is None else "zero_baseline_loss"
    return report


def run_reference_selection_checks():
    """Synthetic selection/reliability checks; mortality never gates concepts."""
    import numpy as np
    a = np.asarray([0, -2, 1, 2, 3, 4, float("nan"), float("inf")])
    target = {"valid": True, "cutoff": float(np.quantile(a[np.isfinite(a) & (a > 0)], .9))}
    assert target["cutoff"] == np.quantile([1, 2, 3, 4], .9)
    assert strict_activation_mask(a, target).tolist() == [False] * 5 + [True, False, False]
    assert strict_activation_mask([target["cutoff"], target["cutoff"] + .1], target).tolist() == [False, True]
    tied = {"valid": True, "cutoff": 1.}
    assert not strict_activation_mask([1, 1, 1], tied).any()
    assert not strict_activation_mask([0, -1], {"valid": False, "cutoff": None}).any()
    codes = np.asarray([[1, 2, 0], [0, 3, 1], [1, 0, 0]])
    assert ordered_reference_candidates(codes, 2) == [0, 1]
    direction = np.asarray([1., 0.])
    cavs = {42: {"valid": True, "cav": direction},
            43: {"valid": True, "cav": -direction},
            44: {"valid": False, "failure_reason": "fixture_unavailable", "cav": None}}
    g = np.asarray([[1., 0.]] * 9 + [[-1., 0.]] + [[-1., 0.]] * 9 + [[1., 0.]])
    masks = {"A": np.asarray([True] * 10 + [False] * 10),
             "R": np.asarray([False] * 10 + [True] * 10)}
    selection = frozen_subgroup_selection(g, cavs, masks, "fixture-factor")
    assert selection["A"]["eligible"] and selection["A"]["primary_tcav"] == .9
    assert selection["R"]["eligible"] and selection["R"]["primary_tcav"] == .1
    assert selection["A"]["diagnostics"][43]["tcav"] == .1
    changed = dict(masks, R=np.ones(20, dtype=bool))
    second = frozen_subgroup_selection(g, cavs, changed, "fixture-factor")
    assert second["A"]["eligible"] and not second["R"]["eligible"]
    assert second["R"]["primary_tcav"] == .5
    assert not frozen_subgroup_selection(g, cavs, {"A": np.zeros(20, bool)}, "fixture-factor")["A"]["eligible"]
    class Rule:
        def mask(self, features):
            return np.asarray(features)[:, 0] > .5
        def to_dict(self):
            return {"fixture_feature": 0, "cutoff": .5}
    X = np.asarray([[1], [1], [0], [0], [1], [0]], dtype=float)
    raw = np.asarray([[2, 0], [2, 1], [-2, 0], [-2, 1], [3, 1], [-3, 1]], dtype=float)
    activation = np.asarray([1, 1, 0, 0, 2, 0], dtype=float)
    fitted = fit_pilot_cavs(raw, X, activation, Rule(), ["a", "b", "c", "d", "e", "f"])
    assert all(fitted[seed]["valid"] for seed in (42, 43, 44))
    assert all(fitted[seed]["positive_patients"] >= 2 and fitted[seed]["negative_patients"] >= 2 for seed in fitted)
    assert all(fitted[seed]["low_activation_cutoff"] == 0 for seed in fitted)
    insufficient = fit_pilot_cavs(raw[:4], X[:4], activation[:4], Rule(), ["a", "a", "c", "d"])
    assert all(not value["valid"] and value["failure_reason"] == "insufficient_distinct_cav_patients" for value in insufficient.values())
    frozen = freeze_reference_objects(target, Rule(), cavs)
    identity = frozen["identity"]
    target["cutoff"] = 100
    direction[:] = 99
    assert_reference_identity(frozen)
    assert frozen["identity"] == identity and frozen["target"]["cutoff"] != 100
    frozen["target"]["cutoff"] = 200
    try:
        assert_reference_identity(frozen)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Frozen reference mutation accepted")

    baseline = fit_constant_baseline([0, 1, 0, 1])
    assert baseline["q"] == .5
    result = subgroup_reliability([0, 1], [.1, .8], ["a", "b"], baseline)
    assert np.isclose(result["brier"], .025) and result["baseline_brier"] == .25
    assert np.isclose(result["brier_skill"], .9)
    assert result["bootstrap"]["replicates"] == 100 and result["bootstrap"]["confidence_level"] == .95
    single = subgroup_reliability([0, 0], [.1, .2], ["a", "b"], baseline)
    assert np.isclose(single["brier"], .025) and single["auroc"] is None
    assert single["sensitivity"] is None and single["specificity"] == 1.0
    zero = subgroup_reliability([0, 0], [0, 0], ["a", "b"], fit_constant_baseline([0, 0]))
    assert zero["baseline_brier"] == 0 and zero["brier_skill"] is None
    assert zero["unavailable_reasons"]["brier_skill"] == "zero_baseline_loss"
    empty_fit = subgroup_reliability([0, 1], [.1, .8], ["a", "b"], fit_constant_baseline([]))
    assert empty_fit["brier"] is not None and empty_fit["baseline_brier"] is None
    empty_eval = subgroup_reliability([], [], [], baseline)
    assert empty_eval["brier"] is None and empty_eval["unavailable_reasons"]["brier"] == "empty_subgroup"
    repeated = subgroup_reliability([0, 0, 1, 1], [.1, .2, .8, .7], ["a", "a", "b", "b"], baseline)
    assert repeated["patients"] == 2 and repeated["records"] == 4
    assert repeated["bootstrap"]["sampling_unit"] == "patient"
    assert repeated["bootstrap"]["paired_model_baseline"] is True
    assert repeated["bootstrap"] == subgroup_reliability([0, 0, 1, 1], [.1, .2, .8, .7], ["a", "a", "b", "b"], baseline)["bootstrap"]
    _check_tcav_selection_rules()
    print("Frozen reference selection, CAV viability and paired subgroup reliability fixtures passed")


def fit_reference_scaler(fitting_embeddings):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    raw = np.asarray(fitting_embeddings, dtype=np.float32)
    if raw.ndim != 2 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Reference scaler needs nonempty finite fitting embeddings")
    # pragmatic: sklearn scaler fits this small reference matrix on CPU.
    return StandardScaler().fit(raw)


def fit_reference_dictionary(fitting_embeddings, config, device, seed=None):
    """Reference discovery only; epochs/seeds/SAE parameters remain configured."""
    import numpy as np
    from sae_compare import train_all_saes
    if config["profile"] not in ("v0", "pilot") or config["epochs"] <= 0:
        raise ValueError("Reference SAE training belongs to v0/pilot profiles only")
    seed = config["sae_seeds"][0] if seed is None else seed
    if type(seed) is not int or seed not in config["sae_seeds"]:
        raise ValueError(f"SAE seed must be one of {config['sae_seeds']}")
    scaler = fit_reference_scaler(fitting_embeddings)
    parameters = config["sae"]
    run, = train_all_saes(1, np.asarray(scaler.transform(fitting_embeddings), dtype=np.float32),
        seeds=[seed], epochs=config["epochs"], model_type=parameters["model_type"],
        alpha=parameters["alpha"], scaling_factor=parameters["scaling_factor"],
        learning_rate=parameters["learning_rate"], weight_decay=parameters["weight_decay"],
        device=device, encoding_batch_size=config["encoding_batch_size"], show_progress=False)
    return {"scaler": scaler, "run": run,
            "scaler_identity": _identity_hash((scaler.mean_, scaler.scale_)),
            "sae_identity": _sae_state_identity(run["model"]),
            "seed": seed, "epochs": config["epochs"],
            "training_note": "full-batch SAE training; encoding batch does not bound training memory"}


def encode_reference_dictionary(dictionary, embeddings, device, batch_size=256):
    import numpy as np
    from sae_compare import encode_sae
    scaler = dictionary["scaler"]
    if _identity_hash((scaler.mean_, scaler.scale_)) != dictionary["scaler_identity"]:
        raise RuntimeError("Frozen reference scaler changed")
    if _sae_state_identity(dictionary["run"]["model"]) != dictionary["sae_identity"]:
        raise RuntimeError("Frozen reference SAE changed: " + _sae_state_difference(dictionary["run"]["model"]))
    return encode_sae(dictionary["run"], np.asarray(scaler.transform(embeddings), dtype=np.float32),
                      device=device, batch_size=batch_size)


def _sae_state_difference(model):
    """Device and per-tensor summary for an identity failure; R1 saw this only late in a long CUDA process."""
    import numpy as np
    state = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    parts = [f"{name} finite={bool(np.isfinite(array).all())} dtype={array.dtype}" for name, array in state.items()]
    return f"device={next(model.parameters()).device}; " + "; ".join(parts)


def _sae_state_identity(model):
    return _identity_hash({name: value.detach().cpu().numpy() for name, value in model.state_dict().items()})


SAE_STATE_ARRAYS = {"encoder.weight": "encoder_weight", "encoder.bias": "encoder_bias", "decoder_bias": "decoder_bias"}
PREDICTOR_OUTPUT_KEYS = ("row_index", "embeddings", "reference_year", "roles_identity", "population_identity",
                         "output_row_identity", "embedding_dim")


def projected_sae_bytes(config, embedding_bound=512):
    """Upper bound of one S unit's cached bytes: float32 SAE state plus float64 scaler vectors."""
    dimension = int(embedding_bound)
    latents = int(dimension * config["sae"]["scaling_factor"])
    return 4 * (latents * dimension + latents + dimension) + 3 * 8 * dimension + 4096


def _sae_discovery_embeddings(population, r, roles, predictor_output):
    """Year-r sae_discovery rows of the cached P(r) output, selected by role and never recomputed."""
    return _role_embeddings(population, r, roles, predictor_output, ("sae_discovery",))


def sae_unit(population, r, roles, predictor_output, config, seed, device, *, expected_scaler_identity=None, fit=None):
    """S(seed, r): reference scaler and one seeded SAE on year-r sae_discovery embeddings of the cached P(r) output."""
    try:
        return _sae_unit(population, int(r), roles, predictor_output, config, seed, device,
                         expected_scaler_identity, fit_reference_dictionary if fit is None else fit)
    finally:
        if device == "cuda":
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()


def _sae_unit(population, r, roles, predictor_output, config, seed, device, expected_scaler_identity, fit):
    import numpy as np
    import torch
    if device not in ("cpu", "cuda"):
        raise ValueError("SAE unit needs a resolved device: cpu or cuda")
    rows, discovery = _sae_discovery_embeddings(population, r, roles, predictor_output)
    started = time.monotonic()
    dictionary = fit(discovery, config, device, seed=seed)
    train_seconds = time.monotonic() - started
    scaler, model, parameters = dictionary["scaler"], dictionary["run"]["model"], config["sae"]
    if dictionary["seed"] != seed or dictionary["epochs"] != config["epochs"]:
        raise RuntimeError("Trained dictionary does not match the requested seed and epochs")
    state = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    if set(state) != set(SAE_STATE_ARRAYS):
        raise RuntimeError(f"Unexpected SAE state tensors: {sorted(state)}")
    if not all(np.isfinite(array).all() for array in state.values()):
        raise FloatingPointError("Nonfinite SAE parameters after training")
    latents, dimension = int(model.num_latents), int(discovery.shape[1])
    if (state["encoder.weight"].shape != (latents, dimension) or state["encoder.bias"].shape != (latents,)
            or state["decoder_bias"].shape != (dimension,)):
        raise RuntimeError("SAE state shapes disagree with the trained model")
    if dictionary["scaler_identity"] != _identity_hash((scaler.mean_, scaler.scale_)):
        raise RuntimeError("Reference scaler identity does not describe the fitted scaler")
    if dictionary["sae_identity"] != _sae_state_identity(model):
        raise RuntimeError("SAE identity does not describe the trained state")
    if expected_scaler_identity is not None and dictionary["scaler_identity"] != expected_scaler_identity:
        raise RuntimeError("Reference scaler differs from the one fitted for this reference year")
    scaled = torch.as_tensor(np.asarray(scaler.transform(discovery), dtype=np.float32), device=device)
    model.to(device).eval()
    with torch.inference_mode():
        codes, reconstruction = model(scaled)
        mse = torch.nn.functional.mse_loss(reconstruction, scaled)
        penalty = parameters["alpha"] * codes.abs().mean()
        active = codes > 0
        summary = torch.stack([mse, penalty, mse + penalty, active.float().mean(),
                               (~active.any(dim=0)).sum().float(), torch.isfinite(codes).all().float(),
                               torch.isfinite(reconstruction).all().float()]).cpu().tolist()
    mse_value, penalty_value, loss_value, active_fraction, dead, codes_finite, reconstruction_finite = summary
    if not all(np.isfinite([mse_value, penalty_value, loss_value])) or not (codes_finite and reconstruction_finite):
        raise FloatingPointError("Nonfinite SAE loss or outputs after training")
    peak = int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
    rows_identity = _identity_hash({"row_index": rows})
    payload = {
        "encoder_weight": state["encoder.weight"], "encoder_bias": state["encoder.bias"],
        "decoder_bias": state["decoder_bias"],
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64), "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "scaler_var": np.asarray(scaler.var_, dtype=np.float64),
        "_rows": int(len(rows)), "reference_year": int(r), "seed": int(seed), "epochs": int(config["epochs"]),
        "latent_count": latents, "embedding_dim": dimension, "decoder_tied_to_encoder": True,
        "sae": {name: parameters[name] for name in ("model_type", "scaling_factor", "alpha", "learning_rate", "weight_decay")},
        "scaler_samples": int(scaler.n_samples_seen_), "scaler_identity": dictionary["scaler_identity"],
        "sae_identity": dictionary["sae_identity"],
        "identity": _identity_hash({"reference_year": int(r), "seed": int(seed), "epochs": int(config["epochs"]),
                                    "scaler": dictionary["scaler_identity"], "sae": dictionary["sae_identity"],
                                    "sae_config": {name: parameters[name] for name in sorted(parameters)}}),
        "final_loss": loss_value, "final_reconstruction_mse": mse_value, "final_sparsity_penalty": penalty_value,
        "final_loss_basis": "trained state on the training matrix after the last optimizer step",
        "active_fraction": active_fraction, "dead_latents_on_discovery": int(dead),
        "discovery_rows": int(len(rows)), "discovery_patients": int(len(np.unique(population["patient_id"][rows]))),
        "discovery_rows_identity": rows_identity,
        "roles_identity": roles["identity"], "population_identity": predictor_output["population_identity"],
        "predictor_context_identity": predictor_output.get("context_identity"),
        "predictor_output_row_identity": predictor_output["output_row_identity"],
        "training": {"device": device, "train_seconds": train_seconds, "batch": "full", "batch_rows": int(len(rows)),
                     "input_bytes": int(discovery.nbytes), "epochs": int(config["epochs"]),
                     "learning_rate": parameters["learning_rate"], "alpha": parameters["alpha"],
                     "weight_decay": parameters["weight_decay"], "peak_gpu_bytes": peak,
                     "loss_history": "unavailable: the trainer keeps per-epoch losses internal"}}
    del dictionary, model, scaled, codes, reconstruction
    return payload


def run_sae_unit(cache, population, roles, config, seed, device, *, fit=None):
    """Run S(seed, r) through the cache; P(r) is loaded from the cache, so a missing predictor unit blocks it."""
    if config_hash(config) != cache.config_hash:
        raise ValueError("Configuration differs from the cache configuration")
    r, first = int(roles["reference_year"]), config["sae_seeds"][0]

    def produce():
        anchor = f"S/{first}/{r}"
        expected = cache.load(anchor)["scaler_identity"] if seed != first and cache.is_complete(anchor) else None
        return sae_unit(population, r, roles, cache.load(f"P/{r}"), config, seed, device,
                        expected_scaler_identity=expected, fit=fit)

    return cache.run(f"S/{seed}/{r}", produce, projected_bytes=projected_sae_bytes(config))


def load_reference_sae(cache, r, seed, device):
    """Rebuild a cached reference SAE and scaler for encoding; refuses any state that changed since training."""
    import numpy as np
    import torch
    from sae import SAE
    from sklearn.preprocessing import StandardScaler
    if device not in ("cpu", "cuda"):
        raise ValueError("Loading a reference SAE needs a resolved device: cpu or cuda")
    r, seed = int(r), int(seed)
    unit = cache.load(f"S/{seed}/{r}")
    if unit["reference_year"] != r or unit["seed"] != seed:
        raise RuntimeError("Cached SAE unit belongs to a different reference year or seed")
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.var_ = (np.array(unit[name], dtype=np.float64)
                                                for name in ("scaler_mean", "scaler_scale", "scaler_var"))
    scaler.n_features_in_, scaler.n_samples_seen_ = len(scaler.mean_), int(unit["scaler_samples"])
    if _identity_hash((scaler.mean_, scaler.scale_)) != unit["scaler_identity"]:
        raise RuntimeError("Cached reference scaler changed")
    weight = np.array(unit["encoder_weight"], dtype=np.float32)
    latents, dimension = weight.shape
    if (latents, dimension) != (unit["latent_count"], unit["embedding_dim"]) or len(scaler.mean_) != dimension:
        raise RuntimeError("Cached SAE shapes disagree with the unit record")
    parameters = unit["sae"]
    model = SAE(data_dimension=dimension, scaling_factor=parameters["scaling_factor"], use_decoder_bias=True,
                type=parameters["model_type"])
    if model.num_latents != latents:
        raise RuntimeError("Rebuilt SAE latent count differs from the cached state")
    model.load_state_dict({"encoder.weight": torch.from_numpy(weight),
                           "encoder.bias": torch.from_numpy(np.array(unit["encoder_bias"], dtype=np.float32)),
                           "decoder_bias": torch.from_numpy(np.array(unit["decoder_bias"], dtype=np.float32))})
    if _sae_state_identity(model) != unit["sae_identity"]:
        raise RuntimeError("Cached reference SAE changed")
    model.to(device).eval()
    return {"scaler": scaler, "run": {"model": model, "model_type": parameters["model_type"]},
            "scaler_identity": unit["scaler_identity"], "sae_identity": unit["sae_identity"], "seed": seed,
            "epochs": int(unit["epochs"]), "reference_year": r, "latent_count": int(latents),
            "embedding_dim": int(dimension), "unit_identity": unit["identity"]}


def load_reference_dictionaries(cache, r, device):
    """Every configured seed's reference SAE for year r, refusing scalers that differ between seeds."""
    dictionaries = {seed: load_reference_sae(cache, r, seed, device) for seed in cache.config["sae_seeds"]}
    if (len({entry["scaler_identity"] for entry in dictionaries.values()}) != 1
            or len({(entry["latent_count"], entry["embedding_dim"]) for entry in dictionaries.values()}) != 1):
        raise RuntimeError("Reference scaler or SAE shape differs between seeds")
    return dictionaries


def encode_with_reference_sae(dictionary, embeddings, device, batch_size=256):
    """Batched float32 activations of raw embeddings under the frozen reference scaler and SAE."""
    import numpy as np
    values = np.asarray(embeddings)
    if values.ndim != 2 or values.shape[1] != dictionary["embedding_dim"]:
        raise ValueError("Embeddings must be two-dimensional with the reference embedding dimension")
    if not np.isfinite(values).all():
        raise ValueError("Embeddings must be finite")
    if not len(values):
        return np.empty((0, dictionary["latent_count"]), dtype=np.float32)
    codes = encode_reference_dictionary(dictionary, values.astype(np.float32, copy=False), device, batch_size)
    if codes.dtype != np.float32 or codes.shape != (len(values), dictionary["latent_count"]) or not np.isfinite(codes).all():
        raise RuntimeError("Reference encoding must be finite float32 activations with one column per latent")
    if dictionary["run"]["model_type"] == "ReLU" and (codes < 0).any():
        raise RuntimeError("ReLU activations must be nonnegative")
    return codes


FAMILY_KINDS = ("full", "partial", "chain", "inconsistent")
FAMILY_COUNT_KEYS = FAMILY_KINDS + ("singleton",)
TIE_TOLERANCE = 1e-12
MATCHING_VIEW_KEYS = ("year", "patient_id", "fingerprint")


def _family_inputs(activations_by_seed, seeds, factor_ids):
    import numpy as np
    seeds = list(seeds)
    if (len(seeds) < 2 or len(activations_by_seed) != len(seeds)
            or not all(isinstance(seed, (int, np.integer)) and not isinstance(seed, bool) for seed in seeds)
            or len(set(int(seed) for seed in seeds)) != len(seeds)):
        raise ValueError("Families need one activation matrix per distinct integer seed and at least two seeds")
    seeds = [int(seed) for seed in seeds]
    matrices = [np.asarray(matrix) for matrix in activations_by_seed]
    if any(matrix.ndim != 2 or matrix.dtype.kind != "f" for matrix in matrices):
        raise ValueError("Activations must be two-dimensional floating-point matrices")
    if len({len(matrix) for matrix in matrices}) != 1 or len(matrices[0]) < 2:
        raise ValueError("Activation matrices must cover identical ordered rows (at least two)")
    if factor_ids is None:
        ids = [np.arange(matrix.shape[1], dtype=np.int64) for matrix in matrices]
    else:
        ids = [np.asarray(values, dtype=np.int64) for values in factor_ids]
        if (len(ids) != len(matrices) or any(values.shape != (matrix.shape[1],) or (values < 0).any()
                                             or len(np.unique(values)) != len(values)
                                             for values, matrix in zip(ids, matrices))):
            raise ValueError("Factor IDs must be unique nonnegative integers, one per activation column of each seed")
    return seeds, matrices, ids


def _zscore_columns(values):
    centered = values - values.mean(dim=0, keepdim=True)
    return centered / (centered * centered).mean(dim=0, keepdim=True).sqrt()


def _correlation(left, right):
    return (left.T @ right / left.shape[0]).clamp(-1.0, 1.0)


def pearson_matrix(left, right, device):
    """Pearson correlation of every left column with every right column over identical ordered rows.

    Float64 on the run device: the matrices hold a few hundred rows by a few hundred latents, so the double-precision
    matmul is cheap even on a consumer GPU, and it keeps CPU and CUDA results equal to far better than 1e-6.
    """
    import numpy as np
    import torch
    if device not in ("cpu", "cuda"):
        raise ValueError("Correlation needs a resolved device: cpu or cuda")
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 2 or right.ndim != 2 or len(left) != len(right) or len(left) < 2:
        raise ValueError("Correlation needs two matrices over the same at least two rows")
    tensors = [torch.as_tensor(np.asarray(values, dtype=np.float64), device=device) for values in (left, right)]
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("Correlation needs finite activations")
    if any(bool((tensor.amax(dim=0) == tensor.amin(dim=0)).any()) for tensor in tensors):
        raise ValueError("Correlation is undefined for a constant column")
    return _correlation(_zscore_columns(tensors[0]), _zscore_columns(tensors[1])).cpu().numpy()


def _pair_links(corr, left_ids, right_ids):
    """Lowest-ID argmax each way and the mutual mask, as device tensors; ties are within TIE_TOLERANCE of the maximum."""
    import torch
    unreachable = torch.iinfo(torch.int64).max

    def lowest(matrix, ids):
        close = matrix >= matrix.amax(dim=1, keepdim=True) - TIE_TOLERANCE
        candidates = torch.where(close, ids[None, :], torch.full_like(close, unreachable, dtype=torch.int64))
        return candidates.argmin(dim=1), close.sum(dim=1) > 1

    forward, forward_tied = lowest(corr, right_ids)
    backward, backward_tied = lowest(corr.T, left_ids)
    mutual = backward[forward] == torch.arange(len(forward), device=corr.device)
    return forward, backward, mutual, forward_tied, backward_tied


def _components(nodes, edges):
    parent = {node: node for node in nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)
    members, edge_count = {}, {}
    for node in nodes:
        members.setdefault(find(node), []).append(node)
    for left, _ in edges:
        root = find(left)
        edge_count[root] = edge_count.get(root, 0) + 1
    return sorted((sorted(group), edge_count.get(root, 0)) for root, group in members.items())


def _component_kind(members, edge_count, seed_count):
    """full: one factor per seed with every pair linked; partial: two factors of different seeds; chain: three or more
    factors of distinct seeds that are not a full clique; inconsistent: two factors of one seed (any component larger
    than the seed count necessarily is); singleton: unmatched."""
    if len(members) == 1:
        return "singleton"
    if len({seed for seed, _ in members}) < len(members):
        return "inconsistent"
    if len(members) == seed_count and edge_count == seed_count * (seed_count - 1) // 2:
        return "full"
    return "partial" if len(members) == 2 else "chain"


def _kind_counts(components, seed_count):
    counts = dict.fromkeys(FAMILY_COUNT_KEYS, 0)
    for members, edge_count in components:
        kind = _component_kind(members, edge_count, seed_count)
        counts[kind] += len(members) if kind == "singleton" else 1
    return counts


def build_factor_families(activations_by_seed, seeds, *, device, id_prefix, factor_ids=None, permutations=0, rng_seed=0):
    """Anchor-free factor families from per-seed activation matrices over identical ordered rows.

    Dead (all zero) and constant nonzero (Pearson undefined) latents are excluded and counted. Pearson is computed
    for every seed pair on the device (float64, z-scored columns, matmul / rows); mutual nearest neighbours link
    factors without a threshold, ties going to the lowest factor ID and being recorded; connected components of the
    link graph are full / partial / chain / inconsistent families or singletons. The chance check re-runs the same
    steps with every seed's row order independently permuted (a seeded numpy generator, so results do not depend on
    the device) and reports observed counts against the permutation mean and maximum. Only activations are read.
    """
    import numpy as np
    import torch
    if device not in ("cpu", "cuda"):
        raise ValueError("Family construction needs a resolved device: cpu or cuda")
    if type(permutations) is not int or permutations < 0:
        raise ValueError("permutations must be a nonnegative integer")
    if not isinstance(id_prefix, str) or not id_prefix:
        raise ValueError("id_prefix must name the reference year and split seed")
    seeds, matrices, ids = _family_inputs(activations_by_seed, seeds, factor_ids)
    rows, seed_count = len(matrices[0]), len(seeds)
    tensors = [torch.as_tensor(np.asarray(matrix, dtype=np.float64), device=device) for matrix in matrices]
    if not all(torch.stack([torch.isfinite(tensor).all() for tensor in tensors]).tolist()):
        raise ValueError("Activations must be finite")
    per_seed, z, usable_ids = [], [], []
    for index, tensor in enumerate(tensors):
        alive = (tensor != 0).any(dim=0)
        constant = alive & (tensor.amax(dim=0) == tensor.amin(dim=0))
        flags = torch.stack([alive, constant]).cpu().numpy()
        alive_mask, constant_mask = flags[0], flags[1]
        usable = alive_mask & ~constant_mask
        usable_ids.append(ids[index][usable])
        per_seed.append({"seed": seeds[index], "latents": int(tensor.shape[1]), "usable": int(usable.sum()),
                         "dead": ids[index][~alive_mask].tolist(),
                         "undefined_correlation": ids[index][constant_mask].tolist()})
        z.append(_zscore_columns(tensor[:, torch.as_tensor(usable, device=device)]) if usable.any() else None)
    id_tensors = [torch.as_tensor(values, device=device) for values in usable_ids]
    pairs = [(i, j) for i in range(seed_count) for j in range(i + 1, seed_count)
             if z[i] is not None and z[j] is not None]
    nodes = [(seeds[i], int(factor)) for i in range(seed_count) for factor in usable_ids[i]]
    position = [{int(factor): at for at, factor in enumerate(values)} for values in usable_ids]

    observed = {}
    for i, j in pairs:
        corr = _correlation(z[i], z[j])
        links = _pair_links(corr, id_tensors[i], id_tensors[j])
        observed[i, j] = [corr.cpu().numpy()] + [tensor.cpu().numpy() for tensor in links]

    def edges_from(i, j, forward, mutual):
        return [((seeds[i], int(usable_ids[i][at])), (seeds[j], int(usable_ids[j][forward[at]])))
                for at in np.flatnonzero(mutual)]

    edges, ties, pair_report = [], [], []
    for i, j in pairs:
        corr, forward, backward, mutual, forward_tied, backward_tied = observed[i, j]
        edges += edges_from(i, j, forward, mutual)
        for direction, matrix, flags, source_ids, target_ids in (
                ("left_to_right", corr, forward_tied, usable_ids[i], usable_ids[j]),
                ("right_to_left", corr.T, backward_tied, usable_ids[j], usable_ids[i])):
            for source in np.flatnonzero(flags):
                tied = target_ids[matrix[source] >= matrix[source].max() - TIE_TOLERANCE]
                ties.append({"pair": [seeds[i], seeds[j]], "direction": direction, "source": int(source_ids[source]),
                             "tied": sorted(int(factor) for factor in tied), "chosen": int(tied.min())})
    for i in range(seed_count):
        for j in range(i + 1, seed_count):
            entry = {"seeds": [seeds[i], seeds[j]], "usable": [len(usable_ids[i]), len(usable_ids[j])],
                     "mutual_pairs": 0, "mutual_correlation": None}
            if (i, j) in observed:
                corr, forward, _, mutual, _, _ = observed[i, j]
                values = corr[np.flatnonzero(mutual), forward[mutual]]
                entry["mutual_pairs"] = int(mutual.sum())
                if len(values):
                    entry["mutual_correlation"] = {"min": float(values.min()), "median": float(np.median(values)),
                                                   "max": float(values.max())}
            pair_report.append(entry)

    def between(first, second):
        (a, left), (b, right) = (seeds.index(first[0]), first[1]), (seeds.index(second[0]), second[1])
        if a < b:
            return float(observed[a, b][0][position[a][left], position[b][right]])
        return float(observed[b, a][0][position[b][right], position[a][left]])

    components = _components(nodes, edges)
    families, singletons = [], []
    for members, edge_count in components:
        kind = _component_kind(members, edge_count, seed_count)
        if kind == "singleton":
            singletons.append(list(members[0]))
            continue
        scores = {member: [between(member, other) for other in members if other[0] != member[0]] for member in members}
        means = {member: float(np.mean(values)) for member, values in scores.items()}
        best = max(means.values())
        medoid = min(member for member in members if means[member] >= best - TIE_TOLERANCE)
        pairwise = [between(a, b) for index, a in enumerate(members) for b in members[index + 1:] if a[0] != b[0]]
        families.append({"id": f"{id_prefix}/" + ",".join(f"{seed}:{factor}" for seed, factor in members),
                         "kind": kind, "members": [list(member) for member in members], "size": len(members),
                         "links": int(edge_count), "min_correlation": float(min(pairwise)),
                         "median_correlation": float(np.median(pairwise)), "medoid": list(medoid),
                         "medoid_mean_correlation": means[medoid]})
    counts = _kind_counts(components, seed_count)

    chance = None
    if permutations:
        generator = np.random.default_rng(rng_seed)
        order = torch.as_tensor(np.stack([[generator.permutation(rows) for _ in seeds] for _ in range(permutations)]),
                                device=device)
        collected = {pair: [] for pair in pairs}
        for index in range(permutations):
            shuffled = [None if values is None else values[order[index, at]] for at, values in enumerate(z)]
            for i, j in pairs:
                forward, _, mutual, _, _ = _pair_links(_correlation(shuffled[i], shuffled[j]), id_tensors[i], id_tensors[j])
                collected[i, j].append(torch.stack([forward, mutual.to(torch.int64)]))
        stacked = {pair: torch.stack(values).cpu().numpy() for pair, values in collected.items()}
        table = []
        for index in range(permutations):
            permuted = [edge for i, j in pairs
                        for edge in edges_from(i, j, stacked[i, j][index, 0], stacked[i, j][index, 1].astype(bool))]
            table.append(_kind_counts(_components(nodes, permuted), seed_count))
        chance = {"permutations": permutations, "rng_seed": list(rng_seed) if np.ndim(rng_seed) else int(rng_seed),
                  "scheme": "every seed's row order independently permuted in every permutation",
                  "observed": dict(counts), "per_permutation": table,
                  "mean": {key: float(np.mean([row[key] for row in table])) for key in FAMILY_COUNT_KEYS},
                  "max": {key: int(max(row[key] for row in table)) for key in FAMILY_COUNT_KEYS}}
    return {"id_prefix": id_prefix, "seeds": seeds, "rows": rows, "device": device, "precision": "float64",
            "tie_tolerance": TIE_TOLERANCE, "per_seed": per_seed, "pairs": pair_report,
            "ties": {"count": len(ties), "events": ties}, "families": families, "singletons": singletons,
            "counts": counts, "chance": chance}


def family_candidates(families, medoid_activations, num_latents, fraction):
    """Full families ranked by medoid positive-activation count (ties by family ID), cut at the latent-count cap."""
    import numpy as np
    cap = candidate_cap(num_latents, fraction)
    full = [family for family in families if family["kind"] == "full"]
    if len({family["id"] for family in full}) != len(full):
        raise ValueError("Family IDs must be unique")
    ranked = []
    for family in full:
        if family["id"] not in medoid_activations:
            raise ValueError(f"Medoid activations missing for family {family['id']}")
        values = np.asarray(medoid_activations[family["id"]], dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Medoid activations must be one finite vector over the matching rows")
        ranked.append((-int((values > 0).sum()), family["id"], family))
    ranked.sort(key=lambda item: (item[0], item[1]))
    records = [{"rank": rank, "family_id": family_id, "medoid": list(family["medoid"]), "positive_count": -negative}
               for rank, (negative, family_id, family) in enumerate(ranked, start=1)]
    return {"cap": cap, "fraction": fraction, "num_latents": int(num_latents), "full_families": len(full),
            "ranking": "medoid positive-activation count on the matching rows, descending; ties by family ID",
            "candidates": records[:cap], "dropped_by_cap": records[cap:]}


def matching_population_view(population):
    """Population fields the family unit may read; outcomes are deliberately absent."""
    return {key: population[key] for key in MATCHING_VIEW_KEYS}


def _role_embeddings(view, r, roles, predictor_output, role_names):
    """Year-r rows of the given reference roles from the cached P(r) output, in population order, never recomputed."""
    import numpy as np
    if roles["reference_year"] != r:
        raise ValueError("Roles belong to a different reference year")
    missing = [key for key in PREDICTOR_OUTPUT_KEYS if key not in predictor_output]
    if missing:
        raise ValueError(f"Predictor output lacks {missing}")
    if predictor_output["reference_year"] != r:
        raise ValueError("Predictor output belongs to a different reference year")
    if (predictor_output["roles_identity"] != roles["identity"]
            or predictor_output["population_identity"] != view["fingerprint"]["population_identity"]):
        raise ValueError("Predictor output was produced for different roles or a different population")
    rows, embeddings = np.asarray(predictor_output["row_index"]), np.asarray(predictor_output["embeddings"])
    if rows.ndim != 1 or embeddings.ndim != 2 or len(embeddings) != len(rows) or embeddings.dtype != np.float32:
        raise ValueError("Predictor output needs row-aligned float32 embeddings")
    if _identity_hash({"row_index": rows}) != predictor_output["output_row_identity"]:
        raise ValueError("Predictor output rows differ from their recorded identity")
    if not np.array_equal(rows, np.flatnonzero(view["year"] >= r)):
        raise ValueError("Predictor output must cover every row of years r..last")
    if embeddings.shape[1] != predictor_output["embedding_dim"]:
        raise ValueError("Predictor output embedding dimension differs from its record")
    chosen = ((view["year"][rows] == r) & np.isin(roles["row_role"][rows], list(role_names))
              & ~roles["is_entrant"][rows])
    positions = np.flatnonzero(chosen)
    if not len(positions):
        raise ValueError(f"No {'/'.join(role_names)} rows in reference year {r}")
    return rows[positions], embeddings[positions]


def projected_family_bytes(config, row_bound=8192, latent_bound=768):
    """Upper bound of one D/r/families unit's cached bytes: candidate activations, member and singleton tables, metadata."""
    cap = candidate_cap(latent_bound, config["candidate_cap_fraction"])
    nodes = len(config["sae_seeds"]) * latent_bound
    return cap * row_bound * 4 + row_bound * 8 + nodes * 5 * 8 + nodes * 2 * 8 + 4 * 2**20


def family_unit(view, r, roles, predictor_output, dictionaries, config, device, *, build=None):
    """D(r) families: encode year-r matching rows with every seed's reference SAE, build families, select candidates."""
    try:
        return _family_unit(view, int(r), roles, predictor_output, dictionaries, config, device,
                            build_factor_families if build is None else build)
    finally:
        if device == "cuda":
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()


def _family_unit(view, r, roles, predictor_output, dictionaries, config, device, build):
    import numpy as np
    if device not in ("cpu", "cuda"):
        raise ValueError("Family unit needs a resolved device: cpu or cuda")
    if set(view) != set(MATCHING_VIEW_KEYS):
        raise ValueError(f"Family unit reads only {MATCHING_VIEW_KEYS}; outcomes are not accepted")
    seeds = list(config["sae_seeds"])
    if list(dictionaries) != seeds:
        raise ValueError("Dictionaries must be given for exactly the configured SAE seeds, in order")
    latent_counts = {entry["latent_count"] for entry in dictionaries.values()}
    if len(latent_counts) != 1 or len({entry["scaler_identity"] for entry in dictionaries.values()}) != 1:
        raise RuntimeError("Reference scaler or latent count differs between seeds")
    num_latents = int(latent_counts.pop())
    rows, embeddings = _role_embeddings(view, r, roles, predictor_output, config["matching_roles"])
    if len(rows) < 2:
        raise ValueError(f"Reference year {r} has fewer than two matching rows")
    batch = config["encoding_batch_size"]
    matrices = [encode_with_reference_sae(dictionaries[seed], embeddings, device, batch) for seed in seeds]
    split_seed = int(config["split_seed"])
    result = build(matrices, seeds, device=device, id_prefix=f"{r}/{split_seed}",
                   permutations=config["chance_permutations"], rng_seed=[split_seed, r])
    seed_at = {seed: index for index, seed in enumerate(seeds)}
    medoids = {family["id"]: matrices[seed_at[family["medoid"][0]]][:, family["medoid"][1]]
               for family in result["families"] if family["kind"] == "full"}
    selection = family_candidates(result["families"], medoids, num_latents, config["candidate_cap_fraction"])
    candidate_activations = np.stack([medoids[record["family_id"]] for record in selection["candidates"]]
                                     ).astype(np.float32) if selection["candidates"] else np.empty((0, len(rows)), np.float32)
    member_table = [[index, seed, factor, int([seed, factor] == family["medoid"]), FAMILY_KINDS.index(family["kind"])]
                    for index, family in enumerate(result["families"]) for seed, factor in family["members"]]
    candidate_table = [[record["rank"], *record["medoid"], record["positive_count"]] for record in selection["candidates"]]
    per_seed = {entry["seed"]: entry for entry in result["per_seed"]}
    construction = {key: result[key] for key in ("id_prefix", "seeds", "rows", "device", "precision", "tie_tolerance",
                                                 "per_seed", "pairs", "ties", "counts", "chance")}
    return {
        "row_index": rows.astype(np.int64), "candidate_activations": candidate_activations,
        "candidate_table": np.asarray(candidate_table, dtype=np.int64).reshape(-1, 4),
        "member_table": np.asarray(member_table, dtype=np.int64).reshape(-1, 5),
        "singleton_table": np.asarray(result["singletons"], dtype=np.int64).reshape(-1, 2),
        "_rows": int(len(rows)), "reference_year": int(r), "split_seed": split_seed, "num_latents": num_latents,
        "member_table_columns": ["family_index", "seed", "factor", "is_medoid", "kind_index"],
        "candidate_table_columns": ["rank", "seed", "factor", "positive_count"], "kinds": list(FAMILY_KINDS),
        "matching_roles": list(config["matching_roles"]), "matching_rows": int(len(rows)),
        "matching_patients": int(len(np.unique(view["patient_id"][rows]))),
        "matching_rows_identity": _identity_hash({"row_index": rows}),
        "construction": construction, "families": result["families"], "selection": selection,
        "dead_latents": {str(seed): len(per_seed[seed]["dead"]) for seed in seeds},
        "undefined_latents": {str(seed): len(per_seed[seed]["undefined_correlation"]) for seed in seeds},
        "scaler_identity": next(iter(dictionaries.values()))["scaler_identity"],
        "sae_identities": {str(seed): dictionaries[seed]["sae_identity"] for seed in seeds},
        "roles_identity": roles["identity"], "population_identity": view["fingerprint"]["population_identity"],
        "predictor_output_row_identity": predictor_output["output_row_identity"],
        "outcomes_read": False}


def run_family_unit(cache, view, roles, config, device, *, build=None):
    """Run D(r)/families through the cache; P(r) and every seed's S unit are dependencies, loaded from the cache."""
    if config_hash(config) != cache.config_hash:
        raise ValueError("Configuration differs from the cache configuration")
    if set(view) != set(MATCHING_VIEW_KEYS):
        raise ValueError(f"Family unit reads only {MATCHING_VIEW_KEYS}; outcomes are not accepted")
    r = int(roles["reference_year"])
    return cache.run(f"D/{r}/families",
                     lambda: family_unit(view, r, roles, cache.load(f"P/{r}"), load_reference_dictionaries(cache, r, device),
                                         config, device, build=build),
                     projected_bytes=projected_family_bytes(config))


RULE_SOURCES = ("high_precision", "semantic")
ENABLED_RULE_SOURCES = ("semantic",)
RULE_VIEW_KEYS = ("year", "patient_id", "X", "feature_names", "fingerprint")
NO_RULE_REASON = "dropped:no_feasible_semantic_rule"
SWEEP_METRICS = ("f2", "precision", "recall", "lift")
RECURRENCE_KEYS = ("max_candidates_per_bootstrap", "family_similarity_threshold", "min_family_recurrence")


def rule_population_view(population):
    """Population fields the rule unit may read; outcomes are deliberately absent."""
    return {key: population[key] for key in RULE_VIEW_KEYS}


def _require_enabled_rule_source(rule_source):
    if rule_source not in RULE_SOURCES:
        raise ValueError(f"Unknown rule_source {rule_source!r}; known sources are {RULE_SOURCES}")
    if rule_source not in ENABLED_RULE_SOURCES:
        raise ValueError(f"rule_source {rule_source!r} is not enabled in this pilot; enabled: {ENABLED_RULE_SOURCES}")


def rule_selection_config(rule):
    """Selection filters exactly as configured; every other search limit keeps the original library default."""
    from semantic_rules import RuleSetSelectionConfig
    return RuleSetSelectionConfig(objective=rule["objective"], min_precision=rule["min_precision"], min_lift=rule["min_lift"],
                                  max_rules=rule["max_rules"], max_rule_length=rule["max_rule_length"],
                                  min_marginal_recall=rule["min_marginal_recall"])


def _backend_settings(overrides=None):
    """Stable rule backend settings (library defaults: 30 bootstraps x 50 trees, depth 15) except the per-job seed."""
    from dataclasses import asdict
    from stable_rule_backend import StableRuleBackendConfig
    settings = asdict(StableRuleBackendConfig(bootstrap_unit="group", random_state=0, **(overrides or {})))
    for name in ("random_state", "show_progress", "progress_desc"):
        del settings[name]
    return settings


def _derived_seed(*parts):
    digest = hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def semantic_winner(rows, rule_source="semantic"):
    """Per-A rows annotated by the original winner ordering, and the primary row among feasible A (None if none)."""
    from temporal_rules import select_activation_winner
    _require_enabled_rule_source(rule_source)
    ordered = select_activation_winner(rows, rule_source=rule_source)
    primary = [row for row in ordered if row["target_role"] == "primary"]
    if len(primary) > 1 or (primary and not primary[0]["valid"]):
        raise RuntimeError("Winner selection must mark at most one feasible primary row")
    return ordered, (primary[0] if primary else None)


def activation_threshold_target(activations, fraction):
    """Positive-only cutoff of the top `fraction` of strictly positive activations; at least two positives are required."""
    import numpy as np
    from semantic_rules import ActivationTargetSpec, fit_activation_target
    spec = ActivationTargetSpec(name=f"A={float(fraction):g}", positive_fraction=float(fraction), minimum_positive_samples=2)
    return fit_activation_target(np.asarray(activations, dtype=float), spec)


def rule_inner_split(patients, activations, seed):
    """Deterministic patient-level 50/50 split of the rule rows, stratified by medoid positive activation.

    The stratum does not depend on A, so every A of a family shares this split; the seed comes from the caller.
    """
    import numpy as np
    ids = np.asarray(_canonical_patient_ids(patients))
    values = np.asarray(activations, dtype=float)
    if ids.ndim != 1 or values.shape != ids.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("Rule rows need one activation and one distinct patient per row")
    return tuple(_stratified_patient_partition(ids, (values > 0).astype(int), (.5, .5), seed))


def _count_conditions(rule_set):
    return sum(len(rule.conditions) for rule in rule_set.rules)


def _sweep_job(job):
    """Stable-backend candidates on the fit half and constrained selection on the tune half, for one (family, A).

    Like the original semantic procedure, candidate occurrences are capped per bootstrap, clustered into rule families on the
    tune half and only families recurring in enough bootstraps are offered to the selection (recurrence=None offers every
    unique candidate). A pure function of its payload (seeded backend, no shared state): serial and pooled runs agree.
    """
    from semantic_compare import RuleSimilarityConfig, cluster_rule_families, recurrent_representatives
    from semantic_experiment import _cap_occurrences
    from semantic_rules import select_rule_set
    from stable_rule_backend import StableRuleBackendConfig, discover_stable_rule_candidates
    row = dict(job["row"])
    discovery = discover_stable_rule_candidates(job["X_fit"], job["y_fit"].astype(int), job["feature_names"],
                                                groups=job["patients_fit"], clinical_group_map={},
                                                config=StableRuleBackendConfig(random_state=row["backend_seed"],
                                                                               **job["backend"]))
    recurrence, offered, families = job["recurrence"], discovery.candidates, None
    if recurrence is not None:
        clustering = cluster_rule_families(_cap_occurrences(discovery.occurrences, recurrence["max_candidates_per_bootstrap"]),
                                           job["X_tune"], RuleSimilarityConfig(similarity_threshold=recurrence["family_similarity_threshold"],
                                                                               min_recurrence=recurrence["min_family_recurrence"]),
                                           total_bootstraps=job["backend"]["n_bootstraps"])
        families, offered = len(clustering.families), recurrent_representatives(clustering.families, recurrence["min_family_recurrence"])
    selector = select_rule_set if job["select"] is None else job["select"]
    selection = selector(offered, job["X_tune"], job["y_tune"].astype(int), job["selection_config"],
                         threshold_name=row["threshold_name"])
    rule_set = selection.rule_set
    feasible = bool(selection.feasible)
    stage = ("selected" if feasible else "no_candidate_rule" if not discovery.occurrences else
             "no_recurrent_family" if not offered else "no_feasible_subset")
    row.update(valid=feasible, feasible=feasible, failure_reason=None if feasible else selection.reason, funnel_stage=stage,
               candidate_count=len(discovery.candidates), family_count=families, offered_count=len(offered),
               backend_warnings=list(discovery.warnings),
               search_method=selection.search_method, rule_count=len(rule_set.rules),
               condition_count=_count_conditions(rule_set), rule_set=rule_set.to_dict(),
               selected_count=int(selection.metrics.n_selected) if feasible else None,
               rule_identity=_identity_hash(rule_set.to_dict()) if feasible else None)
    row.update({name: float(getattr(selection.metrics, name)) if feasible else None for name in SWEEP_METRICS})
    return row


def _run_sweep_jobs(jobs, workers):
    """Results in job order; one process per worker, each limited to one BLAS thread (the forests are single-threaded)."""
    workers = max(1, min(int(workers), len(jobs)))
    if workers == 1:
        return [_sweep_job(job) for job in jobs]
    from joblib import Parallel, delayed, parallel_backend
    with parallel_backend("loky", inner_max_num_threads=1):
        return Parallel(n_jobs=workers, batch_size=1)(delayed(_sweep_job)(job) for job in jobs)


def run_threshold_sweep(X, patients, feature_names, candidates, activations, sweep, rule, *, reference_year, split_seed,
                        workers=1, backend=None, select=None, rule_source="semantic", recurrence=None):
    """Sweep A for every candidate on the rule_discovery rows; per-A rows, winners, drops and frozen rules.

    `candidates` are family records (family_id, medoid, rank); `activations` holds one medoid vector per candidate over
    the given rows. A family without a feasible A is dropped and recorded; nothing replaces it.
    """
    import numpy as np
    _require_enabled_rule_source(rule_source)
    features, ids, values = np.asarray(X, dtype=np.float64), np.asarray(_canonical_patient_ids(patients)), np.asarray(activations)
    if features.ndim != 2 or len(features) != len(ids) or features.shape[1] != len(feature_names) or values.shape != (len(candidates), len(ids)):
        raise ValueError("Rule sweep features, patients, names and candidate activations must align")
    if not np.isfinite(features).all():
        raise ValueError("Rule rows must have finite features")
    selection_config, settings = rule_selection_config(rule), _backend_settings(backend)
    names, entries, jobs, splits = list(feature_names), [], [], {}
    for index, record in enumerate(candidates):
        medoid = np.asarray(values[index], dtype=np.float64)
        if not np.isfinite(medoid).all():
            raise ValueError("Medoid activations on the rule rows must be finite")
        split_derived = _derived_seed("rule_inner_split", split_seed, reference_year, record["family_id"])
        fit_patients, tune_patients = rule_inner_split(ids, medoid, split_derived)
        fit_mask, tune_mask = np.isin(ids, fit_patients), np.isin(ids, tune_patients)
        splits[record["family_id"]] = {
            "seed": split_derived,
            "stratified_by": "medoid_positive_activation", "fit_rows": int(fit_mask.sum()), "tune_rows": int(tune_mask.sum()),
            "fit_patients_identity": _identity_hash(fit_patients), "tune_patients_identity": _identity_hash(tune_patients)}
        for fraction in sweep:
            target = activation_threshold_target(medoid, fraction)
            binary = strict_activation_mask(medoid, target.to_dict())
            row = {"family_id": record["family_id"], "medoid": list(record["medoid"]), "rank": record["rank"],
                   "activation_target": float(fraction), "threshold_name": target.spec.name, "target_valid": bool(target.valid),
                   "cutoff": float(target.cutoff) if target.valid else None, "rows_fit": int(fit_mask.sum()),
                   "rows_tune": int(tune_mask.sum()), "positives_fit": int(binary[fit_mask].sum()),
                   "positives_tune": int(binary[tune_mask].sum()), "positive_rows": int(binary.sum()),
                   "backend_seed": _derived_seed("rule_backend", split_seed, reference_year, record["family_id"], float(fraction)),
                   "valid": False, "feasible": False, "failure_reason": target.invalid_reason,
                   "funnel_stage": "invalid_activation_target", "candidate_count": 0, "family_count": None, "offered_count": 0,
                   "backend_warnings": [], "search_method": None, "rule_count": 0, "condition_count": 0, "rule_set": None,
                   "selected_count": None, "rule_identity": None, **dict.fromkeys(SWEEP_METRICS)}
            if target.valid:
                jobs.append({"row": row, "X_fit": features[fit_mask], "y_fit": binary[fit_mask], "patients_fit": ids[fit_mask],
                             "X_tune": features[tune_mask], "y_tune": binary[tune_mask], "feature_names": names,
                             "backend": settings, "selection_config": selection_config, "select": select,
                             "recurrence": recurrence})
                entries.append(len(jobs) - 1)
            else:
                entries.append(row)
    results = _run_sweep_jobs(jobs, workers) if jobs else []
    rows = [results[entry] if isinstance(entry, int) else entry for entry in entries]
    sweep_rows, families, frozen = [], [], []
    for record in candidates:
        own = [row for row in rows if row["family_id"] == record["family_id"]]
        ordered, winner = semantic_winner(own, rule_source)
        sweep_rows.extend(ordered)
        summary = {"family_id": record["family_id"], "medoid": list(record["medoid"]), "rank": record["rank"],
                   "positive_count": record["positive_count"], "split": splits[record["family_id"]]}
        if winner is None:
            families.append({**summary, "status": NO_RULE_REASON, "winner_activation_target": None,
                             "reasons_by_activation_target": {f"{row['activation_target']:g}": row["failure_reason"] for row in ordered}})
            continue
        families.append({**summary, "status": "frozen", "winner_activation_target": winner["activation_target"],
                         "reasons_by_activation_target": {f"{row['activation_target']:g}": row["failure_reason"] for row in ordered}})
        frozen.append({"family_id": record["family_id"], "medoid": list(record["medoid"]), "reference_year": int(reference_year),
                       "split_seed": int(split_seed), "rule_source": rule_source,
                       "activation_target": winner["activation_target"], "cutoff": winner["cutoff"],
                       "target": {"valid": True, "cutoff": winner["cutoff"], "positive_fraction": winner["activation_target"]},
                       "rule_set": winner["rule_set"], "rule_identity": winner["rule_identity"],
                       "rule_count": winner["rule_count"], "condition_count": winner["condition_count"],
                       "tune_metrics": {name: winner[name] for name in SWEEP_METRICS},
                       "positives_fit": winner["positives_fit"], "positives_tune": winner["positives_tune"],
                       "inner_split": splits[record["family_id"]]})
    return {"rows": sweep_rows, "families": families, "frozen": frozen, "jobs": len(jobs),
            "selection_config": {name: getattr(selection_config, name) for name in selection_config.__dataclass_fields__},
            "backend": settings, "recurrence": None if recurrence is None else dict(recurrence)}


def _frozen_identity(entry):
    return _identity_hash({name: value for name, value in entry.items() if name not in ("identity", "rule")})


def projected_rule_bytes(config, latent_bound=768, rule_bytes=32 * 1024, row_bound=8192):
    """Upper bound of one D/r/rules unit's cached bytes: per-A rows, frozen rules, metadata."""
    cap = candidate_cap(latent_bound, config["candidate_cap_fraction"])
    return cap * len(config["threshold_sweep"]) * rule_bytes + cap * rule_bytes + row_bound * 8 + 4 * 2**20


def rule_unit(view, r, roles, family_output, config, *, workers=1, backend=None, select=None, rule_source=None,
              recurrence="configured"):
    """D(r) rules: semantic threshold sweep on the year-r rule_discovery rows of every candidate family medoid."""
    import numpy as np
    r = int(r)
    if set(view) != set(RULE_VIEW_KEYS):
        raise ValueError(f"Rule unit reads only {RULE_VIEW_KEYS}; outcomes are not accepted")
    rule_source = config["rule"]["rule_source"] if rule_source is None else rule_source
    _require_enabled_rule_source(rule_source)
    configured = isinstance(recurrence, str) and recurrence == "configured"
    if (backend is not None or not configured) and config.get("profile") != "v0":
        raise ValueError("Backend and candidate-filter overrides exist for the v0 fixtures only")
    if configured:
        recurrence = config["rule"]["recurrence"]
    if recurrence is not None and set(recurrence) != set(RECURRENCE_KEYS):
        raise ValueError(f"Recurrence settings must be exactly {RECURRENCE_KEYS}")
    if roles["reference_year"] != r or family_output["reference_year"] != r:
        raise ValueError("Roles or families belong to a different reference year")
    if (family_output["roles_identity"] != roles["identity"]
            or family_output["population_identity"] != view["fingerprint"]["population_identity"]):
        raise ValueError("Family unit was produced for different roles or a different population")
    matching = np.asarray(family_output["row_index"])
    if _identity_hash({"row_index": matching}) != family_output["matching_rows_identity"]:
        raise ValueError("Family unit rows differ from their recorded identity")
    year, patient = view["year"], view["patient_id"]
    expected = np.flatnonzero((year == r) & (roles["row_role"] == "rule_discovery") & ~roles["is_entrant"])
    positions = np.flatnonzero(roles["row_role"][matching] == "rule_discovery")
    if len(expected) < 2 or not np.array_equal(matching[positions], expected):
        raise ValueError("The family unit's rule_discovery rows must be exactly the year-r rule_discovery rows")
    if len(np.unique(patient[expected])) != len(expected):
        raise ValueError("Rule rows need one row per patient")
    selection = family_output["selection"]
    candidates = list(selection["candidates"])
    activations = np.asarray(family_output["candidate_activations"])
    if activations.shape != (len(candidates), len(matching)):
        raise ValueError("Candidate activations must align with the candidates and the matching rows")
    for index, record in enumerate(candidates):
        if int((activations[index] > 0).sum()) != record["positive_count"]:
            raise ValueError(f"Medoid activations disagree with the positive count of {record['family_id']}")
    result = run_threshold_sweep(view["X"][expected], patient[expected], view["feature_names"], candidates,
                                 activations[:, positions].astype(np.float64), config["threshold_sweep"], config["rule"],
                                 reference_year=r, split_seed=config["split_seed"], workers=workers, backend=backend,
                                 select=select, rule_source=rule_source, recurrence=recurrence)
    rows_identity = _identity_hash({"row_index": expected})
    for entry in result["frozen"]:
        entry["rule_rows_identity"] = rows_identity
        entry["identity"] = _frozen_identity(entry)
    sweep = [f"{fraction:g}" for fraction in config["threshold_sweep"]]
    feasible = [family for family in result["families"] if family["status"] == "frozen"]
    return {
        "rule_row_index": expected.astype(np.int64), "_rows": int(len(expected)), "reference_year": r,
        "split_seed": int(config["split_seed"]), "rule_source": rule_source, "sweep": sweep,
        "selection_config": result["selection_config"], "backend_config": result["backend"],
        "backend_overridden": backend is not None, "candidate_filter": result["recurrence"],
        "rule_rows": int(len(expected)), "rule_patients": int(len(np.unique(patient[expected]))),
        "rule_rows_identity": rows_identity,
        "family_matching_rows_identity": family_output["matching_rows_identity"],
        "feature_names_identity": _identity_hash(list(view["feature_names"])), "feature_count": len(view["feature_names"]),
        "candidates": len(candidates), "cap": selection["cap"], "dropped_by_cap": len(selection["dropped_by_cap"]),
        "refilled": False, "jobs": result["jobs"], "sweep_rows": result["rows"], "families": result["families"],
        "frozen": result["frozen"], "frozen_count": len(feasible), "dropped_count": len(candidates) - len(feasible),
        "winner_distribution": {name: sum(family["winner_activation_target"] == float(name) for family in feasible) for name in sweep},
        "no_rule_reason": NO_RULE_REASON, "scaler_identity": family_output["scaler_identity"],
        "sae_identities": family_output["sae_identities"], "roles_identity": roles["identity"],
        "population_identity": view["fingerprint"]["population_identity"], "outcomes_read": False}


def run_rule_unit(cache, view, roles, config, *, workers=None, backend=None, select=None, rule_source=None,
                  recurrence="configured"):
    """Run D(r)/rules through the cache; the families unit is a dependency, loaded from the cache."""
    if config_hash(config) != cache.config_hash:
        raise ValueError("Configuration differs from the cache configuration")
    if set(view) != set(RULE_VIEW_KEYS):
        raise ValueError(f"Rule unit reads only {RULE_VIEW_KEYS}; outcomes are not accepted")
    r = int(roles["reference_year"])
    pool = config["joblib_workers"] if workers is None else workers
    return cache.run(f"D/{r}/rules",
                     lambda: rule_unit(view, r, roles, cache.load(f"D/{r}/families"), config, workers=pool, backend=backend,
                                       select=select, rule_source=rule_source, recurrence=recurrence),
                     projected_bytes=projected_rule_bytes(config))


def load_frozen_rules(cache, r):
    """Frozen reference rules of year r by family ID, each with its RuleSet; refuses any entry whose identity changed."""
    from semantic_rules import RuleSet
    unit = cache.load(f"D/{int(r)}/rules")
    frozen = {}
    for entry in unit["frozen"]:
        rule_set = RuleSet.from_dict(entry["rule_set"])
        if (_identity_hash(rule_set.to_dict()) != entry["rule_identity"]
                or _identity_hash(entry["rule_set"]) != entry["rule_identity"] or _frozen_identity(entry) != entry["identity"]):
            raise RuntimeError(f"Frozen rule of {entry['family_id']} changed")
        frozen[entry["family_id"]] = {**entry, "rule": rule_set}
    return frozen


def rule_mask(rule, X, feature_names=None):
    """Rows selected by a rule set (RuleSet or its dict), evaluated in float64 like the backend that fitted it.

    With feature_names, every condition must name the column it indexes.
    """
    import numpy as np
    from semantic_rules import RuleSet
    rule_set = RuleSet.from_dict(rule) if isinstance(rule, dict) else rule
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Rule evaluation needs a two-dimensional feature matrix")
    for member in rule_set.rules:
        for condition in member.conditions:
            if condition.feature_index >= values.shape[1]:
                raise ValueError(f"Rule indexes column {condition.feature_index} outside {values.shape[1]} columns")
            if feature_names is not None and feature_names[condition.feature_index] != condition.feature_name:
                raise ValueError(f"Column {condition.feature_index} is not {condition.feature_name!r}")
    return rule_set.mask(values)


def rule_text(rule):
    """Readable OR-of-ANDs text of a rule set (RuleSet or its dict)."""
    from semantic_rules import RuleSet
    rule_set = RuleSet.from_dict(rule) if isinstance(rule, dict) else rule
    if not rule_set.rules:
        return "(empty rule set)"
    return " OR ".join("(" + " AND ".join(f"{c.feature_name} {c.operator} {c.threshold:.4g}" for c in member.conditions) + ")"
                       for member in rule_set.rules)


def rule_sweep_summary(unit):
    """Compact per-reference-year sweep summary: candidates, feasible and dropped families, winner A distribution."""
    return {"candidates": unit["candidates"], "feasible": unit["frozen_count"], "dropped": unit["dropped_count"],
            "winner A": {name: count for name, count in unit["winner_distribution"].items() if count},
            "rule rows": unit["rule_rows"], "jobs": unit["jobs"]}


CONTINUATION_SCHEMA = "pilot-continuation-v1"
CONTINUATION_PREFIX = "continuation_"
CONTINUATION_PARITY = {"rtol": 1e-4, "atol": 1e-5}
GRADIENT_REPLAY = {"rtol": 1e-3, "atol": 1e-5}
PORTABLE_OPTION_KEYS = ("is_classification", "average_logits", "num_classes", "class_shifts")
GRADIENT_OUTPUT_KEYS = PREDICTOR_OUTPUT_KEYS + ("probabilities", "gradients", "continuation_identity", "gate_positions")


def projected_continuation_bytes(embedding_bound=512):
    """Upper bound of the serialized head: two linear layers with a hidden width of four times the embedding."""
    dimension = int(embedding_bound)
    hidden = 4 * dimension
    return 4 * (dimension * hidden + hidden + hidden * 10 + 10) + 2**20


def _continuation_identity(meta, arrays):
    return _identity_hash({"meta": {key: value for key, value in meta.items() if key != "identity"},
                           "arrays": {name: arrays[name] for name in sorted(arrays)}})


def export_continuation(continuation):
    """Head tensors plus a JSON description from which the continuation is rebuilt without the predictor."""
    import numpy as np
    import torch
    options = getattr(continuation, "portable_options", None)
    if not isinstance(options, dict) or set(options) != set(PORTABLE_OPTION_KEYS):
        raise TypeError("Continuation lacks the portable parse/aggregate options")
    if not options["is_classification"] or options["num_classes"] != len(continuation.classes):
        raise TypeError("Only classification continuations can be serialized")
    if len(options["class_shifts"]) != 1:
        raise TypeError("Only single-configuration continuations can be serialized")
    head = continuation.head
    if type(head) is not torch.nn.Sequential:
        raise TypeError(f"Unsupported predictor head {type(head).__name__}")
    layers, arrays = [], {}
    for index, module in enumerate(head):
        if type(module) is torch.nn.Linear:
            if module.weight.dtype != torch.float32:
                raise TypeError("Predictor head weights must be float32")
            layers.append({"kind": "linear", "in_features": int(module.in_features),
                           "out_features": int(module.out_features), "bias": module.bias is not None})
            arrays[f"{CONTINUATION_PREFIX}head_{index}_weight"] = module.weight.detach().cpu().numpy().copy()
            if module.bias is not None:
                arrays[f"{CONTINUATION_PREFIX}head_{index}_bias"] = module.bias.detach().cpu().numpy().copy()
        elif type(module) is torch.nn.GELU:
            layers.append({"kind": "gelu", "approximate": str(module.approximate)})
        else:
            raise TypeError(f"Unsupported predictor head layer {type(module).__name__} at position {index}")
    shift = options["class_shifts"][0]
    if shift is not None:
        arrays[f"{CONTINUATION_PREFIX}class_shift"] = shift.detach().cpu().numpy().astype(np.int64)
    labels = continuation.eval_ys
    if labels is not None:
        if not torch.is_tensor(labels):
            raise TypeError("Continuation labels must be a tensor or None")
        arrays[f"{CONTINUATION_PREFIX}eval_ys"] = labels.detach().cpu().numpy().copy()
    meta = {"schema": CONTINUATION_SCHEMA, "layers": layers, "temperature": float(continuation.temperature),
            "classes": [float(value) for value in continuation.classes], "context_identity": continuation.context_identity,
            "reweight": bool(continuation.reweight), "average_logits": bool(options["average_logits"]),
            "num_classes": int(options["num_classes"]), "has_class_shift": shift is not None,
            "has_eval_ys": labels is not None, "array_names": sorted(arrays)}
    meta["identity"] = _continuation_identity(meta, arrays)
    return arrays, meta


def rebuild_continuation(arrays, meta, device):
    """MortalityContinuation rebuilt from exported tensors; refuses any array or description that changed."""
    import numpy as np
    import torch
    if device not in ("cpu", "cuda"):
        raise ValueError("Rebuilding a continuation needs a resolved device: cpu or cuda")
    if meta.get("schema") != CONTINUATION_SCHEMA:
        raise ValueError(f"Unknown continuation schema {meta.get('schema')!r}")
    missing = [name for name in meta["array_names"] if name not in arrays]
    if missing:
        raise RuntimeError(f"Cached continuation lacks {missing}")
    values = {name: np.asarray(arrays[name]) for name in meta["array_names"]}
    if _continuation_identity(meta, values) != meta["identity"]:
        raise RuntimeError("Cached continuation changed")
    modules = []
    for index, layer in enumerate(meta["layers"]):
        if layer["kind"] == "linear":
            module = torch.nn.Linear(layer["in_features"], layer["out_features"], bias=layer["bias"])
            state = {"weight": torch.from_numpy(values[f"{CONTINUATION_PREFIX}head_{index}_weight"].copy())}
            if layer["bias"]:
                state["bias"] = torch.from_numpy(values[f"{CONTINUATION_PREFIX}head_{index}_bias"].copy())
            module.load_state_dict(state)
        elif layer["kind"] == "gelu":
            module = torch.nn.GELU(approximate=layer["approximate"])
        else:
            raise ValueError(f"Unknown head layer {layer['kind']!r}")
        modules.append(module)
    head = torch.nn.Sequential(*modules).to(device)
    shift = torch.from_numpy(values[f"{CONTINUATION_PREFIX}class_shift"].copy()).to(device) if meta["has_class_shift"] else None
    labels = torch.from_numpy(values[f"{CONTINUATION_PREFIX}eval_ys"].copy()).to(device) if meta["has_eval_ys"] else None
    average_logits, num_classes = bool(meta["average_logits"]), int(meta["num_classes"])

    def parse(logits, shifts):
        outputs = []
        for position, class_shift in enumerate(shifts):
            output = logits[:, position:position + 1, :]
            if class_shift is not None:
                output = output[..., class_shift]
            if not average_logits:
                output = torch.nn.functional.softmax(output, dim=-1)
            outputs.append(output)
        return outputs

    def aggregate(outputs, reweight, eval_ys):
        output = torch.mean(torch.cat(outputs, 1), 1, keepdim=True)
        if average_logits:
            output = torch.nn.functional.softmax(output, dim=-1)
        if reweight:
            counts = torch.zeros(num_classes, device=output.device)
            flat = eval_ys.flatten().long().to(output.device)
            counts.scatter_add_(0, flat, torch.ones(len(flat), device=output.device))
            output = output / (counts / counts.sum())
            output = output / output.sum(dim=-1, keepdim=True)
        return torch.transpose(output, 0, 1).squeeze(0)

    continuation = MortalityContinuation(head, parse, aggregate, [shift], meta["temperature"], meta["classes"],
                                         meta["context_identity"], meta["reweight"], labels)
    continuation.portable_options = {"is_classification": True, "average_logits": average_logits,
                                     "num_classes": num_classes, "class_shifts": [shift]}
    continuation.identity = meta["identity"]
    return continuation


def _continuation_embeddings(continuation, embeddings):
    import torch
    device = next(continuation.head.parameters()).device
    values = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    if values.ndim != 2:
        raise ValueError("Embeddings must be (rows, dimension)")
    return values


def batched_death_gradients(continuation, embeddings, batch_rows=4096):
    """Float32 gradients of the final death probability with respect to raw embeddings, batched on the continuation device."""
    import numpy as np
    import torch
    if type(batch_rows) is not int or batch_rows < 1:
        raise ValueError("batch_rows must be a positive integer")
    values = _continuation_embeddings(continuation, embeddings)
    if not len(values):
        return np.empty((0, values.shape[1]), dtype=np.float32)
    gradients = [continuation.death_gradient(values[start:start + batch_rows]) for start in range(0, len(values), batch_rows)]
    return torch.cat(gradients).cpu().numpy().astype(np.float32, copy=False)


def batched_death_probabilities(continuation, embeddings, batch_rows=4096):
    """Float32 (rows, 2) class probabilities of raw embeddings through the continuation, batched on its device."""
    import numpy as np
    import torch
    if type(batch_rows) is not int or batch_rows < 1:
        raise ValueError("batch_rows must be a positive integer")
    values = _continuation_embeddings(continuation, embeddings)
    if not len(values):
        return np.empty((0, len(continuation.classes)), dtype=np.float32)
    with torch.no_grad():
        chunks = [continuation.probabilities(values[start:start + batch_rows]) for start in range(0, len(values), batch_rows)]
    return torch.cat(chunks).cpu().numpy().astype(np.float32, copy=False)


def replay_continuation(continuation, embeddings, probabilities, gradients):
    """Raise RuntimeError unless the continuation reproduces probabilities (parity tolerance) and gradients (finite-difference tolerance)."""
    import numpy as np
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or not len(embeddings):
        raise RuntimeError("Replay needs at least one embedding row")
    replayed = batched_death_probabilities(continuation, embeddings)
    try:
        np.testing.assert_allclose(replayed, np.asarray(probabilities, dtype=np.float32), **CONTINUATION_PARITY)
    except AssertionError as error:
        raise RuntimeError(f"Probability replay differs: {_first_line(error)}") from error
    differentiated = batched_death_gradients(continuation, embeddings)
    try:
        np.testing.assert_allclose(differentiated, np.asarray(gradients, dtype=np.float32), **GRADIENT_REPLAY)
    except AssertionError as error:
        raise RuntimeError(f"Gradient replay differs: {_first_line(error)}") from error
    return {"rows": int(len(embeddings)),
            "probability_max_abs_error": float(np.max(np.abs(replayed.astype(np.float64) - np.asarray(probabilities, dtype=np.float64)))),
            "gradient_max_abs_error": float(np.max(np.abs(differentiated.astype(np.float64) - np.asarray(gradients, dtype=np.float64))))}


def continuation_from_unit(unit, r, device):
    """Rebuild the head of a loaded P(r) unit and verify it on the cached gate rows before anything uses it."""
    import numpy as np
    meta = unit.get("continuation")
    if not isinstance(meta, dict) or unit.get("reference_year") != int(r):
        raise RuntimeError("Predictor unit lacks a serialized continuation for this reference year")
    if meta["identity"] != unit["continuation_identity"] or meta["context_identity"] != unit["context_identity"]:
        raise RuntimeError("Serialized continuation does not describe the unit's frozen predictor")
    continuation = rebuild_continuation({name: unit[name] for name in meta["array_names"] if name in unit}, meta, device)
    positions = np.asarray(unit["gate_positions"], dtype=np.int64)
    continuation.verification = replay_continuation(continuation, unit["embeddings"][positions],
                                                    unit["probabilities"][positions], unit["gradients"][positions])
    return continuation


def load_continuation(cache, r, device):
    """The frozen mortality continuation of reference year r, rebuilt from the cached P(r) unit and verified."""
    r = int(r)
    return continuation_from_unit(cache.load(f"P/{r}"), r, device)


IDEAS = (("idea1", "A"), ("idea2", "R"))
PERFORMANCE_VIEW_KEYS = RULE_VIEW_KEYS + ("y",)
TCAV_ELIGIBILITY_CUTOFFS = (0.1, 0.9)


def performance_population_view(population):
    """Population fields the reference performance unit may read: no gradients, labels only of the rows it selects."""
    return {key: population[key] for key in PERFORMANCE_VIEW_KEYS}


def _require_selection_settings(config):
    if (list(config["cav_seeds"]) != [42, 43, 44] or config["primary_cav_seed"] != 42
            or list(config["tcav_cutoffs"]) != list(TCAV_ELIGIBILITY_CUTOFFS)):
        raise ValueError("Selection supports CAV seeds 42/43/44, primary seed 42 and TCAV cutoffs 0.1/0.9 only")


def _predictor_positions(view, r, roles, predictor_output, role_names):
    """Year-r reference rows of the given roles and their positions in the cached P(r) arrays; only structure is validated."""
    import numpy as np
    r = int(r)
    if roles["reference_year"] != r:
        raise ValueError("Roles belong to a different reference year")
    missing = [key for key in GRADIENT_OUTPUT_KEYS if key not in predictor_output]
    if missing:
        raise ValueError(f"Predictor output lacks {missing}")
    if predictor_output["reference_year"] != r:
        raise ValueError("Predictor output belongs to a different reference year")
    if (predictor_output["roles_identity"] != roles["identity"]
            or predictor_output["population_identity"] != view["fingerprint"]["population_identity"]):
        raise ValueError("Predictor output was produced for different roles or a different population")
    rows = np.asarray(predictor_output["row_index"])
    if _identity_hash({"row_index": rows}) != predictor_output["output_row_identity"]:
        raise ValueError("Predictor output rows differ from their recorded identity")
    if not np.array_equal(rows, np.flatnonzero(view["year"] >= r)):
        raise ValueError("Predictor output must cover every row of years r..last")
    dimension = int(predictor_output["embedding_dim"])
    for name, shape in (("embeddings", (len(rows), dimension)), ("gradients", (len(rows), dimension)),
                        ("probabilities", (len(rows), 2))):
        values = np.asarray(predictor_output[name])
        if values.shape != shape or values.dtype != np.float32:
            raise ValueError(f"Predictor output {name} must be float32 with shape {shape}")
    chosen = (view["year"][rows] == r) & np.isin(roles["row_role"][rows], list(role_names)) & ~roles["is_entrant"][rows]
    positions = np.flatnonzero(chosen)
    return rows[positions], positions


def _family_rows(view, r, roles, family_output):
    """Matching rows, cached candidate activations and their family index from a validated families unit output."""
    import numpy as np
    if roles["reference_year"] != r or family_output["reference_year"] != r:
        raise ValueError("Roles or families belong to a different reference year")
    if (family_output["roles_identity"] != roles["identity"]
            or family_output["population_identity"] != view["fingerprint"]["population_identity"]):
        raise ValueError("Family unit was produced for different roles or a different population")
    matching = np.asarray(family_output["row_index"])
    if _identity_hash({"row_index": matching}) != family_output["matching_rows_identity"] or np.any(np.diff(matching) <= 0):
        raise ValueError("Family unit rows differ from their recorded identity")
    candidates = family_output["selection"]["candidates"]
    activations = np.asarray(family_output["candidate_activations"])
    if activations.shape != (len(candidates), len(matching)):
        raise ValueError("Candidate activations must align with the candidates and the matching rows")
    return matching, activations, {record["family_id"]: index for index, record in enumerate(candidates)}


def _medoid_activations(rows, embeddings, entries, family_rows, dictionaries, config, device):
    """Float64 medoid activations (entries, rows): rows of the families unit come from its cache, others are encoded
    with the medoid's seed SAE in one pass per seed."""
    import numpy as np
    matching, cached_values, index = family_rows
    result = np.empty((len(entries), len(rows)), dtype=np.float64)
    if not len(entries):
        return result
    cached = np.isin(rows, matching)
    where = np.searchsorted(matching, rows[cached])
    for position, entry in enumerate(entries):
        if entry["family_id"] not in index:
            raise ValueError(f"Frozen family {entry['family_id']} is not a candidate of the families unit")
        result[position, cached] = cached_values[index[entry["family_id"]], where]
    fresh = ~cached
    if fresh.any():
        for seed in sorted({int(entry["medoid"][0]) for entry in entries}):
            codes = encode_with_reference_sae(dictionaries[seed], embeddings[fresh], device, config["encoding_batch_size"])
            for position, entry in enumerate(entries):
                if int(entry["medoid"][0]) == seed:
                    result[position, fresh] = codes[:, int(entry["medoid"][1])]
    if not np.isfinite(result).all():
        raise ValueError("Medoid activations must be finite")
    return result


def fit_family_cavs(embeddings, features, activations, rule, patient_ids, row_index, seeds):
    """CAV repetitions of one frozen family on its cav_fit rows: rule positives against rule negatives at the minimum
    activation (positive_fraction .10 puts the negative-pool quantile at zero). A repetition whose prerequisites fail
    is recorded unavailable with its reason."""
    import numpy as np
    rows = np.asarray(row_index)
    fitted = fit_pilot_cavs(embeddings, np.asarray(features, dtype=np.float64), activations, rule, patient_ids,
                            seeds=tuple(int(seed) for seed in seeds))
    result = {}
    for seed in seeds:
        value = fitted[int(seed)]
        valid = bool(value["valid"])
        record = {"valid": valid, "failure_reason": None if valid else str(value.get("failure_reason") or "unavailable_cav"),
                  "cav": np.array(value["cav"], dtype=np.float64) if valid else None}
        for name in ("positive_count", "eligible_negative_count", "positive_training_count", "negative_count",
                     "positive_patients", "negative_patients"):
            record[name] = int(value[name]) if value.get(name) is not None else None
        for name in ("low_activation_cutoff", "cv_accuracy", "cv_auroc"):
            record[name] = float(value[name]) if valid and value.get(name) is not None else None
        for name, indices in (("positive_rows_identity", "positive_indices"), ("negative_rows_identity", "negative_indices")):
            record[name] = _identity_hash({"row_index": rows[np.asarray(value[indices])]}) if valid else None
        result[int(seed)] = record
    return result


def _cav_description(record):
    description = {name: value for name, value in record.items() if name != "cav"}
    description["status"] = "available" if record["valid"] else f"unavailable:{record['failure_reason']}"
    return description


def _selection_description(result, subgroup):
    eligible = bool(result["eligible"])
    reason = None if eligible else (result["unavailable_reason"] or "tcav_within_cutoffs")
    status = "eligible" if eligible else (f"unavailable:{reason}" if result["unavailable_reason"] else f"not_eligible:{reason}")
    return {"subgroup": subgroup, "primary_seed": 42, "tcav": result["primary_tcav"], "eligible": eligible,
            "reason": reason, "status": status, "records": int(result["selection_records"]),
            "diagnostics": {str(seed): dict(entry) for seed, entry in result["diagnostics"].items()}}


def projected_cavs_selection_bytes(config, embedding_bound=512, row_bound=8192, latent_bound=768):
    """Upper bound of one D/r/cavs_selection unit's cached bytes: CAV vectors, subgroup masks, per-family JSON."""
    cap = candidate_cap(latent_bound, config["candidate_cap_fraction"])
    return (cap * len(config["cav_seeds"]) * embedding_bound * 9 + 2 * cap * row_bound + 2 * row_bound * 8
            + cap * 64 * 1024 + 4 * 2**20)


def cavs_selection_unit(view, r, roles, predictor_output, family_output, frozen, dictionaries, config, device, *, fit=None):
    """D(r) CAVs and selection: per frozen family, CAV repetitions on the year-r cav_fit rows and raw seed-42 TCAV on the
    year-r concept_selection rows inside A (idea 1) and inside R (idea 2), eligibility independently per idea."""
    import numpy as np
    r = int(r)
    if device not in ("cpu", "cuda"):
        raise ValueError("CAV selection needs a resolved device: cpu or cuda")
    if set(view) != set(RULE_VIEW_KEYS):
        raise ValueError(f"CAV selection reads only {RULE_VIEW_KEYS}; outcomes are not accepted")
    _require_selection_settings(config)
    fit = fit_family_cavs if fit is None else fit
    family_rows = _family_rows(view, r, roles, family_output)
    cav_rows, cav_positions = _predictor_positions(view, r, roles, predictor_output, ("cav_fit",))
    selection_rows, selection_positions = _predictor_positions(view, r, roles, predictor_output, ("concept_selection",))
    entries, seeds = list(frozen.values()), [int(seed) for seed in config["cav_seeds"]]
    dimension = int(predictor_output["embedding_dim"])
    cav_embeddings = np.asarray(predictor_output["embeddings"])[cav_positions]
    selection_embeddings = np.asarray(predictor_output["embeddings"])[selection_positions]
    selection_gradients = np.asarray(predictor_output["gradients"])[selection_positions]
    cav_activations = _medoid_activations(cav_rows, cav_embeddings, entries, family_rows, dictionaries, config, device)
    selection_activations = _medoid_activations(selection_rows, selection_embeddings, entries, family_rows, dictionaries,
                                                config, device)
    cav_features, selection_features = view["X"][cav_rows].astype(np.float64), view["X"][selection_rows].astype(np.float64)
    patients = view["patient_id"]
    vectors = np.zeros((len(entries), len(seeds), dimension), dtype=np.float64)
    valid = np.zeros((len(entries), len(seeds)), dtype=bool)
    masks = {"A": np.zeros((len(entries), len(selection_rows)), dtype=bool),
             "R": np.zeros((len(entries), len(selection_rows)), dtype=bool)}
    families = []
    for position, entry in enumerate(entries):
        cavs = fit(cav_embeddings, cav_features, cav_activations[position], entry["rule"], patients[cav_rows], cav_rows, seeds)
        if set(cavs) != set(seeds):
            raise RuntimeError("CAV fitting must report every configured seed")
        a_mask = strict_activation_mask(selection_activations[position], entry["target"])
        r_mask = rule_mask(entry["rule"], selection_features, view["feature_names"])
        selection = frozen_subgroup_selection(selection_gradients, cavs, {"A": a_mask, "R": r_mask}, entry["family_id"])
        reference = freeze_reference_objects(entry["target"], entry["rule"], cavs)
        for column, seed in enumerate(seeds):
            valid[position, column] = cavs[seed]["valid"]
            if cavs[seed]["valid"]:
                vectors[position, column] = cavs[seed]["cav"]
        masks["A"][position], masks["R"][position] = a_mask, r_mask
        both = int((a_mask & r_mask).sum())
        families.append({
            "family_id": entry["family_id"], "medoid": [int(value) for value in entry["medoid"]],
            "activation_target": entry["activation_target"], "cutoff": entry["cutoff"], "target": dict(entry["target"]),
            "rule_identity": entry["rule_identity"], "rule_entry_identity": entry["identity"],
            "cav": {str(seed): _cav_description(cavs[seed]) for seed in seeds},
            "selection": {"idea1": _selection_description(selection["A"], "A"), "idea2": _selection_description(selection["R"], "R")},
            "inventory": {"selection_rows": int(len(selection_rows)), "A": int(a_mask.sum()), "R": int(r_mask.sum()),
                          "A_and_R": both, "A_not_R": int(a_mask.sum()) - both, "R_not_A": int(r_mask.sum()) - both,
                          "neither": int((~a_mask & ~r_mask).sum())},
            "reference_identity": reference["identity"]})
    eligible = {idea: [family["family_id"] for family in families if family["selection"][idea]["eligible"]] for idea, _ in IDEAS}
    eligible["both"] = [family_id for family_id in eligible["idea1"] if family_id in eligible["idea2"]]
    return {
        "cav_vectors": vectors, "cav_valid": valid, "cav_row_index": cav_rows.astype(np.int64),
        "selection_row_index": selection_rows.astype(np.int64), "selection_mask_A": masks["A"], "selection_mask_R": masks["R"],
        "_rows": int(len(selection_rows)), "selection_rows": int(len(selection_rows)), "reference_year": r, "cav_seeds": seeds, "primary_cav_seed": 42,
        "tcav_cutoffs": list(TCAV_ELIGIBILITY_CUTOFFS), "embedding_dim": dimension, "families": families, "eligible": eligible,
        "frozen_families": len(entries), "candidate_families": len(family_output["selection"]["candidates"]),
        "cav_rows": int(len(cav_rows)), "cav_patients": int(len(np.unique(patients[cav_rows]))),
        "selection_rows_patients": int(len(np.unique(patients[selection_rows]))),
        "cav_rows_identity": _identity_hash({"row_index": cav_rows}),
        "selection_rows_identity": _identity_hash({"row_index": selection_rows}),
        "negative_pool": "rule-negative rows at the minimum finite medoid activation of the cav_fit rows (low quantile 0)",
        "selection_identity": _identity_hash({"reference_year": r, "eligible": eligible,
                                              "references": [[family["family_id"], family["reference_identity"]] for family in families]}),
        "roles_identity": roles["identity"], "population_identity": view["fingerprint"]["population_identity"],
        "predictor_context_identity": predictor_output.get("context_identity"),
        "continuation_identity": predictor_output["continuation_identity"],
        "predictor_output_row_identity": predictor_output["output_row_identity"],
        "scaler_identity": family_output["scaler_identity"], "sae_identities": family_output["sae_identities"],
        "outcomes_read": False}


def run_cavs_selection_unit(cache, view, roles, config, device, *, fit=None):
    """Run D(r)/cavs_selection through the cache; P(r), families and rules are dependencies loaded from the cache."""
    if config_hash(config) != cache.config_hash:
        raise ValueError("Configuration differs from the cache configuration")
    if set(view) != set(RULE_VIEW_KEYS):
        raise ValueError(f"CAV selection reads only {RULE_VIEW_KEYS}; outcomes are not accepted")
    r = int(roles["reference_year"])
    return cache.run(f"D/{r}/cavs_selection",
                     lambda: cavs_selection_unit(view, r, roles, cache.load(f"P/{r}"), cache.load(f"D/{r}/families"),
                                                 load_frozen_rules(cache, r), load_reference_dictionaries(cache, r, device),
                                                 config, device, fit=fit),
                     projected_bytes=projected_cavs_selection_bytes(config))


def load_frozen_references(cache, r):
    """Frozen reference objects of year r by family ID (target, rule, CAV repetitions, identity, eligibility per idea),
    each rebuilt from the cached units and refused when its identity differs from the one recorded at freezing."""
    r = int(r)
    return _references_from_units(cache.load(f"D/{r}/cavs_selection"), load_frozen_rules(cache, r))


def _references_from_units(unit, rules):
    references = {}
    for position, family in enumerate(unit["families"]):
        entry = rules.get(family["family_id"])
        if entry is None or entry["rule_identity"] != family["rule_identity"] or entry["identity"] != family["rule_entry_identity"]:
            raise RuntimeError(f"Frozen rule of {family['family_id']} differs from the one the CAVs were fitted for")
        cavs = {}
        for column, seed in enumerate(unit["cav_seeds"]):
            record = {name: value for name, value in family["cav"][str(seed)].items() if name != "status"}
            record["cav"] = unit["cav_vectors"][position, column].copy() if unit["cav_valid"][position, column] else None
            cavs[int(seed)] = record
        reference = freeze_reference_objects(entry["target"], entry["rule"], cavs)
        if reference["identity"] != family["reference_identity"]:
            raise RuntimeError(f"Frozen reference of {family['family_id']} changed")
        assert_reference_identity(reference)
        reference.update(family_id=family["family_id"], medoid=list(family["medoid"]),
                         activation_target=family["activation_target"], cutoff=family["cutoff"],
                         eligible={idea: family["selection"][idea]["eligible"] for idea, _ in IDEAS})
        references[family["family_id"]] = reference
    return references


def cavs_selection_table(unit):
    """Compact inventory/selection rows: CAV status per seed, subgroup sizes, raw TCAV, eligibility and reasons per idea."""
    rows = []
    for family in unit["families"]:
        idea1, idea2 = family["selection"]["idea1"], family["selection"]["idea2"]
        rows.append({"family": family["family_id"], "medoid": ":".join(str(value) for value in family["medoid"]),
                     "A": family["activation_target"], "cutoff": round(family["cutoff"], 4),
                     "CAV 42/43/44": "/".join("ok" if family["cav"][str(seed)]["valid"] else "n/a" for seed in unit["cav_seeds"]),
                     "|A|": family["inventory"]["A"], "|R|": family["inventory"]["R"], "|A and R|": family["inventory"]["A_and_R"],
                     "A TCAV": None if idea1["tcav"] is None else round(idea1["tcav"], 3), "idea 1": idea1["status"],
                     "R TCAV": None if idea2["tcav"] is None else round(idea2["tcav"], 3), "idea 2": idea2["status"]})
    return rows


def projected_performance_bytes(config, latent_bound=768):
    """Upper bound of one D/r/reference_performance unit's cached bytes: one metrics record per family and idea."""
    cap = candidate_cap(latent_bound, config["candidate_cap_fraction"])
    return cap * len(IDEAS) * 32 * 1024 + 4 * 2**20


def reference_performance_unit(view, r, roles, predictor_output, family_output, references, dictionaries, config, device):
    """D(r) reference performance on the year-r evaluation rows for every (family, idea) eligible at selection, with the
    baseline q fitted on the year-r tabpfn_context, sae_discovery, rule_discovery and cav_fit rows inside the same frozen subgroup."""
    import numpy as np
    r = int(r)
    if device not in ("cpu", "cuda"):
        raise ValueError("Reference performance needs a resolved device: cpu or cuda")
    if set(view) != set(PERFORMANCE_VIEW_KEYS):
        raise ValueError(f"Reference performance reads only {PERFORMANCE_VIEW_KEYS}")
    if config["classification_threshold"] != 0.5 or config["confidence_level"] != 0.95:
        raise ValueError("Reference performance supports the 0.5 threshold and 95% intervals only")
    for reference in references.values():
        assert_reference_identity(reference)
    family_rows = _family_rows(view, r, roles, family_output)
    fitting_rows, fitting_positions = _predictor_positions(view, r, roles, predictor_output, FITTING_SIDE_ROLE_NAMES)
    evaluation_rows, evaluation_positions = _predictor_positions(view, r, roles, predictor_output, ("evaluation",))
    entries = [reference for reference in references.values() if any(reference["eligible"].values())]
    rows = np.concatenate([fitting_rows, evaluation_rows])
    positions = np.concatenate([fitting_positions, evaluation_positions])
    activations = _medoid_activations(rows, np.asarray(predictor_output["embeddings"])[positions], entries, family_rows,
                                      dictionaries, config, device)
    split = len(fitting_rows)
    labels, patients = view["y"], view["patient_id"]
    fitting_labels, evaluation_labels = labels[fitting_rows], labels[evaluation_rows]
    probabilities = np.asarray(predictor_output["probabilities"])[evaluation_positions, 1].astype(np.float64)
    fitting_features = view["X"][fitting_rows].astype(np.float64)
    evaluation_features = view["X"][evaluation_rows].astype(np.float64)
    results = []
    for position, reference in enumerate(entries):
        for idea, subgroup in IDEAS:
            if not reference["eligible"][idea]:
                continue
            if subgroup == "A":
                fit_mask = strict_activation_mask(activations[position, :split], reference["target"])
                evaluation_mask = strict_activation_mask(activations[position, split:], reference["target"])
            else:
                fit_mask = rule_mask(reference["rule"], fitting_features, view["feature_names"])
                evaluation_mask = rule_mask(reference["rule"], evaluation_features, view["feature_names"])
            baseline = fit_constant_baseline(fitting_labels[fit_mask])
            seed = _derived_seed("reference_performance", int(config["split_seed"]), r, reference["family_id"], idea)
            metrics = subgroup_reliability(evaluation_labels[evaluation_mask], probabilities[evaluation_mask],
                                           patients[evaluation_rows][evaluation_mask], baseline,
                                           config["bootstrap_replicates"], seed)
            results.append({"family_id": reference["family_id"], "medoid": list(reference["medoid"]), "idea": idea,
                            "subgroup": subgroup, "activation_target": reference["activation_target"],
                            "cutoff": reference["cutoff"], "reference_identity": reference["identity"],
                            "evaluation_rows": int(evaluation_mask.sum()), "evaluation_deaths": int(evaluation_labels[evaluation_mask].sum()),
                            "fitting_rows": int(fit_mask.sum()), "fitting_deaths": int(fitting_labels[fit_mask].sum()),
                            "bootstrap_seed": seed, "baseline": baseline, "metrics": metrics})
    for reference in references.values():
        assert_reference_identity(reference)
    not_computed = [{"family_id": reference["family_id"], "idea": idea, "reason": "not_eligible_at_selection"}
                    for reference in references.values() for idea, _ in IDEAS if not reference["eligible"][idea]]
    return {
        "rows": results, "not_computed": not_computed, "_rows": int(len(evaluation_rows)), "reference_year": r,
        "evaluation_role": "evaluation", "fitting_roles": list(FITTING_SIDE_ROLE_NAMES),
        "evaluation_rows": int(len(evaluation_rows)), "evaluation_patients": int(len(np.unique(patients[evaluation_rows]))),
        "evaluation_deaths": int(evaluation_labels.sum()), "fitting_rows": int(len(fitting_rows)),
        "fitting_deaths": int(fitting_labels.sum()),
        "evaluation_rows_identity": _identity_hash({"row_index": evaluation_rows}),
        "fitting_rows_identity": _identity_hash({"row_index": fitting_rows}),
        "bootstrap": {"replicates": int(config["bootstrap_replicates"]), "confidence_level": 0.95, "sampling_unit": "patient",
                      "seed": "sha256 of (reference_performance, split_seed, reference year, family, idea)"},
        "classification_threshold": 0.5, "references": {reference["family_id"]: reference["identity"] for reference in references.values()},
        "roles_identity": roles["identity"], "population_identity": view["fingerprint"]["population_identity"],
        "outcomes_read": "fitting roles and evaluation rows of the reference year only"}


def run_reference_performance_unit(cache, view, roles, config, device):
    """Run D(r)/reference_performance through the cache; frozen references are rebuilt and identity-checked from the cavs unit."""
    if config_hash(config) != cache.config_hash:
        raise ValueError("Configuration differs from the cache configuration")
    if set(view) != set(PERFORMANCE_VIEW_KEYS):
        raise ValueError(f"Reference performance reads only {PERFORMANCE_VIEW_KEYS}")
    r = int(roles["reference_year"])
    return cache.run(f"D/{r}/reference_performance",
                     lambda: reference_performance_unit(view, r, roles, cache.load(f"P/{r}"), cache.load(f"D/{r}/families"),
                                                        load_frozen_references(cache, r), load_reference_dictionaries(cache, r, device),
                                                        config, device),
                     projected_bytes=projected_performance_bytes(config))


def reference_performance_table(unit):
    """Flat rows for the performance section: counts, baseline q, Brier, skill with interval, confusion metrics, calibration."""
    def number(value, digits=4):
        return None if value is None else round(value, digits)
    rows = []
    for row in unit["rows"]:
        metrics, interval = row["metrics"], row["metrics"]["bootstrap"]["intervals"].get("brier_skill")
        rows.append({"family": row["family_id"], "idea": row["idea"], "subgroup": row["subgroup"],
                     "eval rows": row["evaluation_rows"], "deaths": row["evaluation_deaths"],
                     "q (fit rows)": number(row["baseline"]["q"]), "Brier": number(metrics["brier"]),
                     "baseline Brier": number(metrics["baseline_brier"]), "skill": number(metrics["brier_skill"]),
                     "skill 95% CI": None if interval is None else [round(value, 3) for value in interval],
                     "mean risk": number(metrics["mean_risk"]), "prevalence": number(metrics["prevalence"]),
                     "F1": number(metrics["f1"]), "sensitivity": number(metrics["sensitivity"]),
                     "specificity": number(metrics["specificity"]), "ECE": number(metrics["ece"]),
                     "unavailable": {name: reason for name, reason in metrics["unavailable_reasons"].items() if metrics.get(name) is None},
                     "skill CI unavailable": metrics["bootstrap"]["unavailable_reasons"].get("brier_skill") if interval is None else None})
    return rows


def fit_reference_rule(features, activations, target, patient_ids, feature_names, seed=42):
    """Zero-gate rule for a given frozen target inside one patient pool (descriptive rules); reference rules use the sweep."""
    import numpy as np
    from stable_rule_backend import discover_stable_rule_candidates, StableRuleBackendConfig
    from semantic_rules import RuleSetSelectionConfig, select_rule_set
    X = np.asarray(features)
    patients = np.asarray(_canonical_patient_ids(patient_ids))
    binary = strict_activation_mask(activations, target)
    if X.ndim != 2 or len(X) != len(patients) or len(X) != len(binary) or X.shape[1] != len(feature_names):
        raise ValueError("Rule fitting features/activations/patients/names must align")
    if len(np.unique(patients)) != len(patients):
        raise ValueError("Annual rule fitting requires one record per patient")
    fitting, tuning = _stratified_patient_partition(patients, binary.astype(int), (.5, .5), seed)
    fit_mask, tune_mask = np.isin(patients, fitting), np.isin(patients, tuning)
    result = {"rule": None, "failure_reason": None,
              "fitting_patients_identity": _identity_hash(fitting), "tuning_patients_identity": _identity_hash(tuning),
              "backend": {"bootstraps": 30, "trees_per_bootstrap": 50, "max_depth": 15},
              "selection_gates": {"min_precision": 0., "min_lift": 0., "min_marginal_recall": 0.},
              "candidate_count": 0}
    if not fit_mask.any() or not tune_mask.any() or not binary[fit_mask].any() or not binary[tune_mask].any():
        result["failure_reason"] = "insufficient_inner_rule_partitions"
        return result
    discovery = discover_stable_rule_candidates(X[fit_mask], binary[fit_mask].astype(int), feature_names,
        groups=patients[fit_mask], config=StableRuleBackendConfig(random_state=seed))
    result["candidate_count"] = len(discovery.candidates)
    selection = select_rule_set(discovery.candidates, X[tune_mask], binary[tune_mask].astype(int),
        RuleSetSelectionConfig(min_precision=0., min_lift=0., min_marginal_recall=0.), threshold_name="frozen_threshold")
    result["rule"] = selection.rule_set
    result["rule_identity"] = _identity_hash(selection.rule_set.to_dict())
    result["selection_diagnostics"] = {"feasible": selection.feasible, "reason": selection.reason,
                                      "search_method": selection.search_method}
    return result


def run_reference_primitive_checks():
    """Actual frozen encoding and rule backend; SAE training awaits diagnostic smoke."""
    import numpy as np
    import torch
    from sae import SAE
    fit_raw = np.asarray([[-1, 2], [1, 2]], dtype=np.float32)
    scaler = fit_reference_scaler(fit_raw)
    np.testing.assert_allclose(scaler.mean_, [0, 2])
    np.testing.assert_allclose(scaler.scale_, [1, 1])
    model = SAE(data_dimension=2, scaling_factor=1.5, type="ReLU")
    with torch.no_grad():
        model.encoder.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 0.]]))
        model.encoder.bias.zero_()
    dictionary = {"scaler": scaler, "run": {"model": model, "model_type": "ReLU"},
                  "scaler_identity": _identity_hash((scaler.mean_, scaler.scale_)),
                  "sae_identity": _sae_state_identity(model)}
    codes = encode_reference_dictionary(dictionary, np.asarray([[2, 3], [-2, 2]], dtype=np.float32), "cpu", 1)
    np.testing.assert_allclose(codes, [[2, 1, 0], [0, 0, 2]], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(scaler.mean_, [0, 2])
    X = np.asarray([[float(index)] for index in range(40)])
    target = {"valid": True, "cutoff": 19.5}
    rule = fit_reference_rule(X, X[:, 0], target, [f"rule-{index}" for index in range(40)], ["feature"])
    assert rule["failure_reason"] is None and rule["candidate_count"] > 0
    assert rule["rule"].mask(X).shape == (40,)
    assert rule["fitting_patients_identity"] != rule["tuning_patients_identity"]
    assert rule["selection_gates"] == {"min_precision": 0., "min_lift": 0., "min_marginal_recall": 0.}
    print("Frozen fitting-only scaler/actual SAE encoding and patient-inner-split rule backend fixtures passed")


def run_reference_sae_unit_checks(notebook_path="concept_subgroup_drift_pilot.ipynb"):
    """S(seed, r): shared per-year scaler, role-only rows, seed determinism, state round trip, encoding, failures."""
    import numpy as np
    import torch
    presets = _notebook_presets(json.loads(Path(notebook_path).read_text()))
    config = resolve_config(presets, "v0")
    years, seeds = config["reference_year_candidates"], config["sae_seeds"]
    assert config["profile"] == "v0" and years == [2007, 2008, 2009] and seeds == [42, 135, 215]
    population = synthetic_population(years=years, patients_per_year=64)
    roles = {r: reference_year_roles(population, r, config["split_seed"]) for r in years}
    dimension, year, rng = 8, population["year"], np.random.default_rng(11)

    def discovery_rows(r):
        return (year == r) & np.isin(population["patient_id"], roles[r]["reference_roles"]["sae_discovery"])

    def predictor_output(r):
        rows = np.flatnonzero(year >= r)
        embeddings = rng.normal(size=(len(rows), dimension)) * np.linspace(0.5, 2.0, dimension) + np.linspace(-1, 1, dimension)
        return {"row_index": rows.astype(np.int64), "probabilities": np.full((len(rows), 2), 0.5, dtype=np.float32),
                "embeddings": embeddings.astype(np.float32), "reference_year": r, "roles_identity": roles[r]["identity"],
                "population_identity": population["fingerprint"]["population_identity"],
                "output_row_identity": _identity_hash({"row_index": rows}), "embedding_dim": dimension,
                "context_identity": "fixture-context", "_rows": len(rows)}

    def refuses(call, error=ValueError):
        try:
            call()
        except error:
            return
        raise AssertionError(f"Expected {error.__name__}")

    outputs = {r: predictor_output(r) for r in years}
    expected = {r: int(discovery_rows(r).sum()) for r in years}
    assert all(count >= 3 for count in expected.values()), expected
    free = lambda: 100 * 2 ** 30
    base = Path(os.environ["RPI_OUT_DIR"])
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as directory:
        work = Path(directory)

        def open_cache(name, cfg=config):
            cache = UnitCache(work / name, cfg, "source-fixture", free_bytes=free)
            for r in years:
                assert cache.run(f"P/{r}", lambda r=r: dict(outputs[r]))["status"] == "ok"
            return cache

        calls = []

        def counting_fit(embeddings, cfg, device, seed=None):
            calls.append((seed, embeddings.shape))
            return fit_reference_dictionary(embeddings, cfg, device, seed=seed)

        cache = UnitCache(work / "cache", config, "source-fixture", free_bytes=free)
        blocked = run_sae_unit(cache, population, roles[2007], config, 42, "cpu", fit=counting_fit)
        assert (blocked["status"], blocked["reason"]) == ("unavailable", "dep_not_run:P/2007") and not calls
        for r in years:
            assert cache.run(f"P/{r}", lambda r=r: dict(outputs[r]))["status"] == "ok"
        records = {}
        for seed in seeds:
            for r in years:
                record = records[seed, r] = run_sae_unit(cache, population, roles[r], config, seed, "cpu", fit=counting_fit)
                assert record["status"] == "ok" and record["key"] == f"S/{seed}/{r}" and record["stage"] == "S", record
                assert record["rows"] == expected[r] and record["device"] == "cpu" and record["seconds"] >= 0
                assert record["dependencies"] == {f"P/{r}": cache._latest[f"P/{r}"]["id"]}
        assert calls == [(seed, (expected[r], dimension)) for seed in seeds for r in years], "One training per unit on discovery rows"
        keys = [f"S/{seed}/{r}" for seed in seeds for r in years]
        assert [row["status"] for row in cache.completeness_table(keys)] == ["complete"] * 9
        fitted = len(calls)
        assert run_sae_unit(cache, population, roles[2007], config, 42, "cpu", fit=counting_fit)["id"] == records[42, 2007]["id"]
        assert len(calls) == fitted, "A complete S unit was retrained"
        refuses(lambda: run_sae_unit(cache, population, roles[2007], dict(config, epochs=3), 42, "cpu"))

        unit = cache.load("S/42/2007")
        sample = outputs[2007]["embeddings"]
        selected = sample[discovery_rows(2007)[outputs[2007]["row_index"]]]
        latents = unit["encoder_weight"].shape[0]
        assert unit["latent_count"] == latents == int(dimension * config["sae"]["scaling_factor"])
        assert unit["encoder_weight"].dtype == unit["encoder_bias"].dtype == unit["decoder_bias"].dtype == np.float32
        assert unit["encoder_weight"].shape == (latents, dimension) and unit["encoder_bias"].shape == (latents,)
        assert unit["decoder_bias"].shape == (dimension,) and unit["decoder_tied_to_encoder"] is True
        for name in ("scaler_mean", "scaler_scale", "scaler_var"):
            assert unit[name].dtype == np.float64 and unit[name].shape == (dimension,)
        assert (unit["reference_year"], unit["seed"], unit["epochs"], unit["embedding_dim"]) == (2007, 42, config["epochs"], dimension)
        assert unit["discovery_rows"] == expected[2007] == unit["scaler_samples"] and unit["sae"]["model_type"] == "ReLU"
        assert all(len(unit[name]) == 64 for name in ("scaler_identity", "sae_identity", "identity", "discovery_rows_identity"))
        assert unit["roles_identity"] == roles[2007]["identity"] and unit["predictor_context_identity"] == "fixture-context"
        assert np.isfinite([unit["final_loss"], unit["final_reconstruction_mse"], unit["final_sparsity_penalty"]]).all()
        assert unit["final_loss"] > 0 and 0 <= unit["active_fraction"] <= 1 and 0 <= unit["dead_latents_on_discovery"] <= latents
        training = unit["training"]
        assert training["device"] == "cpu" and training["batch"] == "full" and training["batch_rows"] == expected[2007]
        assert training["train_seconds"] >= 0 and training["peak_gpu_bytes"] is None and training["epochs"] == config["epochs"]
        assert training["input_bytes"] == selected.nbytes
        scaled = (selected - unit["scaler_mean"]) / unit["scaler_scale"]
        codes = np.maximum(scaled @ unit["encoder_weight"].T + unit["encoder_bias"], 0)
        reconstruction = codes @ unit["encoder_weight"] + unit["decoder_bias"]
        mse = float(np.mean((reconstruction - scaled) ** 2))
        np.testing.assert_allclose(unit["final_reconstruction_mse"], mse, rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(unit["final_loss"], mse + config["sae"]["alpha"] * float(np.mean(np.abs(codes))), rtol=1e-3, atol=1e-6)
        assert projected_sae_bytes(config, dimension) >= sum(unit[name].nbytes for name in (
            "encoder_weight", "encoder_bias", "decoder_bias", "scaler_mean", "scaler_scale", "scaler_var"))
        assert projected_sae_bytes(config) == 4 * (768 * 512 + 768 + 512) + 24 * 512 + 4096

        for r in years:
            units = [cache.load(f"S/{seed}/{r}") for seed in seeds]
            assert len({entry["scaler_identity"] for entry in units}) == 1, "Scaler must be shared by every seed"
            for name in ("scaler_mean", "scaler_scale", "scaler_var"):
                assert len({entry[name].tobytes() for entry in units}) == 1, name
            rows = outputs[r]["embeddings"][discovery_rows(r)[outputs[r]["row_index"]]]
            np.testing.assert_allclose(units[0]["scaler_mean"], rows.mean(axis=0, dtype=np.float64), rtol=1e-6, atol=1e-9)
            np.testing.assert_allclose(units[0]["scaler_scale"], rows.std(axis=0, dtype=np.float64), rtol=1e-5)
            assert units[0]["scaler_samples"] == len(rows) == expected[r]
        assert len({cache.load(f"S/42/{r}")["scaler_identity"] for r in years}) == 3
        assert len({cache.load(f"S/{seed}/2007")["sae_identity"] for seed in seeds}) == 3, "Seeds must differ"
        stored = cache.load("P/2007")
        again = sae_unit(population, 2007, roles[2007], stored, config, 42, "cpu")
        assert again["sae_identity"] == unit["sae_identity"] and again["identity"] == unit["identity"]
        assert again["scaler_identity"] == unit["scaler_identity"] and again["final_loss"] == unit["final_loss"]
        assert np.array_equal(again["encoder_weight"], unit["encoder_weight"])

        mask = discovery_rows(2007)[outputs[2007]["row_index"]]
        poisoned = dict(stored, embeddings=np.where(mask[:, None], stored["embeddings"], np.float32("nan")))
        assert poisoned["embeddings"].dtype == np.float32 and np.isnan(poisoned["embeddings"][~mask]).all()
        twin = sae_unit(population, 2007, roles[2007], poisoned, config, 42, "cpu")
        assert twin["scaler_identity"] == unit["scaler_identity"] and twin["sae_identity"] == unit["sae_identity"]
        shifted = stored["embeddings"].copy()
        shifted[np.flatnonzero(mask)[0]] += 1.0
        moved = sae_unit(population, 2007, roles[2007], dict(stored, embeddings=shifted), config, 42, "cpu")
        assert moved["scaler_identity"] != unit["scaler_identity"] and moved["sae_identity"] != unit["sae_identity"]

        bad_outputs = [
            ("missing key", {key: value for key, value in stored.items() if key != "roles_identity"}),
            ("roles identity", dict(stored, roles_identity="0" * 64)),
            ("population identity", dict(stored, population_identity="0" * 64)),
            ("float64 embeddings", dict(stored, embeddings=stored["embeddings"].astype(np.float64))),
            ("row identity", dict(stored, output_row_identity="0" * 64)),
            ("dropped row", dict(stored, row_index=stored["row_index"][:-1], embeddings=stored["embeddings"][:-1],
                                 output_row_identity=_identity_hash({"row_index": stored["row_index"][:-1]}))),
            ("embedding dimension", dict(stored, embedding_dim=dimension + 1)),
            ("other year", outputs[2008]),
        ]
        for label, bad in bad_outputs:
            try:
                sae_unit(population, 2007, roles[2007], bad, config, 42, "cpu")
            except ValueError:
                pass
            else:
                raise AssertionError(f"Predictor output accepted: {label}")
        refuses(lambda: sae_unit(population, 2007, roles[2008], stored, config, 42, "cpu"))
        refuses(lambda: sae_unit(population, 2008, roles[2008], stored, config, 42, "cpu"))
        refuses(lambda: sae_unit(population, 2007, roles[2007], stored, config, 7, "cpu"))
        refuses(lambda: sae_unit(population, 2007, roles[2007], stored, dict(config, profile="analysis"), 42, "cpu"))
        for device in ("auto", "cuda:0"):
            refuses(lambda device=device: sae_unit(population, 2007, roles[2007], stored, config, 42, device))
        refuses(lambda: sae_unit(population, 2007, roles[2007], stored, config, 42, "cpu",
                                 expected_scaler_identity="0" * 64), RuntimeError)

        for seed in seeds:
            loaded = load_reference_sae(cache, 2007, seed, "cpu")
            record = cache.load(f"S/{seed}/2007")
            model = loaded["run"]["model"]
            assert _sae_state_identity(model) == record["sae_identity"] == loaded["sae_identity"]
            assert _identity_hash((loaded["scaler"].mean_, loaded["scaler"].scale_)) == record["scaler_identity"]
            assert model.num_latents == loaded["latent_count"] == record["latent_count"] and loaded["epochs"] == config["epochs"]
            assert (loaded["seed"], loaded["reference_year"], loaded["unit_identity"]) == (seed, 2007, record["identity"])
        loaded = load_reference_sae(cache, 2007, 42, "cpu")
        encoded = encode_with_reference_sae(loaded, sample, "cpu", 7)
        assert encoded.dtype == np.float32 and encoded.shape == (len(sample), latents)
        assert np.isfinite(encoded).all() and (encoded >= 0).all() and (encoded > 0).any()
        scaled = (sample - unit["scaler_mean"]) / unit["scaler_scale"]
        np.testing.assert_allclose(encoded, np.maximum(scaled @ unit["encoder_weight"].T + unit["encoder_bias"], 0),
                                   rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(encode_with_reference_sae(loaded, sample, "cpu", 1), encoded, rtol=1e-5, atol=1e-6)
        assert encode_with_reference_sae(loaded, sample[:0], "cpu").shape == (0, latents)
        assert _sae_state_identity(loaded["run"]["model"]) == unit["sae_identity"], "Encoding changed the frozen SAE"
        for bad in (sample[0], sample[:, :-1], np.where(np.arange(len(sample))[:, None] == 0, np.float32("inf"), sample)):
            refuses(lambda bad=bad: encode_with_reference_sae(loaded, bad, "cpu"))
        dictionaries = load_reference_dictionaries(cache, 2007, "cpu")
        assert list(dictionaries) == seeds and len({entry["scaler_identity"] for entry in dictionaries.values()}) == 1
        assert len({entry["sae_identity"] for entry in dictionaries.values()}) == 3

        target = cache._unit_path("S/215/2009") / "arrays.npz"
        original = target.read_bytes()
        with np.load(target) as stored_arrays:
            arrays = {name: stored_arrays[name] for name in stored_arrays.files}
        for name, change in (("encoder_weight", 1e-3), ("scaler_mean", 1.0)):
            np.savez(target, **dict(arrays, **{name: arrays[name] + arrays[name].dtype.type(change)}))
            refuses(lambda: load_reference_sae(cache, 2009, 215, "cpu"), RuntimeError)
            refuses(lambda: load_reference_dictionaries(cache, 2009, "cpu"), RuntimeError)
        target.write_bytes(original)
        assert len(load_reference_dictionaries(cache, 2009, "cpu")) == 3

        failing = open_cache("cache-failure")
        refuses(lambda: load_reference_sae(failing, 2007, 42, "cpu"), KeyError)
        assert run_sae_unit(failing, population, roles[2007], config, 42, "cpu")["status"] == "ok"

        def nan_fit(embeddings, cfg, device, seed=None):
            dictionary = fit_reference_dictionary(embeddings, cfg, device, seed=seed)
            with torch.no_grad():
                dictionary["run"]["model"].encoder.weight[0, 0] = float("nan")
            return dictionary

        def overflow_fit(embeddings, cfg, device, seed=None):
            dictionary = fit_reference_dictionary(embeddings, cfg, device, seed=seed)
            with torch.no_grad():
                dictionary["run"]["model"].encoder.weight.fill_(1e30)
            dictionary["sae_identity"] = _sae_state_identity(dictionary["run"]["model"])
            return dictionary

        record = run_sae_unit(failing, population, roles[2007], config, 135, "cpu", fit=nan_fit)
        assert record["status"] == "failed" and record["reason"].startswith("FloatingPointError: Nonfinite SAE parameters"), record
        record = run_sae_unit(failing, population, roles[2008], config, 135, "cpu", fit=overflow_fit)
        assert record["status"] == "failed" and record["reason"].startswith("FloatingPointError: Nonfinite SAE loss or outputs"), record
        assert not failing.is_complete("S/135/2007") and not failing._unit_path("S/135/2007").exists()
        assert run_sae_unit(failing, population, roles[2007], config, 215, "cpu")["status"] == "ok"
        executed = []
        for key in ("D/2007/families", "D/2007/rules"):
            dependent = failing.run(key, lambda key=key: executed.append(key))
            assert (dependent["status"], dependent["reason"]) == ("unavailable", "dep_failed:S/135/2007"), dependent
        assert not executed, "A dependent of a failed S unit was executed"
        table = {row["unit"]: row["status"] for row in failing.completeness_table(
            ["S/42/2007", "S/135/2007", "S/135/2008", "D/2007/families", "S/42/2008"])}
        assert table["S/42/2007"] == "complete" and table["S/42/2008"] == "not_run (stopped)"
        assert table["S/135/2007"].startswith("failed:FloatingPointError: Nonfinite SAE parameters")
        assert table["S/135/2008"].startswith("failed:FloatingPointError: Nonfinite SAE loss or outputs")
        assert table["D/2007/families"] == "unavailable:dep_failed:S/135/2007"
        refuses(lambda: load_reference_dictionaries(failing, 2007, "cpu"), KeyError)

        diverging = dict(config, sae=dict(config["sae"], learning_rate=1e30))
        diverging["config_hash"] = config_hash(diverging)
        diverged = open_cache("cache-diverged", diverging)
        record = run_sae_unit(diverged, population, roles[2007], diverging, 42, "cpu")
        assert record["status"] == "failed" and record["reason"].startswith("FloatingPointError: Nonfinite"), record
        assert not diverged.is_complete("S/42/2007") and diverged.run("D/2007/families", lambda: {})["status"] == "unavailable"

        def shifted_fit(embeddings, cfg, device, seed=None):
            return fit_reference_dictionary(embeddings + np.float32(1.0), cfg, device, seed=seed)

        unequal = open_cache("cache-scaler")
        assert run_sae_unit(unequal, population, roles[2007], config, 42, "cpu")["status"] == "ok"
        record = run_sae_unit(unequal, population, roles[2007], config, 135, "cpu", fit=shifted_fit)
        assert record["status"] == "failed" and record["reason"].startswith("RuntimeError: Reference scaler differs"), record

        if _cuda_available():
            import gc
            # Initialize persistent backend state before measuring per-unit memory growth.
            warmup = sae_unit(population, 2007, roles[2007], stored, config, 42, "cuda")
            del warmup
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            on_gpu = sae_unit(population, 2007, roles[2007], stored, config, 42, "cuda")
            assert on_gpu["training"]["device"] == "cuda" and on_gpu["training"]["peak_gpu_bytes"] > 0
            after = torch.cuda.memory_allocated()
            assert after <= before + 8 * 2 ** 20, f"GPU memory was not released at unit end: before={before} after={after} delta={after - before} bytes"
            assert on_gpu["scaler_identity"] == unit["scaler_identity"]
            for name in ("encoder_weight", "encoder_bias", "decoder_bias"):
                np.testing.assert_allclose(on_gpu[name], unit[name], rtol=1e-3, atol=1e-4)
            codes_gpu = encode_with_reference_sae(load_reference_sae(cache, 2007, 42, "cuda"), sample, "cuda", 7)
            np.testing.assert_allclose(codes_gpu, encoded, rtol=1e-3, atol=1e-4)
        else:
            print("CUDA tolerance checks skipped: CUDA unavailable here; GPU validation pending")
    print("Reference SAE unit fixtures passed")


def _planted_activations(rows, columns):
    """Nonnegative float32 columns built from orthogonal zero-mean Hadamard basis vectors: exact planted correlations."""
    import numpy as np
    basis = np.array([[1.0]])
    while len(basis) < rows:
        basis = np.block([[basis, basis], [basis, -basis]])
    built = []
    for weights in columns:
        if weights is None:
            built.append(np.zeros(rows))
        elif isinstance(weights, (int, float)):
            built.append(np.full(rows, float(weights)))
        else:
            built.append(sum(weight * basis[:, index] for index, weight in weights.items())
                         + sum(abs(weight) for weight in weights.values()))
    return np.stack(built, axis=1).astype(np.float32)


def _same_family_results(left, right):
    """Equal structure, exact counts and ties, correlations within float64 matmul rounding."""
    import numpy as np
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert [(f["id"], f["kind"], f["members"], f["medoid"]) for f in a["families"]] == [
            (f["id"], f["kind"], f["members"], f["medoid"]) for f in b["families"]]
        scores = lambda result: [(f["min_correlation"], f["median_correlation"], f["medoid_mean_correlation"])
                                 for f in result["families"]]
        np.testing.assert_allclose(scores(a), scores(b), rtol=0, atol=1e-9)
        assert (a["counts"], a["chance"], a["ties"], a["singletons"], a["per_seed"]) == (
            b["counts"], b["chance"], b["ties"], b["singletons"], b["per_seed"])


def _check_family_construction(device):
    """Planted structures through build_factor_families on the given device; returns the results for comparison."""
    import numpy as np
    seeds, rows, collected = [42, 135, 215], 64, []

    def build(columns, **options):
        matrices = [_planted_activations(rows, per_seed) for per_seed in columns]
        result = build_factor_families(matrices, seeds, device=device, id_prefix="2007/42", **options)
        collected.append(result)
        return result

    def refuses(call):
        try:
            call()
        except ValueError:
            return
        raise AssertionError("Invalid family input accepted")

    def family_ids(result, kind):
        return sorted(family["id"] for family in result["families"] if family["kind"] == kind)

    noise = lambda index: 20 + index
    # (a) four planted triangles: same signal plus small seed-specific noise; positions differ per seed.
    order = [[2, 0, 3, 1], [3, 1, 0, 2], [1, 3, 2, 0]]
    noise_weight = {0: (.1, .3, .5), 1: (.2, .05, .4), 2: (.5, .3, .1), 3: (.3, .1, .6)}
    columns = [[{signal + 1: 1.0, noise(10 * s + signal): noise_weight[signal][s]} for signal in order[s]] for s in range(3)]
    result = build(columns)
    assert result["counts"] == {"full": 4, "partial": 0, "chain": 0, "inconsistent": 0, "singleton": 0}, result["counts"]
    assert result["device"] == device and result["precision"] == "float64" and result["chance"] is None
    assert all(entry["dead"] == [] and entry["undefined_correlation"] == [] and entry["usable"] == 4
               for entry in result["per_seed"]) and result["ties"] == {"count": 0, "events": []}
    expected_members = {signal: [[seeds[s], order[s].index(signal)] for s in range(3)] for signal in range(4)}
    expected_ids = sorted("2007/42/" + ",".join(f"{a}:{b}" for a, b in members) for members in expected_members.values())
    assert family_ids(result, "full") == expected_ids, family_ids(result, "full")
    matrices = [_planted_activations(rows, per_seed) for per_seed in columns]
    for family in result["families"]:
        assert family["members"] == sorted(family["members"]) and family["size"] == 3 and family["links"] == 3
        signal = next(s for s, members in expected_members.items() if members == family["members"])
        values = [matrices[seeds.index(seed)][:, factor].astype(np.float64) for seed, factor in family["members"]]
        reference = np.corrcoef(np.stack(values))
        pairwise = [reference[0, 1], reference[0, 2], reference[1, 2]]
        np.testing.assert_allclose([family["min_correlation"], family["median_correlation"]],
                                   [min(pairwise), float(np.median(pairwise))], rtol=0, atol=1e-12)
        means = [(reference[k].sum() - 1) / 2 for k in range(3)]
        assert family["medoid"] == family["members"][int(np.argmax(means))], (family, means)
        smallest_noise = int(np.argmin(noise_weight[signal]))
        assert family["medoid"][0] == seeds[smallest_noise], "Medoid is the member with the least planted noise"
        np.testing.assert_allclose(family["medoid_mean_correlation"], max(means), atol=1e-12)
    assert {tuple(family["medoid"])[0] for family in result["families"]} == {42, 135, 215}

    # (b) a pair present in two seeds only, (a) triangles beside it, and an unmatched latent.
    columns = [[{1: 1.0, noise(0): .1}, {2: 1.0, noise(1): .1}, {5: 1.0, 2: .3}],
               [{1: 1.0, noise(2): .1}, {2: 1.0, noise(3): .1}, {5: 1.0, 1: .3}],
               [{1: 1.0, noise(4): .1}, {2: 1.0, noise(5): .1}, {6: 1.0, 1: .3}]]
    result = build(columns)
    assert result["counts"] == {"full": 2, "partial": 1, "chain": 0, "inconsistent": 0, "singleton": 1}, result["counts"]
    assert family_ids(result, "partial") == ["2007/42/42:2,135:2"] and result["singletons"] == [[215, 2]]
    assert family_ids(result, "full") == ["2007/42/42:0,135:0,215:0", "2007/42/42:1,135:1,215:1"]
    partial, = [family for family in result["families"] if family["kind"] == "partial"]
    assert partial["size"] == 2 and partial["links"] == 1 and partial["min_correlation"] == partial["median_correlation"]
    assert partial["medoid"] == [42, 2], "A two-member tie goes to the lowest (seed, factor)"

    # (d) a chain: one factor per seed, only two of three pairs linked, is not full.
    columns = [[{1: 1.0}, {3: 1.0}], [{1: 1.0, 2: 1.0}], [{2: 1.0}, {3: 1.0, 1: .5}]]
    result = build(columns)
    assert family_ids(result, "chain") == ["2007/42/42:0,135:0,215:0"], result["families"]
    assert family_ids(result, "partial") == ["2007/42/42:1,215:1"] and family_ids(result, "full") == []
    chain, = [family for family in result["families"] if family["kind"] == "chain"]
    assert chain["size"] == 3 and chain["links"] == 2 and result["counts"]["chain"] == 1 and result["counts"]["full"] == 0
    assert chain["medoid"] == [135, 0], "The middle factor correlates most with the others"

    # (c) split/merge: two factors of one seed reach the same component through the other seeds; never forced.
    columns = [[{1: 1.0}, {2: 1.0}], [{1: 1.1, 2: 1.0}], [{2: 1.0}]]
    result = build(columns)
    assert result["counts"] == {"full": 0, "partial": 0, "chain": 0, "inconsistent": 1, "singleton": 0}, result["counts"]
    inconsistent, = result["families"]
    assert inconsistent["kind"] == "inconsistent" and inconsistent["members"] == [[42, 0], [42, 1], [135, 0], [215, 0]]
    assert inconsistent["links"] == 3 and inconsistent["id"] == "2007/42/42:0,42:1,135:0,215:0"
    assert family_candidates(result["families"], {}, 30, .1)["candidates"] == [], "Inconsistent families are never candidates"

    # (e) dead and constant latents are excluded and counted separately; a single-row spike stays usable.
    spike = np.zeros(rows)
    spike[5] = 1.0
    matrices = [_planted_activations(rows, [None, {1: 1.0, noise(0): .1}, 3.0]),
                _planted_activations(rows, [{1: 1.0, noise(1): .1}, None, None]),
                _planted_activations(rows, [2.0, {1: 1.0, noise(2): .1}, {1: 0.0}])]
    matrices[2][:, 2] = spike
    result = build_factor_families(matrices, seeds, device=device, id_prefix="2007/42")
    assert [(entry["seed"], entry["dead"], entry["undefined_correlation"], entry["usable"]) for entry in result["per_seed"]] == [
        (42, [0], [2], 1), (135, [1, 2], [], 1), (215, [], [0], 2)], result["per_seed"]
    assert family_ids(result, "full") == ["2007/42/42:1,135:0,215:1"] and result["singletons"] == [[215, 2]]
    placed = {tuple(member) for family in result["families"] for member in family["members"]} | {
        tuple(node) for node in result["singletons"]}
    assert placed == {(42, 1), (135, 0), (215, 1), (215, 2)}, "Excluded latents must not appear as nodes"
    assert result["counts"] == {"full": 1, "partial": 0, "chain": 0, "inconsistent": 0, "singleton": 1}
    everything_dead = build_factor_families([np.zeros((rows, 2), np.float32)] * 3, seeds, device=device, id_prefix="x")
    assert everything_dead["families"] == [] and everything_dead["counts"]["singleton"] == 0
    assert [entry["dead"] for entry in everything_dead["per_seed"]] == [[0, 1]] * 3

    # (f) exact ties go to the lowest factor ID (not the lowest column) and are recorded.
    duplicated = {1: 1.0, noise(1): .1}
    matrices = [_planted_activations(rows, [{1: 1.0, noise(0): .1}]), _planted_activations(rows, [duplicated, duplicated]),
                _planted_activations(rows, [{1: 1.0, noise(2): .1}])]
    result = build_factor_families(matrices, seeds, device=device, id_prefix="2007/42",
                                   factor_ids=[[3], [9, 4], [8]])
    assert family_ids(result, "full") == ["2007/42/42:3,135:4,215:8"] and result["singletons"] == [[135, 9]], result
    assert result["ties"]["count"] == 2, result["ties"]
    assert sorted((event["pair"][1], event["direction"], event["source"], event["tied"], event["chosen"])
                  for event in result["ties"]["events"]) == [(135, "left_to_right", 3, [4, 9], 4), (215, "right_to_left", 8, [4, 9], 4)]
    twice = build_factor_families(matrices, seeds, device=device, id_prefix="2007/42", factor_ids=[[3], [9, 4], [8]])
    assert twice == result

    # (k) deterministic IDs, members and medoids across runs.
    columns = [[{signal + 1: 1.0, noise(10 * s + signal): .1 * (s + 1)} for signal in order[s]] for s in range(3)]
    runs = [build(columns, permutations=4, rng_seed=[42, 2007]) for _ in range(2)]
    assert json.dumps(runs[0], sort_keys=True) == json.dumps(runs[1], sort_keys=True)

    # (h) permutation counts: reproducible for one seed, different for another, observed above the chance maximum.
    triangles = [[{signal + 1: 1.0, noise(10 * s + signal): .1 * (s + 1)} for signal in
                  np.random.default_rng(100 + s).permutation(8)] for s in range(3)]
    first = build(triangles, permutations=20, rng_seed=[42, 2007])
    again = build(triangles, permutations=20, rng_seed=[42, 2007])
    other = build(triangles, permutations=20, rng_seed=[42, 2008])
    assert first["chance"] == again["chance"] and first["chance"]["per_permutation"] != other["chance"]["per_permutation"]
    chance = first["chance"]
    assert chance["permutations"] == len(chance["per_permutation"]) == 20 and chance["observed"] == first["counts"]
    assert first["counts"]["full"] == 8 and chance["max"]["full"] < first["counts"]["full"], chance
    assert chance["rng_seed"] == [42, 2007] and first["families"] == build(triangles)["families"]
    for key in FAMILY_COUNT_KEYS:
        values = [row[key] for row in chance["per_permutation"]]
        assert chance["mean"][key] == sum(values) / 20 and chance["max"][key] == max(values)
    assert all(set(row) == set(FAMILY_COUNT_KEYS) for row in chance["per_permutation"])
    assert build(triangles, permutations=0)["chance"] is None

    # Invalid inputs.
    good = [_planted_activations(rows, [{1: 1.0}])] * 3
    refuses(lambda: build_factor_families(good, seeds, device="auto", id_prefix="x"))
    refuses(lambda: build_factor_families(good[:2] + [good[0][:-1]], seeds, device=device, id_prefix="x"))
    refuses(lambda: build_factor_families(good, [42, 42, 215], device=device, id_prefix="x"))
    refuses(lambda: build_factor_families(good[:1], seeds[:1], device=device, id_prefix="x"))
    refuses(lambda: build_factor_families(good[:2], seeds, device=device, id_prefix="x"))
    refuses(lambda: build_factor_families(good, seeds, device=device, id_prefix="x", permutations=-1))
    refuses(lambda: build_factor_families(good, seeds, device=device, id_prefix=""))
    refuses(lambda: build_factor_families(good, seeds, device=device, id_prefix="x", factor_ids=[[0], [0], [0, 1]]))
    refuses(lambda: build_factor_families(good, seeds, device=device, id_prefix="x", factor_ids=[[0], [0], [-1]]))
    refuses(lambda: build_factor_families([m.astype(np.int32) for m in good], seeds, device=device, id_prefix="x"))
    nonfinite = good[0].copy()
    nonfinite[0, 0] = np.nan
    refuses(lambda: build_factor_families([nonfinite] + good[1:], seeds, device=device, id_prefix="x"))
    return collected


def _check_pearson(device):
    import numpy as np
    rng = np.random.default_rng(3)
    left = rng.gamma(2.0, size=(50, 7)).astype(np.float32) * (rng.random((50, 7)) > .3)
    right = rng.gamma(1.5, size=(50, 9)).astype(np.float32) * (rng.random((50, 9)) > .2)
    left[:, 0] += 1e-3 * np.arange(50, dtype=np.float32)
    right[:, 0] += 1e-3 * np.arange(50, dtype=np.float32)
    reference = np.corrcoef(left.astype(np.float64).T, right.astype(np.float64).T)[:7, 7:]
    got = pearson_matrix(left, right, device)
    assert got.shape == (7, 9) and got.dtype == np.float64
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-10)
    np.testing.assert_allclose(pearson_matrix(left, left, device).diagonal(), np.ones(7), atol=1e-12)
    for call in (lambda: pearson_matrix(left, np.ones((50, 2), np.float32), device),
                 lambda: pearson_matrix(left, right[:-1], device),
                 lambda: pearson_matrix(left, np.where(np.arange(50)[:, None] == 0, np.nan, right), device),
                 lambda: pearson_matrix(left, right, "auto")):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("Invalid correlation input accepted")
    return got


def _check_candidates():
    def family(identifier, kind="full", medoid=(42, 0)):
        return {"id": identifier, "kind": kind, "medoid": list(medoid), "members": [list(medoid)]}

    import numpy as np
    assert candidate_cap(288, .1) == 29 and candidate_cap(30, .1) == 3 and candidate_cap(31, .1) == 4
    positives = {"b": 7, "a": 7, "c": 10, "d": 3, "e": 9, "p": 100}
    families = [family(name) for name in "bacde"] + [family("p", "partial"), family("i", "inconsistent")]
    vectors = {name: np.array([1.0] * count + [0.0] * (12 - count) + [-1.0]) for name, count in positives.items()}
    chosen = family_candidates(families, vectors, 30, .1)
    assert [(record["family_id"], record["positive_count"]) for record in chosen["candidates"]] == [
        ("c", 10), ("e", 9), ("a", 7)], chosen
    assert [record["rank"] for record in chosen["candidates"]] == [1, 2, 3] and chosen["cap"] == 3
    assert [(record["family_id"], record["rank"]) for record in chosen["dropped_by_cap"]] == [("b", 4), ("d", 5)]
    assert chosen["full_families"] == 5 and chosen["num_latents"] == 30 and chosen["fraction"] == .1
    big = [family(f"f{index:02d}", medoid=(42, index)) for index in range(40)]
    activations = {item["id"]: np.arange(12.0) % (1 + index % 7) for index, item in enumerate(big)}
    wide = family_candidates(big, activations, 288, .1)
    assert wide["cap"] == 29 and len(wide["candidates"]) == 29 and len(wide["dropped_by_cap"]) == 11
    counts = [record["positive_count"] for record in wide["candidates"] + wide["dropped_by_cap"]]
    assert counts == sorted(counts, reverse=True)
    tied = [record["family_id"] for record in wide["candidates"] + wide["dropped_by_cap"] if record["positive_count"] == counts[0]]
    assert tied == sorted(tied)
    few = family_candidates(big[:5], activations, 288, .1)
    assert len(few["candidates"]) == 5 and few["dropped_by_cap"] == [] and few["cap"] == 29
    assert family_candidates([family("p", "partial")], {}, 288, .1)["candidates"] == []
    for bad in ({}, {name: vector[:-1] * np.nan for name, vector in vectors.items()},
                {name: vector.reshape(1, -1) for name, vector in vectors.items()}):
        try:
            family_candidates(families, bad, 30, .1)
        except ValueError:
            continue
        raise AssertionError("Invalid medoid activations accepted")
    try:
        family_candidates(families + [family("a")], vectors, 30, .1)
    except ValueError:
        return
    raise AssertionError("Duplicate family IDs accepted")


def _check_family_units(config, population, years, seeds):
    """D(r)/families through the unit cache on a tiny real SAE system."""
    import inspect
    import numpy as np
    import torch
    roles = {r: reference_year_roles(population, r, config["split_seed"]) for r in years}
    dimension, year, rng = 8, population["year"], np.random.default_rng(23)
    view = matching_population_view(population)
    assert set(view) == set(MATCHING_VIEW_KEYS) and "y" not in view and "post_death" not in view
    names = set(inspect.signature(family_unit).parameters) | set(inspect.signature(run_family_unit).parameters)
    assert not names & {"y", "outcome", "outcomes", "labels", "population"}, names

    def predictor_output(r):
        rows = np.flatnonzero(year >= r)
        embeddings = rng.normal(size=(len(rows), dimension)) * np.linspace(.5, 2., dimension) + np.linspace(-1, 1, dimension)
        return {"row_index": rows.astype(np.int64), "probabilities": np.full((len(rows), 2), .5, dtype=np.float32),
                "embeddings": embeddings.astype(np.float32), "reference_year": r, "roles_identity": roles[r]["identity"],
                "population_identity": population["fingerprint"]["population_identity"],
                "output_row_identity": _identity_hash({"row_index": rows}), "embedding_dim": dimension,
                "context_identity": "fixture-context", "_rows": len(rows)}

    def refuses(call, error=ValueError):
        try:
            call()
        except error:
            return
        raise AssertionError(f"Expected {error.__name__}")

    outputs = {r: predictor_output(r) for r in years}
    free = lambda: 100 * 2 ** 30
    base = Path(os.environ["RPI_OUT_DIR"])
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as directory:
        work = Path(directory)

        def open_cache(name, trained=True):
            cache = UnitCache(work / name, config, "source-fixture", free_bytes=free)
            for r in years:
                assert cache.run(f"P/{r}", lambda r=r: dict(outputs[r]))["status"] == "ok"
            if trained:
                for seed in seeds:
                    for r in years:
                        assert run_sae_unit(cache, population, roles[r], config, seed, "cpu")["status"] == "ok"
            return cache

        calls = []

        def counting_build(*arguments, **options):
            calls.append(options["id_prefix"])
            return build_factor_families(*arguments, **options)

        cache = open_cache("cache", trained=False)
        blocked = run_family_unit(cache, view, roles[2007], config, "cpu", build=counting_build)
        assert (blocked["status"], blocked["reason"]) == ("unavailable", "dep_not_run:S/42/2007") and not calls
        for seed in seeds:
            for r in years:
                assert run_sae_unit(cache, population, roles[r], config, seed, "cpu")["status"] == "ok"
        refuses(lambda: run_family_unit(cache, population, roles[2007], config, "cpu"))
        refuses(lambda: run_family_unit(cache, view, roles[2007], dict(config, epochs=3), "cpu"))
        refuses(lambda: family_unit(population, 2007, roles[2007], outputs[2007],
                                    load_reference_dictionaries(cache, 2007, "cpu"), config, "cpu"))
        records = {}
        for r in years:
            record = records[r] = run_family_unit(cache, view, roles[r], config, "cpu", build=counting_build)
            assert record["status"] == "ok" and record["key"] == f"D/{r}/families" and record["stage"] == "D.families", record
            assert record["dependencies"] == {f"P/{r}": cache._latest[f"P/{r}"]["id"],
                                              **{f"S/{seed}/{r}": cache._latest[f"S/{seed}/{r}"]["id"] for seed in seeds}}
        assert calls == [f"{r}/{config['split_seed']}" for r in years]
        again = run_family_unit(cache, view, roles[2007], config, "cpu", build=counting_build)
        assert again["id"] == records[2007]["id"] and len(calls) == len(years), "A complete D unit was rebuilt"

        for r in years:
            unit = cache.load(f"D/{r}/families")
            reference = roles[r]["reference_roles"]
            members = set(reference["rule_discovery"]) | set(reference["cav_fit"])
            expected = np.flatnonzero((year == r) & np.isin(population["patient_id"], sorted(members)))
            assert expected.size >= 4 and np.array_equal(unit["row_index"], expected), "Matching rows are year-r rule_discovery + cav_fit"
            assert records[r]["rows"] == unit["matching_rows"] == len(expected) and unit["outcomes_read"] is False
            assert unit["matching_rows_identity"] == _identity_hash({"row_index": expected})
            assert unit["matching_roles"] == ["rule_discovery", "cav_fit"] and unit["reference_year"] == r
            assert not set(roles[r]["row_role"][expected]) - {"rule_discovery", "cav_fit"}
            dictionaries = load_reference_dictionaries(cache, r, "cpu")
            latents = unit["num_latents"]
            assert latents == dictionaries[seeds[0]]["latent_count"] == int(dimension * config["sae"]["scaling_factor"])
            sample = outputs[r]["embeddings"][np.searchsorted(outputs[r]["row_index"], expected)]
            matrices = [encode_with_reference_sae(dictionaries[seed], sample, "cpu", config["encoding_batch_size"]) for seed in seeds]
            direct = build_factor_families(matrices, seeds, device="cpu", id_prefix=f"{r}/{config['split_seed']}",
                                           permutations=config["chance_permutations"], rng_seed=[config["split_seed"], r])
            assert unit["families"] == json.loads(json.dumps(direct["families"])), "Unit differs from direct construction"
            assert unit["construction"]["counts"] == direct["counts"] and unit["construction"]["chance"] == direct["chance"]
            assert unit["construction"]["chance"]["permutations"] == config["chance_permutations"] == 5
            assert unit["singleton_table"].tolist() == direct["singletons"]
            per_seed = {entry["seed"]: entry for entry in unit["construction"]["per_seed"]}
            for seed in seeds:
                entry = per_seed[seed]
                assert unit["dead_latents"][str(seed)] == len(entry["dead"]) and unit["undefined_latents"][str(seed)] == len(entry["undefined_correlation"])
                assert entry["usable"] + len(entry["dead"]) + len(entry["undefined_correlation"]) == entry["latents"] == latents
            nodes = sum(family["size"] for family in unit["families"]) + len(unit["singleton_table"])
            assert nodes == sum(entry["usable"] for entry in per_seed.values())
            assert unit["member_table"].dtype == np.int64 and unit["member_table"].shape == (sum(f["size"] for f in unit["families"]), 5)
            selection = unit["selection"]
            assert selection["cap"] == candidate_cap(latents, config["candidate_cap_fraction"]) and len(selection["candidates"]) <= selection["cap"]
            kinds = {family["id"]: family["kind"] for family in unit["families"]}
            assert all(kinds[record["family_id"]] == "full" for record in selection["candidates"] + selection["dropped_by_cap"])
            assert selection["full_families"] == unit["construction"]["counts"]["full"] == len(selection["candidates"]) + len(selection["dropped_by_cap"])
            assert unit["candidate_activations"].dtype == np.float32 and unit["candidate_activations"].shape == (len(selection["candidates"]), len(expected))
            for index, record in enumerate(selection["candidates"]):
                seed, factor = record["medoid"]
                np.testing.assert_array_equal(unit["candidate_activations"][index], matrices[seeds.index(seed)][:, factor])
                assert record["positive_count"] == int((unit["candidate_activations"][index] > 0).sum())
                assert unit["candidate_table"][index].tolist() == [record["rank"], seed, factor, record["positive_count"]]
            counts = [record["positive_count"] for record in selection["candidates"] + selection["dropped_by_cap"]]
            assert counts == sorted(counts, reverse=True)
            assert unit["scaler_identity"] == dictionaries[seeds[0]]["scaler_identity"]
            assert unit["sae_identities"] == {str(seed): dictionaries[seed]["sae_identity"] for seed in seeds}
            assert unit["roles_identity"] == roles[r]["identity"] and unit["split_seed"] == config["split_seed"]
            poisoned = dict(outputs[r], embeddings=np.where(np.isin(outputs[r]["row_index"], expected)[:, None],
                                                           outputs[r]["embeddings"], np.float32("nan")))
            twin = family_unit(view, r, roles[r], poisoned, dictionaries, config, "cpu")
            assert twin["families"] == unit["families"] and twin["selection"] == unit["selection"], "Non-matching rows were read"
            assert twin["matching_rows_identity"] == unit["matching_rows_identity"]
            refuses(lambda: family_unit(view, r, roles[r], outputs[r], dict(reversed(list(dictionaries.items()))), config, "cpu"))
            refuses(lambda: family_unit(view, r, roles[r], outputs[r], dictionaries, config, "auto"))
            refuses(lambda: family_unit(dict(view, y=population["y"]), r, roles[r], outputs[r], dictionaries, config, "cpu"))
        assert projected_family_bytes(config) >= max(sum(value.nbytes for value in cache.load(f"D/{r}/families").values()
                                                         if isinstance(value, np.ndarray)) for r in years)

        failing = open_cache("cache-failure", trained=False)

        def nan_fit(embeddings, cfg, device, seed=None):
            dictionary = fit_reference_dictionary(embeddings, cfg, device, seed=seed)
            with torch.no_grad():
                dictionary["run"]["model"].encoder.weight[0, 0] = float("nan")
            return dictionary

        assert run_sae_unit(failing, population, roles[2007], config, 42, "cpu")["status"] == "ok"
        assert run_sae_unit(failing, population, roles[2007], config, 135, "cpu", fit=nan_fit)["status"] == "failed"
        assert run_sae_unit(failing, population, roles[2007], config, 215, "cpu")["status"] == "ok"
        before = len(calls)
        record = run_family_unit(failing, view, roles[2007], config, "cpu", build=counting_build)
        assert (record["status"], record["reason"]) == ("unavailable", "dep_failed:S/135/2007") and len(calls) == before
        assert not failing.is_complete("D/2007/families") and not failing._unit_path("D/2007/families").exists()
        table = {row["unit"]: row["status"] for row in failing.completeness_table(["D/2007/families", "D/2008/families"])}
        assert table == {"D/2007/families": "unavailable:dep_failed:S/135/2007", "D/2008/families": "not_run (stopped)"}
        for name in ("D/2007/rules", "D/2007/cavs_selection"):
            assert failing.run(name, lambda: {})["status"] == "unavailable"

        def broken_build(*arguments, **options):
            raise FloatingPointError("injected")

        recovered = open_cache("cache-build-failure")
        record = run_family_unit(recovered, view, roles[2007], config, "cpu", build=broken_build)
        assert record["status"] == "failed" and record["reason"] == "FloatingPointError: injected"
        assert not recovered._unit_path("D/2007/families").exists()
        assert run_family_unit(recovered, view, roles[2007], config, "cpu")["status"] == "ok"

        if _cuda_available():
            gpu = family_unit(view, 2007, roles[2007], outputs[2007], load_reference_dictionaries(recovered, 2007, "cuda"),
                              config, "cuda")
            cpu = recovered.load("D/2007/families")
            assert gpu["construction"]["device"] == "cuda" and cpu["construction"]["device"] == "cpu"
            assert [(f["id"], f["kind"], f["medoid"]) for f in gpu["families"]] == [(f["id"], f["kind"], f["medoid"]) for f in cpu["families"]]
            assert gpu["selection"] == cpu["selection"] and gpu["construction"]["chance"] == cpu["construction"]["chance"]
            np.testing.assert_allclose(gpu["candidate_activations"], cpu["candidate_activations"], rtol=1e-3, atol=1e-4)
    return records


def run_family_checks(notebook_path="concept_subgroup_drift_pilot.ipynb"):
    """Anchor-free factor families: planted structures, exclusions, ties, chance check, cap, unit wiring, no outcomes."""
    import numpy as np
    import torch
    presets = _notebook_presets(json.loads(Path(notebook_path).read_text()))
    config = resolve_config(presets, "v0")
    years, seeds = config["reference_year_candidates"], config["sae_seeds"]
    assert config["profile"] == "v0" and years == [2007, 2008, 2009] and seeds == [42, 135, 215]
    assert config["matching_roles"] == ["rule_discovery", "cav_fit"] and config["chance_permutations"] == 5
    on_host = _check_family_construction("cpu")
    _check_pearson("cpu")
    _check_candidates()
    _check_family_units(config, synthetic_population(years=years, patients_per_year=160), years, seeds)
    if _cuda_available():
        _same_family_results(on_host, _check_family_construction("cuda"))
        rng = np.random.default_rng(5)
        mats = [(rng.gamma(1.0, size=(300, 40)) * (rng.random((300, 40)) > .5)).astype(np.float32) for _ in range(3)]
        options = dict(id_prefix="x", permutations=6, rng_seed=[1, 2])
        _same_family_results([build_factor_families(mats, seeds, device="cpu", **options)],
                             [build_factor_families(mats, seeds, device="cuda", **options)])
        np.testing.assert_allclose(_check_pearson("cuda"), _check_pearson("cpu"), rtol=0, atol=1e-9)
        assert torch.cuda.memory_allocated() < 64 * 2 ** 20, "GPU memory was not released after family construction"
    else:
        print("CUDA family equality checks skipped: CUDA unavailable here; GPU validation pending")
    print("Anchor-free family fixtures passed")


def _winner_row(fraction, f2=.5, precision=.5, lift=2., rules=2, conditions=4, valid=True):
    return {"activation_target": fraction, "valid": valid, "f2": f2, "precision": precision, "lift": lift,
            "rule_count": rules, "condition_count": conditions}


def _check_winner_ordering():
    """F2, then precision, lift, fewer rules, fewer conditions, smaller A; only feasible rows compete."""
    from temporal_rules import select_activation_winner
    cases = [
        ([_winner_row(.1, f2=.6), _winner_row(.2, f2=.7), _winner_row(.3, f2=.5)], .2),
        ([_winner_row(.1, f2=.7, precision=.5), _winner_row(.2, f2=.6, precision=.9)], .1),
        ([_winner_row(.1, precision=.6), _winner_row(.2, precision=.7), _winner_row(.3, precision=.65)], .2),
        ([_winner_row(.1, lift=2.), _winner_row(.2, lift=3.), _winner_row(.3, lift=1.5)], .2),
        ([_winner_row(.1, precision=.6, lift=2.), _winner_row(.2, precision=.5, lift=9.)], .1),
        ([_winner_row(.1, rules=3), _winner_row(.2, rules=2), _winner_row(.3, rules=4)], .2),
        ([_winner_row(.1, rules=2, conditions=5), _winner_row(.2, rules=3, conditions=3)], .1),
        ([_winner_row(.1, conditions=5), _winner_row(.2, conditions=4), _winner_row(.3, conditions=6)], .2),
        ([_winner_row(.3), _winner_row(.1), _winner_row(.2)], .1),
        ([_winner_row(.1, f2=.95, valid=False), _winner_row(.4, f2=.4), _winner_row(.5, f2=.3)], .4),
        ([_winner_row(.1, f2=.9, valid=False), _winner_row(.2, f2=.8, valid=False)], None)]
    for rows, expected in cases:
        for ordering in (rows, rows[::-1]):
            direct = select_activation_winner(ordering, rule_source="semantic")
            primary = [row["activation_target"] for row in direct if row["target_role"] == "primary"]
            assert primary == ([] if expected is None else [expected]), (primary, expected)
            ordered, winner = semantic_winner(ordering)
            assert [row["target_role"] for row in ordered] == [row["target_role"] for row in direct]
            assert (None if winner is None else winner["activation_target"]) == expected
            assert len(ordered) == len(ordering) and all(row["rule_source"] == "semantic" for row in ordered)
            assert winner is None or winner["valid"] and winner["eligible_for_primary_cav"]
            assert not [row for row in ordered if row["target_role"] == "secondary" and row["eligible_for_primary_cav"]]


def _check_targets_and_splits():
    """Strict positive-only cutoffs, invalid targets, and the A-independent deterministic inner split."""
    import numpy as np
    target = activation_threshold_target([0, 0, 1, 2, 3, 4, float("nan")], .5)
    assert target.valid and target.cutoff == 2.5 and target.spec.name == "A=0.5" and target.spec.minimum_positive_samples == 2
    assert strict_activation_mask([0, 0, 1, 2, 3, 4, float("nan")], target.to_dict()).tolist() == [False] * 4 + [True, True, False]
    tied = activation_threshold_target([0, 1, 2, 3], .5)
    assert tied.cutoff == 2.0 and strict_activation_mask([0, 1, 2, 3], tied.to_dict()).tolist() == [False, False, False, True]
    for values, reason in (([0, 0, 1.5, 0], "insufficient_positive_fit_activations"), ([0, 0, 0], "no_positive_fit_activations")):
        invalid = activation_threshold_target(values, .1)
        assert not invalid.valid and invalid.invalid_reason == reason
        assert not strict_activation_mask(values, invalid.to_dict()).any()
    patients = [f"p{index:03d}" for index in range(41)]
    values = np.where(np.arange(41) % 3 == 0, 0., 1. + np.arange(41))
    fit, tune = rule_inner_split(patients, values, 7)
    assert (fit, tune) == rule_inner_split(patients, values, 7) and fit != rule_inner_split(patients, values, 8)[0]
    assert not set(fit) & set(tune) and set(fit) | set(tune) == set(patients) and abs(len(fit) - len(tune)) <= 2
    positive = {patient for patient, value in zip(patients, values) if value > 0}
    assert abs(len(positive & set(fit)) - len(positive & set(tune))) <= 1
    assert (fit, tune) == rule_inner_split(patients, np.where(values > 0, 5., 0.), 7), "The split must depend on positivity only"
    for bad in (lambda: rule_inner_split(patients[:-1] + [patients[0]], values, 7), lambda: rule_inner_split(patients, values[:-1], 7)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid inner split input accepted")


def _rule_fixture(patients=2400, seed=11):
    """One reference year with a planted feature rule and synthetic family-unit output for three medoids."""
    import numpy as np
    rng = np.random.default_rng(seed)
    names = [f"feature_{index}" for index in range(10)]
    X = rng.normal(size=(patients, 10)).astype(np.float32)
    X[:, 7] = (rng.random(patients) < .5).astype(np.float32)
    ids = np.asarray([f"rule-{index:05d}" for index in range(patients)])
    population = _build_population(X=X, y=(rng.random(patients) < .2).astype(int), year=np.full(patients, 2007), patient_id=ids,
                                   record_keys=np.asarray([f"{patient}|2007" for patient in ids]), feature_names=names,
                                   feature_selection_max_year=2006)
    roles = reference_year_roles(population, 2007, 42)
    matching = np.flatnonzero(np.isin(roles["row_role"], ["rule_discovery", "cav_fit"]))
    planted_rows = (X[:, 3] > .25) & (X[:, 7] > .5)
    amplitude = rng.random(patients)
    columns = {"planted": np.where(planted_rows, 1. + 2. * amplitude, np.where(rng.random(patients) < .3, .001 + .05 * amplitude, 0.)),
               "noise": np.where(rng.random(patients) < .5, .001 + rng.random(patients), 0.),
               "sparse": np.zeros(patients)}
    columns["sparse"][np.flatnonzero(roles["row_role"] == "rule_discovery")[0]] = 1.
    identifiers = {"planted": ("42:7,135:3,215:5", [42, 7]), "noise": ("42:2,135:9,215:1", [135, 9]),
                   "sparse": ("42:4,135:4,215:4", [215, 1])}
    return {"population": population, "roles": roles, "matching": matching, "columns": columns, "identifiers": identifiers,
            "names": names, "planted_rows": planted_rows}


def _family_output(fixture, columns=None, cap=3, extra=True):
    import numpy as np
    population, roles, matching = fixture["population"], fixture["roles"], fixture["matching"]
    columns = fixture["columns"] if columns is None else columns
    counts = {name: int((values[matching] > 0).sum()) for name, values in columns.items()}
    ranked = sorted(columns, key=lambda name: (-counts[name], fixture["identifiers"][name][0]))
    records = [{"rank": rank, "family_id": "2007/42/" + fixture["identifiers"][name][0], "medoid": fixture["identifiers"][name][1],
                "positive_count": counts[name]} for rank, name in enumerate(ranked, start=1)]
    cut = [{"rank": 4, "family_id": "2007/42/42:0,135:0,215:0", "medoid": [42, 0], "positive_count": 1}] if extra else []
    return {"row_index": matching.astype(np.int64),
            "candidate_activations": np.stack([columns[name][matching] for name in ranked]).astype(np.float32),
            "selection": {"cap": cap, "fraction": .1, "num_latents": 30, "full_families": len(records) + len(cut),
                          "candidates": records, "dropped_by_cap": cut},
            "reference_year": 2007, "split_seed": 42, "_rows": int(len(matching)),
            "matching_rows_identity": _identity_hash({"row_index": matching}), "roles_identity": roles["identity"],
            "population_identity": population["fingerprint"]["population_identity"], "scaler_identity": "fixture-scaler",
            "sae_identities": {"42": "a", "135": "b", "215": "c"}, "outcomes_read": False}


def _unit_json(unit):
    return json.dumps(unit, sort_keys=True, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value))


def _refuses(call, text=None, error=ValueError):
    try:
        call()
    except error as raised:
        assert text is None or text in str(raised), str(raised)
        return
    raise AssertionError(f"Expected {error.__name__}")


def _check_sweep_unit(config, fixture, backend):
    """The (family x A) sweep on a planted rule: winners, drops, no refill, determinism, role-only rows, recorded filters."""
    import inspect
    import numpy as np
    from semantic_rules import RuleSet, RuleSetSelectionConfig, select_rule_set
    population, roles, names = fixture["population"], fixture["roles"], fixture["names"]
    view, output = rule_population_view(population), _family_output(fixture)
    assert set(view) == set(RULE_VIEW_KEYS) and "y" not in view and "post_death" not in view
    forbidden = {"y", "outcome", "outcomes", "labels", "population"}
    assert not (set(inspect.signature(rule_unit).parameters) | set(inspect.signature(run_rule_unit).parameters)) & forbidden
    sweep, ids = config["threshold_sweep"], {name: "2007/42/" + fixture["identifiers"][name][0] for name in fixture["columns"]}
    rule_rows = np.flatnonzero((roles["row_role"] == "rule_discovery") & ~roles["is_entrant"])
    assert len(rule_rows) >= 250

    unit = rule_unit(view, 2007, roles, output, config, workers=1, backend=backend)
    assert unit["rule_source"] == "semantic" and unit["reference_year"] == 2007 and unit["outcomes_read"] is False
    assert np.array_equal(unit["rule_row_index"], rule_rows) and unit["rule_rows"] == unit["_rows"] == len(rule_rows)
    assert unit["rule_rows_identity"] == _identity_hash({"row_index": rule_rows}) and unit["feature_count"] == 10
    assert unit["candidates"] == 3 and unit["cap"] == 3 and unit["dropped_by_cap"] == 1 and unit["refilled"] is False
    assert unit["sweep"] == ["0.1", "0.2", "0.3", "0.4", "0.5"] and unit["backend_overridden"] is True
    assert unit["selection_config"]["objective"] == "f2" and unit["selection_config"]["min_precision"] == .5
    assert unit["backend_config"]["n_bootstraps"] == backend["n_bootstraps"] and unit["backend_config"]["bootstrap_unit"] == "group"
    default = _backend_settings()
    assert (default["n_bootstraps"], default["trees_per_bootstrap"], default["max_depth"]) == (30, 50, 15), "Real backend stays 30x50x15"
    assert unit["sae_identities"] == {"42": "a", "135": "b", "215": "c"} and unit["roles_identity"] == roles["identity"]
    assert unit["candidate_filter"] == config["rule"]["recurrence"] == {
        "max_candidates_per_bootstrap": 20, "family_similarity_threshold": .7, "min_family_recurrence": .5}
    changed = dict(config, rule=dict(config["rule"], recurrence=dict(config["rule"]["recurrence"], min_family_recurrence=.75)))
    assert rule_unit(view, 2007, roles, _family_output(fixture), changed, workers=1, backend=backend)["candidate_filter"][
        "min_family_recurrence"] == .75, "Recurrence settings must come from the configuration"

    rows = unit["sweep_rows"]
    assert len(rows) == 3 * len(sweep) and [row["activation_target"] for row in rows] == sweep * 3, "Every A row of every family is retained"
    by_family = {}
    for row in rows:
        by_family.setdefault(row["family_id"], []).append(row)
    assert set(by_family) == set(ids.values()) and "2007/42/42:0,135:0,215:0" not in by_family, "The cap is never refilled"
    for family_id, own in by_family.items():
        assert [row["target_role"] for row in own].count("primary") <= 1
        assert len({(row["rows_fit"], row["rows_tune"]) for row in own}) == 1
        record = next(item for item in unit["families"] if item["family_id"] == family_id)
        assert record["split"]["fit_rows"] + record["split"]["tune_rows"] == len(rule_rows)
        assert {row["rows_fit"] for row in own} == {record["split"]["fit_rows"]} and {row["rows_tune"] for row in own} == {record["split"]["tune_rows"]}
        for row in own:
            assert row["target_role"] in ("primary", "secondary") and row["rule_source"] == "semantic"
            assert row["valid"] == row["feasible"] and (row["failure_reason"] is None) == row["feasible"]
            assert (row["cutoff"] is None) == (not row["target_valid"])
            assert row["funnel_stage"] == ("selected" if row["feasible"] else row["funnel_stage"] if row["target_valid"] else "invalid_activation_target")
            if row["target_valid"]:
                assert 0 <= row["offered_count"] <= row["family_count"] <= row["candidate_count"]
            if row["feasible"]:
                assert row["precision"] >= .5 and row["lift"] >= 1.5 and 1 <= row["rule_count"] <= 5 and row["condition_count"] >= 1
                assert all(row[name] is not None for name in SWEEP_METRICS)
            else:
                assert row["rule_count"] == 0 and row["target_role"] == "secondary" and not row["eligible_for_primary_cav"]

    sparse = by_family[ids["sparse"]]
    assert all(not row["target_valid"] and not row["valid"] and row["cutoff"] is None for row in sparse)
    assert {row["failure_reason"] for row in sparse} == {"insufficient_positive_fit_activations"} and unit["jobs"] == 2 * len(sweep)
    families = {item["family_id"]: item for item in unit["families"]}
    assert families[ids["sparse"]]["status"] == NO_RULE_REASON == unit["no_rule_reason"] == "dropped:no_feasible_semantic_rule"
    assert families[ids["sparse"]]["winner_activation_target"] is None
    assert set(families[ids["sparse"]]["reasons_by_activation_target"]) == set(unit["sweep"])

    noise = families[ids["noise"]]
    assert noise["status"] == NO_RULE_REASON and noise["winner_activation_target"] is None, "Independent noise must not yield a frozen rule"
    assert all(row["funnel_stage"] in ("no_recurrent_family", "no_feasible_subset") and not row["feasible"] for row in by_family[ids["noise"]])
    planted = families[ids["planted"]]
    assert planted["status"] == "frozen" and planted["winner_activation_target"] in sweep
    primary = [row for row in by_family[ids["planted"]] if row["target_role"] == "primary"]
    assert len(primary) == 1 and primary[0]["activation_target"] == planted["winner_activation_target"]
    feasible = [row for row in by_family[ids["planted"]] if row["feasible"]]
    assert primary[0] is not None and all((row["f2"], row["precision"], row["lift"], -row["rule_count"], -row["condition_count"],
                                           -row["activation_target"]) <= (primary[0]["f2"], primary[0]["precision"], primary[0]["lift"],
                                           -primary[0]["rule_count"], -primary[0]["condition_count"], -primary[0]["activation_target"])
                                          for row in feasible)
    frozen = {entry["family_id"]: entry for entry in unit["frozen"]}
    assert set(frozen) == {family_id for family_id, item in families.items() if item["status"] == "frozen"}
    assert unit["frozen_count"] == len(frozen) == 1 and unit["dropped_count"] == 2 and ids["planted"] in frozen, "Dropped families are not replaced"
    assert sum(unit["winner_distribution"].values()) == len(frozen)
    entry = frozen[ids["planted"]]
    assert entry["activation_target"] == primary[0]["activation_target"] and entry["cutoff"] == primary[0]["cutoff"]
    assert entry["target"] == {"valid": True, "cutoff": entry["cutoff"], "positive_fraction": entry["activation_target"]}
    assert entry["rule_set"]["threshold_name"] == f"A={entry['activation_target']:g}" and entry["medoid"] == [42, 7]
    assert entry["rule_identity"] == _identity_hash(entry["rule_set"]) == primary[0]["rule_identity"]
    assert entry["rule_rows_identity"] == unit["rule_rows_identity"] and entry["identity"] == _frozen_identity(entry)
    assert entry["inner_split"] == planted["split"] and entry["inner_split"]["fit_patients_identity"] != entry["inner_split"]["tune_patients_identity"]
    used = {condition["feature_name"] for member in entry["rule_set"]["rules"] for condition in member["conditions"]}
    assert {"feature_3", "feature_7"} <= used and used <= set(names), used
    selected = rule_mask(entry["rule_set"], population["X"][rule_rows], names)
    truth = fixture["planted_rows"][rule_rows]
    precision, recall = (selected & truth).sum() / selected.sum(), (selected & truth).sum() / truth.sum()
    assert precision >= .8 and recall >= .8, (precision, recall)
    assert np.array_equal(selected, rule_mask(RuleSet.from_dict(entry["rule_set"]), population["X"][rule_rows].astype(np.float64)))

    again = rule_unit(view, 2007, roles, _family_output(fixture), config, workers=1, backend=backend)
    assert _unit_json(again) == _unit_json(unit), "Frozen thresholds, rules and identities must be reproducible"
    assert [item["identity"] for item in again["frozen"]] == [item["identity"] for item in unit["frozen"]]
    raw = rule_unit(view, 2007, roles, _family_output(fixture), config, workers=1, backend=backend, recurrence=None)
    assert raw["candidate_filter"] is None and len(raw["sweep_rows"]) == len(rows) and raw["candidates"] == 3
    assert all(row["offered_count"] == row["candidate_count"] and row["family_count"] is None for row in raw["sweep_rows"] if row["target_valid"])
    pooled = rule_unit(view, 2007, roles, _family_output(fixture), config, workers=2, backend=backend)
    assert _unit_json(pooled) == _unit_json(unit), "The joblib pool must reproduce serial execution exactly"

    cav_rows = np.flatnonzero(roles["row_role"] == "cav_fit")
    assert len(cav_rows) >= 100
    poisoned = _family_output(fixture)
    at_cav = np.isin(fixture["matching"], cav_rows)[None, :]
    values = poisoned["candidate_activations"]
    poisoned["candidate_activations"] = np.where(at_cav, np.where(values > 0, values * 7 + 3, 0), values).astype(np.float32)
    poisoned_X = population["X"].copy()
    poisoned_X[cav_rows] = np.float32("nan")
    assert _unit_json(rule_unit(dict(view, X=poisoned_X), 2007, roles, poisoned, config, workers=1, backend=backend)) == _unit_json(unit), \
        "Only the year-r rule_discovery rows may be read"

    calls = []

    def recording(candidates, X_tune, y_tune, selection_config, *, threshold_name):
        calls.append((selection_config, threshold_name, X_tune.shape, int(y_tune.sum()), _identity_hash(X_tune)))
        return select_rule_set(candidates, X_tune, y_tune, selection_config, threshold_name=threshold_name)

    recorded = rule_unit(view, 2007, roles, _family_output(fixture), config, workers=1, backend=backend, select=recording)
    assert _unit_json(recorded) == _unit_json(unit), "Injected selection must not change results"
    valid_rows = [row for row in recorded["sweep_rows"] if row["target_valid"]]
    assert len(calls) == len(valid_rows) == recorded["jobs"] == 2 * len(sweep)
    configured = config["rule"]
    for (selection_config, name, shape, positives, _), row in zip(calls, valid_rows):
        assert isinstance(selection_config, RuleSetSelectionConfig)
        assert (selection_config.objective, selection_config.min_precision, selection_config.min_lift, selection_config.max_rules,
                selection_config.max_rule_length, selection_config.min_marginal_recall) == ("f2", .5, 1.5, 5, 10, .02)
        assert (selection_config.objective, selection_config.min_precision, selection_config.min_lift, selection_config.max_rules,
                selection_config.max_rule_length, selection_config.min_marginal_recall) == tuple(
                    configured[key] for key in ("objective", "min_precision", "min_lift", "max_rules", "max_rule_length", "min_marginal_recall"))
        assert name == row["threshold_name"] == f"A={row['activation_target']:g}"
        assert shape == (row["rows_tune"], 10) and positives == row["positives_tune"]
    assert {call[1] for call in calls} == {f"A={fraction:g}" for fraction in sweep}
    tuned = {}
    for call, row in zip(calls, valid_rows):
        tuned.setdefault(row["family_id"], set()).add(call[4])
    assert all(len(hashes) == 1 for hashes in tuned.values()) and set(tuned) == {ids["planted"], ids["noise"]}, "One tune half per family"
    medoid = fixture["columns"]["planted"][rule_rows]
    _, expected_tune = rule_inner_split(population["patient_id"][rule_rows], medoid, _derived_seed("rule_inner_split", 42, 2007, ids["planted"]))
    expected_X = population["X"][rule_rows][np.isin(population["patient_id"][rule_rows], expected_tune)].astype(np.float64)
    assert tuned[ids["planted"]] == {_identity_hash(expected_X)}, "The tune half is the seeded patient-stratified half"
    calls.clear()
    stricter = dict(config, rule=dict(configured, min_precision=.6, min_lift=2., max_rules=3, max_rule_length=4, min_marginal_recall=.05))
    rule_unit(view, 2007, roles, _family_output(fixture), stricter, workers=1, backend=backend, select=recording)
    assert {(call[0].min_precision, call[0].min_lift, call[0].max_rules, call[0].max_rule_length, call[0].min_marginal_recall)
            for call in calls} == {(.6, 2., 3, 4, .05)}, "Filters must come from the configuration"
    assert rule_selection_config(configured) == RuleSetSelectionConfig(objective="f2", min_precision=.5, min_lift=1.5, max_rules=5,
                                                                      max_rule_length=10, min_marginal_recall=.02)

    def nothing_feasible(candidates, X_tune, y_tune, selection_config, *, threshold_name):
        return select_rule_set((), X_tune, y_tune, selection_config, threshold_name=threshold_name)

    empty = rule_unit(view, 2007, roles, _family_output(fixture), config, workers=1, backend=backend, select=nothing_feasible)
    assert empty["frozen"] == [] and empty["frozen_count"] == 0 and empty["dropped_count"] == empty["candidates"] == 3
    assert [item["status"] for item in empty["families"]] == [NO_RULE_REASON] * 3 and empty["refilled"] is False
    assert len(empty["sweep_rows"]) == 3 * len(sweep) and not [row for row in empty["sweep_rows"] if row["target_role"] == "primary"]
    assert {row["failure_reason"] for row in empty["sweep_rows"] if row["target_valid"]} == {"no_eligible_candidates"}
    assert sum(empty["winner_distribution"].values()) == 0 and "2007/42/42:0,135:0,215:0" not in {row["family_id"] for row in empty["sweep_rows"]}
    return unit


def _check_rule_sources(config, fixture, backend):
    from copy import deepcopy
    population, roles = fixture["population"], fixture["roles"]
    view, output = rule_population_view(population), _family_output(fixture)
    assert RULE_SOURCES == ("high_precision", "semantic") and ENABLED_RULE_SOURCES == ("semantic",)
    for source in ("high_precision", "unknown", ""):
        _refuses(lambda source=source: rule_unit(view, 2007, roles, output, config, backend=backend, rule_source=source))
        _refuses(lambda source=source: semantic_winner([_winner_row(.1)], source))
    _refuses(lambda: rule_unit(view, 2007, roles, output, config, backend=backend, rule_source="high_precision"), "not enabled in this pilot")
    _refuses(lambda: semantic_winner([_winner_row(.1)], "high_precision"), "not enabled in this pilot")
    _refuses(lambda: rule_unit(view, 2007, roles, output, config, backend=backend, rule_source="other"), "Unknown rule_source")
    changed = deepcopy(config)
    changed["rule"]["rule_source"] = "high_precision"
    _refuses(lambda: rule_unit(view, 2007, roles, output, changed, backend=backend), "not enabled in this pilot")
    _refuses(lambda: rule_unit(dict(view, y=population["y"]), 2007, roles, output, config, backend=backend))
    _refuses(lambda: rule_unit({key: value for key, value in view.items() if key != "X"}, 2007, roles, output, config))
    _refuses(lambda: rule_unit(view, 2007, roles, output, dict(config, profile="pilot"), backend=backend), "fixtures only")
    _refuses(lambda: rule_unit(view, 2007, roles, output, dict(config, profile="pilot"), recurrence=None), "fixtures only")
    _refuses(lambda: rule_unit(view, 2008, roles, output, config, backend=backend), "different reference year")
    _refuses(lambda: rule_unit(view, 2007, roles, dict(output, roles_identity="other"), config, backend=backend), "different roles")
    _refuses(lambda: rule_unit(view, 2007, roles, dict(output, matching_rows_identity="other"), config, backend=backend), "recorded identity")
    shifted = dict(output, row_index=output["row_index"][1:], candidate_activations=output["candidate_activations"][:, 1:])
    shifted["matching_rows_identity"] = _identity_hash({"row_index": shifted["row_index"]})
    _refuses(lambda: rule_unit(view, 2007, roles, shifted, config, backend=backend), "exactly the year-r rule_discovery rows")
    bad = dict(output, candidate_activations=output["candidate_activations"] * 0)
    _refuses(lambda: rule_unit(view, 2007, roles, bad, config, backend=backend), "positive count")


def _check_rule_mask():
    import numpy as np
    from semantic_rules import Condition, Rule, RuleSet
    rule = RuleSet((Rule("r", (Condition(1, "b", ">", .5), Condition(0, "a", "<=", .1)), None),), threshold_name="A=0.1")
    X = np.asarray([[.0625, 1.], [.1, 1.], [0., 0.], [-1., .6]], dtype=np.float32)
    assert rule_mask(rule, X).tolist() == rule_mask(rule.to_dict(), X).tolist() == [True, False, False, True], \
        "float32 .1 exceeds the float64 threshold .1, as in the backend that fitted the rule"
    assert rule_mask(rule, X, ["a", "b"]).dtype == bool and not rule_mask(RuleSet(), X).any()
    _refuses(lambda: rule_mask(rule, X, ["b", "a"]), "is not")
    _refuses(lambda: rule_mask(rule, X[:, :1]), "outside")
    _refuses(lambda: rule_mask(rule, X[0]), "two-dimensional")
    assert rule_text(rule) == "(b > 0.5 AND a <= 0.1)" and rule_text(RuleSet()) == "(empty rule set)"


def _check_rule_units(config, fixture, backend, reference):
    """D(r)/rules through the unit cache: dependencies, resume, unavailable when the families unit failed or never ran."""
    import numpy as np
    population, roles = fixture["population"], fixture["roles"]
    view, output = rule_population_view(population), _family_output(fixture)
    base = Path(os.environ["RPI_OUT_DIR"])
    base.mkdir(parents=True, exist_ok=True)
    free = lambda: 100 * 2 ** 30
    with tempfile.TemporaryDirectory(dir=base) as directory:
        work = Path(directory)

        def open_cache(name):
            cache = UnitCache(work / name, config, "source-fixture", free_bytes=free)
            assert cache.run("P/2007", lambda: {"_rows": 1})["status"] == "ok"
            for seed in config["sae_seeds"]:
                assert cache.run(f"S/{seed}/2007", lambda: {"_rows": 1})["status"] == "ok"
            return cache

        cache = open_cache("cache")
        counted = []

        def counting(candidates, X_tune, y_tune, selection_config, *, threshold_name):
            from semantic_rules import select_rule_set
            counted.append(threshold_name)
            return select_rule_set(candidates, X_tune, y_tune, selection_config, threshold_name=threshold_name)

        blocked = run_rule_unit(cache, view, roles, config, workers=1, backend=backend, select=counting)
        assert (blocked["status"], blocked["reason"]) == ("unavailable", "dep_not_run:D/2007/families") and not counted
        assert cache.run("D/2007/families", lambda: dict(output))["status"] == "ok"
        _refuses(lambda: run_rule_unit(cache, dict(view, y=population["y"]), roles, config, backend=backend))
        _refuses(lambda: run_rule_unit(cache, view, roles, dict(config, epochs=3), backend=backend))
        record = run_rule_unit(cache, view, roles, config, workers=1, backend=backend, select=counting)
        assert record["status"] == "ok" and record["key"] == "D/2007/rules" and record["stage"] == "D.rules", record
        assert record["dependencies"] == {"P/2007": cache._latest["P/2007"]["id"], "D/2007/families": cache._latest["D/2007/families"]["id"]}
        assert record["rows"] == reference["rule_rows"] and len(counted) == reference["jobs"]
        again = run_rule_unit(cache, view, roles, config, workers=1, backend=backend, select=counting)
        assert again["id"] == record["id"] and len(counted) == reference["jobs"], "A complete D unit was rebuilt"
        loaded = cache.load("D/2007/rules")
        assert _unit_json({key: value for key, value in loaded.items() if key != "rule_row_index"}) == _unit_json(
            {key: value for key, value in reference.items() if key != "rule_row_index"})
        assert np.array_equal(loaded["rule_row_index"], reference["rule_row_index"])
        frozen = load_frozen_rules(cache, 2007)
        assert set(frozen) == {entry["family_id"] for entry in reference["frozen"]} and frozen
        for family_id, entry in frozen.items():
            assert entry["rule"].to_dict() == entry["rule_set"] and entry["identity"] == _frozen_identity(entry)
            assert rule_mask(entry["rule"], population["X"][loaded["rule_row_index"]], population["feature_names"]).dtype == bool
        directory_bytes = sum(path.stat().st_size for path in cache._unit_path("D/2007/rules").rglob("*") if path.is_file())
        assert 0 < directory_bytes <= projected_rule_bytes(config), (directory_bytes, projected_rule_bytes(config))
        assert cache.completeness_table(["D/2007/rules", "D/2008/rules"]) == [
            {"unit": "D/2007/rules", "status": "complete"}, {"unit": "D/2008/rules", "status": "not_run (stopped)"}]

        failing = open_cache("cache-failure")

        def broken():
            raise FloatingPointError("injected")

        assert failing.run("D/2007/families", broken)["status"] == "failed"
        before = len(counted)
        record = run_rule_unit(failing, view, roles, config, workers=1, backend=backend, select=counting)
        assert (record["status"], record["reason"]) == ("unavailable", "dep_failed:D/2007/families") and len(counted) == before
        assert not failing.is_complete("D/2007/rules") and not failing._unit_path("D/2007/rules").exists()
        assert failing.completeness_table(["D/2007/rules"]) == [{"unit": "D/2007/rules", "status": "unavailable:dep_failed:D/2007/families"}]
        assert failing.run("D/2007/cavs_selection", lambda: {})["status"] == "unavailable"

        recovered = open_cache("cache-select-failure")
        assert recovered.run("D/2007/families", lambda: dict(output))["status"] == "ok"

        def crashing(candidates, X_tune, y_tune, selection_config, *, threshold_name):
            raise FloatingPointError("injected")

        record = run_rule_unit(recovered, view, roles, config, workers=1, backend=backend, select=crashing)
        assert record["status"] == "failed" and record["reason"] == "FloatingPointError: injected"
        assert not recovered._unit_path("D/2007/rules").exists()
        assert run_rule_unit(recovered, view, roles, config, workers=1, backend=backend)["status"] == "ok"


def run_threshold_sweep_checks(notebook_path="concept_subgroup_drift_pilot.ipynb"):
    """Semantic threshold sweep: winner ordering, drops without refill, frozen rules, role-only rows, filters, parallel parity."""
    import inspect
    presets = _notebook_presets(json.loads(Path(notebook_path).read_text()))
    config = resolve_config(presets, "v0")
    assert config["profile"] == "v0" and config["threshold_sweep"] == [.1, .2, .3, .4, .5]
    assert config["rule"] == {"objective": "f2", "min_precision": .5, "min_lift": 1.5, "max_rules": 5, "max_rule_length": 10,
                              "min_marginal_recall": .02, "rule_source": "semantic",
                              "recurrence": {"max_candidates_per_bootstrap": 20, "family_similarity_threshold": .7,
                                             "min_family_recurrence": .5}} and config["joblib_workers"] <= 4
    assert "P90" not in inspect.getsource(fit_reference_rule) and "P90" not in inspect.getsource(run_reference_primitive_checks)
    _check_winner_ordering()
    _check_targets_and_splits()
    _check_rule_mask()
    fixture, backend = _rule_fixture(), {"n_bootstraps": 8, "trees_per_bootstrap": 10, "max_depth": 6}
    reference = _check_sweep_unit(config, fixture, backend)
    _check_rule_sources(config, fixture, backend)
    _check_rule_units(config, fixture, backend, reference)
    print("Semantic threshold sweep fixtures passed: winner ordering, drop without refill, frozen rules, role-only rows, filters, pool parity")


def _fixture_head(seed, inputs=6, hidden=12, outputs=10):
    import torch
    generator = torch.Generator().manual_seed(seed)
    head = torch.nn.Sequential(torch.nn.Linear(inputs, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, outputs))
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.copy_(0.5 * torch.randn(parameter.shape, generator=generator))
    return head.eval()


def _fork_classifier(average_logits, device):
    """A classifier shell with only the attributes the fork's parser and aggregator read."""
    import numpy as np
    from tabpfn.scripts.estimator.base import TabPFNClassifier
    classifier = object.__new__(TabPFNClassifier)
    for name, value in (("is_classification_", True), ("num_classes_", 2), ("average_logits", bool(average_logits)),
                        ("device_", device), ("softmax_temperature", -0.1), ("classes_", np.asarray([0., 1.]))):
        setattr(classifier, name, value)
    return classifier


def _fork_continuation(head, shift, average_logits, reweight, labels, device):
    from tabpfn.scripts.estimator.configs import EnsembleConfiguration
    kwargs = {"reweight_probs_based_on_train": reweight, "eval_ys": labels, "bar_distribution": None,
              "logit_cancel_masks": [None], "adapted_bar_dists": []}
    return _frozen_continuation(_fork_classifier(average_logits, device), head.to(device),
                                [EnsembleConfiguration(class_shift_configuration=shift)], kwargs, "context-fixture")


def _check_portable_semantics(device):
    """The serialized continuation reproduces the fork's own parser and aggregator for every supported variant."""
    import numpy as np
    import torch
    generator = torch.Generator().manual_seed(3)
    h = torch.randn(17, 6, generator=generator).to(device)
    train_labels = torch.tensor([0., 0., 1., 0., 1., 0., 0.], device=device)
    shifts = {"none": None, "identity": torch.tensor([0, 1], device=device), "swap": torch.tensor([1, 0], device=device)}
    seen = {}
    for average_logits in (False, True):
        for name, shift in shifts.items():
            for reweight in (False, True):
                labels = train_labels if reweight else torch.zeros(0, 1, device=device)
                original = _fork_continuation(_fixture_head(5), shift, average_logits, reweight, labels, device)
                arrays, meta = export_continuation(original)
                rebuilt = rebuild_continuation(arrays, meta, device)
                torch.testing.assert_close(rebuilt.probabilities(h), original.probabilities(h), rtol=1e-5, atol=1e-6)
                if reweight:
                    direction = torch.linspace(-1, 1, 6, device=device)
                    step, gradient = 1e-2, rebuilt.death_gradient(h)
                    with torch.no_grad():
                        central = (rebuilt.probabilities(h + step * direction)[:, 1] - rebuilt.probabilities(h - step * direction)[:, 1]) / (2 * step)
                    torch.testing.assert_close((gradient * direction).sum(dim=1), central, rtol=1e-2, atol=5e-4)
                else:
                    torch.testing.assert_close(rebuilt.death_gradient(h), original.death_gradient(h), rtol=1e-4, atol=1e-6)
                assert rebuilt.identity == meta["identity"] and export_continuation(rebuilt)[1] == meta
                again = export_continuation(rebuilt)[0]
                assert set(again) == set(arrays) and all(np.array_equal(again[key], arrays[key]) for key in arrays)
                seen[(average_logits, name, reweight)] = original.probabilities(h).detach().cpu().numpy()
    assert not np.allclose(seen[(False, "identity", False)], seen[(False, "swap", False)], atol=1e-3), "Class shift unused"
    assert not np.allclose(seen[(False, "none", False)], seen[(False, "none", True)], atol=1e-3), "Reweighting unused"
    original = _fork_continuation(_fixture_head(5), shifts["swap"], False, True, train_labels, device)
    arrays, meta = export_continuation(original)
    expected = original.probabilities(h)
    with torch.no_grad():
        for parameter in original.head.parameters():
            parameter.add_(1.0)
    original.eval_ys.fill_(0.)
    torch.testing.assert_close(rebuild_continuation(arrays, meta, device).probabilities(h), expected, rtol=1e-5, atol=1e-6)
    ys = torch.tensor([0., 1., 1., 0.], device=device)
    from tabpfn.scripts.estimator.configs import EnsembleConfiguration
    configurations = [EnsembleConfiguration(class_shift_configuration=shifts["swap"]),
                      EnsembleConfiguration(class_shift_configuration=None)]
    logits = torch.randn(9, 2, 2, generator=generator).to(device)
    for average_logits in (False, True):
        classifier = _fork_classifier(average_logits, device)
        other = _fork_continuation(_fixture_head(5), None, average_logits, False, torch.zeros(0, 1), device)
        rebuilt = rebuild_continuation(*export_continuation(other), device)
        for reweight in (False, True):
            fork = classifier._aggregate_predictions(classifier._parse_predictions_per_configuration(logits.clone(), configurations),
                reweight_probs_based_on_train=reweight, eval_ys=ys, bar_distribution=None, logit_cancel_masks=[None, None],
                adapted_bar_dists=[])
            mine = rebuilt.aggregate(rebuilt.parse(logits.clone(), [shifts["swap"], None]), reweight, ys)
            torch.testing.assert_close(mine, fork.squeeze(0), rtol=1e-6, atol=1e-7)

    tampered = dict(arrays)
    key = f"{CONTINUATION_PREFIX}head_0_weight"
    tampered[key] = arrays[key] + np.float32(1e-3)
    _refuses(lambda: rebuild_continuation(tampered, meta, device), "changed", RuntimeError)
    _refuses(lambda: rebuild_continuation(arrays, dict(meta, temperature=meta["temperature"] + 1e-3), device), "changed", RuntimeError)
    _refuses(lambda: rebuild_continuation({k: v for k, v in arrays.items() if k != key}, meta, device), "lacks", RuntimeError)
    _refuses(lambda: rebuild_continuation(arrays, dict(meta, schema="other"), device), "schema")
    _refuses(lambda: rebuild_continuation(arrays, meta, "auto"), "resolved device")
    unsupported = _fork_continuation(torch.nn.Sequential(torch.nn.Linear(6, 12), torch.nn.Tanh(), torch.nn.Linear(12, 10)).eval(),
                                     None, False, False, torch.zeros(0, 1), device)
    _refuses(lambda: export_continuation(unsupported), "Unsupported predictor head layer", TypeError)
    wrapped = _fork_continuation(_fixture_head(5), None, False, False, torch.zeros(0, 1), device)
    wrapped.head = torch.nn.Linear(6, 10)
    _refuses(lambda: export_continuation(wrapped), "Unsupported predictor head Linear", TypeError)
    bare = MortalityContinuation(_fixture_head(5), lambda logits, cfg: logits, lambda out, rw, ys: out, [None], 0.1,
                                 [0, 1], "x", False, None)
    _refuses(lambda: export_continuation(bare), "portable", TypeError)
    two = _fork_continuation(_fixture_head(5), None, False, False, torch.zeros(0, 1), device)
    two.portable_options = dict(two.portable_options, class_shifts=[None, None])
    _refuses(lambda: export_continuation(two), "single-configuration", TypeError)
    regression = _fork_continuation(_fixture_head(5), None, False, False, torch.zeros(0, 1), device)
    regression.portable_options = dict(regression.portable_options, is_classification=False)
    _refuses(lambda: export_continuation(regression), "classification", TypeError)


def _continuation_unit(continuation, rows=40, dimension=6, device="cpu", seed=9, gate=8, **changes):
    """A P(r)-shaped payload around a continuation: embeddings, replay arrays, gate positions and the serialized head."""
    import numpy as np
    import torch
    generator = torch.Generator().manual_seed(seed)
    h = torch.randn(rows, dimension, generator=generator).to(device)
    arrays, meta = export_continuation(continuation)
    probabilities = batched_death_probabilities(continuation, h)
    gradients = batched_death_gradients(continuation, h)
    payload = {"embeddings": h.cpu().numpy(), "probabilities": probabilities, "gradients": gradients,
               "gate_positions": np.arange(0, rows, rows // gate, dtype=np.int64)[:gate], "reference_year": 2007,
               **arrays, "continuation": meta, "continuation_identity": meta["identity"],
               "context_identity": meta["context_identity"], "_rows": rows}
    payload.update(changes)
    return payload


def _check_continuation_units(config, device):
    """Batched probability/gradient helpers, P-unit round trip through the cache, replay refusals."""
    import numpy as np
    import torch
    original = _fork_continuation(_fixture_head(11), torch.tensor([1, 0], device=device), False, True,
                                  torch.tensor([0., 1., 0., 0., 1., 0.], device=device), device)
    h = torch.randn(23, 6, generator=torch.Generator().manual_seed(2)).to(device)
    reference_gradients = torch.cat([original.death_gradient(row[None]) for row in h]).cpu().numpy()
    for batch in (1, 4, 23, 100):
        np.testing.assert_allclose(batched_death_gradients(original, h, batch), reference_gradients, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(batched_death_probabilities(original, h.cpu().numpy(), batch),
                                   original.probabilities(h).detach().cpu().numpy(), rtol=1e-5, atol=1e-6)
    assert batched_death_gradients(original, h).dtype == np.float32 and batched_death_probabilities(original, h).shape == (23, 2)
    assert batched_death_gradients(original, h[:0]).shape == (0, 6) and batched_death_probabilities(original, h[:0]).shape == (0, 2)
    _refuses(lambda: batched_death_gradients(original, h, 0), "positive")
    _refuses(lambda: batched_death_probabilities(original, h[0]), "rows, dimension")
    assert not np.isclose(batched_death_gradients(original, h), 0).all()

    base = Path(os.environ["RPI_OUT_DIR"])
    base.mkdir(parents=True, exist_ok=True)
    free = lambda: 100 * 2 ** 30
    payload = _continuation_unit(original, device=device)
    with tempfile.TemporaryDirectory(dir=base) as directory:
        cache = UnitCache(Path(directory) / "cache", config, "source-fixture", free_bytes=free)
        record = cache.run("P/2007", lambda: dict(payload))
        assert record["status"] == "ok" and record["stage_version"] == STAGE_CODE_VERSIONS["P"], record
        loaded = load_continuation(cache, 2007, "cpu")
        assert loaded.identity == payload["continuation_identity"] and loaded.context_identity == "context-fixture"
        assert loaded.verification["rows"] == 8 and loaded.verification["probability_max_abs_error"] <= 1e-5
        assert 0 <= loaded.verification["gradient_max_abs_error"] <= 1e-4
        np.testing.assert_allclose(batched_death_probabilities(loaded, payload["embeddings"]), payload["probabilities"],
                                   rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(batched_death_gradients(loaded, payload["embeddings"], 5), payload["gradients"],
                                   rtol=1e-3, atol=1e-5)
        _refuses(lambda: load_continuation(cache, 2008, "cpu"), None, KeyError)
        _refuses(lambda: continuation_from_unit(cache.load("P/2007"), 2008, "cpu"), "reference year", RuntimeError)
        _refuses(lambda: load_continuation(cache, 2007, "auto"), "resolved device")
        arrays_path = cache._unit_path("P/2007") / "arrays.npz"
        original_bytes = arrays_path.read_bytes()
        with np.load(arrays_path) as stored:
            changed = {name: stored[name] for name in stored.files}
        changed[f"{CONTINUATION_PREFIX}head_2_bias"] = changed[f"{CONTINUATION_PREFIX}head_2_bias"] + np.float32(1e-3)
        np.savez(arrays_path, **changed)
        _refuses(lambda: load_continuation(cache, 2007, "cpu"), "changed", RuntimeError)
        arrays_path.write_bytes(original_bytes)
        assert load_continuation(cache, 2007, "cpu").identity == payload["continuation_identity"]
    perturbed = [("probabilities", lambda value: value + np.float32(1e-3) * (np.arange(value.shape[0])[:, None] == 10), "Probability replay"),
                 ("gradients", lambda value: value * np.float32(1.1) + np.float32(1e-3), "Gradient replay")]
    for name, change, message in perturbed:
        changed = dict(payload)
        changed[name] = change(payload[name])
        _refuses(lambda: continuation_from_unit(changed, 2007, "cpu"), message, RuntimeError)
    for key, value, message in (("continuation_identity", "0" * 64, "does not describe"), ("context_identity", "other", "does not describe"),
                                ("continuation", None, "lacks a serialized")):
        _refuses(lambda: continuation_from_unit(dict(payload, **{key: value}), 2007, "cpu"), message, RuntimeError)
    if _cuda_available():
        on_cpu = rebuild_continuation({k: payload[k] for k in payload["continuation"]["array_names"]}, payload["continuation"], "cpu")
        on_gpu = rebuild_continuation({k: payload[k] for k in payload["continuation"]["array_names"]}, payload["continuation"], "cuda")
        np.testing.assert_allclose(batched_death_probabilities(on_gpu, payload["embeddings"]),
                                   batched_death_probabilities(on_cpu, payload["embeddings"]), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(batched_death_gradients(on_gpu, payload["embeddings"]),
                                   batched_death_gradients(on_cpu, payload["embeddings"]), rtol=1e-3, atol=1e-5)
    else:
        print("CUDA continuation equality skipped: CUDA unavailable here; GPU validation pending")


def _check_predictor_gradients(out, cache, r):
    """A real P(r) unit: gradients cover every eligible row and the cached head reproduces probabilities and gradients."""
    import numpy as np
    assert out["gradients"].dtype == np.float32 and out["gradients"].shape == out["embeddings"].shape
    assert np.isfinite(out["gradients"]).all() and np.abs(out["gradients"]).max() > 0
    assert out["gradient_objective"] == "final_mortality_probability_class_1_wrt_raw_embeddings"
    assert out["continuation_identity"] == out["continuation"]["identity"] and out["gate_positions"].dtype == np.int64
    assert len(out["gate_positions"]) == min(PREDICTOR_GATE_ROWS, len(out["embeddings"])) and out["gate_positions"].max() < len(out["embeddings"])
    assert out["continuation_replay"]["rows"] == len(out["gate_positions"])
    assert 0 <= out["gradient_stability_max_abs_error"] <= 1e-4
    layers = [layer["kind"] for layer in out["continuation"]["layers"]]
    assert layers[0] == "linear" and layers[-1] == "linear" and set(layers) <= {"linear", "gelu"}, layers
    continuation = load_continuation(cache, r, "cpu")
    assert continuation.identity == out["continuation_identity"] and continuation.context_identity == out["context_identity"]
    np.testing.assert_allclose(batched_death_probabilities(continuation, out["embeddings"]), out["probabilities"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(batched_death_gradients(continuation, out["embeddings"], 7), out["gradients"], rtol=1e-3, atol=1e-5)
    arrays, meta = export_continuation(continuation)
    assert meta["identity"] == out["continuation_identity"] and meta == out["continuation"]


def _check_continuation_equivalence(continuation, h, probabilities, device):
    """The head of a real fitted predictor serialized, rebuilt and compared with the executed one."""
    import torch
    arrays, meta = export_continuation(continuation)
    rebuilt = rebuild_continuation(arrays, meta, device)
    torch.testing.assert_close(rebuilt.probabilities(h), probabilities.to(device), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(rebuilt.probabilities(h), continuation.probabilities(h), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rebuilt.death_gradient(h), continuation.death_gradient(h), rtol=1e-3, atol=1e-6)
    assert export_continuation(rebuilt)[1] == meta
    return meta


def _positive_rule(feature, name, operator=">", threshold=.5):
    from semantic_rules import Condition, Rule, RuleSet
    return RuleSet((Rule("r", (Condition(feature, name, operator, threshold),), None),), threshold_name="A=0.1")


def _check_cav_fitting():
    """CAV repetitions: rule positives vs minimum-activation rule negatives, prerequisites, float64 rule semantics."""
    import numpy as np
    rng = np.random.default_rng(4)
    rows, dimension = 90, 5
    features = np.zeros((rows, 3), dtype=np.float32)
    features[:30, 1] = 1.
    activation = np.zeros(rows)
    activation[:30] = 2. + rng.random(30)
    activation[30:50] = .5 + rng.random(20)
    embeddings = rng.normal(size=(rows, dimension)) * .1
    embeddings[:30, 0] += 3.
    patients = np.asarray([f"cav-{i:03d}" for i in range(rows)])
    row_index = np.arange(1000, 1000 + rows)
    rule = _positive_rule(1, "b")
    fitted = fit_family_cavs(embeddings, features, activation, rule, patients, row_index, [42, 43, 44])
    assert list(fitted) == [42, 43, 44] and all(record["valid"] and record["failure_reason"] is None for record in fitted.values())
    for record in fitted.values():
        assert record["cav"].dtype == np.float64 and abs(np.linalg.norm(record["cav"]) - 1) < 1e-9 and record["cav"][0] > .9
        assert record["low_activation_cutoff"] == 0. and record["positive_count"] == 30 and record["eligible_negative_count"] == 40
        assert record["positive_training_count"] == record["negative_count"] == 30 and record["negative_patients"] >= 2
        assert record["positive_patients"] == 30 and record["cv_accuracy"] is not None
    assert not np.array_equal(fitted[42]["cav"], fitted[43]["cav"]), "Seeds must resample the negative pool"
    assert fitted[42]["negative_rows_identity"] != fitted[43]["negative_rows_identity"]
    again = fit_family_cavs(embeddings, features, activation, rule, patients, row_index, [42, 43, 44])
    assert all(np.array_equal(again[seed]["cav"], fitted[seed]["cav"]) and again[seed]["positive_rows_identity"] == fitted[seed]["positive_rows_identity"]
               for seed in fitted)
    shifted = fit_family_cavs(embeddings, features, activation, rule, patients, row_index + 7, [42])
    assert shifted[42]["positive_rows_identity"] != fitted[42]["positive_rows_identity"]
    np.testing.assert_array_equal(shifted[42]["cav"], fitted[42]["cav"])
    assert [record["valid"] for record in fit_family_cavs(embeddings, features, activation, rule, patients, row_index, [42]).values()] == [True]

    edge = np.zeros((rows, 3), dtype=np.float32)
    edge[:, 0] = np.float32(.1)
    at_most = fit_family_cavs(embeddings, edge, activation, _positive_rule(0, "a", "<=", .1), patients, row_index, [42])[42]
    above = fit_family_cavs(embeddings, edge, activation, _positive_rule(0, "a", ">", .1), patients, row_index, [42])[42]
    assert at_most["failure_reason"] == "insufficient_cav_positives" and above["failure_reason"] == "insufficient_cav_negatives", \
        "float32 .1 exceeds the float64 threshold .1, as in the backend that fitted the rule"

    def reason(**changes):
        arguments = dict(embeddings=embeddings, features=features, activations=activation, rule=rule, patient_ids=patients,
                         row_index=row_index, seeds=[42, 43, 44])
        arguments.update(changes)
        result = fit_family_cavs(**arguments)
        assert all(not record["valid"] and record["cav"] is None and record["low_activation_cutoff"] is None for record in result.values())
        assert {record["failure_reason"] for record in result.values()} == {result[42]["failure_reason"]}
        assert _cav_description(result[42])["status"] == f"unavailable:{result[42]['failure_reason']}"
        return result[42]["failure_reason"]

    one_positive = features.copy()
    one_positive[1:30, 1] = 0.
    assert reason(features=one_positive) == "insufficient_cav_positives"
    assert reason(features=np.zeros_like(features)) == "insufficient_cav_positives"
    crowded = activation.copy()
    crowded[30:], crowded[0] = 1., 0.
    assert reason(activations=crowded) == "insufficient_cav_negatives"
    assert reason(activations=np.full(rows, np.nan)) == "insufficient_cav_negatives"
    one_negative = activation.copy()
    one_negative[30], one_negative[31:] = .2, 1.
    assert reason(activations=one_negative) == "insufficient_cav_negatives"
    same_patient = patients.copy()
    same_patient[:30] = "cav-000"
    assert reason(patient_ids=same_patient) == "insufficient_distinct_cav_patients"
    same_negative = patients.copy()
    same_negative[30:] = "cav-099"
    assert reason(patient_ids=same_negative) == "insufficient_distinct_cav_patients"
    infinite = embeddings.copy()
    infinite[3, 2] = np.nan
    assert reason(embeddings=infinite) == "nonfinite_cav_embeddings"
    zero = fit_family_cavs(embeddings * 0, features, activation, rule, patients, row_index, [42])
    assert not zero[42]["valid"] and zero[42]["failure_reason"] == "invalid_cav_direction"
    partial = activation.copy()
    partial[50:60] = np.nan
    mixed = fit_family_cavs(embeddings, features, partial, rule, patients, row_index, [42])[42]
    assert mixed["valid"] and mixed["low_activation_cutoff"] == 0. and mixed["eligible_negative_count"] == 30, \
        "Nonfinite activations leave the negative pool"
    _refuses(lambda: fit_family_cavs(embeddings[:5], features, activation, rule, patients, row_index, [42]), "align")


def _identity_sae_payload(seed, dimension, reference_year=2007):
    """A cached S(seed, r) unit whose encoder maps embedding k to activation relu(x_k)."""
    import numpy as np
    import torch
    from sae import SAE
    latents = int(dimension * 1.5)
    weight = np.zeros((latents, dimension), dtype=np.float32)
    weight[np.arange(dimension), np.arange(dimension)] = 1.
    model = SAE(data_dimension=dimension, scaling_factor=1.5, use_decoder_bias=True, type="ReLU")
    model.load_state_dict({"encoder.weight": torch.from_numpy(weight), "encoder.bias": torch.zeros(latents),
                           "decoder_bias": torch.zeros(dimension)})
    mean, scale = np.zeros(dimension), np.ones(dimension)
    return {"encoder_weight": weight, "encoder_bias": np.zeros(latents, dtype=np.float32),
            "decoder_bias": np.zeros(dimension, dtype=np.float32), "scaler_mean": mean, "scaler_scale": scale,
            "scaler_var": scale.copy(), "_rows": 1, "reference_year": reference_year, "seed": int(seed), "epochs": 2,
            "latent_count": latents, "embedding_dim": dimension,
            "sae": {"model_type": "ReLU", "scaling_factor": 1.5, "alpha": .1, "learning_rate": .001, "weight_decay": 0.},
            "scaler_samples": 10, "scaler_identity": _identity_hash((mean, scale)), "sae_identity": _sae_state_identity(model),
            "identity": f"identity-sae-{seed}"}


FIXTURE_FAMILIES = (("alpha", [42, 0]), ("beta", [135, 1]), ("gamma", [215, 2]), ("delta", [42, 3]), ("epsilon", [135, 4]))
SELECTION_SIGNS = {  # family -> (positive rows of A, positive rows of R) as block offsets
    "alpha": (set(range(0, 9)), set(range(5, 9)) | set(range(10, 18))),
    "beta": (set(range(0, 5)), {10, 11}),
    "gamma": (set(range(0, 8)), set(range(5, 8)) | set(range(10, 17))),
    "delta": (set(), set()),
    "epsilon": (set(), set(range(10, 28)))}


def _selection_world():
    """Year 2007 (reference) and 2008 rows of one patient set with planted rule indicators, activation channels, probabilities.

    Embedding channel k is the activation of family k (identity encoder); block k of the concept_selection rows holds the
    subgroups of family k, so every selection row belongs to one family's subgroups.
    """
    import numpy as np
    from semantic_rules import RuleSet
    patients, dimension, years = 3200, 8, (2007, 2008)
    rng = np.random.default_rng(37)
    ids = np.asarray([f"world-{i:05d}" for i in range(patients)])
    patient, year = np.tile(ids, len(years)), np.repeat(years, patients)
    y = np.concatenate([(rng.random(patients) < .25).astype(int) for _ in years])
    names = [f"feature_{index}" for index in range(dimension)]
    keys = np.asarray([f"{p}|{t}" for p, t in zip(patient, year)])
    X = np.zeros((len(year), dimension), dtype=np.float32)
    X[:, 5:] = rng.normal(size=(len(year), dimension - 5)).astype(np.float32)
    base = _build_population(X=X, y=y, year=year, patient_id=patient, record_keys=keys, feature_names=names,
                             feature_selection_max_year=2006)
    roles = reference_year_roles(base, 2007, 42)
    at = {name: np.flatnonzero((year == 2007) & (roles["row_role"] == name)) for name in REFERENCE_ROLE_NAMES}
    E = np.zeros((len(year), dimension), dtype=np.float32)
    E[:, 5:] = (.01 * rng.normal(size=(len(year), dimension - 5))).astype(np.float32)
    E[year == 2008] = rng.normal(size=(int((year == 2008).sum()), dimension)).astype(np.float32)
    X[year == 2008, :5] = (rng.random((int((year == 2008).sum()), 5)) < .3).astype(np.float32)
    p1 = rng.uniform(.05, .45, size=len(year)).astype(np.float32)
    cav, selection, evaluation = at["cav_fit"], at["concept_selection"], at["evaluation"]
    assert len(cav) >= 120 and len(selection) >= 150 and len(evaluation) >= 300, (len(cav), len(selection), len(evaluation))
    for k, (name, _) in enumerate(FIXTURE_FAMILIES):
        block = cav[20 * k:20 * k + 20] if name != "delta" else cav[60:61]
        X[block, k], E[block, k] = 1., .8
        part = selection[30 * k:30 * k + 30]
        a_rows = part[:10]
        r_rows = part[10:30] if name == "epsilon" else part[5:25]
        E[a_rows, k] = 5.
        X[r_rows, k] = 1.
        E[r_rows[E[r_rows, k] == 0], k] = .5
    pool = lambda rows, label: rows[y[rows] == label]
    tabpfn = at["tabpfn_context"]
    alpha_fit = np.concatenate([pool(tabpfn, 1)[:3], pool(tabpfn, 0)[:7]])
    E[alpha_fit, 0] = 5.
    beta_fit = np.concatenate([pool(tabpfn, 1)[3:7], pool(tabpfn, 0)[7:13]])
    X[beta_fit, 1] = 1.
    epsilon_fit = np.concatenate([pool(at["sae_discovery"], 1)[:2], pool(at["sae_discovery"], 0)[:4]])
    E[epsilon_fit, 4] = 5.
    epsilon_fit_r = np.concatenate([pool(at["rule_discovery"], 1)[:2], pool(at["rule_discovery"], 0)[:3]])
    X[epsilon_fit_r, 4] = 1.
    alive, dead = pool(evaluation, 0), pool(evaluation, 1)
    alpha_a = np.concatenate([dead[:1], alive[:3]])
    E[alpha_a, 0] = 5.
    p1[alpha_a] = [.9, .2, .1, .4]
    alpha_r = np.concatenate([dead[1:3], alive[3:7]])
    X[alpha_r, 0] = 1.
    p1[alpha_r] = [.7, .4, .6, .3, .2, .5]
    beta_r = alive[7:10]
    X[beta_r, 1] = 1.
    p1[beta_r] = [.1, .2, .3]
    epsilon_r = dead[3:6]
    X[epsilon_r, 4] = 1.
    p1[epsilon_r] = [.8, .6, .3]
    population = _population_with(base, X=X)
    assert roles["identity"] == reference_year_roles(population, 2007, 42)["identity"]
    rows = np.flatnonzero(year >= 2007)
    matching = np.flatnonzero((year == 2007) & np.isin(roles["row_role"], ["rule_discovery", "cav_fit"]))
    identifiers = {name: f"2007/42/fixture-{name}" for name, _ in FIXTURE_FAMILIES}
    candidates = [{"rank": rank, "family_id": identifiers[name], "medoid": list(medoid),
                   "positive_count": int((E[matching, k] > 0).sum())} for rank, (k, (name, medoid)) in enumerate(
                       enumerate(FIXTURE_FAMILIES), start=1)]
    family_output = {
        "row_index": matching.astype(np.int64), "candidate_activations": np.stack(
            [np.maximum(E[matching, k], 0.) for k in range(len(FIXTURE_FAMILIES))]).astype(np.float32),
        "selection": {"cap": 29, "fraction": .1, "num_latents": 12, "full_families": 5, "candidates": candidates, "dropped_by_cap": []},
        "reference_year": 2007, "split_seed": 42, "_rows": int(len(matching)),
        "matching_rows_identity": _identity_hash({"row_index": matching}), "roles_identity": roles["identity"],
        "population_identity": population["fingerprint"]["population_identity"], "scaler_identity": _identity_sae_payload(42, dimension)["scaler_identity"],
        "sae_identities": {"42": "a", "135": "b", "215": "c"}, "outcomes_read": False}
    output = {"row_index": rows.astype(np.int64), "embeddings": E[rows].copy(),
              "probabilities": np.stack([1 - p1[rows], p1[rows]], axis=1).astype(np.float32),
              "gradients": rng.normal(size=(len(rows), dimension)).astype(np.float32), "reference_year": 2007,
              "roles_identity": roles["identity"], "population_identity": population["fingerprint"]["population_identity"],
              "output_row_identity": _identity_hash({"row_index": rows}), "embedding_dim": dimension,
              "context_identity": "fixture-context", "continuation_identity": "fixture-continuation",
              "gate_positions": np.arange(8, dtype=np.int64), "_rows": int(len(rows))}
    entries = {}
    for k, (name, medoid) in enumerate(FIXTURE_FAMILIES):
        rule = _positive_rule(k, f"feature_{k}")
        entry = {"family_id": identifiers[name], "medoid": list(medoid), "reference_year": 2007, "split_seed": 42,
                 "rule_source": "semantic", "activation_target": .1, "cutoff": 1.0,
                 "target": {"valid": True, "cutoff": 1.0, "positive_fraction": .1}, "rule_set": rule.to_dict(),
                 "rule_identity": _identity_hash(rule.to_dict()), "rule_count": 1, "condition_count": 1,
                 "rule_rows_identity": "fixture"}
        entry["identity"] = _frozen_identity(entry)
        entries[entry["family_id"]] = {**entry, "rule": rule}
    return {"population": population, "roles": roles, "at": at, "output": output, "family_output": family_output, "frozen": entries,
            "E": E, "X": X, "y": population["y"], "p1": p1, "identifiers": identifiers, "dimension": dimension, "selection": selection,
            "ids": {"alpha_fit": alpha_fit, "alpha_a": alpha_a, "alpha_r": alpha_r, "beta_r": beta_r, "epsilon_r": epsilon_r,
                    "epsilon_fit": epsilon_fit, "epsilon_fit_r": epsilon_fit_r, "beta_fit": beta_fit}}


def _identity_dictionaries(dimension, seeds, device="cpu"):
    import numpy as np
    import torch
    from sae import SAE
    from sklearn.preprocessing import StandardScaler
    result = {}
    for seed in seeds:
        payload = _identity_sae_payload(seed, dimension)
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_, scaler.var_ = payload["scaler_mean"], payload["scaler_scale"], payload["scaler_var"]
        scaler.n_features_in_, scaler.n_samples_seen_ = dimension, 10
        model = SAE(data_dimension=dimension, scaling_factor=1.5, use_decoder_bias=True, type="ReLU")
        model.load_state_dict({"encoder.weight": torch.from_numpy(payload["encoder_weight"]),
                               "encoder.bias": torch.from_numpy(payload["encoder_bias"]),
                               "decoder_bias": torch.from_numpy(payload["decoder_bias"])})
        result[seed] = {"scaler": scaler, "run": {"model": model.to(device).eval(), "model_type": "ReLU"},
                        "scaler_identity": payload["scaler_identity"], "sae_identity": payload["sae_identity"], "seed": seed,
                        "epochs": 2, "reference_year": 2007, "latent_count": payload["latent_count"], "embedding_dim": dimension}
    return result


def _with_gradients(world, vectors):
    """Gradients that give every family block its planted raw TCAV against the fitted seed-42 CAV of that family."""
    import numpy as np
    output = dict(world["output"])
    rows = output["row_index"]
    gradients = np.random.default_rng(3).normal(size=output["gradients"].shape).astype(np.float32)
    for k, (name, _) in enumerate(FIXTURE_FAMILIES):
        if not np.any(vectors[k, 0]):
            continue
        a_positive, r_positive = SELECTION_SIGNS[name]
        part = world["selection"][30 * k:30 * k + 30]
        a_offsets = set(range(10))
        r_offsets = set(range(10, 30)) if name == "epsilon" else set(range(5, 25))
        for offset, row in enumerate(part):
            sign = 1.
            if offset in a_offsets:
                sign = 1. if offset in a_positive else -1.
            if offset in r_offsets and offset not in a_offsets:
                sign = 1. if offset in r_positive else -1.
            gradients[np.searchsorted(rows, row)] = sign * vectors[k, 0]
    output["gradients"] = gradients
    return output


def _selection_config(config):
    return dict(config, tcav_cutoffs=[.1, .9], primary_cav_seed=42, cav_seeds=[42, 43, 44])


def _check_tcav_selection_rules():
    """Raw subgroup TCAV eligibility: inclusive 0.1/0.9 edges, independence per idea, seed 42 decides, reasons."""
    import numpy as np
    direction = np.asarray([1., 0.])
    cavs = {42: {"valid": True, "cav": direction}, 43: {"valid": True, "cav": -direction},
            44: {"valid": False, "failure_reason": "fixture_unavailable", "cav": None}}

    def selection(positives, rows, cav_set=None, zeros=0):
        gradients = np.zeros((rows, 2))
        gradients[:positives, 0] = 1.
        gradients[positives:rows - zeros, 0] = -1.
        return frozen_subgroup_selection(gradients, cavs if cav_set is None else cav_set, {"A": np.ones(rows, dtype=bool)}, "f")["A"]

    for positives, rows, eligible in [(0, 10, True), (1, 10, True), (2, 10, False), (5, 10, False), (8, 10, False), (9, 10, True),
                                      (10, 10, True), (3, 30, True), (4, 30, False), (26, 30, False), (27, 30, True),
                                      (10, 100, True), (11, 100, False), (89, 100, False), (90, 100, True), (1, 1, True)]:
        result = selection(positives, rows)
        assert result["primary_tcav"] == positives / rows and result["eligible"] is eligible, (positives, rows)
    assert selection(1, 10, zeros=9)["primary_tcav"] == .1 and selection(1, 10, zeros=9)["eligible"], "Zero projections are not positive"
    assert selection(0, 10, zeros=10)["primary_tcav"] == 0. and selection(0, 10, zeros=10)["eligible"]
    flipped = selection(9, 10)
    assert flipped["diagnostics"][43]["tcav"] == .1 and flipped["diagnostics"][44]["unavailable_reason"] == "fixture_unavailable"
    assert flipped["eligible"] and flipped["primary_seed"] == 42, "Seeds 43/44 are diagnostics only"
    only_diagnostics = {**cavs, 42: {"valid": False, "failure_reason": "insufficient_cav_positives", "cav": None}}
    refused = selection(9, 10, only_diagnostics)
    assert not refused["eligible"] and refused["primary_tcav"] is None and refused["diagnostics"][43]["tcav"] == .1
    assert refused["unavailable_reason"] == "insufficient_cav_positives"
    empty = frozen_subgroup_selection(np.ones((4, 2)), cavs, {"A": np.zeros(4, dtype=bool)}, "f")["A"]
    assert empty["unavailable_reason"] == "empty_selection_subgroup" and not empty["eligible"]
    gradients = np.zeros((20, 2))
    gradients[:9, 0], gradients[9:20, 0] = 1., -1.
    gradients[10:19, 1], gradients[[0, 9, 19], 1] = 1., -1.
    both = frozen_subgroup_selection(gradients, {42: {"valid": True, "cav": direction}, 43: cavs[43], 44: cavs[44]},
                                     {"A": np.asarray([True] * 10 + [False] * 10), "R": np.asarray([False] * 10 + [True] * 10)}, "f")
    assert both["A"]["primary_tcav"] == .9 and both["A"]["eligible"] and both["R"]["primary_tcav"] == 0. and both["R"]["eligible"]
    mixed = frozen_subgroup_selection(gradients, {42: {"valid": True, "cav": direction}, 43: cavs[43], 44: cavs[44]},
                                      {"A": np.asarray([True] * 10 + [False] * 10), "R": np.asarray([True] * 20)}, "f")
    assert mixed["A"]["eligible"] and not mixed["R"]["eligible"] and mixed["R"]["primary_tcav"] == 9 / 20, "Ideas are independent"
    bad = gradients.copy()
    bad[3, 0] = np.nan
    _refuses(lambda: frozen_subgroup_selection(bad, cavs, {"A": np.ones(20, dtype=bool)}, "f"), "finite")


def _check_selection_descriptions():
    """Eligibility status text: eligible, not eligible with the reason, or unavailable with its reason."""
    import numpy as np
    direction = np.asarray([1., 0.])
    cavs = {42: {"valid": True, "cav": direction}, 43: {"valid": True, "cav": direction}, 44: {"valid": True, "cav": direction}}
    gradients = np.zeros((10, 2))
    gradients[:9, 0], gradients[9:, 0] = 1., -1.
    mask = np.ones(10, dtype=bool)
    eligible = _selection_description(frozen_subgroup_selection(gradients, cavs, {"A": mask}, "f")["A"], "A")
    assert eligible["status"] == "eligible" and eligible["eligible"] and eligible["reason"] is None and eligible["tcav"] == .9
    assert set(eligible["diagnostics"]) == {"42", "43", "44"} and eligible["records"] == 10 and eligible["primary_seed"] == 42
    gradients[:5, 0], gradients[5:, 0] = 1., -1.
    inside = _selection_description(frozen_subgroup_selection(gradients, cavs, {"A": mask}, "f")["A"], "A")
    assert inside["status"] == "not_eligible:tcav_within_cutoffs" and not inside["eligible"] and inside["reason"] == "tcav_within_cutoffs"
    missing = {**cavs, 42: {"valid": False, "failure_reason": "insufficient_cav_negatives", "cav": None}}
    absent = _selection_description(frozen_subgroup_selection(gradients, missing, {"A": mask}, "f")["A"], "A")
    assert absent["status"] == "unavailable:insufficient_cav_negatives" and absent["tcav"] is None and not absent["eligible"]
    empty = _selection_description(frozen_subgroup_selection(gradients, cavs, {"A": ~mask}, "f")["A"], "A")
    assert empty["status"] == "unavailable:empty_selection_subgroup" and empty["records"] == 0


def _unit_arrays_equal(left, right):
    import numpy as np
    assert set(left) == set(right)
    for key, value in left.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, right[key], equal_nan=value.dtype.kind == "f"), key
    assert _unit_json({k: v for k, v in left.items() if not isinstance(v, np.ndarray)}) == _unit_json(
        {k: v for k, v in right.items() if not isinstance(v, np.ndarray)})


def _poisoned_world(world, keep_embeddings, keep_gradients, keep_features):
    """Copies of the P(r) output and population view with NaN (or flipped labels) outside the allowed rows."""
    import numpy as np
    output, rows = dict(world["output"]), world["output"]["row_index"]
    embeddings, gradients = output["embeddings"].copy(), output["gradients"].copy()
    embeddings[~np.isin(rows, keep_embeddings)] = np.nan
    gradients[~np.isin(rows, keep_gradients)] = np.nan
    output["embeddings"], output["gradients"] = embeddings, gradients
    output["probabilities"] = np.full_like(output["probabilities"], np.nan)
    view = rule_population_view(world["population"])
    X = view["X"].copy()
    X[~np.isin(np.arange(len(X)), keep_features)] = np.nan
    view["X"] = X
    return view, output


def _check_selection_unit(config, world):
    """D(r) CAVs and selection on planted subgroups; edges, independence, seed 42, unavailable reasons, rows used."""
    import inspect
    import numpy as np
    config = _selection_config(config)
    population, roles, at = world["population"], world["roles"], world["at"]
    view = rule_population_view(population)
    assert not set(inspect.signature(cavs_selection_unit).parameters) & {"y", "outcomes", "labels", "population"}
    dictionaries = _identity_dictionaries(world["dimension"], [42, 135, 215])
    ids = world["identifiers"]
    first = cavs_selection_unit(view, 2007, roles, world["output"], world["family_output"], world["frozen"], dictionaries, config, "cpu")
    vectors = first["cav_vectors"]
    assert vectors.shape == (5, 3, world["dimension"]) and vectors.dtype == np.float64
    assert first["cav_valid"].tolist() == [[True] * 3] * 3 + [[False] * 3] + [[True] * 3]
    assert not vectors[3].any() and all(abs(np.linalg.norm(vectors[k, j]) - 1) < 1e-9 for k in (0, 1, 2, 4) for j in range(3))
    assert all(vectors[k, 0, k] > .5 for k in (0, 1, 2, 4)), "Each CAV points to its own channel"
    output = _with_gradients(world, vectors)
    unit = cavs_selection_unit(view, 2007, roles, output, world["family_output"], world["frozen"], dictionaries, config, "cpu")
    _unit_arrays_equal({k: v for k, v in unit.items() if k not in ("families", "eligible", "selection_identity")},
                       {k: v for k, v in first.items() if k not in ("families", "eligible", "selection_identity")})
    families = {family["family_id"]: family for family in unit["families"]}
    assert list(families) == list(world["frozen"]) and unit["frozen_families"] == 5 and unit["candidate_families"] == 5

    def check(name, a_tcav, a_status, r_tcav, r_status):
        selection = families[ids[name]]["selection"]
        assert selection["idea1"]["tcav"] == a_tcav and selection["idea1"]["status"] == a_status, (name, selection["idea1"])
        assert selection["idea2"]["tcav"] == r_tcav and selection["idea2"]["status"] == r_status, (name, selection["idea2"])
        assert selection["idea1"]["eligible"] == (a_status == "eligible") and selection["idea2"]["eligible"] == (r_status == "eligible")

    non = "not_eligible:tcav_within_cutoffs"
    check("alpha", .9, "eligible", .6, non)
    check("beta", .5, non, .1, "eligible")
    check("gamma", .8, non, .5, non)
    check("delta", None, "unavailable:insufficient_cav_positives", None, "unavailable:insufficient_cav_positives")
    check("epsilon", 0., "eligible", .9, "eligible")
    assert unit["eligible"] == {"idea1": [ids["alpha"], ids["epsilon"]], "idea2": [ids["beta"], ids["epsilon"]], "both": [ids["epsilon"]]}
    inventory = {name: families[ids[name]]["inventory"] for name, _ in FIXTURE_FAMILIES}
    assert inventory["alpha"] == {"selection_rows": len(world["selection"]), "A": 10, "R": 20, "A_and_R": 5, "A_not_R": 5, "R_not_A": 15,
                                  "neither": len(world["selection"]) - 25}
    assert inventory["epsilon"]["A_and_R"] == 0 and inventory["epsilon"]["R"] == 20 and inventory["delta"]["A"] == 10
    for name in ("alpha", "beta"):
        for seed in ("42", "43", "44"):
            diagnostic = families[ids[name]]["selection"]["idea1"]["diagnostics"][seed]
            assert diagnostic["tcav"] is not None and diagnostic["unavailable_reason"] is None
    assert families[ids["delta"]]["selection"]["idea1"]["diagnostics"]["43"]["unavailable_reason"] == "insufficient_cav_positives"
    assert families[ids["alpha"]]["cav"]["42"]["status"] == "available" and families[ids["delta"]]["cav"]["42"]["status"] == \
        "unavailable:insufficient_cav_positives"
    assert families[ids["delta"]]["cav"]["42"]["positive_count"] == 1 and families[ids["alpha"]]["cav"]["42"]["positive_count"] == 20
    assert unit["negative_pool"].startswith("rule-negative rows at the minimum") and unit["outcomes_read"] is False
    assert unit["cav_rows"] == len(at["cav_fit"]) and unit["selection_rows_identity"] == _identity_hash({"row_index": np.sort(world["selection"])})
    assert np.array_equal(unit["selection_row_index"], np.sort(world["selection"])) and unit["selection_mask_A"].sum(axis=1).tolist() == [10] * 5
    rows = {name: families[ids[name]]["reference_identity"] for name, _ in FIXTURE_FAMILIES}
    assert len(set(rows.values())) == 5
    table = cavs_selection_table(unit)
    assert [row["idea 1"] for row in table][:1] == ["eligible"] and table[3]["CAV 42/43/44"] == "n/a/n/a/n/a" and table[0]["|A and R|"] == 5

    reference = {key: value for key, value in unit.items()}
    again = cavs_selection_unit(view, 2007, roles, output, world["family_output"], world["frozen"], dictionaries, config, "cpu")
    _unit_arrays_equal(again, reference)
    assert unit["selection_identity"] == again["selection_identity"]

    def negated(embeddings, features, activations, rule, patient_ids, row_index, seeds):
        result = fit_family_cavs(embeddings, features, activations, rule, patient_ids, row_index, seeds)
        for seed in (43, 44):
            if result[seed]["valid"]:
                result[seed]["cav"] = -result[seed]["cav"]
        return result

    repeated = cavs_selection_unit(view, 2007, roles, output, world["family_output"], world["frozen"], dictionaries, config, "cpu", fit=negated)
    assert repeated["eligible"] == unit["eligible"], "Seeds 43/44 must not change eligibility"
    alpha = repeated["families"][0]["selection"]["idea1"]["diagnostics"]
    assert alpha["42"]["tcav"] == .9 and alpha["43"]["tcav"] == .1 and alpha["44"]["tcav"] == .1

    def seed_42_unavailable(embeddings, features, activations, rule, patient_ids, row_index, seeds):
        result = fit_family_cavs(embeddings, features, activations, rule, patient_ids, row_index, seeds)
        result[42] = dict(result[42], valid=False, cav=None, failure_reason="fixture_unavailable", low_activation_cutoff=None,
                          positive_rows_identity=None, negative_rows_identity=None)
        return result

    unavailable = cavs_selection_unit(view, 2007, roles, output, world["family_output"], world["frozen"], dictionaries, config, "cpu",
                                      fit=seed_42_unavailable)
    assert unavailable["eligible"] == {"idea1": [], "idea2": [], "both": []}
    selected = unavailable["families"][0]["selection"]["idea1"]
    assert selected["status"] == "unavailable:fixture_unavailable" and selected["diagnostics"]["43"]["tcav"] is not None

    emptied = {name: dict(entry, target={"valid": True, "cutoff": 1e9, "positive_fraction": .1}, cutoff=1e9) if name == ids["alpha"] else entry
               for name, entry in world["frozen"].items()}
    empty = cavs_selection_unit(view, 2007, roles, output, world["family_output"], emptied, dictionaries, config, "cpu")
    assert empty["families"][0]["selection"]["idea1"]["status"] == "unavailable:empty_selection_subgroup"
    assert empty["families"][0]["selection"]["idea2"]["status"] == non and empty["eligible"]["idea1"] == [ids["epsilon"]]
    invalid_target = {name: dict(entry, target={"valid": False, "cutoff": None, "positive_fraction": .1}) if name == ids["alpha"] else entry
                      for name, entry in world["frozen"].items()}
    assert cavs_selection_unit(view, 2007, roles, output, world["family_output"], invalid_target, dictionaries, config, "cpu"
                               )["families"][0]["selection"]["idea1"]["status"] == "unavailable:empty_selection_subgroup"

    nothing = cavs_selection_unit(view, 2007, roles, output, world["family_output"], {}, dictionaries, config, "cpu")
    assert nothing["families"] == [] and nothing["cav_vectors"].shape == (0, 3, world["dimension"]) and nothing["eligible"] == {
        "idea1": [], "idea2": [], "both": []} and cavs_selection_table(nothing) == []

    keep = np.concatenate([at["cav_fit"], at["concept_selection"]])
    view_p, output_p = _poisoned_world(world, keep, at["concept_selection"], keep)
    output_p["gradients"][np.isin(output_p["row_index"], at["concept_selection"])] = output["gradients"][
        np.isin(output["row_index"], at["concept_selection"])]
    poisoned = cavs_selection_unit(view_p, 2007, roles, output_p, world["family_output"], world["frozen"], dictionaries, config, "cpu")
    _unit_arrays_equal(poisoned, unit)
    leaked = dict(output, gradients=output["gradients"].copy())
    leaked["gradients"][np.isin(leaked["row_index"], at["concept_selection"])] = 0.
    assert cavs_selection_unit(view, 2007, roles, leaked, world["family_output"], world["frozen"], dictionaries, config, "cpu"
                               )["eligible"] != unit["eligible"], "Selection gradients must matter"
    nan_gradients = dict(output, gradients=output["gradients"].copy())
    nan_gradients["gradients"][np.flatnonzero(np.isin(output["row_index"], at["concept_selection"]))[0], 0] = np.nan
    _refuses(lambda: cavs_selection_unit(view, 2007, roles, nan_gradients, world["family_output"], world["frozen"], dictionaries, config, "cpu"), "finite")
    nan_cav = dict(output, embeddings=output["embeddings"].copy())
    nan_cav["embeddings"][np.flatnonzero(np.isin(output["row_index"], at["cav_fit"]))[0], 6] = np.nan
    all_unavailable = cavs_selection_unit(view, 2007, roles, nan_cav, world["family_output"], world["frozen"], dictionaries, config, "cpu")
    assert not all_unavailable["cav_valid"].any() and all_unavailable["eligible"] == {"idea1": [], "idea2": [], "both": []}
    assert all_unavailable["families"][0]["selection"]["idea1"]["status"] == "unavailable:nonfinite_cav_embeddings"

    _refuses(lambda: cavs_selection_unit(dict(view, y=population["y"]), 2007, roles, output, world["family_output"], world["frozen"], dictionaries, config, "cpu"), "outcomes")
    _refuses(lambda: cavs_selection_unit(view, 2007, roles, output, world["family_output"], world["frozen"], dictionaries, config, "auto"), "resolved device")
    _refuses(lambda: cavs_selection_unit(view, 2008, roles, output, world["family_output"], world["frozen"], dictionaries, config, "cpu"), "reference year")
    _refuses(lambda: cavs_selection_unit(view, 2007, roles, dict(output, roles_identity="other"), world["family_output"], world["frozen"], dictionaries, config, "cpu"), "different roles")
    _refuses(lambda: cavs_selection_unit(view, 2007, roles, dict(output, gradients=output["gradients"].astype(np.float64)), world["family_output"], world["frozen"], dictionaries, config, "cpu"), "float32")
    _refuses(lambda: cavs_selection_unit(view, 2007, roles, {k: v for k, v in output.items() if k != "gradients"}, world["family_output"], world["frozen"], dictionaries, config, "cpu"), "lacks")
    _refuses(lambda: cavs_selection_unit(view, 2007, roles, output, dict(world["family_output"], matching_rows_identity="x"), world["frozen"], dictionaries, config, "cpu"), "identity")
    stray = {"stray": dict(world["frozen"][ids["alpha"]], family_id="stray")}
    _refuses(lambda: cavs_selection_unit(view, 2007, roles, output, world["family_output"], stray, dictionaries, config, "cpu"), "not a candidate")
    for changed in (dict(config, cav_seeds=[42, 43]), dict(config, primary_cav_seed=43), dict(config, tcav_cutoffs=[.2, .8])):
        _refuses(lambda: cavs_selection_unit(view, 2007, roles, output, world["family_output"], world["frozen"], dictionaries, changed, "cpu"), "Selection supports")
    return unit, output


def _eligible_references(unit, world):
    return _references_from_units(unit, world["frozen"])


def _check_performance_unit(config, world, selection_unit, output):
    """Known Brier/skill, baseline from fitting roles only, single-class and empty subgroups, bootstrap, frozen identity, rows used."""
    import inspect
    import numpy as np
    config = _selection_config(config)
    population, roles, at = world["population"], world["roles"], world["at"]
    view = performance_population_view(population)
    assert set(view) == set(PERFORMANCE_VIEW_KEYS) and "gradients" not in inspect.signature(reference_performance_unit).parameters
    dictionaries = _identity_dictionaries(world["dimension"], [42, 135, 215])
    ids = world["identifiers"]
    references = _eligible_references(selection_unit, world)
    assert {name: reference["eligible"] for name, reference in references.items()}[ids["alpha"]] == {"idea1": True, "idea2": False}

    def run(refs=references, v=view, out=output, cfg=config, family_output=world["family_output"]):
        return reference_performance_unit(v, 2007, roles, out, family_output, refs, dictionaries, cfg, "cpu")

    unit = run()
    rows = {(row["family_id"], row["idea"]): row for row in unit["rows"]}
    assert set(rows) == {(ids["alpha"], "idea1"), (ids["beta"], "idea2"), (ids["epsilon"], "idea1"), (ids["epsilon"], "idea2")}
    assert {(item["family_id"], item["idea"]) for item in unit["not_computed"]} == {
        (ids["alpha"], "idea2"), (ids["beta"], "idea1"), (ids["gamma"], "idea1"), (ids["gamma"], "idea2"),
        (ids["delta"], "idea1"), (ids["delta"], "idea2")}
    y, E, X, p1 = world["y"], world["E"], world["X"], world["p1"]
    fitting = np.flatnonzero((population["year"] == 2007) & np.isin(roles["row_role"], FITTING_SIDE_ROLE_NAMES))
    evaluation = at["evaluation"]
    assert unit["fitting_roles"] == list(FITTING_SIDE_ROLE_NAMES) and unit["evaluation_rows"] == len(evaluation)
    assert unit["fitting_rows"] == len(fitting) and unit["evaluation_rows_identity"] == _identity_hash({"row_index": np.sort(evaluation)})
    alpha = rows[(ids["alpha"], "idea1")]
    assert alpha["baseline"]["q"] == .3 and alpha["baseline"]["fitting_records"] == 10 and alpha["fitting_rows"] == 10
    metrics = alpha["metrics"]
    assert alpha["evaluation_rows"] == 4 and alpha["evaluation_deaths"] == 1 and np.isclose(metrics["brier"], .055)
    assert np.isclose(metrics["baseline_brier"], .19) and np.isclose(metrics["brier_skill"], 1 - .055 / .19)
    assert np.isclose(metrics["mean_risk"], .4) and metrics["prevalence"] == .25 and metrics["baseline_identity"] == alpha["baseline"]["identity"]
    assert (metrics["true_positive"], metrics["false_negative"], metrics["false_positive"], metrics["true_negative"]) == (1, 0, 0, 3)
    assert metrics["f1"] == 1. and metrics["sensitivity"] == 1. and metrics["specificity"] == 1. and metrics["auroc"] == 1.
    assert np.isclose(metrics["ece"], .25 * (.1 + .2 + .4 + .1), atol=1e-6)
    assert metrics["classification_threshold"] == .5 and metrics["bootstrap"]["replicates"] == config["bootstrap_replicates"] == 10
    assert metrics["bootstrap"]["confidence_level"] == .95 and metrics["bootstrap"]["sampling_unit"] == "patient"

    def independent(name, idea):
        k = [item[0] for item in FIXTURE_FAMILIES].index(name)
        if idea == "A":
            fit = fitting[np.maximum(E[fitting, k], 0) > 1.0]
            ev = evaluation[np.maximum(E[evaluation, k], 0) > 1.0]
        else:
            fit, ev = fitting[X[fitting, k] > .5], evaluation[X[evaluation, k] > .5]
        return fit, ev

    for (family_id, idea), row in rows.items():
        name = next(key for key, value in ids.items() if value == family_id)
        fit, ev = independent(name, "A" if idea == "idea1" else "R")
        q = float(y[fit].mean()) if len(fit) else None
        assert row["baseline"]["q"] == q and row["fitting_rows"] == len(fit) and row["evaluation_rows"] == len(ev), (name, idea)
        if len(ev):
            assert np.isclose(row["metrics"]["brier"], np.mean((p1[ev].astype(np.float64) - y[ev]) ** 2))
            assert np.isclose(row["metrics"]["baseline_brier"], np.mean((q - y[ev]) ** 2))
            assert row["metrics"]["records"] == len(ev) and row["metrics"]["mean_risk"] == float(p1[ev].astype(np.float64).mean())
    beta = rows[(ids["beta"], "idea2")]["metrics"]
    assert np.isclose(beta["brier"], (.01 + .04 + .09) / 3) and beta["prevalence"] == 0. and beta["auroc"] is None
    assert beta["sensitivity"] is None and beta["specificity"] == 1. and beta["unavailable_reasons"]["auroc"] == "single_outcome_class"
    assert np.isclose(beta["mean_risk"], .2) and beta["brier"] is not None and beta["bootstrap"]["intervals"]["brier"] is not None
    epsilon_a, epsilon_r = rows[(ids["epsilon"], "idea1")], rows[(ids["epsilon"], "idea2")]
    assert epsilon_a["evaluation_rows"] == 0 and epsilon_a["metrics"]["brier"] is None and epsilon_a["metrics"]["unavailable_reasons"]["brier"] == "empty_subgroup"
    assert epsilon_a["baseline"]["q"] is not None and epsilon_a["metrics"]["bootstrap"]["unavailable_reasons"]["brier"] == "fewer_than_two_patients"
    single = epsilon_r["metrics"]
    assert np.isclose(single["brier"], (.04 + .16 + .49) / 3) and single["prevalence"] == 1. and single["sensitivity"] == 2 / 3
    assert single["specificity"] is None and single["unavailable_reasons"]["specificity"] == "zero_metric_denominator" and single["auroc"] is None
    table = reference_performance_table(unit)
    assert [(row["family"], row["idea"]) for row in table] == [(row["family_id"], row["idea"]) for row in unit["rows"]]
    assert table[0]["eval rows"] == 4 and table[0]["deaths"] == 1 and table[0]["q (fit rows)"] == .3 and table[0]["skill"] == round(1 - .055 / .19, 4)
    assert [row["unavailable"] for row in table if row["eval rows"] == 0] and "brier" in [row for row in table if row["eval rows"] == 0][0]["unavailable"]

    seeds = {row["bootstrap_seed"] for row in unit["rows"]}
    assert len(seeds) == len(unit["rows"]), "Each subgroup resamples with its own derived seed"
    assert _unit_json(run()) == _unit_json(unit), "Bootstrap must be deterministic"
    more = run(cfg=dict(config, bootstrap_replicates=25))
    assert more["rows"][0]["metrics"]["bootstrap"]["replicates"] == 25 and more["rows"][0]["metrics"]["brier"] == metrics["brier"]
    assert more["rows"][0]["metrics"]["bootstrap"]["intervals"] != metrics["bootstrap"]["intervals"]
    assert not set(inspect.signature(reference_performance_unit).parameters) & {"gradients"}

    keep_rows = np.concatenate([fitting, evaluation])
    outside = ~np.isin(output["row_index"], keep_rows)
    garbage = dict(output, embeddings=output["embeddings"].copy(), probabilities=output["probabilities"].copy(),
                   gradients=np.full_like(output["gradients"], np.nan))
    garbage["embeddings"][outside], garbage["probabilities"][outside] = np.nan, np.nan
    unread = np.setdiff1d(np.arange(len(view["y"])), keep_rows)
    view_g = dict(view, X=view["X"].copy(), y=view["y"].copy())
    view_g["X"][unread], view_g["y"][unread] = np.nan, 7
    assert _unit_json(run(v=view_g, out=garbage)) == _unit_json(unit), \
        "Only fitting-role and evaluation rows of the reference year may be read"
    flipped_eval = view["y"].copy()
    flipped_eval[evaluation] = 1 - flipped_eval[evaluation]
    changed_eval = run(v=dict(view, y=flipped_eval))
    changed_rows = {(row["family_id"], row["idea"]): row for row in changed_eval["rows"]}
    assert all(changed_rows[key]["baseline"] == rows[key]["baseline"] for key in rows), "q must not depend on evaluation labels"
    assert changed_rows[(ids["alpha"], "idea1")]["metrics"]["brier"] != metrics["brier"]
    flipped_fit = view["y"].copy()
    flipped_fit[world["ids"]["alpha_fit"]] = 1 - flipped_fit[world["ids"]["alpha_fit"]]
    refit = {(row["family_id"], row["idea"]): row for row in run(v=dict(view, y=flipped_fit))["rows"]}
    assert refit[(ids["alpha"], "idea1")]["baseline"]["q"] == .7 and refit[(ids["alpha"], "idea1")]["metrics"]["brier"] == metrics["brier"]
    for role in ("concept_selection",):
        flipped = view["y"].copy()
        flipped[at[role]] = 1 - flipped[at[role]]
        assert _unit_json(run(v=dict(view, y=flipped))) == _unit_json(unit), f"{role} labels must not be read"
    later = view["y"].copy()
    later[population["year"] > 2007] = 1 - later[population["year"] > 2007]
    assert _unit_json(run(v=dict(view, y=later))) == _unit_json(unit), "Later-year labels must not be read"
    shifted_p = dict(output, probabilities=output["probabilities"].copy())
    shifted_p["probabilities"][np.isin(output["row_index"], at["concept_selection"]), 1] = .5
    assert _unit_json(run(out=shifted_p)) == _unit_json(unit)

    mutated = {name: deepcopy(reference) for name, reference in references.items()}
    mutated[ids["alpha"]]["target"]["cutoff"] = 2.0
    _refuses(lambda: run(refs=mutated), "identity changed", RuntimeError)
    mutated = {name: deepcopy(reference) for name, reference in references.items()}
    mutated[ids["beta"]]["cavs"][42]["cav"][0] += 1e-3
    _refuses(lambda: run(refs=mutated), "identity changed", RuntimeError)
    mutated = {name: deepcopy(reference) for name, reference in references.items()}
    mutated[ids["beta"]]["rule"] = _positive_rule(2, "feature_2")
    _refuses(lambda: run(refs=mutated), "identity changed", RuntimeError)
    mutated = {name: deepcopy(reference) for name, reference in references.items()}
    mutated[ids["alpha"]]["target"]["cutoff"] = 2.0
    _refuses(lambda: reference_performance_unit(view, 2007, roles, output, world["family_output"], mutated, {}, config, "cpu"),
             "identity changed", RuntimeError)
    assert run(refs={ids["alpha"]: references[ids["alpha"]]})["rows"][0]["metrics"]["brier"] == metrics["brier"]
    none_eligible = {name: dict(reference, eligible={"idea1": False, "idea2": False}) for name, reference in references.items()}
    result = run(refs=none_eligible)
    assert result["rows"] == [] and len(result["not_computed"]) == 10 and run(refs={})["rows"] == []
    _refuses(lambda: run(v=dict(view, gradients=output["gradients"])), "reads only")
    _refuses(lambda: run(v=rule_population_view(population)), "reads only")
    _refuses(lambda: run(cfg=dict(config, classification_threshold=.4)), "0.5 threshold")
    _refuses(lambda: reference_performance_unit(view, 2007, roles, output, world["family_output"], references, dictionaries, config, "auto"), "resolved device")
    _refuses(lambda: run(out=dict(output, roles_identity="other")), "different roles")
    return unit


def _check_cav_units(config, world):
    """D(r)/cavs_selection and D(r)/reference_performance through the cache: dependencies, resume, identity, unavailable propagation."""
    import numpy as np
    config = _selection_config(config)
    population, roles = world["population"], world["roles"]
    view, performance_view = rule_population_view(population), performance_population_view(population)
    dictionaries = _identity_dictionaries(world["dimension"], [42, 135, 215])
    dimension = world["dimension"]
    base = Path(os.environ["RPI_OUT_DIR"])
    base.mkdir(parents=True, exist_ok=True)
    free = lambda: 100 * 2 ** 30
    probe = cavs_selection_unit(view, 2007, roles, world["output"], world["family_output"], world["frozen"], dictionaries, config, "cpu")
    output = _with_gradients(world, probe["cav_vectors"])
    direct = cavs_selection_unit(view, 2007, roles, output, world["family_output"], world["frozen"], dictionaries, config, "cpu")
    ids = world["identifiers"]

    def rules_payload(frozen):
        return {"frozen": [{key: value for key, value in entry.items() if key != "rule"} for entry in frozen.values()],
                "families": [{"family_id": name, "status": "frozen"} for name in frozen], "_rows": 1}

    with tempfile.TemporaryDirectory(dir=base) as directory:
        work = Path(directory)

        def open_cache(name, until="rules"):
            cache = UnitCache(work / name, config, "source-fixture", free_bytes=free)
            assert cache.run("P/2007", lambda: dict(output))["status"] == "ok"
            for seed in config["sae_seeds"]:
                assert cache.run(f"S/{seed}/2007", lambda seed=seed: _identity_sae_payload(seed, dimension))["status"] == "ok"
            if until != "P":
                assert cache.run("D/2007/families", lambda: dict(world["family_output"]))["status"] == "ok"
            if until == "rules":
                assert cache.run("D/2007/rules", lambda: rules_payload(world["frozen"]))["status"] == "ok"
            return cache

        cache = open_cache("cache")
        _refuses(lambda: run_cavs_selection_unit(cache, population, roles, config, "cpu"), "outcomes")
        _refuses(lambda: run_cavs_selection_unit(cache, view, roles, dict(config, epochs=3), "cpu"), "differs")
        _refuses(lambda: run_reference_performance_unit(cache, view, roles, config, "cpu"), "reads only")
        blocked = run_reference_performance_unit(cache, performance_view, roles, config, "cpu")
        assert (blocked["status"], blocked["reason"]) == ("unavailable", "dep_not_run:D/2007/cavs_selection")
        record = run_cavs_selection_unit(cache, view, roles, config, "cpu")
        assert record["status"] == "ok" and record["key"] == "D/2007/cavs_selection" and record["stage"] == "D.cavs_selection", record
        assert record["dependencies"] == {key: cache._latest[key]["id"] for key in ("P/2007", "D/2007/families", "D/2007/rules")}
        assert record["rows"] == len(world["selection"])
        stored = cache.load("D/2007/cavs_selection")
        assert stored["eligible"] == {"idea1": [ids["alpha"], ids["epsilon"]], "idea2": [ids["beta"], ids["epsilon"]], "both": [ids["epsilon"]]}
        assert stored["cav_vectors"].shape == (5, 3, dimension) and stored["selection_mask_A"].dtype == bool
        assert _unit_json(stored) == _unit_json(direct), "Cached unit differs from the direct computation"
        assert run_cavs_selection_unit(cache, view, roles, config, "cpu")["id"] == record["id"]
        used = sum(path.stat().st_size for path in cache._unit_path("D/2007/cavs_selection").rglob("*") if path.is_file())
        assert 0 < used <= projected_cavs_selection_bytes(config), (used, projected_cavs_selection_bytes(config))
        references = load_frozen_references(cache, 2007)
        assert list(references) == list(world["frozen"]) and all(reference["identity"] == family["reference_identity"]
               for reference, family in zip(references.values(), stored["families"]))
        assert references[ids["epsilon"]]["eligible"] == {"idea1": True, "idea2": True} and references[ids["delta"]]["cavs"][42]["cav"] is None
        assert references[ids["alpha"]]["target"] == world["frozen"][ids["alpha"]]["target"] and references[ids["alpha"]]["cavs"][42]["valid"]
        np.testing.assert_array_equal(references[ids["alpha"]]["cavs"][43]["cav"], stored["cav_vectors"][0, 1])
        assert_reference_identity(references[ids["alpha"]])
        references[ids["alpha"]]["cavs"][42]["cav"][0] += 1e-6
        _refuses(lambda: assert_reference_identity(references[ids["alpha"]]), "identity changed", RuntimeError)
        path = cache._unit_path("D/2007/cavs_selection") / "arrays.npz"
        original = path.read_bytes()
        with np.load(path) as handle:
            changed = {name: handle[name] for name in handle.files}
        changed["cav_vectors"] = changed["cav_vectors"].copy()
        changed["cav_vectors"][1, 0, 0] += 1e-9
        np.savez(path, **changed)
        _refuses(lambda: load_frozen_references(cache, 2007), "changed", RuntimeError)
        path.write_bytes(original)
        assert len(load_frozen_references(cache, 2007)) == 5

        performance = run_reference_performance_unit(cache, performance_view, roles, config, "cpu")
        assert performance["status"] == "ok" and performance["key"] == "D/2007/reference_performance" and performance["stage"] == "D.reference_performance"
        assert performance["dependencies"] == {key: cache._latest[key]["id"] for key in ("P/2007", "D/2007/cavs_selection")}
        stored_performance = cache.load("D/2007/reference_performance")
        expected = reference_performance_unit(performance_view, 2007, roles, output, world["family_output"],
                                              load_frozen_references(cache, 2007), dictionaries, config, "cpu")
        assert _unit_json(stored_performance) == _unit_json(expected) and len(stored_performance["rows"]) == 4
        assert run_reference_performance_unit(cache, performance_view, roles, config, "cpu")["id"] == performance["id"]
        used = sum(path.stat().st_size for path in cache._unit_path("D/2007/reference_performance").rglob("*") if path.is_file())
        assert 0 < used <= projected_performance_bytes(config)
        assert cache.completeness_table(["D/2007/cavs_selection", "D/2007/reference_performance", "D/2008/cavs_selection"]) == [
            {"unit": "D/2007/cavs_selection", "status": "complete"}, {"unit": "D/2007/reference_performance", "status": "complete"},
            {"unit": "D/2008/cavs_selection", "status": "not_run (stopped)"}]

        for failing in ("failed", "unavailable"):
            broken = open_cache(f"cache-{failing}", until="families")
            if failing == "failed":
                assert broken.run("D/2007/rules", lambda: (_ for _ in ()).throw(FloatingPointError("injected")))["status"] == "failed"
                reason = "dep_failed:D/2007/rules"
            else:
                bad = open_cache("cache-no-families", until="P")
                assert bad.run("D/2007/families", lambda: (_ for _ in ()).throw(FloatingPointError("injected")))["status"] == "failed"
                broken, reason = bad, "dep_failed:D/2007/families"
                assert broken.run("D/2007/rules", lambda: rules_payload(world["frozen"]))["status"] == "unavailable"
            record = run_cavs_selection_unit(broken, view, roles, config, "cpu")
            assert (record["status"], record["reason"]) == ("unavailable", reason), record
            assert not broken.is_complete("D/2007/cavs_selection") and not broken._unit_path("D/2007/cavs_selection").exists()
            follow = run_reference_performance_unit(broken, performance_view, roles, config, "cpu")
            assert (follow["status"], follow["reason"]) == ("unavailable", reason), follow
            assert [row["status"] for row in broken.completeness_table(["D/2007/cavs_selection", "D/2007/reference_performance"])] == [
                f"unavailable:{reason}"] * 2

        crashing = open_cache("cache-crash")

        def broken_fit(*arguments):
            raise FloatingPointError("injected")

        record = run_cavs_selection_unit(crashing, view, roles, config, "cpu", fit=broken_fit)
        assert record["status"] == "failed" and record["reason"] == "FloatingPointError: injected"
        assert not crashing._unit_path("D/2007/cavs_selection").exists()
        follow = run_reference_performance_unit(crashing, performance_view, roles, config, "cpu")
        assert (follow["status"], follow["reason"]) == ("unavailable", "dep_failed:D/2007/cavs_selection")
        assert run_cavs_selection_unit(crashing, view, roles, config, "cpu")["status"] == "ok"
        assert run_reference_performance_unit(crashing, performance_view, roles, config, "cpu")["status"] == "ok"

        empty_cache = UnitCache(work / "cache-empty", config, "source-fixture", free_bytes=free)
        assert empty_cache.run("P/2007", lambda: dict(output))["status"] == "ok"
        for seed in config["sae_seeds"]:
            assert empty_cache.run(f"S/{seed}/2007", lambda seed=seed: _identity_sae_payload(seed, dimension))["status"] == "ok"
        assert empty_cache.run("D/2007/families", lambda: dict(world["family_output"]))["status"] == "ok"
        assert empty_cache.run("D/2007/rules", lambda: rules_payload({}))["status"] == "ok"
        record = run_cavs_selection_unit(empty_cache, view, roles, config, "cpu")
        assert record["status"] == "ok" and empty_cache.load("D/2007/cavs_selection")["families"] == []
        assert run_reference_performance_unit(empty_cache, performance_view, roles, config, "cpu")["status"] == "ok"
        assert empty_cache.load("D/2007/reference_performance")["rows"] == []


def run_cav_selection_checks(notebook_path="concept_subgroup_drift_pilot.ipynb"):
    """Continuation round trip, medoid CAVs, independent TCAV selection per idea and reference performance on planted subgroups."""
    presets = _notebook_presets(json.loads(Path(notebook_path).read_text()))
    config = resolve_config(presets, "v0")
    assert config["profile"] == "v0" and config["cav_seeds"] == [42, 43, 44] and config["primary_cav_seed"] == 42
    assert config["tcav_cutoffs"] == [.1, .9] and config["bootstrap_replicates"] == 10 and config["classification_threshold"] == .5
    assert STAGE_CODE_VERSIONS["P"] >= 2
    _check_portable_semantics("cpu")
    _check_continuation_units(config, "cpu")
    _check_cav_fitting()
    _check_tcav_selection_rules()
    _check_selection_descriptions()
    world = _selection_world()
    unit, output = _check_selection_unit(config, world)
    _check_performance_unit(config, world, unit, output)
    _check_cav_units(config, world)
    print("Continuation round trip, medoid CAVs, independent TCAV selection and reference performance fixtures passed")


def run_identity_checks():
    """Identity depends on typed scientific values, never allocation or dict order."""
    import numpy as np
    import torch
    left = {"tensor": torch.tensor([[1., 2.], [3., 4.]]),
            "nested": {"metadata": [1, None, True], "array": np.asarray([[1., 2.], [3., 4.]], dtype=np.float32)}}
    right = {"nested": {"array": np.asfortranarray(np.asarray([[1., 2.], [3., 4.]], dtype=np.float32)),
                        "metadata": [1, None, True]},
             "tensor": torch.tensor([[1., 2.], [3., 4.]])}
    assert _identity_hash(left) == _identity_hash(right), "Equivalent typed values depend on allocation/dictionary order/layout"
    assert _identity_hash({"a": 1, "b": 2}) == _identity_hash({"b": 2, "a": 1})
    assert _identity_hash(np.asarray([1, 2], dtype=np.float32)) != _identity_hash(np.asarray([1, 2], dtype=np.float64))
    assert _identity_hash(np.asarray([[1, 2]], dtype=np.float32)) != _identity_hash(np.asarray([1, 2], dtype=np.float32))
    assert _identity_hash(np.asarray([1, 2])) != _identity_hash(np.asarray([1, 3]))
    assert _identity_hash((1, 2)) != _identity_hash([1, 2])
    assert _identity_hash(True) != _identity_hash(1)
    assert _identity_hash(np.asarray(["a", "b"])) == _identity_hash(np.asarray(["a", "b"]))
    assert _identity_hash(np.asarray([1, "b"], dtype=object)) == _identity_hash(np.asarray([1, "b"], dtype=object))
    assert _identity_hash(float("nan")) == _identity_hash(float("nan"))
    assert _identity_hash(float("inf")) != _identity_hash(float("-inf"))
    from tabpfn.scripts.estimator.configs import EnsembleConfiguration, PreprocessorConfig
    # Actual configuration classes must have explicit value adapters.
    config = EnsembleConfiguration(class_shift_configuration=torch.tensor([1, 0]),
        feature_shift_configuration=torch.tensor([0, 1]),
        preprocess_transform_configuration=PreprocessorConfig(name="none"),
        styles_configuration=None, subsample_samples_configuration=None)
    assert _identity_hash(config) == _identity_hash(deepcopy(config))
    from tabpfn.scripts.estimator.preprocessing import RemoveConstantFeaturesStep, SequentialFeatureTransformer
    from sklearn.preprocessing import StandardScaler
    step = RemoveConstantFeaturesStep()
    step.fit(np.asarray([[1., 2.], [3., 2.]]), [])
    pipeline = SequentialFeatureTransformer([step])
    pipeline.categorical_features_ = []
    original = _identity_hash(pipeline)
    assert original == _identity_hash(deepcopy(pipeline))
    step.sel_[1] = True
    assert original != _identity_hash(pipeline), "Fitted column selection mutation ignored"
    scaler = StandardScaler().fit(np.asarray([[1., 2.], [3., 4.]]))
    original = _identity_hash(scaler)
    assert original == _identity_hash(deepcopy(scaler))
    scaler.mean_[0] += 1
    assert original != _identity_hash(scaler), "Fitted scaler mutation ignored"
    try:
        _identity_hash(object())
    except TypeError as error:
        assert "unsupported" in str(error).lower()
    else:
        raise AssertionError("Unregistered opaque object accepted for scientific identity")
    from tabpfn.scripts.estimator.preprocessing import ReshapeFeatureDistributionsStep
    singleton = ReshapeFeatureDistributionsStep.none_transformer
    original_hash = _identity_hash(singleton)
    for name in ("func", "inverse_func"):
        original_callback = getattr(singleton, name)
        try:
            setattr(singleton, name, lambda x: x + 1)
            try:
                _identity_hash(singleton)
            except TypeError:
                pass
            else:
                raise AssertionError(f"Known identity singleton accepted changed {name} callback")
        finally:
            setattr(singleton, name, original_callback)
        assert _identity_hash(singleton) == original_hash
    print("Typed identity allocation, order, layout and unsupported-object fixtures passed")


def annual_fitting_mask(patient_ids, reference_roles, experiment_id, seed=42):
    import numpy as np
    _assert_patient_separation(reference_roles)
    mapping = {patient: role for role, patients in reference_roles.items() for patient in patients}
    fitting = {"tabpfn_context", "sae_discovery", "rule_discovery", "cav_fit"}
    return np.asarray([mapping[patient] in fitting if patient in mapping else
                       stable_entrant_role(patient, experiment_id, seed) == "fitting"
                       for patient in _canonical_patient_ids(patient_ids)], dtype=bool)


def rule_activation_agreement(rule_mask, activation_mask):
    import numpy as np
    rule, activation = np.asarray(rule_mask, dtype=bool), np.asarray(activation_mask, dtype=bool)
    if rule.ndim != 1 or rule.shape != activation.shape:
        raise ValueError("Rule/activation fidelity masks must share ordered rows")
    tp = int(np.sum(rule & activation))
    fp = int(np.sum(rule & ~activation))
    fn = int(np.sum(~rule & activation))
    result = {"records": len(rule), "intersection": tp, "rule_records": int(rule.sum()),
              "activation_records": int(activation.sum()), "unavailable_reasons": {}}
    for name, numerator, denominator in (("precision", tp, tp + fp), ("recall", tp, tp + fn),
        ("f2", 5 * tp, 5 * tp + 4 * fn + fp), ("jaccard", tp, tp + fp + fn),
        ("rule_coverage", int(rule.sum()), len(rule)), ("activation_coverage", int(activation.sum()), len(rule))):
        result[name] = numerator / denominator if denominator else None
        if not denominator:
            result["unavailable_reasons"][name] = "zero_metric_denominator"
    return result


def fit_yearly_descriptive_rule(frame, activations, target, reference_roles,
                                experiment_id, feature_names, year, seed=42, rule_fitter=None):
    import numpy as np
    values = np.asarray(activations, dtype=float)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError("Yearly fitting activations must be finite and aligned")
    if "year" not in frame or (frame.year > year).any():
        raise ValueError("Yearly descriptive rules cannot access future rows")
    fitting = annual_fitting_mask(frame.patient_id, reference_roles, experiment_id, seed)
    current_fitting = fitting & (frame.year.to_numpy() == year)
    if not current_fitting.any():
        return {"rule": None, "failure_reason": "empty_current_year_fitting_population", "fitting_records": 0}
    before = _identity_hash(target)
    fitter = fit_reference_rule if rule_fitter is None else rule_fitter
    result = fitter(frame.loc[current_fitting, list(feature_names)].to_numpy(), values[current_fitting],
                    deepcopy(target), frame.loc[current_fitting, "patient_id"].tolist(), feature_names, seed)
    if _identity_hash(target) != before:
        raise RuntimeError("Yearly rule fitting changed the frozen reference threshold")
    result.update(year=int(year), fitting_records=int(current_fitting.sum()),
        fitting_row_identity=_identity_hash(list(zip(frame.loc[current_fitting, "patient_id"], frame.loc[current_fitting, "year"]))),
        frozen_target_identity=before, rule_role="yearly_descriptive_only")
    return result


def fixed_factor_observation(frame, activations, gradients, frozen_reference,
                             factor_identity, factor_id, sae_seed, baseline,
                             reference_statistics, year, reference_year,
                             feature_names, new_rule=None, population="pipeline_unseen",
                             bootstrap_replicates=100, bootstrap_seed=42):
    """Score an already held-out annual cohort; descriptive rules never change A."""
    import numpy as np
    assert_reference_identity(frozen_reference)
    if year < reference_year or "year" not in frame or not (frame.year == year).all():
        raise ValueError("Fixed-factor observation must use exactly its eligible calendar year")
    values, raw_gradients = np.asarray(activations, dtype=float), np.asarray(gradients, dtype=float)
    if values.shape != (len(frame),) or raw_gradients.ndim != 2 or len(raw_gradients) != len(frame):
        raise ValueError("Annual activations/gradients must share the cohort rows")
    if not np.isfinite(values).all() or not np.isfinite(raw_gradients).all():
        raise ValueError("Annual coordinates/activations/gradients must be finite")
    A = strict_activation_mask(values, frozen_reference["target"])
    X = frame[list(feature_names)].to_numpy()
    original = np.asarray(frozen_reference["rule"].mask(X), dtype=bool)
    result = {"year": int(year), "reference_year": int(reference_year), "factor_identity": str(factor_identity),
        "factor_id": int(factor_id), "sae_seed": int(sae_seed), "population": str(population),
        "reference_identity": frozen_reference["identity"], "reference_eligibility_frozen": True,
        "records": len(frame), "patients": len(set(frame.patient_id)),
        "row_identity": _identity_hash(list(zip(frame.patient_id, frame.year))),
        "activation_membership": A.tolist(), "activation_records": int(A.sum()),
        "original_rule_agreement": rule_activation_agreement(original, A),
        "new_rule_agreement": rule_activation_agreement(new_rule.mask(X), A) if new_rule is not None else None,
        "new_rule_unavailable_reason": None if new_rule is not None else "descriptive_rule_unavailable",
        "unavailable_reasons": {}}
    positive = values > 0
    result["activation_prevalence"] = float(A.mean()) if len(A) else None
    result["positive_activation_mean"] = float(values[positive].mean()) if positive.any() else None
    result["activation_subgroup_mean"] = float(values[A].mean()) if A.any() else None
    result["positive_activation_records"] = int(positive.sum())
    for ratio, value_key in (("prevalence_ratio", "activation_prevalence"), ("magnitude_ratio", "positive_activation_mean")):
        current, denominator = result[value_key], reference_statistics.get(value_key)
        result[ratio] = current / denominator if current is not None and denominator is not None and denominator != 0 else None
        if result[ratio] is None:
            result["unavailable_reasons"][ratio] = "zero_reference_denominator" if denominator == 0 else "unavailable_reference_or_current_value"
    result["reference_statistics_identity"] = _identity_hash(reference_statistics)
    result["reliability"] = subgroup_reliability(frame.loc[A, "y"].to_numpy(), frame.loc[A, "p"].to_numpy(),
                                                 frame.loc[A, "patient_id"].tolist(), baseline,
                                                 bootstrap_replicates, bootstrap_seed)
    result["subgroup_influence"] = _frozen_influence(raw_gradients[A], frozen_reference["cavs"])
    result["whole_cohort_influence"] = _frozen_influence(raw_gradients, frozen_reference["cavs"])
    assert_reference_identity(frozen_reference)
    return result


def _frozen_influence(gradients, cavs):
    import numpy as np
    result = {}
    for seed, cav in cavs.items():
        reason = None
        if not len(gradients):
            reason = "empty_subgroup"
        elif not cav["valid"]:
            reason = cav.get("failure_reason", "unavailable_cav")
        else:
            direction = np.asarray(cav["cav"], dtype=float)
            if direction.shape != (gradients.shape[1],) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
                reason = "invalid_cav_direction"
        if reason:
            result[int(seed)] = {"tcav": None, "signed_sensitivity": None, "absolute_sensitivity": None, "unavailable_reason": reason}
        else:
            projected = gradients @ direction
            result[int(seed)] = {"tcav": float(np.mean(projected > 0)), "signed_sensitivity": float(projected.mean()),
                                "absolute_sensitivity": float(np.abs(projected).mean()), "unavailable_reason": None}
    return result


IDEA1_VIEW_KEYS = PERFORMANCE_VIEW_KEYS
IDEA1_QUALITY_FILTERS = ("min_precision", "min_lift", "min_marginal_recall")
IDEA1_RESERVED_COLUMNS = ("patient_id", "year", "y", "p")
IDEA1_UNAVAILABLE_FIT = "empty_current_year_fitting_population"
IDEA1_UNAVAILABLE_SPLIT = "insufficient_inner_rule_partitions"


def idea1_follow_up_years(config, r):
    """Follow-up years t > r that the unit graph assigns D(r)/idea1(t) units to; none for a T0-only reference year."""
    prefix = f"D/{int(r)}/idea1/"
    return [int(key[len(prefix):]) for key in unit_graph(config) if key.startswith(prefix)]


def descriptive_selection_config(rule):
    """Selection settings of the yearly descriptive rules: the reference search limits (objective, rule count and length)
    with every quality filter at zero. The candidate generation (backend, per-bootstrap cap, recurrence) is not a filter."""
    selection = rule_selection_config({**rule, **dict.fromkeys(IDEA1_QUALITY_FILTERS, 0.)})
    if any(getattr(selection, name) != 0 for name in IDEA1_QUALITY_FILTERS):
        raise RuntimeError("Yearly descriptive rules must not use quality filters")
    return selection


def _rule_terms(rule):
    from semantic_rules import RuleSet
    rule_set = RuleSet.from_dict(rule) if isinstance(rule, dict) else rule
    conditions = {f"{c.feature_name} {c.operator} {c.threshold:.4g}" for member in rule_set.rules for c in member.conditions}
    features = {c.feature_name for member in rule_set.rules for c in member.conditions}
    return conditions, features, _identity_hash(rule_set.to_dict())


def describe_rule_change(reference_rule, yearly_rule, X, feature_names):
    """How the yearly descriptive rule differs from the frozen reference rule: identity, conditions and features added or
    removed, and the Jaccard of the two row masks on the given held-out rows."""
    import numpy as np
    reference_conditions, reference_features, reference_identity = _rule_terms(reference_rule)
    yearly_conditions, yearly_features, yearly_identity = _rule_terms(yearly_rule)
    reference_rows = np.asarray(rule_mask(reference_rule, X, feature_names), dtype=bool)
    yearly_rows = np.asarray(rule_mask(yearly_rule, X, feature_names), dtype=bool)
    union = int((reference_rows | yearly_rows).sum())
    return {"identical_rule": reference_identity == yearly_identity,
            "reference_rule_identity": reference_identity, "yearly_rule_identity": yearly_identity,
            "reference_rule": rule_text(reference_rule), "yearly_rule": rule_text(yearly_rule),
            "conditions_added": sorted(yearly_conditions - reference_conditions),
            "conditions_removed": sorted(reference_conditions - yearly_conditions),
            "conditions_kept": len(reference_conditions & yearly_conditions),
            "features_added": sorted(yearly_features - reference_features),
            "features_removed": sorted(reference_features - yearly_features),
            "features_kept": len(reference_features & yearly_features),
            "evaluation_rows": int(len(reference_rows)), "reference_rule_rows": int(reference_rows.sum()),
            "yearly_rule_rows": int(yearly_rows.sum()),
            "mask_jaccard": int((reference_rows & yearly_rows).sum()) / union if union else None,
            "unavailable_reason": None if union else "both_rules_select_no_rows"}


def _population_counts(labels, patients, entrant, a_mask, r_mask):
    """Evaluation rows, patients, deaths and subgroup sizes of returning-reference and entrant patients, kept separate."""
    import numpy as np
    result = {}
    for name, flag in (("returning", ~entrant), ("entrant", entrant)):
        result[name] = {"rows": int(flag.sum()), "patients": int(len(np.unique(patients[flag]))), "deaths": int(labels[flag].sum()),
                        "A_rows": int((a_mask & flag).sum()), "A_deaths": int(labels[a_mask & flag].sum()),
                        "R_rows": int((r_mask & flag).sum()), "R_deaths": int(labels[r_mask & flag].sum())}
    return result


def projected_idea1_bytes(config, latent_bound=768, row_bound=8192, rule_bytes=32 * 1024):
    """Upper bound of one D/r/idea1/t unit's cached bytes: per family a rule, three row masks and the metric records."""
    cap = candidate_cap(latent_bound, config["candidate_cap_fraction"])
    return cap * (rule_bytes + 3 * row_bound + 64 * 1024) + row_bound * 8 + 4 * 2**20


def idea1_year_unit(view, r, t, roles, predictor_output, family_output, references, dictionaries, config, device, *,
                    workers=1, backend=None, select=None, recurrence="configured"):
    """D(r)/idea1(t): for every Idea-1-eligible frozen family, the activation subgroup A_t of follow-up year t under the frozen
    medoid SAE, scaler and threshold; transfer of the frozen rule; a yearly descriptive rule fitted on the year-t permanent
    fitting patients with no quality filters; reliability inside A_t and the frozen rule subgroup R on the held-out year-t
    evaluation patients against the baseline q fixed at the reference; frozen-CAV influence from the cached gradients."""
    import numpy as np
    import pandas
    from semantic_rules import RuleSet
    r, t = int(r), int(t)
    if device not in ("cpu", "cuda"):
        raise ValueError("Idea 1 needs a resolved device: cpu or cuda")
    if set(view) != set(IDEA1_VIEW_KEYS):
        raise ValueError(f"Idea 1 reads only {IDEA1_VIEW_KEYS}")
    if t not in idea1_follow_up_years(config, r):
        raise ValueError(f"Year {t} is not a follow-up year of reference year {r}")
    _require_selection_settings(config)
    if config["classification_threshold"] != 0.5 or config["confidence_level"] != 0.95:
        raise ValueError("Idea 1 supports the 0.5 threshold and 95% intervals only")
    configured = isinstance(recurrence, str) and recurrence == "configured"
    if (backend is not None or select is not None or not configured) and config.get("profile") != "v0":
        raise ValueError("Backend, selection and candidate-filter overrides exist for the v0 fixtures only")
    if configured:
        recurrence = config["rule"]["recurrence"]
    if recurrence is not None and set(recurrence) != set(RECURRENCE_KEYS):
        raise ValueError(f"Recurrence settings must be exactly {RECURRENCE_KEYS}")
    names = list(view["feature_names"])
    if set(names) & set(IDEA1_RESERVED_COLUMNS):
        raise ValueError(f"Feature names must not use the reserved columns {IDEA1_RESERVED_COLUMNS}")
    for reference in references.values():
        assert_reference_identity(reference)
    selection_config, settings = descriptive_selection_config(config["rule"]), _backend_settings(backend)
    family_rows = _family_rows(view, r, roles, family_output)
    reference_fitting, _ = _predictor_positions(view, r, roles, predictor_output, FITTING_SIDE_ROLE_NAMES)
    reference_evaluation, _ = _predictor_positions(view, r, roles, predictor_output, ("evaluation",))
    year_rows, year_role, year_entrant = year_roles(view, roles, t)
    fitting = year_rows[np.where(year_entrant, year_role == "fitting", np.isin(year_role, FITTING_SIDE_ROLE_NAMES))]
    evaluation_flag = year_role == "evaluation"
    evaluation, evaluation_entrant = year_rows[evaluation_flag], year_entrant[evaluation_flag]
    cached_index = np.asarray(predictor_output["row_index"])
    locate = lambda rows: np.searchsorted(cached_index, rows)
    entries = [reference for reference in references.values() if reference["eligible"]["idea1"]]
    rows = np.concatenate([reference_fitting, reference_evaluation, fitting, evaluation])
    bounds = np.cumsum([len(reference_fitting), len(reference_evaluation), len(fitting)])
    activations = _medoid_activations(rows, np.asarray(predictor_output["embeddings"])[locate(rows)], entries, family_rows,
                                      dictionaries, config, device)
    labels, patients, X = view["y"], view["patient_id"], view["X"]
    reference_labels = labels[reference_fitting]
    reference_features = X[reference_fitting].astype(np.float64)
    fitting_features, evaluation_features = X[fitting].astype(np.float64), X[evaluation].astype(np.float64)
    if not np.isfinite(fitting_features).all() or not np.isfinite(evaluation_features).all():
        raise ValueError("Year-t fitting and evaluation features must be finite")
    evaluation_labels, evaluation_patients = labels[evaluation], patients[evaluation]
    probabilities = np.asarray(predictor_output["probabilities"])[locate(evaluation), 1].astype(np.float64)
    gradients = np.asarray(predictor_output["gradients"])[locate(evaluation)].astype(np.float64)
    frame = pandas.DataFrame(evaluation_features, columns=names)
    frame.insert(0, "patient_id", evaluation_patients)
    frame["year"], frame["y"], frame["p"] = t, evaluation_labels, probabilities
    split_seed, replicates = int(config["split_seed"]), int(config["bootstrap_replicates"])
    prepared, jobs = [], []
    for position, reference in enumerate(entries):
        values = activations[position]
        target, family_id = reference["target"], reference["family_id"]
        a_reference = strict_activation_mask(values[:bounds[0]], target)
        r_reference = rule_mask(reference["rule"], reference_features, names)
        reference_values = values[bounds[0]:bounds[1]]
        reference_a = strict_activation_mask(reference_values, target)
        statistics = {"activation_prevalence": float(reference_a.mean()) if len(reference_a) else None,
                      "positive_activation_mean": float(reference_values[reference_values > 0].mean()) if (reference_values > 0).any() else None}
        baselines = {"A": fit_constant_baseline(reference_labels[a_reference]), "R": fit_constant_baseline(reference_labels[r_reference])}
        fit_values = values[bounds[1]:bounds[2]]
        binary = strict_activation_mask(fit_values, target)
        record = {"position": position, "reference": reference, "baselines": baselines, "statistics": statistics,
                  "fit_positives": int(binary.sum()), "job": None, "unavailable": None}
        if not len(fitting):
            record["unavailable"] = IDEA1_UNAVAILABLE_FIT
        else:
            ids = patients[fitting]
            fit_half, tune_half = rule_inner_split(ids, binary.astype(float),
                                                   _derived_seed("idea1_inner_split", split_seed, r, t, family_id))
            fit_mask, tune_mask = np.isin(ids, fit_half), np.isin(ids, tune_half)
            record["inner_split"] = {"stratified_by": "frozen_threshold_target", "fit_rows": int(fit_mask.sum()),
                                     "tune_rows": int(tune_mask.sum()), "fit_positives": int(binary[fit_mask].sum()),
                                     "tune_positives": int(binary[tune_mask].sum()),
                                     "fit_patients_identity": _identity_hash(fit_half), "tune_patients_identity": _identity_hash(tune_half)}
            if not (binary[fit_mask].any() and binary[tune_mask].any() and (~binary[fit_mask]).any()):
                record["unavailable"] = IDEA1_UNAVAILABLE_SPLIT
            else:
                jobs.append({"row": {"backend_seed": _derived_seed("idea1_rule_backend", split_seed, r, t, family_id),
                                     "threshold_name": "frozen_threshold"},
                             "X_fit": fitting_features[fit_mask], "y_fit": binary[fit_mask], "patients_fit": ids[fit_mask],
                             "X_tune": fitting_features[tune_mask], "y_tune": binary[tune_mask], "feature_names": names,
                             "backend": settings, "selection_config": selection_config, "select": select, "recurrence": recurrence})
                record["job"] = len(jobs) - 1
        prepared.append(record)
    fitted = _run_sweep_jobs(jobs, workers) if jobs else []
    families = []
    masks = {name: np.zeros((len(entries), len(evaluation)), dtype=bool) for name in ("A", "R", "yearly")}
    for record in prepared:
        position, reference = record["position"], record["reference"]
        family_id = reference["family_id"]
        yearly, rule_set = {"status": "unavailable", "reason": record["unavailable"], "fitting_rows": int(len(fitting)),
                            "fitting_positives": record["fit_positives"], "inner_split": record.get("inner_split")}, None
        if record["job"] is not None:
            result = fitted[record["job"]]
            yearly.update(candidate_count=result["candidate_count"], family_count=result["family_count"],
                          offered_count=result["offered_count"], funnel_stage=result["funnel_stage"],
                          search_method=result["search_method"], backend_warnings=result["backend_warnings"])
            if result["valid"]:
                rule_set = RuleSet.from_dict(result["rule_set"])
                yearly.update(status="fitted", reason=None, rule_set=result["rule_set"], rule_identity=result["rule_identity"],
                              rule_text=rule_text(rule_set), rule_count=result["rule_count"], condition_count=result["condition_count"],
                              tune_metrics={name: result[name] for name in SWEEP_METRICS})
            else:
                yearly["reason"] = result["failure_reason"] or result["funnel_stage"]
        if yearly["status"] != "fitted":
            yearly["reason"] = str(yearly["reason"])
        seed = lambda subgroup: _derived_seed("idea1_reliability", split_seed, r, t, family_id, subgroup)
        observation = fixed_factor_observation(frame, activations[position, bounds[2]:], gradients, reference, family_id,
                                               reference["medoid"][1], reference["medoid"][0], record["baselines"]["A"],
                                               record["statistics"], t, r, names, new_rule=rule_set,
                                               population="heldout_evaluation", bootstrap_replicates=replicates,
                                               bootstrap_seed=seed("A"))
        a_mask = np.asarray(observation.pop("activation_membership"), dtype=bool)
        r_mask = np.asarray(rule_mask(reference["rule"], evaluation_features, names), dtype=bool)
        reliability_r = subgroup_reliability(evaluation_labels[r_mask], probabilities[r_mask], evaluation_patients[r_mask],
                                             record["baselines"]["R"], replicates, seed("R"))
        masks["A"][position], masks["R"][position] = a_mask, r_mask
        change = None
        if rule_set is not None:
            masks["yearly"][position] = rule_mask(rule_set, evaluation_features, names)
            change = describe_rule_change(reference["rule"], rule_set, evaluation_features, names)
        observation["subgroup_influence"] = {str(key): value for key, value in observation["subgroup_influence"].items()}
        observation["whole_cohort_influence"] = {str(key): value for key, value in observation["whole_cohort_influence"].items()}
        reliability_a = observation.pop("reliability")
        transfer, descriptive = observation.pop("original_rule_agreement"), observation.pop("new_rule_agreement")
        families.append({
            "family_id": family_id, "medoid": [int(value) for value in reference["medoid"]],
            "activation_target": reference["activation_target"], "cutoff": reference["cutoff"],
            "reference_identity": reference["identity"], "reference_statistics": record["statistics"],
            "baselines": record["baselines"], "activation_membership_identity": _identity_hash(a_mask),
            "observation": observation, "original_rule_transfer": transfer,
            "yearly_rule": yearly, "rule_change": change, "yearly_rule_agreement": descriptive,
            "reliability": {"A": reliability_a, "R": reliability_r},
            "populations": _population_counts(evaluation_labels, evaluation_patients, evaluation_entrant, a_mask, r_mask)})
    for reference in references.values():
        assert_reference_identity(reference)
    not_computed = [{"family_id": reference["family_id"], "reason": "not_eligible_at_selection"}
                    for reference in references.values() if not reference["eligible"]["idea1"]]
    fitting_entrant = year_entrant[np.isin(year_rows, fitting)]
    return {
        "evaluation_row_index": evaluation.astype(np.int64), "evaluation_is_entrant": evaluation_entrant,
        "mask_A": masks["A"], "mask_R": masks["R"], "mask_yearly_rule": masks["yearly"],
        "yearly_rule_available": np.asarray([family["yearly_rule"]["status"] == "fitted" for family in families], dtype=bool),
        "_rows": int(len(evaluation)), "reference_year": r, "year": t, "families": families, "not_computed": not_computed,
        "family_ids": [family["family_id"] for family in families], "jobs": len(jobs),
        "fitting": {"rows": int(len(fitting)), "patients": int(len(np.unique(patients[fitting]))),
                    "returning_rows": int((~fitting_entrant).sum()), "entrant_rows": int(fitting_entrant.sum()),
                    "roles": list(FITTING_SIDE_ROLE_NAMES) + ["entrant:fitting"]},
        "evaluation": {"rows": int(len(evaluation)), "patients": int(len(np.unique(evaluation_patients))),
                       "deaths": int(evaluation_labels.sum()), "returning_rows": int((~evaluation_entrant).sum()),
                       "entrant_rows": int(evaluation_entrant.sum()), "role": "evaluation"},
        "fitting_rows_identity": _identity_hash({"row_index": fitting}),
        "evaluation_rows_identity": _identity_hash({"row_index": evaluation}),
        "reference_fitting_rows_identity": _identity_hash({"row_index": reference_fitting}),
        "reference_evaluation_rows_identity": _identity_hash({"row_index": reference_evaluation}),
        "rule_selection": {name: getattr(selection_config, name) for name in selection_config.__dataclass_fields__},
        "backend": settings, "candidate_filter": None if recurrence is None else dict(recurrence),
        "backend_overridden": backend is not None,
        "bootstrap": {"replicates": replicates, "confidence_level": 0.95, "sampling_unit": "patient",
                      "seed": "sha256 of (idea1_reliability, split_seed, reference year, year, family, subgroup)"},
        "classification_threshold": 0.5,
        "references": {reference["family_id"]: reference["identity"] for reference in references.values()},
        "roles_identity": roles["identity"], "population_identity": view["fingerprint"]["population_identity"],
        "predictor_output_row_identity": predictor_output["output_row_identity"],
        "gradient_source": "cached P(r) output; the predictor is not rerun",
        "baseline_source": "reference-year fitting-role rows inside the frozen subgroup; not refitted",
        "outcomes_read": "reference-year fitting-role rows and year-t evaluation rows only"}


def run_idea1_unit(cache, view, roles, config, device, t, *, workers=None, backend=None, select=None, recurrence="configured"):
    """Run D(r)/idea1(t) through the cache; P(r), families and the frozen references are loaded from the cache."""
    if config_hash(config) != cache.config_hash:
        raise ValueError("Configuration differs from the cache configuration")
    if set(view) != set(IDEA1_VIEW_KEYS):
        raise ValueError(f"Idea 1 reads only {IDEA1_VIEW_KEYS}")
    r, t = int(roles["reference_year"]), int(t)
    if t not in idea1_follow_up_years(config, r):
        raise ValueError(f"Year {t} is not a follow-up year of reference year {r}")
    pool = config["joblib_workers"] if workers is None else workers
    return cache.run(f"D/{r}/idea1/{t}",
                     lambda: idea1_year_unit(view, r, t, roles, cache.load(f"P/{r}"), cache.load(f"D/{r}/families"),
                                             load_frozen_references(cache, r), load_reference_dictionaries(cache, r, device),
                                             config, device, workers=pool, backend=backend, select=select, recurrence=recurrence),
                     projected_bytes=projected_idea1_bytes(config))


def idea1_table(unit):
    """Compact rows per frozen family: A_t prevalence and ratio, rule transfer, yearly rule and its change, reliability, TCAV, counts."""
    def number(value, digits=3):
        return None if value is None else round(value, digits)
    rows = []
    for family in unit["families"]:
        observation, transfer = family["observation"], family["original_rule_transfer"]
        yearly, change, descriptive = family["yearly_rule"], family["rule_change"], family["yearly_rule_agreement"]
        a_metrics, r_metrics = family["reliability"]["A"], family["reliability"]["R"]
        populations = family["populations"]
        rows.append({"family": family["family_id"], "t": unit["year"], "A rows": observation["activation_records"],
                     "A prevalence": number(observation["activation_prevalence"]), "prevalence ratio": number(observation["prevalence_ratio"]),
                     "R to A F2": number(transfer["f2"]), "R to A Jaccard": number(transfer["jaccard"]),
                     "yearly rule": yearly["status"] if yearly["status"] == "fitted" else f"unavailable:{yearly['reason']}",
                     "yearly to A F2": None if descriptive is None else number(descriptive["f2"]),
                     "yearly to A Jaccard": None if descriptive is None else number(descriptive["jaccard"]),
                     "rule +/- features": None if change is None else f"+{len(change['features_added'])}/-{len(change['features_removed'])}",
                     "ref/yearly mask Jaccard": None if change is None else number(change["mask_jaccard"]),
                     "A Brier": number(a_metrics["brier"]), "A skill": number(a_metrics["brier_skill"]),
                     "R Brier": number(r_metrics["brier"]), "R skill": number(r_metrics["brier_skill"]),
                     "A TCAV 42": number(observation["subgroup_influence"]["42"]["tcav"]),
                     "eval returning/entrant": f"{populations['returning']['rows']}/{populations['entrant']['rows']}",
                     "unavailable": {"A": {name: reason for name, reason in a_metrics["unavailable_reasons"].items() if a_metrics.get(name) is None},
                                     "R": {name: reason for name, reason in r_metrics["unavailable_reasons"].items() if r_metrics.get(name) is None},
                                     "observation": dict(observation.get("unavailable_reasons", {})),
                                     "rule_transfer": dict(transfer.get("unavailable_reasons", {})),
                                     "yearly_agreement": (dict(descriptive.get("unavailable_reasons", {})) if descriptive is not None
                                                          else {"all": yearly.get("reason", "descriptive_rule_unavailable")}),
                                     "TCAV 42": observation["subgroup_influence"]["42"].get("unavailable_reason")}})
    return rows


IDEA2_FROZEN_VIEW_KEYS = RULE_VIEW_KEYS
IDEA2_YEARLY_VIEW_KEYS = MATCHING_VIEW_KEYS
IDEA2_UNAVAILABLE_FIT = "empty_current_year_fitting_population"
IDEA2_UNAVAILABLE_EVALUATION = "fewer_than_two_evaluation_rows"
IDEA2_UNAVAILABLE_RULE = "empty_rule_subgroup"
IDEA2_UNAVAILABLE_EMPTY_YEAR = "empty_evaluation_population"
IDEA2_LINK_STATES = ("retained", "not_mutual_full", "partial", "chain", "inconsistent", "unmatched")


def idea2_follow_up_years(config, r):
    """Follow-up years t > r that the unit graph assigns D(r)/idea2 units to; none for a T0-only reference year.
    For t = r the yearly view reuses the reference SAEs (see `idea2_reuse_record`) and has no unit."""
    prefix = f"D/{int(r)}/idea2_frozen/"
    return [int(key[len(prefix):]) for key in unit_graph(config) if key.startswith(prefix)]


def _idea2_year_rows(view, roles, t):
    """Year-t row indices of the permanent fitting patients (returning fitting-side roles plus fitting entrants) and of the
    held-out evaluation patients with their entrant flags. Selection patients belong to neither."""
    import numpy as np
    rows, role, entrant = year_roles(view, roles, t)
    fitting = rows[np.where(entrant, role == "fitting", np.isin(role, FITTING_SIDE_ROLE_NAMES))]
    flag = role == "evaluation"
    return fitting, rows[flag], entrant[flag]


def _year_embeddings(view, r, roles, predictor_output, rows):
    """Cached raw embeddings of the given population rows (any year >= r) after validating the cached P(r) output."""
    import numpy as np
    _role_embeddings(view, r, roles, predictor_output, ("evaluation",))
    cached = np.asarray(predictor_output["row_index"])
    position = np.searchsorted(cached, rows)
    if len(rows) and (position.max() >= len(cached) or not np.array_equal(cached[position], rows)):
        raise ValueError("Requested rows are absent from the cached predictor output")
    return np.asarray(predictor_output["embeddings"])[position]


def _activation_statistics(values, in_rule):
    """Prevalence (strictly positive activation) and magnitude of one factor over the rows inside the rule subgroup."""
    import numpy as np
    selected = np.asarray(values, dtype=np.float64)[in_rule]
    if not len(selected):
        return {"rows": 0, "positive_rows": 0, "positive_prevalence": None, "magnitude_mean": None,
                "positive_magnitude_mean": None, "unavailable": IDEA2_UNAVAILABLE_RULE}
    positive = selected > 0
    return {"rows": int(len(selected)), "positive_rows": int(positive.sum()), "positive_prevalence": float(positive.mean()),
            "magnitude_mean": float(selected.mean()),
            "positive_magnitude_mean": float(selected[positive].mean()) if positive.any() else None, "unavailable": None}


def _ratio(value, reference):
    return None if value is None or reference is None or reference == 0 else float(value / reference)


def projected_idea2_frozen_bytes(config, latent_bound=768, row_bound=8192):
    """Upper bound of one D/r/idea2_frozen/t unit's cached bytes: per family two rule masks and the member statistics."""
    cap = candidate_cap(latent_bound, config["candidate_cap_fraction"])
    return cap * (2 * row_bound + 64 * 1024) + 2 * row_bound * 9 + 4 * 2**20


def idea2_frozen_year_unit(view, r, t, roles, predictor_output, family_output, references, dictionaries, config, device):
    """D(r)/idea2_frozen(t): for every Idea-2-eligible frozen family, the frozen rule subgroup R inside the held-out year-t
    evaluation patients and, inside R, the retention of the medoid (share strictly above the frozen threshold) and the
    prevalence and magnitude of every member of the full reference family under its frozen SAE and scaler, each against
    the same quantity on the reference-year evaluation patients. Embeddings and features only; no outcome is read."""
    import numpy as np
    r, t = int(r), int(t)
    if device not in ("cpu", "cuda"):
        raise ValueError("Idea 2 needs a resolved device: cpu or cuda")
    if set(view) != set(IDEA2_FROZEN_VIEW_KEYS):
        raise ValueError(f"Idea 2 frozen view reads only {IDEA2_FROZEN_VIEW_KEYS}; outcomes are not accepted")
    if t not in idea2_follow_up_years(config, r):
        raise ValueError(f"Year {t} is not a follow-up year of reference year {r}")
    seeds = list(config["sae_seeds"])
    if list(dictionaries) != seeds:
        raise ValueError("Dictionaries must be given for exactly the configured SAE seeds, in order")
    for reference in references.values():
        assert_reference_identity(reference)
    _family_rows(view, r, roles, family_output)
    full = {family["id"]: family for family in family_output["families"] if family["kind"] == "full"}
    names = list(view["feature_names"])
    entries = [reference for reference in references.values() if reference["eligible"]["idea2"]]
    for reference in entries:
        family = full.get(reference["family_id"])
        if family is None or [int(value) for value in reference["medoid"]] != [int(value) for value in family["medoid"]]:
            raise ValueError(f"Frozen family {reference['family_id']} is not a full family of the families unit")
    reference_rows, _ = _role_embeddings(view, r, roles, predictor_output, ("evaluation",))
    _, evaluation, evaluation_entrant = _idea2_year_rows(view, roles, t)
    rows = np.concatenate([reference_rows, evaluation])
    split = len(reference_rows)
    embeddings = _year_embeddings(view, r, roles, predictor_output, rows) if len(rows) else None
    needed = sorted({int(seed) for reference in entries for seed, _ in full[reference["family_id"]]["members"]})
    codes = {seed: encode_with_reference_sae(dictionaries[seed], embeddings, device, config["encoding_batch_size"])
             for seed in needed} if len(evaluation) and entries else {}
    features = np.asarray(view["X"])[rows].astype(np.float64) if len(rows) else np.empty((0, len(names)))
    if len(rows) and not np.isfinite(features).all():
        raise ValueError("Features of the reference and year-t evaluation rows must be finite")
    families = []
    masks = {"R": np.zeros((len(entries), len(evaluation)), dtype=bool),
             "R_reference": np.zeros((len(entries), split), dtype=bool)}
    for position, reference in enumerate(entries):
        family = full[reference["family_id"]]
        in_rule = np.asarray(rule_mask(reference["rule"], features, names), dtype=bool) if len(rows) else np.zeros(0, dtype=bool)
        r_reference, r_year = in_rule[:split], in_rule[split:]
        masks["R_reference"][position], masks["R"][position] = r_reference, r_year
        record = {"family_id": reference["family_id"], "medoid": [int(value) for value in reference["medoid"]],
                  "activation_target": reference["activation_target"], "cutoff": reference["cutoff"],
                  "reference_identity": reference["identity"], "rule_mask_identity": _identity_hash(r_year),
                  "status": "ok", "reason": None, "members": [],
                  "rule_rows": {"year": int(r_year.sum()), "reference": int(r_reference.sum())},
                  "populations": {name: {"rows": int(flag.sum()), "R_rows": int((r_year & flag).sum()),
                                         "R_patients": int(len(np.unique(np.asarray(view["patient_id"])[evaluation][r_year & flag])))}
                                  for name, flag in (("returning", ~evaluation_entrant), ("entrant", evaluation_entrant))}}
        if not len(evaluation):
            record.update(status="unavailable", reason=IDEA2_UNAVAILABLE_EMPTY_YEAR)
        for seed, factor in family["members"] if record["status"] == "ok" else []:
            values = codes[int(seed)][:, int(factor)]
            then, now = _activation_statistics(values[:split], r_reference), _activation_statistics(values[split:], r_year)
            member = {"seed": int(seed), "factor": int(factor), "is_medoid": [int(seed), int(factor)] == record["medoid"],
                      "reference": then, "year": now,
                      "prevalence_ratio": _ratio(now["positive_prevalence"], then["positive_prevalence"]),
                      "magnitude_ratio": _ratio(now["magnitude_mean"], then["magnitude_mean"])}
            if member["is_medoid"]:
                above = strict_activation_mask(values.astype(np.float64), reference["target"])
                rate = lambda flag, selected: float(flag[selected].mean()) if selected.any() else None
                reference_rate, year_rate = rate(above[:split], r_reference), rate(above[split:], r_year)
                member["retention"] = {"cutoff": reference["cutoff"], "year_rate": year_rate, "reference_rate": reference_rate,
                                       "ratio": _ratio(year_rate, reference_rate), "year_rows_above": int(above[split:][r_year].sum()),
                                       "reference_rows_above": int(above[:split][r_reference].sum()),
                                       "definition": "share of rule-subgroup rows with medoid activation strictly above the frozen cutoff"}
            record["members"].append(member)
        families.append(record)
    for reference in references.values():
        assert_reference_identity(reference)
    not_computed = [{"family_id": reference["family_id"], "reason": "not_eligible_at_selection"}
                    for reference in references.values() if not reference["eligible"]["idea2"]]
    return {
        "evaluation_row_index": evaluation.astype(np.int64), "evaluation_is_entrant": evaluation_entrant,
        "reference_row_index": reference_rows.astype(np.int64), "mask_R": masks["R"], "mask_R_reference": masks["R_reference"],
        "_rows": int(len(evaluation)), "reference_year": r, "year": t, "families": families, "not_computed": not_computed,
        "family_ids": [family["family_id"] for family in families],
        "evaluation": {"rows": int(len(evaluation)), "returning_rows": int((~evaluation_entrant).sum()),
                       "entrant_rows": int(evaluation_entrant.sum()), "role": "evaluation"},
        "evaluation_rows_identity": _identity_hash({"row_index": evaluation}),
        "reference_rows_identity": _identity_hash({"row_index": reference_rows}),
        "statistics": {"prevalence": "share of rule-subgroup rows with strictly positive activation",
                       "magnitude": "mean activation over all rule-subgroup rows; positive magnitude over positive rows only",
                       "reference": "reference-year evaluation patients inside the same frozen rule"},
        "references": {reference["family_id"]: reference["identity"] for reference in references.values()},
        "scaler_identity": next(iter(dictionaries.values()))["scaler_identity"],
        "sae_identities": {str(seed): dictionaries[seed]["sae_identity"] for seed in seeds},
        "roles_identity": roles["identity"], "population_identity": view["fingerprint"]["population_identity"],
        "predictor_output_row_identity": predictor_output["output_row_identity"],
        "outcomes_read": False}


def run_idea2_frozen_unit(cache, view, roles, config, device, t):
    """Run D(r)/idea2_frozen(t) through the cache; P(r), families, frozen references and dictionaries are loaded from it."""
    if config_hash(config) != cache.config_hash:
        raise ValueError("Configuration differs from the cache configuration")
    if set(view) != set(IDEA2_FROZEN_VIEW_KEYS):
        raise ValueError(f"Idea 2 frozen view reads only {IDEA2_FROZEN_VIEW_KEYS}; outcomes are not accepted")
    r, t = int(roles["reference_year"]), int(t)
    if t not in idea2_follow_up_years(config, r):
        raise ValueError(f"Year {t} is not a follow-up year of reference year {r}")
    return cache.run(f"D/{r}/idea2_frozen/{t}",
                     lambda: idea2_frozen_year_unit(view, r, t, roles, cache.load(f"P/{r}"), cache.load(f"D/{r}/families"),
                                                    load_frozen_references(cache, r), load_reference_dictionaries(cache, r, device),
                                                    config, device),
                     projected_bytes=projected_idea2_frozen_bytes(config))


def idea2_frozen_table(unit):
    """Compact rows per family and member: prevalence and magnitude inside R at year t and at the reference year, medoid retention."""
    def number(value, digits=3):
        return None if value is None else round(value, digits)
    rows = []
    for family in unit["families"]:
        if family["status"] != "ok":
            rows.append({"family": family["family_id"], "t": unit["year"], "status": f"unavailable:{family['reason']}"})
            continue
        for member in family["members"]:
            retention = member.get("retention")
            rows.append({"family": family["family_id"], "t": unit["year"], "member": f"{member['seed']}:{member['factor']}",
                         "medoid": member["is_medoid"], "R rows": member["year"]["rows"],
                         "prevalence": number(member["year"]["positive_prevalence"]),
                         "reference prevalence": number(member["reference"]["positive_prevalence"]),
                         "prevalence ratio": number(member["prevalence_ratio"]),
                         "magnitude": number(member["year"]["magnitude_mean"]),
                         "magnitude ratio": number(member["magnitude_ratio"]),
                         "retention": None if retention is None else number(retention["year_rate"]),
                         "retention ratio": None if retention is None else number(retention["ratio"])})
    return rows


def fit_yearly_dictionary(scaled_fitting, config, device, seed):
    """One yearly SAE on rows that are already under the frozen reference scaler; no scaler is fitted here."""
    import numpy as np
    from sae_compare import train_all_saes
    if config["profile"] not in ("v0", "pilot") or config["epochs"] <= 0:
        raise ValueError("Yearly SAE training belongs to v0/pilot profiles only")
    if type(seed) is not int or seed not in config["sae_seeds"]:
        raise ValueError(f"SAE seed must be one of {config['sae_seeds']}")
    parameters = config["sae"]
    run, = train_all_saes(1, np.asarray(scaled_fitting, dtype=np.float32), seeds=[seed], epochs=config["epochs"],
                          model_type=parameters["model_type"], alpha=parameters["alpha"],
                          scaling_factor=parameters["scaling_factor"], learning_rate=parameters["learning_rate"],
                          weight_decay=parameters["weight_decay"], device=device,
                          encoding_batch_size=config["encoding_batch_size"], show_progress=False)
    return {"run": run, "sae_identity": _sae_state_identity(run["model"]), "seed": seed, "epochs": config["epochs"],
            "training_note": "full-batch SAE training under the frozen reference scaler"}


def projected_idea2_yearly_sae_bytes(config, embedding_bound=512):
    """Upper bound of one D/r/idea2_yearly_sae/seed/t unit's cached bytes: float32 SAE state plus metadata."""
    dimension = int(embedding_bound)
    latents = int(dimension * config["sae"]["scaling_factor"])
    return 4 * (latents * dimension + latents + dimension) + 64 * 1024


def _unavailable_unit(r, t, reason, **fields):
    return {"status": "unavailable", "reason": reason, "_rows": 0, "reference_year": int(r), "year": int(t),
            "outcomes_read": False, **fields}


def idea2_yearly_sae_unit(view, r, t, roles, predictor_output, config, seed, device, reference_dictionary, *, fit=None):
    """D(r)/idea2_yearly_sae(seed, t): one SAE trained on the year-t permanent fitting patients' cached embeddings under
    the frozen reference scaler of S(seed, r); the scaler is never refitted. A year without fitting rows is unavailable."""
    try:
        return _idea2_yearly_sae_unit(view, int(r), int(t), roles, predictor_output, config, int(seed), device,
                                      reference_dictionary, fit_yearly_dictionary if fit is None else fit)
    finally:
        if device == "cuda":
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()


def _idea2_yearly_sae_unit(view, r, t, roles, predictor_output, config, seed, device, reference, fit):
    import numpy as np
    import torch
    if device not in ("cpu", "cuda"):
        raise ValueError("Yearly SAE unit needs a resolved device: cpu or cuda")
    if set(view) != set(IDEA2_YEARLY_VIEW_KEYS):
        raise ValueError(f"Yearly SAE unit reads only {IDEA2_YEARLY_VIEW_KEYS}; outcomes are not accepted")
    if t not in idea2_follow_up_years(config, r):
        raise ValueError(f"Year {t} is not a follow-up year of reference year {r}")
    if seed not in config["sae_seeds"] or reference["seed"] != seed or reference["reference_year"] != r:
        raise ValueError("The reference dictionary must be the S(seed, r) dictionary of the requested seed and year")
    scaler = reference["scaler"]
    if _identity_hash((scaler.mean_, scaler.scale_)) != reference["scaler_identity"]:
        raise RuntimeError("Frozen reference scaler changed")
    fitting, _, _ = _idea2_year_rows(view, roles, t)
    identity = {"reference_year": r, "year": t, "seed": seed, "epochs": int(config["epochs"]),
                "roles_identity": roles["identity"], "population_identity": view["fingerprint"]["population_identity"],
                "predictor_output_row_identity": predictor_output["output_row_identity"],
                "scaler_identity": reference["scaler_identity"], "reference_sae_identity": reference["sae_identity"]}
    if len(fitting) < 2:
        return _unavailable_unit(r, t, IDEA2_UNAVAILABLE_FIT, fitting_rows=int(len(fitting)), **identity)
    embeddings = _year_embeddings(view, r, roles, predictor_output, fitting)
    if not np.isfinite(embeddings).all():
        raise ValueError("Year-t fitting embeddings must be finite")
    scaled = np.asarray(scaler.transform(embeddings), dtype=np.float32)
    started = time.monotonic()
    dictionary = fit(scaled, config, device, seed)
    train_seconds = time.monotonic() - started
    model, parameters = dictionary["run"]["model"], config["sae"]
    if dictionary["seed"] != seed or dictionary["epochs"] != config["epochs"]:
        raise RuntimeError("Trained dictionary does not match the requested seed and epochs")
    state = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    if set(state) != set(SAE_STATE_ARRAYS):
        raise RuntimeError(f"Unexpected SAE state tensors: {sorted(state)}")
    if not all(np.isfinite(array).all() for array in state.values()):
        raise FloatingPointError("Nonfinite SAE parameters after training")
    latents, dimension = int(model.num_latents), int(embeddings.shape[1])
    if (state["encoder.weight"].shape != (latents, dimension) or state["encoder.bias"].shape != (latents,)
            or state["decoder_bias"].shape != (dimension,)):
        raise RuntimeError("SAE state shapes disagree with the trained model")
    if (latents, dimension) != (reference["latent_count"], reference["embedding_dim"]):
        raise RuntimeError("Yearly SAE shape differs from the reference SAE shape")
    if dictionary["sae_identity"] != _sae_state_identity(model):
        raise RuntimeError("SAE identity does not describe the trained state")
    tensor = torch.as_tensor(scaled, device=device)
    model.to(device).eval()
    with torch.inference_mode():
        codes, reconstruction = model(tensor)
        mse = torch.nn.functional.mse_loss(reconstruction, tensor)
        penalty = parameters["alpha"] * codes.abs().mean()
        active = codes > 0
        summary = torch.stack([mse, penalty, mse + penalty, active.float().mean(), (~active.any(dim=0)).sum().float(),
                               torch.isfinite(codes).all().float(), torch.isfinite(reconstruction).all().float()]).cpu().tolist()
    mse_value, penalty_value, loss_value, active_fraction, dead, codes_finite, reconstruction_finite = summary
    if not all(np.isfinite([mse_value, penalty_value, loss_value])) or not (codes_finite and reconstruction_finite):
        raise FloatingPointError("Nonfinite SAE loss or outputs after training")
    peak = int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
    payload = {
        "encoder_weight": state["encoder.weight"], "encoder_bias": state["encoder.bias"], "decoder_bias": state["decoder_bias"],
        "status": "ok", "reason": None, "_rows": int(len(fitting)), **identity,
        "latent_count": latents, "embedding_dim": dimension, "decoder_tied_to_encoder": True,
        "sae": {name: parameters[name] for name in ("model_type", "scaling_factor", "alpha", "learning_rate", "weight_decay")},
        "sae_identity": dictionary["sae_identity"],
        "identity": _identity_hash({**identity, "sae": dictionary["sae_identity"],
                                    "sae_config": {name: parameters[name] for name in sorted(parameters)}}),
        "scaler_source": "frozen reference scaler of S(seed, r); no yearly rescaling",
        "final_loss": loss_value, "final_reconstruction_mse": mse_value, "final_sparsity_penalty": penalty_value,
        "final_loss_basis": "trained state on the training matrix after the last optimizer step",
        "active_fraction": active_fraction, "dead_latents_on_fitting": int(dead),
        "fitting_rows": int(len(fitting)), "fitting_patients": int(len(np.unique(np.asarray(view["patient_id"])[fitting]))),
        "fitting_rows_identity": _identity_hash({"row_index": fitting}),
        "fitting": {"roles": list(FITTING_SIDE_ROLE_NAMES) + ["entrant:fitting"],
                    "returning_rows": int((~roles["is_entrant"][fitting]).sum()),
                    "entrant_rows": int(roles["is_entrant"][fitting].sum())},
        "training": {"device": device, "train_seconds": train_seconds, "batch": "full", "batch_rows": int(len(fitting)),
                     "input_bytes": int(scaled.nbytes), "epochs": int(config["epochs"]), "peak_gpu_bytes": peak,
                     "loss_history": "unavailable: the trainer keeps per-epoch losses internal"},
        "outcomes_read": False}
    del dictionary, model, tensor, codes, reconstruction
    return payload


def run_idea2_yearly_sae_unit(cache, view, roles, config, device, seed, t, *, fit=None):
    """Run D(r)/idea2_yearly_sae(seed, t) through the cache; P(r) and S(seed, r) are loaded from it."""
    if config_hash(config) != cache.config_hash:
        raise ValueError("Configuration differs from the cache configuration")
    if set(view) != set(IDEA2_YEARLY_VIEW_KEYS):
        raise ValueError(f"Yearly SAE unit reads only {IDEA2_YEARLY_VIEW_KEYS}; outcomes are not accepted")
    r, seed, t = int(roles["reference_year"]), int(seed), int(t)
    if t not in idea2_follow_up_years(config, r):
        raise ValueError(f"Year {t} is not a follow-up year of reference year {r}")
    return cache.run(f"D/{r}/idea2_yearly_sae/{seed}/{t}",
                     lambda: idea2_yearly_sae_unit(view, r, t, roles, cache.load(f"P/{r}"), config, seed, device,
                                                   load_reference_sae(cache, r, seed, device), fit=fit),
                     projected_bytes=projected_idea2_yearly_sae_bytes(config))


def load_yearly_sae(cache, r, seed, t, device):
    """Rebuild a cached yearly SAE under the frozen reference scaler of S(seed, r); refuses a changed state or scaler."""
    import numpy as np
    import torch
    from sae import SAE
    if device not in ("cpu", "cuda"):
        raise ValueError("Loading a yearly SAE needs a resolved device: cpu or cuda")
    r, seed, t = int(r), int(seed), int(t)
    unit = cache.load(f"D/{r}/idea2_yearly_sae/{seed}/{t}")
    if unit["status"] != "ok":
        raise RuntimeError(f"Yearly SAE is unavailable: {unit['reason']}")
    reference = load_reference_sae(cache, r, seed, device)
    if (unit["reference_year"], unit["year"], unit["seed"]) != (r, t, seed) or unit["scaler_identity"] != reference["scaler_identity"]:
        raise RuntimeError("Cached yearly SAE belongs to a different year, seed or scaler")
    weight = np.array(unit["encoder_weight"], dtype=np.float32)
    parameters = unit["sae"]
    model = SAE(data_dimension=weight.shape[1], scaling_factor=parameters["scaling_factor"], use_decoder_bias=True,
                type=parameters["model_type"])
    if model.num_latents != weight.shape[0] or weight.shape != (unit["latent_count"], unit["embedding_dim"]):
        raise RuntimeError("Cached yearly SAE shapes disagree with the unit record")
    model.load_state_dict({"encoder.weight": torch.from_numpy(weight),
                           "encoder.bias": torch.from_numpy(np.array(unit["encoder_bias"], dtype=np.float32)),
                           "decoder_bias": torch.from_numpy(np.array(unit["decoder_bias"], dtype=np.float32))})
    if _sae_state_identity(model) != unit["sae_identity"]:
        raise RuntimeError("Cached yearly SAE changed")
    model.to(device).eval()
    return {"scaler": reference["scaler"], "run": {"model": model, "model_type": parameters["model_type"]},
            "scaler_identity": reference["scaler_identity"], "sae_identity": unit["sae_identity"], "seed": seed,
            "epochs": int(unit["epochs"]), "reference_year": r, "year": t, "latent_count": int(weight.shape[0]),
            "embedding_dim": int(weight.shape[1]), "unit_identity": unit["identity"]}


def idea2_pair_encodings(reference, yearly):
    """Reference-family encoding and yearly encoding must cover the same ordered row IDs, seeds and dimensions."""
    import numpy as np
    rows = [np.asarray(encoding["row_index"]) for encoding in (reference, yearly)]
    if rows[0].ndim != 1 or rows[0].shape != rows[1].shape or not np.array_equal(rows[0], rows[1]):
        raise ValueError("Reference and yearly encodings must cover identical ordered row IDs")
    if len(set(rows[0].tolist())) != len(rows[0]) or len(rows[0]) < 2:
        raise ValueError("Encoded rows must be at least two distinct row IDs")
    if list(reference["seeds"]) != list(yearly["seeds"]):
        raise ValueError("Reference and yearly encodings must list the same seeds in the same order")
    matrices = [np.asarray(matrix) for matrix in (*reference["matrices"], *yearly["matrices"])]
    shapes = {matrix.shape for matrix in matrices}
    if len(matrices) != 2 * len(reference["seeds"]) or len(shapes) != 1 or next(iter(shapes))[0] != len(rows[0]):
        raise ValueError("Reference and yearly encodings must have one matrix per seed with identical dimensions")
    return {"rows": int(len(rows[0])), "latents": int(next(iter(shapes))[1]), "row_identity": _identity_hash({"row_index": rows[0]})}


def _family_membership(families, seeds, latents, device):
    """One-hot member indicators per seed (families x latents) and the member counts of every family."""
    import numpy as np
    import torch
    indicator = np.zeros((len(seeds), len(families), latents))
    for index, family in enumerate(families):
        for seed, factor in family["members"]:
            if int(seed) not in seeds or not 0 <= int(factor) < latents:
                raise ValueError("Family members must name a known seed and an existing latent column")
            indicator[seeds.index(int(seed)), index, int(factor)] = 1.0
    tensor = torch.as_tensor(indicator, device=device)
    return tensor, tensor.sum(dim=(0, 2))


def _family_signal(membership, sizes, z):
    """Mean z-scored member signal of each family over the rows (families x rows). The mean pairwise Pearson of two
    families over all member pairs equals the product of their mean signals divided by the row count."""
    import torch
    total = sum(membership[index] @ z[index].T for index in range(len(z)))
    return total / sizes[:, None] if len(sizes) else torch.zeros((0, z[0].shape[0]), dtype=torch.float64, device=z[0].device)


def _mutual_family_links(similarity):
    """Lowest-index (ID-sorted) argmax each way over a finite matrix and the mutual mask; ties within TIE_TOLERANCE."""
    import numpy as np
    if not similarity.size:
        return np.empty(len(similarity), dtype=np.int64), np.empty(similarity.shape[1], dtype=np.int64), np.zeros(len(similarity), dtype=bool)
    forward = (similarity >= similarity.max(axis=1, keepdims=True) - TIE_TOLERANCE).argmax(axis=1)
    backward = (similarity.T >= similarity.T.max(axis=1, keepdims=True) - TIE_TOLERANCE).argmax(axis=1)
    return forward, backward, backward[forward] == np.arange(len(similarity))


def link_reference_to_yearly(reference_matrices, yearly_matrices, seeds, reference_families, yearly_families, *, device,
                             permutations=0, rng_seed=0):
    """Family-level links between the full reference families and the yearly families over identical ordered rows.

    Family similarity is the mean Pearson correlation over every reference-member by yearly-member pair (float64 on the
    device; computed as the product of the families' mean z-scored signals, so one matmul per family pair set). A
    reference family with an inactive (constant) member on these rows has no defined similarity and is unmatched. Mutual
    nearest neighbours are taken among the reference full families and the yearly full families only (ties to the lowest
    family ID, no threshold, no forced assignment); every other reference family reports the state of its most similar
    yearly family: not_mutual_full, partial, chain, inconsistent, or unmatched. The chance check permutes every yearly
    seed's row order independently with a seeded numpy generator, reruns the links and reports observed against permuted.
    """
    import numpy as np
    import torch
    if device not in ("cpu", "cuda"):
        raise ValueError("Family linking needs a resolved device: cpu or cuda")
    if type(permutations) is not int or permutations < 0:
        raise ValueError("permutations must be a nonnegative integer")
    seeds = [int(seed) for seed in seeds]
    matrices = [np.asarray(matrix) for matrix in (*reference_matrices, *yearly_matrices)]
    if (len(reference_matrices) != len(seeds) or len(yearly_matrices) != len(seeds) or len(seeds) < 2
            or any(matrix.ndim != 2 or matrix.dtype.kind != "f" for matrix in matrices)
            or len({matrix.shape for matrix in matrices}) != 1 or matrices[0].shape[0] < 2):
        raise ValueError("Linking needs one floating-point matrix per seed on each side, all over identical ordered rows")
    rows = matrices[0].shape[0]
    for family in (*reference_families, *yearly_families):
        if not family["members"] or any(int(seed) not in seeds or not 0 <= int(factor) < matrices[0].shape[1]
                                        for seed, factor in family["members"]):
            raise ValueError("Family members must name a known seed and an existing latent column")
    reference_full = sorted((family for family in reference_families if family["kind"] == "full"), key=lambda f: f["id"])
    yearly_all = sorted(yearly_families, key=lambda f: f["id"])
    yearly_full = [index for index, family in enumerate(yearly_all) if family["kind"] == "full"]
    tensors = [torch.as_tensor(np.asarray(matrix, dtype=np.float64), device=device) for matrix in matrices]
    if not all(torch.stack([torch.isfinite(tensor).all() for tensor in tensors]).tolist()):
        raise ValueError("Activations must be finite")

    def standardize(tensor):
        constant = tensor.amax(dim=0) == tensor.amin(dim=0)
        centered = tensor - tensor.mean(dim=0, keepdim=True)
        scale = (centered * centered).mean(dim=0, keepdim=True).sqrt()
        return torch.where(constant[None, :], torch.zeros_like(centered), centered / torch.where(constant[None, :], torch.ones_like(scale), scale)), constant

    standardized = [standardize(tensor) for tensor in tensors]
    z_reference, z_yearly = [value for value, _ in standardized[:len(seeds)]], [value for value, _ in standardized[len(seeds):]]
    constant_reference = [flag.cpu().numpy() for _, flag in standardized[:len(seeds)]]
    constant_yearly = [flag.cpu().numpy() for _, flag in standardized[len(seeds):]]
    inactive = {family["id"]: [list(member) for member in family["members"]
                               if constant_reference[seeds.index(int(member[0]))][int(member[1])]] for family in reference_full}
    valid = np.asarray([not inactive[family["id"]] for family in reference_full], dtype=bool)
    yearly_valid = np.asarray([not any(constant_yearly[seeds.index(int(seed))][int(factor)] for seed, factor in family["members"])
                               for family in yearly_all], dtype=bool)
    latents = matrices[0].shape[1]
    reference_membership = _family_membership(reference_full, seeds, latents, device)
    yearly_membership = _family_membership(yearly_all, seeds, latents, device)
    mean_reference = _family_signal(*reference_membership, z_reference)
    mean_yearly = _family_signal(*yearly_membership, z_yearly)
    similarity = (mean_reference @ mean_yearly.T / rows).clamp(-1.0, 1.0).cpu().numpy() if len(yearly_all) and len(reference_full) \
        else np.empty((len(reference_full), len(yearly_all)))
    similarity[~valid, :] = np.nan
    similarity[:, ~yearly_valid] = np.nan

    def links(matrix):
        usable_rows = np.flatnonzero(valid)
        usable_columns = np.asarray([index for index in yearly_full if yearly_valid[index]], dtype=np.int64)
        sub = matrix[np.ix_(usable_rows, usable_columns)]
        forward, backward, mutual = _mutual_family_links(sub)
        return usable_rows, usable_columns, sub, forward, backward, mutual

    usable_rows, usable_columns, sub, forward, backward, mutual = links(similarity)
    position = {int(row): at for at, row in enumerate(usable_rows)}
    records = []
    for index, family in enumerate(reference_full):
        record = {"family_id": family["id"], "members": [list(member) for member in family["members"]],
                  "medoid": list(family["medoid"]), "valid": bool(valid[index]), "retained": False, "state": "unmatched",
                  "reason": None, "matched_family": None, "similarity": None, "runner_up_gap": None, "runner_up_family": None,
                  "best_full_family": None, "best_full_similarity": None, "tie": False, "nearest": None,
                  "inactive_members": inactive[family["id"]]}
        records.append(record)
        if not valid[index]:
            record["reason"] = "reference_member_inactive"
            continue
        row = similarity[index]
        finite = np.flatnonzero(np.isfinite(row))
        if len(finite):
            best = int(finite[(row[finite] >= row[finite].max() - TIE_TOLERANCE).argmax()])
            record["nearest"] = {"family_id": yearly_all[best]["id"], "kind": yearly_all[best]["kind"],
                                 "similarity": float(row[best])}
        if not len(usable_columns):
            record["reason"] = "no_yearly_full_family"
            if record["nearest"]:
                record.update(state=record["nearest"]["kind"], similarity=record["nearest"]["similarity"])
            continue
        at = position[index]
        best_full = int(usable_columns[forward[at]])
        record["best_full_family"], record["best_full_similarity"] = yearly_all[best_full]["id"], float(sub[at, forward[at]])
        record["tie"] = bool((sub[at] >= sub[at].max() - TIE_TOLERANCE).sum() > 1)
        if len(usable_columns) > 1:
            ordered = np.sort(sub[at])[::-1]
            record["runner_up_gap"] = float(ordered[0] - ordered[1])
            second = np.argsort(-sub[at], kind="stable")
            second = [int(c) for c in second if c != forward[at]][0]
            record["runner_up_family"] = yearly_all[int(usable_columns[second])]["id"]
        if mutual[at]:
            record.update(retained=True, state="retained", matched_family=yearly_all[best_full]["id"],
                          similarity=record["best_full_similarity"])
        else:
            record["similarity"] = record["nearest"]["similarity"]
            kind = record["nearest"]["kind"]
            record["state"] = "not_mutual_full" if kind == "full" else kind
    yearly_records = []
    for column_at, column in enumerate(usable_columns):
        row_at = backward[column_at] if len(usable_rows) else None
        linked = [int(usable_rows[at]) for at in range(len(usable_rows)) if mutual[at] and forward[at] == column_at]
        yearly_records.append({"family_id": yearly_all[int(column)]["id"],
                               "best_reference_family": None if row_at is None else reference_full[int(usable_rows[row_at])]["id"],
                               "best_similarity": None if row_at is None else float(sub[row_at, column_at]),
                               "linked_reference_family": reference_full[linked[0]]["id"] if linked else None,
                               "yearly_only": not linked})

    def mean_link_similarity(sub_matrix, mutual_mask, forward_index):
        values = sub_matrix[np.flatnonzero(mutual_mask), forward_index[mutual_mask]]
        return float(values.mean()) if len(values) else None

    chance = None
    if permutations:
        generator = np.random.default_rng(rng_seed)
        order = torch.as_tensor(np.stack([[generator.permutation(rows) for _ in seeds] for _ in range(permutations)]), device=device)
        full_families = [yearly_all[index] for index in yearly_full]
        full_at = torch.as_tensor(np.asarray(yearly_full, dtype=np.int64), device=device)
        full_membership = (yearly_membership[0].index_select(1, full_at), yearly_membership[1].index_select(0, full_at))
        stacked = []
        if len(reference_full) and full_families:
            for index in range(permutations):
                shuffled = [z_yearly[at][order[index, at]] for at in range(len(seeds))]
                signal = _family_signal(*full_membership, shuffled)
                stacked.append((mean_reference @ signal.T / rows).clamp(-1.0, 1.0))
            stacked = torch.stack(stacked).cpu().numpy()
        table = []
        for index in range(permutations):
            if len(reference_full) and full_families:
                matrix = stacked[index].copy()
                matrix[~valid, :] = np.nan
                matrix[:, ~yearly_valid[yearly_full]] = np.nan
                keep_columns = np.flatnonzero(yearly_valid[yearly_full])
                sub_matrix = matrix[np.ix_(np.flatnonzero(valid), keep_columns)]
                f, _, m = _mutual_family_links(sub_matrix)
                table.append({"retained": int(m.sum()), "mean_link_similarity": mean_link_similarity(sub_matrix, m, f)})
            else:
                table.append({"retained": 0, "mean_link_similarity": None})
        defined = [row["mean_link_similarity"] for row in table if row["mean_link_similarity"] is not None]
        observed_similarity = mean_link_similarity(sub, mutual, forward) if len(usable_columns) else None
        chance = {"permutations": permutations, "rng_seed": list(rng_seed) if np.ndim(rng_seed) else int(rng_seed),
                  "scheme": "every yearly seed's row order independently permuted in every permutation; reference encoding fixed",
                  "observed": {"retained": int(mutual.sum()), "mean_link_similarity": observed_similarity},
                  "per_permutation": table,
                  "mean": {"retained": float(np.mean([row["retained"] for row in table])),
                           "mean_link_similarity": float(np.mean(defined)) if defined else None},
                  "max": {"retained": int(max(row["retained"] for row in table)),
                          "mean_link_similarity": float(max(defined)) if defined else None}}
    counts = {state: sum(record["state"] == state for record in records) for state in IDEA2_LINK_STATES}
    return {"seeds": seeds, "rows": int(rows), "device": device, "precision": "float64", "tie_tolerance": TIE_TOLERANCE,
            "reference_ids": [family["id"] for family in reference_full], "yearly_ids": [family["id"] for family in yearly_all],
            "yearly_kinds": [family["kind"] for family in yearly_all], "similarity": similarity, "links": records,
            "yearly_links": yearly_records, "state_counts": counts, "retained": int(counts["retained"]), "chance": chance}


def idea2_reuse_record(family_output, r):
    """The t = r yearly view: the reference SAEs and families are reused as they are, labelled as reuse, no retraining."""
    return {"reference_year": int(r), "year": int(r), "mode": "reference_reuse",
            "label": "t = r: yearly view reuses the reference SAEs and families (no retraining, no linking)",
            "links": [{"family_id": family["id"], "retained": True, "state": "reused", "matched_family": family["id"],
                       "similarity": None, "runner_up_gap": None}
                      for family in family_output["families"] if family["kind"] == "full"]}


def projected_idea2_yearly_families_bytes(config, row_bound=8192, latent_bound=768):
    """Upper bound of one D/r/idea2_yearly_families/t unit's cached bytes: similarity matrix, row index, family and link records."""
    return (8 * latent_bound * 3 * latent_bound + row_bound * 9 + 3 * latent_bound * 4096
            + config["chance_permutations"] * 1024 + 4 * 2**20)


def idea2_yearly_families_unit(view, r, t, roles, predictor_output, family_output, reference_dictionaries, yearly_dictionaries,
                               config, device, *, build=None):
    """D(r)/idea2_yearly_families(t): yearly families built anchor-free from the three yearly SAEs on the eligible held-out
    year-t evaluation rows (embeddings only), the reference families encoded on the same ordered rows, and the
    family-level mutual-nearest-neighbour links with retained yes/no, similarity, runner-up gap and states. A yearly
    dictionary may be a reason string (that seed's yearly SAE was unavailable), which makes the unit unavailable."""
    try:
        return _idea2_yearly_families_unit(view, int(r), int(t), roles, predictor_output, family_output, reference_dictionaries,
                                           yearly_dictionaries, config, device, build_factor_families if build is None else build)
    finally:
        if device == "cuda":
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()


def _idea2_yearly_families_unit(view, r, t, roles, predictor_output, family_output, reference_dictionaries, yearly_dictionaries,
                                config, device, build):
    import numpy as np
    if device not in ("cpu", "cuda"):
        raise ValueError("Yearly family unit needs a resolved device: cpu or cuda")
    if set(view) != set(IDEA2_YEARLY_VIEW_KEYS):
        raise ValueError(f"Yearly family unit reads only {IDEA2_YEARLY_VIEW_KEYS}; outcomes are not accepted")
    if t not in idea2_follow_up_years(config, r):
        raise ValueError(f"Year {t} is not a follow-up year of reference year {r}")
    seeds = list(config["sae_seeds"])
    if list(reference_dictionaries) != seeds or list(yearly_dictionaries) != seeds:
        raise ValueError("Dictionaries must be given for exactly the configured SAE seeds, in order")
    _family_rows(view, r, roles, family_output)
    _, evaluation, entrant = _idea2_year_rows(view, roles, t)
    base = {"reference_year": r, "year": t, "mode": "yearly_retrained", "reuse": False}
    unavailable = [f"{seed}:{value}" for seed, value in yearly_dictionaries.items() if isinstance(value, str)]
    if unavailable:
        return _unavailable_unit(r, t, "yearly_sae_unavailable:" + ",".join(unavailable), **base)
    if len(evaluation) < 2:
        return _unavailable_unit(r, t, IDEA2_UNAVAILABLE_EVALUATION, evaluation_rows=int(len(evaluation)), **base)
    latent_counts = {entry["latent_count"] for entry in (*reference_dictionaries.values(), *yearly_dictionaries.values())}
    if (len(latent_counts) != 1 or len({entry["scaler_identity"] for entry in (*reference_dictionaries.values(),
                                                                              *yearly_dictionaries.values())}) != 1):
        raise RuntimeError("Reference and yearly SAEs must share the frozen reference scaler and latent count")
    embeddings = _year_embeddings(view, r, roles, predictor_output, evaluation)
    batch = config["encoding_batch_size"]
    reference_encoding = {"row_index": evaluation, "seeds": seeds,
                          "matrices": [encode_with_reference_sae(reference_dictionaries[seed], embeddings, device, batch) for seed in seeds]}
    yearly_encoding = {"row_index": evaluation, "seeds": seeds,
                       "matrices": [encode_with_reference_sae(yearly_dictionaries[seed], embeddings, device, batch) for seed in seeds]}
    paired = idea2_pair_encodings(reference_encoding, yearly_encoding)
    split_seed = int(config["split_seed"])
    result = build(yearly_encoding["matrices"], seeds, device=device, id_prefix=f"{r}/{split_seed}/{t}",
                   permutations=config["chance_permutations"], rng_seed=[split_seed, r, t])
    linked = link_reference_to_yearly(reference_encoding["matrices"], yearly_encoding["matrices"], seeds, family_output["families"],
                                      result["families"], device=device, permutations=config["chance_permutations"],
                                      rng_seed=[split_seed, r, t, 1])
    construction = {key: result[key] for key in ("id_prefix", "seeds", "rows", "device", "precision", "tie_tolerance",
                                                 "per_seed", "pairs", "ties", "counts", "chance")}
    per_seed = {entry["seed"]: entry for entry in result["per_seed"]}
    return {
        "evaluation_row_index": evaluation.astype(np.int64), "evaluation_is_entrant": entrant,
        "similarity": linked["similarity"].astype(np.float64), "_rows": int(len(evaluation)), **base,
        "status": "ok", "reason": None, "reference_family_ids": linked["reference_ids"],
        "yearly_family_ids": linked["yearly_ids"], "yearly_family_kinds": linked["yearly_kinds"],
        "links": linked["links"], "yearly_links": linked["yearly_links"], "state_counts": linked["state_counts"],
        "retained": linked["retained"], "link_chance": linked["chance"], "link_precision": linked["precision"],
        "similarity_definition": "mean Pearson over all reference-member by yearly-member pairs on the shared evaluation rows",
        "construction": construction, "families": result["families"], "singletons": result["singletons"],
        "dead_latents": {str(seed): len(per_seed[seed]["dead"]) for seed in seeds},
        "undefined_latents": {str(seed): len(per_seed[seed]["undefined_correlation"]) for seed in seeds},
        "latent_count": paired["latents"], "row_pair_identity": paired["row_identity"],
        "evaluation": {"rows": int(len(evaluation)), "returning_rows": int((~entrant).sum()),
                       "entrant_rows": int(entrant.sum()), "patients": int(len(np.unique(np.asarray(view["patient_id"])[evaluation]))),
                       "role": "evaluation"},
        "evaluation_rows_identity": _identity_hash({"row_index": evaluation}),
        "scaler_identity": next(iter(reference_dictionaries.values()))["scaler_identity"],
        "reference_sae_identities": {str(seed): reference_dictionaries[seed]["sae_identity"] for seed in seeds},
        "yearly_sae_identities": {str(seed): yearly_dictionaries[seed]["sae_identity"] for seed in seeds},
        "roles_identity": roles["identity"], "population_identity": view["fingerprint"]["population_identity"],
        "predictor_output_row_identity": predictor_output["output_row_identity"], "outcomes_read": False}


def run_idea2_yearly_families_unit(cache, view, roles, config, device, t, *, build=None):
    """Run D(r)/idea2_yearly_families(t) through the cache; the reference families, dictionaries and the yearly SAE units
    of every seed are loaded from it, and an unavailable yearly SAE makes this unit unavailable with its reason."""
    if config_hash(config) != cache.config_hash:
        raise ValueError("Configuration differs from the cache configuration")
    if set(view) != set(IDEA2_YEARLY_VIEW_KEYS):
        raise ValueError(f"Yearly family unit reads only {IDEA2_YEARLY_VIEW_KEYS}; outcomes are not accepted")
    r, t = int(roles["reference_year"]), int(t)
    if t not in idea2_follow_up_years(config, r):
        raise ValueError(f"Year {t} is not a follow-up year of reference year {r}")

    def produce():
        yearly = {}
        for seed in config["sae_seeds"]:
            stored = cache.load(f"D/{r}/idea2_yearly_sae/{seed}/{t}")
            yearly[seed] = load_yearly_sae(cache, r, seed, t, device) if stored["status"] == "ok" else str(stored["reason"])
        return idea2_yearly_families_unit(view, r, t, roles, cache.load(f"P/{r}"), cache.load(f"D/{r}/families"),
                                          load_reference_dictionaries(cache, r, device), yearly, config, device, build=build)

    return cache.run(f"D/{r}/idea2_yearly_families/{t}", produce, projected_bytes=projected_idea2_yearly_families_bytes(config))


def idea2_link_table(unit):
    """Compact rows per reference full family: retained yes/no, matched yearly family, similarity, runner-up gap, state."""
    def number(value, digits=3):
        return None if value is None else round(value, digits)
    if unit["status"] != "ok":
        return [{"t": unit["year"], "status": f"unavailable:{unit['reason']}"}]
    return [{"family": link["family_id"], "t": unit["year"], "retained": link["retained"], "state": link["state"],
             "similarity": number(link["similarity"]), "runner-up gap": number(link["runner_up_gap"]),
             "matched yearly family": link["matched_family"] or "-"} for link in unit["links"]]


GROUP_NAMES = ("R_and_A", "R_and_not_A", "not_R_and_A", "not_R_and_not_A")
GROUPS_VIEW_KEYS = PERFORMANCE_VIEW_KEYS
GROUPS_EXAMPLE_LIMIT = 4
GROUPS_CONTROL_DRAWS = 5
GROUPS_ACTIVATION_EDGE = 1e-4
GROUPS_ACTIVATION_AGREEMENT = {"rtol": 1e-4, "atol": 1e-4}
GROUPS_RESIDUAL_TOLERANCE = 1e-9
GROUPS_UNAVAILABLE_EMPTY = "empty_evaluation_population"


def groups_follow_up_years(config, r):
    """Follow-up years t > r that the unit graph assigns D(r)/groups_interventions(t) units to; none for a T0-only reference year."""
    prefix = f"D/{int(r)}/groups_interventions/"
    return [int(key[len(prefix):]) for key in unit_graph(config) if key.startswith(prefix)]


def partition_groups(a_mask, r_mask):
    """Boolean membership (..., 4, rows) of R and A in GROUP_NAMES order (R and A, R and not A, not R and A, neither);
    every row belongs to exactly one group, asserted."""
    import torch
    a, r = torch.as_tensor(a_mask), torch.as_tensor(r_mask)
    if a.dtype != torch.bool or r.dtype != torch.bool:
        raise ValueError("Subgroup masks must be boolean")
    if a.shape != r.shape or a.ndim not in (1, 2):
        raise ValueError("A and R masks must share one ordered row axis (rows) or (families, rows)")
    membership = torch.stack([r & a, r & ~a, ~r & a, ~r & ~a], dim=-2)
    if not bool((membership.sum(dim=-2) == 1).all()):
        raise RuntimeError("The four groups must partition the rows")
    return membership


def group_feature_summary(membership, features):
    """Rows, per-feature mean and population standard deviation (denominator = group rows) of each group (families, 4, ...)
    in float64 by one batched product on the tensors' device; empty groups give NaN."""
    import torch
    if membership.ndim != 3 or features.ndim != 2 or membership.shape[-1] != features.shape[0]:
        raise ValueError("Membership (families, 4, rows) and features (rows, columns) must share the row axis")
    flags = membership.to(torch.float64).reshape(-1, membership.shape[-1])
    values = features.to(torch.float64)
    counts = flags.sum(dim=-1)
    first, second = flags @ values, flags @ (values * values)
    divisor = counts[:, None]
    mean = first / divisor
    deviation = torch.sqrt(torch.clamp(second / divisor - mean * mean, min=0.))
    empty = (counts == 0)[:, None]
    shape = (*membership.shape[:2], values.shape[1])
    return {"rows": counts.reshape(membership.shape[:2]), "mean": torch.where(empty, torch.nan, mean).reshape(shape),
            "std": torch.where(empty, torch.nan, deviation).reshape(shape)}


def group_coactivation(membership, activations):
    """For every group of every family and every factor h: rows with strictly positive activation and the activation sum
    over the group's rows (families, 4, factors). Prevalence = positive rows / group rows; magnitude = sum / group rows;
    positive magnitude = sum / positive rows (activations are nonnegative)."""
    import torch
    values = torch.as_tensor(activations).to(torch.float64)
    if membership.ndim != 3 or values.ndim != 2 or values.shape[1] != membership.shape[-1]:
        raise ValueError("Membership (families, 4, rows) and activations (factors, rows) must share the row axis")
    if bool((values < 0).any()):
        raise ValueError("Coactivation needs nonnegative activations")
    flags = membership.to(torch.float64)
    return {"positive_rows": torch.einsum("fgn,hn->fgh", flags, (values > 0).to(torch.float64)),
            "activation_sum": torch.einsum("fgn,hn->fgh", flags, values)}


def select_group_examples(membership, activations, limit=GROUPS_EXAMPLE_LIMIT):
    """Deterministic examples per family and group from the family's own medoid activations only: rows of the group with
    strictly positive activation, highest activation first, ties by lowest row position (ascending population order).
    Returns positions (families, 4, k) and validity (families, 4, k); labels and probabilities are not inputs."""
    import torch
    values = torch.as_tensor(activations).to(torch.float64)
    if membership.ndim != 3 or values.shape != (membership.shape[0], membership.shape[2]):
        raise ValueError("Activations (families, rows) must align with membership (families, 4, rows)")
    if type(limit) is not int or limit < 1:
        raise ValueError("The example limit must be a positive integer")
    own = values[:, None, :]
    score = torch.where(membership & (own > 0), own, torch.full_like(own, -1.))
    order = torch.sort(score, dim=-1, descending=True, stable=True).indices[..., :min(limit, membership.shape[-1])]
    return {"positions": order, "valid": torch.gather(score, -1, order) > 0}


def _dictionary_tensors(dictionary, device):
    import torch
    model = dictionary["run"]["model"]
    if dictionary["run"]["model_type"] != "ReLU" or getattr(model, "type", None) != "ReLU":
        raise ValueError("Decoder-contribution removal supports the tied ReLU SAE only")
    scaler = dictionary["scaler"]
    weight = model.encoder.weight.detach().to(device=device, dtype=torch.float64)
    return {"weight": weight, "encoder_bias": model.encoder.bias.detach().to(device=device, dtype=torch.float64),
            "decoder_bias": model.decoder_bias.detach().to(device=device, dtype=torch.float64),
            "mean": torch.as_tensor(scaler.mean_, dtype=torch.float64, device=device),
            "scale": torch.as_tensor(scaler.scale_, dtype=torch.float64, device=device), "model": model}


def remove_factor_contribution(dictionary, embeddings, factors, activations):
    """Residual-preserving removal of one SAE factor's decoder contribution from raw embeddings, batched on their device.

    With scaler u = (h - mean) / scale, the tied ReLU decoder x_hat = codes @ W + b and reconstruction error e = u - x_hat,
    the contribution of factor k is c_k = a_k W[k]; the removal in raw coordinates is scale * c_k and the intervened
    embedding is h - scale * c_k. The reconstruction error and decoder bias are never replaced, so zero activation returns
    the input unchanged. The same result is rebuilt from the explicit decomposition scale * (x_hat - c_k + e) + mean and
    both the activation and the decoder orientation are verified against the SAE itself."""
    import torch
    values = torch.as_tensor(embeddings)
    device = values.device
    parts = _dictionary_tensors(dictionary, device)
    weight = parts["weight"]
    h = values.to(torch.float64)
    index = torch.as_tensor(factors, dtype=torch.long, device=device)
    a = torch.as_tensor(activations, dtype=torch.float64, device=device)
    if h.ndim != 2 or h.shape[1] != weight.shape[1] or index.shape != (len(h),) or a.shape != (len(h),):
        raise ValueError("Embeddings (rows, dimension) need one factor and one activation per row")
    if len(h) and (int(index.min()) < 0 or int(index.max()) >= weight.shape[0]):
        raise ValueError("Factor index outside the SAE latents")
    if not bool(torch.isfinite(h).all()) or not bool(torch.isfinite(a).all()) or bool((a < 0).any()):
        raise ValueError("Embeddings must be finite and activations finite and nonnegative")
    with torch.no_grad():
        scaled = (h - parts["mean"]) / parts["scale"]
        codes = torch.relu(scaled @ weight.T + parts["encoder_bias"])
        reconstruction = codes @ weight + parts["decoder_bias"]
        residual = scaled - reconstruction
        contribution = a[:, None] * weight[index]
        removal = parts["scale"] * contribution
        removed = h - removal
        explicit = parts["scale"] * (reconstruction - contribution + residual) + parts["mean"]
        decoded = parts["model"].decode(codes.to(parts["model"].encoder.weight.dtype)).to(torch.float64)
        own = codes[torch.arange(len(h), device=device), index]
    return {"embeddings": removed, "removal": removal, "residual": parts["scale"] * residual,
            "residual_form_gap": (explicit - removed).abs().amax(dim=-1) if len(h) else removed.new_zeros(0),
            "activation_gap": (own - a).abs(), "decoder_gap": (decoded - reconstruction).abs().amax(dim=-1) if len(h) else removed.new_zeros(0)}


def reconstruction_mse(dictionary, embeddings):
    """Per-row mean squared reconstruction error of the frozen SAE in the scaled coordinates, float64 on the embeddings' device."""
    import torch
    values = torch.as_tensor(embeddings)
    parts = _dictionary_tensors(dictionary, values.device)
    with torch.no_grad():
        scaled = (values.to(torch.float64) - parts["mean"]) / parts["scale"]
        codes = torch.relu(scaled @ parts["weight"].T + parts["encoder_bias"])
        return ((scaled - (codes @ parts["weight"] + parts["decoder_bias"])) ** 2).mean(dim=-1)


def control_directions(seed, examples, draws, dimension):
    """Reproducible random directions (examples, draws, dimension) drawn once on the host, independent of the run device."""
    import numpy as np
    return np.random.default_rng(int(seed)).standard_normal((int(examples), int(draws), int(dimension)))


def norm_matched_controls(removal, directions):
    """Control removal vectors (examples, draws, dimension): each draw points along its random direction and has exactly
    the norm of the example's decoder-contribution removal."""
    import torch
    unit = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    return torch.linalg.vector_norm(removal, dim=-1)[:, None, None] * unit


def _single_host_transfer(tensors):
    """One device-to-host copy of several tensors (float64 concatenation split back by shape)."""
    import torch
    names = list(tensors)
    if not names:
        return {}
    flat = torch.cat([tensors[name].detach().to(torch.float64).reshape(-1) for name in names]).cpu().numpy()
    result, start = {}, 0
    for name in names:
        size = int(tensors[name].numel())
        result[name] = flat[start:start + size].reshape(tuple(tensors[name].shape))
        start += size
    return result


def _death_probabilities(continuation, values, batch_rows):
    """Death-class probabilities of raw embeddings through the serialized head, in batches, kept on the continuation's device."""
    import torch
    with torch.no_grad():
        chunks = [continuation.probabilities(values[start:start + batch_rows])[:, 1] for start in range(0, len(values), batch_rows)]
    return torch.cat(chunks) if chunks else values.new_zeros(0)


def projected_groups_interventions_bytes(config, latent_bound=768, row_bound=8192, feature_bound=1024):
    """Upper bound of one D/r/groups_interventions/t unit's cached bytes: per family three row arrays, group feature means and
    standard deviations, coactivation counts and sums, per-example intervention arrays and the metric records."""
    cap = candidate_cap(latent_bound, config["candidate_cap_fraction"])
    groups, examples = len(GROUP_NAMES), cap * len(GROUP_NAMES) * GROUPS_EXAMPLE_LIMIT
    arrays = cap * (3 * row_bound + groups * feature_bound * 2 * 4 + groups * cap * 16 + groups * 8)
    per_example = 8 * (14 + 2 * GROUPS_CONTROL_DRAWS + len(config["cav_seeds"]))
    return arrays + row_bound * 9 + examples * per_example + cap * (groups * 24 * 1024 + 64 * 1024) + 4 * 2**20


def _check_dependency(payload, name, expected, evaluation, r, t, roles, view):
    import numpy as np
    if payload["reference_year"] != r or payload["year"] != t:
        raise ValueError(f"The {name} payload belongs to a different reference year or follow-up year")
    if payload["roles_identity"] != roles["identity"] or payload["population_identity"] != view["fingerprint"]["population_identity"]:
        raise ValueError(f"The {name} payload was produced for different roles or a different population")
    if list(payload["family_ids"]) != list(expected):
        raise ValueError(f"The {name} payload does not follow the families eligible for it")
    if (not np.array_equal(np.asarray(payload["evaluation_row_index"]), evaluation)
            or payload["evaluation_rows_identity"] != _identity_hash({"row_index": evaluation})):
        raise ValueError(f"The {name} payload covers different evaluation rows")


def groups_interventions_unit(view, r, t, roles, predictor_output, family_output, references, dictionaries, continuation,
                              idea1, idea2_frozen, config, device):
    """D(r)/groups_interventions(t): for every family eligible for either idea, the partition of the held-out year-t evaluation
    rows by the frozen rule R and the frozen activation subgroup A_t (R and A, R and not A, not R and A, neither) with counts,
    performance against baselines fixed at the reference, same-year R/A overlap, original-feature distributions, coactivation of
    every candidate medoid, deterministic reference-defined examples (at most four per family and group, chosen from medoid
    activations only) and, for those examples, verified residual-preserving removal of the medoid's decoder contribution with
    norm-matched controls, zero-removal parity and probabilities/Brier losses through the cached continuation. A and R are
    recomputed from the frozen objects and checked against the Idea 1 and Idea 2 payloads. No causal claim."""
    import numpy as np
    import torch
    r, t = int(r), int(t)
    if device not in ("cpu", "cuda"):
        raise ValueError("Groups need a resolved device: cpu or cuda")
    if set(view) != set(GROUPS_VIEW_KEYS):
        raise ValueError(f"Groups read only {GROUPS_VIEW_KEYS}")
    if t not in groups_follow_up_years(config, r):
        raise ValueError(f"Year {t} is not a follow-up year of reference year {r}")
    _require_selection_settings(config)
    if config["classification_threshold"] != 0.5 or config["confidence_level"] != 0.95:
        raise ValueError("Groups support the 0.5 threshold and 95% intervals only")
    seeds = list(config["sae_seeds"])
    if list(dictionaries) != seeds:
        raise ValueError("Dictionaries must be given for exactly the configured SAE seeds, in order")
    if len({entry["scaler_identity"] for entry in dictionaries.values()}) != 1:
        raise ValueError("Reference dictionaries must share one frozen scaler")
    if next(continuation.head.parameters()).device.type != device:
        raise ValueError("The continuation must live on the run device")
    for reference in references.values():
        assert_reference_identity(reference)
    family_rows = _family_rows(view, r, roles, family_output)
    names = list(view["feature_names"])
    entries = [reference for reference in references.values() if any(reference["eligible"].values())]
    not_computed = [{"family_id": reference["family_id"], "reason": "not_eligible_at_selection"}
                    for reference in references.values() if not any(reference["eligible"].values())]
    reference_fitting, _ = _predictor_positions(view, r, roles, predictor_output, FITTING_SIDE_ROLE_NAMES)
    _, evaluation, evaluation_entrant = _idea2_year_rows(view, roles, t)
    count = len(evaluation)
    _check_dependency(idea1, "idea1", [reference["family_id"] for reference in references.values() if reference["eligible"]["idea1"]],
                      evaluation, r, t, roles, view)
    _check_dependency(idea2_frozen, "idea2_frozen",
                      [reference["family_id"] for reference in references.values() if reference["eligible"]["idea2"]],
                      evaluation, r, t, roles, view)
    frozen_family = {family["family_id"]: family for family in idea2_frozen["families"]}
    identity = {"reference_year": r, "year": t, "roles_identity": roles["identity"],
                "population_identity": view["fingerprint"]["population_identity"],
                "predictor_output_row_identity": predictor_output["output_row_identity"],
                "continuation_identity": continuation.identity, "scaler_identity": next(iter(dictionaries.values()))["scaler_identity"],
                "references": {reference["family_id"]: reference["identity"] for reference in references.values()},
                "evaluation_rows_identity": _identity_hash({"row_index": evaluation}),
                "reference_fitting_rows_identity": _identity_hash({"row_index": reference_fitting})}
    base = {"reference_year": r, "year": t, "not_computed": not_computed, **identity}

    def record(reference, status, reason, **fields):
        return {"family_id": reference["family_id"], "medoid": [int(value) for value in reference["medoid"]],
                "activation_target": reference["activation_target"], "cutoff": reference["cutoff"],
                "reference_identity": reference["identity"], "eligible": dict(reference["eligible"]),
                "status": status, "reason": reason, "array_index": None, **fields}

    if not count:
        return {**base, "status": "unavailable", "reason": GROUPS_UNAVAILABLE_EMPTY, "_rows": 0,
                "evaluation_row_index": evaluation.astype(np.int64), "evaluation_is_entrant": evaluation_entrant,
                "families": [record(reference, "unavailable", GROUPS_UNAVAILABLE_EMPTY) for reference in entries],
                "family_ids": [], "outcomes_read": False}
    families, usable = [], []
    for reference in entries:
        frozen = frozen_family.get(reference["family_id"])
        if reference["eligible"]["idea2"] and (frozen is None or frozen["status"] != "ok"):
            reason = "missing" if frozen is None else frozen["reason"]
            families.append(record(reference, "unavailable", f"idea2_frozen_unavailable:{reason}"))
        else:
            usable.append(reference)
            families.append(None)
    split = len(reference_fitting)
    rows = np.concatenate([reference_fitting, evaluation])
    embeddings = _year_embeddings(view, r, roles, predictor_output, rows)
    cached_index = np.asarray(predictor_output["row_index"])
    locate = np.searchsorted(cached_index, evaluation)
    labels, patients = np.asarray(view["y"]), np.asarray(view["patient_id"])
    reference_labels = labels[reference_fitting]
    reference_features = np.asarray(view["X"])[reference_fitting].astype(np.float64)
    features = np.asarray(view["X"])[evaluation].astype(np.float64)
    if not np.isfinite(features).all() or not np.isfinite(reference_features).all():
        raise ValueError("Features of the reference fitting and year-t evaluation rows must be finite")
    dimension, total = int(embeddings.shape[1]), len(usable)
    activations = _medoid_activations(rows, embeddings, usable, family_rows, dictionaries, config, device)
    a_mask = np.zeros((total, count), dtype=bool)
    r_mask = np.zeros((total, count), dtype=bool)
    a_reference = np.zeros((total, split), dtype=bool)
    r_reference = np.zeros((total, split), dtype=bool)
    for position, reference in enumerate(usable):
        a_mask[position] = strict_activation_mask(activations[position, split:], reference["target"])
        r_mask[position] = rule_mask(reference["rule"], features, names)
        a_reference[position] = strict_activation_mask(activations[position, :split], reference["target"])
        r_reference[position] = rule_mask(reference["rule"], reference_features, names)
    where = {reference["family_id"]: position for position, reference in enumerate(usable)}
    consistency = {reference["family_id"]: {"idea1": None, "idea2": None} for reference in usable}
    for position, family_id in enumerate(idea1["family_ids"]):
        if family_id not in where:
            continue
        mine = where[family_id]
        theirs_a, theirs_r = np.asarray(idea1["mask_A"])[position], np.asarray(idea1["mask_R"])[position]
        edge = np.abs(activations[mine, split:] - usable[mine]["cutoff"]) <= GROUPS_ACTIVATION_EDGE * (1 + abs(usable[mine]["cutoff"]))
        if not np.array_equal(theirs_r, r_mask[mine]) or bool(((theirs_a != a_mask[mine]) & ~edge).any()):
            raise RuntimeError(f"Idea 1 membership of {family_id} differs from the frozen definitions on the same rows")
        consistency[family_id]["idea1"] = {"A_rows_checked": count, "A_edge_rows_differing": int((theirs_a != a_mask[mine]).sum()),
                                           "R_rows_differing": 0}
    for family in idea2_frozen["families"]:
        if family["status"] == "ok":
            mine = where[family["family_id"]]
            theirs = np.asarray(idea2_frozen["mask_R"])[idea2_frozen["family_ids"].index(family["family_id"])]
            if not np.array_equal(theirs, r_mask[mine]):
                raise RuntimeError(f"Idea 2 rule membership of {family['family_id']} differs from the frozen rule on the same rows")
            consistency[family["family_id"]]["idea2"] = {"R_rows_checked": count, "R_rows_differing": 0}
    reference_membership = partition_groups(a_reference, r_reference).numpy()
    split_seed, replicates = int(config["split_seed"]), int(config["bootstrap_replicates"])
    cav_seeds = [int(seed) for seed in config["cav_seeds"]]
    cav_all = np.full((total, len(cav_seeds), dimension), np.nan)
    for position, reference in enumerate(usable):
        for column, seed in enumerate(cav_seeds):
            value = reference["cavs"][seed]["cav"]
            if value is not None:
                cav_all[position, column] = np.asarray(value, dtype=np.float64)
    medoid_seed = np.asarray([int(reference["medoid"][0]) for reference in usable], dtype=np.int64)
    medoid_factor = np.asarray([int(reference["medoid"][1]) for reference in usable], dtype=np.int64)
    gradients = np.asarray(predictor_output["gradients"])[locate].astype(np.float64)
    probabilities = np.asarray(predictor_output["probabilities"])[locate].astype(np.float64)
    evaluation_labels = labels[evaluation]
    draws = GROUPS_CONTROL_DRAWS
    control_seed = _derived_seed("groups_controls", split_seed, r, t)
    if total:
        place = torch.device(device)
        membership = partition_groups(torch.as_tensor(a_mask, device=place), torch.as_tensor(r_mask, device=place))
        own = torch.as_tensor(activations[:, split:], device=place)
        summary = group_feature_summary(membership, torch.as_tensor(features, device=place))
        coactive = group_coactivation(membership, own)
        emb_eval = torch.as_tensor(np.ascontiguousarray(embeddings[split:]), dtype=torch.float32, device=place)
        error = torch.empty((total, count), dtype=torch.float64, device=place)
        for seed in sorted({int(value) for value in medoid_seed}):
            chosen = np.flatnonzero(medoid_seed == seed)
            error[torch.as_tensor(chosen, device=place)] = reconstruction_mse(dictionaries[seed], emb_eval)
        flags = membership.to(torch.float64)
        reconstruction = (flags * error[:, None, :]).sum(dim=-1) / flags.sum(dim=-1)
        reconstruction = torch.where(flags.sum(dim=-1) == 0, torch.nan, reconstruction)
        picked = select_group_examples(membership, own)
        family_e, group_e, rank_e = picked["valid"].nonzero(as_tuple=True)
        position_e = picked["positions"][family_e, group_e, rank_e]
        examples = int(family_e.numel())
        transfer = {"group_code": (membership.to(torch.float64) * torch.arange(4, device=place, dtype=torch.float64)[None, :, None]).sum(dim=1),
                    "group_rows": summary["rows"], "feature_mean": summary["mean"], "feature_std": summary["std"],
                    "positive_rows": coactive["positive_rows"], "activation_sum": coactive["activation_sum"],
                    "reconstruction_mse": reconstruction, "family_e": family_e, "group_e": group_e, "rank_e": rank_e,
                    "position_e": position_e}
        if examples:
            h = emb_eval[position_e]
            activation_e = own[family_e, position_e]
            seed_e, factor_e = (torch.as_tensor(medoid_seed, device=place)[family_e],
                                torch.as_tensor(medoid_factor, device=place)[family_e])
            removed, removal, residual = (torch.empty((examples, dimension), dtype=torch.float64, device=place) for _ in range(3))
            zero = torch.empty_like(removed)
            gaps = {name: torch.empty(examples, dtype=torch.float64, device=place) for name in ("residual_form_gap", "activation_gap", "decoder_gap")}
            for seed in sorted({int(value) for value in medoid_seed}):
                chosen = (seed_e == seed).nonzero(as_tuple=True)[0]
                if not len(chosen):
                    continue
                result = remove_factor_contribution(dictionaries[seed], h[chosen], factor_e[chosen], activation_e[chosen])
                nothing = remove_factor_contribution(dictionaries[seed], h[chosen], factor_e[chosen], torch.zeros_like(activation_e[chosen]))
                removed[chosen], removal[chosen], residual[chosen], zero[chosen] = (
                    result["embeddings"], result["removal"], result["residual"], nothing["embeddings"])
                for name in gaps:
                    gaps[name][chosen] = result[name]
            directions = torch.as_tensor(control_directions(control_seed, examples, draws, dimension), device=place)
            controls = norm_matched_controls(removal, directions)
            h64 = h.to(torch.float64)
            batch = torch.cat([h, removed.to(torch.float32), zero.to(torch.float32),
                               (h64[:, None, :] - controls).to(torch.float32).reshape(-1, dimension)])
            outcome = _death_probabilities(continuation, batch, int(config["inference_batch_rows"]))
            cav = torch.as_tensor(cav_all, device=place)[family_e]
            transfer.update(
                activation_e=activation_e, removal_norm=torch.linalg.vector_norm(removal, dim=-1),
                residual_norm=torch.linalg.vector_norm(residual, dim=-1),
                control_norm_gap=(torch.linalg.vector_norm(controls, dim=-1) - torch.linalg.vector_norm(removal, dim=-1)[:, None]).abs().amax(),
                zero_embedding_gap=(zero.to(torch.float32).to(torch.float64) - h64).abs().amax(),
                probability=outcome, cav_projection=torch.einsum("ed,ecd->ec", removal, cav),
                first_order_change=-(torch.as_tensor(gradients, device=place)[position_e] * removal).sum(dim=-1),
                **gaps)
        host = _single_host_transfer(transfer)
    else:
        host = {"group_code": np.zeros((0, count)), "group_rows": np.zeros((0, 4)), "feature_mean": np.zeros((0, 4, features.shape[1])),
                "feature_std": np.zeros((0, 4, features.shape[1])), "positive_rows": np.zeros((0, 4, 0)), "activation_sum": np.zeros((0, 4, 0)),
                "reconstruction_mse": np.zeros((0, 4)), "family_e": np.zeros(0), "group_e": np.zeros(0),
                "rank_e": np.zeros(0), "position_e": np.zeros(0)}
        examples = 0
    code = host["group_code"].astype(np.uint8)
    member = code[:, None, :] == np.arange(4)[None, :, None]
    family_e, group_e, rank_e, position_e = (host[name].astype(np.int64) for name in ("family_e", "group_e", "rank_e", "position_e"))
    example_arrays = {"example_family": family_e, "example_group": group_e, "example_rank": rank_e,
                      "example_row_index": evaluation[position_e].astype(np.int64)}
    if examples:
        outcome = host["probability"]
        cached = probabilities[position_e, 1]
        before, after, zeroed = outcome[:examples], outcome[examples:2 * examples], outcome[2 * examples:3 * examples]
        control = outcome[3 * examples:].reshape(examples, draws)
        label = evaluation_labels[position_e].astype(np.float64)
        zero_gap = float(np.max(np.abs(zeroed - cached)))
        gates = {"zero_removal_embedding_max_abs": float(host["zero_embedding_gap"]),
                 "zero_removal_probability_max_abs": zero_gap,
                 "continuation_replay_max_abs": float(np.max(np.abs(before - cached))),
                 "residual_form_max_abs": float(host["residual_form_gap"].max()),
                 "activation_max_abs_gap": float(host["activation_gap"].max()),
                 "decoder_max_abs_gap": float(host["decoder_gap"].max()),
                 "control_norm_max_abs_gap": float(host["control_norm_gap"])}
        if gates["zero_removal_embedding_max_abs"] != 0.:
            raise RuntimeError("Zero removal changed the embeddings")
        if gates["residual_form_max_abs"] > GROUPS_RESIDUAL_TOLERANCE * max(1., float(np.abs(embeddings[split:]).max())):
            raise RuntimeError("Residual-preserving removal differs from the explicit decomposition")
        try:
            np.testing.assert_allclose(zeroed, cached, **CONTINUATION_PARITY)
            np.testing.assert_allclose(before, cached, **CONTINUATION_PARITY)
        except AssertionError as error:
            raise RuntimeError(f"Un-intervened embeddings do not reproduce the cached predictions: {_first_line(error)}") from error
        if float(np.max(host["activation_gap"] - GROUPS_ACTIVATION_AGREEMENT["atol"] - GROUPS_ACTIVATION_AGREEMENT["rtol"] * host["activation_e"])) > 0:
            raise RuntimeError("Cached medoid activation differs from the SAE's own encoding")
        if gates["decoder_max_abs_gap"] > 1e-4 * max(1., float(np.abs(embeddings[split:]).max())):
            raise RuntimeError("The manual decoder differs from the SAE decoder")
        if gates["control_norm_max_abs_gap"] > 1e-9 * max(1., float(host["removal_norm"].max())):
            raise RuntimeError("Controls are not norm matched")
        example_arrays.update(
            example_activation=host["activation_e"], example_label=label.astype(np.int8),
            removal_norm=host["removal_norm"], residual_norm=host["residual_norm"], probability_cached=cached,
            probability_before=before, probability_removed=after, probability_zero_removal=zeroed, probability_control=control,
            loss_before=(before - label) ** 2, loss_removed=(after - label) ** 2, loss_control=(control - label[:, None]) ** 2,
            cav_projection=host["cav_projection"], first_order_change=host["first_order_change"])
    else:
        gates = {name: None for name in ("zero_removal_embedding_max_abs", "zero_removal_probability_max_abs", "continuation_replay_max_abs",
                                         "residual_form_max_abs", "activation_max_abs_gap", "decoder_max_abs_gap", "control_norm_max_abs_gap")}
        for name in ("example_activation", "removal_norm", "residual_norm", "probability_cached", "probability_before",
                     "probability_removed", "probability_zero_removal", "first_order_change", "loss_before", "loss_removed"):
            example_arrays[name] = np.zeros(0)
        example_arrays.update(example_label=np.zeros(0, dtype=np.int8), probability_control=np.zeros((0, draws)),
                              loss_control=np.zeros((0, draws)), cav_projection=np.zeros((0, len(cav_seeds))))
    group_rows = host["group_rows"].astype(np.int64)
    evaluation_patients = patients[evaluation]
    p_death = probabilities[:, 1]
    for position, reference in enumerate(usable):
        entry = record(reference, "ok", None)
        entry["array_index"] = position
        agreement = rule_activation_agreement(r_mask[position], a_mask[position])
        groups = {}
        for index, name in enumerate(GROUP_NAMES):
            flag, fitted = member[position, index], reference_membership[position, index]
            baseline = fit_constant_baseline(reference_labels[fitted])
            seed = _derived_seed("groups_reliability", split_seed, r, t, reference["family_id"], name)
            groups[name] = {"rows": int(flag.sum()), "patients": int(len(np.unique(evaluation_patients[flag]))),
                            "deaths": int(evaluation_labels[flag].sum()), "returning_rows": int((flag & ~evaluation_entrant).sum()),
                            "entrant_rows": int((flag & evaluation_entrant).sum()), "reference_fitting_rows": int(fitted.sum()),
                            "baseline": baseline, "bootstrap_seed": seed,
                            "reliability": subgroup_reliability(evaluation_labels[flag], p_death[flag], evaluation_patients[flag],
                                                                baseline, replicates, seed),
                            "positive_example_rows": int(host["positive_rows"][position, index, position]),
                            "examples": [int(value) for value in np.flatnonzero((family_e == position) & (group_e == index))]}
            if int(group_rows[position, index]) != groups[name]["rows"]:
                raise RuntimeError("Group sizes differ between the device and the host")
        entry.update(overlap=agreement, consistency=consistency[reference["family_id"]], groups=groups)
        families[families.index(None)] = entry
    ok = [family for family in families if family["status"] == "ok"]
    if [family["family_id"] for family in ok] != [reference["family_id"] for reference in usable]:
        raise RuntimeError("Usable families lost their order")
    interventions = {"examples": int(examples), "limit": GROUPS_EXAMPLE_LIMIT, "control_draws": draws, "control_seed": control_seed,
                     "removed": "decoder contribution of the medoid factor (scale * activation * decoder row) from the raw embedding",
                     "residual": "reconstruction error and decoder bias are never replaced", "gates": gates}
    return {
        **base, "status": "ok", "reason": None, "_rows": int(count), "families": families, "family_ids": [family["family_id"] for family in ok],
        "evaluation_row_index": evaluation.astype(np.int64), "evaluation_is_entrant": evaluation_entrant,
        "mask_A": a_mask, "mask_R": r_mask, "group_code": code,
        "feature_mean": host["feature_mean"].astype(np.float32), "feature_std": host["feature_std"].astype(np.float32),
        "coactivation_positive_rows": host["positive_rows"].astype(np.int64), "coactivation_activation_sum": host["activation_sum"],
        "reconstruction_mse": host["reconstruction_mse"], **example_arrays,
        "cav_seeds": cav_seeds, "feature_names": names, "group_names": list(GROUP_NAMES), "interventions": interventions,
        "evaluation": {"rows": int(count), "patients": int(len(np.unique(evaluation_patients))), "deaths": int(evaluation_labels.sum()),
                       "returning_rows": int((~evaluation_entrant).sum()), "entrant_rows": int(evaluation_entrant.sum()), "role": "evaluation"},
        "statistics": {"partition": "R = frozen rule rows, A = medoid activation strictly above the frozen cutoff; four disjoint groups per family",
                       "feature_distribution": "mean and population standard deviation (denominator = group rows) of each original feature",
                       "coactivation": "per candidate medoid inside each group: prevalence = positive rows / group rows, magnitude = activation sum / group rows, positive magnitude = activation sum / positive rows",
                       "jaccard": "same-year mask overlap |R and A| / |R or A|; not an across-year identity overlap",
                       "examples": "highest medoid activation first among group rows with positive activation, ties by population row order; no label or probability is read",
                       "reconstruction_mse": "mean squared scaled-space reconstruction error of the medoid's SAE over the group rows"},
        "bootstrap": {"replicates": replicates, "confidence_level": 0.95, "sampling_unit": "patient",
                      "seed": "sha256 of (groups_reliability, split_seed, reference year, year, family, group)"},
        "baseline_source": "reference-year fitting-role rows inside the same frozen group definition; not refitted",
        "classification_threshold": 0.5, "gradient_source": "cached P(r) output; the predictor is not rerun",
        "outcomes_read": "reference-year fitting-role rows (baselines) and year-t evaluation rows (performance, example losses) only"}


def run_groups_interventions_unit(cache, view, roles, config, device, t):
    """Run D(r)/groups_interventions(t) through the cache; P(r), families, frozen references, dictionaries, the serialized
    continuation and the Idea 1 and Idea 2 frozen payloads are loaded from it."""
    if config_hash(config) != cache.config_hash:
        raise ValueError("Configuration differs from the cache configuration")
    if set(view) != set(GROUPS_VIEW_KEYS):
        raise ValueError(f"Groups read only {GROUPS_VIEW_KEYS}")
    r, t = int(roles["reference_year"]), int(t)
    if t not in groups_follow_up_years(config, r):
        raise ValueError(f"Year {t} is not a follow-up year of reference year {r}")
    return cache.run(f"D/{r}/groups_interventions/{t}",
                     lambda: groups_interventions_unit(view, r, t, roles, cache.load(f"P/{r}"), cache.load(f"D/{r}/families"),
                                                       load_frozen_references(cache, r), load_reference_dictionaries(cache, r, device),
                                                       load_continuation(cache, r, device), cache.load(f"D/{r}/idea1/{t}"),
                                                       cache.load(f"D/{r}/idea2_frozen/{t}"), config, device),
                     projected_bytes=projected_groups_interventions_bytes(config))


def groups_table(unit):
    """Compact rows per family and group: rows (returning/entrant), deaths, mean risk, Brier, skill, same-year R/A Jaccard, reconstruction error."""
    def number(value, digits=3):
        return None if value is None else round(value, digits)
    if unit["status"] != "ok":
        return [{"t": unit["year"], "status": f"unavailable:{unit['reason']}"}]
    rows = []
    for family in unit["families"]:
        if family["status"] != "ok":
            rows.append({"family": family["family_id"], "t": unit["year"], "status": f"unavailable:{family['reason']}"})
            continue
        for index, name in enumerate(GROUP_NAMES):
            group = family["groups"][name]
            metrics = group["reliability"]
            mse = float(unit["reconstruction_mse"][family["array_index"], index])
            rows.append({"family": family["family_id"], "t": unit["year"], "group": name, "rows": group["rows"],
                         "returning/entrant": f"{group['returning_rows']}/{group['entrant_rows']}", "deaths": group["deaths"],
                         "mean risk": number(metrics["mean_risk"]), "Brier": number(metrics["brier"]),
                         "skill": number(metrics["brier_skill"]), "R/A Jaccard": number(family["overlap"]["jaccard"]),
                         "recon MSE": None if mse != mse else round(mse, 4),
                         "unavailable": {**{name: reason for name, reason in metrics["unavailable_reasons"].items() if metrics.get(name) is None},
                                         **family["overlap"].get("unavailable_reasons", {}),
                                         **({"reconstruction_mse": "empty_group" if not group["rows"] else "nonfinite_reconstruction_mse"}
                                            if mse != mse else {})}})
    return rows


def interventions_table(unit):
    """Compact rows per example: activation, removal and residual norms, probability and Brier-loss changes against the
    norm-matched controls' mean, the CAV projection of the removal (seed 42) and the first-order change it predicts."""
    def number(value, digits=4):
        return None if value is None or value != value else round(float(value), digits)
    if unit["status"] != "ok":
        return [{"t": unit["year"], "status": f"unavailable:{unit['reason']}"}]
    ids = {family["array_index"]: family["family_id"] for family in unit["families"] if family["status"] == "ok"}
    column = unit["cav_seeds"].index(42) if 42 in unit["cav_seeds"] else 0
    rows = []
    for index in range(len(unit["example_family"])):
        rows.append({"family": ids[int(unit["example_family"][index])], "t": unit["year"],
                     "group": GROUP_NAMES[int(unit["example_group"][index])], "rank": int(unit["example_rank"][index]),
                     "row": int(unit["example_row_index"][index]), "activation": number(unit["example_activation"][index]),
                     "removal norm": number(unit["removal_norm"][index]), "residual norm": number(unit["residual_norm"][index]),
                     "p before": number(unit["probability_before"][index]), "p removed": number(unit["probability_removed"][index]),
                     "dp removed": number(unit["probability_removed"][index] - unit["probability_before"][index]),
                     "dp control mean": number(float(unit["probability_control"][index].mean()) - float(unit["probability_before"][index])),
                     "dBrier removed": number(unit["loss_removed"][index] - unit["loss_before"][index]),
                     "CAV projection 42": number(unit["cav_projection"][index][column]),
                     "first-order dp": number(unit["first_order_change"][index])})
    return rows


def coactivation_table(unit):
    """Compact rows per family, group and candidate medoid: prevalence, magnitude and positive magnitude with their denominators."""
    def number(value, digits=3):
        return None if value is None else round(float(value), digits)
    if unit["status"] != "ok":
        return [{"t": unit["year"], "status": f"unavailable:{unit['reason']}"}]
    computed = [family for family in unit["families"] if family["status"] == "ok"]
    ids = [family["family_id"] for family in computed]
    rows = []
    for position, family in enumerate(computed):
        family_id = family["family_id"]
        for index, name in enumerate(GROUP_NAMES):
            size = family["groups"][name]["rows"]
            for other, other_id in enumerate(ids):
                positive = int(unit["coactivation_positive_rows"][position, index, other])
                total = float(unit["coactivation_activation_sum"][position, index, other])
                rows.append({"family": family_id, "t": unit["year"], "group": name, "medoid of": other_id, "group rows": size,
                             "positive rows": positive, "prevalence": number(positive / size) if size else None,
                             "magnitude": number(total / size) if size else None,
                             "positive magnitude": number(total / positive) if positive else None})
    return rows


TEMPORAL_HOLDOUT_YEARS = 2
TEMPORAL_MODELS = ("persistence", "history", "history_concept")
TEMPORAL_SUBGROUP_GROUPS = {"A": ("R_and_A", "not_R_and_A"), "R": ("R_and_A", "R_and_not_A")}
TEMPORAL_CONCEPTS = {"A": ("activation_prevalence", "tcav_primary", "rule_transfer_f2", "reconstruction_mse"),
                     "R": ("medoid_retention", "medoid_prevalence", "medoid_magnitude", "reconstruction_mse")}
TEMPORAL_SOURCES = {"idea1": "idea1", "idea2": "idea2_frozen"}
TEMPORAL_CONSISTENCY_TOLERANCE = 1e-9
TEMPORAL_NO_UNCERTAINTY = ("aggregate unit payloads carry no patient rows: paired patient-aware resampling is unavailable here "
                           "and the seed dispersion of the explanations is not an uncertainty of the forecast")


def temporal_calendar(config, holdout_years=TEMPORAL_HOLDOUT_YEARS):
    """Common calendar boundary of every reference experiment: the final `holdout_years` distinct target years of the
    configured follow-up grid are held out and all earlier target years train. The boundary comes from the configuration,
    never from which units happen to be complete, so an interrupted run cannot move it."""
    if config.get("label_availability") != "end_of_year":
        raise ValueError("Forecasts assume labels complete at the end of each calendar year")
    if type(holdout_years) is not int or holdout_years < 1:
        raise ValueError("holdout_years must be a positive integer")
    targets = sorted({t for r in config["reference_year_candidates"] for t in idea1_follow_up_years(config, r)})
    holdout = targets[-holdout_years:]
    return {"target_years": targets, "holdout_years": holdout, "training_years": [t for t in targets if t < holdout[0]] if holdout else [],
            "last_year": int(max(config["reference_year_candidates"])), "label_availability": config["label_availability"],
            "definition": "target year t+1 of a transition t -> t+1; forecast origin is the end of year t"}


def _temporal_unit(units, key):
    """(payload, None) for an available unit payload, else (None, label); an absent unit is `not_run (stopped)`."""
    value = units.get(key, "not_run (stopped)")
    if isinstance(value, str):
        return None, value
    if not isinstance(value, dict):
        raise TypeError(f"Unit {key} must be a payload dict or a status label")
    if value.get("status") == "unavailable":
        return None, f"unavailable:{value.get('reason')}"
    if value.get("_rows") == 0:
        return None, "unavailable:empty_evaluation_population"
    return value, None


def _pooled(values, weights):
    """Row-weighted mean over the entries with positive weight and a finite value; None when nothing is available."""
    import numpy as np
    pairs = [(float(v), float(w)) for v, w in zip(values, weights) if v is not None and w and np.isfinite(v)]
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total else None


def _group_value(groups, family, subgroup, name):
    """Pooled Brier or reconstruction error of the subgroup (A or R) from the four disjoint groups of one family."""
    selected = [family["groups"][group] for group in TEMPORAL_SUBGROUP_GROUPS[subgroup]]
    weights = [group["rows"] for group in selected]
    if name == "brier":
        return _pooled([group["reliability"]["brier"] for group in selected], [group["reliability"]["records"] for group in selected])
    index = [GROUP_NAMES.index(group) for group in TEMPORAL_SUBGROUP_GROUPS[subgroup]]
    return _pooled([float(groups["reconstruction_mse"][family["array_index"], i]) for i in index], weights)


def _agreed(values):
    """The single reference-year value shared by every follow-up payload of one trajectory."""
    found = [value for value in values if value is not None]
    if not found:
        return None, "no_reference_value"
    if len(set(found)) > 1:
        return None, "inconsistent_reference_value"
    return found[0], None


def _entry(parts, kind, family_id, key="families", id_key="family_id"):
    payload, label = parts[kind]
    if payload is None:
        return None, f"{kind}:{label}"
    entry = next((item for item in payload[key] if item[id_key] == family_id), None)
    return (entry, None) if entry is not None else (None, f"{kind}:family_absent")


def _temporal_year(idea, family_id, parts, seed):
    """One trajectory-year record: the Brier loss and every concept metric with the reason when unavailable. The loss of
    A comes from the Idea 1 payload, that of R from the pooled four groups (cross-checked against Idea 1 where both exist)."""
    subgroup = dict(IDEAS)[idea]
    record = {"loss": None, "rows": None, "reason": None, "concepts": dict.fromkeys(TEMPORAL_CONCEPTS[subgroup]),
              "concept_reasons": {}}
    idea1, why1 = _entry(parts, "idea1", family_id)
    group_family, why_groups = _entry(parts, "groups", family_id)
    if group_family is not None and group_family["status"] != "ok":
        group_family, why_groups = None, f"groups:{group_family['reason']}"
    if subgroup == "A":
        if idea1 is None:
            record["reason"] = why1
        else:
            metrics = idea1["reliability"]["A"]
            if metrics["brier"] is None:
                record["reason"] = f"idea1:{metrics['unavailable_reasons'].get('brier', 'brier_unavailable')}"
            else:
                record.update(loss=float(metrics["brier"]), rows=int(metrics["records"]))
    elif group_family is None:
        record["reason"] = why_groups
    else:
        pooled = _group_value(parts["groups"][0], group_family, "R", "brier")
        rows = sum(group_family["groups"][group]["reliability"]["records"] for group in TEMPORAL_SUBGROUP_GROUPS["R"])
        if pooled is None:
            record["reason"] = "groups:empty_subgroup"
        elif idea1 is not None and idea1["reliability"]["R"]["brier"] is not None and (
                idea1["reliability"]["R"]["records"] != rows or abs(idea1["reliability"]["R"]["brier"] - pooled) > TEMPORAL_CONSISTENCY_TOLERANCE):
            record["reason"] = "inconsistent_sources:R_brier_differs_between_idea1_and_groups"
        else:
            record.update(loss=pooled, rows=int(rows))
    concepts, reasons = record["concepts"], record["concept_reasons"]
    if subgroup == "A":
        if idea1 is None:
            reasons.update(dict.fromkeys(("activation_prevalence", "tcav_primary", "rule_transfer_f2"), why1))
        else:
            concepts["activation_prevalence"] = idea1["observation"]["activation_prevalence"]
            tcav = idea1["observation"]["subgroup_influence"][seed]
            concepts["tcav_primary"], reasons["tcav_primary"] = tcav["tcav"], tcav["unavailable_reason"]
            concepts["rule_transfer_f2"] = idea1["original_rule_transfer"]["f2"]
    else:
        frozen, why = _entry(parts, "idea2", family_id)
        if frozen is not None and frozen["status"] != "ok":
            frozen, why = None, f"idea2:{frozen['reason']}"
        medoid = [member for member in frozen["members"] if member["is_medoid"]] if frozen is not None else []
        if frozen is None or len(medoid) != 1:
            reasons.update(dict.fromkeys(("medoid_retention", "medoid_prevalence", "medoid_magnitude"), why or "idea2:medoid_not_unique"))
        else:
            member = medoid[0]
            concepts["medoid_retention"] = member["retention"]["year_rate"]
            concepts["medoid_prevalence"] = member["year"]["positive_prevalence"]
            concepts["medoid_magnitude"] = member["year"]["magnitude_mean"]
            for name in ("medoid_prevalence", "medoid_magnitude"):
                reasons[name] = member["year"]["unavailable"]
    if group_family is None:
        reasons["reconstruction_mse"] = why_groups
    else:
        concepts["reconstruction_mse"] = _group_value(parts["groups"][0], group_family, subgroup, "mse")
        if concepts["reconstruction_mse"] is None:
            reasons["reconstruction_mse"] = "groups:empty_subgroup"
    for name, value in concepts.items():
        if value is None and not reasons.get(name):
            reasons[name] = "unavailable"
        elif value is not None:
            reasons.pop(name, None)
    return record


def _temporal_reference(idea, family_id, performance, units, r, years, seed):
    """The year-r record: loss from the reference performance unit, concept values from the reference fields that every
    follow-up payload of the trajectory repeats (they must agree)."""
    subgroup = dict(IDEAS)[idea]
    payload, label = performance
    record = {"loss": None, "rows": None, "reason": None, "concepts": dict.fromkeys(TEMPORAL_CONCEPTS[subgroup]), "concept_reasons": {}}
    if payload is None:
        record["reason"] = f"reference_performance:{label}"
    else:
        row = next((item for item in payload["rows"] if item["family_id"] == family_id and item["idea"] == idea), None)
        if row is None:
            record["reason"] = "reference_performance:row_absent"
        elif row["metrics"]["brier"] is None:
            record["reason"] = f"reference_performance:{row['metrics']['unavailable_reasons'].get('brier', 'brier_unavailable')}"
        else:
            record.update(loss=float(row["metrics"]["brier"]), rows=int(row["evaluation_rows"]))
    values = {name: [] for name in TEMPORAL_CONCEPTS[subgroup]}
    for t in years:
        if subgroup == "A":
            payload, _ = _temporal_unit(units, f"D/{r}/idea1/{t}")
            entry = next((item for item in payload["families"] if item["family_id"] == family_id), None) if payload is not None else None
            if entry is not None:
                values["activation_prevalence"].append(entry["reference_statistics"]["activation_prevalence"])
        else:
            payload, _ = _temporal_unit(units, f"D/{r}/idea2_frozen/{t}")
            entry = next((item for item in payload["families"] if item["family_id"] == family_id and item["status"] == "ok"), None) \
                if payload is not None else None
            medoid = [member for member in entry["members"] if member["is_medoid"]] if entry is not None else []
            if len(medoid) == 1:
                values["medoid_retention"].append(medoid[0]["retention"]["reference_rate"])
                values["medoid_prevalence"].append(medoid[0]["reference"]["positive_prevalence"])
                values["medoid_magnitude"].append(medoid[0]["reference"]["magnitude_mean"])
    for name in TEMPORAL_CONCEPTS[subgroup]:
        record["concepts"][name], reason = _agreed(values.get(name, []))
        if record["concepts"][name] is None:
            record["concept_reasons"][name] = reason if name in values else "no_reference_value"
    return record


def temporal_trajectories(config, units):
    """One trajectory per (reference year, family, idea) with a record for every year r..last: loss, rows and concept metrics,
    or the reason they are unavailable (unit label or payload status). Nothing is imputed; a missing year stays missing."""
    seed, last = str(config["primary_cav_seed"]), int(max(config["reference_year_candidates"]))
    trajectories = []
    for r in config["reference_year_candidates"]:
        years = idea1_follow_up_years(config, r)
        if not years:
            continue
        performance = _temporal_unit(units, f"D/{r}/reference_performance")
        parts = {t: {"idea1": _temporal_unit(units, f"D/{r}/idea1/{t}"), "idea2": _temporal_unit(units, f"D/{r}/idea2_frozen/{t}"),
                     "groups": _temporal_unit(units, f"D/{r}/groups_interventions/{t}")} for t in years}
        found = set()
        if performance[0] is not None:
            found.update((row["family_id"], row["idea"]) for row in performance[0]["rows"])
        for t in years:
            if parts[t]["idea1"][0] is not None:
                found.update((family["family_id"], "idea1") for family in parts[t]["idea1"][0]["families"])
            if parts[t]["idea2"][0] is not None:
                found.update((family["family_id"], "idea2") for family in parts[t]["idea2"][0]["families"])
        for family_id, idea in sorted(found):
            records = {r: _temporal_reference(idea, family_id, performance, units, r, years, seed)}
            for t in range(r + 1, last + 1):
                records[t] = _temporal_year(idea, family_id, parts[t], seed) if t in parts else {
                    "loss": None, "rows": None, "reason": "year_not_in_configuration",
                    "concepts": dict.fromkeys(TEMPORAL_CONCEPTS[dict(IDEAS)[idea]]),
                    "concept_reasons": dict.fromkeys(TEMPORAL_CONCEPTS[dict(IDEAS)[idea]], "year_not_in_configuration")}
            trajectories.append({"key": f"{r}/{family_id}/{idea}", "reference_year": int(r), "family_id": family_id, "idea": idea,
                                 "subgroup": dict(IDEAS)[idea], "years": records})
    return trajectories


def temporal_transitions(trajectory, last_year):
    """Transitions t -> t+1 for every origin from the reference year through last_year - 1. A transition exists only when
    both consecutive years have a loss; otherwise it is listed as skipped with the reason. Years are never bridged."""
    years, rows = trajectory["years"], []
    for origin in range(trajectory["reference_year"], int(last_year)):
        before, after = years[origin], years[origin + 1]
        reasons = [f"{label}_loss_missing:{record['reason']}" for label, record in (("origin", before), ("target", after))
                   if record["loss"] is None]
        item = {"origin": origin, "target": origin + 1, "status": "skipped" if reasons else "ok",
                "reason": "; ".join(reasons) if reasons else None, "loss_origin": None, "loss_target": None, "delta_loss": None,
                "concepts_origin": dict(before["concepts"]), "concepts_target": dict(after["concepts"])}
        if not reasons:
            item.update(loss_origin=before["loss"], loss_target=after["loss"], delta_loss=after["loss"] - before["loss"])
        rows.append(item)
    return rows


def temporal_ols(design, response, columns):
    """Least squares with an intercept. Unavailable (never extrapolated or regularised) without a finite design, at least one
    residual degree of freedom or full column rank."""
    import numpy as np
    y = np.asarray(response, dtype=np.float64).reshape(-1)
    n, parameters = len(y), int(columns) + 1
    x = np.asarray(design, dtype=np.float64).reshape(n, int(columns))
    result = {"status": "unavailable", "reason": None, "n": n, "parameters": parameters, "residual_df": n - parameters,
              "rank": None, "coefficients": None}
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        result["reason"] = "nonfinite_inputs"
    elif n - parameters < 1:
        result["reason"] = "insufficient_residual_df"
    else:
        full = np.column_stack([np.ones(n), x])
        result["rank"] = int(np.linalg.matrix_rank(full))
        if result["rank"] < parameters:
            result["reason"] = "rank_deficient"
        else:
            result.update(status="ok", coefficients=[float(value) for value in np.linalg.lstsq(full, y, rcond=None)[0]])
    return result


def _ols_predict(fit, design):
    import numpy as np
    x = np.asarray(design, dtype=np.float64).reshape(len(design), fit["parameters"] - 1)
    return np.column_stack([np.ones(len(x)), x]) @ np.asarray(fit["coefficients"])


def temporal_forecast(transitions, calendar, concept=None):
    """One-year-ahead forecast of delta_loss = L[t+1] - L[t] for one trajectory. Models: persistence (delta 0), OLS on L[t]
    (history) and, for a concept, OLS on L[t] and C[t]. Fitted only on transitions whose target year is earlier than every
    held-out target year; scored by absolute error on the held-out transitions; all models use the same transitions, so the
    differences are paired. A transition without the concept at its origin is excluded for every model of that concept."""
    import numpy as np
    holdout, training = list(calendar["holdout_years"]), list(calendar["training_years"])
    usable, excluded = [], []
    for item in transitions:
        if item["status"] != "ok":
            excluded.append({"origin": item["origin"], "target": item["target"], "reason": item["reason"]})
        elif concept is not None and item["concepts_origin"].get(concept) is None:
            excluded.append({"origin": item["origin"], "target": item["target"], "reason": f"concept_missing_at_origin:{concept}"})
        else:
            usable.append(item)
    if any(item["target"] not in holdout and item["target"] not in training for item in usable):
        raise ValueError("A transition target lies outside the calendar")
    train = sorted((item for item in usable if item["target"] in training), key=lambda item: item["target"])
    test = sorted((item for item in usable if item["target"] in holdout), key=lambda item: item["target"])
    if train and test and max(item["target"] for item in train) >= min(item["target"] for item in test):
        raise RuntimeError("Fitting transitions must end strictly before the held-out calendar years")
    result = {"concept": concept, "status": "unavailable", "reason": None, "excluded": excluded,
              "training": {"n": len(train), "target_years": [item["target"] for item in train]},
              "holdout": {"n": len(test), "target_years": [item["target"] for item in test]}, "models": {}, "mae": {},
              "paired": {}, "transitions": []}
    if not test:
        result["reason"] = "no_holdout_transitions"
        return result
    result["status"] = "ok"
    delta = np.asarray([item["delta_loss"] for item in test])
    history_columns = lambda items: [[item["loss_origin"]] for item in items]
    concept_columns = lambda items: [[item["loss_origin"], item["concepts_origin"][concept]] for item in items]
    specs = {"history": (history_columns, 1)}
    if concept is not None:
        specs["history_concept"] = (concept_columns, 2)
    predictions = {"persistence": np.zeros(len(test))}
    result["models"]["persistence"] = {"status": "ok", "reason": None}
    for name, (columns, width) in specs.items():
        fit = temporal_ols(columns(train), [item["delta_loss"] for item in train], width)
        result["models"][name] = fit
        if fit["status"] == "ok":
            predictions[name] = _ols_predict(fit, columns(test))
    errors = {name: np.abs(delta - value) for name, value in predictions.items()}
    result["mae"] = {name: (float(errors[name].mean()) if name in errors else None) for name in TEMPORAL_MODELS
                     if name == "persistence" or name in specs}
    for name in specs:
        if name not in errors:
            result["models"][name]["reason"] = result["models"][name]["reason"] or "unavailable"
    for first, second in (("history", "persistence"), ("history_concept", "history"), ("history_concept", "persistence")):
        if first in errors and second in errors:
            result["paired"][f"{first}_minus_{second}"] = float((errors[first] - errors[second]).mean())
    for index, item in enumerate(test):
        result["transitions"].append({"origin": item["origin"], "target": item["target"], "delta_loss": float(delta[index]),
                                      "prediction": {name: float(value[index]) for name, value in predictions.items()},
                                      "abs_error": {name: float(value[index]) for name, value in errors.items()}})
    return result


def _pearson(left, right):
    import numpy as np
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if len(x) < 3:
        return None, "fewer_than_three_transitions"
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, "constant_series"
    return float(np.corrcoef(x, y)[0, 1]), None


def temporal_patterns(trajectory, concept):
    """Descriptive, forecast-free view of every actual consecutive transition: deterioration (delta_loss > 0, no further
    threshold) beside the signed concept change in the preceding, same and following interval, so that concept changes
    before, concurrent with, after or without deterioration all stay visible, plus the concurrent Pearson association."""
    years, rows, pairs = trajectory["years"], [], []

    def change(start):
        if start not in years or start + 1 not in years:
            return None
        a, b = years[start]["concepts"].get(concept), years[start + 1]["concepts"].get(concept)
        return None if a is None or b is None else b - a

    for origin in range(trajectory["reference_year"], max(years)):
        before, after = years[origin], years[origin + 1]
        if before["loss"] is None or after["loss"] is None:
            continue
        delta = after["loss"] - before["loss"]
        rows.append({"origin": origin, "target": origin + 1, "delta_loss": delta,
                     "deterioration": "deterioration" if delta > 0 else "no_deterioration",
                     "concept_change_before": change(origin - 1), "concept_change_concurrent": change(origin),
                     "concept_change_after": change(origin + 1)})
        if rows[-1]["concept_change_concurrent"] is not None:
            pairs.append((rows[-1]["concept_change_concurrent"], delta))
    value, reason = _pearson([p[0] for p in pairs], [p[1] for p in pairs])
    return {"concept": concept, "rows": rows, "concurrent": {"n": len(pairs), "pearson": value, "unavailable": reason,
                                                            "scope": "all consecutive transitions; descriptive, not a forecast"}}


def temporal_overlap(config, units):
    """Evaluation-row overlap of the reference experiments per target calendar year. One row is one patient-year, so rows
    shared within a year are patients shared within that year. Overlapping experiments are not independent replicates."""
    import numpy as np
    rows = {}
    for r in config["reference_year_candidates"]:
        for t in idea1_follow_up_years(config, r):
            for kind in ("idea1", "idea2_frozen", "groups_interventions"):
                payload, _ = _temporal_unit(units, f"D/{r}/{kind}/{t}")
                if payload is not None and "evaluation_row_index" in payload:
                    rows[(r, t)] = (kind, np.unique(np.asarray(payload["evaluation_row_index"], dtype=np.int64)))
                    break
    by_year = {}
    for t in sorted({year for _, year in rows}):
        members = sorted(r for r, year in rows if year == t)
        pairs = []
        for position, first in enumerate(members):
            for second in members[position + 1:]:
                shared = len(np.intersect1d(rows[(first, t)][1], rows[(second, t)][1], assume_unique=True))
                union = len(rows[(first, t)][1]) + len(rows[(second, t)][1]) - shared
                pairs.append({"reference_years": [int(first), int(second)], "shared_rows": int(shared), "union_rows": int(union),
                              "jaccard": shared / union if union else None})
        by_year[int(t)] = {"reference_years": [int(r) for r in members], "rows": {str(r): int(len(rows[(r, t)][1])) for r in members},
                           "sources": {str(r): rows[(r, t)][0] for r in members}, "pairs": pairs}
    return {"by_year": by_year, "independent_replicates": False,
            "note": "Experiments of different reference years evaluate the same calendar years and may share patients; families, "
                    "ideas and SAE seeds inside one experiment share the same patients. Counts of trajectories are not "
                    "counts of independent replicates."}


def temporal_summary(forecasts, calendar):
    """Descriptive counts and mean held-out MAE per (idea, concept) over trajectories with paired evidence; never a score."""
    groups = {}
    for entry in forecasts:
        groups.setdefault((entry["idea"], entry["forecast"]["concept"]), []).append(entry)
    rows = []
    for (idea, concept), entries in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        evaluated = [e["forecast"] for e in entries if e["forecast"]["status"] == "ok"]
        fitted = [f for f in evaluated if "history" in f["mae"] and f["mae"]["history"] is not None
                  and (concept is None or f["mae"].get("history_concept") is not None)]
        row = {"idea": idea, "concept": concept, "trajectories": len(entries), "with_holdout": len(evaluated), "with_fitted_models": len(fitted),
               "mean_mae": {name: (sum(f["mae"][name] for f in fitted) / len(fitted) if fitted else None)
                            for name in TEMPORAL_MODELS if name == "persistence" or name == "history" or concept is not None}}
        if concept is not None:
            better = [f["paired"]["history_concept_minus_history"] < 0 for f in fitted]
            row["concept_lower_mae_than_history"] = sum(better)
        rows.append(row)
    return {"rows": rows, "holdout_years": list(calendar["holdout_years"]), "independent_replicates": False,
            "uncertainty": {"status": "unavailable", "reason": TEMPORAL_NO_UNCERTAINTY}}


def temporal_analysis(config, units):
    """Analysis-stage computation over completed D units (not a cached unit): trajectories, consecutive-year transitions,
    calendar-holdout forecasts of delta_loss against persistence, descriptive patterns, overlap and availability labels.
    `units` maps unit keys (reference_performance, idea1, idea2_frozen and groups_interventions) to a payload dict or a
    status label; an absent key is `not_run (stopped)`."""
    calendar = temporal_calendar(config)
    trajectories = temporal_trajectories(config, units)
    forecasts, patterns = [], []
    for trajectory in trajectories:
        transitions = temporal_transitions(trajectory, calendar["last_year"])
        for concept in (None, *TEMPORAL_CONCEPTS[trajectory["subgroup"]]):
            forecasts.append({"trajectory": trajectory["key"], "reference_year": trajectory["reference_year"],
                              "family_id": trajectory["family_id"], "idea": trajectory["idea"],
                              "forecast": temporal_forecast(transitions, calendar, concept)})
        for concept in TEMPORAL_CONCEPTS[trajectory["subgroup"]]:
            patterns.append({"trajectory": trajectory["key"], "idea": trajectory["idea"], **temporal_patterns(trajectory, concept)})
    availability = []
    for r in config["reference_year_candidates"]:
        years = idea1_follow_up_years(config, r)
        for key in ([f"D/{r}/reference_performance"] if years else []) + [f"D/{r}/{kind}/{t}" for t in years
                                                                            for kind in ("idea1", "idea2_frozen", "groups_interventions")]:
            payload, label = _temporal_unit(units, key)
            availability.append({"unit": key, "status": "ok" if payload is not None else label})
    holdout = {str(year): sorted({e["reference_year"] for e in forecasts if e["forecast"]["concept"] is None
                                  and year in e["forecast"]["holdout"]["target_years"]}) for year in calendar["holdout_years"]}
    return {"calendar": calendar, "trajectories": trajectories, "forecasts": forecasts, "patterns": patterns,
            "overlap": temporal_overlap(config, units), "availability": availability,
            "holdout_reference_years": holdout, "summary": temporal_summary(forecasts, calendar),
            "policy": {"history_baseline": "persistence (delta_loss = 0)", "ols": "delta_loss[t+1] ~ L[t] versus ~ L[t] + C[t], intercept, "
                       "paired absolute errors on the common held-out calendar years", "composite_score": None, "alarm_threshold": None}}


def temporal_units(cache, config):
    """Completed payloads (or status labels) of every unit the temporal analysis reads, from the unit cache."""
    keys = []
    for r in config["reference_year_candidates"]:
        years = idea1_follow_up_years(config, r)
        keys += ([f"D/{r}/reference_performance"] if years else []) + [f"D/{r}/{kind}/{t}" for t in years
                                                                      for kind in ("idea1", "idea2_frozen", "groups_interventions")]
    status = {row["unit"]: row["status"] for row in cache.completeness_table(keys)}
    return {key: cache.load(key) if status[key] == "complete" else status[key] for key in keys}


def run_temporal_analysis(cache, config):
    """Temporal analysis over the completed units of a cache; missing, unavailable and failed units are labelled, never filled."""
    if config_hash(config) != cache.config_hash:
        raise ValueError("Configuration differs from the cache configuration")
    return temporal_analysis(config, temporal_units(cache, config))


def temporal_tables(result):
    """Compact row lists: forecasts (paired held-out MAE), concurrent associations, patterns, overlap and availability."""
    def number(value, digits=4):
        return None if value is None else round(value, digits)
    forecast = []
    for entry in result["forecasts"]:
        item, mae, paired = entry["forecast"], entry["forecast"]["mae"], entry["forecast"]["paired"]
        status = item["status"] if item["status"] == "ok" else f"unavailable:{item['reason']}"
        models = {name: (fit["status"] if fit["status"] == "ok" else f"unavailable:{fit['reason']}") for name, fit in item["models"].items()}
        forecast.append({"trajectory": entry["trajectory"], "concept": item["concept"], "status": status,
                         "train n": item["training"]["n"], "holdout n": item["holdout"]["n"],
                         "MAE persistence": number(mae.get("persistence")), "MAE history": number(mae.get("history")),
                         "MAE history+concept": number(mae.get("history_concept")),
                         "d(concept-history)": number(paired.get("history_concept_minus_history")),
                         "history": models.get("history"), "history+concept": models.get("history_concept")})
    concurrent = [{"trajectory": entry["trajectory"], "concept": entry["concept"], "n": entry["concurrent"]["n"],
                   "pearson": number(entry["concurrent"]["pearson"]), "unavailable": entry["concurrent"]["unavailable"]}
                  for entry in result["patterns"]]
    patterns = [{"trajectory": entry["trajectory"], "concept": entry["concept"], "origin": row["origin"], "target": row["target"],
                 "delta_loss": number(row["delta_loss"]), "pattern": row["deterioration"],
                 "dC before": number(row["concept_change_before"]), "dC concurrent": number(row["concept_change_concurrent"]),
                 "dC after": number(row["concept_change_after"])} for entry in result["patterns"] for row in entry["rows"]]
    overlap = [{"t": year, "reference years": item["reference_years"], "pair": pair["reference_years"], "shared rows": pair["shared_rows"],
                "union rows": pair["union_rows"], "jaccard": number(pair["jaccard"])}
               for year, item in sorted(result["overlap"]["by_year"].items()) for pair in item["pairs"]]
    return {"forecast": forecast, "concurrent": concurrent, "patterns": patterns, "overlap": overlap,
            "availability": list(result["availability"]), "summary": result["summary"]["rows"]}


def run_fixed_factor_checks():
    """Changing descriptive rules must never move the reference activation anchor."""
    import numpy as np
    import pandas as pd
    roles = {"tabpfn_context": ["context"], "sae_discovery": ["discovery"],
             "rule_discovery": ["rule-fit"], "cav_fit": ["cav-fit"],
             "concept_selection": ["selection"], "evaluation": ["held-a", "held-b"]}
    ids = ["context", "rule-fit", "selection", "held-a", "held-b"]
    fitting = annual_fitting_mask(ids, roles, "ref2007")
    assert fitting.tolist() == [True, True, False, False, False]
    entrants = [f"entrant-{i}" for i in range(32)]
    entrance_mask = annual_fitting_mask(entrants, roles, "ref2007")
    assert entrance_mask.tolist() == [stable_entrant_role(patient, "ref2007") == "fitting" for patient in entrants]
    assert entrance_mask.tolist() == annual_fitting_mask(entrants, roles, "ref2007").tolist()
    agreement = rule_activation_agreement([True, True, False, False], [True, False, True, False])
    assert agreement["precision"] == agreement["recall"] == agreement["f2"] == .5
    assert agreement["jaccard"] == 1 / 3 and agreement["rule_coverage"] == .5
    assert rule_activation_agreement([], [])["precision"] is None
    class Rule:
        def __init__(self, feature):
            self.feature = feature
        def mask(self, X):
            return np.asarray(X)[:, self.feature] > .5
        def to_dict(self):
            return {"feature": self.feature, "operator": ">", "threshold": .5}
    reference = freeze_reference_objects({"valid": True, "cutoff": .5}, Rule(0),
        {42: {"valid": True, "cav": np.asarray([1., 0.])},
         43: {"valid": True, "cav": np.asarray([1., 0.])},
         44: {"valid": True, "cav": np.asarray([1., 0.])}})
    frame = pd.DataFrame({"patient_id": ["held-a", "held-b"], "year": [2008, 2008],
                          "feature0": [1., 0.], "feature1": [0., 1.], "y": [1, 0], "p": [.7, .2]})
    gradients = np.asarray([[-1., 0.], [1., 0.]])
    values = np.asarray([.6, .4])
    fit_frame = pd.DataFrame({"patient_id": ["context", "rule-fit", "selection", "held-a", "context"],
        "year": [2008, 2008, 2008, 2008, 2007], "feature0": [1., 0., 10., 20., 30.],
        "feature1": [0., 1., 10., 20., 30.]})
    calls = []
    def capture_fit(features, activation, target, patients, feature_names, seed):
        calls.append((features.copy(), activation.copy(), list(patients), deepcopy(target)))
        return {"rule": Rule(1), "failure_reason": None}
    yearly_rule = fit_yearly_descriptive_rule(fit_frame, [.6, .4, .9, .9, .9], reference["target"],
        roles, "ref2007", ["feature0", "feature1"], 2008, rule_fitter=capture_fit)
    assert calls[0][2] == ["context", "rule-fit"] and calls[0][0].shape == (2, 2)
    assert yearly_rule["fitting_records"] == 2 and yearly_rule["rule_role"] == "yearly_descriptive_only"
    future = fit_frame.copy()
    future.loc[future.index[-1], "year"] = 2009
    try:
        fit_yearly_descriptive_rule(future, [.6, .4, .9, .9, .9], reference["target"],
            roles, "ref2007", ["feature0", "feature1"], 2008, rule_fitter=capture_fit)
    except ValueError:
        pass
    else:
        raise AssertionError("Yearly fitting accessed a future population")
    first = fixed_factor_observation(frame, values, gradients, reference, "eligible-reference-factor", 0, 42,
        fit_constant_baseline([0, 1]), {"activation_prevalence": .5, "positive_activation_mean": .5},
        2008, 2007, ["feature0", "feature1"], new_rule=Rule(1))
    assert first["activation_membership"] == [True, False]
    assert first["original_rule_agreement"]["jaccard"] == 1
    assert first["new_rule_agreement"]["jaccard"] == 0
    assert first["activation_prevalence"] == .5 and first["positive_activation_mean"] == .5
    assert first["activation_subgroup_mean"] == .6
    assert first["subgroup_influence"][42]["tcav"] == 0
    assert first["whole_cohort_influence"][42]["tcav"] == .5
    assert first["reference_eligibility_frozen"] is True
    assert first["factor_identity"] == "eligible-reference-factor" and first["sae_seed"] == 42
    changed_rule = fixed_factor_observation(frame, values, gradients, reference, "eligible-reference-factor", 0, 42,
        fit_constant_baseline([0, 1]), {"activation_prevalence": 0., "positive_activation_mean": 0.},
        2008, 2007, ["feature0", "feature1"], new_rule=Rule(0))
    assert changed_rule["activation_membership"] == first["activation_membership"]
    assert changed_rule["prevalence_ratio"] is None and changed_rule["magnitude_ratio"] is None
    assert changed_rule["unavailable_reasons"]["prevalence_ratio"] == "zero_reference_denominator"
    empty = fixed_factor_observation(frame, [0., .1], gradients, reference, "eligible-reference-factor", 0, 42,
        fit_constant_baseline([0, 1]), {"activation_prevalence": .5, "positive_activation_mean": .5},
        2008, 2007, ["feature0", "feature1"])
    assert empty["reliability"]["brier"] is None and empty["subgroup_influence"][42]["tcav"] is None
    try:
        fixed_factor_observation(frame, values, gradients, reference, "eligible-reference-factor", 0, 42,
            fit_constant_baseline([0, 1]), {"activation_prevalence": .5, "positive_activation_mean": .5},
            2009, 2007, ["feature0", "feature1"])
    except ValueError:
        pass
    else:
        raise AssertionError("A different/future calendar population was accepted")
    reference["target"]["cutoff"] = .7
    try:
        fixed_factor_observation(frame, values, gradients, reference, "eligible-reference-factor", 0, 42,
            fit_constant_baseline([0, 1]), {"activation_prevalence": .5, "positive_activation_mean": .5},
            2008, 2007, ["feature0", "feature1"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Changed activation anchor accepted")
    print("Fixed-factor anchors, annual fitting isolation and held-out fidelity/influence fixtures passed")


IDEA1_BACKEND = {"n_bootstraps": 8, "trees_per_bootstrap": 10, "max_depth": 6}


def _check_idea1_settings(config):
    """Zero quality filters on the select config, follow-up years only, graph dependencies, pure rule-change description."""
    import numpy as np
    reference = rule_selection_config(config["rule"])
    descriptive = descriptive_selection_config(config["rule"])
    assert (reference.min_precision, reference.min_lift, reference.min_marginal_recall) == (.5, 1.5, .02)
    assert (descriptive.min_precision, descriptive.min_lift, descriptive.min_marginal_recall) == (0, 0, 0)
    for name in ("objective", "max_rules", "max_rule_length"):
        assert getattr(descriptive, name) == getattr(reference, name), name
    assert descriptive_selection_config({**config["rule"], "min_precision": .9, "min_lift": 3., "min_marginal_recall": .5}) == descriptive
    assert IDEA1_QUALITY_FILTERS == ("min_precision", "min_lift", "min_marginal_recall")
    pilot = {"reference_year_candidates": list(range(2007, 2016)), "t0_only_reference_years": [2015], "sae_seeds": [42, 135, 215]}
    assert idea1_follow_up_years(pilot, 2007) == list(range(2008, 2016)) and idea1_follow_up_years(pilot, 2014) == [2015]
    assert idea1_follow_up_years(pilot, 2015) == [] and idea1_follow_up_years(config, 2007) == [2008, 2009]
    assert idea1_follow_up_years(config, 2008) == [2009] and idea1_follow_up_years(config, 2009) == []
    graph = unit_graph(pilot)
    assert graph["D/2007/idea1/2010"] == ("P/2007", "D/2007/rules", "D/2007/cavs_selection")
    assert "D/2007/idea1/2007" not in graph and not [key for key in graph if key.startswith("D/2015/idea1")]
    assert unit_stage("D/2007/idea1/2010") == "D.idea1"

    X = np.asarray([[1., 0., 0.], [1., 1., 0.], [0., 1., 0.], [0., 0., 0.]])
    names = ["a", "b", "c"]
    first, second, third = _positive_rule(0, "a"), _positive_rule(1, "b"), _positive_rule(2, "c")
    same = describe_rule_change(first, first, X, names)
    assert same["identical_rule"] and same["mask_jaccard"] == 1 and same["conditions_added"] == same["conditions_removed"] == []
    moved = describe_rule_change(first, second, X, names)
    assert not moved["identical_rule"] and moved["conditions_added"] == ["b > 0.5"] and moved["conditions_removed"] == ["a > 0.5"]
    assert moved["features_added"] == ["b"] and moved["features_removed"] == ["a"] and moved["features_kept"] == 0
    assert moved["mask_jaccard"] == 1 / 3 and moved["evaluation_rows"] == 4 and (moved["reference_rule_rows"], moved["yearly_rule_rows"]) == (2, 2)
    nothing = describe_rule_change(third, third, X, names)
    assert nothing["mask_jaccard"] is None and nothing["unavailable_reason"] == "both_rules_select_no_rows" and nothing["identical_rule"]
    _refuses(lambda: describe_rule_change(first, second, X, ["a", "x", "c"]), "is not")


def _idea1_world():
    """The selection world with entrant patients in 2008 and activations of 2008 driven by features: alpha keeps feature_0
    (no drift), epsilon moves from feature_4 to feature_3."""
    import numpy as np
    world = _selection_world()
    population, roles = world["population"], world["roles"]
    year = population["year"]
    renamed = (year == 2008) & (np.arange(len(year)) % 4 == 0)
    patient = population["patient_id"].astype("<U16")
    patient[renamed] = np.char.add("entrant-", np.char.partition(patient[renamed], "-")[:, 2])
    keys = np.asarray([f"{p}|{t}" for p, t in zip(patient, year)])
    X, E = world["X"].copy(), world["E"].copy()
    later, rng = year == 2008, np.random.default_rng(101)
    for factor, driver in ((0, 0), (4, 3)):
        high = X[later, driver] > .5
        E[later, factor] = np.where(high, 2.5 + rng.random(int(later.sum())), -1. + .3 * rng.random(int(later.sum())))
    population = _population_with(population, X=X, patient_id=patient, record_keys=keys)
    roles2 = reference_year_roles(population, 2007, 42)
    assert np.array_equal(roles2["row_role"][year == 2007], roles["row_role"][year == 2007]) and roles2["is_entrant"][later].any()
    identity = population["fingerprint"]["population_identity"]
    rows = world["output"]["row_index"]
    output = dict(world["output"], embeddings=E[rows].copy(), roles_identity=roles2["identity"], population_identity=identity)
    family_output = dict(world["family_output"], roles_identity=roles2["identity"], population_identity=identity)
    return dict(world, population=population, roles=roles2, X=X, E=E, y=population["y"], output=output, family_output=family_output)


def _idea1_rows(world):
    import numpy as np
    population, roles = world["population"], world["roles"]
    year, role, entrant = population["year"], roles["row_role"], roles["is_entrant"]
    later = year == 2008
    return {"fit_r": np.flatnonzero((year == 2007) & np.isin(role, FITTING_SIDE_ROLE_NAMES)),
            "eval_r": np.flatnonzero((year == 2007) & (role == "evaluation")),
            "fit_t": np.flatnonzero(later & np.where(entrant, role == "fitting", np.isin(role, FITTING_SIDE_ROLE_NAMES))),
            "eval_t": np.flatnonzero(later & (role == "evaluation")),
            "selection_t": np.flatnonzero(later & (role == "concept_selection"))}


def _idea1_poison(world, view, output, *, embeddings, gradients, features, labels):
    """Copies of the view and cached P(r) output with NaN (features, embeddings, gradients, probabilities) or flipped labels
    outside the allowed rows; probabilities stay only on the given gradient rows."""
    import numpy as np
    rows = output["row_index"]
    result, poisoned = dict(output), dict(view)
    result["embeddings"] = np.where(np.isin(rows, embeddings)[:, None], output["embeddings"], np.nan).astype(np.float32)
    result["gradients"] = np.where(np.isin(rows, gradients)[:, None], output["gradients"], np.nan).astype(np.float32)
    result["probabilities"] = np.where(np.isin(rows, gradients)[:, None], output["probabilities"], np.nan).astype(np.float32)
    X = view["X"].copy()
    X[~np.isin(np.arange(len(X)), features)] = np.nan
    poisoned["X"] = X
    y = view["y"].copy()
    flip = ~np.isin(np.arange(len(y)), labels)
    y[flip] = 1 - y[flip]
    poisoned["y"] = y
    return poisoned, result


def _idea1_equal(left, right):
    _unit_arrays_equal(left, right)


def _check_idea1_unit(config, world):
    """Frozen A_t, rule transfer, unfiltered yearly rules on fitting patients only, held-out reliability at the fixed baseline."""
    import copy
    import inspect
    import numpy as np
    from semantic_rules import select_rule_set
    population, roles = world["population"], world["roles"]
    view = performance_population_view(population)
    dimension, ids, rows = world["dimension"], world["identifiers"], _idea1_rows(world)
    dictionaries = _identity_dictionaries(dimension, [42, 135, 215])
    rule_view = rule_population_view(population)
    probe = cavs_selection_unit(rule_view, 2007, roles, world["output"], world["family_output"], world["frozen"], dictionaries, config, "cpu")
    output = _with_gradients(world, probe["cav_vectors"])
    selection = cavs_selection_unit(rule_view, 2007, roles, output, world["family_output"], world["frozen"], dictionaries, config, "cpu")
    assert selection["eligible"]["idea1"] == [ids["alpha"], ids["epsilon"]] and ids["beta"] in selection["eligible"]["idea2"]
    references = _references_from_units(selection, world["frozen"])
    snapshot = copy.deepcopy(references)
    calls = []

    def counting(candidates, X_tune, y_tune, selection_config, *, threshold_name):
        calls.append({"X_tune": X_tune.copy(), "config": selection_config, "name": threshold_name})
        return select_rule_set(candidates, X_tune, y_tune, selection_config, threshold_name=threshold_name)

    def run(v=view, out=output, refs=references, dicts=dictionaries, t=2008, family_output=None, **keywords):
        keywords.setdefault("backend", IDEA1_BACKEND)
        keywords.setdefault("workers", 1)
        return idea1_year_unit(v, 2007, t, roles, out, world["family_output"] if family_output is None else family_output,
                               refs, dicts, config, "cpu", **keywords)

    unit = run(select=counting)
    fitting, evaluation = rows["fit_t"], rows["eval_t"]
    assert len(calls) == unit["jobs"] == 2 and all(call["name"] == "frozen_threshold" for call in calls)
    for call in calls:
        used = call["config"]
        assert (used.min_precision, used.min_lift, used.min_marginal_recall) == (0, 0, 0), "Yearly rules must not be filtered"
        assert (used.objective, used.max_rules, used.max_rule_length) == (config["rule"]["objective"], 5, 10)
    assert unit["rule_selection"]["min_precision"] == unit["rule_selection"]["min_lift"] == unit["rule_selection"]["min_marginal_recall"] == 0
    assert unit["candidate_filter"] == config["rule"]["recurrence"] and unit["reference_year"] == 2007 and unit["year"] == 2008
    assert [family["family_id"] for family in unit["families"]] == [ids["alpha"], ids["epsilon"]] == unit["family_ids"]
    assert {item["family_id"] for item in unit["not_computed"]} == {ids["beta"], ids["gamma"], ids["delta"]}
    assert {item["reason"] for item in unit["not_computed"]} == {"not_eligible_at_selection"}

    t_rows = population["year"] == 2008
    mask = annual_fitting_mask(population["patient_id"][t_rows], roles["reference_roles"], 2007, 42)
    assert np.array_equal(np.flatnonzero(t_rows)[mask], fitting), "Year-t fitting patients differ from the permanent fitting assignment"
    assert unit["fitting_rows_identity"] == _identity_hash({"row_index": fitting})
    assert unit["evaluation_rows_identity"] == _identity_hash({"row_index": evaluation})
    assert np.array_equal(unit["evaluation_row_index"], evaluation) and unit["_rows"] == len(evaluation)
    entrant = roles["is_entrant"]
    assert unit["fitting"]["rows"] == len(fitting) and unit["fitting"]["entrant_rows"] == int(entrant[fitting].sum()) > 0
    assert unit["fitting"]["returning_rows"] == int((~entrant[fitting]).sum()) > 0
    assert unit["evaluation"]["entrant_rows"] == int(entrant[evaluation].sum()) > 0
    assert unit["evaluation"]["returning_rows"] == int((~entrant[evaluation]).sum()) > 0
    assert np.array_equal(unit["evaluation_is_entrant"], entrant[evaluation])
    assert not set(fitting) & set(evaluation) and not set(fitting) & set(rows["selection_t"])
    assert not set(population["patient_id"][fitting]) & set(population["patient_id"][evaluation])
    for call in calls:
        tune = {row.tobytes() for row in call["X_tune"]}
        fit_bytes = {row.tobytes() for row in population["X"][fitting].astype(np.float64)}
        held = {row.tobytes() for row in population["X"][np.concatenate([evaluation, rows["selection_t"]])].astype(np.float64)}
        assert tune <= fit_bytes and not tune & held, "A yearly rule was tuned on rows outside the year-t fitting patients"

    E, X, y, p1, k = world["E"], population["X"], population["y"], world["p1"], {"alpha": 0, "epsilon": 4}
    cavs42 = {name: references[ids[name]]["cavs"][42]["cav"] for name in k}
    for index, name in enumerate(("alpha", "epsilon")):
        family, factor = unit["families"][index], k[name]
        expected = E[evaluation, factor] > 1.0
        assert np.array_equal(unit["mask_A"][index], expected), "A_t must be the frozen medoid activation above the frozen threshold"
        assert family["activation_membership_identity"] == _identity_hash(expected) and family["cutoff"] == 1.0
        reference_rule_rows = X[evaluation, factor] > .5
        assert np.array_equal(unit["mask_R"][index], reference_rule_rows)
        assert family["original_rule_transfer"] == rule_activation_agreement(reference_rule_rows, expected)
        assert family["observation"]["activation_records"] == int(expected.sum()) and family["observation"]["records"] == len(evaluation)
        assert family["observation"]["population"] == "heldout_evaluation" and family["observation"]["reference_eligibility_frozen"]
        baseline = family["baselines"]["A"]
        assert baseline["q"] == y[rows["fit_r"]][E[rows["fit_r"], factor] > 1.0].mean() and baseline["identity"] == family["reliability"]["A"]["baseline_identity"]
        assert family["baselines"]["R"]["q"] == y[rows["fit_r"]][X[rows["fit_r"], factor] > .5].mean()
        for subgroup, selected in (("A", expected), ("R", reference_rule_rows)):
            metrics = family["reliability"][subgroup]
            assert metrics["records"] == int(selected.sum()) and np.isclose(metrics["brier"], np.mean((p1[evaluation][selected] - y[evaluation][selected]) ** 2))
            assert metrics["baseline_q"] == family["baselines"][subgroup]["q"] and metrics["bootstrap"]["replicates"] == config["bootstrap_replicates"]
            assert metrics["bootstrap"]["baseline_refitted"] is False
        reference_values = np.maximum(E[rows["eval_r"], factor], 0.)
        assert np.isclose(family["reference_statistics"]["activation_prevalence"], (reference_values > 1.0).mean())
        projected = output["gradients"][np.searchsorted(output["row_index"], evaluation)][expected].astype(np.float64) @ cavs42[name]
        assert np.isclose(family["observation"]["subgroup_influence"]["42"]["tcav"], np.mean(projected > 0))
        assert family["observation"]["subgroup_influence"]["43"]["tcav"] is not None
        for population_name, flag in (("returning", ~entrant[evaluation]), ("entrant", entrant[evaluation])):
            counts = family["populations"][population_name]
            assert counts["rows"] == int(flag.sum()) and counts["deaths"] == int(y[evaluation][flag].sum())
            assert counts["A_rows"] == int((expected & flag).sum()) and counts["R_rows"] == int((reference_rule_rows & flag).sum())
            assert counts["A_deaths"] == int(y[evaluation][expected & flag].sum()) and counts["patients"] <= counts["rows"]
        assert family["populations"]["returning"]["A_rows"] + family["populations"]["entrant"]["A_rows"] == int(expected.sum())
        assert family["yearly_rule"]["status"] == "fitted" and family["yearly_rule"]["fitting_rows"] == len(fitting)
        split = family["yearly_rule"]["inner_split"]
        assert split["fit_rows"] + split["tune_rows"] == len(fitting) and split["stratified_by"] == "frozen_threshold_target"
        assert unit["yearly_rule_available"][index]
        yearly_rows = unit["mask_yearly_rule"][index]
        change = family["rule_change"]
        assert family["yearly_rule_agreement"] == rule_activation_agreement(yearly_rows, expected)
        assert change["evaluation_rows"] == len(evaluation) and change["yearly_rule_rows"] == int(yearly_rows.sum())
    alpha, epsilon = unit["families"]
    assert alpha["original_rule_transfer"]["jaccard"] == 1 and alpha["rule_change"]["features_added"] == []
    assert alpha["rule_change"]["features_removed"] == [] and alpha["rule_change"]["features_kept"] == 1
    assert alpha["rule_change"]["mask_jaccard"] == 1 and alpha["yearly_rule_agreement"]["jaccard"] == 1
    assert epsilon["original_rule_transfer"]["jaccard"] < .5
    assert epsilon["rule_change"]["features_added"] == ["feature_3"] and epsilon["rule_change"]["features_removed"] == ["feature_4"]
    assert epsilon["rule_change"]["mask_jaccard"] < .5 and epsilon["yearly_rule_agreement"]["jaccard"] >= .95
    assert not epsilon["rule_change"]["identical_rule"] and epsilon["yearly_rule"]["rule_text"].startswith("(feature_3 >")

    performance = reference_performance_unit(view, 2007, roles, output, world["family_output"], references, dictionaries, config, "cpu")
    reference_rows = {(row["family_id"], row["idea"]): row for row in performance["rows"]}
    for family in unit["families"]:
        assert family["baselines"]["A"] == reference_rows[(family["family_id"], "idea1")]["baseline"], "Baseline must equal the reference one"
    assert unit["families"][1]["baselines"]["R"] == reference_rows[(ids["epsilon"], "idea2")]["baseline"]

    assert unit["bootstrap"]["replicates"] == config["bootstrap_replicates"] and unit["classification_threshold"] == .5
    assert "continuation" not in " ".join(inspect.signature(idea1_year_unit).parameters) and "gradients" not in inspect.signature(idea1_year_unit).parameters
    source = inspect.getsource(idea1_year_unit)
    assert not any(name in source for name in ("load_continuation", "continuation_from_unit", "batched_death", "replay_continuation"))
    table = idea1_table(unit)
    assert [row["family"] for row in table] == unit["family_ids"] and all(row["t"] == 2008 for row in table)
    assert table[1]["yearly rule"] == "fitted" and table[1]["rule +/- features"] == "+1/-1" and table[0]["eval returning/entrant"].count("/") == 1
    json.dumps(table)

    assert _unit_json(run()) == _unit_json(unit) and _unit_json(run(workers=2)) == _unit_json(unit), "Pool or rerun changed the unit"
    _idea1_equal(run(workers=2), unit)

    keep = {"embeddings": np.concatenate([rows["fit_r"], rows["eval_r"], rows["fit_t"], rows["eval_t"]]),
            "gradients": rows["eval_t"], "features": np.concatenate([rows["fit_r"], rows["fit_t"], rows["eval_t"]]),
            "labels": np.concatenate([rows["fit_r"], rows["eval_t"]])}
    poisoned_view, poisoned_output = _idea1_poison(world, view, output, **keep)
    _idea1_equal(run(poisoned_view, poisoned_output), unit)

    flipped = view["y"].copy()
    flipped[population["year"] == 2008] = 1 - flipped[population["year"] == 2008]
    changed = run(dict(view, y=flipped))
    assert np.array_equal(changed["mask_A"], unit["mask_A"]) and np.array_equal(changed["mask_R"], unit["mask_R"])
    assert np.array_equal(changed["mask_yearly_rule"], unit["mask_yearly_rule"])
    for new, old in zip(changed["families"], unit["families"]):
        for name in ("activation_membership_identity", "yearly_rule", "rule_change", "original_rule_transfer", "yearly_rule_agreement", "baselines"):
            assert new[name] == old[name], f"Year-t labels changed {name}"
    assert changed["families"][0]["reliability"]["A"]["brier"] != unit["families"][0]["reliability"]["A"]["brier"]
    refit = view["y"].copy()
    refit[rows["fit_t"]] = 1 - refit[rows["fit_t"]]
    assert [family["baselines"] for family in run(dict(view, y=refit))["families"]] == [family["baselines"] for family in unit["families"]]
    moved = view["y"].copy()
    moved[rows["fit_r"]] = 1 - moved[rows["fit_r"]]
    assert run(dict(view, y=moved))["families"][0]["baselines"]["A"]["q"] != unit["families"][0]["baselines"]["A"]["q"]

    other_eval = view["X"].copy()
    other_eval[evaluation] = np.random.default_rng(5).normal(size=other_eval[evaluation].shape).astype(np.float32)
    shifted = run(dict(view, X=other_eval))
    assert [family["yearly_rule"] for family in shifted["families"]] == [family["yearly_rule"] for family in unit["families"]]
    assert not np.array_equal(shifted["mask_R"], unit["mask_R"])
    other_fit = view["X"].copy()
    other_fit[fitting] = other_fit[fitting][np.random.default_rng(6).permutation(len(fitting))]
    assert run(dict(view, X=other_fit))["families"][1]["yearly_rule"].get("rule_text") != epsilon["yearly_rule"]["rule_text"]

    scaled = dict(output, gradients=(output["gradients"] * np.float32(-1)))
    assert run(out=scaled)["families"][0]["observation"]["subgroup_influence"]["42"]["tcav"] != alpha["observation"]["subgroup_influence"]["42"]["tcav"]

    for name, mutate in (("target", lambda refs: refs[ids["alpha"]]["target"].update(cutoff=.7)),
                         ("CAV", lambda refs: refs[ids["epsilon"]]["cavs"][42]["cav"].__setitem__(0, 9.))):
        altered = copy.deepcopy(references)
        mutate(altered)
        _refuses(lambda: run(refs=altered), "identity changed", RuntimeError)
    for seed, damage, text in ((42, lambda d: setattr(d["scaler"], "mean_", d["scaler"].mean_ + 1.), "scaler changed"),
                               (135, lambda d: d["run"]["model"].encoder.weight.data.__setitem__((0, 0), 3.), "SAE changed")):
        altered = copy.deepcopy(dictionaries)
        damage(altered[seed])
        _refuses(lambda: run(dicts=altered), text, RuntimeError)
    assert _unit_json(references) == _unit_json(snapshot) and all(
        _reference_object_identity(reference) == reference["identity"] for reference in references.values())

    for refused, keywords in (("follow-up", {"t": 2007}), ("follow-up", {"t": 2006}), ("follow-up", {"t": 2010})):
        _refuses(lambda: run(**keywords), refused)
    _refuses(lambda: idea1_year_unit(view, 2009, 2009, roles, output, world["family_output"], references, dictionaries, config, "cpu"), "follow-up")
    _refuses(lambda: run(rule_view), "reads only")
    _refuses(lambda: run(dict(view, y=view["y"], extra=1)), "reads only")
    _refuses(lambda: idea1_year_unit(view, 2007, 2008, roles, output, world["family_output"], references, dictionaries,
                                     dict(config, profile="pilot"), "cpu", backend=IDEA1_BACKEND), "v0 fixtures only")
    _refuses(lambda: idea1_year_unit(view, 2007, 2008, roles, output, world["family_output"], references, dictionaries, config, "auto"), "resolved device")
    _refuses(lambda: run(v=dict(view, feature_names=["y"] + list(view["feature_names"][1:]))), "reserved")

    absent = run(t=2009, select=counting)
    assert absent["_rows"] == 0 and absent["fitting"]["rows"] == 0 and absent["evaluation"]["rows"] == 0 and absent["jobs"] == 0 and len(calls) == 2
    for family in absent["families"]:
        assert family["yearly_rule"]["status"] == "unavailable" and family["yearly_rule"]["reason"] == IDEA1_UNAVAILABLE_FIT
        assert family["rule_change"] is None and family["yearly_rule_agreement"] is None
        assert family["reliability"]["A"]["brier"] is None and family["reliability"]["A"]["unavailable_reasons"]["brier"] == "empty_subgroup"
        assert family["observation"]["subgroup_influence"]["42"]["unavailable_reason"] == "empty_subgroup"
    assert absent["mask_A"].shape == (2, 0) and not absent["yearly_rule_available"].any()

    thin = dict(output)
    thin_embeddings = output["embeddings"].copy()
    at_t = np.isin(output["row_index"], np.flatnonzero(population["year"] == 2008))
    thin_embeddings[at_t, 4] = 0.
    thin["embeddings"] = thin_embeddings
    before = len(calls)
    sparse = run(out=thin, select=counting)
    assert len(calls) == before + 1 and sparse["jobs"] == 1
    assert sparse["families"][0] == unit["families"][0] or _unit_json(sparse["families"][0]) == _unit_json(unit["families"][0])
    starved = sparse["families"][1]
    assert starved["yearly_rule"]["status"] == "unavailable" and starved["yearly_rule"]["reason"] == IDEA1_UNAVAILABLE_SPLIT
    assert starved["rule_change"] is None and starved["yearly_rule_agreement"] is None and starved["observation"]["activation_records"] == 0
    assert not sparse["mask_A"][1].any() and not sparse["mask_yearly_rule"][1].any() and list(sparse["yearly_rule_available"]) == [True, False]
    assert starved["observation"]["new_rule_unavailable_reason"] == "descriptive_rule_unavailable"
    return unit, output, references


def _check_idea1_cache(config, world, unit, output):
    """D(r)/idea1(t) through the cache: dependencies, resume, cached gradients only, unavailable when a dependency failed."""
    import numpy as np
    population, roles = world["population"], world["roles"]
    view, rule_view = performance_population_view(population), rule_population_view(population)
    dimension, ids = world["dimension"], world["identifiers"]
    base = Path(os.environ["RPI_OUT_DIR"])
    base.mkdir(parents=True, exist_ok=True)
    free = lambda: 100 * 2 ** 30
    rules = {"frozen": [{key: value for key, value in entry.items() if key != "rule"} for entry in world["frozen"].values()],
             "families": [{"family_id": name, "status": "frozen"} for name in world["frozen"]], "_rows": 1}
    forbidden = ("load_continuation", "continuation_from_unit", "batched_death_gradients", "batched_death_probabilities",
                 "rebuild_continuation", "replay_continuation")
    module = globals()
    with tempfile.TemporaryDirectory(dir=base) as directory:
        work = Path(directory)

        def open_cache(name, until="rules"):
            cache = UnitCache(work / name, config, "source-fixture", free_bytes=free)
            assert cache.run("P/2007", lambda: dict(output))["status"] == "ok"
            for seed in config["sae_seeds"]:
                assert cache.run(f"S/{seed}/2007", lambda seed=seed: _identity_sae_payload(seed, dimension))["status"] == "ok"
            if until != "P":
                assert cache.run("D/2007/families", lambda: dict(world["family_output"]))["status"] == "ok"
            if until == "rules":
                assert cache.run("D/2007/rules", lambda: dict(rules))["status"] == "ok"
            return cache

        counted = []

        def counting(candidates, X_tune, y_tune, selection_config, *, threshold_name):
            from semantic_rules import select_rule_set
            counted.append(threshold_name)
            return select_rule_set(candidates, X_tune, y_tune, selection_config, threshold_name=threshold_name)

        cache = open_cache("cache")
        _refuses(lambda: run_idea1_unit(cache, rule_view, roles, config, "cpu", 2008), "reads only")
        _refuses(lambda: run_idea1_unit(cache, view, roles, dict(config, epochs=3), "cpu", 2008), "differs")
        _refuses(lambda: run_idea1_unit(cache, view, roles, config, "cpu", 2007), "follow-up")
        blocked = run_idea1_unit(cache, view, roles, config, "cpu", 2008, backend=IDEA1_BACKEND, select=counting)
        assert (blocked["status"], blocked["reason"]) == ("unavailable", "dep_not_run:D/2007/cavs_selection") and not counted
        assert run_cavs_selection_unit(cache, rule_view, roles, config, "cpu")["status"] == "ok"
        originals = {name: module[name] for name in forbidden}

        def refuse(name):
            def raiser(*arguments, **keywords):
                raise AssertionError(f"{name} was called")
            return raiser

        try:
            module.update({name: refuse(name) for name in forbidden})
            record = run_idea1_unit(cache, view, roles, config, "cpu", 2008, workers=1, backend=IDEA1_BACKEND, select=counting)
        finally:
            module.update(originals)
        assert record["status"] == "ok" and record["key"] == "D/2007/idea1/2008" and record["stage"] == "D.idea1", record
        assert record["dependencies"] == {key: cache._latest[key]["id"] for key in ("P/2007", "D/2007/rules", "D/2007/cavs_selection")}
        assert record["rows"] == unit["_rows"] and len(counted) == 2
        again = run_idea1_unit(cache, view, roles, config, "cpu", 2008, workers=1, backend=IDEA1_BACKEND, select=counting)
        assert again["id"] == record["id"] and len(counted) == 2, "A complete D unit was rebuilt"
        stored = cache.load("D/2007/idea1/2008")
        assert _unit_json(stored) == _unit_json(unit), "Cached unit differs from the direct computation"
        assert np.array_equal(stored["mask_A"], unit["mask_A"]) and stored["mask_A"].dtype == bool
        used = sum(path.stat().st_size for path in cache._unit_path("D/2007/idea1/2008").rglob("*") if path.is_file())
        assert 0 < used <= projected_idea1_bytes(config), (used, projected_idea1_bytes(config))
        empty = run_idea1_unit(cache, view, roles, config, "cpu", 2009, workers=1, backend=IDEA1_BACKEND, select=counting)
        assert empty["status"] == "ok" and empty["rows"] == 0 and len(counted) == 2
        assert cache.completeness_table(["D/2007/idea1/2008", "D/2007/idea1/2009", "D/2008/idea1/2009"]) == [
            {"unit": "D/2007/idea1/2008", "status": "complete"}, {"unit": "D/2007/idea1/2009", "status": "complete"},
            {"unit": "D/2008/idea1/2009", "status": "not_run (stopped)"}]

        crashed = open_cache("cache-crash")

        def broken_fit(*arguments):
            raise FloatingPointError("injected")

        assert run_cavs_selection_unit(crashed, rule_view, roles, config, "cpu", fit=broken_fit)["status"] == "failed"
        before = len(counted)
        record = run_idea1_unit(crashed, view, roles, config, "cpu", 2008, backend=IDEA1_BACKEND, select=counting)
        assert (record["status"], record["reason"]) == ("unavailable", "dep_failed:D/2007/cavs_selection") and len(counted) == before
        assert not crashed.is_complete("D/2007/idea1/2008") and not crashed._unit_path("D/2007/idea1/2008").exists()
        assert crashed.completeness_table(["D/2007/idea1/2008"]) == [
            {"unit": "D/2007/idea1/2008", "status": "unavailable:dep_failed:D/2007/cavs_selection"}]

        lost = open_cache("cache-rules", until="families")
        assert lost.run("D/2007/rules", lambda: (_ for _ in ()).throw(FloatingPointError("injected")))["status"] == "failed"
        record = run_idea1_unit(lost, view, roles, config, "cpu", 2008, backend=IDEA1_BACKEND, select=counting)
        assert (record["status"], record["reason"]) == ("unavailable", "dep_failed:D/2007/rules") and len(counted) == before

        failing = open_cache("cache-select")
        assert run_cavs_selection_unit(failing, rule_view, roles, config, "cpu")["status"] == "ok"

        def crashing(candidates, X_tune, y_tune, selection_config, *, threshold_name):
            raise FloatingPointError("injected")

        record = run_idea1_unit(failing, view, roles, config, "cpu", 2008, workers=1, backend=IDEA1_BACKEND, select=crashing)
        assert record["status"] == "failed" and record["reason"] == "FloatingPointError: injected"
        assert not failing._unit_path("D/2007/idea1/2008").exists()
        assert run_idea1_unit(failing, view, roles, config, "cpu", 2008, workers=1, backend=IDEA1_BACKEND)["status"] == "ok"


def run_idea1_checks(notebook_path="concept_subgroup_drift_pilot.ipynb"):
    """Idea 1 on planted follow-up years: frozen A_t, original-rule transfer, unfiltered yearly rules, held-out reliability."""
    presets = _notebook_presets(json.loads(Path(notebook_path).read_text()))
    config = resolve_config(presets, "v0")
    assert config["profile"] == "v0" and config["bootstrap_replicates"] == 10 and config["joblib_workers"] <= 4
    config = _selection_config(config)
    _check_idea1_settings(config)
    world = _idea1_world()
    unit, output, _ = _check_idea1_unit(config, world)
    _check_idea1_cache(config, world, unit, output)
    print("Idea 1 fixtures passed: frozen A_t, rule transfer, unfiltered yearly rules on fitting patients, held-out reliability, cached gradients")


def _check_idea2_settings(presets, config):
    """Graph dependencies of the three Idea 2 units, follow-up years, stage names, shared view keys, chance permutations."""
    import inspect
    pilot = {"reference_year_candidates": list(range(2007, 2016)), "t0_only_reference_years": [2015], "sae_seeds": [42, 135, 215]}
    assert idea2_follow_up_years(pilot, 2007) == list(range(2008, 2016)) and idea2_follow_up_years(pilot, 2014) == [2015]
    assert idea2_follow_up_years(pilot, 2015) == [] and idea2_follow_up_years(config, 2007) == [2008, 2009]
    assert idea2_follow_up_years(config, 2008) == [2009] and idea2_follow_up_years(config, 2009) == []
    graph = unit_graph(pilot)
    assert graph["D/2007/idea2_frozen/2010"] == ("P/2007", "D/2007/families", "D/2007/cavs_selection")
    for seed in pilot["sae_seeds"]:
        assert graph[f"D/2007/idea2_yearly_sae/{seed}/2010"] == ("P/2007", f"S/{seed}/2007")
    assert graph["D/2007/idea2_yearly_families/2010"] == ("P/2007", "D/2007/families", *(
        f"D/2007/idea2_yearly_sae/{seed}/2010" for seed in pilot["sae_seeds"]))
    for stage in ("idea2_frozen", "idea2_yearly_sae", "idea2_yearly_families"):
        assert [key for key in graph if key.startswith(f"D/2007/{stage}/")] and stage in "".join(UNIT_STAGES)
        assert not [key for key in graph if key.startswith(f"D/2015/{stage}") or key.startswith(f"D/2007/{stage}/2007")]
    assert unit_stage("D/2007/idea2_yearly_sae/42/2010") == "D.idea2_yearly_sae"
    assert unit_stage("D/2007/idea2_frozen/2010") == "D.idea2_frozen" and unit_stage("D/2007/idea2_yearly_families/2010") == "D.idea2_yearly_families"
    assert set(STAGE_CODE_VERSIONS) <= set(UNIT_STAGES)
    assert (IDEA2_YEARLY_VIEW_KEYS, IDEA2_FROZEN_VIEW_KEYS) == (MATCHING_VIEW_KEYS, RULE_VIEW_KEYS)
    assert not {"y", "post_death", "gradients", "probabilities"} & (set(IDEA2_YEARLY_VIEW_KEYS) | set(IDEA2_FROZEN_VIEW_KEYS))
    for function in (idea2_frozen_year_unit, idea2_yearly_sae_unit, idea2_yearly_families_unit, link_reference_to_yearly,
                     run_idea2_frozen_unit, run_idea2_yearly_sae_unit, run_idea2_yearly_families_unit):
        assert not set(inspect.signature(function).parameters) & {"y", "outcome", "outcomes", "labels", "population"}, function
    assert resolve_config(presets, "pilot")["chance_permutations"] == 20 and config["chance_permutations"] == 5
    assert IDEA2_LINK_STATES == ("retained", "not_mutual_full", "partial", "chain", "inconsistent", "unmatched")
    labelled = idea2_reuse_record({"families": [{"id": "a", "kind": "full"}, {"id": "b", "kind": "partial"}]}, 2007)
    assert labelled["mode"] == "reference_reuse" and labelled["year"] == labelled["reference_year"] == 2007
    assert "reuse" in labelled["label"] and [(link["family_id"], link["state"], link["retained"]) for link in labelled["links"]] == [("a", "reused", True)]
    assert all(link["similarity"] is None for link in labelled["links"])


def _idea2_planted(rows=128):
    """Planted reference-year families, their year-t encoding (family E inactive) and yearly encodings over identical rows:
    A kept, B only partial, C drifted, D claimed by a yearly family that prefers A, E inactive, one yearly-only family."""
    seeds = [42, 135, 215]
    noise = lambda index: 20 + index
    weight = lambda seed, signal: (.1, .2, .3)[seed] * (1 + .1 * signal)
    order = [[2, 0, 3, 1, 4], [3, 1, 0, 4, 2], [1, 3, 4, 2, 0]]
    reference_year = [_planted_activations(rows, [{signal + 1: 1.0, noise(10 * s + signal): weight(s, signal)} for signal in order[s]])
                      for s in range(3)]
    families = build_factor_families(reference_year, seeds, device="cpu", id_prefix="2007/42")
    assert families["counts"] == {"full": 5, "partial": 0, "chain": 0, "inconsistent": 0, "singleton": 0}
    at_t = [_planted_activations(rows, [None if signal == 4 else {signal + 1: 1.0, noise(10 * s + signal): weight(s, signal)}
                                        for signal in order[s]]) for s in range(3)]
    yearly = []
    for s in range(3):
        columns = [{1: 1.0, noise(30 + s): .1 + .05 * s}, {3: 1.0, 7: 1.0, noise(33 + s): .1}, {1: 1.0, 4: .5, noise(36 + s): .1},
                   {9: 1.0, noise(39 + s): .1}]
        columns.append({2: 1.0, noise(42 + s): .1} if s < 2 else {12: 1.0, noise(45): .1})
        yearly.append(_planted_activations(rows, columns))
    built = build_factor_families(yearly, seeds, device="cpu", id_prefix="2007/42/2008", permutations=5, rng_seed=[42, 2007, 2008])
    assert built["counts"] == {"full": 4, "partial": 1, "chain": 0, "inconsistent": 0, "singleton": 1}, built["counts"]
    signal_of = {}
    for family in families["families"]:
        seed, factor = family["members"][0]
        signal_of[family["id"]] = order[seeds.index(seed)][factor] + 1
    return {"seeds": seeds, "rows": rows, "reference": families, "at_t": at_t, "yearly": yearly, "built": built, "signal_of": signal_of}


def _mean_pairwise(reference_matrices, yearly_matrices, seeds, left, right):
    """Independent oracle: mean np.corrcoef over every left-member by right-member pair."""
    import numpy as np
    values = []
    for a_seed, a_factor in left["members"]:
        for b_seed, b_factor in right["members"]:
            a = reference_matrices[seeds.index(a_seed)][:, a_factor].astype(np.float64)
            b = yearly_matrices[seeds.index(b_seed)][:, b_factor].astype(np.float64)
            if a.max() == a.min() or b.max() == b.min():
                return np.nan
            values.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(values))


def _oracle_permuted_links(reference_matrices, yearly_matrices, seeds, reference_families, yearly_families, rng_seed, permutations):
    """Independent chance oracle: the documented scheme with np.corrcoef (every yearly seed's rows permuted independently,
    draws in permutation-major, seed-minor order) and a naive mutual-nearest-neighbour search."""
    import numpy as np
    generator = np.random.default_rng(rng_seed)
    rows = reference_matrices[0].shape[0]
    full_reference = sorted((f for f in reference_families if f["kind"] == "full"), key=lambda f: f["id"])
    full_yearly = sorted((f for f in yearly_families if f["kind"] == "full"), key=lambda f: f["id"])
    table = []
    for _ in range(permutations):
        shuffled = [matrix[generator.permutation(rows)] for matrix in yearly_matrices]
        sim = np.asarray([[_mean_pairwise(reference_matrices, shuffled, seeds, f, g) for g in full_yearly] for f in full_reference])
        valid = ~np.isnan(sim).any(axis=1)
        sub = sim[valid]
        forward, backward = sub.argmax(axis=1), sub.argmax(axis=0)
        mutual = [at for at in range(len(sub)) if backward[forward[at]] == at]
        table.append({"retained": len(mutual), "mean_link_similarity": float(np.mean([sub[at, forward[at]] for at in mutual])) if mutual else None})
    return table


def _same_chance_rows(left, right):
    """Equal retained counts; link similarities within float64 matmul rounding (CPU against CUDA)."""
    assert [row["retained"] for row in left] == [row["retained"] for row in right]
    for a, b in zip(left, right):
        assert (a["mean_link_similarity"] is None) == (b["mean_link_similarity"] is None)
        assert a["mean_link_similarity"] is None or abs(a["mean_link_similarity"] - b["mean_link_similarity"]) < 1e-9


def _check_idea2_links(device):
    """Family-level similarity and mutual-nearest-neighbour links on planted structures; returns results for comparison."""
    import numpy as np
    world = _idea2_planted()
    seeds, reference, built, signal_of = world["seeds"], world["reference"], world["built"], world["signal_of"]
    result = link_reference_to_yearly(world["at_t"], world["yearly"], seeds, reference["families"], built["families"], device=device,
                                      permutations=20, rng_seed=[42, 2007, 2008, 1])
    assert result["device"] == device and result["precision"] == "float64" and result["seeds"] == seeds and result["rows"] == 128
    by_signal = {signal_of[link["family_id"]]: link for link in result["links"]}
    yearly = {family["id"]: family for family in built["families"]}
    ids = result["reference_ids"]
    assert ids == sorted(ids) and result["yearly_ids"] == sorted(yearly) and result["similarity"].shape == (5, 5)
    reference_by_id = {family["id"]: family for family in reference["families"]}
    for row, family_id in enumerate(ids):
        for column, yearly_id in enumerate(result["yearly_ids"]):
            expected = _mean_pairwise(world["at_t"], world["yearly"], seeds, reference_by_id[family_id], yearly[yearly_id])
            got = result["similarity"][row, column]
            assert (np.isnan(expected) and np.isnan(got)) or abs(expected - got) < 1e-9, (family_id, yearly_id, expected, got)
    assert np.isnan(result["similarity"][ids.index(next(i for i in ids if signal_of[i] == 5))]).all(), "Inactive family has no similarity"

    def full_columns():
        return [i for i, kind in enumerate(result["yearly_kinds"]) if kind == "full"]

    a, b, c, d, e = (by_signal[signal] for signal in (1, 2, 3, 4, 5))
    for link in (a, c):
        assert link["retained"] and link["state"] == "retained" and link["matched_family"] == link["best_full_family"]
        assert link["similarity"] == link["best_full_similarity"] and not link["tie"] and link["reason"] is None
    assert a["matched_family"] != c["matched_family"] and a["similarity"] > .95 and .6 < c["similarity"] < .75, "A kept; C drifted yet retained"
    assert not d["retained"] and d["state"] == "not_mutual_full" and d["matched_family"] is None
    assert d["best_full_family"] not in (a["matched_family"], c["matched_family"]) and .35 < d["similarity"] < .5
    assert not b["retained"] and b["state"] == "partial" and b["matched_family"] is None and b["similarity"] > .85
    assert yearly[b["nearest"]["family_id"]]["kind"] == "partial" and b["nearest"]["kind"] == "partial"
    assert (e["retained"], e["state"], e["valid"], e["reason"], e["similarity"], e["runner_up_gap"]) == (False, "unmatched", False, "reference_member_inactive", None, None)
    assert len(e["inactive_members"]) == 3 and e["matched_family"] is None
    for link in (a, c, d):
        row = result["similarity"][ids.index(link["family_id"])]
        top = sorted((row[i] for i in full_columns()), reverse=True)
        assert abs(link["runner_up_gap"] - (top[0] - top[1])) < 1e-12 and link["runner_up_gap"] > 0
    assert result["state_counts"] == {"retained": 2, "not_mutual_full": 1, "partial": 1, "chain": 0, "inconsistent": 0, "unmatched": 1}
    assert result["retained"] == 2 and len(result["links"]) == 5
    assert {record["family_id"]: record["yearly_only"] for record in result["yearly_links"]}[d["best_full_family"]] is True
    assert sum(not record["yearly_only"] for record in result["yearly_links"]) == 2 and len(result["yearly_links"]) == 4
    assert {record["linked_reference_family"] for record in result["yearly_links"]} - {None} == {a["family_id"], c["family_id"]}
    chance = result["chance"]
    assert chance["permutations"] == 20 == len(chance["per_permutation"]) and chance["observed"]["retained"] == 2
    assert chance["rng_seed"] == [42, 2007, 2008, 1] and "independently permuted" in chance["scheme"]
    assert abs(chance["mean"]["retained"] - np.mean([row["retained"] for row in chance["per_permutation"]])) < 1e-12
    assert chance["max"]["retained"] == max(row["retained"] for row in chance["per_permutation"])
    assert chance["mean"]["mean_link_similarity"] < chance["observed"]["mean_link_similarity"] - .3, "Planted links must beat permuted links"
    oracle = _oracle_permuted_links(world["at_t"], world["yearly"], seeds, reference["families"], built["families"], [42, 2007, 2008, 1], 20)
    assert [row["retained"] for row in oracle] == [row["retained"] for row in chance["per_permutation"]], "Chance scheme differs from the documented one"
    for expected, got in zip(oracle, chance["per_permutation"]):
        assert (expected["mean_link_similarity"] is None) == (got["mean_link_similarity"] is None)
        assert expected["mean_link_similarity"] is None or abs(expected["mean_link_similarity"] - got["mean_link_similarity"]) < 1e-9
    assert len({row["retained"] for row in oracle}) > 1, "The permutations must vary"
    again = link_reference_to_yearly(world["at_t"], world["yearly"], seeds, reference["families"], built["families"], device=device,
                                     permutations=20, rng_seed=[42, 2007, 2008, 1])
    assert again["chance"] == chance and _unit_json(again["links"]) == _unit_json(result["links"]), "The chance check must be reproducible"
    different = link_reference_to_yearly(world["at_t"], world["yearly"], seeds, reference["families"], built["families"], device=device,
                                         permutations=20, rng_seed=[1, 2, 3])
    assert different["chance"]["per_permutation"] != chance["per_permutation"] and different["links"] == result["links"], \
        "The seed changes the permutations only, never the observed links"
    assert link_reference_to_yearly(world["at_t"], world["yearly"], seeds, reference["families"], built["families"], device=device)["chance"] is None

    # No yearly full family: a split/merge yearly family is inconsistent, never forced onto a reference family.
    rows = 128
    reference_two = [_planted_activations(rows, [{1: 1.0, 20 + s: .1}, {2: 1.0, 30 + s: .1}]) for s in range(3)]
    yearly_two = [_planted_activations(rows, columns) for columns in
                  ([{1: 1.0, 40: .1}, {2: 1.0, 41: .1}], [{1: 1.1, 2: 1.0, 42: .1}, None], [{2: 1.0, 43: .1}, None])]
    families_two = build_factor_families(reference_two, seeds, device=device, id_prefix="2007/42")
    yearly_families = build_factor_families(yearly_two, seeds, device=device, id_prefix="2007/42/2008")
    assert yearly_families["counts"]["inconsistent"] == 1 and yearly_families["counts"]["full"] == 0
    split = link_reference_to_yearly(reference_two, yearly_two, seeds, families_two["families"], yearly_families["families"],
                                     device=device, permutations=3, rng_seed=0)
    inconsistent, = yearly_families["families"]
    assert [(link["state"], link["retained"], link["reason"], link["matched_family"], link["runner_up_gap"], link["best_full_family"])
            for link in split["links"]] == [("inconsistent", False, "no_yearly_full_family", None, None, None)] * 2
    for link in split["links"]:
        family = next(f for f in families_two["families"] if f["id"] == link["family_id"])
        assert link["nearest"]["family_id"] == inconsistent["id"]
        assert abs(link["similarity"] - _mean_pairwise(reference_two, yearly_two, seeds, family, inconsistent)) < 1e-9
    assert split["yearly_links"] == [] and split["chance"]["max"]["retained"] == 0 and split["state_counts"]["inconsistent"] == 2

    # Exact ties between yearly families go to the lowest family ID and are flagged; the other family stays yearly-only.
    twin = [_planted_activations(rows, [{1: 1.0, 40 + s: .1}] * 2) for s in range(3)]
    for s in range(3):
        twin[s][:, 1] = twin[s][:, 0]
    family = {"id": "ref/a", "kind": "full", "members": [[seed, 0] for seed in seeds], "medoid": [42, 0]}
    first = {"id": "y/a", "kind": "full", "members": [[seed, 0] for seed in seeds], "medoid": [42, 0]}
    second = {"id": "y/b", "kind": "full", "members": [[seed, 1] for seed in seeds], "medoid": [42, 1]}
    tied = link_reference_to_yearly(twin, twin, seeds, [family], [second, first], device=device)
    link, = tied["links"]
    assert tied["yearly_ids"] == ["y/a", "y/b"] and link["retained"] and link["matched_family"] == "y/a" and link["tie"]
    assert abs(link["runner_up_gap"]) < TIE_TOLERANCE and link["runner_up_family"] == "y/b"
    assert {record["family_id"]: record["yearly_only"] for record in tied["yearly_links"]} == {"y/a": False, "y/b": True}

    # Empty sides, an unusable yearly family and invalid inputs.
    nothing = link_reference_to_yearly(world["at_t"], world["yearly"], seeds, [], built["families"], device=device, permutations=2)
    assert nothing["links"] == [] and nothing["similarity"].shape == (0, 5) and nothing["chance"]["max"]["retained"] == 0
    bare = link_reference_to_yearly(world["at_t"], world["yearly"], seeds, reference["families"], [], device=device, permutations=2)
    assert {link["state"] for link in bare["links"] if link["valid"]} == {"unmatched"} and bare["retained"] == 0
    assert {link["reason"] for link in bare["links"] if link["valid"]} == {"no_yearly_full_family"} and bare["similarity"].shape == (5, 0)
    good = (world["at_t"], world["yearly"], seeds, reference["families"], built["families"])
    _refuses(lambda: link_reference_to_yearly(*good, device="auto"), "resolved device")
    _refuses(lambda: link_reference_to_yearly(*good, device=device, permutations=-1), "nonnegative")
    _refuses(lambda: link_reference_to_yearly(world["at_t"][:2], world["yearly"], seeds, *good[3:], device=device), "one floating-point matrix per seed")
    _refuses(lambda: link_reference_to_yearly([m[:64] for m in world["at_t"]], world["yearly"], seeds, *good[3:], device=device), "identical ordered rows")
    _refuses(lambda: link_reference_to_yearly([m.astype(int) for m in world["at_t"]], world["yearly"], seeds, *good[3:], device=device), "floating-point")
    broken = [dict(family, members=[[42, 99]]) for family in reference["families"]]
    _refuses(lambda: link_reference_to_yearly(world["at_t"], world["yearly"], seeds, broken, built["families"], device=device), "existing latent column")
    return {"links": result["links"], "similarity": result["similarity"], "chance": chance, "tied": tied["links"], "split": split["links"]}


def _check_idea2_pairing():
    """Reference-family and yearly encodings must share the same ordered row IDs, seeds and dimensions."""
    import numpy as np
    rows = np.arange(10, 30)
    matrices = [np.zeros((20, 6), dtype=np.float32) for _ in range(3)]
    reference = {"row_index": rows, "seeds": [42, 135, 215], "matrices": matrices}
    yearly = {"row_index": rows.copy(), "seeds": [42, 135, 215], "matrices": [m.copy() for m in matrices]}
    paired = idea2_pair_encodings(reference, yearly)
    assert paired == {"rows": 20, "latents": 6, "row_identity": _identity_hash({"row_index": rows})}
    _refuses(lambda: idea2_pair_encodings(reference, dict(yearly, row_index=rows[::-1].copy())), "identical ordered row IDs")
    _refuses(lambda: idea2_pair_encodings(reference, dict(yearly, row_index=rows + 1)), "identical ordered row IDs")
    _refuses(lambda: idea2_pair_encodings(reference, dict(yearly, row_index=rows[:-1])), "identical ordered row IDs")
    _refuses(lambda: idea2_pair_encodings(dict(reference, row_index=np.repeat(rows[:1], 20)), dict(yearly, row_index=np.repeat(rows[:1], 20))), "distinct row IDs")
    _refuses(lambda: idea2_pair_encodings(reference, dict(yearly, seeds=[42, 215, 135])), "same seeds")
    _refuses(lambda: idea2_pair_encodings(reference, dict(yearly, matrices=[m[:, :5] for m in matrices])), "identical dimensions")
    _refuses(lambda: idea2_pair_encodings(reference, dict(yearly, matrices=matrices[:2])), "identical dimensions")
    _refuses(lambda: idea2_pair_encodings(reference, dict(yearly, matrices=[m[:19] for m in matrices])), "identical dimensions")


def _idea2_world():
    """The Idea 1 world with a full family (three members) per fixture medoid; member channels differ so that every member
    has its own prevalence and magnitude. Identity SAEs: factor k < 8 is relu(embedding channel k), the rest are dead."""
    world = _idea1_world()
    members = {"alpha": [[42, 0], [135, 5], [215, 6]], "beta": [[42, 5], [135, 1], [215, 6]], "gamma": [[42, 2], [135, 2], [215, 2]],
               "delta": [[42, 3], [135, 3], [215, 3]], "epsilon": [[42, 6], [135, 4], [215, 7]]}
    families = [{"id": world["identifiers"][name], "kind": "full", "members": members[name], "size": 3, "links": 3,
                 "min_correlation": .5, "median_correlation": .6, "medoid": list(medoid), "medoid_mean_correlation": .7}
                for name, medoid in FIXTURE_FAMILIES]
    for family, (_, medoid) in zip(families, FIXTURE_FAMILIES):
        assert family["medoid"] in family["members"]
    return dict(world, family_output=dict(world["family_output"], families=families), members=members)


def _check_idea2_frozen(config, world):
    """Medoid retention and per-member prevalence/magnitude inside the frozen rule on held-out year-t rows, reference-year
    comparison, no outcomes, no rows outside the reference and year-t evaluation patients."""
    import copy
    import numpy as np
    population, roles = world["population"], world["roles"]
    rule_view, performance_view = rule_population_view(population), performance_population_view(population)
    dimension, ids, rows = world["dimension"], world["identifiers"], _idea1_rows(world)
    dictionaries = _identity_dictionaries(dimension, [42, 135, 215])
    probe = cavs_selection_unit(rule_view, 2007, roles, world["output"], world["family_output"], world["frozen"], dictionaries, config, "cpu")
    output = _with_gradients(world, probe["cav_vectors"])
    selection = cavs_selection_unit(rule_view, 2007, roles, output, world["family_output"], world["frozen"], dictionaries, config, "cpu")
    assert selection["eligible"]["idea2"] == [ids["beta"], ids["epsilon"]]
    references = _references_from_units(selection, world["frozen"])
    snapshot = copy.deepcopy(references)
    family_output = world["family_output"]

    def run(view=rule_view, out=output, refs=references, dicts=dictionaries, t=2008, device="cpu", families=family_output):
        return idea2_frozen_year_unit(view, 2007, t, roles, out, families, refs, dicts, config, device)

    unit = run()
    evaluation, reference_rows = rows["eval_t"], rows["eval_r"]
    assert unit["family_ids"] == [ids["beta"], ids["epsilon"]] and unit["outcomes_read"] is False
    assert {item["family_id"] for item in unit["not_computed"]} == {ids["alpha"], ids["gamma"], ids["delta"]}
    assert {item["reason"] for item in unit["not_computed"]} == {"not_eligible_at_selection"}
    assert np.array_equal(unit["evaluation_row_index"], evaluation) and np.array_equal(unit["reference_row_index"], reference_rows)
    assert unit["_rows"] == len(evaluation) and unit["reference_year"] == 2007 and unit["year"] == 2008
    entrant = roles["is_entrant"]
    assert np.array_equal(unit["evaluation_is_entrant"], entrant[evaluation]) and entrant[evaluation].any() and (~entrant[evaluation]).any()
    assert unit["evaluation_rows_identity"] == _identity_hash({"row_index": evaluation})
    assert not set(evaluation) & set(rows["fit_t"]) and not set(evaluation) & set(rows["selection_t"])
    assert unit["scaler_identity"] == dictionaries[42]["scaler_identity"]
    E, X = world["E"], population["X"]
    medoid_differs = 0
    for position, name in enumerate(("beta", "epsilon")):
        family, feature = unit["families"][position], {"beta": 1, "epsilon": 4}[name]
        in_t, in_r = X[evaluation, feature] > .5, X[reference_rows, feature] > .5
        assert family["status"] == "ok" and in_t.any() and in_r.any()
        assert np.array_equal(unit["mask_R"][position], in_t) and np.array_equal(unit["mask_R_reference"][position], in_r)
        assert family["rule_rows"] == {"year": int(in_t.sum()), "reference": int(in_r.sum())} and family["rule_mask_identity"] == _identity_hash(in_t)
        for flag_name, flag in (("returning", ~entrant[evaluation]), ("entrant", entrant[evaluation])):
            counts = family["populations"][flag_name]
            assert counts["rows"] == int(flag.sum()) and counts["R_rows"] == int((in_t & flag).sum()) and counts["R_patients"] <= counts["R_rows"]
        assert [(m["seed"], m["factor"]) for m in family["members"]] == [tuple(m) for m in world["members"][name]]
        assert sum(m["is_medoid"] for m in family["members"]) == 1
        medoid_member = [m for m in family["members"] if m["is_medoid"]][0]
        assert [medoid_member["seed"], medoid_member["factor"]] == family["medoid"] == {"beta": [135, 1], "epsilon": [135, 4]}[name]
        for member in family["members"]:
            values_t, values_r = np.maximum(E[evaluation, member["factor"]], 0.)[in_t], np.maximum(E[reference_rows, member["factor"]], 0.)[in_r]
            for key, values in (("year", values_t), ("reference", values_r)):
                stats = member[key]
                assert stats["rows"] == len(values) and stats["positive_rows"] == int((values > 0).sum()) and stats["unavailable"] is None
                assert np.isclose(stats["positive_prevalence"], (values > 0).mean()) and np.isclose(stats["magnitude_mean"], values.mean())
                if (values > 0).any():
                    assert np.isclose(stats["positive_magnitude_mean"], values[values > 0].mean())
                else:
                    assert stats["positive_magnitude_mean"] is None
            for key, now, then in (("prevalence_ratio", (values_t > 0).mean(), (values_r > 0).mean()), ("magnitude_ratio", values_t.mean(), values_r.mean())):
                assert (member[key] is None) if then == 0 else np.isclose(member[key], now / then), key
            assert ("retention" in member) == member["is_medoid"]
            if member["is_medoid"]:
                cutoff = 1.0
                above_t, above_r = np.maximum(E[evaluation, member["factor"]], 0.)[in_t] > cutoff, np.maximum(E[reference_rows, member["factor"]], 0.)[in_r] > cutoff
                retention = member["retention"]
                assert retention["cutoff"] == cutoff and retention["year_rows_above"] == int(above_t.sum()) and retention["reference_rows_above"] == int(above_r.sum())
                assert np.isclose(retention["year_rate"], above_t.mean()) and np.isclose(retention["reference_rate"], above_r.mean())
                assert (retention["ratio"] is None) if above_r.mean() == 0 else np.isclose(retention["ratio"], above_t.mean() / above_r.mean())
            else:
                medoid_differs += abs(member["year"]["positive_prevalence"] - medoid_member["year"]["positive_prevalence"]) > 1e-9 \
                    or abs(member["year"]["magnitude_mean"] - medoid_member["year"]["magnitude_mean"]) > 1e-9
    assert medoid_differs >= 2, "Non-medoid members must be measured on their own activations"
    table = idea2_frozen_table(unit)
    assert len(table) == 6 and sum(row["medoid"] for row in table) == 2 and {row["family"] for row in table} == {ids["beta"], ids["epsilon"]}
    assert all(row["retention"] is None for row in table if not row["medoid"])

    allowed = np.concatenate([reference_rows, evaluation])
    view, poisoned = _poisoned_world(world, allowed, [], allowed)
    twin = run(view=view, out=poisoned)
    _unit_arrays_equal(unit, twin)
    assert not np.isfinite(poisoned["embeddings"][~np.isin(poisoned["row_index"], allowed)]).any(), "The poison must cover every other row"
    _refuses(lambda: run(view=performance_view), "outcomes are not accepted")
    _refuses(lambda: run(view=dict(rule_view, post_death=population["y"])), "outcomes are not accepted")
    _refuses(lambda: run(t=2007), "follow-up")
    _refuses(lambda: run(t=2010), "follow-up")
    _refuses(lambda: run(device="auto"), "resolved device")
    _refuses(lambda: run(dicts=dict(reversed(list(dictionaries.items())))), "exactly the configured")
    broken = dict(family_output, families=[dict(family, kind="partial") if family["id"] == ids["beta"] else family
                                           for family in family_output["families"]])
    _refuses(lambda: run(families=broken), "not a full family")
    empty = run(t=2009)
    assert empty["_rows"] == 0 and [(f["status"], f["reason"], f["members"]) for f in empty["families"]] == [("unavailable", IDEA2_UNAVAILABLE_EMPTY_YEAR, [])] * 2
    assert idea2_frozen_table(empty)[0]["status"] == f"unavailable:{IDEA2_UNAVAILABLE_EMPTY_YEAR}"
    assert references.keys() == snapshot.keys() and all(references[key]["identity"] == snapshot[key]["identity"] for key in references)
    return unit, output


def _check_idea2_frozen_cache(config, world, unit, output):
    """D(r)/idea2_frozen(t) through the cache: dependencies, resume, size, cached embeddings only, failure propagation."""
    population, roles = world["population"], world["roles"]
    view = rule_view = rule_population_view(population)
    dimension = world["dimension"]
    base = Path(os.environ["RPI_OUT_DIR"])
    base.mkdir(parents=True, exist_ok=True)
    free = lambda: 100 * 2 ** 30
    rules = {"frozen": [{key: value for key, value in entry.items() if key != "rule"} for entry in world["frozen"].values()],
             "families": [{"family_id": name, "status": "frozen"} for name in world["frozen"]], "_rows": 1}
    forbidden = ("load_continuation", "continuation_from_unit", "batched_death_gradients", "batched_death_probabilities",
                 "rebuild_continuation", "replay_continuation")
    module = globals()
    with tempfile.TemporaryDirectory(dir=base) as directory:
        work = Path(directory)

        def open_cache(name, until="rules"):
            cache = UnitCache(work / name, config, "source-fixture", free_bytes=free)
            assert cache.run("P/2007", lambda: dict(output))["status"] == "ok"
            for seed in config["sae_seeds"]:
                assert cache.run(f"S/{seed}/2007", lambda seed=seed: _identity_sae_payload(seed, dimension))["status"] == "ok"
            if until != "P":
                assert cache.run("D/2007/families", lambda: dict(world["family_output"]))["status"] == "ok"
            if until == "rules":
                assert cache.run("D/2007/rules", lambda: dict(rules))["status"] == "ok"
            return cache

        cache = open_cache("cache")
        _refuses(lambda: run_idea2_frozen_unit(cache, performance_population_view(population), roles, config, "cpu", 2008), "outcomes are not accepted")
        _refuses(lambda: run_idea2_frozen_unit(cache, view, roles, dict(config, epochs=3), "cpu", 2008), "differs")
        _refuses(lambda: run_idea2_frozen_unit(cache, view, roles, config, "cpu", 2007), "follow-up")
        blocked = run_idea2_frozen_unit(cache, view, roles, config, "cpu", 2008)
        assert (blocked["status"], blocked["reason"]) == ("unavailable", "dep_not_run:D/2007/cavs_selection")
        assert run_cavs_selection_unit(cache, rule_view, roles, config, "cpu")["status"] == "ok"
        originals = {name: module[name] for name in forbidden}

        def refuse(name):
            def raiser(*arguments, **keywords):
                raise AssertionError(f"{name} was called")
            return raiser

        try:
            module.update({name: refuse(name) for name in forbidden})
            record = run_idea2_frozen_unit(cache, view, roles, config, "cpu", 2008)
        finally:
            module.update(originals)
        assert record["status"] == "ok" and record["key"] == "D/2007/idea2_frozen/2008" and record["stage"] == "D.idea2_frozen", record
        assert record["dependencies"] == {key: cache._latest[key]["id"] for key in ("P/2007", "D/2007/families", "D/2007/cavs_selection")}
        assert record["rows"] == unit["_rows"]
        again = run_idea2_frozen_unit(cache, view, roles, config, "cpu", 2008)
        assert again["id"] == record["id"], "A complete D unit was rebuilt"
        stored = cache.load("D/2007/idea2_frozen/2008")
        _unit_arrays_equal(stored, unit)
        assert _unit_json(stored) == _unit_json(unit) and stored["mask_R"].dtype == bool
        used = sum(path.stat().st_size for path in cache._unit_path("D/2007/idea2_frozen/2008").rglob("*") if path.is_file())
        assert 0 < used <= projected_idea2_frozen_bytes(config), (used, projected_idea2_frozen_bytes(config))
        empty = run_idea2_frozen_unit(cache, view, roles, config, "cpu", 2009)
        assert empty["status"] == "ok" and empty["rows"] == 0
        assert cache.completeness_table(["D/2007/idea2_frozen/2008", "D/2007/idea2_frozen/2009", "D/2008/idea2_frozen/2009"]) == [
            {"unit": "D/2007/idea2_frozen/2008", "status": "complete"}, {"unit": "D/2007/idea2_frozen/2009", "status": "complete"},
            {"unit": "D/2008/idea2_frozen/2009", "status": "not_run (stopped)"}]

        crashed = open_cache("cache-crash")

        def broken_fit(*arguments):
            raise FloatingPointError("injected")

        assert run_cavs_selection_unit(crashed, rule_view, roles, config, "cpu", fit=broken_fit)["status"] == "failed"
        record = run_idea2_frozen_unit(crashed, view, roles, config, "cpu", 2008)
        assert (record["status"], record["reason"]) == ("unavailable", "dep_failed:D/2007/cavs_selection")
        assert not crashed._unit_path("D/2007/idea2_frozen/2008").exists()
        assert crashed.completeness_table(["D/2007/idea2_frozen/2008"]) == [
            {"unit": "D/2007/idea2_frozen/2008", "status": "unavailable:dep_failed:D/2007/cavs_selection"}]


def _idea2_yearly_world(config):
    """Tiny structured population, cached P(r) outputs with sparse nonnegative factor structure and nontrivial scaling,
    real reference SAEs and families through the unit cache (r = 2007 and 2008)."""
    import numpy as np
    years = config["reference_year_candidates"]
    population = synthetic_population(years=years, patients_per_year=160)
    roles = {r: reference_year_roles(population, r, config["split_seed"]) for r in years}
    dimension, year, rng = 8, population["year"], np.random.default_rng(23)
    mixing = rng.normal(size=(6, dimension))

    def predictor_output(r):
        rows = np.flatnonzero(year >= r)
        latent = (rng.random((len(rows), 6)) < .3) * rng.gamma(2., 1., size=(len(rows), 6))
        embeddings = latent @ mixing + .05 * rng.normal(size=(len(rows), dimension)) + np.linspace(-1, 1, dimension)
        return {"row_index": rows.astype(np.int64), "probabilities": np.full((len(rows), 2), .5, dtype=np.float32),
                "embeddings": embeddings.astype(np.float32), "reference_year": r, "roles_identity": roles[r]["identity"],
                "population_identity": population["fingerprint"]["population_identity"],
                "output_row_identity": _identity_hash({"row_index": rows}), "embedding_dim": dimension,
                "context_identity": "fixture-context", "_rows": len(rows)}

    return {"population": population, "roles": roles, "outputs": {r: predictor_output(r) for r in years}, "dimension": dimension}


def _check_idea2_yearly(config):
    """Yearly SAEs and yearly families through the unit cache on a tiny real SAE system."""
    import numpy as np
    import torch
    from sklearn.preprocessing import StandardScaler
    config = dict(config, epochs=30)
    config["config_hash"] = config_hash(config)
    years, seeds = config["reference_year_candidates"], config["sae_seeds"]
    world = _idea2_yearly_world(config)
    population, roles, outputs = world["population"], world["roles"], world["outputs"]
    view, year = matching_population_view(population), population["year"]
    free = lambda: 100 * 2 ** 30
    base = Path(os.environ["RPI_OUT_DIR"])
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as directory:
        work = Path(directory)

        def open_cache(name, *, trained=True, families=True, chosen=(2007, 2008)):
            cache = UnitCache(work / name, config, "source-fixture", free_bytes=free)
            for r in years:
                assert cache.run(f"P/{r}", lambda r=r: dict(outputs[r]))["status"] == "ok"
            if trained:
                for seed in seeds:
                    for r in chosen:
                        assert run_sae_unit(cache, population, roles[r], config, seed, "cpu")["status"] == "ok"
                if families:
                    for r in chosen:
                        assert run_family_unit(cache, view, roles[r], config, "cpu")["status"] == "ok"
            return cache

        cache = open_cache("cache", trained=False)
        for seed in seeds:
            for r in (2007, 2008):
                assert run_sae_unit(cache, population, roles[r], config, seed, "cpu")["status"] == "ok"
        early = run_idea2_yearly_families_unit(cache, view, roles[2007], config, "cpu", 2008)
        assert (early["status"], early["reason"]) == ("unavailable", "dep_not_run:D/2007/families")
        for r in (2007, 2008):
            assert run_family_unit(cache, view, roles[r], config, "cpu")["status"] == "ok"
        early = run_idea2_yearly_families_unit(cache, view, roles[2007], config, "cpu", 2008)
        assert (early["status"], early["reason"]) == ("unavailable", "dep_not_run:D/2007/idea2_yearly_sae/42/2008")

        _refuses(lambda: run_idea2_yearly_sae_unit(cache, population, roles[2007], config, "cpu", 42, 2008), "outcomes are not accepted")
        _refuses(lambda: run_idea2_yearly_sae_unit(cache, view, roles[2007], dict(config, epochs=3), "cpu", 42, 2008), "differs")
        _refuses(lambda: run_idea2_yearly_sae_unit(cache, view, roles[2007], config, "cpu", 42, 2007), "follow-up")
        _refuses(lambda: run_idea2_yearly_sae_unit(cache, view, roles[2009], config, "cpu", 42, 2009), "follow-up")
        _refuses(lambda: run_idea2_yearly_families_unit(cache, dict(view, y=population["y"]), roles[2007], config, "cpu", 2008), "outcomes are not accepted")
        _refuses(lambda: run_idea2_yearly_families_unit(cache, view, roles[2007], config, "cpu", 2007), "follow-up")

        captured = []

        def capturing_fit(scaled, cfg, device, seed):
            captured.append((seed, scaled.copy()))
            return fit_yearly_dictionary(scaled, cfg, device, seed)

        for seed in seeds:
            record = run_idea2_yearly_sae_unit(cache, view, roles[2007], config, "cpu", seed, 2008, fit=capturing_fit)
            assert record["status"] == "ok" and record["key"] == f"D/2007/idea2_yearly_sae/{seed}/2008" and record["stage"] == "D.idea2_yearly_sae"
            assert record["dependencies"] == {"P/2007": cache._latest["P/2007"]["id"], f"S/{seed}/2007": cache._latest[f"S/{seed}/2007"]["id"]}
        assert [seed for seed, _ in captured] == seeds

        # Yearly SAEs: year-t permanent fitting patients, frozen reference scaler, no yearly rescaling.
        t_rows = np.flatnonzero(year == 2008)
        fitting_mask = annual_fitting_mask(population["patient_id"][t_rows], roles[2007]["reference_roles"], 2007, 42)
        fitting = t_rows[fitting_mask]
        assert len(fitting) >= 20 and roles[2007]["is_entrant"][fitting].any() and (~roles[2007]["is_entrant"][fitting]).any()
        embeddings = outputs[2007]["embeddings"][np.searchsorted(outputs[2007]["row_index"], fitting)]
        for seed, scaled in captured:
            reference = cache.load(f"S/{seed}/2007")
            scaler = StandardScaler()
            scaler.mean_, scaler.scale_ = reference["scaler_mean"], reference["scaler_scale"]
            assert np.abs(reference["scaler_mean"]).max() > .1 and not np.allclose(reference["scaler_scale"], 1), "The scaler must be nontrivial"
            frozen = (embeddings.astype(np.float64) - scaler.mean_) / scaler.scale_
            np.testing.assert_allclose(scaled, frozen.astype(np.float32), rtol=0, atol=1e-6)
            yearly_standardized = StandardScaler().fit_transform(embeddings)
            assert not np.allclose(scaled, yearly_standardized, atol=1e-3), "Yearly rescaling is not allowed"
            unit = cache.load(f"D/2007/idea2_yearly_sae/{seed}/2008")
            assert unit["status"] == "ok" and unit["scaler_identity"] == reference["scaler_identity"] and unit["seed"] == seed
            assert unit["fitting_rows"] == len(fitting) == unit["_rows"] and unit["fitting_rows_identity"] == _identity_hash({"row_index": fitting})
            assert unit["fitting"]["entrant_rows"] == int(roles[2007]["is_entrant"][fitting].sum()) and unit["outcomes_read"] is False
            assert unit["latent_count"] == reference["latent_count"] and unit["encoder_weight"].shape == reference["encoder_weight"].shape
            assert unit["sae_identity"] != reference["sae_identity"] and unit["epochs"] == config["epochs"] == 30
            assert np.isfinite(unit["final_loss"]) and unit["decoder_tied_to_encoder"] and "frozen reference scaler" in unit["scaler_source"]
            from sae_compare import train_all_saes
            run, = train_all_saes(1, scaled, seeds=[seed], epochs=config["epochs"], model_type=config["sae"]["model_type"], alpha=config["sae"]["alpha"],
                                  scaling_factor=config["sae"]["scaling_factor"], learning_rate=config["sae"]["learning_rate"],
                                  weight_decay=config["sae"]["weight_decay"], device="cpu", encoding_batch_size=config["encoding_batch_size"], show_progress=False)
            for name, key in (("encoder.weight", "encoder_weight"), ("encoder.bias", "encoder_bias"), ("decoder_bias", "decoder_bias")):
                assert np.array_equal(run["model"].state_dict()[name].numpy(), unit[key]), f"{key}: yearly SAE is not the configured-epoch fit of the captured matrix"
            loaded = load_yearly_sae(cache, 2007, seed, 2008, "cpu")
            assert loaded["scaler_identity"] == reference["scaler_identity"] and loaded["year"] == 2008 and loaded["sae_identity"] == unit["sae_identity"]
            sample = outputs[2007]["embeddings"][:20]
            direct = encode_with_reference_sae(loaded, sample, "cpu", 7)
            with torch.no_grad():
                manual = loaded["run"]["model"](torch.as_tensor(((sample.astype(np.float64) - scaler.mean_) / scaler.scale_).astype(np.float32)))[0].numpy()
            np.testing.assert_allclose(direct, manual, rtol=0, atol=1e-5)
        assert projected_idea2_yearly_sae_bytes(config) >= sum(path.stat().st_size for path in cache._unit_path("D/2007/idea2_yearly_sae/42/2008").rglob("*") if path.is_file())
        poisoned = dict(outputs[2007], embeddings=np.where(np.isin(outputs[2007]["row_index"], fitting)[:, None], outputs[2007]["embeddings"], np.float32("nan")))
        dictionaries = load_reference_dictionaries(cache, 2007, "cpu")
        twin = idea2_yearly_sae_unit(view, 2007, 2008, roles[2007], poisoned, config, 42, "cpu", dictionaries[42])
        stored = cache.load("D/2007/idea2_yearly_sae/42/2008")
        for key in ("encoder_weight", "encoder_bias", "decoder_bias"):
            assert np.array_equal(twin[key], stored[key]), f"{key}: rows outside the year-t fitting patients were read or training is not deterministic"
        assert twin["sae_identity"] == stored["sae_identity"] and twin["identity"] == stored["identity"]
        _refuses(lambda: idea2_yearly_sae_unit(dict(view, y=population["y"]), 2007, 2008, roles[2007], outputs[2007], config, 42, "cpu", dictionaries[42]), "outcomes are not accepted")
        _refuses(lambda: idea2_yearly_sae_unit(view, 2007, 2008, roles[2007], outputs[2007], config, 42, "cpu", dictionaries[135]), "requested seed")
        _refuses(lambda: idea2_yearly_sae_unit(view, 2007, 2008, roles[2007], outputs[2007], config, 42, "auto", dictionaries[42]), "resolved device")
        _refuses(lambda: idea2_yearly_sae_unit(view, 2007, 2008, roles[2007], outputs[2007], config, 7, "cpu", dictionaries[42]), "requested seed")

        # Yearly families on the eligible held-out year-t evaluation rows only.
        calls = []

        def counting_build(*arguments, **options):
            calls.append(options["id_prefix"])
            return build_factor_families(*arguments, **options)

        paired_calls, original_pair = [], idea2_pair_encodings

        def recording_pair(reference, yearly):
            paired_calls.append((np.asarray(reference["row_index"]).copy(), np.asarray(yearly["row_index"]).copy(),
                                 [m.shape for m in (*reference["matrices"], *yearly["matrices"])]))
            return original_pair(reference, yearly)

        try:
            globals()["idea2_pair_encodings"] = recording_pair
            record = run_idea2_yearly_families_unit(cache, view, roles[2007], config, "cpu", 2008, build=counting_build)
        finally:
            globals()["idea2_pair_encodings"] = original_pair
        assert len(paired_calls) == 1 and np.array_equal(paired_calls[0][0], paired_calls[0][1]) and len({shape for shape in paired_calls[0][2]}) == 1, \
            "The unit must assert identical ordered rows and dimensions between the reference and yearly encodings"
        assert record["status"] == "ok" and record["key"] == "D/2007/idea2_yearly_families/2008" and record["stage"] == "D.idea2_yearly_families", record
        assert record["dependencies"] == {"P/2007": cache._latest["P/2007"]["id"], "D/2007/families": cache._latest["D/2007/families"]["id"],
                                          **{f"D/2007/idea2_yearly_sae/{seed}/2008": cache._latest[f"D/2007/idea2_yearly_sae/{seed}/2008"]["id"] for seed in seeds}}
        assert calls == ["2007/42/2008"]
        again = run_idea2_yearly_families_unit(cache, view, roles[2007], config, "cpu", 2008, build=counting_build)
        assert again["id"] == record["id"] and len(calls) == 1, "A complete D unit was rebuilt"
        unit = cache.load("D/2007/idea2_yearly_families/2008")
        evaluation = t_rows[roles[2007]["row_role"][t_rows] == "evaluation"]
        assert np.array_equal(paired_calls[0][0], evaluation)
        assert len(evaluation) >= 10 and np.array_equal(unit["evaluation_row_index"], evaluation) and unit["_rows"] == record["rows"] == len(evaluation)
        assert (year[unit["evaluation_row_index"]] == 2008).all() and not (year[unit["evaluation_row_index"]] > 2008).any(), "No future rows"
        assert not set(unit["evaluation_row_index"]) & set(fitting) and unit["outcomes_read"] is False and unit["status"] == "ok"
        assert np.array_equal(unit["evaluation_is_entrant"], roles[2007]["is_entrant"][evaluation])
        assert unit["evaluation"]["returning_rows"] + unit["evaluation"]["entrant_rows"] == len(evaluation) and unit["mode"] == "yearly_retrained" and unit["reuse"] is False
        assert unit["evaluation_rows_identity"] == _identity_hash({"row_index": evaluation}) and unit["row_pair_identity"] == unit["evaluation_rows_identity"]
        reference_dicts = load_reference_dictionaries(cache, 2007, "cpu")
        yearly_dicts = {seed: load_yearly_sae(cache, 2007, seed, 2008, "cpu") for seed in seeds}
        sample = outputs[2007]["embeddings"][np.searchsorted(outputs[2007]["row_index"], evaluation)]
        reference_matrices = [encode_with_reference_sae(reference_dicts[seed], sample, "cpu", config["encoding_batch_size"]) for seed in seeds]
        yearly_matrices = [encode_with_reference_sae(yearly_dicts[seed], sample, "cpu", config["encoding_batch_size"]) for seed in seeds]
        assert {matrix.shape for matrix in reference_matrices + yearly_matrices} == {(len(evaluation), unit["latent_count"])}
        direct = build_factor_families(yearly_matrices, seeds, device="cpu", id_prefix="2007/42/2008", permutations=config["chance_permutations"],
                                       rng_seed=[config["split_seed"], 2007, 2008])
        assert unit["families"] == json.loads(json.dumps(direct["families"])), "Yearly families differ from the generic construction"
        assert unit["construction"]["counts"] == direct["counts"] and unit["construction"]["chance"] == direct["chance"]
        assert unit["construction"]["chance"]["permutations"] == config["chance_permutations"] and unit["singletons"] == direct["singletons"]
        assert unit["construction"]["id_prefix"] == "2007/42/2008" and all(f["id"].startswith("2007/42/2008/") for f in unit["families"])
        family_unit_output = cache.load("D/2007/families")
        linked = link_reference_to_yearly(reference_matrices, yearly_matrices, seeds, family_unit_output["families"], direct["families"],
                                          device="cpu", permutations=config["chance_permutations"], rng_seed=[config["split_seed"], 2007, 2008, 1])
        assert unit["links"] == json.loads(json.dumps(linked["links"])) and unit["yearly_links"] == json.loads(json.dumps(linked["yearly_links"]))
        assert np.array_equal(unit["similarity"], linked["similarity"], equal_nan=True) and unit["link_chance"] == json.loads(json.dumps(linked["chance"]))
        assert unit["link_chance"]["permutations"] == config["chance_permutations"] and unit["state_counts"] == linked["state_counts"]
        assert unit["reference_family_ids"] == linked["reference_ids"] and unit["yearly_family_ids"] == linked["yearly_ids"]
        full_reference = [f for f in family_unit_output["families"] if f["kind"] == "full"]
        assert len(full_reference) >= 1 and len(unit["links"]) == len(full_reference) == unit["similarity"].shape[0] and unit["similarity"].dtype == np.float64
        assert sum(unit["state_counts"].values()) == len(full_reference) and set(unit["state_counts"]) == set(IDEA2_LINK_STATES)
        for link in unit["links"]:
            row = unit["reference_family_ids"].index(link["family_id"])
            family = next(f for f in full_reference if f["id"] == link["family_id"])
            for column, yearly_id in enumerate(unit["yearly_family_ids"]):
                other = next(f for f in direct["families"] if f["id"] == yearly_id)
                expected = _mean_pairwise(reference_matrices, yearly_matrices, seeds, family, other)
                got = unit["similarity"][row, column]
                assert (np.isnan(expected) and np.isnan(got)) or abs(expected - got) < 1e-9
            assert link["state"] in IDEA2_LINK_STATES and link["retained"] == (link["state"] == "retained")
        assert unit["scaler_identity"] == reference_dicts[42]["scaler_identity"] == yearly_dicts[135]["scaler_identity"]
        assert unit["reference_sae_identities"] == {str(seed): reference_dicts[seed]["sae_identity"] for seed in seeds}
        assert unit["yearly_sae_identities"] == {str(seed): yearly_dicts[seed]["sae_identity"] for seed in seeds}
        assert all(unit["yearly_sae_identities"][key] != unit["reference_sae_identities"][key] for key in unit["yearly_sae_identities"])
        table = idea2_link_table(unit)
        assert len(table) == len(full_reference) and {row["retained"] for row in table} <= {True, False}
        assert projected_idea2_yearly_families_bytes(config) >= sum(path.stat().st_size for path in cache._unit_path("D/2007/idea2_yearly_families/2008").rglob("*") if path.is_file())
        reuse = idea2_reuse_record(family_unit_output, 2007)
        assert [link["family_id"] for link in reuse["links"]] == [f["id"] for f in full_reference] and reuse["mode"] == "reference_reuse"

        # Only the year-t evaluation rows are read: every other row (reference year, other roles, the future year) is NaN.
        poisoned = dict(outputs[2007], embeddings=np.where(np.isin(outputs[2007]["row_index"], evaluation)[:, None], outputs[2007]["embeddings"], np.float32("nan")))
        outside = ~np.isin(poisoned["row_index"], evaluation)
        assert np.isnan(poisoned["embeddings"][outside]).all() and np.isnan(poisoned["embeddings"][year[poisoned["row_index"]] == 2009]).all()
        assert {2007, 2008, 2009} <= set(year[poisoned["row_index"]][outside].tolist()), "The poison must cover reference-year, other-role and future rows"
        twin = idea2_yearly_families_unit(view, 2007, 2008, roles[2007], poisoned, family_unit_output, reference_dicts, yearly_dicts, config, "cpu")
        _unit_arrays_equal({k: v for k, v in unit.items()}, twin)
        _refuses(lambda: idea2_yearly_families_unit(dict(view, y=population["y"]), 2007, 2008, roles[2007], outputs[2007], family_unit_output,
                                                    reference_dicts, yearly_dicts, config, "cpu"), "outcomes are not accepted")
        _refuses(lambda: idea2_yearly_families_unit(view, 2007, 2008, roles[2007], outputs[2007], family_unit_output,
                                                    reference_dicts, dict(reversed(list(yearly_dicts.items()))), config, "cpu"), "exactly the configured")
        _refuses(lambda: idea2_yearly_families_unit(view, 2007, 2008, roles[2007], outputs[2007], family_unit_output, reference_dicts, yearly_dicts,
                                                    config, "auto"), "resolved device")
        shifted = dict(yearly_dicts[42], scaler_identity="different")
        _refuses(lambda: idea2_yearly_families_unit(view, 2007, 2008, roles[2007], outputs[2007], family_unit_output, reference_dicts,
                                                    {**yearly_dicts, 42: shifted}, config, "cpu"), "frozen reference scaler", RuntimeError)

        # A yearly SAE that is unavailable, failed, or a build that fails: labelled, never partial.
        starved_roles = dict(roles[2007], row_role=np.where(year == 2008, "evaluation", roles[2007]["row_role"]))
        starved_roles["identity"] = _identity_hash({"starved": roles[2007]["identity"]})
        starved_output = dict(outputs[2007], roles_identity=starved_roles["identity"])
        starved = idea2_yearly_sae_unit(view, 2007, 2008, starved_roles, starved_output, config, 42, "cpu", dictionaries[42])
        assert (starved["status"], starved["reason"], starved["_rows"], starved["fitting_rows"]) == ("unavailable", IDEA2_UNAVAILABLE_FIT, 0, 0)
        assert "encoder_weight" not in starved and starved["outcomes_read"] is False
        cascade = idea2_yearly_families_unit(view, 2007, 2008, roles[2007], outputs[2007], family_unit_output, reference_dicts,
                                             {**yearly_dicts, 135: str(starved["reason"])}, config, "cpu")
        assert (cascade["status"], cascade["reason"], cascade["_rows"]) == ("unavailable", f"yearly_sae_unavailable:135:{IDEA2_UNAVAILABLE_FIT}", 0)
        assert idea2_link_table(cascade) == [{"t": 2008, "status": f"unavailable:{cascade['reason']}"}]
        lonely = dict(roles[2007], row_role=np.where(year == 2008, np.where(roles[2007]["row_role"] == "evaluation", "concept_selection", roles[2007]["row_role"]),
                                                     roles[2007]["row_role"]))
        lonely["identity"] = _identity_hash({"lonely": roles[2007]["identity"]})
        lonely_output = dict(outputs[2007], roles_identity=lonely["identity"])
        thin = idea2_yearly_families_unit(view, 2007, 2008, lonely, lonely_output, dict(family_unit_output, roles_identity=lonely["identity"]),
                                          reference_dicts, yearly_dicts, config, "cpu")
        assert (thin["status"], thin["reason"], thin["_rows"]) == ("unavailable", IDEA2_UNAVAILABLE_EVALUATION, 0)

        failing = open_cache("cache-failure")

        def nan_fit(scaled, cfg, device, seed):
            dictionary = fit_yearly_dictionary(scaled, cfg, device, seed)
            with torch.no_grad():
                dictionary["run"]["model"].encoder.weight[0, 0] = float("nan")
            return dictionary

        assert run_idea2_yearly_sae_unit(failing, view, roles[2007], config, "cpu", 42, 2008)["status"] == "ok"
        assert run_idea2_yearly_sae_unit(failing, view, roles[2007], config, "cpu", 135, 2008, fit=nan_fit)["status"] == "failed"
        assert run_idea2_yearly_sae_unit(failing, view, roles[2007], config, "cpu", 215, 2008)["status"] == "ok"
        before = len(calls)
        record = run_idea2_yearly_families_unit(failing, view, roles[2007], config, "cpu", 2008, build=counting_build)
        assert (record["status"], record["reason"]) == ("unavailable", "dep_failed:D/2007/idea2_yearly_sae/135/2008") and len(calls) == before
        assert not failing.is_complete("D/2007/idea2_yearly_families/2008") and not failing._unit_path("D/2007/idea2_yearly_families/2008").exists()
        assert failing.completeness_table(["D/2007/idea2_yearly_families/2008"]) == [
            {"unit": "D/2007/idea2_yearly_families/2008", "status": "unavailable:dep_failed:D/2007/idea2_yearly_sae/135/2008"}]
        assert failing.run("D/2007/groups_interventions/2008", lambda: {})["status"] == "unavailable"

        def broken_build(*arguments, **options):
            raise FloatingPointError("injected")

        crashed = open_cache("cache-build")
        for seed in seeds:
            assert run_idea2_yearly_sae_unit(crashed, view, roles[2007], config, "cpu", seed, 2008)["status"] == "ok"
        record = run_idea2_yearly_families_unit(crashed, view, roles[2007], config, "cpu", 2008, build=broken_build)
        assert record["status"] == "failed" and record["reason"] == "FloatingPointError: injected"
        assert not crashed._unit_path("D/2007/idea2_yearly_families/2008").exists()
        assert run_idea2_yearly_families_unit(crashed, view, roles[2007], config, "cpu", 2008)["status"] == "ok"

        # The latest follow-up year of the later reference year, through the same units.
        for seed in seeds:
            assert run_idea2_yearly_sae_unit(cache, view, roles[2008], config, "cpu", seed, 2009)["status"] == "ok"
        last = run_idea2_yearly_families_unit(cache, view, roles[2008], config, "cpu", 2009)
        assert last["status"] == "ok" and cache.load("D/2008/idea2_yearly_families/2009")["year"] == 2009
        assert cache.load("D/2008/idea2_yearly_families/2009")["construction"]["id_prefix"] == "2008/42/2009"

        if _cuda_available():
            gpu_dicts = load_reference_dictionaries(cache, 2007, "cuda")
            gpu_yearly = {seed: load_yearly_sae(cache, 2007, seed, 2008, "cuda") for seed in seeds}
            gpu = idea2_yearly_families_unit(view, 2007, 2008, roles[2007], outputs[2007], family_unit_output, gpu_dicts, gpu_yearly, config, "cuda")
            assert gpu["construction"]["device"] == "cuda" and gpu["status"] == "ok"
            assert [(f["id"], f["kind"], f["medoid"]) for f in gpu["families"]] == [(f["id"], f["kind"], f["medoid"]) for f in unit["families"]]
            assert [(l["family_id"], l["retained"], l["state"], l["matched_family"]) for l in gpu["links"]] == [
                (l["family_id"], l["retained"], l["state"], l["matched_family"]) for l in unit["links"]]
            np.testing.assert_allclose(gpu["similarity"], unit["similarity"], rtol=0, atol=1e-6, equal_nan=True)
            _same_chance_rows(gpu["link_chance"]["per_permutation"], unit["link_chance"]["per_permutation"])
            assert torch.cuda.memory_allocated() < 64 * 2 ** 20, "GPU memory was not released after the yearly family unit"
    return unit


def run_idea2_checks(notebook_path="concept_subgroup_drift_pilot.ipynb"):
    """Idea 2: frozen-rule view (medoid retention, member prevalence/magnitude), yearly SAEs under the frozen scaler, yearly
    families, family-level mutual-nearest-neighbour links with the chance check, same-row and no-future-row guarantees."""
    import numpy as np
    import torch
    presets = _notebook_presets(json.loads(Path(notebook_path).read_text()))
    config = resolve_config(presets, "v0")
    assert config["profile"] == "v0" and config["bootstrap_replicates"] == 10 and config["joblib_workers"] <= 4
    config = _selection_config(config)
    _check_idea2_settings(presets, config)
    _check_idea2_pairing()
    on_host = _check_idea2_links("cpu")
    world = _idea2_world()
    unit, output = _check_idea2_frozen(config, world)
    _check_idea2_frozen_cache(config, world, unit, output)
    _check_idea2_yearly(config)
    if _cuda_available():
        on_device = _check_idea2_links("cuda")
        assert [(l["family_id"], l["retained"], l["state"], l["matched_family"]) for l in on_device["links"]] == [
            (l["family_id"], l["retained"], l["state"], l["matched_family"]) for l in on_host["links"]]
        np.testing.assert_allclose(on_device["similarity"], on_host["similarity"], rtol=0, atol=1e-9, equal_nan=True)
        _same_chance_rows(on_device["chance"]["per_permutation"], on_host["chance"]["per_permutation"])
        assert torch.cuda.memory_allocated() < 64 * 2 ** 20, "GPU memory was not released after family linking"
    else:
        print("CUDA Idea 2 equality checks skipped: CUDA unavailable here; GPU validation pending")
    print("Idea 2 fixtures passed: same-row pairing, frozen-rule retention and member statistics, yearly SAEs under the frozen scaler, "
          "yearly families, family-level links with chance check, no future rows, labelled reuse")


def _check_groups_settings(config):
    """Follow-up years and graph dependencies of the groups unit, stage name, group order, example and control constants."""
    pilot = {"reference_year_candidates": list(range(2007, 2016)), "t0_only_reference_years": [2015], "sae_seeds": [42, 135, 215]}
    assert groups_follow_up_years(pilot, 2007) == list(range(2008, 2016)) and groups_follow_up_years(pilot, 2014) == [2015]
    assert groups_follow_up_years(pilot, 2015) == [] and groups_follow_up_years(config, 2007) == [2008, 2009]
    assert groups_follow_up_years(config, 2008) == [2009] and groups_follow_up_years(config, 2009) == []
    graph = unit_graph(pilot)
    assert graph["D/2007/groups_interventions/2010"] == ("P/2007", "D/2007/cavs_selection", "D/2007/idea1/2010", "D/2007/idea2_frozen/2010")
    assert "D/2007/groups_interventions/2007" not in graph and not [key for key in graph if key.startswith("D/2015/groups")]
    assert unit_stage("D/2007/groups_interventions/2010") == "D.groups_interventions"
    assert STAGE_CODE_VERSIONS.get("D.groups_interventions", 1) == 1
    assert GROUP_NAMES == ("R_and_A", "R_and_not_A", "not_R_and_A", "not_R_and_not_A")
    assert GROUPS_EXAMPLE_LIMIT == 4 and GROUPS_CONTROL_DRAWS >= 3 and GROUPS_VIEW_KEYS == PERFORMANCE_VIEW_KEYS
    assert config["inference_batch_rows"] > 0 and config["cav_seeds"] == [42, 43, 44]


def _groups_sae_payload(seed, dimension, reference_year=2007):
    """A cached S(seed, r) unit with a tied ReLU SAE whose weights are a perturbed identity, a nonzero scaler mean and a
    non-unit scaler scale, so that residuals, bias and inverse scaling all matter."""
    import numpy as np
    import torch
    from sae import SAE
    rng = np.random.default_rng(1000 + int(seed))
    latents = int(dimension * 1.5)
    weight = (.15 * rng.normal(size=(latents, dimension))).astype(np.float32)
    weight[np.arange(dimension), np.arange(dimension)] += 1.
    bias = (.05 * rng.normal(size=latents)).astype(np.float32)
    decoder_bias = (.1 * rng.normal(size=dimension)).astype(np.float32)
    mean, scale = .2 * np.arange(dimension) / dimension, .7 + .1 * np.arange(dimension)
    model = SAE(data_dimension=dimension, scaling_factor=1.5, use_decoder_bias=True, type="ReLU")
    model.load_state_dict({"encoder.weight": torch.from_numpy(weight), "encoder.bias": torch.from_numpy(bias),
                           "decoder_bias": torch.from_numpy(decoder_bias)})
    return {"encoder_weight": weight, "encoder_bias": bias, "decoder_bias": decoder_bias, "scaler_mean": mean, "scaler_scale": scale,
            "scaler_var": scale ** 2, "_rows": 1, "reference_year": reference_year, "seed": int(seed), "epochs": 2,
            "latent_count": latents, "embedding_dim": dimension,
            "sae": {"model_type": "ReLU", "scaling_factor": 1.5, "alpha": .1, "learning_rate": .001, "weight_decay": 0.},
            "scaler_samples": 10, "scaler_identity": _identity_hash((mean, scale)), "sae_identity": _sae_state_identity(model),
            "identity": f"groups-sae-{seed}"}


def _groups_dictionaries(payloads, device="cpu"):
    import torch
    from sae import SAE
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    result = {}
    for seed, payload in payloads.items():
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_, scaler.var_ = payload["scaler_mean"], payload["scaler_scale"], payload["scaler_var"]
        scaler.n_features_in_, scaler.n_samples_seen_ = payload["embedding_dim"], 10
        model = SAE(data_dimension=payload["embedding_dim"], scaling_factor=1.5, use_decoder_bias=True, type="ReLU")
        model.load_state_dict({"encoder.weight": torch.from_numpy(np.array(payload["encoder_weight"])),
                               "encoder.bias": torch.from_numpy(np.array(payload["encoder_bias"])),
                               "decoder_bias": torch.from_numpy(np.array(payload["decoder_bias"]))})
        result[seed] = {"scaler": scaler, "run": {"model": model.to(device).eval(), "model_type": "ReLU"},
                        "scaler_identity": payload["scaler_identity"], "sae_identity": payload["sae_identity"], "seed": seed,
                        "epochs": 2, "reference_year": 2007, "latent_count": payload["latent_count"],
                        "embedding_dim": payload["embedding_dim"]}
    return result


def _groups_codes(payload, h):
    """Independent float64 numpy encoding under a cached SAE payload and its scaler."""
    import numpy as np
    u = (np.asarray(h, dtype=np.float64) - payload["scaler_mean"]) / payload["scaler_scale"]
    return np.maximum(u @ np.asarray(payload["encoder_weight"], dtype=np.float64).T + np.asarray(payload["encoder_bias"], dtype=np.float64), 0.)


def _check_group_primitives(device):
    """Partition truth table, feature distributions, coactivation and deterministic example selection against numpy."""
    import inspect
    import numpy as np
    import torch
    place = torch.device(device)
    a = torch.tensor([True, True, False, False], device=place)
    r = torch.tensor([True, False, True, False], device=place)
    membership = partition_groups(a, r)
    assert membership.shape == (4, 4) and membership.dtype == torch.bool
    code = (membership.to(torch.int64) * torch.arange(4, device=place)[:, None]).sum(dim=0).tolist()
    assert code == [0, 2, 1, 3], code
    assert membership.sum(dim=0).tolist() == [1, 1, 1, 1] and membership.sum(dim=1).tolist() == [1, 1, 1, 1]
    rng = np.random.default_rng(5)
    flags_a, flags_r = rng.random((3, 200)) < .3, rng.random((3, 200)) < .5
    flags_a[1], flags_r[1] = False, True
    many = partition_groups(torch.as_tensor(flags_a, device=place), torch.as_tensor(flags_r, device=place))
    for family in range(3):
        for index, expected in enumerate((flags_r & flags_a, flags_r & ~flags_a, ~flags_r & flags_a, ~flags_r & ~flags_a)):
            assert np.array_equal(many[family, index].cpu().numpy(), expected[family])
    assert int(many.sum()) == 3 * 200
    _refuses(lambda: partition_groups(torch.ones(4), torch.ones(4, dtype=torch.bool)), "boolean")
    _refuses(lambda: partition_groups(torch.ones(4, dtype=torch.bool), torch.ones(5, dtype=torch.bool)), "ordered row axis")
    _refuses(lambda: partition_groups(torch.ones((1, 2, 3), dtype=torch.bool), torch.ones((1, 2, 3), dtype=torch.bool)), "ordered row axis")

    features = rng.normal(size=(200, 6)) * np.arange(1, 7) + np.arange(6)
    summary = group_feature_summary(many, torch.as_tensor(features, device=place))
    mean, deviation, counts = summary["mean"].cpu().numpy(), summary["std"].cpu().numpy(), summary["rows"].cpu().numpy()
    assert mean.shape == deviation.shape == (3, 4, 6) and counts.shape == (3, 4)
    for family in range(3):
        for index, expected in enumerate((flags_r & flags_a, flags_r & ~flags_a, ~flags_r & flags_a, ~flags_r & ~flags_a)):
            rows = expected[family]
            assert counts[family, index] == rows.sum()
            if rows.any():
                np.testing.assert_allclose(mean[family, index], features[rows].mean(axis=0), rtol=1e-12)
                np.testing.assert_allclose(deviation[family, index], features[rows].std(axis=0), rtol=1e-9)
            else:
                assert np.isnan(mean[family, index]).all() and np.isnan(deviation[family, index]).all()
    assert counts[1, 0] == 0 and counts[1, 2] == 0 and np.isnan(mean[1, 0]).all(), "Empty groups must be NaN, not zero"
    _refuses(lambda: group_feature_summary(many, torch.zeros((199, 6))), "share the row axis")

    activations = np.abs(rng.normal(size=(5, 200))) * (rng.random((5, 200)) < .6)
    coactive = group_coactivation(many, torch.as_tensor(activations, device=place))
    positive, total = coactive["positive_rows"].cpu().numpy(), coactive["activation_sum"].cpu().numpy()
    for family in range(3):
        for index, expected in enumerate((flags_r & flags_a, flags_r & ~flags_a, ~flags_r & flags_a, ~flags_r & ~flags_a)):
            for other in range(5):
                rows = expected[family]
                assert positive[family, index, other] == int((activations[other, rows] > 0).sum())
                np.testing.assert_allclose(total[family, index, other], activations[other, rows].sum(), rtol=1e-12, atol=1e-12)
    assert positive.shape == total.shape == (3, 4, 5)
    _refuses(lambda: group_coactivation(many, torch.as_tensor(-activations, device=place)), "nonnegative")

    own = np.array([[0., 3., 3., 1., 5., 0., 3., 2., 0., 4.]])
    in_group = np.array([[False, True, True, True, True, True, True, False, True, True]])
    in_other = np.array([[True, False, False, False, False, False, False, True, False, False]])
    block = torch.as_tensor(np.stack([in_group, in_other, np.zeros_like(in_group), np.zeros_like(in_group)], axis=1), device=place)
    chosen = select_group_examples(block, torch.as_tensor(own, device=place))
    positions, valid = chosen["positions"].cpu().numpy(), chosen["valid"].cpu().numpy()
    assert positions.shape == (1, 4, 4) and valid.shape == (1, 4, 4)
    assert positions[0, 0].tolist() == [4, 9, 1, 2] and valid[0, 0].all(), "Highest activation first, ties by lowest row"
    assert valid[0, 1].tolist() == [True, False, False, False] and positions[0, 1, 0] == 7, "Only rows with positive activation"
    assert not valid[0, 2].any() and not valid[0, 3].any(), "Empty groups give no examples"
    more = select_group_examples(block, torch.as_tensor(own, device=place), 2)
    assert more["positions"][0, 0].tolist() == [4, 9] and more["valid"].shape == (1, 4, 2)
    assert select_group_examples(block, torch.as_tensor(own, device=place), 50)["positions"].shape == (1, 4, 10)
    _refuses(lambda: select_group_examples(block, torch.as_tensor(own[:, :9], device=place)), "align")
    _refuses(lambda: select_group_examples(block, torch.as_tensor(own, device=place), 0), "positive integer")
    assert set(inspect.signature(select_group_examples).parameters) == {"membership", "activations", "limit"}, "Selection may not take labels or risks"

    pieces = _single_host_transfer({"a": torch.arange(6, device=place).reshape(2, 3), "b": torch.tensor(2.5, device=place),
                                    "c": torch.tensor([True, False], device=place), "d": torch.zeros((0, 3), device=place)})
    assert pieces["a"].tolist() == [[0, 1, 2], [3, 4, 5]] and float(pieces["b"]) == 2.5 and pieces["c"].tolist() == [1., 0.]
    assert pieces["d"].shape == (0, 3) and _single_host_transfer({}) == {}


def _check_residual_preserving_removal(device):
    """Inverse-scaled decoder-contribution removal keeps the reconstruction error and decoder bias: the un-intervened input
    is reproduced exactly, the result equals the explicit decomposition, reconstruction replacement and unconverted scaled
    subtraction are different, and controls are norm matched and reproducible."""
    import numpy as np
    import torch
    from sae import SAE
    place = torch.device(device)
    dimension = 6
    payload = _groups_sae_payload(7, dimension)
    dictionary = _groups_dictionaries({7: payload}, device)[7]
    rng = np.random.default_rng(3)
    h = (3 * rng.normal(size=(24, dimension)) + 1.).astype(np.float32)
    codes = _groups_codes(payload, h)
    factors = np.argmax(codes, axis=1)
    activation = codes[np.arange(len(h)), factors]
    assert (activation > 0).all()
    mean, scale = payload["scaler_mean"], payload["scaler_scale"]
    weight = np.asarray(payload["encoder_weight"], dtype=np.float64)
    bias = np.asarray(payload["decoder_bias"], dtype=np.float64)
    scaled = (h.astype(np.float64) - mean) / scale
    reconstruction = codes @ weight + bias
    error = scaled - reconstruction
    assert np.abs(error).max() > .05, "The fixture needs a real reconstruction error"
    contribution = activation[:, None] * weight[factors]
    out = remove_factor_contribution(dictionary, torch.as_tensor(h, device=place), factors, activation)
    removed = out["embeddings"].cpu().numpy()
    np.testing.assert_allclose(out["removal"].cpu().numpy(), scale * contribution, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(removed, h.astype(np.float64) - scale * contribution, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(removed - h.astype(np.float64), -out["removal"].cpu().numpy(), rtol=1e-9, atol=1e-12, err_msg="Only the contribution is subtracted")
    explicit = scale * (reconstruction - contribution + error) + mean
    np.testing.assert_allclose(removed, explicit, rtol=1e-6, atol=1e-8)
    assert float(out["residual_form_gap"].max()) < 1e-9 and float(out["activation_gap"].max()) < 1e-5 and float(out["decoder_gap"].max()) < 1e-4
    np.testing.assert_allclose(out["residual"].cpu().numpy(), scale * error, rtol=1e-6, atol=1e-8)
    replacement = scale * (reconstruction - contribution) + mean
    np.testing.assert_allclose(replacement - removed, -scale * error, rtol=1e-6, atol=1e-8)
    assert np.linalg.norm(replacement - removed, axis=1).min() > 1e-3, "Reconstruction replacement drops the residual"
    assert np.linalg.norm(h - contribution - removed, axis=1).min() > 1e-3, "Subtracting the scaled contribution without inverse scaling differs"
    assert not np.allclose(removed, h.astype(np.float64) - contribution)

    zero = remove_factor_contribution(dictionary, torch.as_tensor(h, device=place), factors, np.zeros(len(h)))
    assert np.array_equal(zero["embeddings"].to(torch.float32).cpu().numpy(), h), "Zero removal must reproduce the embedding exactly"
    assert float(zero["removal"].abs().max()) == 0.
    batched = remove_factor_contribution(dictionary, torch.as_tensor(h[:5], device=place), factors[:5], activation[:5])
    np.testing.assert_allclose(batched["embeddings"].cpu().numpy(), removed[:5], rtol=1e-12, atol=1e-12)
    empty = remove_factor_contribution(dictionary, torch.zeros((0, dimension), device=place), np.zeros(0, dtype=np.int64), np.zeros(0))
    assert empty["embeddings"].shape == (0, dimension) and empty["residual_form_gap"].shape == (0,)

    identity = _identity_dictionaries(dimension, [42], device)[42]
    columns = np.zeros((3, dimension), dtype=np.float32)
    columns[:, 1] = [2., -1., .5]
    simple = remove_factor_contribution(identity, torch.as_tensor(columns, device=place), [1, 1, 1], [2., 0., .5])
    np.testing.assert_allclose(simple["embeddings"].cpu().numpy()[:, 1], [0., -1., 0.], atol=1e-12)
    np.testing.assert_allclose(simple["residual"].cpu().numpy()[:, 1], [0., -1., 0.], atol=1e-12)

    topk = {"run": {"model": SAE(data_dimension=dimension, type="TopK", k=2, k_aux=2), "model_type": "TopK"}, "scaler": dictionary["scaler"]}
    _refuses(lambda: remove_factor_contribution(topk, torch.as_tensor(h, device=place), factors, activation), "tied ReLU")
    _refuses(lambda: remove_factor_contribution(dictionary, torch.as_tensor(h, device=place), factors, -activation), "nonnegative")
    _refuses(lambda: remove_factor_contribution(dictionary, torch.as_tensor(h, device=place), factors + 99, activation), "outside")
    _refuses(lambda: remove_factor_contribution(dictionary, torch.as_tensor(h[:, :4], device=place), factors, activation), "one factor and one activation")
    broken = h.copy()
    broken[0, 0] = np.nan
    _refuses(lambda: remove_factor_contribution(dictionary, torch.as_tensor(broken, device=place), factors, activation), "finite")
    mse = reconstruction_mse(dictionary, torch.as_tensor(h, device=place)).cpu().numpy()
    np.testing.assert_allclose(mse, (error ** 2).mean(axis=1), rtol=1e-9)

    first, second = control_directions(5, 4, 3, dimension), control_directions(5, 4, 3, dimension)
    assert np.array_equal(first, second) and not np.array_equal(first, control_directions(6, 4, 3, dimension))
    assert first.shape == (4, 3, dimension)
    removal = out["removal"][:4]
    controls = norm_matched_controls(removal, torch.as_tensor(first, device=place))
    assert controls.shape == (4, 3, dimension)
    np.testing.assert_allclose(torch.linalg.vector_norm(controls, dim=-1).cpu().numpy(),
                               np.repeat(torch.linalg.vector_norm(removal, dim=-1).cpu().numpy()[:, None], 3, axis=1), rtol=1e-12)
    cosine = (controls * removal[:, None, :]).sum(-1).cpu().numpy() / np.linalg.norm(controls.cpu().numpy(), axis=-1) / np.linalg.norm(removal.cpu().numpy(), axis=-1)[:, None]
    assert (np.abs(cosine) < .999).all(), "A control must not be the removal itself"
    return {"removed": removed, "factors": factors, "activation": activation}


def _groups_world(config):
    """The Idea 2 world with real reference SAEs (perturbed identity weights, nonzero scaler mean, non-unit scale), activations
    consistent with them, and a P(r) output whose probabilities and gradients come from a serialized head so that the cached
    continuation reproduces them. The CAV selection payload is built from planted gradients (eligibility: alpha idea 1; beta
    idea 2; epsilon both; gamma and delta neither)."""
    import numpy as np
    import torch
    world = _idea2_world()
    dimension, population, roles = world["dimension"], world["population"], world["roles"]
    seeds = list(config["sae_seeds"])
    payloads = {seed: _groups_sae_payload(seed, dimension) for seed in seeds}
    dictionaries = _groups_dictionaries(payloads)
    matching = world["family_output"]["row_index"]
    activations = np.stack([np.maximum(_groups_codes(payloads[medoid[0]], world["E"][matching])[:, medoid[1]], 0.)
                            for _, medoid in FIXTURE_FAMILIES]).astype(np.float32)
    family_output = dict(world["family_output"], candidate_activations=activations)
    family_output["selection"] = dict(family_output["selection"], candidates=[
        dict(candidate, positive_count=int((activations[index] > 0).sum())) for index, candidate in enumerate(family_output["selection"]["candidates"])])
    continuation = _fork_continuation(_fixture_head(23, inputs=dimension), torch.tensor([1, 0]), False, True,
                                      torch.tensor([0., 1., 0., 0., 1., 0.]), "cpu")
    arrays, meta = export_continuation(continuation)
    planted = dict(world["output"])
    embeddings = planted["embeddings"]
    output = dict(planted, probabilities=batched_death_probabilities(continuation, embeddings),
                  gradients=batched_death_gradients(continuation, embeddings), **arrays, continuation=meta,
                  continuation_identity=meta["identity"], context_identity=meta["context_identity"])
    rule_view = rule_population_view(population)
    probe = cavs_selection_unit(rule_view, 2007, roles, planted, family_output, world["frozen"], dictionaries, config, "cpu")
    planted = _with_gradients(dict(world, output=planted), probe["cav_vectors"])
    selection = cavs_selection_unit(rule_view, 2007, roles, planted, family_output, world["frozen"], dictionaries, config, "cpu")
    references = _references_from_units(selection, world["frozen"])
    view = performance_population_view(population)
    return dict(world, family_output=family_output, output=output, payloads=payloads, dictionaries=dictionaries,
                continuation=continuation_from_unit(output, 2007, "cpu"), selection_unit=selection, references=references,
                view=view, rule_view=rule_view, rows=_idea1_rows(world), continuation_meta=meta)


def _groups_dependencies(config, world):
    """Idea 1 and Idea 2 frozen payloads of the world for year 2008, computed directly."""
    view, roles, references = world["view"], world["roles"], world["references"]
    unit1 = idea1_year_unit(view, 2007, 2008, roles, world["output"], world["family_output"], references, world["dictionaries"],
                            config, "cpu", backend=IDEA1_BACKEND, workers=1)
    unit2 = idea2_frozen_year_unit(world["rule_view"], 2007, 2008, roles, world["output"], world["family_output"], references,
                                   world["dictionaries"], config, "cpu")
    return unit1, unit2


def _groups_run(world, config, unit1, unit2, **changes):
    arguments = dict(view=world["view"], r=2007, t=2008, roles=world["roles"], predictor_output=world["output"],
                     family_output=world["family_output"], references=world["references"], dictionaries=world["dictionaries"],
                     continuation=world["continuation"], idea1=unit1, idea2_frozen=unit2, config=config, device="cpu")
    arguments.update(changes)
    return groups_interventions_unit(**arguments)


def _groups_oracle(world, config):
    """Independent numpy reference of every group quantity for the usable families of year 2008, from the cached SAE payloads."""
    import numpy as np
    rows = world["rows"]["eval_t"]
    E, X, labels = world["E"], world["X"], world["population"]["y"]
    order = [name for name, _ in FIXTURE_FAMILIES if any(world["references"][world["identifiers"][name]]["eligible"].values())]
    oracle = {"rows": rows, "order": order, "families": {}}
    for name in order:
        index = [fixture for fixture, _ in FIXTURE_FAMILIES].index(name)
        seed, factor = FIXTURE_FAMILIES[index][1]
        payload = world["payloads"][seed]
        activation = _groups_codes(payload, E[rows])[:, factor]
        a = activation > 1.0
        r = X[rows, index].astype(np.float64) > .5
        oracle["families"][name] = {"index": index, "seed": seed, "factor": factor, "activation": activation, "A": a, "R": r,
                                    "code": np.where(r & a, 0, np.where(r & ~a, 1, np.where(~r & a, 2, 3)))}
    return oracle


def _check_groups_unit(config, world, unit1, unit2):
    """Groups of the planted year against an independent oracle: masks, partition, counts, performance, features, coactivation,
    examples, interventions, controls, zero removal, continuation parity, determinism, refusals."""
    import copy
    import inspect
    import numpy as np
    import torch
    population, roles = world["population"], world["roles"]
    ids, rows = world["identifiers"], world["rows"]
    unit = _groups_run(world, config, unit1, unit2)
    oracle = _groups_oracle(world, config)
    order = oracle["order"]
    assert order == ["alpha", "beta", "epsilon"], order
    evaluation = rows["eval_t"]
    assert unit["status"] == "ok" and unit["reason"] is None and unit["_rows"] == len(evaluation) == 405
    assert unit["family_ids"] == [ids[name] for name in order] == [family["family_id"] for family in unit["families"]]
    assert {item["family_id"] for item in unit["not_computed"]} == {ids["gamma"], ids["delta"]}
    assert {item["reason"] for item in unit["not_computed"]} == {"not_eligible_at_selection"}
    assert np.array_equal(unit["evaluation_row_index"], evaluation) and unit["evaluation_rows_identity"] == _identity_hash({"row_index": evaluation})
    assert np.array_equal(unit["evaluation_is_entrant"], roles["is_entrant"][evaluation]) and unit["evaluation_is_entrant"].any()
    assert unit["group_names"] == list(GROUP_NAMES) and unit["cav_seeds"] == [42, 43, 44]
    labels, patients, p1 = population["y"], population["patient_id"], world["output"]["probabilities"][np.searchsorted(world["output"]["row_index"], evaluation), 1].astype(np.float64)
    entrant = roles["is_entrant"][evaluation]
    assert unit["evaluation"]["rows"] == 405 and unit["evaluation"]["entrant_rows"] == int(entrant.sum())
    sizes = []
    for position, name in enumerate(order):
        family, expected = unit["families"][position], oracle["families"][name]
        assert family["status"] == "ok" and family["array_index"] == position and family["family_id"] == ids[name]
        assert np.array_equal(unit["mask_A"][position], expected["A"]) and np.array_equal(unit["mask_R"][position], expected["R"]), name
        assert np.array_equal(unit["group_code"][position], expected["code"]), name
        counts = [int((expected["code"] == index).sum()) for index in range(4)]
        sizes.append(counts)
        assert sum(counts) == len(evaluation), (name, counts)
        if name == "alpha":
            assert counts[1] == counts[2] == 0 < min(counts[0], counts[3]), counts  # the rule and the activation coincide for alpha
        else:
            assert min(counts) > 0, (name, counts)
        agreement = rule_activation_agreement(expected["R"], expected["A"])
        assert family["overlap"] == agreement and family["overlap"]["jaccard"] == counts[0] / (counts[0] + counts[1] + counts[2])
        for index, group in enumerate(GROUP_NAMES):
            flag = expected["code"] == index
            record = family["groups"][group]
            assert record["rows"] == counts[index]
            assert record["returning_rows"] == int((flag & ~entrant).sum()) and record["entrant_rows"] == int((flag & entrant).sum())
            assert record["deaths"] == int(labels[evaluation][flag].sum()) and record["patients"] == len(np.unique(patients[evaluation][flag]))
            metrics = record["reliability"]
            assert metrics["records"] == counts[index]
            if counts[index]:
                np.testing.assert_allclose(metrics["brier"], ((p1[flag] - labels[evaluation][flag]) ** 2).mean(), rtol=1e-12)
                np.testing.assert_allclose(metrics["mean_risk"], p1[flag].mean(), rtol=1e-12)
            else:
                assert metrics["brier"] is None and metrics["unavailable_reasons"]["brier"] == "empty_subgroup", "Empty groups are unavailable, never zero"
                assert record["patients"] == 0 and record["deaths"] == 0 and record["examples"] == []
            fitting = rows["fit_r"]
            reference_a = _groups_codes(world["payloads"][expected["seed"]], world["E"][fitting])[:, expected["factor"]] > 1.0
            reference_r = world["X"][fitting, expected["index"]] > .5
            members = {0: reference_r & reference_a, 1: reference_r & ~reference_a, 2: ~reference_r & reference_a, 3: ~reference_r & ~reference_a}[index]
            assert record["reference_fitting_rows"] == int(members.sum())
            if members.any():
                np.testing.assert_allclose(record["baseline"]["q"], labels[fitting][members].mean(), rtol=1e-12)
            else:
                assert record["baseline"]["q"] is None and record["baseline"]["unavailable_reason"] == "empty_fitting_subgroup"
                assert metrics["unavailable_reasons"]["brier_skill"] == ("empty_subgroup" if not counts[index] else "empty_fitting_subgroup")
            assert metrics["baseline_q"] == record["baseline"]["q"]
    # no outcome-free quantity depends on labels; features and coactivation against numpy
    features = world["X"][evaluation].astype(np.float64)
    for position, name in enumerate(order):
        expected = oracle["families"][name]
        for index in range(4):
            flag = expected["code"] == index
            if flag.any():
                np.testing.assert_allclose(unit["feature_mean"][position, index], features[flag].mean(axis=0), rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(unit["feature_std"][position, index], features[flag].std(axis=0), rtol=1e-4, atol=1e-6)
            else:
                assert np.isnan(unit["feature_mean"][position, index]).all() and np.isnan(unit["feature_std"][position, index]).all()
            for other, other_name in enumerate(order):
                values = oracle["families"][other_name]["activation"]
                values = np.maximum(values, 0.)
                assert unit["coactivation_positive_rows"][position, index, other] == int((values[flag] > 0).sum())
                np.testing.assert_allclose(unit["coactivation_activation_sum"][position, index, other], values[flag].sum(), rtol=1e-4, atol=1e-5)
            own = np.maximum(expected["activation"], 0.)[flag]
            assert unit["families"][position]["groups"][GROUP_NAMES[index]]["positive_example_rows"] == int((own > 0).sum())
    # reconstruction error of the medoid's SAE on the group rows
    for position, name in enumerate(order):
        expected = oracle["families"][name]
        payload = world["payloads"][expected["seed"]]
        h = world["E"][evaluation].astype(np.float64)
        scaled = (h - payload["scaler_mean"]) / payload["scaler_scale"]
        codes = _groups_codes(payload, h)
        error = ((scaled - (codes @ np.asarray(payload["encoder_weight"], dtype=np.float64) + payload["decoder_bias"])) ** 2).mean(axis=1)
        for index in range(4):
            if (expected["code"] == index).any():
                np.testing.assert_allclose(unit["reconstruction_mse"][position, index], error[expected["code"] == index].mean(), rtol=1e-6)
            else:
                assert np.isnan(unit["reconstruction_mse"][position, index])

    # deterministic examples: highest activation first, ties by row order, at most four, positive activation only
    expected_examples = []
    for position, name in enumerate(order):
        expected = oracle["families"][name]
        for index in range(4):
            flag = np.flatnonzero((expected["code"] == index) & (expected["activation"] > 0))
            ranked = sorted(flag, key=lambda item: (-expected["activation"][item], item))[:GROUPS_EXAMPLE_LIMIT]
            expected_examples += [(position, index, rank, int(evaluation[item]), item) for rank, item in enumerate(ranked)]
    got = list(zip(unit["example_family"].tolist(), unit["example_group"].tolist(), unit["example_rank"].tolist(), unit["example_row_index"].tolist()))
    assert got == [item[:4] for item in expected_examples], (got[:6], expected_examples[:6])
    examples = len(expected_examples)
    assert examples and unit["interventions"]["examples"] == examples and 0 < max(group["examples"].__len__() for family in unit["families"] for group in family["groups"].values()) <= GROUPS_EXAMPLE_LIMIT
    assert {group for _, group, *_ in expected_examples} == {0, 1, 2, 3}, "Every group should contribute examples in the fixture"

    # interventions against numpy: removal, residual, continuation probabilities, controls, losses, CAV projections
    continuation = world["continuation"]
    h = world["E"][unit["example_row_index"]].astype(np.float64)
    removal, residual, removed, activation = [], [], [], []
    for example, (position, index, rank, row, item) in enumerate(expected_examples):
        expected = oracle["families"][order[position]]
        payload = world["payloads"][expected["seed"]]
        weight = np.asarray(payload["encoder_weight"], dtype=np.float64)
        a = expected["activation"][item]
        delta = payload["scaler_scale"] * a * weight[expected["factor"]]
        u = (h[example] - payload["scaler_mean"]) / payload["scaler_scale"]
        codes = np.maximum(u @ weight.T + payload["encoder_bias"], 0.)
        e = u - (codes @ weight + payload["decoder_bias"])
        removal.append(delta), residual.append(payload["scaler_scale"] * e), removed.append(h[example] - delta), activation.append(a)
    removal, residual, removed, activation = map(np.asarray, (removal, residual, removed, activation))
    np.testing.assert_allclose(unit["example_activation"], activation, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(unit["removal_norm"], np.linalg.norm(removal, axis=1), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(unit["residual_norm"], np.linalg.norm(residual, axis=1), rtol=1e-4, atol=1e-6)
    assert unit["residual_norm"].min() > 1e-3, "The fixture residual must be nonzero for the preservation checks to mean anything"
    cached = world["output"]["probabilities"][np.searchsorted(world["output"]["row_index"], unit["example_row_index"]), 1]
    np.testing.assert_allclose(unit["probability_cached"], cached, rtol=1e-6)
    np.testing.assert_allclose(unit["probability_before"], cached, **CONTINUATION_PARITY)
    np.testing.assert_allclose(unit["probability_zero_removal"], cached, **CONTINUATION_PARITY)
    np.testing.assert_allclose(unit["probability_removed"], batched_death_probabilities(continuation, removed.astype(np.float32))[:, 1], rtol=1e-3, atol=1e-5)
    assert np.abs(unit["probability_removed"] - unit["probability_before"]).max() > 1e-3, "Removal must change some prediction"
    draws = GROUPS_CONTROL_DRAWS
    directions = control_directions(unit["interventions"]["control_seed"], examples, draws, h.shape[1])
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    controls = np.linalg.norm(removal, axis=1)[:, None, None] * directions
    np.testing.assert_allclose(unit["probability_control"], batched_death_probabilities(
        continuation, (h[:, None, :] - controls).reshape(-1, h.shape[1]).astype(np.float32))[:, 1].reshape(examples, draws), rtol=1e-3, atol=1e-5)
    assert unit["interventions"]["control_seed"] == _derived_seed("groups_controls", 42, 2007, 2008)
    label = labels[unit["example_row_index"]]
    assert np.array_equal(unit["example_label"], label)
    np.testing.assert_allclose(unit["loss_before"], (unit["probability_before"] - label) ** 2, rtol=1e-12)
    np.testing.assert_allclose(unit["loss_removed"], (unit["probability_removed"] - label) ** 2, rtol=1e-12)
    np.testing.assert_allclose(unit["loss_control"], (unit["probability_control"] - label[:, None]) ** 2, rtol=1e-12)
    gradients = world["output"]["gradients"][np.searchsorted(world["output"]["row_index"], unit["example_row_index"])].astype(np.float64)
    np.testing.assert_allclose(unit["first_order_change"], -(gradients * removal).sum(axis=1), rtol=1e-3, atol=1e-6)
    for position, name in enumerate(order):
        reference = world["references"][ids[name]]
        for column, seed in enumerate(unit["cav_seeds"]):
            chosen = unit["example_family"] == position
            if reference["cavs"][seed]["cav"] is None:
                assert np.isnan(unit["cav_projection"][chosen, column]).all()
            else:
                np.testing.assert_allclose(unit["cav_projection"][chosen, column], removal[chosen] @ np.asarray(reference["cavs"][seed]["cav"], dtype=np.float64), rtol=1e-4, atol=1e-6)
    gates = unit["interventions"]["gates"]
    assert gates["zero_removal_embedding_max_abs"] == 0.0 and gates["residual_form_max_abs"] < 1e-9
    assert gates["control_norm_max_abs_gap"] < 1e-9 and gates["activation_max_abs_gap"] < 1e-4 and gates["decoder_max_abs_gap"] < 1e-4
    assert gates["zero_removal_probability_max_abs"] < 1e-4 and gates["continuation_replay_max_abs"] < 1e-4
    assert unit["interventions"]["limit"] == GROUPS_EXAMPLE_LIMIT and unit["interventions"]["control_draws"] == draws

    # A_t is the strict frozen-cutoff mask of the shared helper on the year-t and reference fitting rows
    seen, original = [], strict_activation_mask

    def recording(values, target):
        seen.append(len(values))
        return original(values, target)

    try:
        globals()["strict_activation_mask"] = recording
        _groups_run(world, config, unit1, unit2)
    finally:
        globals()["strict_activation_mask"] = original
    assert seen.count(405) == 3 and seen.count(len(rows["fit_r"])) == 3, seen

    # a CAV repetition that is unavailable at the reference stays unavailable (NaN projection), never zero
    assert not np.isnan(unit["cav_projection"]).any()
    refs = {key: dict(value, cavs=dict(value["cavs"])) for key, value in world["references"].items()}
    refs[ids["beta"]]["cavs"][44] = {**refs[ids["beta"]]["cavs"][44], "valid": False, "failure_reason": "injected", "cav": None}
    refs[ids["beta"]]["identity"] = _reference_object_identity(refs[ids["beta"]])
    dropped = _groups_run(world, config, unit1, unit2, references=refs)
    chosen = dropped["example_family"] == order.index("beta")
    assert chosen.any() and np.isnan(dropped["cav_projection"][chosen, 2]).all() and not np.isnan(dropped["cav_projection"][chosen, :2]).any()
    assert np.array_equal(dropped["cav_projection"][~chosen], unit["cav_projection"][~chosen])
    assert interventions_table(dropped)[int(np.flatnonzero(chosen)[0])]["CAV projection 42"] is not None

    # consistency with the Idea 1 and Idea 2 payloads is recorded and exact on the rule side
    for name in order:
        family = unit["families"][order.index(name)]
        reference = world["references"][ids[name]]
        assert (family["consistency"]["idea1"] is not None) == reference["eligible"]["idea1"], name
        assert (family["consistency"]["idea2"] is not None) == reference["eligible"]["idea2"], name
        if family["consistency"]["idea1"] is not None:
            assert family["consistency"]["idea1"]["R_rows_differing"] == 0
            assert np.array_equal(unit1["mask_A"][unit1["family_ids"].index(ids[name])], unit["mask_A"][order.index(name)])
        if family["consistency"]["idea2"] is not None:
            assert np.array_equal(unit2["mask_R"][unit2["family_ids"].index(ids[name])], unit["mask_R"][order.index(name)])
    assert unit["outcomes_read"].startswith("reference-year fitting-role rows") and unit["classification_threshold"] == 0.5
    assert unit["baseline_source"].startswith("reference-year fitting-role rows")

    # determinism and tables
    again = _groups_run(world, config, unit1, unit2)
    _unit_arrays_equal(again, unit)
    table = groups_table(unit)
    assert len(table) == 12 and {row["group"] for row in table} == set(GROUP_NAMES) and sum(row["rows"] for row in table) == 3 * 405 and sum(row["rows"] == 0 for row in table) == 2
    assert len(interventions_table(unit)) == examples and len(coactivation_table(unit)) == 3 * 4 * 3
    assert groups_table(dict(unit, status="unavailable", reason="x")) == [{"t": 2008, "status": "unavailable:x"}]

    # no outcome leakage: nothing but the loss columns and reliability depends on labels; only allowed rows are read
    permitted = np.concatenate([rows["fit_r"], rows["eval_t"]])
    view = world["view"]
    flipped = view["y"].copy()
    flipped[rows["eval_t"]] = 1 - flipped[rows["eval_t"]]
    other = _groups_run(world, config, unit1, unit2, view=dict(view, y=flipped))
    for name in ("mask_A", "mask_R", "group_code", "feature_mean", "feature_std", "coactivation_positive_rows", "coactivation_activation_sum",
                 "reconstruction_mse", "example_family", "example_group", "example_rank", "example_row_index", "example_activation",
                 "removal_norm", "residual_norm", "probability_before", "probability_removed", "probability_control", "cav_projection"):
        assert np.array_equal(other[name], unit[name], equal_nan=True), f"{name} depends on outcomes"
    assert not np.array_equal(other["loss_before"], unit["loss_before"]) and not np.array_equal(other["example_label"], unit["example_label"])
    poisoned_view, poisoned = dict(view), dict(world["output"])
    outside = ~np.isin(np.arange(len(view["y"])), permitted)
    X = view["X"].copy()
    X[outside] = np.nan
    y = view["y"].copy()
    y[outside] = 1 - y[outside]
    poisoned_view.update(X=X, y=y)
    cached_rows = world["output"]["row_index"]
    for name, allowed in (("embeddings", permitted), ("gradients", rows["eval_t"]), ("probabilities", rows["eval_t"])):
        poisoned[name] = np.where(np.isin(cached_rows, allowed)[:, None], world["output"][name], np.nan).astype(np.float32)
    clean = _groups_run(world, config, unit1, unit2, view=poisoned_view, predictor_output=poisoned)
    _unit_arrays_equal(clean, unit)
    assert not set(poisoned_view) - set(PERFORMANCE_VIEW_KEYS)
    forbidden = {"outcome", "outcomes", "labels", "population", "y"}
    assert not set(inspect.signature(groups_interventions_unit).parameters) & forbidden
    selection_rows = rows["selection_t"]
    assert len(selection_rows) and not np.isin(selection_rows, permitted).any() and not np.isin(rows["fit_t"], permitted).any()

    # the cached predictions must be reproduced by the un-intervened embeddings
    shifted = dict(world["output"], probabilities=np.clip(world["output"]["probabilities"] + np.float32(.02), 0, 1).astype(np.float32))
    _refuses(lambda: _groups_run(world, config, unit1, unit2, predictor_output=shifted), "do not reproduce the cached predictions", RuntimeError)

    # refusals
    _refuses(lambda: _groups_run(world, config, unit1, unit2, view=world["rule_view"]), "read only")
    _refuses(lambda: _groups_run(world, config, unit1, unit2, t=2007), "follow-up")
    _refuses(lambda: _groups_run(world, config, unit1, unit2, device="auto"), "resolved device")
    _refuses(lambda: _groups_run(world, config, unit1, unit2, device="cuda"), "run device")
    _refuses(lambda: _groups_run(world, config, unit1, unit2, dictionaries={42: world["dictionaries"][42]}), "configured SAE seeds")
    _refuses(lambda: _groups_run(world, dict(config, classification_threshold=.4), unit1, unit2), "0.5 threshold")
    _refuses(lambda: _groups_run(world, config, unit1, unit2, idea1=dict(unit1, year=2009)), "different reference year or follow-up year")
    _refuses(lambda: _groups_run(world, config, unit1, unit2, idea1=dict(unit1, roles_identity="other")), "different roles")
    _refuses(lambda: _groups_run(world, config, unit1, unit2, idea1=dict(unit1, family_ids=unit1["family_ids"][:1])), "does not follow the families eligible")
    _refuses(lambda: _groups_run(world, config, unit1, unit2, idea2_frozen=dict(unit2, family_ids=list(reversed(unit2["family_ids"])))), "does not follow the families eligible")
    _refuses(lambda: _groups_run(world, config, unit1, unit2, idea1=dict(unit1, evaluation_row_index=unit1["evaluation_row_index"][::-1].copy())), "different evaluation rows")
    scrambled = {key: dict(value) for key, value in world["references"].items()}
    scrambled[ids["alpha"]]["target"] = {"valid": True, "cutoff": 0.1}
    _refuses(lambda: _groups_run(world, config, unit1, unit2, references=scrambled), "identity changed", RuntimeError)

    # disagreement with the Idea 1 / Idea 2 payloads is an error, never silently reconciled
    mask_a = unit1["mask_A"].copy()
    near = np.abs(oracle["families"]["alpha"]["activation"] - 1.0) > 1e-3
    mask_a[0, np.flatnonzero(near)[0]] ^= True
    _refuses(lambda: _groups_run(world, config, unit1, unit2, idea1=dict(unit1, mask_A=mask_a)), "Idea 1 membership", RuntimeError)
    mask_r = unit1["mask_R"].copy()
    mask_r[1, 3] ^= True
    _refuses(lambda: _groups_run(world, config, unit1, unit2, idea1=dict(unit1, mask_R=mask_r)), "Idea 1 membership", RuntimeError)
    mask_two = unit2["mask_R"].copy()
    mask_two[0, 5] ^= True
    _refuses(lambda: _groups_run(world, config, unit1, unit2, idea2_frozen=dict(unit2, mask_R=mask_two)), "Idea 2 rule membership", RuntimeError)

    # unavailability inside the Idea 2 payload (the cache record stays ok) is consumed per family; the yearly rule status of Idea 1 is not
    broken = copy.deepcopy(unit2["families"])
    for family in broken:
        if family["family_id"] == ids["beta"]:
            family.update(status="unavailable", reason="empty_rule_subgroup")
    partial = _groups_run(world, config, unit1, dict(unit2, families=broken))
    assert partial["status"] == "ok" and partial["family_ids"] == [ids["alpha"], ids["epsilon"]]
    by_id = {family["family_id"]: family for family in partial["families"]}
    assert by_id[ids["beta"]]["status"] == "unavailable" and by_id[ids["beta"]]["reason"] == "idea2_frozen_unavailable:empty_rule_subgroup"
    assert by_id[ids["beta"]]["array_index"] is None and "groups" not in by_id[ids["beta"]]
    assert [family["family_id"] for family in partial["families"]] == [ids[name] for name in order]
    assert partial["mask_A"].shape == (2, 405) and partial["group_code"].shape == (2, 405) and partial["feature_mean"].shape[0] == 2
    assert partial["coactivation_positive_rows"].shape == (2, 4, 2) and partial["interventions"]["examples"] < examples
    assert groups_table(partial)[0]["group"] == "R_and_A" and any(row.get("status", "").startswith("unavailable:idea2_frozen") for row in groups_table(partial))
    assert np.array_equal(partial["group_code"][0], unit["group_code"][0]) and np.array_equal(partial["group_code"][1], unit["group_code"][2])
    assert by_id[ids["alpha"]]["groups"] == unit["families"][0]["groups"]
    quiet_rule = copy.deepcopy(unit1["families"])
    for family in quiet_rule:
        family["yearly_rule"] = {"status": "unavailable", "reason": "insufficient_inner_rule_partitions"}
    same = _groups_run(world, config, dict(unit1, families=quiet_rule), unit2)
    _unit_arrays_equal(same, unit)
    everything = copy.deepcopy(unit2["families"])
    for family in everything:
        family.update(status="unavailable", reason="empty_current_year_fitting_population")
    none = _groups_run(world, config, unit1, dict(unit2, families=everything))
    assert none["status"] == "ok" and none["family_ids"] == [ids["alpha"]] and none["mask_A"].shape == (1, 405)

    # batching never changes the continuation result
    h_all = world["E"][evaluation].astype(np.float32)
    for batch in (1, 7, 405, 4096):
        np.testing.assert_allclose(_death_probabilities(continuation, torch.as_tensor(h_all), batch).numpy(),
                                   batched_death_probabilities(continuation, h_all)[:, 1], rtol=1e-6, atol=1e-7)
    assert _death_probabilities(continuation, torch.zeros((0, h_all.shape[1])), 4).shape == (0,)
    return unit


def _check_groups_empty_and_idle(world, config):
    """A follow-up year without evaluation rows and a year without eligible families are explicit, not zero-filled."""
    import numpy as np
    unit1_empty = idea1_year_unit(world["view"], 2007, 2009, world["roles"], world["output"], world["family_output"], world["references"],
                                  world["dictionaries"], config, "cpu", backend=IDEA1_BACKEND, workers=1)
    unit2_empty = idea2_frozen_year_unit(world["rule_view"], 2007, 2009, world["roles"], world["output"], world["family_output"],
                                         world["references"], world["dictionaries"], config, "cpu")
    empty = _groups_run(world, config, unit1_empty, unit2_empty, t=2009)
    assert empty["status"] == "unavailable" and empty["reason"] == GROUPS_UNAVAILABLE_EMPTY and empty["_rows"] == 0
    assert [family["status"] for family in empty["families"]] == ["unavailable"] * 3 and empty["family_ids"] == []
    assert {family["reason"] for family in empty["families"]} == {GROUPS_UNAVAILABLE_EMPTY}
    assert empty["evaluation_row_index"].shape == (0,) and "mask_A" not in empty
    assert groups_table(empty) == [{"t": 2009, "status": f"unavailable:{GROUPS_UNAVAILABLE_EMPTY}"}]
    assert interventions_table(empty)[0]["status"].startswith("unavailable") and coactivation_table(empty)[0]["status"].startswith("unavailable")
    # no eligible family at all
    none_eligible = {key: dict(value, eligible={"idea1": False, "idea2": False}) for key, value in world["references"].items()}
    first = idea1_year_unit(world["view"], 2007, 2008, world["roles"], world["output"], world["family_output"], none_eligible,
                            world["dictionaries"], config, "cpu", backend=IDEA1_BACKEND, workers=1)
    second = idea2_frozen_year_unit(world["rule_view"], 2007, 2008, world["roles"], world["output"], world["family_output"], none_eligible,
                                    world["dictionaries"], config, "cpu")
    idle = _groups_run(world, config, first, second, references=none_eligible)
    assert idle["status"] == "ok" and idle["family_ids"] == [] and idle["families"] == [] and idle["_rows"] == 405
    assert idle["mask_A"].shape == (0, 405) and idle["group_code"].shape == (0, 405) and idle["interventions"]["examples"] == 0
    assert idle["feature_mean"].shape == idle["feature_std"].shape == (0, 4, world["X"].shape[1]) and idle["coactivation_positive_rows"].shape == (0, 4, 0)
    assert idle["example_row_index"].shape == (0,) and idle["probability_control"].shape == (0, GROUPS_CONTROL_DRAWS)
    assert len(idle["not_computed"]) == 5 and groups_table(idle) == []
    assert interventions_table(idle) == [] and coactivation_table(idle) == []
    return idle


def _check_groups_cache(config, world, unit, unit1, unit2):
    """D(r)/groups_interventions(t) through the cache: dependencies, serialized continuation use, resume, size, failure and
    unavailability propagation, payload-level unavailability of an Idea 2 family consumed while the cache record stays ok."""
    import numpy as np
    population, roles = world["population"], world["roles"]
    view, rule_view = world["view"], world["rule_view"]
    dimension = world["dimension"]
    base = Path(os.environ["RPI_OUT_DIR"])
    base.mkdir(parents=True, exist_ok=True)
    free = lambda: 100 * 2 ** 30
    rules = {"frozen": [{key: value for key, value in entry.items() if key != "rule"} for entry in world["frozen"].values()],
             "families": [{"family_id": name, "status": "frozen"} for name in world["frozen"]], "_rows": 1}
    module = globals()
    with tempfile.TemporaryDirectory(dir=base) as directory:
        work = Path(directory)

        def open_cache(name, until="cavs"):
            cache = UnitCache(work / name, config, "source-fixture", free_bytes=free)
            assert cache.run("P/2007", lambda: dict(world["output"]))["status"] == "ok"
            for seed in config["sae_seeds"]:
                assert cache.run(f"S/{seed}/2007", lambda seed=seed: dict(world["payloads"][seed]))["status"] == "ok"
            assert cache.run("D/2007/families", lambda: dict(world["family_output"]))["status"] == "ok"
            assert cache.run("D/2007/rules", lambda: dict(rules))["status"] == "ok"
            if until == "cavs":
                assert cache.run("D/2007/cavs_selection", lambda: dict(world["selection_unit"]))["status"] == "ok"
            return cache

        cache = open_cache("cache")
        _refuses(lambda: run_groups_interventions_unit(cache, rule_view, roles, config, "cpu", 2008), "read only")
        _refuses(lambda: run_groups_interventions_unit(cache, view, roles, dict(config, epochs=3), "cpu", 2008), "differs")
        _refuses(lambda: run_groups_interventions_unit(cache, view, roles, config, "cpu", 2007), "follow-up")
        blocked = run_groups_interventions_unit(cache, view, roles, config, "cpu", 2008)
        assert (blocked["status"], blocked["reason"]) == ("unavailable", "dep_not_run:D/2007/idea1/2008")
        assert run_idea1_unit(cache, view, roles, config, "cpu", 2008, workers=1, backend=IDEA1_BACKEND)["status"] == "ok"
        blocked = run_groups_interventions_unit(cache, view, roles, config, "cpu", 2008)
        assert (blocked["status"], blocked["reason"]) == ("unavailable", "dep_not_run:D/2007/idea2_frozen/2008")
        assert run_idea2_frozen_unit(cache, rule_view, roles, config, "cpu", 2008)["status"] == "ok"

        calls = {"load_continuation": 0}
        originals = {name: module[name] for name in ("load_continuation", "fit_frozen_predictor", "capture_frozen_predictions", "predictor_unit")}

        def counting(*arguments, **keywords):
            calls["load_continuation"] += 1
            return originals["load_continuation"](*arguments, **keywords)

        def refuse(name):
            def raiser(*arguments, **keywords):
                raise AssertionError(f"{name} was called")
            return raiser

        try:
            module.update(load_continuation=counting, fit_frozen_predictor=refuse("fit_frozen_predictor"),
                          capture_frozen_predictions=refuse("capture_frozen_predictions"), predictor_unit=refuse("predictor_unit"))
            record = run_groups_interventions_unit(cache, view, roles, config, "cpu", 2008)
            again = run_groups_interventions_unit(cache, view, roles, config, "cpu", 2008)
        finally:
            module.update(originals)
        assert calls["load_continuation"] == 1, "The serialized continuation of P(r) is loaded once per unit run"
        assert record["status"] == "ok" and record["key"] == "D/2007/groups_interventions/2008" and record["stage"] == "D.groups_interventions", record
        assert record["dependencies"] == {key: cache._latest[key]["id"] for key in ("P/2007", "D/2007/cavs_selection", "D/2007/idea1/2008", "D/2007/idea2_frozen/2008")}
        assert record["rows"] == unit["_rows"] and again["id"] == record["id"], "A complete D unit was rebuilt"
        stored = cache.load("D/2007/groups_interventions/2008")
        _unit_arrays_equal(stored, unit)
        assert _unit_json(stored) == _unit_json(unit) and stored["mask_A"].dtype == bool and stored["group_code"].dtype == np.uint8
        used = sum(path.stat().st_size for path in cache._unit_path("D/2007/groups_interventions/2008").rglob("*") if path.is_file())
        assert 0 < used <= projected_groups_interventions_bytes(config), (used, projected_groups_interventions_bytes(config))
        assert projected_groups_interventions_bytes(config) < 64 * 2**20
        assert run_idea1_unit(cache, view, roles, config, "cpu", 2009, workers=1, backend=IDEA1_BACKEND)["status"] == "ok"
        assert run_idea2_frozen_unit(cache, rule_view, roles, config, "cpu", 2009)["status"] == "ok"
        empty = run_groups_interventions_unit(cache, view, roles, config, "cpu", 2009)
        assert empty["status"] == "ok" and empty["rows"] == 0
        assert cache.load("D/2007/groups_interventions/2009")["status"] == "unavailable"
        assert cache.completeness_table(["D/2007/groups_interventions/2008", "D/2007/groups_interventions/2009", "D/2008/groups_interventions/2009"]) == [
            {"unit": "D/2007/groups_interventions/2008", "status": "complete"}, {"unit": "D/2007/groups_interventions/2009", "status": "complete"},
            {"unit": "D/2008/groups_interventions/2009", "status": "not_run (stopped)"}]

        # interrupted run: the failed attempt leaves no unit, the retry completes
        retry = open_cache("cache-retry")
        assert run_idea1_unit(retry, view, roles, config, "cpu", 2008, workers=1, backend=IDEA1_BACKEND)["status"] == "ok"
        assert run_idea2_frozen_unit(retry, rule_view, roles, config, "cpu", 2008)["status"] == "ok"
        try:
            module["load_continuation"] = lambda *arguments, **keywords: (_ for _ in ()).throw(FloatingPointError("injected"))
            crashed = run_groups_interventions_unit(retry, view, roles, config, "cpu", 2008)
        finally:
            module["load_continuation"] = originals["load_continuation"]
        assert crashed["status"] == "failed" and crashed["reason"] == "FloatingPointError: injected"
        assert not retry._unit_path("D/2007/groups_interventions/2008").exists() and not retry.is_complete("D/2007/groups_interventions/2008")
        assert run_groups_interventions_unit(retry, view, roles, config, "cpu", 2008)["status"] == "ok"
        _unit_arrays_equal(retry.load("D/2007/groups_interventions/2008"), unit)

        # failed dependencies make the unit unavailable without running it
        for name, breaker, expected in (
                ("cache-cavs", "cavs", "dep_failed:D/2007/cavs_selection"), ("cache-idea1", "idea1", "dep_failed:D/2007/idea1/2008"),
                ("cache-frozen", "frozen", "dep_failed:D/2007/idea2_frozen/2008")):
            broken = open_cache(name, until="rules")
            if breaker == "cavs":
                assert broken.run("D/2007/cavs_selection", lambda: (_ for _ in ()).throw(FloatingPointError("injected")))["status"] == "failed"
            else:
                assert broken.run("D/2007/cavs_selection", lambda: dict(world["selection_unit"]))["status"] == "ok"
            if breaker == "idea1":
                assert broken.run("D/2007/idea1/2008", lambda: (_ for _ in ()).throw(FloatingPointError("injected")))["status"] == "failed"
            elif breaker != "cavs":
                assert run_idea1_unit(broken, view, roles, config, "cpu", 2008, workers=1, backend=IDEA1_BACKEND)["status"] == "ok"
            if breaker == "frozen":
                assert broken.run("D/2007/idea2_frozen/2008", lambda: (_ for _ in ()).throw(FloatingPointError("injected")))["status"] == "failed"
            record = run_groups_interventions_unit(broken, view, roles, config, "cpu", 2008)
            assert (record["status"], record["reason"]) == ("unavailable", expected), record
            assert not broken._unit_path("D/2007/groups_interventions/2008").exists()
            assert broken.completeness_table(["D/2007/groups_interventions/2008"]) == [
                {"unit": "D/2007/groups_interventions/2008", "status": f"unavailable:{expected}"}]

        # an unavailable family inside a cache-ok Idea 2 payload is consumed per family
        doctored = open_cache("cache-doctored")
        assert run_idea1_unit(doctored, view, roles, config, "cpu", 2008, workers=1, backend=IDEA1_BACKEND)["status"] == "ok"
        real = run_idea2_frozen_unit(doctored, rule_view, roles, config, "cpu", 2008)
        assert real["status"] == "ok"
        payload = doctored.load("D/2007/idea2_frozen/2008")
        for family in payload["families"]:
            if family["family_id"] == world["identifiers"]["beta"]:
                family.update(status="unavailable", reason="empty_rule_subgroup")
        patched = UnitCache(work / "cache-doctored2", config, "source-fixture", free_bytes=free)
        for key in ("P/2007", "S/42/2007", "S/135/2007", "S/215/2007", "D/2007/families", "D/2007/rules", "D/2007/cavs_selection", "D/2007/idea1/2008"):
            source = doctored.load(key)
            assert patched.run(key, lambda source=source: dict(source))["status"] == "ok"
        assert patched.run("D/2007/idea2_frozen/2008", lambda: dict(payload))["status"] == "ok"
        record = run_groups_interventions_unit(patched, view, roles, config, "cpu", 2008)
        assert record["status"] == "ok", "Payload-level unavailability must not fail or block the unit"
        result = patched.load("D/2007/groups_interventions/2008")
        by_id = {family["family_id"]: family for family in result["families"]}
        assert by_id[world["identifiers"]["beta"]]["reason"] == "idea2_frozen_unavailable:empty_rule_subgroup"
        assert result["family_ids"] == [world["identifiers"]["alpha"], world["identifiers"]["epsilon"]] and record["rows"] == 405


def _check_groups_cuda(config, world, unit, unit1, unit2):
    """CUDA-only tolerance block: the same unit on the GPU with the continuation and dictionaries moved to the device."""
    import numpy as np
    import torch
    on_cpu = _check_group_primitives("cpu")
    _check_group_primitives("cuda")
    left, right = _check_residual_preserving_removal("cpu"), _check_residual_preserving_removal("cuda")
    np.testing.assert_allclose(left["removed"], right["removed"], rtol=1e-9, atol=1e-9)
    arrays, meta = export_continuation(world["continuation"])
    continuation = rebuild_continuation(arrays, meta, "cuda")
    gpu = _groups_run(world, config, unit1, unit2, dictionaries=_groups_dictionaries(world["payloads"], "cuda"), continuation=continuation, device="cuda")
    assert gpu["status"] == "ok" and gpu["family_ids"] == unit["family_ids"]
    for name in ("mask_A", "mask_R", "group_code", "example_family", "example_group", "example_rank", "example_row_index"):
        assert np.array_equal(gpu[name], unit[name]), name
    for name in ("feature_mean", "feature_std", "coactivation_activation_sum", "reconstruction_mse", "removal_norm", "residual_norm",
                 "probability_before", "probability_removed", "probability_control", "loss_removed", "first_order_change"):
        np.testing.assert_allclose(gpu[name], unit[name], rtol=1e-4, atol=1e-5, equal_nan=True, err_msg=name)
    np.testing.assert_allclose(gpu["cav_projection"], unit["cav_projection"], rtol=1e-4, atol=1e-5, equal_nan=True)
    assert [family["groups"] for family in gpu["families"]] == [family["groups"] for family in unit["families"]]
    assert torch.cuda.memory_allocated() < 256 * 2 ** 20, "GPU memory was not released after the groups unit"


def run_group_intervention_checks(notebook_path="concept_subgroup_drift_pilot.ipynb"):
    """Four groups (partition, counts, performance, overlap), original-feature distributions, coactivation, deterministic
    examples and residual-preserving, verified decoder-contribution removal with norm-matched controls and zero-removal parity
    through the serialized continuation, on planted follow-up data with an independent numpy oracle."""
    presets = _notebook_presets(json.loads(Path(notebook_path).read_text()))
    config = resolve_config(presets, "v0")
    assert config["profile"] == "v0" and config["bootstrap_replicates"] == 10 and config["joblib_workers"] <= 4
    config = _selection_config(config)
    _check_groups_settings(config)
    _check_group_primitives("cpu")
    _check_residual_preserving_removal("cpu")
    world = _groups_world(config)
    assert world["selection_unit"]["eligible"]["idea1"] == [world["identifiers"]["alpha"], world["identifiers"]["epsilon"]]
    assert world["selection_unit"]["eligible"]["idea2"] == [world["identifiers"]["beta"], world["identifiers"]["epsilon"]]
    unit1, unit2 = _groups_dependencies(config, world)
    unit = _check_groups_unit(config, world, unit1, unit2)
    _check_groups_empty_and_idle(world, config)
    _check_groups_cache(config, world, unit, unit1, unit2)
    if _cuda_available():
        _check_groups_cuda(config, world, unit, unit1, unit2)
    else:
        print("CUDA group/intervention equality checks skipped: CUDA unavailable here; GPU validation pending")
    print("Group fixtures passed: four-group partition and counts, performance at the reference baseline, feature distributions, "
          "coactivation, deterministic examples, residual-preserving removal with controls, zero-removal parity, cache resume and propagation")


TEMPORAL_LAW = (-0.05, -0.2, 0.2)
TEMPORAL_REFERENCE_CONCEPTS = ("activation_prevalence", "medoid_retention", "medoid_prevalence", "medoid_magnitude")
TEMPORAL_FIXTURE_FAMILIES = (("fa", ("idea1",)), ("fb", ("idea2",)), ("fc", ("idea1", "idea2")))


def _temporal_pilot_config(last=2013):
    return {"reference_year_candidates": list(range(2007, last + 1)), "t0_only_reference_years": [last],
            "sae_seeds": [42, 135, 215], "primary_cav_seed": 42, "label_availability": "end_of_year"}


def _check_temporal_calendar(config):
    """The common calendar boundary comes from the configuration only."""
    pilot = _temporal_pilot_config(2015)
    calendar = temporal_calendar(pilot)
    assert calendar["target_years"] == list(range(2008, 2016)) and calendar["holdout_years"] == [2014, 2015]
    assert calendar["training_years"] == list(range(2008, 2014)) and calendar["last_year"] == 2015
    assert max(calendar["training_years"]) < min(calendar["holdout_years"])
    assert temporal_calendar(pilot, 1)["holdout_years"] == [2015] and temporal_calendar(pilot, 1)["training_years"] == list(range(2008, 2015))
    small = temporal_calendar(_temporal_pilot_config(2013))
    assert small["holdout_years"] == [2012, 2013] and small["training_years"] == [2008, 2009, 2010, 2011]
    v0 = temporal_calendar(config)
    assert v0["target_years"] == [2008, 2009] and v0["holdout_years"] == [2008, 2009] and v0["training_years"] == []
    assert temporal_calendar(dict(pilot, reference_year_candidates=[2007, 2009, 2010], t0_only_reference_years=[2010]))["target_years"] == [2009, 2010]
    for bad in (0, -1, 1.5, "2", None):
        _refuses(lambda bad=bad: temporal_calendar(pilot, bad), "holdout_years")
    _refuses(lambda: temporal_calendar(dict(pilot, label_availability="next_year")), "end of each calendar year")
    _refuses(lambda: temporal_calendar({key: value for key, value in pilot.items() if key != "label_availability"}), "end of each")


def _temporal_oracle_ols(rows, width):
    """Normal-equation least squares, independent of numpy's lstsq/rank code path."""
    import numpy as np
    design = np.asarray([[1.0, *row[:width]] for row in rows])
    return np.linalg.solve(design.T @ design, design.T @ np.asarray([row[-1] for row in rows]))


def _check_temporal_math():
    """OLS prerequisites and values, Pearson support and pooling, against closed forms."""
    import numpy as np
    rng = np.random.default_rng(7)
    x = rng.normal(size=(9, 2))
    y = 0.3 + x @ np.asarray([0.5, -0.2]) + 0.05 * rng.normal(size=9)
    fit = temporal_ols(x, y, 2)
    beta = _temporal_oracle_ols([(*row, value) for row, value in zip(x, y)], 2)
    assert fit["status"] == "ok" and fit["n"] == 9 and fit["parameters"] == 3 and fit["residual_df"] == 6 and fit["rank"] == 3
    np.testing.assert_allclose(fit["coefficients"], beta, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(_ols_predict(fit, x), np.column_stack([np.ones(9), x]) @ beta, rtol=1e-10)
    one = temporal_ols(x[:, :1], y, 1)
    slope = np.cov(x[:, 0], y, bias=True)[0, 1] / x[:, 0].var()
    np.testing.assert_allclose(one["coefficients"], [y.mean() - slope * x[:, 0].mean(), slope], rtol=1e-10)
    exact = temporal_ols(x, 1.0 + 2.0 * x[:, 0] - 3.0 * x[:, 1], 2)
    np.testing.assert_allclose(exact["coefficients"], [1.0, 2.0, -3.0], atol=1e-10)
    short = temporal_ols(x[:3], y[:3], 2)
    assert (short["status"], short["reason"], short["residual_df"]) == ("unavailable", "insufficient_residual_df", 0) and short["coefficients"] is None
    assert temporal_ols(x[:4], y[:4], 2)["status"] == "ok" and temporal_ols(x[:2, :1], y[:2], 1)["reason"] == "insufficient_residual_df"
    assert temporal_ols(np.empty((0, 1)), [], 1)["reason"] == "insufficient_residual_df"
    constant = temporal_ols(np.column_stack([x[:, 0], np.full(9, 0.4)]), y, 2)
    assert (constant["status"], constant["reason"], constant["rank"]) == ("unavailable", "rank_deficient", 2)
    collinear = temporal_ols(np.column_stack([x[:, 0], 2 * x[:, 0]]), y, 2)
    assert collinear["reason"] == "rank_deficient"
    bad = x.copy()
    bad[0, 0] = np.nan
    assert temporal_ols(bad, y, 2)["reason"] == "nonfinite_inputs"
    value, reason = _pearson(x[:, 0], y)
    np.testing.assert_allclose(value, np.corrcoef(x[:, 0], y)[0, 1], rtol=1e-12)
    assert _pearson([1, 2], [3, 4]) == (None, "fewer_than_three_transitions") and _pearson([1, 1, 1], [1, 2, 3]) == (None, "constant_series")
    assert _pooled([1.0, 3.0], [1, 3]) == 2.5 and _pooled([1.0, None], [4, 0]) == 1.0 and _pooled([None, None], [0, 0]) is None
    assert _pooled([2.0, 5.0], [0, 0]) is None and _pooled([float("nan"), 4.0], [2, 2]) == 4.0
    assert _agreed([None, 0.5, 0.5]) == (0.5, None) and _agreed([0.5, 0.6]) == (None, "inconsistent_reference_value")
    assert _agreed([None]) == (None, "no_reference_value") and _agreed([]) == (None, "no_reference_value")


def _temporal_series(config):
    """Planted law delta_loss[t] = a + b L[t] + c C[t] with C the first concept of the subgroup; other concepts are noise."""
    import numpy as np
    last, a, b, c = int(max(config["reference_year_candidates"])), *TEMPORAL_LAW
    series = {}
    for r in config["reference_year_candidates"]:
        for index, (family, ideas) in enumerate(_TEMPORAL_FIXTURE_FAMILIES_LOCAL):
            for idea in ideas:
                names, years = TEMPORAL_CONCEPTS[dict(IDEAS)[idea]], list(range(r, last + 1))
                rng = np.random.default_rng(1000 * r + 10 * index + (idea == "idea2"))
                concepts = {name: {t: float(v) for t, v in zip(years, rng.uniform(0.2, 0.8, len(years)))} for name in names}
                loss = {r: 0.15 + 0.01 * (r - 2007)}
                for t in years[:-1]:
                    loss[t + 1] = loss[t] + a + b * loss[t] + c * concepts[names[0]][t]
                series[(r, family, idea)] = {"loss": loss, "concepts": concepts}
    return series


_TEMPORAL_FIXTURE_FAMILIES_LOCAL = TEMPORAL_FIXTURE_FAMILIES


def _temporal_rows(r, t):
    import numpy as np
    return (t - 2000) * 1000 + np.arange(20 * (r - 2007), 20 * (r - 2007) + 60, dtype=np.int64)


def _temporal_sizes(t):
    """Disjoint group sizes of one year: R and A, R and not A, not R and A, neither."""
    return [10 + t % 3, 12 + t % 4, 8 + t % 2, 30 - t % 3 - t % 4 - t % 2]


def _temporal_units(config, series):
    """Payload-shaped unit contents consumed by the temporal analysis, built from the planted series (extra keys that a real
    payload carries are omitted; unread values are poisoned with 9.0 or a wrong CAV seed to expose wrong reads)."""
    import numpy as np
    units, groups_names = {}, list(GROUP_NAMES)
    for r in config["reference_year_candidates"]:
        follow = idea1_follow_up_years(config, r)
        if not follow:
            continue
        entries = {key: value for key, value in series.items() if key[0] == r}
        units[f"D/{r}/reference_performance"] = {"reference_year": r, "_rows": 60, "rows": [
            {"family_id": f, "idea": idea, "subgroup": dict(IDEAS)[idea], "evaluation_rows": 60 - 7,
             "metrics": {"brier": value["loss"][r], "unavailable_reasons": {}}} for (_, f, idea), value in sorted(entries.items())]}
        for t in follow:
            rows, sizes = _temporal_rows(r, t), _temporal_sizes(t)
            n_a, n_r = sizes[0] + sizes[2], sizes[0] + sizes[1]
            idea1_families, idea2_families, group_families = [], [], []
            mse = np.full((len(_TEMPORAL_FIXTURE_FAMILIES_LOCAL), 4), 0.2)
            for position, (f, ideas) in enumerate(_TEMPORAL_FIXTURE_FAMILIES_LOCAL):
                a_series, r_series = entries.get((r, f, "idea1")), entries.get((r, f, "idea2"))
                target_r = r_series["loss"][t] if r_series else 0.21
                if a_series:
                    a_names = TEMPORAL_CONCEPTS["A"]
                    idea1_families.append({
                        "family_id": f, "reference_statistics": {"activation_prevalence": a_series["concepts"][a_names[0]][r]},
                        "observation": {"activation_prevalence": a_series["concepts"][a_names[0]][t],
                                        "subgroup_influence": {"42": {"tcav": a_series["concepts"][a_names[1]][t], "unavailable_reason": None},
                                                               "43": {"tcav": 0.99, "unavailable_reason": None}}},
                        "original_rule_transfer": {"f2": a_series["concepts"][a_names[2]][t]},
                        "reliability": {"A": {"brier": a_series["loss"][t], "records": n_a, "unavailable_reasons": {}},
                                        "R": {"brier": target_r, "records": n_r, "unavailable_reasons": {}}}})
                if r_series:
                    r_names = TEMPORAL_CONCEPTS["R"]
                    values = r_series["concepts"]
                    idea2_families.append({"family_id": f, "status": "ok", "reason": None, "members": [
                        {"seed": 42, "factor": 3, "is_medoid": False, "reference": {"positive_prevalence": 9.0, "magnitude_mean": 9.0},
                         "year": {"positive_prevalence": 9.0, "magnitude_mean": 9.0, "unavailable": None}},
                        {"seed": 135, "factor": 1, "is_medoid": True,
                         "reference": {"positive_prevalence": values[r_names[1]][r], "magnitude_mean": values[r_names[2]][r]},
                         "year": {"positive_prevalence": values[r_names[1]][t], "magnitude_mean": values[r_names[2]][t], "unavailable": None},
                         "retention": {"year_rate": values[r_names[0]][t], "reference_rate": values[r_names[0]][r]}}]})
                a_mse = a_series["concepts"]["reconstruction_mse"][t] if a_series else 0.3
                r_mse = r_series["concepts"]["reconstruction_mse"][t] if r_series else 0.4
                b0 = target_r + 0.01
                b1 = (target_r * n_r - b0 * sizes[0]) / sizes[1]
                m0 = 0.5
                mse[position, 0] = m0
                mse[position, 1] = (r_mse * n_r - m0 * sizes[0]) / sizes[1]
                mse[position, 2] = (a_mse * n_a - m0 * sizes[0]) / sizes[2]
                briers = [b0, b1, 0.25, 0.35]
                group_families.append({"family_id": f, "status": "ok", "reason": None, "array_index": position, "groups": {
                    name: {"rows": sizes[index], "reliability": {"brier": briers[index], "records": sizes[index]}}
                    for index, name in enumerate(groups_names)}})
            units[f"D/{r}/idea1/{t}"] = {"year": t, "reference_year": r, "_rows": len(rows), "evaluation_row_index": rows,
                                         "families": idea1_families, "family_ids": [item["family_id"] for item in idea1_families]}
            units[f"D/{r}/idea2_frozen/{t}"] = {"year": t, "reference_year": r, "_rows": len(rows), "evaluation_row_index": rows,
                                                "families": idea2_families, "family_ids": [item["family_id"] for item in idea2_families]}
            units[f"D/{r}/groups_interventions/{t}"] = {"status": "ok", "reason": None, "year": t, "reference_year": r, "_rows": len(rows),
                                                        "evaluation_row_index": rows, "families": group_families, "reconstruction_mse": mse}
    return units


def _forecast_oracle(loss, concept, r, last, calendar):
    """Held-out MAE of persistence, history OLS and history+concept OLS from plain dictionaries."""
    import numpy as np
    rows = [(t, t + 1, loss[t], None if concept is None else concept[t], loss[t + 1] - loss[t]) for t in range(r, last)
            if t in loss and t + 1 in loss and (concept is None or t in concept)]
    train = [row for row in rows if row[1] in calendar["training_years"]]
    test = [row for row in rows if row[1] in calendar["holdout_years"]]
    result = {"n_train": len(train), "n_test": len(test), "persistence": None, "history": None, "history_concept": None}
    if not test:
        return result
    delta = np.asarray([row[4] for row in test])
    result["persistence"] = float(np.abs(delta).mean())
    for name, width in (("history", 1), ("history_concept", 2)):
        if name == "history_concept" and concept is None or len(train) - (width + 1) < 1:
            continue
        beta = _temporal_oracle_ols([(*(row[2:2 + 1] if width == 1 else (row[2], row[3])), row[4]) for row in train], width)
        design = np.asarray([[1.0, row[2]] if width == 1 else [1.0, row[2], row[3]] for row in test])
        result[name] = float(np.abs(delta - design @ beta).mean())
    return result


def _check_against_oracle(result, config, series, missing_loss=(), missing_concept=()):
    """Every trajectory-concept forecast of the analysis equals the independent oracle on the same, gap-free, transitions."""
    import numpy as np
    calendar, last = temporal_calendar(config), int(max(config["reference_year_candidates"]))
    checked = 0
    for entry in result["forecasts"]:
        r, family, idea = entry["reference_year"], entry["family_id"], entry["trajectory"].split("/")[2]
        assert entry["trajectory"] == f"{r}/{family}/{idea}"
        planted = series[(r, family, idea)]
        loss = {t: value for t, value in planted["loss"].items() if (r, family, idea, t) not in missing_loss}
        concept = entry["forecast"]["concept"]
        values = None
        if concept is not None:
            values = {t: value for t, value in planted["concepts"][concept].items()
                      if (concept in TEMPORAL_REFERENCE_CONCEPTS or t != r) and (r, family, idea, concept, t) not in missing_concept}
        expected, forecast = _forecast_oracle(loss, values, r, last, calendar), entry["forecast"]
        assert forecast["training"]["n"] == expected["n_train"] and forecast["holdout"]["n"] == expected["n_test"], (entry["trajectory"], concept)
        assert (forecast["status"] == "ok") == (expected["n_test"] > 0), (entry["trajectory"], concept)
        for name in ("persistence", "history", "history_concept"):
            got = forecast["mae"].get(name)
            if expected[name] is None:
                assert got is None, (entry["trajectory"], concept, name, got)
            else:
                np.testing.assert_allclose(got, expected[name], rtol=1e-9, atol=1e-12, err_msg=f"{entry['trajectory']} {concept} {name}")
                checked += 1
    return checked


def _same_rows_oracle(config, missing=()):
    """Python-set row overlap per target year, with the same row-source preference as the analysis."""
    result = {}
    for t in sorted({t for r in config["reference_year_candidates"] for t in idea1_follow_up_years(config, r)}):
        members = [r for r in config["reference_year_candidates"] if t in idea1_follow_up_years(config, r) and (r, t) not in missing]
        pairs = []
        for i, first in enumerate(members):
            for second in members[i + 1:]:
                a, b = set(_temporal_rows(first, t).tolist()), set(_temporal_rows(second, t).tolist())
                pairs.append({"reference_years": [first, second], "shared_rows": len(a & b), "union_rows": len(a | b), "jaccard": len(a & b) / len(a | b)})
        result[t] = {"reference_years": members, "pairs": pairs}
    return result


def _check_temporal_world(world_config):
    """Full planted world: trajectories, consecutive transitions, calendar holdout, chronology, boundary, overlap, determinism."""
    import copy
    import numpy as np
    config = world_config
    series = _temporal_series(config)
    units = _temporal_units(config, series)
    calendar = temporal_calendar(config)
    result = temporal_analysis(config, units)
    assert result["calendar"] == calendar and calendar["holdout_years"] == [2012, 2013] and calendar["training_years"] == [2008, 2009, 2010, 2011]
    assert sorted(item["key"] for item in result["trajectories"]) == sorted(
        f"{r}/{f}/{idea}" for r in range(2007, 2013) for f, ideas in TEMPORAL_FIXTURE_FAMILIES for idea in ideas)
    assert not [item for item in result["trajectories"] if item["reference_year"] == 2013], "T0-only reference years have no transitions"
    assert all(len(item["years"]) == 2013 - item["reference_year"] + 1 for item in result["trajectories"])
    checked = _check_against_oracle(result, config, series)
    assert checked > 140, checked
    # planted law: the exact history+concept model recovers the coefficients and beats history and persistence on every holdout
    by_key = {(entry["trajectory"], entry["forecast"]["concept"]): entry["forecast"] for entry in result["forecasts"]}
    exact = by_key[("2007/fc/idea2", "medoid_retention")]
    np.testing.assert_allclose(exact["models"]["history_concept"]["coefficients"], TEMPORAL_LAW, atol=1e-9)
    assert exact["mae"]["history_concept"] < 1e-9 < exact["mae"]["history"] and exact["mae"]["persistence"] > 1e-4
    assert exact["training"] == {"n": 4, "target_years": [2008, 2009, 2010, 2011]} and exact["holdout"]["target_years"] == [2012, 2013]
    assert exact["models"]["history_concept"]["residual_df"] == 1 and exact["models"]["history"]["residual_df"] == 2
    assert exact["paired"]["history_concept_minus_history"] < 0 and exact["paired"]["history_concept_minus_persistence"] < 0
    np.testing.assert_allclose(exact["paired"]["history_minus_persistence"], exact["mae"]["history"] - exact["mae"]["persistence"], atol=1e-12)
    # prerequisites: three training transitions fit the history model only; two fit neither; none leaves persistence alone
    assert by_key[("2008/fb/idea2", "medoid_retention")]["models"]["history"]["status"] == "ok"
    short = by_key[("2008/fb/idea2", "medoid_retention")]["models"]["history_concept"]
    assert (short["status"], short["reason"], short["residual_df"]) == ("unavailable", "insufficient_residual_df", 0)
    two = by_key[("2009/fb/idea2", None)]
    assert two["models"]["history"]["reason"] == "insufficient_residual_df" and two["mae"]["history"] is None and two["mae"]["persistence"] is not None
    none = by_key[("2011/fb/idea2", None)]
    assert none["training"]["n"] == 0 and none["holdout"]["n"] == 2 and none["status"] == "ok" and none["mae"]["history"] is None
    last_only = by_key[("2012/fb/idea2", None)]
    assert last_only["holdout"] == {"n": 1, "target_years": [2013]} and last_only["mae"]["persistence"] is not None
    # reference values are absent for the concepts without a reference-year quantity: the first transition is excluded, not imputed
    reconstruction = by_key[("2007/fc/idea2", "reconstruction_mse")]
    assert reconstruction["excluded"] == [{"origin": 2007, "target": 2008, "reason": "concept_missing_at_origin:reconstruction_mse"}]
    assert reconstruction["training"]["n"] == 3 and reconstruction["models"]["history_concept"]["reason"] == "insufficient_residual_df"
    assert by_key[("2007/fc/idea1", "tcav_primary")]["training"]["n"] == 3
    # the cross-year concept at the reference year equals the planted reference quantity
    trajectory = next(item for item in result["trajectories"] if item["key"] == "2007/fc/idea2")
    assert trajectory["years"][2007]["loss"] == series[(2007, "fc", "idea2")]["loss"][2007]
    for name in TEMPORAL_REFERENCE_CONCEPTS[1:]:
        assert trajectory["years"][2007]["concepts"][name] == series[(2007, "fc", "idea2")]["concepts"][name][2007]
    assert trajectory["years"][2007]["concepts"]["reconstruction_mse"] is None and trajectory["years"][2007]["concept_reasons"]["reconstruction_mse"]
    for year in range(2008, 2014):
        for name, value in series[(2007, "fc", "idea2")]["concepts"].items():
            np.testing.assert_allclose(trajectory["years"][year]["concepts"][name], value[year], rtol=1e-12)
        np.testing.assert_allclose(trajectory["years"][year]["loss"], series[(2007, "fc", "idea2")]["loss"][year], rtol=1e-12)
    # only actual consecutive years, chronologically ordered, training strictly before the held-out years
    for item in result["trajectories"]:
        transitions = temporal_transitions(item, 2013)
        assert [(t["origin"], t["target"]) for t in transitions] == [(y, y + 1) for y in range(item["reference_year"], 2013)]
        assert all(t["status"] == "ok" for t in transitions)
    for entry in result["forecasts"]:
        forecast = entry["forecast"]
        years = forecast["training"]["target_years"]
        assert years == sorted(years) and set(years) <= set(calendar["training_years"])
        assert set(forecast["holdout"]["target_years"]) <= set(calendar["holdout_years"])
        if years and forecast["holdout"]["target_years"]:
            assert max(years) < min(forecast["holdout"]["target_years"])
        assert [row["target"] for row in forecast["transitions"]] == forecast["holdout"]["target_years"]
    transitions = temporal_transitions(trajectory, 2013)
    tampered = dict(calendar, training_years=calendar["training_years"] + [2012])
    _refuses(lambda: temporal_forecast(transitions, tampered, None), "strictly before", error=RuntimeError)
    _refuses(lambda: temporal_forecast(transitions, dict(calendar, holdout_years=[2013]), None), "outside the calendar")
    # no holdout-year information is used in fitting: poisoning years >= 2012 leaves every fit untouched
    poisoned = copy.deepcopy(series)
    rng = np.random.default_rng(99)
    for planted in poisoned.values():
        for year in (2012, 2013):
            if year in planted["loss"]:
                planted["loss"][year] = float(rng.uniform(0.5, 2.0))
            for values in planted["concepts"].values():
                if year in values:
                    values[year] = float(rng.uniform(-3, 3))
    other = temporal_analysis(config, _temporal_units(config, poisoned))
    changed = 0
    for left, right in zip(result["forecasts"], other["forecasts"]):
        assert left["trajectory"] == right["trajectory"] and left["forecast"]["concept"] == right["forecast"]["concept"]
        for name, fit in left["forecast"]["models"].items():
            assert fit == right["forecast"]["models"][name], (left["trajectory"], name)
        assert left["forecast"]["training"] == right["forecast"]["training"]
        first_left = [row for row in left["forecast"]["transitions"] if row["target"] == 2012]
        first_right = [row for row in right["forecast"]["transitions"] if row["target"] == 2012]
        for a, b in zip(first_left, first_right):
            assert a["prediction"] == b["prediction"], "the first held-out forecast uses only its origin-year history"
            changed += a["delta_loss"] != b["delta_loss"]
    assert changed > 80, changed
    sensitive = copy.deepcopy(series)
    for planted in sensitive.values():
        if 2010 in planted["loss"]:
            planted["loss"][2010] += 0.2
    moved = temporal_analysis(config, _temporal_units(config, sensitive))
    assert any(a["forecast"]["models"]["history"] != b["forecast"]["models"]["history"]
               for a, b in zip(result["forecasts"], moved["forecasts"])), "fits must react to training-year changes"
    # the boundary is configuration-defined: an interrupted run does not move it
    stopped = {key: value for key, value in units.items() if not key.endswith("/2013")}
    interrupted = temporal_analysis(config, stopped)
    assert interrupted["calendar"] == calendar and interrupted["holdout_reference_years"]["2013"] == []
    shortened = {(entry["trajectory"], entry["forecast"]["concept"]): entry["forecast"] for entry in interrupted["forecasts"]}
    assert shortened[("2007/fc/idea2", None)]["holdout"] == {"n": 1, "target_years": [2012]}
    assert shortened[("2007/fc/idea2", None)]["models"]["history"] == by_key[("2007/fc/idea2", None)]["models"]["history"]
    assert shortened[("2012/fc/idea2", None)]["status"] == "unavailable" and shortened[("2012/fc/idea2", None)]["reason"] == "no_holdout_transitions"
    assert shortened[("2012/fc/idea2", None)]["mae"] == {}
    labels = {row["unit"]: row["status"] for row in interrupted["availability"]}
    assert labels["D/2007/idea1/2013"] == labels["D/2012/groups_interventions/2013"] == "not_run (stopped)"
    # the same holdout years and the same reference experiments are used by every trajectory (common boundary)
    assert result["holdout_reference_years"] == {"2012": list(range(2007, 2012)), "2013": list(range(2007, 2013))}
    # overlap: shared evaluation rows per target year and pair against Python sets; never independent replicates
    expected = _same_rows_oracle(config)
    overlap = result["overlap"]
    assert overlap["independent_replicates"] is False and "independent replicates" in overlap["note"]
    assert sorted(overlap["by_year"]) == sorted(expected)
    for year, item in expected.items():
        got = overlap["by_year"][year]
        assert got["reference_years"] == item["reference_years"] and len(got["pairs"]) == len(item["pairs"])
        for left, right in zip(got["pairs"], item["pairs"]):
            assert left["reference_years"] == right["reference_years"] and left["shared_rows"] == right["shared_rows"]
            assert left["union_rows"] == right["union_rows"] and abs(left["jaccard"] - right["jaccard"]) < 1e-12
        assert set(got["sources"].values()) == {"idea1"}
    pair = {tuple(p["reference_years"]): p for p in overlap["by_year"][2012]["pairs"]}
    assert pair[(2007, 2008)]["shared_rows"] == 40 and pair[(2007, 2009)]["shared_rows"] == 20 and pair[(2007, 2010)]["shared_rows"] == 0
    assert pair[(2007, 2010)]["jaccard"] == 0 and abs(pair[(2007, 2008)]["jaccard"] - 40 / 80) < 1e-12
    # concurrent / before / after patterns on one trajectory against hand computation
    pattern = next(item for item in result["patterns"] if item["trajectory"] == "2007/fc/idea2" and item["concept"] == "medoid_retention")
    loss, concept = series[(2007, "fc", "idea2")]["loss"], series[(2007, "fc", "idea2")]["concepts"]["medoid_retention"]
    assert [row["origin"] for row in pattern["rows"]] == list(range(2007, 2013))
    for row in pattern["rows"]:
        t = row["origin"]
        np.testing.assert_allclose(row["delta_loss"], loss[t + 1] - loss[t], atol=1e-14)
        assert row["deterioration"] == ("deterioration" if loss[t + 1] - loss[t] > 0 else "no_deterioration")
        np.testing.assert_allclose(row["concept_change_concurrent"], concept[t + 1] - concept[t], atol=1e-14)
        assert (row["concept_change_before"] is None) == (t == 2007) and (row["concept_change_after"] is None) == (t == 2012)
        if t > 2007:
            np.testing.assert_allclose(row["concept_change_before"], concept[t] - concept[t - 1], atol=1e-14)
        if t < 2012:
            np.testing.assert_allclose(row["concept_change_after"], concept[t + 2] - concept[t + 1], atol=1e-14)
    assert {row["deterioration"] for row in pattern["rows"]} == {"deterioration", "no_deterioration"}
    deltas = [loss[t + 1] - loss[t] for t in range(2007, 2013)]
    np.testing.assert_allclose(pattern["concurrent"]["pearson"], np.corrcoef([concept[t + 1] - concept[t] for t in range(2007, 2013)], deltas)[0, 1], rtol=1e-10)
    assert pattern["concurrent"]["n"] == 6 and pattern["concurrent"]["unavailable"] is None
    recon = next(item for item in result["patterns"] if item["trajectory"] == "2007/fc/idea2" and item["concept"] == "reconstruction_mse")
    assert recon["concurrent"]["n"] == 5 and recon["rows"][0]["concept_change_concurrent"] is None and recon["rows"][1]["concept_change_before"] is None
    # summary: counts and means only, never a combined score or an alarm threshold
    assert result["policy"]["composite_score"] is None and result["policy"]["alarm_threshold"] is None
    assert result["summary"]["independent_replicates"] is False and result["summary"]["uncertainty"]["status"] == "unavailable"
    row = next(item for item in result["summary"]["rows"] if item["idea"] == "idea2" and item["concept"] == "medoid_retention")
    group = [entry["forecast"] for entry in result["forecasts"] if entry["idea"] == "idea2" and entry["forecast"]["concept"] == "medoid_retention"
             and entry["forecast"]["mae"].get("history_concept") is not None]
    assert row["trajectories"] == 12 and row["with_holdout"] == 12 and row["with_fitted_models"] == len(group) == 2
    np.testing.assert_allclose(row["mean_mae"]["history_concept"], np.mean([f["mae"]["history_concept"] for f in group]))
    assert row["concept_lower_mae_than_history"] == 2
    tables = temporal_tables(result)
    assert set(tables) == {"forecast", "concurrent", "patterns", "overlap", "availability", "summary"}
    assert len(tables["forecast"]) == len(result["forecasts"]) and len(tables["overlap"]) == sum(len(item["pairs"]) for item in overlap["by_year"].values())
    assert json.dumps(tables, allow_nan=False) and json.dumps(result, sort_keys=True, allow_nan=False)
    # deterministic and independent of the order of the units
    shuffled = dict(reversed(list(units.items())))
    assert json.dumps(temporal_analysis(config, shuffled), sort_keys=True) == json.dumps(result, sort_keys=True)
    assert temporal_analysis(config, copy.deepcopy(units))["forecasts"] == result["forecasts"]
    return series, units, result


def _check_temporal_missing(config, series, units):
    """Missing years are labelled and never imputed or bridged; payload-level unavailability of every consumed payload."""
    import copy
    import numpy as np
    unit = copy.deepcopy(units)
    # P1 groups payload unavailable for the year
    unit["D/2007/groups_interventions/2010"] = {"status": "unavailable", "reason": "empty_evaluation_population", "_rows": 0, "families": [],
                                                "evaluation_row_index": np.zeros(0, dtype=np.int64)}
    # P2 Idea 1 unit failed; P3 groups unit never ran
    unit["D/2007/idea1/2011"] = "failed:RuntimeError: boom"
    del unit["D/2008/groups_interventions/2009"]
    # P4 groups family unavailable because its Idea 2 family was; P5 Idea 2 payload family unavailable
    group = unit["D/2009/groups_interventions/2010"]["families"][1]
    assert group["family_id"] == "fb"
    group.update(status="unavailable", reason="idea2_frozen_unavailable:empty_current_year_fitting_population", array_index=None)
    unit["D/2008/idea2_frozen/2011"]["families"][1].update(status="unavailable", reason="empty_evaluation_population")
    # P6 Idea 1 payload without evaluation rows; P7 Idea 1 A subgroup empty
    unit["D/2009/idea1/2012"]["_rows"] = 0
    unit["D/2010/idea1/2011"]["families"][0]["reliability"]["A"].update(brier=None, records=0, unavailable_reasons={"brier": "empty_subgroup"})
    # P8 one empty R group: the pooled values come from the other group and equal the planted ones
    sizes = _temporal_sizes(2012)
    groups = unit["D/2011/groups_interventions/2012"]
    family = groups["families"][2]
    assert family["family_id"] == "fc"
    target = series[(2011, "fc", "idea2")]
    family["groups"]["R_and_A"].update(rows=0, reliability={"brier": None, "records": 0})
    family["groups"]["R_and_not_A"].update(rows=sizes[0] + sizes[1], reliability={"brier": target["loss"][2012], "records": sizes[0] + sizes[1]})
    groups["reconstruction_mse"][2, 0] = np.nan
    groups["reconstruction_mse"][2, 1] = target["concepts"]["reconstruction_mse"][2012]
    groups["reconstruction_mse"][2, 2] = series[(2011, "fc", "idea1")]["concepts"]["reconstruction_mse"][2012]
    # P9 both R groups empty
    for name in ("R_and_A", "R_and_not_A"):
        unit["D/2010/groups_interventions/2012"]["families"][1]["groups"][name].update(rows=0, reliability={"brier": None, "records": 0})
    # P12 groups payload unavailable for a reason of its own although it carries rows
    unit["D/2010/groups_interventions/2013"].update(status="unavailable", reason="synthetic_reason", _rows=7)
    # P10 the Idea 1 and groups R losses disagree; P11 the reference value differs between two Idea 1 payloads
    unit["D/2007/idea1/2009"]["families"][1]["reliability"]["R"]["brier"] += 1e-3
    unit["D/2007/idea1/2008"]["families"][0]["reference_statistics"]["activation_prevalence"] += 0.05
    result = temporal_analysis(config, unit)
    trajectories = {item["key"]: item for item in result["trajectories"]}
    labels = {row["unit"]: row["status"] for row in result["availability"]}
    assert labels["D/2007/groups_interventions/2010"] == "unavailable:empty_evaluation_population"
    assert labels["D/2007/idea1/2011"] == "failed:RuntimeError: boom" and labels["D/2008/groups_interventions/2009"] == "not_run (stopped)"
    assert labels["D/2009/idea1/2012"] == "unavailable:empty_evaluation_population" and labels["D/2007/idea1/2010"] == "ok"
    assert labels["D/2010/groups_interventions/2013"] == "unavailable:synthetic_reason"
    years = trajectories["2010/fc/idea2"]["years"]
    assert years[2013]["loss"] is None and years[2013]["reason"] == "groups:unavailable:synthetic_reason"
    for key in ("2007/fb/idea2", "2007/fc/idea2"):
        years = trajectories[key]["years"]
        assert years[2010]["loss"] is None and years[2010]["reason"] == "groups:unavailable:empty_evaluation_population", years[2010]
        assert years[2011]["loss"] is not None
    years = trajectories["2007/fb/idea2"]["years"]
    transitions = temporal_transitions(trajectories["2007/fb/idea2"], 2013)
    assert [(t["origin"], t["target"]) for t in transitions if t["status"] == "ok"] == [(2007, 2008), (2008, 2009), (2011, 2012), (2012, 2013)]
    assert all(t["target"] - t["origin"] == 1 for t in transitions)
    assert {t["origin"]: t["reason"] for t in transitions if t["status"] == "skipped"} == {
        2009: "target_loss_missing:groups:unavailable:empty_evaluation_population",
        2010: "origin_loss_missing:groups:unavailable:empty_evaluation_population"}
    forecast = next(e["forecast"] for e in result["forecasts"] if e["trajectory"] == "2007/fb/idea2" and e["forecast"]["concept"] is None)
    assert forecast["training"] == {"n": 2, "target_years": [2008, 2009]} and forecast["models"]["history"]["reason"] == "insufficient_residual_df"
    assert [item["origin"] for item in forecast["excluded"]] == [2009, 2010] and forecast["holdout"]["n"] == 2
    for key in ("2007/fa/idea1", "2007/fc/idea1"):
        years = trajectories[key]["years"]
        assert years[2011]["loss"] is None and years[2011]["reason"] == "idea1:failed:RuntimeError: boom"
        assert years[2011]["concepts"]["reconstruction_mse"] is not None
        assert years[2011]["concept_reasons"]["activation_prevalence"] == "idea1:failed:RuntimeError: boom"
    assert trajectories["2007/fc/idea2"]["years"][2011]["loss"] is not None, "the R loss comes from the groups, not from a failed Idea 1 unit"
    years = trajectories["2008/fb/idea2"]["years"]
    assert years[2009]["loss"] is None and years[2009]["reason"] == "groups:not_run (stopped)"
    assert trajectories["2008/fa/idea1"]["years"][2009]["loss"] is not None
    assert trajectories["2008/fa/idea1"]["years"][2009]["concept_reasons"]["reconstruction_mse"] == "groups:not_run (stopped)"
    years = trajectories["2009/fb/idea2"]["years"]
    assert years[2010]["loss"] is None and years[2010]["reason"] == "groups:idea2_frozen_unavailable:empty_current_year_fitting_population"
    assert years[2010]["concepts"]["medoid_retention"] is not None, "the Idea 2 payload itself is fine"
    years = trajectories["2008/fc/idea2"]["years"]
    assert years[2011]["loss"] is not None and years[2011]["concepts"]["medoid_retention"] is None
    assert years[2011]["concept_reasons"]["medoid_retention"] == "idea2:empty_evaluation_population" and years[2011]["concepts"]["reconstruction_mse"] is not None
    years = trajectories["2009/fa/idea1"]["years"]
    assert years[2012]["loss"] is None and years[2012]["reason"] == "idea1:unavailable:empty_evaluation_population"
    years = trajectories["2010/fa/idea1"]["years"]
    assert years[2011]["loss"] is None and years[2011]["reason"] == "idea1:empty_subgroup"
    years = trajectories["2011/fc/idea2"]["years"]
    np.testing.assert_allclose(years[2012]["loss"], series[(2011, "fc", "idea2")]["loss"][2012], rtol=1e-12)
    np.testing.assert_allclose(years[2012]["concepts"]["reconstruction_mse"], series[(2011, "fc", "idea2")]["concepts"]["reconstruction_mse"][2012], rtol=1e-12)
    np.testing.assert_allclose(trajectories["2011/fc/idea1"]["years"][2012]["concepts"]["reconstruction_mse"],
                               series[(2011, "fc", "idea1")]["concepts"]["reconstruction_mse"][2012], rtol=1e-12)
    years = trajectories["2010/fb/idea2"]["years"]
    assert years[2012]["loss"] is None and years[2012]["reason"] == "groups:empty_subgroup"
    assert years[2012]["concepts"]["reconstruction_mse"] is None and years[2012]["concept_reasons"]["reconstruction_mse"] == "groups:empty_subgroup"
    years = trajectories["2007/fc/idea2"]["years"]
    assert years[2009]["loss"] is None and years[2009]["reason"].startswith("inconsistent_sources:"), years[2009]
    assert trajectories["2007/fa/idea1"]["years"][2009]["loss"] is not None
    first = trajectories["2007/fa/idea1"]["years"][2007]
    assert first["concepts"]["activation_prevalence"] is None and first["concept_reasons"]["activation_prevalence"] == "inconsistent_reference_value"
    assert trajectories["2007/fc/idea1"]["years"][2007]["concepts"]["activation_prevalence"] is not None
    # every trajectory-concept forecast still equals the oracle on the surviving consecutive pairs
    four = (("fa", "idea1"), ("fb", "idea2"), ("fc", "idea1"), ("fc", "idea2"))
    missing_loss = {(2007, "fb", "idea2", 2010), (2007, "fc", "idea2", 2010), (2007, "fa", "idea1", 2011), (2007, "fc", "idea1", 2011),
                    (2008, "fb", "idea2", 2009), (2008, "fc", "idea2", 2009), (2009, "fb", "idea2", 2010), (2009, "fa", "idea1", 2012),
                    (2009, "fc", "idea1", 2012), (2010, "fa", "idea1", 2011), (2010, "fb", "idea2", 2012), (2007, "fc", "idea2", 2009),
                    (2010, "fb", "idea2", 2013), (2010, "fc", "idea2", 2013)}
    missing_concept = {(2007, f, idea, "reconstruction_mse", 2010) for f, idea in four}
    missing_concept |= {(2007, f, "idea1", name, 2011) for f in ("fa", "fc") for name in TEMPORAL_CONCEPTS["A"][:3]}
    missing_concept |= {(2008, f, idea, "reconstruction_mse", 2009) for f, idea in four}
    missing_concept |= {(2010, f, idea, "reconstruction_mse", 2013) for f, idea in four}
    missing_concept |= {(2009, "fb", "idea2", "reconstruction_mse", 2010), (2010, "fb", "idea2", "reconstruction_mse", 2012),
                        (2007, "fa", "idea1", "activation_prevalence", 2007)}
    missing_concept |= {(2008, "fc", "idea2", name, 2011) for name in TEMPORAL_CONCEPTS["R"][:3]}
    missing_concept |= {(2009, f, "idea1", name, 2012) for f in ("fa", "fc") for name in TEMPORAL_CONCEPTS["A"][:3]}
    checked = _check_against_oracle(result, config, series, missing_loss, missing_concept)
    assert checked > 100, checked
    overlap = result["overlap"]["by_year"]
    assert overlap[2011]["sources"]["2007"] == "idea2_frozen" and overlap[2011]["sources"]["2008"] == "idea1"
    assert overlap[2010]["reference_years"] == [2007, 2008, 2009] and overlap[2012]["sources"]["2009"] == "idea2_frozen"
    expected = _same_rows_oracle(config)
    for year, item in expected.items():
        for left, right in zip(overlap[year]["pairs"], item["pairs"]):
            assert left["shared_rows"] == right["shared_rows"], "a fallback row source gives the same evaluation rows"
    # every consumed payload absent: labelled, no trajectory, no crash, nothing zero-filled
    empty = temporal_analysis(config, {})
    assert empty["trajectories"] == [] and empty["forecasts"] == [] and empty["overlap"]["by_year"] == {}
    assert {row["status"] for row in empty["availability"]} == {"not_run (stopped)"} and empty["summary"]["rows"] == []
    only = temporal_analysis(config, {"D/2007/reference_performance": units["D/2007/reference_performance"]})
    track = next(item for item in only["trajectories"] if item["key"] == "2007/fc/idea2")
    assert track["years"][2007]["loss"] is not None and all(track["years"][t]["loss"] is None for t in range(2008, 2014))
    assert all(entry["forecast"]["status"] == "unavailable" for entry in only["forecasts"])
    nothing = temporal_analysis(config, {"D/2007/reference_performance": "failed:RuntimeError: gate"})
    assert nothing["trajectories"] == []
    assert {row["unit"]: row["status"] for row in nothing["availability"]}["D/2007/reference_performance"] == "failed:RuntimeError: gate"
    _refuses(lambda: temporal_analysis(config, {"D/2007/idea1/2008": 5}), "payload dict or a status label", error=TypeError)
    # a year outside the configured grid is a labelled gap, not a crash and not bridged
    gappy = dict(config, reference_year_candidates=[2007, 2009, 2010, 2011, 2012, 2013])
    kept = {key: value for key, value in units.items() if not key.startswith("D/2008/") and not key.endswith("/2008")}
    gapped = temporal_analysis(gappy, kept)
    track = next(item for item in gapped["trajectories"] if item["key"] == "2007/fc/idea2")
    assert track["years"][2008]["loss"] is None and track["years"][2008]["reason"] == "year_not_in_configuration"
    assert [(t["origin"], t["target"]) for t in temporal_transitions(track, 2013) if t["status"] == "ok"] == [(2009, 2010), (2010, 2011), (2011, 2012), (2012, 2013)]
    assert gapped["calendar"]["target_years"] == [2009, 2010, 2011, 2012, 2013] and gapped["calendar"]["holdout_years"] == [2012, 2013]
    return result


def _check_temporal_cache(config, series_config):
    """Analysis-stage computation over a real unit cache: complete units load, failed/unavailable/not-run units are labelled,
    the manifest and the unit graph are untouched."""
    import copy
    base = Path(os.environ["RPI_OUT_DIR"])
    base.mkdir(parents=True, exist_ok=True)
    series = _temporal_series(series_config)
    payloads = _temporal_units(series_config, series)
    assert not [key for key in unit_graph(config) if "temporal" in key] and not [stage for stage in UNIT_STAGES if "temporal" in stage]
    free = lambda: 100 * 2 ** 30
    with tempfile.TemporaryDirectory(dir=base) as directory:
        cache = UnitCache(Path(directory) / "cache", config, "source-fixture", free_bytes=free)

        def boom():
            raise RuntimeError("boom")

        for key in unit_order(config):
            if key == "D/2008/idea2_frozen/2009":
                continue
            fn = boom if key == "D/2007/idea1/2008" else (lambda key=key: dict(payloads.get(key, {"_rows": 1})))
            cache.run(key, fn)
        status = {row["unit"]: row["status"] for row in cache.completeness_table()}
        assert status["D/2007/idea1/2008"] == "failed:RuntimeError: boom" and status["D/2008/idea2_frozen/2009"] == "not_run (stopped)"
        assert status["D/2007/groups_interventions/2008"] == "unavailable:dep_failed:D/2007/idea1/2008"
        assert status["D/2008/groups_interventions/2009"] == "unavailable:dep_not_run:D/2008/idea2_frozen/2009"
        manifest = (cache.directory / "units.jsonl").read_bytes()
        units = temporal_units(cache, config)
        assert sorted(units) == sorted(["D/2007/reference_performance", "D/2007/idea1/2008", "D/2007/idea2_frozen/2008", "D/2007/groups_interventions/2008",
                                        "D/2007/idea1/2009", "D/2007/idea2_frozen/2009", "D/2007/groups_interventions/2009",
                                        "D/2008/reference_performance", "D/2008/idea1/2009", "D/2008/idea2_frozen/2009", "D/2008/groups_interventions/2009"])
        assert units["D/2007/idea1/2008"] == "failed:RuntimeError: boom" and units["D/2008/idea2_frozen/2009"] == "not_run (stopped)"
        assert units["D/2007/groups_interventions/2008"] == "unavailable:dep_failed:D/2007/idea1/2008"
        assert isinstance(units["D/2007/idea1/2009"], dict) and units["D/2007/idea1/2009"]["evaluation_row_index"].dtype.kind == "i"
        result = run_temporal_analysis(cache, config)
        direct = {key: (payloads[key] if isinstance(value, dict) else value) for key, value in units.items()}
        assert json.dumps(result, sort_keys=True, allow_nan=False) == json.dumps(temporal_analysis(config, direct), sort_keys=True, allow_nan=False)
        labels = {row["unit"]: row["status"] for row in result["availability"]}
        assert labels["D/2007/idea1/2008"] == "failed:RuntimeError: boom" and labels["D/2007/idea1/2009"] == "ok"
        assert labels["D/2008/groups_interventions/2009"] == "unavailable:dep_not_run:D/2008/idea2_frozen/2009"
        trajectories = {item["key"]: item for item in result["trajectories"]}
        assert trajectories["2007/fa/idea1"]["years"][2008]["reason"] == "idea1:failed:RuntimeError: boom"
        assert trajectories["2007/fa/idea1"]["years"][2009]["loss"] is not None
        assert trajectories["2007/fb/idea2"]["years"][2008]["reason"] == "groups:unavailable:dep_failed:D/2007/idea1/2008"
        assert trajectories["2008/fa/idea1"]["years"][2009]["loss"] is not None and trajectories["2008/fb/idea2"]["years"][2009]["loss"] is None
        assert result["calendar"]["holdout_years"] == [2008, 2009] and result["calendar"]["training_years"] == []
        assert all(entry["forecast"]["training"]["n"] == 0 for entry in result["forecasts"])
        assert (cache.directory / "units.jsonl").read_bytes() == manifest, "the temporal analysis must not write units"
        _refuses(lambda: run_temporal_analysis(cache, dict(config, epochs=3)), "differs")
    return result


def _check_temporal_contract(config):
    """Real payloads (the T17 to T19 fixture world and the reference performance unit) feed the analysis; every extracted number
    is compared with the raw planted data or the payload field it is defined by."""
    import numpy as np
    world = _groups_world(config)
    unit1, unit2 = _groups_dependencies(config, world)
    groups = _groups_run(world, config, unit1, unit2)
    performance = reference_performance_unit(world["view"], 2007, world["roles"], world["output"], world["family_output"],
                                             world["references"], world["dictionaries"], config, "cpu")
    units = {"D/2007/reference_performance": performance, "D/2007/idea1/2008": unit1, "D/2007/idea2_frozen/2008": unit2,
             "D/2007/groups_interventions/2008": groups}
    result = temporal_analysis(config, units)
    ids = world["identifiers"]
    keys = sorted(item["key"] for item in result["trajectories"])
    assert keys == sorted([f"2007/{ids['alpha']}/idea1", f"2007/{ids['beta']}/idea2", f"2007/{ids['epsilon']}/idea1", f"2007/{ids['epsilon']}/idea2"]), keys
    oracle = _groups_oracle(world, config)
    rows = oracle["rows"]
    population = world["population"]
    labels = population["y"][rows].astype(np.float64)
    probability = world["output"]["probabilities"][np.searchsorted(world["output"]["row_index"], rows), 1].astype(np.float64)
    brier = lambda flag: float(((probability[flag] - labels[flag]) ** 2).mean())
    by_family = {ids[name]: name for name in ("alpha", "beta", "epsilon")}
    stored = {(row["family_id"], row["idea"]): row for row in performance["rows"]}
    for item in result["trajectories"]:
        name, idea = by_family[item["family_id"]], item["idea"]
        expected = oracle["families"][name]
        years = item["years"]
        assert sorted(years) == [2007, 2008, 2009] and item["subgroup"] == dict(IDEAS)[idea]
        reference = stored[(item["family_id"], idea)]
        if reference["metrics"]["brier"] is None:
            assert years[2007]["loss"] is None and years[2007]["reason"] == "reference_performance:empty_subgroup" and reference["evaluation_rows"] == 0
        else:
            np.testing.assert_allclose(years[2007]["loss"], reference["metrics"]["brier"], rtol=1e-12)
            assert years[2007]["rows"] == reference["evaluation_rows"]
        assert (item["family_id"], idea) != (ids["epsilon"], "idea1") or years[2007]["loss"] is None, "the planted empty reference subgroup stays labelled"
        flag = expected["A"] if idea == "idea1" else expected["R"]
        np.testing.assert_allclose(years[2008]["loss"], brier(flag), rtol=1e-10)
        assert years[2008]["rows"] == int(flag.sum()) and years[2008]["reason"] is None
        assert years[2009]["loss"] is None
        assert years[2009]["reason"] == ("idea1:not_run (stopped)" if idea == "idea1" else "groups:not_run (stopped)")
        position = oracle["order"].index(name)
        counts = [int((expected["code"] == index).sum()) for index in range(4)]
        picks = {"A": (0, 2), "R": (0, 1)}[item["subgroup"]]
        weights = [counts[i] for i in picks]
        errors = [float(groups["reconstruction_mse"][position, i]) for i in picks]
        pooled = sum(w * e for w, e in zip(weights, errors) if w) / sum(weights)
        np.testing.assert_allclose(years[2008]["concepts"]["reconstruction_mse"], pooled, rtol=1e-12)
        family1 = next((f for f in unit1["families"] if f["family_id"] == item["family_id"]), None)
        family2 = next((f for f in unit2["families"] if f["family_id"] == item["family_id"]), None)
        if idea == "idea1":
            observation = family1["observation"]
            assert years[2008]["concepts"]["activation_prevalence"] == observation["activation_prevalence"] == float(expected["A"].mean())
            assert years[2007]["concepts"]["activation_prevalence"] == family1["reference_statistics"]["activation_prevalence"]
            assert years[2008]["concepts"]["tcav_primary"] == observation["subgroup_influence"]["42"]["tcav"]
            assert years[2008]["concepts"]["rule_transfer_f2"] == family1["original_rule_transfer"]["f2"]
        else:
            medoid = [member for member in family2["members"] if member["is_medoid"]]
            assert len(medoid) == 1 and years[2008]["concepts"]["medoid_retention"] == medoid[0]["retention"]["year_rate"]
            assert years[2008]["concepts"]["medoid_prevalence"] == medoid[0]["year"]["positive_prevalence"]
            assert years[2008]["concepts"]["medoid_magnitude"] == medoid[0]["year"]["magnitude_mean"]
            assert years[2007]["concepts"]["medoid_retention"] == medoid[0]["retention"]["reference_rate"]
            assert years[2007]["concepts"]["medoid_prevalence"] == medoid[0]["reference"]["positive_prevalence"]
    both = [item for item in result["trajectories"] if item["family_id"] == ids["epsilon"]]
    assert len(both) == 2
    transitions = temporal_transitions(next(item for item in both if item["idea"] == "idea2"), 2009)
    assert [(t["origin"], t["target"], t["status"]) for t in transitions] == [(2007, 2008, "ok"), (2008, 2009, "skipped")]
    transitions = temporal_transitions(next(item for item in both if item["idea"] == "idea1"), 2009)
    assert [t["status"] for t in transitions] == ["skipped", "skipped"] and transitions[0]["reason"].startswith("origin_loss_missing:reference_performance:")
    assert result["overlap"]["by_year"][2008]["reference_years"] == [2007] and result["overlap"]["by_year"][2008]["sources"] == {"2007": "idea1"}
    assert np.array_equal(unit1["evaluation_row_index"], rows)
    assert json.dumps(temporal_tables(result), allow_nan=False)
    # a differing R loss between the Idea 1 and the groups payload (same rows, same frozen rule) is refused, not averaged
    import copy
    tampered = copy.deepcopy(units)
    epsilon = next(f for f in tampered["D/2007/idea1/2008"]["families"] if f["family_id"] == ids["epsilon"])
    epsilon["reliability"]["R"]["brier"] += 1e-3
    mutated = temporal_analysis(config, tampered)
    track = next(item for item in mutated["trajectories"] if item["key"] == f"2007/{ids['epsilon']}/idea2")
    assert track["years"][2008]["loss"] is None and track["years"][2008]["reason"].startswith("inconsistent_sources")
    return result


def run_temporal_checks(notebook_path="concept_subgroup_drift_pilot.ipynb"):
    """Temporal analysis on synthetic payloads: consecutive-year transitions with labelled (never imputed or bridged) missing
    years, strictly chronological fitting before a common configured calendar holdout, no holdout information in fitting,
    persistence and OLS baselines against an independent numpy oracle, overlap reporting, payload-level unavailability of
    the Idea 1, Idea 2 and groups payloads, the analysis-stage placement over a unit cache, and real unit payloads."""
    presets = _notebook_presets(json.loads(Path(notebook_path).read_text()))
    config = resolve_config(presets, "v0")
    assert config["profile"] == "v0" and config["label_availability"] == "end_of_year"
    _check_temporal_calendar(config)
    _check_temporal_math()
    world = _temporal_pilot_config(2013)
    series, units, _ = _check_temporal_world(world)
    _check_temporal_missing(world, series, units)
    _check_temporal_cache(config, config)
    _check_temporal_contract(_selection_config(config))
    print("Temporal fixtures passed: calendar boundary from configuration, consecutive transitions with labelled gaps, chronological "
          "fitting without holdout information, persistence/OLS against numpy, overlap reporting, payload-level unavailability, "
          "analysis-stage cache consumption, real unit payload contract")


def report_completeness(cache):
    """Report committed units and payload-level scientific unavailability separately."""
    rows = []
    for entry in cache.completeness_table():
        key, status = entry["unit"], entry["status"]
        record = cache._latest.get(key, {})
        row = dict(entry, cache_status=status, reference_year=int(key.split("/")[-1] if key.startswith("S/") else key.split("/")[1]),
                   seconds=record.get("seconds"), peak_gpu_bytes=record.get("peak_gpu_bytes"),
                   source_hash=record.get("source_hash"), stage_version=record.get("stage_version"))
        if status == "complete":
            path = cache._unit_path(key) / "meta.json"
            payload = json.loads(path.read_text())
            if payload.get("status") == "unavailable":
                row["status"] = f"unavailable:{payload.get('reason', 'unknown')}"
            row["unavailable_families"] = [{"family": family.get("family_id", family.get("id")),
                                           "reason": family.get("reason", "unknown")}
                                          for family in payload.get("families", []) if family.get("status") == "unavailable"]
        rows.append(row)
    return rows


def report_tables(cache, config):
    """Read compact, provenance-bearing report rows from completed units only."""
    if config_hash(config) != cache.config_hash:
        raise ValueError("Configuration differs from the cache configuration")
    tables = {name: [] for name in ("support", "families", "members", "permutation", "sweep", "rules", "selection",
                                   "performance", "idea1", "idea2_frozen", "links", "groups", "profiles", "coactivation", "interventions")}
    tables["completeness"] = report_completeness(cache)
    converters = {"D.cavs_selection": ("selection", cavs_selection_table),
                  "D.reference_performance": ("performance", reference_performance_table),
                  "D.idea1": ("idea1", idea1_table), "D.idea2_frozen": ("idea2_frozen", idea2_frozen_table),
                  "D.idea2_yearly_families": ("links", idea2_link_table)}
    # pragmatic: CPU formatting of compact aggregate rows; vectorize if report rows exceed one million.
    for entry in tables["completeness"]:
        key, r, stage = entry["unit"], entry["reference_year"], unit_stage(entry["unit"])
        prefix = {"unit": key, "reference_year": r, "status": entry["status"]}
        if entry["cache_status"] != "complete":
            targets = [converters[stage][0]] if stage in converters else (["groups", "interventions"] if stage == "D.groups_interventions" else [])
            for name in targets:
                tables[name].append(prefix)
            continue
        # Load array-free metadata except for the few table adapters that consume compact cached arrays.
        unit = (cache.load(key) if stage == "D.groups_interventions" else json.loads((cache._unit_path(key) / "meta.json").read_text()))
        def extend(name, rows):
            tables[name].extend({**prefix, **{column: value for column, value in row.items() if column not in ("row", "patient_id", "row_index")}}
                                for row in rows)
        if stage == "P":
            extend("support", [{"rows": unit.get("_rows"), "context_rows": unit.get("context_rows"),
                                "context_patients": unit.get("context_patients"), "context_deaths": unit.get("context_deaths"),
                                "rows_per_year": unit.get("rows_per_year"), "embedding_dim": unit.get("embedding_dim"),
                                "population_identity": unit.get("population_identity"), "roles_identity": unit.get("roles_identity")}])
        elif stage == "D.families":
            for family in unit.get("families", []):
                extend("families", [family])
                extend("members", [{"family": family["id"], "seed": seed, "factor": factor,
                                    "medoid": [seed, factor] == family["medoid"], "kind": family["kind"]}
                                   for seed, factor in family["members"]])
            construction = unit.get("construction", {})
            extend("permutation", [{"view": "reference", "counts": construction.get("counts"),
                                     "chance": construction.get("chance"), "dead": unit.get("dead_latents"),
                                     "undefined": unit.get("undefined_latents")}])
        elif stage == "D.rules":
            extend("sweep", unit.get("sweep_rows", []))
            extend("rules", [{"family": item["family_id"], "medoid": item["medoid"],
                              "activation_target": item["activation_target"], "cutoff": item["cutoff"],
                              "rule": rule_text(item["rule_set"]), "rule_identity": item["rule_identity"],
                              "identity": item["identity"]} for item in unit.get("frozen", [])])
        elif stage in converters:
            name, converter = converters[stage]
            if entry["status"].startswith("unavailable:"):
                extend(name, [{}])
            else:
                extend(name, converter(unit))
            if stage == "D.idea2_yearly_families" and unit.get("status") == "ok":
                extend("permutation", [{"view": "yearly", "year": unit["year"], "chance": unit.get("link_chance"),
                                         "counts": unit.get("construction", {}).get("counts")}])
        elif stage == "D.groups_interventions":
            extend("groups", groups_table(unit))
            extend("interventions", interventions_table(unit))
            extend("coactivation", coactivation_table(unit))
            if unit["status"] == "ok":
                for family in unit["families"]:
                    if family["status"] != "ok":
                        continue
                    position = family["array_index"]
                    extend("profiles", [{"family": family["family_id"], "year": unit["year"], "group": group,
                                         "features": list(unit["feature_names"]), "rows": family["groups"][group]["rows"],
                                         "mean": unit["feature_mean"][position, index].tolist(),
                                         "std": unit["feature_std"][position, index].tolist(),
                                         "reason": None if family["groups"][group]["rows"] else "empty_group"}
                                        for index, group in enumerate(GROUP_NAMES)])
    temporal = run_temporal_analysis(cache, config)
    tables.update({"temporal_" + name: rows for name, rows in temporal_tables(temporal).items()})
    tables["trajectories"] = [{"reference_year": item["reference_year"], "family": item["family_id"], "idea": item["idea"],
                               "subgroup": item["subgroup"], "year": year, **record}
                              for item in temporal["trajectories"] for year, record in item["years"].items()]
    tables["reference_summary"] = [{"reference_year": r, "complete": sum(row["status"] == "complete" for row in tables["completeness"] if row["reference_year"] == r),
                                    "expected": sum(row["reference_year"] == r for row in tables["completeness"]),
                                    "families": sum(row["reference_year"] == r for row in tables["families"]),
                                    "independent_replicates": False}
                                   for r in config["reference_year_candidates"]]
    return _report_json(tables)


def _report_json(value):
    """JSON-safe scalar aggregates: undefined values remain null, never zero."""
    import math
    if isinstance(value, dict):
        return {str(key): _report_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_report_json(item) for item in value]
    if hasattr(value, "item"):
        return _report_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def export_report(tables, output):
    """Write compact JSON/CSV reports without patient identifiers or cache arrays."""
    import csv
    import io
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tables = _report_json(tables)
    _atomic_json(output / "report.json", tables)
    written = ["report.json"]
    for name, rows in tables.items():
        columns = list(dict.fromkeys(column for row in rows for column in row)) or ["status"]
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
        path = output / ("completeness.csv" if name == "completeness" else f"tables/{name}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
        with os.fdopen(fd, "w") as handle:
            handle.write(stream.getvalue()); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        written.append(str(path.relative_to(output)))
    return written


def overlap_cpu_gpu(cpu_job, gpu_jobs):
    """Compute a CPU payload concurrently with main-thread GPU jobs; worker never commits cache units."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(cpu_job)
        results = [job() for job in gpu_jobs]
        return future.result(), results


def resolve_device(requested):
    """Resolve the device once; explicit CUDA never falls back."""
    import logging
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    available = _cuda_available() if requested != "cpu" else False
    if requested == "cuda" and not available:
        raise RuntimeError("device=cuda requested but CUDA is unavailable")
    if requested == "auto" and not available:
        logging.getLogger(__name__).warning("device=auto: CUDA unavailable, running on CPU")
    return ("cuda" if available else "cpu") if requested == "auto" else requested


def run_report_unavailability_checks():
    """Compact report rows preserve explicit reasons for undefined metrics."""
    import numpy as np
    reliability = {"mean_risk": None, "brier": None, "brier_skill": None,
                   "unavailable_reasons": {"mean_risk": "empty_subgroup", "brier": "empty_subgroup", "brier_skill": "empty_subgroup"}}
    groups = {name: {"rows": 0, "returning_rows": 0, "entrant_rows": 0, "deaths": 0, "reliability": reliability}
              for name in GROUP_NAMES}
    unit = {"status": "ok", "year": 2008, "families": [{"status": "ok", "family_id": "empty-family", "array_index": 0,
            "groups": groups, "overlap": {"jaccard": None, "unavailable_reasons": {"jaccard": "zero_metric_denominator"}}}],
            "reconstruction_mse": np.full((1, 4), np.nan)}
    for row in groups_table(unit):
        assert row["Brier"] is None and row["skill"] is None
        assert row["unavailable"]["brier"] == row["unavailable"]["brier_skill"] == "empty_subgroup"
        assert row["unavailable"]["reconstruction_mse"] == "empty_group"
    print("Report unavailability fixtures passed: null metrics retain explicit reasons")


def run_read_only_cache_checks(notebook_path="concept_subgroup_drift_pilot.ipynb"):
    """Read-only cache opening never creates, repairs or writes artifacts, including after a hard stop."""
    config = resolve_config(_notebook_presets(json.loads(Path(notebook_path).read_text())), "v0")
    base = Path(os.environ["RPI_OUT_DIR"])
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as directory:
        root = Path(directory) / "cache"
        cache = UnitCache(root, config, "source-fixture", free_bytes=lambda: 100 * 2**30)
        cache.run("P/2007", lambda: {"_rows": 1})
        before = cache._manifest.read_bytes()
        readonly = UnitCache(root, config, "source-fixture", read_only=True)
        assert readonly.is_complete("P/2007") and readonly.load("P/2007")["_rows"] == 1
        _refuses(lambda: readonly.run("P/2007", lambda: {}), "read-only", PermissionError)
        _refuses(lambda: readonly._append("P/2007", "failed", "refused"), "read-only", PermissionError)
        assert cache._manifest.read_bytes() == before
        truncated = before + b'{"partial":'
        cache._manifest.write_bytes(truncated)
        _refuses(lambda: UnitCache(root, config, "source-fixture", read_only=True), "truncated", ValueError)
        assert cache._manifest.read_bytes() == truncated, "Read-only open repaired truncated manifest"
        missing = Path(directory) / "absent"
        _refuses(lambda: UnitCache(missing, config, "source-fixture", read_only=True), "absent", FileNotFoundError)
        assert not missing.exists(), "Read-only open created a missing cache"
    print("Read-only cache fixtures passed: intact reads, write rejection, truncated-manifest refusal, no implicit creation")


def run_orchestration_checks(notebook_path="concept_subgroup_drift_pilot.ipynb"):
    """Nonempty planted reports, stopped analysis, payload unavailability, sanitized exports and concurrent scheduling."""
    import copy
    import threading
    notebook = json.loads(Path(notebook_path).read_text())
    config = _selection_config(resolve_config(_notebook_presets(notebook), "v0"))
    world = _groups_world(config)
    world["family_output"] = dict(world["family_output"], families=[
        {"id": item["family_id"], "kind": "full", "medoid": item["medoid"],
         "members": [[seed, item["medoid"][1]] for seed in config["sae_seeds"]],
         "min_correlation": .8, "median_correlation": .9}
        for item in world["family_output"]["selection"]["candidates"]])
    idea1, idea2 = _groups_dependencies(config, world)
    groups = _groups_run(world, config, idea1, idea2)
    performance = reference_performance_unit(world["view"], 2007, world["roles"], world["output"],
                                             world["family_output"], world["references"], world["dictionaries"], config, "cpu")
    rules = {"frozen": [{key: value for key, value in entry.items() if key != "rule"}
                        for entry in world["frozen"].values()],
             "families": [{"family_id": name, "status": "frozen"} for name in world["frozen"]], "_rows": 1}
    payloads = {"P/2007": world["output"], "D/2007/families": world["family_output"], "D/2007/rules": rules,
                "D/2007/cavs_selection": world["selection_unit"], "D/2007/reference_performance": performance,
                "D/2007/idea1/2008": idea1, "D/2007/idea2_frozen/2008": idea2,
                "D/2007/groups_interventions/2008": groups}
    payloads.update({f"S/{seed}/2007": payload for seed, payload in world["payloads"].items()})
    base = Path(os.environ["RPI_OUT_DIR"])
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as directory:
        cache = UnitCache(Path(directory) / "cache", config, "source-fixture", free_bytes=lambda: 100 * 2**30)
        for key in unit_order(config):
            if key in payloads:
                assert cache.run(key, lambda key=key: dict(payloads[key]))["status"] == "ok", key
        before = cache._manifest.read_bytes()
        completeness = report_completeness(cache)
        by_unit = {row["unit"]: row for row in completeness}
        assert by_unit["D/2007/groups_interventions/2008"]["status"] == "complete"
        assert by_unit["P/2008"]["status"] == "not_run (stopped)"
        tables = report_tables(cache, config)
        for name in ("members", "selection", "performance", "idea1", "idea2_frozen", "groups", "interventions", "temporal_forecast"):
            assert tables[name], f"Planted eligible families missing from {name}"
        assert len(tables["temporal_forecast"]) >= 4
        assert all(row["unit"] in payloads or row["status"] != "complete"
                   for name in ("performance", "groups", "idea1", "idea2_frozen") for row in tables[name])
        assert not any("row" in row or "patient_id" in row for row in tables["interventions"])
        assert cache._manifest.read_bytes() == before, "Report-only analysis mutated the cache"
        written = export_report(tables, Path(directory) / "out")
        assert "completeness.csv" in written and "report.json" in written
        exported = json.loads((Path(directory) / "out" / "report.json").read_text())
        assert exported["performance"] and exported["interventions"]
        assert exported["completeness"] == tables["completeness"]
        assert cache._manifest.read_bytes() == before
        unavailable = dict(groups, status="unavailable", reason="empty_evaluation_population")
        cache._append("D/2007/groups_interventions/2008", "failed", "fixture-reset")
        assert cache.run("D/2007/groups_interventions/2008", lambda: unavailable)["status"] == "ok"
        row = next(row for row in report_completeness(cache) if row["unit"] == "D/2007/groups_interventions/2008")
        assert row["status"] == "unavailable:empty_evaluation_population" and row["cache_status"] == "complete"
        unavailable_tables = report_tables(cache, config)
        assert any(row["status"] == "unavailable:empty_evaluation_population" for row in unavailable_tables["groups"])
        cache.run("D/2008/families", lambda: {})
        assert next(row for row in report_completeness(cache) if row["unit"] == "D/2008/families")["status"].startswith("unavailable:")
        cache._append("P/2008", "failed", "RuntimeError: planted gate failure")
        assert next(row for row in report_completeness(cache) if row["unit"] == "P/2008")["status"] == "failed:RuntimeError: planted gate failure"
        _refuses(lambda: report_tables(cache, dict(config, epochs=3)), "differs")

    entered, gpu_started = threading.Event(), threading.Event()
    caller = threading.get_ident()
    def cpu_job():
        assert threading.get_ident() != caller
        entered.set()
        assert gpu_started.wait(10), "CPU/GPU jobs did not overlap"
        return {"result": "cpu-payload"}
    def gpu_job():
        assert threading.get_ident() == caller, "GPU/cache owner moved to worker thread"
        assert entered.wait(10)
        gpu_started.set()
        return "gpu-unit"
    cpu, gpu = overlap_cpu_gpu(cpu_job, [gpu_job])
    assert cpu == {"result": "cpu-payload"} and gpu == ["gpu-unit"]
    def failed_cpu():
        raise RuntimeError("planted CPU failure")
    _refuses(lambda: overlap_cpu_gpu(failed_cpu, []), "planted CPU failure", RuntimeError)
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "Scientific orchestration pending" not in sources
    assert "def compute_downstream" in sources and "pilot.overlap_cpu_gpu" in sources
    assert "pilot.report_tables" in sources and "pilot.export_report" in sources
    print("Orchestration fixtures passed: eligible exports, stopped analysis, scientific unavailability, read-only cache, CPU/GPU overlap")


def run_unit_cache_checks():
    """Unit order/dependencies, atomic resumable cache, hash refusal, stage versions, disk guard, labels."""
    import numpy as np
    pilot = {"reference_year_candidates": list(range(2007, 2016)), "t0_only_reference_years": [2015],
             "sae_seeds": [42, 135, 215]}
    years, seeds = pilot["reference_year_candidates"], pilot["sae_seeds"]
    graph, order = unit_graph(pilot), unit_order(pilot)
    assert order == list(graph) and len(order) == len(set(order)) == 324

    def reference_block(r, followups):
        keys = [f"D/{r}/{name}" for name in ("families", "rules", "cavs_selection", "reference_performance")]
        for t in followups:
            keys += [f"D/{r}/idea1/{t}", f"D/{r}/idea2_frozen/{t}"]
            keys += [f"D/{r}/idea2_yearly_sae/{seed}/{t}" for seed in seeds]
            keys += [f"D/{r}/idea2_yearly_families/{t}", f"D/{r}/groups_interventions/{t}"]
        return keys
    assert order[:9] == [f"P/{r}" for r in years]
    assert order[9:36] == [f"S/{seed}/{r}" for seed in seeds for r in years]
    assert order[36:] == [key for r in years for key in reference_block(r, [t for t in years if t > r])]
    assert "D/2015/reference_performance" in order and not [key for key in order if key.startswith("D/2015/idea")]
    early = unit_graph(dict(pilot, t0_only_reference_years=[2014, 2015]))
    assert "D/2014/families" in early and not [key for key in early if key.startswith(("D/2014/idea", "D/2014/groups"))]
    assert graph["P/2007"] == () and graph["S/135/2008"] == ("P/2008",)
    assert set(graph["D/2007/families"]) == {"P/2007"} | {f"S/{seed}/2007" for seed in seeds}
    assert {f"D/2007/idea2_yearly_sae/{seed}/2010" for seed in seeds} <= set(graph["D/2007/idea2_yearly_families/2010"])
    assert {f"S/{seed}/2007" for seed in seeds} == {dep for seed in seeds for dep in graph[f"D/2007/idea2_yearly_sae/{seed}/2010"]
                                                    if dep.startswith("S/")}
    assert all(set(graph[f"D/2007/idea2_yearly_sae/{seed}/2010"]) == {"P/2007", f"S/{seed}/2007"} for seed in seeds)
    assert set(graph["D/2007/groups_interventions/2010"]) == {"P/2007", "D/2007/cavs_selection", "D/2007/idea1/2010",
                                                              "D/2007/idea2_frozen/2010"}, "Frozen groups must not wait for yearly SAEs"
    for key, dependencies in graph.items():
        assert set(dependencies) <= set(order[:order.index(key)]), key
        assert not key.startswith("D/") or f"P/{key.split('/')[1]}" in dependencies, key
    assert [unit_stage(key) for key in ("P/2007", "S/42/2007", "D/2007/families", "D/2007/idea1/2008",
            "D/2007/idea2_yearly_sae/42/2008", "D/2007/groups_interventions/2008")] == [
            "P", "S", "D.families", "D.idea1", "D.idea2_yearly_sae", "D.groups_interventions"]
    for bad in ("", "X/1", "D/2007"):
        try:
            unit_stage(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unknown unit key accepted: {bad!r}")

    config = dict(pilot, reference_year_candidates=[2007, 2008, 2009], t0_only_reference_years=[2009],
                  epochs=2, device="cpu", disk_guard_gib=15, joblib_workers=2)
    assert config_hash(config) == config_hash(dict(config, disk_guard_gib=1, joblib_workers=9))
    assert config_hash(config) != config_hash(dict(config, epochs=3))
    base = Path(os.environ["RPI_OUT_DIR"]) if os.environ.get("RPI_OUT_DIR") else None
    if base is not None:
        base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as work:
        root = Path(work)
        free = {"bytes": 100 * 2**30}
        versions = {}
        calls = []

        source_root = root / "source-resume"
        first_source = UnitCache(source_root, config, "source-1", free_bytes=lambda: free["bytes"])
        first_predictor = first_source.run("P/2007", lambda: {"_rows": 1, "note": "first-source"})
        first_sae = first_source.run("S/42/2007", lambda: {"_rows": 1})
        changed_source = UnitCache(source_root, config, "source-2", free_bytes=lambda: free["bytes"])
        assert not changed_source.is_complete("P/2007"), "Source change silently reused stale predictor unit"
        assert not changed_source.is_complete("S/42/2007"), "Source change silently reused stale dependent unit"
        recomputed = changed_source.run("P/2007", lambda: {"_rows": 1, "note": "second-source"})
        assert recomputed["id"] != first_predictor["id"] and changed_source.load("P/2007")["note"] == "second-source"
        recomputed_sae = changed_source.run("S/42/2007", lambda: {"_rows": 1})
        assert recomputed_sae["id"] != first_sae["id"] and recomputed_sae["dependencies"]["P/2007"] == recomputed["id"]

        def open_cache(cfg=config, **keywords):
            keywords.setdefault("stage_versions", versions)
            return UnitCache(root, cfg, "source-1", free_bytes=lambda: free["bytes"], **keywords)

        def unit(key):
            def produce():
                calls.append(key)
                return {"values": np.arange(4) + len(calls), "note": key, "_rows": 4}
            return produce

        cache = open_cache()
        assert cache.directory == root / config_hash(config) and (cache.directory / "config.json").is_file()
        record = cache.run("P/2007", unit("P/2007"))
        assert record["status"] == "ok" and record["rows"] == 4 and record["key"] == "P/2007"
        assert record["stage"] == "P" and record["stage_version"] == STAGE_CODE_VERSIONS["P"]
        assert record["config_hash"] == config_hash(config) and record["source_hash"] == "source-1"
        assert record["device"] == "cpu" and record["seconds"] >= 0 and record["dependencies"] == {}
        assert record["start_utc"] and record["end_utc"]
        assert record["peak_gpu_bytes"] is None or record["peak_gpu_bytes"] >= 0
        loaded = cache.load("P/2007")
        np.testing.assert_array_equal(loaded["values"], np.arange(4) + 1)
        assert loaded["note"] == "P/2007" and cache.is_complete("P/2007")
        resumed = open_cache()
        before = len(calls)
        assert resumed.is_complete("P/2007") and resumed.run("P/2007", unit("P/2007"))["id"] == record["id"]
        assert len(calls) == before, "Complete unit was recomputed on resume"
        try:
            cache.run("X/1", unit("X/1"))
        except ValueError:
            pass
        else:
            raise AssertionError("Unknown unit accepted")

        for interruption in (KeyboardInterrupt, SystemExit):
            def interrupted():
                calls.append("interrupted")
                raise interruption()
            try:
                cache.run("P/2008", interrupted)
            except interruption:
                pass
            else:
                raise AssertionError("Interruption was swallowed")
            assert not cache.is_complete("P/2008")
            assert cache.completeness_table(["P/2008"])[0]["status"] == "not_run (stopped)"
        leftover, partial = cache._tmp_path("P/2008"), cache._unit_path("P/2008")
        leftover.mkdir(parents=True, exist_ok=True)
        (leftover / "arrays.npz").write_bytes(b"partial")
        partial.mkdir(parents=True, exist_ok=True)
        (partial / "meta.json").write_text("{")
        assert not cache.is_complete("P/2008") and not open_cache().is_complete("P/2008")
        assert cache.run("P/2008", unit("P/2008"))["status"] == "ok"
        assert not leftover.exists() and sorted(path.name for path in partial.iterdir()) == ["arrays.npz", "meta.json"]
        assert cache.load("P/2008")["note"] == "P/2008"

        def boom():
            calls.append("P/2009")
            raise ValueError("boom")
        record = cache.run("P/2009", boom)
        assert record["status"] == "failed" and record["reason"] == "ValueError: boom"
        assert not cache._unit_path("P/2009").exists() and not cache.is_complete("P/2009")
        calls_before = len(calls)
        for key in ("S/42/2009", "D/2009/families"):
            record = cache.run(key, unit(key))
            assert record["status"] == "unavailable" and record["reason"] == "dep_failed:P/2009", key
        assert len(calls) == calls_before, "Dependent of a failed unit was executed"
        for seed in seeds:
            cache.run(f"S/{seed}/2007", unit(f"S/{seed}/2007"))
        failing = "S/135/2008"
        assert cache.run("S/42/2008", unit("S/42/2008"))["status"] == "ok"
        assert cache.run(failing, lambda: (_ for _ in ()).throw(RuntimeError("fit diverged")))["reason"] == "RuntimeError: fit diverged"
        for key in ("D/2008/families", "D/2008/rules"):
            record = cache.run(key, unit(key))
            assert (record["status"], record["reason"]) == ("unavailable", "dep_failed:S/135/2008"), key
        table = {row["unit"]: row["status"] for row in cache.completeness_table(
            ["P/2007", "P/2009", "S/42/2009", "D/2008/rules", "S/215/2008", "D/2007/families"])}
        assert table == {"P/2007": "complete", "P/2009": "failed:ValueError: boom",
                         "S/42/2009": "unavailable:dep_failed:P/2009",
                         "D/2008/rules": "unavailable:dep_failed:S/135/2008",
                         "S/215/2008": "not_run (stopped)", "D/2007/families": "not_run (stopped)"}
        assert [row["unit"] for row in cache.completeness_table()] == list(unit_order(config))

        chain = ["P/2007", "S/42/2007", "S/135/2007", "S/215/2007", "D/2007/families", "D/2007/rules"]
        for key in chain:
            assert cache.run(key, unit(key))["status"] == "ok"
        assert all(cache.is_complete(key) for key in chain + ["P/2008"])
        try:
            open_cache(stage_versions={"unknown_stage": 2})
        except ValueError:
            pass
        else:
            raise AssertionError("Unknown stage version accepted")
        versions["D.families"] = 2
        bumped = open_cache()
        assert not bumped.is_complete("D/2007/families") and not bumped.is_complete("D/2007/rules")
        assert all(bumped.is_complete(key) for key in chain[:4] + ["P/2008"])
        calls_before = len(calls)
        assert bumped.run("D/2007/families", unit("D/2007/families"))["stage_version"] == 2
        assert not bumped.is_complete("D/2007/rules"), "Dependent of a recomputed unit stayed fresh"
        for key in chain:
            assert bumped.run(key, unit(key))["status"] == "ok"
        assert calls[calls_before:] == ["D/2007/families", "D/2007/rules"]
        versions["S"] = 2
        bumped = open_cache()
        assert all(bumped.is_complete(key) for key in ("P/2007", "P/2008"))
        assert not any(bumped.is_complete(key) for key in chain[1:])
        calls_before = len(calls)
        for key in chain:
            bumped.run(key, unit(key))
        assert calls[calls_before:] == chain[1:]
        cache = open_cache()
        assert all(cache.is_complete(key) for key in chain)

        other = open_cache(dict(config, epochs=3))
        assert other.directory != cache.directory and not other.is_complete("P/2007")
        for expected in (config_hash(dict(config, epochs=3)), "0" * 64):
            try:
                open_cache(expected_hash=expected)
            except ValueError:
                pass
            else:
                raise AssertionError("Resume with a different configuration hash accepted")
        assert open_cache(expected_hash=config_hash(config)).is_complete("P/2007")
        stored = cache.directory / "config.json"
        original = stored.read_text()
        stored.write_text(json.dumps({"epochs": 99}))
        try:
            open_cache()
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatching stored configuration accepted")
        stored.write_text(original)
        assert open_cache(dict(config, disk_guard_gib=20)).directory == cache.directory

        target = "D/2007/cavs_selection"
        free["bytes"] = 16 * 2**30
        calls_before = len(calls)
        try:
            cache.run(target, unit(target), projected_bytes=2 * 2**30)
        except DiskGuardStop:
            pass
        else:
            raise AssertionError("Disk guard did not stop the loop")
        assert len(calls) == calls_before and not cache.is_complete(target)
        assert cache.completeness_table([target])[0]["status"] == "not_run (stopped)"
        stopped = [json.loads(line) for line in (cache.directory / "units.jsonl").read_text().splitlines()
                   if json.loads(line)["key"] == target]
        assert stopped[-1]["status"] == "stopped" and stopped[-1]["reason"] == "disk_guard"
        free["bytes"] = 17 * 2**30
        try:
            open_cache(dict(config, disk_guard_gib=20)).run(target, unit(target))
        except DiskGuardStop:
            pass
        else:
            raise AssertionError("Changed guard was ignored")
        assert cache.run(target, unit(target), projected_bytes=2 * 2**30)["status"] == "ok"
        assert cache.completeness_table([target])[0]["status"] == "complete"

        with open(cache.directory / "units.jsonl", "a") as stream:
            stream.write('{"key": "D/2007/refe')
        crashed = open_cache()
        assert crashed.run("D/2007/reference_performance", unit("D/2007/reference_performance"))["status"] == "ok"
        assert open_cache().is_complete("D/2007/reference_performance") and open_cache().is_complete(target)
        import shutil
        shutil.rmtree(crashed._unit_path("D/2007/reference_performance"))
        assert not open_cache().is_complete("D/2007/reference_performance")
        assert open_cache().run("D/2007/reference_performance", unit("D/2007/reference_performance"))["status"] == "ok"
    print("Unit order, atomic resumable cache, hash refusal, stage versions, disk guard and labels fixtures passed")
