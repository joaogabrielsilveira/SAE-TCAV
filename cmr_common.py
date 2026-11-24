"""Read-only identities, protected rows, shared patient draws and private freezes."""
import ast
import atexit
import fcntl
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import csv
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import struct
import subprocess
import sys
import time
import zipfile

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hash(path, expected):
    if file_hash(path) != expected.removeprefix("sha256:"):
        raise ValueError("Source/input hash mismatch")


def verify_order(actual, expected):
    if not np.array_equal(actual, expected) or len(np.unique(actual)) != len(actual):
        raise ValueError("ordered row identity mismatch")


def verify_roles(patients, roles):
    assigned = {}
    for patient, role in zip(patients, roles, strict=True):
        if role and assigned.setdefault(str(patient), str(role)) != str(role):
            raise ValueError("Patient role overlap/change")
    return assigned


def patient_weights(patients, seeds=range(3001, 3501)):
    """One multinomial draw per distinct patient, reused for every patient row."""
    unique, inverse = np.unique(np.asarray(patients).astype(str), return_inverse=True)
    seeds = tuple(seeds)
    if not len(unique):
        return np.zeros((len(seeds), 0), dtype=np.int64)
    return np.stack([np.random.default_rng(seed).multinomial(
        len(unique), np.full(len(unique), 1 / len(unique)))[inverse] for seed in seeds])


def read_array_rows(path, name, rows=None):
    """Index stored NPY members in-place; never decompress/read protected rows."""
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(name + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("Refuse compressed patient archive; cannot isolate rows")
        with open(path, "rb") as f:
            f.seek(info.header_offset)
            header = f.read(30)
            name_size, extra_size = struct.unpack_from("<HH", header, 26)
            f.seek(name_size + extra_size, 1)
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                raise ValueError("Unsupported NPY header")
            offset = f.tell()
    if dtype.hasobject:
        raise ValueError("Object patient arrays refused")
    array = np.memmap(path, dtype=dtype, mode="r", offset=offset,
                      shape=shape, order="F" if fortran else "C")
    return np.array(array if rows is None else array[np.asarray(rows, dtype=int)], copy=True)


def public_records(records, fields):
    """Only explicit scalar aggregate columns may leave private storage."""
    forbidden = re.compile(r"patient|embedding|gradient|decoder|direction|mask|prediction|hash", re.I)
    if any(forbidden.search(key) for key in fields):
        raise ValueError("Private field in export schema")
    clean = []
    for record in records:
        if set(record) - set(fields):
            raise ValueError("Unknown aggregate field")
        row = {}
        for key, value in record.items():
            if isinstance(value, np.generic):
                value = value.item()
            if value is not None and type(value) not in (str, int, float, bool):
                raise TypeError("Only scalar aggregate values may be exported")
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("Nonfinite public value; use null with status")
            if isinstance(value, str) and (len(value) > 512 or re.search(r"[a-f0-9]{64}", value)):
                raise ValueError("Private identifier or unbounded text in export")
            row[key] = value
        clean.append(row)
    return clean


def publish(ctx, records, fields):
    import pandas as pd
    rows = public_records(records, fields)
    frame = pd.DataFrame(rows, columns=sorted(fields))
    public = ctx["out"] / "public"
    public.mkdir(exist_ok=True)
    number = len(list(public.glob("table-*.json")))
    (public / f"table-{number:03d}.json").write_text(json.dumps(rows, allow_nan=False))
    frame.to_csv(public / f"table-{number:03d}.csv", index=False)
    return frame


def resolve_device(requested):
    import torch
    if requested not in ("cpu", "cuda"):
        raise ValueError("CMR_DEVICE must be cpu or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def check_budget(ctx):
    elapsed = time.monotonic() - ctx["started"]
    totals = {"all": 0., "smoke": 0.}
    if "budget_file" in ctx:
        with ctx["budget_file"].open("a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            entries = json.loads(f.read() or "{}")
            for key, value in entries.items():
                if key != str(ctx["out"]):
                    seconds = value.get("seconds", time.time() - value["started"])
                    totals["all"] += seconds
                    if value["mode"] == "smoke":
                        totals["smoke"] += seconds
    if elapsed + totals["all"] >= 8 * 3600 or (ctx["mode"] == "smoke" and elapsed + totals["smoke"] >= 20 * 60):
        raise TimeoutError("Scientific budget expired; retain partial coverage")
    if shutil.disk_usage(ctx["out"]).free < 15 * 2**30:
        raise RuntimeError("Free disk below 15 GiB")


def verify_execution_source(ctx, branch):
    root = ctx["root"]
    if Path(sys.executable).resolve() != (root / ".venv/bin/python").resolve():
        raise RuntimeError("Notebook kernel is not approved Fiji venv")
    if Path.cwd().resolve() != root:
        raise RuntimeError("Source/import root differs from execution root")
    names = {"measurement": "concept_measurement_reliability", "explanations": "concept_explanations",
             "groups": "concept_factor_groups"}
    source_names = ["cmr_common.py", names[branch] + ".ipynb", "pilot_follow_ups.ipynb",
                    "concept_subgroup_drift_pilot.py", "concept_subgroup_drift_pilot.ipynb"]
    ctx["source_hashes"] = {name: file_hash(root / name) for name in source_names}
    for path in root.glob("*.py"):
        ctx["source_hashes"][path.name] = file_hash(path)
    for name in ("concept_subgroup_drift_pilot.py", "concept_subgroup_drift_pilot.ipynb", "pilot_follow_ups.ipynb"):
        pinned = subprocess.check_output(["git", "show", "16425ee:" + name], cwd=root)
        if hashlib.sha256(pinned).hexdigest() != ctx["source_hashes"][name]:
            raise ValueError("Historical source parity mismatch")


def resolve_input_roots(ctx):
    for key in ("CMR_PILOT_ROOT", "CMR_BUILDS_ROOT", "CMR_POPULATION", "CMR_PREVIOUS_RUNS"):
        if not os.environ.get(key):
            raise ValueError(key + " required")
        ctx[key] = Path(os.environ[key]).expanduser().resolve(strict=True)


def prepare_run_output(ctx):
    root = ctx["root"]
    out = Path(os.environ["CMR_OUT"]).expanduser().resolve()
    if out == root or root in out.parents:
        raise ValueError("Outputs must stay outside source checkout")
    study_root = Path.home() / "rpi-runs/SAE-TCAV/concept-measurement-reliability"
    if out.parent.parent != study_root:
        raise ValueError("Outputs must use the approved work-item/run/branch namespace")
    if out.exists():
        existing = list(out.iterdir())
        if any(path.name != "private" or not path.is_dir() or any(path.iterdir()) for path in existing):
            raise ValueError("Use a new immutable output namespace")
    out.mkdir(parents=True, exist_ok=True)
    (out / "private").mkdir(exist_ok=True)
    (out / "public").mkdir(exist_ok=True)
    ctx.update(out=out, device=resolve_device(os.environ.get("CMR_DEVICE", "cuda")))
    ctx["budget_file"] = out.parent.parent / "study-budget.json"
    if ctx["device"] != "cuda":
        raise RuntimeError("Scientific modes require CUDA")


def record_run_environment(ctx):
    import torch
    ctx["accelerator"] = {"device": ctx["device"], "name": torch.cuda.get_device_name(),
                          "memory": torch.cuda.get_device_properties(0).total_memory,
                          "free_memory": torch.cuda.mem_get_info()[0], "cpu_workers": os.cpu_count(),
                          "host_available_bytes": os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")}
    ctx["environment"] = {key: importlib.metadata.version(key) for key in
                          ("torch", "numpy", "pandas", "scikit-learn", "nbconvert", "ipykernel")}
    historical = json.loads((ctx["CMR_PREVIOUS_RUNS"] / "develop-last_3/run-manifest.json").read_text())
    old = historical["environment"]
    versions = old.get("packages", old)
    if versions.get("scikit-learn") != ctx["environment"]["scikit-learn"]:
        raise ValueError("Historical sklearn library parity mismatch")


def setup_run(branch):
    root = Path(__file__).resolve().parent
    mode = os.environ.get("CMR_MODE", "check")
    if mode not in ("check", "smoke", "develop", "evaluate"):
        raise ValueError("Unknown CMR_MODE")
    ctx = {"mode": mode, "started": time.monotonic(), "branch": branch,
           "root": root, "input_files": {}, "coverage": []}
    if mode == "check":
        return {**ctx, "device": "cpu", "out": None, "source_hashes": {}}
    verify_execution_source(ctx, branch)
    resolve_input_roots(ctx)
    prepare_run_output(ctx)
    record_run_environment(ctx)
    check_budget(ctx)
    finish_run(ctx, "running", {})
    atexit.register(lambda: finish_run(ctx, "failed", {}) if ctx.get("status") == "running" else None)
    return ctx


def finish_run(ctx, status, counts):
    if ctx["mode"] == "check":
        return
    ctx["status"] = status
    if "budget_file" in ctx:
        with ctx["budget_file"].open("a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            entries = json.loads(f.read() or "{}")
            entry = entries.setdefault(str(ctx["out"]), {"mode": ctx["mode"], "started": time.time()})
            if status != "running":
                entry["seconds"] = time.monotonic() - ctx["started"]
            entry["status"] = status
            f.seek(0)
            f.truncate()
            f.write(json.dumps(entries))
            f.flush()
    root = ctx["root"]
    result = {"status": status, "branch": ctx["branch"], "mode": ctx["mode"],
              "seconds": time.monotonic() - ctx["started"], "counts": counts,
              "base_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
              "source_hashes": ctx["source_hashes"], "input_hashes": ctx["input_files"],
              "environment": ctx["environment"], "accelerator": ctx["accelerator"],
              "command": sys.argv, "interpreter": sys.executable,
              "seeds": {"solver": 42, "negative": [42, 43, 44], "joint": list(range(1001, 1051)),
                        "labels": list(range(2001, 2006)), "patients": list(range(3001, 3501)),
                        "sae": [42, 135, 215]},
              "updated_utc": datetime.now(timezone.utc).isoformat(), "coverage": ctx["coverage"]}
    (ctx["out"] / "private/run-manifest.json").write_text(json.dumps(result, allow_nan=False, indent=2))


def scan_notebook(ctx, path):
    """Validate executed outputs before a byte-identical notebook may become public."""
    path = Path(path).resolve(strict=True)
    if path.parent != ctx["out"] / "private":
        raise ValueError("Report must originate in private output")
    notebook = json.loads(path.read_text())
    # Labels shown in tables must already be public: CSV headers (covering empty tables) and
    # cell values (pivot labels); "..." is pandas' truncation marker.
    allowed_fields = {"check", "result", "status", "name", "passed", "mode", "device", "..."}
    for table in (ctx["out"] / "public").glob("table-*.csv"):
        with table.open(newline="") as f:
            for row in csv.reader(f):
                allowed_fields.update(row)
    forbidden = re.compile(r"patient_id|patient_hash|row_index|encoder_weight|decoder_weight|cav_vectors|traceback", re.I)
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            if output["output_type"] not in ("display_data", "execute_result", "stream"):
                raise ValueError("Error/unknown notebook output")
            data = output.get("data", {})
            if set(data) - {"text/plain", "text/html", "image/png"}:
                raise ValueError("Unknown notebook MIME output")
            for kind in ("text/plain", "text/html"):
                text = data.get(kind, [])
                text = text if isinstance(text, str) else "".join(text)
                if forbidden.search(text) or re.search(r"<script|<iframe|application/javascript", text, re.I):
                    raise ValueError("Private/unsafe notebook output")
                if re.search(r"\barray\s*\(|\btensor\s*\(", text):
                    raise ValueError("Unschematized notebook array output")
                if kind == "text/html":
                    headers = re.findall(r"<th[^>]*>(.*?)</th>", text, flags=re.S)
                    headers = [html.unescape(re.sub(r"<[^>]+>", "", value)).strip() for value in headers]
                    if any(value and not value.isdigit() and value not in allowed_fields for value in headers):
                        raise ValueError("Unknown notebook table column")
                if kind == "text/plain" and text.lstrip().startswith(("{", "[")):
                    raise ValueError("Unschematized notebook collection output")
            stream = output.get("text", [])
            stream = stream if isinstance(stream, str) else "".join(stream)
            if forbidden.search(stream) or re.search(r"\barray\s*\(|\btensor\s*\(", stream):
                raise ValueError("Private notebook stream")
    target = ctx["out"] / "public" / "executed.ipynb"
    shutil.copyfile(path, target)
    return file_hash(target)


def freeze_write(ctx, payload):
    """Freeze fitted private objects plus exact source/input provenance; never overwrite."""
    check_budget(ctx)
    path = ctx["out"] / "private/freeze.pkl"
    if path.exists():
        raise FileExistsError("Freeze already exists")
    roots = {key: str(value) for key, value in ctx.items() if key.startswith("CMR_")}
    data = pickle.dumps({"branch": ctx["branch"], "roots": roots, "source_hashes": ctx["source_hashes"],
                         "input_files": ctx["input_files"], "payload": payload}, protocol=5)
    with path.open("xb") as f:
        f.write(data)
    identity = hashlib.sha256(data).hexdigest()
    path.with_suffix(".sha256").write_text(identity)
    return identity


def freeze_read(ctx):
    path = Path(os.environ["CMR_FREEZE"]).expanduser().resolve(strict=True)
    verify_hash(path, path.with_suffix(".sha256").read_text().strip())
    # Pickle is private trusted local output, never an imported external model.
    if "private" not in path.parts or path.is_symlink():
        raise ValueError("Freeze must be a private local artifact")
    value = pickle.loads(path.read_bytes())
    if value["branch"] != ctx["branch"] or value["source_hashes"] != ctx["source_hashes"]:
        raise ValueError("Freeze source/branch identity mismatch")
    if value["roots"] != {key: str(v) for key, v in ctx.items() if key.startswith("CMR_")}:
        raise ValueError("Freeze input root identity mismatch")
    for name, expected in value["input_files"].items():
        verify_hash(name, expected)
    return value["payload"]


def effective_build(variant, reference):
    if reference == 2007 or (variant == "baseline" and reference <= 2013):
        return "r1"
    if variant == "last_3" and reference in (2008, 2009):
        return "all_history"
    return variant


def legacy_descriptions(root):
    """Only the verified historical functions used for yearly descriptions; no notebook execution."""
    notebook = json.loads((root / "pilot_follow_ups.ipynb").read_text())
    source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    names = {"canonical_hash", "ordered_rows_hash", "explanation_features", "patient_partition", "development_rows",
             "candidate_policies", "outer_recurrence", "fidelity", "cap_by_meaning",
             "fit_member_explanation", "yearly_description"}
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"np": np, "hashlib": hashlib, "json": json, "CFG": {"split_seed": 42}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "historical-description-functions", "exec"), namespace)
    return namespace


def open_caches(ctx):
    import concept_subgroup_drift_pilot as pilot
    if Path(pilot.__file__).resolve().parent != ctx["root"]:
        raise ValueError("Mutable or foreign import root")
    variants = ("last_3", "all_history") if ctx["branch"] == "groups" else ("baseline", "last_3", "all_history")
    roots = {"r1": ctx["CMR_PILOT_ROOT"], **{v: ctx["CMR_BUILDS_ROOT"] / v for v in variants}}
    sources = {name: ctx["source_hashes"][name] for name in
               ("concept_subgroup_drift_pilot.py", "concept_subgroup_drift_pilot.ipynb")}
    combined = digest(sources)
    caches = {}
    reference_config = json.loads((roots["r1"] / "out/resolved-config.json").read_text())
    excluded = {"context_window", "reference_year_candidates", "predictor_cpu_from", "config_hash", "profile",
                "device", "data_path", "cache_dir", "output_dir", "disk_guard_gib", "joblib_workers", "inference_batch_rows"}
    shared = lambda value: {key: item for key, item in value.items() if key not in excluded}
    windows = {"baseline": "reference_only", "last_3": "last_3", "all_history": "all_history"}
    for build, path in roots.items():
        f = path / "out/resolved-config.json"
        config = json.loads(f.read_text())
        if build != "r1" and (config.get("context_window", "reference_only") != windows[build]
                               or shared(config) != shared(reference_config)):
            raise ValueError("Build context/configuration parity mismatch")
        source = "8a71a6759b29e3b967c64cd5cd5e696a47a13a18cfe4112791dc0e229d9231d7" if build == "r1" else combined
        cache = pilot.UnitCache(path / "unit-cache", config, source,
                               expected_hash=config["config_hash"], read_only=True)
        ctx["input_files"][str(f)] = file_hash(f)
        ctx["input_files"][str(cache.directory / "units.jsonl")] = file_hash(cache.directory / "units.jsonl")
        caches[build] = cache
    return caches


def cached_role_patients(population, cache, reference, ctx, paths, predictor_meta):
    import concept_subgroup_drift_pilot as pilot
    role_of = {}
    for role, (key, array_name, identity_name) in paths.items():
        if not cache.is_complete(key):
            raise KeyError("Missing frozen role assignment dependency: " + key)
        directory = cache._unit_path(key)
        meta = json.loads((directory / "meta.json").read_text())
        if meta["roles_identity"] != predictor_meta["roles_identity"]:
            raise ValueError("Frozen role object identity mismatch")
        rows = read_array_rows(directory / "arrays.npz", array_name)
        if (pilot._identity_hash({"row_index": rows}) != meta[identity_name]
                or not np.all(population["year"][rows] == reference)):
            raise ValueError("Frozen role row identity mismatch")
        for row in rows:
            patient = str(population["patient_id"][row])
            if patient in role_of and role_of[patient] != role:
                raise ValueError("Frozen patient role overlap")
            role_of[patient] = role
        for name in ("meta.json", "arrays.npz"):
            f = directory / name
            ctx["input_files"][str(f)] = file_hash(f)

    return role_of


def permanent_cached_roles(population, cache, reference, role_of):
    import concept_subgroup_drift_pilot as pilot
    reference_patients = set(population["patient_id"][population["year"] == reference].astype(str))
    roles = np.full(len(population["year"]), "", dtype="<U20")
    for i in np.flatnonzero(population["year"] >= reference):
        patient = str(population["patient_id"][i])
        roles[i] = (role_of.get(patient, "protected") if patient in reference_patients else
                    pilot.stable_entrant_role(patient, reference, cache.config["split_seed"]))

    return roles


def evaluation_role_key(reference):
    """Reference-year evaluation rows are saved, outcome-free, in the first later idea2_frozen unit."""
    return f"D/{reference}/idea2_frozen/{reference + 1}"


def frozen_roles(population, cache, reference, ctx):
    """Use only the cached row assignments for roles this study can access."""
    paths = {"rule_discovery": (f"D/{reference}/rules", "rule_row_index", "rule_rows_identity"),
             "cav_fit": (f"D/{reference}/cavs_selection", "cav_row_index", "cav_rows_identity"),
             "concept_selection": (f"D/{reference}/cavs_selection", "selection_row_index", "selection_rows_identity")}
    if ctx["mode"] == "evaluate":
        paths["evaluation"] = (evaluation_role_key(reference), "reference_row_index", "reference_rows_identity")
    predictor_meta = json.loads((cache._unit_path(f"P/{reference}") / "meta.json").read_text())

    role_of = cached_role_patients(population, cache, reference, ctx, paths, predictor_meta)
    roles = permanent_cached_roles(population, cache, reference, role_of)
    return {"row_role": roles, "identity": predictor_meta["roles_identity"]}


def validate_family_inventory(ctx, build, reference, aliases, refs, families):
    """Check saved correspondence IDs and frozen object identities before alias reuse."""
    expected = {name: sorted([list(map(int, member)) for member in families[name]["members"]])
                for name, value in refs.items() if any(value["eligible"].values())}
    for variant in aliases:
        path = ctx["CMR_PREVIOUS_RUNS"] / f"evaluate-{variant}/evaluate-{reference}.json"
        saved = json.loads(path.read_text())
        if saved["build"] != build:
            raise ValueError("Saved effective-build alias mismatch")
        members = {row["family"]: sorted(row["members"]) for row in saved["correspondence"]}
        if members != expected:
            raise ValueError("Saved correspondence family/member universe mismatch")
        identities = {row["family"]: row["reference_identity"] for row in saved["temporal"]["families"]}
        if any(identities.get(name) != refs[name]["identity"] for name in expected):
            raise ValueError("Saved alias cutoff/rule/CAV identity mismatch")
        ctx["input_files"][str(path)] = file_hash(path)


def load_study_population(ctx):
    import concept_subgroup_drift_pilot as pilot
    population = pilot.load_pilot_population(ctx["CMR_POPULATION"],
        "c8804591c93087241b9533dce77dd54b2f1e77b23a36400527459cfa8f99add7")
    ctx["input_files"][str(ctx["CMR_POPULATION"])] = file_hash(ctx["CMR_POPULATION"])
    first = {}
    for patient, year in zip(population["patient_id"], population["year"], strict=True):
        first[str(patient)] = min(first.get(str(patient), int(year)), int(year))

    return population, first


def study_universe(ctx, branch):
    variants = ("last_3", "all_history") if branch == "groups" or ctx["mode"] == "smoke" else ("last_3", "all_history", "baseline")
    universe = {}
    for variant in variants:
        for reference in range(2007, 2016 if branch == "groups" or ctx["mode"] == "smoke" else 2014):
            universe.setdefault((effective_build(variant, reference), reference), []).append(variant)

    return universe


def study_inventory(ctx, branch, caches, population):
    import concept_subgroup_drift_pilot as pilot
    universe = study_universe(ctx, branch)
    inventory = []
    for (build, reference), aliases in universe.items():
        cache = caches[build]
        required = [f"P/{reference}", f"D/{reference}/families", f"D/{reference}/rules",
                    f"D/{reference}/cavs_selection", *[f"S/{s}/{reference}" for s in (42, 135, 215)]]
        if ctx["mode"] == "evaluate":
            required.append(evaluation_role_key(reference))
        missing = [key for key in required if key not in cache.graph or not cache.is_complete(key)]
        if missing:
            ctx["coverage"].append({"build": build, "reference": reference, "status": "missing_dependency", "n": len(missing)})
            continue
        refs = pilot.load_frozen_references(cache, reference)
        families = {item["id"]: item for item in cache.load(f"D/{reference}/families")["families"]}
        validate_family_inventory(ctx, build, reference, aliases, refs, families)
        roles = frozen_roles(population, cache, reference, ctx)
        verify_roles(population["patient_id"], roles["row_role"])
        discovery = np.flatnonzero((population["year"] == reference) & (roles["row_role"] == "rule_discovery"))
        for family, frozen in refs.items():
            if (branch == "groups" or ctx["mode"] == "smoke") and not any(frozen["eligible"].values()):
                continue
            counts = frozen["cavs"][42]
            support = (min(counts.get("positive_count", 0), counts.get("eligible_negative_count", 0)), len(discovery))
            inventory.append((support, build, reference, aliases, family, frozen, families[family], roles))

    return inventory


def smoke_inventory(inventory, branch):
    ordered = sorted(inventory, key=lambda x: (x[0], x[1], x[2], x[4]))
    primary = [x for x in ordered if x[2] <= 2013]
    unshared = [x for x in primary if x[1] == "all_history" and x[2] >= 2010]
    secondary = [x for x in ordered if x[2] in (2014, 2015)]
    wanted = [primary[0], unshared[len(unshared)//2], secondary[-1]]
    if branch != "groups":
        wanted = [x for x in wanted if x[2] <= 2013]
    inventory = list({(x[1], x[2], x[4]): x for x in wanted}.values())

    return inventory


def record_cache_inputs(ctx, cache, reference):
    for key in [f"P/{reference}", f"D/{reference}/families", f"D/{reference}/rules",
                f"D/{reference}/cavs_selection", *[f"S/{s}/{reference}" for s in (42, 135, 215)]]:
        for f in cache._unit_path(key).iterdir():
            if f.is_file():
                ctx["input_files"][str(f)] = file_hash(f)


def verified_predictor_archive(population, cache, reference, roles):
    import concept_subgroup_drift_pilot as pilot
    directory = cache._unit_path(f"P/{reference}")
    meta = json.loads((directory / "meta.json").read_text())
    archive = directory / "arrays.npz"
    rows = read_array_rows(archive, "row_index")
    verify_order(rows, np.flatnonzero(population["year"] >= reference))
    if meta["roles_identity"] != roles["identity"] or meta["population_identity"] != population["fingerprint"]["population_identity"]:
        raise ValueError("Population/role identity mismatch")
    if pilot._identity_hash({"row_index": rows}) != meta["output_row_identity"]:
        raise ValueError("Cached ordered row identity mismatch")

    return meta, archive, rows


def allowed_predictor_positions(ctx, branch, population, rows, roles, reference):
    role = roles["row_role"][rows]
    years = population["year"][rows]
    if ctx["mode"] == "evaluate":
        allowed = (role == "evaluation") | ((role == "rule_discovery") & (branch != "groups"))
        if branch == "groups":
            allowed &= years == reference
    else:
        names = ["rule_discovery", "cav_fit", "concept_selection"] if branch == "measurement" else ["rule_discovery"]
        allowed = (years == reference) & np.isin(role, names)
    positions = np.flatnonzero(allowed)

    return positions


def selected_population(population, rows, positions, roles, first):
    selected = rows[positions]
    base = {"X": population["X"][selected], "feature_names": list(population["feature_names"]),
            "patients": population["patient_id"][selected].astype(str), "years": population["year"][selected],
            "roles": roles["row_role"][selected], "row_index": selected,
            "first_observed_year": np.array([first[str(p)] for p in population["patient_id"][selected]])}

    return base


def load_predictor_values(ctx, branch, population, archive, positions, base):
    selected = base["row_index"]
    base["embeddings"] = read_array_rows(archive, "embeddings", positions)
    for name in ("gradients", "probabilities"):
        base[name] = (read_array_rows(archive, name, positions) if branch == "measurement"
                      else np.empty((len(selected), 0)))
    base["labels"] = np.full(len(selected), np.nan)
    if ctx["mode"] == "evaluate" and branch == "measurement":
        legal = base["years"] <= 2014
        base["labels"][legal] = population["y"][selected[legal]]


def encode_frozen_dictionaries(ctx, cache, reference, base):
    import concept_subgroup_drift_pilot as pilot
    dictionaries = pilot.load_reference_dictionaries(cache, reference, ctx["device"])
    base["codes"], base["decoders"] = {}, {}
    for seed, dictionary in dictionaries.items():
        base["codes"][seed] = pilot.encode_with_reference_sae(dictionary, base["embeddings"], ctx["device"], 256)
        base["decoders"][seed] = dictionary["run"]["model"].encoder.weight.detach().cpu().numpy().copy()
        dictionary["run"]["model"].to("cpu")
    base["scale"] = dictionaries[42]["scaler"].scale_.copy()

    base["dictionary_ids"] = {str(s): dictionaries[s]["unit_identity"] for s in dictionaries}


def load_frozen_head(ctx, meta, archive, base):
    import concept_subgroup_drift_pilot as pilot
    selected = base["row_index"]
    continuation_meta = meta["continuation"]
    arrays = {name: read_array_rows(archive, name) for name in continuation_meta["array_names"]}
    base["continuation"] = pilot.rebuild_continuation(arrays, continuation_meta, ctx["device"])
    if len(selected):
        pilot.replay_continuation(base["continuation"], base["embeddings"][:8],
                                 base["probabilities"][:8], base["gradients"][:8])


def load_allowed_build(ctx, branch, population, cache, reference, roles, first):
    record_cache_inputs(ctx, cache, reference)
    meta, archive, rows = verified_predictor_archive(population, cache, reference, roles)
    positions = allowed_predictor_positions(ctx, branch, population, rows, roles, reference)
    base = selected_population(population, rows, positions, roles, first)
    load_predictor_values(ctx, branch, population, archive, positions, base)
    encode_frozen_dictionaries(ctx, cache, reference, base)
    base["continuation"] = None
    if branch == "measurement":
        load_frozen_head(ctx, meta, archive, base)
    return base


def discovery_split(ctx, cache, base, reference, family, seed, factor):
    import concept_subgroup_drift_pilot as pilot
    patients, years, role = base["patients"], base["years"], base["roles"]
    rule = np.flatnonzero((years == reference) & (role == "rule_discovery"))
    if ctx["mode"] != "evaluate":
        expected_rule_rows = cache.load(f"D/{reference}/rules")["rule_row_index"]
        verify_order(base["row_index"][rule], expected_rule_rows)
    fit = tune = np.array([], dtype=int)
    if len(rule):
        split_seed = pilot._derived_seed("rule_inner_split", cache.config["split_seed"], reference, family)
        fit_ids, tune_ids = pilot.rule_inner_split(patients[rule], base["codes"][seed][rule, factor], split_seed)
        fit, tune = rule[np.isin(patients[rule], fit_ids)], rule[np.isin(patients[rule], tune_ids)]
        if set(patients[fit]) & set(patients[tune]):
            raise ValueError("Inner patient role overlap")

    return fit, tune


def family_unit(ctx, base, cache, population, build, reference, aliases, family, frozen, family_record, roles):
    seed, factor = map(int, frozen["medoid"])
    years, role = base["years"], base["roles"]
    fit, tune = discovery_split(ctx, cache, base, reference, family, seed, factor)
    unit = {**base, "key": f"{build}/{reference}/{family}", "aliases": aliases,
            "reference": reference, "family": family, "seed": seed, "factor": factor,
            "cutoff": frozen["cutoff"], "eligible": frozen["eligible"], "historical": frozen,
            "activation": base["codes"][seed][:, factor], "rule_mask": frozen["rule"].mask(base["X"]),
            "rule_text": str(frozen["rule"].to_dict()), "rule_dict": frozen["rule"].to_dict(),
            "members": {int(s): int(k) for s, k in family_record["members"]},
            "fit": fit, "tune": tune,
            "cav_fit": np.flatnonzero((years == reference) & (role == "cav_fit")),
            "selection": np.flatnonzero((years == reference) & (role == "concept_selection")),
            "evaluation": np.flatnonzero(role == "evaluation")}
    # Freeze identities do not depend on whether reference or evaluation rows are exposed.
    unit["identity"] = {"build": build, "reference": reference, "family": family,
                        "reference_object": frozen["identity"], "roles": roles["identity"],
                        "population": population["fingerprint"]["population_identity"],
                        "dictionary": base["dictionary_ids"], "cutoff": frozen["cutoff"],
                        "members": family_record["members"], "config": cache.config_hash}

    return unit


def attach_yearly_description(ctx, unit, cache, aliases, namespaces):
    import concept_subgroup_drift_pilot as pilot
    recipe_path = ctx["CMR_PREVIOUS_RUNS"] / f"develop-{aliases[0]}/freeze.json"
    ctx["input_files"][str(recipe_path)] = file_hash(recipe_path)
    recipe = json.loads(recipe_path.read_text())
    if recipe["transform"] != {"kind": "counts"} or recipe["allocation"] != "original":
        raise ValueError("Historical count/original-allocation policy parity mismatch")
    recipe = {**recipe, "backend": pilot._backend_settings(),
              "selection": vars(pilot.rule_selection_config(cache.config["rule"]))}
    unit.update(yearly_recipe=recipe, yearly_description=namespaces["yearly_description"])


def iter_units(ctx, branch):
    """Yield exact family units after row/role identities, with mechanically masked inputs."""
    if ctx["mode"] == "evaluate":
        freeze_read(ctx)
    caches = open_caches(ctx)
    population, first = load_study_population(ctx)
    namespaces = legacy_descriptions(ctx["root"]) if branch != "groups" else None
    inventory = study_inventory(ctx, branch, caches, population)
    if ctx["mode"] == "smoke" and inventory:
        inventory = smoke_inventory(inventory, branch)
    loaded = {}
    for _, build, reference, aliases, family, frozen, family_record, roles in inventory:
        check_budget(ctx)
        cache = caches[build]
        if (build, reference) not in loaded:
            loaded.clear()
            loaded[(build, reference)] = load_allowed_build(ctx, branch, population, cache, reference, roles, first)
        base = loaded[(build, reference)]
        unit = family_unit(ctx, base, cache, population, build, reference, aliases, family, frozen, family_record, roles)
        if branch != "groups":
            attach_yearly_description(ctx, unit, cache, aliases, namespaces)
        ctx["coverage"].append({"key": unit["key"], "aliases": aliases, "status": "measured", "rows": len(base["patients"])})
        yield unit
