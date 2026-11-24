"""Identity and provenance checks for the saved temporal comparison inputs."""

import hashlib
import csv
import json
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from tabpfn.model.encoders import MinMaxScalingEncoderStep, Time2VecEncoderStep


_FITTED_IDENTITY_FIELDS = (
    "X_", "y_", "_categorical_features", "additional_x_", "categorical_features",
    "classes_", "c_processed_", "cached_preprocessors_", "cached_shuffles_",
    "data_mean_", "data_std_", "average_logits", "softmax_temperature",
    "ensemble_configurations_", "transformer_predict_kwargs", "preprocess_transforms",
    "feature_shift_decoder", "normalize_with_test", "normalize_to_ranking",
    "normalize_to_clip", "remove_outliers", "fit_at_predict_time", "fp16_inference",
    "sklearn_compatible_precision", "pilot_checkpoint_sha256_", "pilot_domain_policy_",
)


def matched_joint_inputs(classifier, ensemble_configurations, eval_xs, eval_ys, eval_position,
                         additional_xs=None, additional_ys=None, bar_dist=None,
                         cache_trainset_transforms=False):
    """Build a joint input with the cached clinical transform and shuffle, without fitting."""
    if (not classifier.is_classification_ or bar_dist is not None or cache_trainset_transforms
            or len(ensemble_configurations) != 1 or len(classifier.ensemble_configurations_) != 1
            or ensemble_configurations[0] is not classifier.ensemble_configurations_[0]
            or len(classifier.cached_preprocessors_) != 1 or len(classifier.cached_shuffles_) != 1
            or eval_position <= 0 or eval_position >= len(eval_xs)):
        raise ValueError("unsupported matched joint preprocessing contract")
    preprocessor, shuffle = classifier.cached_preprocessors_[0], classifier.cached_shuffles_[0]
    if preprocessor is None or shuffle is None:
        raise ValueError("missing fitted clinical transform or shuffle")
    train, train_categories = preprocessor.transform(eval_xs[:eval_position, 0])
    query, query_categories = preprocessor.transform(eval_xs[eval_position:, 0], is_test=True)
    if train_categories != query_categories:
        raise ValueError("clinical categories differ between context and query")
    combined = np.concatenate([train, query], axis=0).astype(np.float32)
    features, categories = shuffle.transform(torch.tensor(combined))
    features = features[:, None]
    config = ensemble_configurations[0]
    labels = eval_ys.clone()
    shift = config.class_shift_configuration
    if shift is not None:
        labels = labels.long()
        mask = labels != classifier.semisupervised_indicator
        labels[mask] = classifier._map_classes(labels[mask], shift)
    additional_xs = additional_xs or {}
    additional_ys = additional_ys or {}
    inputs_x = {key: [value] for key, value in additional_xs.items()}
    inputs_y = {key: [value] for key, value in additional_ys.items()}
    subsample = config.subsample_samples_configuration
    if subsample is not None:
        features = torch.cat((features[subsample], features[eval_position:]))
        labels = labels[subsample]
        for key, values in inputs_x.items():
            values[0] = torch.cat((values[0][subsample], values[0][eval_position:]))
    return [features], [labels], [categories], [None], [None], inputs_x, inputs_y


@contextmanager
def matched_joint_execution(classifier):
    """Temporarily use the frozen ensemble and transforms in the joint model call."""
    from concept_subgroup_drift_pilot import predictor_context_identity
    if getattr(classifier, "fit_at_predict_time", True) or not getattr(classifier, "cached_models_", None):
        raise ValueError("matched joint inference requires a fitted cached classifier")
    names = ("get_ensemble_configurations", "build_transformer_input_for_each_configuration")
    prior = {name: classifier.__dict__[name] for name in names if name in classifier.__dict__}
    prior_fit_at_predict_time = classifier.fit_at_predict_time
    before = predictor_context_identity(classifier)
    try:
        classifier.fit_at_predict_time = True
        classifier.get_ensemble_configurations = lambda *args, **kwargs: classifier.ensemble_configurations_
        classifier.build_transformer_input_for_each_configuration = partial(matched_joint_inputs, classifier)
        yield classifier
    finally:
        classifier.fit_at_predict_time = prior_fit_at_predict_time
        for name in names:
            if name in prior:
                setattr(classifier, name, prior[name])
            elif name in classifier.__dict__:
                delattr(classifier, name)
        if predictor_context_identity(classifier) != before:
            raise RuntimeError("matched joint execution changed the frozen fitted classifier")


@contextmanager
def observe_transformer_inputs(classifier):
    """Observe the exact preprocessed features and categorical indices used by the model."""
    name = "_batch_predict"
    present = name in classifier.__dict__
    previous = classifier.__dict__.get(name)
    original = getattr(classifier, name)
    observed = []

    def capture(*args, **kwargs):
        observed.append({"inputs": [value.detach().clone() for value in kwargs["inputs"]],
                         "categorical_inds": deepcopy(kwargs["categorical_inds"])})
        return original(*args, **kwargs)

    try:
        setattr(classifier, name, capture)
        yield observed
    finally:
        if present:
            setattr(classifier, name, previous)
        else:
            delattr(classifier, name)


class _FrozenUnits:
    """Read named frozen units needed by the reference rule and dictionary helpers."""

    def __init__(self, directory, config):
        self.directory, self.config = directory, config

    def load(self, key):
        path = self.directory.joinpath(*key.split("/"))
        result = json.loads((path / "meta.json").read_text())
        archive = path / "arrays.npz"
        if archive.is_file():
            with np.load(archive, allow_pickle=False) as arrays:
                result.update({name: arrays[name] for name in arrays.files})
        return result


def select_primary_builds(builds):
    """Select the saved builds serving the seven primary reference years."""
    selected = [record for record in builds if "last_3" in record["variants"]]
    references = [int(record["reference"]) for record in selected]
    if len(references) != len(set(references)) or set(references) != set(range(2007, 2014)):
        raise ValueError("saved primary builds do not uniquely cover references 2007–2013")
    return selected


def tabpfn_source_identity(package_root):
    """Fingerprint the installed fork's Python source tree, excluding weights and caches."""
    root = Path(package_root)
    paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not paths:
        return None
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_hash = inventory_inputs({"source": path}, {})["files"]["source"]["sha256"]
        digest.update(bytes.fromhex(file_hash))
    return {"method": "tabpfn-python-tree-sha256-v1", "sha256": digest.hexdigest()}


def tabpfn_identity_matches(historical, current):
    """Require the same recorded fork source digest; package version alone is insufficient."""
    return (isinstance(historical, dict) and isinstance(current, dict)
            and historical.get("method") == current.get("method") == "tabpfn-python-tree-sha256-v1"
            and isinstance(historical.get("sha256"), str) and len(historical["sha256"]) == 64
            and historical == current)


def audit_frozen_inputs(ledger_path, project_root=Path("."), history_root=None):
    """Audit the existing population, saved builds and reference inputs without inference."""
    import concept_subgroup_drift_pilot as pilot
    from importlib import resources
    import tabpfn

    root = Path(project_root)
    history = Path(history_root) if history_root is not None else Path.home() / "rpi-runs" / "SAE-TCAV"
    study = root / "stats/concept_agreement_study"
    index_path = study / "patient_table/index.json"
    index = json.loads(index_path.read_text())
    primary_builds = select_primary_builds(index["builds"])
    with Path(ledger_path).open(newline="") as stream:
        ledger = [row for row in csv.DictReader(stream) if row["in_62_primary_tree_basis"] == "1"]
    cases = audit_ledger(ledger, primary_builds)
    primary_concepts = [item for item in index["concepts"] if "last_3" in item["variants"]]
    indexed = {item["concept"]: item for item in primary_concepts}
    if len(indexed) != len(primary_concepts) or set(indexed) != {item["full_identity"] for item in cases}:
        raise ValueError("work 0 ledger and saved concept identities disagree")
    population_path = root / "stats/temporal_robustness/_population/prepared.pkl"
    expected_population_hash = "c8804591c93087241b9533dce77dd54b2f1e77b23a36400527459cfa8f99add7"
    population = pilot.load_pilot_population(population_path, expected_population_hash)
    source_population = {"patient_id": population["patient_id"], "year": population["year"],
                         "label": population["y"], "features": population["X"],
                         "feature_names": population["feature_names"]}
    population_keys = _row_keys(source_population, "population")
    row_lookup = {key: position for position, key in enumerate(population_keys)}
    checkpoint = Path(str(resources.files(tabpfn))) / "model_cache/tabpfn_dist_model_1.cpkt"
    package_root = Path(str(resources.files(tabpfn)))
    common_paths = {"ledger": ledger_path, "index": index_path, "population": population_path,
                    "checkpoint": checkpoint, "pilot_source": root / "concept_subgroup_drift_pilot.py",
                    "pilot_notebook": root / "concept_subgroup_drift_pilot.ipynb",
                    "semantic_rules": root / "semantic_rules.py", "sae_source": root / "sae.py",
                    "predictor_wrapper": root / "tabpfn_model.py",
                    "tabpfn_encoders": package_root / "model/encoders.py",
                    "tabpfn_utils": package_root / "utils.py",
                    "tabpfn_estimator": package_root / "scripts/estimator/base.py",
                    "tabpfn_preprocessing": package_root / "scripts/estimator/preprocessing.py",
                    "tabpfn_model_builder": package_root / "scripts/model_builder.py"}
    common = inventory_inputs(common_paths, {"historical_tabpfn_package": None,
                                              "current_tabpfn_source_identity": tabpfn_source_identity(package_root)})
    systems, coverage = [], []
    for record in primary_builds:
        build, reference = record["build"], int(record["reference"])
        run = history / ("r1-pilot-fe945-20261009" if build == "r1" else f"pilot-ideas-followups/builds/{build}")
        config_path = run / "out/resolved-config.json"
        manifest_path = run / "out/run-manifest.json"
        config = json.loads(config_path.read_text())
        unit = run / "unit-cache" / config["config_hash"] / "units"
        frozen = _FrozenUnits(unit, config)
        rules = pilot.load_frozen_rules(frozen, reference)
        dictionaries = pilot.load_reference_dictionaries(frozen, reference, "cpu")
        predictor = unit / "P" / str(reference)
        metadata_path, archive_path = predictor / "meta.json", predictor / "arrays.npz"
        meta = json.loads(metadata_path.read_text())
        if meta["reference_year"] != reference:
            raise ValueError(f"wrong reference for {build}/{reference}")
        roles = pilot.reference_year_roles(population, reference, seed=config["split_seed"])
        if roles["identity"] != meta["roles_identity"]:
            raise ValueError(f"role identity mismatch for {build}/{reference}")
        year = np.asarray(population["year"])
        window = config.get("context_window", "reference_only")
        first = {"reference_only": reference, "last_3": max(pilot.FIRST_CONTEXT_YEAR, reference - 2),
                 "all_history": pilot.FIRST_CONTEXT_YEAR}[window]
        context = ((year == reference) & (roles["row_role"] == "tabpfn_context") & ~roles["is_entrant"])
        context |= (year >= first) & (year < reference) & ~population["post_death"]
        rows = np.flatnonzero(context)
        context_frame = pilot._prediction_frame(population, rows, pilot._feature_columns(population["X"].shape[1]))
        context_identity = pilot._identity_hash(list(zip(context_frame.patient_id, context_frame.year)))
        if context_identity != meta["context_rows_identity"]:
            raise ValueError(f"context row identity mismatch for {build}/{reference}")
        actual_path = study / "patient_table" / record["file"]
        held_path = study / "year_input" / record["file"]
        with np.load(actual_path, allow_pickle=False) as actual, np.load(held_path, allow_pickle=False) as held:
            row_report = audit_paired_rows(actual, held, source_population)
            positions = np.asarray([row_lookup[key] for key in _row_keys(actual, "actual")])
            with np.load(archive_path, allow_pickle=False) as saved:
                archive_rows = saved["row_index"]
                expected_rows = np.flatnonzero(year >= reference)
                if (not np.array_equal(archive_rows, expected_rows)
                        or pilot._identity_hash({"row_index": archive_rows}) != meta["output_row_identity"]):
                    raise ValueError(f"predictor archive row identity mismatch for {build}/{reference}")
                at = np.searchsorted(archive_rows, positions)
                if (np.any(at >= len(archive_rows)) or not np.array_equal(archive_rows[at], positions)
                        or not np.array_equal(saved["probabilities"][at, 1], actual["probability"])):
                    raise ValueError(f"saved mortality probabilities mismatch for {build}/{reference}")
                if not np.isfinite(saved["embeddings"][at]).all():
                    raise ValueError(f"nonfinite saved embeddings for {build}/{reference}")
                embeddings = saved["embeddings"][at].copy()
                precision = {name: str(saved[name].dtype) for name in ("probabilities", "embeddings")}
            encoded = {seed: pilot.encode_with_reference_sae(dictionary, embeddings, "cpu")
                       for seed, dictionary in dictionaries.items()}
            for case in (case for case in cases if case["build"] == build and case["reference_year"] == reference):
                concept = indexed[case["full_identity"]]
                concept_path = study / "patient_table/concepts" / concept["file"]
                with np.load(concept_path, allow_pickle=False) as saved_concept:
                    activation = saved_concept["activation"]
                    if (not np.isfinite(activation).all() or len(activation) != len(positions)
                            or not np.array_equal(saved_concept["in_A"], activation > concept["cutoff"])
                            or not np.array_equal(activation.astype(np.float16),
                                                  actual[f"codes_{concept['seed']}"][:, concept["factor"]])):
                        raise ValueError(f"frozen activation mismatch for {case['full_identity']}")
                    if saved_concept["in_R"].shape != (len(positions),):
                        raise ValueError(f"frozen rule rows mismatch for {case['full_identity']}")
                    rule = rules[case["family"]]["rule"]
                    if not np.array_equal(saved_concept["in_R"],
                                          pilot.rule_mask(rule, actual["features"], actual["feature_names"])):
                        raise ValueError(f"frozen rule evaluation mismatch for {case['full_identity']}")
                    replayed = encoded[concept["seed"]][:, concept["factor"]]
                    if not np.allclose(activation, replayed, rtol=1e-4, atol=1e-5):
                        raise ValueError(f"reference SAE replay mismatch for {case['full_identity']}")
                    activation_error = float(np.max(np.abs(activation - replayed))) if len(activation) else 0.0
                    disagrees = (activation > concept["cutoff"]) != (replayed > concept["cutoff"])
                    cutoff_disagreements = int(np.sum(disagrees))
                    by_year = {str(int(yr)): int(np.sum(disagrees & (actual["year"] == yr)))
                               for yr in np.unique(actual["year"])}
                    boundary_error = float(np.max(np.abs(activation[disagrees] - replayed[disagrees]))) if cutoff_disagreements else None
                coverage.append({"concept": case["full_identity"], "build": build, "reference": reference,
                                 "aliases": ",".join(case["aliases"]),
                                 "fit_positive": case["fit_positive"],
                                 "cav_support_status": case["measurement_reference_status"],
                                 "sae_replay_max_abs_error": activation_error,
                                 "sae_cutoff_disagreements": cutoff_disagreements,
                                 "sae_cutoff_disagreements_by_year": by_year,
                                 "sae_disagreement_max_abs_error": boundary_error,
                                 "activation_mask_source": "saved historical float32 activation and original strict cutoff",
                                 "concept_file_sha256": inventory_inputs({"concept": concept_path}, {})["files"]["concept"]["sha256"]})
        files = {"config": config_path, "manifest": manifest_path, "predictor_meta": metadata_path,
                 "predictor_archive": archive_path, "actual": actual_path, "held": held_path,
                 "rules_meta": unit / "D" / str(reference) / "rules/meta.json",
                 **{f"sae_{seed}_{name}": unit / "S" / str(seed) / str(reference) / name
                    for seed in (42, 135, 215) for name in ("meta.json", "arrays.npz")}}
        file_report = inventory_inputs(files, {})["files"]
        manifest = json.loads(manifest_path.read_text())
        systems.append({"build": build, "reference": reference, "variants": record["variants"],
                        "death_threshold": record["death_threshold"], "context_rows": len(rows),
                        "context_years": sorted(np.unique(year[rows]).astype(int).tolist()),
                        "domain_origin": first, "context_rows_identity": context_identity,
                        "roles_identity": roles["identity"], "saved_context_identity": meta["context_identity"],
                        "saved_checkpoint_sha256": meta["checkpoint_sha256"],
                        "current_checkpoint_matches": common["files"]["checkpoint"]["sha256"] == meta["checkpoint_sha256"],
                        "saved_device": meta["device"], "saved_batch_rows": meta["batch_rows"],
                        "saved_source_hashes": manifest.get("source_file_hashes"),
                        "historical_tabpfn_source_identity": (manifest.get("dependency_source_identities") or {}).get("tabpfn") or "unknown",
                        "saved_environment": manifest.get("environment"), "saved_git": manifest.get("git"),
                        "archive_dtypes": precision, "rows": row_report, "files": file_report})
    return {"common": common, "systems": systems, "coverage": coverage}


def pack_parity_probes(saved, population, reference, domain_origin):
    """Pack the first saved row for each year, in actual/held order."""
    years = np.asarray(saved["year"])
    lookup = {(str(patient), int(year)): i for i, (patient, year) in
              enumerate(zip(population["patient_id"], population["year"]))}
    rows, domains, settings, missing = [], [], [], []
    for year in range(reference, 2015):
        selected = np.flatnonzero(years == year)
        if not len(selected):
            missing.append(year)
            continue
        index = selected[0]
        row = lookup[(str(saved["patient_id"][index]), year)]
        for arm, supplied in (("actual", year), ("held", reference)):
            rows.append(row)
            domains.append(supplied - domain_origin)
            settings.append({"year": year, "input": arm})
    return {"features": np.ascontiguousarray(population["X"][rows], dtype=np.float32),
            "domains": np.asarray(domains, dtype=np.int64), "settings": settings,
            "missing_years": missing}


def compare_packed_parity(system, probes, cached, joint, cached_inputs, joint_inputs):
    """Compare every packed query at the model boundary and in its outputs."""
    count = len(probes["settings"])
    expected = [{"year": year, "input": arm} for year in range(system["reference"], 2015)
                if year not in probes["missing_years"] for arm in ("actual", "held")]
    if probes["settings"] != expected:
        raise ValueError("parity probe mapping is missing or reordered")
    cf = cached_inputs[0]["inputs"][0]
    jf = joint_inputs[0]["inputs"][0][-count:]
    categories_equal = cached_inputs[0]["categorical_inds"] == joint_inputs[0]["categorical_inds"]
    cp, jp = np.asarray(_identity_value(cached["probabilities"])), np.asarray(_identity_value(joint["probabilities"]))
    ce, je = np.asarray(_identity_value(cached["embeddings"])), np.asarray(_identity_value(joint["embeddings"]))
    ct = cached["time_values"].detach().cpu().numpy()
    jt = joint["time_values"][-count:].detach().cpu().numpy()
    if any(len(item) != count for item in (cf, jf, cp, jp, ce, je, ct, jt)):
        raise ValueError("packed parity output rows do not match probe mapping")
    reports = []
    for i, setting in enumerate(probes["settings"]):
        clinical_equal = torch.equal(cf[i], jf[i])
        probability_error = float(np.max(np.abs(cp[i] - jp[i])))
        embedding_error = float(np.max(np.abs(ce[i] - je[i])))
        time_error = float(np.max(np.abs(ct[i] - jt[i])))
        passed = (clinical_equal and categories_equal and np.allclose(cp[i], jp[i], rtol=1e-4, atol=1e-5)
                  and np.allclose(ce[i], je[i], rtol=1e-4, atol=1e-5)
                  and np.allclose(ct[i], jt[i], rtol=1e-4, atol=1e-5))
        control = "not applicable"
        if setting == {"year": system["reference"], "input": "held"}:
            if i == 0 or probes["settings"][i - 1] != {"year": system["reference"], "input": "actual"}:
                raise ValueError("reference probe mapping is missing or reordered")
            control_passed = (np.allclose(cp[i], cp[i - 1], rtol=1e-4, atol=1e-5)
                              and np.allclose(ce[i], ce[i - 1], rtol=1e-4, atol=1e-5))
            passed = passed and control_passed
            control = "pass" if control_passed else "fail"
        reports.append({"build": system["build"], "reference": system["reference"], **setting,
                        "rows": 1, "clinical_features_equal": clinical_equal,
                        "categorical_indices_equal": categories_equal, "time_max_abs_error": time_error,
                        "reference_actual_held_control": control, "probability_max_abs_error": probability_error,
                        "embedding_max_abs_error": embedding_error, "parity": "pass" if passed else "fail"})
    return reports


def _initialize_context_classifier(system, device):
    """Create the frozen single-fast classifier before fitting its context cache."""
    from importlib import resources
    import tabpfn
    import concept_subgroup_drift_pilot as pilot
    from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig

    config = json.loads(Path(system["files"]["config"]["path"]).read_text())
    checkpoint = Path(str(resources.files(tabpfn))) / "model_cache/tabpfn_dist_model_1.cpkt"
    paths = TabPFNModelPathsConfig(paths=[str(checkpoint)], task_type="dist_shift_multiclass")
    classifier = get_best_tabpfn(task_type="dist_shift_multiclass", model_type="single_fast",
                                paths_config=paths, debug=False, device=device)
    classifier.seed = int(config["split_seed"])
    classifier.fit_at_predict_time = False
    classifier.sklearn_compatible_precision = False
    classifier.fp16_inference = False
    classifier.normalize_with_test = False
    classifier.pilot_precision_ = pilot.configure_predictor_precision(device)
    classifier.pilot_checkpoint_sha256_ = system["saved_checkpoint_sha256"]
    classifier.pilot_domain_policy_ = ("calendar_year_minus_reference_year"
                                       if config.get("context_window", "reference_only") == "reference_only"
                                       else "calendar_year_minus_first_context_year")
    if hasattr(classifier, "show_progress"):
        classifier.show_progress = False
    classifier.init_model_and_get_model_config()
    return classifier, config


def _native_positional_embeddings(fitted_predictor):
    """Read the executed cached model's positional tensor after native historical fitting."""
    models = getattr(fitted_predictor.get("classifier"), "cached_models_", ())
    if len(models) != 1:
        raise ValueError("native state reconstruction requires one executed cached model")
    state = getattr(models[0], "cached_embeddings", None)
    if (not torch.is_tensor(state) or state.ndim != 2 or state.numel() == 0
            or state.dtype != torch.float32 or not torch.isfinite(state).all().item()):
        raise ValueError("native fitted model lacks a finite float32 positional cache")
    return state.detach().clone()


def _identity_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _identity_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_identity_value(item) for item in value)
    if isinstance(value, list):
        return [_identity_value(item) for item in value]
    return value


def _clinical_input_identity(pilot, X, y, additional_x, classes):
    additional = _identity_value(additional_x)
    if isinstance(additional, dict) and "dist_shift_domain" in additional:
        domain = np.asarray(additional["dist_shift_domain"]).reshape(-1)
        if not np.isfinite(domain).all() or not np.array_equal(domain, np.rint(domain)):
            raise ValueError("fitted time-domain inputs are invalid")
        additional["dist_shift_domain"] = domain.astype(np.int64)
    class_values = np.asarray(_identity_value(classes)).reshape(-1)
    return pilot._identity_hash({"X": np.asarray(_identity_value(X), dtype=np.float32),
                                 "y": np.asarray(_identity_value(y), dtype=np.float32),
                                 "additional_x": additional,
                                 "classes": [int(value) for value in class_values]})


def _current_scoring_context(pilot, population, system, positional_state=None):
    config = json.loads(Path(system["files"]["config"]["path"]).read_text())
    reference = int(system["reference"])
    years = np.asarray(population["year"])
    roles = pilot.reference_year_roles(population, reference, seed=config["split_seed"])
    first = int(system["domain_origin"])
    context_mask = ((years == reference) & (roles["row_role"] == "tabpfn_context")
                    & ~roles["is_entrant"])
    context_mask |= (years >= first) & (years < reference) & ~population["post_death"]
    context_rows = np.flatnonzero(context_mask)
    columns = pilot._feature_columns(population["X"].shape[1])
    context = pilot._prediction_frame(population, context_rows, columns)
    y = np.ascontiguousarray(population["y"][context_rows], dtype=np.float32)
    context["y"] = y
    X = np.ascontiguousarray(context[columns].to_numpy(dtype=np.float32))
    origin = reference if config.get("context_window", "reference_only") == "reference_only" else first
    additional_x = {"dist_shift_domain": np.ascontiguousarray(years[context_rows] - origin, dtype=np.int64)}
    context_identity = pilot._identity_hash(list(zip(context.patient_id, context.year)))
    bindings = {"system_identity": f"{system['build']}/{reference}",
                "clinical_input_identity": _clinical_input_identity(pilot, X, y, additional_x, np.array([0, 1])),
                "context_rows_identity": context_identity,
                "checkpoint_sha256": system["saved_checkpoint_sha256"],
                "positional_state": ({"sha256": positional_state.get("sha256"),
                                      "shape": positional_state.get("shape"),
                                      "dtype": str(positional_state.get("dtype", ""))}
                                     if isinstance(positional_state, dict) else {})}
    return context, columns, context_rows, X, y, additional_x, bindings, config


def _comparable_predictor_identity(pilot, classifier, context_rows_identity, system_identity="unknown"):
    """Expose device-independent fitted fields while retaining opaque and device identities."""
    import r1_2007_device_diagnostic as diagnostic

    models = getattr(classifier, "cached_models_", ())
    if len(models) != 1:
        raise ValueError("comparable identity requires one executed cached model")
    model = models[0]
    fields = pilot.predictor_identity_report(classifier).get("field_digests", {})
    required = _FITTED_IDENTITY_FIELDS
    if any(name not in fields for name in required):
        raise ValueError("comparable predictor identity lacks explicit fitted fields")
    parameter_hash = pilot._identity_hash({name: value.detach().cpu() for name, value in model.named_parameters()})
    positional = getattr(model, "cached_embeddings", None)
    if positional is None:
        raise ValueError("comparable predictor has no executed positional state")
    device = str(next(model.parameters()).device)
    execution = diagnostic._execution_state(model)
    depth = sum(item["num_layers"] for item in execution["layer_stacks"])
    dropout = tuple(item["probability"] for item in execution["attention_dropout"])
    identity_digest = lambda names: hashlib.sha256(json.dumps(
        {name: fields[name] for name in names}, sort_keys=True).encode()).hexdigest()
    clinical_input_identity = _clinical_input_identity(
        pilot, classifier.X_, classifier.y_, classifier.additional_x_, classifier.classes_)
    configuration_fields = ("_categorical_features", "categorical_features", "softmax_temperature",
        "average_logits", "ensemble_configurations_", "transformer_predict_kwargs", "normalize_with_test",
        "normalize_to_ranking", "normalize_to_clip", "remove_outliers", "fit_at_predict_time",
        "fp16_inference", "sklearn_compatible_precision", "pilot_checkpoint_sha256_", "pilot_domain_policy_")
    transform_fields = ("c_processed_", "cached_preprocessors_", "cached_shuffles_", "data_mean_", "data_std_",
                        "preprocess_transforms", "feature_shift_decoder")
    return {"opaque_identity": pilot.predictor_context_identity(classifier), "device": device,
            "system_identity": system_identity,
            "clinical_input_identity": clinical_input_identity, "context_rows_identity": context_rows_identity,
            "checkpoint_sha256": classifier.pilot_checkpoint_sha256_,
            "learned_parameter_identity": parameter_hash,
            "configuration_identity": identity_digest(configuration_fields),
            "clinical_transform_identity": identity_digest(transform_fields),
            "fitted_fields_identity": identity_digest(_FITTED_IDENTITY_FIELDS),
            "precision": {"dtype": "fp32", "amp": "off", "tf32": False},
            "positional_state": {"sha256": _numpy_tensor_sha256(positional.detach().cpu().numpy()),
                                  "shape": list(positional.shape), "dtype": str(positional.dtype)},
            "execution_state": {"eval": not model.training, "depth": depth, "dropout": list(dropout)}}


def _current_state_input_fingerprints(audit, system):
    fingerprints = {}
    for section, files in (("common", audit["common"]["files"]), ("system", system["files"])):
        for name, entry in files.items():
            if entry.get("status") != "available" or not entry.get("sha256"):
                raise ValueError("generated state requires complete current input fingerprints")
            fingerprints[f"{section}:{name}"] = entry["sha256"]
    for row in audit["coverage"]:
        if (row["build"], int(row["reference"])) == (system["build"], int(system["reference"])):
            fingerprints[f"concept:{row['concept']}"] = row["concept_file_sha256"]
    return fingerprints


def _generated_state_for_system(audit, system, population, device, source_sha256, root=Path(".")):
    """Load the validated r1 tensor or capture a native CPU candidate before any fit."""
    build, reference = system["build"], int(system["reference"])
    fingerprints = _current_state_input_fingerprints(audit, system)
    input_sha256 = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
    if build == "r1" and reference == 2007:
        common, files = audit["common"]["files"], system["files"]
        recovered_source = (Path(root) / "stats/concept_agreement_study/extensions/r1_2007_replay"
                            / "recovered/concept_subgroup_drift_pilot.py")
        diagnostic_inputs = {
            "actual": files["actual"]["sha256"], "checkpoint": common["checkpoint"]["sha256"],
            "config": files["config"]["sha256"], "held": files["held"]["sha256"],
            "index": common["index"]["sha256"], "manifest": files["manifest"]["sha256"],
            "pilot_source": common["pilot_source"]["sha256"],
            "population": common["population"]["sha256"],
            "predictor_archive": files["predictor_archive"]["sha256"],
            "predictor_meta": files["predictor_meta"]["sha256"],
            "predictor_wrapper": common["predictor_wrapper"]["sha256"],
            "recovered_pilot_source": _file_sha256(recovered_source),
        }
        diagnostic = (Path(root) / "stats/concept_agreement_study/extensions/r1_2007_replay"
                      / "device_state_diagnostic")
        independent = (Path(root) / "stats/concept_agreement_study/extensions/temporal_input_attribution"
                       / "generated_state/evidence/independent-verification.public.json")
        loaded = load_r1_diagnostic_positional_state(
            diagnostic, diagnostic_inputs, independent_verification_path=independent)
        positional = torch.from_numpy(loaded["positional_tensor"].copy()).to(device=device)
        origin = "validated_r1_gpu_diagnostic"
        state_source_sha256 = loaded["provenance"]["source_sha256"]
        device_origin = "cuda:0"
    else:
        import concept_subgroup_drift_pilot as pilot
        config = json.loads(Path(system["files"]["config"]["path"]).read_text())
        roles = pilot.reference_year_roles(population, reference, seed=int(config["split_seed"]))
        years = np.asarray(population["year"])
        first = int(system["domain_origin"])
        context_mask = ((years == reference) & (roles["row_role"] == "tabpfn_context")
                        & ~roles["is_entrant"])
        context_mask |= ((years >= first) & (years < reference) & ~population["post_death"])
        context_rows = np.flatnonzero(context_mask)
        columns = pilot._feature_columns(population["X"].shape[1])
        context = pilot._prediction_frame(population, context_rows, columns)
        context["y"] = population["y"][context_rows]
        fit_options = ({} if config.get("context_window", "reference_only") == "reference_only"
                       else {"domain_origin": first})
        native_predictor = pilot.fit_frozen_predictor(
            context, columns, device, seed=int(config["split_seed"]), **fit_options)
        positional = _native_positional_embeddings(native_predictor)
        origin = "native_cpu_reconstruction_candidate"
        state_source_sha256 = source_sha256
        device_origin = device
        del native_predictor
    if positional.device.type != device or positional.dtype != torch.float32:
        raise ValueError("generated positional state could not be placed on the requested device")
    provenance = {"origin": origin, "build": build, "reference": reference,
                  "device": device, "device_origin": device_origin,
                  "source_sha256": source_sha256, "state_source_sha256": state_source_sha256,
                  "input_fingerprints": fingerprints, "input_sha256": input_sha256,
                  "historical_environment": "unknown", "historical_dependency_identity": "unknown",
                  "positional_state": {"shape": list(positional.shape), "dtype": str(positional.dtype),
                                        "sha256": _numpy_tensor_sha256(positional.detach().cpu().numpy())}}
    return {"positional_tensor": positional, "provenance": provenance}


def check_system_parity(system, population, device, positional_embeddings):
    """Check a small outcome-blind batch per year and input against a matched joint call."""
    import time
    import concept_subgroup_drift_pilot as pilot
    from tabpfn_model import infer_model_additional_x_info, make_dist_tensor

    reference, build = system["reference"], system["build"]
    classifier, config = _initialize_context_classifier(system, device)
    roles = pilot.reference_year_roles(population, reference, seed=config["split_seed"])
    year = np.asarray(population["year"])
    first = system["domain_origin"]
    context_mask = (year == reference) & (roles["row_role"] == "tabpfn_context") & ~roles["is_entrant"]
    context_mask |= (year >= first) & (year < reference) & ~population["post_death"]
    context_rows = np.flatnonzero(context_mask)
    columns = pilot._feature_columns(population["X"].shape[1])
    context = pilot._prediction_frame(population, context_rows, columns)
    prepared = prepare_context_classifier(classifier, positional_embeddings)
    parameter_hash = pilot._identity_hash({name: value.detach().cpu() for name, value in prepared.model.named_parameters()})
    started = time.monotonic()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with pilot.predictor_sdpa_context(device):
        prepared.fit(np.ascontiguousarray(context[columns].to_numpy(dtype=np.float32)),
                     np.ascontiguousarray(population["y"][context_rows], dtype=np.float32),
                     additional_x={"dist_shift_domain": np.ascontiguousarray(year[context_rows] - first, dtype=np.int64)})
    cached_parameter_hash = pilot._identity_hash({name: value.detach().cpu() for name, value in prepared.cached_models_[0].named_parameters()})
    if parameter_hash != cached_parameter_hash:
        raise RuntimeError("learned model parameters changed during cache construction")
    if device == "cuda":
        torch.cuda.synchronize()
    fit_seconds = time.monotonic() - started
    identity = lambda: pilot.predictor_context_identity(prepared)
    actual_path = Path(system["files"]["actual"]["path"])
    saved = np.load(actual_path, allow_pickle=False)
    try:
        probes = pack_parity_probes(saved, population, reference, first)
    finally:
        saved.close()
    if probes["missing_years"]:
        raise ValueError(f"missing parity probe years: {probes['missing_years']}")
    additional_device, shape = infer_model_additional_x_info(prepared)
    domains = make_dist_tensor(probes["domains"], additional_device, shape)
    def predict():
        with pilot.predictor_sdpa_context(device):
            return prepared.predict_full(probes["features"], additional_x={"dist_shift_domain": domains},
                                         get_additional_outputs=["test_embeddings"])
    started = time.monotonic()
    with observe_transformer_inputs(prepared) as cached_inputs:
        cached = capture_same_call(prepared, prepared.cached_models_[0], predict, identity, pilot._frozen_continuation)
    with matched_joint_execution(prepared):
        with observe_transformer_inputs(prepared) as joint_inputs:
            joint = capture_same_call(prepared, prepared.model_processed_, predict, identity, pilot._frozen_continuation)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    reports = compare_packed_parity(system, probes, cached, joint, cached_inputs, joint_inputs)
    return {"build": build, "reference": reference, "fit_seconds": fit_seconds,
            "query_check_seconds": elapsed, "learned_parameter_sha256": parameter_hash,
            "comparable_predictor_identity": _comparable_predictor_identity(
                pilot, prepared, system["context_rows_identity"], f"{system['build']}/{system['reference']}"),
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else None,
            "parity": reports}


def check_historical_reconstruction(system, population, device, positional_embeddings):
    """Compare small current-source cached calls with the saved historical arrays."""
    import time
    import concept_subgroup_drift_pilot as pilot

    reference, build = system["reference"], system["build"]
    config = json.loads(Path(system["files"]["config"]["path"]).read_text())
    roles = pilot.reference_year_roles(population, reference, seed=config["split_seed"])
    year = np.asarray(population["year"])
    first = system["domain_origin"]
    context_mask = (year == reference) & (roles["row_role"] == "tabpfn_context") & ~roles["is_entrant"]
    context_mask |= (year >= first) & (year < reference) & ~population["post_death"]
    context_rows = np.flatnonzero(context_mask)
    columns = pilot._feature_columns(population["X"].shape[1])
    context = pilot._prediction_frame(population, context_rows, columns)
    context["y"] = population["y"][context_rows]
    window = config.get("context_window", "reference_only")
    fit_options = {} if window == "reference_only" else {"domain_origin": first}
    started = time.monotonic()
    predictor = _fit_frozen_predictor_with_state(
        context, columns, system, device, int(config["split_seed"]), positional_embeddings, **fit_options)
    if device == "cuda":
        torch.cuda.synchronize()
    fit_seconds = time.monotonic() - started
    lookup = {(str(patient), int(yr)): i for i, (patient, yr) in enumerate(zip(population["patient_id"], year))}
    reports = []
    with np.load(system["files"]["actual"]["path"], allow_pickle=False) as actual, \
            np.load(system["files"]["held"]["path"], allow_pickle=False) as held, \
            np.load(system["files"]["predictor_archive"]["path"], allow_pickle=False) as archived:
        held_lookup = {key: i for i, key in enumerate(_row_keys(held, "held"))}
        archive_rows = archived["row_index"]
        for evaluation_year in sorted(set(actual["year"].tolist()) & set(range(reference, 2015))):
            at = int(np.flatnonzero(actual["year"] == evaluation_year)[0])
            key = (str(actual["patient_id"][at]), int(evaluation_year))
            population_row = lookup[key]
            archive_at = int(np.searchsorted(archive_rows, population_row))
            frame = pilot._prediction_frame(population, [population_row], columns)
            for arm, supplied in (("actual", evaluation_year), ("held", reference)):
                changed = frame.copy()
                changed["year"] = supplied
                observed = pilot.capture_frozen_predictions(predictor, changed, device)
                probability = observed["probabilities"].detach().cpu().numpy()[0, 1]
                if arm == "actual":
                    expected = actual["probability"][at]
                    embedding = observed["embeddings"].detach().cpu().numpy()[0]
                    embedding_error = float(np.max(np.abs(embedding - archived["embeddings"][archive_at])))
                    embedding_match = bool(np.allclose(embedding, archived["embeddings"][archive_at], rtol=1e-4, atol=1e-5))
                elif evaluation_year == reference:
                    expected, embedding_error, embedding_match = actual["probability"][at], None, None
                else:
                    if key not in held_lookup:
                        raise ValueError(f"missing held archive row for {build}/{reference}/{evaluation_year}")
                    expected = held["probability_swapped"][held_lookup[key]]
                    embedding_error, embedding_match = None, None
                reports.append({"build": build, "reference": reference, "year": int(evaluation_year), "input": arm,
                                "rows": 1, "kind": "current_source_reconstruction_not_exact_historical_batch",
                                "probability_max_abs_error": float(abs(probability - expected)),
                                "probability_within_fp32_tolerance": bool(np.isclose(probability, expected, rtol=1e-4, atol=1e-5)),
                                "embedding_max_abs_error": embedding_error, "embedding_within_fp32_tolerance": embedding_match,
                                "held_embedding_status": "not archived" if arm == "held" else "not applicable"})
    return {"fit_seconds": fit_seconds, "replay": reports,
            "comparable_predictor_identity": _comparable_predictor_identity(
                pilot, predictor["classifier"], system["context_rows_identity"],
                f"{system['build']}/{system['reference']}")}


def _finite_max_abs_error(left, right):
    """Return a JSON-safe finite difference, or None when it cannot be measured."""
    left, right = np.asarray(left), np.asarray(right)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    if difference.size == 0 or not np.isfinite(difference).all():
        return None
    return float(np.max(difference))


def _system_device(system, device, r1_device):
    if device != "cpu" or r1_device not in ("cpu", "cuda"):
        raise ValueError("system device policy requires CPU and r1_device cpu or cuda")
    return r1_device if (system["build"], int(system["reference"])) == ("r1", 2007) else device


def _validate_device_evidence(systems, policy, states, identities, scope, parity, device, r1_device):
    expected = {f"{s['build']}/{s['reference']}": _system_device(s, device, r1_device) for s in systems}
    if policy != expected or set(states) != set(expected) or set(identities) != set(expected):
        raise RuntimeError("system device policy or evidence is missing or changed")
    for key, wanted in expected.items():
        pair = identities[key]
        if (states[key].get("device") != wanted or not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(str(item.get("device", "")).split(":")[0] != wanted for item in pair)):
            raise RuntimeError("fitted state or identity device does not match system device")
    for row in list(scope) + list(parity):
        key = f"{row['build']}/{int(row['reference'])}"
        if key not in expected or row.get("device") != expected[key]:
            raise RuntimeError("replay or parity device does not match system device")


def _validate_sae_device_evidence(systems, policy, scope, sae_device):
    expected = {f"{s['build']}/{s['reference']}": sae_device for s in systems}
    if sae_device not in ("cpu", "cuda") or policy != expected:
        raise RuntimeError("SAE device policy is missing or changed")
    for row in scope:
        key = f"{row['build']}/{int(row['reference'])}"
        if key not in expected or row.get("sae_device") != expected[key]:
            raise RuntimeError("replay SAE device does not match system device")


def _bounded_scoped_replay(system, population, device, positional_embeddings, state_provenance,
                           progress=None, sae_device="cpu"):
    """Check original actual companions and candidate held companions before readiness."""
    import concept_subgroup_drift_pilot as pilot
    import r1_2007_replay as replay_helpers

    if progress is None:
        progress = {}
    progress.update({"stage": "bounded_start", "proof": [], "scope": [],
                     "actual_proof_rows": 0, "held_proof_rows": 0})
    build, reference = system["build"], int(system["reference"])
    config = json.loads(Path(system["files"]["config"]["path"]).read_text())
    year = np.asarray(population["year"])
    first = int(system["domain_origin"])
    roles = pilot.reference_year_roles(population, reference, seed=int(config["split_seed"]))
    context_mask = (year == reference) & (roles["row_role"] == "tabpfn_context") & ~roles["is_entrant"]
    context_mask |= (year >= first) & (year < reference) & ~population["post_death"]
    context_rows = np.flatnonzero(context_mask)
    columns = pilot._feature_columns(population["X"].shape[1])
    context = pilot._prediction_frame(population, context_rows, columns)
    context["y"] = population["y"][context_rows]
    fit_options = {} if config.get("context_window", "reference_only") == "reference_only" else {"domain_origin": first}
    progress["stage"] = "native_fit_started"
    predictor = _fit_frozen_predictor_with_state(
        context, columns, system, device, int(config["split_seed"]), positional_embeddings, **fit_options)
    progress["stage"] = "native_fit_complete"
    with np.load(system["files"]["actual"]["path"], allow_pickle=False) as actual, \
            np.load(system["files"]["held"]["path"], allow_pickle=False) as held, \
            np.load(system["files"]["predictor_archive"]["path"], allow_pickle=False) as archived:
        archive_rows = archived["row_index"].copy()
        actual_batch_size = int(system.get("saved_batch_rows", 4096))
        progress["stage"] = "actual_selection_started"
        actual_batch = replay_helpers.select_query_batch(
            population, archive_rows, reference, batch_size=actual_batch_size)
        actual_probes = replay_helpers.select_probe_rows(
            population, actual_batch, actual, reference, range(reference, 2015))
        progress["stage"] = "actual_selection_complete"
        actual_population_rows = actual_batch
        actual_frame = pilot._prediction_frame(population, actual_population_rows, columns)
        progress["stage"] = "actual_capture_started"
        actual_capture = pilot.capture_frozen_predictions(predictor, actual_frame, device)
        progress["stage"] = "actual_capture_complete"
        actual_probabilities = np.asarray(_identity_value(actual_capture["probabilities"]), dtype=np.float32)
        actual_embeddings = np.asarray(_identity_value(actual_capture["embeddings"]), dtype=np.float32)
        population_lookup = {(str(patient), int(yr)): i for i, (patient, yr)
                             in enumerate(zip(population["patient_id"], year))}
        archived_position = {int(row): i for i, row in enumerate(archive_rows)}
        actual_keys = _row_keys(actual, "actual")
        actual_proofs = []
        actual_scope = []
        actual_batch_hash = _row_order_sha256(population["patient_id"][actual_batch], year[actual_batch])
        settings = {"model_type": "single_fast", "device": device, "sae_device": sae_device,
                    "precision": {"dtype": "fp32", "amp": "off", "tf32": False},
                    "actual_batch_rows": actual_batch_size, "held_batch_rows": 64,
                    "held_precheck_rows": 32, "query_schedule": "archive-companion-order"}
        settings_sha256 = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        progress["stage"] = "actual_proof_checks_started"
        for probe_year, selected in actual_probes.items():
            table_row = selected["actual_table_row"]
            population_row = selected["population_row"]
            batch_position = selected["batch_position"]
            archive_position = archived_position[population_row]
            actual_probability = actual_probabilities[batch_position, 1]
            archived_probability = actual["probability"][table_row]
            probability_error = _finite_max_abs_error(actual_probability, archived_probability)
            p_ok = bool(probability_error is not None and np.isclose(
                actual_probability, archived_probability, rtol=1e-4, atol=1e-5))
            actual_embedding = actual_embeddings[batch_position]
            archived_embedding = archived["embeddings"][archive_position]
            embedding_error = _finite_max_abs_error(actual_embedding, archived_embedding)
            e_ok = bool(embedding_error is not None and np.allclose(
                actual_embedding, archived_embedding, rtol=1e-4, atol=1e-5))
            scope = {"build": build, "reference": reference, "year": int(probe_year), "arm": "actual",
                     "rows_sha256": actual_batch_hash, "rows": len(actual_batch),
                     "schedule_id": f"original-actual-{actual_batch_size}-ordered-v1",
                     "source_sha256": state_provenance["source_sha256"],
                     "state_source_sha256": state_provenance["state_source_sha256"],
                     "input_sha256": state_provenance["input_sha256"],
                     "positional_sha256": state_provenance["positional_state"]["sha256"],
                     "device": device, "sae_device": sae_device, "settings_sha256": settings_sha256,
                     "precision": "fp32", "schedule_certainty": "exact"}
            actual_scope.append(scope)
            proof = {**scope, "probability_within_tolerance": p_ok,
                     "embedding_within_tolerance": e_ok,
                     "saved_code_serialization_match": False,
                     "probability_max_abs_error": probability_error,
                     "embedding_max_abs_error": embedding_error}
            actual_proofs.append(proof)
            progress["proof"].append(proof)
            progress["scope"].append(scope)
            progress["actual_proof_rows"] = len(actual_proofs)
            progress["stage"] = "actual_proof_row_recorded"
        progress["stage"] = "held_selection_started"
        held_selection = replay_helpers.select_held_batch(held, actual, reference,
                                                          batch_size=64, precheck_size=32)
        progress["stage"] = "held_selection_complete"
        precheck_actual_rows = held_selection["precheck_rows"]
        precheck_population_rows = np.asarray([population_lookup[actual_keys[int(row)]]
                                               for row in precheck_actual_rows], dtype=np.int64)
        precheck_frame = pilot._prediction_frame(population, precheck_population_rows, columns)
        progress["stage"] = "held_precheck_capture_started"
        precheck_capture = pilot.capture_frozen_predictions(predictor, precheck_frame, device)
        precheck_probabilities = np.asarray(_identity_value(precheck_capture["probabilities"]), dtype=np.float32)[:, 1]
        precheck_expected = np.asarray(actual["probability"], dtype=np.float32)[precheck_actual_rows]
        precheck_pass = bool(np.isfinite(precheck_probabilities).all()
                             and np.isfinite(precheck_expected).all()
                             and np.allclose(precheck_probabilities, precheck_expected, rtol=1e-4, atol=1e-5))
        progress["stage"] = "held_precheck_complete"
        held_actual_rows = held_selection["held_actual_rows"]
        held_population_rows = np.asarray([population_lookup[actual_keys[int(row)]] for row in held_actual_rows],
                                          dtype=np.int64)
        held_frame = pilot._prediction_frame(population, held_population_rows, columns)
        held_frame["year"] = reference
        progress["stage"] = "held_capture_started"
        held_capture = pilot.capture_frozen_predictions(predictor, held_frame, device)
        held_probabilities = np.asarray(_identity_value(held_capture["probabilities"]), dtype=np.float32)
        held_embeddings = np.asarray(_identity_value(held_capture["embeddings"]), dtype=np.float32)
        held_probes = replay_helpers.select_held_probe_rows(held, held_selection,
                                                            range(reference + 1, 2015))
        progress["stage"] = "held_probe_selection_complete"
        run = Path(system["files"]["config"]["path"]).parent.parent
        frozen = _FrozenUnits(run / "unit-cache" / config["config_hash"] / "units", config)
        if sae_device == "cuda":
            pilot.configure_predictor_precision(sae_device)
        dictionaries = pilot.load_reference_dictionaries(frozen, reference, "cpu")
        progress["stage"] = "held_code_encoding_started"
        held_codes = {seed: pilot.encode_with_reference_sae(dictionary, held_embeddings, sae_device)
                      for seed, dictionary in dictionaries.items()}
        progress["stage"] = "held_code_encoding_complete"
        held_batch_hash = _row_order_sha256(held["patient_id"][held_selection["held_rows"]],
                                            held["year"][held_selection["held_rows"]])
        held_scope, held_proofs = [], []
        for probe_year, selected in held_probes.items():
            archive_row, batch_position = selected["archive_row"], selected["batch_position"]
            held_probability = held_probabilities[batch_position, 1]
            archived_held_probability = held["probability_swapped"][archive_row]
            probability_error = _finite_max_abs_error(held_probability, archived_held_probability)
            p_ok = bool(probability_error is not None and np.isclose(
                held_probability, archived_held_probability, rtol=1e-4, atol=1e-5))
            code_ok = all(np.array_equal(held_codes[seed][held_selection["held_rows"]].astype(
                held[f"codes_{seed}"].dtype), held[f"codes_{seed}"][held_selection["held_rows"]])
                          for seed in held_codes)
            # A separate singleton held schedule checks fresh raw coordinates where the source archive has none.
            single_row = held_actual_rows[batch_position]
            single_population = population_lookup[actual_keys[int(single_row)]]
            single_frame = pilot._prediction_frame(population, [single_population], columns)
            single_frame["year"] = reference
            progress["stage"] = "held_singleton_capture_started"
            singleton = pilot.capture_frozen_predictions(predictor, single_frame, device)
            single_embedding = np.asarray(_identity_value(singleton["embeddings"]), dtype=np.float32)[0]
            singleton_finite = bool(np.isfinite(single_embedding).all()
                                    and np.isfinite(held_embeddings[batch_position]).all())
            embedding_ok = bool(singleton_finite and np.allclose(
                single_embedding, held_embeddings[batch_position], rtol=1e-4, atol=1e-5))
            scope = {"build": build, "reference": reference, "year": int(probe_year), "arm": "held",
                     "rows_sha256": held_batch_hash, "rows": 64,
                     "schedule_id": "candidate-held-64-with-actual-32-precheck-v1",
                     "source_sha256": state_provenance["source_sha256"],
                     "state_source_sha256": state_provenance["state_source_sha256"],
                     "input_sha256": state_provenance["input_sha256"],
                     "positional_sha256": state_provenance["positional_state"]["sha256"],
                     "device": device, "sae_device": sae_device, "settings_sha256": settings_sha256,
                     "precision": "fp32", "schedule_certainty": "candidate"}
            held_scope.append(scope)
            proof = {**scope, "probability_within_tolerance": p_ok,
                     "embedding_within_tolerance": embedding_ok,
                     "saved_code_serialization_match": code_ok,
                     "probability_max_abs_error": probability_error}
            held_proofs.append(proof)
            progress["proof"].append(proof)
            progress["scope"].append(scope)
            progress["held_proof_rows"] = len(held_proofs)
            progress["stage"] = "held_proof_row_recorded"
    proof = actual_proofs + held_proofs
    scope = actual_scope + held_scope
    progress["stage"] = "final_scope_validation_started"
    validation = validate_scoped_replay_evidence(proof, scope)
    progress["stage"] = "final_scope_validation_complete"
    comparable = _comparable_predictor_identity(pilot, predictor["classifier"], system["context_rows_identity"],
                                                f"{system['build']}/{system['reference']}")
    validation.update({"scope": scope, "proof": proof,
                       "actual_batch_rows": actual_batch_size,
                       "held_batch_rows": 64, "held_precheck_rows": 32,
                       "held_actual_precheck_pass": precheck_pass,
                       "comparable_predictor_identity": comparable})
    if not precheck_pass:
        raise ValueError("held candidate actual precheck failed")
    progress["stage"] = "bounded_complete"
    return validation


def _check_execution_paths(device="cpu", ledger=None, output=None, systems=None, r1_device="cpu", sae_device="cpu"):
    """Audit frozen inputs and run bounded outcome-blind parity/reconstruction checks."""
    import gc
    import subprocess
    import time
    from datetime import datetime, timezone
    import concept_subgroup_drift_pilot as pilot

    root = Path(".")
    destination = (Path(output) if output is not None else
                   root / "stats/concept_agreement_study/extensions/temporal_input_attribution/generated_state/primary_checks")
    ledger = (Path(ledger) if ledger is not None
              else root / "stats/concept_agreement_study/extensions/temporal_input_attribution/checks/coverage-ledger.csv")
    if (destination / "provenance.json").exists():
        raise FileExistsError("check output already exists; choose a separate output directory")
    _system_device({"build": "r1", "reference": 2007}, device, r1_device)
    if sae_device not in ("cpu", "cuda"):
        raise ValueError("SAE device must be cpu or cuda")
    device = pilot.resolve_device(device)
    r1_device = pilot.resolve_device(r1_device)
    sae_device = pilot.resolve_device(sae_device)
    started, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    audit = audit_frozen_inputs(ledger, root)
    destination.mkdir(parents=True, exist_ok=True)
    population = pilot.load_pilot_population(root / "stats/temporal_robustness/_population/prepared.pkl",
                                             "c8804591c93087241b9533dce77dd54b2f1e77b23a36400527459cfa8f99add7")
    selected = {(str(build), int(year)) for build, year in systems} if systems is not None else None
    found = {(s["build"], s["reference"]) for s in audit["systems"]}
    if selected is not None and not selected <= found:
        raise ValueError("requested systems absent from saved index")
    parity, replay, replay_scope, timing, errors = [], [], [], [], []
    generated_state_provenance = {}
    comparable_predictor_identities = {}
    system_devices = {f"{s['build']}/{s['reference']}": _system_device(s, device, r1_device)
                      for s in audit["systems"] if selected is None or (s["build"], s["reference"]) in selected}
    system_sae_devices = {key: sae_device for key in system_devices}
    current_source_hash = inventory_inputs({"source": Path(__file__)}, {})["files"]["source"]["sha256"]
    current_pilot_source = audit["common"]["files"].get("pilot_source", {})
    current_pilot_source_hash = (current_pilot_source.get("sha256")
                                 if current_pilot_source.get("status") == "available" else None)
    for system in audit["systems"]:
        key = (system["build"], system["reference"])
        if selected is not None and key not in selected:
            continue
        system_device = system_devices[f"{key[0]}/{key[1]}"]
        bounded_progress = {}
        bounded_rows_extended = False
        try:
            if not system["current_checkpoint_matches"]:
                raise ValueError("current checkpoint differs from historical checkpoint")
            state_record = _generated_state_for_system(audit, system, population, system_device,
                                                       current_source_hash, root=root)
            generated_state_provenance[f"{key[0]}/{key[1]}"] = state_record["provenance"]
            result = check_system_parity(system, population, system_device, state_record["positional_tensor"])
            parity.extend({**row, "device": system_device} for row in result["parity"])
            timing.append({"build": key[0], "reference": key[1], "context_fit_seconds": result["fit_seconds"],
                           "peak_gpu_bytes": result["peak_gpu_bytes"],
                           "learned_parameter_sha256": result["learned_parameter_sha256"],
                           "query_check_seconds": result["query_check_seconds"]})
            gc.collect()
            if system_device == "cuda":
                torch.cuda.empty_cache()
            scoped = _bounded_scoped_replay(system, population, system_device,
                                            state_record["positional_tensor"], state_record["provenance"],
                                            progress=bounded_progress, sae_device=sae_device)
            replay.extend(scoped["proof"])
            replay_scope.extend(scoped["scope"])
            bounded_rows_extended = True
            timing[-1]["held_actual_precheck_pass"] = scoped["held_actual_precheck_pass"]
            timing[-1]["actual_companion_rows"] = scoped["actual_batch_rows"]
            timing[-1]["held_companion_rows"] = scoped["held_batch_rows"]
            timing[-1]["held_precheck_rows"] = scoped["held_precheck_rows"]
            comparable_predictor_identities[f"{key[0]}/{key[1]}"] = [
                result["comparable_predictor_identity"], scoped["comparable_predictor_identity"]]
        except Exception as error:
            if not bounded_rows_extended:
                replay.extend(bounded_progress.get("proof", []))
                replay_scope.extend(bounded_progress.get("scope", []))
            errors.append({"build": key[0], "reference": key[1], **sanitize_public_failure(error),
                           "bounded_stage": bounded_progress.get("stage", "before_bounded_replay"),
                           "actual_proof_rows": int(bounded_progress.get("actual_proof_rows", 0)),
                           "held_proof_rows": int(bounded_progress.get("held_proof_rows", 0))})
        finally:
            gc.collect()
            if system_device == "cuda" or sae_device == "cuda":
                torch.cuda.empty_cache()
    for system in audit["systems"]:
        hashes = system["saved_source_hashes"] or {}
        system["saved_source_hashes"] = hashes if hashes else "unknown"
        system["current_pilot_source_matches"] = (hashes.get("concept_subgroup_drift_pilot.py")
                                                   == current_pilot_source_hash
                                                   if isinstance(hashes, dict) and "concept_subgroup_drift_pilot.py" in hashes
                                                   and current_pilot_source_hash is not None
                                                   else "unknown")
        packages = (system["saved_environment"] or {}).get("packages", {})
        system["historical_tabpfn_package"] = packages.get("tabpfn") or "unknown"
    expected = {(system["build"], system["reference"], year, arm)
                for system in audit["systems"] for year in range(system["reference"], 2015)
                for arm in ("actual", "held")}
    seen_parity = {(row["build"], row["reference"], row["year"], row["input"]) for row in parity}
    complete_parity = (len(seen_parity) == len(parity) and seen_parity == expected
                       and all(row["parity"] == "pass" for row in parity)
                       and all(row.get("reference_actual_held_control") == "pass"
                               for row in parity if row["year"] == row["reference"] and row["input"] == "held"))
    provenance_complete = all(system["current_checkpoint_matches"] for system in audit["systems"])
    try:
        scoped_validation = validate_scoped_replay_evidence(replay, replay_scope)
    except ValueError:
        scoped_validation = None
    all_systems_selected = selected is None
    generated_system_keys = set(generated_state_provenance)
    identity_system_keys = set(comparable_predictor_identities)
    comparable_identities_complete = (identity_system_keys == generated_system_keys
        and all(isinstance(pair, (list, tuple)) and len(pair) == 2
                and comparable_predictor_identities_match(pair[0], pair[1])
                for pair in comparable_predictor_identities.values()))
    try:
        selected_records = [s for s in audit["systems"] if selected is None or (s["build"], s["reference"]) in selected]
        _validate_device_evidence(selected_records, system_devices, generated_state_provenance,
                                  comparable_predictor_identities, replay_scope, parity, device, r1_device)
        _validate_sae_device_evidence(selected_records, system_sae_devices, replay_scope, sae_device)
        devices_complete = True
    except RuntimeError:
        devices_complete = False
    scoring_ready = bool(scoped_validation and complete_parity and provenance_complete
                         and not errors and all_systems_selected
                         and len(generated_state_provenance) == len(audit["systems"])
                         and comparable_identities_complete and devices_complete)
    audit["checks"] = {"device": device, "r1_device": r1_device, "system_devices": system_devices,
                       "sae_device": sae_device, "system_sae_devices": system_sae_devices,
                       "selected_systems": sorted(selected) if selected is not None else "all",
                       "systems_attempted": len(found if selected is None else selected), "errors": errors,
                       "parity_rows": len(parity), "parity_failures": sum(row["parity"] != "pass" for row in parity),
                       "complete_parity": complete_parity,
                       "comparable_predictor_identities_match": comparable_identities_complete,
                       "system_devices_match": devices_complete,
                       "conditional_numerical_reconstruction": ("pass" if scoring_ready else
                           "unavailable: complete generated-state scope and primary proof are not present"),
                       "historical_environment": "unknown",
                       "historical_exact_replay": "unavailable: historical dependency identity and exact producer schedule are unknown",
                       "historical_provenance_complete": False, "current_source_and_checkpoint_match": provenance_complete,
                       "ready_for_scoring": scoring_ready,
                       "source_sha256": inventory_inputs({"source": Path(__file__)}, {})["files"]["source"]["sha256"],
                       "reason": ("bounded generated-state actual/held/code proof passed" if scoring_ready else
                                  "generated-state evidence, complete primary scope, parity and held/code proof are required before readiness")}
    audit["generated_state_provenance"] = generated_state_provenance
    audit["comparable_predictor_identities"] = comparable_predictor_identities
    audit["replay_scope"] = replay_scope
    (destination / "provenance.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")

    def write_table(name, rows):
        if not rows:
            (destination / name).write_text("")
            return
        fields = sorted({key for row in rows for key in row})
        with (destination / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                                 for key, value in row.items()})
    write_table("coverage.csv", audit["coverage"])
    write_table("parity.csv", parity)
    write_table("historical-replay.csv", replay)
    (destination / "scoped-replay.json").write_text(json.dumps(replay, indent=2, allow_nan=False) + "\n")
    (destination / "timing.json").write_text(json.dumps({"systems": timing, "errors": errors,
                                                         "scope": "bounded checks, not a full-run projection"}, indent=2) + "\n")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    git_changes = subprocess.run(["git", "status", "--porcelain", "--", "temporal_input_attribution.py",
                                  "tests/test_temporal_input_attribution.py"], capture_output=True, text=True, check=True).stdout.splitlines()
    (destination / "run-record.json").write_text(json.dumps({"what": "bounded temporal input checks",
        "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(), "device": device,
        "r1_device": r1_device, "system_devices": system_devices,
        "sae_device": sae_device, "system_sae_devices": system_sae_devices,
        "commit": revision, "uncommitted_code_files": git_changes,
        "source_sha256": inventory_inputs({"source": Path(__file__)}, {})["files"]["source"]["sha256"],
        "elapsed_seconds": time.monotonic() - clock,
        "outputs": ["provenance.json", "coverage.csv", "parity.csv", "historical-replay.csv",
                    "scoped-replay.json", "timing.json"]}, indent=2) + "\n")
    return audit["checks"]


def check_execution_paths(device="cpu", ledger=None, output=None, systems=None, r1_device="cpu", sae_device="cpu"):
    """Run bounded checks and retain provenance if a new check directory fails."""
    import subprocess
    import time
    from datetime import datetime, timezone

    destination = (Path(output) if output is not None
                   else Path("stats/concept_agreement_study/extensions/temporal_input_attribution/generated_state/primary_checks"))
    writable = not destination.exists() or (destination.is_dir() and all(
        item.name == "coverage-ledger.csv" or (item.name == "run" and item.is_dir()) for item in destination.iterdir()))
    started, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    try:
        return _check_execution_paths(device, ledger, output, systems, r1_device, sae_device)
    except Exception as error:
        if writable:
            destination.mkdir(parents=True, exist_ok=True)
            revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
            changes = subprocess.run(["git", "status", "--porcelain", "--", "temporal_input_attribution.py",
                                      "tests/test_temporal_input_attribution.py"], capture_output=True, text=True, check=True).stdout.splitlines()
            record = {"status": "failed", "device": device, "r1_device": r1_device,
                      "sae_device": sae_device, "started_utc": started,
                      "ended_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - clock,
                      "commit": revision, "uncommitted_code_files": changes,
                      "source_sha256": inventory_inputs({"source": Path(__file__)}, {})["files"]["source"]["sha256"],
                      **sanitize_public_failure(error), "outputs": []}
            (destination / "run-record.json").write_text(json.dumps(record, indent=2) + "\n")
        raise


PRIMARY_COMPARISON = Path("stats/concept_agreement_study/extensions/temporal_input_attribution/primary")


def comparable_predictor_identities_match(left, right):
    """Compare explicit fitted inputs across devices while retaining opaque identities."""
    required = ("opaque_identity", "device", "system_identity", "clinical_input_identity", "context_rows_identity",
                "checkpoint_sha256", "learned_parameter_identity", "configuration_identity",
                "clinical_transform_identity", "fitted_fields_identity", "precision", "positional_state", "execution_state")
    if any(not isinstance(item, dict) or any(key not in item for key in required)
           for item in (left, right)):
        return False
    if (not left["opaque_identity"] or not right["opaque_identity"]
            or not left["device"] or not right["device"]):
        return False
    identity_fields = ("system_identity", "clinical_input_identity", "context_rows_identity", "checkpoint_sha256",
                       "learned_parameter_identity", "configuration_identity", "clinical_transform_identity")
    if any(not left[key] or not right[key] for key in identity_fields):
        return False
    for item in (left, right):
        positional = item["positional_state"]
        execution = item["execution_state"]
        if (not isinstance(positional, dict) or not all(positional.get(key) for key in ("sha256", "shape", "dtype"))
                or not isinstance(execution, dict) or not execution.get("eval")
                or execution.get("depth") is None or execution.get("dropout") is None):
            return False
    if left["precision"] != {"dtype": "fp32", "amp": "off", "tf32": False}:
        return False
    if right["precision"] != {"dtype": "fp32", "amp": "off", "tf32": False}:
        return False
    comparable = tuple(key for key in required if key not in ("opaque_identity", "device"))
    return all(left[key] == right[key] for key in comparable)


def class_one_probabilities(probabilities, expected_rows):
    """Read class-one probabilities from the archived vector or binary class matrix layout."""
    values = np.asarray(probabilities)
    if values.ndim == 1 and values.shape == (expected_rows,):
        result = values
    elif values.ndim == 2 and values.shape == (expected_rows, 2):
        result = values[:, 1]
    else:
        raise ValueError("unsupported probability layout")
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("invalid class-one probabilities")
    return result


_REPLAY_SCOPE_FIELDS = ("build", "reference", "year", "arm", "rows_sha256", "rows",
                        "schedule_id", "source_sha256", "state_source_sha256", "input_sha256", "positional_sha256",
                        "device", "settings_sha256", "precision", "schedule_certainty")


def validate_scoped_replay_evidence(evidence, expected_scope):
    """Require unique actual and genuine held proof rows bound to current state and inputs."""
    if not expected_scope:
        raise ValueError("replay evidence must be complete and unique for a nonempty scope")
    expected_by_key = {}
    for scope in expected_scope:
        if any(field not in scope for field in _REPLAY_SCOPE_FIELDS):
            raise ValueError("expected replay scope is incomplete")
        key = (scope["build"], int(scope["reference"]), int(scope["year"]), scope["arm"])
        if key in expected_by_key:
            raise ValueError("expected replay scope is not complete and unique")
        expected_by_key[key] = scope
    for build, reference in {(key[0], key[1]) for key in expected_by_key}:
        keys = [key for key in expected_by_key if key[:2] == (build, reference)]
        if (not any(key[3] == "actual" for key in keys)
                or not any(key[3] == "held" and key[2] > reference for key in keys)):
            raise ValueError("replay scope requires actual and genuine later-year held evidence")
    observed = {}
    for proof in evidence:
        if any(field not in proof for field in (*_REPLAY_SCOPE_FIELDS, "probability_within_tolerance")):
            raise ValueError("replay evidence scope is incomplete")
        arm, year, reference = proof["arm"], int(proof["year"]), int(proof["reference"])
        if arm == "held" and year <= reference:
            raise ValueError("held evidence must be a genuine later-year held row")
        if arm not in ("actual", "held"):
            raise ValueError("replay evidence arm is unsupported")
        key = (proof["build"], reference, year, arm)
        if key in observed:
            raise ValueError("replay evidence must be complete and unique")
        observed[key] = proof
        if proof["schedule_certainty"] not in ("exact", "candidate"):
            raise ValueError("replay evidence schedule certainty is invalid")
    if set(observed) != set(expected_by_key):
        raise ValueError("replay evidence must be complete and unique for its declared scope")
    for key, proof in observed.items():
        expected = expected_by_key[key]
        if any(proof.get(field) != expected.get(field) for field in _REPLAY_SCOPE_FIELDS):
            raise ValueError("replay evidence scope does not match current inputs and settings")
        if "sae_device" in expected and proof.get("sae_device") != expected["sae_device"]:
            raise ValueError("replay SAE device does not match current settings")
        if proof.get("probability_within_tolerance") is not True:
            raise ValueError("replay probability check failed")
        if proof["arm"] == "actual" and proof.get("embedding_within_tolerance") is not True:
            raise ValueError("actual replay embedding check failed")
        if proof["arm"] == "held" and proof.get("embedding_within_tolerance") is not True:
            raise ValueError("held replay embedding check failed")
        if proof["arm"] == "held" and proof.get("saved_code_serialization_match") is not True:
            raise ValueError("held replay saved-code serialization check failed")
    return {"conditional_numerical_reconstruction": "pass", "historical_environment": "unknown",
            "historical_exact_replay": "unavailable: historical dependency identity is unknown",
            "schedule_certainty": sorted({proof["schedule_certainty"] for proof in observed.values()})}


def sanitize_public_failure(error):
    """Keep private exception text out of public readiness records."""
    return {"error_type": type(error).__name__,
            "message": "bounded check failed; private exception details are not persisted"}


_DIAGNOSTIC_TRUSTED_HASHES = {
    "completion_sha256": "2f324b1f02623c21a6e7194b3d203b8a1e6aad29ab2ba8795fc21f3bc23c24c5",
    "report_sha256": "43e60b2c4837473906326127f8892105041ebfd677331bca2090bfe33ee43f39",
    "independent_sha256": "c906f663c83e9b331b5a6548b6a135e4990710003e3a28ff3264a9c0c3d05767",
    "module_sha256": "0aafbfb91f78e54b90c9b44d7ed760b3ec14bc62a98e3794b0469b2a403752b3",
    "gpu_native_sha256": "2607ac5b21a4683562892d46ebf57befda314d2210805d0441371f1b57744fd8",
    "current_pilot_source_sha256": "a44dbc87b66927c7517d8509469cfc948ca5a2d3b3a50d55b5a50915a2da6a12",
}


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy_tensor_sha256(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode())
    torch_dtype = {"float32": "torch.float32"}.get(str(array.dtype), str(array.dtype))
    digest.update(torch_dtype.encode())
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def load_r1_diagnostic_positional_state(diagnostic_directory, expected_input_files, trusted_hashes=None,
                                       independent_verification_path=None):
    """Validate the completed diagnostic chain before reading only its named GPU state array."""
    directory = Path(diagnostic_directory)
    trusted = dict(_DIAGNOSTIC_TRUSTED_HASHES if trusted_hashes is None else trusted_hashes)
    required_trust = ("completion_sha256", "report_sha256", "independent_sha256", "module_sha256",
                      "gpu_native_sha256", "current_pilot_source_sha256")
    if any(not trusted.get(key) for key in required_trust):
        raise ValueError("trusted diagnostic fingerprints are incomplete")
    paths = {name: directory / name for name in (
        "completion.json", "report.json", "independent-verification.public.json",
        "report.csv", "report.md", "gpu_native.npz")}
    if independent_verification_path is not None:
        paths["independent-verification.public.json"] = Path(independent_verification_path)
    completion_path, report_path, independent_path = (paths[name] for name in
        ("completion.json", "report.json", "independent-verification.public.json"))
    for key, path in (("completion_sha256", completion_path), ("report_sha256", report_path),
                      ("independent_sha256", independent_path), ("gpu_native_sha256", paths["gpu_native.npz"])):
        if not path.is_file() or _file_sha256(path) != trusted[key]:
            raise ValueError("diagnostic artifact fingerprint mismatch")
    completion = json.loads(completion_path.read_text())
    report = json.loads(report_path.read_text())
    independent = json.loads(independent_path.read_text())
    if (completion.get("status") != "completed" or report.get("status") != "completed"
            or report.get("schema") != "r1-2007-device-state-diagnostic-v1"
            or completion.get("module_sha256") != trusted["module_sha256"]
            or report.get("module_sha256") != trusted["module_sha256"]
            or report.get("source_sha256", {}).get("current") != trusted["current_pilot_source_sha256"]
            or report.get("input_fingerprints", {}).get("source_hashes", {}).get("current")
            != trusted["current_pilot_source_sha256"]):
        raise ValueError("diagnostic completion or source identity is invalid")
    if (completion.get("summary_outputs", {}).get("report.json") != trusted["report_sha256"]
            or completion.get("outputs", {}).get("gpu_native") != trusted["gpu_native_sha256"]
            or completion.get("summary_outputs", {}).get("report.csv") != _file_sha256(paths["report.csv"])
            or completion.get("summary_outputs", {}).get("report.md") != _file_sha256(paths["report.md"])):
        raise ValueError("diagnostic completion output chain is invalid")
    if (independent.get("status") != "independent_saved_output_verification_pass"
            or independent.get("completion") != completion
            or independent.get("summary_checksums") != "pass"
            or independent.get("targets_equal_original_archive") != "pass"
            or independent.get("all_report_per_year_errors_and_combined_tolerances") != "pass"):
        raise ValueError("independent diagnostic verification is incomplete")
    comparisons = independent.get("comparisons", {})
    for name in ("gpu_4096_vs_archive", "cpu_gpu_position_8_vs_archive"):
        result = comparisons.get(name, {})
        if (result.get("probes") != 8 or result.get("probability_pass") != 8
                or result.get("embedding_pass") != 8):
            raise ValueError("diagnostic actual-probe output verification is incomplete")
    fingerprints = report.get("input_fingerprints", {}).get("input_files", {})
    if (not expected_input_files or any(fingerprints.get(name) != value
                                        for name, value in expected_input_files.items())):
        raise ValueError("diagnostic source or input fingerprints are stale")
    if set(expected_input_files) != set(fingerprints):
        raise ValueError("diagnostic input fingerprint scope is incomplete")
    schedule = report.get("input_fingerprints", {}).get("schedule", {})
    expected_schedule = {"actual_batch_rows": 4096, "actual_probe_years": list(range(2007, 2015)),
                         "held_batch_rows": 64, "held_probe_years": list(range(2008, 2015)),
                         "precheck_rows": 32}
    if schedule != expected_schedule:
        raise ValueError("diagnostic actual-probe schedule is not the approved scope")
    gpu = report.get("arms", {}).get("gpu_native", {})
    cpu_position = report.get("arms", {}).get("cpu_gpu_position", {})
    state = gpu.get("after_state", {}).get("positional_state", {})
    injected_state = cpu_position.get("after_state", {}).get("positional_state", {})
    comparable_state_keys = ("mode", "cache_name", "shape", "dtype", "sha256", "registered_buffer")
    gpu_device_pair_matches = (gpu.get("device") == "cuda" and state.get("device") == "cuda:0")
    cpu_device_pair_matches = (cpu_position.get("device") == "cpu" and injected_state.get("device") == "cpu")
    if (not state or any(state.get(key) != injected_state.get(key) for key in comparable_state_keys)
            or cpu_position.get("before_state") != cpu_position.get("after_state")
            or gpu.get("before_state") != gpu.get("after_state")
            or not gpu_device_pair_matches or not cpu_device_pair_matches
            or state.get("dtype") != "torch.float32"
            or state.get("mode") != "subspace" or gpu.get("private_output_sha256") != trusted["gpu_native_sha256"]):
        raise ValueError("diagnostic generated-state observation is inconsistent")
    with np.load(paths["gpu_native.npz"], allow_pickle=False) as arrays:
        if "positional_tensor" not in arrays.files:
            raise ValueError("diagnostic positional tensor is missing")
        positional_tensor = arrays["positional_tensor"].copy()
    if (list(positional_tensor.shape) != state.get("shape") or positional_tensor.dtype != np.float32
            or _numpy_tensor_sha256(positional_tensor) != state.get("sha256")):
        raise ValueError("diagnostic positional tensor identity does not match its fitted observation")
    input_sha256 = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
    return {"positional_tensor": positional_tensor,
            "provenance": {"origin": "validated_r1_gpu_diagnostic", "build": "r1", "reference": 2007,
                           "device": "cuda:0", "source_sha256": trusted["current_pilot_source_sha256"],
                           "diagnostic_module_sha256": trusted["module_sha256"],
                           "historical_environment": "unknown", "historical_dependency_identity": "unknown",
                           "input_fingerprints": dict(fingerprints), "input_sha256": input_sha256,
                           "positional_state": {key: state[key] for key in ("shape", "dtype", "sha256")}}}


def verify_execution_checks(report, current, parity, replay, device, source_sha256, expected_scope=None,
                            r1_device="cpu", sae_device="cpu"):
    """Recheck proof rows and fingerprints rather than trusting a persisted ready flag."""
    status = report.get("checks", {})
    current_state = current.get("generated_state_provenance")
    reported_state = report.get("generated_state_provenance")
    if not isinstance(current_state, dict) or not current_state or reported_state != current_state:
        raise RuntimeError("generated state evidence is missing, stale, or changed")
    reported_identities = report.get("comparable_predictor_identities")
    identity_bindings = current.get("comparable_predictor_identity_bindings")
    if (not isinstance(reported_identities, dict) or not reported_identities
            or not isinstance(identity_bindings, dict)
            or set(reported_identities) != set(current_state) or set(identity_bindings) != set(current_state)):
        raise RuntimeError("comparable fitted predictor identities are missing or stale")
    for key, pair in reported_identities.items():
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not comparable_predictor_identities_match(pair[0], pair[1])):
            raise RuntimeError("explicit cross-device fitted predictor fields do not match")
        binding = identity_bindings[key]
        for identity in pair:
            if (not isinstance(binding, dict)
                    or any(identity.get(name) != binding.get(name) for name in
                           ("system_identity", "clinical_input_identity", "context_rows_identity", "checkpoint_sha256"))
                    or identity.get("positional_state") != binding.get("positional_state")):
                raise RuntimeError("fitted predictor identity does not match the current system identity binding")
    if expected_scope is None or not expected_scope:
        raise RuntimeError("generated state replay scope must be rebuilt from current inputs")
    if r1_device != "cpu" or "system_devices" in status:
        _validate_device_evidence(current["systems"], status.get("system_devices"), current_state,
                                  reported_identities, expected_scope, parity, device, r1_device)
        if status.get("r1_device") != r1_device:
            raise RuntimeError("execution checks used a different r1 device")
    if sae_device != "cpu" or "system_sae_devices" in status:
        _validate_sae_device_evidence(current["systems"], status.get("system_sae_devices"), expected_scope, sae_device)
        if status.get("sae_device") != sae_device:
            raise RuntimeError("execution checks used a different SAE device")
    if (not status.get("ready_for_scoring") or status.get("conditional_numerical_reconstruction") != "pass"
            or status.get("historical_environment") != "unknown"
            or status.get("historical_exact_replay") == "pass" or status.get("errors")
            or status.get("selected_systems") != "all"):
        raise RuntimeError("execution checks do not establish conditional generated-state readiness")
    state_by_system = {}
    for item in current_state.values():
        if not isinstance(item, dict):
            raise RuntimeError("generated state evidence is invalid")
        key = (item.get("build"), int(item.get("reference", -1)))
        state = item.get("positional_state", {})
        if (key in state_by_system or item.get("origin") not in
                ("validated_r1_gpu_diagnostic", "native_cpu_reconstruction_candidate")
                or not isinstance(item.get("input_fingerprints"), dict) or not item.get("input_fingerprints")
                or not item.get("input_sha256") or not item.get("source_sha256") or not state.get("sha256")
                or not state.get("shape") or state.get("dtype") != "torch.float32"):
            raise RuntimeError("generated state evidence is incomplete or duplicated")
        state_by_system[key] = state["sha256"]
    for scope in expected_scope:
        key = (scope.get("build"), int(scope.get("reference", -1)))
        if state_by_system.get(key) != scope.get("positional_sha256"):
            raise RuntimeError("generated state digest does not match replay scope")
        item = next(value for value in current_state.values()
                    if (value.get("build"), int(value.get("reference", -1))) == key)
        if (item.get("input_sha256") != scope.get("input_sha256")
                or item.get("state_source_sha256", item.get("source_sha256"))
                != scope.get("state_source_sha256")):
            raise RuntimeError("generated state input or source fingerprint does not match replay scope")
    try:
        validate_scoped_replay_evidence(replay, expected_scope)
    except ValueError as error:
        raise RuntimeError(str(error)) from None
    if status.get("device") != device:
        raise RuntimeError("execution checks used a different device")
    if status.get("source_sha256") != source_sha256:
        raise RuntimeError("execution checks are stale: comparison source changed")
    old_systems = {(s["build"], s["reference"]): s for s in report["systems"]}
    new_systems = {(s["build"], s["reference"]): s for s in current["systems"]}
    if len(old_systems) != len(report["systems"]) or not set(new_systems) <= set(old_systems):
        raise RuntimeError("execution checks are stale: systems changed")
    for name, file in current["common"]["files"].items():
        if file["status"] != "available" or file["sha256"] != report["common"]["files"].get(name, {}).get("sha256"):
            raise RuntimeError(f"execution checks are stale: {name} changed")
    for key, system in new_systems.items():
        previous = old_systems[key]
        if not system["current_checkpoint_matches"]:
            raise RuntimeError(f"historical provenance unresolved for {key}")
        current_dependency = current.get("common", {}).get("provenance", {}).get("current_tabpfn_source_identity")
        recorded_dependency = report.get("common", {}).get("provenance", {}).get("current_tabpfn_source_identity")
        if (current_dependency != recorded_dependency
                or system.get("historical_tabpfn_source_identity") != previous.get("historical_tabpfn_source_identity")):
            raise RuntimeError(f"current dependency source identity changed for {key}")
        if any(value["status"] != "available" or value["sha256"] != previous["files"].get(name, {}).get("sha256")
               for name, value in system["files"].items()):
            raise RuntimeError(f"execution checks are stale for {key}")
    old_concepts = {row["concept"]: row["concept_file_sha256"] for row in report["coverage"]}
    if (len(old_concepts) != len(report["coverage"])
            or any(old_concepts.get(row["concept"]) != row["concept_file_sha256"] for row in current["coverage"])):
        raise RuntimeError("execution checks are stale: concept inputs changed")
    expected = {(build, reference, year, arm) for build, reference in new_systems
                for year in range(reference, 2015) for arm in ("actual", "held")}
    def row_key(row):
        return row["build"], int(row["reference"]), int(row["year"]), row["input"]
    parity = [row for row in parity if (row["build"], int(row["reference"])) in new_systems]
    replay = [row for row in replay if (row["build"], int(row["reference"])) in new_systems]
    if (len(parity) != len(expected) or {row_key(row) for row in parity} != expected
            or any(row["parity"] != "pass" for row in parity)
            or any(row.get("reference_actual_held_control") != "pass" for row in parity
                   if int(row["year"]) == int(row["reference"]) and row["input"] == "held")):
        raise RuntimeError("execution checks lack complete matched parity evidence")
    def true(value):
        return value is True or value == "True"
    return True


def _row_order_sha256(patients, years):
    payload = [[str(patient), int(year)] for patient, year in zip(patients, years)]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _rebuild_replay_scope(audit, states, source_sha256, device, r1_device="cpu", sae_device="cpu"):
    """Rebuild actual and later-held row scope from current archives and generated state."""
    import r1_2007_replay as replay_helpers

    scope = []
    for system in audit["systems"]:
        system_device = _system_device(system, device, r1_device)
        key = f"{system['build']}/{system['reference']}"
        provenance = states[key]["provenance"]
        settings = {"model_type": "single_fast", "device": system_device, "sae_device": sae_device,
                    "precision": {"dtype": "fp32", "amp": "off", "tf32": False},
                    "actual_batch_rows": int(system.get("saved_batch_rows", 4096)),
                    "held_batch_rows": 64, "held_precheck_rows": 32,
                    "query_schedule": "archive-companion-order"}
        settings_sha256 = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        with np.load(system["files"]["actual"]["path"], allow_pickle=False) as actual, \
                np.load(system["files"]["held"]["path"], allow_pickle=False) as held, \
                np.load(system["files"]["predictor_archive"]["path"], allow_pickle=False) as archived:
            population = audit["population"]
            actual_rows = replay_helpers.select_query_batch(
                population, archived["row_index"], int(system["reference"]),
                batch_size=int(system.get("saved_batch_rows", 4096)))
            actual_probes = replay_helpers.select_probe_rows(
                population, actual_rows, actual, int(system["reference"]),
                range(int(system["reference"]), 2015))
            actual_hash = _row_order_sha256(population["patient_id"][actual_rows],
                                            population["year"][actual_rows])
            held_selection = replay_helpers.select_held_batch(held, actual, int(system["reference"]), 64, 32)
            held_probes = replay_helpers.select_held_probe_rows(
                held, held_selection, range(int(system["reference"]) + 1, 2015))
            held_rows = held_selection["held_rows"]
            held_hash = _row_order_sha256(held["patient_id"][held_rows], held["year"][held_rows])
            for year in actual_probes:
                scope.append({"build": system["build"], "reference": int(system["reference"]),
                              "year": int(year), "arm": "actual", "rows_sha256": actual_hash,
                              "rows": len(actual_rows),
                              "schedule_id": f"original-actual-{len(actual_rows)}-ordered-v1",
                              "source_sha256": source_sha256,
                              "state_source_sha256": provenance["state_source_sha256"],
                              "input_sha256": provenance["input_sha256"],
                              "positional_sha256": provenance["positional_state"]["sha256"],
                              "device": system_device, "sae_device": sae_device, "settings_sha256": settings_sha256,
                              "precision": "fp32", "schedule_certainty": "exact"})
            for year in held_probes:
                scope.append({"build": system["build"], "reference": int(system["reference"]),
                              "year": int(year), "arm": "held", "rows_sha256": held_hash,
                              "rows": len(held_rows),
                              "schedule_id": "candidate-held-64-with-actual-32-precheck-v1",
                              "source_sha256": source_sha256,
                              "state_source_sha256": provenance["state_source_sha256"],
                              "input_sha256": provenance["input_sha256"],
                              "positional_sha256": provenance["positional_state"]["sha256"],
                              "device": system_device, "sae_device": sae_device, "settings_sha256": settings_sha256,
                              "precision": "fp32", "schedule_certainty": "candidate"})
    return scope


def _comparison_inputs(variants, device, r1_device="cpu", checks_dir=None, sae_device="cpu"):
    import concept_subgroup_drift_pilot as pilot
    generated_checks = (Path(checks_dir) if checks_dir is not None else
                        Path("stats/concept_agreement_study/extensions/temporal_input_attribution/generated_state/primary_checks"))
    checks = generated_checks / "provenance.json"
    if not checks.is_file():
        raise RuntimeError("generated-state execution checks are unavailable; scientific scoring is blocked")
    readiness_provenance_sha256 = _file_sha256(checks)
    report = json.loads(checks.read_text())
    if variants != ("last_3",):
        raise ValueError("primary plan only permits the last_3 variant")
    if not report.get("checks", {}).get("ready_for_scoring"):
        raise RuntimeError("generated-state execution checks do not establish conditional readiness; scientific scoring is blocked")
    _system_device({"build": "r1", "reference": 2007}, device, r1_device)
    if sae_device not in ("cpu", "cuda"):
        raise ValueError("SAE device must be cpu or cuda")
    resolved = pilot.resolve_device(device)
    resolved_r1 = pilot.resolve_device(r1_device)
    resolved_sae = pilot.resolve_device(sae_device)
    ledger = Path("stats/concept_agreement_study/extensions/temporal_input_attribution/checks/coverage-ledger.csv")
    audit = audit_frozen_inputs(ledger)
    population = pilot.load_pilot_population("stats/temporal_robustness/_population/prepared.pkl",
                                             "c8804591c93087241b9533dce77dd54b2f1e77b23a36400527459cfa8f99add7")
    audit["population"] = population
    with (checks.parent / "parity.csv").open(newline="") as stream:
        parity = list(csv.DictReader(stream))
    replay_path = checks.parent / "scoped-replay.json"
    if not replay_path.is_file():
        raise RuntimeError("generated-state checks lack scoped replay proof")
    replay = json.loads(replay_path.read_text())
    source_hash = inventory_inputs({"source": Path(__file__)}, {})["files"]["source"]["sha256"]
    state_records = {}
    current_fingerprints = {}
    identity_bindings = {}
    for system in audit["systems"]:
        key = (system["build"], int(system["reference"]))
        state_key = f"{key[0]}/{key[1]}"
        state = _generated_state_for_system(audit, system, population,
                                            _system_device(system, resolved, resolved_r1), source_hash)
        expected_state = report.get("generated_state_provenance", {}).get(state_key)
        if not isinstance(expected_state, dict) or expected_state != state["provenance"]:
            raise RuntimeError("generated state evidence is stale for current inputs")
        state_records[state_key] = state
        current_fingerprints[state_key] = state["provenance"]["input_fingerprints"]
        _, _, _, _, _, _, bindings, _ = _current_scoring_context(
            pilot, population, system, state["provenance"].get("positional_state"))
        if bindings["context_rows_identity"] != system["context_rows_identity"]:
            raise RuntimeError("current system context rows do not match the audited context identity")
        identity_bindings[state_key] = bindings
    expected_scope = _rebuild_replay_scope(audit, state_records, source_hash, resolved, resolved_r1, resolved_sae)
    audit["generated_state_provenance"] = {key: value["provenance"] for key, value in state_records.items()}
    audit["comparable_predictor_identity_bindings"] = identity_bindings
    verify_execution_checks(report, audit, parity, replay, resolved, source_hash, expected_scope, resolved_r1, resolved_sae)
    if _file_sha256(checks) != readiness_provenance_sha256:
        raise RuntimeError("execution checks changed during verification")
    index = json.loads(Path("stats/concept_agreement_study/patient_table/index.json").read_text())
    systems = [system for system in audit["systems"] if set(system["variants"]) & set(variants)]
    keys = {(system["build"], system["reference"]) for system in systems}
    concepts = [concept for concept in index["concepts"] if (concept["build"], concept["reference"]) in keys
                and set(concept["variants"]) & set(variants)]
    patient_sets = []
    fingerprints = {}
    for system in systems:
        key = (system["build"], system["reference"])
        with np.load(system["files"]["actual"]["path"], allow_pickle=False) as stored:
            patient_sets.append(stored["patient_id"][stored["year"] <= 2014].astype(str))
        payload = {"system": system, "concepts": [row for row in audit["coverage"]
                                                  if (row["build"], row["reference"]) == key],
                   "source_sha256": source_hash, "common": audit["common"]["files"],
                   "system_device": _system_device(system, resolved, resolved_r1),
                   "sae_device": resolved_sae,
                   "readiness_provenance_sha256": readiness_provenance_sha256}
        fingerprints[key] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {"systems": systems, "concepts": concepts, "patient_sets": patient_sets,
            "fingerprints": fingerprints, "population": population,
            "generated_state_provenance": state_records,
            "expected_comparable_predictor_identities": report.get("comparable_predictor_identities"),
            "current_input_fingerprints": current_fingerprints, "source_sha256": source_hash,
            "readiness_provenance_sha256": readiness_provenance_sha256}


def _fit_frozen_predictor_with_state(context_frame, feature_names, system, device, seed,
                                     positional_embeddings, domain_origin=None):
    """Fit the frozen single-fast predictor after installing this system's state."""
    import hashlib
    import tabpfn
    from importlib import resources
    from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig
    import concept_subgroup_drift_pilot as pilot

    years = context_frame.year.to_numpy(dtype=np.int64)
    if len(context_frame) == 0 or set(context_frame.y.unique()) != {0, 1}:
        raise ValueError("Reference context requires both mortality classes")
    origin = int(years[0]) if domain_origin is None else int(domain_origin)
    if years.min() < origin or (domain_origin is None and len(np.unique(years)) != 1):
        raise ValueError("Context rows do not match the declared domain origin")
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
    classifier.pilot_precision_ = pilot.configure_predictor_precision(device)
    classifier.pilot_checkpoint_sha256_ = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    classifier.pilot_domain_policy_ = ("calendar_year_minus_reference_year" if domain_origin is None
                                       else "calendar_year_minus_first_context_year")
    if hasattr(classifier, "show_progress"):
        classifier.show_progress = False
    classifier.init_model_and_get_model_config()
    prepared = prepare_context_classifier(classifier, positional_embeddings, use_context_scaler=False)
    X = np.ascontiguousarray(context_frame[list(feature_names)].to_numpy(dtype=np.float32))
    y = np.ascontiguousarray(context_frame.y.to_numpy(dtype=np.float32))
    with pilot.predictor_sdpa_context(device):
        prepared.fit(X, y, additional_x={"dist_shift_domain": np.ascontiguousarray(years - origin, dtype=np.int64)})
    if len(prepared.ensemble_configurations_) != 1 or len(prepared.cached_models_) != 1:
        raise RuntimeError("Pilot requires exactly one frozen executed configuration/model")
    if list(prepared.classes_) != [0, 1]:
        raise RuntimeError("Death class mapping differs from explicit [0, 1]")
    return {"classifier": prepared, "context_identity": pilot.predictor_context_identity(prepared),
            "context_rows": pilot._identity_hash(list(zip(context_frame.patient_id, context_frame.year))),
            "reference_year": int(years.max()), "domain_origin": origin,
            "feature_names": tuple(feature_names), "checkpoint_sha256": classifier.pilot_checkpoint_sha256_,
            "settings": {"model_type": "single_fast", "fit_at_predict_time": False,
                         "sklearn_compatible_precision": False, "fp16_inference": False,
                         "precision": classifier.pilot_precision_, "domain_policy": classifier.pilot_domain_policy_,
                         "seed": int(seed)}}


def _score_system(inputs, system, device, sae_device="cpu"):
    """Score the four frozen arms for one already-verified system."""
    import gc
    from importlib import resources
    import concept_subgroup_drift_pilot as pilot
    import tabpfn
    from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig
    from tabpfn_model import infer_model_additional_x_info, make_dist_tensor

    reference, build = system["reference"], system["build"]
    state_key = f"{build}/{reference}"
    state_record = inputs.get("generated_state_provenance", {}).get(state_key)
    if not isinstance(state_record, dict):
        raise RuntimeError("generated state evidence is missing for this scoring system")
    positional_embeddings = state_record.get("positional_tensor")
    state_provenance = state_record.get("provenance", {})
    if (not torch.is_tensor(positional_embeddings) or positional_embeddings.ndim != 2
            or positional_embeddings.dtype != torch.float32 or not torch.isfinite(positional_embeddings).all().item()
            or positional_embeddings.device.type != device
            or state_provenance.get("build") != build or int(state_provenance.get("reference", -1)) != reference
            or state_provenance.get("device") != device
            or state_provenance.get("positional_state", {}).get("sha256")
            != _numpy_tensor_sha256(positional_embeddings.detach().cpu().numpy())):
        raise RuntimeError("generated state evidence is stale or invalid for this scoring system")
    expected_fingerprints = inputs.get("current_input_fingerprints", {}).get(state_key)
    expected_source = inputs.get("source_sha256")
    if (not state_provenance.get("source_sha256") or not state_provenance.get("input_sha256")
            or not isinstance(expected_fingerprints, dict) or not expected_fingerprints
            or state_provenance.get("input_fingerprints") != expected_fingerprints
            or state_provenance.get("source_sha256") != expected_source):
        raise RuntimeError("generated state evidence is not bound to current source and inputs")
    population = inputs["population"]
    expected_pair = inputs.get("expected_comparable_predictor_identities", {}).get(state_key)
    if (not isinstance(expected_pair, (list, tuple)) or len(expected_pair) != 2
            or not comparable_predictor_identities_match(expected_pair[0], expected_pair[1])
            or any(str(identity.get("device", "")).split(":")[0] != device for identity in expected_pair)):
        raise RuntimeError("expected fitted predictor identity proof is missing or incomplete")
    context, columns, context_rows, context_x, context_y, additional_x, current_binding, config = (
        _current_scoring_context(pilot, population, system, state_provenance.get("positional_state")))
    if (current_binding["context_rows_identity"] != system["context_rows_identity"]
            or any(identity.get("system_identity") != current_binding["system_identity"]
                   or identity.get("clinical_input_identity") != current_binding["clinical_input_identity"]
                   or identity.get("context_rows_identity") != current_binding["context_rows_identity"]
                   or identity.get("checkpoint_sha256") != current_binding["checkpoint_sha256"]
                   or identity.get("positional_state") != current_binding["positional_state"]
                   for identity in expected_pair)):
        raise RuntimeError("expected fitted predictor identity does not match current scoring inputs")
    years = np.asarray(population["year"])
    first = system["domain_origin"]
    population_lookup = {(str(patient), int(year)): i for i, (patient, year) in enumerate(zip(population["patient_id"], years))}
    with np.load(system["files"]["actual"]["path"], allow_pickle=False) as actual, \
            np.load(system["files"]["held"]["path"], allow_pickle=False) as held, \
            np.load(system["files"]["predictor_archive"]["path"], allow_pickle=False) as predictor_archive:
        patient_id, year = actual["patient_id"].copy(), actual["year"].copy()
        rows = np.array([population_lookup[key] for key in _row_keys(actual, "actual")])
        archive_rows = predictor_archive["row_index"]
        archive_position = np.searchsorted(archive_rows, rows)
        if (np.any(archive_position >= len(archive_rows))
                or not np.array_equal(archive_rows[archive_position], rows)):
            raise ValueError(f"predictor archive rows changed for {build}/{reference}")
        archived_actual_embedding = predictor_archive["embeddings"][archive_position].copy()
        archived_actual_probability = actual["probability"].copy()
        held_positions = held["positions"].copy()
        if not np.array_equal(held_positions, np.flatnonzero(year > reference)):
            raise ValueError(f"held archive coverage differs for {build}/{reference}")
        held_probability = held["probability_swapped"].copy()
        held_codes = {seed: held[f"codes_{seed}"].copy() for seed in (42, 135, 215)}
        features, labels = actual["features"].copy(), actual["label"].copy()
        feature_names = actual["feature_names"].copy()
        if (build, reference, device) == ("r1", 2007, "cuda"):
            import r1_2007_replay as replay_helpers
            original_rows = replay_helpers.select_query_batch(
                population, archive_rows, reference, batch_size=int(system["saved_batch_rows"]))
            selection = replay_helpers.select_held_batch(held, actual, reference, 64, 32)
            precheck_rows = selection["precheck_rows"]
    history_device = device
    fit_options = {} if config.get("context_window", "reference_only") == "reference_only" else {"domain_origin": first}
    history = _fit_frozen_predictor_with_state(
        context, columns, system, history_device, int(config["split_seed"]), positional_embeddings, **fit_options)
    if (history["checkpoint_sha256"] != system["saved_checkpoint_sha256"]
            or history["context_rows"] != system["context_rows_identity"]):
        raise RuntimeError(f"historical predictor identity mismatch for {build}/{reference}")
    history_identity = _comparable_predictor_identity(
        pilot, history["classifier"], history["context_rows"], state_key)
    if (not comparable_predictor_identities_match(history_identity, expected_pair[1])
            or not comparable_predictor_identities_match(history_identity, expected_pair[0])
            or str(history_identity.get("device", "")).split(":")[0] != device):
        raise RuntimeError(f"fresh historical fitted predictor identity mismatch for {build}/{reference}")
    if (build, reference, device) == ("r1", 2007, "cuda"):
        original_frame = pilot._prediction_frame(population, original_rows, columns)
        pilot.capture_frozen_predictions(history, original_frame, history_device)
        precheck_frame = pilot._prediction_frame(population, rows[precheck_rows], columns)
        precheck = pilot.capture_frozen_predictions(history, precheck_frame, history_device)
        precheck_probability = _identity_value(precheck["probabilities"])[:, 1]
        if not np.allclose(precheck_probability, archived_actual_probability[precheck_rows], rtol=1e-4, atol=1e-5):
            raise RuntimeError("historical held actual precheck failed")
        del precheck
    held_replay_embeddings = []
    held_replay_probabilities = []
    # The held reconstruction uses the separately qualified candidate 64-row schedule.
    frame = pilot._prediction_frame(population, rows[held_positions], columns)
    frame["year"] = reference
    for start in range(0, len(frame), 64):
        captured = pilot.capture_frozen_predictions(history, frame.iloc[start:start + 64], history_device)
        held_replay_probabilities.append(captured["probabilities"][:, 1].detach().cpu().numpy())
        held_replay_embeddings.append(captured["embeddings"].detach().cpu().numpy())
    held_replay_probabilities = np.concatenate(held_replay_probabilities)
    held_replay_embeddings = np.concatenate(held_replay_embeddings)
    held_probability_error = float(np.max(np.abs(held_replay_probabilities - held_probability)))
    if not np.allclose(held_replay_probabilities, held_probability, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"historical held probabilities do not replay for {build}/{reference}; max absolute error {held_probability_error}")
    # Both arms use one ordered candidate 64-row schedule for matched fresh outputs.
    historical_probabilities, historical_embeddings = {}, {}
    for arm in ("actual", "held"):
        frame = pilot._prediction_frame(population, rows, columns)
        if arm == "held":
            frame["year"] = reference
        probabilities, coordinates = [], []
        for start in range(0, len(frame), 64):
            captured = pilot.capture_frozen_predictions(history, frame.iloc[start:start + 64], history_device)
            probabilities.append(captured["probabilities"][:, 1].detach().cpu().numpy())
            coordinates.append(captured["embeddings"].detach().cpu().numpy())
        historical_probabilities[arm] = np.concatenate(probabilities)
        historical_embeddings[arm] = np.concatenate(coordinates)
    actual_probability_error = float(np.max(np.abs(historical_probabilities["actual"] - archived_actual_probability)))
    actual_embedding_error = float(np.max(np.abs(historical_embeddings["actual"] - archived_actual_embedding)))
    if (not np.allclose(historical_probabilities["actual"], archived_actual_probability, rtol=1e-4, atol=1e-5)
            or not np.allclose(historical_embeddings["actual"], archived_actual_embedding, rtol=1e-4, atol=1e-5)):
        raise RuntimeError(f"historical actual fresh schedule does not match the archive for {build}/{reference}; "
                           f"max probability error {actual_probability_error}, max embedding error {actual_embedding_error}")
    fresh_held_probability_error = float(np.max(np.abs(historical_probabilities["held"][held_positions] - held_replay_probabilities)))
    fresh_held_embedding_error = float(np.max(np.abs(historical_embeddings["held"][held_positions] - held_replay_embeddings)))
    if (not np.allclose(historical_probabilities["held"][held_positions], held_replay_probabilities, rtol=1e-4, atol=1e-5)
            or not np.allclose(historical_embeddings["held"][held_positions], held_replay_embeddings, rtol=1e-4, atol=1e-5)):
        raise RuntimeError(f"historical held fresh schedule does not match replay for {build}/{reference}; "
                           f"max probability error {fresh_held_probability_error}, max embedding error {fresh_held_embedding_error}")
    reference_rows = year == reference
    if (not np.allclose(historical_probabilities["actual"][reference_rows], historical_probabilities["held"][reference_rows], rtol=1e-4, atol=1e-5)
            or not np.allclose(historical_embeddings["actual"][reference_rows], historical_embeddings["held"][reference_rows], rtol=1e-4, atol=1e-5)):
        raise RuntimeError("historical reference actual/held control failed")
    del history
    if device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    run = Path(system["files"]["config"]["path"]).parent.parent
    frozen = _FrozenUnits(run / "unit-cache" / config["config_hash"] / "units", config)
    if sae_device == "cuda":
        pilot.configure_predictor_precision(sae_device)
    dictionaries = pilot.load_reference_dictionaries(frozen, reference, "cpu")
    held_reencoded = {seed: pilot.encode_with_reference_sae(dictionary, held_replay_embeddings, sae_device)
                      for seed, dictionary in dictionaries.items()}
    for seed, code in held_reencoded.items():
        if not np.array_equal(code.astype(held_codes[seed].dtype), held_codes[seed]):
            differences = int(np.sum(code.astype(held_codes[seed].dtype) != held_codes[seed]))
            raise RuntimeError(f"historical held codes do not replay at saved precision for {build}/{reference}, "
                               f"seed {seed}; differing entries {differences}")
    if sae_device == "cuda":
        torch.cuda.empty_cache()
    classifier, _ = _initialize_context_classifier(system, device)
    prepared = prepare_context_classifier(classifier, positional_embeddings)
    parameter_hash = pilot._identity_hash({name: value.detach().cpu() for name, value in prepared.model.named_parameters()})
    with pilot.predictor_sdpa_context(device):
        prepared.fit(np.ascontiguousarray(context[columns].to_numpy(dtype=np.float32)),
                     np.ascontiguousarray(population["y"][context_rows], dtype=np.float32),
                     additional_x={"dist_shift_domain": np.ascontiguousarray(years[context_rows] - first, dtype=np.int64)})
    if parameter_hash != pilot._identity_hash({name: value.detach().cpu() for name, value in prepared.cached_models_[0].named_parameters()}):
        raise RuntimeError("learned parameters changed during consistent cache construction")
    consistent_identity = _comparable_predictor_identity(pilot, prepared, system["context_rows_identity"], state_key)
    if (not comparable_predictor_identities_match(consistent_identity, expected_pair[0])
            or not comparable_predictor_identities_match(consistent_identity, expected_pair[1])
            or not comparable_predictor_identities_match(history_identity, consistent_identity)
            or str(consistent_identity.get("device", "")).split(":")[0] != device):
        raise RuntimeError(f"fresh consistent fitted predictor identity mismatch for {build}/{reference}")
    fitted_identity = pilot.predictor_context_identity(prepared)
    domain_device, shape = infer_model_additional_x_info(prepared)
    probabilities, embeddings = {}, {}
    for arm in ("actual", "held"):
        probabilities[arm], embeddings[arm] = [], []
        supplied_years = year if arm == "actual" else np.full(len(year), reference)
        for start in range(0, len(year), 64):
            block = slice(start, start + 64)
            domain = make_dist_tensor(np.ascontiguousarray(supplied_years[block] - first, dtype=np.int64), domain_device, shape)
            def predict():
                with pilot.predictor_sdpa_context(device):
                    return prepared.predict_full(features[block], additional_x={"dist_shift_domain": domain},
                                                 get_additional_outputs=["test_embeddings"])
            captured = capture_same_call(prepared, prepared.cached_models_[0], predict,
                                         lambda: pilot.predictor_context_identity(prepared), pilot._frozen_continuation)
            probabilities[arm].append(np.asarray(_identity_value(captured["probabilities"]))[:, 1])
            embeddings[arm].append(np.asarray(_identity_value(captured["embeddings"]))[:, 0, :])
        probabilities[arm] = np.concatenate(probabilities[arm])
        embeddings[arm] = np.concatenate(embeddings[arm])
    if pilot.predictor_context_identity(prepared) != fitted_identity:
        raise RuntimeError("consistent predictor identity changed during scoring")
    reference_rows = year == reference
    if (not np.allclose(probabilities["actual"][reference_rows], probabilities["held"][reference_rows], rtol=1e-4, atol=1e-5)
            or not np.allclose(embeddings["actual"][reference_rows], embeddings["held"][reference_rows], rtol=1e-4, atol=1e-5)):
        raise RuntimeError("reference actual/held control failed")
    del captured, prepared, classifier
    if device == "cuda" or sae_device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    scored = {"patient_id": patient_id, "year": year, "label": labels, "features": features,
              "feature_names": feature_names,
              "archive_actual_probability_max_abs_error": np.array(actual_probability_error),
              "archive_actual_embedding_max_abs_error": np.array(actual_embedding_error),
              "archive_held_probability_max_abs_error": np.array(held_probability_error),
              "fresh_held_probability_max_abs_error": np.array(fresh_held_probability_error),
              "fresh_held_embedding_max_abs_error": np.array(fresh_held_embedding_error),
              "archived_historical_device": np.array(system["saved_device"]),
              "historical_replay_device": np.array(device),
              "sae_device": np.array(sae_device),
              "historical_device_identity": np.array("match" if system["saved_device"] == device
                                                       else "unresolved: recorded and replay devices differ")}
    for arm in ("actual", "held"):
        scored[f"historical_{arm}_probability"] = historical_probabilities[arm]
        scored[f"historical_{arm}_embedding"] = historical_embeddings[arm]
        for seed, dictionary in dictionaries.items():
            scored[f"historical_{arm}_codes_{seed}"] = pilot.encode_with_reference_sae(dictionary, historical_embeddings[arm], sae_device)
    for arm in ("actual", "held"):
        scored[f"consistent_{arm}_probability"] = probabilities[arm]
        scored[f"consistent_{arm}_embedding"] = embeddings[arm]
        for seed, dictionary in dictionaries.items():
            scored[f"consistent_{arm}_codes_{seed}"] = pilot.encode_with_reference_sae(dictionary, embeddings[arm], sae_device)
    return scored


def _concept_cells(inputs, system, concept, scored):
    concept_path = Path("stats/concept_agreement_study/patient_table/concepts") / concept["file"]
    with np.load(concept_path, allow_pickle=False) as saved:
        rule = saved["in_R"].copy()
    if len(rule) != len(scored["year"]):
        raise ValueError("concept rows no longer match the scored patient table")
    seed, factor = concept["seed"], concept["factor"]
    key = (system["build"], system["reference"], system["saved_context_identity"], system["saved_checkpoint_sha256"])
    cells_by_year = {}
    for year in sorted(set(scored["year"].tolist()) & set(range(system["reference"], 2015))):
        mask = scored["year"] == year
        base = {"system": key, "concept": concept["concept"], "patient_id": scored["patient_id"][mask],
                "year": scored["year"][mask], "label": scored["label"][mask],
                "features": scored["features"][mask], "feature_names": scored["feature_names"], "rule": rule[mask]}
        cells = {}
        for processing in ("historical", "consistent"):
            for arm in ("actual", "held"):
                prefix = f"{processing}_{arm}"
                activation = scored[f"{prefix}_codes_{seed}"][:, factor]
                cells[(processing, arm)] = {**base, "probability": scored[f"{prefix}_probability"][mask],
                                            "embedding": scored[f"{prefix}_embedding"][mask],
                                            "activation": activation[mask]}
        cells_by_year[int(year)] = cells
    return cells_by_year


def _archive_activation_diagnostics(concept, scored):
    """Compare saved historical activations with fresh matched-schedule activations."""
    path = Path("stats/concept_agreement_study/patient_table/concepts") / concept["file"]
    with np.load(path, allow_pickle=False) as archive:
        saved = archive["activation"]
        fresh = scored[f"historical_actual_codes_{concept['seed']}"][:, concept["factor"]]
        if saved.shape != fresh.shape or not np.isfinite(saved).all() or not np.isfinite(fresh).all():
            raise ValueError("historical activation archive identity mismatch")
        disagreement = (saved > concept["cutoff"]) != (fresh > concept["cutoff"])
        return {int(year): {"rows": int(np.sum(scored["year"] == year)),
                            "strict_cutoff_disagreements": int(np.sum(disagreement & (scored["year"] == year))),
                            "max_absolute_error": float(np.max(np.abs(saved[scored["year"] == year] - fresh[scored["year"] == year])))}
                for year in np.unique(scored["year"])}


def _run_comparison(variants=("last_3",), device="cpu", output=None, reuse_primary=False,
                    r1_device="cpu", checks_dir=None, sae_device="cpu"):
    """Score a declared panel only after current provenance and parity checks pass."""
    import subprocess
    import time
    from datetime import datetime, timezone

    variants = tuple(variants)
    inputs = (_comparison_inputs(variants, device) if r1_device == "cpu" and checks_dir is None and sae_device == "cpu"
              else _comparison_inputs(variants, device, r1_device, checks_dir, sae_device))
    destination = Path(output) if output is not None else Path("stats/concept_agreement_study/extensions/temporal_input_attribution") / "_".join(variants)
    if destination.exists() and any(item.name != "run" or not item.is_dir() for item in destination.iterdir()):
        raise FileExistsError("comparison output already contains files")
    bootstrap = patient_bootstrap(inputs["patient_sets"])
    destination.mkdir(parents=True, exist_ok=True)
    private = destination / "private"
    private.mkdir(mode=0o700)
    started, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    changes = subprocess.run(["git", "status", "--porcelain", "--", "temporal_input_attribution.py",
                              "tests/test_temporal_input_attribution.py"], capture_output=True, text=True, check=True).stdout.splitlines()
    initial = {"status": "running", "variants": variants, "device": device, "r1_device": r1_device,
               "sae_device": sae_device,
               "readiness_provenance_sha256": inputs.get("readiness_provenance_sha256"),
               "started_utc": started,
               "commit": revision, "uncommitted_code_files": changes,
               "source_sha256": inventory_inputs({"source": Path(__file__)}, {})["files"]["source"]["sha256"],
               "systems": [{"build": system["build"], "reference": system["reference"],
                            "historical_device": system.get("saved_device", "unknown"),
                            "archived_historical_device": system.get("saved_device", "unknown"),
                            "historical_replay_device": _system_device(system, device, r1_device),
                            "scientific_device": _system_device(system, device, r1_device),
                            "sae_device": sae_device,
                            "consistent_device": _system_device(system, device, r1_device)}
                           for system in inputs["systems"]], "outputs": ["private/"]}
    (destination / "run-record.json").write_text(json.dumps(initial, indent=2) + "\n")
    cases, system_records = [], []
    for system in inputs["systems"]:
        key = (system["build"], system["reference"])
        system_device = _system_device(system, device, r1_device)
        fingerprint = inputs["fingerprints"][key]
        stem = f"{key[0]}_{key[1]}"
        if not all(character.isalnum() or character == "_" for character in stem):
            raise ValueError("invalid saved build name")
        reusable = PRIMARY_COMPARISON / "private" / f"{stem}.json"
        if reuse_primary and reusable.exists():
            meta = json.loads(reusable.read_text())
            if meta.get("fingerprint") != fingerprint or meta.get("system") != [key[0], key[1]]:
                raise ValueError(f"reuse identity mismatch for {stem}")
            archive_path = reusable.with_suffix(".npz")
            archive_hash = inventory_inputs({"archive": archive_path}, {})["files"]["archive"]["sha256"]
            if archive_hash is None or meta.get("archive_sha256") != archive_hash:
                raise ValueError(f"reuse archive checksum mismatch for {stem}")
            with np.load(archive_path, allow_pickle=False) as archive:
                scored = {name: archive[name] for name in archive.files}
            reused = True
        else:
            scored = (_score_system(inputs, system, system_device) if sae_device == "cpu"
                      else _score_system(inputs, system, system_device, sae_device))
            reused = False
        archive_path = private / f"{stem}.npz"
        np.savez_compressed(archive_path, **scored)
        archive_hash = inventory_inputs({"archive": archive_path}, {})["files"]["archive"]["sha256"]
        (private / f"{stem}.json").write_text(json.dumps({"fingerprint": fingerprint, "system": list(key),
                                                        "archive_sha256": archive_hash}) + "\n")
        threshold_variants = set(system["variants"]) & set(variants)
        thresholds = {float(system["death_threshold"][variant]) for variant in threshold_variants}
        if len(thresholds) != 1:
            raise ValueError(f"aliases for {stem} have different frozen thresholds")
        threshold = thresholds.pop()
        for concept in inputs["concepts"]:
            if (concept["build"], concept["reference"]) != key:
                continue
            cells_by_year = _concept_cells(inputs, system, concept, scored)
            archive_diagnostics = _archive_activation_diagnostics(concept, scored)
            yearly = {int(year): summarize_four_cells(cells, threshold, concept["cutoff"], bootstrap)
                      for year, cells in sorted(cells_by_year.items())}
            if not yearly:
                cases.append({"concept": concept["concept"], "build": key[0], "reference": key[1],
                              "unavailable_reason": "no eligible evaluation rows"})
                continue
            for missing in sorted(set(range(key[1], 2015)) - set(yearly)):
                cases.append({"concept": concept["concept"], "build": key[0], "reference": key[1],
                              "year": missing, "unavailable_reason": "no eligible evaluation rows"})
            for year, summary in yearly.items():
                if year > 2014 or year < key[1]:
                    raise ValueError("year outside frozen mortality period")
                year_changes = (reference_to_later(yearly[key[1]], summary) if year > key[1] and key[1] in yearly
                                else {"unavailable_reason": "missing reference-year rows"} if year > key[1] else None)
                cases.append({"concept": concept["concept"], "build": key[0], "reference": key[1],
                              "year": year, "cells": summary["cells"], "contrasts": summary["contrasts"],
                              "reference_to_later": year_changes,
                              "historical_archive_activation": archive_diagnostics.get(year)})
        system_records.append({"build": key[0], "reference": key[1], "input_fingerprint": fingerprint,
                               "historical_device": system.get("saved_device", "unknown"),
                               "archived_historical_device": system.get("saved_device", "unknown"),
                               "historical_replay_device": system_device, "scientific_device": system_device,
                               "sae_device": sae_device,
                               "consistent_device": system_device, "reused_primary": reused})

    def public(value):
        if isinstance(value, dict):
            return {"/".join(key) if isinstance(key, tuple) else str(key): public(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [public(item) for item in value]
        if isinstance(value, np.generic):
            return public(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    (destination / "aggregate.json").write_text(json.dumps(public({"cases": cases, "systems": system_records}), indent=2, allow_nan=False) + "\n")
    record = {"status": "completed_with_unavailable" if any("unavailable_reason" in case for case in cases) else "completed",
              "variants": variants, "device": device, "r1_device": r1_device, "sae_device": sae_device,
              "started_utc": started,
              "readiness_provenance_sha256": inputs.get("readiness_provenance_sha256"),
              "ended_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - clock,
              "commit": revision, "uncommitted_code_files": changes, "systems": system_records,
              "source_sha256": inventory_inputs({"source": Path(__file__)}, {})["files"]["source"]["sha256"],
              "concept_years": sum("cells" in case for case in cases), "outputs": ["aggregate.json", "private/"]}
    (destination / "run-record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def run_comparison(variants=("last_3",), device="cpu", output=None, reuse_primary=False,
                   r1_device="cpu", checks_dir=None, sae_device="cpu"):
    """Run the gated comparison and preserve a failure record if output creation began."""
    import subprocess
    import time
    from datetime import datetime, timezone

    variants = tuple(variants)
    if variants != ("last_3",):
        raise ValueError("primary plan only permits the last_3 variant")
    _system_device({"build": "r1", "reference": 2007}, device, r1_device)
    if sae_device not in ("cpu", "cuda"):
        raise ValueError("SAE device must be cpu or cuda")
    destination = Path(output) if output is not None else Path("stats/concept_agreement_study/extensions/temporal_input_attribution") / "_".join(variants)
    existed = destination.exists()
    launcher_only = existed and destination.is_dir() and all(item.name == "run" and item.is_dir() for item in destination.iterdir())
    started, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    try:
        return _run_comparison(variants, device, destination, reuse_primary, r1_device, checks_dir, sae_device)
    except Exception as error:
        if (not existed or launcher_only) and destination.is_dir():
            revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
            changes = subprocess.run(["git", "status", "--porcelain", "--", "temporal_input_attribution.py",
                                      "tests/test_temporal_input_attribution.py"], capture_output=True, text=True, check=True).stdout.splitlines()
            record_path = destination / "run-record.json"
            record = json.loads(record_path.read_text()) if record_path.exists() else {}
            record.update({"status": "failed", "variants": variants, "device": device,
                      "r1_device": r1_device, "sae_device": sae_device, "started_utc": started,
                      "ended_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - clock,
                      "commit": revision, "uncommitted_code_files": changes,
                      "source_sha256": inventory_inputs({"source": Path(__file__)}, {})["files"]["source"]["sha256"],
                      **sanitize_public_failure(error), "outputs": ["private/"]})
            record_path.write_text(json.dumps(record, indent=2) + "\n")
        raise


def validate_four_cells(cells):
    expected = {(processing, arm) for processing in ("historical", "consistent")
                for arm in ("actual", "held")}
    if set(cells) != expected:
        raise ValueError("all four processing/input cells are required")
    base = cells[("historical", "actual")]
    for field in ("system", "concept"):
        if field not in base or base[field] is None or base[field] == "":
            raise ValueError(f"missing {field} identity")
    keys = _row_keys(base, "historical actual")
    if len(np.unique(base["year"])) != 1:
        raise ValueError("each four-cell summary must contain one evaluation year")
    for name, cell in cells.items():
        if _row_keys(cell, str(name)) != keys:
            raise ValueError("patient/year order differs across cells")
        for field in ("features", "label", "feature_names", "rule"):
            left, right = np.asarray(base[field]), np.asarray(cell[field])
            if left.shape != right.shape or left.dtype != right.dtype or left.tobytes() != right.tobytes():
                raise ValueError(f"{field} differs across cells")
        for field in ("system", "concept"):
            if (field in base) != (field in cell) or (field in base and base[field] != cell[field]):
                raise ValueError(f"{field} differs across cells")
        n = len(keys)
        for field in ("probability", "activation"):
            value = np.asarray(cell[field])
            if value.shape != (n,) or not np.isfinite(value).all():
                raise ValueError(f"invalid {field}")
        probability = np.asarray(cell["probability"])
        if np.any((probability < 0) | (probability > 1)):
            raise ValueError("probability outside [0, 1]")
        embedding = np.asarray(cell["embedding"])
        if embedding.ndim != 2 or len(embedding) != n or not np.isfinite(embedding).all():
            raise ValueError("invalid embedding")
        if embedding.shape != np.asarray(base["embedding"]).shape:
            raise ValueError("embedding shape differs across cells")
        if np.asarray(cell["features"]).ndim != 2 or len(cell["features"]) != n:
            raise ValueError("invalid features")
        if np.asarray(cell["rule"]).shape != (n,) or np.asarray(cell["rule"]).dtype != np.bool_:
            raise ValueError("invalid rule")
        if np.asarray(cell["label"]).shape != (n,) or not np.isin(cell["label"], [0, 1]).all():
            raise ValueError("invalid mortality label")
    return True


def _ratio(numerator, denominator):
    numerator, denominator = float(numerator), float(denominator)
    return {"estimate": numerator / denominator if denominator else np.nan,
            "numerator": numerator, "denominator": denominator,
            "reason": None if denominator else "zero denominator"}


def cell_metrics(cell, threshold, cutoff, weights=None, include_descriptive=True):
    """Frozen threshold, rule and activation summaries for a single cell."""
    from sklearn.metrics import roc_auc_score

    probability = np.asarray(cell["probability"], dtype=float)
    activation = np.asarray(cell["activation"], dtype=float)
    labels, rule = np.asarray(cell["label"]), np.asarray(cell["rule"])
    n = len(probability)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if (not np.isfinite(threshold) or not 0 <= threshold <= 1 or not np.isfinite(cutoff)
            or probability.shape != (n,) or activation.shape != (n,) or labels.shape != (n,)
            or rule.shape != (n,) or rule.dtype != np.bool_ or w.shape != (n,)
            or not np.isfinite(w).all() or np.any(w < 0) or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1)) or not np.isfinite(activation).all()
            or not np.isin(labels, [0, 1]).all()):
        raise ValueError("invalid cell metrics inputs")
    death, flagged, active = labels == 1, probability >= threshold, activation > cutoff
    count = lambda mask: float(w @ np.asarray(mask, dtype=float))
    total, deaths, survivors = float(w.sum()), count(death), count(~death)
    tp, fp, fn, tn = (count(death & flagged), count(~death & flagged),
                      count(death & ~flagged), count(~death & ~flagged))
    both, rule_only, activation_only, neither = (count(rule & active), count(rule & ~active),
                                                  count(~rule & active), count(~rule & ~active))
    entropy = np.zeros(n, dtype=float)
    middle = (probability > 0) & (probability < 1)
    p = probability[middle]
    entropy[middle] = -p * np.log(p) - (1 - p) * np.log1p(-p)
    counts = {"patients": total, "deaths": deaths, "survivors": survivors,
              "caught_deaths": tp, "false_positives": fp, "false_negatives": fn,
              "true_negatives": tn, "rule_and_active": both, "rule_only": rule_only,
              "activation_only": activation_only, "neither": neither,
              "rule_positive": both + rule_only, "activation_positive": both + activation_only}
    metrics = {name: _ratio(value, 1) for name, value in counts.items()}
    metrics.update({
        "mean_probability": _ratio(w @ probability, total),
        "mean_entropy": _ratio(w @ entropy, total),
        "mean_activation": _ratio(w @ activation, total),
        "agreement": _ratio(both, both + rule_only),
        "rule_recall": _ratio(both, both + activation_only),
        "rule_f2": _ratio(5 * both, 5 * both + 4 * activation_only + rule_only),
        "overlap": _ratio(both, both + rule_only + activation_only),
        "rule_rate": _ratio(both + rule_only, total),
        "activation_rate": _ratio(both + activation_only, total),
        "false_positive_rate": _ratio(fp, survivors),
        "false_negative_rate": _ratio(fn, deaths),
        "death_recall": _ratio(tp, deaths),
        "death_precision": _ratio(tp, tp + fp),
        "death_f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "mortality": _ratio(deaths, total),
    })
    if include_descriptive:
        metrics["brier"] = _ratio(w @ ((probability - labels) ** 2), total)
        if deaths and survivors:
            metrics["auroc"] = {"estimate": float(roc_auc_score(labels, probability, sample_weight=w)),
                                "numerator": None, "denominator": None, "reason": None}
        else:
            metrics["auroc"] = {"estimate": np.nan, "numerator": None, "denominator": None,
                                "reason": "both outcome classes required"}
    return metrics


def patient_bootstrap(patient_sets, draws=1000, seed=20261106):
    """Draw one multiplicity for each patient shared by all supplied panels."""
    sets = [np.asarray(values).astype(str) for values in patient_sets]
    if not sets or draws <= 0:
        raise ValueError("patient bootstrap needs patients and positive draws")
    patients = np.unique(np.concatenate(sets))
    if len(patients) == 0:
        raise ValueError("patient bootstrap needs patients")
    weights = np.random.default_rng(seed).multinomial(len(patients), np.full(len(patients), 1 / len(patients)), size=draws)
    return patients, weights


def _interval(estimate, draws):
    values = np.asarray(draws, dtype=float)
    valid = values[np.isfinite(values)]
    low, high = np.percentile(valid, [2.5, 97.5]) if len(valid) and np.isfinite(estimate) else (np.nan, np.nan)
    return {"estimate": float(estimate), "low": float(low), "high": float(high),
            "valid_draws": len(valid), "undefined_draws": len(values) - len(valid),
            "reason": None if np.isfinite(estimate) else "undefined component"}


def _contrast_values(values):
    h = values[("historical", "actual")] - values[("historical", "held")]
    n = values[("consistent", "actual")] - values[("consistent", "held")]
    return {"historical_actual_minus_held": h, "consistent_actual_minus_held": n,
            "normalization_sensitivity": n - h,
            "consistent_minus_historical_actual": values[("consistent", "actual")] - values[("historical", "actual")],
            "consistent_minus_historical_held": values[("consistent", "held")] - values[("historical", "held")]}


def summarize_four_cells(cells, threshold, cutoff, bootstrap):
    """Summarize one concept/system/year with a shared patient bootstrap."""
    validate_four_cells(cells)
    patients, draws = np.asarray(bootstrap[0]).astype(str), np.asarray(bootstrap[1])
    if (patients.ndim != 1 or len(np.unique(patients)) != len(patients)
            or not np.array_equal(patients, np.sort(patients)) or draws.ndim != 2
            or draws.shape[1] != len(patients) or len(draws) == 0
            or not np.issubdtype(draws.dtype, np.integer) or np.any(draws < 0)):
        raise ValueError("invalid shared patient bootstrap")
    row_patients = np.asarray(cells[("historical", "actual")]["patient_id"]).astype(str)
    positions = np.searchsorted(patients, row_patients)
    if np.any(positions >= len(patients)) or not np.array_equal(patients[positions], row_patients):
        raise ValueError("patient missing from shared bootstrap")
    identity = hashlib.sha256()
    for patient in patients:
        encoded = patient.encode("utf-8")
        identity.update(len(encoded).to_bytes(8, "big"))
        identity.update(encoded)
    identity.update(np.ascontiguousarray(draws).tobytes())
    point = {key: cell_metrics(cell, threshold, cutoff) for key, cell in cells.items()}
    metric_names = [name for name in point[("historical", "actual")] if name not in ("auroc", "brier")]
    cell_draws = {key: {name: [] for name in metric_names} for key in cells}
    for draw in draws:
        weights = draw[positions]
        for key, cell in cells.items():
            result = cell_metrics(cell, threshold, cutoff, weights, include_descriptive=False)
            for name in metric_names:
                cell_draws[key][name].append(result[name]["estimate"])
    for key in cells:
        for name in metric_names:
            point[key][name].update(_interval(point[key][name]["estimate"], cell_draws[key][name]))
            if point[key][name]["denominator"] == 0:
                point[key][name]["reason"] = "zero denominator"
    contrasts, contrast_draws = {}, {}
    for name in metric_names:
        values = {key: point[key][name]["estimate"] for key in cells}
        draw_values = {key: np.asarray(cell_draws[key][name]) for key in cells}
        estimates, samples = _contrast_values(values), _contrast_values(draw_values)
        contrasts[name] = {label: {**_interval(estimates[label], samples[label]),
                                   "components": {key: {field: point[key][name][field]
                                                        for field in ("estimate", "numerator", "denominator", "reason")}
                                                  for key in cells}}
                           for label in estimates}
        contrast_draws[name] = samples
    displacement, displacement_draws, per_patient = {}, {}, {}
    for field in ("probability", "activation"):
        arrays = {key: np.asarray(cell[field], dtype=float) for key, cell in cells.items()}
        per_patient[field] = _contrast_values(arrays)
    embedding_pairs = {
        "historical_actual_minus_held": (("historical", "actual"), ("historical", "held")),
        "consistent_actual_minus_held": (("consistent", "actual"), ("consistent", "held")),
        "consistent_minus_historical_actual": (("consistent", "actual"), ("historical", "actual")),
        "consistent_minus_historical_held": (("consistent", "held"), ("historical", "held")),
    }
    for label, (left, right) in embedding_pairs.items():
        distances = np.linalg.norm(np.asarray(cells[left]["embedding"], dtype=float)
                                   - np.asarray(cells[right]["embedding"], dtype=float), axis=1)
        estimate = _ratio(distances.sum(), len(distances))
        samples = np.array([_ratio(draw[positions] @ distances, draw[positions].sum())["estimate"] for draw in draws])
        displacement[label] = {**estimate, **_interval(estimate["estimate"], samples)}
        displacement_draws[label] = samples
        per_patient.setdefault("embedding_displacement", {})[label] = distances
    contrasts["embedding_displacement"] = displacement
    contrast_draws["embedding_displacement"] = displacement_draws
    base = cells[("historical", "actual")]
    return {"system": base["system"], "concept": base["concept"], "year": int(base["year"][0]),
            "threshold": float(threshold), "cutoff": float(cutoff),
            "cells": point, "contrasts": contrasts, "bootstrap_identity": identity.hexdigest(),
            "per_patient": per_patient, "draws": {"cells": cell_draws, "contrasts": contrast_draws}}


def reference_to_later(reference, later):
    """Subtract reference estimates inside each shared patient-cluster draw."""
    if reference["bootstrap_identity"] != later["bootstrap_identity"]:
        raise ValueError("reference and later need the same shared patient bootstrap")
    for field in ("system", "concept", "threshold", "cutoff"):
        if reference[field] != later[field]:
            raise ValueError(f"reference and later {field} mismatch")
    if reference["year"] >= later["year"]:
        raise ValueError("later year must follow reference year")
    result = {"cells": {}, "contrasts": {}}
    for key, metrics in later["cells"].items():
        result["cells"][key] = {}
        for name in later["draws"]["cells"][key]:
            samples = (np.asarray(later["draws"]["cells"][key][name])
                       - np.asarray(reference["draws"]["cells"][key][name]))
            estimate = metrics[name]["estimate"] - reference["cells"][key][name]["estimate"]
            result["cells"][key][name] = {**_interval(estimate, samples),
                                           "reference": reference["cells"][key][name],
                                           "later": metrics[name]}
    for name, measures in later["contrasts"].items():
        result["contrasts"][name] = {}
        for key, metric in measures.items():
            samples = (np.asarray(later["draws"]["contrasts"][name][key])
                       - np.asarray(reference["draws"]["contrasts"][name][key]))
            estimate = metric["estimate"] - reference["contrasts"][name][key]["estimate"]
            result["contrasts"][name][key] = {**_interval(estimate, samples),
                                               "reference": reference["contrasts"][name][key],
                                               "later": metric}
    return result


def audit_roles(saved, original):
    saved_roles = np.asarray(saved["row_role"])
    original_roles = np.asarray(original["row_role"])
    allowed = {"", "tabpfn_context", "sae_discovery", "rule_discovery", "cav_fit",
               "concept_selection", "evaluation", "fitting"}
    if saved["identity"] != original["identity"] or saved_roles.shape != original_roles.shape:
        raise ValueError("roles identity or order mismatch")
    if not set(original_roles.tolist()) <= allowed:
        raise ValueError("roles contain an unknown original role")
    projected = np.where(np.isin(original_roles, ["tabpfn_context", "sae_discovery"]),
                         "protected", original_roles)
    if not np.array_equal(saved_roles, projected):
        raise ValueError("roles identity or order mismatch")


class ContextMinMaxScaler(MinMaxScalingEncoderStep):
    """Retain extrema from context so cached queries use the joint scaling rule."""

    def _fit(self, x, single_eval_pos, **kwargs):
        if single_eval_pos is None or single_eval_pos <= 0 or single_eval_pos > len(x):
            raise RuntimeError("context positions are required to fit temporal scaling")
        context = x[:single_eval_pos]
        self.context_min = torch.min(context, dim=0, keepdim=True)[0].detach().clone()
        self.context_max = torch.max(context, dim=0, keepdim=True)[0].detach().clone()

    def _transform(self, x, single_eval_pos, **kwargs):
        if not hasattr(self, "context_min"):
            raise RuntimeError("context temporal bounds have not been fitted")
        span = self.context_max - self.context_min
        denominator = span + (span < 1e-16).float()
        scaled = self.min + ((x - self.context_min) * (self.max - self.min)) / denominator
        return (torch.clamp(scaled, min=self.min - 5, max=self.max + 5),)


def with_context_scaler(model):
    """Copy a model before cache construction and replace only its temporal scaler."""
    if getattr(model, "cache_trainset_representation", False):
        raise ValueError("install the context scaler before cache construction")
    copy = deepcopy(model)
    matches = [(name, step) for name, step in copy.encoder.named_children()
               if isinstance(step, MinMaxScalingEncoderStep)
               and list(step.in_keys) == ["dist_shift_domain"]]
    if len(matches) != 1:
        raise ValueError("expected exactly one temporal scaler")
    name, previous = matches[0]
    if not previous.normalize_on_train_only:
        raise ValueError("temporal scaler must normalize on context only")
    replacement = ContextMinMaxScaler(normalize_on_train_only=True, min=previous.min, max=previous.max,
                                      in_keys=previous.in_keys, out_keys=previous.out_keys)
    copy.encoder._modules[name] = replacement
    return copy


def _install_positional_embeddings(model, positional_embeddings):
    """Prescribe subspace vectors on model instances before native cache construction."""
    from tabpfn.model.transformer import PerFeatureTransformer

    if (not torch.is_tensor(positional_embeddings) or positional_embeddings.ndim != 2
            or positional_embeddings.numel() == 0 or positional_embeddings.dtype != torch.float32
            or not torch.isfinite(positional_embeddings).all().item()):
        raise ValueError("prescribed positional embeddings must be a nonempty finite float32 matrix")
    prescribed = positional_embeddings.detach().clone()
    transformers = [module for module in model.modules()
                    if isinstance(module, PerFeatureTransformer)
                    and module.feature_positional_embedding == "subspace"]
    if not transformers:
        raise ValueError("model has no subspace positional transformer")
    for transformer in transformers:
        width = transformer.feature_positional_embedding_embeddings.out_features
        if prescribed.shape[1] != width:
            raise ValueError("prescribed positional embedding width is incompatible with model")
        transformer._pilot_prescribed_positional_embeddings = prescribed.clone()
        transformer._pilot_positional_embeddings_identity = prescribed.clone()

        def add_embeddings(self, x, y, num_features, seq_len, cache_embeddings=False,
                           use_cached_embeddings=False):
            expected = self._pilot_prescribed_positional_embeddings
            identity = self._pilot_positional_embeddings_identity
            if not torch.equal(expected, identity):
                raise RuntimeError("prescribed positional state changed after installation")
            if (x.ndim != 4 or expected.shape != (x.shape[2], x.shape[3])
                    or expected.dtype != x.dtype or expected.device != x.device):
                raise ValueError("prescribed positional embeddings are incompatible with model input: "
                                 f"state={tuple(expected.shape)}/{expected.dtype}/{expected.device}, "
                                 f"input={tuple(x.shape)}/{x.dtype}/{x.device}")
            if use_cached_embeddings:
                cached = getattr(self, "cached_embeddings", None)
                if cached is None or not torch.equal(cached, expected):
                    raise RuntimeError("cached positional embeddings differ from prescribed state")
                x += cached[None, None]
                return x, y
            x += expected[None, None]
            self.cached_embeddings = expected.clone() if cache_embeddings else None
            return x, y

        transformer.add_embeddings = MethodType(add_embeddings, transformer)


def prepare_context_classifier(classifier, positional_embeddings=None, use_context_scaler=True):
    """Copy an initialized classifier with native or context-scaled model preprocessing."""
    if getattr(classifier, "cached_models_", None):
        raise ValueError("classifier already has a context cache")
    if not hasattr(classifier, "model_processed_") or not hasattr(classifier, "c_processed_"):
        raise ValueError("classifier must be initialized before preparing its model")
    prepared = deepcopy(classifier)
    prepared.model = (with_context_scaler(classifier.model_processed_) if use_context_scaler
                      else prepared.model_processed_)
    if positional_embeddings is not None:
        _install_positional_embeddings(prepared.model, positional_embeddings)
    prepared.c = deepcopy(classifier.c_processed_)
    return prepared


def capture_same_call(classifier, model, predict, identity, continuation_factory):
    """Observe one model call without changing its parser, hooks, or configuration."""
    times = [module for module in model.modules() if isinstance(module, Time2VecEncoderStep)]
    if len(times) != 1:
        raise ValueError("expected exactly one Time2Vec encoder")
    head = model.decoder_dict["standard"]
    before = identity()
    captured = {"head_inputs": [], "time_inputs": [], "configurations": []}
    names = ("_parse_predictions_per_configuration", "_aggregate_predictions")
    originals = {name: getattr(classifier, name) for name in names}
    previous = {name: classifier.__dict__[name] for name in names if name in classifier.__dict__}
    handles = []

    def observe_parse(predictions, ensemble_configurations):
        captured["configurations"].append(deepcopy(ensemble_configurations))
        return originals[names[0]](predictions, ensemble_configurations)

    def observe_aggregate(ensemble_outputs, **kwargs):
        captured["aggregate_kwargs"] = {key: value.detach().clone() if torch.is_tensor(value)
                                        else deepcopy(value) for key, value in kwargs.items()}
        return originals[names[1]](ensemble_outputs, **kwargs)

    try:
        handles.append(head.register_forward_pre_hook(
            lambda module, args: captured["head_inputs"].append(args[0].detach().clone())))
        handles.append(times[0].register_forward_pre_hook(
            lambda module, args: captured["time_inputs"].append(args[0]["dist_shift_domain"].detach().clone())))
        classifier._parse_predictions_per_configuration = observe_parse
        classifier._aggregate_predictions = observe_aggregate
        with torch.inference_mode():
            result = predict()
    finally:
        for handle in handles:
            handle.remove()
        for name in names:
            if name in previous:
                setattr(classifier, name, previous[name])
            elif name in classifier.__dict__:
                delattr(classifier, name)
    if identity() != before:
        raise RuntimeError("prediction changed fitted context or configuration")
    if any(len(captured[key]) != 1 for key in ("head_inputs", "time_inputs", "configurations")):
        raise RuntimeError("prediction did not execute one head, temporal encoder and configuration")
    embeddings = torch.as_tensor(result["test_embeddings"])
    if embeddings.ndim != 3 or embeddings.shape[1] != 1:
        raise RuntimeError("expected one configuration of decoder embeddings")
    torch.testing.assert_close(captured["head_inputs"][0].to(embeddings.device), embeddings,
                               rtol=1e-4, atol=1e-5)
    device = next(head.parameters()).device
    coordinates = embeddings[:, 0, :].to(device).clone()
    probabilities = torch.as_tensor(result["proba"], device=device, dtype=torch.float32)
    continuation = continuation_factory(classifier, head, captured["configurations"][0],
                                        captured.get("aggregate_kwargs", {}), before)
    with torch.no_grad():
        replay = continuation.probabilities(coordinates)
    torch.testing.assert_close(replay, probabilities, rtol=1e-4, atol=1e-5)
    if not torch.isfinite(coordinates).all() or not torch.isfinite(probabilities).all():
        raise RuntimeError("nonfinite same-call embeddings or probabilities")
    if identity() != before:
        raise RuntimeError("continuation changed fitted context or configuration")
    return {"probabilities": result["proba"], "embeddings": result["test_embeddings"],
            "time_values": captured["time_inputs"][0], "configurations": captured["configurations"][0],
            "aggregate_kwargs": captured.get("aggregate_kwargs", {}), "continuation": continuation,
            "replay_max_absolute_error": float((replay - probabilities).abs().max().cpu())}


def audit_ledger(rows, builds):
    """Validate work 0's full concept keys against the saved build aliases."""
    alias_owners = {}
    for build in builds:
        key = (str(build["build"]), int(build["reference"]))
        for alias in build["variants"]:
            alias_key = (int(build["reference"]), alias)
            if alias_key in alias_owners:
                raise ValueError(f"duplicate build alias {alias_key}")
            alias_owners[alias_key] = key

    identities = set()
    result = []
    for row in rows:
        build, reference, family = str(row["build"]), int(row["reference_year"]), str(row["family"])
        identity = f"{build}/{reference}/{family}"
        if row["full_identity"] != identity:
            raise ValueError(f"invalid full identity {row['full_identity']!r}")
        if identity in identities:
            raise ValueError(f"duplicate full identity {identity}")
        identities.add(identity)
        aliases = [alias.strip() for alias in row["aliases"].split(",") if alias.strip()]
        if len(aliases) != len(set(aliases)):
            raise ValueError(f"duplicate alias for {identity}")
        for alias in set(aliases) & {"last_3", "all_history"}:
            if alias_owners.get((reference, alias)) != (build, reference):
                raise ValueError(f"alias {alias} does not identify {build}/{reference}")
        result.append({**row, "reference_year": reference, "aliases": aliases})
    return result


def _row_keys(table, name):
    patients = np.asarray(table["patient_id"])
    years = np.asarray(table["year"])
    if patients.ndim != 1 or years.ndim != 1 or len(patients) != len(years):
        raise ValueError(f"invalid {name} patient/year columns")
    keys = list(zip(patients.astype(str), years.tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate patient/year in {name}")
    return keys


def audit_paired_rows(actual, held, population, last_outcome_year=2014):
    """Check archived rows against their original clinical rows and held positions.

    Population can contain additional rows; archive order remains authoritative.
    The held year is always the true evaluation year, not the supplied domain.
    """
    actual_keys = _row_keys(actual, "actual")
    held_keys = _row_keys(held, "held")
    population_keys = _row_keys(population, "population")
    lookup = {key: i for i, key in enumerate(population_keys)}
    try:
        indices = np.array([lookup[key] for key in actual_keys], dtype=int)
    except KeyError as exc:
        raise ValueError(f"actual patient/year absent from population: {exc}") from exc
    for name in ("features",):
        values, source = np.asarray(actual[name]), np.asarray(population[name])
        if len(values) != len(actual_keys) or len(source) != len(population_keys):
            raise ValueError(f"invalid {name} row count")
        selected = source[indices]
        if (values.dtype != selected.dtype or values.shape != selected.shape
                or values.tobytes(order="C") != selected.tobytes(order="C")):
            raise ValueError(f"{name} mismatch")
    labels = np.asarray(actual["label"])
    original_labels = np.asarray(population["label"])
    if labels.shape != (len(actual_keys),) or len(original_labels) != len(population_keys):
        raise ValueError("invalid label row count")
    if labels.dtype != np.dtype(np.float32):
        raise ValueError("label archive must use the writer's float32 serialization")
    years = np.asarray(actual["year"])
    eligible = years <= last_outcome_year
    if (not np.isfinite(labels[eligible]).all() or not np.isnan(labels[~eligible]).all()
            or not np.array_equal(labels[eligible], original_labels[indices][eligible].astype(np.float32))):
        raise ValueError("label values or documented outcome-year mask mismatch")
    if not np.array_equal(actual["feature_names"], population["feature_names"]):
        raise ValueError("feature_names mismatch")
    positions = np.asarray(held["positions"])
    if (positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer)
            or len(positions) != len(held_keys) or np.any(positions < 0)
            or np.any(positions >= len(actual_keys)) or len(set(positions.tolist())) != len(positions)):
        raise ValueError("invalid held positions")
    if [actual_keys[i] for i in positions] != held_keys:
        raise ValueError("held patient/year does not match actual positions")
    label_years = {str(year): {"rows": int(np.sum(years == year)),
                               "masked": int(np.sum(np.isnan(labels[years == year])))}
                   for year in np.unique(years)}
    return {"actual_rows": len(actual_keys), "held_rows": len(held_keys),
            "actual_rows_without_held": len(actual_keys) - len(held_keys),
            "archive_label_dtype": str(labels.dtype), "population_label_dtype": str(original_labels.dtype),
            "label_years": label_years}


def inventory_inputs(paths, provenance):
    """Hash only explicitly requested inputs and keep unavailable provenance visible."""
    files = {}
    for name, value in paths.items():
        path = Path(value)
        entry = {"path": str(path), "status": "missing", "sha256": None}
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            entry.update(status="available", sha256=digest.hexdigest(), size_bytes=path.stat().st_size)
        files[name] = entry
    return {"files": files, "provenance": {k: "unknown" if v is None else v
                                           for k, v in provenance.items()}}
