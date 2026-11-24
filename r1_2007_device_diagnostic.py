"""Observe and control fitted TabPFN positional embeddings for diagnosis."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import numpy as np


_IDENTITY_FIELDS = (
    "X_", "y_", "_categorical_features", "additional_x_", "categorical_features",
    "classes_", "c_processed_", "cached_preprocessors_", "cached_shuffles_",
    "data_mean_", "data_std_", "average_logits", "softmax_temperature",
    "ensemble_configurations_", "transformer_predict_kwargs", "preprocess_transforms",
    "feature_shift_decoder", "normalize_with_test", "normalize_to_ranking",
    "normalize_to_clip", "remove_outliers", "fit_at_predict_time", "fp16_inference",
    "sklearn_compatible_precision", "pilot_checkpoint_sha256_", "pilot_domain_policy_",
)


def observe_fitted_state(pilot, system):
    import torch

    from r1_2007_replay import _state_report

    classifier = system["classifier"]
    cached_models = getattr(classifier, "cached_models_", None)
    if not isinstance(cached_models, (list, tuple)) or len(cached_models) != 1:
        raise ValueError("fitted predictor requires exactly one cached learned model")
    model = cached_models[0]
    mode = getattr(model, "feature_positional_embedding", None)
    if mode != "subspace":
        raise ValueError(f"expected subspace positional embeddings, got {mode!r}")
    embeddings = getattr(model, "cached_embeddings", None)
    if embeddings is None:
        raise ValueError("cached positional embeddings are missing")
    if (not torch.is_tensor(embeddings) or embeddings.ndim != 2
            or embeddings.shape[0] == 0 or embeddings.shape[1] == 0):
        raise ValueError("positional embeddings must be a nonempty matrix")
    if embeddings.dtype != torch.float32:
        raise ValueError("positional embeddings must use float32")
    if not torch.isfinite(embeddings).all().item():
        raise ValueError("nonfinite positional embeddings")

    positional_sha = _tensor_sha256(embeddings)
    buffers = []
    named_buffers = getattr(model, "named_buffers", None)
    if callable(named_buffers):
        for name, buffer in named_buffers():
            identity = _tensor_identity(buffer)
            if buffer.dtype.is_floating_point or buffer.dtype.is_complex:
                identity["nonfinite_element_count"] = int((~torch.isfinite(buffer)).sum().item())
            else:
                identity["nonfinite_element_count"] = 0
            buffers.append({"name": name, **identity})
    learned_state = _state_report(pilot, system)
    execution_state = _execution_state(model)
    return {
        "positional_state": {
            "mode": mode,
            "cache_name": "cached_embeddings",
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "device": str(embeddings.device),
            "sha256": positional_sha,
            "registered_buffer": any(item["name"] == "cached_embeddings" for item in buffers),
        },
        "learned_parameter_identity": learned_state["learned_parameter_identity"],
        "learned_parameter_count": learned_state["learned_parameter_count"],
        "registered_buffers": buffers,
        "execution_state": execution_state,
    }


@contextmanager
def scoped_cached_embedding_injection(embeddings):
    import torch

    if not torch.is_tensor(embeddings) or embeddings.ndim != 2 or embeddings.numel() == 0:
        raise ValueError("injected positional embeddings must be a nonempty matrix")
    if embeddings.dtype != torch.float32:
        raise ValueError("injected positional embeddings must use float32")
    if not torch.isfinite(embeddings).all().item():
        raise ValueError("injected positional embeddings contain nonfinite values")

    transformer_class = _per_feature_transformer_class()
    original_add_embeddings = transformer_class.add_embeddings
    trace = {"context_calls": 0, "reuse_calls": 0, "cached_tensor_reused": False,
             "injected_tensor_sha256": _tensor_sha256(embeddings)}

    def add_embeddings(self, x, y, num_features, seq_len, cache_embeddings=False,
                       use_cached_embeddings=False):
        if use_cached_embeddings:
            cached = getattr(self, "cached_embeddings", None)
            if cached is None:
                raise RuntimeError("query requested positional cache before context injection")
            if cached is not embeddings:
                raise RuntimeError("query did not reuse the injected positional tensor")
            if (cached.shape != (x.shape[2], x.shape[3])
                    or cached.dtype != x.dtype or cached.device != x.device):
                raise ValueError("cached positional embeddings are incompatible with query input")
            trace["reuse_calls"] += 1
            trace["cached_tensor_reused"] = True
            return original_add_embeddings(self, x, y, num_features, seq_len,
                                           cache_embeddings=cache_embeddings,
                                           use_cached_embeddings=use_cached_embeddings)

        if not cache_embeddings:
            raise RuntimeError("positional injection must happen during context cache construction")
        if getattr(self, "feature_positional_embedding", None) != "subspace":
            raise ValueError("positional injection requires subspace embedding mode")
        if x.ndim != 4 or embeddings.shape != (x.shape[2], x.shape[3]):
            raise ValueError("injected positional embeddings shape does not match context input")
        if embeddings.dtype != x.dtype or embeddings.device != x.device:
            raise ValueError("injected positional embeddings dtype/device do not match context input")
        x += embeddings[None, None]
        self.cached_embeddings = embeddings
        trace["context_calls"] += 1
        return x, y

    transformer_class.add_embeddings = add_embeddings
    try:
        yield trace
    finally:
        transformer_class.add_embeddings = original_add_embeddings


def _per_feature_transformer_class():
    from tabpfn.model.transformer import PerFeatureTransformer

    return PerFeatureTransformer


def _execution_state(model):
    named_modules = getattr(model, "named_modules", None)
    modules = list(named_modules()) if callable(named_modules) else [("", model)]
    eval_modules = 0
    layer_stacks = []
    attention_dropout = []
    for name, module in modules:
        if hasattr(module, "training"):
            if module.training:
                raise ValueError(f"fitted model module {name!r} is not in eval mode")
            eval_modules += 1
        if hasattr(module, "num_layers") and hasattr(module, "min_num_layers_layer_dropout"):
            num_layers = int(module.num_layers)
            minimum = module.min_num_layers_layer_dropout
            if minimum != num_layers:
                raise ValueError(f"fitted layer stack {name!r} can randomly omit layers")
            layer_stacks.append({"name": name, "num_layers": num_layers,
                                 "min_num_layers_layer_dropout": int(minimum)})
        if hasattr(module, "dropout_p"):
            probability = module.dropout_p
            if probability not in (None, 0, 0.0):
                raise ValueError(f"fitted attention module {name!r} has nonzero dropout")
            attention_dropout.append({"name": name, "probability": probability})
    if not eval_modules:
        raise ValueError("fitted transformer exposes no module eval state")
    if not layer_stacks:
        raise ValueError("fitted transformer exposes no layer-depth state")
    if not attention_dropout:
        raise ValueError("fitted transformer exposes no attention-dropout state")
    return {"module_eval": True, "eval_module_count": eval_modules,
            "layer_stacks": layer_stacks, "attention_dropout": attention_dropout}


def _tensor_sha256(tensor):
    import torch

    tensor = tensor.detach().contiguous().cpu()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(raw)
    return digest.hexdigest()


def _tensor_identity(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "device": str(tensor.device), "sha256": _tensor_sha256(tensor)}


def _prepare_diagnostic():
    """Load and checksum the approved CPU feasibility baseline before any new fit."""
    import r1_2007_replay as replay

    inputs = replay._load_r1_inputs()
    baseline_runtime = replay._require_replay_runtime(inputs)
    baseline_fingerprint = replay._execution_fingerprint(inputs, baseline_runtime)
    baseline_path = (Path.cwd() / "stats/concept_agreement_study/extensions"
                     / "r1_2007_replay/feasibility_fix3")
    baseline = replay._load_feasibility(baseline_path, baseline_fingerprint)
    inputs["baseline_runtime"] = baseline_runtime
    inputs["baseline_evidence"] = {
        "completion_sha256": replay.file_sha256(baseline_path / "completion.json"),
        "run_sha256": replay.file_sha256(baseline_path / "run.json"),
        "probe_sha256": replay.file_sha256(baseline_path / "recovered_actual_probes.npz"),
        "status": json.loads((baseline_path / "completion.json").read_text())["status"],
    }
    return inputs, baseline_fingerprint, baseline


def _capture_device_arm(inputs, arm, injected_embeddings=None):
    """Fit one fresh recovered-source model and capture its approved probe schedule."""
    from time import monotonic
    import gc
    import r1_2007_replay as replay
    import torch

    if arm not in ("gpu_native", "cpu_native", "cpu_gpu_position"):
        raise ValueError(f"unknown diagnostic arm: {arm}")
    device = "cuda" if arm == "gpu_native" else "cpu"
    pilot = inputs["recovered_pilot"]
    population, prepared = inputs["population"], inputs["prepared"]
    columns = pilot._feature_columns(population["X"].shape[1])
    context_rows = prepared["context_rows"]
    context = replay._frame_for_rows(pilot, inputs, context_rows)
    context["y"] = population["y"][context_rows]
    context_x = np.ascontiguousarray(context[columns].to_numpy(dtype=np.float32))
    context_y = np.ascontiguousarray(context["y"].to_numpy(dtype=np.float32))
    clinical_identity = hashlib.sha256(
        str(context_x.shape).encode() + context_x.tobytes()
        + str(context_y.shape).encode() + context_y.tobytes()).hexdigest()

    if arm == "gpu_native":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for the native GPU diagnostic arm")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
    replay._seed_replay_runtime(42)
    precision = pilot.configure_predictor_precision(device)
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("precision helper did not disable TF32")

    def fit():
        return pilot.fit_frozen_predictor(context, columns, device, seed=42)

    started = monotonic()
    injection_trace = None
    if arm == "cpu_gpu_position":
        if injected_embeddings is None:
            raise ValueError("controlled CPU arm requires the native GPU positional tensor")
        with scoped_cached_embedding_injection(injected_embeddings) as injection_trace:
            system = fit()
    else:
        system = fit()
    if (system.get("reference_year") != 2007
            or system.get("checkpoint_sha256") != replay.CHECKPOINT_SHA256
            or system.get("settings", {}).get("seed") != 42
            or system.get("settings", {}).get("domain_policy") != "calendar_year_minus_reference_year"
            or system.get("context_rows") != prepared["context_identity"]):
        raise RuntimeError("fitted diagnostic arm failed reference, checkpoint, seed, or context identity")
    if system.get("settings", {}).get("precision", precision).get("amp", "off") != "off":
        raise RuntimeError("diagnostic arm did not retain AMP-off precision")
    before = observe_fitted_state(pilot, system)
    if arm == "cpu_gpu_position" and before["positional_state"]["sha256"] != _tensor_sha256(injected_embeddings):
        raise RuntimeError("controlled CPU arm did not retain the injected GPU positional tensor")

    capture = pilot.capture_frozen_predictions
    gate_rows = np.asarray(inputs["gate_rows"], dtype=np.int64)
    gate = replay._frame_for_rows(pilot, inputs, gate_rows)
    base = capture(system, gate, device)
    repeated = capture(system, gate, device)
    reversed_batch = capture(system, gate.iloc[::-1], device)
    singles = [capture(system, gate.iloc[index:index + 1], device) for index in range(len(gate))]
    base_pair = replay._checked_capture(base, len(gate), "diagnostic reference gate")
    repeated_pair = replay._checked_capture(repeated, len(gate), "diagnostic repeated gate")
    reverse_pair = replay._checked_capture(reversed_batch, len(gate), "diagnostic reversed gate")
    singleton_pair = [replay._checked_capture(item, 1, "diagnostic singleton gate") for item in singles]
    for index in range(2):
        expected = base_pair[index]
        observed = (repeated_pair[index], reverse_pair[index][::-1],
                    np.concatenate([item[index] for item in singleton_pair]))
        if any(not np.allclose(value, expected, rtol=1e-4, atol=1e-5) for value in observed):
            raise RuntimeError("diagnostic reference gate failed its original repeat/order/singleton checks")
    del base, repeated, reversed_batch, singles, gate

    if arm == "gpu_native":
        query_rows = np.asarray(prepared["query_rows"], dtype=np.int64)
        if len(query_rows) != 4096:
            raise RuntimeError("native GPU query batch is not the original 4096 rows")
        full = capture(system, replay._frame_for_rows(pilot, inputs, query_rows), device)
        full_probabilities, full_embeddings = replay._checked_capture(full, 4096, "native GPU 4096 batch")
        positions = [prepared["probes"][year]["batch_position"] for year in range(2007, 2015)]
        full_probe_probabilities = full_probabilities[positions]
        full_probe_embeddings = full_embeddings[positions]
    else:
        full_probe_probabilities = full_probe_embeddings = None

    years = np.arange(2007, 2015, dtype=np.int64)
    probe_rows = [prepared["probes"][int(year)]["population_row"] for year in years]
    if arm == "cpu_gpu_position":
        with scoped_cached_embedding_injection(injected_embeddings) as query_injection_trace:
            probe = capture(system, replay._frame_for_rows(pilot, inputs, probe_rows), device)
    else:
        query_injection_trace = None
        probe = capture(system, replay._frame_for_rows(pilot, inputs, probe_rows), device)
    probabilities, embeddings = replay._checked_capture(probe, len(years), f"{arm} matched eight probes")
    after = observe_fitted_state(pilot, system)
    if (after["learned_parameter_identity"] != before["learned_parameter_identity"]
            or after["positional_state"]["sha256"] != before["positional_state"]["sha256"]):
        raise RuntimeError("learned parameters or positional cache changed during diagnostic captures")
    buffer_before = [(item["name"], item["sha256"]) for item in before["registered_buffers"]]
    buffer_after = [(item["name"], item["sha256"]) for item in after["registered_buffers"]]
    if buffer_after != buffer_before:
        raise RuntimeError("registered fitted context buffers changed during diagnostic captures")
    position_tensor = None
    if arm == "gpu_native":
        position_tensor = system["classifier"].cached_models_[0].cached_embeddings.detach().cpu().clone()
    if arm == "cpu_gpu_position" and injection_trace is not None:
        if (not query_injection_trace["cached_tensor_reused"]
                or query_injection_trace["reuse_calls"] < 1):
            raise RuntimeError("controlled CPU query did not reuse its injected positional tensor")
    gpu_memory = None
    if device == "cuda":
        torch.cuda.synchronize()
        device_properties = torch.cuda.get_device_properties(0)
        gpu_memory = {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                      "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                      "device_name": str(device_properties.name),
                      "total_memory_bytes": int(device_properties.total_memory)}

    result = {
        "arm": arm, "device": device, "years": years,
        "probabilities": probabilities.astype(np.float32, copy=False),
        "embeddings": embeddings.astype(np.float32, copy=False),
        "full_probabilities": (None if full_probe_probabilities is None
                               else full_probe_probabilities.astype(np.float32, copy=False)),
        "full_embeddings": (None if full_probe_embeddings is None
                            else full_probe_embeddings.astype(np.float32, copy=False)),
        "positional_tensor": position_tensor,
        "before_state": before, "after_state": after,
        "learned_parameter_identity": before["learned_parameter_identity"],
        "predictor_identity": replay._state_report(pilot, system),
        "clinical_input_identity": clinical_identity,
        "context_identity": system["context_identity"],
        "context_rows_identity": system["context_rows"],
        "checkpoint_sha256": system["checkpoint_sha256"],
        "settings": system["settings"], "precision": precision,
        "gate_trace": ["base", "repeat", "reverse", "singletons"],
        "injection_trace": (None if injection_trace is None else {
            "context_calls": injection_trace["context_calls"],
            "reuse_calls": query_injection_trace["reuse_calls"],
            "cached_tensor_reused": query_injection_trace["cached_tensor_reused"],
            "injected_tensor_sha256": injection_trace["injected_tensor_sha256"]}),
        "elapsed_seconds": monotonic() - started,
        "gpu_memory": gpu_memory,
    }
    del system
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return result


def _condition_identity(condition):
    settings = condition["settings"]
    setting_keys = ("model_type", "seed", "fit_at_predict_time", "fp16_inference",
                    "sklearn_compatible_precision", "domain_policy")
    fields = condition["predictor_identity"].get("field_digests", {})
    comparable_fields = _IDENTITY_FIELDS
    missing = [key for key in comparable_fields if key not in fields]
    if missing:
        raise ValueError(f"arm identity report lacks required fitted fields: {missing}")
    return (condition["learned_parameter_identity"], condition["clinical_input_identity"],
            condition["context_rows_identity"],
            condition["checkpoint_sha256"], tuple((key, settings.get(key)) for key in setting_keys),
            tuple((key, fields[key]) for key in comparable_fields))


def _save_arm(output, name, condition, targets):
    path = output / f"{name}.npz"
    content = {key: condition[key] for key in ("years", "probabilities", "embeddings")}
    content["target_probabilities"] = np.asarray(targets["probabilities"], dtype=np.float32)
    content["target_embeddings"] = np.asarray(targets["embeddings"], dtype=np.float32)
    if condition.get("full_probabilities") is not None:
        content["full_probabilities"] = condition["full_probabilities"]
        content["full_embeddings"] = condition["full_embeddings"]
    if condition.get("positional_tensor") is not None:
        content["positional_tensor"] = condition["positional_tensor"]
    np.savez_compressed(path, **content)
    return replay_file_sha256(path)


def replay_file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_arm(condition, output_hash):
    return {key: value for key, value in {
        "device": condition.get("device"), "checkpoint_sha256": condition.get("checkpoint_sha256"),
        "clinical_input_identity": condition.get("clinical_input_identity"),
        "context_identity": condition.get("context_identity"),
        "context_rows_identity": condition.get("context_rows_identity"),
        "learned_parameter_identity": condition.get("learned_parameter_identity"),
        "fitted_field_digest_count": len(condition.get("predictor_identity", {}).get("field_digests", {})),
        "settings": condition.get("settings"), "precision": condition.get("precision"),
        "before_state": condition.get("before_state"), "after_state": condition.get("after_state"),
        "gate_trace": condition.get("gate_trace"), "injection_trace": condition.get("injection_trace"),
        "elapsed_seconds": condition.get("elapsed_seconds"), "private_output_sha256": output_hash,
        "gpu_memory": condition.get("gpu_memory"),
    }.items() if value is not None}


def _validate_saved_baseline(baseline):
    years = np.asarray(baseline.get("years"))
    probabilities = np.asarray(baseline.get("probabilities"))
    embeddings = np.asarray(baseline.get("embeddings"))
    target_probabilities = np.asarray(baseline.get("target_probabilities"))
    target_embeddings = np.asarray(baseline.get("target_embeddings"))
    expected_years = np.arange(2007, 2015, dtype=np.int64)
    if (years.shape != expected_years.shape or years.dtype.kind not in "iu"
            or not np.array_equal(years, expected_years)
            or probabilities.shape != (8, 2) or probabilities.dtype != np.float32
            or embeddings.ndim != 2 or embeddings.shape[0] != 8 or embeddings.dtype != np.float32
            or target_probabilities.shape not in ((8,), (8, 2))
            or target_probabilities.dtype != np.float32
            or target_embeddings.shape != embeddings.shape or target_embeddings.dtype != np.float32
            or not all(np.isfinite(array).all() for array in
                       (probabilities, embeddings, target_probabilities, target_embeddings))
            or not bool(np.asarray(baseline.get("target_embeddings_available", False)).item())):
        raise ValueError("saved feasibility probe identities or array contracts are invalid")


def _array_metrics(probabilities, embeddings, target_probabilities, target_embeddings):
    probabilities = np.asarray(probabilities)
    embeddings = np.asarray(embeddings)
    target_probabilities = np.asarray(target_probabilities)
    target_embeddings = np.asarray(target_embeddings)
    if (probabilities.ndim != 2 or probabilities.shape[1] != 2
            or target_probabilities.shape not in ((len(probabilities),), probabilities.shape)
            or embeddings.ndim != 2 or target_embeddings.shape != embeddings.shape
            or len(embeddings) != len(probabilities)
            or not all(array.dtype == np.float32 for array in
                       (probabilities, embeddings, target_probabilities, target_embeddings))
            or not all(np.isfinite(array).all() for array in
                       (probabilities, embeddings, target_probabilities, target_embeddings))):
        raise ValueError("invalid diagnostic comparison axes, dtype, or finite values")
    target_class1 = target_probabilities[:, 1] if target_probabilities.ndim == 2 else target_probabilities
    probability_difference = probabilities[:, 1].astype(np.float64) - target_class1.astype(np.float64)
    embedding_difference = embeddings.astype(np.float64) - target_embeddings.astype(np.float64)
    def relative_l2(diff, reference):
        denominator = float(np.linalg.norm(reference.astype(np.float64).reshape(-1)))
        return float(np.linalg.norm(diff.reshape(-1)) / denominator) if denominator else None
    return {
        "count": int(len(probabilities)),
        "class1_probability_max_abs_error": float(np.max(np.abs(probability_difference))),
        "class1_probability_relative_l2": relative_l2(probability_difference, target_class1),
        "class1_probability_within_combined_tolerance": bool(np.allclose(
            probabilities[:, 1], target_class1, rtol=1e-4, atol=1e-5)),
        "raw_embedding_max_abs_error": float(np.max(np.abs(embedding_difference))),
        "raw_embedding_relative_l2": relative_l2(embedding_difference, target_embeddings),
        "raw_embedding_within_combined_tolerance": bool(np.allclose(
            embeddings, target_embeddings, rtol=1e-4, atol=1e-5)),
        "per_probe": [{
            "year": int(2007 + index),
            "class1_probability_abs_error": float(abs(float(probabilities[index, 1]) - float(target_class1[index]))),
            "class1_probability_within_combined_tolerance": bool(np.isclose(
                probabilities[index, 1], target_class1[index], rtol=1e-4, atol=1e-5)),
            "raw_embedding_max_abs_error": float(np.max(np.abs(embeddings[index] - target_embeddings[index]))),
            "raw_embedding_within_combined_tolerance": bool(np.allclose(
                embeddings[index], target_embeddings[index], rtol=1e-4, atol=1e-5)),
        } for index in range(len(probabilities))],
    }


def run_diagnostic(output):
    """Run the matched CUDA/CPU device-state diagnostic; caller must authorize launch."""
    import datetime
    import r1_2007_replay as replay
    from time import monotonic

    output = Path(output)
    if output.exists():
        if ({path.name for path in output.iterdir()} != {"run"}
                or not (output / "run").is_dir()):
            raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    else:
        output.mkdir(parents=True)
    inputs = fingerprint = baseline = None
    arms, output_hashes = {}, {}
    started = monotonic()
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record = {"schema": "r1-2007-device-state-diagnostic-v1", "status": "running",
              "arms": {}, "comparisons": {}, "identity_comparisons": {}, "baseline_fingerprint": None,
              "source_sha256": {}, "dependency_identity": None,
              "module_sha256": replay_file_sha256(__file__), "started_utc": started_utc}
    _write_json(output / "completion.json", {"status": "running"})
    stage = "baseline validation"
    try:
        inputs, fingerprint, baseline = _prepare_diagnostic()
        record["baseline_fingerprint"] = fingerprint
        record["source_sha256"] = inputs.get("source_hashes", {})
        record["dependency_identity"] = inputs.get("dependency_identity")
        record["input_fingerprints"] = inputs.get("fingerprint_facts", {})
        record["baseline_evidence"] = inputs.get("baseline_evidence", {})
        record["baseline_runtime"] = inputs.get("baseline_runtime", {})
        record["runtime"] = {"device_arms": ["cuda:0", "cpu", "cpu"], "threads": 1,
                              "precision": "fp32", "seed": 42, "batch_rows": 4096,
                              "probe_rows": 8}
        _validate_saved_baseline(baseline)
        archive_targets = {"probabilities": baseline["target_probabilities"],
                           "embeddings": baseline["target_embeddings"]}
        saved_cpu = {"probabilities": baseline["probabilities"],
                     "embeddings": baseline["embeddings"]}
        if not bool(np.asarray(baseline.get("target_embeddings_available", True)).item()):
            raise ValueError("saved CPU feasibility baseline lacks verified raw embedding targets")
        for arm_name in ("gpu_native", "cpu_native", "cpu_gpu_position"):
            stage = {"gpu_native": "native CUDA capture",
                     "cpu_native": "native CPU capture",
                     "cpu_gpu_position": "controlled CPU capture"}[arm_name]
            injected = arms["gpu_native"].get("positional_tensor") if arm_name == "cpu_gpu_position" else None
            condition = _capture_device_arm(inputs, arm_name, injected)
            arms[arm_name] = condition
            output_hashes[arm_name] = _save_arm(output, arm_name, condition, archive_targets)
            record["arms"][arm_name] = _public_arm(condition, output_hashes[arm_name])
            record["partial_outputs"] = dict(output_hashes)
            if arm_name != "gpu_native" and _condition_identity(condition) != _condition_identity(arms["gpu_native"]):
                record["identity_comparisons"][arm_name] = {
                    "status": "mismatch",
                    "qualification": "fitted input, predictor configuration or learned parameter identity differed"}
                raise ValueError("arm identity mismatch across device-state conditions")
            if arm_name != "gpu_native":
                record["identity_comparisons"][arm_name] = {"status": "matched"}
            if arm_name == "cpu_gpu_position" and condition["before_state"]["positional_state"]["sha256"] != _tensor_sha256(injected):
                raise ValueError("controlled arm identity mismatch for injected positional state")

        stage = "comparison and report generation"
        comparisons = {}
        comparisons["gpu_4096_vs_archive"] = _array_metrics(
            arms["gpu_native"]["full_probabilities"], arms["gpu_native"]["full_embeddings"],
            archive_targets["probabilities"], archive_targets["embeddings"])
        comparisons["gpu_4096_vs_saved_cpu_4096"] = _array_metrics(
            arms["gpu_native"]["full_probabilities"], arms["gpu_native"]["full_embeddings"],
            saved_cpu["probabilities"], saved_cpu["embeddings"])
        comparisons["gpu_8_vs_archive"] = _array_metrics(
            arms["gpu_native"]["probabilities"], arms["gpu_native"]["embeddings"],
            archive_targets["probabilities"], archive_targets["embeddings"])
        comparisons["gpu_8_vs_saved_cpu_4096"] = _array_metrics(
            arms["gpu_native"]["probabilities"], arms["gpu_native"]["embeddings"],
            saved_cpu["probabilities"], saved_cpu["embeddings"])
        comparisons["gpu_8_vs_gpu_4096"] = _array_metrics(
            arms["gpu_native"]["probabilities"], arms["gpu_native"]["embeddings"],
            arms["gpu_native"]["full_probabilities"], arms["gpu_native"]["full_embeddings"])
        for name in ("cpu_native", "cpu_gpu_position"):
            comparisons[f"{name}_8_vs_archive"] = _array_metrics(
                arms[name]["probabilities"], arms[name]["embeddings"],
                archive_targets["probabilities"], archive_targets["embeddings"])
            comparisons[f"{name}_8_vs_saved_cpu_4096"] = _array_metrics(
                arms[name]["probabilities"], arms[name]["embeddings"],
                saved_cpu["probabilities"], saved_cpu["embeddings"])
        comparisons["cpu_gpu_position_8_vs_cpu_native_8"] = _array_metrics(
            arms["cpu_gpu_position"]["probabilities"], arms["cpu_gpu_position"]["embeddings"],
            arms["cpu_native"]["probabilities"], arms["cpu_native"]["embeddings"])
        comparisons["cpu_gpu_position_8_vs_gpu_8"] = _array_metrics(
            arms["cpu_gpu_position"]["probabilities"], arms["cpu_gpu_position"]["embeddings"],
            arms["gpu_native"]["probabilities"], arms["gpu_native"]["embeddings"])
        record.update({"status": "completed", "comparisons": comparisons,
                       "elapsed_seconds": monotonic() - started,
                       "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "tolerances": {"rtol": 1e-4, "atol": 1e-5,
                                      "criterion": "combined elementwise tolerance"}})
        launcher_path = output / "run" / "run.json"
        if launcher_path.is_file():
            launcher = json.loads(launcher_path.read_text())
            dirty_files = launcher.get("uncommitted_code_sha256", {})
            record["launcher_identity"] = {
                "commit": launcher.get("commit"),
                "dirty_file_count": len(dirty_files),
                "dirty_files_sha256": hashlib.sha256(json.dumps(
                    dirty_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            }
        import resource
        record["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _write_outputs(output, record)
        summaries = {name: replay_file_sha256(output / name)
                     for name in ("report.json", "report.csv", "report.md")}
        _write_json(output / "completion.json", {"status": "completed", "module_sha256": record["module_sha256"],
                                                  "outputs": output_hashes,
                                                  "summary_outputs": summaries})
        return record
    except BaseException as error:
        record.update({"status": "failed", "elapsed_seconds": monotonic() - started,
                       "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "error": _safe_error(error, stage),
                       "partial_outputs": dict(output_hashes)})
        import resource
        record["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _write_outputs(output, record)
        summaries = {name: replay_file_sha256(output / name)
                     for name in ("report.json", "report.csv", "report.md")}
        _write_json(output / "completion.json", {"status": "failed",
                                                  "module_sha256": record["module_sha256"],
                                                  "partial_outputs": dict(output_hashes),
                                                  "summary_outputs": summaries,
                                                  "error": record["error"]})
        raise


def _safe_error(error, stage):
    """Return only fixed public categories; exception text may contain private row data."""
    allowed_stages = {
        "baseline validation", "native CUDA capture", "native CPU capture",
        "controlled CPU capture", "comparison and report generation",
    }
    safe_stage = stage if stage in allowed_stages else "diagnostic execution"
    return {"type": type(error).__name__, "message": f"diagnostic failed during {safe_stage}"}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_outputs(output, record):
    _write_json(output / "report.json", record)
    rows = ["record_type,comparison,year,count,class1_probability_max_abs_error,class1_probability_relative_l2,"
            "class1_probability_within_combined_tolerance,raw_embedding_max_abs_error,"
            "raw_embedding_relative_l2,raw_embedding_within_combined_tolerance"]
    for name, metrics in record.get("comparisons", {}).items():
        rows.append(",".join(["aggregate", name, ""] + [str(metrics.get(key, "")) for key in (
            "count", "class1_probability_max_abs_error", "class1_probability_relative_l2",
            "class1_probability_within_combined_tolerance", "raw_embedding_max_abs_error",
            "raw_embedding_relative_l2", "raw_embedding_within_combined_tolerance")]))
        for probe in metrics.get("per_probe", []):
            rows.append(",".join(["probe", name, str(probe["year"]), "1",
                str(probe["class1_probability_abs_error"]), "",
                str(probe["class1_probability_within_combined_tolerance"]),
                str(probe["raw_embedding_max_abs_error"]), "",
                str(probe["raw_embedding_within_combined_tolerance"])]))
    (output / "report.csv").write_text("\n".join(rows) + "\n")
    lines = ["# r1 2007 device-state diagnostic", "", f"Status: {record.get('status')}", "",
             "Probe outputs and positional tensors remain in private Fiji NPZ files.", "",
             f"Diagnostic module SHA256: `{record.get('module_sha256')}`", "",
             "| Comparison | N | Max class-1 error | Relative L2 class-1 | Class-1 tolerance | Max embedding error | Relative L2 embedding | Embedding tolerance |",
             "|---|---:|---:|---:|---|---:|---:|---|"]
    for name, metrics in record.get("comparisons", {}).items():
        lines.append("| " + " | ".join(str(value) for value in (
            name, metrics.get("count", ""), metrics.get("class1_probability_max_abs_error", ""),
            metrics.get("class1_probability_relative_l2", ""),
            metrics.get("class1_probability_within_combined_tolerance", ""),
            metrics.get("raw_embedding_max_abs_error", ""), metrics.get("raw_embedding_relative_l2", ""),
            metrics.get("raw_embedding_within_combined_tolerance", ""))) + " |")
    if record.get("error"):
        lines.extend(["", f"Failure: `{record['error'].get('type')}` — {record['error'].get('message')}"])
    lines.extend(["", "## Per-probe errors", "",
                  "| Comparison | Year | Class-1 absolute error | Class-1 tolerance | Raw embedding maximum error | Embedding tolerance |",
                  "|---|---:|---:|---|---:|---|"])
    for name, metrics in record.get("comparisons", {}).items():
        for probe in metrics.get("per_probe", []):
            lines.append("| " + " | ".join(str(value) for value in (
                name, probe["year"], probe["class1_probability_abs_error"],
                probe["class1_probability_within_combined_tolerance"],
                probe["raw_embedding_max_abs_error"],
                probe["raw_embedding_within_combined_tolerance"])) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
