import hashlib
import numpy as np
import pytest
import torch

import r1_2007_device_diagnostic as diagnostic
import r1_2007_replay as replay


class FakePerFeatureTransformer(torch.nn.Module):
    def __init__(self, cached_embeddings=None):
        super().__init__()
        self.feature_positional_embedding = "subspace"
        self.cached_embeddings = cached_embeddings
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
        self.register_buffer("running_state", torch.tensor([3.0], dtype=torch.float32))
        self.num_layers = 1
        self.min_num_layers_layer_dropout = 1
        self.dropout_p = 0.0
        self.eval()

    def add_embeddings(self, x, y, num_features, seq_len, cache_embeddings=False,
                       use_cached_embeddings=False):
        if use_cached_embeddings and self.cached_embeddings is not None:
            x = x + self.cached_embeddings[None, None]
            return x, y
        generated = torch.zeros((x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
        x = x + generated[None, None]
        self.cached_embeddings = generated if cache_embeddings else None
        return x, y


class FakePilot:
    @staticmethod
    def predictor_identity_report(classifier):
        return {"identity": "fake-fitted-identity", "field_digests": {"context": "same"}}

    @staticmethod
    def _identity_hash(value):
        digest = hashlib.sha256()
        for name, tensor in sorted(value.items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def _fake_system(cached_embeddings):
    model = FakePerFeatureTransformer(cached_embeddings)
    return {"classifier": type("Classifier", (), {
        "model": model,
        "cached_models_": [model],
        "pilot_checkpoint_sha256_": replay.CHECKPOINT_SHA256,
    })()}


def test_observe_fitted_state_separates_positional_cache_from_learned_parameters():
    cached = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    result = diagnostic.observe_fitted_state(FakePilot, _fake_system(cached))

    positional = result["positional_state"]
    assert positional["mode"] == "subspace"
    assert positional["shape"] == [2, 3]
    assert positional["dtype"] == "torch.float32"
    assert positional["sha256"]
    assert positional["sha256"] != result["learned_parameter_identity"]
    assert result["registered_buffers"][0]["name"] == "running_state"
    assert "cached_embeddings" not in result["registered_buffers"]
    assert result["execution_state"]["module_eval"] is True
    assert result["execution_state"]["layer_stacks"][0]["min_num_layers_layer_dropout"] == 1
    assert result["execution_state"]["attention_dropout"][0]["probability"] == 0.0
    assert not any(isinstance(value, torch.Tensor) for value in result.values())


@pytest.mark.parametrize("cached, message", [
    (None, "cached positional embeddings are missing"),
    (torch.tensor([[float("nan")]], dtype=torch.float32), "nonfinite positional embeddings"),
    (torch.ones(1, dtype=torch.float32), "positional embeddings must be a nonempty matrix"),
])
def test_observe_rejects_missing_nonfinite_or_incompatible_positional_cache(cached, message):
    with pytest.raises(ValueError, match=message):
        diagnostic.observe_fitted_state(FakePilot, _fake_system(cached))


def test_scoped_injection_caches_before_context_and_reuses_for_query(monkeypatch):
    monkeypatch.setattr(diagnostic, "_per_feature_transformer_class",
                        lambda: FakePerFeatureTransformer)
    injected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    original = FakePerFeatureTransformer.add_embeddings
    model = FakePerFeatureTransformer()
    context_x = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

    with diagnostic.scoped_cached_embedding_injection(injected) as trace:
        assert FakePerFeatureTransformer.add_embeddings is not original
        transformed, _ = model.add_embeddings(context_x, None, 2, 2, cache_embeddings=True)
        assert model.cached_embeddings is injected
        torch.testing.assert_close(transformed[0, 0], injected)
        query_x = torch.zeros_like(context_x)
        reused, _ = model.add_embeddings(query_x, None, 2, 2, use_cached_embeddings=True)
        torch.testing.assert_close(reused[0, 0], injected)
        assert trace["context_calls"] == 1
        assert trace["reuse_calls"] == 1
        assert trace["cached_tensor_reused"] is True

    assert FakePerFeatureTransformer.add_embeddings is original


def test_scoped_injection_rejects_bad_shape_and_restores_method(monkeypatch):
    monkeypatch.setattr(diagnostic, "_per_feature_transformer_class",
                        lambda: FakePerFeatureTransformer)
    original = FakePerFeatureTransformer.add_embeddings
    model = FakePerFeatureTransformer()
    x = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="injected positional embeddings shape"):
        with diagnostic.scoped_cached_embedding_injection(torch.ones((1, 3), dtype=torch.float32)):
            model.add_embeddings(x, None, 2, 2, cache_embeddings=True)

    assert FakePerFeatureTransformer.add_embeddings is original


def test_scoped_injection_rejects_non_subspace_mode_and_restores_method(monkeypatch):
    monkeypatch.setattr(diagnostic, "_per_feature_transformer_class",
                        lambda: FakePerFeatureTransformer)
    original = FakePerFeatureTransformer.add_embeddings
    model = FakePerFeatureTransformer()
    model.feature_positional_embedding = "normal_rand_vec"
    x = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="requires subspace embedding mode"):
        with diagnostic.scoped_cached_embedding_injection(torch.ones((2, 3), dtype=torch.float32)):
            model.add_embeddings(x, None, 2, 2, cache_embeddings=True)

    assert FakePerFeatureTransformer.add_embeddings is original


def test_recovered_cpu_fit_caches_injected_tensor_before_query_reuses_it():
    from pathlib import Path
    import gc
    import pandas as pd

    recovered_path = (Path(__file__).resolve().parents[1] / "stats/concept_agreement_study/extensions"
                      / "r1_2007_replay/recovered/concept_subgroup_drift_pilot.py")
    pilot = replay.load_isolated_source(
        recovered_path, replay.RECOVERED_SOURCE_SHA256, module_name="_test_recovered_positional_injection")
    generator = np.random.default_rng(42)
    feature_names = ["f0", "f1", "f2"]
    context = pd.DataFrame(generator.normal(size=(32, 3)).astype(np.float32), columns=feature_names)
    context["patient_id"] = [f"synthetic-context-{index}" for index in range(32)]
    context["year"] = 2007
    context["y"] = np.arange(32, dtype=np.int64) % 2
    query = context.iloc[:2].drop(columns="y").copy()
    query["patient_id"] = ["synthetic-query-0", "synthetic-query-1"]
    query["year"] = 2008

    baseline = pilot.fit_frozen_predictor(context, feature_names, "cpu", seed=42)
    baseline_state = diagnostic.observe_fitted_state(pilot, baseline)
    generated = baseline["classifier"].cached_models_[0].cached_embeddings
    injected = generated.detach().cpu().clone()
    injected[0, 0] += 0.25
    del baseline
    gc.collect()

    with diagnostic.scoped_cached_embedding_injection(injected) as trace:
        system = pilot.fit_frozen_predictor(context, feature_names, "cpu", seed=42)
        fitted_state = diagnostic.observe_fitted_state(pilot, system)
        prediction = pilot.capture_frozen_predictions(system, query, "cpu")
    post_capture_state = diagnostic.observe_fitted_state(pilot, system)

    assert trace["context_calls"] >= 1
    assert trace["reuse_calls"] >= 1
    assert trace["cached_tensor_reused"] is True
    assert fitted_state["positional_state"]["sha256"] == diagnostic._tensor_sha256(injected)
    assert fitted_state["positional_state"]["sha256"] != baseline_state["positional_state"]["sha256"]
    assert fitted_state["learned_parameter_identity"] == baseline_state["learned_parameter_identity"]
    assert post_capture_state["positional_state"]["sha256"] == fitted_state["positional_state"]["sha256"]
    assert np.asarray(prediction["probabilities"].detach().cpu()).shape == (2, 2)


def _orchestration_condition(name, identity="same", position=None):
    fields = {key: f"{key}-same" for key in _REQUIRED_IDENTITY_FIELDS}
    if identity != "same":
        fields["X_"] = f"X_-{identity}"
    return {
        "name": name,
        "years": np.arange(2007, 2015, dtype=np.int64),
        "probabilities": np.tile(np.asarray([[0.4, 0.6]], dtype=np.float32), (8, 1)),
        "embeddings": np.ones((8, 192), dtype=np.float32),
        "positional_tensor": position,
        "before_state": {"positional_state": {"sha256": diagnostic._tensor_sha256(position)}}
                        if position is not None else {"positional_state": {"sha256": "same"}},
        "context_identity": f"device-specific-full-context-{name}",
        "predictor_identity": {"field_digests": fields},
        "learned_parameter_identity": "same-learned-parameters",
        "clinical_input_identity": "same-clinical-input",
        "context_rows_identity": "same-context-rows",
        "checkpoint_sha256": "same-checkpoint",
        "settings": {"model_type": "single_fast", "seed": 42,
                     "fit_at_predict_time": False, "fp16_inference": False,
                     "sklearn_compatible_precision": False,
                     "domain_policy": "calendar_year_minus_reference_year"},
        "gate_trace": ["gate", "repeat", "reverse", "singletons"],
    }


_REQUIRED_IDENTITY_FIELDS = (
    "X_", "y_", "_categorical_features", "additional_x_", "categorical_features",
    "classes_", "c_processed_", "cached_preprocessors_", "cached_shuffles_",
    "data_mean_", "data_std_", "average_logits", "softmax_temperature",
    "ensemble_configurations_", "transformer_predict_kwargs", "preprocess_transforms",
    "feature_shift_decoder", "normalize_with_test", "normalize_to_ranking",
    "normalize_to_clip", "remove_outliers", "fit_at_predict_time", "fp16_inference",
    "sklearn_compatible_precision", "pilot_checkpoint_sha256_", "pilot_domain_policy_",
)


def test_run_diagnostic_executes_matched_arms_in_order_and_keeps_private_tensors_private(
        tmp_path, monkeypatch):
    import json

    injected = torch.ones((2, 3), dtype=torch.float32, device="cpu")
    calls = []
    baseline = {"target_probabilities": np.full(8, 0.61, dtype=np.float32),
                "years": np.arange(2007, 2015, dtype=np.int64),
                "target_embeddings": np.ones((8, 192), dtype=np.float32),
                "probabilities": np.tile(np.asarray([[0.35, 0.65]], dtype=np.float32), (8, 1)),
                "embeddings": np.ones((8, 192), dtype=np.float32),
                "target_embeddings_available": np.asarray(True)}
    fake_inputs = {"baseline": {}, "source_hashes": {"recovered": "sha"},
                   "fingerprint_facts": {"input_files": {"source": "sha"}}}
    monkeypatch.setattr(diagnostic, "_prepare_diagnostic", lambda: (fake_inputs, "fp", baseline))

    def capture(inputs, arm, injected_embeddings=None):
        calls.append((arm, injected_embeddings is not None))
        condition = _orchestration_condition(arm, position=injected if arm == "gpu_native" else None)
        if arm == "cpu_gpu_position":
            condition["before_state"]["positional_state"]["sha256"] = diagnostic._tensor_sha256(injected_embeddings)
        if arm == "gpu_native":
            condition["full_probabilities"] = condition["probabilities"]
            condition["full_embeddings"] = condition["embeddings"]
        return condition

    monkeypatch.setattr(diagnostic, "_capture_device_arm", capture)
    result = diagnostic.run_diagnostic(tmp_path / "out")

    assert calls == [("gpu_native", False), ("cpu_native", False), ("cpu_gpu_position", True)]
    assert result["status"] == "completed"
    public = json.loads((tmp_path / "out" / "report.json").read_text())
    assert public["status"] == "completed"
    assert public["comparisons"]["gpu_8_vs_archive"]["class1_probability_max_abs_error"] == pytest.approx(0.01)
    assert public["comparisons"]["gpu_8_vs_saved_cpu_4096"]["class1_probability_max_abs_error"] == pytest.approx(0.05)
    assert len(public["comparisons"]["gpu_8_vs_archive"]["per_probe"]) == 8
    assert all("probabilities" not in arm and "embeddings" not in arm
               and "positional_tensor" not in arm for arm in public["arms"].values())
    assert not any(isinstance(value, torch.Tensor) for value in public.values())


def test_run_diagnostic_records_identity_failure_as_partial_and_does_not_continue(
        tmp_path, monkeypatch):
    import json

    calls = []
    baseline = {"target_embeddings": np.ones((8, 192), dtype=np.float32),
                "years": np.arange(2007, 2015, dtype=np.int64),
                "target_probabilities": np.full(8, 0.6, dtype=np.float32),
                "probabilities": np.tile(np.asarray([[0.4, 0.6]], dtype=np.float32), (8, 1)),
                "embeddings": np.ones((8, 192), dtype=np.float32),
                "target_embeddings_available": np.asarray(True)}
    fake_inputs = {"baseline": {}, "source_hashes": {"recovered": "sha"},
                   "fingerprint_facts": {"input_files": {"source": "sha"}}}
    monkeypatch.setattr(diagnostic, "_prepare_diagnostic", lambda: (fake_inputs, "fp", baseline))

    def capture(inputs, arm, injected_embeddings=None):
        calls.append(arm)
        condition = _orchestration_condition(arm, identity="changed" if arm == "cpu_native" else "same",
                                             position=torch.ones((2, 3)) if arm == "gpu_native" else None)
        if arm == "cpu_gpu_position":
            condition["before_state"]["positional_state"]["sha256"] = diagnostic._tensor_sha256(injected_embeddings)
        return condition

    monkeypatch.setattr(diagnostic, "_capture_device_arm", capture)
    with pytest.raises(ValueError, match="arm identity mismatch"):
        diagnostic.run_diagnostic(tmp_path / "partial")

    assert calls == ["gpu_native", "cpu_native"]
    completion = json.loads((tmp_path / "partial" / "completion.json").read_text())
    report = json.loads((tmp_path / "partial" / "report.json").read_text())
    assert completion["status"] == report["status"] == "failed"
    assert set(report["arms"]) == {"gpu_native", "cpu_native"}


def test_cross_device_identity_ignores_full_context_hash_but_requires_fitted_fields():
    gpu = _orchestration_condition("gpu_native")
    cpu = _orchestration_condition("cpu_native")
    assert gpu["context_identity"] != cpu["context_identity"]
    assert diagnostic._condition_identity(gpu) == diagnostic._condition_identity(cpu)

    cpu["predictor_identity"]["field_digests"]["X_"] = "different-transformed-input"
    assert diagnostic._condition_identity(gpu) != diagnostic._condition_identity(cpu)


def test_run_diagnostic_refuses_existing_outputs(tmp_path, monkeypatch):
    output = tmp_path / "exists"
    output.mkdir()
    (output / "report.json").write_text("{}");
    monkeypatch.setattr(diagnostic, "_prepare_diagnostic", lambda: pytest.fail("must validate output first"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnostic.run_diagnostic(output)


def test_array_metrics_uses_combined_tolerance_and_reports_relative_l2():
    reference_probabilities = np.asarray([[0.25, 0.75], [0.5, 0.5]], dtype=np.float32)
    observed_probabilities = reference_probabilities.copy()
    observed_probabilities[0, 1] += 1.0e-6
    reference_embeddings = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    observed_embeddings = reference_embeddings.copy()
    observed_embeddings[0, 0] += 1.0e-3

    metrics = diagnostic._array_metrics(observed_probabilities, observed_embeddings,
                                        reference_probabilities, reference_embeddings)

    assert metrics["count"] == 2
    assert metrics["class1_probability_within_combined_tolerance"] is True
    assert metrics["class1_probability_relative_l2"] > 0
    assert metrics["raw_embedding_within_combined_tolerance"] is False
    assert metrics["raw_embedding_relative_l2"] > 0


def test_array_metrics_accepts_archived_class1_vector_and_emits_per_year_errors():
    probabilities = np.tile(np.asarray([[0.3, 0.7]], dtype=np.float32), (8, 1))
    embeddings = np.ones((8, 192), dtype=np.float32)
    archived_class1 = np.full(8, 0.7, dtype=np.float32)
    archived_embeddings = np.ones((8, 192), dtype=np.float32)

    metrics = diagnostic._array_metrics(probabilities, embeddings,
                                        archived_class1, archived_embeddings)

    assert metrics["count"] == 8
    assert metrics["class1_probability_within_combined_tolerance"] is True
    assert len(metrics["per_probe"]) == 8
    assert [row["year"] for row in metrics["per_probe"]] == list(range(2007, 2015))


def test_run_diagnostic_rejects_invalid_saved_baseline_before_fitting_any_arm(tmp_path, monkeypatch):
    calls = []
    bad_baseline = {"years": np.asarray([2007], dtype=np.int64),
                    "probabilities": np.zeros((1, 2), dtype=np.float32),
                    "embeddings": np.zeros((1, 192), dtype=np.float32),
                    "target_probabilities": np.zeros(1, dtype=np.float32),
                    "target_embeddings": np.zeros((1, 192), dtype=np.float32),
                    "target_embeddings_available": np.asarray(True)}
    monkeypatch.setattr(diagnostic, "_prepare_diagnostic",
                        lambda: ({"source_hashes": {}, "fingerprint_facts": {}}, "fp", bad_baseline))
    monkeypatch.setattr(diagnostic, "_capture_device_arm",
                        lambda *args, **kwargs: calls.append(args[1]))

    with pytest.raises(ValueError, match="saved feasibility probe identities"):
        diagnostic.run_diagnostic(tmp_path / "invalid-baseline")

    assert calls == []


def test_run_diagnostic_never_exports_private_exception_text(tmp_path, monkeypatch):
    secret = "patient-identifier-PRIVATE-9381 feature-value-PRIVATE"

    def fail_preparation():
        raise RuntimeError(f"failed on {secret}")

    monkeypatch.setattr(diagnostic, "_prepare_diagnostic", fail_preparation)
    output = tmp_path / "private-error"
    with pytest.raises(RuntimeError, match="patient-identifier-PRIVATE"):
        diagnostic.run_diagnostic(output)

    for name in ("report.json", "report.md", "completion.json"):
        assert secret not in (output / name).read_text()
    report = (output / "report.json").read_text()
    assert "diagnostic failed during baseline validation" in report
