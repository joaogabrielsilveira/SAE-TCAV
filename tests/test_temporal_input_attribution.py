import numpy as np
import pytest

from temporal_input_attribution import audit_ledger, audit_paired_rows, audit_roles, inventory_inputs


def ledger_row(build="r1", reference=2007, family="2007/42/a", aliases="last_3,all_history"):
    return {"build": build, "reference_year": str(reference), "family": family,
            "full_identity": f"{build}/{reference}/{family}", "aliases": aliases}


def test_full_keys_allow_same_family_in_different_builds_and_reject_alias_collision():
    rows = [ledger_row(), ledger_row("all_history", 2007, aliases="all_history")]
    builds = [{"build": "r1", "reference": 2007, "variants": ["last_3", "all_history"]}]
    assert len(audit_ledger(rows[:1], builds)) == 1
    with pytest.raises(ValueError, match="alias"):
        audit_ledger(rows, builds)
    with pytest.raises(ValueError, match="duplicate"):
        audit_ledger(rows[:1] * 2, builds)
    rows[1]["aliases"] = "baseline"
    assert len(audit_ledger(rows, builds)) == 2


def arrays():
    return ({"patient_id": np.array(["a", "b"]), "year": np.array([2007, 2008]),
             "label": np.array([0, 1], dtype=np.float32), "features": np.array([[1., 2.], [3., 4.]]),
             "feature_names": np.array(["x", "y"])},
            {"patient_id": np.array(["b"]), "year": np.array([2008]), "positions": np.array([1])},
            {"patient_id": np.array(["a", "b"]), "year": np.array([2007, 2008]),
             "label": np.array([0, 1]), "features": np.array([[1., 2.], [3., 4.]]),
             "feature_names": np.array(["x", "y"])})


def test_exact_patient_year_clinical_label_identity_and_positions():
    actual, held, population = arrays()
    assert audit_paired_rows(actual, held, population)["actual_rows_without_held"] == 1
    held["year"][0] = 2007
    with pytest.raises(ValueError, match="held"):
        audit_paired_rows(actual, held, population)
    actual, held, population = arrays()
    population["features"][1, 0] = 9
    with pytest.raises(ValueError, match="features"):
        audit_paired_rows(actual, held, population)
    actual, held, population = arrays()
    population["label"][0] = 1
    with pytest.raises(ValueError, match="label"):
        audit_paired_rows(actual, held, population)
    actual, held, population = arrays()
    actual["label"] = actual["label"].astype(np.int64)
    with pytest.raises(ValueError, match="label"):
        audit_paired_rows(actual, held, population)
    actual, held, population = arrays()
    actual["patient_id"][1] = "a"
    actual["year"][1] = 2007
    with pytest.raises(ValueError, match="duplicate"):
        audit_paired_rows(actual, held, population)


def test_inventory_only_requested_files_and_explicit_unknowns(tmp_path):
    path = tmp_path / "actual.npz"
    path.write_bytes(b"archive")
    audit = inventory_inputs({"actual": path, "held": tmp_path / "missing.npz"},
                             {"historical_package": None})
    assert audit["files"]["actual"]["sha256"]
    assert audit["files"]["held"]["status"] == "missing"
    assert audit["provenance"]["historical_package"] == "unknown"
    assert set(audit["files"]) == {"actual", "held"}


def test_saved_roles_require_matching_identity_and_order():
    saved = {"identity": "frozen", "row_role": np.array(["protected", "evaluation"])}
    original = np.array(["tabpfn_context", "evaluation"])
    audit_roles(saved, {"identity": "frozen", "row_role": original.copy()})
    with pytest.raises(ValueError, match="roles"):
        audit_roles(saved, {"identity": "other", "row_role": original})
    with pytest.raises(ValueError, match="roles"):
        audit_roles(saved, {"identity": "frozen", "row_role": original[::-1]})


def test_documented_outcome_mask_and_serialization():
    actual, held, population = arrays()
    actual["year"][1] = held["year"][0] = population["year"][1] = 2015
    actual["label"][1] = np.nan
    report = audit_paired_rows(actual, held, population)
    assert report["label_years"]["2015"] == {"rows": 1, "masked": 1}
    actual["label"][1] = 1
    with pytest.raises(ValueError, match="label"):
        audit_paired_rows(actual, held, population)


def test_projected_roles_only_protect_original_fitting_side():
    saved = {"identity": "id", "row_role": np.array(["protected", "protected", "fitting", ""])}
    original = {"identity": "id", "row_role": np.array(["tabpfn_context", "sae_discovery", "fitting", ""])}
    audit_roles(saved, original)
    original["row_role"][1] = "unexpected"
    with pytest.raises(ValueError, match="roles"):
        audit_roles(saved, original)


def test_temporal_scaler_isolated_context_fit_and_clipping():
    import torch
    from torch import nn
    from tabpfn.model.encoders import MinMaxScalingEncoderStep, SequentialEncoder
    from temporal_input_attribution import with_context_scaler

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = SequentialEncoder(MinMaxScalingEncoderStep(
                normalize_on_train_only=True, in_keys=["dist_shift_domain"], out_keys=["dist_shift_domain"]))
            self.weight = nn.Parameter(torch.tensor([3.]))
            self.cache_trainset_representation = False

    original = Model()
    model = with_context_scaler(original)
    scaler = model.encoder[0]
    assert scaler is not original.encoder[0]
    assert torch.equal(model.weight, original.weight)
    context = torch.tensor([0., 2.]).reshape(-1, 1, 1)
    joint = torch.tensor([0., 2., 4., -20., 20.]).reshape(-1, 1, 1)
    scaler({"dist_shift_domain": context.clone()}, cache_trainset_representation=True, single_eval_pos=2)
    cached = scaler({"dist_shift_domain": joint[2:].clone()}, cache_trainset_representation=True, single_eval_pos=0)["dist_shift_domain"]
    observed = scaler({"dist_shift_domain": joint.clone()}, cache_trainset_representation=False, single_eval_pos=2)["dist_shift_domain"]
    torch.testing.assert_close(cached[:, 0, 0], torch.tensor([2., -5., 6.]))
    torch.testing.assert_close(observed[2:], cached)
    constant = torch.tensor([3., 3., 4.]).reshape(-1, 1, 1)
    scaled = scaler({"dist_shift_domain": constant}, cache_trainset_representation=False, single_eval_pos=2)["dist_shift_domain"]
    torch.testing.assert_close(scaled[:, 0, 0], torch.tensor([0., 0., 1.]))
    with pytest.raises(RuntimeError, match="context"):
        with_context_scaler(original).encoder[0]({"dist_shift_domain": context}, cache_trainset_representation=True, single_eval_pos=0)


@pytest.mark.parametrize("fail", [False, True])
def test_capture_same_call_observes_and_cleans_up(fail):
    import torch
    from torch import nn
    from tabpfn.model.encoders import Time2VecEncoderStep
    from temporal_input_attribution import capture_same_call

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.time = Time2VecEncoderStep(time_emb_size=2, weight_init_alpha=1, weight_init_beta=1,
                gradient_multiplier=1, in_keys=["dist_shift_domain"], out_keys=["dist_shift_domain"])
            self.decoder_dict = nn.ModuleDict({"standard": nn.Linear(2, 2)})

    class Classifier:
        def _parse_predictions_per_configuration(self, predictions, ensemble_configurations):
            return predictions
        def _aggregate_predictions(self, outputs, **kwargs):
            return outputs

    model, classifier = Model(), Classifier()
    def predict():
        state = model.time({"dist_shift_domain": torch.tensor([[[2.]]])}, cache_trainset_representation=False, single_eval_pos=1)
        h = state["dist_shift_domain"]
        logits = model.decoder_dict["standard"](h)
        parsed = classifier._parse_predictions_per_configuration(logits, ["fixed"])
        prob = classifier._aggregate_predictions(parsed).softmax(-1)
        if fail:
            raise ValueError("prediction failed")
        return {"proba": prob.squeeze(1), "test_embeddings": h}

    def continuation_factory(classifier, head, configurations, kwargs, identity):
        class Continuation:
            def probabilities(self, embeddings):
                return head(embeddings).softmax(-1)
        return Continuation()

    if fail:
        with pytest.raises(ValueError, match="prediction failed"):
            capture_same_call(classifier, model, predict, lambda: "fixed", continuation_factory)
    else:
        result = capture_same_call(classifier, model, predict, lambda: "fixed", continuation_factory)
        assert result["time_values"].item() == 2
        assert result["configurations"] == ["fixed"]
        assert result["replay_max_absolute_error"] == 0
    assert "_parse_predictions_per_configuration" not in classifier.__dict__
    assert "_aggregate_predictions" not in classifier.__dict__
    assert not model.time._forward_pre_hooks and not model.decoder_dict["standard"]._forward_pre_hooks


def test_installed_classifier_context_cache_and_capture():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable: installed predictor lifecycle runs on Fiji GPU")
    from importlib import resources
    from pathlib import Path
    import tabpfn
    from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig
    from tabpfn_model import infer_model_additional_x_info, make_dist_tensor
    from concept_subgroup_drift_pilot import predictor_context_identity, predictor_sdpa_context, configure_predictor_precision, _frozen_continuation
    from temporal_input_attribution import prepare_context_classifier, capture_same_call, ContextMinMaxScaler, matched_joint_execution, observe_transformer_inputs

    checkpoint = Path(str(resources.files(tabpfn))) / "model_cache" / "tabpfn_dist_model_1.cpkt"
    paths = TabPFNModelPathsConfig(paths=[str(checkpoint)], task_type="dist_shift_multiclass")
    classifier = get_best_tabpfn(task_type="dist_shift_multiclass", model_type="single_fast",
                                paths_config=paths, debug=False, device="cuda")
    classifier.seed = 42
    classifier.fit_at_predict_time = False
    classifier.sklearn_compatible_precision = False
    classifier.fp16_inference = False
    classifier.normalize_with_test = False
    import hashlib
    classifier.pilot_checkpoint_sha256_ = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    classifier.pilot_precision_ = configure_predictor_precision("cuda")
    classifier.pilot_domain_policy_ = "calendar_year_minus_first_context_year"
    classifier.init_model_and_get_model_config()
    original = {name: tensor.detach().clone() for name, tensor in classifier.model_processed_.state_dict().items()}
    prepared = prepare_context_classifier(classifier)
    context = np.arange(96, dtype=np.float32).reshape(32, 3) / 96
    labels = np.tile([0, 1], 16)
    domains = np.tile([0, 2], 16).astype(np.int64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with predictor_sdpa_context("cuda"):
        prepared.fit(context, labels, additional_x={"dist_shift_domain": domains})
    model = prepared.cached_models_[0]
    scalers = [m for m in model.modules() if isinstance(m, ContextMinMaxScaler)]
    assert len(scalers) == 1
    assert torch.all(scalers[0].context_min == 0) and torch.all(scalers[0].context_max == 2)
    additional_device, shape = infer_model_additional_x_info(prepared)
    query_domain = make_dist_tensor(np.array([4], dtype=np.int64), additional_device, shape)
    def predict():
        with predictor_sdpa_context("cuda"):
            return prepared.predict_full(context[:1], additional_x={"dist_shift_domain": query_domain},
                                         get_additional_outputs=["test_embeddings"])
    with observe_transformer_inputs(prepared) as cached_inputs:
        result = capture_same_call(prepared, model, predict, lambda: predictor_context_identity(prepared), _frozen_continuation)
    torch.testing.assert_close(result["time_values"], torch.full_like(result["time_values"], 2.))
    assert result["probabilities"].shape[0] == 1 and result["embeddings"].shape[0] == 1
    with matched_joint_execution(prepared):
        with observe_transformer_inputs(prepared) as joint_inputs:
            joint = capture_same_call(prepared, prepared.model_processed_, predict,
                                      lambda: predictor_context_identity(prepared), _frozen_continuation)
    torch.testing.assert_close(joint["time_values"][-1], torch.full_like(joint["time_values"][-1], 2.))
    assert joint["probabilities"].shape[0] == 1 and joint["embeddings"].shape[0] == 1
    np.testing.assert_allclose(joint["probabilities"], result["probabilities"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(joint["embeddings"], result["embeddings"], rtol=1e-4, atol=1e-5)
    assert torch.equal(joint_inputs[0]["inputs"][0][-1:], cached_inputs[0]["inputs"][0])
    assert joint_inputs[0]["categorical_inds"] == cached_inputs[0]["categorical_inds"]


def test_native_generated_state_survives_fit_copy_and_joint_execution():
    import torch
    from importlib import resources
    from pathlib import Path
    import tabpfn
    from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig
    from temporal_input_attribution import prepare_context_classifier, matched_joint_execution
    from tabpfn.model.transformer import PerFeatureTransformer

    checkpoint = Path(str(resources.files(tabpfn))) / "model_cache" / "tabpfn_dist_model_1.cpkt"
    paths = TabPFNModelPathsConfig(paths=[str(checkpoint)], task_type="dist_shift_multiclass")
    classifier = get_best_tabpfn(task_type="dist_shift_multiclass", model_type="single_fast",
                                paths_config=paths, debug=False, device="cpu")
    classifier.seed = 42
    classifier.fit_at_predict_time = False
    classifier.sklearn_compatible_precision = False
    classifier.fp16_inference = False
    classifier.normalize_with_test = False
    classifier.init_model_and_get_model_config()
    classifier.pilot_checkpoint_sha256_ = "synthetic-checkpoint"
    classifier.pilot_precision_ = {"device": "cpu", "amp": "off", "tf32": False}
    classifier.pilot_domain_policy_ = "synthetic-calendar-domain"
    transformers = [module for module in classifier.model_processed_.modules()
                    if isinstance(module, PerFeatureTransformer)
                    and module.feature_positional_embedding == "subspace"]
    assert len(transformers) == 1
    transformer = transformers[0]
    original_state = {name: value.detach().clone()
                      for name, value in classifier.model_processed_.state_dict().items()}
    width = transformer.feature_positional_embedding_embeddings.out_features
    prescribed = torch.full((3, width), 0.375, dtype=torch.float32)
    with pytest.raises(ValueError, match="width is incompatible"):
        prepare_context_classifier(classifier, positional_embeddings=torch.zeros((3, width + 1)))
    prepared = prepare_context_classifier(classifier, positional_embeddings=prescribed)
    joint_transformer = next(module for module in prepared.model.modules()
                             if isinstance(module, PerFeatureTransformer)
                             and module.feature_positional_embedding == "subspace")
    assert joint_transformer is not transformer
    context = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16
    labels = torch.tensor([0, 1] * 4, dtype=torch.float32)
    domains = np.arange(len(context), dtype=np.int64)
    prepared.fit(context.numpy(), labels.numpy(),
                 additional_x={"dist_shift_domain": domains})

    cached_transformer = next(module for module in prepared.cached_models_[0].modules()
                              if isinstance(module, PerFeatureTransformer)
                              and module.feature_positional_embedding == "subspace")
    torch.testing.assert_close(cached_transformer.cached_embeddings, prescribed)
    assert cached_transformer is not transformer
    assert not hasattr(transformer, "cached_embeddings")
    query = context[:1].numpy()
    query_domain = {"dist_shift_domain": np.array([[[8]]], dtype=np.int64)}
    cache_deltas = []
    original_cached_method = cached_transformer.__dict__["add_embeddings"]
    cached_add_embeddings = cached_transformer.add_embeddings

    def observe_cached_add(x, y, *args, **kwargs):
        before = x.detach().clone()
        result = cached_add_embeddings(x, y, *args, **kwargs)
        cache_deltas.append(result[0].detach() - before)
        return result

    cached_transformer.add_embeddings = observe_cached_add
    try:
        prepared.predict_full(query, additional_x=query_domain)
    finally:
        cached_transformer.add_embeddings = original_cached_method

    joint_deltas = []
    original_joint_method = joint_transformer.__dict__["add_embeddings"]
    joint_add_embeddings = joint_transformer.add_embeddings

    def observe_joint_add(x, y, *args, **kwargs):
        before = x.detach().clone()
        result = joint_add_embeddings(x, y, *args, **kwargs)
        joint_deltas.append(result[0].detach() - before)
        return result

    joint_transformer.add_embeddings = observe_joint_add
    try:
        with matched_joint_execution(prepared):
            prepared.predict_full(query, additional_x=query_domain)
    finally:
        joint_transformer.add_embeddings = original_joint_method
    assert len(cache_deltas) == len(joint_deltas) == 1
    torch.testing.assert_close(cache_deltas[0], prescribed[None, None].expand_as(cache_deltas[0]))
    torch.testing.assert_close(joint_deltas[0], prescribed[None, None].expand_as(joint_deltas[0]))
    torch.testing.assert_close(cached_transformer.cached_embeddings, prescribed)

    names = ("get_ensemble_configurations", "build_transformer_input_for_each_configuration")
    previous = {name: prepared.__dict__[name] for name in names if name in prepared.__dict__}
    with pytest.raises(RuntimeError, match="joint sentinel"):
        with matched_joint_execution(prepared):
            raise RuntimeError("joint sentinel")
    assert prepared.fit_at_predict_time is False
    assert all((name in previous and prepared.__dict__[name] is previous[name])
               or (name not in previous and name not in prepared.__dict__) for name in names)

    changed = torch.zeros_like(cached_transformer.cached_embeddings)
    changed[0, 0] = 1
    cached_transformer.cached_embeddings = changed
    with pytest.raises(RuntimeError, match="differ from prescribed state"):
        prepared.predict_full(query, additional_x=query_domain)
    assert not prepared.fit_at_predict_time
    for name, tensor in classifier.model_processed_.state_dict().items():
        torch.testing.assert_close(tensor, original_state[name], rtol=0, atol=0)


def test_scoring_builder_and_explicit_identity_match_real_native_cpu_predictors(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import torch
    from importlib import resources
    from pathlib import Path
    import tabpfn
    from tabpfn.model.encoders import MinMaxScalingEncoderStep
    import concept_subgroup_drift_pilot as pilot
    import temporal_input_attribution as module
    from temporal_input_attribution import ContextMinMaxScaler

    checkpoint = Path(str(resources.files(tabpfn))) / "model_cache" / "tabpfn_dist_model_1.cpkt"
    checkpoint_sha = module._file_sha256(checkpoint)
    config_path = tmp_path / "resolved-config.json"
    config_path.write_text(json.dumps({"split_seed": 42, "context_window": "reference_only"}))
    patients = [f"native-{index}" for index in range(8)]
    context = pd.DataFrame({"f0": np.arange(8, dtype=np.float32) / 8,
                            "f1": np.arange(8, dtype=np.float32)[::-1] / 8,
                            "y": np.array([0, 1] * 4, dtype=np.float32),
                            "patient_id": patients, "year": np.full(8, 2013, dtype=np.int64)})
    rows_identity = pilot._identity_hash(list(zip(patients, [2013] * 8)))
    system = {"build": "last_3", "reference": 2013, "domain_origin": 2013,
              "saved_checkpoint_sha256": checkpoint_sha,
              "context_rows_identity": rows_identity, "files": {"config": {"path": str(config_path)}}}
    width = 192
    prescribed = torch.full((3, width), 0.375, dtype=torch.float32)
    native_predictor = pilot.fit_frozen_predictor(context, ("f0", "f1"), "cpu", seed=42)
    native_state = module._native_positional_embeddings(native_predictor)
    cached_state = native_predictor["classifier"].cached_models_[0].cached_embeddings
    torch.testing.assert_close(native_state, cached_state, rtol=0, atol=0)
    assert native_state.shape == (3, width)
    historical = module._fit_frozen_predictor_with_state(
        context, ("f0", "f1"), system, "cpu", 42, prescribed)
    classifier, _ = module._initialize_context_classifier(system, "cpu")
    consistent = module.prepare_context_classifier(classifier, prescribed)
    with pilot.predictor_sdpa_context("cpu"):
        consistent.fit(context[["f0", "f1"]].to_numpy(dtype=np.float32),
                       context["y"].to_numpy(dtype=np.float32),
                       additional_x={"dist_shift_domain": np.zeros(len(context), dtype=np.int64)})
    historical_model = historical["classifier"].cached_models_[0]
    normalized_model = consistent.cached_models_[0]
    historical_scalers = [step for step in historical_model.modules()
                          if isinstance(step, MinMaxScalingEncoderStep)
                          and list(step.in_keys) == ["dist_shift_domain"]]
    normalized_scalers = [step for step in normalized_model.modules()
                          if isinstance(step, MinMaxScalingEncoderStep)
                          and list(step.in_keys) == ["dist_shift_domain"]]
    assert len(historical_scalers) == len(normalized_scalers) == 1
    assert type(historical_scalers[0]) is MinMaxScalingEncoderStep
    assert isinstance(normalized_scalers[0], ContextMinMaxScaler)
    torch.testing.assert_close(historical_model.cached_embeddings, prescribed, rtol=0, atol=0)
    torch.testing.assert_close(normalized_model.cached_embeddings, prescribed, rtol=0, atol=0)
    query_domains = torch.tensor([20., -20.], dtype=torch.float32).reshape(-1, 1, 1)
    historical_time = historical_scalers[0]({"dist_shift_domain": query_domains.clone()},
                                               cache_trainset_representation=True,
                                               single_eval_pos=0)["dist_shift_domain"]
    normalized_time = normalized_scalers[0]({"dist_shift_domain": query_domains.clone()},
                                              cache_trainset_representation=True,
                                              single_eval_pos=0)["dist_shift_domain"]
    torch.testing.assert_close(historical_time[:, 0, 0], torch.tensor([20., -20.]))
    torch.testing.assert_close(normalized_time[:, 0, 0], torch.tensor([6., -5.]))
    assert not torch.equal(historical_time, normalized_time)
    history_identity = module._comparable_predictor_identity(
        pilot, historical["classifier"], rows_identity, "last_3/2013")
    normalized_identity = module._comparable_predictor_identity(pilot, consistent, rows_identity, "last_3/2013")
    expected_input_identity = module._clinical_input_identity(
        pilot, context[["f0", "f1"]].to_numpy(dtype=np.float32), context["y"].to_numpy(dtype=np.float32),
        {"dist_shift_domain": np.zeros(len(context), dtype=np.int64)}, historical["classifier"].classes_)
    assert history_identity["clinical_input_identity"] == expected_input_identity, (
        type(historical["classifier"].X_), type(historical["classifier"].y_),
        module._identity_value(historical["classifier"].additional_x_),
        module._identity_value(historical["classifier"].classes_))
    assert module.comparable_predictor_identities_match(history_identity, normalized_identity)
    assert history_identity["fitted_fields_identity"] == normalized_identity["fitted_fields_identity"]
    original_temperature = consistent.softmax_temperature
    consistent.softmax_temperature = 1.2345
    changed_temperature_identity = module._comparable_predictor_identity(
        pilot, consistent, rows_identity, "last_3/2013")
    assert changed_temperature_identity["fitted_fields_identity"] != history_identity["fitted_fields_identity"]
    assert not module.comparable_predictor_identities_match(history_identity, changed_temperature_identity)
    consistent.softmax_temperature = original_temperature
    changed_state = prescribed.clone()
    changed_state[0, 0] += 1
    changed = module._fit_frozen_predictor_with_state(
        context, ("f0", "f1"), system, "cpu", 42, changed_state)
    changed_identity = module._comparable_predictor_identity(
        pilot, changed["classifier"], rows_identity, "last_3/2013")
    assert not module.comparable_predictor_identities_match(history_identity, changed_identity)

    population = {"X": context[["f0", "f1"]].to_numpy(dtype=np.float32),
                  "y": context["y"].to_numpy(dtype=np.float32),
                  "year": context["year"].to_numpy(dtype=np.int64),
                  "patient_id": np.asarray(patients), "post_death": np.zeros(len(context), dtype=bool)}
    monkeypatch.setattr(pilot, "reference_year_roles", lambda *args, **kwargs: {
        "row_role": np.full(len(context), "tabpfn_context"), "is_entrant": np.zeros(len(context), dtype=bool)})
    state_sha = module._numpy_tensor_sha256(prescribed.numpy())
    positional_state = {"sha256": state_sha, "shape": list(prescribed.shape), "dtype": str(prescribed.dtype)}
    _, _, _, _, _, _, current_binding, _ = module._current_scoring_context(
        pilot, population, system, positional_state)
    assert current_binding["positional_state"] == history_identity["positional_state"], (
        current_binding["positional_state"], history_identity["positional_state"])
    persisted_pair = json.loads(json.dumps([normalized_identity, history_identity]))
    current, report, parity, replay, scope = _conditional_check_fixture("cpu")
    current["generated_state_provenance"]["last_3/2013"]["positional_state"] = positional_state
    report["generated_state_provenance"]["last_3/2013"]["positional_state"] = positional_state
    current["comparable_predictor_identity_bindings"]["last_3/2013"] = current_binding
    current["comparable_predictor_identities"]["last_3/2013"] = persisted_pair
    report["comparable_predictor_identities"]["last_3/2013"] = persisted_pair
    for row in scope + replay:
        row["positional_sha256"] = state_sha
    assert module.verify_execution_checks(report, current, parity, replay, "cpu", "source-hash", scope)
    assert module.comparable_predictor_identities_match(history_identity, persisted_pair[1])
    wrong_dtype = {**current_binding, "positional_state": {**positional_state, "dtype": "torch.float16"}}
    wrong_current = {**current, "comparable_predictor_identity_bindings": {"last_3/2013": wrong_dtype}}
    with pytest.raises(RuntimeError, match="current system identity binding"):
        module.verify_execution_checks(report, wrong_current, parity, replay, "cpu", "source-hash", scope)


def four_cells():
    from copy import deepcopy
    base = {"patient_id": np.array(["a", "b", "c"]), "year": np.array([2008] * 3),
            "system": ("r1", 2007, "context-checkpoint-config"), "concept": "r1/2007/2007/42/a",
            "label": np.array([0, 1, 1]), "features": np.array([[1.], [2.], [3.]], dtype=np.float32),
            "feature_names": np.array(["clinical"]), "rule": np.array([True, True, False]),
            "probability": np.array([.5, .8, .2]), "activation": np.array([.5, 1., 0.]),
            "embedding": np.array([[0., 0.], [1., 0.], [2., 0.]])}
    cells = {(processing, arm): deepcopy(base) for processing in ("historical", "consistent")
             for arm in ("actual", "held")}
    cells[("historical", "held")]["activation"] = np.array([0., 1., 1.])
    cells[("consistent", "held")]["activation"] = np.array([1., 1., 0.])
    cells[("consistent", "held")]["embedding"] += 1.
    return cells


def test_frozen_threshold_cutoff_confusion_and_entropy():
    from temporal_input_attribution import cell_metrics
    cell = four_cells()[("historical", "actual")]
    result = cell_metrics(cell, threshold=.5, cutoff=.5)
    assert result["false_positives"]["estimate"] == 1
    assert result["false_negatives"]["estimate"] == 1
    assert result["caught_deaths"]["estimate"] == 1
    assert result["death_f1"]["estimate"] == .5
    assert result["agreement"]["numerator"] == 1 and result["agreement"]["denominator"] == 2
    assert result["rule_f2"]["estimate"] == pytest.approx(5 / 6)
    cell["probability"] = np.array([0., 1., .5])
    result = cell_metrics(cell, .5, .5)
    assert result["mean_probability"]["estimate"] == .5
    assert result["mean_entropy"]["estimate"] == pytest.approx(np.log(2) / 3)
    cell["rule"][:] = False
    assert cell_metrics(cell, .5, .5)["agreement"]["reason"] == "zero denominator"


def test_joint_preprocessing_reuses_fitted_transform_shuffle_and_configuration():
    import torch
    from types import SimpleNamespace
    from temporal_input_attribution import matched_joint_inputs

    class Preprocessor:
        def transform(self, value, is_test=False):
            return np.asarray(value) + (10 if is_test else 3), [0]
        def fit(self, *args):
            raise AssertionError("must not refit the saved preprocessor")
    class Shuffle:
        def transform(self, value):
            return value[:, [1, 0]], [1]
        def fit(self, *args):
            raise AssertionError("must not refit the saved shuffle")
    config = SimpleNamespace(class_shift_configuration=None, subsample_samples_configuration=None)
    classifier = SimpleNamespace(cached_preprocessors_=[Preprocessor()], cached_shuffles_=[Shuffle()],
                                 ensemble_configurations_=[config], is_classification_=True)
    x = torch.tensor([[[1., 2.]], [[3., 4.]], [[5., 6.]]])
    y = torch.tensor([[0.], [1.]])
    domain = torch.tensor([[[0.]], [[2.]], [[4.]]])
    output = matched_joint_inputs(classifier, [config], x, y, 2, {"dist_shift_domain": domain})
    torch.testing.assert_close(output[0][0][:, 0], torch.tensor([[5., 4.], [7., 6.], [16., 15.]]))
    assert output[2] == [[1]] and output[5]["dist_shift_domain"][0] is domain


def test_four_cell_rejects_nonfinite_and_changed_rows():
    from temporal_input_attribution import validate_four_cells
    cells = four_cells()
    validate_four_cells(cells)
    cells[("consistent", "held")]["features"][0, 0] += 1
    with pytest.raises(ValueError, match="features"):
        validate_four_cells(cells)
    cells = four_cells()
    cells[("consistent", "held")]["year"][0] = 2007
    with pytest.raises(ValueError, match="patient/year"):
        validate_four_cells(cells)
    cells = four_cells()
    cells[("consistent", "held")]["probability"][0] = np.nan
    with pytest.raises(ValueError, match="probability"):
        validate_four_cells(cells)
    cells = four_cells()
    for cell in cells.values():
        del cell["system"]
    with pytest.raises(ValueError, match="system"):
        validate_four_cells(cells)


def test_shared_cluster_draws_four_cell_contrasts_and_later_change():
    from temporal_input_attribution import patient_bootstrap, summarize_four_cells, reference_to_later
    patients, weights = patient_bootstrap([np.array(["a", "b"]), np.array(["a", "c"])], draws=5, seed=7)
    assert patients.tolist() == ["a", "b", "c"] and weights.shape == (5, 3)
    assert np.all(weights.sum(axis=1) == 3)
    bootstrap = (np.array(["a", "b", "c"]), np.array([[1, 1, 1], [3, 0, 0], [0, 0, 3]]))
    reference = summarize_four_cells(four_cells(), .5, .5, bootstrap)
    assert reference["contrasts"]["agreement"]["historical_actual_minus_held"]["estimate"] == 0
    change = reference["contrasts"]["agreement"]["normalization_sensitivity"]
    assert change["estimate"] == -.5 and change["valid_draws"] == 2 and change["undefined_draws"] == 1
    assert reference["contrasts"]["embedding_displacement"]["consistent_actual_minus_held"]["estimate"] == pytest.approx(np.sqrt(2))
    assert reference["contrasts"]["embedding_displacement"]["consistent_minus_historical_held"]["estimate"] == pytest.approx(np.sqrt(2))
    assert reference["contrasts"]["agreement"]["normalization_sensitivity"]["components"][("historical", "actual")]["denominator"] == 2
    later = four_cells()
    for cell in later.values():
        cell["year"] += 1
    later[("consistent", "held")]["activation"][0] = 0
    later = summarize_four_cells(later, .5, .5, bootstrap)
    delta = reference_to_later(reference, later)
    assert delta["contrasts"]["agreement"]["normalization_sensitivity"]["estimate"] == .5
    assert delta["contrasts"]["agreement"]["normalization_sensitivity"]["valid_draws"] == 2
    assert delta["cells"][("historical", "actual")]["death_f1"]["reference"]["denominator"] == 4
    assert delta["cells"][("historical", "actual")]["death_f1"]["later"]["denominator"] == 4
    different = four_cells()
    for cell in different.values():
        cell["system"] = ("other", 2007, "context-checkpoint-config")
        cell["year"] += 1
    with pytest.raises(ValueError, match="system"):
        reference_to_later(reference, summarize_four_cells(different, .5, .5, bootstrap))
    assert "auroc" not in reference["contrasts"] and "brier" not in reference["contrasts"]


def test_comparison_refuses_missing_parity_before_scoring(tmp_path):
    from temporal_input_attribution import run_comparison
    with pytest.raises(RuntimeError, match="checks"):
        run_comparison(output=tmp_path / "panel")
    assert not (tmp_path / "panel").exists()


def test_comparison_runs_synthetic_panel_and_rejects_stale_reuse(tmp_path, monkeypatch):
    import json
    import temporal_input_attribution as module
    cases = {2008: four_cells(), 2009: four_cells()}
    for cell in cases[2009].values():
        cell["year"] += 1
    system = {"build": "r1", "reference": 2008, "variants": ["last_3", "all_history"],
              "death_threshold": {"last_3": .5, "all_history": .5}}
    concept = {"concept": "r1/2007/2007/42/a", "build": "r1", "reference": 2008, "cutoff": .5}
    inputs = {"systems": [system], "concepts": [concept], "patient_sets": [np.array(["a", "b", "c"])],
              "fingerprints": {("r1", 2008): "inputs-v1"}}
    monkeypatch.setattr(module, "_comparison_inputs", lambda variants, device: inputs)
    monkeypatch.setattr(module, "_score_system", lambda *args: {"fixture": np.array([7])})
    monkeypatch.setattr(module, "_concept_cells", lambda *args: cases)
    monkeypatch.setattr(module, "_archive_activation_diagnostics", lambda *args: {})
    primary = tmp_path / "primary"
    (primary / "run").mkdir(parents=True)
    result = module.run_comparison(output=primary)
    assert result["status"] == "completed_with_unavailable" and result["concept_years"] == 2
    assert (primary / "run").is_dir()
    public = (primary / "aggregate.json").read_text()
    assert '"patient_id"' not in public and '"per_patient"' not in public
    assert json.loads(public)["cases"][0]["concept"] == concept["concept"]
    monkeypatch.setattr(module, "PRIMARY_COMPARISON", primary)
    def no_rescoring(*args):
        raise AssertionError("valid primary cache should be reused")
    monkeypatch.setattr(module, "_score_system", no_rescoring)
    reused = module.run_comparison(variants=("last_3",), output=tmp_path / "valid_reuse", reuse_primary=True)
    assert reused["systems"][0]["reused_primary"] is True
    inputs["fingerprints"][("r1", 2008)] = "changed-inputs"
    (tmp_path / "all_history" / "run").mkdir(parents=True)
    with pytest.raises(ValueError, match="reuse"):
        module.run_comparison(variants=("last_3",), output=tmp_path / "all_history", reuse_primary=True)
    assert json.loads((tmp_path / "all_history" / "run-record.json").read_text())["status"] == "failed"
    inputs["fingerprints"][("r1", 2008)] = "inputs-v1"
    (primary / "private" / "r1_2008.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="reuse archive checksum"):
        module.run_comparison(variants=("last_3",), output=tmp_path / "corrupt_reuse", reuse_primary=True)


def _conditional_check_fixture(device="cpu"):
    from copy import deepcopy

    file = {"status": "available", "sha256": "input-hash"}
    dependency = {"method": "tabpfn-python-tree-sha256-v1", "sha256": "a" * 64}
    system = {"build": "last_3", "reference": 2013, "files": {"predictor": file},
              "current_checkpoint_matches": True, "current_pilot_source_matches": False,
              "historical_tabpfn_package": "unknown", "historical_tabpfn_source_identity": "unknown"}
    state = {"build": "last_3", "reference": 2013, "origin": "native_cpu_reconstruction_candidate",
             "source_sha256": "state-source", "input_sha256": "state-inputs",
             "input_fingerprints": {"predictor": "input-hash"}, "device": "cpu",
             "positional_state": {"sha256": "position", "shape": [3, 2], "dtype": "torch.float32"}}
    identity = {"opaque_identity": "opaque-cuda", "device": "cuda:0",
                "system_identity": "last_3/2013",
                "clinical_input_identity": "clinical-input", "context_rows_identity": "ordered-context",
                "checkpoint_sha256": "checkpoint", "learned_parameter_identity": "learned",
                "fitted_fields_identity": "fitted-fields",
                "configuration_identity": "configuration", "clinical_transform_identity": "clinical-transform",
                "precision": {"dtype": "fp32", "amp": "off", "tf32": False},
                "positional_state": {"sha256": "position", "shape": [3, 2], "dtype": "float32"},
                "execution_state": {"eval": True, "depth": 12, "dropout": 0.0}}
    cpu_identity = {**deepcopy(identity), "opaque_identity": "opaque-cpu", "device": "cpu"}
    current = {"systems": [system], "common": {"files": {"checkpoint": file},
               "provenance": {"current_tabpfn_source_identity": dependency}},
               "coverage": [{"concept": "full-key", "concept_file_sha256": "concept-hash"}],
               "generated_state_provenance": {"last_3/2013": state},
               "comparable_predictor_identities": {"last_3/2013": [identity, cpu_identity]},
               "comparable_predictor_identity_bindings": {"last_3/2013": {
                   "system_identity": "last_3/2013", "clinical_input_identity": "clinical-input",
                   "context_rows_identity": "ordered-context",
                   "checkpoint_sha256": "checkpoint",
                   "positional_state": {"sha256": "position", "shape": [3, 2], "dtype": "float32"}}}}
    report = deepcopy(current)
    report["checks"] = {"ready_for_scoring": True, "conditional_numerical_reconstruction": "pass",
                        "historical_environment": "unknown", "errors": [], "selected_systems": "all",
                        "historical_exact_replay": "unavailable: dependency identity unknown",
                        "device": device, "source_sha256": "source-hash"}
    expected = []
    for year, arm, row_hash in ((2013, "actual", "actual-ref"),
                                (2014, "actual", "actual-held-year"),
                                (2014, "held", "held-year")):
        expected.append({"build": "last_3", "reference": 2013, "year": year, "arm": arm,
                         "rows_sha256": row_hash, "rows": 2, "schedule_id": f"{arm}-{year}-candidate",
                         "source_sha256": "source-hash", "state_source_sha256": "state-source",
                         "input_sha256": "state-inputs", "positional_sha256": "position",
                         "device": device, "settings_sha256": "settings", "precision": "fp32",
                         "schedule_certainty": "candidate"})
    replay = [{**scope, "probability_within_tolerance": True,
               "embedding_within_tolerance": True,
               "saved_code_serialization_match": scope["arm"] == "held"} for scope in expected]
    parity = [{"build": "last_3", "reference": 2013, "year": year, "input": arm, "parity": "pass",
               "reference_actual_held_control": "pass" if year == 2013 and arm == "held" else "not applicable"}
              for year in (2013, 2014) for arm in ("actual", "held")]
    return current, report, parity, replay, expected


def test_execution_gate_rechecks_device_source_and_frozen_input_hashes():
    from copy import deepcopy
    from temporal_input_attribution import verify_execution_checks
    current, report, parity, replay, scope = _conditional_check_fixture("cuda")
    assert verify_execution_checks(report, current, parity, replay, "cuda", "source-hash", scope)
    with pytest.raises(RuntimeError, match="source"):
        verify_execution_checks(report, current, parity, replay, "cuda", "changed", scope)
    with pytest.raises(RuntimeError, match="device"):
        verify_execution_checks(report, current, parity, replay, "cpu", "source-hash", scope)
    changed = deepcopy(current)
    changed["systems"][0]["files"]["predictor"]["sha256"] = "changed"
    with pytest.raises(RuntimeError, match="stale"):
        verify_execution_checks(report, changed, parity, replay, "cuda", "source-hash", scope)
    mismatch = deepcopy(report)
    mismatch["systems"][0]["historical_tabpfn_source_identity"] = {"sha256": "b" * 64}
    with pytest.raises(RuntimeError, match="dependency"):
        verify_execution_checks(mismatch, current, parity, replay, "cuda", "source-hash", scope)
    changed_identity = deepcopy(current)
    changed_identity["comparable_predictor_identities"]["last_3/2013"][1]["positional_state"]["sha256"] = "changed"
    changed_report = deepcopy(report)
    changed_report["comparable_predictor_identities"] = deepcopy(changed_identity["comparable_predictor_identities"])
    with pytest.raises(RuntimeError, match="explicit cross-device"):
        verify_execution_checks(changed_report, changed_identity, parity, replay, "cuda", "source-hash", scope)

    shared_tamper = deepcopy(current)
    for item in shared_tamper["comparable_predictor_identities"]["last_3/2013"]:
        item["clinical_input_identity"] = "another-system-clinical-input"
    shared_tamper_report = deepcopy(report)
    shared_tamper_report["comparable_predictor_identities"] = deepcopy(
        shared_tamper["comparable_predictor_identities"])
    with pytest.raises(RuntimeError, match="current system identity binding"):
        verify_execution_checks(shared_tamper_report, shared_tamper, parity, replay, "cuda", "source-hash", scope)

    wrong_system_pair = deepcopy(current)
    for item in wrong_system_pair["comparable_predictor_identities"]["last_3/2013"]:
        item["positional_state"] = {"sha256": "other-system-position", "shape": [3, 2], "dtype": "float32"}
    wrong_system_report = deepcopy(report)
    wrong_system_report["comparable_predictor_identities"] = deepcopy(
        wrong_system_pair["comparable_predictor_identities"])
    with pytest.raises(RuntimeError, match="current system identity binding"):
        verify_execution_checks(wrong_system_report, wrong_system_pair, parity, replay, "cuda", "source-hash", scope)


def test_primary_selection_keeps_shared_builds_and_rejects_extra_contexts():
    from temporal_input_attribution import select_primary_builds
    builds = [{"build": "r1", "reference": 2007, "variants": ["last_3", "all_history"]}]
    builds += [{"build": "all_history", "reference": year, "variants": ["last_3", "all_history"]}
               for year in (2008, 2009)]
    builds += [{"build": "last_3", "reference": year, "variants": ["last_3"]} for year in range(2010, 2014)]
    builds += [{"build": "all_history", "reference": year, "variants": ["all_history"]} for year in range(2010, 2014)]
    selected = select_primary_builds(builds)
    assert {(item["build"], item["reference"]) for item in selected} == {
        ("r1", 2007), ("all_history", 2008), ("all_history", 2009),
        *(('last_3', year) for year in range(2010, 2014))}
    with pytest.raises(ValueError, match="primary"):
        select_primary_builds(builds[:-5])


def test_primary_gate_ignores_dropped_contexts_but_requires_primary_proof():
    from copy import deepcopy
    from temporal_input_attribution import verify_execution_checks
    current, report, parity, replay, scope = _conditional_check_fixture("cpu")
    primary = current["systems"][0]
    dropped = {**deepcopy(primary), "build": "all_history"}
    report["systems"].append(dropped)
    report["coverage"].append({"concept": "dropped-key", "concept_file_sha256": "other"})
    report["checks"]["source_sha256"] = "source"
    for item in scope:
        item["source_sha256"] = "source"
    for item in replay:
        item["source_sha256"] = "source"
    assert verify_execution_checks(report, current, parity, replay, "cpu", "source", scope)
    with pytest.raises(RuntimeError, match="parity"):
        verify_execution_checks(report, current, parity[:1], replay, "cpu", "source", scope)


def test_generated_identity_and_scoped_replay_require_explicit_matching_proof():
    from copy import deepcopy
    from temporal_input_attribution import (
        comparable_predictor_identities_match,
        class_one_probabilities,
        validate_scoped_replay_evidence,
    )

    identity = {
        "opaque_identity": "cuda-context-hash",
        "device": "cuda:0",
        "system_identity": "r1/2007",
        "clinical_input_identity": "clinical-input",
        "context_rows_identity": "ordered-context",
        "checkpoint_sha256": "checkpoint",
        "fitted_fields_identity": "fitted-fields",
        "learned_parameter_identity": "learned",
        "configuration_identity": "configuration",
        "clinical_transform_identity": "clinical-transform",
        "precision": {"dtype": "fp32", "amp": "off", "tf32": False},
        "positional_state": {"sha256": "position", "shape": [67, 192], "dtype": "float32"},
        "execution_state": {"eval": True, "depth": 12, "dropout": 0.0},
    }
    cpu_identity = {**deepcopy(identity), "opaque_identity": "cpu-context-hash", "device": "cpu"}
    assert comparable_predictor_identities_match(identity, cpu_identity)
    cpu_identity["positional_state"] = {**identity["positional_state"], "sha256": "changed"}
    assert not comparable_predictor_identities_match(identity, cpu_identity)
    cpu_identity = {**deepcopy(identity), "opaque_identity": "cpu-context-hash", "device": "cpu",
                    "precision": {"dtype": "fp32", "amp": "on", "tf32": False}}
    assert not comparable_predictor_identities_match(identity, cpu_identity)
    missing_field = {**deepcopy(identity), "clinical_transform_identity": None}
    assert not comparable_predictor_identities_match(missing_field, missing_field)
    missing_state = {**deepcopy(identity), "positional_state": {}}
    assert not comparable_predictor_identities_match(missing_state, missing_state)
    expected_probabilities = np.array([.1, .7], dtype=np.float32)
    np.testing.assert_array_equal(class_one_probabilities(expected_probabilities, 2), expected_probabilities)
    matrix_probabilities = np.array([[.9, .1], [.3, .7]], dtype=np.float32)
    np.testing.assert_array_equal(class_one_probabilities(matrix_probabilities, 2), expected_probabilities)
    with pytest.raises(ValueError, match="probability layout"):
        class_one_probabilities(np.ones((2, 3), dtype=np.float32), 2)

    expected = [
        {"build": "r1", "reference": 2007, "year": 2007, "arm": "actual", "rows_sha256": "actual-rows",
         "rows": 12, "schedule_id": "actual-context-v1", "source_sha256": "work-source",
         "state_source_sha256": "source-of-position", "input_sha256": "actual-input",
         "positional_sha256": "position", "device": "cpu",
         "settings_sha256": "settings", "precision": "fp32", "schedule_certainty": "candidate"},
        {"build": "r1", "reference": 2007, "year": 2008, "arm": "held", "rows_sha256": "held-rows",
         "rows": 7, "schedule_id": "held-candidate-v1", "source_sha256": "work-source",
         "state_source_sha256": "source-of-position", "input_sha256": "held-input",
         "positional_sha256": "position", "device": "cpu",
         "settings_sha256": "settings", "precision": "fp32", "schedule_certainty": "candidate"},
    ]
    evidence = [{**scope, "probability_within_tolerance": True,
                 "embedding_within_tolerance": True,
                 "saved_code_serialization_match": scope["arm"] == "held"}
                for scope in expected]
    result = validate_scoped_replay_evidence(evidence, expected)
    assert result["conditional_numerical_reconstruction"] == "pass"
    assert result["historical_environment"] == "unknown"
    with pytest.raises(ValueError, match="actual and genuine later-year held"):
        validate_scoped_replay_evidence(evidence[:1], expected[:1])
    with pytest.raises(ValueError, match="nonempty scope"):
        validate_scoped_replay_evidence([], [])
    for field, value in (("source_sha256", "stale-source"), ("input_sha256", "stale-input"),
                         ("rows_sha256", "reordered-rows"), ("settings_sha256", "changed-settings"),
                         ("schedule_id", "different-producer-schedule")):
        stale = deepcopy(evidence)
        stale[0][field] = value
        with pytest.raises(ValueError, match="scope"):
            validate_scoped_replay_evidence(stale, expected)
    missing_held = deepcopy(evidence)
    missing_held[1]["year"] = 2007
    with pytest.raises(ValueError, match="genuine later-year held"):
        validate_scoped_replay_evidence(missing_held, expected)
    failed_code = deepcopy(evidence)
    failed_code[1]["saved_code_serialization_match"] = False
    with pytest.raises(ValueError, match="saved-code"):
        validate_scoped_replay_evidence(failed_code, expected)
    failed_held_embedding = deepcopy(evidence)
    failed_held_embedding[1]["embedding_within_tolerance"] = False
    with pytest.raises(ValueError, match="held replay embedding"):
        validate_scoped_replay_evidence(failed_held_embedding, expected)
    wrong_arm = deepcopy(evidence)
    wrong_arm[0]["arm"] = "held"
    with pytest.raises(ValueError, match="genuine later-year held"):
        validate_scoped_replay_evidence(wrong_arm, expected)
    forged = deepcopy(evidence)
    forged.pop()
    with pytest.raises(ValueError, match="complete and unique"):
        validate_scoped_replay_evidence(forged, expected)


def test_ready_flag_without_generated_state_or_scoped_rows_is_rejected():
    from temporal_input_attribution import verify_execution_checks

    report = {"systems": [], "coverage": [], "common": {"files": {}, "provenance": {}},
              "checks": {"ready_for_scoring": True, "historical_exact_replay": "pass", "errors": [],
                         "selected_systems": "all", "device": "cpu", "source_sha256": "source"}}
    current = {"systems": [], "coverage": [], "common": {"files": {}, "provenance": {}}}
    with pytest.raises(RuntimeError, match="generated state"):
        verify_execution_checks(report, current, [], [], "cpu", "source")


def test_system_scorer_rejects_missing_generated_state_before_fitting(tmp_path, monkeypatch):
    import temporal_input_attribution as module

    inputs, system, calls = _synthetic_scoring_fixture(tmp_path, monkeypatch, batch_sensitive=False)
    inputs["generated_state_provenance"].clear()
    with pytest.raises(RuntimeError, match="generated state evidence"):
        module._score_system(inputs, system, "cpu")
    assert calls == []


def test_diagnostic_chain_reads_only_the_bound_positional_tensor(tmp_path):
    import hashlib
    import json
    from temporal_input_attribution import load_r1_diagnostic_positional_state

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_json(path, value):
        path.write_text(json.dumps(value, sort_keys=True))

    tensor = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor_sha = hashlib.sha256(str((2, 3)).encode() + b"torch.float32" + tensor.view(np.uint8).tobytes()).hexdigest()
    archive_path = tmp_path / "gpu_native.npz"
    np.savez_compressed(archive_path, positional_tensor=tensor, other_private_array=np.array([987654]))
    archive_sha = digest(archive_path)
    report_md = tmp_path / "report.md"
    report_csv = tmp_path / "report.csv"
    report_md.write_text("synthetic report")
    report_csv.write_text("synthetic,csv\n")
    source_sha = "current-source"
    module_sha = "diagnostic-module"
    expected_inputs = {"actual": "actual-input", "held": "held-input", "checkpoint": "checkpoint"}
    state = {"mode": "subspace", "cache_name": "cached_embeddings", "shape": [2, 3],
             "dtype": "torch.float32", "device": "cuda:0", "sha256": tensor_sha,
             "registered_buffer": False}
    arm = {"device": "cuda", "private_output_sha256": archive_sha,
           "before_state": {"positional_state": state}, "after_state": {"positional_state": state}}
    cpu_state = {**state, "device": "cpu"}
    cpu_arm = {"device": "cpu", "private_output_sha256": "cpu-private-output",
               "before_state": {"positional_state": cpu_state}, "after_state": {"positional_state": cpu_state}}
    report = {"status": "completed", "schema": "r1-2007-device-state-diagnostic-v1",
              "module_sha256": module_sha, "source_sha256": {"current": source_sha},
              "input_fingerprints": {"input_files": expected_inputs, "source_hashes": {"current": source_sha},
                                     "schedule": {"actual_batch_rows": 4096, "actual_probe_years": list(range(2007, 2015)),
                                                  "held_batch_rows": 64, "held_probe_years": list(range(2008, 2015)),
                                                  "precheck_rows": 32}},
              "arms": {"gpu_native": arm, "cpu_gpu_position": cpu_arm}}
    report_path = tmp_path / "report.json"
    write_json(report_path, report)
    completion = {"status": "completed", "module_sha256": module_sha,
                  "outputs": {"gpu_native": archive_sha},
                  "summary_outputs": {"report.json": digest(report_path), "report.md": digest(report_md),
                                      "report.csv": digest(report_csv)}}
    completion_path = tmp_path / "completion.json"
    write_json(completion_path, completion)
    independent = {"status": "independent_saved_output_verification_pass", "completion": completion,
                   "summary_checksums": "pass", "targets_equal_original_archive": "pass",
                   "all_report_per_year_errors_and_combined_tolerances": "pass",
                   "comparisons": {name: {"probes": 8, "probability_pass": 8, "embedding_pass": 8}
                                   for name in ("gpu_4096_vs_archive", "cpu_gpu_position_8_vs_archive")}}
    independent_path = tmp_path / "independent-verification.public.json"
    write_json(independent_path, independent)
    trusted = {"completion_sha256": digest(completion_path), "report_sha256": digest(report_path),
               "independent_sha256": digest(independent_path), "module_sha256": module_sha,
               "gpu_native_sha256": archive_sha, "current_pilot_source_sha256": source_sha}

    result = load_r1_diagnostic_positional_state(tmp_path, expected_inputs, trusted_hashes=trusted)
    np.testing.assert_array_equal(result["positional_tensor"], tensor)
    assert result["provenance"]["origin"] == "validated_r1_gpu_diagnostic"
    assert result["provenance"]["historical_environment"] == "unknown"
    stale_inputs = {**expected_inputs, "actual": "stale-input"}
    with pytest.raises(ValueError, match="fingerprints are stale"):
        load_r1_diagnostic_positional_state(tmp_path, stale_inputs, trusted_hashes=trusted)
    (tmp_path / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact fingerprint mismatch"):
        load_r1_diagnostic_positional_state(tmp_path, expected_inputs, trusted_hashes=trusted)


def test_real_public_diagnostic_metadata_reaches_guarded_tensor_boundary(monkeypatch):
    import json
    from pathlib import Path
    import temporal_input_attribution as module

    evidence = Path("stats/concept_agreement_study/extensions/temporal_input_attribution/generated_state/test-evidence/r1-public")
    completion = evidence / "completion.json"
    report = evidence / "report.json"
    independent = evidence / "independent-verification.public.json"
    archive = evidence / "gpu_native.npz"
    assert all(path.is_file() for path in (completion, report, independent))
    archive.touch(exist_ok=True)  # Empty placeholder; the private archive is never opened or copied.
    expected_inputs = json.loads(report.read_text())["input_fingerprints"]["input_files"]
    original_hash = module._file_sha256
    def metadata_hash(path):
        if Path(path).name == "gpu_native.npz":
            return module._DIAGNOSTIC_TRUSTED_HASHES["gpu_native_sha256"]
        return original_hash(path)
    monkeypatch.setattr(module, "_file_sha256", metadata_hash)
    def guarded_load(*args, **kwargs):
        raise RuntimeError("guarded tensor-array access reached")
    monkeypatch.setattr(module.np, "load", guarded_load)
    with pytest.raises(RuntimeError, match="guarded tensor-array access reached"):
        module.load_r1_diagnostic_positional_state(
            evidence, expected_inputs, independent_verification_path=independent)


def test_execution_report_compares_saved_pilot_hash_to_current_pilot_source(tmp_path, monkeypatch):
    import copy
    import json
    import concept_subgroup_drift_pilot as pilot
    import temporal_input_attribution as module

    def write_report(saved_hash, expected_match, name):
        audit = {"common": {"files": {"pilot_source": {"status": "available",
                                                           "sha256": "current-pilot-source"}}},
                 "systems": [{"build": "r1", "reference": 2007,
                              "saved_source_hashes": {"concept_subgroup_drift_pilot.py": saved_hash},
                              "saved_environment": {"packages": {}}, "current_checkpoint_matches": True}],
                 "coverage": []}
        monkeypatch.setattr(module, "audit_frozen_inputs", lambda *args: copy.deepcopy(audit))
        monkeypatch.setattr(pilot, "load_pilot_population", lambda *args: {})
        output = tmp_path / name
        module._check_execution_paths("cpu", ledger=tmp_path / "ledger.csv", output=output, systems=[])
        saved = json.loads((output / "provenance.json").read_text())
        assert saved["systems"][0]["current_pilot_source_matches"] is expected_match

    write_report("current-pilot-source", True, "matching")
    write_report("old-pilot-source", False, "changed")


def test_execution_check_producer_requires_complete_matching_identity_pairs(tmp_path, monkeypatch):
    import copy
    import temporal_input_attribution as module
    import concept_subgroup_drift_pilot as pilot

    def identity(opaque, clinical="clinical"):
        return {"opaque_identity": opaque, "device": "cpu", "system_identity": "last_3/2013",
                "clinical_input_identity": clinical, "context_rows_identity": "rows",
                "checkpoint_sha256": "checkpoint", "learned_parameter_identity": "learned",
                "configuration_identity": "config", "clinical_transform_identity": "transform",
                "fitted_fields_identity": "fitted-fields",
                "precision": {"dtype": "fp32", "amp": "off", "tf32": False},
                "positional_state": {"sha256": "position", "shape": [3, 2], "dtype": "float32"},
                "execution_state": {"eval": True, "depth": 12, "dropout": 0.0}}

    matching = [identity("historical"), identity("normalized")]
    mismatched = [copy.deepcopy(matching[0]), identity("normalized", clinical="changed")]
    incomplete = [copy.deepcopy(matching[0]), {"opaque_identity": "normalized"}]
    audit = {"systems": [{"build": "last_3", "reference": 2013,
                           "current_checkpoint_matches": True, "saved_source_hashes": {},
                           "saved_environment": {}}],
             "common": {"files": {"pilot_source": {"status": "available", "sha256": "pilot"}}},
             "coverage": []}
    monkeypatch.setattr(module, "audit_frozen_inputs", lambda *args: copy.deepcopy(audit))
    monkeypatch.setattr(pilot, "load_pilot_population", lambda *args: object())
    monkeypatch.setattr(module, "inventory_inputs", lambda *args: {"files": {"source": {"sha256": "source"}}})
    monkeypatch.setattr(module, "_generated_state_for_system", lambda *args, **kwargs: {
        "provenance": {"build": "last_3", "reference": 2013, "device": "cpu"}, "positional_tensor": object()})
    parity = [{"build": "last_3", "reference": 2013, "year": year, "input": arm,
               "parity": "pass",
               "reference_actual_held_control": "pass" if year == 2013 and arm == "held" else "not applicable"}
              for year in (2013, 2014) for arm in ("actual", "held")]
    monkeypatch.setattr(module, "check_system_parity", lambda *args: {
        "parity": copy.deepcopy(parity), "fit_seconds": 1.0, "peak_gpu_bytes": 0,
        "learned_parameter_sha256": "learned", "query_check_seconds": 0.0,
        "comparable_predictor_identity": copy.deepcopy(matching[0])})

    def check_with(pair, name):
        scope = [{"build": "last_3", "reference": 2013, "year": 2013, "arm": "actual", "device": "cpu", "sae_device": "cpu"},
                 {"build": "last_3", "reference": 2013, "year": 2014, "arm": "held", "device": "cpu", "sae_device": "cpu"}]
        replay = [{**row, "probability_within_tolerance": True,
                   "embedding_within_tolerance": True, "saved_code_serialization_match": True}
                  for row in scope]
        scoped = {"proof": replay, "scope": scope, "held_actual_precheck_pass": True,
                  "actual_batch_rows": 4096, "held_batch_rows": 64, "held_precheck_rows": 32}
        if pair is not None:
            scoped["comparable_predictor_identity"] = pair[1] if len(pair) > 1 else pair[0]
        monkeypatch.setattr(module, "_bounded_scoped_replay", lambda *args, **kwargs: copy.deepcopy(scoped))
        monkeypatch.setattr(module, "validate_scoped_replay_evidence", lambda *args: {
            "conditional_numerical_reconstruction": "pass"})
        return module._check_execution_paths(
            "cpu", ledger=tmp_path / "ledger.csv", output=tmp_path / name)

    matching_checks = check_with(matching, "matching")
    assert matching_checks["ready_for_scoring"] is True, matching_checks
    mismatched_checks = check_with(mismatched, "mismatched")
    assert mismatched_checks["ready_for_scoring"] is False
    assert mismatched_checks["comparable_predictor_identities_match"] is False
    incomplete_checks = check_with(incomplete, "incomplete")
    assert incomplete_checks["ready_for_scoring"] is False
    assert incomplete_checks["comparable_predictor_identities_match"] is False
    assert check_with(None, "missing")["ready_for_scoring"] is False


def test_public_failure_does_not_include_private_exception_text():
    import json
    from temporal_input_attribution import sanitize_public_failure

    failure = sanitize_public_failure(ValueError("PRIVATE-PATIENT-YEAR-SENTINEL"))
    assert failure["error_type"] == "ValueError"
    assert "PRIVATE-PATIENT-YEAR-SENTINEL" not in json.dumps(failure)


def test_defaults_use_cpu_primary_and_reject_robustness_before_input_loading(monkeypatch, tmp_path):
    import inspect
    import temporal_input_attribution as module
    assert inspect.signature(module.check_execution_paths).parameters["device"].default == "cpu"
    assert inspect.signature(module.run_comparison).parameters["device"].default == "cpu"
    monkeypatch.setattr(module, "_comparison_inputs", lambda *args: pytest.fail("robustness loaded"))
    with pytest.raises(ValueError, match="last_3"):
        module.run_comparison(variants=("all_history",), output=tmp_path / "dropped")


def test_mixed_policy_routes_only_r1_and_rejects_invalid_requests():
    import inspect
    import temporal_input_attribution as module

    assert inspect.signature(module.check_execution_paths).parameters["r1_device"].default == "cpu"
    assert inspect.signature(module.run_comparison).parameters["r1_device"].default == "cpu"
    assert "checks_dir" in inspect.signature(module.run_comparison).parameters
    assert module._system_device({"build": "r1", "reference": 2007}, "cpu", "cuda") == "cuda"
    assert module._system_device({"build": "last_3", "reference": 2013}, "cpu", "cuda") == "cpu"
    with pytest.raises(ValueError):
        module._system_device({"build": "r1", "reference": 2007}, "cpu", "auto")


def test_mixed_policy_rejects_wrong_or_missing_device_evidence():
    from copy import deepcopy
    import temporal_input_attribution as module

    system = {"build": "r1", "reference": 2007}
    policy = {"r1/2007": "cuda"}
    states = {"r1/2007": {"device": "cuda"}}
    identities = {"r1/2007": [{"device": "cuda:0"}, {"device": "cuda:0"}]}
    scope = [{"build": "r1", "reference": 2007, "device": "cuda"}]
    parity = [{"build": "r1", "reference": 2007, "device": "cuda"}]
    module._validate_device_evidence([system], policy, states, identities, scope, parity, "cpu", "cuda")
    for changed in ("policy", "states", "identities", "scope", "parity"):
        values = deepcopy([policy, states, identities, scope, parity])
        index = {"policy": 0, "states": 1, "identities": 2, "scope": 3, "parity": 4}[changed]
        if changed == "policy":
            values[index]["r1/2007"] = "cpu"
        elif changed == "states":
            values[index]["r1/2007"]["device"] = "cpu"
        elif changed == "identities":
            values[index]["r1/2007"][0]["device"] = "cpu"
        else:
            values[index][0].pop("device")
        with pytest.raises(RuntimeError, match="device"):
            module._validate_device_evidence([system], *values, "cpu", "cuda")


def test_sae_device_policy_rejects_missing_or_stale_evidence():
    import inspect
    import temporal_input_attribution as module

    assert inspect.signature(module.check_execution_paths).parameters["sae_device"].default == "cpu"
    assert inspect.signature(module.run_comparison).parameters["sae_device"].default == "cpu"
    systems = [{"build": "last_3", "reference": 2008}]
    policy = {"last_3/2008": "cuda"}
    scope = [{"build": "last_3", "reference": 2008, "sae_device": "cuda"}]
    module._validate_sae_device_evidence(systems, policy, scope, "cuda")
    with pytest.raises(RuntimeError, match="SAE device"):
        module._validate_sae_device_evidence(systems, {"last_3/2008": "cpu"}, scope, "cuda")
    with pytest.raises(RuntimeError, match="SAE device"):
        module._validate_sae_device_evidence(systems, policy, [{"build": "last_3", "reference": 2008}], "cuda")


def test_replay_proof_rejects_missing_or_tampered_sae_device():
    from copy import deepcopy
    import temporal_input_attribution as module

    _, _, _, replay, scope = _conditional_check_fixture("cpu")
    for row in scope + replay:
        row["sae_device"] = "cuda"
    module.validate_scoped_replay_evidence(replay, scope)
    for replacement in (None, "cpu"):
        changed = deepcopy(replay)
        if replacement is None:
            changed[-1].pop("sae_device")
        else:
            changed[-1]["sae_device"] = replacement
        with pytest.raises(ValueError, match="SAE device"):
            module.validate_scoped_replay_evidence(changed, scope)


def test_scoring_uses_one_sae_device_for_archive_and_all_four_cells(tmp_path, monkeypatch):
    import concept_subgroup_drift_pilot as pilot
    import temporal_input_attribution as module

    inputs, system, _ = _synthetic_scoring_fixture(tmp_path, monkeypatch, batch_sensitive=False)
    devices = []
    def encode(dictionary, values, device):
        devices.append(device)
        return np.asarray(values, dtype=np.float32) * 2
    monkeypatch.setattr(pilot, "encode_with_reference_sae", encode)
    module._score_system(inputs, system, "cpu", sae_device="cuda")
    assert devices == ["cuda"] * 15


def _synthetic_scoring_fixture(tmp_path, monkeypatch, batch_sensitive, held_sensitive=False):
    import copy
    import json
    from contextlib import nullcontext
    import pandas as pd
    import torch
    import concept_subgroup_drift_pilot as pilot
    import tabpfn.best_models as best_models
    import tabpfn_model
    import temporal_input_attribution as module

    # Two reference-year and 64 later query rows force a partial final batch.
    n = 66
    features = np.arange(n + 1, dtype=np.float32).reshape(-1, 1) / 100
    years = np.array([2007, 2007, 2007] + [2008] * 64)
    patients = np.array([f"synthetic-{i}" for i in range(n + 1)])
    population = {"X": features, "year": years, "y": np.arange(n + 1) % 2,
                  "patient_id": patients, "post_death": np.zeros(n + 1, dtype=bool)}
    calls = []

    def response(x, batch, supplied_years=None):
        is_held = supplied_years is not None and np.all(np.asarray(supplied_years) == 2007)
        marker = batch / 1000 if batch_sensitive or (held_sensitive and is_held) else 0
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        probability = .2 + x / 100 + marker
        embedding = np.column_stack((x * 2 + marker, x * 3 + marker)).astype(np.float32)
        return probability.astype(np.float32), embedding

    archive_p, archive_e = response(features, n + 1)
    later_p, later_e = response(features[3:], 64, np.full(64, 2007))
    root = tmp_path / "run"
    (root / "out").mkdir(parents=True)
    config_path = root / "out/resolved-config.json"
    config_path.write_text(json.dumps({"split_seed": 42, "context_window": "reference_only", "config_hash": "fixture"}))
    actual_path, held_path, archive_path = tmp_path / "actual.npz", tmp_path / "held.npz", tmp_path / "predictor.npz"
    np.savez(actual_path, patient_id=patients[1:], year=years[1:], label=population["y"][1:].astype(np.float32),
             features=features[1:], feature_names=np.array(["clinical"]), probability=archive_p[1:])
    np.savez(held_path, positions=np.arange(2, n), patient_id=patients[3:], year=years[3:],
             probability_swapped=later_p, **{f"codes_{seed}": (later_e * 2).astype(np.float16) for seed in (42, 135, 215)})
    np.savez(archive_path, row_index=np.arange(n + 1), probabilities=np.column_stack((1 - archive_p, archive_p)), embeddings=archive_e)

    roles = {"reference_year": 2007, "row_role": np.array(["tabpfn_context"] + ["evaluation"] * n),
             "is_entrant": np.zeros(n + 1, dtype=bool)}
    monkeypatch.setattr(pilot, "reference_year_roles", lambda *args, **kwargs: roles)
    expected_pair = []
    history_model = type("IdentityModel", (), {})()
    def fit_history(context, columns, system, device, seed, positional_embeddings, **kwargs):
        calls.append(("history_device", device))
        calls.append(("prescribed_state", module._numpy_tensor_sha256(positional_embeddings.numpy())))
        return {"context_identity": "context", "checkpoint_sha256": "checkpoint", "context_rows": system["context_rows_identity"],
                "classifier": history_model}
    monkeypatch.setattr(module, "_fit_frozen_predictor_with_state", fit_history)
    monkeypatch.setattr(pilot, "configure_predictor_precision", lambda device: {})
    monkeypatch.setattr(pilot, "predictor_context_identity", lambda classifier: "context")
    monkeypatch.setattr(pilot, "predictor_sdpa_context", lambda device: nullcontext())
    monkeypatch.setattr(pilot, "load_reference_dictionaries", lambda *args: {seed: seed for seed in (42, 135, 215)})
    monkeypatch.setattr(pilot, "encode_with_reference_sae", lambda dictionary, x, device: np.asarray(x, dtype=np.float32) * 2)

    def capture_history(system, frame, device):
        calls.append(("historical", frame["f0"].to_numpy().copy(), frame["year"].to_numpy().copy()))
        p, embedding = response(frame["f0"].to_numpy(), len(frame), frame["year"].to_numpy())
        return {"probabilities": torch.from_numpy(np.column_stack((1 - p, p))), "embeddings": torch.from_numpy(embedding)}
    monkeypatch.setattr(pilot, "capture_frozen_predictions", capture_history)

    class FakeClassifier:
        def __init__(self):
            self.model = torch.nn.Linear(1, 1)
            self.cached_models_ = [self.model]
            self.comparable_identity = copy.deepcopy(expected_pair[0])
        def init_model_and_get_model_config(self):
            pass
        def fit(self, *args, **kwargs):
            return self
        def predict_full(self, x, additional_x, get_additional_outputs):
            domains = np.asarray(additional_x["dist_shift_domain"]).reshape(-1)
            calls.append(("consistent", np.asarray(x)[:, 0].copy(), domains.copy() + 2007))
            p, embedding = response(x[:, 0], len(x), domains + 2007)
            return {"proba": np.column_stack((1 - p, p)), "test_embeddings": torch.from_numpy(embedding[:, None, :])}
    monkeypatch.setattr(best_models, "get_best_tabpfn", lambda **kwargs: FakeClassifier())
    def prepare_with_state(classifier, positional_embeddings=None):
        calls.append(("prescribed_state", None if positional_embeddings is None else
                      module._numpy_tensor_sha256(positional_embeddings.detach().cpu().numpy())))
        return classifier
    monkeypatch.setattr(module, "prepare_context_classifier", prepare_with_state)
    monkeypatch.setattr(module, "_comparable_predictor_identity",
                        lambda pilot, classifier, context_rows_identity, system_identity="unknown":
                        classifier.comparable_identity)
    monkeypatch.setattr(module, "capture_same_call", lambda classifier, model, predict, identity, continuation: {
        "probabilities": predict_result["proba"], "embeddings": predict_result["test_embeddings"]}
        if (predict_result := predict()) is not None else None)
    monkeypatch.setattr(tabpfn_model, "infer_model_additional_x_info", lambda classifier: ("cpu", (1, 1, 1)))
    monkeypatch.setattr(tabpfn_model, "make_dist_tensor", lambda domains, device, shape: domains)
    system = {"reference": 2007, "build": "r1", "domain_origin": 2007, "saved_device": "cpu",
              "saved_context_identity": "context", "saved_checkpoint_sha256": "checkpoint",
              "context_rows_identity": "rows",
              "files": {name: {"path": str(path)} for name, path in
                        (("config", config_path), ("actual", actual_path), ("held", held_path), ("predictor_archive", archive_path))}}
    state_tensor = torch.tensor([[.25, -.5]], dtype=torch.float32)
    state_sha = module._numpy_tensor_sha256(state_tensor.numpy())
    state_provenance = {"build": "r1", "reference": 2007, "origin": "native_cpu_reconstruction_candidate",
                        "device": "cpu", "source_sha256": "fixture-source", "input_sha256": "fixture-input",
                        "input_fingerprints": {"actual": "fixture-actual"},
                        "positional_state": {"sha256": state_sha, "shape": [1, 2], "dtype": "torch.float32"}}
    _, _, _, _, _, _, binding, _ = module._current_scoring_context(pilot, population, system,
                                                                    state_provenance["positional_state"])
    system["context_rows_identity"] = binding["context_rows_identity"]
    def expected_identity(opaque):
        return {**binding, "opaque_identity": opaque, "device": "cpu",
                "learned_parameter_identity": "learned-parameters", "configuration_identity": "config",
                "clinical_transform_identity": "transform", "fitted_fields_identity": "all-diagnostic-fields",
                "precision": {"dtype": "fp32", "amp": "off", "tf32": False},
                "execution_state": {"eval": True, "depth": 12, "dropout": 0.0}}
    expected_pair.extend((expected_identity("consistent-opaque"), expected_identity("historical-opaque")))
    history_model.comparable_identity = copy.deepcopy(expected_pair[1])
    inputs = {"population": population,
              "source_sha256": "fixture-source",
              "current_input_fingerprints": {"r1/2007": {"actual": "fixture-actual"}},
              "expected_comparable_predictor_identities": {"r1/2007": copy.deepcopy(expected_pair)},
              "fresh_identity_models": {"historical": history_model},
              "generated_state_provenance": {"r1/2007": {"positional_tensor": state_tensor,
                                                              "provenance": state_provenance}}}
    return inputs, system, calls


def test_real_system_scorer_matches_all_fresh_batch_schedules_and_activations(tmp_path, monkeypatch):
    import temporal_input_attribution as module
    inputs, system, calls = _synthetic_scoring_fixture(tmp_path, monkeypatch, batch_sensitive=False)
    scored = module._score_system(inputs, system, "cpu")
    historical = [call for call in calls if call[0] == "historical"]
    consistent = [call for call in calls if call[0] == "consistent"]
    # The held archive replay is separate; the last four historical calls are the scientific cells.
    assert [len(call[1]) for call in historical[-4:]] == [64, 2, 64, 2]
    assert [len(call[1]) for call in consistent] == [64, 2, 64, 2]
    state_sha = inputs["generated_state_provenance"]["r1/2007"]["provenance"]["positional_state"]["sha256"]
    assert [call[1] for call in calls if call[0] == "prescribed_state"] == [state_sha, state_sha]
    for h, n in zip(historical[-4:], consistent):
        np.testing.assert_array_equal(h[1], n[1])
        np.testing.assert_array_equal(h[2], n[2])
    np.testing.assert_array_equal(scored["historical_actual_codes_42"], scored["historical_actual_embedding"] * 2)
    path = tmp_path / "stats/concept_agreement_study/patient_table/concepts"
    path.mkdir(parents=True)
    np.savez(path / "fixture.npz", activation=np.full(66, 99, np.float32), in_R=np.ones(66, bool))
    monkeypatch.chdir(tmp_path)
    concept = {"file": "fixture.npz", "seed": 42, "factor": 0, "concept": "full-key", "cutoff": .5}
    cells = module._concept_cells(inputs, system, concept, scored)
    np.testing.assert_array_equal(cells[2007][("historical", "actual")]["activation"],
                                  scored["historical_actual_codes_42"][:2, 0])
    assert module._archive_activation_diagnostics(concept, scored)[2007]["strict_cutoff_disagreements"] == 2


def test_real_system_scorer_rejects_batch_sensitive_archive_mismatch(tmp_path, monkeypatch):
    import temporal_input_attribution as module
    inputs, system, calls = _synthetic_scoring_fixture(tmp_path, monkeypatch, batch_sensitive=True)
    with pytest.raises(RuntimeError, match="historical actual.*archive"):
        module._score_system(inputs, system, "cpu")


def test_scorer_requires_expected_identity_pair_before_fitting(tmp_path, monkeypatch):
    import temporal_input_attribution as module
    inputs, system, calls = _synthetic_scoring_fixture(tmp_path, monkeypatch, batch_sensitive=False)
    inputs["expected_comparable_predictor_identities"].clear()
    with pytest.raises(RuntimeError, match="expected fitted predictor identity proof"):
        module._score_system(inputs, system, "cpu")
    assert not any(call[0] == "history_device" for call in calls)


def test_scorer_rechecks_fresh_historical_and_consistent_fitted_identities(tmp_path, monkeypatch):
    import temporal_input_attribution as module
    inputs, system, calls = _synthetic_scoring_fixture(tmp_path / "second", monkeypatch, batch_sensitive=False)
    inputs["fresh_identity_models"]["historical"].comparable_identity["fitted_fields_identity"] = "tampered-y-or-setting"
    with pytest.raises(RuntimeError, match="fresh historical fitted predictor identity"):
        module._score_system(inputs, system, "cpu")

    inputs, system, calls = _synthetic_scoring_fixture(tmp_path, monkeypatch, batch_sensitive=False)
    original_initialize = module._initialize_context_classifier
    def initialize_with_changed_setting(system, device):
        classifier, config = original_initialize(system, device)
        classifier.comparable_identity["fitted_fields_identity"] = "tampered-softmax-temperature"
        return classifier, config
    monkeypatch.setattr(module, "_initialize_context_classifier", initialize_with_changed_setting)
    with pytest.raises(RuntimeError, match="fresh consistent fitted predictor identity"):
        module._score_system(inputs, system, "cpu")


def test_fresh_historical_scoring_uses_requested_cpu_even_when_archive_used_cuda(tmp_path, monkeypatch):
    import temporal_input_attribution as module
    inputs, system, calls = _synthetic_scoring_fixture(tmp_path, monkeypatch, batch_sensitive=False)
    system["saved_device"] = "cuda"
    scored = module._score_system(inputs, system, "cpu")
    assert ("history_device", "cpu") in calls
    assert scored["archived_historical_device"].item() == "cuda"
    assert scored["historical_replay_device"].item() == "cpu"
    assert scored["historical_device_identity"].item() == "unresolved: recorded and replay devices differ"


def test_fresh_held_batch_must_match_separate_historical_held_replay(tmp_path, monkeypatch):
    import temporal_input_attribution as module
    inputs, system, calls = _synthetic_scoring_fixture(tmp_path, monkeypatch, batch_sensitive=False, held_sensitive=True)
    with pytest.raises(RuntimeError, match="historical held fresh.*replay"):
        module._score_system(inputs, system, "cpu")


def test_bounded_scoped_producer_checks_original_actual_and_candidate_held_batches(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import concept_subgroup_drift_pilot as pilot
    import temporal_input_attribution as module

    years = np.array([2007] + [2008 + index % 7 for index in range(64)], dtype=np.int64)
    patients = np.array([f"bounded-{index}" for index in range(65)])
    features = np.arange(65, dtype=np.float32).reshape(-1, 1) / 100
    population = {"patient_id": patients, "year": years, "X": features,
                  "feature_names": np.array(["f0"]), "y": (np.arange(65) % 2).astype(np.float32),
                  "post_death": np.zeros(65, dtype=bool)}
    actual_probability = .2 + features[:, 0]
    embeddings = np.column_stack((features[:, 0] * 2, features[:, 0] * 3)).astype(np.float32)
    actual_path, held_path, predictor_path = (tmp_path / name for name in
                                               ("actual.npz", "held.npz", "predictor.npz"))
    np.savez(actual_path, patient_id=patients, year=years, label=population["y"], features=features,
             feature_names=np.array(["f0"]), probability=actual_probability)
    held_rows = np.arange(1, 65)
    held_years = years[held_rows]
    held_probabilities = .2 + features[held_rows, 0]
    held_embeddings = embeddings[held_rows]
    np.savez(held_path, positions=held_rows, patient_id=patients[held_rows], year=held_years,
             probability_swapped=held_probabilities,
             **{f"codes_{seed}": (held_embeddings * 2).astype(np.float16) for seed in (42, 135, 215)})
    np.savez(predictor_path, row_index=np.arange(65), embeddings=embeddings)
    config_path = tmp_path / "resolved-config.json"
    config_path.write_text(json.dumps({"split_seed": 42, "context_window": "reference_only",
                                      "config_hash": "fixture"}))
    roles = {"row_role": np.array(["tabpfn_context"] + ["evaluation"] * 64),
             "is_entrant": np.zeros(65, dtype=bool)}
    monkeypatch.setattr(pilot, "reference_year_roles", lambda *args, **kwargs: roles)
    monkeypatch.setattr(module, "_fit_frozen_predictor_with_state", lambda *args, **kwargs:
                        {"classifier": object()})
    monkeypatch.setattr(pilot, "load_reference_dictionaries", lambda *args: {seed: seed for seed in (42, 135, 215)})
    monkeypatch.setattr(pilot, "encode_with_reference_sae", lambda dictionary, values, device:
                        np.asarray(values, dtype=np.float32) * 2)
    def capture(system, frame, device):
        values = frame["f0"].to_numpy(dtype=np.float32)
        probability = .2 + values
        output_embeddings = np.column_stack((values * 2, values * 3)).astype(np.float32)
        return {"probabilities": np.column_stack((1 - probability, probability)).astype(np.float32),
                "embeddings": output_embeddings}
    monkeypatch.setattr(pilot, "capture_frozen_predictions", capture)
    monkeypatch.setattr(module, "_comparable_predictor_identity", lambda *args: {"fixture": "identity"})
    system = {"build": "r1", "reference": 2007, "domain_origin": 2007, "saved_batch_rows": 65,
              "context_rows_identity": "rows",
              "files": {name: {"path": str(path)} for name, path in
                        (("actual", actual_path), ("held", held_path), ("predictor_archive", predictor_path),
                         ("config", config_path))}}
    state = {"source_sha256": "source", "state_source_sha256": "state-source", "input_sha256": "input",
             "positional_state": {"sha256": "position"}}
    result = module._bounded_scoped_replay(system, population, "cpu", None, state)
    assert len(result["scope"]) == 15  # Eight actual probes plus seven later-year held probes.
    assert result["held_actual_precheck_pass"] is True
    actual = [row for row in result["scope"] if row["arm"] == "actual"]
    held = [row for row in result["scope"] if row["arm"] == "held"]
    assert all(row["schedule_certainty"] == "exact" and row["rows"] == 65 for row in actual)
    assert all(row["schedule_certainty"] == "candidate" and row["rows"] == 64 for row in held)
    assert module.validate_scoped_replay_evidence(result["proof"], result["scope"])[
        "conditional_numerical_reconstruction"] == "pass"
    tampered = [dict(row) for row in result["proof"]]
    tampered[-1]["schedule_id"] = "actual-64"
    with pytest.raises(ValueError, match="scope"):
        module.validate_scoped_replay_evidence(tampered, result["scope"])

    # Drive the real bounded producer and outer persistence path with synthetic inputs.
    # A bad probability must retain its computed proof rows while readiness remains false.
    with np.load(actual_path, allow_pickle=False) as archive:
        actual_arrays = {name: archive[name].copy() for name in archive.files}
    actual_arrays["probability"] = actual_arrays["probability"] + np.float32(0.25)
    np.savez(actual_path, **actual_arrays)
    audit_system = {**system, "current_checkpoint_matches": True, "saved_source_hashes": {},
                    "saved_environment": {}, "context_rows_identity": "synthetic-context"}
    audit = {"systems": [audit_system], "common": {"files": {
                "pilot_source": {"status": "available", "sha256": "synthetic-pilot"}}},
             "coverage": []}
    monkeypatch.setattr(module, "audit_frozen_inputs", lambda *args: audit)
    monkeypatch.setattr(pilot, "load_pilot_population", lambda *args: population)
    monkeypatch.setattr(module, "inventory_inputs", lambda *args: {"files": {"source": {"sha256": "synthetic-source"}}})
    monkeypatch.setattr(module, "_generated_state_for_system", lambda *args, **kwargs: {
        "provenance": state, "positional_tensor": None})
    all_parity = [{"build": "r1", "reference": 2007, "year": year, "input": arm,
                   "parity": "pass",
                   "reference_actual_held_control": "pass" if year == 2007 and arm == "held" else "not applicable"}
                  for year in range(2007, 2015) for arm in ("actual", "held")]
    monkeypatch.setattr(module, "check_system_parity", lambda *args: {
        "parity": all_parity, "fit_seconds": 0.0, "peak_gpu_bytes": 0,
        "learned_parameter_sha256": "synthetic-learned", "query_check_seconds": 0.0,
        "comparable_predictor_identity": {"fixture": "identity"}})
    mismatch_output = tmp_path / "mismatch-output"
    mismatch_checks = module._check_execution_paths(
        "cpu", ledger=tmp_path / "ledger.csv", output=mismatch_output, systems=[("r1", 2007)])
    mismatch_report = json.loads((mismatch_output / "provenance.json").read_text())
    assert mismatch_checks["ready_for_scoring"] is False
    assert len(mismatch_report["replay_scope"]) == 15
    mismatch_proof = json.loads((mismatch_output / "scoped-replay.json").read_text())
    assert len(mismatch_proof) == 15
    mismatch_keys = {(row["build"], row["reference"], row["year"], row["arm"])
                     for row in mismatch_report["replay_scope"]}
    assert len(mismatch_keys) == 15
    mismatch_error = mismatch_report["checks"]["errors"][0]
    assert mismatch_error["bounded_stage"] == "final_scope_validation_started"
    assert mismatch_error["actual_proof_rows"] == 8 and mismatch_error["held_proof_rows"] == 7
    assert mismatch_report["checks"]["errors"][0]["message"] == (
        "bounded check failed; private exception details are not persisted")

    # A later synthetic exception must preserve the eight completed actual rows and
    # never serialize its patient/year sentinel.
    import r1_2007_replay as replay_helpers
    original_held_selector = replay_helpers.select_held_batch
    def fail_held_selection(*args, **kwargs):
        raise ValueError("PRIVATE-PATIENT-YEAR-SENTINEL")
    monkeypatch.setattr(replay_helpers, "select_held_batch", fail_held_selection)
    partial_output = tmp_path / "partial-output"
    partial_checks = module._check_execution_paths(
        "cpu", ledger=tmp_path / "ledger.csv", output=partial_output, systems=[("r1", 2007)])
    partial_report = json.loads((partial_output / "provenance.json").read_text())
    serialized_partial = json.dumps(partial_report)
    assert partial_checks["ready_for_scoring"] is False
    assert len(partial_report["replay_scope"]) == 8
    assert all(row["arm"] == "actual" for row in partial_report["replay_scope"])
    assert "PRIVATE-PATIENT-YEAR-SENTINEL" not in serialized_partial
    partial_error = partial_report["checks"]["errors"][0]
    assert partial_error["bounded_stage"] == "held_selection_started"
    assert partial_error["actual_proof_rows"] == 8 and partial_error["held_proof_rows"] == 0
    assert partial_error["message"] == (
        "bounded check failed; private exception details are not persisted")
    monkeypatch.setattr(replay_helpers, "select_held_batch", original_held_selector)

    # A nonfinite archived held probability must remain a failed, strict-JSON proof.
    with np.load(held_path, allow_pickle=False) as archive:
        held_arrays = {name: archive[name].copy() for name in archive.files}
    held_arrays["probability_swapped"][0] = np.nan
    np.savez(held_path, **held_arrays)
    nonfinite_output = tmp_path / "nonfinite-output"
    nonfinite_checks = module._check_execution_paths(
        "cpu", ledger=tmp_path / "ledger.csv", output=nonfinite_output, systems=[("r1", 2007)])
    nonfinite_report = json.loads((nonfinite_output / "provenance.json").read_text())
    nonfinite_proof = json.loads((nonfinite_output / "scoped-replay.json").read_text())
    held_proof = [row for row in nonfinite_proof if row["arm"] == "held"]
    assert nonfinite_checks["ready_for_scoring"] is False
    assert len(nonfinite_report["replay_scope"]) == 15
    assert any(row["probability_within_tolerance"] is False
               and row["probability_max_abs_error"] is None for row in held_proof)
    assert "NaN" not in (nonfinite_output / "scoped-replay.json").read_text()

    # The same unavailable-error representation covers actual probabilities and embeddings.
    with np.load(actual_path, allow_pickle=False) as archive:
        actual_arrays = {name: archive[name].copy() for name in archive.files}
    actual_arrays["probability"][0] = np.inf
    np.savez(actual_path, **actual_arrays)
    actual_nonfinite_output = tmp_path / "actual-nonfinite-output"
    actual_nonfinite_checks = module._check_execution_paths(
        "cpu", ledger=tmp_path / "ledger.csv", output=actual_nonfinite_output, systems=[("r1", 2007)])
    actual_nonfinite_proof = json.loads((actual_nonfinite_output / "scoped-replay.json").read_text())
    actual_nonfinite = [row for row in actual_nonfinite_proof
                        if row["arm"] == "actual" and row["year"] == 2007]
    assert actual_nonfinite_checks["ready_for_scoring"] is False
    assert len(actual_nonfinite) == 1
    assert actual_nonfinite[0]["probability_within_tolerance"] is False
    assert actual_nonfinite[0]["probability_max_abs_error"] is None

    with np.load(predictor_path, allow_pickle=False) as archive:
        predictor_arrays = {name: archive[name].copy() for name in archive.files}
    predictor_arrays["embeddings"][0, 0] = np.inf
    np.savez(predictor_path, **predictor_arrays)
    embedding_nonfinite_output = tmp_path / "embedding-nonfinite-output"
    embedding_nonfinite_checks = module._check_execution_paths(
        "cpu", ledger=tmp_path / "ledger.csv", output=embedding_nonfinite_output, systems=[("r1", 2007)])
    embedding_nonfinite_proof = json.loads((embedding_nonfinite_output / "scoped-replay.json").read_text())
    embedding_nonfinite = [row for row in embedding_nonfinite_proof
                           if row["arm"] == "actual" and row["year"] == 2007]
    assert embedding_nonfinite_checks["ready_for_scoring"] is False
    assert len(embedding_nonfinite) == 1
    assert embedding_nonfinite[0]["embedding_within_tolerance"] is False
    assert embedding_nonfinite[0]["embedding_max_abs_error"] is None


def test_packed_probes_cover_each_setting_and_compare_rows():
    import torch
    from temporal_input_attribution import pack_parity_probes, compare_packed_parity
    population = {"patient_id": np.array(["a", "b", "c"]), "year": np.array([2012, 2013, 2014]),
                  "X": np.array([[1.], [2.], [3.]], dtype=np.float32)}
    saved = {"patient_id": population["patient_id"], "year": population["year"]}
    probes = pack_parity_probes(saved, population, 2012, 2012)
    np.testing.assert_array_equal(probes["features"][:, 0], [1, 1, 2, 2, 3, 3])
    np.testing.assert_array_equal(probes["domains"], [0, 0, 1, 0, 2, 0])
    assert [(row["year"], row["input"]) for row in probes["settings"]] == [
        (2012, "actual"), (2012, "held"), (2013, "actual"), (2013, "held"), (2014, "actual"), (2014, "held")]
    assert probes["missing_years"] == []
    values = torch.tensor([[.1, .9], [.1, .9], [.2, .8], [.3, .7], [.4, .6], [.5, .5]])
    cached = {"probabilities": values.numpy(), "embeddings": values[:, None, :], "time_values": torch.tensor(probes["domains"]).reshape(-1, 1, 1)}
    joint = {"probabilities": values.clone().numpy(), "embeddings": values[:, None, :].clone(), "time_values": cached["time_values"].clone()}
    boundary = [{"inputs": [torch.from_numpy(probes["features"])[:, None, :]], "categorical_inds": [[0]]}]
    report = compare_packed_parity({"build": "b", "reference": 2012}, probes, cached, joint, boundary, boundary)
    assert len(report) == 6 and all(row["parity"] == "pass" for row in report)
    assert report[1]["reference_actual_held_control"] == "pass"
    joint["probabilities"][4, 1] += .1
    report = compare_packed_parity({"build": "b", "reference": 2012}, probes, cached, joint, boundary, boundary)
    assert report[4]["parity"] == "fail" and report[3]["parity"] == "pass"
    probes["settings"][2], probes["settings"][3] = probes["settings"][3], probes["settings"][2]
    with pytest.raises(ValueError, match="missing or reordered"):
        compare_packed_parity({"build": "b", "reference": 2012}, probes, cached, joint, boundary, boundary)
    saved = {"patient_id": population["patient_id"][:2], "year": population["year"][:2]}
    assert pack_parity_probes(saved, population, 2012, 2012)["missing_years"] == [2014]
