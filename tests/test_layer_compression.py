import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_info

import concept_subgroup_drift_pilot as pilot
import flchain_probe as probe
import layer_compression as lc
import prospective_audit as pa
import tcav_geometry as geometry

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE_PARAMS = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not CUDA_AVAILABLE, reason="CUDA unavailable on this host; the Fiji run executes this variant")),
]
FORWARD_TOL = {"rtol": 1e-4, "atol": 1e-5}
GRADIENT_TOL = {"rtol": 1e-3, "atol": 1e-5}
FD_STEP = 1e-2
FD_TOL = {"rtol": 1e-3, "atol": 1e-5}
PROBE_ROWS = 8
EMBEDDING_DIM = 192
LAST_LAYER = 11


def tiny_fit(device):
    rng = np.random.default_rng(11)
    years = np.tile([2007, 2008, 2009], 20)
    X = rng.normal(size=(60, 6)).astype(np.float32)
    y = np.tile([0, 1], 30)
    fitted = pa.fit_dr_tabpfn_context(X, y, years, domain_origin=2007, device=device, use_context_scaler=True)
    return fitted, X[:PROBE_ROWS], years[:PROBE_ROWS]


def hooked_layer_outputs(fitted, X, years):
    """One predict call; each of the 12 layer outputs is recorded for test rows (rows, tokens, dimension)."""
    layers = fitted["classifier"].cached_models_[0].transformer_encoder.layers
    recorded = [[] for _ in layers]
    handles = [layer.register_forward_hook(lambda module, args, output, index=index: recorded[index].append(output[0].clone()))
               for index, layer in enumerate(layers)]
    try:
        captured = probe.capture_flchain_batches(fitted, X, years, batch_size=len(X))
    finally:
        for handle in handles:
            handle.remove()
    return captured, [values[0] for values in recorded]


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_replay_matches_cached_forward(device):
    fitted, X, years = tiny_fit(device)
    captured, hooked = hooked_layer_outputs(fitted, X, years)
    np.testing.assert_allclose(hooked[LAST_LAYER][:, -1, :].cpu().numpy(), captured["embeddings"], **FORWARD_TOL)
    weights = lc.clone_replay_weights(fitted["classifier"])
    replay = lc.replay_layers(weights, hooked[0].clone(), 0)
    assert len(replay) == LAST_LAYER + 1
    for layer in range(1, LAST_LAYER + 1):
        torch.testing.assert_close(replay[layer], hooked[layer], **FORWARD_TOL)
    probability = lc.death_probability(captured["continuation"], replay[-1])
    np.testing.assert_allclose(probability.detach().cpu().numpy(), captured["probabilities"][:, 1], **FORWARD_TOL)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_layer_gradients_match_finite_differences(device):
    fitted, X, years = tiny_fit(device)
    captured, hooked = hooked_layer_outputs(fitted, X, years)
    continuation = lc.double_continuation(captured["continuation"])
    weights = lc.double_replay_weights(lc.clone_replay_weights(fitted["classifier"]))
    first = hooked[0].double()
    _, gradients = lc.layer_gradients(weights, continuation, first)
    replay = lc.replay_layers(weights, first, 0)
    rng = np.random.default_rng(7)
    for layer in range(LAST_LAYER + 1):
        direction = rng.normal(size=(PROBE_ROWS, EMBEDDING_DIM))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        direction = torch.as_tensor(direction, dtype=torch.float64, device=first.device)
        analytic = (gradients[layer] * direction).sum(dim=1)
        base = replay[layer].detach()
        plus, minus = base.clone(), base.clone()
        plus[:, -1, :] += FD_STEP * direction
        minus[:, -1, :] -= FD_STEP * direction
        p_plus = lc.death_probability(continuation, lc.replay_layers(weights, plus, layer)[-1]).double()
        p_minus = lc.death_probability(continuation, lc.replay_layers(weights, minus, layer)[-1]).double()
        finite = (p_plus - p_minus) / (2 * FD_STEP)
        torch.testing.assert_close(analytic.detach(), finite.detach(), **FD_TOL)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_layer_11_gradient_matches_pilot(device):
    fitted, X, years = tiny_fit(device)
    captured, hooked = hooked_layer_outputs(fitted, X, years)
    weights = lc.clone_replay_weights(fitted["classifier"])
    _, gradients = lc.layer_gradients(weights, captured["continuation"], hooked[0].clone())
    expected = pilot.batched_death_gradients(captured["continuation"], captured["embeddings"], batch_rows=PROBE_ROWS)
    np.testing.assert_allclose(gradients[LAST_LAYER].cpu().numpy(), expected, **GRADIENT_TOL)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_replay_rows_independent(device):
    fitted, X, years = tiny_fit(device)
    captured, hooked = hooked_layer_outputs(fitted, X, years)
    continuation = captured["continuation"]
    weights = lc.clone_replay_weights(fitted["classifier"])
    first = hooked[0].clone()
    row = 3
    batch_probability, batch_gradients = lc.layer_gradients(weights, continuation, first)
    alone_probability, alone_gradients = lc.layer_gradients(weights, continuation, first[row:row + 1])
    torch.testing.assert_close(alone_probability, batch_probability[row:row + 1], **FORWARD_TOL)
    for layer in range(LAST_LAYER + 1):
        torch.testing.assert_close(alone_gradients[layer], batch_gradients[layer][row:row + 1], **GRADIENT_TOL)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_replay_leaves_model_unchanged(device):
    fitted, X, years = tiny_fit(device)
    classifier = fitted["classifier"]
    layers = classifier.cached_models_[0].transformer_encoder.layers
    cache_before = [layer.self_attn_between_items._kv_cache.clone() for layer in layers]
    identity_before = pilot.predictor_context_identity(classifier)
    before, hooked = hooked_layer_outputs(fitted, X, years)
    weights = lc.clone_replay_weights(classifier)
    lc.layer_gradients(weights, before["continuation"], hooked[0].clone())
    after = probe.capture_flchain_batches(fitted, X, years, batch_size=PROBE_ROWS)
    assert pilot.predictor_context_identity(classifier) == identity_before
    np.testing.assert_array_equal(after["probabilities"], before["probabilities"])
    for layer, cache in zip(layers, cache_before):
        assert torch.equal(layer.self_attn_between_items._kv_cache, cache)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_capture_shapes_and_token_count(device):
    fitted, X, years = tiny_fit(device)
    classifier = fitted["classifier"]
    layers = classifier.cached_models_[0].transformer_encoder.layers
    captured, hooked = hooked_layer_outputs(fitted, X, years)
    layer_capture = lc.capture_layers(fitted, X, years, batch_size=PROBE_ROWS)
    tokens = lc.token_count(classifier)
    assert tokens == hooked[0].shape[1]
    assert layer_capture["target"].shape == (PROBE_ROWS, LAST_LAYER + 1, EMBEDDING_DIM)
    assert layer_capture["feature_mean"].shape == (PROBE_ROWS, LAST_LAYER + 1, EMBEDDING_DIM)
    assert layer_capture["layer0"].shape == (PROBE_ROWS, tokens, EMBEDDING_DIM)
    assert layer_capture["probabilities"].shape == (PROBE_ROWS, 2)
    torch.testing.assert_close(layer_capture["layer0"], hooked[0], **FORWARD_TOL)
    for index, layer_output in enumerate(hooked):
        np.testing.assert_allclose(layer_capture["target"][:, index], layer_output[:, -1, :].cpu().numpy(), **FORWARD_TOL)
        np.testing.assert_allclose(layer_capture["feature_mean"][:, index],
                                   layer_output[:, :-1, :].mean(dim=1).cpu().numpy(), **FORWARD_TOL)
    assert all(len(layer._forward_hooks) == 0 for layer in layers)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_capture_layer_11_matches_existing(device):
    fitted, X, years = tiny_fit(device)
    existing = probe.capture_flchain_batches(fitted, X, years, batch_size=PROBE_ROWS)
    layer_capture = lc.capture_layers(fitted, X, years, batch_size=PROBE_ROWS)
    np.testing.assert_allclose(layer_capture["target"][:, LAST_LAYER], existing["embeddings"], **FORWARD_TOL)
    np.testing.assert_allclose(layer_capture["probabilities"], existing["probabilities"], **FORWARD_TOL)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_capture_batching_gives_same_result(device):
    fitted, X, years = tiny_fit(device)
    whole = lc.capture_layers(fitted, X, years, batch_size=PROBE_ROWS)
    chunked = lc.capture_layers(fitted, X, years, batch_size=3)
    for key in ("target", "feature_mean", "probabilities"):
        np.testing.assert_allclose(chunked[key], whole[key], **FORWARD_TOL)
    torch.testing.assert_close(chunked["layer0"], whole["layer0"], **FORWARD_TOL)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_capture_layer0_feeds_replay(device):
    fitted, X, years = tiny_fit(device)
    layer_capture = lc.capture_layers(fitted, X, years, batch_size=PROBE_ROWS)
    weights = lc.clone_replay_weights(fitted["classifier"])
    replay = lc.replay_layers(weights, layer_capture["layer0"].clone(), 0)
    for layer in range(1, LAST_LAYER + 1):
        np.testing.assert_allclose(replay[layer][:, -1, :].detach().cpu().numpy(),
                                   layer_capture["target"][:, layer], **FORWARD_TOL)
        np.testing.assert_allclose(replay[layer][:, :-1, :].mean(dim=1).detach().cpu().numpy(),
                                   layer_capture["feature_mean"][:, layer], **FORWARD_TOL)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_hooks_removed_on_error(device):
    fitted, X, years = tiny_fit(device)
    layers = fitted["classifier"].cached_models_[0].transformer_encoder.layers
    with pytest.raises(ValueError, match="lengths differ"):
        lc.capture_layers(fitted, X, years[:-1], batch_size=PROBE_ROWS)
    assert all(len(layer._forward_hooks) == 0 for layer in layers)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_token_norm_is_sqrt_emsize(device):
    fitted, X, years = tiny_fit(device)
    layer_capture = lc.capture_layers(fitted, X, years, batch_size=PROBE_ROWS)
    expected = np.sqrt(EMBEDDING_DIM)
    np.testing.assert_allclose(np.linalg.norm(layer_capture["target"], axis=-1), expected, rtol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(layer_capture["layer0"].cpu().numpy(), axis=-1), expected, rtol=1e-4)


def synthetic_geometry_arrays(rng, rows=120, layers=3, dim=8):
    target = rng.normal(size=(rows, layers, dim))
    gradients = rng.normal(size=(rows, layers, dim)) * rng.uniform(0.5, 2.0, size=(rows, layers, 1))
    probabilities = rng.uniform(0.05, 0.95, size=rows)
    scale_rows = rng.normal(size=(80, layers, dim))
    feature_mean = rng.normal(size=(rows, layers, dim))
    scale_feature_mean = rng.normal(size=(80, layers, dim))
    return target, gradients, probabilities, scale_rows, feature_mean, scale_feature_mean


def test_layer_geometry_table_at_last_layer_matches_geometry_summary():
    rng = np.random.default_rng(21)
    target, gradients, probabilities, scale_rows, feature_mean, scale_feature_mean = synthetic_geometry_arrays(rng)
    table = lc.layer_geometry_table(target, gradients, probabilities, scale_rows,
                                    feature_mean=feature_mean, scale_feature_mean=scale_feature_mean)
    last = len(target[0]) - 1
    subset = table[(table["layer"] == last) & (table["token"] == "target")]
    expected = pa.flatten_geometry(pa.geometry_summary(target[:, last], gradients[:, last],
                                                        probabilities, scale_rows[:, last]))
    assert list(zip(subset["section"], subset["key"], subset["metric"])) == \
        list(zip(expected["section"], expected["key"], expected["metric"]))
    np.testing.assert_allclose(subset["value"].to_numpy(dtype=float), expected["value"].to_numpy(dtype=float),
                               rtol=1e-9, atol=1e-12, equal_nan=True)


def test_layer_geometry_table_feature_mean_has_spectrum_only_and_zero_column_count():
    rng = np.random.default_rng(22)
    target, gradients, probabilities, scale_rows, feature_mean, scale_feature_mean = synthetic_geometry_arrays(rng)
    scale_feature_mean[:, 1, 2] = 3.0
    table = lc.layer_geometry_table(target, gradients, probabilities, scale_rows,
                                    feature_mean=feature_mean, scale_feature_mean=scale_feature_mean)
    mean_rows = table[table["token"] == "feature_mean"]
    assert set(mean_rows["section"]) <= {"spectrum", "standardization_zero_columns"}
    assert set(mean_rows["layer"]) == {0, 1, 2}
    zero_counts = mean_rows[mean_rows["section"] == "standardization_zero_columns"].set_index("layer")["value"]
    assert zero_counts.loc[1] == 1 and zero_counts.loc[0] == 0
    target_rows = table[table["token"] == "target"]
    assert {"coherence", "dimension", "energy", "random"} <= set(target_rows["section"])


def test_unit_weight_metrics_equal_analyze_layer_geometry():
    rng = np.random.default_rng(23)
    emb = rng.normal(size=(120, 8))
    grad = rng.normal(size=(120, 8))
    grad[3] = 0.0
    grad[5, 2] = np.nan
    prob = rng.uniform(0.05, 0.95, size=120)
    prob[7] = 0.0
    scale = rng.uniform(0.5, 2.0, size=8)
    result = probe.analyze_layer_geometry(emb, grad, prob, scale, rng=np.random.default_rng(1))
    ones = np.ones(len(emb))
    spectrum = lc.weighted_spectrum(emb, scale, ones)
    for chart in ("raw", "standardized"):
        for metric in ("embedding_pr", "top1_var_share", "top5_var_share", "top10_var_share"):
            assert spectrum[chart][metric] == pytest.approx(result["spectrum"][chart][metric], rel=1e-6)
    log_odds = geometry.log_odds_gradients(grad, prob)
    for gradient_name, values in (("probability", grad), ("log_odds", log_odds)):
        metrics = lc.weighted_gradient_metrics(values, scale, ones)
        for chart in ("raw", "standardized"):
            key = (chart, gradient_name)
            assert metrics[(chart, "resultant_length")] == pytest.approx(
                result["coherence"][key]["resultant_length"], rel=1e-6)
            assert metrics[(chart, "centered_pr")] == pytest.approx(result["dimension"][key]["centered_pr"], rel=1e-6)
            assert metrics[(chart, "uncentered_pr")] == pytest.approx(result["dimension"][key]["uncentered_pr"], rel=1e-6)


def test_integer_weights_equal_repeated_rows():
    rng = np.random.default_rng(24)
    emb = rng.normal(size=(40, 5))
    grad = rng.normal(size=(40, 5))
    grad[4, 1] = np.nan
    scale = rng.uniform(0.5, 2.0, size=5)
    counts = rng.integers(0, 4, size=40)
    repeated = np.repeat(np.arange(40), counts)
    weighted_spectrum = lc.weighted_spectrum(emb, scale, counts.astype(float))
    repeated_spectrum = lc.weighted_spectrum(emb[repeated], scale, np.ones(len(repeated)))
    assert weighted_spectrum["raw"]["embedding_pr"] == pytest.approx(repeated_spectrum["raw"]["embedding_pr"], rel=1e-9)
    assert weighted_spectrum["standardized"]["top5_var_share"] == pytest.approx(
        repeated_spectrum["standardized"]["top5_var_share"], rel=1e-9)
    weighted = lc.weighted_gradient_metrics(grad, scale, counts.astype(float))
    repeated_metrics = lc.weighted_gradient_metrics(grad[repeated], scale, np.ones(len(repeated)))
    for key in weighted:
        assert weighted[key] == pytest.approx(repeated_metrics[key], rel=1e-9)


def bootstrap_inputs(rng, fit_y, eval_y, *, signal=0.0, layers=3, dim=6):
    """Fitting rows and one evaluation stratum; the first target coordinate shifts by signal times the label."""
    def tokens(labels):
        values = rng.normal(size=(len(labels), layers, dim))
        values[:, :, 0] += signal * np.asarray(labels)[:, None]
        return values

    fit = {"patient_ids": [f"fit-{index}" for index in range(len(fit_y))], "y": np.asarray(fit_y),
           "target": tokens(fit_y), "feature_mean": rng.normal(size=(len(fit_y), layers, dim))}
    evaluations = {"2011-2013": {
        "patient_ids": [f"eval-{index}" for index in range(len(eval_y))], "y": np.asarray(eval_y),
        "probabilities": rng.uniform(0.05, 0.95, size=len(eval_y)),
        "gradients": rng.normal(size=(len(eval_y), layers, dim)),
        "target": tokens(eval_y), "feature_mean": rng.normal(size=(len(eval_y), layers, dim)),
    }}
    scale_rows = rng.normal(size=(80, layers, dim))
    scale_feature_mean = rng.normal(size=(80, layers, dim))
    return fit, evaluations, scale_rows, scale_feature_mean


def stratum_geometry_table(evaluations, scale_rows, scale_feature_mean):
    """Geometry table per evaluation stratum, as the notebook builds it before the bootstrap."""
    frames = []
    for name, data in evaluations.items():
        table = lc.layer_geometry_table(data["target"], data["gradients"], data["probabilities"], scale_rows,
                                        feature_mean=data["feature_mean"], scale_feature_mean=scale_feature_mean)
        table.insert(0, "stratum", name)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def bootstrap_probe_fits(fit, evaluations, scale_rows, scale_feature_mean, draws=2):
    geometry_table = stratum_geometry_table(evaluations, scale_rows, scale_feature_mean)
    return lc.bootstrap_layer_curves(fit, evaluations, scale_rows, scale_feature_mean,
                                     geometry_table=geometry_table, draws=draws, seed=9)


def target_probe_auroc(result, layer=0, stratum="2011-2013"):
    fits = result["probe_fits"]
    row = fits[(fits["token"] == "target") & (fits["layer"] == layer) & (fits["stratum"] == stratum)]
    return float(row["auroc"].iloc[0])


def test_bootstrap_probe_separates_planted_signal_from_no_signal():
    labels = np.random.default_rng(31).integers(0, 2, size=160)
    planted = bootstrap_probe_fits(*bootstrap_inputs(np.random.default_rng(3), labels[:80], labels[80:], signal=2.0))
    no_signal = bootstrap_probe_fits(*bootstrap_inputs(np.random.default_rng(3), labels[:80], labels[80:], signal=0.0))
    assert target_probe_auroc(planted) > target_probe_auroc(no_signal)


def test_bootstrap_probe_fits_on_fitting_rows_only(monkeypatch):
    labels = np.random.default_rng(32).integers(0, 2, size=160)
    fit, evaluations, scale_rows, scale_feature_mean = bootstrap_inputs(
        np.random.default_rng(4), labels[:80], labels[80:], signal=1.0)
    seen_fits = []

    class RecordingCV(LogisticRegressionCV):
        def fit(self, X_fit, y_fit, sample_weight=None):
            seen_fits.append(np.asarray(X_fit))
            return super().fit(X_fit, y_fit, sample_weight=sample_weight)

    monkeypatch.setattr(lc, "LogisticRegressionCV", RecordingCV)
    bootstrap_probe_fits(fit, evaluations, scale_rows, scale_feature_mean)
    assert [len(X_fit) for X_fit in seen_fits] == [80] * 6
    for X_fit in seen_fits:
        np.testing.assert_allclose(X_fit.mean(axis=0), 0.0, atol=1e-9)


def test_bootstrap_probe_C_and_auroc_ignore_evaluation_rows_for_choice():
    labels = np.random.default_rng(33).integers(0, 2, size=160)
    fit, evaluations, scale_rows, scale_feature_mean = bootstrap_inputs(
        np.random.default_rng(5), labels[:80], labels[80:], signal=1.0)
    first = bootstrap_probe_fits(fit, evaluations, scale_rows, scale_feature_mean)
    evaluations["2011-2013"]["target"] = np.random.default_rng(6).normal(size=(80, 3, 6)) * 3.0
    changed = bootstrap_probe_fits(fit, evaluations, scale_rows, scale_feature_mean)
    first_fits = first["probe_fits"].set_index(["token", "layer", "stratum"])
    changed_fits = changed["probe_fits"].set_index(["token", "layer", "stratum"])
    for token in ("target", "feature_mean"):
        for layer in range(3):
            key = (token, layer, "2011-2013")
            assert changed_fits.loc[key, "C"] == first_fits.loc[key, "C"]
    assert target_probe_auroc(changed) != target_probe_auroc(first)


def test_choose_probe_C_matches_default_cross_validation_grid():
    rng = np.random.default_rng(34)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + rng.normal(size=120) > 0).astype(int)
    folds = StratifiedKFold(n_splits=lc.PROBE_CV_FOLDS, shuffle=True, random_state=lc.PROBE_CV_SEED)
    default_choice = LogisticRegressionCV(cv=folds, scoring="roc_auc").fit(X, y).C_[0]
    assert lc.choose_probe_C(X, y) == pytest.approx(default_choice, rel=1e-12)


def test_probe_fits_flag_C_at_grid_edge():
    labels = np.random.default_rng(35).integers(0, 2, size=160)
    fit, evaluations, scale_rows, scale_feature_mean = bootstrap_inputs(
        np.random.default_rng(7), labels[:80], labels[80:], signal=1.0)
    fits = bootstrap_probe_fits(fit, evaluations, scale_rows, scale_feature_mean)["probe_fits"]
    assert "C_at_grid_edge" in fits.columns
    token_rows = fits[fits["token"].isin(["target", "feature_mean"])]
    edges = token_rows["C"].isin([lc.PROBE_C_GRID[0], lc.PROBE_C_GRID[-1]])
    assert list(token_rows["C_at_grid_edge"].astype(bool)) == list(edges)


def test_bootstrap_setup_runs_under_thread_limit(monkeypatch):
    labels = np.random.default_rng(36).integers(0, 2, size=160)
    fit, evaluations, scale_rows, scale_feature_mean = bootstrap_inputs(
        np.random.default_rng(8), labels[:80], labels[80:], signal=1.0)
    blas_threads_seen = []
    original_prepare_layer = lc.prepare_layer

    def recording_prepare_layer(*args, **kwargs):
        blas_threads_seen.append(max(entry["num_threads"] for entry in threadpool_info()
                                     if entry["user_api"] == "blas"))
        return original_prepare_layer(*args, **kwargs)

    monkeypatch.setattr(lc, "prepare_layer", recording_prepare_layer)
    bootstrap_probe_fits(fit, evaluations, scale_rows, scale_feature_mean)
    assert len(blas_threads_seen) == 3
    assert max(blas_threads_seen) <= lc.BOOTSTRAP_THREAD_LIMIT


def test_bootstrap_random_shares_come_from_geometry_table_and_match_direct_analysis(monkeypatch):
    labels = np.random.default_rng(37).integers(0, 2, size=160)
    fit, evaluations, scale_rows, scale_feature_mean = bootstrap_inputs(
        np.random.default_rng(9), labels[:80], labels[80:], signal=1.0)
    geometry_table = stratum_geometry_table(evaluations, scale_rows, scale_feature_mean)
    analysis_calls = []
    original_analysis = probe.analyze_layer_geometry

    def counting_analysis(*args, **kwargs):
        analysis_calls.append(1)
        return original_analysis(*args, **kwargs)

    monkeypatch.setattr(probe, "analyze_layer_geometry", counting_analysis)
    curves = lc.bootstrap_layer_curves(fit, evaluations, scale_rows, scale_feature_mean,
                                       geometry_table=geometry_table, draws=2, seed=9)["layer_curves"]
    assert analysis_calls == []
    data = evaluations["2011-2013"]
    scale_1, _ = probe.standardization_scale(scale_rows[:, 1])
    direct = original_analysis(data["target"][:, 1], data["gradients"][:, 1], data["probabilities"], scale_1,
                               rng=np.random.default_rng(probe.DIRECTION_SEED), n_random=probe.RANDOM_DIRECTIONS,
                               top_axes=probe.TOP_AXES, interior=probe.INTERIOR_CUTS)
    for family in geometry.DIRECTION_FAMILIES:
        row = curves[(curves["layer"] == 1) & (curves["token"] == "target") & (curves["chart"] == "raw")
                     & (curves["metric"] == f"random_{family}_interior_share")].iloc[0]
        assert row["estimate"] == pytest.approx(direct["random"][("raw", family)]["interior_share"], rel=1e-9)


def test_weighted_gradient_metrics_identical_rows_match_analyze_layer_geometry():
    rng = np.random.default_rng(25)
    # Exactly representable values and power-of-two scales, so the centred covariance is exactly zero.
    grad = np.tile(np.array([1.0, 2.0, -0.5, 0.25, 3.0, -1.0, 0.0, 2.0]), (30, 1))
    emb = rng.normal(size=(30, 8))
    prob = rng.uniform(0.05, 0.95, size=30)
    scale = np.array([2.0, 0.5, 1.0, 4.0, 0.25, 2.0, 1.0, 0.5])
    result = probe.analyze_layer_geometry(emb, grad, prob, scale, rng=np.random.default_rng(1))
    metrics = lc.weighted_gradient_metrics(grad, scale, np.ones(30))
    for chart in ("raw", "standardized"):
        expected = result["dimension"][(chart, "probability")]
        assert np.isnan(metrics[(chart, "centered_pr")]) and np.isnan(expected["centered_pr"])
        assert metrics[(chart, "uncentered_pr")] == pytest.approx(expected["uncentered_pr"], rel=1e-6)


def test_bootstrap_layer_curves_shapes_and_point_estimates():
    rng = np.random.default_rng(5)
    target, gradients, probabilities, scale_rows, feature_mean, scale_feature_mean = synthetic_geometry_arrays(
        rng, rows=60, layers=3, dim=6)
    fit = {
        "patient_ids": [f"fit-{index}" for index in range(60)],
        "y": np.tile([0, 1], 30),
        "target": target[:60],
        "feature_mean": feature_mean[:60],
    }
    eval_rows = 40
    evaluations = {"2011-2013": {
        "patient_ids": [f"eval-{index % 25}" for index in range(eval_rows)],
        "y": np.tile([0, 1], 20),
        "probabilities": probabilities[:eval_rows],
        "gradients": gradients[:eval_rows],
        "target": target[:eval_rows],
        "feature_mean": feature_mean[:eval_rows],
    }}
    result = lc.bootstrap_layer_curves(fit, evaluations, scale_rows, scale_feature_mean,
                                       geometry_table=stratum_geometry_table(evaluations, scale_rows, scale_feature_mean),
                                       draws=5, seed=9)
    curves, probe_fits = result["layer_curves"], result["probe_fits"]
    assert {"stratum", "layer", "token", "chart", "metric", "estimate", "low", "high", "valid_draws",
            "difference_from_layer_11", "difference_low", "difference_high", "rows", "deaths"} <= set(curves.columns)
    assert set(curves["layer"].dropna().astype(int)) == {0, 1, 2}
    assert set(probe_fits["token"]) == {"target", "feature_mean", "model"}
    assert set(probe_fits.dropna(subset=["layer"])["layer"].astype(int)) == {0, 1, 2}
    scale_1, _ = probe.standardization_scale(scale_rows[:, 1])
    spectrum = lc.weighted_spectrum(target[:eval_rows, 1], scale_1, np.ones(eval_rows))
    row = curves[(curves["layer"] == 1) & (curves["token"] == "target") & (curves["chart"] == "raw")
                 & (curves["metric"] == "embedding_pr")].iloc[0]
    assert row["estimate"] == pytest.approx(spectrum["raw"]["embedding_pr"], rel=1e-9)
    assert row["valid_draws"] <= 5
    assert row["rows"] == eval_rows and row["deaths"] == 20


def test_check_disjoint_rows_refuses_overlap():
    lc.check_disjoint_rows(["c1", "c2"], ["f1", "f2"], ["e1", "e2"])
    with pytest.raises(ValueError, match="overlap"):
        lc.check_disjoint_rows(["c1", "f1"], ["f1", "f2"], ["e1"])
    with pytest.raises(ValueError, match="overlap"):
        lc.check_disjoint_rows(["c1"], ["f1"], ["f1"])


def numpy_of(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def test_historical_fit_predictions_match_pilot_frozen_predictor():
    rng = np.random.default_rng(13)
    names = [f"f{index}" for index in range(6)]
    years = np.tile([2007, 2008, 2009], 20)
    X = rng.normal(size=(60, 6)).astype(np.float32)
    y = np.tile([0, 1], 30)
    context = pd.DataFrame(X, columns=names)
    context["patient_id"], context["year"], context["y"] = [f"c{index}" for index in range(60)], years, y
    system = pilot.fit_frozen_predictor(context, names, "cpu", seed=42, domain_origin=2007)
    query_X = rng.normal(size=(8, 6)).astype(np.float32)
    query_years = np.tile([2009, 2010], 4)
    query = pd.DataFrame(query_X, columns=names)
    query["patient_id"], query["year"] = [f"q{index}" for index in range(8)], query_years
    expected = numpy_of(pilot.capture_frozen_predictions(system, query, "cpu")["probabilities"])
    fitted = pa.fit_dr_tabpfn_context(X, y, years, domain_origin=2007, device="cpu", use_context_scaler=False)
    actual = probe.capture_flchain_batches(fitted, query_X, query_years, batch_size=8)["probabilities"]
    np.testing.assert_allclose(actual, expected.reshape(actual.shape), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("use_context_scaler", [True, False])
def test_capture_layers_works_on_sus_style_fit(use_context_scaler):
    rng = np.random.default_rng(17)
    years = np.tile([2009, 2010, 2011], 20)
    X = rng.normal(size=(60, 6)).astype(np.float32)
    y = np.tile([0, 1], 30)
    fitted = pa.fit_dr_tabpfn_context(X, y, years, domain_origin=2009, device="cpu",
                                      use_context_scaler=use_context_scaler)
    query_X = X[:6]
    query_years = np.tile([2011, 2012], 3)
    layer_capture = lc.capture_layers(fitted, query_X, query_years, batch_size=4)
    existing = pa.capture_scored_rows(fitted, query_X, query_years, batch_size=4)
    assert layer_capture["target"].shape == (6, 12, 192)
    np.testing.assert_allclose(layer_capture["target"][:, 11], existing["embeddings"], **FORWARD_TOL)
    np.testing.assert_allclose(layer_capture["probabilities"][:, 1], existing["probabilities"], **FORWARD_TOL)
    assert all(len(layer._forward_hooks) == 0 for layer in fitted["classifier"].cached_models_[0].transformer_encoder.layers)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_layer_fd_check_reports_every_layer(device):
    fitted, X, years = tiny_fit(device)
    captured, hooked = hooked_layer_outputs(fitted, X, years)
    weights = lc.clone_replay_weights(fitted["classifier"])
    report = lc.layer_fd_check(weights, captured["continuation"], hooked[0].clone(), rows=np.arange(PROBE_ROWS), seed=7)
    assert len(report["max_abs_error"]) == LAST_LAYER + 1
    assert report["passed"]
