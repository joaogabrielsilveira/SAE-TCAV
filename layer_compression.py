"""Layer replay of the cached DR-TabPFN test-row computation, for gradients at every layer.

The cached model is inference-only, so its layers cannot be differentiated. These functions
recompute layers 1..11 for test rows from ordinary copies of the weights and of the head-0
cached keys and values of the context rows.
"""
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import flchain_probe as probe
import prospective_audit as pa
import tcav_geometry as geometry

FD_STEP = 1e-2
FD_TOLERANCE = {"rtol": 1e-3, "atol": 1e-5}
TOKENS = ("target", "feature_mean")
SPECTRUM_METRICS = ("embedding_pr", "top1_var_share", "top5_var_share", "top10_var_share")
PROBE_CV_FOLDS = 5
PROBE_CV_SEED = 42
PROBE_C_GRID = np.logspace(-4, 4, 10)
BOOTSTRAP_THREAD_LIMIT = 1


def check_replayable_layer(layer):
    """Refuse layers whose settings the replay does not implement."""
    features, items = layer.self_attn_between_features, layer.self_attn_between_items
    checks = [
        not layer.pre_norm,
        layer.second_mlp is None,
        features.softmax_scale is None and items.softmax_scale is None,
        features.dropout_p in (None, 0.0) and items.dropout_p in (None, 0.0),
        all(norm.weight is None and norm.bias is None for norm in layer.layer_norms),
        items._kv_cache is not None and items._k_cache is None,
    ]
    if not all(checks):
        raise ValueError("layer uses settings the layer replay does not implement")


def layer_replay_weights(layer):
    """Ordinary tensors with one layer's weights and the head-0 cached item keys and values."""
    check_replayable_layer(layer)
    features, items = layer.self_attn_between_features, layer.self_attn_between_items
    cache = items._kv_cache
    return {
        "feature_qkv": features._w_qkv.detach().clone(),
        "feature_out": features._w_out.detach().clone(),
        "item_query": items._w_qkv[0].detach().clone(),
        "item_out": items._w_out.detach().clone(),
        "item_key": cache[:, :, 0, 0, :].detach().clone(),
        "item_value": cache[:, :, 1, 0, :].detach().clone(),
        "mlp_in": layer.mlp.linear1.weight.detach().clone(),
        "mlp_out": layer.mlp.linear2.weight.detach().clone(),
        "norm_eps": float(layer.layer_norms[0].eps),
    }


def clone_replay_weights(classifier):
    """Replay weights for layers 1..11 of the cached model; must run outside inference mode."""
    if torch.is_inference_mode_enabled():
        raise RuntimeError("clone replay weights outside torch.inference_mode")
    layers = classifier.cached_models_[0].transformer_encoder.layers
    return {index: layer_replay_weights(layers[index]) for index in range(1, len(layers))}


def double_replay_weights(weights):
    """Float64 copies of the replay weights, for the finite-difference check only."""
    return {
        index: {name: value.double() if torch.is_tensor(value) else value for name, value in layer.items()}
        for index, layer in weights.items()
    }


def double_continuation(continuation):
    """Copy of the head continuation with a float64 head; its probabilities() still returns float32."""
    doubled = copy.copy(continuation)
    doubled.head = copy.deepcopy(continuation.head).double()
    return doubled


def feature_attention(state, weights):
    """Attention among the feature tokens of each row, with the residual added."""
    qkv = torch.einsum("nts,jhds->ntjhd", state, weights["feature_qkv"])
    query, key, value = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
    mixed = F.scaled_dot_product_attention(query, key, value)
    return state + torch.einsum("nhtd,hds->nts", mixed, weights["feature_out"])


def item_attention(state, weights):
    """Attention from each test row to the cached context rows, per token, with the residual added."""
    tokens = state.transpose(0, 1)
    query = torch.einsum("tns,hds->tnhd", tokens, weights["item_query"]).transpose(1, 2)
    heads = query.shape[1]
    key = weights["item_key"][:, None].expand(-1, heads, -1, -1)
    value = weights["item_value"][:, None].expand(-1, heads, -1, -1)
    mixed = F.scaled_dot_product_attention(query, key, value)
    attended = torch.einsum("thnd,hds->tns", mixed, weights["item_out"]).transpose(0, 1)
    return state + attended


def feed_forward(state, weights):
    """Two bias-free linear maps with GELU between them, with the residual added."""
    hidden = F.gelu(state @ weights["mlp_in"].T)
    return state + hidden @ weights["mlp_out"].T


def replay_layer(state, weights):
    """One post-norm layer: feature attention, item attention and feed-forward, each followed by LayerNorm."""
    for sublayer in (feature_attention, item_attention, feed_forward):
        state = sublayer(state, weights)
        state = F.layer_norm(state, state.shape[-1:], eps=weights["norm_eps"])
    return state


def replay_layers(weights, layer_output, first_layer):
    """Outputs from first_layer to the last layer; the first entry is layer_output itself."""
    outputs = [layer_output]
    for layer in range(first_layer + 1, max(weights) + 1):
        outputs.append(replay_layer(outputs[-1], weights[layer]))
    return outputs


def death_probability(continuation, last_output):
    """Death probability of the head applied to the target token of the last replayed layer."""
    return continuation.probabilities(last_output[:, -1, :])[:, 1]


def layer_gradients(weights, continuation, first_layer_output):
    """Death probability and target-token gradients at every layer, from one backward pass."""
    leaf = first_layer_output.detach().clone().requires_grad_(True)
    outputs = replay_layers(weights, leaf, 0)
    for output in outputs:
        output.retain_grad()
    probability = death_probability(continuation, outputs[-1])
    probability.sum().backward()
    return probability.detach(), [output.grad[:, -1, :] for output in outputs]


def layer_fd_check(weights, continuation, first_layer_output, *, rows, seed):
    """Central finite differences of each layer's target-token gradient, on some rows, in float64.

    For layer L the target token of that layer is moved along a random unit direction, the layers above
    it are replayed, and the change in death probability is compared with the analytic gradient projected
    on the same direction. Returns the largest absolute error per layer and whether every layer is within
    FD_TOLERANCE.
    """
    double_weights = double_replay_weights(weights)
    double_head = double_continuation(continuation)
    first = first_layer_output[torch.as_tensor(rows, device=first_layer_output.device)].double()
    _, gradients = layer_gradients(double_weights, double_head, first)
    replay = replay_layers(double_weights, first, 0)
    rng = np.random.default_rng(seed)
    errors, passed = [], []
    for layer, output in enumerate(replay):
        direction = rng.normal(size=(len(rows), first.shape[-1]))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        direction = torch.as_tensor(direction, dtype=torch.float64, device=first.device)
        analytic = (gradients[layer] * direction).sum(dim=1)
        plus, minus = output.detach().clone(), output.detach().clone()
        plus[:, -1, :] += FD_STEP * direction
        minus[:, -1, :] -= FD_STEP * direction
        p_plus = death_probability(double_head, replay_layers(double_weights, plus, layer)[-1]).double()
        p_minus = death_probability(double_head, replay_layers(double_weights, minus, layer)[-1]).double()
        finite = (p_plus - p_minus) / (2 * FD_STEP)
        errors.append(float((analytic - finite).abs().max()))
        passed.append(bool(torch.allclose(analytic, finite, **FD_TOLERANCE)))
    return {"max_abs_error": errors, "passed": all(passed)}


def token_count(classifier):
    """Tokens per test row, read from the item-attention key/value cache of layer 0."""
    layers = classifier.cached_models_[0].transformer_encoder.layers
    return int(layers[0].self_attn_between_items._kv_cache.shape[0])


def record_layer_output(recorded, index):
    """Forward hook that appends one layer's output for the test rows of each capture batch."""
    def hook(module, args, output):
        recorded[index].append(output[0].detach().clone())
    return hook


def capture_layers(fitted, X, years, *, batch_size=256):
    """Every layer's target and feature-mean tokens, the layer-0 output and the death probability.

    One capture call per batch; hooks are removed after success or error.
    """
    layers = fitted["classifier"].cached_models_[0].transformer_encoder.layers
    recorded = [[] for _ in layers]
    handles = [layer.register_forward_hook(record_layer_output(recorded, index))
               for index, layer in enumerate(layers)]
    try:
        captured = probe.capture_flchain_batches(fitted, X, years, batch_size=batch_size)
    finally:
        for handle in handles:
            handle.remove()
    batch_count = len(recorded[0])
    if any(len(outputs) != batch_count for outputs in recorded):
        raise RuntimeError("layers did not run once per capture batch")
    tokens = torch.cat([
        torch.stack([recorded[index][batch] for index in range(len(layers))], dim=1)
        for batch in range(batch_count)
    ])
    return {
        "target": tokens[:, :, -1, :].cpu().numpy(),
        "feature_mean": tokens[:, :, :-1, :].mean(dim=2).cpu().numpy(),
        "layer0": tokens[:, 0],
        "probabilities": captured["probabilities"],
        "continuation": captured["continuation"],
    }


def check_disjoint_rows(context_ids, fitting_ids, evaluation_ids):
    """Refuse row sets that share a row: context, probe fitting and evaluation rows must not overlap."""
    named_sets = {"context": context_ids, "fitting": fitting_ids, "evaluation": evaluation_ids}
    names = list(named_sets)
    for position, first in enumerate(names):
        for second in names[position + 1:]:
            shared = np.intersect1d(np.asarray(named_sets[first]), np.asarray(named_sets[second]))
            if len(shared):
                raise ValueError(f"row sets overlap: {first} and {second} share {len(shared)} rows")


def weighted_covariance(values, weights):
    """Covariance of rows where row i counts weights[i] times (divides by the total weight)."""
    total = weights.sum()
    centred = values - (weights @ values) / total
    return (centred * weights[:, None]).T @ centred / total


def weighted_spectrum(embeddings, scale, weights):
    """Spectrum block of analyze_layer_geometry, with row i counted weights[i] times, per chart."""
    values = np.asarray(embeddings, dtype=np.float64)
    blocks = {}
    for chart, chart_scale in (("raw", np.ones_like(scale)), ("standardized", scale)):
        covariance = weighted_covariance(values / chart_scale, weights)
        variances = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0.0)
        total = variances.sum()
        block = {"n_rows": int(np.count_nonzero(weights))}
        block["embedding_pr"] = geometry.participation_ratio(covariance) if total > 0 else np.nan
        for top in (1, 5, 10):
            block[f"top{top}_var_share"] = variances[:top].sum() / total if total > 0 else np.nan
        blocks[chart] = block
    return blocks


def weighted_gradient_metrics(gradients, scale, weights):
    """Resultant length and participation ratios of gradient rows with row weights, per chart.

    Same definitions as analyze_layer_geometry: unit rows (finite, nonzero) for the resultant length,
    finite rows for the centred and uncentred participation ratios.
    """
    grads = np.asarray(gradients, dtype=np.float64)
    metrics = {}
    for chart, chart_scale in (("raw", np.ones_like(scale)), ("standardized", scale)):
        chart_grads = grads * chart_scale
        finite = np.isfinite(chart_grads).all(axis=1)
        values = np.where(finite[:, None], chart_grads, 0.0)
        moment_weights = weights * finite
        if moment_weights.sum() <= 0:
            metrics.update({(chart, name): np.nan for name in ("resultant_length", "centered_pr", "uncentered_pr")})
            continue
        centred = weighted_covariance(values, moment_weights)
        uncentred = (values * moment_weights[:, None]).T @ values / moment_weights.sum()
        norms = np.linalg.norm(values, axis=1)
        unit_mask = finite & (norms > 0)
        unit = values / np.where(unit_mask, norms, 1.0)[:, None]
        unit_weights = weights * unit_mask
        metrics[(chart, "resultant_length")] = (
            float(np.linalg.norm(unit_weights @ unit / unit_weights.sum())) if unit_weights.sum() > 0 else np.nan)
        has_spread = float(np.sum(centred * centred.T)) > 0
        metrics[(chart, "centered_pr")] = geometry.participation_ratio(centred) if has_spread else np.nan
        metrics[(chart, "uncentered_pr")] = (
            geometry.participation_ratio(uncentred) if float(np.sum(uncentred * uncentred.T)) > 0 else np.nan)
    return metrics


def long_geometry(result, layer, token):
    """flatten_geometry output with the layer and the token kind as columns."""
    table = pa.flatten_geometry(result)
    table.insert(0, "token", token)
    table.insert(0, "layer", layer)
    return table


def layer_geometry_table(target, gradients, probabilities, scale_embeddings, *, feature_mean, scale_feature_mean):
    """Long geometry table per layer: target token (Work 2 block) and feature-mean token (spectrum only).

    Standardization scales come from the scale rows of each layer; zero-SD columns are counted per layer.
    """
    frames = []
    for layer in range(target.shape[1]):
        scale, zero_columns = probe.standardization_scale(scale_embeddings[:, layer])
        result = probe.analyze_layer_geometry(
            target[:, layer], gradients[:, layer], probabilities, scale,
            rng=np.random.default_rng(probe.DIRECTION_SEED), n_random=probe.RANDOM_DIRECTIONS,
            top_axes=probe.TOP_AXES, interior=probe.INTERIOR_CUTS)
        result["standardization_zero_columns"] = zero_columns
        frames.append(long_geometry(result, layer, "target"))
        mean_scale, mean_zero_columns = probe.standardization_scale(scale_feature_mean[:, layer])
        mean_result = {
            "spectrum": weighted_spectrum(feature_mean[:, layer], mean_scale, np.ones(len(feature_mean))),
            "standardization_zero_columns": mean_zero_columns,
        }
        frames.append(long_geometry(mean_result, layer, "feature_mean"))
    return pd.concat(frames, ignore_index=True)


def standardize_by_fitting_rows(fit_X):
    """Mean and standard deviation of the fitting rows (zero-SD columns keep scale 1)."""
    scale, _ = probe.standardization_scale(fit_X)
    return fit_X.mean(axis=0), scale


def choose_probe_C(fit_X, fit_y):
    """Regularization strength chosen by cross-validation on the fitting rows only."""
    folds = StratifiedKFold(n_splits=PROBE_CV_FOLDS, shuffle=True, random_state=PROBE_CV_SEED)
    return float(LogisticRegressionCV(Cs=PROBE_C_GRID, cv=folds, scoring="roc_auc").fit(fit_X, fit_y).C_[0])


def at_grid_edge(C):
    """True when the chosen C is the smallest or the largest value of the probe's C grid."""
    return bool(C in (PROBE_C_GRID[0], PROBE_C_GRID[-1]))


def fit_probe_or_none(fit_X, fit_y, *, C, fit_weight):
    """L2 logistic probe on the fitting rows with row weights; None when a weighted class is missing."""
    present = fit_weight > 0
    if len(np.unique(fit_y[present])) < 2:
        return None
    return LogisticRegression(C=C).fit(fit_X, fit_y, sample_weight=fit_weight)


def weighted_auroc(labels, scores, weight=None):
    """AUROC with row weights; NaN when a weighted class is missing."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    weight = np.ones(len(labels)) if weight is None else np.asarray(weight, dtype=np.float64)
    present = weight > 0
    if len(np.unique(labels[present])) < 2:
        return np.nan
    return float(roc_auc_score(labels[present], scores[present], sample_weight=weight[present]))


def prepare_layer(layer, fit, evaluations, scale_embeddings, scale_feature_mean):
    """Standardized probe inputs, chosen C and gradient arrays for one layer."""
    target_scale, _ = probe.standardization_scale(scale_embeddings[:, layer])
    mean_scale, _ = probe.standardization_scale(scale_feature_mean[:, layer])
    prepared = {"target_scale": target_scale, "feature_mean_scale": mean_scale, "tokens": {}, "strata": {}}
    for token in TOKENS:
        fit_values = np.asarray(fit[token][:, layer], dtype=np.float64)
        mean, scale = standardize_by_fitting_rows(fit_values)
        fit_std = (fit_values - mean) / scale
        prepared["tokens"][token] = {
            "fit_std": fit_std,
            "C": choose_probe_C(fit_std, fit["y"]),
            "eval_std": {name: (np.asarray(data[token][:, layer], dtype=np.float64) - mean) / scale
                         for name, data in evaluations.items()},
        }
    for name, data in evaluations.items():
        probability = np.asarray(data["probabilities"], dtype=np.float64)
        gradients = np.asarray(data["gradients"][:, layer], dtype=np.float64)
        target = np.asarray(data["target"][:, layer], dtype=np.float64)
        prepared["strata"][name] = {
            "y": np.asarray(data["y"]),
            "probabilities": probability,
            "target": target,
            "feature_mean": np.asarray(data["feature_mean"][:, layer], dtype=np.float64),
            "gradient_probability": gradients,
            "gradient_log_odds": geometry.log_odds_gradients(gradients, probability),
        }
    return prepared


def layer_values(prepared, fit_y, fit_weight, eval_weights):
    """Curve metrics of one layer for one set of row weights, keyed by (stratum, token, chart, metric)."""
    values = {}
    for token in TOKENS:
        setup = prepared["tokens"][token]
        model = fit_probe_or_none(setup["fit_std"], fit_y, C=setup["C"], fit_weight=fit_weight)
        for name, data in prepared["strata"].items():
            scores = (model.predict_proba(setup["eval_std"][name])[:, 1] if model is not None
                      else np.full(len(data["y"]), np.nan))
            values[(name, token, "none", "probe_auroc")] = weighted_auroc(data["y"], scores, eval_weights[name])
    for name, data in prepared["strata"].items():
        weight = eval_weights[name]
        for token, scale in (("target", prepared["target_scale"]), ("feature_mean", prepared["feature_mean_scale"])):
            for chart, block in weighted_spectrum(data[token], scale, weight).items():
                for metric in SPECTRUM_METRICS:
                    values[(name, token, chart, metric)] = block[metric]
        for gradient_name, gradients in (("probability", data["gradient_probability"]),
                                         ("log_odds", data["gradient_log_odds"])):
            for (chart, metric), value in weighted_gradient_metrics(
                    gradients, prepared["target_scale"], weight).items():
                values[(name, "target", chart, f"{gradient_name}_{metric}")] = value
    return values


def interval(draws):
    """Percentile interval over the finite draws, and how many draws were finite."""
    finite = draws[np.isfinite(draws)]
    if not len(finite):
        return np.nan, np.nan, 0
    return (float(np.quantile(finite, pa.LOW_QUANTILE)), float(np.quantile(finite, pa.HIGH_QUANTILE)), len(finite))


def bootstrap_layer_curves(fit, evaluations, scale_embeddings, scale_feature_mean, *,
                           geometry_table, draws=pa.BOOTSTRAP_DRAWS, seed=pa.BOOTSTRAP_SEED):
    """Layer curves with one joint patient bootstrap over the fitting and evaluation patients.

    fit: dict with patient_ids, y, target and feature_mean arrays (rows, layers, dimension).
    evaluations: {stratum: dict with patient_ids, y, probabilities, gradients, target, feature_mean}.
    geometry_table: layer_geometry_table output for each stratum, with a stratum column. Its random-direction
    rows are the point values of the random-direction shares, so the geometry is not computed again here.
    Each draw refits every layer's probe on the weighted fitting rows (C fixed from the point fit) and
    recomputes the geometry on the weighted evaluation rows with the standardization scales fixed.
    Returns the curve table (estimate, interval, paired difference from the last layer) and the probe fits.
    """
    fit_y = np.asarray(fit["y"])
    universe, counts = pa.patient_draw_counts(
        np.concatenate([np.asarray(fit["patient_ids"], dtype=str)]
                       + [np.asarray(data["patient_ids"], dtype=str) for data in evaluations.values()]),
        draws=draws, seed=seed)
    fit_codes = pa.row_patient_codes(fit["patient_ids"], universe)
    eval_codes = {name: pa.row_patient_codes(data["patient_ids"], universe) for name, data in evaluations.items()}
    layers = range(scale_embeddings.shape[1])
    last_layer = layers[-1]
    ones_fit = np.ones(len(fit_y))
    point_eval = {name: np.ones(len(data["y"])) for name, data in evaluations.items()}
    point_model = {name: weighted_auroc(data["y"], data["probabilities"]) for name, data in evaluations.items()}
    draw_model = {name: np.full(draws, np.nan) for name in evaluations}

    with threadpool_limits(limits=BOOTSTRAP_THREAD_LIMIT):
        prepared = {layer: prepare_layer(layer, fit, evaluations, scale_embeddings, scale_feature_mean)
                    for layer in layers}
        point = {layer: layer_values(prepared[layer], fit_y, ones_fit, point_eval) for layer in layers}
        draw_values = {layer: {key: np.full(draws, np.nan) for key in point[layer]} for layer in layers}
        for draw in range(draws):
            fit_weight = counts[draw, fit_codes].astype(np.float64)
            eval_weights = {name: counts[draw, codes].astype(np.float64) for name, codes in eval_codes.items()}
            for name, data in evaluations.items():
                draw_model[name][draw] = weighted_auroc(data["y"], data["probabilities"], eval_weights[name])
            for layer in layers:
                for key, value in layer_values(prepared[layer], fit_y, fit_weight, eval_weights).items():
                    draw_values[layer][key][draw] = value

    rows = []
    for name, data in evaluations.items():
        estimate = point_model[name]
        low, high, valid = interval(draw_model[name])
        rows.append(curve_row(name, None, "model", "none", "death_auroc", estimate, low, high, valid,
                              np.nan, np.nan, np.nan, len(data["y"]), int(np.sum(data["y"]))))
    for layer in layers:
        for key, estimate in point[layer].items():
            name, token, chart, metric = key
            low, high, valid = interval(draw_values[layer][key])
            difference = draw_values[layer][key] - draw_values[last_layer][key]
            difference_low, difference_high, _ = interval(difference)
            rows.append(curve_row(name, layer, token, chart, metric, estimate, low, high, valid,
                                  estimate - point[last_layer][key], difference_low, difference_high,
                                  len(evaluations[name]["y"]), int(np.sum(evaluations[name]["y"]))))
    rows.extend(random_share_rows(geometry_table, evaluations))
    return {"layer_curves": pd.DataFrame(rows), "probe_fits": probe_fit_table(prepared, point, fit_y, evaluations)}


def curve_row(stratum, layer, token, chart, metric, estimate, low, high, valid_draws,
              difference, difference_low, difference_high, rows, deaths):
    """One line of the layer curve table, columns as in the plan's output schema."""
    return {"stratum": stratum, "layer": layer, "token": token, "chart": chart, "metric": metric,
            "estimate": estimate, "low": low, "high": high, "valid_draws": valid_draws,
            "difference_from_layer_11": difference, "difference_low": difference_low,
            "difference_high": difference_high, "rows": rows, "deaths": deaths}


def random_share_rows(geometry_table, evaluations):
    """Random-direction interior shares with their direction-sampling standard errors (no row bootstrap),
    read from the geometry table rows of each stratum."""
    shares = geometry_table[(geometry_table["section"] == "random") & (geometry_table["token"] == "target")
                            & geometry_table["metric"].isin(["interior_share", "interior_se"])]
    rows = []
    for name, data in evaluations.items():
        for entry in shares[shares["stratum"] == name].itertuples(index=False):
            chart, family = entry.key.split("|")
            rows.append(curve_row(name, int(entry.layer), "target", chart, f"random_{family}_{entry.metric}",
                                  float(entry.value), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                  len(data["y"]), int(np.sum(data["y"]))))
    return rows


def probe_fit_table(prepared, point, fit_y, evaluations):
    """Chosen C, whether C sits at an edge of its grid, row counts and point AUROC per layer, token and stratum,
    plus the model's own AUROC rows."""
    rows = []
    for layer, layer_point in point.items():
        for token in TOKENS:
            chosen = prepared[layer]["tokens"][token]["C"]
            for name, data in evaluations.items():
                rows.append({"stratum": name, "layer": layer, "token": token,
                             "C": chosen, "C_at_grid_edge": at_grid_edge(chosen), "fit_rows": len(fit_y),
                             "fit_deaths": int(np.sum(fit_y)), "eval_rows": len(data["y"]),
                             "eval_deaths": int(np.sum(data["y"])),
                             "auroc": layer_point[(name, token, "none", "probe_auroc")]})
    for name, data in evaluations.items():
        rows.append({"stratum": name, "layer": np.nan, "token": "model", "C": np.nan, "C_at_grid_edge": np.nan,
                     "fit_rows": len(fit_y), "fit_deaths": int(np.sum(fit_y)), "eval_rows": len(data["y"]),
                     "eval_deaths": int(np.sum(data["y"])),
                     "auroc": weighted_auroc(data["y"], data["probabilities"])})
    return pd.DataFrame(rows)
