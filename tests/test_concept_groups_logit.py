import numpy as np
import pytest
import torch

import concept_groups_logit as logit
import concept_groups_over_time as study


def toy(seed=0, rows=40, factors=6, dimensions=5):
    rng = np.random.default_rng(seed)
    codes = np.maximum(rng.normal(size=(rows, factors)), 0)
    codes[:3] = 0  # wholly inactive rows stay undefined on both scales
    decoder = rng.normal(size=(factors, dimensions))
    scale = rng.uniform(0.5, 2, size=dimensions)
    gradients = rng.normal(size=(rows, dimensions))
    probability = rng.uniform(0.005, 0.6, size=rows)
    return codes, decoder, scale, gradients, probability


def test_log_odds_loading_is_probability_loading_over_pq():
    codes, decoder, scale, gradients, p = toy()
    factors = (0, 2, 5)
    prob = study.contribution_measures(codes, decoder, scale, gradients, factors)
    odds = study.contribution_measures(codes, decoder, scale, logit.log_odds_gradients(gradients, p), factors)
    pq = torch.as_tensor(p * (1 - p))
    torch.testing.assert_close(odds["loading"], prob["loading"] / pq, rtol=1e-12, atol=0)
    torch.testing.assert_close(odds["member_loading"], prob["member_loading"] / pq[:, None], rtol=1e-12, atol=0)
    torch.testing.assert_close(odds["normalized_sensitivity"], prob["normalized_sensitivity"] / pq,
                               rtol=1e-12, atol=0, equal_nan=True)
    for name in ("contribution", "norm", "cancellation", "member_norm", "member_activation"):
        torch.testing.assert_close(odds[name], prob[name], rtol=0, atol=0, equal_nan=True)


def test_conversion_matches_autograd_of_logit():
    torch.manual_seed(0)
    weight = torch.randn(5, dtype=torch.float64)
    x = torch.randn(30, 5, dtype=torch.float64, requires_grad=True)
    p = torch.sigmoid(x @ weight - 2)
    dp, = torch.autograd.grad(p.sum(), x)
    dlogit, = torch.autograd.grad(torch.logit(torch.sigmoid(x @ weight - 2)).sum(), x)
    np.testing.assert_allclose(logit.log_odds_gradients(dp.numpy(), p.detach().numpy()), dlogit.numpy(), rtol=1e-10)


def test_conversion_rejects_probabilities_at_bounds():
    with pytest.raises(ValueError):
        logit.log_odds_gradients(np.ones((2, 3)), np.array([0.2, 1.0]))


def test_midpoint_decomposition_exact_on_log_odds():
    codes, decoder, scale, g0, p0 = toy(1)
    _, _, _, g1, p1 = toy(2)
    codes1 = codes + np.abs(np.random.default_rng(3).normal(size=codes.shape))
    factors = (1, 3)
    l0, l1 = logit.log_odds_gradients(g0, p0), logit.log_odds_gradients(g1, p1)
    first = study.contribution_measures(codes, decoder, scale, l0, factors)
    second = study.contribution_measures(codes1, decoder, scale, l1, factors)
    parts = study.midpoint_decomposition(l0, l1, first["contribution"], second["contribution"])
    torch.testing.assert_close(parts["loading_change"], second["loading"] - first["loading"], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(parts["loading_change"], parts["contribution_change"] + parts["gradient_change"],
                               rtol=1e-12, atol=1e-12)


def test_low_risk_tail_shares():
    loading = np.array([10.0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    p = np.array([0.01, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2])
    tail = logit.low_risk_tail(loading, p)
    assert tail["low_risk_decile_rows"] == 1
    assert tail["low_risk_decile_share_log_odds"] == pytest.approx(10 / 19)
    weighted = loading * p * (1 - p)
    assert tail["low_risk_decile_share_probability"] == pytest.approx(weighted[0] / weighted.sum())


RUN = logit.EXTENSION / "run"


@pytest.mark.skipif(not (RUN / "logit-manifest.json").exists() or not logit.ORIGINAL.exists(),
                    reason="needs the full original and log-odds runs (Fiji)")
def test_gradient_free_outputs_identical_to_original_run():
    differences = logit.gradient_free_differences(logit.ORIGINAL, RUN)
    for name, values in differences.items():
        assert values["definedness_equal"], name
        for column in ("value", "low", "high"):
            assert values[column] <= 1e-12, (name, column, values[column])


@pytest.mark.skipif(not (RUN / "logit-manifest.json").exists(), reason="needs the log-odds run (Fiji)")
def test_run_decomposition_exact_and_conversion_checked():
    import json
    import pandas as pd
    manifest = json.loads((RUN / "logit-manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["study_source_sha256"] == logit.CORRECTED_SOURCE_SHA256
    assert all(check["p_min"] > 0 and check["p_max"] < 1 for check in manifest["build_checks"])
    table = pd.read_csv(RUN / "transitions.csv", usecols=["concept", "pair", "kind", "side", "start", "subgroup",
                                                          "transition", "metric", "value"])
    wide = table.set_index(["concept", "pair", "kind", "side", "start", "subgroup", "transition",
                            "metric"])["value"].unstack("metric")
    residual = wide.loading_change - wide.contribution_change - wide.gradient_change
    assert np.nanmax(np.abs(residual.to_numpy())) <= 1e-9 * max(1, np.nanmax(np.abs(wide.loading_change.to_numpy())))
