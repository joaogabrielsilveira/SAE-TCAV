"""Focused checks of the concept-study check modules (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import rankdata

from concept_study import family_controls, input_drift, planted_shift, review_sheet


def test_psi_is_zero_for_the_same_sample():
    x = np.random.default_rng(0).poisson(1.0, size=(200, 3)).astype(float)
    np.testing.assert_allclose(input_drift.psi(x, x), 0.0, atol=1e-12)


def test_psi_is_large_when_the_current_column_is_all_zero():
    reference = np.random.default_rng(1).integers(1, 10, size=(200, 1)).astype(float)
    assert input_drift.psi(reference, np.zeros((200, 1)))[0] > 1.0


def test_psi_two_bin_hand_case():
    # Reference has no positive values: bins are "zero" and "positive". Counts + 0.5 per bin.
    reference, current = np.zeros((4, 1)), np.array([[0.0], [0.0], [1.0], [1.0]])
    p, q = np.array([4.5, 0.5]) / 5, np.array([2.5, 2.5]) / 5
    expected = np.sum((q - p) * np.log(q / p))
    assert input_drift.psi(reference, current)[0] == pytest.approx(expected)


def test_partial_spearman_with_groups_is_within_group_demeaned_rank_correlation():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.array(["a", "b", "c"]), 10)
    x, y = rng.normal(size=30), rng.normal(size=30)
    rx, ry = rankdata(x), rankdata(y)
    for g in np.unique(groups):
        rx[groups == g] -= rx[groups == g].mean()
        ry[groups == g] -= ry[groups == g].mean()
    expected = (rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    assert input_drift.partial_spearman(x, y, groups) == pytest.approx(expected)


def test_concept_bootstrap_keeps_duplicated_concepts_as_separate_groups(monkeypatch):
    monkeypatch.setattr(input_drift, "BOOT", 50)
    frame = pd.DataFrame({"concept": ["a", "a", "b", "b"], "value": [1.0, 2.0, 3.0, 4.0]})
    point, low, high = input_drift.concept_bootstrap(frame, lambda f: f["concept"].nunique())
    assert (point, low, high) == (2, 2, 2)


def test_a_concept_in_two_variants_counts_in_both():
    frame = pd.DataFrame({"variants": ["last_3,all_history", "last_3", "all_history"]})
    np.testing.assert_array_equal(input_drift.in_variant(frame, "last_3"), [True, True, False])
    np.testing.assert_array_equal(input_drift.in_variant(frame, "all_history"), [True, False, True])


def test_matched_columns_are_nearest_prevalence_and_not_reused():
    prevalence = np.array([0.50, 0.48, 0.10, 0.49, 0.90])
    # Rule column 0 takes the nearest free column (1); rule column 3 then takes the nearest remaining one (2).
    assert input_drift.matched_columns(prevalence, [0, 3]) == [1, 2]


def test_hd_stop_half_zeroes_half_of_the_recorded_rows_only():
    names = [planted_shift.HD, "OTHER"]
    features = np.array([[1, 5], [0, 5], [2, 0], [3, 1], [0, 2], [4, 0], [1, 3]], dtype=float)
    x, column, changed = planted_shift.shifted_features(features, names, "hd_stop_half", "OTHER",
                                                        np.random.default_rng(0))
    recorded = features[:, 0] > 0
    assert column == planted_shift.HD and changed == recorded.sum() // 2
    assert ((x[:, 0] == 0) & recorded).sum() == recorded.sum() // 2
    np.testing.assert_array_equal(x[~recorded, 0], features[~recorded, 0])
    zeroed = (x[:, 0] == 0) & recorded
    np.testing.assert_array_equal(x[recorded & ~zeroed, 0], features[recorded & ~zeroed, 0])
    np.testing.assert_array_equal(x[:, 1], features[:, 1])


def test_placebo_is_the_unused_feature_with_prevalence_closest_to_haemodialysis():
    names = [planted_shift.HD, "USED", "FAR", "NEAR"]
    features = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]], dtype=float)
    index = {"concepts": [{"rule_features": [planted_shift.HD, "USED"]}]}
    assert planted_shift.placebo_feature(index, features, names) == "NEAR"


def test_death_f1_hand_case():
    p, y = np.array([0.9, 0.6, 0.2, 0.1]), np.array([1, 0, 1, 0])
    # Predicted deaths: rows 0, 1; one true positive; F1 = 2 * 1 / (2 + 2).
    assert planted_shift.death_f1(p, y, 0.5) == pytest.approx(0.5)


def test_rotated_model_keeps_gram_matrix_and_biases():
    from sae import SAE
    torch.manual_seed(0)
    model = SAE(data_dimension=8, scaling_factor=1.5, use_decoder_bias=True, type="ReLU")
    rotated = family_controls.rotated_model(model, np.random.default_rng(0))
    w, v = model.encoder.weight.detach(), rotated.encoder.weight.detach()
    assert not torch.allclose(w, v)
    torch.testing.assert_close(v @ v.T, w @ w.T, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rotated.encoder.bias, model.encoder.bias)
    torch.testing.assert_close(rotated.decoder_bias, model.decoder_bias)


def test_untrained_model_is_the_seeded_construction():
    from sae import SAE
    torch.manual_seed(99)
    trained = SAE(data_dimension=8, scaling_factor=1.5, use_decoder_bias=True, type="ReLU")
    for sae in (None, {"model_type": "ReLU", "scaling_factor": 1.5}):
        model = family_controls.untrained_model(trained, 7, sae)
        torch.manual_seed(7)
        expected = SAE(data_dimension=8, scaling_factor=1.5, use_decoder_bias=True, type="ReLU")
        for name, value in expected.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value)


def test_training_displacement_hand_case():
    initial = np.eye(12)[:2]                      # rows e0, e1
    trained = initial + np.eye(12)[[11, 11]] * [[1], [0]]   # training adds e11 to the first row
    directions = np.eye(12)[:, [11, 5, 6, 7, 8, 0, 1, 2, 3, 4, 9, 10]]  # top-5 = e11, e5..e8
    out = family_controls.training_displacement(initial, trained, directions)
    assert out["relative_change"] == pytest.approx(1 / np.sqrt(2))
    assert out["change_in_top5"] == pytest.approx(1.0)
    assert out["init_in_top5"] == pytest.approx(0.0) and out["init_in_top10"] == pytest.approx(1.0)
    assert out["trained_in_top5"] == pytest.approx(1 / 3)


def test_rule_text_two_rules_joined_by_or():
    rule = {"rules": [
        {"conditions": [{"feature_name": "A ", "operator": ">", "threshold": 0.5}]},
        {"conditions": [{"feature_name": "B", "op": "<=", "value": 2.0},
                        {"feature_name": "C", "operator": ">", "threshold": 1.23456}]},
    ]}
    assert review_sheet.rule_text(rule) == "(A > 0.5) OR (B <= 2 AND C > 1.23)"


def test_top_features_orders_by_absolute_value_skips_nan_and_strips_names():
    row = np.array([0.1, np.nan, -0.5, 0.3])
    names = ["a", "b", "DIAGN_I10 ", " d"]
    assert review_sheet.top_features(row, names, k=3) == "DIAGN_I10 -0.50; d +0.30; a +0.10"
