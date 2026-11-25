import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr

import output_fidelity as fidelity

ALPHAS = np.logspace(-3, 4, 22)


def teacher_from_probability_one(probability_one: np.ndarray) -> np.ndarray:
    return np.column_stack([1 - probability_one, probability_one])


def test_teacher_log_odds_is_log_ratio_and_finite_on_observed_range():
    probability_one = np.linspace(0.005, 0.6, 50)
    teacher = teacher_from_probability_one(probability_one)
    log_odds = fidelity.teacher_log_odds(teacher)
    np.testing.assert_allclose(log_odds, np.log(probability_one / (1 - probability_one)))
    assert np.isfinite(log_odds).all()


def test_fit_readout_recovers_known_log_odds_map_and_keeps_patients_in_one_fold():
    rng = np.random.default_rng(0)
    features = rng.standard_normal((300, 4))
    true_log_odds = features @ np.array([0.8, -0.5, 0.0, 0.3]) - 2.0
    teacher = teacher_from_probability_one(expit(true_log_odds))
    patient_ids = rng.integers(0, 60, 300)
    readout = fidelity.fit_readout(features, teacher, patient_ids, ALPHAS, 5)
    predicted = fidelity.predict_probability(readout, features)
    np.testing.assert_allclose(predicted, expit(true_log_odds), atol=1e-3)
    assert readout["alpha_on_grid_edge"]           # noise-free data: the smallest strength wins


def test_fit_readout_chooses_strength_with_lowest_grouped_cross_validated_error():
    rng = np.random.default_rng(1)
    features = rng.standard_normal((300, 4))
    noisy_log_odds = features @ np.array([0.8, -0.5, 0.0, 0.3]) - 2.0 + rng.normal(0, 1, 300)
    teacher = teacher_from_probability_one(expit(noisy_log_odds))
    patient_ids = rng.integers(0, 60, 300)
    readout = fidelity.fit_readout(features, teacher, patient_ids, ALPHAS, 5)
    assert len(readout["cv_mse"]) == len(ALPHAS)
    assert readout["alpha"] == ALPHAS[np.argmin(readout["cv_mse"])]


def test_patient_folds_keep_each_patient_in_one_fold_and_test_each_row_once():
    rng = np.random.default_rng(2)
    patient_ids = rng.integers(0, 40, 120)
    folds = fidelity.patient_folds(patient_ids, 5)
    assert len(folds) == 5
    for train, test in folds:
        assert not set(patient_ids[train]) & set(patient_ids[test])
    test_rows = np.concatenate([test for _, test in folds])
    assert sorted(test_rows) == list(range(len(patient_ids)))


def test_bootstrap_row_zero_is_plain_error_and_weight_two_counts_twice():
    predicted = np.array([0.1, 0.4, 0.2, 0.9])
    truth = np.array([0.2, 0.3, 0.1, 0.5])
    teacher = teacher_from_probability_one(truth)
    weights = np.array([[1, 1, 1, 1], [1, 2, 1, 0]], dtype=float)
    errors = fidelity.bootstrap_errors(predicted, teacher, weights)
    np.testing.assert_allclose(errors["probability_mse"][0], np.mean((predicted - truth) ** 2))
    np.testing.assert_allclose(errors["probability_mae"][0], np.mean(np.abs(predicted - truth)))
    rows = [0, 1, 1, 2]
    predicted_logit = np.log(predicted[rows] / (1 - predicted[rows]))
    truth_logit = np.log(truth[rows] / (1 - truth[rows]))
    np.testing.assert_allclose(errors["probability_mse"][1], np.mean((predicted[rows] - truth[rows]) ** 2))
    np.testing.assert_allclose(errors["probability_mae"][1], np.mean(np.abs(predicted[rows] - truth[rows])))
    np.testing.assert_allclose(errors["log_odds_mse"][1], np.mean((predicted_logit - truth_logit) ** 2))


def test_constant_benchmark_mse_on_fit_rows_is_variance_about_fit_mean():
    rng = np.random.default_rng(3)
    truth = rng.uniform(0.005, 0.6, 200)
    constant = np.full(200, truth.mean())
    errors = fidelity.bootstrap_errors(constant, teacher_from_probability_one(truth), np.ones((1, 200)))
    np.testing.assert_allclose(errors["probability_mse"][0], truth.var())


def test_bootstrap_spearman_row_zero_is_plain_and_weight_two_counts_twice():
    predicted = np.array([0.1, 0.4, 0.2, 0.9, 0.3])
    truth = np.array([0.2, 0.3, 0.1, 0.5, 0.4])
    teacher = teacher_from_probability_one(truth)
    weights = np.array([[1, 1, 1, 1, 1], [1, 2, 1, 0, 1]], dtype=float)
    spearman = fidelity.bootstrap_spearman(predicted, teacher, weights)
    np.testing.assert_allclose(spearman[0], spearmanr(predicted, truth)[0])
    rows = [0, 1, 1, 2, 4]
    np.testing.assert_allclose(spearman[1], spearmanr(predicted[rows], truth[rows])[0])


def test_stratum_agreement_counts_and_shares_on_hand_built_example():
    truth = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    predicted = np.array([0.1, 0.1, 0.6, 0.2, 0.9, 0.05])
    table = fidelity.stratum_agreement(teacher_from_probability_one(truth), predicted, [0.25, 0.45])
    first, second = table.iloc[0], table.iloc[1]
    assert (first["rows"], first["teacher_above"], first["readout_above"], first["both_above"]) == (6, 4, 2, 2)
    assert first["share_of_teacher_above_also_readout_above"] == 0.5
    assert first["share_of_readout_above_also_teacher_above"] == 1.0
    assert (second["teacher_above"], second["readout_above"], second["both_above"]) == (2, 2, 1)
    assert second["share_of_teacher_above_also_readout_above"] == 0.5
    assert second["share_of_readout_above_also_teacher_above"] == 0.5
