import numpy as np

import tcav_geometry as geometry


def test_per_fit_tcav_counts_zeros_in_denominator_and_drops_failed_fits():
    sensitivity = np.array([[1.0, np.nan, -1.0, 0.5],
                            [0.0, np.nan, 2.0, 0.5],
                            [-3.0, np.nan, 1.0, -0.5],
                            [2.0, np.nan, 0.0, 0.5]])
    groups = np.array([[True, True], [True, False], [True, True], [False, True]])
    years = np.array([2007, 2007, 2008, 2008])
    tcav = geometry.per_fit_tcav(sensitivity, groups, years, (2007, 2008, 2009))
    assert tcav.shape == (2, 3, 3)
    np.testing.assert_allclose(tcav[0, 0], [0.5, 0.5, 1.0])     # zero counts in the denominator
    np.testing.assert_allclose(tcav[0, 1], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(tcav[1, 1], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(tcav[1, 0], [1.0, 0.0, 1.0])
    assert np.isnan(tcav[:, 2]).all()                            # empty year


def test_sensitivity_comparison_reports_difference_and_sign_flips():
    gradients = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    directions = np.array([[1.0, 0.0], [np.nan, np.nan]])
    stored = np.array([[1.0, np.nan], [0.0, np.nan], [-1e-9, np.nan]])
    result = geometry.compare_sensitivities(gradients, directions, stored)
    assert result["valid_fits"] == 1
    np.testing.assert_allclose(result["max_abs_difference"], 1.0 + 1e-9)
    assert result["sign_disagreements"] == 1
    assert result["nonzero_rows"] == 2


def test_log_odds_gradient_keeps_signs_and_marks_saturated_rows():
    gradients = np.array([[0.2, -0.1], [0.3, 0.4], [1.0, 1.0], [0.5, -0.5]])
    probability = np.array([0.5, 0.1, 1.0, 0.0])
    log_odds = geometry.log_odds_gradients(gradients, probability)
    np.testing.assert_allclose(log_odds[0], [0.8, -0.4])
    np.testing.assert_allclose(log_odds[1], np.array([0.3, 0.4]) / 0.09)
    assert np.isnan(log_odds[2:]).all()
    assert (np.sign(log_odds[:2]) == np.sign(gradients[:2])).all()


def test_replay_summary_uses_pilot_tolerances():
    stored_p = np.array([[0.9, 0.1], [0.5, 0.5]])
    stored_g = np.array([[1.0, -2.0], [0.0, 3.0]])
    close = geometry.replay_summary(stored_p + 5e-6, stored_p, stored_g * (1 + 5e-4), stored_g)
    assert close["probability_within_tolerance"] and close["gradient_within_tolerance"]
    np.testing.assert_allclose(close["probability_max_abs_error"], 5e-6, rtol=1e-6)
    far = geometry.replay_summary(stored_p + 1e-3, stored_p, stored_g * 1.01, stored_g)
    assert not far["probability_within_tolerance"] and not far["gradient_within_tolerance"]
    np.testing.assert_allclose(far["gradient_max_relative_error"], 0.01, rtol=1e-6)


def test_shared_gradient_predicts_tcav_exactly():
    rng = np.random.default_rng(0)
    shared = rng.standard_normal(6)
    gradients = rng.uniform(0.1, 2.0, (40, 1)) * shared
    directions = rng.standard_normal((25, 6))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    summary = geometry.coherence(gradients)
    np.testing.assert_allclose(summary["resultant_length"], 1.0)
    np.testing.assert_allclose(summary["leading_share"], 1.0)
    np.testing.assert_allclose(summary["cosines"], 1.0)
    observed = geometry.positive_share(gradients @ directions.T, np.ones((40, 1), dtype=bool))[0]
    np.testing.assert_array_equal(observed, geometry.predicted_tcav(summary["mean_direction"], directions))
    assert set(observed) <= {0.0, 1.0}
    angles = geometry.boundary_angle(summary["mean_direction"], directions)
    assert ((angles > 0) == (observed == 1)).all()


def test_opposite_gradients_have_no_predicted_side():
    gradients = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    summary = geometry.coherence(gradients)
    assert summary["rows_used"] == 2 and summary["rows_dropped"] == 1
    assert summary["resultant_length"] == 0
    assert np.isnan(geometry.predicted_tcav(summary["mean_direction"], np.eye(2))).all()


def test_random_direction_families_live_in_their_subspace():
    rng = np.random.default_rng(1)
    embeddings = rng.standard_normal((200, 8)) * np.arange(8, 0, -1)
    mean, axes, variances = geometry.principal_axes(embeddings)
    assert (np.diff(variances) <= 0).all()
    np.testing.assert_allclose(axes.T @ axes, np.eye(8), atol=1e-10)
    for family in geometry.DIRECTION_FAMILIES:
        directions = geometry.random_directions(rng, 50, axes, 3, family)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
        top_part = np.linalg.norm(directions @ axes[:, :3], axis=1)
        if family == "top":
            np.testing.assert_allclose(top_part, 1.0)
        if family == "complement":
            np.testing.assert_allclose(top_part, 0.0, atol=1e-12)


def test_interior_share_and_standard_error():
    tcav = np.array([0.0, 0.05, 0.1, 0.5, 0.9, 1.0, 0.95, 0.3, np.nan])
    share, error = geometry.interior_share(tcav)
    np.testing.assert_allclose(share, 2 / 8)
    np.testing.assert_allclose(error, np.sqrt(0.25 * 0.75 / 8))


def test_projection_split_sums_to_full_derivative():
    rng = np.random.default_rng(2)
    _, axes, _ = geometry.principal_axes(rng.standard_normal((100, 7)) * np.arange(1, 8))
    gradients, directions = rng.standard_normal((30, 7)), rng.standard_normal((5, 7))
    inside, outside = geometry.split_direction(directions, axes, 3)
    np.testing.assert_allclose(gradients @ inside.T + gradients @ outside.T, gradients @ directions.T)
    np.testing.assert_allclose(inside @ outside.T, 0.0, atol=1e-12)
    np.testing.assert_allclose(outside @ axes[:, :3], 0.0, atol=1e-12)


def test_participation_ratio_of_known_spectrum():
    np.testing.assert_allclose(geometry.participation_ratio(np.diag([1.0, 1.0, 1.0, 0.0])), 3.0)
    np.testing.assert_allclose(geometry.participation_ratio(np.diag([1.0, 0.0, 0.0])), 1.0)
    rotation, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((4, 4)))
    matrix = rotation @ np.diag([4.0, 1.0, 0.0, 0.0]) @ rotation.T
    np.testing.assert_allclose(geometry.participation_ratio(matrix), 25 / 17)


def test_energy_share_of_isotropic_gradients_is_near_chance():
    rng = np.random.default_rng(4)
    axes = np.linalg.qr(rng.standard_normal((10, 10)))[0]
    gradients = rng.standard_normal((20000, 10))
    np.testing.assert_allclose(geometry.energy_share(gradients, axes, 3), 0.3, atol=0.01)
    np.testing.assert_allclose(geometry.energy_share(gradients @ axes[:, :3] @ axes[:, :3].T, axes, 3), 1.0)


def test_standardized_chart_keeps_derivative():
    rng = np.random.default_rng(5)
    gradients, directions = rng.standard_normal((20, 6)), rng.standard_normal((4, 6))
    scale = rng.uniform(0.1, 5.0, 6)
    chart_gradients, chart_directions = geometry.standardized_chart(gradients, directions, scale)
    np.testing.assert_allclose(chart_gradients @ chart_directions.T, gradients @ directions.T)


def test_perturbation_baselines_and_orthogonality():
    rng = np.random.default_rng(6)
    embeddings = rng.standard_normal((50, 8)) * np.arange(8, 0, -1) + 3.0
    mean, axes, _ = geometry.principal_axes(embeddings)
    length = geometry.top_axes_length(embeddings, mean, axes, 3)
    centred = (embeddings - mean) @ axes[:, :3]
    np.testing.assert_allclose(length, np.sqrt(np.mean(np.sum(centred ** 2, axis=1))))
    noise = geometry.noise_draws(rng, embeddings, axes, 3)
    for family in geometry.PERTURBATION_FAMILIES:
        baseline = 1.0 if family == "dilation" else 0.0
        np.testing.assert_allclose(geometry.perturb(embeddings, mean, noise, length, baseline, family), embeddings)
    moved = geometry.perturb(embeddings, mean, noise, length, 2.0, "off_axis") - embeddings
    np.testing.assert_allclose(moved @ axes[:, :3], 0.0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(moved, axis=1), 2.0 * length)
    full = geometry.perturb(embeddings, mean, noise, length, 0.5, "full_space") - embeddings
    np.testing.assert_allclose(np.linalg.norm(full, axis=1), 0.5 * length)
    np.testing.assert_allclose(geometry.perturb(embeddings, mean, noise, length, 0.0, "dilation"),
                               np.broadcast_to(mean, embeddings.shape))


def test_build_rng_is_repeatable_and_separate_per_build_and_stream():
    first = geometry.build_rng(7, "r1", 2007, 0).standard_normal(5)
    np.testing.assert_array_equal(first, geometry.build_rng(7, "r1", 2007, 0).standard_normal(5))
    others = [geometry.build_rng(8, "r1", 2007, 0), geometry.build_rng(7, "r2", 2007, 0),
              geometry.build_rng(7, "r1", 2008, 0), geometry.build_rng(7, "r1", 2007, 1)]
    for other in others:
        assert not np.array_equal(first, other.standard_normal(5))
