import numpy as np

from comparison_runner import _scale_embeddings_from_semantic_fit
from sae_compare import high_activation_matrix, high_activation_profiles


def test_scaler_fits_semantic_test_subset_and_transforms_all_splits():
    train = np.asarray([[100.0, 1000.0], [200.0, 2000.0]])
    test = np.asarray(
        [[0.0, 10.0], [2.0, 14.0], [50.0, 90.0], [80.0, 120.0]]
    )
    fit_indices = np.asarray([0, 1])

    train_scaled, test_scaled, scaler = _scale_embeddings_from_semantic_fit(
        train, test, fit_indices
    )

    np.testing.assert_allclose(scaler.mean_, [1.0, 12.0])
    np.testing.assert_allclose(scaler.scale_, [1.0, 2.0])
    np.testing.assert_allclose(test_scaled[fit_indices].mean(axis=0), 0.0)
    np.testing.assert_allclose(train_scaled, scaler.transform(train))
    np.testing.assert_allclose(test_scaled, scaler.transform(test))


def test_profiles_use_positive_values_strict_thresholds_and_dead_factors():
    concepts = np.asarray(
        [
            [-4.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 0.0, 2.0, 2.0],
            [3.0, 0.0, 3.0, 2.0],
        ]
    )
    profiles = high_activation_profiles(concepts, (70, 80, 90))

    assert tuple(profiles) == (70, 80, 90)
    assert np.isinf(profiles[70]["thresholds"][1])
    assert not profiles[70]["masks"][:, 1].any()
    assert not profiles[70]["masks"][:3, 2].any()
    assert profiles[90]["thresholds"][3] == 2.0
    assert not profiles[90]["masks"][:, 3].any()
    for percentile in profiles:
        expected = concepts > profiles[percentile]["thresholds"]
        np.testing.assert_array_equal(profiles[percentile]["masks"], expected)
    np.testing.assert_array_equal(
        high_activation_matrix(concepts, perc=90), profiles[90]["masks"]
    )
