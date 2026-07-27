import numpy as np
import pytest

from semantic_splits import semantic_test_subsplits


def test_grouped_split_has_no_patient_leakage_and_is_deterministic():
    patients = np.repeat(np.arange(40), 3)
    y = np.tile([0, 0, 1], 40)
    first = semantic_test_subsplits(y, patients, rng_seed=17)
    second = semantic_test_subsplits(y, patients, rng_seed=17)
    names = ["idx_semantic_fit", "idx_semantic_select", "idx_tcav_eval", "idx_semantic_final"]
    seen = set()
    for name in names:
        assert np.array_equal(first[name], second[name])
        split_patients = set(patients[first[name]])
        assert not seen & split_patients
        seen |= split_patients
    assert seen == set(patients)


def test_grouped_split_rejects_too_few_groups():
    with pytest.raises(ValueError, match="four"):
        semantic_test_subsplits(np.array([0, 1, 0]), np.array([1, 2, 3]))
