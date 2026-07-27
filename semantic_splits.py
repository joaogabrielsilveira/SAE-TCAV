"""Leakage-safe record splits for semantic SAE experiments."""

from __future__ import annotations

import numpy as np


def semantic_test_subsplits(
    y_test: np.ndarray,
    patient_ids: np.ndarray,
    rng_seed: int = 42,
    fractions: tuple[float, float, float, float] = (0.33, 0.335, 0.1675, 0.1675),
) -> dict[str, np.ndarray]:
    """Split patient groups into fit, selection, TCAV, and final partitions."""

    y = np.asarray(y_test, dtype=int)
    groups = np.asarray(patient_ids)
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(groups):
        raise ValueError("y_test and patient_ids must be aligned one-dimensional arrays")
    fractions_array = np.asarray(fractions, dtype=float)
    if len(fractions_array) != 4 or np.any(fractions_array <= 0):
        raise ValueError("fractions must contain four positive values")
    fractions_array /= fractions_array.sum()

    unique_groups, inverse = np.unique(groups.astype(str), return_inverse=True)
    if len(unique_groups) < 4:
        raise ValueError("At least four patient groups are required")

    rng = np.random.default_rng(rng_seed)
    assignments = np.full(len(unique_groups), -1, dtype=int)
    group_positive = np.bincount(inverse, weights=y) > 0

    # Allocate each outcome stratum separately. Largest-remainder rounding
    # keeps configured proportions without requiring sklearn in this module.
    for stratum in (False, True):
        members = np.flatnonzero(group_positive == stratum)
        rng.shuffle(members)
        exact_counts = fractions_array * len(members)
        counts = np.floor(exact_counts).astype(int)
        remainder_order = np.argsort(-(exact_counts - counts), kind="stable")
        for partition in remainder_order[: len(members) - int(counts.sum())]:
            counts[partition] += 1
        start = 0
        for partition, count in enumerate(counts):
            assignments[members[start : start + count]] = partition
            start += count

    # Very small strata can leave a partition empty. Move one deterministic
    # group from largest donor so every role remains represented.
    partition_counts = np.bincount(assignments, minlength=4)
    for empty_partition in np.flatnonzero(partition_counts == 0):
        donor = int(np.argmax(partition_counts))
        donor_groups = np.flatnonzero(assignments == donor)
        moved_group = int(donor_groups[-1])
        assignments[moved_group] = empty_partition
        partition_counts[donor] -= 1
        partition_counts[empty_partition] += 1

    indices = [np.flatnonzero(assignments[inverse] == partition) for partition in range(4)]
    if sum(map(len, indices)) != len(y):
        raise AssertionError("Grouped split lost records")
    patient_sets = [set(groups[index].tolist()) for index in indices]
    if any(patient_sets[i] & patient_sets[j] for i in range(4) for j in range(i + 1, 4)):
        raise AssertionError("Patient leakage between semantic splits")

    fit, select, tcav_eval, final = (np.asarray(index, dtype=int) for index in indices)
    return {
        "idx_semantic_fit": fit,
        "idx_semantic_select": select,
        "idx_tcav_eval": tcav_eval,
        "idx_semantic_final": final,
        "idx_test_discover": fit,
        "idx_test_cav_train": select,
        "idx_test_tcav_eval": tcav_eval,
        "idx_test_held_out": final,
    }
