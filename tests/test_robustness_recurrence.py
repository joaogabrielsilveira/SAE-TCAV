from robustness_recurrence import compute_factor_recurrence


def _cosine(run_i, run_j, factor_i, factor_j, cosine, overlaps):
    return {
        "run_i": run_i,
        "run_j": run_j,
        "factor_i": factor_i,
        "factor_j": factor_j,
        "cos_sim": cosine,
        **{f"overlap_p{p}": value for p, value in zip((70, 80, 90), overlaps)},
    }


def _overlap(run_i, run_j, factor_i, factor_j, percentile, score):
    return {
        "run_i": run_i,
        "run_j": run_j,
        "factor_i": factor_i,
        "factor_j": factor_j,
        "percentile": percentile,
        "overlap": score,
    }


def test_recurrence_uses_r_minus_one_reverse_orientation_and_missing_failures():
    cosine = [
        _cosine(0, 1, 0, 1, 0.60, (0.70, 0.70, 0.1)),
        _cosine(0, 2, 0, 0, 0.59, (0.70, 0.1, 0.70)),
        _cosine(1, 2, 1, 0, 0.90, (0.1, 0.70, 0.70)),
    ]
    overlap = [
        _overlap(left, right, 0, 0, percentile, 0.70)
        for left, right in ((0, 1), (0, 2), (1, 2))
        for percentile in (70, 80, 90)
    ]

    primary, secondary, highlights = compute_factor_recurrence(
        {0: 2, 1: 2, 2: 1}, cosine, overlap
    )
    run0_factor0 = next(
        row for row in primary if row["run_id"] == 0 and row["factor_id"] == 0
    )
    run1_factor1 = next(
        row for row in primary if row["run_id"] == 1 and row["factor_id"] == 1
    )
    missing = next(
        row for row in primary if row["run_id"] == 0 and row["factor_id"] == 1
    )

    assert run0_factor0["comparison_count"] == 2
    assert run0_factor0["cosine_recurrence"] == 0.5
    assert run0_factor0["cosine_recurrent"] is False
    assert run1_factor1["cosine_recurrence"] == 1.0
    assert missing["assigned_count"] == 0
    assert missing["cosine_recurrence"] == 0.0
    assert all(row["comparison_count"] == 2 for row in secondary)
