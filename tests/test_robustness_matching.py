import numpy as np
import pytest

from comparison_cache import ComparisonCache
from comparison_runner import (
    ComparisonRunnerConfig,
    DefaultComparisonAdapter,
    MatchingRunnerConfig,
    _SAEData,
    _matching_artifact_rows,
    _select_matching_rows,
)
from robustness_matching import analyze_run_pair
from robustness_recurrence import analyze_robustness_artifacts


def _profiles(left_masks, right_masks):
    return (
        {70: left_masks[0], 80: left_masks[1], 90: left_masks[2]},
        {70: right_masks[0], 80: right_masks[1], 90: right_masks[2]},
    )


def test_matching_supports_different_percentile_assignments_and_rectangles():
    left_directions = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    right_directions = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    left_p70 = np.asarray(
        [[1, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=bool
    )
    right_p70 = np.asarray(
        [[0, 1], [0, 1], [1, 0], [1, 0]], dtype=bool
    )
    left_p80 = left_p70.copy()
    right_p80 = np.asarray(
        [[1, 0], [1, 0], [0, 1], [0, 1]], dtype=bool
    )
    left_p90 = np.zeros_like(left_p70)
    right_p90 = np.zeros_like(right_p70)
    left_profiles, right_profiles = _profiles(
        (left_p70, left_p80, left_p90),
        (right_p70, right_p80, right_p90),
    )

    analysis = analyze_run_pair(
        left_directions,
        right_directions,
        left_profiles,
        right_profiles,
        top_k=3,
    )

    assert analysis.cosine_assignment.pairs == ((0, 0), (1, 1))
    assert analysis.overlap_assignments[70].pairs != analysis.cosine_assignment.pairs
    assert analysis.overlap_assignments[70].pairs != analysis.overlap_assignments[80].pairs
    assert analysis.cosine_assignment.left_to_right[2] is None
    assert np.all(analysis.overlaps[90] == 0)
    assert len(analysis.nearest_hungarian_gaps["cosine"]) == 5


def test_nearest_neighbors_are_deterministic_and_report_collisions_reciprocity_gaps():
    directions = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    targets = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    masks = np.eye(3, dtype=bool)
    target_masks = np.asarray([[1, 0], [0, 1], [0, 0]], dtype=bool)
    analysis = analyze_run_pair(
        directions,
        targets,
        {70: masks, 80: masks, 90: masks},
        {70: target_masks, 80: target_masks, 90: target_masks},
        top_k=3,
    )
    cosine = analysis.nearest_neighbors["cosine"]
    left_best = [row for row in cosine if row.source_side == "left" and row.rank == 1]

    assert [row.target_factor for row in left_best] == [0, 0, 1]
    assert left_best[0].target_collision_count_raw == 2
    assert left_best[1].target_collision_count_raw == 2
    assert left_best[0].reciprocal_raw is True
    assert left_best[1].reciprocal_raw is False
    tied = [
        row for row in cosine
        if row.source_side == "left" and row.source_factor == 0
    ]
    assert [row.target_factor for row in tied] == [0, 1]
    gaps = analysis.nearest_hungarian_gaps["cosine"]
    assert any(row.hungarian_target is None for row in gaps if row.source_side == "left")


def test_invalid_profile_coverage_and_top_k_fail():
    directions = np.eye(2)
    masks = np.eye(2, dtype=bool)
    with pytest.raises(ValueError, match="top_k"):
        analyze_run_pair(directions, directions, {70: masks}, {70: masks}, 0)
    with pytest.raises(ValueError, match="identical"):
        analyze_run_pair(directions, directions, {70: masks}, {80: masks}, 1)


def test_runner_writes_all_pair_artifacts_and_reuses_cache_for_threshold_changes(
    tmp_path,
):
    masks = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=bool)
    runs = []
    for run_id, directions in enumerate(
        (np.eye(2), np.asarray([[0.9, 0.1], [0.1, 0.9]]), np.eye(2))
    ):
        runs.append(
            {
                "idx": run_id,
                "seed": 100 + run_id,
                "decoder_directions": directions,
                "high_activation_profiles": {
                    percentile: {
                        "masks": masks,
                        "thresholds": np.asarray([1.0, 1.0]),
                    }
                    for percentile in (70, 80, 90)
                },
                "high_activation_matrix": masks,
            }
        )
    sae_data = _SAEData(runs=runs, activations={})
    cache = ComparisonCache(tmp_path / "cache")
    adapter = DefaultComparisonAdapter(cache)
    config = ComparisonRunnerConfig(
        matching=MatchingRunnerConfig(
            cosine_analysis_threshold=0.95,
            overlap_analysis_threshold=0.95,
        ),
        show_progress=False,
    )

    first_workspace = tmp_path / "first"
    first_workspace.mkdir()
    np.savez_compressed(first_workspace / "high_activation_profiles.npz", value=masks)
    all_matches, selected = adapter.match(sae_data, config, first_workspace)

    assert len(all_matches) == 6
    assert len(selected) <= len(all_matches)
    manifest = __import__("json").loads(
        (first_workspace / "matching/manifest.json").read_text()
    )
    assert len(manifest["run_pairs"]) == 3
    assert {
        tuple((row["run_i"], row["run_j"])) for row in manifest["run_pairs"]
    } == {(0, 1), (0, 2), (1, 2)}
    assert (first_workspace / "matching/matching_diagnostics.csv").is_file()
    (first_workspace / "sae_manifest.json").write_text(
        __import__("json").dumps(
            [{"run_id": run_id, "n_factors": 2} for run_id in range(3)]
        )
    )
    recurrence_manifest = analyze_robustness_artifacts(
        first_workspace, tmp_path / "analysis", save_plots=False
    )
    assert recurrence_manifest["matching_percentiles"] == [70, 80, 90]
    assert (tmp_path / "analysis/factor_recurrence_primary.csv").is_file()

    second_workspace = tmp_path / "second"
    second_workspace.mkdir()
    np.savez_compressed(second_workspace / "high_activation_profiles.npz", value=masks)
    changed = ComparisonRunnerConfig(
        matching=MatchingRunnerConfig(
            cosine_analysis_threshold=0.2,
            overlap_analysis_threshold=0.8,
            alternative_score_deltas=(0.02, 0.20),
        ),
        show_progress=False,
    )
    adapter.match(sae_data, changed, second_workspace)

    matching_events = [event for event in cache.events if event.stage == "matching_pair"]
    assert [event.status for event in matching_events[-3:]] == ["hit", "hit", "hit"]


def test_threshold_equality_and_rank_two_three_delta_flags_pass():
    left = np.asarray([[1.0, 0.0]])
    right = np.asarray(
        [[1.0, 0.0], [0.95, np.sqrt(1 - 0.95**2)], [0.90, np.sqrt(1 - 0.90**2)]]
    )
    left_masks = np.ones((10, 1), dtype=bool)
    right_masks = np.column_stack(
        (
            np.ones(10, dtype=bool),
            np.asarray([1] * 7 + [0] * 3, dtype=bool),
            np.asarray([1] * 6 + [0] * 4, dtype=bool),
        )
    )
    analysis = analyze_run_pair(
        left,
        right,
        {p: left_masks for p in (70, 80, 90)},
        {p: right_masks for p in (70, 80, 90)},
        3,
    )
    rows = _matching_artifact_rows(
        analysis,
        0,
        1,
        MatchingRunnerConfig(
            cosine_analysis_threshold=0.60,
            overlap_analysis_threshold=0.70,
        ),
    )["nearest_neighbors"]
    cosine_left = [
        row
        for row in rows
        if row["metric"] == "cosine" and row["source_side"] == "left"
    ]
    overlap_left = [
        row
        for row in rows
        if row["metric"] == "overlap_p70" and row["source_side"] == "left"
    ]

    assert cosine_left[1]["valid_alternative_delta_0_05"] is True
    assert cosine_left[2]["valid_alternative_delta_0_10"] is True
    assert overlap_left[1]["score"] == 0.70
    assert overlap_left[1]["passes_threshold"] is True


def test_overlap_selector_keeps_percentile_with_most_factors_and_high_tie_break():
    directions = np.eye(2)
    left_masks = np.asarray(
        [[1, 0], [1, 0], [0, 1], [0, 1]], dtype=bool
    )
    same = left_masks.copy()
    one_match = np.asarray(
        [[1, 0], [1, 0], [0, 0], [0, 0]], dtype=bool
    )
    swapped = left_masks[:, ::-1]
    analysis = analyze_run_pair(
        directions,
        directions,
        {70: left_masks, 80: left_masks, 90: left_masks},
        {70: same, 80: one_match, 90: swapped},
        2,
    )

    all_rows, selected, percentile = _select_matching_rows(
        analysis,
        0,
        1,
        MatchingRunnerConfig(criterion="overlap"),
    )

    assert percentile == 90
    assert len(selected) == 2
    assert {(row["original_concept"], row["best_pair"]) for row in all_rows} == {
        (0, 1),
        (1, 0),
    }
    assert all(row["overlap_percentile"] == 90 for row in selected)


def test_cosine_selector_uses_cosine_analysis_threshold():
    left = np.eye(2)
    right = np.asarray([[1.0, 0.0], [np.sqrt(0.75), 0.5]])
    masks = np.eye(2, dtype=bool)
    analysis = analyze_run_pair(
        left,
        right,
        {p: masks for p in (70, 80, 90)},
        {p: masks for p in (70, 80, 90)},
        2,
    )

    all_rows, selected, percentile = _select_matching_rows(
        analysis,
        0,
        1,
        MatchingRunnerConfig(cosine_analysis_threshold=0.60),
    )

    assert len(all_rows) == 2
    assert len(selected) == 1
    assert selected[0]["cos_sim"] == 1.0
    assert percentile == 90
