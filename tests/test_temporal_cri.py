from dataclasses import replace

import pytest

from temporal_cri import (
    CRIAnalysisConfig, _preservation, _ratio_utility, aggregate_family_scores,
    build_family_universe, compute_member_utilities, evaluate_loyo, summarize_system_scores,
)
from temporal_unified_analysis import UnifiedAnalysisConfig


def _row(**changes):
    row = {
        "reference_year": 2007, "patient_split_seed": 42, "factor_family_uid": "family",
        "member_sae_seed": 42, "member_factor_id": 3, "activation_target": .5,
        "cohort_view": "all_comer", "temporal_distance": 0, "test_year": 2007,
        "matching_view": "intersection", "rule_source": "semantic", "target_role": "primary",
        "cosine_threshold": .6, "overlap_percentile": 70, "overlap_threshold": .7,
        "geometric_factor_recurrence": .75, "f2": .8, "jaccard": .6,
        "prevalence_ratio": 1., "activation_magnitude": 2., "status": "stable",
    }
    row.update(changes)
    return row


def test_utilities_are_symmetric_and_reference_floor_is_explicit():
    assert _ratio_utility(.5, 1) == pytest.approx(_ratio_utility(2., 1))
    assert _preservation(.4, .8, 1e-6) == pytest.approx(.5)
    assert _preservation(.4, 0., 1e-6) is None


def test_dead_member_stays_in_frozen_denominator_and_scores_zero_retention():
    unified = UnifiedAnalysisConfig(quiescent_seconds=0)
    config = CRIAnalysisConfig()
    reference = _row()
    dead = _row(temporal_distance=1, test_year=2008, status="dead_absent", f2=.4, jaccard=.3,
                prevalence_ratio=0., activation_magnitude=0.)
    universe = build_family_universe([reference, dead], unified, config)
    members = compute_member_utilities([reference, dead], universe, [
        {"activation_target": .5, "metric": "prevalence", "tau": 1},
        {"activation_target": .5, "metric": "activation", "tau": 1},
    ], unified, config)
    result = next(row for row in members if row["temporal_distance"] == 1)
    assert result["utility_available"] is True
    assert result["u_prevalence"] == result["u_activation"] == 0.
    families = aggregate_family_scores(universe, members)
    assert next(row for row in families if row["temporal_distance"] == 1)["family_complete"] is True


def test_insufficient_support_is_coverage_missing_not_fake_degradation():
    unified = UnifiedAnalysisConfig(quiescent_seconds=0)
    config = CRIAnalysisConfig()
    rows = [_row(), _row(temporal_distance=1, test_year=2008, status="insufficient_future_support")]
    universe = build_family_universe(rows, unified, config)
    members = compute_member_utilities(rows, universe, [
        {"activation_target": .5, "metric": "prevalence", "tau": 1},
        {"activation_target": .5, "metric": "activation", "tau": 1},
    ], unified, config)
    families = aggregate_family_scores(universe, members)
    system = summarize_system_scores(families, config)
    result = next(row for row in system if row["temporal_distance"] == 1)
    assert result["coverage"] == 0.
    assert result["median_cri"] is None


def test_family_one_vote_not_one_vote_per_member():
    config = CRIAnalysisConfig()
    families = [
        {"reference_year": 2007, "patient_split_seed": 42, "factor_family_uid": "a", "activation_target": .5,
         "cohort_view": "all_comer", "temporal_distance": 1, "family_complete": True, "cri_arithmetic": .2,
         "cri_geometric": .2, "semantic_cohort_utility": .2, "activation_utility": .2},
        {"reference_year": 2007, "patient_split_seed": 42, "factor_family_uid": "b", "activation_target": .5,
         "cohort_view": "all_comer", "temporal_distance": 1, "family_complete": True, "cri_arithmetic": .8,
         "cri_geometric": .8, "semantic_cohort_utility": .8, "activation_utility": .8},
    ]
    result = summarize_system_scores(families, config)[0]
    assert result["median_cri"] == pytest.approx(.5)


def test_loyo_never_leaks_held_reference_rows_into_training_fold():
    rows = []
    for reference in (2007, 2008, 2009):
        for distance in (1, 2):
            rows.append({"reference_year": reference, "patient_split_seed": 42,
                         "cohort_view": "all_comer", "activation_target": .5,
                         "temporal_distance": distance, "target": .5, "f1_d": .6,
                         "delta_f1_d": -.1, "cri_d": .7, "delta_cri_d": -.1})
    _, folds, _ = evaluate_loyo(rows)
    assert all(row["entire_reference_held_out"] for row in folds)
    assert {row["held_out_reference_year"] for row in folds} == {2007, 2008, 2009}
