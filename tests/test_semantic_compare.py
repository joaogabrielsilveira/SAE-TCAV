import numpy as np
import pytest

from semantic_compare import (
    cluster_rule_families,
    compare_rule_sets_symmetric,
    recurrent_representatives,
    rule_similarity,
)
from semantic_rules import Condition, Rule, RuleSet
from stable_rule_backend import CandidateRuleOccurrence


def _rule(
    rule_id,
    *,
    feature_index=0,
    feature_name="creatinine",
    operator=">",
    threshold=0.5,
    normalized_threshold=0.5,
    clinical_groups=("renal",),
):
    return Rule(
        rule_id,
        (
            Condition(
                feature_index,
                feature_name,
                operator,
                threshold,
                clinical_groups=clinical_groups,
                normalized_threshold=normalized_threshold,
            ),
        ),
    )


def _occurrence(rule, *, bootstrap_id, tree_id=0, leaf_id=1):
    return CandidateRuleOccurrence(
        rule=rule,
        source="test",
        bootstrap_id=bootstrap_id,
        tree_id=tree_id,
        bootstrap_seed=100 + bootstrap_id,
        tree_seed=1000 + tree_id,
        leaf_id=leaf_id,
        fit_sample_count=100,
        fit_positive_count=25,
        fit_selected_count=25,
        fit_true_positive_count=20,
        fit_precision=0.8,
        fit_recall=0.8,
        oob_sample_count=30,
        oob_selected_count=8,
        oob_true_positive_count=6,
        oob_precision=0.75,
        oob_recall=0.6,
    )


def test_rule_equivalence_uses_cohorts_groups_thresholds_and_direction():
    signal = np.linspace(0.0, 1.0, 101)
    # Columns deliberately encode correlated clinical substitutes.
    X = np.column_stack((signal, signal))
    baseline = _rule("baseline")
    nearby_threshold = _rule(
        "nearby", threshold=0.52, normalized_threshold=0.52
    )
    correlated_feature = _rule(
        "correlated",
        feature_index=1,
        feature_name="egfr",
    )
    opposite_direction = _rule("opposite", operator="<=")

    nearby = rule_similarity(baseline, nearby_threshold, X)
    substitute = rule_similarity(baseline, correlated_feature, X)
    opposite = rule_similarity(baseline, opposite_direction, X)

    assert nearby.total >= 0.70
    assert nearby.cohort_jaccard > 0.9
    assert substitute.total >= 0.70
    assert substitute.exact_feature_jaccard == 0.0
    assert substitute.clinical_group_jaccard == 1.0
    assert opposite.total < 0.70
    assert opposite.threshold_direction_compatibility == 0.0

    families = cluster_rule_families(
        (
            _occurrence(baseline, bootstrap_id=0),
            _occurrence(nearby_threshold, bootstrap_id=0, tree_id=1),
            _occurrence(correlated_feature, bootstrap_id=2),
            _occurrence(opposite_direction, bootstrap_id=1),
        ),
        X,
        total_bootstraps=4,
    )
    assert sorted(len(family.occurrences) for family in families) == [1, 3]


def test_recurrence_counts_bootstraps_not_tree_occurrences():
    signal = np.linspace(0.0, 1.0, 101)
    X = np.column_stack((signal, signal))
    occurrences = (
        _occurrence(_rule("base"), bootstrap_id=0, tree_id=0),
        _occurrence(
            _rule("same-bootstrap", threshold=0.51, normalized_threshold=0.51),
            bootstrap_id=0,
            tree_id=1,
        ),
        _occurrence(
            _rule("clinical-substitute", feature_index=1, feature_name="egfr"),
            bootstrap_id=2,
        ),
    )

    families = cluster_rule_families(occurrences, X, total_bootstraps=4)

    assert len(families) == 1
    family = families[0]
    assert len(family.occurrences) == 3
    assert family.recurrence_count == 2
    assert family.recurrence_frequency == 0.5
    assert family.feature_recurrence == {"creatinine": 0.25, "egfr": 0.25}
    assert family.clinical_group_recurrence == {"renal": 0.5}
    assert family.cohort_overlap_stability > 0.9
    assert recurrent_representatives(families, 0.5) == (
        family.representative_rule,
    )
    assert recurrent_representatives(families, 0.5001) == ()


def test_symmetric_transfer_retains_directional_asymmetry():
    X_final = np.arange(10, dtype=float).reshape(-1, 1)
    broad = _rule(
        "broad", threshold=3.5, normalized_threshold=None, clinical_groups=("lab",)
    )
    narrow = _rule(
        "narrow", threshold=6.5, normalized_threshold=None, clinical_groups=("lab",)
    )
    broad_set = RuleSet((broad,), threshold_name="broad")
    narrow_set = RuleSet((narrow,), threshold_name="broad")
    target_i = broad_set.mask(X_final)
    target_j = narrow_set.mask(X_final)

    transfer = compare_rule_sets_symmetric(
        broad_set, target_i, narrow_set, target_j, X_final
    )

    assert transfer.i_to_j.precision == 0.5
    assert transfer.i_to_j.recall == 1.0
    assert transfer.i_to_j.f2 == pytest.approx(5.0 / 6.0)
    assert transfer.j_to_i.precision == 1.0
    assert transfer.j_to_i.recall == 0.5
    assert transfer.j_to_i.f2 == pytest.approx(5.0 / 9.0)
    assert transfer.mean["f2"] == pytest.approx(25.0 / 36.0)
    assert transfer.minimum["f2"] == pytest.approx(5.0 / 9.0)
    assert transfer.target_cohort_jaccard == 0.5
    assert transfer.selected_cohort_jaccard == 0.5
    assert transfer.exact_feature_equal
    assert transfer.clinical_group_equal


def test_symmetric_transfer_requires_one_aligned_final_record_set():
    X_final = np.arange(10, dtype=float).reshape(-1, 1)
    rule_set = RuleSet((_rule("rule", threshold=4.5),))
    target = rule_set.mask(X_final)

    with pytest.raises(ValueError, match="aligned one-dimensional"):
        compare_rule_sets_symmetric(rule_set, target, rule_set, target[:-1], X_final)
    with pytest.raises(ValueError, match="identical records"):
        compare_rule_sets_symmetric(rule_set, target, rule_set, target, X_final[:-1])


def test_identical_rules_and_targets_have_perfect_bidirectional_transfer():
    X_final = np.arange(10, dtype=float).reshape(-1, 1)
    rule_set = RuleSet((_rule("rule", threshold=4.5),))
    target = rule_set.mask(X_final)

    transfer = compare_rule_sets_symmetric(
        rule_set, target, rule_set, target.copy(), X_final
    )

    assert transfer.i_to_j.f2 == 1.0
    assert transfer.j_to_i.f2 == 1.0
    assert transfer.mean["f2"] == 1.0
    assert transfer.minimum["f2"] == 1.0
    assert transfer.target_cohort_jaccard == 1.0
    assert transfer.selected_cohort_jaccard == 1.0
