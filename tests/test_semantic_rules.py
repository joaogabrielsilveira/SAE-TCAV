import numpy as np

from semantic_rules import (
    ActivationTargetSpec,
    Condition,
    Rule,
    RuleSet,
    RuleSetSelectionConfig,
    binary_metrics,
    evaluate_rule_set,
    fit_activation_target,
    select_rule_set,
)


def _rule(rule_id, feature, operator, threshold):
    return Rule(rule_id, (Condition(feature, f"x{feature}", operator, threshold),))


def test_or_metrics_use_union_not_mean_of_rule_metrics():
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]], dtype=float)
    y = np.array([0, 1, 1, 1, 0], dtype=bool)
    left = _rule("left", 0, ">", 0.5)
    right = _rule("right", 1, ">", 0.5)
    aggregate = evaluate_rule_set(RuleSet((left, right)), X, y)
    individual_recall_mean = np.mean([
        evaluate_rule_set(RuleSet((left,)), X, y).recall,
        evaluate_rule_set(RuleSet((right,)), X, y).recall,
    ])
    assert aggregate.n_selected == 4
    assert aggregate.recall == 1.0
    assert aggregate.recall != individual_recall_mean
    assert aggregate.precision == 0.75


def test_activation_cutoff_is_fit_once_then_frozen():
    target = fit_activation_target(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        ActivationTargetSpec("core", 0.2),
    )
    original_cutoff = target.cutoff
    selected = target.apply(np.array([0.0, 3.1, 1000.0]))
    assert target.cutoff == original_cutoff
    assert selected.tolist() == [False, False, True]


def test_constrained_selection_uses_union_and_supports_both_objectives():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=bool)
    candidates = [_rule("narrow", 0, ">", 6.5), _rule("broad", 0, ">", 3.5)]
    common = dict(min_precision=0.5, min_lift=1.0, max_rules=1, min_marginal_recall=0.0)
    for objective in ("f2", "recall"):
        result = select_rule_set(
            candidates, X, y, RuleSetSelectionConfig(objective=objective, **common)
        )
        assert result.feasible
        assert result.rule_set.rules[0].rule_id == "broad"
        assert result.metrics.recall == 1.0


def test_no_valid_rule_and_no_positive_cases_are_explicit():
    X = np.arange(4, dtype=float).reshape(-1, 1)
    no_candidates = select_rule_set([], X, np.array([0, 1, 0, 1]))
    assert not no_candidates.feasible
    assert no_candidates.reason == "no_eligible_candidates"
    no_positives = select_rule_set([_rule("all", 0, ">", -1)], X, np.zeros(4))
    assert not no_positives.feasible
    assert no_positives.reason == "no_positive_selection_targets"


def test_binary_metrics_handles_empty_and_rare_targets():
    empty = binary_metrics(np.zeros(5), np.zeros(5))
    assert empty.precision == empty.recall == empty.f2 == empty.lift == 0.0
    assert empty.jaccard == 1.0
    rare = binary_metrics(np.array([1, 0, 0, 0]), np.array([1, 1, 0, 0]))
    assert rare.recall == 1.0
    assert rare.precision == 0.5
    assert rare.lift == 2.0
