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


def _conjunctive_rule(rule_id, conditions):
    return Rule(
        rule_id,
        tuple(
            Condition(feature, f"x{feature}", operator, threshold)
            for feature, operator, threshold in conditions
        ),
    )


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


def test_selection_diagnostics_record_candidate_and_support_funnel_counts():
    X = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    short = _rule("short", 0, ">", 0.5)
    long = _conjunctive_rule("long", [(0, ">", 0.5), (1, ">", 0.5)])
    result = select_rule_set(
        [short, long],
        X,
        np.array([0, 1, 1]),
        RuleSetSelectionConfig(
            min_precision=0.0,
            min_lift=0.0,
            max_rule_length=1,
            min_marginal_recall=0.0,
        ),
    )
    assert result.feasible
    assert result.diagnostics.n_input_candidates == 2
    assert result.diagnostics.n_eligible_candidates == result.n_candidates == 1
    assert result.diagnostics.n_excluded_by_rule_length == 1
    assert result.diagnostics.n_positive_selection_targets == 2
    assert result.diagnostics.constraint_rescues.applicable == ()


def test_rule_length_failure_has_an_exclusive_reason_and_overlapping_rescue_flag():
    X = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    candidate = _conjunctive_rule(
        "long", [(0, ">", 0.5), (1, ">", 0.5)]
    )
    result = select_rule_set(
        [candidate],
        X,
        np.array([0, 1, 1]),
        RuleSetSelectionConfig(
            min_precision=0.0,
            min_lift=0.0,
            max_rule_length=1,
            min_marginal_recall=0.0,
        ),
    )
    rescues = result.diagnostics.constraint_rescues
    assert result.reason == "no_eligible_candidates"
    assert rescues.is_applicable("max_rule_length")
    assert rescues.is_rescued("max_rule_length")
    assert rescues.evaluated_count("max_rule_length") == 1


def test_each_final_constraint_reports_leave_one_out_rescue():
    # Both rules reuse the same false positives, so neither meets precision
    # alone while their union does. This isolates max_rules and marginal recall.
    X = np.array(
        [[1, 0], [0, 1], [1, 1], [1, 1]],
        dtype=float,
    )
    y = np.array([1, 1, 0, 0], dtype=bool)
    candidates = [_rule("left", 0, ">", 0.5), _rule("right", 1, ">", 0.5)]
    max_rules = select_rule_set(
        candidates,
        X,
        y,
        RuleSetSelectionConfig(
            min_precision=0.5,
            min_lift=0.0,
            max_rules=1,
            min_marginal_recall=0.0,
        ),
    )
    assert not max_rules.feasible
    assert max_rules.diagnostics.constraint_rescues.is_rescued("max_rules")

    marginal = select_rule_set(
        candidates,
        X,
        y,
        RuleSetSelectionConfig(
            min_precision=0.5,
            min_lift=0.0,
            max_rules=2,
            min_marginal_recall=0.75,
        ),
    )
    assert not marginal.feasible
    assert marginal.diagnostics.constraint_rescues.is_rescued(
        "min_marginal_recall"
    )

    one_rule = [_rule("half", 0, ">", 0.5)]
    precision = select_rule_set(
        one_rule,
        np.array([[1], [0], [1], [0]], dtype=float),
        np.array([1, 1, 0, 0]),
        RuleSetSelectionConfig(
            min_precision=0.75,
            min_lift=0.0,
            min_marginal_recall=0.0,
        ),
    )
    assert precision.diagnostics.constraint_rescues.is_rescued("min_precision")

    lift = select_rule_set(
        one_rule,
        np.array([[1], [0], [1], [0]], dtype=float),
        np.array([1, 1, 0, 0]),
        RuleSetSelectionConfig(
            min_precision=0.0,
            min_lift=1.5,
            min_marginal_recall=0.0,
        ),
    )
    assert lift.diagnostics.constraint_rescues.is_rescued("min_lift")


def test_rescue_diagnostics_do_not_change_a_feasible_baseline_selection():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=bool)
    candidates = [_rule("narrow", 0, ">", 6.5), _rule("broad", 0, ">", 3.5)]
    result = select_rule_set(
        candidates,
        X,
        y,
        RuleSetSelectionConfig(
            min_precision=0.5,
            min_lift=1.0,
            max_rules=1,
            min_marginal_recall=0.0,
        ),
    )
    assert result.rule_set.rules[0].rule_id == "broad"
    assert result.diagnostics.constraint_rescues.applicable == ()


def test_leave_one_out_rescue_flags_can_overlap_for_one_failed_selection():
    X = np.array([[1, 1], [1, 0], [0, 0]], dtype=float)
    y = np.array([1, 0, 0], dtype=bool)
    candidates = [
        _rule("short-imprecise", 0, ">", 0.5),
        _conjunctive_rule(
            "long-precise", [(0, ">", 0.5), (1, ">", 0.5)]
        ),
    ]
    result = select_rule_set(
        candidates,
        X,
        y,
        RuleSetSelectionConfig(
            min_precision=0.75,
            min_lift=0.0,
            max_rule_length=1,
            min_marginal_recall=0.0,
        ),
    )
    rescues = result.diagnostics.constraint_rescues
    assert result.reason == "no_subset_satisfies_constraints"
    assert set(rescues.rescued) == {"max_rule_length", "min_precision"}
