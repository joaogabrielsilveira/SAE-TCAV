"""Known-answer checks for the scientific calculations."""

import numpy as np
import pytest

import pandas as pd

from error_profiles import (population_rows, select_rows, join_timing,
                            fit_directions, baseline_values, prediction_groups,
                            patient_bootstrap, row_weights, mix_within, record_length_groups,
                            shift_scores, score_risk, expected_errors, year_profile,
                            profile_contrasts, crossfit_ridge, annual_benchmarks,
                            performance_table, rate_splits, rule_error_rates)
from error_profiles import (weighted_spearman, training_target, fit_tree, rule_key,
                            context_presence, effective_dimension, log_odds_sensitivity)


def test_population_join_uses_patient_year_identity_and_checks_values():
    population = {"patient_id": np.array(["a", "b", "a"]),
                  "year": np.array([2010, 2010, 2011]),
                  "X": np.array([[1., 0.], [0., 2.], [3., 0.]]),
                  "y": np.array([0., 1., 0.]),
                  "feature_names": np.array(["x", "z"])}
    cached = {"patient_id": np.array(["a", "b"]), "year": np.array([2011, 2010]),
              "features": population["X"][[2, 1]], "label": np.array([0., 1.]),
              "feature_names": population["feature_names"]}
    np.testing.assert_array_equal(population_rows(cached, population), [2, 1])
    changed = {**cached, "features": cached["features"] + 1}
    with pytest.raises(ValueError, match="features"):
        population_rows(changed, population)


def test_roles_keep_evaluation_and_fitting_entrants_disjoint_through_2014():
    population = {"patient_id": np.array(["a", "a", "a", "b", "c", "d"]),
                  "year": np.array([2010, 2011, 2015, 2011, 2011, 2010])}
    roles = {"row_role": np.array(["evaluation", "evaluation", "evaluation", "fitting",
                                   "evaluation", "cav_fit"]),
             "is_entrant": np.array([False, False, False, True, True, False])}
    selected = select_rows(population, roles, 2010)
    np.testing.assert_array_equal(selected["evaluation"], [0, 1, 4])
    np.testing.assert_array_equal(selected["entrants"], [3])
    np.testing.assert_array_equal(selected["direction_fit"], [5])
    assert not set(population["patient_id"][selected["evaluation"]]) & set(
        population["patient_id"][selected["entrants"]])


def test_missing_timing_is_retained_as_missing_and_counted():
    rows = pd.DataFrame({"patient_id": ["a", "b"], "year": [2010, 2010]})
    timing = pd.DataFrame({"patient_id": ["a"], "year": [2010], "active_months": [3.]}).set_index(
        ["patient_id", "year"])
    joined = join_timing(rows, timing)
    assert joined.loc[0, "active_months"] == 3
    assert pd.isna(joined.loc[1, "active_months"])
    assert joined["timing_observed"].tolist() == [True, False]


def test_direction_recipe_balances_classes_normalizes_and_keeps_failures():
    labels = np.repeat([False, True], 8)
    embeddings = np.column_stack([np.where(labels, 2., -2.), np.arange(16) / 100])
    patients = np.array([str(i) for i in range(16)])
    result = fit_directions(embeddings, labels, patients, [1001, 1002])
    np.testing.assert_allclose(np.linalg.norm(result["directions"], axis=1), 1.)
    assert (result["directions"][:, 0] > .9).all()
    assert result["matching"] == result["nonmatching"] == 8
    np.testing.assert_array_equal(result["directions"],
                                  fit_directions(embeddings, labels, patients, [1001, 1002])["directions"])
    unavailable = fit_directions(embeddings, np.arange(16) == 0, patients, [1001])
    assert np.isnan(unavailable["directions"]).all()
    assert unavailable["fit_status"] == ["not fittable at the reference year"]


def test_baseline_uses_frozen_inclusive_threshold_and_exact_error_counts():
    result = baseline_values(np.array([1, 1, 0, 0]), np.array([.2, .5, .5, .1]), .5)
    assert [result[k] for k in ("n", "deaths", "true_positives", "false_negatives",
                               "false_positives")] == [4, 2, 1, 1, 1]
    assert result["recall"] == .5
    assert result["brier"] == pytest.approx((.8 ** 2 + .5 ** 2 + .5 ** 2 + .1 ** 2) / 4)


def test_choosing_entrants_exclude_context_exposed_gap_returners():
    population = {"patient_id": np.array(["gap", "gap", "new", "evaluation", "evaluation"]),
                  "year": np.array([2008, 2011, 2011, 2010, 2011])}
    roles = {"row_role": np.array(["", "fitting", "fitting", "evaluation", "evaluation"]),
             "is_entrant": np.array([False, True, True, False, False])}
    selected = select_rows(population, roles, 2010)
    np.testing.assert_array_equal(selected["entrants"], [2])
    np.testing.assert_array_equal(selected["evaluation"], [3, 4])


def test_death_only_timing_row_is_zero_care_but_absent_row_is_unknown():
    rows = pd.DataFrame({"patient_id": ["death_only", "absent"], "year": [2010, 2010]})
    timing = pd.DataFrame({"patient_id": ["death_only"], "year": [2010],
                           "events_all": [np.nan], "active_months": [np.nan],
                           "death_month": [4.]}).set_index(["patient_id", "year"])
    joined = join_timing(rows, timing)
    assert joined.loc[0, "active_months"] == 0
    assert joined.loc[0, "timing_observed"]
    assert joined.loc[0, "death_only_timing"]
    assert pd.isna(joined.loc[1, "active_months"])
    assert not joined.loc[1, "timing_observed"]


def test_four_groups_keep_threshold_ties_flagged():
    groups = prediction_groups(np.array([1, 1, 0, 0]), np.array([.1, .5, .5, .1]), .5)
    assert groups.tolist() == ["missed_deaths", "caught_deaths", "false_alarms", "correct_survivors"]


def test_bootstrap_shares_patient_multiplicity_across_years_and_predictors():
    bootstrap = patient_bootstrap([np.array(["a", "b"]), np.array(["b", "c"])], 9, 17)
    weights = row_weights(np.array(["a", "b", "a", "c", "b"]), bootstrap)
    np.testing.assert_array_equal(weights[:, 0], weights[:, 2])
    np.testing.assert_array_equal(weights[:, 1], weights[:, 4])
    np.testing.assert_array_equal(bootstrap[1].sum(axis=1), np.repeat(3, 10))
    assert np.any(bootstrap[1][1:] != 1)


def test_mix_within_exact_accounting_keeps_empty_group_as_mix():
    result = mix_within(np.array([10., 0.]), np.array([2., 0.]),
                        np.array([5., 5.]), np.array([2., 5.]))
    assert result["change"] == pytest.approx(.5)
    assert result["mix"] == pytest.approx(.35)
    assert result["within"] == pytest.approx(.15)
    assert result["mix"] + result["within"] == pytest.approx(result["change"])
    assert result["empty_group_patients"] == 5
    groups = record_length_groups(np.arange(13), np.ones(13))
    np.testing.assert_array_equal(groups, [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    np.testing.assert_array_equal(record_length_groups(np.arange(13), np.zeros(13)), np.arange(13))


def test_uniform_shift_matches_reference_gap_and_reports_impossible_target():
    y = np.array([0, 1, 0, 1]); p = np.array([.2, .6, .1, .3]); years = np.array([2010, 2010, 2011, 2011])
    shifted, diagnostics = shift_scores(y, p, years, 2010, np.ones(4))
    assert shifted[2:].mean() - y[2:].mean() == pytest.approx(p[:2].mean() - y[:2].mean())
    assert diagnostics[2011]["status"] == "fitted"
    impossible, diagnostics = shift_scores(np.array([0, 0, 1, 1]), np.array([.9, .9, .2, .2]),
                                           years, 2010, np.ones(4))
    assert np.isnan(impossible[2:]).all()
    assert diagnostics[2011]["status"] == "impossible calibration target"


def test_score_benchmark_expected_denominators_and_unfittable_years():
    risk = np.array([.1, .4, .8]); flagged = np.array([False, False, True])
    result = expected_errors(risk, flagged, np.ones(3))
    assert result["missed_death_rate"] == pytest.approx(.5 / 1.3)
    assert result["false_alarm_rate"] == pytest.approx(.2 / 1.7)
    fitted = score_risk(np.array([0, 1, 0, 1]), np.array([0., .3, .6, 1.]))
    assert fitted["status"] == "fitted" and fitted["endpoint_count"] == 2
    assert np.isfinite(fitted["risk"]).all()
    unavailable = score_risk(np.zeros(4), np.array([.1, .2, .4, .7]))
    assert unavailable["status"] == "single outcome class"
    assert np.isnan(unavailable["risk"]).all()


def test_profiles_and_consecutive_changes_keep_undefined_patient_draws():
    frame = pd.DataFrame({"patient_id": ["a", "b", "a", "b"], "year": [2010, 2010, 2011, 2011],
                           "label": [1, 1, 1, 1], "probability": [.1, .8, .1, .8]})
    weights = np.array([[1, 1, 1, 1], [2, 0, 2, 0], [0, 2, 0, 2]])
    profile, draws = year_profile(frame, {"activation": np.array([2., 4., 8., 12.])}, weights, .5, 2010)
    changes = profile_contrasts(profile, draws, 2010)
    row = changes[(changes["view"] == "same_outcome_deaths") & (changes["kind"] == "change") &
                  (changes["start"] == "2010") & (changes["end"] == "later")].iloc[0]
    assert row["value"] == -2
    assert row["undefined_draws"] == 2
    absent = profile[(profile["period"] == "2010") & (profile["group"] == "false_alarms")].iloc[0]
    assert absent["rows"] == 0 and np.isnan(absent["value"])


def test_ridge_excludes_each_patient_from_own_reference_and_later_prediction():
    patients = np.array([str(i) for i in range(12)] * 2)
    years = np.repeat([2010, 2011], 12)
    features = np.tile(np.arange(12.)[:, None], (2, 1))
    activation = features[:, 0] * 2
    before = crossfit_ridge(features, activation, patients, years, 2010, 3, 3, (.1, 1., 10.))
    poisoned = activation.copy(); poisoned[0] = 100000.
    after = crossfit_ridge(features, poisoned, patients, years, 2010, 3, 3, (.1, 1., 10.))
    np.testing.assert_array_equal(before["prediction"][[0, 12]], after["prediction"][[0, 12]])
    assert before["folds"][0] == before["folds"][12]
    assert np.isfinite(before["prediction"]).all()


def test_annual_analysis_exposes_rate_counts_failed_draws_and_inside_outside_errors():
    frame = pd.DataFrame({"patient_id": ["a", "b", "c", "d"] * 2,
                           "year": [2010] * 4 + [2011] * 4,
                           "label": [1, 1, 0, 0] * 2,
                           "probability": [.1, .8, .6, .2, .2, .9, .7, .3]})
    weights = np.array([[1] * 8, [2, 2, 0, 0] * 2, [0, 0, 2, 2] * 2])
    risks, shifted, diagnostics = annual_benchmarks(frame, weights, 2010)
    assert np.isnan(risks[1:]).all()
    assert set(diagnostics.loc[diagnostics["draw"] > 0, "status"]) == {"single outcome class"}
    table = performance_table(frame, weights, .5, 2010, shifted)
    row = table[(table["period"] == "2010") & (table["version"] == "frozen") &
                (table["metric"] == "recall")].iloc[0]
    assert "numerator" in row and "denominator" in row
    assert row["value"] == .5 and row["numerator"] == 1 and row["denominator"] == 2
    membership = np.array([True, False, True, False] * 2)
    rates = rule_error_rates(frame, membership, weights, .5, 2010, risks)
    inside = rates[(rates["period"] == "2010") & (rates["side"] == "deaths") &
                   (rates["group"] == "inside") & (rates["version"] == "observed")].iloc[0]
    assert inside["value"] == 1 and inside["numerator"] == inside["denominator"] == 1
    split = rate_splits(frame, membership, weights, .5, 2010, risks)
    expected = split[(split["version"] == "score_alone") & (split["component"] == "change")]
    assert (expected["undefined_draws"] == 2).all()


def test_visible_notebook_cells_compile():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "error_profiles_over_time.ipynb"
    notebook = json.loads(path.read_text())
    for number, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:cell-{number}", "exec")


def test_descriptive_notebook_cells_produce_all_eight_figures_on_complete_fixture(tmp_path):
    import json
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import error_profiles as ep

    rng = np.random.default_rng(12)
    cases = []
    for ref in range(2010, 2014):
        views = {}
        for name, years, count in (("evaluation", range(ref, 2015), 12), ("entrants", range(ref + 1, 2015), 8)):
            calendar = np.repeat(list(years), count)
            ids = np.tile([f"{name}-{i}" for i in range(count)], len(years))
            frame = pd.DataFrame({"patient_id": ids, "year": calendar,
                                   "label": np.tile([1, 1, 0, 0], len(calendar) // 4),
                                   "probability": np.tile([.2, .8, .6, .1], len(calendar) // 4),
                                   "active_months": np.tile([0, 3, 6, 12], len(calendar) // 4)})
            views[name] = {"frame": frame, "features": rng.poisson(2, (len(frame), 500)).astype(float)}
        activation = views["evaluation"]["features"][:, 0]
        concept = {"record": {"concept": f"last_3/{ref}/fixture"},
                   "arrays": {"activation": activation, "in_R": activation > 0, "in_A": activation > 2,
                              "sensitivity_all_negatives": rng.normal(size=(len(activation), 50))},
                   "direction_fit": {"fit_status": ["measured"] * 50, "matching": 8, "nonmatching": 4}}
        cases.append({"reference": ref, "threshold": .5, "views": views,
                      "feature_names": np.array([f"feature-{i}" for i in range(500)]),
                      "concepts": [concept], "timing_ready": True})
    namespace = {"ep": ep, "np": np, "pd": pd, "plt": plt, "display": lambda *args: None,
                 "cases": cases, "OUTPUT": tmp_path, "SEED": 17, "BOOTSTRAP_DRAWS": 3,
                 "RIDGE_FOLDS": 3, "RIDGE_ALPHAS": (.1, 1.), "study": {}}
    path = Path(__file__).resolve().parents[1] / "error_profiles_over_time.ipynb"
    started = False
    for cell in json.loads(path.read_text())["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and source.startswith("## 1."):
            started = True
        if cell["cell_type"] == "markdown" and source.startswith("## 6."):
            break
        if started and cell["cell_type"] == "code":
            exec(compile(source, path.name, "exec"), namespace)
            plt.close("all")
    expected = {1, 2, 3, 4, 5, 6, 10, 11}
    found = {int(p.name.split("_")[1]) for p in tmp_path.glob("figure_*.png") if p.stat().st_size > 0}
    assert found == expected
    assert len(pd.read_csv(tmp_path / "2010_all_500_feature_changes_entrant_order.csv").query(
        "view == 'same_outcome_deaths'")) == 500


def test_weighted_rank_correlation_matches_explicit_resampled_ties():
    from scipy.stats import spearmanr

    values = np.array([[0., 2.], [0., 1.], [2., 0.], [4., 0.]])
    target, weights = np.array([1., 3., 2., 2.]), np.array([3, 1, 2, 0])
    rows = np.repeat(np.arange(4), weights)
    expected = [spearmanr(values[rows, j], target[rows]).statistic for j in range(2)]
    np.testing.assert_allclose(weighted_spearman(values, target, weights).ravel(), expected)
    assert np.isnan(weighted_spearman(np.ones((4, 1)), target, weights)).all()


def test_slice_nuisance_uses_training_only_and_keeps_unseen_month_undefined():
    training = pd.DataFrame({"year": [2011] * 4, "active_months": [1, 1, 2, 2],
                             "label": [0, 1, 0, 1], "probability": [.1, .8, .2, .7]})
    other = pd.DataFrame({"year": [2011, 2011], "active_months": [1, 3],
                          "label": [0, 1], "probability": [.3, .6]})
    before, heldout, diagnostics = training_target(training, other)
    poisoned = other.assign(label=1 - other["label"])
    after, _, _ = training_target(training, poisoned)
    np.testing.assert_array_equal(before, after)
    assert np.isfinite(heldout[0]) and np.isnan(heldout[1])
    assert diagnostics["undefined_heldout_rows"].sum() == 1


def test_pruned_tree_never_forces_a_feature_description():
    X = np.arange(16.)[:, None]
    result = fit_tree(X, np.ones(16), np.arange(16).astype(str), ["feature"], 4, 42)
    assert result["status"] == "no feature description" and result["rules"] == []
    assert result["fit_count"] > 0


def test_rule_identity_and_context_absence_distinguish_definitions_and_counts():
    assert rule_key(((0, "<=", .5),)) != rule_key(((0, "<=", .6),))
    assert rule_key(((0, "<=", .5), (1, ">", 2.))) == rule_key(((1, ">", 2.), (0, "<=", .5)))
    labels = np.array([1, 0, 0])
    assert context_presence(np.array([False, False, False]), labels)["absence"] == "no matching context rows"
    assert context_presence(np.array([False, True, False]), labels)["absence"] == "matching context rows but no deaths"


def test_v6_sensitivity_is_patient_median_before_patient_profile():
    sensitivity = np.array([[0., 0., 100.], [0., 100., 0.], [100., 0., 0.]])
    np.testing.assert_array_equal(log_odds_sensitivity(sensitivity, np.repeat(.5, 3)), [0., 0., 0.])
    values = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    assert effective_dimension(values) == pytest.approx(2.)


def test_rank_matrix_matches_scalar_ranks_and_links_match_weighted_spearman():
    from error_profiles import tie_layout, weighted_rank_matrix, weighted_ranks, factor_links
    rng = np.random.default_rng(3)
    values = np.column_stack([rng.integers(0, 3, 40), np.zeros(40), rng.normal(size=40)]).astype(float)
    weights = rng.integers(0, 3, 40).astype(float)
    ranks = weighted_rank_matrix(tie_layout(values), weights)
    for j in range(3):
        keep = weights > 0
        np.testing.assert_allclose(ranks[keep, j], weighted_ranks(values[:, j], weights)[keep])
    frame = pd.DataFrame({"year": np.repeat([2010, 2011], 20)})
    target = rng.normal(size=40)
    links = factor_links(values, ["a", "b", "c"], {"plain": target}, weights[None, :], frame, 2010)
    later = links.query("period == 'later' and kind == 'link'").set_index("factor")["value"]
    expected = weighted_spearman(values[20:], target[20:], weights[20:]).ravel()
    np.testing.assert_allclose(later.to_numpy(), expected)
    assert np.isnan(later["b"])
    change = links.query("kind == 'change'").set_index("factor")["value"]
    first = links.query("period == '2010'").set_index("factor")["value"]
    np.testing.assert_allclose(change.dropna(), (later - first).dropna())


def test_adjusted_target_centres_each_year_month_cell_and_keeps_unknown_months():
    from error_profiles import choosing_targets
    frame = pd.DataFrame({"year": [2011, 2011, 2011, 2012, 2012], "active_months": [3., 3., np.nan, 3., 3.],
                          "label": [1, 0, 1, 0, 1], "probability": [.5, .2, .4, .3, .6]})
    risks = np.array([[.4, .3, .5, .2, .7], [.1, .1, .1, .1, .1]])
    weights = np.array([[1., 1., 1., 1., 1.], [2., 1., 0., 1., 0.]])
    targets = choosing_targets(frame, risks, weights)
    np.testing.assert_allclose(targets["plain"], frame["label"] - frame["probability"])
    adjusted = targets["adjusted"]
    assert np.isnan(adjusted[:, 2]).all()
    for d in range(2):
        for cell in ([0, 1], [3, 4]):
            w = weights[d, cell]
            if w.sum():
                assert np.sum(w * adjusted[d, cell]) == pytest.approx(0.)


def test_context_shares_separate_absence_cases_and_count_later_missed_deaths():
    from error_profiles import context_shares
    frame = pd.DataFrame({"year": [2010, 2011, 2011, 2011, 2012], "label": [1, 1, 1, 0, 0],
                          "probability": [.9, .1, .9, .9, .1]})
    member = np.array([True, True, False, True, False])
    row = context_shares(np.array([True, False]), np.array([0, 1]), member, frame, np.ones((1, 5)), .5, 2010)
    assert row["absence"] == "matching context rows but no deaths" and row["context_death_share"] == 0
    assert row["later_deaths_share"] == .5 and row["later_missed_deaths_share"] == 1.
    assert row["later_false_alarms_share"] == 1. and row["later_survivors_share"] == .5


def test_slice_report_splits_newcomers_from_returning_and_keeps_reference_year():
    from error_profiles import slice_report
    frame = pd.DataFrame({"patient_id": list("abcd"), "year": [2010, 2011, 2011, 2011], "first_year": [2010, 2011, 2010, 2010],
                          "label": [1, 1, 0, 0], "probability": [.2, .3, .6, .1], "active_months": [12., 2., 12., np.nan]})
    report = slice_report(frame, np.array([True, True, True, False]), np.ones((1, 4)), np.full((1, 4), .25), .5, 2010)
    share = report.query("quantity == 'share'").set_index(["period", "cohort"])
    assert share.loc[("2011", "newcomers"), "value"] == 1. and share.loc[("2011", "returning"), "value"] == .5
    assert share.loc[("2010", "all"), "value"] == 1.
    gap = report.query("quantity == 'observed_minus_predicted' and period == '2011' and cohort == 'all'").iloc[0]
    assert gap["value"] == pytest.approx(((1 - .3) + (0 - .6)) / 2) and gap["missed_deaths"] == 1 and gap["false_alarms"] == 1


def test_new_rule_sensitivity_is_undefined_at_endpoints_and_for_failed_fits():
    from error_profiles import projected_sensitivity, fit_spread
    directions = np.array([[1., 0.], [np.nan, np.nan]])
    values = projected_sensitivity(np.array([[1., 2.], [1., 2.]]), np.array([.5, 1.]), directions)
    assert values[0, 0] == pytest.approx(4.) and np.isnan(values[1]).all() and np.isnan(values[:, 1]).all()
    spread = fit_spread(values, np.array([True, True]), np.array([2010, 2010]), 2010)
    assert spread["fit_spread_fits_2010"] == 1 and spread["fit_spread_p05_2010"] == pytest.approx(4.)


def test_full_notebook_produces_all_twelve_figures_and_evidence_on_complete_fixture(tmp_path):
    import json
    import time
    import warnings
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import error_profiles as ep

    rng = np.random.default_rng(5)
    cases = []
    for ref in range(2010, 2014):
        views = {}
        for name, years, count in (("evaluation", range(ref, 2015), 16), ("entrants", range(ref + 1, 2015), 16),
                                   ("direction_fit", range(ref, ref + 1), 24)):
            calendar = np.repeat(list(years), count)
            ids = np.tile([f"{name}-{i}" for i in range(count)], len(years))
            n = len(calendar)
            features = rng.poisson(2, (n, 500)).astype(float)
            frame = pd.DataFrame({"patient_id": ids, "year": calendar, "label": (features[:, 5] >= 4).astype(int),
                                  "probability": np.clip(rng.uniform(.05, .9, n), 0, 1),
                                  "active_months": np.tile([0, 3, 6, 12], n // 4).astype(float),
                                  "first_year": np.where(np.arange(n) % 3 == 0, calendar, years[0])})
            codes = {seed: np.maximum(features[:, :3] * (1 + i) - 1 + rng.normal(0, .1, (n, 3)), 0)
                     for i, seed in enumerate((42, 135, 215))}
            views[name] = {"frame": frame, "features": features, "codes": codes,
                           "embeddings": features[:, :8] + rng.normal(0, .5, (n, 8)), "gradients": rng.normal(size=(n, 8))}
        evaluation = views["evaluation"]
        rule = ((0, ">", 1.5),)
        activation = evaluation["codes"][42][:, 0]
        concept = {"record": {"concept": f"last_3/{ref}/fixture", "seed": 42, "factor": 0}, "rule": rule,
                   "arrays": {"activation": activation, "in_R": ep.rule_mask(rule, evaluation["features"]), "in_A": activation > 2,
                              "sensitivity_all_negatives": rng.normal(size=(len(activation), 50))},
                   "direction_fit": {"fit_status": ["measured"] * 50, "matching": 8, "nonmatching": 4}}
        context = {"features": rng.poisson(2, (60, 500)).astype(float), "labels": np.tile([1, 0, 0], 20),
                   "years": np.repeat([ref - 2, ref - 1], 30), "patients": np.arange(60)}
        cases.append({"reference": ref, "threshold": .5, "views": views, "concepts": [concept], "context": context,
                      "feature_names": np.array([f"feature-{i}" for i in range(500)]), "timing_ready": True,
                      "families": [{"family": "fixture", "members": [(42, 0), (135, 0), (215, 0)]}]})
    namespace = {"ep": ep, "np": np, "pd": pd, "plt": plt, "warnings": warnings, "time": time, "Path": Path,
                 "display": lambda *args: None, "cases": cases, "OUTPUT": tmp_path, "SEED": 17, "BOOTSTRAP_DRAWS": 3,
                 "RIDGE_FOLDS": 3, "RIDGE_ALPHAS": (.1, 1.), "TREE_FOLDS": 3, "WORKERS": 1, "DIRECTION_FITS": 3,
                 "TOLERANCES": {}, "timings": {}, "started": time.perf_counter(), "manifest": {"status": "running"},
                 "study": {"composition": pd.DataFrame(), "source_hashes": {}, "input_hashes": {}, "load_seconds": 0.}}
    path = Path(__file__).resolve().parents[1] / "error_profiles_over_time.ipynb"
    started = False
    for cell in json.loads(path.read_text())["cells"]:
        source = "".join(cell["source"])
        started = started or (cell["cell_type"] == "markdown" and source.startswith("## 1."))
        if started and cell["cell_type"] == "code":
            exec(compile(source, path.name, "exec"), namespace)
            plt.close("all")
    found = {int(p.name.split("_")[1]) for p in tmp_path.glob("figure_*.png") if p.stat().st_size > 0}
    assert found == set(range(1, 13))
    links = pd.read_csv(tmp_path / "2010_factor_links.csv")
    assert set(links["target"]) == {"plain", "score_alone", "adjusted"} and links["factor"].nunique() == 9
    descriptions = pd.read_csv(tmp_path / "2010_factor_descriptions.csv")
    assert len(descriptions) == 9
    context = pd.read_csv(tmp_path / "2010_context_shares.csv")
    assert "existing" in set(context["kind"])
    evidence = pd.read_csv(tmp_path / "2010_evidence_table.csv.gz")
    assert any(len(pd.read_csv(tmp_path / f"{r}_slices.csv")) for r in range(2010, 2014))
    sensitivity_rows = evidence.query("measure == 'log_odds_sensitivity'").groupby(["kind", "group"]).size()
    assert sensitivity_rows.nunique() == 1 and len(sensitivity_rows) == evidence.groupby(["kind", "group"]).ngroups
    assert {"existing"} <= set(evidence["kind"]) and json.loads((tmp_path / "run_manifest.json").read_text())["direction_fits_per_new_rule"] == 3


def test_slice_search_survives_a_heldout_month_unseen_in_training():
    rng = np.random.default_rng(9)
    n = 120
    X = rng.normal(size=(n, 3))
    frame = pd.DataFrame({"year": np.repeat([2011, 2012], n // 2), "active_months": np.tile([3., 6., 12.], n // 3),
                          "label": (X[:, 0] > .8).astype(int), "probability": rng.uniform(.05, .3, n)})
    frame.loc[7, "active_months"] = 0.  # one patient alone in its (year, month) cell
    patients = np.arange(n).astype(str)
    result = fit_tree(X, None, patients, ["a", "b", "c"], 4, 1, frame=frame)
    assert not result["status"].startswith("not estimable")
    assert result["cv_diagnostics"]["undefined_heldout_rows"].sum() >= 1


def test_predictors_cover_every_last_three_year_reference():
    import error_profiles as ep
    assert sorted(ep.PREDICTORS) == list(range(2007, 2014))
    assert sum(count for _, count in ep.PREDICTORS.values()) == 62
    assert ep.last_three_years(2007) == {2007}
    assert ep.last_three_years(2008) == {2007, 2008}
    assert ep.last_three_years(2009) == {2007, 2008, 2009}
    assert ep.last_three_years(2013) == {2011, 2012, 2013}
