import numpy as np
import pandas as pd
import pytest

import error_profiles as ep
import error_explorer_export as ex


def synthetic(rows=240, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"patient_id": [f"p{i % 90}" for i in range(rows)],
                          "year": rng.choice([2010, 2011, 2012], rows),
                          "label": rng.integers(0, 2, rows),
                          "probability": rng.uniform(.01, .99, rows),
                          "active_months": rng.integers(0, 13, rows).astype(float)})
    bootstrap = ep.patient_bootstrap([frame["patient_id"].to_numpy()], 30, seed)
    return frame, ep.row_weights(frame["patient_id"], bootstrap), rng


def test_change_pairs_reference_first_then_consecutive_without_duplicates():
    assert ex.change_pairs(["2010", "2011", "2012", "later"], 2010) == [
        ("2010", "2011"), ("2010", "2012"), ("2010", "later"), ("2011", "2012")]


def test_rule_from_key_inverts_rule_key():
    rule = [(3, "<=", 4.069970534681328), (0, ">", .5723834906576124)]
    assert ep.rule_key(ex.rule_from_key(ep.rule_key(rule))) == ep.rule_key(rule)


def test_mean_over_predictors_keeps_undefined_draws_undefined():
    out = ex.mean_over_predictors([np.array([1., 2., np.nan]), np.array([3., 4., 5.])])
    np.testing.assert_array_equal(out[:2], [2., 3.])
    assert np.isnan(out[2])


def test_within_concept_features_match_profile_contrasts_on_rule_rows():
    frame, weights, rng = synthetic()
    member = rng.random(len(frame)) < .7
    quantities = {"a": rng.normal(size=len(frame)), "b": rng.integers(0, 3, len(frame)).astype(float)}
    mine, _ = ex.within_concept_features(frame, member, weights, .5, 2010, quantities)
    sub = frame[member].reset_index(drop=True)
    profile, draws = ep.year_profile(sub, {k: v[member] for k, v in quantities.items()}, weights[:, member], .5, 2010)
    theirs = ep.profile_contrasts(profile, draws, 2010)
    names = {"deaths": "same_outcome_deaths", "survivors": "same_outcome_survivors", "all": "all"}
    joined = mine.assign(view=mine["view"].map(names)).merge(
        theirs, on=["quantity", "view", "kind", "start", "end"], suffixes=("", "_nb"), validate="one_to_one")
    assert len(joined) == len(mine)
    np.testing.assert_allclose(joined["value"], joined["value_nb"], equal_nan=True)
    np.testing.assert_allclose(joined["low"], joined["low_nb"], equal_nan=True)


def test_factor_link_draws_reproduce_notebook_factor_links():
    frame, weights, rng = synthetic(seed=1)
    values = rng.normal(size=(len(frame), 4))
    risks, _, _ = ep.annual_benchmarks(frame, weights, 2010)
    targets = {k: v for k, v in ep.choosing_targets(frame, risks, weights).items() if k in ("plain", "adjusted")}
    draws = ex.factor_link_draws(values, targets, weights, frame, 2010)
    saved = ep.factor_links(values, list("wxyz"), targets, weights, frame, 2010).query("kind == 'link'")
    for (period, target), matrix in draws.items():
        expected = saved.query("period == @period and target == @target")["value"].to_numpy()
        np.testing.assert_allclose(ep.summarize_matrix(matrix)["value"], expected, equal_nan=True)


def test_performance_draws_reproduce_performance_table():
    frame, weights, _ = synthetic(seed=2)
    _, shifted, _ = ep.annual_benchmarks(frame, weights, 2010)
    rows = pd.DataFrame(ex.performance_rows(ex.performance_draws(frame, weights, .5, 2010, shifted), 2010, "true_year"))
    rows["period"] = np.where(rows["kind"] == "level", rows["end"], rows["start"] + "→" + rows["end"])
    saved = ep.performance_table(frame, weights, .5, 2010, shifted)
    joined = rows.merge(saved, on=["period", "version", "metric"], suffixes=("", "_nb"), validate="one_to_one")
    assert len(joined) == len(saved)
    np.testing.assert_allclose(joined["value"], joined["value_nb"], equal_nan=True)
    np.testing.assert_allclose(joined["high"], joined["high_nb"], equal_nan=True)


def test_cohort_errors_split_rates_by_first_year():
    frame = pd.DataFrame({"year": [2010, 2011, 2011, 2011], "first_year": [2010, 2011, 2010, 2010],
                          "label": [1, 0, 0, 1], "probability": [.1, .9, .2, .9], "patient_id": list("abcd")})
    weights = np.ones((2, 4))
    rows = pd.DataFrame(ex.cohort_errors(frame, weights, .5, 2010)).set_index(["period", "cohort", "side"])
    assert rows.loc[("2011", "newcomers", "survivors"), "value"] == 1.
    assert rows.loc[("2011", "returning", "survivors"), "value"] == 0.
    assert rows.loc[("2011", "returning", "deaths"), "value"] == 0.
