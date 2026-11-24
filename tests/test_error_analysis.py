"""Unit tests of the error-analysis extension (pure functions, synthetic data)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from concept_study import error_analysis as ea


def test_record_timing_counts_months_and_death_month():
    events = pd.DataFrame({
        "patient_id": [1, 1, 1, 1, 2, 3, 9],
        "date": pd.to_datetime(["2012-01-01", "2012-01-01", "2012-03-01", "2012-03-01", "2012-07-01", "2013-02-01",
                                "2012-05-01"]),
        "event": ["EVENT_A", "DIAGN_B", "EVENT_A", "DEATH", "EVENT_A", "DEATH", "EVENT_A"]})
    table = ea.record_timing(events, {"1", "2", "3"})
    first = table.loc[("1", 2012)]
    assert (first.events_all, first.active_months, first.first_month, first.last_month, first.death_month) == (3, 2, 1, 3, 3)
    assert np.isnan(table.loc[("2", 2012)].death_month)
    assert np.isnan(table.loc[("3", 2013)].active_months) and table.loc[("3", 2013)].death_month == 2   # death only
    assert "9" not in table.index.get_level_values(0)


def test_previous_year_rows():
    lag = ea.previous_year_rows(np.array(["a", "a", "b", "a"]), np.array([2010, 2011, 2011, 2013]))
    assert lag.tolist() == [-1, 0, -1, -1]


def test_patient_weights_follow_the_patient():
    weights = ea.patient_weights(np.array(["a", "b", "a", "c"]), replicates=50, seed=1)
    assert weights.shape == (51, 4) and (weights[0] == 1).all()
    assert np.array_equal(weights[:, 0], weights[:, 2])
    assert np.allclose(weights[:, [0, 1, 3]].sum(1), 3)


def test_auroc_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    label = rng.integers(0, 2, 200)
    score = rng.integers(0, 6, 200).astype(float)                # many ties
    weights = np.vstack([np.ones(200), rng.integers(0, 4, (5, 200))]).astype(np.float32)
    values = ea.auroc(label, score, weights)
    for row in range(len(weights)):
        assert values[row] == pytest.approx(roc_auc_score(label, score, sample_weight=weights[row]), abs=1e-6)


def test_auroc_inside_strata_ignores_between_stratum_pairs():
    label = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.9, 0.1, 0.1])                       # score only tells the stratum
    strata = np.array([0, 0, 1, 1])
    weights = np.ones((1, 4), dtype=np.float32)
    assert ea.auroc(label, score, weights)[0] == pytest.approx(0.5)
    assert ea.auroc(label, score, weights, strata)[0] == pytest.approx(0.5)
    score = np.array([0.9, 0.8, 0.1, 0.0])                       # also right inside each stratum
    assert ea.auroc(label, score, weights, strata)[0] == pytest.approx(1.0)
    assert np.isnan(ea.auroc(np.array([0, 0]), np.array([0.1, 0.2]), np.ones((1, 2), dtype=np.float32))[0])


def test_estimate_point_and_interval():
    out = ea.estimate("x", np.array([5.0, *np.arange(101.0), np.nan]))
    assert out["x"] == 5 and out["x_low"] == pytest.approx(2.5) and out["x_high"] == pytest.approx(97.5)


def rows_frame(label, **scores):
    n = len(label)
    return pd.DataFrame({"patient_id": [str(i) for i in range(n)], "label": label, "context_mortality": 0.25,
                         "set": ["evaluation", "discovery"] * (n // 2), **scores})


def test_scores_table_skill_and_differences():
    label = np.array([1, 0, 0, 0] * 10)
    rows = rows_frame(label, model=np.where(label == 1, 0.8, 0.1), constant=np.full(40, 0.25))
    weights = ea.patient_weights(rows.patient_id.to_numpy(), replicates=20)
    table = {r["score"]: r for r in ea.scores_table(rows, weights, ["model", "constant"])}
    assert table["model"]["auroc"] == pytest.approx(1.0) and table["model"]["deaths"] == 10
    assert table["constant"]["brier_skill"] == pytest.approx(0.0, abs=1e-6)
    assert table["constant"]["auroc_minus_model"] == pytest.approx(-0.5)
    assert table["model"]["brier"] == pytest.approx(0.25 * 0.04 + 0.75 * 0.01)


def test_scores_table_keeps_only_rows_with_every_score():
    label = np.array([1, 0, 1, 0, 1, 0])
    rows = rows_frame(label, model=np.linspace(0.9, 0.1, 6), other=[0.5, 0.4, np.nan, np.nan, 0.6, 0.1])
    table = ea.scores_table(rows, ea.patient_weights(rows.patient_id.to_numpy(), replicates=5), ["model", "other"])
    assert table[0]["rows"] == 4 and table[0]["deaths"] == 2


def test_timing_scores_separate_record_length_from_model():
    months = np.array([3, 3, 12, 12] * 20)
    label = np.array([1, 1, 0, 0] * 20)                           # death decided by record length alone
    rows = rows_frame(label, model=np.where(months == 3, 0.6, 0.05), boosting=np.where(months == 3, 0.6, 0.05))
    rows["active_months"], rows["last_month"], rows["feature_events"] = months, months, months * 2.0
    rows["death_month"] = np.where(label == 1, 3.0, np.nan)
    table = ea.step2_timing_scores(rows, ea.patient_weights(rows.patient_id.to_numpy(), replicates=10))
    both = table[table.set == "both"].set_index("score").auroc
    assert both["model"] == pytest.approx(1.0) and both["fewer months with care events"] == pytest.approx(1.0)
    assert np.isnan(both["model, pairs with the same number of months"])    # no death/survivor pair shares a length
    assert both["model, deaths of quarter 1 against all survivors"] == pytest.approx(1.0)
    assert np.isnan(both["model, deaths of quarter 4 against all survivors"])


def test_pooled_contrast_is_within_concept():
    # two concepts, one replicate row (point estimate only); residual = (deaths - expected) / n
    sums = {"n_on": np.array([[10.0], [10.0]]), "deaths_on": np.array([[4.0], [1.0]]),
            "expected_on": np.array([[2.0], [1.0]]), "n_off": np.array([[10.0], [30.0]]),
            "deaths_off": np.array([[1.0], [3.0]]), "expected_off": np.array([[1.0], [3.0]])}
    out = ea.pooled_contrast(sums, np.array([[0, 1]]))
    weights = np.array([5.0, 7.5])
    assert out["residual_difference_on_minus_off"] == pytest.approx((weights * np.array([0.2, 0.0])).sum() / weights.sum())
    assert out["observed_over_expected_on"] == pytest.approx(5 / 3)
    assert out["observed_over_expected_off"] == pytest.approx(1.0)


def test_concept_contrast_sums_inside_the_rule_only():
    concepts = [{"concept": "c", "positions": np.array([0, 1, 2, 3]), "in_R": np.array([True, True, True, False]),
                 "in_A": np.array([True, False, False, True])}]
    y, p = np.array([1.0, 0.0, 1.0, 1.0]), np.array([0.5, 0.2, 0.1, 0.9])
    sums = ea.concept_contrast(concepts, y, p, np.ones((1, 4), dtype=np.float32), np.ones(4, bool))
    assert (sums["n_on"][0, 0], sums["deaths_on"][0, 0], sums["expected_on"][0, 0]) == pytest.approx((1, 1, 0.5))
    assert (sums["n_off"][0, 0], sums["deaths_off"][0, 0], sums["expected_off"][0, 0]) == pytest.approx((2, 1, 0.3))


def test_study_rows_and_context_rows():
    population = {"year": np.array([2011, 2012, 2013, 2013, 2014, 2015, 2013]),
                  "post_death": np.array([False, False, False, False, False, False, False])}
    roles = {"reference_year": 2012,
             "row_role": np.array(["", "tabpfn_context", "evaluation", "fitting", "fitting", "evaluation", "fitting"]),
             "is_entrant": np.array([False, False, False, True, True, False, False])}
    assert ea.study_rows(population, roles).tolist() == [2, 3, 4]          # later years to 2014; fitting only if entrant
    predictor = {"context_window": "last_3", "context_years": [2011, 2012], "context_rows": 2}
    assert ea.context_rows(population, roles, predictor).tolist() == [0, 1]
    with pytest.raises(ValueError):
        ea.context_rows(population, roles, {**predictor, "context_rows": 3})


def test_auroc_of_no_rows_is_undefined():
    values = ea.auroc(np.zeros(0), np.zeros(0), np.ones((3, 0), dtype=np.float32))
    assert values.shape == (3,) and np.isnan(values).all()
