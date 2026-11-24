"""Focused checks of the concept agreement study helpers (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge

from concept_study import associations, bootstrap, cohort, decomposition, features, forecasts, measures, tables


def test_composition_split_adds_up_and_returning_part_is_paired_change():
    # Patients a, b in 2010 and 2011; c only in 2010; d only in 2011. One value per row.
    patients = np.array(["a", "b", "c", "a", "b", "d"])
    years = np.array([2010, 2010, 2010, 2011, 2011, 2011])
    value = np.array([1.0, 2.0, 9.0, 2.0, 4.0, 0.0])
    year_axis = (2010, 2011)
    present = np.stack([np.isin(patients, patients[years == y]) for y in year_axis], 1)
    means = np.zeros((2, 3))
    for k, year in enumerate(year_axis):
        for s in range(3):
            mask = (years == year) & (np.ones(6, bool) if s == 0 else present[:, s - 1])
            means[k, s] = value[mask].mean()
    split = decomposition.split_change(means, year_axis, [(2010, 2011)])
    total, returning = split[0]
    assert total == pytest.approx(2.0 - 4.0)
    assert returning == pytest.approx(((2 - 1) + (4 - 2)) / 2)
    composition = total - returning
    assert total == pytest.approx(returning + composition)


def test_bootstrap_weight_is_shared_by_every_row_of_a_patient():
    unique, weights = bootstrap.patient_weights(np.array(["p1", "p2", "p3"]), replicates=20, seed=1)
    rows_concept_1 = bootstrap.row_weights(np.array(["p2", "p1", "p2"]), unique, weights)
    rows_concept_2 = bootstrap.row_weights(np.array(["p2"]), unique, weights)
    np.testing.assert_array_equal(rows_concept_1[:, 0], rows_concept_1[:, 2])
    np.testing.assert_array_equal(rows_concept_1[:, 0], rows_concept_2[:, 0])
    assert np.all(weights[0] == 1) and np.all(weights[1:].sum(1) == 3)


def test_weighted_sums_match_direct_computation():
    rng = np.random.default_rng(0)
    w = rng.integers(0, 3, (5, 7)).astype(np.float32)
    masks = rng.random((7, 4)) > 0.5
    values = rng.normal(size=(7, 3)).astype(np.float32)
    expected = np.einsum("br,rc,rm->bcm", w, masks.astype(np.float32), values)
    np.testing.assert_allclose(bootstrap.weighted_sums(w, masks, values).cpu().numpy(), expected, rtol=1e-5, atol=1e-5)


def test_smd_hand_example_and_undefined_for_constant_feature():
    mean_1, var_1 = torch.tensor([3.0, 1.0]), torch.tensor([1.0, 0.0])
    mean_2, var_2 = torch.tensor([1.0, 1.0]), torch.tensor([3.0, 0.0])
    value = features.smd(mean_1, var_1, mean_2, var_2)
    assert value[0].item() == pytest.approx(2 / np.sqrt(2))
    assert np.isnan(value[1].item())


def test_spearman_and_partial_rank_match_hand_computation():
    rng = np.random.default_rng(3)
    x, y, z = rng.normal(size=40), rng.normal(size=40), rng.normal(size=40)
    y = y + x + z
    assert associations.spearman(x[None], y[None])[0] == pytest.approx(spearmanr(x, y).statistic)
    design = np.column_stack([np.ones(40), rankdata(z)])
    rx = rankdata(x) - design @ np.linalg.lstsq(design, rankdata(x), rcond=None)[0]
    ry = rankdata(y) - design @ np.linalg.lstsq(design, rankdata(y), rcond=None)[0]
    expected = np.corrcoef(rx, ry)[0, 1]
    assert associations.correlation(x[None], y[None], numeric=[z[None]])[0] == pytest.approx(expected)
    assert associations.correlation(x[None], y[None])[0] == pytest.approx(spearmanr(x, y).statistic)


def test_ridge_matches_sklearn():
    rng = np.random.default_rng(4)
    train = pd.DataFrame(rng.normal(size=(30, 2)), columns=["a", "b"])
    train["t"] = train.a * 2 + rng.normal(size=30)
    test = pd.DataFrame(rng.normal(size=(5, 2)), columns=["a", "b"])
    prediction = forecasts.fit_predict(train, test, ["a", "b"], "t", 10.0)
    mean, scale = train[["a", "b"]].mean().to_numpy(), train[["a", "b"]].std(ddof=0).to_numpy()
    expected = Ridge(alpha=10.0).fit((train[["a", "b"]] - mean) / scale, train.t).predict((test - mean) / scale)
    np.testing.assert_allclose(prediction, expected)


def test_horizon_join_pairs_exact_years_and_stops_at_2014():
    point = pd.DataFrame({"concept": "c", "group": "A", "year": [2011, 2012, 2013, 2014, 2015]})
    cube = {"brier": np.arange(10, dtype=float).reshape(2, 5)}
    joined, joined_cube = associations.horizon_panel(point, cube, 2, ["brier"])
    assert joined.year.tolist() == [2011, 2012]
    assert joined.future_year.tolist() == [2013, 2014]
    np.testing.assert_array_equal(joined_cube["future_brier_change"], np.full((2, 2), 2.0))


def test_group_measures_from_sums():
    # Two labelled rows: p = (0.2, 0.9), y = (0, 1); threshold 0.5 predicts the death correctly.
    sums = np.zeros(len(measures.BASE_COLUMNS))
    column = {name: i for i, name in enumerate(measures.BASE_COLUMNS)}
    sums[column["count"]] = sums[column["labelled"]] = 2
    sums[column["deaths"]] = 1
    sums[column["squared_error"]] = 0.2 ** 2 + 0.1 ** 2
    sums[column["predicted_death"]] = sums[column["predicted_and_died"]] = 1
    result = measures.group_measures(sums)
    assert result["brier"] == pytest.approx(0.025)
    assert result["death_f1"] == pytest.approx(1.0)


def test_logit_sensitivity_is_probability_gradient_over_p_one_minus_p():
    # Gradient of p along a direction equals p(1-p) times the gradient of the log-odds.
    p = np.array([0.02, 0.1, 0.5, 0.9])
    logit_gradient = np.array([[1.0, -2.0], [0.5, 0.0], [3.0, 1.0], [-1.0, 4.0]])
    probability_gradient = (logit_gradient * (p * (1 - p))[:, None]).astype(np.float32)
    concept = {"sensitivity_all_negatives": np.concatenate([probability_gradient, np.full((4, 1), np.nan)], 1)}
    values, fits = measures.sensitivity_values(concept, "all_negatives_logit", p)
    assert fits == 2                                            # the invalid (NaN) fit is dropped
    assert np.allclose(values[:, :2], logit_gradient, rtol=1e-5)
    assert np.allclose(values[:, 2:4], np.abs(logit_gradient), rtol=1e-5)
    assert np.array_equal(values[:, 4:], (logit_gradient > 0).astype(np.float32))
    plain, _ = measures.sensitivity_values(concept, "all_negatives")
    assert np.allclose(plain[:, :2], probability_gradient)


def test_removal_brier_change_splits_into_outcome_free_and_death_parts():
    p = np.array([0.1, 0.4, 0.05, 0.3, 0.2])
    q = np.array([0.15, 0.2, 0.05, 0.5, 0.1])                   # probability after removing the factor
    label = np.array([1.0, 0.0, np.nan, 1.0, 0.0])               # one unlabelled row
    y = np.nan_to_num(label)
    build = {"label": label, "probability": p, "reference_mortality": 0.1, "reconstruction_error_1": np.zeros(5)}
    concept = {"activation": np.zeros(5), "removal_probability_change": q - p,
               "removal_brier_change": (q - y) ** 2 - (p - y) ** 2}
    rows = measures.base_values(concept, build, seed=1, threshold=0.5)
    group = measures.group_measures(rows.sum(0))
    labelled = np.isfinite(label)
    expected = np.mean(((q - y) ** 2 - (p - y) ** 2)[labelled])
    assert np.isclose(group["removal_brier_change"], expected)
    assert np.isclose(group["removal_brier_outcome_free"], np.mean((q ** 2 - p ** 2)[labelled]))
    assert np.isclose(group["removal_brier_death_part"], np.mean((-2 * y * (q - p))[labelled]))
    assert np.isclose(group["removal_brier_outcome_free"] + group["removal_brier_death_part"], expected)


def test_presence_and_cell_masks_are_ordered_groups_years_subsets():
    build = {"patient_id": np.array(["a", "b", "a", "c"]), "year": np.array([2007, 2007, 2008, 2009])}
    concept = {"in_A": np.array([True, False, True, False]), "in_R": np.array([True, True, False, False])}
    years, groups, subsets = measures.YEARS, measures.GROUPS, measures.SUBSETS
    present = measures.presence(build)
    assert present[0, years.index(2008)] and present[2, years.index(2007)]       # patient a: 2007 and 2008
    assert not present[1, years.index(2008)] and not present[3, years.index(2007)]
    cells = measures.cell_masks(concept, build).reshape(4, len(groups), len(years), len(subsets))
    # Row 1 (patient b, 2007) is a rule patient without strong activation.
    assert cells[1, groups.index("R+A-"), years.index(2007), 0]
    assert not cells[1, groups.index("A"), years.index(2007), 0]
    assert not cells[1, groups.index("R+A-"), years.index(2007), subsets.index("present_in_2008")]
    assert cells[0, groups.index("R+A+"), years.index(2007), subsets.index("present_in_2008")]
    membership = {"A": concept["in_A"], "R": concept["in_R"], "R+A+": concept["in_R"] & concept["in_A"],
                  "R+A-": concept["in_R"] & ~concept["in_A"], "R-A+": ~concept["in_R"] & concept["in_A"],
                  "neither": ~concept["in_R"] & ~concept["in_A"], "all": np.ones(4, bool)}
    for r in range(4):
        for g, group in enumerate(groups):
            for y, year in enumerate(years):
                for k in range(len(subsets)):
                    expected = membership[group][r] and build["year"][r] == year and (k == 0 or present[r, k - 1])
                    assert cells[r, g, y, k] == expected


def test_cohort_metrics_and_weighted_auroc_match_a_direct_loop():
    rng = np.random.default_rng(7)
    n = 25
    p = rng.random(n).round(1)                                   # rounding creates ties
    y = (rng.random(n) < 0.35).astype(float)
    w = rng.integers(0, 3, (6, n)).astype(float)
    w[0] = 1
    result = cohort.metrics(p, y, threshold=0.5, constant=0.3, weights=w)
    for b in range(len(w)):
        brier = np.sum(w[b] * (p - y) ** 2) / w[b].sum()
        predicted = p >= 0.5
        f1 = 2 * np.sum(w[b] * predicted * y) / (np.sum(w[b] * predicted) + np.sum(w[b] * y))
        wins = total = 0.0
        for i in range(n):
            for j in range(n):
                if y[i] == 1 and y[j] == 0:
                    pair = w[b, i] * w[b, j]
                    wins += pair * (1.0 if p[i] > p[j] else 0.5 if p[i] == p[j] else 0.0)
                    total += pair
        assert result["brier"][b] == pytest.approx(brier)
        assert result["brier_skill"][b] == pytest.approx(1 - brier / (np.sum(w[b] * (0.3 - y) ** 2) / w[b].sum()))
        assert result["death_f1"][b] == pytest.approx(f1)
        assert result["auroc"][b] == pytest.approx(wins / total)
        assert result["mortality"][b] == pytest.approx(np.sum(w[b] * y) / w[b].sum())


def test_adjustments_of_one_comparison_share_one_row_set():
    rng = np.random.default_rng(5)
    n = 40
    point = pd.DataFrame({"concept": np.repeat(["c1", "c2", "c3", "c4"], 10),
                          "predictor_year": np.tile(["p1", "p2", "p3", "p4", "p5"], 8)})
    cube = {name: rng.normal(size=(3, n)) for name in ("agreement", "brier", "mortality", "log_n", "mean_risk")}
    cube["mortality"][:, :6] = np.nan                             # e.g. 2015 rows: no outcomes
    cube["agreement"][:, 6] = np.nan
    cells = [{"feature": "agreement", "target": "brier", "controls": a, "mask": np.ones(n, bool)}
             for a in ("none", "within_concept", "full")]
    table = associations.grid(point, cube, cells, jobs=1)
    assert table.rows.tolist() == [33, 33, 33]
    keep = np.arange(n) >= 7
    assert table.r[0] == pytest.approx(spearmanr(cube["agreement"][0, keep], cube["brier"][0, keep]).statistic)
    # A comparison without the outcome-based controls keeps the rows where only mortality is missing.
    alone = associations.grid(point, cube, cells[:2], jobs=1)
    assert alone.rows.tolist() == [39, 39]


def test_change_correlations_control_for_the_change_in_mean_risk():
    assert associations.controls_for("full_and_risk", "agreement", "brier")[-1] == "mean_risk"
    assert associations.controls_for("full_and_risk", "agreement_change", "brier_change")[-1] == "mean_risk_change"
    assert associations.controls_for("within_concept_and_risk", "agreement_change", "tcav_change") == \
        ["concept", "mean_risk_change"]
    # Removal measures are levels even though their names end in '_change'.
    assert associations.controls_for("full_and_risk", "removal_brier_change", "brier")[-1] == "mean_risk"
    assert associations.controls_for("full_and_risk", "agreement_change", "removal_brier_change_change")[-1] == \
        "mean_risk_change"


def test_concept_resampling_relabels_each_draw_and_correlation_uses_the_labels():
    concepts = np.array(["a", "a", "b", "c", "c", "c", "d", "d"])
    draws = bootstrap.concept_resamples(concepts, replicates=40, seed=2)
    np.testing.assert_array_equal(draws[0][0], np.arange(8))
    duplicated = 0
    for rows, labels in draws[1:]:
        assert sorted(np.unique(labels)) == [0, 1, 2, 3]           # four draws, one label each
        for k in range(4):
            drawn = concepts[rows[labels == k]]
            assert len(set(drawn)) == 1 and len(drawn) == (concepts == drawn[0]).sum()   # all rows of one concept
        duplicated += len(set(concepts[rows])) < 4
    assert duplicated > 0
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(41, 8)), rng.normal(size=(41, 8))
    values = associations.correlation(x, y, concepts, concept_indicators=True, seed=2)
    for b, (rows, labels) in enumerate(draws):
        residuals = []
        for v in (x[b, rows], y[b, rows]):
            ranks = rankdata(v)
            residuals.append(ranks - pd.Series(ranks).groupby(labels).transform("mean").to_numpy())
        spread = min(np.sum(r ** 2) for r in residuals)
        if spread < 1e-9 or len(rows) - len(set(labels)) < 3:
            continue
        assert values[b] == pytest.approx(np.corrcoef(residuals[0], residuals[1])[0, 1])
    summary = associations.summarize(values)
    assert summary["replicate_median"] == pytest.approx(np.nanmedian(values[1:]))
    assert np.isnan(associations.summarize(values[:1])["undefined_share"])


def test_forecast_models_use_identical_rows_and_compare_within_variant_and_group():
    rng = np.random.default_rng(8)
    years = np.repeat(np.arange(2004, 2014), 8)
    panel = pd.DataFrame({"year": years, "future_year": years + 1, "concept": np.tile(list("abcdefgh"), 10),
                          "t": rng.normal(size=80), "h": rng.normal(size=80), "b": rng.normal(size=80)})
    panel.loc[::5, "b"] = np.nan                                  # block feature missing on some rows
    scores = forecasts.chronological_forecasts(panel, "t", ["h"], {"+ b": ["b"]}, horizon=1, alphas=(10.0,))
    assert (scores.status == "fitted").any()
    for _, part in scores.groupby("target_year"):
        assert part.train_rows.nunique() == 1 and part.test_rows.nunique() == 1
    two = pd.concat([scores.assign(variant="v1", group="R"), scores.assign(variant="v2", group="R", mse=scores.mse * 2)])
    compared = forecasts.compare_to_history(two)
    assert len(compared) == 2 * (scores.status == "fitted").sum()
    history = {v: compared[compared.variant == v].set_index(["model", "target_year"]).mse_history for v in ("v1", "v2")}
    np.testing.assert_allclose(history["v2"], 2 * history["v1"].loc[history["v2"].index])


def test_composition_split_refuses_outcome_measures():
    for name in ("brier", "death_f1", "mortality", "brier_skill", "removal_brier_change"):
        assert name not in measures.SPLIT_GROUP_MEASURES
        with pytest.raises(ValueError):
            tables.split_table([], {name: f"group|{name}"})
        with pytest.raises(ValueError):
            tables.split_replicates([], f"group|{name}", "R")
    assert {"mean_risk", "mean_activation", "n"} <= set(measures.SPLIT_GROUP_MEASURES)
