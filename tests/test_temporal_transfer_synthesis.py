import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from sklearn.svm import SVC

import concept_candidates as cc
import concept_regions as cr
import temporal_transfer_synthesis as ts

NAN = float("nan")
AMBIGUOUS = "candidate split/ambiguity"
UNMATCHED = "unmatched"


def synthetic_export():
    """Small Ideas-style export: one family-year with three yearly SAEs' worth of matching entries."""
    family = "2008/42/42:283,135:131"
    matching = [
        {"seed": 42, "dictionary_identity": "id-a", "matching_patients": 756, "possible_merges": [93],
         "status": "available",
         "members": [{"reference_factor": 283, "status": AMBIGUOUS,
                      "candidates": [{"factor": 283, "pearson": 0.97},
                                     {"factor": 31, "pearson": 0.93},
                                     {"factor": 124, "pearson": 0.90}]}]},
        {"seed": 43, "dictionary_identity": "id-b", "matching_patients": 700, "possible_merges": [],
         "status": "available",
         "members": [{"reference_factor": 135, "status": AMBIGUOUS,
                      "candidates": [{"factor": 7, "pearson": 0.80},
                                     {"factor": 8, "pearson": 0.75},
                                     {"factor": 9, "pearson": NAN}]}]},
        {"seed": 44, "dictionary_identity": "id-c", "matching_patients": 690, "possible_merges": [],
         "status": "available",
         "members": [{"reference_factor": 215, "status": UNMATCHED,
                      "candidates": [{"factor": 1, "pearson": NAN},
                                     {"factor": 2, "pearson": NAN},
                                     {"factor": 3, "pearson": NAN}]}]},
    ]
    return {
        "build": "all_history",
        "reference": 2008,
        "temporal": {
            "families": [{
                "family": family, "reference": 2008, "year": 2009,
                "rows": 314, "R_rows": 20, "A_rows": 14, "retention": 0.6,
                "idea_status": {"idea1": "eligible", "idea2": "eligible"},
                "yearly_description": {"status": "available"},
                "yearly_representations": matching,
            }],
            "members": [{"family": family, "reference": 2008, "year": 2009, "seed": 42, "factor": 283,
                         "retention": 0.6, "cutoff": 2.3, "cutoff_status": "available",
                         "inside_A": {"rows": 14}, "inside_R": {"rows": 20}, "positive_fraction": 0.1}],
        },
    }


def test_check_hashes_raises_on_mismatch():
    with pytest.raises(ValueError, match="trajectories.csv"):
        ts.check_hashes({"trajectories.csv": "aa", "run.json": "bb"},
                        {"trajectories.csv": "xx", "run.json": "bb"})


def test_check_hashes_passes_on_equal_and_missing_is_mismatch():
    ts.check_hashes({"run.json": "bb"}, {"run.json": "bb", "extra": "cc"})
    with pytest.raises(ValueError, match="run.json"):
        ts.check_hashes({"run.json": "bb"}, {})


def test_combined_hash_recipe(tmp_path):
    first = tmp_path / "b.npz"
    second = tmp_path / "a.npz"
    first.write_bytes(b"second file bytes")
    second.write_bytes(b"first file bytes")
    sorted_paths = sorted([first, second])
    joined = "".join(hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted_paths)
    expected = hashlib.sha256(joined.encode()).hexdigest()
    assert ts.combined_hash([first, second]) == expected
    assert ts.combined_hash([second, first]) == expected
    assert cr.sha256(first) == hashlib.sha256(first.read_bytes()).hexdigest()


def test_export_path_follows_decision_4():
    root = Path("/exports")
    assert ts.export_path(root, "r1", 2007) == Path("/exports/evaluate-baseline/evaluate-2007.json")
    assert ts.export_path(root, "all_history", 2008) == Path("/exports/evaluate-all_history/evaluate-2008.json")
    assert ts.export_path(root, "last_3", 2011) == Path("/exports/evaluate-last_3/evaluate-2011.json")


def test_family_year_rows_keep_counts_and_eligibility():
    rows = ts.family_year_rows(synthetic_export(), "all_history")
    assert len(rows) == 1
    row = rows[0]
    assert row["concept"] == "all_history/2008/2008/42/42:283,135:131"
    assert (row["year"], row["rows"], row["R_rows"], row["A_rows"]) == (2009, 314, 20, 14)
    assert row["idea1_status"] == "eligible" and row["idea2_status"] == "eligible"


def test_description_rows_keep_status():
    rows = ts.description_rows(synthetic_export(), "all_history")
    assert rows == [{"build": "all_history", "reference": 2008, "year": 2009,
                     "family": "2008/42/42:283,135:131",
                     "concept": "all_history/2008/2008/42/42:283,135:131", "status": "available"}]


def test_matching_rows_top_k_gap_merges_and_mechanical_status():
    rows = {(row["seed"], row["reference_factor"]): row
            for row in ts.matching_rows(synthetic_export(), "all_history")}
    three = rows[(42, 283)]
    assert three["candidate_factors"] == (283, 31, 124)
    assert three["pearson_1"] == 0.97 and three["pearson_3"] == 0.90
    assert three["gap_1_2"] == pytest.approx(0.04)
    assert three["merges"] == 1
    assert three["mechanical_status"] == AMBIGUOUS
    assert three["export_status"] == AMBIGUOUS

    two = rows[(43, 135)]
    assert two["mechanical_status"] == AMBIGUOUS
    assert math.isnan(two["pearson_3"])
    assert two["gap_1_2"] == pytest.approx(0.05)

    none = rows[(44, 215)]
    assert none["mechanical_status"] == UNMATCHED
    assert math.isnan(none["gap_1_2"]) and math.isnan(none["pearson_1"])


def test_matching_rows_single_finite_candidate_is_its_own_status():
    export = synthetic_export()
    export["temporal"]["families"][0]["yearly_representations"][0]["members"][0]["candidates"] = [
        {"factor": 283, "pearson": 0.9}, {"factor": 31, "pearson": NAN}, {"factor": 124, "pearson": NAN}]
    row = ts.matching_rows(export, "all_history")[0]
    assert row["mechanical_status"] == "single candidate"
    assert math.isnan(row["gap_1_2"])


def test_duplicate_gaps_zero_when_equal_and_largest_gap_otherwise():
    first = ts.family_year_rows(synthetic_export(), "all_history")
    assert ts.duplicate_gaps(first, first, ["concept"], ["retention", "rows"]) == 0.0

    second = [dict(row, retention=row["retention"] + 0.25, rows=row["rows"] + 2) for row in first]
    assert ts.duplicate_gaps(first, second, ["concept"], ["retention", "rows"]) == pytest.approx(2.0)


def test_duplicate_gaps_key_missing_from_one_copy_is_infinite():
    first = ts.family_year_rows(synthetic_export(), "all_history")
    assert ts.duplicate_gaps(first, [], ["concept"], ["retention"]) == math.inf


def test_absent_concept_has_status_row_not_zero():
    absent = ts.status_row({"concept": "all_history/2011/2011/9", "year": 2012, "part": "ideas"},
                           "no Ideas record (not eligible for either idea)", ["retention", "rows"])
    assert absent["status"].startswith("no Ideas record")
    assert math.isnan(absent["retention"]) and math.isnan(absent["rows"])

    measured = ts.status_row({"concept": "c", "year": 2012, "part": "ideas"}, "measured",
                             ["retention", "rows"], {"retention": 0.5, "rows": 12})
    assert measured["retention"] == 0.5 and measured["rows"] == 12


def test_candidate_coverage_counts_distinct_factors_within_one_yearly_sae_only():
    def entry(build, seed, factors):
        return {"build": build, "reference": 2008, "seed": seed, "year": 2009,
                "candidate_factors": tuple(factors)}

    matching = [
        entry("all_history", 42, (283, 31, 124)),
        entry("all_history", 42, (31, 9, 10)),
        entry("all_history", 43, (283, 31, 124)),
    ]
    coverage = {(row["seed"]): row for row in ts.candidate_coverage(matching, dictionary_size=288)}
    assert coverage[42]["distinct_candidate_factors"] == 5
    assert coverage[43]["distinct_candidate_factors"] == 3
    assert coverage[42]["dictionary_size"] == 288
    assert coverage[42]["share"] == pytest.approx(5 / 288)


POPULATION = np.array(["p1", "p2", "p3", "p4", "p5", "p6"])


def year_rows(patient_ids, in_R, in_A, activation, linear):
    return {"patient_ids": np.array(patient_ids), "in_R": np.array(in_R, bool),
            "in_A": np.array(in_A, bool), "activation": np.array(activation, float),
            "linear": np.array(linear, float)}


def brute_weighted_auroc(labels, scores, weights):
    """Hand oracle: weighted share of positive-negative pairs ranked correctly, ties counted half."""
    wins = total = 0.0
    for i in np.flatnonzero(labels):
        for j in np.flatnonzero(~labels):
            pair = weights[i] * weights[j]
            total += pair
            wins += pair * (1.0 if scores[i] > scores[j] else 0.5 if scores[i] == scores[j] else 0.0)
    return wins / total


def test_year_draws_use_population_draw_counts_for_each_row():
    rng = np.random.default_rng(0)
    count = cc.patient_weights(POPULATION, seeds=[11, 12, 13])
    patients = ["p3", "p1", "p6", "p2", "p5", "p4"]
    rows = year_rows(patients, [1, 1, 0, 0, 0, 0], [1, 0, 1, 1, 0, 0],
                     rng.normal(size=6), rng.normal(size=6))
    out = ts.year_auroc_draws(rows, count, POPULATION)
    positions = [int(p[1:]) - 1 for p in patients]
    for d in range(count.shape[0]):
        weights = count[d, positions]
        expected = brute_weighted_auroc(rows["in_R"], rows["activation"], weights)
        assert out["draws"]["auroc_activation"][d] == pytest.approx(expected, abs=1e-12)


def test_shared_patient_has_the_same_draw_count_in_every_year():
    rng = np.random.default_rng(1)
    count = cc.patient_weights(POPULATION, seeds=[21, 22, 23, 24])
    activation = rng.normal(size=6)
    linear = rng.normal(size=6)
    in_R = [1, 1, 0, 0, 0, 1]
    in_A = [1, 0, 1, 0, 0, 1]
    first = ts.year_auroc_draws(year_rows(["p1", "p2", "p3", "p4", "p5", "p6"], in_R, in_A, activation, linear),
                                count, POPULATION)
    reversed_order = list(reversed(range(6)))
    second = ts.year_auroc_draws(
        year_rows([f"p{i + 1}" for i in reversed_order], [in_R[i] for i in reversed_order],
                  [in_A[i] for i in reversed_order], activation[reversed_order], linear[reversed_order]),
        count, POPULATION)
    for key in ("auroc_activation", "auroc_linear", "p_a_given_r", "p_a_given_not_r"):
        np.testing.assert_allclose(first["draws"][key], second["draws"][key], rtol=0, atol=1e-12)


def test_patient_table_ids_alone_raise_for_draw_counts():
    count = cc.patient_weights(POPULATION, seeds=[31, 32])
    rows = year_rows(["p1", "p2", "p3", "p4"], [1, 0, 1, 0], [1, 1, 0, 0], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4])
    table_ids = ["p1", "p2", "p3", "p4"]
    with pytest.raises(ValueError):
        ts.year_auroc_draws(rows, count, table_ids)
    with pytest.raises(ValueError):
        ts.auroc_changes({2012: rows}, 2012, count, table_ids)


def test_auroc_hand_count_with_tied_scores():
    rows = year_rows(["p1", "p2", "p3", "p4", "p5", "p6"], [1, 1, 0, 0, 0, 0], [1, 0, 1, 1, 1, 0],
                     [0, 1, 0, 2, 0, 0], [0, 1, 0, 2, 0, 0])
    out = ts.year_auroc_draws(rows, np.ones((1, 6)), POPULATION)
    assert out["point"]["auroc_activation"] == pytest.approx(4.5 / 8)
    assert out["draws"]["auroc_linear"][0] == pytest.approx(4.5 / 8)


def test_p_a_given_r_and_not_r_use_rule_and_non_rule_rows_by_draw():
    rows = year_rows(["p1", "p2", "p3", "p4", "p5", "p6"], [1, 1, 0, 0, 0, 0], [1, 0, 1, 1, 1, 0],
                     [0, 1, 0, 2, 0, 0], [0, 1, 0, 2, 0, 0])
    out = ts.year_auroc_draws(rows, np.ones((1, 6)), POPULATION)
    assert out["draws"]["p_a_given_r"][0] == pytest.approx(0.5)
    assert out["draws"]["p_a_given_not_r"][0] == pytest.approx(3 / 4)


def test_zero_shares_by_rule_group_match_hand_counts():
    rows = year_rows(["p1", "p2", "p3", "p4", "p5", "p6"], [1, 1, 0, 0, 0, 0], [1, 0, 1, 1, 1, 0],
                     [0, 1, 0, 2, 0, 0], [0, 1, 0, 2, 0, 0])
    out = ts.year_auroc_draws(rows, np.ones((1, 6)), POPULATION)
    assert out["zero_share"]["R"] == pytest.approx(0.5)
    assert out["zero_share"]["not_R"] == pytest.approx(3 / 4)


def test_auroc_changes_are_taken_from_reference_draw_by_draw_with_paired_difference():
    activation = np.array([0, 1, 0, 2, 0, 0], float)
    in_R = [1, 1, 0, 0, 0, 0]
    in_A = [1, 0, 1, 1, 1, 0]
    reference_rows = year_rows(["p1", "p2", "p3", "p4", "p5", "p6"], in_R, in_A, activation, activation)
    later_rows = year_rows(["p1", "p2", "p3", "p4", "p5", "p6"], in_R, in_A, np.zeros(6), activation)
    out = ts.auroc_changes({2009: reference_rows, 2010: later_rows}, 2009, np.ones((1, 6)), POPULATION)
    assert out["changes"][2009]["change_auroc_activation"][0] == pytest.approx(0.0)
    assert out["changes"][2010]["change_auroc_activation"][0] == pytest.approx(0.5 - 4.5 / 8)
    assert out["changes"][2010]["change_auroc_linear"][0] == pytest.approx(0.0)
    assert out["changes"][2010]["paired_auroc_difference"][0] == pytest.approx(0.5 - 4.5 / 8)


def fitted_pipeline(preprocessing, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(60, 3)) * np.array([0.5, 2.0, 4.0])
    y = np.zeros(60, bool)
    y[:20] = True
    X[y] += 1.5
    model = cr.probe_pipeline(preprocessing, SVC(C=1.0, kernel="rbf", gamma="scale"))
    model.fit(X, y)
    new_rows = np.random.default_rng(seed + 1).normal(size=(15, 3)) * np.array([0.5, 2.0, 4.0])
    return model, new_rows


@pytest.mark.parametrize("preprocessing", ["raw", "standardised"])
def test_kernel_terms_reproduce_decision_function(preprocessing):
    model, rows = fitted_pipeline(preprocessing)
    out = ts.kernel_terms(model, rows)
    np.testing.assert_allclose(out["score"], model.decision_function(rows), rtol=0, atol=1e-9)


@pytest.mark.parametrize("preprocessing", ["raw", "standardised"])
def test_rows_far_from_every_support_vector_score_the_intercept(preprocessing):
    model, rows = fitted_pipeline(preprocessing)
    out = ts.kernel_terms(model, rows + 1e3)
    intercept = float(model[-1].intercept_[0])
    np.testing.assert_allclose(out["score"], intercept, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["kernel_term"], 0.0, rtol=0, atol=1e-12)


def test_scaled_distances_use_the_pipelines_own_scaler():
    model, rows = fitted_pipeline("standardised")
    out = ts.kernel_terms(model, rows)
    scaled = model[0].transform(rows)
    support = model[-1].support_vectors_
    gamma = float(model[-1]._gamma)
    squared = ((scaled[:, None, :] - support[None, :, :]) ** 2).sum(-1)
    positive = model[-1].dual_coef_[0] > 0
    np.testing.assert_allclose(out["gamma_sq_distance_all"], gamma * squared.min(1), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(out["gamma_sq_distance_positive"], gamma * squared[:, positive].min(1),
                               rtol=1e-9, atol=1e-12)
    unscaled = ((rows[:, None, :] - support[None, :, :]) ** 2).sum(-1)
    assert not np.allclose(out["gamma_sq_distance_all"], gamma * unscaled.min(1))


def test_ratio_to_reference_is_one_at_the_reference_year():
    values = {2010: np.array([0.5, 2.0, 4.0]), 2011: np.array([1.0, 1.0, 8.0])}
    out = ts.ratio_to_reference(values, 2010)
    np.testing.assert_allclose(out[2010], 1.0)
    np.testing.assert_allclose(out[2011], [2.0, 0.5, 2.0])


POP8 = np.array([f"p{i}" for i in range(1, 9)])


def overlap_rows(in_R, missed_region, missed_other, patient_ids=POP8):
    """Rows for one year: region score -1 (missed) or 1 (not missed); other reading as given."""
    return {"patient_ids": np.array(patient_ids), "in_R": np.array(in_R, bool),
            "region_score": np.where(np.array(missed_region, bool), -1.0, 1.0),
            "other_missed": np.array(missed_other, bool)}


def test_identical_missed_sets_give_contrast_one():
    in_R = [1, 1, 1, 1, 1, 1, 0, 0]
    missed = [1, 1, 0, 0, 0, 0, 0, 0]
    out = ts.missed_overlap({2012: overlap_rows(in_R, missed, missed)}, 2012, np.ones((1, 8)), POP8)
    point = out["per_year"][2012]["point"]
    assert point["p_n_given_m"] == pytest.approx(1.0)
    assert point["p_n_given_not_m"] == pytest.approx(0.0)
    assert point["contrast"] == pytest.approx(1.0)


def test_independent_missed_sets_give_contrast_zero():
    in_R = [1] * 8
    missed_region = [1, 1, 1, 1, 0, 0, 0, 0]
    missed_other = [1, 0, 1, 0, 1, 0, 1, 0]
    out = ts.missed_overlap({2012: overlap_rows(in_R, missed_region, missed_other)}, 2012, np.ones((1, 8)), POP8)
    point = out["per_year"][2012]["point"]
    assert point["p_n_given_m"] == pytest.approx(0.5)
    assert point["p_n_given_not_m"] == pytest.approx(0.5)
    assert point["contrast"] == pytest.approx(0.0)


def test_shares_match_hand_counts_and_ignore_non_rule_rows():
    in_R = [1, 1, 1, 1, 1, 1, 0, 0]
    missed_region = [1, 1, 0, 0, 0, 0, 0, 0]
    missed_other = [1, 0, 1, 1, 1, 0, 1, 1]
    out = ts.missed_overlap({2012: overlap_rows(in_R, missed_region, missed_other)}, 2012, np.ones((1, 8)), POP8)
    point = out["per_year"][2012]["point"]
    assert point["p_n_given_m"] == pytest.approx(1 / 2)
    assert point["p_n_given_not_m"] == pytest.approx(3 / 4)
    assert point["p_n_given_r"] == pytest.approx(4 / 6)
    assert point["contrast"] == pytest.approx(-0.25)
    assert out["per_year"][2012]["counts"] == {"R_rows": 6, "region_misses": 2, "other_misses": 4}


def test_reference_value_is_subtracted_draw_by_draw_with_defined_fraction():
    count = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 0, 0, 0, 0, 1, 1]], float)
    in_R = [1, 1, 1, 1, 1, 1, 0, 0]
    missed = [1, 1, 0, 0, 0, 0, 0, 0]
    reference_rows = overlap_rows(in_R, missed, [1, 0, 1, 1, 1, 0, 0, 0])
    later_rows = overlap_rows(in_R, missed, [1, 1, 0, 0, 0, 0, 0, 0])
    out = ts.missed_overlap({2012: reference_rows, 2013: later_rows}, 2012, count, POP8)
    assert out["reference_status"] == "2 missed rows"
    assert out["reference_undefined_fraction"]["p_n_given_not_m"] == pytest.approx(0.5)
    change = out["per_year"][2013]["change"]
    np.testing.assert_allclose(change["p_n_given_m"], [0.5, 0.5], atol=1e-12)
    assert out["per_year"][2013]["defined_fraction"]["p_n_given_m"] == pytest.approx(1.0)
    assert change["p_n_given_not_m"][0] == pytest.approx(-0.75)
    assert np.isnan(change["p_n_given_not_m"][1])
    assert out["per_year"][2013]["defined_fraction"]["p_n_given_not_m"] == pytest.approx(0.5)


def test_reference_with_no_misses_reports_later_values_as_all_new():
    in_R = [1, 1, 1, 1, 1, 1, 0, 0]
    later_missed = [1, 1, 0, 0, 0, 0, 0, 0]
    out = ts.missed_overlap({2012: overlap_rows(in_R, [0] * 8, [0] * 8),
                             2013: overlap_rows(in_R, later_missed, later_missed)}, 2012, np.ones((1, 8)), POP8)
    assert out["reference_status"] == "no reference misses"
    later = out["per_year"][2013]
    assert later["label"] == "all later misses are new"
    assert later["change"] is None
    assert later["point"]["p_n_given_m"] == pytest.approx(1.0)
    assert not np.isnan(later["draws"]["p_n_given_m"]).any()


def test_undefined_later_years_say_why_and_reference_all_missed_is_labelled():
    in_R = [1, 1, 1, 1, 1, 1, 0, 0]
    all_missed = [1, 1, 1, 1, 1, 1, 0, 0]
    none_missed = [0] * 8
    out = ts.missed_overlap({2012: overlap_rows(in_R, all_missed, [1, 0, 1, 0, 1, 0, 0, 0]),
                             2013: overlap_rows(in_R, none_missed, [1, 0, 1, 0, 1, 0, 0, 0]),
                             2014: overlap_rows(in_R, all_missed, [1, 0, 1, 0, 1, 0, 0, 0])},
                            2012, np.ones((1, 8)), POP8)
    assert out["reference_status"] == "misses all"
    assert out["reference_undefined_fraction"]["p_n_given_not_m"] == pytest.approx(1.0)
    none_year = out["per_year"][2013]
    assert np.isnan(none_year["point"]["p_n_given_m"])
    assert none_year["reason"] == "region misses no R rows"
    all_year = out["per_year"][2014]
    assert np.isnan(all_year["point"]["p_n_given_not_m"])
    assert all_year["reason"] == "region misses all R rows"
    assert all_year["point"]["p_n_given_m"] == pytest.approx(3 / 6)


def profile_for(column, in_R, bump=0):
    """Export-style profile of one column over the rule rows (bump shifts the positive count)."""
    rows = int(in_R.sum())
    positives = int((column[in_R] > 0).sum()) + bump
    return {"positive_prevalence": [positives / rows], "rows": rows,
            "mean_magnitude": [float(np.mean(column[in_R]))], "status": "available"}


def yearly_fixture(bump_candidate=None):
    """Evaluation activations of one yearly SAE (8 rows, 6 latents); frozen member is latent 2, candidates 4, 1, 5."""
    rng = np.random.default_rng(5)
    activations = np.maximum(rng.normal(size=(8, 6)), 0).astype(np.float32)
    in_R = np.array([1, 1, 1, 0, 0, 0, 1, 0], bool)
    candidate_factors = [4, 1, 5]
    candidates = []
    for rank, factor in enumerate(candidate_factors):
        bump = 1 if rank == bump_candidate else 0
        candidates.append({"factor": factor, "pearson": 0.9 - 0.1 * rank,
                           "profile_inside_R": profile_for(activations[:, factor], in_R, bump)})
    entry = {"reference": 2008, "seed": 42, "year": 2009, "concept": "all_history/2008/2008/42/42:283",
             "reference_factor": 2, "dictionary_identity": "id-a", "candidates": candidates}
    return entry, activations, in_R


def fake_loader(identity="id-a", encoding_batch=None, error=None):
    def load_unit(reference, seed, year):
        if error is not None:
            raise RuntimeError(error)
        return {"unit_identity": identity, "encoding_batch": encoding_batch}
    return load_unit


def fake_encoder(activations, calls):
    def encode(unit, batch_size):
        calls.append(batch_size)
        return activations
    return encode


def test_loader_error_is_unmeasured_with_its_text_and_no_encoding():
    entry, activations, in_R = yearly_fixture()
    calls = []
    out = ts.yearly_candidate_auroc(entry, fake_loader(error="Yearly SAE is unavailable: missing"),
                                    fake_encoder(activations, calls), activations[:, 2], in_R, POP8,
                                    np.ones((1, 8)), POP8)
    assert out["status"] == "unmeasured: Yearly SAE is unavailable: missing"
    assert all(row["status"] == out["status"] for row in out["candidates"])
    assert calls == []


def test_identity_mismatch_is_unmeasured_and_not_encoded():
    entry, activations, in_R = yearly_fixture()
    calls = []
    out = ts.yearly_candidate_auroc(entry, fake_loader(identity="id-other"), fake_encoder(activations, calls),
                                    activations[:, 2], in_R, POP8, np.ones((1, 8)), POP8)
    assert out["status"] == "unmeasured: identity differs"
    assert calls == []


def test_profile_mismatch_marks_one_candidate_and_the_build_continues():
    entry, activations, in_R = yearly_fixture(bump_candidate=1)
    out = ts.yearly_candidate_auroc(entry, fake_loader(), fake_encoder(activations, []), activations[:, 2],
                                    in_R, POP8, np.ones((1, 8)), POP8)
    assert out["status"] == "measured"
    statuses = [row["status"] for row in out["candidates"]]
    assert statuses == ["measured", "unmeasured: profile differs", "measured"]
    assert out["candidates"][1]["rows_differ"] == 1
    assert out["candidates"][0]["auroc_draws"] is not None
    assert out["frozen_auroc_draws"].shape == (1,)


def test_candidates_keep_the_given_pearson_order():
    entry, activations, in_R = yearly_fixture()
    entry["candidates"] = list(reversed(entry["candidates"]))
    out = ts.yearly_candidate_auroc(entry, fake_loader(), fake_encoder(activations, []), activations[:, 2],
                                    in_R, POP8, np.ones((1, 8)), POP8)
    assert [row["factor"] for row in out["candidates"]] == [5, 1, 4]
    assert [row["rank"] for row in out["candidates"]] == [1, 2, 3]


def test_encoding_batch_comes_from_cache_config_else_default_and_is_recorded():
    entry, activations, in_R = yearly_fixture()
    calls = []
    out = ts.yearly_candidate_auroc(entry, fake_loader(encoding_batch=64), fake_encoder(activations, calls),
                                    activations[:, 2], in_R, POP8, np.ones((1, 8)), POP8)
    assert calls == [64] and out["encoding_batch"] == 64 and out["encoding_batch_source"] == "cache config"

    calls = []
    out = ts.yearly_candidate_auroc(entry, fake_loader(), fake_encoder(activations, calls),
                                    activations[:, 2], in_R, POP8, np.ones((1, 8)), POP8)
    assert calls == [256] and out["encoding_batch"] == 256 and out["encoding_batch_source"] == "default"


def test_difference_is_candidate_minus_frozen_member_draw_by_draw():
    entry, activations, in_R = yearly_fixture()
    count = cc.patient_weights(POP8, seeds=[7, 8])
    out = ts.yearly_candidate_auroc(entry, fake_loader(), fake_encoder(activations, []), activations[:, 2],
                                    in_R, POP8, count, POP8)
    frozen = [brute_weighted_auroc(in_R, activations[:, 2], count[d]) for d in range(2)]
    np.testing.assert_allclose(out["frozen_auroc_draws"], frozen, atol=1e-12)
    first = out["candidates"][0]
    expected_first = [brute_weighted_auroc(in_R, activations[:, 4], count[d]) for d in range(2)]
    np.testing.assert_allclose(first["auroc_draws"], expected_first, atol=1e-12)
    np.testing.assert_allclose(first["difference_draws"], np.array(expected_first) - np.array(frozen), atol=1e-12)


def export_profile(column, in_R):
    """Export profile as written by activation_profile (pilot_follow_ups.ipynb cell 22, code line 2-3):
    positive_prevalence = (selected > 0).mean(), a fraction of the rule rows, not a count; mean_magnitude =
    selected.mean() over all rule rows, zeros included. The same definition appears in
    concept_subgroup_drift_pilot.py:7068-7078 (_activation_statistics: positive_prevalence = positive.mean(),
    magnitude_mean = selected.mean())."""
    selected = column[in_R]
    return {"rows": len(selected), "positive_prevalence": [float((selected > 0).mean())],
            "mean_magnitude": [float(selected.mean())], "status": "available"}


def test_profile_check_follows_activation_profile_definition():
    column = np.array([0.0, 0.0, 0.0, 2.0, 5.0, 0.0])
    in_R = np.array([1, 1, 1, 1, 0, 0], bool)
    profile = export_profile(column, in_R)
    assert profile["positive_prevalence"] == [0.25] and profile["mean_magnitude"] == [0.5]
    assert ts.profile_check(column[in_R], profile)["status"] == "measured"


def test_profile_check_rejects_positive_only_mean_and_prevalence_count():
    column = np.array([0.0, 0.0, 0.0, 2.0])
    positive_only_mean = {"rows": 4, "positive_prevalence": [0.25], "mean_magnitude": [2.0], "status": "available"}
    prevalence_as_count = {"rows": 4, "positive_prevalence": [1.0], "mean_magnitude": [0.5], "status": "available"}
    assert ts.profile_check(column, positive_only_mean)["status"] == "unmeasured: profile differs"
    assert ts.profile_check(column, prevalence_as_count)["status"] == "unmeasured: profile differs"


def test_anchor_row_passes_within_tolerance_and_treats_nan_pairs_as_equal():
    passed = ts.anchor_row("linear AUROC low", np.array([0.5, np.nan]), np.array([0.5 + 1e-13, np.nan]), 1e-12)
    assert passed["passed"] is True and passed["max_gap"] < 1e-12
    failed = ts.anchor_row("linear AUROC low", np.array([0.5, 0.6]), np.array([0.5, 0.6 + 1e-6]), 1e-12)
    assert failed["passed"] is False and failed["max_gap"] == pytest.approx(1e-6)


def test_two_by_two_by_two_counts_rule_rows_only():
    in_R = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
    region = np.array([1, 1, 0, 0, 1, 1, 1, 1], bool)
    other = np.array([1, 0, 1, 1, 1, 1, 1, 1], bool)
    in_A = np.array([1, 1, 1, 0, 1, 1, 1, 1], bool)
    table = ts.two_by_two_by_two(in_R, region, other, in_A)
    assert len(table) == 8
    assert sum(table.values()) == 4
    assert table[(True, True, True)] == 1
    assert table[(True, False, True)] == 1
    assert table[(False, True, False)] == 1
    assert table[(True, True, False)] == 0


def test_medoid_gap_is_the_largest_absolute_difference():
    assert ts.medoid_gap(np.array([0.1, 0.5]), np.array([0.1, 0.5])) == 0.0
    assert ts.medoid_gap(np.array([0.1, 0.5]), np.array([0.1, 0.75])) == pytest.approx(0.25)


def test_rank_correlation_drops_pairs_with_nan_and_reports_n():
    rho = ts.rank_correlation(np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0, 30.0, 40.0]))
    assert rho == {"rho": pytest.approx(1.0), "n": 4}
    with_nan = ts.rank_correlation(np.array([1.0, 2.0, 3.0, np.nan]), np.array([10.0, 20.0, 30.0, 5.0]))
    assert with_nan == {"rho": pytest.approx(1.0), "n": 3}


def test_json_ready_maps_non_finite_to_none_and_numpy_to_python():
    value = {"a": np.float64(np.nan), "b": np.inf, "c": np.int64(3), "d": np.array([1.0, np.nan]), "e": np.bool_(True)}
    assert ts.json_ready(value) == {"a": None, "b": None, "c": 3, "d": [1.0, None], "e": True}


def test_write_run_record_writes_strict_json_atomically(tmp_path):
    target = tmp_path / "run.json"
    ts.write_run_record(target, {"status": "completed", "gap": float("nan"), "n": np.int64(2)})
    assert json.loads(target.read_text()) == {"status": "completed", "gap": None, "n": 2}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]


def test_relative_to_project_never_returns_a_home_path(tmp_path):
    project = tmp_path / "SAE-TCAV"
    home = tmp_path / "home"
    assert ts.relative_to_project(project / "stats/x.csv", project, home) == "stats/x.csv"
    assert ts.relative_to_project(home / "rpi-runs/y.json", project, home) == "~/rpi-runs/y.json"
    with pytest.raises(ValueError):
        ts.relative_to_project(tmp_path / "elsewhere/z.json", project, home)


def test_decision_gap_compares_float64_rows_so_float32_input_does_not_show_a_gap():
    model, rows = fitted_pipeline("standardised")
    rows32 = rows.astype(np.float32)
    assert ts.decision_gap(model, rows32) <= 1e-9
    assert ts.decision_gap(model, rows) <= 1e-9


def test_standardiser_keeps_float32_input_in_float32_which_is_the_cause():
    model, rows = fitted_pipeline("standardised")
    assert model[0].transform(rows.astype(np.float32)).dtype == np.float32
    assert model[0].transform(rows).dtype == np.float64


def test_failed_anchors_lists_the_names_that_did_not_pass():
    rows = [ts.anchor_row("a", [1.0], [1.0], 1e-9), ts.anchor_row("b", [1.0], [2.0], 1e-9)]
    assert ts.failed_anchors(rows) == ["b"]
    assert ts.failed_anchors([]) == []


def test_reference_year_row_is_not_labelled_as_later_when_reference_has_no_misses():
    in_R = [1, 1, 1, 1, 1, 1, 0, 0]
    out = ts.missed_overlap({2012: overlap_rows(in_R, [0] * 8, [0] * 8),
                             2013: overlap_rows(in_R, [1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0])},
                            2012, np.ones((1, 8)), POP8)
    assert out["per_year"][2012]["label"] == ""
    assert out["per_year"][2013]["label"] == "all later misses are new"


def test_every_undefined_contrast_has_a_reason():
    in_R = [1, 1, 1, 1, 1, 1, 0, 0]
    cases = {
        "region misses none": ([0] * 8, [1, 0, 1, 0, 0, 0, 0, 0]),
        "region misses all": ([1, 1, 1, 1, 1, 1, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0]),
        "defined": ([1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 0, 0, 0, 0]),
    }
    for name, (missed, other) in cases.items():
        out = ts.missed_overlap({2012: overlap_rows(in_R, missed, other)}, 2012, np.ones((1, 8)), POP8)
        entry = out["per_year"][2012]
        if np.isnan(entry["point"]["contrast"]):
            assert entry["reason"] in ("region misses no R rows", "region misses all R rows"), name
        else:
            assert entry["reason"] == "", name


def test_empty_export_profile_is_a_status_not_a_crash():
    column = np.array([0.0, 1.0, 2.0])
    empty = {"rows": 0, "positive_prevalence": None, "mean_magnitude": None, "status": "empty subgroup"}
    result = ts.profile_check(column[np.array([False, False, False])], empty)
    assert result["status"] == "unmeasured: no export profile (empty subgroup)"
    assert result["rows_differ"] is None


def test_missing_export_profile_is_a_status_not_a_crash():
    result = ts.profile_check(np.array([1.0, 2.0]), None)
    assert result["status"] == "unmeasured: no export profile (missing)"


def test_yearly_entry_with_empty_profile_on_one_candidate_continues_the_build():
    entry, activations, in_R = yearly_fixture()
    entry["candidates"][1]["profile_inside_R"] = {"rows": 0, "positive_prevalence": None,
                                                  "mean_magnitude": None, "status": "empty subgroup"}
    out = ts.yearly_candidate_auroc(entry, fake_loader(), fake_encoder(activations, []), activations[:, 2],
                                    in_R, POP8, np.ones((1, 8)), POP8)
    assert out["status"] == "measured"
    assert [row["status"] for row in out["candidates"]] == [
        "measured", "unmeasured: no export profile (empty subgroup)", "measured"]


def test_yearly_entry_with_no_candidate_list_does_not_crash():
    entry, activations, in_R = yearly_fixture()
    entry["candidates"] = None
    out = ts.yearly_candidate_auroc(entry, fake_loader(), fake_encoder(activations, []), activations[:, 2],
                                    in_R, POP8, np.ones((1, 8)), POP8)
    assert out["status"] == "measured" and out["candidates"] == []


def test_anchor_row_reports_observed_and_expected_values():
    row = ts.anchor_row("draw matrix low/high", [0.5, 0.25, 0.75], [0.5, 0.2500001, 0.75], 1e-12)
    assert row["observed"] == [0.5, 0.25, 0.75]
    assert row["expected"] == [0.5, 0.2500001, 0.75]
    assert row["passed"] is False


def test_no_rule_patients_is_undefined_not_no_misses():
    in_R = [0] * 8
    out = ts.missed_overlap({2012: overlap_rows(in_R, [0] * 8, [0] * 8),
                             2013: overlap_rows(in_R, [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0])},
                            2012, np.ones((1, 8)), POP8)
    assert out["reference_status"] == "undefined (no rule patients that year)"
    assert out["per_year"][2013]["counts"]["R_rows"] == 0
    assert out["per_year"][2013]["reason"] == "undefined (no rule patients that year)"


def test_member_rows_keep_seed_retention_for_the_seed_spread():
    rows = ts.member_rows(synthetic_export(), "all_history")
    assert len(rows) == 1
    row = rows[0]
    assert (row["factor"], row["seed"], row["retention"]) == (283, 42, 0.6)
    assert row["concept"] == "all_history/2008/2008/42/42:283,135:131"
    assert (row["A_rows"], row["R_rows"]) == (14, 20)


def test_year_group_puts_years_between_reference_and_2012_before_2012():
    # Fix 14: a 2009 row from a 2008 reference must not be labelled 2012-2014
    assert ts.year_group(2009, 2008) == "before 2012"
    assert ts.year_group(2011, 2007) == "before 2012"
    assert ts.year_group(2009, 2008) != "2012-2014"


def test_year_group_2012_to_2014_are_calendar_years_only():
    assert ts.year_group(2012, 2008) == "2012-2014"
    assert ts.year_group(2013, 2012) == "2012-2014"
    assert ts.year_group(2014, 2010) == "2012-2014"


def test_year_group_reference_and_2015_are_their_own_groups():
    assert ts.year_group(2008, 2008) == "reference"
    assert ts.year_group(2015, 2009) == "2015 separate"
    assert ts.year_group(2015, 2013) == "2015 separate"


def test_year_group_rejects_years_outside_the_groups():
    with pytest.raises(ValueError):
        ts.year_group(2007, 2008)
    with pytest.raises(ValueError):
        ts.year_group(2016, 2013)
