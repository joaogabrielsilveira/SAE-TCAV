import numpy as np
import pandas as pd
import pytest
import torch

import flchain_probe as probe
import prospective_audit as pa


REFERENCE_ROLES = ("tabpfn_context", "sae_discovery", "rule_discovery", "cav_fit", "concept_selection", "evaluation")


def _events(rows):
    frame = pd.DataFrame(rows, columns=["patient_id", "date", "event"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _population(rows):
    frame = pd.DataFrame(rows, columns=["patient_id", "year", "y"])
    return {"patient_id": frame["patient_id"].to_numpy(dtype=str),
            "year": frame["year"].to_numpy(dtype=np.int64),
            "y": frame["y"].to_numpy(dtype=np.int64)}


def _frame(rows):
    return pd.DataFrame(rows, columns=["row", "patient_id", "year", "role", "outcome_valid"])


def _synthetic_population(patients=60, seed=3):
    rng = np.random.default_rng(seed)
    rows, first = [], []
    for k in range(patients):
        pid = f"P{k:03d}"
        death = int(rng.integers(2007, 2014)) if rng.random() < 0.3 else None
        for year in range(2007, 2011):
            if death is not None and year > death:
                break
            rows.append((pid, year, int(death == year)))
            first.append(np.nan if death is None else float(death))
    population = _population(rows)
    return population, np.asarray(first)


def test_first_death_years_reads_full_history_including_pre_2007_deaths():
    events = _events([
        (5, "1999-03-01", "DEATH"), (5, "2008-01-01", "EVENT_A"),
        (7, "2011-06-01", "DEATH"), (7, "2010-06-01", "DEATH"), (9, "2009-01-01", "EVENT_A"),
    ])
    first = pa.first_death_years(events)
    assert first.loc["5"] == 1999
    assert first.loc["7"] == 2010
    assert "9" not in first.index


def test_first_death_year_by_row_maps_never_died_to_nan_and_rejects_unknown_patients():
    events = _events([(5, "1999-03-01", "DEATH"), (9, "2009-01-01", "EVENT_A"), (7, "2010-01-01", "DEATH")])
    population = {"patient_id": np.asarray(["5", "9", "7"]), "year": np.asarray([2008, 2008, 2008])}
    values = pa.first_death_year_by_row(population, events)
    np.testing.assert_array_equal(values[[0, 2]], [1999.0, 2010.0])
    assert np.isnan(values[1])
    with pytest.raises(ValueError):
        pa.first_death_year_by_row({"patient_id": np.asarray(["11"]), "year": np.asarray([2008])}, events)


def test_landmark_cohort_removes_pre_2007_and_same_year_deaths_and_keeps_alive_rows():
    population = _population([
        ("A", 2008, 0),   # died 1999: removed
        ("B", 2008, 0),   # died 2009: kept, next-year death
        ("B", 2009, 1),   # died 2009: same-year death, removed
        ("C", 2008, 0),   # never died: kept, no death
        ("C", 2014, 0),   # never died: kept, representation only
        ("C", 2015, 0),   # input year 2015 dropped
        ("D", 2007, 0),   # died 2008, no 2008 row in the population: kept
        ("E", 2013, 0),   # died 2014: kept, last outcome-valid input year
    ])
    first = np.array([1999, 2009, 2009, np.nan, np.nan, np.nan, 2008, 2014], dtype=float)
    cohort = pa.landmark_cohort(population, first)
    assert cohort.row.tolist() == [1, 3, 4, 6, 7]
    np.testing.assert_array_equal(cohort.y_next.to_numpy(), [1, 0, np.nan, 1, 1])
    assert cohort.outcome_valid.tolist() == [True, True, False, True, True]
    assert cohort.representation_only.tolist() == [False, False, True, False, False]
    assert (cohort.first_death_year_raw.isna() | (cohort.first_death_year_raw > cohort.year)).all()


def test_cohort_checks_pass_on_a_valid_cohort():
    population = _population([("A", 2008, 0), ("B", 2014, 0), ("C", 2013, 0)])
    cohort = pa.landmark_cohort(population, np.array([np.nan, np.nan, 2014.0]))
    checks = pa.cohort_check_rows(cohort)
    assert checks["passed"].all()


def test_add_roles_assigns_one_role_per_patient_and_covers_reference_year():
    population, first = _synthetic_population()
    cohort = pa.landmark_cohort(population, first)
    cohort, roles = pa.add_roles(cohort, population, 2009, seed=42)
    assert set(cohort.role) <= set(REFERENCE_ROLES) | {"fitting", ""}
    assigned = cohort[cohort.year >= 2009]
    assert (assigned.groupby("patient_id").role.nunique() == 1).all()
    reference = roles["reference_roles"]
    assert set(reference) == set(REFERENCE_ROLES)


def test_context_mask_is_last_3_window_with_reference_year_rows_from_context_role_only():
    cohort = _frame([
        (0, "A", 2007, "", True),
        (1, "A", 2008, "", True),
        (2, "A", 2009, "tabpfn_context", True),
        (3, "B", 2008, "", True),
        (4, "B", 2009, "evaluation", True),
        (5, "C", 2010, "tabpfn_context", True),
        (6, "D", 2007, "", True),
    ])
    mask = pa.context_mask(cohort, 2009)
    assert mask.tolist() == [True, True, True, True, False, False, True]
    assert pa.context_first_year(2009) == 2007
    assert pa.context_first_year(2010) == 2008
    assert pa.context_first_year(2007) == 2007
    assert pa.context_first_year(2012) == 2010


def test_selection_and_evaluation_masks_exclude_context_and_censored_rows():
    cohort = _frame([
        (0, "A", 2009, "rule_discovery", True),
        (1, "B", 2009, "cav_fit", True),
        (2, "C", 2009, "concept_selection", True),
        (3, "D", 2009, "evaluation", True),
        (4, "E", 2009, "sae_discovery", True),
        (5, "F", 2009, "tabpfn_context", True),
        (6, "D", 2014, "evaluation", False),
        (7, "G", 2010, "evaluation", True),
        (8, "H", 2008, "", True),
    ])
    assert np.flatnonzero(pa.selection_mask(cohort, 2009)).tolist() == [0, 1, 2]
    assert np.flatnonzero(pa.evaluation_mask(cohort, 2009)).tolist() == [3, 7]


def test_death_f1_threshold_picks_best_flag_cutoff_from_given_rows_only():
    labels = np.array([1, 0, 1, 0])
    probabilities = np.array([0.9, 0.6, 0.4, 0.1])
    assert pa.death_f1_threshold(labels, probabilities) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        pa.death_f1_threshold(np.zeros(3, dtype=int), np.array([0.2, 0.5, 0.7]))


def test_prior_context_counts_count_each_patients_context_rows():
    cohort = _frame([
        (0, "A", 2007, "", True), (1, "A", 2008, "", True), (2, "A", 2009, "evaluation", True),
        (3, "B", 2009, "evaluation", True),
    ])
    context = np.array([True, True, False, False])
    assert pa.prior_context_counts(cohort, context).tolist() == [2, 2, 2, 0]


def test_binary_entropy_in_nats_and_mean_entropy_in_metrics():
    flat = pa.longitudinal_metrics(np.array([0, 1, 0, 1]), np.full(4, 0.5), np.ones(4), 0.9, 0.5)
    assert flat["mean_entropy_nats"] == pytest.approx(np.log(2))
    edge = pa.longitudinal_metrics(np.array([0, 1]), np.array([0.0, 1.0]), np.ones(2), 0.5, 0.5)
    assert edge["mean_entropy_nats"] == pytest.approx(0.0)
    p = 0.2
    shifted = pa.longitudinal_metrics(np.array([0, 1]), np.full(2, p), np.ones(2), 0.5, 0.5)
    assert shifted["mean_entropy_nats"] == pytest.approx(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


def test_recalibration_recovers_known_slope_and_intercept_from_grouped_weights():
    probabilities = np.array([0.1, 0.2, 0.4, 0.7])
    target = 1.0 / (1.0 + np.exp(-(0.5 + 2.0 * np.log(probabilities / (1 - probabilities)))))
    p = np.repeat(probabilities, 2)
    y = np.tile([1, 0], len(probabilities))
    w = np.empty(len(p))
    w[0::2] = target
    w[1::2] = 1 - target
    slope, intercept = pa.recalibration_slope_intercept(y, p, w)
    assert slope == pytest.approx(2.0, rel=1e-3)
    assert intercept == pytest.approx(0.5, rel=1e-3)


def test_patient_draws_pair_rows_by_patient_id_not_row_order():
    ids = np.array(["A", "B", "A", "C", "B", "D"])
    probs = np.array([0.1, 0.4, 0.2, 0.7, 0.5, 0.9])
    labels = np.array([0, 1, 0, 1, 0, 1])
    universe, counts = pa.patient_draw_counts(np.unique(ids), draws=25, seed=7)
    codes = pa.row_patient_codes(ids, universe)
    weights = counts[:, codes]
    assert (weights[:, 0] == weights[:, 2]).all()
    assert (weights[:, 1] == weights[:, 4]).all()

    point, draws = pa.metric_draws(labels, probs, codes, counts, 0.5, 0.5)
    point_b, draws_b = pa.metric_draws(labels, probs, codes, counts, 0.5, 0.5)
    diff_point, diff_draws = pa.difference_draws(point, draws, point_b, draws_b)
    assert (diff_draws.fillna(0.0).to_numpy() == 0.0).all()

    perm = np.array([5, 2, 0, 3, 1, 4])
    codes_perm = pa.row_patient_codes(ids[perm], universe)
    point_p, draws_p = pa.metric_draws(labels[perm], probs[perm], codes_perm, counts, 0.5, 0.5)
    pd.testing.assert_frame_equal(draws, draws_p)


def test_bootstrap_constants_and_seed_are_reproducible():
    assert pa.BOOTSTRAP_DRAWS == 500
    assert pa.BOOTSTRAP_SEED == 20261106
    universe = np.array(["P1", "P2", "P3"])
    _, first = pa.patient_draw_counts(universe, draws=pa.BOOTSTRAP_DRAWS, seed=pa.BOOTSTRAP_SEED)
    _, again = pa.patient_draw_counts(universe, draws=pa.BOOTSTRAP_DRAWS, seed=pa.BOOTSTRAP_SEED)
    _, other = pa.patient_draw_counts(universe, draws=pa.BOOTSTRAP_DRAWS, seed=1)
    assert first.shape == (500, 3)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
    assert (first.sum(axis=1) == 3).all()


def test_summarize_draws_returns_point_interval_and_valid_count():
    point = pd.Series({"auroc": 0.7, "brier": 0.1})
    draws = pd.DataFrame({"auroc": np.r_[np.linspace(0.6, 0.8, 99), np.nan], "brier": np.full(100, 0.1)})
    table = pa.summarize_draws(point, draws)
    assert table.loc["auroc", "valid_draws"] == 99
    assert table.loc["auroc", "ci_low"] < 0.7 < table.loc["auroc", "ci_high"]
    assert table.loc["brier", "ci_low"] == pytest.approx(0.1)


def test_timing_features_fill_missing_months_and_flag_no_care_events():
    events = _events([
        (1, "2012-01-05", "EVENT_A"), (1, "2012-03-05", "EVENT_B"), (2, "2012-07-01", "DEATH"),
    ])
    cohort = _frame([(0, "1", 2012, "evaluation", True), (1, "2", 2012, "evaluation", True)])
    timing = pa.timing_features(cohort, events)
    assert timing["last_month"].tolist() == [3, 0]
    assert timing["active_months"].tolist() == [2, 0]
    assert timing["no_care_event"].tolist() == [False, True]


def test_baseline_scores_return_five_probability_vectors():
    rng = np.random.default_rng(0)
    X_context = rng.integers(0, 5, size=(80, 6)).astype(np.float64)
    y_context = np.tile([0, 1], 40)
    X_eval = rng.integers(0, 5, size=(20, 6)).astype(np.float64)
    timing_context = pd.DataFrame({"last_month": rng.integers(1, 13, 80).astype(float),
                                   "active_months": rng.integers(1, 9, 80).astype(float)})
    timing_eval = pd.DataFrame({"last_month": rng.integers(1, 13, 20).astype(float),
                                "active_months": rng.integers(1, 9, 20).astype(float)})
    scores = pa.baseline_scores(X_context, y_context, timing_context, X_eval, timing_eval)
    assert set(scores) == {"logistic", "xgboost", "total_count", "last_month", "active_months"}
    for values in scores.values():
        assert values.shape == (20,)
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()


def test_export_splits_metadata_round_trips_masks_roles_and_thresholds(tmp_path):
    population = _population([("A", 2008, 0), ("B", 2014, 0), ("C", 2009, 1), ("D", 2015, 0)])
    first = np.array([np.nan, np.nan, 2009.0, 2015.0])
    cohort = pa.landmark_cohort(population, first)
    cohort["role"] = ["tabpfn_context", "evaluation"]
    cohort["is_entrant"] = False
    roles = {"row_role": np.array(["tabpfn_context", "", "evaluation", "evaluation"])}
    context = pa.context_mask(cohort, 2009)
    path = tmp_path / "prospective_splits_and_metadata.npz"
    pa.export_splits_metadata(path, population, first, cohort, {2009: roles}, {2009: context},
                              {2009: 0.3})
    with np.load(path) as saved:
        assert saved["in_cohort"].tolist() == [True, True, False, False]
        assert saved["outcome_valid"].tolist() == [True, False, False, False]
        assert np.isnan(saved["y_next"][1])
        assert saved["y_next"][0] == 0.0
        assert saved["role_2009"].tolist() == roles["row_role"].tolist()
        assert saved["last_3_context_2009"].tolist() == [True, False, False, False]
        assert float(saved["threshold_2009"]) == pytest.approx(0.3)
        assert saved["first_death_year_raw"][2] == 2009.0


CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE_PARAMS = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not CUDA_AVAILABLE, reason="CUDA unavailable on this host; the Fiji run executes this variant")),
]


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_dr_tabpfn_tiny_fit_parity_fd_and_capture_contract(device):
    rng = np.random.default_rng(11)
    years = np.tile([2007, 2008, 2009], 20)
    X = rng.normal(size=(60, 6)).astype(np.float32)
    y = np.tile([0, 1], 30)
    fitted = pa.fit_dr_tabpfn_context(X, y, years, domain_origin=2007, device=device, use_context_scaler=True)
    assert fitted["subsample_rows"] == fitted["context_rows"] == 60
    probe_rows = np.arange(8)
    parity = pa.parity_check(fitted, X[probe_rows], years[probe_rows])
    assert parity["passed"], parity
    captured = pa.capture_scored_rows(fitted, X[probe_rows], years[probe_rows], batch_size=4)
    assert captured["probabilities"].shape == (8,)
    assert ((captured["probabilities"] >= 0) & (captured["probabilities"] <= 1)).all()
    full = probe.capture_flchain_batches(fitted, X[probe_rows], years[probe_rows], batch_size=4)["probabilities"]
    np.testing.assert_allclose(captured["probabilities"], full[:, 1], rtol=0, atol=1e-6)
    assert captured["embeddings"].shape[0] == 8
    gradients, fd_report = pa.gradients_with_fd(captured, grad_batch_size=4, fd_rows=4)
    assert np.isfinite(gradients).all()
    assert fd_report["passed"], fd_report


def test_geometry_summary_flatten_lists_every_section():
    rng = np.random.default_rng(5)
    emb = rng.normal(size=(200, 8))
    grad = np.outer(rng.normal(size=200), rng.normal(size=8))
    prob = np.clip(rng.random(200), 0.05, 0.95)
    sae_embeddings = rng.normal(size=(150, 8))
    result = pa.geometry_summary(emb, grad, prob, sae_embeddings)
    table = pa.flatten_geometry(result)
    assert {"spectrum", "coherence", "dimension", "energy", "random"} <= set(table["section"])
    assert table["value"].map(np.isfinite).any()


def test_load_population_refuses_wrong_sha_before_unpickling(tmp_path):
    path = tmp_path / "prepared.pkl"
    path.write_bytes(b"not a pickle")
    with pytest.raises(Exception, match="sha256"):
        pa.load_population(path, "0" * 64)
    with pytest.raises(ValueError):
        pa.load_population(path, "not-a-digest")
