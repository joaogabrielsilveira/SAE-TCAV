"""Known-answer checks for the planted-change checks of the error profile."""

import numpy as np
import pandas as pd
import pytest

import error_profiles as ep
import error_profile_checks as checks


class FirstFeatureRule:
    def mask(self, features):
        return np.asarray(features)[:, 0] > 0


def synthetic_side(n=120, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"patient_id": [f"p{i}" for i in range(n)], "label": (rng.random(n) < .2).astype(float),
                          "probability": rng.uniform(.01, .6, n).astype(np.float32),
                          "active_months": rng.integers(0, 13, n).astype(float)})
    features = np.column_stack([rng.integers(0, 3, n), rng.normal(size=n)]).astype(np.float32)
    codes = {42: rng.normal(size=(n, 4)).astype(np.float16)}
    concepts = [{"record": {"concept": "c1", "seed": 42, "factor": 2, "cutoff": .5}, "rule": FirstFeatureRule()}]
    return checks.make_side(frame, features, codes, concepts), features, codes, concepts


def bootstrap_for(side, draws=50):
    return ep.patient_bootstrap([side["frame"]["patient_id"].to_numpy()], draws, 3)


def test_tilt_hits_target_mean_and_keeps_death_rate():
    side, *_ = synthetic_side()
    months, labels = side["frame"]["active_months"].to_numpy(), side["frame"]["label"].to_numpy()
    survivors = labels == 0
    for drop in (.5, 1., 2.):
        w, lam = checks.tilt_weights(months, labels, drop)
        assert lam > 0
        assert np.average(months[survivors], weights=w[survivors]) == pytest.approx(months[survivors].mean() - drop, abs=1e-9)
        assert w[survivors].sum() == pytest.approx(survivors.sum())
        np.testing.assert_array_equal(w[~survivors], 1.)
        assert (w @ labels) / w.sum() == pytest.approx(labels.mean())


def test_tilt_never_gives_missing_timing_a_month():
    months, labels = np.array([0., 3., np.nan, 12., 6., 2.]), np.array([0, 0, 0, 0, 1, 0])
    w, _ = checks.tilt_weights(months, labels, 1.)
    assert w[2] == 1. and w[4] == 1.
    known = np.isfinite(months) & (labels == 0)
    assert w[known].sum() == pytest.approx(known.sum())


def test_uniform_shift_is_exactly_undone_by_shift_scores():
    side, *_ = synthetic_side()
    after = checks.shifted_side(side, .8)
    assert np.all(after["frame"]["probability"] > side["frame"]["probability"])
    part = checks.pair(side, after, .3, "x", bootstrap_for(side))
    before_rows = part["frame"]["year"].to_numpy() == 0
    np.testing.assert_allclose(part["shifted"][:, ~before_rows], part["shifted"][:, before_rows], atol=1e-9)
    np.testing.assert_allclose(part["shifted"][0, ~before_rows], side["frame"]["probability"], atol=1e-7)
    moved = checks.what_moved([part]).set_index(["level", "measure"])
    for metric in checks.METRICS:
        assert moved.loc[("after score shift", metric), ["value", "low", "high"]].abs().max() < 1e-7
    assert moved.loc[("what moved", "false_alarm_rate"), "value"] > 0


def test_alignment_returns_rows_and_fails_on_any_mismatch():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"patient_id": ["a", "b", "c", "d"], "year": [2011, 2011, 2012, 2011],
                          "label": [0., 1., 0., 0.], "probability": rng.random(4).astype(np.float32)})
    view = {"frame": frame, "features": rng.normal(size=(4, 3)).astype(np.float32)}
    order = [3, 0, 1]
    table = {"patient_id": np.array(["x", "d", "a", "b"]), "year": np.array([2010, 2011, 2011, 2011]),
             "label": np.r_[0., frame["label"].to_numpy()[order]],
             "probability": np.r_[0., frame["probability"].to_numpy()[order]],
             "features": np.vstack([np.zeros((1, 3)), view["features"][order]])}
    tolerance = {"rtol": 1e-4, "atol": 1e-5}
    np.testing.assert_array_equal(checks.aligned_rows(view, table, np.array([1, 2, 3]), 2011, tolerance), order)
    for key, change in (("probability", .01), ("features", 1.)):
        changed = {**table, key: table[key].copy()}
        changed[key][2] += change
        with pytest.raises(ValueError):
            checks.aligned_rows(view, changed, np.array([1, 2, 3]), 2011, tolerance)
    with pytest.raises(ValueError, match="different patients"):
        checks.aligned_rows(view, {**table, "patient_id": np.array(["x", "d", "a", "z"])}, np.array([1, 2, 3]), 2011, tolerance)


def test_membership_is_recomputed_on_planted_features_and_codes():
    side, features, codes, concepts = synthetic_side()
    np.testing.assert_array_equal(side["members"][("c1", "rule")], features[:, 0] > 0)
    np.testing.assert_array_equal(side["members"][("c1", "factor active")], codes[42][:, 2].astype(np.float32) > .5)
    stopped = features.copy()
    stopped[:, 0] = 0
    quiet = {42: codes[42].copy()}
    quiet[42][:, 2] = 0
    planted = checks.make_side(side["frame"], stopped, quiet, concepts)
    assert planted["members"][("c1", "rule")].sum() == 0 and side["members"][("c1", "rule")].sum() > 0
    assert planted["members"][("c1", "factor active")].sum() == 0


def test_split_halves_reproducible_by_seed_and_disjoint():
    side, *_ = synthetic_side(101)
    a1, b1 = checks.split_halves(side, [7, 0])
    a2, b2 = checks.split_halves(side, [7, 0])
    a3, _ = checks.split_halves(side, [8, 0])
    pd.testing.assert_frame_equal(a1["frame"], a2["frame"])
    pd.testing.assert_frame_equal(b1["frame"], b2["frame"])
    assert not a1["frame"]["patient_id"].equals(a3["frame"]["patient_id"])
    ids = set(a1["frame"]["patient_id"]) | set(b1["frame"]["patient_id"])
    assert len(a1["frame"]) == 50 and len(b1["frame"]) == 51 and len(ids) == 101
    np.testing.assert_array_equal(a1["members"][("c1", "rule")], a1["features"][:, 0] > 0)


def test_flag_coded_scores_and_pooled_counts_match_the_profile_functions():
    side, *_ = synthetic_side()
    after = checks.shifted_side(side, .4)
    part = checks.pair(side, after, .3, "x", bootstrap_for(side))
    frame, member = part["frame"], part["members"][("c1", "rule")]
    real = ep.rate_splits(frame, member, part["weights"], .3, 0, part["risks"])
    coded = ep.rate_splits(checks.flag_frame([part]), member, part["weights"], .5, 0, part["risks"])
    pd.testing.assert_frame_equal(real, coded)
    single = checks.pooled_metrics([part["metrics"][("frozen", 1)]])
    for metric in checks.METRICS:
        np.testing.assert_allclose(single[metric], part["metrics"][("frozen", 1)][metric])
    performance = ep.performance_table(frame, part["weights"], .3, 0, part["shifted"]).query("period == '0→1'")
    moved = checks.what_moved([part])
    for version, level in (("frozen", "what moved"), ("shifted", "after score shift")):
        for metric in checks.METRICS:
            expected = performance.query("version == @version and metric == @metric").iloc[0]
            got = moved.query("level == @level and measure == @metric").iloc[0]
            np.testing.assert_allclose(got[["value", "low", "high"]].astype(float),
                                       expected[["value", "low", "high"]].astype(float), atol=1e-12)


def test_group_rates_match_rule_error_rates():
    side, *_ = synthetic_side()
    part = checks.pair(side, checks.shifted_side(side, .4), .3, "x", bootstrap_for(side))
    member = part["members"][("c1", "rule")]
    rates = checks.group_rates(part["frame"], member, part["weights"], part["risks"])
    table = ep.rule_error_rates(part["frame"], member, part["weights"], .3, 0, part["risks"])
    for row in table[table["period"] != "later"].itertuples():
        draws = rates[(int(row.period), row.group, row.side, row.version)]
        summary = ep.summarize_draws(draws)
        np.testing.assert_allclose([summary["value"], summary["low"], summary["high"]], [row.value, row.low, row.high])


def test_record_length_mix_only_under_tilt_and_zero_mix_under_uniform_shift():
    side, *_ = synthetic_side(300)
    w, _ = checks.tilt_weights(side["frame"]["active_months"], side["frame"]["label"], 1.)
    tilted = checks.pair(side, {**side, "weight": w}, .3, "x", bootstrap_for(side))
    split = checks.record_length([tilted]).set_index("measure")
    assert abs(split.loc["survivors|within", "value"]) < 1e-12
    assert split.loc["survivors|mix", "value"] == pytest.approx(split.loc["survivors|change", "value"])
    assert abs(split.loc["deaths|change", "value"]) < 1e-12
    shifted = checks.pair(side, checks.shifted_side(side, .5), .3, "x", bootstrap_for(side))
    assert np.abs(checks.record_length([shifted]).query("component == 'mix'")[["value", "low", "high"]].to_numpy()).max() < 1e-12


def test_pooled_metrics_over_predictors_with_different_thresholds_sum_their_own_flags():
    a, *_ = synthetic_side(120, seed=1)
    b, *_ = synthetic_side(90, seed=2)
    b = {**b, "frame": b["frame"].assign(patient_id=[f"q{i}" for i in range(90)])}
    boot = ep.patient_bootstrap([a["frame"]["patient_id"].to_numpy(), b["frame"]["patient_id"].to_numpy()], 20, 3)
    parts = [checks.pair(a, checks.shifted_side(a, .5), .2, "a", boot), checks.pair(b, checks.shifted_side(b, .5), .4, "b", boot)]
    moved = checks.what_moved(parts).set_index(["level", "measure"])
    tp = fp = deaths = survivors = 0
    for side, threshold in ((checks.shifted_side(a, .5), .2), (checks.shifted_side(b, .5), .4)):
        y, flag = side["frame"]["label"].to_numpy(), side["frame"]["probability"].to_numpy() >= threshold
        tp, fp = tp + np.sum(flag & (y == 1)), fp + np.sum(flag & (y == 0))
        deaths, survivors = deaths + np.sum(y == 1), survivors + np.sum(y == 0)
    assert moved.loc[("what moved", "recall"), "after"] == pytest.approx(tp / deaths)
    assert moved.loc[("what moved", "false_alarm_rate"), "after"] == pytest.approx(fp / survivors)
    aurocs = [p["metrics"][("frozen", 1)]["auroc"].iloc[0] for p in parts]
    assert moved.loc[("what moved", "auroc"), "after"] == pytest.approx(np.mean(aurocs))


def test_pooled_record_length_strata_keep_predictors_apart():
    a, *_ = synthetic_side(300, seed=4)
    b, *_ = synthetic_side(300, seed=5)
    b = {**b, "frame": b["frame"].assign(patient_id=[f"q{i}" for i in range(300)])}
    boot = ep.patient_bootstrap([a["frame"]["patient_id"].to_numpy(), b["frame"]["patient_id"].to_numpy()], 20, 3)
    parts = []
    for side, threshold, drop, name in ((a, .15, .5, "a"), (b, .45, 2., "b")):
        w, _ = checks.tilt_weights(side["frame"]["active_months"], side["frame"]["label"], drop)
        parts.append(checks.pair(side, {**side, "weight": w}, threshold, name, boot))
    pooled = checks.record_length(parts).set_index("measure")
    assert abs(pooled.loc["survivors|within", "value"]) < 1e-12
    months_only = np.concatenate([ep.record_length_groups(p["frame"]["active_months"], p["frame"]["label"]) for p in parts])
    mixed = ep.rate_splits(checks.flag_frame(parts), months_only, np.hstack([p["weights"] for p in parts]), .5, 0)
    mixed = mixed[(mixed["end"] == "1") & (mixed["side"] == "survivors")].set_index("component")
    assert abs(mixed.loc["within", "value"]) > 1e-4


def test_flag_flips_counts_and_patients_without_the_feature():
    frame = pd.DataFrame({"patient_id": list("abcdef"), "label": [1., 1., 0., 0., 0., 1.],
                          "probability": [.6, .2, .7, .1, .4, .9], "active_months": 1.})
    after = frame.assign(probability=[.2, .6, .1, .1, .5, .9])
    side = lambda f: {"frame": f}
    had = np.array([True, True, True, False, True, False])
    flips = checks.flag_flips(side(frame), side(after), .5, had)
    assert flips == {"deaths_newly_flagged": 1, "deaths_no_longer_flagged": 1, "survivors_newly_flagged": 1,
                     "survivors_no_longer_flagged": 1, "patients_with_feature": 4, "flips_without_feature": 0,
                     "max_score_change_without_feature": 0., "max_score_change_with_feature": pytest.approx(.6)}
    assert set(checks.flag_flips(side(frame), side(after), .5)) == {
        "deaths_newly_flagged", "deaths_no_longer_flagged", "survivors_newly_flagged", "survivors_no_longer_flagged"}


def test_observed_minus_score_alone_on_hand_computed_example():
    # One rule holding everyone; a survivor becomes flagged after, no death changes flag.
    frame = pd.DataFrame({"patient_id": list("abcdabcd"), "label": [1., 1., 0., 0.] * 2, "year": [0] * 4 + [1] * 4,
                          "flagged": [True, False, False, False, True, False, True, False], "active_months": 1.})
    frame["probability"] = frame["flagged"].astype(float)
    risks = np.array([[.5, .5, .2, .2] * 2])
    part = {"predictor": "x", "frame": frame, "weights": np.ones((1, 8)), "risks": risks,
            "members": {("c", "rule"): np.ones(8, bool)}}
    table = checks.concept_changes(part)["concepts"].set_index("measure")
    # Missed-death rate: observed 1/2 -> 1/2; score-alone .9/1.4 -> .7/1.4.
    assert table.loc["rule|inside|deaths|observed", "value"] == pytest.approx(0.)
    assert table.loc["rule|inside|deaths|score_alone", "value"] == pytest.approx(-.2 / 1.4)
    assert table.loc["rule|inside|deaths|observed_minus_score_alone", "value"] == pytest.approx(.2 / 1.4)
    # False-alarm rate: observed 0 -> 1/2; score-alone .5/2.6 -> 1.3/2.6.
    assert table.loc["rule|inside|survivors|observed_minus_score_alone", "value"] == pytest.approx(.5 - .8 / 2.6)
    assert table.loc["rule|inside|deaths|observed", "deaths_in_group_before"] == 2
    assert table.loc["rule|inside|deaths|observed", "source"] == checks.SOURCE["rule"]
