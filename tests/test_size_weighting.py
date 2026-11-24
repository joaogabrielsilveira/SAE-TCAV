"""Size-weighted rank partial correlation (synthetic) and the reproduction anchor (frozen caches)."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from concept_study import associations, input_drift, paths, size_weighting as sw


def _data(seed=0, rows=60):
    rng = np.random.default_rng(seed)
    concepts = np.repeat(np.array(list("abcdef")), rows // 6)
    x = rng.normal(size=rows) + (concepts == "a")
    y = 0.5 * x + rng.normal(size=rows)
    z = rng.normal(size=rows)
    years = rng.integers(0, 4, rows).astype(str)
    return x, y, z, concepts, years


def test_unit_weight_ranks_are_average_ranks_minus_half():
    values = np.array([3.0, 1.0, 3.0, 2.0, 5.0, 1.0])
    np.testing.assert_allclose(sw.weighted_ranks(values, np.ones(6)), rankdata(values) - 0.5)


def test_integer_weight_ranks_equal_repeated_rows():
    values, weights = np.array([2.0, 1.0, 2.0, 4.0]), np.array([1, 3, 2, 1])
    repeated = rankdata(np.repeat(values, weights)) - 0.5
    first = np.concatenate([[0], np.cumsum(weights)[:-1]])
    np.testing.assert_allclose(sw.weighted_ranks(values, weights.astype(float)), repeated[first])


def test_unit_weights_reproduce_frozen_partial_rank():
    x, y, z, concepts, years = _data()
    for numeric, indicators in (([], []), ([z], []), ([], [concepts]), ([z], [concepts, years])):
        expected = associations._partial_rank(x, y, numeric, indicators)
        assert sw.weighted_partial_rank(x, y, np.ones_like(x), numeric, indicators) == pytest.approx(expected, abs=1e-12)


def test_integer_weights_equal_unweighted_on_repeated_rows():
    x, y, z, concepts, years = _data(1)
    w = np.random.default_rng(1).integers(1, 5, len(x))
    weighted = sw.weighted_partial_rank(x, y, w.astype(float), [z], [concepts])
    rep = lambda v: np.repeat(v, w)
    plain = associations._partial_rank(rep(x), rep(y), [rep(z)], [rep(concepts)])
    assert weighted == pytest.approx(plain, abs=1e-12)


def test_weighted_correlation_without_controls_is_weighted_pearson_of_weighted_ranks():
    x, y, *_ = _data(2)
    w = np.random.default_rng(2).uniform(0.1, 3, len(x))
    rx, ry = sw.weighted_ranks(x, w), sw.weighted_ranks(y, w)
    c = np.cov(np.vstack([rx, ry]), aweights=w)
    expected = c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])
    assert sw.weighted_partial_rank(x, y, w) == pytest.approx(expected, abs=1e-12)


def test_weights_scale_free_and_zero_weight_drops_the_row():
    x, y, z, concepts, _ = _data(3)
    w = np.random.default_rng(3).uniform(0.1, 3, len(x))
    base = sw.weighted_partial_rank(x, y, w, [z], [concepts])
    assert sw.weighted_partial_rank(x, y, 7 * w, [z], [concepts]) == pytest.approx(base, abs=1e-12)
    w0 = w.copy()
    w0[5] = 0
    keep = np.arange(len(x)) != 5
    assert sw.weighted_partial_rank(x, y, w0, [z], [concepts]) == pytest.approx(
        sw.weighted_partial_rank(x[keep], y[keep], w[keep], [z[keep]], [concepts[keep]]), abs=1e-12)


def test_change_weight():
    np.testing.assert_allclose(sw.change_weight([10, 4, 0], [10, 12, 5]), [5.0, 3.0, 0.0])


# ---------------------------------------------------------------- anchor on the frozen caches

needs_caches = pytest.mark.skipif(not (paths.OUTPUT / "answers.csv").exists(), reason="frozen study outputs absent")


@needs_caches
def test_unweighted_all_cells_reproduce_frozen_point_estimates():
    records, point, cube = sw.load_panel()
    frozen = sw.frozen_table().set_index(["estimate", "variant"])
    for task in sw.correlation_cells(point, {k: v[:1] for k, v in cube.items()}):   # point estimate only
        function, args, _ = task
        if args[5] != "none" or args[6] != 1:
            continue
        row = function(*args)
        expected = frozen.loc[(row["estimate"], row["variant"])]
        assert row["rows"] == expected.frozen_rows
        assert row["r"] == pytest.approx(expected.frozen_r, abs=1e-10)


@needs_caches
def test_unweighted_all_cells_reproduce_frozen_step_and_drift():
    records, point, cube = sw.load_panel()
    frozen = sw.frozen_table().set_index(["estimate", "variant"])
    steps = sw.step_table(records, point, cube)
    for row in steps[(steps.weighting == "none") & (steps.min_n == 1)].itertuples():
        expected = frozen.loc[(row.estimate, "last_3")]
        assert (row.r, row.low, row.high) == pytest.approx(       # cached values are float32
            (expected.frozen_r, expected.frozen_low, expected.frozen_high), abs=1e-7)
    frame = sw.drift_frame()
    for variant in paths.VARIANTS:
        k = frame[input_drift.in_variant(frame, variant)].dropna(subset=["rule_feature_psi"])
        assert sw.drift_statistic(k, weighted=False) == pytest.approx(frozen.loc[(sw.DRIFT, variant)].frozen_r, abs=1e-10)
