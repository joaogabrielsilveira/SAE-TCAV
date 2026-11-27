"""Unit tests for calibration_and_layer_diagnostics.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import calibration_and_layer_diagnostics as diag
import prospective_audit as pa


def test_newton_recalibration_parity_and_separation_guard():
    rng = np.random.default_rng(123)
    logits = rng.normal(-1.5, 0.8, size=180)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, np.clip(1.0 / (1.0 + np.exp(-(0.2 + 0.9 * logits))), 0.01, 0.99))
    w = rng.integers(1, 4, size=180).astype(float)
    w[0] = 0.0

    from sklearn.linear_model import LogisticRegression
    clipped = np.clip(probs, diag.PROB_CLIP_EPS, 1.0 - diag.PROB_CLIP_EPS)
    x_logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    tight = LogisticRegression(penalty=None, tol=1e-12, max_iter=5000).fit(x_logit, y, sample_weight=w)

    sk_slope, sk_int = pa.recalibration_slope_intercept(y, probs, w)
    nr_slope, nr_int, status = diag.newton_recalibration_slope_intercept(y, probs, w)
    assert status == "ok"
    assert np.isclose(nr_slope, float(tight.coef_[0, 0]), atol=1e-6)
    assert np.isclose(nr_int, float(tight.intercept_[0]), atol=1e-6)
    assert np.isclose(nr_slope, sk_slope, atol=5e-4)
    assert np.isclose(nr_int, sk_int, atol=5e-4)

    # Single class -> NaN + single_class
    y_one = np.zeros(20, dtype=int)
    s_sc, i_sc, st_sc = diag.newton_recalibration_slope_intercept(y_one, probs[:20], np.ones(20))
    assert np.isnan(s_sc) and np.isnan(i_sc) and st_sc == "single_class"

    # Complete separation -> NaN + separated
    p_sep = np.linspace(0.05, 0.95, 20)
    y_sep = (p_sep > 0.5).astype(int)
    s_sep, i_sep, st_sep = diag.newton_recalibration_slope_intercept(y_sep, p_sep, np.ones(20))
    assert np.isnan(s_sep) and np.isnan(i_sep) and st_sep == "separated"

    # Inverted separation (auc == 0.0) -> NaN + separated
    s_inv, i_inv, st_inv = diag.newton_recalibration_slope_intercept(1 - y_sep, p_sep, np.ones(20))
    assert np.isnan(s_inv) and np.isnan(i_inv) and st_inv == "separated"


def test_calibration_metrics_and_paired_bootstrap_and_base_rate():
    rng = np.random.default_rng(42)
    n = 120
    pids = np.array([f"P{i // 2:04d}" for i in range(n)])
    logits = rng.normal(-2.0, 0.9, size=n)
    pa_arr = 1.0 / (1.0 + np.exp(-logits))
    ph_arr = 1.0 / (1.0 + np.exp(-(logits - 0.35)))
    y = rng.binomial(1, pa_arr)
    # Include an extreme-clipped probability on a nonzero weight test separately
    base_rate = 0.08
    m_act = diag.calibration_metrics(y, pa_arr, np.ones(n), base_rate=base_rate, threshold=0.25)
    m_held = diag.calibration_metrics(y, ph_arr, np.ones(n), base_rate=base_rate, threshold=0.25)

    exp_brier = np.mean((pa_arr - y) ** 2)
    exp_brier_ref = np.mean((base_rate - y) ** 2)
    assert np.isclose(m_act["brier"], exp_brier)
    assert np.isclose(m_act["brier_skill"], 1.0 - exp_brier / exp_brier_ref)
    assert np.isclose(m_act["calibration_in_the_large"], pa_arr.mean() - y.mean())
    assert m_act["clipped_rows"] == 0
    assert m_act["recalibration_status"] == "ok"

    # Clipped row tracking
    pa_clip = pa_arr.copy()
    pa_clip[0] = 0.0
    m_clip = diag.calibration_metrics(y, pa_clip, np.ones(n), base_rate=base_rate, threshold=None)
    assert m_clip["clipped_rows"] == 1
    assert "recall" not in m_clip

    boot = diag.paired_calibration_bootstrap(
        y, pa_arr, ph_arr, pids, base_rate=base_rate, threshold=0.25, draws=40, seed=20261106
    )
    citl_diff = boot[(boot["quantity"] == "actual_minus_held") & (boot["metric"] == "calibration_in_the_large")].iloc[0]
    mp_diff = boot[(boot["quantity"] == "actual_minus_held") & (boot["metric"] == "mean_probability")].iloc[0]
    assert np.isclose(citl_diff["estimate"], mp_diff["estimate"])
    assert np.isclose(citl_diff["ci_low"], mp_diff["ci_low"])
    assert np.isclose(citl_diff["ci_high"], mp_diff["ci_high"])
    assert np.isclose(citl_diff["estimate"], m_act["mean_probability"] - m_held["mean_probability"])


def test_reconcile_prospective_scope_a_calibration():
    perf_df = pd.DataFrame([
        {"reference_year": 2008, "scaling": "consistent", "input_year": 2008, "stratum": "all",
         "metric": "rows", "estimate": 100.0, "ci_low": 90.0, "ci_high": 110.0, "valid_draws": 500},
        {"reference_year": 2008, "scaling": "consistent", "input_year": 2008, "stratum": "all",
         "metric": "deaths", "estimate": 10.0, "ci_low": 5.0, "ci_high": 16.0, "valid_draws": 500},
        {"reference_year": 2008, "scaling": "consistent", "input_year": 2008, "stratum": "all",
         "metric": "mean_probability", "estimate": 0.09, "ci_low": 0.07, "ci_high": 0.11, "valid_draws": 500},
        {"reference_year": 2008, "scaling": "consistent", "input_year": "later_inputs", "stratum": "all",
         "metric": "rows", "estimate": 200.0, "ci_low": 180.0, "ci_high": 220.0, "valid_draws": 500},
        {"reference_year": 2008, "scaling": "consistent", "input_year": "later_inputs", "stratum": "all",
         "metric": "deaths", "estimate": 20.0, "ci_low": 12.0, "ci_high": 28.0, "valid_draws": 500},
        {"reference_year": 2008, "scaling": "consistent", "input_year": "later_inputs", "stratum": "all",
         "metric": "mean_probability", "estimate": 0.14, "ci_low": 0.12, "ci_high": 0.16, "valid_draws": 500},
        {"reference_year": 2008, "scaling": "consistent", "input_year": "later_inputs", "stratum": "all",
         "metric": "brier", "estimate": 0.080, "ci_low": 0.070, "ci_high": 0.090, "valid_draws": 500},
    ])
    sens_df = pd.DataFrame([
        # Wrong comparison that must be excluded
        {"reference_year": 2008, "scaling": "consistent", "comparison": "input_year_minus_reference_year",
         "input_year": "later_inputs", "stratum": "all", "metric": "mean_probability",
         "estimate": 0.999, "ci_low": 0.9, "ci_high": 1.0, "valid_draws": 500},
        {"reference_year": 2008, "scaling": "consistent", "comparison": "actual_minus_held_at_reference",
         "input_year": 2008, "stratum": "all", "metric": "mean_probability",
         "estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "valid_draws": 500},
        {"reference_year": 2008, "scaling": "consistent", "comparison": "actual_minus_held_at_reference",
         "input_year": "later_inputs", "stratum": "all", "metric": "mean_probability",
         "estimate": 0.04, "ci_low": 0.03, "ci_high": 0.05, "valid_draws": 500},
        {"reference_year": 2008, "scaling": "consistent", "comparison": "actual_minus_held_at_reference",
         "input_year": "later_inputs", "stratum": "all", "metric": "brier",
         "estimate": -0.002, "ci_low": -0.005, "ci_high": 0.001, "valid_draws": 500},
    ])
    out = diag.reconcile_prospective_scope_a_calibration(perf_df, sens_df)
    brier_held = out[(out["input_year"] == "later_inputs") & (out["quantity"] == "held") & (out["metric"] == "brier")].iloc[0]
    assert np.isclose(brier_held["estimate"], 0.082)
    assert np.isnan(brier_held["ci_low"]) and brier_held["status"] == "point_only_recovered_from_csv"

    citl_lvl = out[(out["input_year"] == "later_inputs") & (out["quantity"] == "actual") & (out["metric"] == "calibration_in_the_large")].iloc[0]
    assert np.isclose(citl_lvl["estimate"], 0.14 - 0.10)
    assert citl_lvl["status"] == "level_ci_unavailable_no_joint_draws"

    citl_diff = out[(out["input_year"] == "later_inputs") & (out["quantity"] == "actual_minus_held") & (out["metric"] == "calibration_in_the_large")].iloc[0]
    assert np.isclose(citl_diff["estimate"], 0.04) and np.isclose(citl_diff["ci_low"], 0.03) and citl_diff["status"] == "ok"

    ll = out[(out["input_year"] == "later_inputs") & (out["quantity"] == "actual") & (out["metric"] == "log_loss")].iloc[0]
    assert np.isnan(ll["estimate"]) and ll["status"] == "unavailable_no_per_row_probs"

    ref_rows = out[out["input_year"].astype(str) == "2008"]
    assert ref_rows["is_reference_year_contrast"].all()


def test_natural_spline_and_partial_pc1_planted_linear_vs_nonlinear():
    rng = np.random.default_rng(77)
    n = 450
    years = np.repeat([2011, 2012, 2013], n // 3)
    pids = np.array([f"P{i:04d}" for i in range(n)])
    # Year shifts logit monotonically, and PC1 is quadratic in logit + tiny noise (no separate year effect)
    logit = (years - 2012) * 1.4 + rng.normal(0.0, 0.25, size=n)
    probs = 1.0 / (1.0 + np.exp(-logit))
    emb = np.zeros((n, 6), dtype=np.float64)
    emb[:, 0] = 3.0 * (logit ** 2) + rng.normal(0.0, 0.2, size=n)
    emb[:, 1:] = rng.normal(0.0, 0.05, size=(n, 5))

    B = diag.natural_cubic_spline_basis(logit, df=4)
    assert B.shape == (n, 3)

    res = diag.partial_pc1_year_analysis(
        emb, years, probs, pids, spline_df=4, draws=30, seed=20261106, run_refit_sensitivity=False
    )
    fixed = res[res["mode"] == "fixed_pc1"].set_index("metric")
    assert fixed.loc["r2_logit_spline", "estimate"] > fixed.loc["r2_logit_linear", "estimate"] + 0.2
    assert fixed.loc["delta_r2_year_cat_given_logit_spline", "estimate"] < 0.05
    assert np.isclose(
        fixed.loc["residual_var_after_logit_spline", "estimate"],
        1.0 - fixed.loc["r2_logit_spline", "estimate"],
        atol=1e-12,
    )
    assert fixed.loc["eta_squared_year_cat", "estimate"] >= 0.0


def test_partial_pc1_weighted_refit_alignment_and_drift():
    rng = np.random.default_rng(99)
    n = 180
    years = np.repeat([2011, 2012, 2013], n // 3)
    pids = np.array([f"P{i:04d}" for i in range(n)])
    # Near-degenerate PC1 and PC2 eigenvalues -> bootstrap flips leading eigenvector -> high axis_drift_share
    emb_deg = np.zeros((n, 5), dtype=np.float64)
    emb_deg[:, 0] = rng.normal(0.0, 1.01, size=n)
    emb_deg[:, 1] = rng.normal(0.0, 1.00, size=n)
    probs = 1.0 / (1.0 + np.exp(-((years - 2012) * 0.5 + 0.5 * emb_deg[:, 0])))

    res_deg = diag.partial_pc1_year_analysis(
        emb_deg, years, probs, pids, draws=35, seed=11, run_refit_sensitivity=True, min_axis_cos=0.9
    )
    drift_row = res_deg[(res_deg["mode"] == "refit_pc1_aligned") & (res_deg["metric"] == "axis_drift_share")].iloc[0]
    assert drift_row["estimate"] > 0.1

    # Dominant PC1 -> zero drift
    emb_dom = np.zeros((n, 5), dtype=np.float64)
    emb_dom[:, 0] = (years - 2012) * 5.0 + rng.normal(0.0, 0.2, size=n)
    emb_dom[:, 1:] = rng.normal(0.0, 0.05, size=(n, 4))
    res_dom = diag.partial_pc1_year_analysis(
        emb_dom, years, probs, pids, draws=30, seed=11, run_refit_sensitivity=True, min_axis_cos=0.9
    )
    drift_dom = res_dom[(res_dom["mode"] == "refit_pc1_aligned") & (res_dom["metric"] == "axis_drift_share")].iloc[0]
    assert drift_dom["estimate"] == 0.0


def test_continuous_tcav_exact_rng_parity_and_equal_n():
    import flchain_probe as probe

    rng = np.random.default_rng(101)
    n, d = 90, 16
    emb = rng.normal(0.0, 1.0, size=(n, d))
    scale = rng.uniform(0.5, 2.0, size=d)
    # Near-1D aligned gradients so saturation is strong
    g_base = np.ones(d) / np.sqrt(d)
    grad = g_base[None, :] + rng.normal(0.0, 0.05, size=(n, d))
    probs = rng.uniform(0.05, 0.45, size=n)

    ref_geom = probe.analyze_layer_geometry(
        emb, grad, probs, scale,
        rng=np.random.default_rng(diag.DIRECTION_SEED),
        n_random=120, top_axes=5, interior=(0.1, 0.9),
    )
    df = diag.continuous_tcav_saturation_for_layer(
        emb, grad, scale, direction_count=120, direction_seed=diag.DIRECTION_SEED, top_axes=5
    )
    for _, row in df.iterrows():
        ref = ref_geom["random"][(row["chart"], row["family"])]
        assert row["interior_share"] == ref["interior_share"]
        assert row["interior_se"] == ref["interior_se"]
        assert 0.0 <= row["mean_tcav_binary_entropy_nats"] <= np.log(2.0)
        assert row["mean_tcav_binary_entropy_se"] >= 0.0
        assert 0.0 <= row["mean_abs_dev_from_half"] <= 0.5
        assert row["exact_endpoint_share"] == ref["exact_binary_share"]

    sub = diag.continuous_tcav_equal_n_subsamples(
        emb, grad, scale, subsample_n=46, repeats=3, subsample_seed=20261127,
        direction_count=60, direction_seed=diag.DIRECTION_SEED, top_axes=5,
    )
    assert len(sub) == 6
    assert (sub["subsample_n"] == 46).all()
    assert (sub["repeats"] == 3).all()


def test_analytical_null_resultant_matches_monte_carlo():
    rng = np.random.default_rng(2026)
    for dim, n_rows in ((2, 46), (192, 46), (192, 200)):
        theo = diag.analytical_null_resultant(n_rows, dimension=dim)
        # Draw UNIT_TEST_NULL_DRAWS = 2000 batches of n_rows isotropic unit vectors
        raw = rng.standard_normal((2000, n_rows, dim))
        unit = raw / np.linalg.norm(raw, axis=2, keepdims=True)
        mean_u = unit.mean(axis=1)
        norms = np.linalg.norm(mean_u, axis=1)
        emp_rms = float(np.sqrt(np.mean(norms ** 2)))
        emp_mean = float(np.mean(norms))
        assert np.isclose(emp_rms, theo["rms_floor"], rtol=0.04)
        assert np.isclose(emp_mean, theo["clt_expected_floor"], rtol=0.04)


def test_non_nested_dispersion_summary_guards():
    df_good = pd.DataFrame([
        {"dataset": "sus", "scaling_scheme": "consistent", "layer": 11, "chart": "raw",
         "stratum": "2011", "probability_resultant_length": 0.96, "prob_std": 0.05,
         "prob_iqr": 0.04, "logit_std": 0.4, "logit_iqr": 0.3, "prevalence": 0.07, "status": "ok"},
        {"dataset": "sus", "scaling_scheme": "consistent", "layer": 11, "chart": "raw",
         "stratum": "2012", "probability_resultant_length": 0.98, "prob_std": 0.03,
         "prob_iqr": 0.02, "logit_std": 0.25, "logit_iqr": 0.2, "prevalence": 0.08, "status": "ok"},
        {"dataset": "sus", "scaling_scheme": "consistent", "layer": 11, "chart": "raw",
         "stratum": "2013", "probability_resultant_length": 0.99, "prob_std": 0.02,
         "prob_iqr": 0.015, "logit_std": 0.15, "logit_iqr": 0.1, "prevalence": 0.09, "status": "ok"},
    ])
    summary = diag.non_nested_dispersion_summary(df_good)
    assert "p_value" not in summary.columns
    row_ps = summary[summary["covariate"] == "prob_std"].iloc[0]
    assert np.isclose(row_ps["spearman_rho"], -1.0)
    assert row_ps["n_strata"] == 3
    row_prev = summary[summary["covariate"] == "prevalence"].iloc[0]
    assert row_prev["covariate_role"] == "descriptive_covariate"

    # Reject duplicate/nested stratum in same group
    df_dup = pd.concat([df_good, df_good.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        diag.non_nested_dispersion_summary(df_dup)


def test_input_feature_spectrum_perm_null_and_ref_sensitivity():
    rng = np.random.default_rng(55)
    n, p = 150, 12
    z = rng.normal(0.0, 1.0, size=n)
    X_eval = np.column_stack([z + rng.normal(0.0, 0.2, size=n) for _ in range(p)])
    X_ref = rng.normal(0.0, 1.0, size=(80, p))
    # Zero variance in ref col 0 -> must be dropped only in ref-standardized sensitivity, with share reported
    X_ref[:, 0] = 0.0
    # Extreme low ref variance in col 1 -> high max_eval_to_ref_var_ratio
    X_ref[:, 1] *= 0.05

    spec = diag.input_feature_spectrum(X_eval, X_ref_scale=X_ref, log1p=False, perm_draws=50, perm_seed=20261126)
    assert spec["active_columns"] == p
    assert spec["ref_active_columns"] == p - 1
    assert spec["corr_matrix_pr"] < spec["perm_null_pr_mean"]
    assert spec["max_eval_to_ref_var_ratio"] > 50.0
    assert 0.0 < spec["dropped_eval_var_share"] < 1.0

    # Column-wise permutation preserves marginals so independent columns approach null PR
    X_indep = rng.normal(0.0, 1.0, size=(150, p))
    spec_indep = diag.input_feature_spectrum(X_indep, perm_draws=50, perm_seed=20261126)
    assert abs(spec_indep["corr_matrix_pr"] - spec_indep["perm_null_pr_mean"]) < 1.5

    # log1p rejects negatives
    with pytest.raises(ValueError):
        diag.input_feature_spectrum(X_eval, log1p=True)


def test_multiplicity_summary_and_sus_notebook_scanner(tmp_path):
    import nbformat

    t1 = pd.DataFrame([
        {"is_reference_year_contrast": True, "ci_low": 0.0, "ci_high": 0.0},
        {"is_reference_year_contrast": False, "ci_low": 0.01, "ci_high": 0.05},
        {"is_reference_year_contrast": False, "ci_low": -0.02, "ci_high": 0.03},
    ])
    mult = diag.multiplicity_summary({"work1": t1}, alpha=0.05)
    row_w1 = mult[mult["table"] == "work1"].iloc[0]
    assert row_w1["total_intervals"] == 2
    assert row_w1["excluding_zero"] == 1
    assert np.isclose(row_w1["chance_expected_at_alpha"], 0.10)

    # Leakage scanner catches patient_id in output and prohibited wording
    nb = nbformat.v4.new_notebook()
    c_clean = nbformat.v4.new_code_cell("print('aggregate mean = 0.08')")
    c_clean.outputs = [nbformat.v4.new_output("stream", name="stdout", text="aggregate mean = 0.08\n")]
    nb.cells = [nbformat.v4.new_markdown_cell("Descriptive summary."), c_clean]
    clean_path = tmp_path / "clean.ipynb"
    nbformat.write(nb, clean_path)
    chk_clean = diag.scan_notebook_for_sus_row_leakage(clean_path)
    assert chk_clean["passed"].all()

    c_bad = nbformat.v4.new_code_cell("print('patient_id')")
    c_bad.outputs = [nbformat.v4.new_output("stream", name="stdout", text="patient_id\n123456\n")]
    nb.cells.append(nbformat.v4.new_markdown_cell("Not " + "caus" + "al phrasing here."))
    nb.cells.append(c_bad)
    bad_path = tmp_path / "bad.ipynb"
    nbformat.write(nb, bad_path)
    chk_bad = diag.scan_notebook_for_sus_row_leakage(bad_path)
    assert not chk_bad["passed"].all()

