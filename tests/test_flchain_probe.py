import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import flchain_probe as probe


def _synthetic_flchain_df(rows_per_year: int = 24, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    idx = 1
    for year in range(1995, 2004):
        for i in range(rows_per_year):
            death = int((i + year) % 3 == 0)
            if year >= 2002 and not death and i % 5 == 0:
                futime = 400
            elif death:
                futime = 300 if i % 2 == 0 else 1500
            else:
                futime = 2200
            records.append({
                "Unnamed: 0": idx,
                "age": float(50 + rng.integers(0, 35)),
                "sex": "M" if i % 2 == 0 else "F",
                "sample.yr": year,
                "kappa": float(1.0 + rng.random()),
                "lambda": float(1.2 + rng.random()),
                "flc.grp": float(1 + (i % 10)),
                "creatinine": np.nan if i % 5 == 0 else float(0.8 + 0.3 * rng.random()),
                "mgus": float(1 if i % 11 == 0 else 0),
                "futime": int(futime),
                "death": death,
                "chapter": "Circulatory" if death else np.nan,
            })
            idx += 1
    return pd.DataFrame(records)


def test_load_and_feature_matrix_excludes_leakage_and_preserves_nan(tmp_path):
    raw = _synthetic_flchain_df(rows_per_year=10)
    csv_path = tmp_path / "flchain.csv"
    raw.to_csv(csv_path, index=False)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    loaded = probe.load_flchain_dataframe(csv_path, expected_sha256=digest)
    assert len(loaded) == len(raw)
    assert np.isnan(loaded["creatinine"]).sum() == np.isnan(raw["creatinine"]).sum()
    np.testing.assert_array_equal(loaded["sex_male"].to_numpy(), (raw["sex"] == "M").astype(np.float32))

    with pytest.raises(ValueError):
        probe.load_flchain_dataframe(csv_path, expected_sha256="0" * 64)

    X = probe.extract_feature_matrix(loaded)
    assert X.shape == (len(raw), 7)
    assert X.dtype == np.float32
    assert X.flags["C_CONTIGUOUS"]
    assert np.isnan(X[:, probe.FLCHAIN_FEATURES.index("creatinine")]).sum() == np.isnan(raw["creatinine"]).sum()

    for forbidden in ("futime", "chapter", "death", "sample.yr"):
        bad = list(probe.FLCHAIN_FEATURES[:-1]) + [forbidden]
        with pytest.raises(ValueError):
            probe.extract_feature_matrix(loaded, tuple(bad))


def test_build_target_censoring_and_horizon():
    df = pd.DataFrame({
        "Unnamed: 0": [1, 2, 3, 4, 5],
        "age": [60, 65, 70, 75, 80],
        "sex": ["F", "M", "F", "M", "F"],
        "sample.yr": [1997, 1997, 1997, 1997, 1997],
        "kappa": [1.0, 1.1, 1.2, 1.3, 1.4],
        "lambda": [1.0, 1.1, 1.2, 1.3, 1.4],
        "flc.grp": [1, 2, 3, 4, 5],
        "creatinine": [1.0, np.nan, 1.1, 1.2, 0.9],
        "mgus": [0, 0, 1, 0, 0],
        "futime": [500, 1095, 1500, 900, 1095],
        "death": [1, 1, 1, 0, 0],
        "chapter": ["Other", "Other", "Other", np.nan, np.nan],
    })
    df["subject_id"] = df["Unnamed: 0"]
    df["sex_male"] = (df["sex"] == "M").astype(np.float32)

    kept_3y, y_3y = probe.build_flchain_target(df, 1095)
    np.testing.assert_array_equal(kept_3y["subject_id"].to_numpy(), [1, 2, 3, 5])
    np.testing.assert_array_equal(y_3y, [1, 1, 0, 0])

    kept_raw, y_raw = probe.build_flchain_target(df, None)
    assert len(kept_raw) == 5
    np.testing.assert_array_equal(y_raw, [1, 1, 1, 0, 0])


def test_split_chronological_disjoint_deterministic_and_stratified(tmp_path):
    raw = _synthetic_flchain_df(rows_per_year=24)
    csv_path = tmp_path / "flchain.csv"
    raw.to_csv(csv_path, index=False)
    df = probe.load_flchain_dataframe(csv_path, expected_sha256=None)
    kept, y = probe.build_flchain_target(df, probe.PRIMARY_HORIZON_DAYS)

    split_a = probe.split_flchain_chronological(kept, y)
    split_b = probe.split_flchain_chronological(kept, y)
    for key in ("context_idx", "ref_fit_idx", "ref_eval_idx", "future_idx"):
        np.testing.assert_array_equal(split_a[key], split_b[key])

    parts = [set(split_a[k].tolist()) for k in ("context_idx", "ref_fit_idx", "ref_eval_idx", "future_idx")]
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            assert not (parts[i] & parts[j])
    assert sum(len(p) for p in parts) == len(kept)

    prior_rows = set(np.flatnonzero(kept["sample.yr"].to_numpy() < 1997).tolist())
    assert prior_rows <= parts[0]
    ref_total = int((kept["sample.yr"] == 1997).sum())
    assert len(split_a["ref_fit_idx"]) + len(split_a["ref_eval_idx"]) == ref_total - (len(split_a["context_idx"]) - len(prior_rows))
    assert abs(len(split_a["ref_fit_idx"]) - len(split_a["ref_eval_idx"])) <= 1
    assert set(["1997_eval", "1998", "1999", "2000", "2001", "2002-2003", "1998-2001", "1998-2003"]) <= set(split_a["strata"])


def test_censoring_audit_table_counts(tmp_path):
    raw = _synthetic_flchain_df(rows_per_year=15)
    csv_path = tmp_path / "flchain.csv"
    raw.to_csv(csv_path, index=False)
    df = probe.load_flchain_dataframe(csv_path, expected_sha256=None)
    audit = probe.censoring_audit_table(df, horizons=(1095, 1825))
    pooled = audit[(audit["stratum"] == "2002-2003") & (audit["horizon_days"] == 1095)].iloc[0]
    expected_censored = int(
        (raw["sample.yr"].isin([2002, 2003]) & (raw["death"] == 0) & (raw["futime"] < 1095)).sum()
    )
    assert pooled["total_rows"] == 30
    assert pooled["censored_rows"] == expected_censored
    assert pooled["kept_rows"] == 30 - expected_censored
    raw_row = audit[(audit["stratum"] == "all") & (audit["target"] == "raw_death")].iloc[0]
    assert raw_row["censored_rows"] == 0
    assert raw_row["kept_rows"] == len(df)


def test_binary_entropy_nats_edges_and_validation():
    ent = probe.binary_entropy_nats(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(ent, [0.0, np.log(2.0), 0.0], atol=1e-12)
    for bad in ([-0.1, 0.5], [0.5, 1.1], [np.nan, 0.2]):
        with pytest.raises(ValueError):
            probe.binary_entropy_nats(np.array(bad))


def test_paired_bootstrap_sensitivity_identical_and_shifted():
    rng = np.random.default_rng(11)
    y = np.array([0, 1] * 25)
    p_held = np.clip(0.15 + 0.35 * y + rng.normal(0, 0.05, len(y)), 0.01, 0.99)
    identical = probe.paired_bootstrap_sensitivity(p_held, p_held.copy(), y, draws=200, seed=5)
    assert identical["mean_entropy_diff"]["estimate"] == 0.0
    assert identical["mean_entropy_diff"]["ci_low"] == 0.0
    assert identical["mean_entropy_diff"]["ci_high"] == 0.0
    assert identical["auroc_diff"]["estimate"] == 0.0

    p_actual = np.full_like(p_held, 0.5)
    shifted = probe.paired_bootstrap_sensitivity(p_actual, p_held, y, draws=200, seed=5)
    assert shifted["mean_entropy_diff"]["ci_low"] > 0.0

    single_class = probe.paired_bootstrap_sensitivity(p_actual, p_held, np.zeros_like(y), draws=50, seed=5)
    assert np.isnan(single_class["auroc_diff"]["estimate"])


def test_analyze_layer_geometry_known_rank_one_gradient():
    rng = np.random.default_rng(12)
    embeddings = np.zeros((60, 8), dtype=np.float32)
    embeddings[:, 0] = rng.normal(0.0, 3.0, 60)
    embeddings[:, 1] = rng.normal(0.0, 1.0, 60)
    gradients = np.tile(np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32), (60, 1))
    probabilities = np.full(60, 0.3, dtype=np.float32)
    ref_fit_emb = embeddings.copy()
    ref_fit_emb[:, 7] = 0.0
    scale, zero_cols = probe.standardization_scale(ref_fit_emb)
    assert zero_cols >= 1
    assert scale[7] == 1.0

    out = probe.analyze_layer_geometry(
        embeddings, gradients, probabilities, scale, rng=np.random.default_rng(9), n_random=200, top_axes=2
    )
    np.testing.assert_allclose(out["coherence"][("raw", "probability")]["resultant_length"], 1.0)
    np.testing.assert_allclose(out["dimension"][("raw", "probability")]["uncentered_pr"], 1.0)
    assert out["random"][("raw", "isotropic")]["exact_binary_share"] == 1.0
    assert out["random"][("raw", "isotropic")]["interior_share"] == 0.0


def test_cpu_baselines_handle_nan_and_return_metrics(tmp_path):
    raw = _synthetic_flchain_df(rows_per_year=30)
    csv_path = tmp_path / "flchain.csv"
    raw.to_csv(csv_path, index=False)
    df = probe.load_flchain_dataframe(csv_path, expected_sha256=None)
    kept, y = probe.build_flchain_target(df, probe.PRIMARY_HORIZON_DAYS)
    split = probe.split_flchain_chronological(kept, y)
    X = probe.extract_feature_matrix(kept)
    ctx = split["context_idx"]
    eval_splits = {name: (X[idx], y[idx]) for name, idx in split["strata"].items() if len(np.unique(y[idx])) == 2}
    table = probe.fit_and_score_cpu_baselines(X[ctx], y[ctx], eval_splits, draws=50)
    assert set(table["model"]) == {"LogisticRegression", "HistGradientBoosting"}
    assert np.isfinite(table["auroc"]).all()
    assert np.isfinite(table["brier"]).all()


def test_real_csv_loader_and_split_counts_if_present():
    csv_path = Path("data/flchain/free_light_chain_mortality.csv")
    if not csv_path.is_file():
        pytest.skip("FLChain CSV not present in local data/flchain/")
    df = probe.load_flchain_dataframe(csv_path)
    assert df.shape[0] == 7874
    kept_3y, y_3y = probe.build_flchain_target(df, 1095)
    split_3y = probe.split_flchain_chronological(kept_3y, y_3y)
    assert len(split_3y["context_idx"]) == 5390
    assert len(split_3y["ref_fit_idx"]) == 340
    assert len(split_3y["ref_eval_idx"]) == 340
    assert len(split_3y["future_idx"]) == 1673

    kept_5y, y_5y = probe.build_flchain_target(df, 1825)
    split_5y = probe.split_flchain_chronological(kept_5y, y_5y)
    assert len(split_5y["context_idx"]) == 5365
    assert len(split_5y["ref_fit_idx"]) == 337
    assert len(split_5y["ref_eval_idx"]) == 338
    assert len(split_5y["future_idx"]) == 1639


def test_dr_tabpfn_synthetic_cpu_fit_parity_and_gradients(tmp_path):
    raw = _synthetic_flchain_df(rows_per_year=24, seed=99)
    csv_path = tmp_path / "flchain.csv"
    raw.to_csv(csv_path, index=False)
    df = probe.load_flchain_dataframe(csv_path, expected_sha256=None)
    kept, y = probe.build_flchain_target(df, probe.PRIMARY_HORIZON_DAYS)
    split = probe.split_flchain_chronological(kept, y)
    X = probe.extract_feature_matrix(kept)
    years = kept["sample.yr"].to_numpy(dtype=np.int64)

    ctx = split["context_idx"]
    fitted = probe.fit_dr_tabpfn_flchain(X[ctx], y[ctx], years[ctx], device="cpu", use_context_scaler=True)
    assert fitted["subsample_rows"] == fitted["context_rows"]

    probe_idx = np.concatenate([split["ref_eval_idx"][:4], split["future_idx"][:8]])
    parity = probe.check_matched_joint_parity(fitted, X[probe_idx], years[probe_idx])
    assert parity["passed"], parity

    captured = probe.capture_flchain_batches(fitted, X[probe_idx], years[probe_idx], batch_size=6)
    assert captured["probabilities"].shape == (len(probe_idx), 2)
    assert captured["embeddings"].shape == (len(probe_idx), 192)
    assert captured["max_replay_error"] <= 1e-4

    grads, fd_report = probe.compute_gradients_and_fd_check(
        captured["continuation"], captured["embeddings"], grad_batch_size=6, fd_rows=4
    )
    assert grads.shape == (len(probe_idx), 192)
    assert fd_report["passed"], fd_report

    comp = probe.build_sus_comparison_table({"1997_eval": {"raw_embedding_pr": 2.0}})
    assert len(comp) == len(probe.SUS_WORK2_BENCHMARKS)
    assert "multi-factor" in comp["scope_caveat"].iloc[0]
