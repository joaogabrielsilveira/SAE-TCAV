"""Analytic examples for the frozen concept-group extension."""
import numpy as np
import pytest
import torch

import concept_groups_over_time as study


def test_opposing_members_are_visible_despite_zero_group_loading():
    result = study.contribution_measures(
        np.array([[2., 1.], [0., 0.]]),
        np.array([[1., 0.], [-2., 0.]]),
        np.array([3., 2.]),
        np.array([[1., 1.], [1., 1.]]),
        (0, 1),
    )
    assert "loading" in result
    np.testing.assert_allclose(result["loading"], [0., 0.])
    np.testing.assert_allclose(result["member_loading"], [[6., -6.], [0., 0.]])
    assert np.isnan(result["normalized_sensitivity"]).all()
    assert result["cancellation"][0] == 1.


def test_scaled_contribution_and_loading_match_independent_oracle():
    rng = np.random.default_rng(8)
    codes, decoder, gradients = rng.random((7, 4)), rng.normal(size=(4, 3)), rng.normal(size=(7, 3))
    scale = np.array([2., 5., 0.3])
    result = study.contribution_measures(codes, decoder, scale, gradients, (0, 3))
    contribution = sum(codes[:, j, None] * decoder[j] * scale for j in (0, 3))
    loading = np.sum(gradients * contribution, axis=1)
    np.testing.assert_allclose(result["contribution"], contribution, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(result["loading"], loading, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(result["member_loading"].sum(1), loading, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(result["normalized_sensitivity"], loading / np.linalg.norm(contribution, axis=1))


def test_midpoint_decomposition_is_exact_for_changing_both_inputs():
    rng = np.random.default_rng(7)
    values = [rng.normal(size=(13, 5)) for _ in range(4)]
    result = study.midpoint_decomposition(*values)
    np.testing.assert_allclose(result["loading_change"], result["contribution_change"] + result["gradient_change"],
                               atol=1e-9, rtol=1e-9)


def test_pairing_does_not_turn_absent_patients_into_zero_observations():
    ids = np.array(["a", "b", "c", "b", "c", "d"])
    years = np.array([2010, 2010, 2010, 2011, 2011, 2011])
    membership = np.array([1, 1, 0, 0, 1, 1], dtype=bool)
    first, second = study.pair_patients(ids, years, 2010, 2011)
    np.testing.assert_array_equal(first, [1, 2])
    np.testing.assert_array_equal(second, [3, 4])
    transitions = study.subgroup_transitions(ids, years, membership, 2010, 2011)
    assert transitions["cohort_absent_after"] == transitions["cohort_absent_before"] == 1
    assert transitions["10"].sum() == transitions["01"].sum() == 1
    assert transitions["11"].sum() == transitions["00"].sum() == 0
    fixed = study.reference_cohort_mask(ids, years, membership, 2010)
    np.testing.assert_array_equal(fixed, [1, 1, 0, 1, 0, 0])


def test_duplicate_patient_years_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        study.pair_patients(["a", "a"], [2010, 2010], 2010, 2011)


def test_global_bootstrap_reuses_patient_multiplicity_across_universes():
    unique, weights = study.global_patient_bootstrap([["a", "b"], ["b", "c"]], draws=19, seed=2)
    first = study.row_bootstrap_weights(["a", "b", "b"], unique, weights)
    second = study.row_bootstrap_weights(["c", "b"], unique, weights)
    np.testing.assert_array_equal(first[:, 1], first[:, 2])
    np.testing.assert_array_equal(first[:, 1], second[:, 1])
    np.testing.assert_array_equal(weights[1:].sum(1), np.full(19, 3))
    with pytest.raises(ValueError, match="absent"):
        study.row_bootstrap_weights(["outside"], unique, weights)


def test_weighted_means_keep_zero_loading_and_undefined_direction_distinct():
    values = [[0., np.nan], [2., 4.]]
    weights = [[1., 1.], [2., 0.], [0., 0.]]
    result = study.weighted_means(values, weights, [True, True])
    np.testing.assert_allclose(result, [[1., 4.], [0., np.nan], [np.nan, np.nan]], equal_nan=True)


def test_disagreement_and_interval_retain_undefined_draws():
    error = study.weighted_disagreement([[1., 0.], [0., 0.]], [[0., 1.], [0., 0.]],
                                        [[1., 1.], [0., 1.], [2., 0.]], [True, True])
    np.testing.assert_allclose(error, [1., np.nan, 1.], equal_nan=True)
    summary = study.interval(error)
    assert summary["value"] == 1.
    assert summary["undefined_draw_fraction"] == 0.5
    assert summary["defined_draws"] == 1


def test_private_or_nested_public_outputs_are_rejected():
    with pytest.raises(ValueError, match="private"):
        study.scalar_records([{"patient_id": "a"}], ["patient_id"])
    with pytest.raises(ValueError, match="scalars"):
        study.scalar_records([{"value": np.array([1., 2.])}], ["value"])
    assert study.scalar_records([{"value": np.float64(np.nan), "status": "undefined"}], ["value", "status"]) == [
        {"value": None, "status": "undefined"}
    ]


def test_resolver_validates_request_and_explicit_cuda(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError):
        study.resolve_device("tpu")
    with pytest.raises(RuntimeError, match="unavailable"):
        study.resolve_device("cuda")
    assert study.resolve_device("auto") == "cpu"
    assert "CUDA unavailable" in caplog.text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable locally; run on Fiji")
def test_cpu_cuda_math_parity():
    rng = np.random.default_rng(19)
    args = (rng.random((20, 5)), rng.normal(size=(5, 7)), np.arange(1, 8), rng.normal(size=(20, 7)), (1, 4))
    cpu = study.contribution_measures(*args, device="cpu")
    gpu = study.contribution_measures(*args, device="cuda")
    for name in cpu:
        torch.testing.assert_close(cpu[name], gpu[name].cpu(), atol=1e-9, rtol=1e-9, equal_nan=True)


def test_frozen_candidates_use_true_size_for_controls_without_reselection():
    entry = {"true": {"selected": ((1, 2), (3,)), "anchors": (1, 3)},
             "controls": [{"anchors": (4, 5), "size_winners": {(2, 1): ((4, 6), (5,)), (1, 1): ((4,), (5,))}}]}
    candidates = study.frozen_candidates(entry)
    assert candidates[0] == ("selected", ((1, 2), (3,)))
    assert candidates[1] == ("singleton", ((1,), (3,)))
    assert candidates[2] == ("control_1", ((4, 6), (5,)))
    assert candidates[3:] == [("control_2", None), ("control_3", None)]


def test_evaluation_selection_is_invariant_to_future_training_values():
    base = {"roles": np.array(["evaluation", "rule_discovery", "evaluation", "evaluation"]),
            "years": np.array([2010, 2011, 2015, 2016])}
    rows = study.eligible_evaluation_rows(base, 2010)
    np.testing.assert_array_equal(rows, [0, 2])
    base["years"][1] = 2010
    np.testing.assert_array_equal(study.eligible_evaluation_rows(base, 2010), [0, 2])


def test_immature_outcomes_and_identity_mismatch_fail_closed():
    labels = study.protected_labels([1., 0., 1.], [2014, 2015, 2016])
    np.testing.assert_array_equal(labels, [1., np.nan, np.nan])
    study.verify_identity({"seeds": (42, 135)}, {"seeds": [42, 135]})
    with pytest.raises(ValueError, match="identity mismatch"):
        study.verify_identity({"population": "first"}, {"population": "different"})


def test_common_direction_and_brier_use_identical_observations():
    result = {"normalized_sensitivity": torch.tensor([2., 100., 4.], dtype=torch.float64)}
    records, replicates = study.common_direction_measures(
        {"kind": "selected"}, result, np.array([True, False, True]), np.array([2010, 2010, 2010]),
        {"R": np.ones(3, dtype=bool)}, np.array([0.1, 0.9, 0.3]), np.array([0., 1., 0.]),
        np.ones((3, 3)), "cpu")
    np.testing.assert_allclose(replicates[(2010, "R", "common_active_absolute_direction")], 3.)
    np.testing.assert_allclose(replicates[(2010, "R", "common_active_brier")], 0.05)
    assert all(row["common_active_rows"] == 2 for row in records)


def test_source_inventory_does_not_substitute_unmatched_concepts(tmp_path):
    import json
    import pickle
    root = tmp_path / "source"
    directory = root / "stats/concept_agreement_study/patient_table"
    directory.mkdir(parents=True)
    concept = {"concept": "last_3/2010/family", "family": "family", "build": "last_3", "reference": 2010, "variants": ["last_3"]}
    (directory / "index.json").write_text(json.dumps({"concepts": [concept], "builds": []}))
    freeze = tmp_path / "freeze.pkl"
    freeze.write_bytes(pickle.dumps({"payload": {"last_3/2010/family/42-135": {"true": {"selected": ((1,), (2,))}}}}))
    _, _, rows = study.source_inventory(root, freeze)
    assert len(rows) == 3
    assert sum(row["status"] == "available" for row in rows) == 1
    assert sum(row["status"] == "missing inherited group" for row in rows) == 2


def test_family_baseline_deduplicates_anchors_and_retains_missing_pairs():
    metadata = {"c": {"reference": 2010, "variants": "last_3"}}
    values = {
        ("c", "42-135", "singleton", "left", 2011, "R", "absolute_loading"): np.array([1., 2.]),
        ("c", "42-135", "singleton", "right", 2011, "R", "absolute_loading"): np.array([5., 6.]),
        ("c", "42-215", "singleton", "left", 2011, "R", "absolute_loading"): np.array([1., 2.]),
        ("c", "42-215", "singleton", "right", 2011, "R", "absolute_loading"): np.array([9., 10.]),
    }
    records, aggregated = study.family_aggregation(values, metadata)
    np.testing.assert_allclose(aggregated[("c", "singleton", 2011, "R", "absolute_loading")], [5., 6.])
    assert records[0]["available_pairs_or_anchors"] == 3


def test_rank_associations_do_not_manufacture_constant_or_unidentified_results():
    assert np.isnan(study.rank_correlation([1., 1., 1.], [1., 2., 3.]))
    assert np.isnan(study.rank_correlation([1., 2., 3.], [1., 2., 3.], [3., 4., 5.]))
    assert study.rank_correlation([1., 2., 3., 4.], [4., 3., 2., 1.]) == -1.


def test_absence_counters_distinguish_subgroup_from_the_whole_cohort():
    result = study.subgroup_transitions(
        ["absent_nonmember", "returner", "returner", "new_member"],
        [2010, 2010, 2011, 2011], [False, True, True, True], 2010, 2011)
    assert result["cohort_absent_after"] == result["cohort_absent_before"] == 1
    assert result["subgroup_absent_after"] == 0
    assert result["subgroup_absent_before"] == 1
    assert result["11"].sum() == 1


def test_missing_loader_family_cannot_be_reported_as_complete_coverage():
    inventory = [{"concept": "first", "status": "available"}, {"concept": "second", "status": "available"},
                 {"concept": "no_inherited_group", "status": "missing inherited group"}]
    study.validate_inventory_coverage(["first", "second"], inventory)
    with pytest.raises(ValueError, match="coverage mismatch"):
        study.validate_inventory_coverage(["first"], inventory)


def test_controls_do_not_claim_true_anchor_common_active_support():
    records, values = study.common_direction_measures(
        {"kind": "control_1"}, {"normalized_sensitivity": torch.tensor([float("nan"), 2.])},
        np.array([True, True]), np.array([2010, 2010]), {"R": np.array([True, True])},
        np.array([0.1, 0.9]), np.array([0., 1.]), np.ones((3, 2)), "cpu")
    assert records == []
    assert values == {}


def test_private_archive_cannot_write_past_its_disk_allowance(tmp_path):
    path = tmp_path / "replicates.npz"
    with pytest.raises(OSError, match="allowance"):
        study.budgeted_npz(path, {"large": np.random.default_rng(1).normal(size=1000)}, byte_limit=300)
    assert path.stat().st_size <= 300


def test_nonfinite_gradient_cannot_silently_drop_one_loading_patient():
    with pytest.raises(ValueError, match="nonfinite raw"):
        study.contribution_measures(
            np.ones((3, 1)), np.ones((1, 2)), np.ones(2),
            np.array([[1., 1.], [np.nan, 1.], [1., 1.]]), (0,))


def test_overflow_cannot_be_reclassified_as_an_empty_metric():
    with pytest.raises(ValueError, match="overflow"):
        study.contribution_measures(
            np.full((3, 1), 1e308), np.full((1, 2), 1e308), np.ones(2), np.ones((3, 2)), (0,))


def test_raw_validation_allows_only_immature_missing_labels():
    base = {"gradients": np.ones((3, 2)), "embeddings": np.ones((3, 2)), "scale": np.ones(2),
            "codes": {seed: np.ones((3, 1)) for seed in (42, 135, 215)},
            "decoders": {seed: np.ones((1, 2)) for seed in (42, 135, 215)},
            "probabilities": np.array([[0.9, 0.1], [0.3, 0.7], [0.8, 0.2]]),
            "labels": np.array([0., 1., np.nan]), "years": np.array([2013, 2014, 2015])}
    study.validate_raw_build(base, [0, 1, 2])
    base["labels"][0] = np.nan
    with pytest.raises(ValueError, match="mature evaluation labels"):
        study.validate_raw_build(base, [0, 1, 2])
    base["labels"][0] = 0.
    base["probabilities"][1, 1] = 1.2
    with pytest.raises(ValueError, match="probabilities"):
        study.validate_raw_build(base, [0, 1, 2])


def test_private_archive_round_trips_numeric_replicates_without_pickle(tmp_path):
    path = tmp_path / "replicates.npz"
    arrays = {"family/absolute_loading": np.array([1., 2., np.nan]), "count": np.array([0, 3, 5])}
    study.budgeted_npz(path, arrays, byte_limit=10000)
    with np.load(path, allow_pickle=False) as stored:
        assert set(stored.files) == set(arrays)
        for name, expected in arrays.items():
            np.testing.assert_array_equal(stored[name], expected)


@pytest.mark.parametrize("redundant", ["zero", "one", "duplicate"])
def test_rank_projection_is_invariant_to_redundant_control_columns(redundant):
    rng = np.random.default_rng(29)
    values, targets = rng.permutation(24).astype(float), rng.permutation(24).astype(float)
    control = np.repeat(np.arange(4), 6).astype(float)
    expected = study.rank_correlation(values, targets, control)
    extra = {"zero": np.zeros(24), "one": np.ones(24), "duplicate": control}[redundant]
    assert np.isfinite(expected)
    np.testing.assert_allclose(study.rank_correlation(values, targets, np.column_stack((control, extra))),
                               expected, rtol=1e-12, atol=1e-12)


def test_rank_projection_keeps_observed_families_after_missing_rows_are_filtered():
    rng = np.random.default_rng(31)
    values, targets = rng.permutation(18).astype(float), rng.permutation(18).astype(float)
    family = np.repeat(["absent", "second", "third"], 6)
    values[family == "absent"] = np.nan
    full_encoding = np.column_stack((family == "second", family == "third"))
    observed_encoding = (family == "third").astype(float)
    expected = study.rank_correlation(values, targets, observed_encoding)
    assert np.isfinite(expected)
    np.testing.assert_allclose(study.rank_correlation(values, targets, full_encoding), expected,
                               rtol=1e-12, atol=1e-12)


def test_redundant_rank_projection_matches_an_orthogonal_column_space_oracle():
    from scipy.stats import rankdata
    rng = np.random.default_rng(37)
    values, targets = rng.permutation(21), rng.permutation(21)
    control = np.repeat(np.arange(3), 7)
    controls = np.column_stack((control, control, np.ones(21), np.zeros(21)))
    design = np.column_stack((np.ones(21), np.apply_along_axis(rankdata, 0, controls)))
    basis, singular, _ = np.linalg.svd(design, full_matrices=False)
    tolerance = max(design.shape) * np.finfo(float).eps * singular[0]
    basis = basis[:, singular > tolerance]
    ranked = np.column_stack((rankdata(values), rankdata(targets)))
    residuals = ranked - basis @ (basis.T @ ranked)
    expected = np.corrcoef(residuals.T)[0, 1]
    np.testing.assert_allclose(study.rank_correlation(values, targets, controls), expected,
                               rtol=1e-12, atol=1e-12)


def test_full_rank_projection_preserves_existing_separate_fit_results():
    from scipy.stats import rankdata
    rng = np.random.default_rng(41)
    values, targets, control = [rng.permutation(27).astype(float) for _ in range(3)]
    design = np.column_stack((np.ones(27), rankdata(control)))
    x, y = rankdata(values), rankdata(targets)
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose(study.rank_correlation(values, targets, control), np.corrcoef(rx, ry)[0, 1],
                               rtol=1e-12, atol=1e-12)


def test_rank_projection_keeps_insufficient_residual_degrees_undefined():
    values, targets = np.array([1., 3., 2., 5., 4.]), np.array([5., 2., 4., 1., 3.])
    controls = np.column_stack(([0., 0., 1., 1., 0.], [0., 0., 0., 0., 1.]))
    assert np.isnan(study.rank_correlation(values, targets, controls))
    assert np.isnan(study.rank_correlation(np.arange(8), np.array([2, 4, 1, 3, 8, 5, 7, 6]), np.arange(8)))


def _association_report_fixture(root):
    import hashlib
    import json
    root.mkdir(exist_ok=True)
    (root / "associations.csv").write_text("metric,value\nspearman,0.1\n")
    (root / "run-manifest.json").write_text(json.dumps({"status": "completed", "config_sha256": "a" * 64}))
    correction = root / "association-correction"
    correction.mkdir()
    (correction / "associations.csv").write_text("metric,value\nspearman,0.2\n")
    manifest = {"status": "completed", "parent_config_sha256": "a" * 64,
                "original_associations_sha256": hashlib.sha256((root / "associations.csv").read_bytes()).hexdigest(),
                "corrected_associations_sha256": hashlib.sha256((correction / "associations.csv").read_bytes()).hexdigest()}
    (correction / "correction-manifest.json").write_text(json.dumps(manifest))
    return correction, manifest


def test_report_without_a_correction_reads_the_original_associations(tmp_path):
    (tmp_path / "associations.csv").write_text("metric,value\nspearman,0.1\n")
    assert study.load_report(tmp_path)["associations"].value.iloc[0] == 0.1


def test_report_reads_only_the_fixed_completed_verified_correction(tmp_path):
    import json
    correction, manifest = _association_report_fixture(tmp_path)
    arbitrary = tmp_path / "unrelated.csv"
    arbitrary.write_text("metric,value\nspearman,999\n")
    manifest["corrected_associations_path"] = str(arbitrary)
    (correction / "correction-manifest.json").write_text(json.dumps(manifest))
    assert study.load_report(tmp_path)["associations"].value.iloc[0] == 0.2
    assert (tmp_path / "associations.csv").read_text() == "metric,value\nspearman,0.1\n"


@pytest.mark.parametrize("invalid", ["pending", "missing_manifest", "corrupt_manifest", "parent_config",
                                      "original_table", "corrected_table", "missing_table", "incomplete_parent"])
def test_report_rejects_unverified_correction_instead_of_silently_falling_back(tmp_path, invalid):
    import json
    correction, manifest = _association_report_fixture(tmp_path)
    path = correction / "correction-manifest.json"
    if invalid == "pending":
        manifest["status"] = "running"
        path.write_text(json.dumps(manifest))
    elif invalid == "missing_manifest":
        path.unlink()
    elif invalid == "corrupt_manifest":
        path.write_text("{broken")
    elif invalid == "parent_config":
        manifest["parent_config_sha256"] = "b" * 64
        path.write_text(json.dumps(manifest))
    elif invalid == "original_table":
        (tmp_path / "associations.csv").write_text("metric,value\nspearman,0.3\n")
    elif invalid == "corrected_table":
        (correction / "associations.csv").write_text("metric,value\nspearman,0.4\n")
    elif invalid == "missing_table":
        (correction / "associations.csv").unlink()
    else:
        (tmp_path / "run-manifest.json").write_text(json.dumps({"status": "running", "config_sha256": "a" * 64}))
    with pytest.raises(ValueError, match="correction"):
        study.load_report(tmp_path)
