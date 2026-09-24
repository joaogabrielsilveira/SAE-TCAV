import json
import os
from pathlib import Path

import numpy as np
import pytest

from temporal_performance_windows import (
    ProbabilityResult,
    WindowExperimentConfig,
    _auroc,
    _binary_scores,
    _configure_cuda_allocator,
    _cuda_oom_backoff_batch_size,
    _is_cuda_out_of_memory,
    _macro_f1,
    _trajectory_rows,
    argmax_binary_labels,
    build_diagnostic_rows,
    build_prediction_indices,
    build_training_indices,
    compare_legacy_parent_row,
    death_probabilities,
    effective_window_years,
    exposure_cohort_masks,
    load_window_experiment,
    metric_bundle,
    model_domain_mapping,
    post_death_exclusion_mask,
    run_window_experiment,
    select_frozen_threshold,
)
from temporal_robustness import TemporalPopulation
from temporal_splits import split_reference_patients


def _population(years=(2007, 2008, 2009)):
    patients, observed_years, outcomes = [], [], []
    for year in years:
        for number in range(20):
            patients.append(f"p{number}")
            observed_years.append(year)
            outcomes.append(int(number < 10 and year == years[0]))
    count = len(patients)
    return TemporalPopulation(
        X=np.arange(count * 2, dtype=float).reshape(count, 2),
        outcomes=np.asarray(outcomes),
        years=np.asarray(observed_years),
        patient_ids=np.asarray(patients),
        feature_names=("x", "z"),
        first_eligible_year={f"p{number}": years[0] for number in range(20)},
        record_keys=np.asarray([f"r{i}" for i in range(count)]),
        feature_selection_max_year=2006,
    )


def test_post_death_keeps_first_death_and_audits_later_rows():
    keep, audit = post_death_exclusion_mask(
        ["a", "a", "a", "b"], [2007, 2008, 2009, 2009], [0, 1, 1, 0]
    )
    assert keep.tolist() == [True, True, False, True]
    assert audit == [{
        "row_index": 2,
        "patient_id": "a",
        "year": 2009,
        "first_death_year": 2008,
        "reason": "after_first_observed_death",
    }]


def test_window_aliases_and_domains_are_separate():
    assert effective_window_years("last_5", 2008) == (2007, 2008)
    assert effective_window_years("all_history", 2008) == (2007, 2008)
    assert model_domain_mapping([2007, 2009], [2015]) == {2007: 0, 2009: 1, 2015: 2}


def test_training_uses_prior_rows_but_excludes_current_validation_and_evaluation():
    population = _population()
    reference = np.flatnonzero(population.years == 2008)
    roles = split_reference_patients(
        population.patient_ids[reference],
        np.asarray([int(int(patient[1:]) < 10) for patient in population.patient_ids[reference]]),
        seed=42,
    )
    global_roles = {name: reference[index] for name, index in roles.items()}
    keep = np.ones(len(population.X), dtype=bool)
    train = build_training_indices(
        population=population,
        reference_year=2008,
        logical_window="last_2",
        global_roles=global_roles,
        common_keep=keep,
    )
    assert set(np.flatnonzero(population.years == 2007)).issubset(train)
    assert not set(global_roles["rule_selection_cav"]) & set(train)
    assert not set(global_roles["t0_evaluation"]) & set(train)
    assert np.max(population.years[train]) == 2008


def test_legacy_prediction_protocol_preserves_parent_query_order():
    population = _population()
    validation = np.asarray([20, 23])
    evaluation = np.asarray([25, 41, 58])
    legacy = build_prediction_indices(
        population,
        2008,
        validation,
        evaluation,
        exact_parent_protocol=True,
    )
    common = build_prediction_indices(
        population,
        2008,
        validation,
        evaluation,
        exact_parent_protocol=False,
    )
    assert legacy.tolist() == list(range(20, 60))
    assert common.tolist() == [20, 23, 25, 41, 58]


def test_cuda_oom_detection_and_allocator_configuration(monkeypatch):
    error = RuntimeError("CUDA out of memory")
    assert _is_cuda_out_of_memory(error) is True
    assert _is_cuda_out_of_memory(RuntimeError("host out of memory")) is False
    assert _cuda_oom_backoff_batch_size(
        error,
        8192,
        cuda_requested=True,
        exact_parent_protocol=False,
    ) == 4096
    # Exact legacy predictions retain the parent's fixed batch boundaries.
    assert _cuda_oom_backoff_batch_size(
        error,
        1024,
        cuda_requested=True,
        exact_parent_protocol=True,
    ) is None
    config = WindowExperimentConfig(cuda_allocator_config="max_split_size_mb:64")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    _configure_cuda_allocator(config)
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:64"


def test_threshold_ties_use_precision_then_higher_threshold():
    selected = select_frozen_threshold([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])
    assert selected["threshold"] == pytest.approx(0.7)
    assert selected["death_f1"] == pytest.approx(0.8)


def test_sorted_threshold_scan_matches_brute_force():
    rng = np.random.default_rng(7)
    for _ in range(20):
        truth = rng.integers(0, 2, size=30)
        probability = rng.choice(np.linspace(0.1, 0.9, 9), size=30)
        candidates = []
        for threshold in np.unique(probability):
            predicted = probability >= threshold
            true_positive = np.count_nonzero((truth == 1) & predicted)
            false_positive = np.count_nonzero((truth == 0) & predicted)
            false_negative = np.count_nonzero((truth == 1) & ~predicted)
            precision = true_positive / (true_positive + false_positive)
            denominator = 2 * true_positive + false_positive + false_negative
            f1 = 2 * true_positive / denominator if denominator else 0.0
            candidates.append((f1, precision, threshold))
        expected = max(candidates, key=lambda row: (row[0], row[1], row[2]))
        actual = select_frozen_threshold(truth, probability)
        assert (actual["death_f1"], actual["death_precision"], actual["threshold"]) == pytest.approx(expected)


def test_class_order_and_argmax_half_tie_match_existing_behavior():
    result = ProbabilityResult(
        probabilities=np.asarray([[0.4, 0.6], [0.5, 0.5], [0.2, 0.8]]),
        classes=np.asarray([1, 0]),
    )
    assert death_probabilities(result).tolist() == [0.4, 0.5, 0.2]
    assert argmax_binary_labels(result).tolist() == [0, 1, 0]


def test_argmax_preserves_native_probability_precision_near_half():
    below = np.nextafter(0.5, 0.0)
    above = np.nextafter(0.5, 1.0)
    result = ProbabilityResult(
        probabilities=np.asarray([[below, above]], dtype=np.float64),
        classes=np.asarray([0, 1]),
    )
    assert np.asarray(result.probabilities, dtype=np.float32)[0, 0] == np.asarray(
        result.probabilities, dtype=np.float32
    )[0, 1]
    assert argmax_binary_labels(result).tolist() == [1]


def test_metrics_handle_calibration_degeneracy_and_label_oracle():
    values = metric_bundle([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], 0.5)
    assert values["calibration_intercept"] is None
    assert values["calibration_failure_reason"] == "constant_probability"
    assert values["death_f1_oracle"] >= values["death_f1_at_0_5"]
    assert "oracle_minus_frozen_f1" in values


def test_insufficient_support_nulls_scores_and_is_excluded_from_trajectories():
    invalid = metric_bundle(
        [0, 0, 0, 1],
        [0.1, 0.2, 0.3, 0.9],
        0.5,
        minimum_deaths=2,
        minimum_survivors=2,
    )
    assert invalid["valid"] is False
    assert invalid["death_f1_at_0_5"] is None
    assert invalid["death_average_precision"] is None
    assert invalid["observed_death_prevalence"] == pytest.approx(0.25)
    rows = [{
        "reference_year": 2007,
        "patient_split_seed": 42,
        "window": "all_history",
        "test_year": 2007,
        "temporal_distance": 0,
        "cohort": "all_comer",
        **invalid,
    }]
    assert _trajectory_rows(rows, WindowExperimentConfig()) == []


def test_diagnostics_use_change_from_distance_zero_without_fixed_score_cutoffs():
    baseline = {
        "reference_year": 2007, "patient_split_seed": 42, "window": "all_history",
        "test_year": 2007, "temporal_distance": 0, "cohort": "all_comer", "valid": True,
        "death_f1_at_0_5": 0.4, "death_f1_at_frozen_threshold": 0.5,
        "death_average_precision": 0.6, "death_f1_oracle": 0.55,
        "brier_score": 0.1, "observed_death_prevalence": 0.1,
        "predicted_positive_rate_at_0_5": 0.1,
        "predicted_positive_rate_at_frozen_threshold": 0.12,
        "calibration_intercept": 0.0, "calibration_slope": 1.0,
    }
    future = {
        **baseline, "test_year": 2008, "temporal_distance": 1,
        "death_f1_at_0_5": 0.2, "death_average_precision": 0.62,
        "death_f1_oracle": 0.57,
    }
    rows = build_diagnostic_rows([baseline, future])
    assert rows[0]["classification"] == "reference_baseline"
    assert rows[1]["classification"] == "default_threshold_or_scaling_failure_pattern"
    assert rows[1]["delta_death_f1_at_0_5"] == pytest.approx(-0.2)
    assert "requires_cluster_interval_confirmation" in rows[1]["inferential_status"]


def test_legacy_parity_compares_parent_retained_metrics():
    actual = {
        "reference_year": 2007, "patient_split_seed": 42, "test_year": 2008,
        "temporal_distance": 1, "record_count": 100, "death_count": 10,
        "survivor_count": 90, "death_f1_at_0_5": 0.25,
        "macro_f1_at_0_5": 0.55, "observed_death_prevalence": 0.1,
    }
    expected = {
        "sample_count": 100, "death_count": 10, "survivor_count": 90,
        "death_f1": 0.25, "macro_f1": 0.55, "prevalence": 0.1,
    }
    assert compare_legacy_parent_row(actual, expected)["parity"] is True
    expected["death_f1"] = 0.20
    mismatch = compare_legacy_parent_row(actual, expected)
    assert mismatch["parity"] is False
    assert mismatch["death_f1_at_0_5_difference"] == pytest.approx(0.05)
    assert mismatch["accepted"] is False


def test_legacy_parity_accepts_only_reproducible_exact_half_tie_without_changing_metrics():
    truth = np.asarray([1, 1, 1, 0, 0, 0, 0, 0])
    hard = np.asarray([1, 1, 1, 1, 1, 1, 0, 0])
    probability = np.asarray([0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.5, 0.1])
    actual = {
        "reference_year": 2009, "patient_split_seed": 44, "test_year": 2010,
        "temporal_distance": 1, "record_count": 8, "death_count": 3,
        "survivor_count": 5, "death_f1_at_0_5": _binary_scores(truth, hard)["f1"],
        "macro_f1_at_0_5": _macro_f1(truth, hard),
        "observed_death_prevalence": 3 / 8,
    }
    parent_labels = hard.copy()
    parent_labels[6] = 1
    expected = {
        "sample_count": 8, "death_count": 3, "survivor_count": 5,
        "death_f1": _binary_scores(truth, parent_labels)["f1"],
        "macro_f1": _macro_f1(truth, parent_labels), "prevalence": 3 / 8,
    }
    reconciled = compare_legacy_parent_row(
        actual,
        expected,
        y_true=truth,
        death_probability=probability,
        hard_labels_at_half=hard,
    )
    assert reconciled["parity"] is False
    assert reconciled["accepted"] is True
    assert reconciled["acceptance_reason"] == "parent_aggregates_reproduced_by_exact_0_5_tie_assignment"
    assert reconciled["exact_half_probability_count"] == 1
    assert reconciled["half_tie_reconciliation_solution_count"] == 1
    assert reconciled["half_tie_minimum_label_changes"] == 1
    assert actual["death_f1_at_0_5"] == _binary_scores(truth, hard)["f1"]


def test_legacy_parity_does_not_reconcile_near_half_probability():
    truth = np.asarray([1, 0])
    hard = np.asarray([1, 0])
    actual = {
        "reference_year": 2009, "patient_split_seed": 44, "test_year": 2010,
        "temporal_distance": 1, "record_count": 2, "death_count": 1,
        "survivor_count": 1, "death_f1_at_0_5": 1.0,
        "macro_f1_at_0_5": 1.0, "observed_death_prevalence": 0.5,
    }
    expected = {
        "sample_count": 2, "death_count": 1, "survivor_count": 1,
        "death_f1": 2 / 3, "macro_f1": 1 / 3, "prevalence": 0.5,
    }
    result = compare_legacy_parent_row(
        actual,
        expected,
        y_true=truth,
        death_probability=np.asarray([0.8, np.nextafter(0.5, 1.0)]),
        hard_labels_at_half=hard,
    )
    assert result["accepted"] is False
    assert result["exact_half_probability_count"] == 0


def test_rank_auroc_preserves_half_credit_for_ties():
    assert _auroc(
        np.asarray([0, 1, 0, 1]),
        np.asarray([0.1, 0.8, 0.8, 0.9]),
    ) == pytest.approx(0.875)


def test_exposure_cohorts_are_exact():
    masks = exposure_cohort_masks(["new", "train", "threshold", "both"], ["train", "both"], ["threshold", "both"])
    assert masks["pipeline_unseen"].tolist() == [True, False, False, False]
    assert masks["returning_model_seen"].tolist() == [False, True, False, True]
    assert masks["threshold_only_seen"].tolist() == [False, False, True, False]


class _Adapter:
    def __init__(self):
        patients = np.asarray([f"p{i}" for i in range(40)])
        outcomes = np.asarray([0, 1] * 20)
        self.population = TemporalPopulation(
            X=np.column_stack((outcomes, np.arange(40))),
            outcomes=outcomes,
            years=np.full(40, 2007),
            patient_ids=patients,
            feature_names=("signal", "row"),
            first_eligible_year={patient: 2007 for patient in patients},
            record_keys=np.asarray([f"record-{i}" for i in range(40)]),
            feature_selection_max_year=2006,
        )
        self.calls = 0

    def load_population(self, config):
        return self.population

    def fit_predict(self, *, population, predict_indices, **kwargs):
        self.calls += 1
        death = np.where(population.outcomes[predict_indices] == 1, 0.8, 0.2)
        return ProbabilityResult(np.column_stack((1 - death, death)), np.asarray([0, 1]))


def test_resumable_runner_writes_checksummed_isolated_artifacts(tmp_path: Path):
    adapter = _Adapter()
    config = WindowExperimentConfig(
        artifact_dir=str(tmp_path),
        reference_years=(2007,),
        patient_split_seeds=(42,),
        windows=("reference_only_common", "last_2", "all_history"),
        final_evaluation_year=2007,
        bootstrap_replicates=5,
        show_progress=False,
    )
    result = run_window_experiment(config, adapter=adapter, fail_fast=True)
    assert result["complete"] is True
    # Common aliases share one fitted probability cache.
    assert adapter.calls == 1
    resumed = run_window_experiment(config, adapter=adapter, fail_fast=True)
    assert resumed["complete"] is True
    assert adapter.calls == 1
    loaded = load_window_experiment(result["manifest_path"])
    assert loaded["artifacts"]["yearly_metrics"]
    assert Path(tmp_path, "latest_manifest.json").is_file()
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert set(manifest["artifacts"]) >= {
        "population_exclusions", "role_exposure_audit", "thresholds",
        "record_probabilities", "yearly_metrics", "paired_window_contrasts",
        "diagnostic_classifications", "legacy_parent_metric_parity",
    }
    assert manifest["source_fingerprints"]["tabpfn_model.py"]
    assert "numerical_environment" in manifest
    log_path = Path(result["artifact_dir"], manifest["persistent_log"])
    assert any(
        phase in log_path.read_text()
        for phase in ("phase=job_complete", "phase=job_resume")
    )
