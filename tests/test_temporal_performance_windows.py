import json
from pathlib import Path

import numpy as np
import pytest

from temporal_performance_windows import (
    ProbabilityResult,
    WindowExperimentConfig,
    _auroc,
    argmax_binary_labels,
    build_training_indices,
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
    assert argmax_binary_labels(result).tolist() == [1, 0, 1]


def test_metrics_handle_calibration_degeneracy_and_label_oracle():
    values = metric_bundle([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], 0.5)
    assert values["calibration_intercept"] is None
    assert values["calibration_failure_reason"] == "constant_probability"
    assert values["death_f1_oracle"] >= values["death_f1_at_0_5"]
    assert "oracle_minus_frozen_f1" in values


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
        windows=("legacy_reference_only", "reference_only_common", "last_2", "all_history"),
        final_evaluation_year=2007,
        bootstrap_replicates=5,
        show_progress=False,
    )
    result = run_window_experiment(config, adapter=adapter, fail_fast=True)
    assert result["complete"] is True
    # Common aliases share one fitted probability cache.
    assert adapter.calls == 2
    resumed = run_window_experiment(config, adapter=adapter, fail_fast=True)
    assert resumed["complete"] is True
    assert adapter.calls == 2
    loaded = load_window_experiment(result["manifest_path"])
    assert loaded["artifacts"]["yearly_metrics"]
    assert Path(tmp_path, "latest_manifest.json").is_file()
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert set(manifest["artifacts"]) >= {
        "population_exclusions", "role_exposure_audit", "thresholds",
        "record_probabilities", "yearly_metrics", "paired_window_contrasts",
        "diagnostic_classifications",
    }
