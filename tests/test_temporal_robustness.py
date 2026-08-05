import json
from pathlib import Path

import numpy as np
import pytest

from robustness_matching import analyze_run_pair
from temporal_cohorts import (
    assign_future_provenance,
    cohort_masks,
    metric_delta,
    relative_domain_map,
)
from temporal_config import (
    TemporalMatchingConfig,
    TemporalRetentionConfig,
    TemporalRobustnessConfig,
    TemporalSupportConfig,
)
from temporal_matching import build_canonical_factor_views, factor_family_uid
from temporal_metrics import (
    common_patient_jaccard,
    prevalence_retention,
    support_aware_classification_metrics,
)
from temporal_robustness import TemporalPopulation, run_temporal_robustness
from temporal_rules import fit_canonical_targets, select_activation_winner
from temporal_splits import ROLE_FRACTIONS, split_reference_patients
from temporal_cav import temporal_tcav


def test_five_way_split_is_deterministic_stratified_and_patient_disjoint():
    patients = np.asarray([f"p{i}" for i in range(80)])
    outcomes = np.asarray([0] * 40 + [1] * 40)
    first = split_reference_patients(patients, outcomes, seed=42)
    second = split_reference_patients(patients, outcomes, seed=42)

    assert tuple(first) == tuple(name for name, _ in ROLE_FRACTIONS)
    assert all(np.array_equal(first[name], second[name]) for name in first)
    sets = [set(patients[indices]) for indices in first.values()]
    assert not any(sets[i] & sets[j] for i in range(5) for j in range(i + 1, 5))
    assert sum(len(indices) for indices in first.values()) == 80
    assert len(first["tabpfn_context"]) == 40
    assert all(len(first[name]) == 10 for name, _ in ROLE_FRACTIONS[1:])


def test_future_provenance_is_exhaustive_and_pipeline_unseen_is_inclusive_union():
    role_patients = {
        "tabpfn_context": {"fit"},
        "sae_discovery": set(),
        "rule_discovery": set(),
        "rule_selection_cav": set(),
        "t0_evaluation": {"t0"},
    }
    labels = assign_future_provenance(
        ["t0", "fit", "new", "prior"],
        {"t0": 2007, "fit": 2007, "new": 2008, "prior": 2005},
        2007,
        role_patients,
    )
    assert labels.tolist() == [
        "returning_t0", "returning_fitting", "new_entrant", "prior_nonreference_returner"
    ]
    masks = cohort_masks(labels)
    assert masks["all_comer"].sum() == 4
    assert masks["pipeline_unseen"].tolist() == [True, False, True, True]


def test_domains_and_delta_names_preserve_estimand_semantics():
    assert relative_domain_map(2008, [2008, 2009, 2011]) == {2008: 0, 2009: 1, 2011: 3}
    paired = metric_delta(0.7, complete_t0_value=0.9, paired_t0_value=0.8, cohort_view="returning_t0")
    benchmark = metric_delta(0.7, complete_t0_value=0.9, paired_t0_value=None, cohort_view="new_entrant")
    assert paired == {"paired_delta": pytest.approx(-0.1), "benchmark_delta": None, "delta_not_defined": False}
    assert benchmark == {"paired_delta": None, "benchmark_delta": pytest.approx(-0.2), "delta_not_defined": False}


def test_canonical_targets_positive_only_and_rule_winner_ties_are_deterministic():
    targets = fit_canonical_targets(
        [np.nan, -3, 0, 1, 2, 3, 4, np.inf], [0.25, 0.5], minimum_positive_samples=4
    )
    assert targets[0.25].cutoff == pytest.approx(3.25)
    assert targets[0.25].apply(np.asarray([3.24, 3.25])).tolist() == [False, True]
    rows = select_activation_winner(
        [
            {"activation_target": 0.2, "precision": 0.9, "recall": 0.3, "f2": 0.4, "lift": 2, "condition_count": 2, "selected_count": 10, "minimum_support": 10},
            {"activation_target": 0.1, "precision": 0.9, "recall": 0.3, "f2": 0.4, "lift": 2, "condition_count": 2, "selected_count": 10, "minimum_support": 10},
            {"activation_target": 0.5, "precision": 1, "recall": 1, "f2": 1, "lift": 3, "condition_count": 1, "selected_count": 10, "minimum_support": 10, "forced_fallback": True},
        ],
        rule_source="high_precision",
    )
    assert [row["activation_target"] for row in rows if row["target_role"] == "primary"] == [0.1]
    assert next(row for row in rows if row.get("forced_fallback"))["eligible_for_primary_cav"] is False


def test_multithreshold_matching_is_canonical_exact_and_missing_counts_fail():
    directions = np.eye(2)
    masks = {
        70: np.asarray([[1, 0], [1, 0], [0, 1]], dtype=bool),
        80: np.asarray([[1, 0], [0, 0], [0, 1]], dtype=bool),
        90: np.asarray([[1, 0], [0, 0], [0, 0]], dtype=bool),
    }
    analysis = analyze_run_pair(directions, directions, masks, masks, top_k=1)
    views = build_canonical_factor_views(
        reference_year=2007,
        patient_split_seed=42,
        canonical_sae_seed=42,
        sae_seeds=(42, 43, 44, 45, 46),
        canonical_factor_count=2,
        analyses_by_member_seed={43: analysis, 44: analysis, 45: analysis},
        cosine_thresholds=(0.6,),
        overlap_percentiles=(70, 80, 90),
        overlap_thresholds=(0.7,),
    )
    uid = factor_family_uid(2007, 42, 42, 0)
    cosine = next(row for row in views["recurrence"] if row["factor_family_uid"] == uid and row["matching_view"] == "cosine_qualified")
    assert cosine["comparison_count"] == 4
    assert cosine["pass_count"] == 3
    assert cosine["recurrence"] == 0.75
    assert cosine["recurrent"] is True
    assert any(row["matching_criterion"] == "canonical_identity" for row in views["family_members"])


@pytest.mark.parametrize(
    "args, expected",
    [
        ((9, 100, 50, 100), "insufficient_reference_support"),
        ((10, 100, 2, 20), "insufficient_future_support"),
        ((10, 100, 0, 100), "dead_absent"),
        ((10, 100, 7, 100), "underused"),
        ((10, 100, 15, 100), "stable"),
        ((10, 100, 16, 100), "overused"),
    ],
)
def test_prevalence_status_precedence_and_boundaries(args, expected):
    assert prevalence_retention(*args, TemporalRetentionConfig())["status"] == expected


def test_support_aware_f1_and_common_patient_jaccard():
    unsupported = support_aware_classification_metrics([0, 0, 1], [0, 1, 1], minimum_deaths=2, minimum_survivors=2)
    assert unsupported["failure_reason"] == "insufficient_class_support"
    assert unsupported["macro_f1"] is None
    result = common_patient_jaccard(
        [f"p{i}" for i in range(30)], [i < 10 for i in range(30)],
        [f"p{i}" for i in range(30)], [i < 5 for i in range(30)],
    )
    assert result["valid"] is True
    assert result["jaccard"] == 0.5


def test_tcav_neutral_band_boundaries_are_inclusive_without_significance_gate():
    cav = np.asarray([1.0])
    at_lower = temporal_tcav(cav, np.asarray([[1.0], [1.0], [-1.0], [-1.0], [-1.0]]))
    at_upper = temporal_tcav(cav, np.asarray([[1.0], [1.0], [1.0], [-1.0], [-1.0]]))
    assert at_lower == {"tcav": 0.4, "tcav_direction": "neutral", "directional": False}
    assert at_upper == {"tcav": 0.6, "tcav_direction": "neutral", "directional": False}


class _FakeTemporalAdapter:
    def __init__(self, population):
        self.population = population
        self.calls = []

    def load_population(self, config):
        return self.population

    def run_reference_experiment(self, **kwargs):
        self.calls.append(kwargs)
        expected = kwargs["population"].years[kwargs["evaluation_indices"]] - kwargs["reference_year"]
        return {
            "stage_domains": {stage: expected for stage in ("predictions", "embeddings", "gradients", "activations")},
            "performance": [{"reference_year": kwargs["reference_year"], "test_year": kwargs["reference_year"], "macro_f1": 1.0}],
        }


def test_parent_runner_namespaces_manifests_and_keeps_diagonal(tmp_path):
    patients = np.asarray([f"p{i}" for i in range(80)] * 2)
    years = np.asarray([2007] * 80 + [2008] * 80)
    outcomes = np.asarray(([0] * 40 + [1] * 40) * 2)
    population = TemporalPopulation(
        X=np.arange(320, dtype=float).reshape(160, 2),
        outcomes=outcomes,
        years=years,
        patient_ids=patients,
        feature_names=("a", "b"),
        first_eligible_year={patient: 2007 for patient in patients},
        feature_selection_max_year=2006,
    )
    adapter = _FakeTemporalAdapter(population)
    config = TemporalRobustnessConfig(
        artifact_dir=str(tmp_path),
        reference_years=(2007, 2008),
        patient_split_seeds=(42,),
        sae_seeds=(42, 43),
        maximum_split_attempts=2,
        bootstrap_replicates=2,
        support=TemporalSupportConfig(
            context_deaths=1, context_survivors=1, t0_deaths=1, t0_survivors=1,
            factor_positive_activations=1, target_high_records_per_role=1,
            selected_rule_records=1, cav_positive_records=1, cav_negative_records=1,
        ),
    )
    result = run_temporal_robustness(config, adapter=adapter)

    assert len(result["successful_experiments"]) == 2
    assert not result["skipped_references"]
    for row in result["successful_experiments"]:
        manifest = json.loads(Path(row["manifest"]).read_text())
        assert manifest["temporal_domain_map"][str(row["reference_year"])] == 0
        assert Path(row["manifest"]).parent.joinpath("performance.json").is_file()
    assert Path(result["artifact_dir"], "parent_manifest.json").is_file()


def test_threshold_scalar_migration_and_future_feature_rejection():
    with pytest.warns(DeprecationWarning):
        migrated = TemporalMatchingConfig.from_dict({"cosine_analysis_threshold": 0.65})
    assert migrated.cosine_analysis_thresholds == (0.65,)
    with pytest.raises(ValueError, match="feature vocabulary"):
        TemporalRobustnessConfig(reference_years=(2007,), feature_selection_max_year=2008)


def test_robustness_notebook_is_artifact_only():
    notebook = json.loads(Path("robustness_analysis.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    forbidden = (
        "fit_dr_tabpfn(", "train_all_saes(", "fit_activation_target(",
        "train_temporal_cav(", "get_model_gradients(",
        "compute_factor_recurrence(", "segmented_breakpoint(",
    )
    assert not any(call in source for call in forbidden)
