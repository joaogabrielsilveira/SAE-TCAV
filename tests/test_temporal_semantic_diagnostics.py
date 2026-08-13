import pytest

from temporal_config import TemporalRobustnessConfig
from temporal_production import _semantic_selection_diagnostic_rows
from temporal_reporting import _semantic_selection_summary


def _model(stage, *, rescued=()):
    rescue_names = (
        "max_rule_length",
        "max_rules",
        "min_marginal_recall",
        "min_precision",
        "min_lift",
    )
    return {
        "run_id": "0",
        "factor_id": 7,
        "target": {"name": "top_30pct_positive", "positive_fraction": 0.3},
        "valid": stage == "selected",
        "reason": None if stage == "selected" else "no_subset_satisfies_constraints",
        "selection": {
            "diagnostics": {
                "n_input_candidates": 4,
                "n_eligible_candidates": 3,
                "n_excluded_by_rule_length": 1,
                "n_positive_selection_targets": 8,
            },
        },
        "selection_diagnostics": {
            "funnel_stage": stage,
            "ablation_eligible": stage == "no_feasible_subset",
            **{
                f"rescued_without_{name}": name in rescued
                for name in rescue_names
            },
        },
    }


def test_temporal_semantic_diagnostics_flatten_and_summarize_overlapping_rescues():
    config = TemporalRobustnessConfig(
        reference_years=(2007,),
        patient_split_seeds=(42,),
        sae_seeds=(42, 43),
    )
    family_members = [{
        "member_sae_seed": 42,
        "member_factor_id": 7,
        "factor_family_uid": "2007/42/42/7",
    }]
    models = [
        _model("no_feasible_subset", rescued=("min_precision", "max_rules")),
        {**_model("selected"), "factor_id": 8},
    ]

    rows = _semantic_selection_diagnostic_rows(
        models,
        family_members,
        reference_year=2007,
        patient_split_seed=42,
        config=config,
    )
    assert rows[0]["factor_family_uid"] == "2007/42/42/7"
    assert rows[0]["n_excluded_by_rule_length"] == 1
    assert rows[0]["rescued_without_min_precision"] is True
    assert rows[0]["rescued_without_max_rules"] is True

    summary = _semantic_selection_summary(rows)
    overall = [row for row in summary if row["summary_scope"] == "overall"]
    funnel = {
        row["criterion"]: row
        for row in overall
        if row["summary_kind"] == "exclusive_funnel"
    }
    assert funnel["no_feasible_subset"]["factor_target_count"] == 1
    assert funnel["selected"]["factor_target_count"] == 1
    assert sum(row["factor_target_count"] for row in funnel.values()) == 2

    rescues = {
        row["criterion"]: row
        for row in overall
        if row["summary_kind"] == "overlapping_ablation_rescue"
    }
    assert rescues["min_precision"]["factor_target_count"] == 1
    assert rescues["max_rules"]["factor_target_count"] == 1
    assert rescues["min_precision"]["denominator_factor_target_count"] == 1
    assert rescues["min_precision"]["factor_target_rate"] == pytest.approx(1.0)

