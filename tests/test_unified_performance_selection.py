from temporal_unified_analysis import choose_f1_variant


def _row(variant, cohort, distance, death_f1, macro_f1=0.5):
    return {
        "variant": variant,
        "cohort_view": cohort,
        "temporal_distance": distance,
        "death_f1": death_f1,
        "macro_f1": macro_f1,
    }


def test_consecutive_zero_f1_triggers_sensitivity_without_replacing_primary():
    rows = [
        _row("original", "all_comer", 0, 0.04, 0.48),
        _row("original", "all_comer", 1, 0.0, 0.45),
        _row("original", "all_comer", 2, 0.0, 0.44),
        _row("balanced_context", "all_comer", 0, 0.29, 0.55),
        _row("balanced_context", "all_comer", 1, 0.20, 0.52),
        _row("balanced_context", "all_comer", 2, 0.16, 0.50),
    ]

    primary, audit = choose_f1_variant(rows)

    all_comer = next(row for row in audit if row["cohort_view"] == "all_comer")
    assert all_comer["sensitivity_triggered"] is True
    assert all_comer["fallback_triggered"] is False
    assert all_comer["primary_variant"] == "original"
    assert all_comer["sensitivity_variant"] == "balanced_context"
    assert all_comer["balanced_sensitivity_available"] is True
    assert [row["death_f1"] for row in primary] == [0.04, 0.0, 0.0]
    assert {row["selected_variant"] for row in primary} == {"original"}


def test_sensitivity_trigger_does_not_require_balanced_rows_to_select_primary():
    rows = [
        _row("original", "pipeline_unseen", 1, 0.0),
        _row("original", "pipeline_unseen", 2, 0.0),
    ]

    primary, audit = choose_f1_variant(rows)

    unseen = next(
        row for row in audit if row["cohort_view"] == "pipeline_unseen"
    )
    assert unseen["sensitivity_triggered"] is True
    assert unseen["balanced_sensitivity_available"] is False
    assert len(primary) == 2
    assert all(row["selected_variant"] == "original" for row in primary)
