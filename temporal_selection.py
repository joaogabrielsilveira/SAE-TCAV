"""Reference-only configuration ranking and isolated post-hoc sensitivity labels."""

from __future__ import annotations

from typing import Mapping, Sequence


REFERENCE_SELECTION_FIELDS = frozenset(
    {
        "recurrent_factor_count",
        "median_recurrence",
        "raw_matching_score",
        "matching_agreement",
        "reference_rule_availability",
        "reference_rule_quality",
    }
)


def rank_reference_configurations(
    rows: Sequence[Mapping[str, object]],
    *,
    ranking_fields: Sequence[str] = (
        "recurrent_factor_count",
        "median_recurrence",
        "raw_matching_score",
        "matching_agreement",
        "reference_rule_availability",
        "reference_rule_quality",
    ),
) -> list[dict[str, object]]:
    fields = tuple(ranking_fields)
    forbidden = set(fields) - REFERENCE_SELECTION_FIELDS
    if forbidden:
        raise ValueError(f"future or undeclared quantities cannot select matching configuration: {sorted(forbidden)}")
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: tuple(-float(row.get(field, 0)) for field in fields)
        + (str(row.get("configuration_id", "")),),
    )
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
        row["selection_scope"] = "reference_only"
    return ranked


def label_post_hoc_future(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            **dict(row),
            "selection_scope": "post_hoc_future",
            "confirmatory_eligible": False,
        }
        for row in rows
    ]
