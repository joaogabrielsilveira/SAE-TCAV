"""Canonical-seed factor families and independent multi-threshold views."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def factor_family_uid(
    reference_year: int, patient_split_seed: int, canonical_sae_seed: int, canonical_factor_id: int
) -> str:
    return "/".join(
        str(value)
        for value in (reference_year, patient_split_seed, canonical_sae_seed, canonical_factor_id)
    )


def build_canonical_factor_views(
    *,
    reference_year: int,
    patient_split_seed: int,
    canonical_sae_seed: int,
    sae_seeds: Sequence[int],
    canonical_factor_count: int,
    analyses_by_member_seed: Mapping[int, object],
    cosine_thresholds: Sequence[float],
    overlap_percentiles: Sequence[int],
    overlap_thresholds: Sequence[float],
) -> dict[str, list[dict[str, object]]]:
    """Derive every threshold view from compute-once pair matrices/assignments."""

    member_seeds = tuple(int(seed) for seed in sae_seeds if int(seed) != canonical_sae_seed)
    denominator = len(member_seeds)
    if denominator < 1:
        raise ValueError("canonical recurrence requires at least one other SAE seed")
    membership = []
    threshold_rows = []
    recurrence = []
    winner_rows = []

    for factor in range(int(canonical_factor_count)):
        uid = factor_family_uid(reference_year, patient_split_seed, canonical_sae_seed, factor)
        membership.append(
            {
                "reference_year": reference_year,
                "patient_split_seed": patient_split_seed,
                "canonical_sae_seed": canonical_sae_seed,
                "canonical_factor_id": factor,
                "factor_family_uid": uid,
                "member_sae_seed": canonical_sae_seed,
                "member_factor_id": factor,
                "matching_criterion": "canonical_identity",
                "activation_percentile": None,
                "matching_threshold": None,
                "match_score": 1.0,
            }
        )
        observations = {}
        for member_seed in member_seeds:
            analysis = analyses_by_member_seed.get(member_seed)
            if analysis is None:
                observations[member_seed] = None
                continue
            cosine_target = analysis.cosine_assignment.left_to_right[factor]
            cosine_score = None if cosine_target is None else float(analysis.cosine[factor, cosine_target])
            overlap = {}
            for percentile in overlap_percentiles:
                assignment = analysis.overlap_assignments[int(percentile)]
                target = assignment.left_to_right[factor]
                overlap[int(percentile)] = (
                    target,
                    None if target is None else float(analysis.overlaps[int(percentile)][factor, target]),
                )
            observations[member_seed] = (cosine_target, cosine_score, overlap)
            if cosine_target is not None:
                membership.append(
                    {
                        "reference_year": reference_year,
                        "patient_split_seed": patient_split_seed,
                        "canonical_sae_seed": canonical_sae_seed,
                        "canonical_factor_id": factor,
                        "factor_family_uid": uid,
                        "member_sae_seed": member_seed,
                        "member_factor_id": int(cosine_target),
                        "matching_criterion": "cosine",
                        "activation_percentile": None,
                        "matching_threshold": None,
                        "match_score": cosine_score,
                    }
                )
            for percentile, (target, score) in overlap.items():
                if target is not None:
                    membership.append(
                        {
                            "reference_year": reference_year,
                            "patient_split_seed": patient_split_seed,
                            "canonical_sae_seed": canonical_sae_seed,
                            "canonical_factor_id": factor,
                            "factor_family_uid": uid,
                            "member_sae_seed": member_seed,
                            "member_factor_id": int(target),
                            "matching_criterion": "overlap",
                            "activation_percentile": percentile,
                            "matching_threshold": None,
                            "match_score": score,
                        }
                    )

        cosine_counts = {}
        for threshold in cosine_thresholds:
            count = sum(
                observation is not None
                and observation[0] is not None
                and observation[1] >= threshold
                for observation in observations.values()
            )
            cosine_counts[float(threshold)] = int(count)
            recurrence.append(_recurrence_row(uid, factor, "cosine_qualified", count, denominator, cosine_threshold=threshold))

        overlap_counts = {}
        for percentile in overlap_percentiles:
            for threshold in overlap_thresholds:
                count = sum(
                    observation is not None
                    and observation[2][int(percentile)][0] is not None
                    and observation[2][int(percentile)][1] >= threshold
                    for observation in observations.values()
                )
                overlap_counts[(int(percentile), float(threshold))] = int(count)
                recurrence.append(
                    _recurrence_row(
                        uid, factor, "overlap_qualified", count, denominator,
                        overlap_percentile=percentile, overlap_threshold=threshold,
                    )
                )

        for overlap_threshold in overlap_thresholds:
            winner = max(
                overlap_percentiles,
                key=lambda percentile: (
                    overlap_counts[(int(percentile), float(overlap_threshold))],
                    int(percentile),
                ),
            )
            winner_rows.append(
                {
                    "factor_family_uid": uid,
                    "overlap_threshold": float(overlap_threshold),
                    "winning_percentile": int(winner),
                    "qualified_pair_count": overlap_counts[(int(winner), float(overlap_threshold))],
                }
            )

        for cosine_threshold in cosine_thresholds:
            for percentile in overlap_percentiles:
                for overlap_threshold in overlap_thresholds:
                    intersection_count = 0
                    cosine_exclusive_count = 0
                    overlap_exclusive_count = 0
                    for member_seed, observation in observations.items():
                        cosine_pass = bool(
                            observation is not None
                            and observation[0] is not None
                            and observation[1] >= cosine_threshold
                        )
                        overlap_pass = bool(
                            observation is not None
                            and observation[2][int(percentile)][0] is not None
                            and observation[2][int(percentile)][1] >= overlap_threshold
                        )
                        same_target = bool(
                            observation is not None
                            and observation[0] == observation[2][int(percentile)][0]
                        )
                        intersection = cosine_pass and overlap_pass and same_target
                        intersection_count += intersection
                        cosine_exclusive_count += cosine_pass and not intersection
                        overlap_exclusive_count += overlap_pass and not intersection
                        threshold_rows.extend(
                            _member_view_rows(
                                uid, factor, member_seed, observation,
                                cosine_threshold, percentile, overlap_threshold,
                                cosine_pass, overlap_pass, intersection,
                            )
                        )
                    for view, count in (
                        ("intersection", intersection_count),
                        ("cosine_exclusive", cosine_exclusive_count),
                        ("overlap_exclusive", overlap_exclusive_count),
                    ):
                        recurrence.append(
                            _recurrence_row(
                                uid, factor, view, count, denominator,
                                cosine_threshold=cosine_threshold,
                                overlap_percentile=percentile,
                                overlap_threshold=overlap_threshold,
                            )
                        )
    return {
        "family_members": membership,
        "threshold_membership": threshold_rows,
        "recurrence": recurrence,
        "overlap_percentile_winners": winner_rows,
    }


def _recurrence_row(uid, factor, view, count, denominator, **thresholds):
    value = count / denominator
    return {
        "factor_family_uid": uid,
        "canonical_factor_id": factor,
        "matching_view": view,
        "comparison_count": denominator,
        "pass_count": int(count),
        "recurrence": value,
        "recurrent": bool(value > 0.50),
        "cosine_threshold": thresholds.get("cosine_threshold"),
        "overlap_percentile": thresholds.get("overlap_percentile"),
        "overlap_threshold": thresholds.get("overlap_threshold"),
    }


def _member_view_rows(
    uid, factor, member_seed, observation, cosine_threshold, percentile,
    overlap_threshold, cosine_pass, overlap_pass, intersection,
):
    cosine_target = None if observation is None else observation[0]
    overlap_target = None if observation is None else observation[2][int(percentile)][0]
    base = {
        "factor_family_uid": uid,
        "canonical_factor_id": factor,
        "member_sae_seed": member_seed,
        "cosine_threshold": float(cosine_threshold),
        "overlap_percentile": int(percentile),
        "overlap_threshold": float(overlap_threshold),
        "cosine_member_factor_id": cosine_target,
        "overlap_member_factor_id": overlap_target,
    }
    return [
        {**base, "matching_view": "cosine_qualified", "qualified": cosine_pass},
        {**base, "matching_view": "overlap_qualified", "qualified": overlap_pass},
        {**base, "matching_view": "intersection", "qualified": intersection},
        {**base, "matching_view": "cosine_exclusive", "qualified": cosine_pass and not intersection},
        {**base, "matching_view": "overlap_exclusive", "qualified": overlap_pass and not intersection},
    ]


def union_factor_membership(rows: Sequence[Mapping[str, object]]) -> set[tuple[str, int, int]]:
    """Return artifacts to fit once across all qualified threshold views."""

    result = set()
    for row in rows:
        if row.get("qualified"):
            factor = (
                row.get("cosine_member_factor_id")
                if row["matching_view"] in {"cosine_qualified", "cosine_exclusive"}
                else row.get("overlap_member_factor_id")
            )
            if factor is not None:
                result.add((str(row["factor_family_uid"]), int(row["member_sae_seed"]), int(factor)))
    return result
