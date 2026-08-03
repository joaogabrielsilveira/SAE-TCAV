"""Pure multi-percentile matching analysis for pairs of SAE runs.

``analyze_run_pair`` is the module's public seam.  It computes every geometric
quantity before policy thresholds are applied, so callers can cache its result
and change reporting thresholds without recomputing matrices or assignments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _scipy_assignment
except ImportError:  # Lightweight fallback keeps this pure module testable.
    _scipy_assignment = None


@dataclass(frozen=True)
class Assignment:
    """One maximum-weight one-to-one assignment."""

    pairs: tuple[tuple[int, int], ...]
    left_to_right: tuple[int | None, ...]
    right_to_left: tuple[int | None, ...]


@dataclass(frozen=True)
class NearestNeighbor:
    """One ranked directed candidate with threshold-independent diagnostics."""

    source_side: str
    source_factor: int
    target_factor: int
    rank: int
    score: float
    reciprocal_raw: bool
    target_collision_count_raw: int


@dataclass(frozen=True)
class NearestHungarianGap:
    """Best directed neighbor compared with the source's Hungarian partner."""

    source_side: str
    source_factor: int
    nearest_target: int | None
    nearest_score: float | None
    hungarian_target: int | None
    hungarian_score: float | None
    nearest_minus_hungarian: float | None


@dataclass(frozen=True)
class PairMatchingAnalysis:
    """Complete raw analysis for one unordered pair of SAE runs."""

    cosine: np.ndarray
    overlaps: dict[int, np.ndarray]
    cosine_assignment: Assignment
    overlap_assignments: dict[int, Assignment]
    nearest_neighbors: dict[str, tuple[NearestNeighbor, ...]]
    nearest_hungarian_gaps: dict[str, tuple[NearestHungarianGap, ...]]
    top_k: int


def analyze_run_pair(
    left_directions: np.ndarray,
    right_directions: np.ndarray,
    left_profiles: Mapping[int, np.ndarray],
    right_profiles: Mapping[int, np.ndarray],
    top_k: int,
) -> PairMatchingAnalysis:
    """Compute unfiltered assignments and directed rankings for one run pair."""

    if int(top_k) < 1:
        raise ValueError("top_k must be positive")
    left = _directions(left_directions, "left_directions")
    right = _directions(right_directions, "right_directions")
    if left.shape[1] != right.shape[1]:
        raise ValueError("direction dimensions must match")
    percentiles = _validate_profiles(
        left_profiles,
        right_profiles,
        left.shape[0],
        right.shape[0],
    )

    cosine = _cosine_matrix(left, right)
    overlaps = {
        percentile: _overlap_matrix(
            np.asarray(left_profiles[percentile], dtype=bool),
            np.asarray(right_profiles[percentile], dtype=bool),
        )
        for percentile in percentiles
    }
    cosine_assignment = _assignment(cosine)
    overlap_assignments = {
        percentile: _assignment(matrix)
        for percentile, matrix in overlaps.items()
    }

    nearest_neighbors: dict[str, tuple[NearestNeighbor, ...]] = {}
    nearest_hungarian_gaps: dict[str, tuple[NearestHungarianGap, ...]] = {}
    metric_inputs = {"cosine": (cosine, cosine_assignment)}
    metric_inputs.update(
        {
            f"overlap_p{percentile}": (
                overlaps[percentile],
                overlap_assignments[percentile],
            )
            for percentile in percentiles
        }
    )
    for metric, (matrix, assignment) in metric_inputs.items():
        candidates = _directed_rankings(matrix, int(top_k))
        nearest_neighbors[metric] = candidates
        nearest_hungarian_gaps[metric] = _nearest_hungarian_gaps(
            matrix, assignment, candidates
        )

    result = PairMatchingAnalysis(
        cosine=cosine,
        overlaps=overlaps,
        cosine_assignment=cosine_assignment,
        overlap_assignments=overlap_assignments,
        nearest_neighbors=nearest_neighbors,
        nearest_hungarian_gaps=nearest_hungarian_gaps,
        top_k=int(top_k),
    )
    validate_analysis(result)
    return result


def validate_analysis(analysis: PairMatchingAnalysis) -> None:
    """Reject corrupt cached analyses before artifact publication."""

    cosine = np.asarray(analysis.cosine)
    if cosine.ndim != 2 or min(cosine.shape, default=0) < 1:
        raise ValueError("cosine matrix must be non-empty and two-dimensional")
    _validate_score_matrix(cosine, -1.0, 1.0, "cosine")
    percentiles = tuple(sorted(analysis.overlaps))
    if not percentiles:
        raise ValueError("at least one overlap percentile is required")
    for percentile in percentiles:
        matrix = np.asarray(analysis.overlaps[percentile])
        if matrix.shape != cosine.shape:
            raise ValueError("overlap matrix shape does not match cosine")
        _validate_score_matrix(matrix, 0.0, 1.0, f"overlap_p{percentile}")
    _validate_assignment(analysis.cosine_assignment, cosine.shape)
    if set(analysis.overlap_assignments) != set(percentiles):
        raise ValueError("overlap assignment percentile coverage is incomplete")
    for assignment in analysis.overlap_assignments.values():
        _validate_assignment(assignment, cosine.shape)

    expected_metrics = {"cosine"} | {
        f"overlap_p{percentile}" for percentile in percentiles
    }
    if set(analysis.nearest_neighbors) != expected_metrics:
        raise ValueError("nearest-neighbor metric coverage is incomplete")
    if set(analysis.nearest_hungarian_gaps) != expected_metrics:
        raise ValueError("nearest/Hungarian metric coverage is incomplete")
    expected_sources = sum(cosine.shape)
    for metric, rows in analysis.nearest_neighbors.items():
        matrix = cosine if metric == "cosine" else analysis.overlaps[int(metric[9:])]
        expected_rows = (
            matrix.shape[0] * min(analysis.top_k, matrix.shape[1])
            + matrix.shape[1] * min(analysis.top_k, matrix.shape[0])
        )
        if len(rows) != expected_rows:
            raise ValueError(f"{metric} nearest-neighbor rank coverage is incomplete")
        groups: dict[tuple[str, int], list[NearestNeighbor]] = {}
        for row in rows:
            shape = matrix.shape if row.source_side == "left" else matrix.T.shape
            if row.source_side not in {"left", "right"}:
                raise ValueError("invalid nearest-neighbor source side")
            if not 0 <= row.source_factor < shape[0]:
                raise ValueError("nearest-neighbor source factor is out of bounds")
            if not 0 <= row.target_factor < shape[1]:
                raise ValueError("nearest-neighbor target factor is out of bounds")
            if not 1 <= row.rank <= min(analysis.top_k, shape[1]):
                raise ValueError("nearest-neighbor rank is invalid")
            indexed = (
                matrix[row.source_factor, row.target_factor]
                if row.source_side == "left"
                else matrix[row.target_factor, row.source_factor]
            )
            if not np.isclose(row.score, indexed, rtol=0.0, atol=1e-12):
                raise ValueError("nearest-neighbor score does not match matrix")
            groups.setdefault((row.source_side, row.source_factor), []).append(row)
        for group in groups.values():
            ordered = sorted(group, key=lambda row: row.rank)
            if [row.rank for row in ordered] != list(range(1, len(ordered) + 1)):
                raise ValueError("nearest-neighbor ranks are not contiguous")
            expected = sorted(group, key=lambda row: (-row.score, row.target_factor))
            if [row.target_factor for row in ordered] != [
                row.target_factor for row in expected
            ]:
                raise ValueError("nearest-neighbor ranking is not deterministic")
        if len(analysis.nearest_hungarian_gaps[metric]) != expected_sources:
            raise ValueError(f"{metric} nearest/Hungarian coverage is incomplete")


def _directions(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or min(array.shape, default=0) < 1:
        raise ValueError(f"{name} must be non-empty and two-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_profiles(
    left: Mapping[int, np.ndarray],
    right: Mapping[int, np.ndarray],
    n_left_factors: int,
    n_right_factors: int,
) -> tuple[int, ...]:
    left_keys = tuple(sorted(int(key) for key in left))
    right_keys = tuple(sorted(int(key) for key in right))
    if not left_keys or left_keys != right_keys:
        raise ValueError("profile percentiles must be non-empty and identical")
    row_count: int | None = None
    for percentile in left_keys:
        left_matrix = np.asarray(left[percentile])
        right_matrix = np.asarray(right[percentile])
        if left_matrix.ndim != 2 or right_matrix.ndim != 2:
            raise ValueError("profile masks must be two-dimensional")
        if left_matrix.shape[1] != n_left_factors:
            raise ValueError("left profile factor count does not match directions")
        if right_matrix.shape[1] != n_right_factors:
            raise ValueError("right profile factor count does not match directions")
        if left_matrix.shape[0] != right_matrix.shape[0]:
            raise ValueError("profile masks must describe identical records")
        if row_count is None:
            row_count = left_matrix.shape[0]
        elif left_matrix.shape[0] != row_count:
            raise ValueError("profile row count differs across percentiles")
    return left_keys


def _cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    denominator = left_norm * right_norm.T
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.divide(
            left @ right.T,
            denominator,
            out=np.zeros((left.shape[0], right.shape[0]), dtype=float),
            where=denominator > 0,
        )
    return np.clip(matrix, -1.0, 1.0)


def _overlap_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    intersection = left.astype(np.int64).T @ right.astype(np.int64)
    union = left.sum(axis=0)[:, None] + right.sum(axis=0)[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros(intersection.shape, dtype=float),
        where=union > 0,
    )


def _assignment(matrix: np.ndarray) -> Assignment:
    rows, columns = _linear_sum_assignment(matrix)
    pairs = tuple(sorted((int(row), int(column)) for row, column in zip(rows, columns)))
    left_to_right: list[int | None] = [None] * matrix.shape[0]
    right_to_left: list[int | None] = [None] * matrix.shape[1]
    for left, right in pairs:
        left_to_right[left] = right
        right_to_left[right] = left
    return Assignment(pairs, tuple(left_to_right), tuple(right_to_left))


def _linear_sum_assignment(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if _scipy_assignment is not None:
        return _scipy_assignment(matrix, maximize=True)

    # O(n^3) rectangular Hungarian fallback.  Algorithm expects rows <= columns.
    transposed = matrix.shape[0] > matrix.shape[1]
    scores = matrix.T if transposed else matrix
    row_count, column_count = scores.shape
    costs = float(np.max(scores)) - scores
    u = np.zeros(row_count + 1, dtype=float)
    v = np.zeros(column_count + 1, dtype=float)
    matching = np.zeros(column_count + 1, dtype=int)
    predecessor = np.zeros(column_count + 1, dtype=int)
    for row in range(1, row_count + 1):
        matching[0] = row
        minimum = np.full(column_count + 1, np.inf, dtype=float)
        used = np.zeros(column_count + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            current_row = matching[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[current_row - 1, candidate - 1]
                    - u[current_row]
                    - v[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(column_count + 1):
                if used[candidate]:
                    u[matching[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matching[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matching[column] = matching[previous]
            column = previous
            if column == 0:
                break
    pairs = sorted(
        (int(row - 1), int(column - 1))
        for column, row in enumerate(matching[1:], start=1)
        if row != 0
    )
    rows = np.asarray([pair[0] for pair in pairs], dtype=int)
    columns = np.asarray([pair[1] for pair in pairs], dtype=int)
    if transposed:
        return columns, rows
    return rows, columns


def _directed_rankings(
    matrix: np.ndarray, top_k: int
) -> tuple[NearestNeighbor, ...]:
    left_rankings = _rank_side(matrix, top_k)
    right_rankings = _rank_side(matrix.T, top_k)
    left_best = {source: targets[0][0] for source, targets in left_rankings.items()}
    right_best = {source: targets[0][0] for source, targets in right_rankings.items()}
    left_collisions = _collision_counts(left_best)
    right_collisions = _collision_counts(right_best)
    rows: list[NearestNeighbor] = []
    for source_side, rankings, reverse_best, collisions in (
        ("left", left_rankings, right_best, left_collisions),
        ("right", right_rankings, left_best, right_collisions),
    ):
        for source, candidates in rankings.items():
            for rank, (target, score) in enumerate(candidates, start=1):
                rows.append(
                    NearestNeighbor(
                        source_side=source_side,
                        source_factor=source,
                        target_factor=target,
                        rank=rank,
                        score=score,
                        reciprocal_raw=(
                            rank == 1 and reverse_best.get(target) == source
                        ),
                        target_collision_count_raw=collisions.get(target, 0),
                    )
                )
    return tuple(rows)


def _rank_side(matrix: np.ndarray, top_k: int) -> dict[int, list[tuple[int, float]]]:
    rankings: dict[int, list[tuple[int, float]]] = {}
    target_ids = np.arange(matrix.shape[1])
    for source in range(matrix.shape[0]):
        order = np.lexsort((target_ids, -matrix[source]))
        rankings[source] = [
            (int(target), float(matrix[source, target]))
            for target in order[: min(top_k, matrix.shape[1])]
        ]
    return rankings


def _collision_counts(best_targets: Mapping[int, int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for target in best_targets.values():
        counts[target] = counts.get(target, 0) + 1
    return counts


def _nearest_hungarian_gaps(
    matrix: np.ndarray,
    assignment: Assignment,
    candidates: tuple[NearestNeighbor, ...],
) -> tuple[NearestHungarianGap, ...]:
    best = {
        (row.source_side, row.source_factor): row
        for row in candidates
        if row.rank == 1
    }
    rows: list[NearestHungarianGap] = []
    for side, assignment_targets in (
        ("left", assignment.left_to_right),
        ("right", assignment.right_to_left),
    ):
        for source, hungarian_target in enumerate(assignment_targets):
            nearest = best.get((side, source))
            hungarian_score = None
            if hungarian_target is not None:
                hungarian_score = float(
                    matrix[source, hungarian_target]
                    if side == "left"
                    else matrix[hungarian_target, source]
                )
            nearest_score = None if nearest is None else nearest.score
            gap = (
                None
                if nearest_score is None or hungarian_score is None
                else nearest_score - hungarian_score
            )
            rows.append(
                NearestHungarianGap(
                    source_side=side,
                    source_factor=source,
                    nearest_target=(None if nearest is None else nearest.target_factor),
                    nearest_score=nearest_score,
                    hungarian_target=hungarian_target,
                    hungarian_score=hungarian_score,
                    nearest_minus_hungarian=gap,
                )
            )
    return tuple(rows)


def _validate_score_matrix(
    matrix: np.ndarray, minimum: float, maximum: float, name: str
) -> None:
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} matrix contains non-finite values")
    tolerance = 1e-12
    if np.any(matrix < minimum - tolerance) or np.any(matrix > maximum + tolerance):
        raise ValueError(f"{name} matrix scores lie outside [{minimum}, {maximum}]")


def _validate_assignment(assignment: Assignment, shape: tuple[int, int]) -> None:
    if len(assignment.pairs) != min(shape):
        raise ValueError("Hungarian assignment has incomplete coverage")
    left = [pair[0] for pair in assignment.pairs]
    right = [pair[1] for pair in assignment.pairs]
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise ValueError("Hungarian assignment is not unique")
    if any(not 0 <= value < shape[0] for value in left):
        raise ValueError("Hungarian left factor is out of bounds")
    if any(not 0 <= value < shape[1] for value in right):
        raise ValueError("Hungarian right factor is out of bounds")
    if len(assignment.left_to_right) != shape[0]:
        raise ValueError("Hungarian left map has invalid length")
    if len(assignment.right_to_left) != shape[1]:
        raise ValueError("Hungarian right map has invalid length")
    for left_factor, right_factor in assignment.pairs:
        if assignment.left_to_right[left_factor] != right_factor:
            raise ValueError("Hungarian left map differs from assignment pairs")
        if assignment.right_to_left[right_factor] != left_factor:
            raise ValueError("Hungarian right map differs from assignment pairs")
    if sum(value is not None for value in assignment.left_to_right) != len(
        assignment.pairs
    ):
        raise ValueError("Hungarian left map has extra assignments")
    if sum(value is not None for value in assignment.right_to_left) != len(
        assignment.pairs
    ):
        raise ValueError("Hungarian right map has extra assignments")
