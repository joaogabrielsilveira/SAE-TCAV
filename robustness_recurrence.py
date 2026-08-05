"""Strict artifact validation and factor recurrence for robustness analysis."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from semantic_artifacts import array_fingerprint


def compute_factor_recurrence(
    factor_counts: Mapping[int, int],
    cosine_matches: Sequence[Mapping[str, Any]],
    overlap_matches: Sequence[Mapping[str, Any]],
    *,
    percentiles: Sequence[int] = (70, 80, 90),
    cosine_threshold: float = 0.60,
    overlap_threshold: float = 0.70,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute primary, secondary-overlap, and highlighted recurrence rows."""

    runs = tuple(sorted(int(run) for run in factor_counts))
    if len(runs) < 2:
        raise ValueError("recurrence requires at least two runs")
    normalized_percentiles = tuple(int(value) for value in percentiles)
    denominator = len(runs) - 1
    cosine_index = _oriented_index(cosine_matches, percentile=None)
    overlap_indices = {
        percentile: _oriented_index(overlap_matches, percentile=percentile)
        for percentile in normalized_percentiles
    }

    primary: list[dict[str, Any]] = []
    secondary: list[dict[str, Any]] = []
    highlights: list[dict[str, Any]] = []
    for run in runs:
        for factor in range(int(factor_counts[run])):
            target_runs = [target for target in runs if target != run]
            assigned = 0
            cosine_passes = 0
            overlap_passes = {percentile: 0 for percentile in normalized_percentiles}
            for target_run in target_runs:
                row = cosine_index.get((run, target_run, factor))
                if row is None:
                    continue
                assigned += 1
                cosine_passes += float(row["cos_sim"]) >= cosine_threshold
                for percentile in normalized_percentiles:
                    passed = (
                        float(row[f"overlap_p{percentile}"])
                        >= overlap_threshold
                    )
                    overlap_passes[percentile] += passed

            row = {
                "run_id": run,
                "factor_id": factor,
                "comparison_count": denominator,
                "assigned_count": assigned,
                "cosine_pass_count": cosine_passes,
                "cosine_recurrence": cosine_passes / denominator,
                "cosine_recurrent": bool(cosine_passes / denominator > 0.50),
            }
            for percentile in normalized_percentiles:
                recurrence = overlap_passes[percentile] / denominator
                row[f"primary_overlap_p{percentile}_pass_count"] = (
                    overlap_passes[percentile]
                )
                row[f"primary_overlap_p{percentile}_recurrence"] = recurrence
                row[f"primary_overlap_p{percentile}_recurrent"] = bool(
                    recurrence > 0.50
                )
            primary.append(row)
            _append_highlights(highlights, row, normalized_percentiles)

            for percentile in normalized_percentiles:
                matches = overlap_indices[percentile]
                assigned_secondary = 0
                pass_count = 0
                for target_run in target_runs:
                    match = matches.get((run, target_run, factor))
                    if match is None:
                        continue
                    assigned_secondary += 1
                    pass_count += float(match["overlap"]) >= overlap_threshold
                recurrence = pass_count / denominator
                secondary.append(
                    {
                        "run_id": run,
                        "factor_id": factor,
                        "percentile": percentile,
                        "comparison_count": denominator,
                        "assigned_count": assigned_secondary,
                        "pass_count": pass_count,
                        "recurrence": recurrence,
                        "recurrent": bool(recurrence > 0.50),
                    }
                )
    return primary, secondary, highlights


def analyze_robustness_artifacts(
    runner_dir: str | Path,
    output_dir: str | Path,
    *,
    save_plots: bool = True,
) -> dict[str, Any]:
    """Validate matching bundle, export recurrence tables, return manifest data."""

    root = Path(runner_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matching_dir = root / "matching"
    required = (
        "manifest.json",
        "cosine_hungarian_matches.json",
        "overlap_hungarian_matches.json",
        "nearest_neighbor_candidates.json",
        "matching_diagnostics.json",
    )
    missing = [name for name in required if not (matching_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"matching artifacts missing: {missing}")

    manifest = _read_json(matching_dir / "manifest.json")
    if manifest.get("scope") != "all":
        raise ValueError("robustness recurrence requires matching.scope='all'")
    n_runs = int(manifest["n_runs"])
    expected_pairs = n_runs * (n_runs - 1) // 2
    if int(manifest.get("expected_run_pair_count", -1)) != expected_pairs:
        raise ValueError("matching manifest expected pair count is invalid")
    entries = list(manifest.get("run_pairs", []))
    coverage = {(int(row["run_i"]), int(row["run_j"])) for row in entries}
    expected_coverage = {
        (left, right)
        for left in range(n_runs)
        for right in range(left + 1, n_runs)
    }
    if len(entries) != expected_pairs or coverage != expected_coverage:
        raise ValueError("matching manifest lacks exact R choose 2 run-pair coverage")
    profile_path = root / str(manifest["profile_artifact"])
    if (
        not profile_path.is_file()
        or _sha256(profile_path) != manifest["profile_artifact_fingerprint"]
    ):
        raise ValueError("high-activation profile artifact fingerprint mismatch")

    matrices = _load_and_validate_matrices(root, entries, manifest)
    cosine_rows = _read_json(matching_dir / "cosine_hungarian_matches.json")
    overlap_rows = _read_json(matching_dir / "overlap_hungarian_matches.json")
    nearest_rows = _read_json(matching_dir / "nearest_neighbor_candidates.json")
    diagnostic_rows = _read_json(matching_dir / "matching_diagnostics.json")
    _validate_stored_scores(
        matrices, cosine_rows, overlap_rows, nearest_rows, diagnostic_rows
    )

    sae_manifest = _read_json(root / "sae_manifest.json")
    factor_counts = {
        int(row["run_id"]): int(row["n_factors"]) for row in sae_manifest
    }
    if set(factor_counts) != set(range(n_runs)):
        raise ValueError("SAE manifest run coverage differs from matching manifest")
    percentiles = tuple(int(value) for value in manifest["analysis_percentiles"])
    primary, secondary, highlights = compute_factor_recurrence(
        factor_counts,
        cosine_rows,
        overlap_rows,
        percentiles=percentiles,
        cosine_threshold=float(manifest["cosine_threshold"]),
        overlap_threshold=float(manifest["overlap_threshold"]),
    )

    summaries = _nearest_exports(diagnostic_rows)
    files = {
        "factor_recurrence_primary": "factor_recurrence_primary.csv",
        "factor_recurrence_secondary_overlap": (
            "factor_recurrence_secondary_overlap.csv"
        ),
        "recurrence_highlights": "recurrence_highlights.csv",
        "nearest_neighbor_summary": "nearest_neighbor_summary.csv",
        "nearest_neighbor_collisions": "nearest_neighbor_collisions.csv",
        "nearest_neighbor_ambiguities": "nearest_neighbor_ambiguities.csv",
    }
    _write_csv(output / files["factor_recurrence_primary"], primary)
    _write_csv(output / files["factor_recurrence_secondary_overlap"], secondary)
    _write_csv(output / files["recurrence_highlights"], highlights)
    _write_csv(output / files["nearest_neighbor_summary"], summaries["summary"])
    _write_csv(output / files["nearest_neighbor_collisions"], summaries["collisions"])
    _write_csv(output / files["nearest_neighbor_ambiguities"], summaries["ambiguities"])
    plot_files = _write_plots(output, primary, secondary, diagnostic_rows) if save_plots else []

    return {
        "matching_manifest_fingerprint": _sha256(matching_dir / "manifest.json"),
        "matching_thresholds": {
            "cosine": float(manifest["cosine_threshold"]),
            "overlap": float(manifest["overlap_threshold"]),
        },
        "matching_percentiles": list(percentiles),
        "recurrence_counts": {
            "primary_highlights": len(highlights),
            "secondary_recurrent": sum(bool(row["recurrent"]) for row in secondary),
        },
        "nearest_neighbor_collision_count": len(summaries["collisions"]),
        "nearest_neighbor_ambiguity_count": len(summaries["ambiguities"]),
        "matching_output_files": {**files, "plots": plot_files},
    }


def _oriented_index(
    rows: Sequence[Mapping[str, Any]], percentile: int | None
) -> dict[tuple[int, int, int], dict[str, Any]]:
    index: dict[tuple[int, int, int], dict[str, Any]] = {}
    for raw in rows:
        if percentile is not None and int(raw["percentile"]) != percentile:
            continue
        row = dict(raw)
        left_run, right_run = int(row["run_i"]), int(row["run_j"])
        left_factor, right_factor = int(row["factor_i"]), int(row["factor_j"])
        left_key = (left_run, right_run, left_factor)
        right_key = (right_run, left_run, right_factor)
        if left_key in index or right_key in index:
            raise ValueError("matching assignment contains duplicate source factors")
        index[left_key] = {
            **row,
            "target_factor": right_factor,
        }
        index[right_key] = {
            **row,
            "target_factor": left_factor,
        }
    return index


def _append_highlights(
    rows: list[dict[str, Any]],
    primary: Mapping[str, Any],
    percentiles: Sequence[int],
) -> None:
    metrics = {
        "cosine": primary["cosine_recurrence"],
        **{
            f"primary_overlap_p{percentile}": primary[
                f"primary_overlap_p{percentile}_recurrence"
            ]
            for percentile in percentiles
        },
    }
    for metric, recurrence in metrics.items():
        if float(recurrence) > 0.50:
            rows.append(
                {
                    "run_id": primary["run_id"],
                    "factor_id": primary["factor_id"],
                    "metric": metric,
                    "recurrence": recurrence,
                }
            )


def _load_and_validate_matrices(
    root: Path,
    entries: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    percentiles = tuple(int(value) for value in manifest["analysis_percentiles"])
    expected_names = {"cosine"} | {
        f"overlap_p{percentile}" for percentile in percentiles
    }
    result: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for entry in entries:
        relative = Path(str(entry["filename"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("matching matrix filename is unsafe")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"matching matrix missing: {path}")
        with np.load(path, allow_pickle=False) as values:
            if set(values.files) != expected_names:
                raise ValueError(f"matrix keys differ from manifest: {path}")
            matrices = {name: np.asarray(values[name]) for name in values.files}
        shape = tuple(int(value) for value in entry["shape"])
        if (
            int(entry["row_count"]) != shape[0]
            or int(entry["column_count"]) != shape[1]
        ):
            raise ValueError(f"matrix row/column metadata mismatch: {path}")
        for name, matrix in matrices.items():
            if matrix.shape != shape or not np.isfinite(matrix).all():
                raise ValueError(f"invalid {name} matrix in {path}")
            expected = entry["fingerprints"][name]
            if array_fingerprint(matrix) != expected:
                raise ValueError(f"{name} fingerprint mismatch in {path}")
        result[(int(entry["run_i"]), int(entry["run_j"]))] = matrices
    return result


def _validate_stored_scores(
    matrices: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    cosine_rows: Sequence[Mapping[str, Any]],
    overlap_rows: Sequence[Mapping[str, Any]],
    nearest_rows: Sequence[Mapping[str, Any]],
    diagnostic_rows: Sequence[Mapping[str, Any]],
) -> None:
    for row in cosine_rows:
        matrix_set = matrices[(int(row["run_i"]), int(row["run_j"]))]
        left, right = int(row["factor_i"]), int(row["factor_j"])
        _same_score(row["cos_sim"], matrix_set["cosine"][left, right])
        for name, matrix in matrix_set.items():
            if name.startswith("overlap_p"):
                _same_score(row[name], matrix[left, right])
    for row in overlap_rows:
        matrix_set = matrices[(int(row["run_i"]), int(row["run_j"]))]
        left, right = int(row["factor_i"]), int(row["factor_j"])
        name = f"overlap_p{int(row['percentile'])}"
        _same_score(row["overlap"], matrix_set[name][left, right])
        _same_score(row["cos_sim"], matrix_set["cosine"][left, right])
    for row in nearest_rows:
        pair = tuple(sorted((int(row["source_run"]), int(row["target_run"]))))
        matrix = matrices[pair][str(row["metric"])]
        source, target = int(row["source_factor"]), int(row["target_factor"])
        indexed = (
            matrix[source, target]
            if row["source_side"] == "left"
            else matrix[target, source]
        )
        _same_score(row["score"], indexed)
    for row in diagnostic_rows:
        if row.get("hungarian_target") is None:
            continue
        pair = tuple(sorted((int(row["source_run"]), int(row["target_run"]))))
        matrix = matrices[pair][str(row["metric"])]
        source, target = int(row["source_factor"]), int(row["hungarian_target"])
        indexed = (
            matrix[source, target]
            if row["source_side"] == "left"
            else matrix[target, source]
        )
        _same_score(row["hungarian_score"], indexed)


def _same_score(stored: Any, indexed: Any) -> None:
    if not np.isclose(float(stored), float(indexed), rtol=0.0, atol=1e-12):
        raise ValueError("stored matching score differs from indexed matrix value")


def _nearest_exports(
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    rows = [dict(row) for row in diagnostics]
    collisions = [
        row
        for row in rows
        if int(row.get("target_collision_count_threshold_qualified", 0)) > 1
    ]
    delta_fields = sorted(
        {key for row in rows for key in row if key.startswith("valid_alternative_delta_")}
    )
    ambiguities = [row for row in rows if any(bool(row[field]) for field in delta_fields)]
    summary: list[dict[str, Any]] = []
    for metric in sorted({str(row["metric"]) for row in rows}):
        metric_rows = [row for row in rows if str(row["metric"]) == metric]
        valid = [row for row in metric_rows if bool(row["threshold_valid"])]
        gaps = [
            float(row["nearest_minus_hungarian"])
            for row in metric_rows
            if row.get("nearest_minus_hungarian") is not None
        ]
        summary.append(
            {
                "metric": metric,
                "source_factor_count": len(metric_rows),
                "threshold_valid_count": len(valid),
                "threshold_valid_rate": len(valid) / len(metric_rows),
                "reciprocal_qualified_rate": (
                    sum(bool(row["reciprocal_threshold_qualified"]) for row in valid)
                    / len(valid)
                    if valid
                    else 0.0
                ),
                "collision_rate": sum(
                    int(row.get("target_collision_count_threshold_qualified", 0)) > 1
                    for row in valid
                ) / len(valid) if valid else 0.0,
                "ambiguity_rate": sum(
                    any(bool(row[field]) for field in delta_fields) for row in valid
                ) / len(valid) if valid else 0.0,
                "mean_nearest_minus_hungarian": float(np.mean(gaps)) if gaps else None,
            }
        )
    return {"summary": summary, "collisions": collisions, "ambiguities": ambiguities}


def _write_plots(
    output: Path,
    primary: Sequence[Mapping[str, Any]],
    secondary: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> list[str]:
    import matplotlib.pyplot as plt

    files: list[str] = []
    histogram_specs = {
        "recurrence_distributions.png": [
            [float(row["cosine_recurrence"]) for row in primary],
            [float(row["recurrence"]) for row in secondary],
        ],
        "nearest_hungarian_gaps.png": [
            [
                float(row["nearest_minus_hungarian"])
                for row in diagnostics
                if row.get("nearest_minus_hungarian") is not None
            ]
        ],
    }
    for filename, datasets in histogram_specs.items():
        figure, axis = plt.subplots(figsize=(7, 4))
        nonempty = [values for values in datasets if values]
        if nonempty:
            axis.hist(nonempty, bins=20, alpha=0.65)
        axis.set_title(filename.removesuffix(".png").replace("_", " ").title())
        figure.tight_layout()
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)
        files.append(filename)

    recurrence_fields = [
        "cosine_recurrence",
        *sorted(
            {
                key
                for row in primary
                for key in row
                if key.startswith("primary_overlap_p")
                and key.endswith("_recurrence")
            }
        ),
    ]
    highlighted_counts = {
        field.removesuffix("_recurrence"): sum(
            float(row[field]) > 0.50 for row in primary
        )
        for field in recurrence_fields
    }
    _bar_plot(
        plt,
        output / "highlighted_factor_counts.png",
        highlighted_counts,
        "Highlighted Factor Counts",
    )
    files.append("highlighted_factor_counts.png")

    delta_fields = sorted(
        {
            key
            for row in diagnostics
            for key in row
            if key.startswith("valid_alternative_delta_")
        }
    )
    for filename, title, predicate in (
        (
            "reciprocal_rates.png",
            "Threshold-qualified Reciprocal Rates",
            lambda row: bool(row["reciprocal_threshold_qualified"]),
        ),
        (
            "collision_rates.png",
            "Threshold-qualified Collision Rates",
            lambda row: int(
                row.get("target_collision_count_threshold_qualified", 0)
            ) > 1,
        ),
        (
            "ambiguity_rates.png",
            "Threshold-qualified Ambiguity Rates",
            lambda row: any(bool(row[field]) for field in delta_fields),
        ),
    ):
        rates = {}
        for metric in sorted({str(row["metric"]) for row in diagnostics}):
            valid = [
                row
                for row in diagnostics
                if str(row["metric"]) == metric and bool(row["threshold_valid"])
            ]
            rates[metric] = (
                sum(predicate(row) for row in valid) / len(valid) if valid else 0.0
            )
        _bar_plot(plt, output / filename, rates, title)
        files.append(filename)
    return files


def _bar_plot(plt: Any, path: Path, values: Mapping[str, float], title: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    labels = list(values)
    axis.bar(range(len(labels)), [values[label] for label in labels])
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
