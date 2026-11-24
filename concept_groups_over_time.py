"""Read-only numerical tools for following frozen concept groups over time.

Loading is the gradient dotted with the represented decoder contribution. It is a
first-order local quantity, not TCAV and not a finite factor-removal effect.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """Resolve the sole device choice, failing on unavailable explicit CUDA."""
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("device=cuda requested but CUDA is unavailable")
    if requested == "auto":
        if not available:
            LOGGER.warning("device=auto: CUDA unavailable, running on CPU")
        return "cuda" if available else "cpu"
    return requested


def contribution_measures(
    codes: Any,
    decoder: Any,
    scale: Any,
    gradients: Any,
    factors: Sequence[int],
    *,
    device: str = "cpu",
    epsilon: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Return group and constituent contributions on the requested device.

    The decoder has shape factors by embedding dimensions; scale multiplies its
    columns to restore raw embedding coordinates. All reductions use float64.
    Cancellation is one minus group norm divided by summed constituent norms.
    It is undefined for a wholly inactive group. Normalized sensitivity is
    undefined for a zero or cancelled group vector.
    """
    selected = tuple(int(factor) for factor in factors)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("factors must be nonempty and unique")
    a = torch.as_tensor(codes, dtype=torch.float64, device=device)
    d = torch.as_tensor(decoder, dtype=torch.float64, device=device)
    s = torch.as_tensor(scale, dtype=torch.float64, device=device)
    g = torch.as_tensor(gradients, dtype=torch.float64, device=device)
    if a.ndim != 2 or d.ndim != 2 or g.ndim != 2:
        raise ValueError("codes, decoder and gradients must be matrices")
    if a.shape[1] != d.shape[0] or g.shape != (a.shape[0], d.shape[1]):
        raise ValueError("incompatible code, decoder and gradient shapes")
    if s.shape != (d.shape[1],) or min(selected) < 0 or max(selected) >= a.shape[1]:
        raise ValueError("invalid scale or factor index")
    if not all(bool(torch.isfinite(value).all()) for value in (a, d, s, g)):
        raise ValueError("nonfinite raw codes, decoder, scale or gradients")
    with torch.inference_mode():
        raw_decoder = d[list(selected)] * s
        selected_codes = a[:, list(selected)]
        contribution = selected_codes @ raw_decoder
        member_loading = selected_codes * (g @ raw_decoder.T)
        loading = member_loading.sum(dim=1)
        norm = torch.linalg.vector_norm(contribution, dim=1)
        member_norm = selected_codes.abs() * torch.linalg.vector_norm(raw_decoder, dim=1)
        total_member_norm = member_norm.sum(dim=1)
        if not all(bool(torch.isfinite(value).all()) for value in
                   (contribution, member_loading, loading, norm, member_norm, total_member_norm)):
            raise ValueError("nonfinite contribution calculation or numerical overflow")
        normalized = torch.where(norm > epsilon, loading / norm.clamp_min(epsilon), torch.nan)
        cancellation = torch.where(
            total_member_norm > epsilon,
            1 - norm / total_member_norm.clamp_min(epsilon),
            torch.nan,
        )
        if bool((~torch.isfinite(normalized) & (norm > epsilon)).any()):
            raise ValueError("nonfinite normalized sensitivity on active observations")
    return {
        "contribution": contribution,
        "loading": loading,
        "normalized_sensitivity": normalized,
        "norm": norm,
        "member_loading": member_loading,
        "member_norm": member_norm,
        "member_activation": selected_codes,
        "cancellation": cancellation.clamp(0, 1),
    }


def midpoint_decomposition(
    first_gradient: Any,
    second_gradient: Any,
    first_contribution: Any,
    second_contribution: Any,
    *,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Exactly partition loading change into contribution and gradient terms."""
    g0, g1, c0, c1 = [
        torch.as_tensor(value, dtype=torch.float64, device=device)
        for value in (first_gradient, second_gradient, first_contribution, second_contribution)
    ]
    if g0.shape != g1.shape or g0.shape != c0.shape or g0.shape != c1.shape or g0.ndim != 2:
        raise ValueError("paired gradients and contributions must have identical matrix shapes")
    result = {
        "loading_change": (g1 * c1 - g0 * c0).sum(dim=1),
        "contribution_change": ((g0 + g1) * (c1 - c0) / 2).sum(dim=1),
        "gradient_change": ((c0 + c1) * (g1 - g0) / 2).sum(dim=1),
    }
    if not all(bool(torch.isfinite(value).all()) for value in result.values()):
        raise ValueError("nonfinite midpoint decomposition or numerical overflow")
    torch.testing.assert_close(result["loading_change"], result["contribution_change"] + result["gradient_change"],
                               rtol=1e-9, atol=1e-9)
    return result


def validate_patient_years(patient_ids: Any, years: Any) -> None:
    """Reject duplicate patient-years before temporal matching."""
    ids, times = np.asarray(patient_ids).astype(str), np.asarray(years)
    if ids.ndim != 1 or times.shape != ids.shape:
        raise ValueError("patient ids and years must be aligned vectors")
    pairs = list(zip(ids.tolist(), times.tolist()))
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate patient-year")


def pair_patients(patient_ids: Any, years: Any, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned row positions for observed patients at both endpoints."""
    validate_patient_years(patient_ids, years)
    ids, times = np.asarray(patient_ids).astype(str), np.asarray(years)
    first_rows = np.flatnonzero(times == start)
    second_rows = np.flatnonzero(times == end)
    _, first, second = np.intersect1d(ids[first_rows], ids[second_rows], return_indices=True)
    return first_rows[first], second_rows[second]


def subgroup_transitions(
    patient_ids: Any, years: Any, membership: Any, start: int, end: int
) -> dict[str, Any]:
    """Separate four membership transitions from genuinely absent observations."""
    first, second = pair_patients(patient_ids, years, start, end)
    mask = np.asarray(membership, dtype=bool)
    times = np.asarray(years)
    if mask.shape != times.shape:
        raise ValueError("membership must align with patient-year rows")
    before, after = mask[first], mask[second]
    return {
        "first_rows": first,
        "second_rows": second,
        "11": before & after,
        "10": before & ~after,
        "01": ~before & after,
        "00": ~before & ~after,
        "cohort_absent_after": int(np.sum(times == start) - len(first)),
        "cohort_absent_before": int(np.sum(times == end) - len(second)),
        "subgroup_absent_after": int(mask[np.setdiff1d(np.flatnonzero(times == start), first)].sum()),
        "subgroup_absent_before": int(mask[np.setdiff1d(np.flatnonzero(times == end), second)].sum()),
    }


def reference_cohort_mask(
    patient_ids: Any, years: Any, membership: Any, reference: int
) -> np.ndarray:
    """Keep observed rows of patients in the original reference-year subgroup."""
    validate_patient_years(patient_ids, years)
    ids, times = np.asarray(patient_ids).astype(str), np.asarray(years)
    mask = np.asarray(membership, dtype=bool)
    if mask.shape != ids.shape:
        raise ValueError("membership must align with patient-year rows")
    reference_ids = ids[(times == reference) & mask]
    return np.isin(ids, reference_ids)


def global_patient_bootstrap(
    patient_sets: Iterable[Any], draws: int = 500, seed: int = 20261106
) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial patient weights over one global evaluation-patient union.

    Row zero is the point estimate; every remaining row is one ordinary
    patient-cluster bootstrap draw with exactly union-size samples.
    """
    arrays = [np.asarray(values).astype(str) for values in patient_sets]
    if not arrays or draws < 1:
        raise ValueError("patient sets and at least one bootstrap draw are required")
    unique_ids = np.unique(np.concatenate(arrays))
    if not len(unique_ids):
        raise ValueError("global patient union must not be empty")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(unique_ids), np.full(len(unique_ids), 1 / len(unique_ids)), size=draws)
    weights = np.vstack([np.ones((1, len(unique_ids)), dtype=np.int64), counts])
    return unique_ids, weights


def row_bootstrap_weights(patient_ids: Any, unique_ids: Any, weights: Any) -> np.ndarray:
    """Project global weights onto rows without redrawing for years or families."""
    ids = np.asarray(patient_ids).astype(str)
    unique = np.asarray(unique_ids).astype(str)
    positions = np.searchsorted(unique, ids)
    if len(unique) == 0 or np.any(positions >= len(unique)):
        raise ValueError("row patient absent from global union")
    if np.any(unique[positions] != ids):
        raise ValueError("row patient absent from global union")
    matrix = np.asarray(weights)
    if matrix.ndim != 2 or matrix.shape[1] != len(unique):
        raise ValueError("bootstrap weights do not align with global ids")
    return matrix[:, positions]


def weighted_means(values: Any, weights: Any, mask: Any, *, device: str = "cpu") -> torch.Tensor:
    """Reduce many draws and metrics on device, keeping missing metrics undefined."""
    x = torch.as_tensor(values, dtype=torch.float64, device=device)
    if x.ndim == 1:
        x = x[:, None]
    w = torch.as_tensor(weights, dtype=torch.float64, device=device)
    selected = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if x.ndim != 2 or w.ndim != 2 or w.shape[1] != len(x) or selected.shape != (len(x),):
        raise ValueError("values, weights and mask must share their row dimension")
    valid = torch.isfinite(x) & selected[:, None]
    denominator = w @ valid.to(torch.float64)
    numerator = w @ torch.where(valid, x, 0)
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1), torch.nan)


def weighted_disagreement(left: Any, right: Any, weights: Any, mask: Any, *, device: str = "cpu") -> torch.Tensor:
    """Return normalized squared contribution disagreement, with shared weights."""
    left = torch.as_tensor(left, dtype=torch.float64, device=device)
    right = torch.as_tensor(right, dtype=torch.float64, device=device)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("paired contribution matrices must share a shape")
    numerator = ((left - right) ** 2).sum(dim=1)
    denominator = (left ** 2 + right ** 2).sum(dim=1)
    values = weighted_means(torch.stack((numerator, denominator), dim=1), weights, mask, device=device)
    return torch.where(values[:, 1] > 1e-12, values[:, 0] / values[:, 1].clamp_min(1e-12), torch.nan)


def interval(values: Any) -> dict[str, float | int | None]:
    """Summarize point plus bootstrap replicates, retaining undefined draws."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("expected point followed by bootstrap replicates")
    finite = array[1:][np.isfinite(array[1:])]
    bounds = np.quantile(finite, (0.025, 0.975)) if len(finite) else (None, None)
    return {
        "value": float(array[0]) if np.isfinite(array[0]) else None,
        "low": None if bounds[0] is None else float(bounds[0]),
        "high": None if bounds[1] is None else float(bounds[1]),
        "defined_draws": int(len(finite)),
        "undefined_draw_fraction": float(1 - len(finite) / (len(array) - 1)),
    }


def scalar_records(records: Iterable[Mapping[str, Any]], allowed_fields: Sequence[str]) -> list[dict[str, Any]]:
    """Validate a scalar-only public schema; forbid patient-level field names."""
    prohibited = {"patient_id", "patient_ids", "patients", "embedding", "embeddings", "gradients", "codes", "contribution"}
    allowed = set(allowed_fields)
    if allowed & prohibited:
        raise ValueError("private field requested in public export")
    output = []
    for record in records:
        if not set(record) <= allowed:
            raise ValueError("unexpected public export field")
        clean = {}
        for key, value in record.items():
            if isinstance(value, np.generic):
                value = value.item()
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError("public exports must contain only scalars")
            clean[key] = None if isinstance(value, float) and not np.isfinite(value) else value
        output.append(clean)
    return output



def frozen_candidates(entry: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Read true, singleton and equally searched size-matched frozen controls."""
    selected = entry["true"]["selected"]
    anchors = entry["true"]["anchors"]
    output = [("selected", selected), ("singleton", tuple((a,) for a in anchors))]
    size = tuple(map(len, selected)) if selected is not None else None
    for position in range(3):
        controls = entry.get("controls", [])
        control = controls[position] if position < len(controls) else None
        choice = control["size_winners"].get(size) if control is not None and size is not None else None
        output.append((f"control_{position + 1}", choice))
    return output


def eligible_evaluation_rows(base: Mapping[str, Any], reference: int, last_year: int = 2015) -> np.ndarray:
    """Expose only held-out rows from reference through the observation boundary."""
    years = np.asarray(base["years"])
    return np.flatnonzero((np.asarray(base["roles"]) == "evaluation") & (years >= reference) & (years <= last_year))


def protected_labels(labels: Any, years: Any, last_outcome_year: int = 2014) -> np.ndarray:
    """Mask immature outcomes before any performance calculation."""
    labels = np.asarray(labels, dtype=float).copy()
    labels[np.asarray(years) > last_outcome_year] = np.nan
    return labels


def file_sha256(path: Any) -> str:
    """Fingerprint one consumed file without loading it all into memory."""
    import hashlib
    from pathlib import Path
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Any, value: Any) -> None:
    """Write a complete JSON artifact atomically and reject nonfinite numbers."""
    import json
    import os
    import tempfile
    from pathlib import Path
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def canonical_identity(identity: Mapping[str, Any]) -> str:
    """Normalize tuple/list representation without weakening identity checks."""
    import json
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported identity value {type(value)}")
    return json.dumps(identity, sort_keys=True, default=convert)


def verify_identity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Fail closed on any mismatch in source objects, patient roles or choices."""
    if canonical_identity(actual) != canonical_identity(expected):
        raise ValueError("frozen group input identity mismatch")


def source_inventory(study_root: Any, group_freeze: Any) -> tuple[dict, dict, list[dict]]:
    """Join inherited frozen groups to the morning study's explicit concept index."""
    import json
    import pickle
    from pathlib import Path
    root, freeze_path = Path(study_root), Path(group_freeze)
    index = json.loads((root / "stats/concept_agreement_study/patient_table/index.json").read_text())
    sidecar = freeze_path.with_suffix(".sha256")
    if sidecar.exists() and file_sha256(freeze_path) != sidecar.read_text().strip():
        raise ValueError("group freeze checksum mismatch")
    with freeze_path.open("rb") as handle:
        freeze = pickle.load(handle)["payload"]
    records = []
    for concept in index["concepts"]:
        for pair in ((42, 135), (42, 215), (135, 215)):
            key = f"{concept['concept']}/{pair[0]}-{pair[1]}"
            entry = freeze.get(key)
            records.append({
                "concept": concept["concept"], "family": concept["family"],
                "build": concept["build"], "reference": concept["reference"],
                "variants": ",".join(concept["variants"]),
                "pair": f"{pair[0]}-{pair[1]}", "key": key,
                "status": "available" if entry is not None else "missing inherited group",
                "selected": entry is not None and entry["true"]["selected"] is not None,
            })
    return index, freeze, records



def validate_inventory_coverage(raw_concepts: Iterable[str], inventory: list[dict]) -> None:
    """Reject silent loss of any matched source family before selecting smoke cases."""
    expected = {row["concept"] for row in inventory if row["status"] == "available"}
    actual = set(raw_concepts)
    if actual != expected:
        missing, unexpected = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(f"raw loader inventory coverage mismatch: missing={missing}, unexpected={unexpected}")


def validate_raw_build(base: Mapping[str, Any], rows: Any, last_outcome_year: int = 2014) -> None:
    """Fail before reductions if consumed raw rows contain invalid numerical data.

    Labels after the mature-outcome boundary are never validated or consumed.
    Their missingness is expected; missing earlier evaluation labels are errors.
    A wholly inactive contribution remains valid and yields undefined direction.
    """
    selected = np.asarray(rows, dtype=int)
    for name in ("gradients", "embeddings"):
        if name in base and not np.isfinite(np.asarray(base[name])[selected]).all():
            raise ValueError(f"nonfinite raw {name} on evaluation rows")
    if not np.isfinite(np.asarray(base["scale"])).all():
        raise ValueError("nonfinite raw scale")
    for seed in (42, 135, 215):
        if not np.isfinite(np.asarray(base["codes"][seed])[selected]).all():
            raise ValueError(f"nonfinite raw codes for seed {seed}")
        if not np.isfinite(np.asarray(base["decoders"][seed])).all():
            raise ValueError(f"nonfinite raw decoder for seed {seed}")
    probability = np.asarray(base["probabilities"])[selected]
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("raw probabilities must be finite and within [0, 1]")
    labels = np.asarray(base["labels"])[selected]
    mature = np.asarray(base["years"])[selected] <= last_outcome_year
    if not np.isfinite(labels[mature]).all() or np.any(~np.isin(labels[mature], [0, 1])):
        raise ValueError("mature evaluation labels must be finite binary outcomes")

def validate_source_alignment(base: Mapping[str, Any], rows: np.ndarray, cached: Mapping[str, Any]) -> None:
    """Assert raw evaluation rows exactly reproduce the morning patient table."""
    if not np.array_equal(np.asarray(base["patients"])[rows].astype(str), cached["patient_id"].astype(str)):
        raise ValueError("morning patient identities/order differ")
    if not np.array_equal(np.asarray(base["years"])[rows], cached["year"]):
        raise ValueError("morning patient years/order differ")
    np.testing.assert_allclose(base["probabilities"][rows, 1], cached["probability"], rtol=1e-4, atol=1e-5)


def tensor_numpy(value: Any) -> np.ndarray:
    """Move a finished reduction to the CPU at an explicit stage boundary."""
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def aggregate_rows(values: Any, weights: Any, mask: Any, device: str) -> np.ndarray:
    """Compute a batch of replicate means with one device reduction."""
    return tensor_numpy(weighted_means(values, weights, mask, device=device))


def metric_record(common: dict, metric: str, values: Any, **extra: Any) -> dict:
    """Build one scalar public result from point and replicate estimates."""
    return {**common, **extra, "metric": metric, **interval(tensor_numpy(values))}


def performance_values(probability: Any, labels: Any) -> np.ndarray:
    """Performance/context values on the exact same rows as the description."""
    probability, labels = np.asarray(probability), np.asarray(labels)
    return np.column_stack(((probability - labels) ** 2, labels, probability))


def annual_measures(
    common: dict, result: dict, gradients: Any, years: np.ndarray, masks: dict,
    probability: np.ndarray, labels: np.ndarray, weights: Any, device: str,
) -> tuple[list[dict], dict]:
    """Annual group loading, direction, cancellation and same-population context."""
    output, replicates = [], {}
    loading = result["loading"]
    direction = result["normalized_sensitivity"]
    norm = result["norm"]
    gradient_norm = torch.linalg.vector_norm(torch.as_tensor(gradients, dtype=torch.float64, device=device), dim=1)
    if not bool(torch.isfinite(gradient_norm).all()):
        raise ValueError("nonfinite gradient norm or numerical overflow")
    data = torch.stack((loading, loading.abs(), direction, direction.abs(), norm,
                        result["cancellation"], gradient_norm), dim=1)
    metric_names = ("signed_loading", "absolute_loading", "signed_direction", "absolute_direction",
                    "contribution_norm", "cancellation", "gradient_norm")
    context = torch.as_tensor(performance_values(probability, labels), dtype=torch.float64, device=device)
    for year in sorted(np.unique(years)):
        for subgroup, subgroup_mask in masks.items():
            mask = subgroup_mask & (years == year)
            means = aggregate_rows(data, weights, mask, device)
            context_means = aggregate_rows(context, weights, mask, device)
            row_common = {**common, "year": int(year), "subgroup": subgroup, "rows": int(mask.sum())}
            counts = tensor_numpy(weights[:, mask].sum(axis=1)).astype(float)
            replicates[(int(year), subgroup, "log_support")] = np.log1p(counts)
            output.append(metric_record(row_common, "support", counts))
            for position, metric in enumerate(metric_names):
                output.append(metric_record(row_common, metric, means[:, position]))
                replicates[(int(year), subgroup, metric)] = means[:, position]
            for position, metric in enumerate(("brier", "mortality", "mean_risk")):
                output.append(metric_record(row_common, metric, context_means[:, position]))
                replicates[(int(year), subgroup, metric)] = context_means[:, position]
            active = mask & tensor_numpy(torch.isfinite(direction))
            active_context = aggregate_rows(context, weights, active, device)
            for position, metric in enumerate(("active_brier", "active_mortality", "active_mean_risk")):
                output.append(metric_record(row_common, metric, active_context[:, position], active_rows=int(active.sum())))
                replicates[(int(year), subgroup, metric)] = active_context[:, position]
            positive = torch.where(torch.isfinite(direction), (direction > 0).to(torch.float64), torch.nan)
            output.append(metric_record(row_common, "positive_direction_fraction",
                                        aggregate_rows(positive, weights, mask, device)[:, 0], active_rows=int(active.sum())))
    return output, replicates


def member_measures(common: dict, result: dict, factors: Sequence[int], years: np.ndarray,
                    masks: dict, weights: Any, device: str) -> list[dict]:
    """Expose each fixed constituent alongside the sum, including opposing signs."""
    output = []
    values = torch.cat((result["member_activation"], result["member_loading"], result["member_loading"].abs()), dim=1)
    size = len(factors)
    for year in sorted(np.unique(years)):
        for subgroup, membership in masks.items():
            mask = membership & (years == year)
            means = aggregate_rows(values, weights, mask, device)
            for position, factor in enumerate(factors):
                for block, metric in enumerate(("activation", "signed_loading", "absolute_loading")):
                    output.append(metric_record({**common, "year": int(year), "subgroup": subgroup,
                                                 "factor": int(factor), "rows": int(mask.sum())},
                                                metric, means[:, block * size + position]))
    return output


def temporal_measures(common: dict, result: dict, gradients: Any, ids: np.ndarray,
                      years: np.ndarray, masks: dict, weights: Any, device: str,
                      reference: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Separate returning patients, membership transitions and fixed-cohort attrition."""
    paired, transitions, fixed = [], [], []
    contribution = result["contribution"]
    loading = tensor_numpy(result["loading"])
    for subgroup, membership in masks.items():
        reference_mask = reference_cohort_mask(ids, years, membership, reference)
        original_n = int(np.sum(reference_mask & (years == reference)))
        for year in sorted(np.unique(years)):
            selected = reference_mask & (years == year)
            values = aggregate_rows(np.abs(loading), weights, selected, device)[:, 0]
            fixed.append(metric_record({**common, "year": int(year), "subgroup": subgroup,
                                        "rows": int(selected.sum()), "reference_rows": original_n,
                                        "retained_fraction": float(selected.sum() / original_n) if original_n else None},
                                       "absolute_loading", values))
        for start in sorted(np.unique(years)):
            end = int(start) + 1
            if end not in years:
                continue
            table = subgroup_transitions(ids, years, membership, int(start), end)
            first, second = table["first_rows"], table["second_rows"]
            common_rows = {**common, "start": int(start), "year": end, "subgroup": subgroup,
                           "returning_rows": len(first), "cohort_absent_after": table["cohort_absent_after"],
                           "cohort_absent_before": table["cohort_absent_before"],
                           "subgroup_absent_after": table["subgroup_absent_after"],
                           "subgroup_absent_before": table["subgroup_absent_before"]}
            start_mask, end_mask = membership & (years == start), membership & (years == end)
            full_start = aggregate_rows(np.abs(loading), weights, start_mask, device)[:, 0]
            full_end = aggregate_rows(np.abs(loading), weights, end_mask, device)[:, 0]
            returning_start = aggregate_rows(np.abs(loading[first]), weights[:, first], membership[first], device)[:, 0]
            returning_end = aggregate_rows(np.abs(loading[second]), weights[:, second], membership[second], device)[:, 0]
            total = full_end - full_start
            returning = returning_end - returning_start
            for metric, value in (("total_absolute_loading_change", total),
                                  ("returning_absolute_loading_change", returning),
                                  ("composition_absolute_loading_change", total - returning)):
                paired.append(metric_record(common_rows, metric, value))
            decomposition = midpoint_decomposition(np.asarray(gradients)[first], np.asarray(gradients)[second],
                                                   contribution[first], contribution[second], device=device)
            for transition in ("11", "10", "01", "00"):
                member = table[transition]
                transition_common = {**common_rows, "transition": transition, "rows": int(member.sum())}
                for metric, values in decomposition.items():
                    mean = aggregate_rows(values, weights[:, first], member, device)[:, 0]
                    transitions.append(metric_record(transition_common, metric, mean))
                loading_change = np.abs(loading[second]) - np.abs(loading[first])
                transitions.append(metric_record(transition_common, "absolute_loading_change",
                                                   aggregate_rows(loading_change, weights[:, first], member, device)[:, 0]))
    return paired, transitions, fixed


def representation_measures(common: dict, candidates: dict, years: np.ndarray, masks: dict,
                            weights: Any, device: str) -> tuple[list[dict], dict]:
    """Score fixed pair contributions on each year's identical subgroup rows."""
    records, replicates = [], {}
    for year in sorted(np.unique(years)):
        for subgroup, membership in masks.items():
            mask = membership & (years == year)
            errors = {}
            for kind, sides in candidates.items():
                errors[kind] = tensor_numpy(weighted_disagreement(sides[0]["contribution"], sides[1]["contribution"],
                                                                 weights, mask, device=device))
            for kind in ("selected", "singleton", "control_1", "control_2", "control_3"):
                error = errors.get(kind, np.full(len(weights), np.nan))
                row_common = {**common, "kind": kind, "year": int(year), "subgroup": subgroup, "rows": int(mask.sum())}
                for metric, values in (("E", error),
                                       ("delta_singleton", error - errors.get("singleton", np.nan)),
                                       ("delta_selected", error - errors.get("selected", np.nan))):
                    records.append(metric_record(row_common, metric, values))
                    replicates[(int(year), subgroup, kind, metric)] = values
    return records, replicates


def family_aggregation(case_replicates: dict, metadata: dict) -> tuple[list[dict], dict]:
    """Median across pair summaries; singleton anchors receive one vote per seed.

    Group summaries first average the two middle observations of the two sides
    (the median for two values), then take the median across three pairs. The
    singleton baseline deduplicates repeated seeds before its three-seed median.
    Incomplete pairs stay visible through expected/available pair counts.
    """
    from collections import defaultdict
    output, family_values = [], {}
    grouped = defaultdict(dict)
    for key, values in case_replicates.items():
        concept, pair, kind, side, year, subgroup, metric = key
        grouped[(concept, kind, year, subgroup, metric)][(pair, side)] = values
    for key, side_values in grouped.items():
        concept, kind, year, subgroup, metric = key
        if kind == "singleton":
            unique = {}
            for (pair, side), values in side_values.items():
                seed = int(pair.split("-")[0 if side == "left" else 1])
                if seed in unique:
                    np.testing.assert_allclose(unique[seed], values, equal_nan=True, atol=1e-9, rtol=1e-9)
                unique[seed] = values
            members = list(unique.values())
            available = len(unique)
        else:
            pairs = sorted(set(pair for pair, _ in side_values))
            members = []
            for pair in pairs:
                sides = [side_values[(pair, side)] for side in ("left", "right") if (pair, side) in side_values]
                if len(sides) == 2:
                    members.append(np.median(np.stack(sides), axis=0))
            available = len(members)
        if not members:
            continue
        values = np.median(np.stack(members), axis=0)
        common = {**metadata[concept], "concept": concept, "kind": kind, "year": year,
                  "subgroup": subgroup, "available_pairs_or_anchors": available, "expected_pairs_or_anchors": 3}
        output.append(metric_record(common, metric, values))
        family_values[(concept, kind, year, subgroup, metric)] = values
    return output, family_values


def rank_correlation(values: Any, targets: Any, controls: Any = None) -> float:
    """Descriptive Spearman/partial-rank association on the observed column space.

    Redundant control coefficients need not be individually identifiable. Their
    shared projection must leave at least three residual degrees of freedom and
    nonzero variation in both ranked residuals.
    """
    from scipy.stats import rankdata
    x, y = np.asarray(values, dtype=float), np.asarray(targets, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    control = np.empty((len(x), 0)) if controls is None else np.asarray(controls, dtype=float)
    if control.ndim == 1:
        control = control[:, None]
    keep &= np.isfinite(control).all(axis=1)
    x, y, control = x[keep], y[keep], control[keep]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    x, y = rankdata(x), rankdata(y)
    if control.shape[1]:
        design = np.column_stack((np.ones(len(x)), np.apply_along_axis(rankdata, 0, control)))
        ranked = np.column_stack((x, y))
        coefficients, _, effective_rank, _ = np.linalg.lstsq(design, ranked, rcond=None)
        if len(x) - effective_rank < 3:
            return np.nan
        # Redundant encodings do not change the uniquely defined projection.
        # The effective column-space rank determines residual degrees of freedom.
        residuals = ranked - design @ coefficients
        x, y = residuals[:, 0], residuals[:, 1]
    if np.linalg.norm(x - x.mean()) < 1e-10 or np.linalg.norm(y - y.mean()) < 1e-10:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def association_records(family_values: dict, metadata: dict, outcome_last_year: int) -> list[dict]:
    """Relate concurrent changes; intervals share patient draws across all rows.

    The per-concept results require three varying adjacent transitions. Pooled
    estimates are exploratory and adjust for family/reference/year, mortality,
    support context and mean predicted risk only when the design is identifiable.
    """
    output, collections = [], {}
    for concept in sorted(metadata):
        for kind in ("selected", "singleton"):
            for subgroup in ("all", "A", "R"):
                for sensitivity, target in (("absolute_loading", "brier"), ("common_active_absolute_direction", "common_active_brier")):
                    years = sorted(year for c, k, year, g, m in family_values
                                   if (c, k, g, m) == (concept, kind, subgroup, sensitivity) and year <= outcome_last_year)
                    transitions = []
                    for end in years:
                        start = end - 1
                        context_prefix = "common_active_" if sensitivity.startswith("common_active_") else ""
                        required = (sensitivity, target, f"{context_prefix}mortality", f"{context_prefix}mean_risk", f"{context_prefix}log_support")
                        if all((concept, kind, y, subgroup, metric) in family_values for y in (start, end) for metric in required):
                            delta = [family_values[(concept, kind, end, subgroup, metric)] -
                                     family_values[(concept, kind, start, subgroup, metric)] for metric in required]
                            transitions.append((end, delta))
                    if not transitions:
                        continue
                    matrix = np.stack([np.stack(delta, axis=1) for _, delta in transitions], axis=1)
                    for adjustment, columns in (("none", []), ("mortality_risk_and_support", [2, 3, 4])):
                        values = np.array([rank_correlation(draw[:, 0], draw[:, 1], draw[:, columns] if columns else None)
                                           for draw in matrix])
                        common = {**metadata[concept], "concept": concept, "kind": kind, "subgroup": subgroup,
                                  "feature": sensitivity, "target": target, "adjustment": adjustment,
                                  "level": "within_concept_changes", "transitions": len(transitions)}
                        output.append(metric_record(common, "spearman", values))
                    key = (kind, subgroup, sensitivity, target)
                    for variant in metadata[concept]["variants"].split(","):
                        collections.setdefault((variant, *key), []).extend((concept, year, matrix[:, position, :])
                                                                        for position, (year, _) in enumerate(transitions))
    for (variant, kind, subgroup, sensitivity, target), rows in collections.items():
        concepts = sorted(set(concept for concept, _, _ in rows))
        years = sorted(set(year for _, year, _ in rows))
        # pragmatic: these small descriptive rank regressions run on CPU; GPU is used for patient reductions.
        family_design = np.array([[float(concept == family) for family in concepts[1:]] for concept, _, _ in rows])
        year_design = np.array([[float(year == value) for value in years[1:]] for _, year, _ in rows])
        matrix = np.stack([values for _, _, values in rows], axis=1)
        for adjustment in ("none", "family_and_year", "family_year_mortality_and_support", "family_year_mortality_risk_and_support"):
            values = []
            for draw in matrix:
                controls = None
                if adjustment != "none":
                    controls = np.column_stack((family_design, year_design))
                    if adjustment.endswith("and_support"):
                        columns = [2, 3, 4] if "_risk_" in adjustment else [2, 4]
                        controls = np.column_stack((controls, draw[:, columns]))
                values.append(rank_correlation(draw[:, 0], draw[:, 1], controls))
            output.append(metric_record({"concept": "pooled", "variant": variant, "kind": kind, "subgroup": subgroup,
                                         "feature": sensitivity, "target": target, "adjustment": adjustment,
                                         "level": "pooled_changes", "transitions": len(rows), "families": len(concepts)},
                                        "spearman", values))
    return output


def representation_summary(replicates: dict, metadata: dict) -> list[dict]:
    """Average later lags within each case, then report dependent-case medians.

    Patient draws are already joint across all cases; case summaries are never
    bootstrapped as independent observations. Controls retain their frozen labels.
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for (concept, pair, year, subgroup, kind, metric), values in replicates.items():
        if year <= metadata[concept]["reference"]:
            continue
        if metric == "delta_singleton" and kind == "selected":
            grouped[(concept, pair, subgroup, "selected_minus_singleton_E")].append(values)
        elif metric == "delta_selected" and kind.startswith("control_"):
            grouped[(concept, pair, subgroup, f"selected_minus_{kind}_E")].append(-values)
    output, pooled = [], defaultdict(list)
    for (concept, pair, subgroup, metric), values in grouped.items():
        array = np.stack(values)
        valid = np.isfinite(array)
        denominator = valid.sum(axis=0)
        average = np.divide(np.where(valid, array, 0).sum(axis=0), denominator,
                            out=np.full(array.shape[1], np.nan), where=denominator > 0)
        output.append(metric_record({**metadata[concept], "concept": concept, "pair": pair,
                                     "subgroup": subgroup, "later_years": len(values),
                                     "level": "case_mean_over_later_years"}, metric, average))
        for variant in metadata[concept]["variants"].split(","):
            pooled[(variant, subgroup, metric)].append(average)
    for (variant, subgroup, metric), values in pooled.items():
        array = np.stack(values)
        finite = np.isfinite(array)
        median = np.full(array.shape[1], np.nan)
        for draw in range(array.shape[1]):
            if finite[:, draw].any():
                median[draw] = np.median(array[finite[:, draw], draw])
        output.append(metric_record({"variant": variant, "subgroup": subgroup,
                                     "level": "median_of_dependent_case_means", "cases_total": len(values),
                                     "cases_computable": int(finite[:, 0].sum()),
                                     "patient_draws_shared_across_cases": True}, metric, median))
    return output



def association_table_path(output_dir: Any) -> Any:
    """Choose a completed, verified fixed-path correction or the original table.

    A correction directory signals an attempted correction: missing, pending or
    mismatched provenance fails closed instead of quietly showing old results.
    Manifest-supplied paths are never used to locate the corrected table.
    """
    import json
    from pathlib import Path
    root = Path(output_dir)
    original = root / "associations.csv"
    correction = root / "association-correction"
    if not correction.exists() and not correction.is_symlink():
        return original
    manifest_path = correction / "correction-manifest.json"
    corrected = correction / "associations.csv"
    try:
        if correction.is_symlink() or not correction.is_dir():
            raise ValueError("correction must use the fixed regular directory")
        if manifest_path.is_symlink() or corrected.is_symlink():
            raise ValueError("correction provenance and table must be regular files")
        manifest = json.loads(manifest_path.read_text())
        parent = json.loads((root / "run-manifest.json").read_text())
        config_hash = parent.get("config_sha256")
        if manifest.get("status") != "completed" or parent.get("status") != "completed":
            raise ValueError("correction and parent run must both be completed")
        if not isinstance(config_hash, str) or len(config_hash) != 64:
            raise ValueError("correction parent configuration fingerprint is missing")
        if manifest.get("parent_config_sha256") != config_hash:
            raise ValueError("correction parent configuration fingerprint mismatch")
        if manifest.get("original_associations_sha256") != file_sha256(original):
            raise ValueError("correction original association-table fingerprint mismatch")
        if manifest.get("corrected_associations_sha256") != file_sha256(corrected):
            raise ValueError("correction association-table fingerprint mismatch")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise ValueError(f"invalid association correction: {error}") from error
    return corrected

def load_report(output_dir: Any) -> dict[str, Any]:
    """Read compact aggregate tables; no patient-level arrays are loaded."""
    from pathlib import Path
    import pandas as pd
    root = Path(output_dir)
    names = ("inventory", "annual", "representation", "members", "paired", "transitions", "fixed_cohort",
             "families", "associations", "summary")
    output = {}
    for name in names:
        path = association_table_path(root) if name == "associations" else root / f"{name}.csv"
        try:
            output[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
        except pd.errors.EmptyDataError:
            output[name] = pd.DataFrame()
    return output


def common_direction_measures(common: dict, result: dict, common_active: np.ndarray, years: np.ndarray,
                              masks: dict, probability: np.ndarray, labels: np.ndarray,
                              weights: Any, device: str) -> tuple[list[dict], dict]:
    """Compare methods' directions and Brier on one fixed common-active support.

    Support intersects selected groups and original anchors across every available
    seed pair for the family. Thus repeated singleton seeds retain identical rows.
    This companion does not replace the full-subgroup loading/Brier summaries.
    Controls retain their own-active diagnostics because the selected/anchor
    intersection does not guarantee that a control contribution is nonzero.
    """
    if common.get("kind") not in ("selected", "singleton"):
        return [], {}
    output, replicates = [], {}
    direction = result["normalized_sensitivity"]
    context = torch.as_tensor(performance_values(probability, labels), dtype=torch.float64, device=device)
    for year in sorted(np.unique(years)):
        for subgroup, membership in masks.items():
            mask = membership & (years == year) & common_active
            values = aggregate_rows(direction.abs(), weights, mask, device)[:, 0]
            row_common = {**common, "year": int(year), "subgroup": subgroup,
                          "rows": int((membership & (years == year)).sum()), "common_active_rows": int(mask.sum())}
            counts = tensor_numpy(weights[:, mask].sum(axis=1)).astype(float)
            replicates[(int(year), subgroup, "common_active_log_support")] = np.log1p(counts)
            output.append(metric_record(row_common, "common_active_support", counts))
            output.append(metric_record(row_common, "common_active_absolute_direction", values))
            replicates[(int(year), subgroup, "common_active_absolute_direction")] = values
            means = aggregate_rows(context, weights, mask, device)
            for position, metric in enumerate(("common_active_brier", "common_active_mortality", "common_active_mean_risk")):
                output.append(metric_record(row_common, metric, means[:, position]))
                replicates[(int(year), subgroup, metric)] = means[:, position]
    return output, replicates


def original_fingerprints(study_root: Any, group_freeze: Any) -> dict:
    """Snapshot original sources and data-directory listings without modifying them."""
    from pathlib import Path
    import hashlib
    import json
    root = Path(study_root)
    sources = [root / "concept_agreement_study.ipynb", root / "concept_agreement_study.executed.ipynb",
               root / "cmr_common.py", root / "concept_subgroup_drift_pilot.py",
               root / "concept_subgroup_drift_pilot.ipynb", Path(group_freeze)]
    sources += sorted((root / "concept_study").glob("*.py"))
    hashes = {str(path): file_sha256(path) for path in sources if path.is_file()}
    listings = {}
    directories = [root / "stats/concept_agreement_study", root / "stats/temporal_robustness/_population"]
    for directory in directories:
        listing = [(str(path.relative_to(directory)), path.stat().st_size, path.stat().st_mtime_ns)
                   for path in sorted(directory.rglob("*")) if path.is_file()]
        listings[str(directory)] = {"method": "listing", "files": len(listing),
                                   "sha256": hashlib.sha256(json.dumps(listing).encode()).hexdigest()}
    return {"source_sha256": hashes, "data_listings": listings}


def check_reference_representation(records: list[dict], study_root: Any, tolerance: float = 1e-6) -> list[dict]:
    """Reproduce November reference-year full/A errors before trusting later years."""
    from pathlib import Path
    import pandas as pd
    published = Path(study_root).parent / "rpi-runs/SAE-TCAV/concept-measurement-reliability/20261103-v7-evaluate/concept_factor_groups/public/table-001.csv"
    if not published.exists():
        raise FileNotFoundError("reference group evaluation table is required")
    previous = pd.read_csv(published)
    checks = []
    for row in records:
        if row["metric"] != "E" or row["year"] != row["reference"] or row["subgroup"] == "R":
            continue
        stratum = "full" if row["subgroup"] == "all" else "original_high"
        selected = previous[(previous.concept == row["family"]) & (previous.reference == row["reference"]) &
                            (previous.pair == row["pair"]) & (previous.kind == row["kind"]) & (previous.stratum == stratum)]
        # A family string can appear in both context variants; aliases identify its build.
        selected = selected[selected.aliases.str.split(",").apply(lambda aliases: row["build"] in aliases or
                            (row["build"] == "r1" and "last_3" in aliases))]
        if len(selected) != 1:
            raise ValueError(f"reference table join is not unique for {row['concept']} {row['pair']}")
        expected = selected.iloc[0].E
        actual = row["value"]
        if actual is None and not np.isfinite(expected):
            difference = None
        else:
            if actual is None or not np.isfinite(expected):
                raise ValueError("reference disagreement definedness mismatch")
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            difference = float(abs(actual - expected))
        checks.append({"concept": row["concept"], "pair": row["pair"], "kind": row["kind"],
                       "subgroup": row["subgroup"], "absolute_difference": difference, "status": "passed"})
    return checks


def write_tables(output_dir: Any, tables: dict[str, list[dict]], max_output_gib: float = 2) -> None:
    """Write scalar aggregate tables in bounded chunks without crossing the ceiling."""
    from pathlib import Path
    import pandas as pd
    root = Path(output_dir)
    total_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    ceiling = int(max_output_gib * 2 ** 30)
    for name, records in tables.items():
        fields = sorted(set().union(*(record.keys() for record in records))) if records else []
        with (root / f"{name}.csv").open("wb") as handle:
            for first in range(0, max(1, len(records)), 10000):
                clean = scalar_records(records[first:first + 10000], fields)
                payload = pd.DataFrame(clean, columns=fields).to_csv(index=False, header=first == 0).encode()
                if total_bytes + len(payload) > ceiling:
                    raise OSError("aggregate tables would exceed the configured artifact ceiling")
                handle.write(payload)
                total_bytes += len(payload)



def budgeted_npz(path: Any, arrays: Mapping[str, Any], byte_limit: int) -> None:
    """Reject an archive write before it crosses its remaining disk allowance."""
    from pathlib import Path
    if byte_limit < 1:
        raise OSError("no artifact budget remains for private replicate archive")
    class LimitedArchive:
        def __init__(self, handle):
            self.handle = handle

        def write(self, payload):
            if self.handle.tell() + len(payload) > byte_limit:
                raise OSError("private replicate archive would exceed artifact allowance")
            return self.handle.write(payload)

        def __getattr__(self, name):
            return getattr(self.handle, name)
    import zipfile
    with Path(path).open("w+b") as handle:
        with zipfile.ZipFile(LimitedArchive(handle), mode="w", compression=zipfile.ZIP_DEFLATED,
                             allowZip64=True) as archive:
            for name, values in arrays.items():
                with archive.open(f"{name}.npy", mode="w", force_zip64=True) as member:
                    np.lib.format.write_array(member, np.asanyarray(values), allow_pickle=False)

def execute_analysis(arguments: Any) -> dict:
    """Run inventory, bounded smoke or the approved complete temporal evaluation."""
    import datetime
    import importlib.metadata
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import time

    root = Path(arguments.study_root).expanduser().resolve()
    output = Path(arguments.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError("choose a new results directory; existing results are read-only")
    if not output.is_relative_to(root / "analysis_outputs/concept_groups_over_time"):
        raise ValueError("new results must be under repository analysis_outputs/concept_groups_over_time")
    if arguments.last_year > 2015 or arguments.outcome_last_year > 2014:
        raise ValueError("observation/outcome boundaries may not exceed 2015/2014")
    if arguments.command == "smoke" and (arguments.max_units > 3 or arguments.max_seconds > 300):
        raise ValueError("smoke is limited to three fixed cases and five minutes")
    if arguments.max_seconds > 7200 or arguments.max_output_gib > 2:
        raise ValueError("approved ceilings are two hours and 2 GiB")
    if shutil.disk_usage(root).free < arguments.max_output_gib * 2 ** 30:
        raise OSError("insufficient free space for the configured output ceiling")
    output.mkdir(parents=True)
    start = time.monotonic()
    device = resolve_device(arguments.device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sources = original_fingerprints(root, arguments.group_freeze)
    config = vars(arguments).copy()
    command = " ".join(sys.argv)
    import hashlib
    manifest = {
        "status": "running", "smoke": arguments.command == "smoke", "command": command,
        "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
        "source_sha256": file_sha256(__file__), "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "requested_device": arguments.device, "resolved_device": device,
        "python": sys.version, "executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas", "scipy")},
        "seed": arguments.seed, "bootstrap_draws": arguments.bootstrap_draws,
        "launcher": None, "source_snapshot_kind": "authorized_uncommitted_files",
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name() if device == "cuda" else None,
        "tf32": torch.backends.cuda.matmul.allow_tf32, "amp": "off",
    }
    (output / "source").mkdir()
    shutil.copyfile(__file__, output / "source/concept_groups_over_time.py")
    notebook = root / "concept_groups_over_time.ipynb"
    if notebook.exists():
        shutil.copyfile(notebook, output / "source/concept_groups_over_time.ipynb")
    atomic_json(output / "run-manifest.json", manifest)
    tables = {name: [] for name in ("inventory", "annual", "representation", "members", "paired", "transitions",
                                   "fixed_cohort", "families", "associations", "summary")}
    stages, reference_checks = [], []
    utilization_samples = []
    import threading
    sample_stop = threading.Event()
    def sample_gpu():
        while not sample_stop.is_set():
            try:
                text = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                                "--format=csv,noheader,nounits"], text=True, timeout=3)
                utilization_samples.append({"elapsed_seconds": time.monotonic() - start, "gpu_utilization_and_memory_mib": text.strip()})
            except (OSError, subprocess.SubprocessError):
                utilization_samples.append({"elapsed_seconds": time.monotonic() - start, "status": "unavailable"})
            sample_stop.wait(1.0)
    sampler = threading.Thread(target=sample_gpu, daemon=True) if device == "cuda" else None
    if sampler is not None:
        sampler.start()
    def guard():
        if time.monotonic() - start > arguments.max_seconds:
            raise TimeoutError("configured analysis wall-time ceiling reached")
        size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if size > arguments.max_output_gib * 2 ** 30:
            raise OSError("configured analysis artifact-size ceiling reached")
    try:
        index, freeze, inventory = source_inventory(root, arguments.group_freeze)
        tables["inventory"] = inventory
        if arguments.command == "inventory":
            write_tables(output, tables, arguments.max_output_gib)
            manifest["inventory_concepts"] = len(index["concepts"])
            manifest["matched_families"] = len({row["concept"] for row in inventory if row["status"] == "available"})
            manifest["matched_pairs"] = sum(row["status"] == "available" for row in inventory)
        else:
            # Source modules are imported without executing notebook orchestration.
            sys.path.insert(0, str(root))
            from concept_study import loading, measures, paths
            cached_ids = [measures.load_build(record)["patient_id"] for record in index["builds"]]
            unique_ids, global_weights = global_patient_bootstrap(cached_ids, arguments.bootstrap_draws, arguments.seed)
            inputs = loading.open_inputs(device)
            available = [item for item in inputs["inventory"] if any(
                row["concept"] == loading.concept_key(item["build"], item["reference"], item["family"]) and row["status"] == "available"
                for row in inventory)]
            validate_inventory_coverage(
                (loading.concept_key(item["build"], item["reference"], item["family"]) for item in available), inventory)
            if arguments.command == "smoke":
                # Structural choice: the earliest available family in distinct builds, then fill deterministically.
                chosen, builds = [], set()
                for item in available:
                    if item["build"] not in builds and len(chosen) < arguments.max_units:
                        chosen.append(item)
                        builds.add(item["build"])
                available = chosen
            family_replicates, representation_replicates, metadata = {}, {}, {}
            current, base, row_positions, ids, years, gradients, probability, labels, row_weights = (None,) * 9
            for item in available:
                guard()
                build_key = (item["build"], item["reference"])
                stage_start = time.monotonic()
                if current != build_key:
                    base = loading.load_build(inputs, *build_key, item["roles"])
                    row_positions = eligible_evaluation_rows(base, item["reference"], arguments.last_year)
                    validate_raw_build(base, row_positions, arguments.outcome_last_year)
                    ids = np.asarray(base["patients"])[row_positions].astype(str)
                    years = np.asarray(base["years"])[row_positions]
                    validate_patient_years(ids, years)
                    build_record = next(record for record in index["builds"] if (record["build"], record["reference"]) == build_key)
                    cached = measures.load_build(build_record)
                    # The source cache includes all eligible rows through 2015; compare before optional truncation.
                    full_rows = eligible_evaluation_rows(base, item["reference"], 2015)
                    validate_source_alignment(base, full_rows, cached)
                    gradients = np.asarray(base["gradients"])[row_positions]
                    probability = np.asarray(base["probabilities"])[row_positions, 1]
                    labels = protected_labels(np.asarray(base["labels"])[row_positions], years, arguments.outcome_last_year)
                    row_weights = torch.as_tensor(row_bootstrap_weights(ids, unique_ids, global_weights),
                                                  dtype=torch.float64, device=device)
                    device_codes = {seed: torch.as_tensor(np.asarray(base["codes"][seed])[row_positions], dtype=torch.float64, device=device)
                                    for seed in (42, 135, 215)}
                    device_decoders = {seed: torch.as_tensor(base["decoders"][seed], dtype=torch.float64, device=device)
                                       for seed in (42, 135, 215)}
                    device_scale = torch.as_tensor(base["scale"], dtype=torch.float64, device=device)
                    device_gradients = torch.as_tensor(gradients, dtype=torch.float64, device=device)
                    current = build_key
                unit = loading.concept_unit(inputs, base, item)
                concept = unit["key"]
                meta = {"family": item["family"], "build": item["build"], "reference": item["reference"],
                        "variants": ",".join(item["variants"])}
                metadata[concept] = meta
                masks = {"all": np.ones(len(ids), dtype=bool), "A": np.asarray(unit["activation"])[row_positions] > unit["cutoff"],
                         "R": np.asarray(unit["rule_mask"])[row_positions].astype(bool)}
                prepared = {}
                contribution_cache = {}
                common_active = np.ones(len(ids), dtype=bool)
                for seeds in ((42, 135), (42, 215), (135, 215)):
                    pair = f"{seeds[0]}-{seeds[1]}"
                    entry = freeze.get(f"{concept}/{pair}")
                    if entry is None:
                        continue
                    verify_identity(unit["identity"], entry["identity"])
                    candidates = {}
                    for kind, groups in frozen_candidates(entry):
                        if groups is None:
                            continue
                        sides = []
                        for seed, factors in zip(seeds, groups):
                            cache_key = (seed, tuple(factors))
                            if cache_key not in contribution_cache:
                                contribution_cache[cache_key] = contribution_measures(
                                    device_codes[seed], device_decoders[seed], device_scale, device_gradients, factors, device=device)
                            sides.append(contribution_cache[cache_key])
                        candidates[kind] = sides
                        if kind in ("selected", "singleton"):
                            for side in sides:
                                common_active &= tensor_numpy(side["norm"] > 1e-12)
                    prepared[pair] = (entry, candidates)
                for pair, (entry, candidates) in prepared.items():
                    guard()
                    common = {**meta, "concept": concept, "pair": pair}
                    representation, repr_values = representation_measures(common, candidates, years, masks, row_weights, device)
                    tables["representation"].extend(representation)
                    for key, value in repr_values.items():
                        year, subgroup, kind, metric = key
                        representation_replicates[(concept, pair, year, subgroup, kind, metric)] = value
                    reference_checks.extend(check_reference_representation(representation, root))
                    candidate_groups = dict(frozen_candidates(entry))
                    for kind, sides in candidates.items():
                        for position, result in enumerate(sides):
                            side = ("left", "right")[position]
                            side_common = {**common, "kind": kind, "side": side}
                            annual, annual_values = annual_measures(side_common, result, device_gradients, years, masks,
                                                                   probability, labels, row_weights, device)
                            normalized, normalized_values = common_direction_measures(side_common, result, common_active, years,
                                                                                     masks, probability, labels, row_weights, device)
                            annual_values.update(normalized_values)
                            tables["annual"].extend(annual + normalized)
                            for (year, subgroup, metric), values in annual_values.items():
                                family_replicates[(concept, pair, kind, side, year, subgroup, metric)] = values
                            tables["members"].extend(member_measures(side_common, result, candidate_groups[kind][position], years,
                                                                      masks, row_weights, device))
                            paired, transitions, fixed = temporal_measures(side_common, result, gradients, ids, years, masks,
                                                                          row_weights, device, item["reference"])
                            tables["paired"].extend(paired)
                            tables["transitions"].extend(transitions)
                            tables["fixed_cohort"].extend(fixed)
                if device == "cuda":
                    torch.cuda.synchronize()
                stages.append({"concept": concept, "seconds": time.monotonic() - stage_start, "device": device,
                               "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
                               "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else 0})
                atomic_json(output / "stages.json", stages)
                LOGGER.info("completed %s (%d pairs)", concept, len(prepared))
            tables["families"], family_values = family_aggregation(family_replicates, metadata)
            tables["associations"] = association_records(family_values, metadata, arguments.outcome_last_year)
            tables["summary"] = representation_summary(representation_replicates, metadata)
            manifest["input_sha256"] = dict(inputs["ctx"]["input_files"])
            for source_path, expected in manifest["input_sha256"].items():
                if file_sha256(source_path) != expected:
                    raise ValueError("consumed frozen input changed during analysis")
            manifest["completed_families"] = len(metadata)
            manifest["completed_pairs"] = len({(key[0], key[1]) for key in representation_replicates})
            manifest["inventory_concepts"] = len(index["concepts"])
            manifest["matched_families"] = len({row["concept"] for row in inventory if row["status"] == "available"})
            manifest["matched_pairs"] = sum(row["status"] == "available" for row in inventory)
            if arguments.command == "evaluate" and (manifest["completed_families"] != manifest["matched_families"] or
                                                       manifest["completed_pairs"] != manifest["matched_pairs"]):
                raise ValueError("full evaluation did not complete every matched source family and pair")
            guard()
            # Reserve 32 MiB for bounded run metadata (at most 7200 utilization samples,
            # 2760 reference checks, source hashes and final manifests). The archive
            # and CSV writers each refuse bytes before crossing the remaining cap.
            metadata_reserve = 32 * 2 ** 20
            ceiling = int(arguments.max_output_gib * 2 ** 30)
            used_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
            private = output / "private"
            private.mkdir(mode=0o700)
            budgeted_npz(private / "family_replicates.npz", {
                "|".join(map(str, key)): value for key, value in family_values.items()
            }, ceiling - used_bytes - metadata_reserve)
            write_tables(output, tables, (ceiling - metadata_reserve) / 2 ** 30)
            atomic_json(output / "reference_checks.json", reference_checks)
        final_sources = original_fingerprints(root, arguments.group_freeze)
        atomic_json(output / "source_integrity.json", {"before": sources, "after": final_sources, "unchanged": sources == final_sources})
        if final_sources != sources:
            raise ValueError("original source or data fingerprint changed during extension")
        guard()
        manifest["status"] = "completed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        sample_stop.set()
        if sampler is not None:
            sampler.join(timeout=4)
        atomic_json(output / "gpu_utilization.json", utilization_samples)
        if not (output / "source_integrity.json").exists():
            final_sources = original_fingerprints(root, arguments.group_freeze)
            atomic_json(output / "source_integrity.json", {"before": sources, "after": final_sources, "unchanged": sources == final_sources})
        manifest["elapsed_seconds"] = time.monotonic() - start
        manifest["end_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(output / "run-manifest.json", manifest)
    return manifest


def main() -> None:
    """Command-line entry point with explicit source, device and resource choices."""
    import argparse
    import datetime
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "smoke", "evaluate"))
    parser.add_argument("--study-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--group-freeze", default=str(Path.home() / "rpi-runs/SAE-TCAV/concept-measurement-reliability/20261103-v7-develop/concept_factor_groups/private/freeze.pkl"))
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-units", type=int, default=3)
    parser.add_argument("--max-seconds", type=int, default=300)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--last-year", type=int, default=2015)
    parser.add_argument("--outcome-last-year", type=int, default=2014)
    parser.add_argument("--max-output-gib", type=float, default=2)
    parser.add_argument("--smoke", action="store_true", help="use the same bounded three-family smoke path")
    arguments = parser.parse_args()
    if arguments.smoke:
        arguments.command = "smoke"
    if arguments.output_dir is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        arguments.output_dir = str(Path(arguments.study_root) / "analysis_outputs/concept_groups_over_time" / f"{timestamp}-{arguments.command}")
    if arguments.bootstrap_draws < 1 or arguments.max_units < 1 or arguments.max_seconds < 1 or arguments.max_output_gib <= 0:
        parser.error("bootstrap draws, units and resource limits must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = execute_analysis(arguments)
    print(f"{manifest['status']}: {arguments.output_dir}")


if __name__ == "__main__":
    main()
