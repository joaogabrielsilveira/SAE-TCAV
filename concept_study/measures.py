"""Concept, influence and performance measures per concept, patient group and year.

Every measure is a ratio of weighted sums over patient rows. The sums are taken with
the joint bootstrap weights, so replicate 0 (all weights one) is the point estimate and
replicates 1..B give the patient intervals, all in one GPU pass per concept.

Cells are (group, year, subset). Subset 0 is every evaluation patient of that year;
subset k is only the patients who also have a row in year YEARS[k-1]. The subsets give
the "returning patient" versions used by the composition split. The split is stored only
for measures that use no outcomes: the returning patients at the start year are those still
present at the end year, so patients who die in the start year are never among them, and an
outcome measure (Brier, mortality, Death-F1, removal Brier change, ...) would be biased by survival.

Cached replicate files carry the hash of index.json they were computed from; a file built
from another patient table is refused on load and recomputed by compute_all.
"""
import functools
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from concept_study import bootstrap, paths

YEARS = tuple(range(2007, paths.LAST_YEAR + 1))
GROUPS = paths.GROUPS
SUBSETS = ("all",) + tuple(f"present_in_{year}" for year in YEARS)
BASE_COLUMNS = ("count", "labelled", "deaths", "risk_labelled", "squared_error", "squared_error_constant",
                "predicted_death", "predicted_and_died", "risk", "activation", "reconstruction_error",
                "removal_risk_change", "removal_brier_change", "removal_brier_outcome_free", "removal_brier_death_part")
GROUP_MEASURES = ("n", "deaths", "mortality", "brier", "brier_skill", "death_f1", "calibration_gap", "mean_risk",
                  "mean_activation", "reconstruction_error", "removal_risk_change", "removal_brier_change",
                  "removal_brier_outcome_free", "removal_brier_death_part")
CONCEPT_MEASURES = ("agreement", "rule_recall", "rule_f2", "overlap", "activation_rate", "rule_rate", "n_A", "n_R")
INFLUENCE_MEASURES = ("signed_sensitivity", "absolute_sensitivity", "tcav", "share_fits_positive")
PROFILE_GROUPS = ("A", "R", "all")
OUTCOME_MEASURES = ("deaths", "mortality", "brier", "brier_skill", "death_f1", "calibration_gap",
                    "removal_brier_change", "removal_brier_outcome_free", "removal_brier_death_part")
SPLIT_GROUP_MEASURES = tuple(name for name in GROUP_MEASURES if name not in OUTCOME_MEASURES)


# ---------------------------------------------------------------- loading

def load_index() -> dict:
    return json.loads((paths.PATIENT_TABLE / "index.json").read_text())


def index_hash() -> str:
    """Hash of index.json; stored in derived cache files to detect a rebuilt patient table."""
    return hashlib.sha256((paths.PATIENT_TABLE / "index.json").read_bytes()).hexdigest()[:16]


def save_atomic(path: Path, **arrays) -> None:
    """np.savez under a temporary name, then rename (readers never see a partial file)."""
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def stored_hash(path: Path) -> str:
    """index_hash stored in a derived cache file ('' when missing)."""
    with np.load(path) as stored:
        return str(stored["index_hash"]) if "index_hash" in stored.files else ""


def load_build(record: dict) -> dict:
    with np.load(paths.PATIENT_TABLE / record["file"]) as stored:
        return {name: stored[name] for name in stored.files}


def load_concept(record: dict) -> dict:
    with np.load(paths.PATIENT_TABLE / "concepts" / record["file"]) as stored:
        return {name: stored[name] for name in stored.files}


def build_record(index: dict, concept: dict) -> dict:
    return next(b for b in index["builds"] if b["build"] == concept["build"] and b["reference"] == concept["reference"])


def death_threshold(build: dict) -> float:
    """Frozen Death-F1 threshold of the variants served by the build; they must agree."""
    values = {build["death_threshold"][variant] for variant in build["variants"]}
    if len(values) != 1:
        raise ValueError(f"{build['build']} {build['reference']}: variants disagree on the Death-F1 threshold {values}")
    return values.pop()


def all_patient_ids(index: dict) -> np.ndarray:
    ids = [load_build(record)["patient_id"] for record in index["builds"]]
    return np.unique(np.concatenate(ids).astype(str))


# ---------------------------------------------------------------- per-row inputs

def group_masks(concept: dict) -> np.ndarray:
    """(rows, groups) membership in A, R, the four agreement groups and the whole cohort."""
    a, r = concept["in_A"], concept["in_R"]
    columns = {"A": a, "R": r, "R+A+": r & a, "R+A-": r & ~a, "R-A+": ~r & a, "neither": ~r & ~a,
               "all": np.ones_like(a)}
    return np.stack([columns[name] for name in GROUPS], axis=1)


def presence(build: dict) -> np.ndarray:
    """(rows, years) True when the row's patient has an evaluation row in that year."""
    present = np.zeros((len(build["year"]), len(YEARS)), dtype=bool)
    for k, year in enumerate(YEARS):
        ids = set(build["patient_id"][build["year"] == year])
        present[:, k] = np.isin(build["patient_id"], list(ids))
    return present


def cell_masks(concept: dict, build: dict) -> np.ndarray:
    """(rows, groups * years * subsets) masks in that order."""
    groups = group_masks(concept)
    years = np.stack([build["year"] == year for year in YEARS], axis=1)
    subsets = np.concatenate([np.ones((len(years), 1), dtype=bool), presence(build)], axis=1)
    cells = groups[:, :, None, None] & years[:, None, :, None] & subsets[:, None, None, :]
    return cells.reshape(len(years), -1)


def base_values(concept: dict, build: dict, seed: int, threshold: float) -> np.ndarray:
    """(rows, BASE_COLUMNS) per-row values; missing labels contribute zero and are not counted."""
    label = build["label"].astype(np.float64)
    labelled = np.isfinite(label)
    y = np.where(labelled, label, 0.0)
    p = build["probability"].astype(np.float64)
    constant = float(build["reference_mortality"])
    predicted = (p >= threshold) & labelled
    removal_brier = np.where(labelled, concept["removal_brier_change"], 0.0)
    # Removal Brier change split exactly in two: (q - y)^2 - (p - y)^2 = (q^2 - p^2) - 2y(q - p),
    # with q the probability after removal. The first part uses no outcomes; the second only deaths.
    q = p + concept["removal_probability_change"].astype(np.float64)
    outcome_free = (q ** 2 - p ** 2) * labelled
    death_part = -2 * y * (q - p) * labelled
    columns = {
        "count": np.ones_like(p), "labelled": labelled.astype(float), "deaths": y,
        "risk_labelled": p * labelled, "squared_error": (p - y) ** 2 * labelled,
        "squared_error_constant": (constant - y) ** 2 * labelled,
        "predicted_death": predicted.astype(float), "predicted_and_died": predicted * y,
        "risk": p, "activation": concept["activation"], "reconstruction_error": build[f"reconstruction_error_{seed}"],
        "removal_risk_change": concept["removal_probability_change"], "removal_brier_change": removal_brier,
        "removal_brier_outcome_free": outcome_free, "removal_brier_death_part": death_part,
    }
    return np.stack([columns[name] for name in BASE_COLUMNS], axis=1)


def sensitivity_values(concept: dict, estimator: str, probability: np.ndarray = None) -> tuple:
    """(rows, 3 * fits) signed, absolute and positive-indicator values of the valid fits.

    The stored sensitivity is the gradient of the predicted probability along the CAV. An estimator
    ending in "_logit" uses the same fits on the log-odds scale: gradient / (p * (1 - p)) per row,
    which removes the part of the probability gradient that only reflects the patient's predicted risk.
    """
    logit = estimator.endswith("_logit")
    name = f"sensitivity_{estimator[:-len('_logit')] if logit else estimator}"
    if name not in concept:
        return None, 0
    s = concept[name]
    s = s[:, np.isfinite(s).all(axis=0)]
    if logit:
        p = probability.astype(np.float64)
        s = (s / (p * (1 - p))[:, None]).astype(np.float32)
    return np.concatenate([s, np.abs(s), (s > 0).astype(np.float32)], axis=1), s.shape[1]


# ---------------------------------------------------------------- ratios

def ratio(numerator, denominator):
    """numerator / denominator, NaN where the denominator is zero (works for arrays and tensors)."""
    if isinstance(numerator, torch.Tensor):
        return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-30),
                           torch.full_like(numerator, float("nan")))
    with np.errstate(all="ignore"):
        return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1), np.nan)


def group_measures(sums: np.ndarray) -> dict:
    """GROUP_MEASURES from base sums (..., BASE_COLUMNS)."""
    s = {name: sums[..., i] for i, name in enumerate(BASE_COLUMNS)}
    mortality = ratio(s["deaths"], s["labelled"])
    brier = ratio(s["squared_error"], s["labelled"])
    constant = ratio(s["squared_error_constant"], s["labelled"])
    return {
        "n": s["count"], "deaths": s["deaths"], "mortality": mortality, "brier": brier,
        "brier_skill": 1 - ratio(brier, constant),
        "death_f1": ratio(2 * s["predicted_and_died"], s["predicted_death"] + s["deaths"]),
        "calibration_gap": ratio(s["risk_labelled"], s["labelled"]) - mortality,
        "mean_risk": ratio(s["risk"], s["count"]),
        "mean_activation": ratio(s["activation"], s["count"]),
        "reconstruction_error": ratio(s["reconstruction_error"], s["count"]),
        "removal_risk_change": ratio(s["removal_risk_change"], s["count"]),
        "removal_brier_change": ratio(s["removal_brier_change"], s["labelled"]),
        "removal_brier_outcome_free": ratio(s["removal_brier_outcome_free"], s["labelled"]),
        "removal_brier_death_part": ratio(s["removal_brier_death_part"], s["labelled"]),
    }


def concept_measures(counts: np.ndarray) -> dict:
    """CONCEPT_MEASURES from group counts (..., groups) of one year and subset."""
    n = {name: counts[..., i] for i, name in enumerate(GROUPS)}
    both, rule_only, activation_only = n["R+A+"], n["R+A-"], n["R-A+"]
    return {
        "agreement": ratio(both, n["R"]),
        "rule_recall": ratio(both, n["A"]),
        "rule_f2": ratio(5 * both, 5 * both + 4 * activation_only + rule_only),
        "overlap": ratio(both, both + rule_only + activation_only),
        "activation_rate": ratio(n["A"], n["all"]),
        "rule_rate": ratio(n["R"], n["all"]),
        "n_A": n["A"], "n_R": n["R"],
    }


def median_over_fits(values: torch.Tensor) -> torch.Tensor:
    """Median over the last axis (average of the two middle values, as numpy.median)."""
    ordered = torch.sort(values, dim=-1).values
    k = values.shape[-1]
    return (ordered[..., (k - 1) // 2] + ordered[..., k // 2]) / 2


def influence_measures(sums: torch.Tensor, counts: torch.Tensor, fits: int) -> tuple:
    """Median-over-fits influence measures and the per-fit point values.

    sums: (replicates, cells, 3 * fits) of signed, absolute, positive; counts: (replicates, cells).
    """
    per_fit = ratio(sums, counts[..., None].expand_as(sums))
    signed, absolute, positive = per_fit[..., :fits], per_fit[..., fits:2 * fits], per_fit[..., 2 * fits:]
    summary = {
        "signed_sensitivity": median_over_fits(signed),
        "absolute_sensitivity": median_over_fits(absolute),
        "tcav": median_over_fits(positive),
        "share_fits_positive": torch.where(torch.isfinite(signed[..., 0]), (signed > 0).float().mean(-1),
                                           torch.full_like(signed[..., 0], float("nan"))),
    }
    point = {"signed_sensitivity": signed[0], "absolute_sensitivity": absolute[0], "tcav": positive[0]}
    return ({k: v.cpu().numpy() for k, v in summary.items()}, {k: v.cpu().numpy() for k, v in point.items()})


def profile_shift(code_sums: np.ndarray, counts: np.ndarray, reference_position: int) -> np.ndarray:
    """1 - cosine(group mean SAE activation profile in a year, same group's profile in the reference year).

    code_sums: (replicates, groups, years, factors); counts: (replicates, groups, years).
    """
    profile = ratio(code_sums, counts[..., None])
    reference = profile[:, :, reference_position:reference_position + 1, :]
    dot = np.nansum(profile * reference, axis=-1)
    norms = np.linalg.norm(np.nan_to_num(profile), axis=-1) * np.linalg.norm(np.nan_to_num(reference), axis=-1)
    shift = 1 - ratio(dot, norms)
    shift[counts <= 0] = np.nan
    return shift


# ---------------------------------------------------------------- one concept, every replicate

def _cells(array):
    """Reshape (replicates, groups * years * subsets, ...) to (replicates, groups, years, subsets, ...)."""
    return array.reshape((array.shape[0], len(GROUPS), len(YEARS), len(SUBSETS)) + array.shape[2:])


def transition_effects(concept: dict, build: dict, weights: np.ndarray, estimator: str, pairs: list) -> dict:
    """Among patients in R at both ends of a consecutive pair: sensitivity change of those who
    stopped strongly activating (A 1->0) minus those who kept activating (A 1->1).

    Returns arrays (replicates, pairs) for the difference in signed and absolute sensitivity change,
    activation change and predicted-risk change, and the point counts of both transition groups.
    """
    from concept_study import decomposition
    values, fits = sensitivity_values(concept, estimator, build["probability"])
    risk = build["probability"].astype(np.float64)
    result = {name: np.full((len(weights), len(pairs)), np.nan) for name in
              ("signed_difference", "absolute_difference", "activation_difference", "risk_difference")}
    result.update(stopped_n=np.zeros(len(pairs)), kept_n=np.zeros(len(pairs)))
    if values is None:
        return result
    signed = values[:, :fits]
    for p, (start, end) in enumerate(pairs):
        if end != start + 1:
            continue
        table = decomposition.transition_table(build["patient_id"], build["year"], concept["in_A"],
                                               concept["in_R"], start, end)
        stayed_in_rule = table["R_start"] & table["R_end"] & table["A_start"]
        stopped, kept = stayed_in_rule & ~table["A_end"], stayed_in_rule & table["A_end"]
        result["stopped_n"][p], result["kept_n"][p] = stopped.sum(), kept.sum()
        if not stopped.any() or not kept.any():
            continue
        first, second = table["start_rows"], table["end_rows"]
        change = np.concatenate([signed[second] - signed[first],
                                 np.abs(signed[second]) - np.abs(signed[first]),
                                 (concept["activation"][second] - concept["activation"][first])[:, None],
                                 (risk[second] - risk[first])[:, None]], axis=1)
        masks = np.stack([stopped, kept], axis=1)
        sums = bootstrap.weighted_sums(weights[:, first], masks, np.concatenate([np.ones((len(first), 1)), change], 1))
        means = ratio(sums[..., 1:], sums[..., :1].expand_as(sums[..., 1:]))
        difference = means[:, 0] - means[:, 1]
        result["signed_difference"][:, p] = median_over_fits(difference[:, :fits]).cpu().numpy()
        result["absolute_difference"][:, p] = median_over_fits(difference[:, fits:2 * fits]).cpu().numpy()
        result["activation_difference"][:, p] = difference[:, -2].cpu().numpy()
        result["risk_difference"][:, p] = difference[:, -1].cpu().numpy()
    return result


def compute_concept(index: dict, record: dict, unique_ids: np.ndarray, patient_weight_matrix: np.ndarray) -> dict:
    """Every measure of one concept for the point estimate (replicate 0) and all bootstrap replicates."""
    from concept_study import decomposition
    build_info = build_record(index, record)
    build, concept = load_build(build_info), load_concept(record)
    weights = bootstrap.row_weights(build["patient_id"], unique_ids, patient_weight_matrix)
    masks = cell_masks(concept, build)
    reference_position = YEARS.index(record["reference"])
    pairs = decomposition.comparison_pairs(record["reference"], YEARS, paths.LAST_YEAR)
    out = {"pairs": np.array(pairs)}

    base = _cells(bootstrap.weighted_sums(weights, masks, base_values(
        concept, build, record["seed"], death_threshold(build_info))).cpu().numpy())
    counts = base[..., 0]                                        # (replicates, groups, years, subsets)
    for name, value in group_measures(base).items():
        out[f"group|{name}"] = value[..., 0]
        if name in SPLIT_GROUP_MEASURES:
            out[f"split|group|{name}"] = decomposition.split_change(value, YEARS, pairs)
    for name, value in concept_measures(np.moveaxis(counts, 1, -1)).items():
        out[f"concept|{name}"] = value[..., 0]
        out[f"split|concept|{name}"] = decomposition.split_change(value, YEARS, pairs)

    for estimator in paths.INFLUENCE_ESTIMATORS:
        values, fits = sensitivity_values(concept, estimator, build["probability"])
        if values is None:
            continue
        sums = bootstrap.weighted_sums(weights, masks, values)
        summary, per_fit = influence_measures(sums, torch.as_tensor(counts.reshape(len(weights), -1), device=sums.device), fits)
        del sums
        for name, value in summary.items():
            value = _cells(value)
            out[f"influence|{estimator}|{name}"] = value[..., 0]
            out[f"split|influence|{estimator}|{name}"] = decomposition.split_change(value, YEARS, pairs)
        for name, value in per_fit.items():
            out[f"per_fit|{estimator}|{name}"] = value.reshape(len(GROUPS), len(YEARS), len(SUBSETS), fits)[:, :, 0]
        for name, value in transition_effects(concept, build, weights, estimator, pairs).items():
            out[f"transition|{estimator}|{name}"] = value

    codes = build[f"codes_{record['seed']}"].astype(np.float32)
    profile_masks = group_masks(concept)[:, [GROUPS.index(g) for g in PROFILE_GROUPS]]
    year_masks = np.stack([build["year"] == year for year in YEARS], axis=1)
    cells = (profile_masks[:, :, None] & year_masks[:, None, :]).reshape(len(codes), -1)
    code_sums = bootstrap.weighted_sums(weights, cells, codes).cpu().numpy()
    code_sums = code_sums.reshape(len(weights), len(PROFILE_GROUPS), len(YEARS), -1)
    profile_counts = counts[..., 0][:, [GROUPS.index(g) for g in PROFILE_GROUPS]]
    out["profile|shift"] = profile_shift(code_sums, profile_counts, reference_position)
    return {name: np.asarray(value, dtype=np.float32) for name, value in out.items()}


def compute_all(force: bool = False) -> list:
    """Compute and cache every concept's replicate file (again when built from another index.json).
    Returns the list of concept records."""
    index, current = load_index(), index_hash()
    paths.REPLICATES.mkdir(parents=True, exist_ok=True)
    unique_ids, weight_matrix = bootstrap.patient_weights(all_patient_ids(index))
    for record in index["concepts"]:
        target = paths.REPLICATES / record["file"]
        if target.exists() and not force and stored_hash(target) == current:
            continue
        save_atomic(target, index_hash=current, **compute_concept(index, record, unique_ids, weight_matrix))
    _replicates_from_file.cache_clear()
    return index["concepts"]


@functools.lru_cache(maxsize=None)
def _replicates_from_file(file_name: str) -> dict:
    with np.load(paths.REPLICATES / file_name) as stored:
        if "index_hash" not in stored.files or str(stored["index_hash"]) != index_hash():
            raise ValueError(f"{file_name}: replicates were built from another patient table; run measures.compute_all()")
        return {name: stored[name] for name in stored.files if name != "index_hash"}


def load_replicates(record: dict) -> dict:
    """Cached replicate arrays of one concept (read from disk once per session)."""
    return _replicates_from_file(record["file"])
