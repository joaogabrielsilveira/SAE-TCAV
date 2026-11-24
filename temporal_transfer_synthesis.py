"""Inputs and readers for the temporal transfer synthesis: hash gate, Ideas export readers,
status rows and factor coverage. Reuses the Work 3-4 helpers in concept_regions (cr).
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import concept_candidates as cc
import concept_regions as cr

TOP_K = 3
DEFAULT_ENCODING_BATCH = 256
EXPORT_FOLDERS = {"r1": "baseline", "all_history": "all_history", "last_3": "last_3"}
AMBIGUOUS = "candidate split/ambiguity"
UNMATCHED = "unmatched"
SINGLE = "single candidate"
NO_REFERENCE_MISSES = "no reference misses"
NO_RULE_PATIENTS = "undefined (no rule patients that year)"
ALL_LATER_NEW = "all later misses are new"
REGION_MISSES_NONE = "region misses no R rows"
REGION_MISSES_ALL = "region misses all R rows"
TRANSFER_YEARS = (2012, 2013, 2014)
REPRESENTATION_ONLY_YEAR = 2015


def year_group(year, reference):
    """Plan decision 2 labels: the reference year; years between the reference and 2012 ("before 2012");
    calendar 2012 to 2014 ("2012-2014"); and 2015 on its own. Other years raise, so no row is mislabelled."""
    year, reference = int(year), int(reference)
    if year == reference:
        return "reference"
    if year == REPRESENTATION_ONLY_YEAR:
        return "2015 separate"
    if year in TRANSFER_YEARS:
        return "2012-2014"
    if reference < year < TRANSFER_YEARS[0]:
        return "before 2012"
    raise ValueError(f"year {year} has no group for reference {reference}")


def check_hashes(expected, actual):
    """Raise ValueError naming every recorded input whose sha256 is missing or differs."""
    mismatched = [name for name, value in expected.items() if actual.get(name) != value]
    if mismatched:
        raise ValueError("input hash mismatch: " + ", ".join(sorted(mismatched)))


def combined_hash(paths):
    """SHA-256 of the joined per-file SHA-256s of the sorted files (Work 4 patient-table recipe)."""
    joined = "".join(cr.sha256(path) for path in sorted(paths))
    return hashlib.sha256(joined.encode()).hexdigest()


def export_path(export_root, build, reference):
    """Ideas export read for one Work 3 build (decision 4): r1 reads the baseline export."""
    folder = EXPORT_FOLDERS[build]
    return Path(export_root) / f"evaluate-{folder}" / f"evaluate-{reference}.json"


def concept_key(build, reference, family):
    return f"{build}/{reference}/{family}"


def to_float(value):
    return math.nan if value is None else float(value)


def family_year_rows(export, build):
    """One row per family-year: counts, retention and the two Ideas eligibility statuses."""
    rows = []
    for family in export["temporal"]["families"]:
        rows.append({
            "build": build,
            "reference": family["reference"],
            "year": family["year"],
            "family": family["family"],
            "concept": concept_key(build, family["reference"], family["family"]),
            "rows": family["rows"],
            "R_rows": family["R_rows"],
            "A_rows": family["A_rows"],
            "retention": family["retention"],
            "idea1_status": family["idea_status"]["idea1"],
            "idea2_status": family["idea_status"]["idea2"],
        })
    return rows


def member_rows(export, build):
    """One row per family member, seed and year: the seed's retention, cutoff and row counts inside A and R.
    The seed spread (plan 3a) is the spread of `retention` over the seeds of one family-year."""
    rows = []
    for member in export["temporal"]["members"]:
        rows.append({
            "build": build,
            "reference": member["reference"],
            "year": member["year"],
            "family": member["family"],
            "concept": concept_key(build, member["reference"], member["family"]),
            "factor": member["factor"],
            "seed": member["seed"],
            "retention": member["retention"],
            "cutoff": member["cutoff"],
            "cutoff_status": member["cutoff_status"],
            "A_rows": member["inside_A"]["rows"],
            "R_rows": member["inside_R"]["rows"],
        })
    return rows


def description_rows(export, build):
    """One row per family-year: the status of its yearly description."""
    rows = []
    for family in export["temporal"]["families"]:
        rows.append({
            "build": build,
            "reference": family["reference"],
            "year": family["year"],
            "family": family["family"],
            "concept": concept_key(build, family["reference"], family["family"]),
            "status": family["yearly_description"]["status"],
        })
    return rows


def mechanical_status(finite_count):
    """Status from the count of finite correlations alone (no minimum similarity)."""
    if finite_count >= 2:
        return AMBIGUOUS
    if finite_count == 1:
        return SINGLE
    return UNMATCHED


def matching_rows(export, build):
    """One row per family, yearly SAE and member: top-k Pearson in the export's order, the top-1 minus
    top-2 gap, the number of possible merges and the mechanical status next to the export's status."""
    rows = []
    for family in export["temporal"]["families"]:
        for entry in family["yearly_representations"]:
            for member in entry["members"]:
                candidates = member["candidates"]
                pearsons = [to_float(candidate["pearson"]) for candidate in candidates]
                finite_count = sum(math.isfinite(value) for value in pearsons)
                top = pearsons[:TOP_K] + [math.nan] * (TOP_K - len(pearsons))
                gap = top[0] - top[1] if math.isfinite(top[0]) and math.isfinite(top[1]) else math.nan
                rows.append({
                    "build": build,
                    "reference": family["reference"],
                    "year": family["year"],
                    "family": family["family"],
                    "concept": concept_key(build, family["reference"], family["family"]),
                    "seed": entry["seed"],
                    "dictionary_identity": entry["dictionary_identity"],
                    "reference_factor": member["reference_factor"],
                    "candidate_factors": tuple(candidate["factor"] for candidate in candidates[:TOP_K]),
                    "pearson_1": top[0],
                    "pearson_2": top[1],
                    "pearson_3": top[2],
                    "gap_1_2": gap,
                    "merges": len(entry["possible_merges"]),
                    "mechanical_status": mechanical_status(finite_count),
                    "export_status": member["status"],
                })
    return rows


def row_key(row, key_fields):
    return tuple(row[field] for field in key_fields)


def value_gap(first, second):
    """Absolute difference; both missing is 0, one missing is an infinite gap."""
    if math.isnan(first) and math.isnan(second):
        return 0.0
    if math.isnan(first) or math.isnan(second):
        return math.inf
    return abs(first - second)


def duplicate_gaps(first_rows, second_rows, key_fields, value_fields):
    """Largest absolute difference in value_fields between two copies of the same rows, matched on
    key_fields. Keys present in only one copy give an infinite gap."""
    second_by_key = {row_key(row, key_fields): row for row in second_rows}
    first_keys = {row_key(row, key_fields) for row in first_rows}
    if first_keys != set(second_by_key):
        return math.inf
    gaps = [0.0]
    for row in first_rows:
        other = second_by_key[row_key(row, key_fields)]
        for field in value_fields:
            gaps.append(value_gap(to_float(row[field]), to_float(other[field])))
    return max(gaps)


def status_row(identity, status, value_fields, values=None):
    """One row with its identity fields and status. Values are NaN when there is no measurement,
    so an absent concept never reads as 0."""
    row = dict(identity)
    row["status"] = status
    for field in value_fields:
        row[field] = values[field] if values is not None and field in values else math.nan
    return row


def candidate_coverage(matching, dictionary_size):
    """Distinct factor indices of one yearly SAE that any member's top-k candidates name.
    Reference-anchored, not a census: indices are compared only within one SAE."""
    factors_by_sae = {}
    for row in matching:
        sae = (row["build"], row["reference"], row["seed"], row["year"])
        factors_by_sae.setdefault(sae, set()).update(row["candidate_factors"])
    rows = []
    for (build, reference, seed, year), factors in sorted(factors_by_sae.items()):
        rows.append({
            "build": build,
            "reference": reference,
            "seed": seed,
            "year": year,
            "distinct_candidate_factors": len(factors),
            "dictionary_size": dictionary_size,
            "share": len(factors) / dictionary_size,
        })
    return rows


def zero_share(activation, mask):
    """Unweighted share of exact-zero activations among the rows in mask; NaN when mask is empty."""
    activation = np.asarray(activation, float)
    mask = np.asarray(mask, bool)
    return float(cr.ratio(np.sum((activation == 0) & mask), np.sum(mask)))


def paired_measures(in_R, in_A, activation, linear, weights):
    """Per draw: AUROC for R of the factor activation and of the linear score, P(A | R) and P(A | not R)."""
    scores = np.vstack([activation, linear])
    auroc = cc.weighted_auroc(in_R, scores, weights)
    return {
        "auroc_activation": auroc[0],
        "auroc_linear": auroc[1],
        "p_a_given_r": cr.agreement_given_rule(in_R, in_A, weights),
        "p_a_given_not_r": cr.agreement_given_rule(~in_R, in_A, weights),
    }


def year_auroc_draws(rows, count_matrix, population_ids):
    """Draws and points of one year's evaluation rows. Draw counts are read by patient id from the
    population's draw matrix, so a patient in two years carries the same count in both."""
    in_R = np.asarray(rows["in_R"], bool)
    in_A = np.asarray(rows["in_A"], bool)
    activation = np.asarray(rows["activation"], float)
    linear = np.asarray(rows["linear"], float)
    weights = cr.draw_counts_for_rows(count_matrix, population_ids, rows["patient_ids"])
    unit_weights = np.ones((1, len(in_R)))
    draws = paired_measures(in_R, in_A, activation, linear, weights)
    point = paired_measures(in_R, in_A, activation, linear, unit_weights)
    return {
        "draws": draws,
        "point": {name: float(values[0]) for name, values in point.items()},
        "zero_share": {"R": zero_share(activation, in_R), "not_R": zero_share(activation, ~in_R)},
    }


def auroc_changes(years, reference, count_matrix, population_ids):
    """Per year: draws of the measures, and their change from the reference year draw by draw, with the
    paired difference (activation change minus linear change)."""
    per_year = {year: year_auroc_draws(rows, count_matrix, population_ids) for year, rows in years.items()}
    reference_draws = per_year[reference]["draws"]
    changes = {}
    for year, result in per_year.items():
        change = {f"change_{name}": values - reference_draws[name] for name, values in result["draws"].items()}
        change["paired_auroc_difference"] = change["change_auroc_activation"] - change["change_auroc_linear"]
        changes[year] = change
    return {"per_year": per_year, "changes": changes}


def kernel_terms(pipeline, X):
    """Score, kernel term and scaled squared distances of a fitted SVC pipeline (optional scaler, then
    an RBF SVC), computed in float64 on the device. Scores follow the pipeline's own transforms, and the
    distances are gamma times the squared distance to the nearest support vector (all, and positive only)."""
    import torch

    device = cc.device()
    Z = np.asarray(X, float)
    for _, step in pipeline.steps[:-1]:
        Z = step.transform(Z)
    svc = pipeline.steps[-1][1]
    gamma = float(svc._gamma)  # private sklearn attribute: sklearn is pinned at 1.5.2, and the run checks the score against decision_function at 1e-9
    rows = torch.as_tensor(Z, dtype=torch.float64, device=device)
    support = torch.as_tensor(svc.support_vectors_, dtype=torch.float64, device=device)
    alpha = torch.as_tensor(svc.dual_coef_[0], dtype=torch.float64, device=device)
    squared = torch.cdist(rows, support) ** 2
    kernel_term = torch.exp(-gamma * squared) @ alpha
    positive = alpha > 0
    return {
        "score": (kernel_term + float(svc.intercept_[0])).cpu().numpy(),
        "kernel_term": kernel_term.cpu().numpy(),
        "gamma_sq_distance_all": (gamma * squared.min(dim=1).values).cpu().numpy(),
        "gamma_sq_distance_positive": (gamma * squared[:, positive].min(dim=1).values).cpu().numpy(),
    }


def ratio_to_reference(values_by_year, reference_year):
    """Each year's values divided by the reference year's values, elementwise; 1 at the reference year."""
    reference = values_by_year[reference_year]
    return {year: cr.ratio(values, reference) for year, values in values_by_year.items()}


def overlap_draws(in_R, missed_region, missed_other, weights):
    """Per draw, among rule rows: P(other misses | region misses), P(other misses | region does not miss),
    P(other misses | R) and their contrast. NaN where a conditioning set has no weight."""
    in_R = np.asarray(in_R, bool)
    missed_region = np.asarray(missed_region, bool)
    missed_other = np.asarray(missed_other, bool)
    p_given_m = cr.agreement_given_rule(in_R & missed_region, missed_other, weights)
    p_given_not_m = cr.agreement_given_rule(in_R & ~missed_region, missed_other, weights)
    return {
        "p_n_given_m": p_given_m,
        "p_n_given_not_m": p_given_not_m,
        "p_n_given_r": cr.agreement_given_rule(in_R, missed_other, weights),
        "contrast": p_given_m - p_given_not_m,
    }


def year_overlap(rows, count_matrix, population_ids):
    """Draws, point values and counts of one year's rule rows. The region misses a row when its score is 0 or below."""
    in_R = np.asarray(rows["in_R"], bool)
    missed_region = np.asarray(rows["region_score"], float) <= 0
    missed_other = np.asarray(rows["other_missed"], bool)
    weights = cr.draw_counts_for_rows(count_matrix, population_ids, rows["patient_ids"])
    draws = overlap_draws(in_R, missed_region, missed_other, weights)
    point = overlap_draws(in_R, missed_region, missed_other, np.ones((1, len(in_R))))
    region_misses = int((in_R & missed_region).sum())
    rule_rows = int(in_R.sum())
    if rule_rows == 0:
        reason = NO_RULE_PATIENTS
    elif region_misses == 0:
        reason = REGION_MISSES_NONE
    elif region_misses == rule_rows:
        reason = REGION_MISSES_ALL
    else:
        reason = ""
    return {
        "draws": draws,
        "point": {name: float(values[0]) for name, values in point.items()},
        "counts": {"R_rows": rule_rows, "region_misses": region_misses,
                   "other_misses": int((in_R & missed_other).sum())},
        "reason": reason,
    }


def reference_status(year_result):
    """Reference status of the region's misses among rule rows: none, all, or a count."""
    misses, rule_rows = year_result["counts"]["region_misses"], year_result["counts"]["R_rows"]
    if rule_rows == 0:
        return NO_RULE_PATIENTS
    if misses == 0:
        return NO_REFERENCE_MISSES
    if misses == rule_rows:
        return "misses all"
    return f"{misses} missed rows"


def missed_overlap(years, reference, count_matrix, population_ids):
    """Overlap of the region's misses with the other reading's misses, per year, by draw. Later values are
    changes from the reference year when the reference has misses; with no reference misses they are reported
    as they are, labelled "all later misses are new". Each change carries the share of draws where it is defined."""
    per_year = {year: year_overlap(rows, count_matrix, population_ids) for year, rows in years.items()}
    reference_result = per_year[reference]
    status = reference_status(reference_result)
    measures = list(reference_result["draws"])
    reference_undefined = {name: float(np.mean(~np.isfinite(reference_result["draws"][name]))) for name in measures}
    entries = {}
    for year, result in per_year.items():
        entry = {"draws": result["draws"], "point": result["point"], "counts": result["counts"],
                 "reason": result["reason"]}
        if status == NO_REFERENCE_MISSES:
            entry["label"] = "" if year == reference else ALL_LATER_NEW
            entry["change"] = None
        else:
            entry["change"] = {name: result["draws"][name] - reference_result["draws"][name] for name in measures}
            entry["defined_fraction"] = {name: float(np.mean(np.isfinite(entry["change"][name]))) for name in measures}
        entries[year] = entry
    return {"reference_status": status, "reference_undefined_fraction": reference_undefined, "per_year": entries}


def unmeasured_candidates(entry, status):
    """Candidate rows that all carry the same status (a build-level or identity failure)."""
    return [{"rank": rank, "factor": candidate["factor"], "pearson": candidate.get("pearson"), "status": status,
             "auroc_draws": None, "difference_draws": None}
            for rank, candidate in enumerate((entry.get("candidates") or [])[:TOP_K], start=1)]


def profile_check(column_R, profile):
    """Compare a candidate's recomputed profile on the rule rows with the export's profile_inside_R:
    positive counts exactly, mean magnitude within rtol 1e-5 and atol 1e-6. rows_differ is the gap in
    positive counts (a lower bound on the rows that differ) plus any gap in row counts."""
    if profile is None or profile.get("positive_prevalence") is None or profile.get("mean_magnitude") is None:
        # The Ideas code writes None for an empty rule subgroup ("empty subgroup", rows 0; pilot_follow_ups.ipynb cell 22
        # activation_profile); there is then nothing to compare, so the candidate is unmeasured with its reason.
        reason = "missing" if profile is None else profile.get("status") or "missing fields"
        return {"status": f"unmeasured: no export profile ({reason})", "rows_differ": None,
                "positive_count": None, "recomputed_positive_count": int(np.sum(column_R > 0)),
                "mean_magnitude_export": None, "mean_magnitude_recomputed": float(np.mean(column_R)) if len(column_R) else math.nan}
    export_rows = int(profile["rows"])
    export_positive = int(round(float(profile["positive_prevalence"][0]) * export_rows))
    export_mean = float(profile["mean_magnitude"][0])
    positive = int(np.sum(column_R > 0))
    mean = float(np.mean(column_R)) if len(column_R) else math.nan
    rows_differ = abs(export_positive - positive) + abs(export_rows - len(column_R))
    matches = (export_rows == len(column_R) and export_positive == positive
               and np.isclose(export_mean, mean, rtol=1e-5, atol=1e-6))
    return {"status": "measured" if matches else "unmeasured: profile differs", "rows_differ": rows_differ,
            "positive_count": export_positive, "recomputed_positive_count": positive,
            "mean_magnitude_export": export_mean, "mean_magnitude_recomputed": mean}


def yearly_candidate_auroc(entry, load_unit, encode, frozen_activation, in_R, patient_ids, count_matrix,
                           population_ids):
    """AUROC for R of the frozen member and of its top-k yearly candidates, by draw, on one year's evaluation
    rows. Only loads and encodes; candidates keep the export's Pearson order. A loader error or an identity
    mismatch marks the whole entry unmeasured; a profile mismatch marks that candidate only."""
    try:
        unit = load_unit(entry["reference"], entry["seed"], entry["year"])
    except Exception as error:
        reason = str(error) or type(error).__name__
        return {"status": f"unmeasured: {reason}", "candidates": unmeasured_candidates(entry, f"unmeasured: {reason}")}
    if unit["unit_identity"] != entry["dictionary_identity"]:
        return {"status": "unmeasured: identity differs",
                "candidates": unmeasured_candidates(entry, "unmeasured: identity differs")}
    cache_batch = unit.get("encoding_batch")
    batch = cache_batch or DEFAULT_ENCODING_BATCH
    activations = np.asarray(encode(unit, batch), float)
    in_R = np.asarray(in_R, bool)
    weights = cr.draw_counts_for_rows(count_matrix, population_ids, patient_ids)
    checks = []
    candidates = (entry.get("candidates") or [])[:TOP_K]
    for rank, candidate in enumerate(candidates, start=1):
        column = activations[:, candidate["factor"]]
        check = profile_check(column[in_R], candidate.get("profile_inside_R"))
        checks.append({"rank": rank, "factor": candidate["factor"], "pearson": candidate["pearson"], **check,
                       "column": column})
    measured = [row for row in checks if row["status"] == "measured"]
    scores = np.vstack([np.asarray(frozen_activation, float)] + [row["column"] for row in measured])
    draws = cc.weighted_auroc(in_R, scores, weights)
    point = cc.weighted_auroc(in_R, scores, np.ones((1, len(in_R))))[:, 0]
    frozen_draws = draws[0]
    candidates = []
    measured_position = 0
    for row in checks:
        out_row = {key: value for key, value in row.items() if key != "column"}
        if row["status"] == "measured":
            measured_position += 1
            candidate_draws = draws[measured_position]
            out_row["auroc_draws"] = candidate_draws
            out_row["difference_draws"] = candidate_draws - frozen_draws
            out_row["auroc_point"] = float(point[measured_position])
        else:
            out_row["auroc_draws"] = None
            out_row["difference_draws"] = None
        candidates.append(out_row)
    return {"status": "measured", "reference": entry["reference"], "seed": entry["seed"], "year": entry["year"],
            "concept": entry["concept"], "reference_factor": entry["reference_factor"],
            "encoding_batch": batch, "encoding_batch_source": "cache config" if cache_batch else "default",
            "frozen_auroc_draws": frozen_draws, "frozen_auroc_point": float(point[0]), "candidates": candidates}


def anchor_row(name, observed, expected, atol):
    """Compare observed values with a stored reference: largest absolute gap, NaN pairs count as equal.
    A failed row stops the build that owns it; the caller decides what to report."""
    observed = np.asarray(observed, float)
    expected = np.asarray(expected, float)
    both_nan = np.isnan(observed) & np.isnan(expected)
    gap = np.where(both_nan, 0.0, np.abs(observed - expected))
    gap = np.where(np.isnan(observed) != np.isnan(expected), np.inf, gap)
    largest = float(np.max(gap)) if gap.size else 0.0
    return {"anchor": name, "max_gap": largest, "atol": atol, "passed": bool(largest <= atol),
            "observed": observed.tolist(), "expected": expected.tolist()}


def two_by_two_by_two(in_R, region_missed, other_missed, in_A):
    """Counts of rule rows by (region misses, other misses, in A): the 2 x 2 x 2 table of check (c)."""
    in_R = np.asarray(in_R, bool)
    region_missed = np.asarray(region_missed, bool)
    other_missed = np.asarray(other_missed, bool)
    in_A = np.asarray(in_A, bool)
    table = {}
    for region in (True, False):
        for other in (True, False):
            for member in (True, False):
                mask = in_R & (region_missed == region) & (other_missed == other) & (in_A == member)
                table[(region, other, member)] = int(mask.sum())
    return table


def medoid_gap(reference_encoding, patient_activation):
    """Largest absolute difference between the medoid's reference encoding and the patient table activation."""
    return float(np.max(np.abs(np.asarray(reference_encoding, float) - np.asarray(patient_activation, float))))


def rank_correlation(x, y):
    """Spearman rank correlation over the pairs where both values are finite, with the number of pairs."""
    from scipy.stats import spearmanr

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return {"rho": math.nan, "n": int(finite.sum())}
    return {"rho": float(spearmanr(x[finite], y[finite]).statistic), "n": int(finite.sum())}


def json_ready(value):
    """Plain JSON values: numpy scalars and arrays become Python values; NaN and infinity become None."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_run_record(path, payload):
    """Write the run record as strict JSON (NaN and infinity become null)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(json_ready(payload), handle, sort_keys=True, indent=2, allow_nan=False)


def relative_to_project(path, project_root, home):
    """Path relative to the project folder, or to ~ for files in the home folder; never an absolute home path.
    Anything else is an error, so no record can hold an absolute path by accident."""
    path = Path(path)
    for root, prefix in ((Path(project_root), ""), (Path(home), "~/")):
        try:
            return prefix + path.relative_to(root).as_posix()
        except ValueError:
            continue
    raise ValueError(f"path is outside the project and home folders: {path.name}")


def decision_gap(pipeline, X):
    """Largest gap between the kernel score and the pipeline's decision_function, both on float64 rows.
    A StandardScaler keeps float32 input in float32, so float32 rows would give a gap of about 1e-8 that is
    not in the kernel arithmetic; the rows are cast before either score is computed."""
    rows = np.asarray(X, float)
    return float(np.max(np.abs(kernel_terms(pipeline, rows)["score"] - pipeline.decision_function(rows))))


def failed_anchors(rows):
    """Names of the anchor rows that did not pass; a build with any of them stops its checks."""
    return [row["anchor"] for row in rows if not row["passed"]]
