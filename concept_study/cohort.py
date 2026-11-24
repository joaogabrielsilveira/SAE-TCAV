"""Whole-cohort performance per reference predictor and year, and the year-input comparison.

Whole-cohort metrics use the same joint patient bootstrap as the concept measures.
"""
import numpy as np
import pandas as pd

from concept_study import bootstrap, measures, paths, year_input


def weighted_auroc(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """AUROC for each weight row: weighted share of (death, survivor) pairs ranked correctly (ties count 1/2)."""
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        return np.full(len(weights), np.nan)
    wins = (positive[:, None] > negative[None, :]) + 0.5 * (positive[:, None] == negative[None, :])
    wp, wn = weights[:, labels == 1], weights[:, labels == 0]
    numerator = ((wp @ wins) * wn).sum(1)
    denominator = wp.sum(1) * wn.sum(1)
    return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1), np.nan)


def metrics(probability: np.ndarray, label: np.ndarray, threshold: float, constant: float,
            weights: np.ndarray) -> dict:
    """Brier, Brier skill, Death-F1 at the frozen threshold, AUROC, mortality, mean risk (per replicate)."""
    w = weights
    total = w.sum(1)
    deaths = w @ label
    predicted = (probability >= threshold).astype(float)
    brier = w @ (probability - label) ** 2 / total
    brier_constant = w @ (constant - label) ** 2 / total
    true_positive = w @ (predicted * label)
    f1_denominator = w @ predicted + deaths
    return {
        "brier": brier, "brier_skill": 1 - brier / brier_constant,
        "death_f1": np.where(f1_denominator > 0, 2 * true_positive / np.where(f1_denominator > 0, f1_denominator, 1), np.nan),
        "auroc": weighted_auroc(probability, label, w),
        "mortality": deaths / total, "mean_risk": w @ probability / total,
    }


def cohort_table(index: dict) -> pd.DataFrame:
    """Whole-cohort metrics per build and year up to 2014, with joint-bootstrap intervals."""
    unique_ids, weight_matrix = bootstrap.patient_weights(measures.all_patient_ids(index))
    rows = []
    for record in index["builds"]:
        build = measures.load_build(record)
        weights = bootstrap.row_weights(build["patient_id"], unique_ids, weight_matrix)
        threshold = measures.death_threshold(record)
        for year in range(record["reference"], paths.LAST_OUTCOME_YEAR + 1):
            at = build["year"] == year
            values = metrics(build["probability"][at].astype(float), build["label"][at].astype(float), threshold,
                             float(build["reference_mortality"]), weights[:, at])
            row = {"build": record["build"], "reference": record["reference"], "year": year,
                   "n": int(at.sum()), "deaths": int(build["label"][at].sum())}
            for name, value in values.items():
                low, high = bootstrap.percentile_interval(value[1:])
                row.update({name: value[0], f"{name}_low": low, f"{name}_high": high})
            rows.append(row)
    table = pd.DataFrame(rows)
    table["years_since_reference"] = table.year - table.reference
    return table


def year_input_table(index: dict) -> tuple:
    """True versus reference-year input: whole-cohort metrics and every concept's A size / agreement.

    Returns (cohort comparison table, concept comparison table). Cohort differences are swapped - true,
    with joint-bootstrap intervals; concept rows are point counts only (no intervals). Both A groups use
    the precision of the stored swapped codes: the true-year activation is rounded to it before the cutoff,
    so rounding alone cannot move a patient across the cutoff.
    """
    unique_ids, weight_matrix = bootstrap.patient_weights(measures.all_patient_ids(index))
    cohort_rows, concept_rows = [], []
    for record in index["builds"]:
        swapped = year_input.load_year_input(record["build"], record["reference"])
        if swapped is None:
            continue
        build = measures.load_build(record)
        positions = swapped["positions"]
        if not (np.array_equal(build["patient_id"][positions], swapped["patient_id"].astype(str))
                and np.array_equal(build["year"][positions], swapped["year"])):
            raise ValueError(f"{record['file']}: year-input rows do not match the build's patient-years")
        weights = bootstrap.row_weights(build["patient_id"][positions], unique_ids, weight_matrix)
        threshold = measures.death_threshold(record)
        for year in sorted(set(swapped["year"])):
            at = swapped["year"] == year
            row = {"build": record["build"], "reference": record["reference"], "year": int(year), "n": int(at.sum())}
            if year <= paths.LAST_OUTCOME_YEAR:
                label = build["label"][positions][at].astype(float)
                true = metrics(build["probability"][positions][at].astype(float), label, threshold,
                               float(build["reference_mortality"]), weights[:, at])
                moved = metrics(swapped["probability_swapped"][at].astype(float), label, threshold,
                                float(build["reference_mortality"]), weights[:, at])
                for name in ("brier", "death_f1", "auroc", "mean_risk"):
                    difference = moved[name] - true[name]
                    low, high = bootstrap.percentile_interval(difference[1:])
                    row.update({f"{name}_true": true[name][0], f"{name}_swapped": moved[name][0],
                                f"{name}_difference": difference[0], f"{name}_low": low, f"{name}_high": high})
            cohort_rows.append(row)
        for concept_record in [c for c in index["concepts"]
                               if c["build"] == record["build"] and c["reference"] == record["reference"]]:
            concept = measures.load_concept(concept_record)
            codes = swapped[f"codes_{concept_record['seed']}"][:, concept_record["factor"]]
            in_a_swapped = codes.astype(np.float32) > concept_record["cutoff"]
            true_activation = concept["activation"][positions].astype(codes.dtype).astype(np.float32)
            in_a, in_r = true_activation > concept_record["cutoff"], concept["in_R"][positions]
            for year in sorted(set(swapped["year"])):
                at = swapped["year"] == year
                n_r = in_r[at].sum()
                concept_rows.append({
                    "concept": concept_record["concept"], "variants": ",".join(concept_record["variants"]),
                    "reference": record["reference"], "year": int(year),
                    "n_A_true": int(in_a[at].sum()), "n_A_swapped": int(in_a_swapped[at].sum()), "n_R": int(n_r),
                    "agreement_true": (in_a & in_r)[at].sum() / n_r if n_r else np.nan,
                    "agreement_swapped": (in_a_swapped & in_r)[at].sum() / n_r if n_r else np.nan})
    cohort = pd.DataFrame(cohort_rows)
    concepts = pd.DataFrame(concept_rows)
    for frame in (cohort, concepts):
        if len(frame):
            frame["years_since_reference"] = frame.year - frame.reference
    return cohort, concepts
