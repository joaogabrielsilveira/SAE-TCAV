"""Readers for results published by earlier runs (November study, October follow-ups)."""
import json

import pandas as pd

from concept_study import paths


def november_evaluation() -> pd.DataFrame:
    """v7d evaluation long table (legacy C=0.1 fits): sources annual, removal, frozen_fit_trajectory, ..."""
    return pd.read_csv(paths.EVALUATE_PUBLIC / "table-001.csv", low_memory=False)


def november_fits() -> pd.DataFrame:
    """v7c reference-year fit table: one row per concept, estimator, C, fit and subgroup (with AUROC)."""
    table = pd.read_csv(paths.DEVELOP_PUBLIC / "table-001.csv", low_memory=False)
    return table[table.source == "joint"]


def october_annual(variant: str) -> pd.DataFrame:
    """October 3-fit annual measures per family, subgroup and year (one variant)."""
    rows = []
    for path in sorted((paths.PREVIOUS_RUNS / f"evaluate-{variant}").glob("evaluate-20*.json")):
        saved = json.loads(path.read_text())
        for record in saved["annual"]:
            row = {"variant": variant, "family": record["family"], "reference": record["reference"],
                   "year": record["year"], "group": record["subgroup"], "tcav": record["tcav"],
                   "signed": record["signed_sensitivity"], "absolute": record["absolute_sensitivity"],
                   "rule_f2": record["rule_f2"], "activation_rate": record["activation_prevalence"],
                   "yearly_alignment": record["yearly_alignment"]}
            for seed, influence in (record.get("influence") or {}).items():
                if isinstance(influence, dict):
                    row[f"tcav_{seed}"] = influence.get("tcav")
            rows.append(row)
    return pd.DataFrame(rows)


def october_cohort(variant: str) -> pd.DataFrame:
    """October whole-cohort performance per reference predictor and year (predictor.json)."""
    saved = json.loads((paths.PREVIOUS_RUNS / f"evaluate-{variant}/predictor.json").read_text())
    rows = []
    for reference, records in saved.items():
        for record in records:
            if "metrics" not in record:          # 2015 rows carry no outcome metrics
                continue
            metrics = record["metrics"]
            rows.append({"variant": variant, "reference": int(record["reference"]), "year": int(record["year"]),
                         "brier": metrics.get("brier"), "auroc": record.get("auroc"),
                         "death_f1": (metrics.get("at_frozen_threshold") or {}).get("f1"),
                         "threshold": record.get("frozen_threshold"), "deaths": metrics.get("deaths")})
    return pd.DataFrame(rows)


def fresh_rule_fidelity() -> pd.DataFrame:
    """Idea 1 saved measure: F2 of a rule fitted to each year (held out), per concept-year."""
    table = november_evaluation()
    fresh = table[table.source == "yearly_rule"][["concept", "year", "value", "status"]]
    return fresh.rename(columns={"value": "fresh_rule_f2"})


def saved_measures(records: list) -> pd.DataFrame:
    """Fresh-rule F2 (Idea 1) and October cross-seed yearly alignment (Idea 2) per concept-year-group.

    The October alignment of a concept shared by both variants is read from the primary variant's run.
    """
    fresh = fresh_rule_fidelity()
    october = pd.concat([october_annual(v) for v in paths.VARIANTS], ignore_index=True)
    frames = []
    for record in records:
        variant = paths.PRIMARY_VARIANT if paths.PRIMARY_VARIANT in record["variants"] else record["variants"][0]
        for group in ("A", "R"):
            part = october[(october.family == record["family"]) & (october.variant == variant)
                           & (october.group == group)]
            if part.year.duplicated().any():
                raise ValueError(f"{record['concept']} {group}: several October rows per year")
            base = pd.DataFrame({"concept": record["concept"], "group": group, "year": part.year,
                                 "yearly_alignment": part.yearly_alignment.astype(float)})
            frames.append(base)
    alignment = pd.concat(frames, ignore_index=True)
    return alignment.merge(fresh[["concept", "year", "fresh_rule_f2"]], on=["concept", "year"], how="outer")


def estimator_auroc() -> pd.DataFrame:
    """Per concept: median held-out AUROC over the 50 fits for all-negative and legacy C=0.1."""
    fits = november_fits()
    fits = fits[(fits.C == paths.CAV_C) & (fits.subgroup == "A")]
    table = fits.groupby(["concept", "variants", "method"]).auroc.median().unstack("method")
    return table.reset_index()
