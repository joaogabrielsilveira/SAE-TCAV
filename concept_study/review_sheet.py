"""Clinician review sheet: one entry per primary (last_3) concept, for a nephrologist to judge.

Per concept: the frozen rule in words, group sizes and agreement at the reference year and at
the last labelled year, and the features that most distinguish two groups (SMD: difference in
mean feature count over the pooled SD; positive = higher in the first group):
- the factor's patients (A) from everyone else, at the reference year;
- the rule's patients (R) from everyone else, at the reference year;
- patients who activate the factor without the rule (R-A+) from patients in neither group
  (median SMD over the labelled years from the reference year on);
- rule patients without the factor (R+A-) from rule patients with it (R+A+), compared within each
  year (cross-sectional, not the same patients followed); median SMD over the labelled years after
  the reference year.

    .venv/bin/python -m concept_study.review_sheet
"""
import numpy as np
import pandas as pd

from concept_study import features, loading, measures, paths

OUTPUT = paths.OUTPUT / "checks" / "review_sheet"
TOP = 5


def rule_text(rule_dict):
    parts = []
    for rule in rule_dict.get("rules", []):
        conditions = []
        for c in rule.get("conditions", []):
            op = c.get("operator", c.get("op", ""))
            value = c.get("threshold", c.get("value", ""))
            value = f"{value:.3g}" if isinstance(value, float) else value
            conditions.append(f"{c['feature_name'].strip()} {op} {value}".strip())
        parts.append(" AND ".join(conditions))
    return " OR ".join(f"({p})" for p in parts) if len(parts) > 1 else (parts[0] if parts else str(rule_dict))


def top_features(smd_row, names, k=TOP):
    """The k features with the largest |SMD| (NaN skipped), as "name +value; ..."."""
    finite = np.where(np.isfinite(smd_row), smd_row, 0)
    order = np.argsort(-np.abs(finite), kind="stable")[:k]
    return "; ".join(f"{names[j].strip()} {finite[j]:+.2f}" for j in order if finite[j] != 0)


def agreement(a, r, year, y):
    """Share of rule patients in year y who activate the factor."""
    m = (year == y) & r
    return float(a[m].mean()) if m.any() else np.nan


def median_over_years(matrix, years, width):
    """Median SMD over the given year positions (all NaN when no year is defined)."""
    return np.nanmedian(matrix[years], axis=0) if np.isfinite(matrix[years]).any() else np.full(width, np.nan)


def main():
    inputs = loading.open_inputs("cpu")
    frozen = {loading.concept_key(i["build"], i["reference"], i["family"]): i["frozen"] for i in inputs["inventory"]}
    index = measures.load_index()
    years = measures.YEARS
    g = {name: k for k, name in enumerate(measures.GROUPS)}
    rows = []
    for record in index["concepts"]:
        if "last_3" not in record["variants"]:
            continue
        f = features.load_features(record)
        concept = measures.load_concept(record)
        build = measures.load_build(measures.build_record(index, record))
        names = list(f["feature_names"])
        ref = years.index(record["reference"])
        labelled = [k for k, y in enumerate(years) if y <= paths.LAST_OUTCOME_YEAR and y >= record["reference"]]
        after = [k for k in labelled if k != ref]
        last = labelled[-1]
        a, r, year = concept["in_A"], concept["in_R"], build["year"]
        rows.append({
            "concept": record["concept"], "reference": record["reference"],
            "factor": f"seed {record['seed']} factor {record['factor']}",
            "rule": rule_text(frozen[record["concept"]]["rule"].to_dict()),
            "n_A_reference": int(f["count"][g["A"], ref]), "n_R_reference": int(f["count"][g["R"], ref]),
            "agreement_reference": agreement(a, r, year, record["reference"]),
            "agreement_last": agreement(a, r, year, years[last]),
            "last_year": years[last],
            "factor_patients_vs_rest": top_features(f["smd_vs_rest"][g["A"], ref], names),
            "rule_patients_vs_rest": top_features(f["smd_vs_rest"][g["R"], ref], names),
            "factor_without_rule_vs_neither": top_features(
                median_over_years(f["smd_activation_only_vs_neither"], labelled, len(names)), names),
            "rule_without_factor_vs_with": top_features(median_over_years(f["smd_stopped_vs_kept"], after, len(names)), names),
        })
    table = pd.DataFrame(rows).sort_values(["reference", "concept"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "review_sheet.csv", index=False)
    lines = ["# Clinician review sheet: last_3 concepts", "",
             "For each concept please judge: (1) does the rule describe one clinically coherent group?",
             "(2) do the factor's patients look like the same clinical idea as the rule?",
             "(3) are the patients with the factor but not the rule clinically similar to the rule group",
             "(should the rule be broader)? (4) in the years after the reference, what distinguishes rule",
             "patients without the factor from rule patients with it (compared within each year, not the",
             "same patients followed), and is that difference clinically meaningful?",
             "", "Numbers after features are standardized mean differences of the feature counts: difference in",
             "mean count divided by the pooled SD (positive = higher mean count in the first group).", ""]
    for _, row in table.iterrows():
        lines += [f"## {row['concept']}", "",
                  f"- Reference year {row['reference']}, {row['factor']}",
                  f"- Rule: `{row['rule']}`",
                  f"- At reference: {row['n_A_reference']} factor patients, {row['n_R_reference']} rule patients; "
                  f"agreement {row['agreement_reference']:.2f} -> {row['agreement_last']:.2f} in {row['last_year']}",
                  f"- Factor patients vs rest: {row['factor_patients_vs_rest']}",
                  f"- Rule patients vs rest: {row['rule_patients_vs_rest']}",
                  f"- Factor without rule vs neither: {row['factor_without_rule_vs_neither']}",
                  f"- Rule patients without the factor vs with (after reference): {row['rule_without_factor_vs_with']}",
                  "- Clinician notes: (1)    (2)    (3)    (4)", ""]
    (OUTPUT / "review_sheet.md").write_text("\n".join(lines))
    print(len(table), "concepts")


if __name__ == "__main__":
    main()
