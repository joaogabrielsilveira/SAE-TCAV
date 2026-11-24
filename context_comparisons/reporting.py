"""Report matched comparisons with explicit missingness and effective contexts."""

import json

import pandas as pd

from context_intervention.analysis import performance
from context_intervention.evaluation import paired_patient_bootstrap, prediction_transitions, temporal_summary


KEYS = ["build", "reference", "year", "patient_id", "sampling_seed"]
OBJECTIVE_PREFIXES = ("death_f1_", "brier_skill_", "average_precision_")


def summarize_choices(output):
    rows = []
    for path in sorted(output.glob("metric_choices_*.json")):
        result = json.loads(path.read_text())
        for decision in result["decisions"]:
            row = {key: value for key, value in decision.items() if key not in ("choice", "effective_policy")}
            row.update({f"chosen_{key}": value for key, value in decision["choice"].items()})
            row.update(decision.get("effective_policy", {}))
            row["new_model_inferences"] = result["new_model_inferences"]
            rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        table.to_csv(output / "metric_choices.csv", index=False)
        coverage = []
        for objective, subset in table.groupby("objective"):
            changed, cached = subset["different_from_death_f1"], subset["evaluation"] == "cached"
            coverage.append({"objective": objective, "decisions": len(subset),
                "changed_choices": int(changed.sum()), "changed_choices_evaluated": int((changed & cached).sum()),
                "changed_choices_missing": int((changed & ~cached).sum()),
                "unchanged_choices_evaluated": int((~changed & cached).sum()),
                "unchanged_choices_missing": int((~changed & ~cached).sum()),
                "interpretation": "test coverage depends on selected policy; no global objective ranking or equivalence claim"})
        pd.DataFrame(coverage).to_csv(output / "objective_coverage.csv", index=False)
    return table


def paired_rows(rows, method, control, seed, common_size=False):
    subset = rows[(rows["method"] == method) & (rows["sampling_seed"] == seed)].copy()
    baseline = rows[(rows["method"] == control) & (rows["sampling_seed"] == seed)].copy()
    if common_size:
        subset = subset[subset["common_size_eligible"] & ~subset["fallback"]]
        baseline = baseline[baseline["common_size_eligible"] & ~baseline["fallback"]]
    baseline = baseline[KEYS + ["probability", "context_size"]].rename(
        columns={"probability": "baseline_probability", "context_size": "control_context_size"})
    matched = subset.merge(baseline, on=KEYS, how="inner", validate="one_to_one")
    if common_size:
        matched = matched[matched["context_size"] == matched["control_context_size"]]
    return matched


def summarize(output, repetitions=1000):
    """Use frozen thresholds only; suppress unequal secondary tuning entirely.

    Original-run tuned thresholds remain in copied raw data, but the added arms
    do not yet have equivalent tuning. Objective summaries are confined to
    matched available cases, whose policy-dependent coverage is shown separately.
    """
    summarize_choices(output)
    files = sorted(output.glob("predictions_*_20??_20??.csv.gz"))
    if not files:
        raise ValueError("No completed comparison predictions found")
    # Objective arms re-use the original study's sampler; they are compared only with the
    # original context and with their own Death-F1 choice, never with corrected sampling arms.
    files += sorted(output.glob("metric_predictions_*_20??_20??.csv.gz"))
    rows = pd.concat([pd.read_csv(path, dtype={"patient_id": str}) for path in files], ignore_index=True)
    if rows.duplicated(KEYS + ["method"]).any():
        raise ValueError("Duplicate comparison predictions")
    if ((rows["year"] > 2014) | (rows["year"] <= rows["reference"])).any():
        raise ValueError("Invalid comparison evaluation year")
    required = {"effective_form", "context_size", "fallback", "common_size_eligible"}
    if not required.issubset(rows.columns) or rows[list(required)].isna().any().any():
        raise ValueError("Comparison predictions lack effective-policy metadata")
    for column in ("fallback", "common_size_eligible"):
        if not rows[column].isin([True, False]).all():
            raise ValueError(f"Invalid boolean policy metadata: {column}")
        rows[column] = rows[column].astype(bool)
    metrics, transitions, summaries, policies, intervals, coverage, additional = [], [], [], [], [], [], []
    for (method, seed), subset in rows.groupby(["method", "sampling_seed"], sort=True):
        objective_method = method.startswith(OBJECTIVE_PREFIXES)
        scope = "matched available cases; selection-dependent coverage" if objective_method else "available reference-year cells"
        for (reference, year), cell in subset.groupby(["reference", "year"], sort=True):
            if cell["threshold"].nunique() != 1:
                raise ValueError("Expected one frozen threshold per evaluation cell")
            common = {"method": method, "seed": seed, "reference": reference, "year": year, "scope": scope}
            policy = {key: cell[key].iloc[0] for key in required}
            policies.append({**common, **policy, "secondary_threshold_status": "unavailable in added comparison report"})
            values = performance(cell["label"], cell["probability"], cell["threshold"].iloc[0],
                                 cell["reference_mortality"].iloc[0])
            metrics.append({**common, **policy, "threshold_type": "threshold", **values})
        original = paired_rows(rows, method, "original", seed)
        for (reference, year), cell in original.groupby(["reference", "year"]):
            transitions.append({"method": method, "seed": seed, "reference": reference, "year": year,
                **prediction_transitions(cell["label"], cell["baseline_probability"], cell["probability"],
                                         cell["threshold"].iloc[0])})
        if not objective_method:
            _, summary = temporal_summary(subset)
            summaries.append({"method": method, "seed": seed, "scope": scope, **summary})
        controls = []
        if method != "original":
            controls.append(("original", False, "original context; " + scope))
        sampled = "hard_sampling" in method or "graded_sampling" in method
        if sampled and not objective_method:
            controls.append(("uniform_sampling", True, "paired common-size sampling; full-context fallbacks excluded"))
        elif method.endswith("_direct") and "random" not in method:
            controls.append((method.replace("_direct", "_random_direct"), False, "matched random direct removal"))
        if "graded_sampling" in method and not objective_method:
            controls.append((method.replace("graded_sampling", "hard_sampling"), True, "common-size graded versus hard sampling"))
        comparison_control = None
        for prefix, replacement in (("latest_year_", "cumulative_"), ("native_context_", "cumulative_"),
                                    ("brier_skill_", "death_f1_"), ("average_precision_", "death_f1_")):
            if method.startswith(prefix) and "random" not in method:
                comparison_control = method.replace(prefix, replacement, 1)
                controls.append((comparison_control, False, "one-change comparison; " + scope))
        for control, common_size, comparison_scope in controls:
            matched = paired_rows(rows, method, control, seed, common_size)
            cells = len(matched[["reference", "year"]].drop_duplicates())
            record = {"method": method, "control": control, "seed": seed, "scope": comparison_scope,
                      "paired_reference_year_cells": cells, "patient_rows": len(matched),
                      "available_method_cells": len(subset[["reference", "year"]].drop_duplicates()),
                      "status": "matched available cases" if len(matched) else "unavailable; no matched eligible control"}
            coverage.append(record)
            if matched.empty:
                continue
            values = paired_patient_bootstrap(matched, repetitions=repetitions)
            target = additional if control == comparison_control else intervals
            target.extend({**record, "metric": metric, **value} for metric, value in values.items())
    for filename, records in (("intervention_metrics.csv.gz", metrics), ("prediction_transitions.csv.gz", transitions),
                              ("temporal_summaries.csv.gz", summaries), ("paired_intervals.csv.gz", intervals),
                              ("comparison_intervals.csv.gz", additional)):
        pd.DataFrame(records).to_csv(output / filename, index=False, compression="gzip")
    pd.DataFrame(coverage).to_csv(output / "comparison_coverage.csv", index=False)
    pd.DataFrame(policies).to_csv(output / "effective_policies.csv", index=False)
    return {"completed_reference_year_cells": len(rows[["build", "reference", "year"]].drop_duplicates()),
            "expected_reference_year_cells": 21, "methods": sorted(rows["method"].unique()),
            "threshold_comparison": "frozen thresholds only; secondary tuning unavailable for added arms",
            "objective_comparison": "matched available cases only; missingness depends on selected policy"}
