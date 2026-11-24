"""Summarize saved intervention predictions without re-running models."""

from pathlib import Path

import numpy as np
import pandas as pd

from context_intervention.analysis import performance
from context_intervention.evaluation import paired_patient_bootstrap, prediction_transitions, temporal_summary


def summarize(output, bootstrap_repetitions=1000):
    """Write cell metrics, transitions and patient-bootstrap differences."""
    output = Path(output)
    files = sorted(output.glob("predictions_*_20??_20??.csv.gz"))
    if not files:
        raise ValueError("No completed intervention predictions found")
    rows = pd.concat([pd.read_csv(path, dtype={"patient_id": str}) for path in files], ignore_index=True)
    keys = ["build", "reference", "year", "patient_id", "sampling_seed"]
    if rows.duplicated(keys + ["method"]).any():
        raise ValueError("Duplicate prediction rows")
    if (rows["year"] > 2014).any() or (rows["year"] <= rows["reference"]).any():
        raise ValueError("Predictions include an invalid evaluation year")
    baseline = rows.loc[rows["method"] == "original", keys + ["probability"]].rename(
        columns={"probability": "baseline_probability"})
    rows = rows.merge(baseline, on=keys, how="left", validate="many_to_one")
    if rows["baseline_probability"].isna().any():
        raise ValueError("An intervention prediction lacks a matched original")
    cell_metrics, transition_rows, summaries, intervals = [], [], [], []
    for (method, seed), subset in rows.groupby(["method", "sampling_seed"], sort=True):
        for (reference, year), cell in subset.groupby(["reference", "year"], sort=True):
            common = {"method": method, "seed": seed, "reference": reference, "year": year}
            for threshold_name in ("threshold", "tuned_threshold"):
                values = cell[threshold_name].unique()
                if len(values) != 1:
                    raise ValueError("Each method and year needs one decision threshold")
                metrics = performance(cell["label"], cell["probability"], values[0], cell["reference_mortality"].iloc[0])
                cell_metrics.append({**common, "threshold_type": threshold_name, **metrics})
            transition_rows.append({**common, **prediction_transitions(cell["label"], cell["baseline_probability"],
                                                                      cell["probability"], cell["threshold"].iloc[0])})
        _, summary = temporal_summary(subset)
        summaries.append({"method": method, "seed": seed, **summary})
        if method != "original":
            result = paired_patient_bootstrap(subset, repetitions=bootstrap_repetitions)
            intervals.extend({"method": method, "seed": seed, "control": "original", "metric": metric, **value}
                             for metric, value in result.items())
            if "hard_sampling" in method or "graded_sampling" in method:
                control_name = "uniform_sampling"
            elif "_direct" in method and "random" not in method:
                control_name = method.replace("_direct", "_random_direct", 1)
            else:
                control_name = None
            if control_name:
                control = rows[(rows["method"] == control_name) & (rows["sampling_seed"] == seed)]
                control = control[keys + ["probability"]].rename(columns={"probability": "control_probability"})
                matched = subset.merge(control, on=keys, how="inner", validate="one_to_one")
                if not matched.empty:
                    matched = matched.copy()
                    matched["baseline_probability"] = matched["control_probability"]
                    result = paired_patient_bootstrap(matched, repetitions=bootstrap_repetitions)
                    intervals.extend({"method": method, "seed": seed, "control": control_name,
                                      "metric": metric, **value} for metric, value in result.items())
            if "graded_sampling" in method:
                hard_name = method.replace("graded_sampling", "hard_sampling")
                control = rows[(rows["method"] == hard_name) & (rows["sampling_seed"] == seed)]
                control = control[keys + ["probability"]].rename(columns={"probability": "control_probability"})
                matched = subset.merge(control, on=keys, how="inner", validate="one_to_one")
                if not matched.empty:
                    matched["baseline_probability"] = matched["control_probability"]
                    result = paired_patient_bootstrap(matched, repetitions=bootstrap_repetitions)
                    intervals.extend({"method": method, "seed": seed, "control": hard_name,
                                      "metric": metric, **value} for metric, value in result.items())
    pd.DataFrame(cell_metrics).to_csv(output / "intervention_metrics.csv.gz", index=False, compression="gzip")
    pd.DataFrame(transition_rows).to_csv(output / "prediction_transitions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(summaries).to_csv(output / "temporal_summaries.csv.gz", index=False, compression="gzip")
    pd.DataFrame(intervals).to_csv(output / "paired_intervals.csv.gz", index=False, compression="gzip")
    cells = rows[["build", "reference", "year"]].drop_duplicates()
    return {"completed_reference_year_cells": len(cells), "expected_reference_year_cells": 21,
            "methods": sorted(rows["method"].unique().tolist())}
