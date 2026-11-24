"""Per later year: distinct evaluation deaths, missed deaths and false alarms inside at least one
factor rule or slice that has no context deaths. Counts only (aggregates). Run after
error_explorer_export.py, from ~/SAE-TCAV:  .venv/bin/python -u -B error_explorer_absent.py"""
from pathlib import Path

import numpy as np
import pandas as pd

import error_profiles as ep
from error_explorer_export import TOLERANCES, rule_from_key

OUTPUT = Path("stats/error_explorer")
rows = []
for case in ep.load_study(TOLERANCES)["cases"]:
    ref, evaluation = case["reference"], case["views"]["evaluation"]
    frame = evaluation["frame"]
    context = pd.read_csv(OUTPUT / "context_by_year.csv").query("reference == @ref")
    slices = pd.read_csv(f"stats/error_profiles_over_time/{ref}_slices.csv").set_index("slice")["rule"]
    absent = context[(context["absence"] != "context deaths present") & (context["kind"] != "existing")][["kind", "group"]].drop_duplicates()
    union = np.zeros(len(frame), bool)
    for kind, group in absent.itertuples(index=False):
        union |= ep.rule_mask(rule_from_key(slices[group] if kind == "slice" else group), evaluation["features"])
    y, p, years = frame["label"].to_numpy(), frame["probability"].to_numpy(), frame["year"].to_numpy()
    flagged = p >= case["threshold"]
    for year in sorted(set(years) - {ref}) + ["later"]:
        at = years > ref if year == "later" else years == year
        for name, mask in (("deaths", y == 1), ("missed_deaths", (y == 1) & ~flagged), ("false_alarms", (y == 0) & flagged)):
            rows.append({"reference": ref, "period": str(year), "outcome": name, "absent_groups": len(absent),
                         "inside": int((at & mask & union).sum()), "all": int((at & mask).sum())})
pd.DataFrame(rows).to_csv(OUTPUT / "absent_union_by_year.csv", index=False)
print("written", len(rows))
