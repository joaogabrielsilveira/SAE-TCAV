"""Log-odds companion of concept_groups_over_time.

The original study's loading uses the cached gradient of the predicted death
probability p with respect to the raw embedding. For a binary model
d logit(p)/dx = (dp/dx) / (p (1 - p)) exactly, and every sensitivity-type
quantity is linear in the per-row gradient. This module reruns the original
`execute_analysis` unchanged, except that the gradient rows are divided by
p (1 - p) at the single point where a build is loaded. Gradient-free measures
(contribution, norm, cancellation, activation, support, Brier) are unchanged.

Commands:
    evaluate  full rerun into stats/concept_agreement_study/extensions/group_logit/run
    smoke     the original three-family smoke path into .../group_logit/smoke
    compare   probability vs log-odds tables and figures in .../group_logit/
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import pathlib
import time
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import concept_groups_over_time as study

ROOT = Path(__file__).resolve().parent
EXTENSION = ROOT / "stats/concept_agreement_study/extensions/group_logit"
ORIGINAL = ROOT / "analysis_outputs/concept_groups_over_time/20261108-full"
# Source of record for the corrected association logic (association-correction/source).
CORRECTED_SOURCE_SHA256 = "bae28527dbe5ff4b78258a4a00f61812c8107d29ecfae4f5b1850d7b80133913"
CONSUMED_SOURCES = {"cmr_common.py", "concept_subgroup_drift_pilot.py", "__init__.py", "loading.py",
                    "measures.py", "paths.py"}
ASSOCIATION_KEYS = ["concept", "kind", "subgroup", "feature", "target", "adjustment", "level", "variant"]


def log_odds_gradients(gradients, probability) -> np.ndarray:
    """Per-row d logit(p)/dx from dp/dx; p must lie strictly inside (0, 1)."""
    p = np.asarray(probability, dtype=np.float64)
    if not np.all((p > 0) & (p < 1)):
        raise ValueError("log-odds conversion needs probabilities strictly inside (0, 1)")
    return np.asarray(gradients, dtype=np.float64) / (p * (1 - p))[:, None]


def build_check(base: dict, build: str, reference: int, rows: np.ndarray, checked_rows: int = 64) -> dict:
    """Probability range on the evaluation rows and an autograd check of the conversion.

    The converted cached gradient is compared with autograd of logit(p) through
    the frozen continuation itself (p = the study's probability column 1).
    """
    p = np.asarray(base["probabilities"])[rows, 1].astype(np.float64)
    weight = 1 / (p * (1 - p))
    sample = rows[:checked_rows]
    continuation = base["continuation"]
    device = next(continuation.head.parameters()).device
    h = torch.as_tensor(np.asarray(base["embeddings"])[sample], dtype=torch.float32, device=device)
    with torch.enable_grad():
        h.requires_grad_(True)
        gradient, = torch.autograd.grad(torch.logit(continuation.probabilities(h)[:, 1]).sum(), h)
    autograd = gradient.detach().cpu().numpy().astype(np.float64)
    converted = np.asarray(base["gradients"])[sample]
    np.testing.assert_allclose(converted, autograd, rtol=2e-3, atol=1e-5 * weight.max())
    return {"build": build, "reference": int(reference), "evaluation_rows": int(len(rows)),
            "p_min": float(p.min()), "p_q01": float(np.quantile(p, 0.01)), "p_q10": float(np.quantile(p, 0.1)),
            "p_median": float(np.median(p)), "p_q99": float(np.quantile(p, 0.99)), "p_max": float(p.max()),
            "max_inverse_pq": float(weight.max()),
            "autograd_rows_checked": int(len(sample)),
            "autograd_max_abs_error": float(np.abs(converted - autograd).max()),
            "autograd_max_abs_gradient": float(np.abs(autograd).max())}


def low_risk_tail(loading: np.ndarray, probability: np.ndarray) -> dict:
    """Share of summed |loading| from the cell's lowest-risk decile and from its largest row."""
    pq = probability * (1 - probability)
    output = {"rows": int(len(loading))}
    if not len(loading):
        return output
    low = probability <= np.quantile(probability, 0.1)
    output["low_risk_decile_rows"] = int(low.sum())
    for scale, values in (("log_odds", np.abs(loading)), ("probability", np.abs(loading) * pq)):
        total = values.sum()
        output[f"low_risk_decile_share_{scale}"] = float(values[low].sum() / total) if total > 0 else None
        output[f"largest_row_share_{scale}"] = float(values.max() / total) if total > 0 else None
    return output


@contextlib.contextmanager
def log_odds_scale(last_year: int, checks: list, tails: list):
    """Patch the original run at three seams; everything else is the original code.

    1. Builds are loaded with gradients divided by p (1 - p) per row.
    2. Outputs may live only under the extension (the original allows only analysis_outputs).
    3. The integrity fingerprint covers the consumed sources and the concept-study
       data outside extensions/, where this run and sibling extensions write.
    The annual-measure hook only records the low-risk tail of point loadings.
    """
    from concept_study import loading
    real_load_build = loading.load_build
    real_annual = study.annual_measures
    real_fingerprints = study.original_fingerprints
    real_relative = pathlib.PurePath.is_relative_to

    def load_build(inputs, build, reference, roles):
        base = real_load_build(inputs, build, reference, roles)
        base["gradients"] = log_odds_gradients(base["gradients"], np.asarray(base["probabilities"])[:, 1])
        rows = study.eligible_evaluation_rows(base, reference, last_year)
        checks.append(build_check(base, build, reference, rows))
        return base

    def annual_measures(common, result, gradients, years, masks, probability, *args, **kwargs):
        if common["kind"] in ("selected", "singleton"):
            loading_values = study.tensor_numpy(result["loading"])
            for year in sorted(np.unique(years)):
                for subgroup, membership in masks.items():
                    rows = membership & (years == year)
                    tails.append({**common, "year": int(year), "subgroup": subgroup,
                                  **low_risk_tail(loading_values[rows], probability[rows])})
        return real_annual(common, result, gradients, years, masks, probability, *args, **kwargs)

    def original_fingerprints(study_root, group_freeze):
        output = real_fingerprints(study_root, group_freeze)
        # Other agents edit unrelated concept_study modules on Fiji; keep the consumed sources only.
        output["source_sha256"] = {path: digest for path, digest in output["source_sha256"].items()
                                   if Path(path).name in CONSUMED_SOURCES or path == str(group_freeze)}
        directory = Path(study_root) / "stats/concept_agreement_study"
        listing = [(str(path.relative_to(directory)), path.stat().st_size, path.stat().st_mtime_ns)
                   for path in sorted(directory.rglob("*"))
                   if path.is_file() and path.relative_to(directory).parts[0] != "extensions"]
        output["data_listings"][str(directory)] = {
            "method": "listing outside extensions/", "files": len(listing),
            "sha256": hashlib.sha256(json.dumps(listing).encode()).hexdigest()}
        return output

    def is_relative_to(self, other):
        if pathlib.PurePath(other) == ROOT / "analysis_outputs/concept_groups_over_time":
            return real_relative(self, EXTENSION)
        return real_relative(self, other)

    with mock.patch.object(loading, "load_build", load_build), \
            mock.patch.object(study, "annual_measures", annual_measures), \
            mock.patch.object(study, "original_fingerprints", original_fingerprints), \
            mock.patch.object(pathlib.PurePath, "is_relative_to", is_relative_to):
        yield


def run(command: str = "evaluate") -> dict:
    """Rerun the original evaluation with the original configuration on the log-odds scale."""
    import pandas as pd
    import shutil
    if study.file_sha256(study.__file__) != CORRECTED_SOURCE_SHA256:
        raise ValueError("imported concept_groups_over_time is not the corrected source of record")
    original = json.loads((ORIGINAL / "run-manifest.json").read_text())
    config = dict(original["config"])
    output = EXTENSION / ("run" if command == "evaluate" else "smoke")
    config.update(command=command, output_dir=str(output), study_root=str(ROOT), max_output_gib=0.95)
    if command == "smoke":
        config.update(max_units=3, max_seconds=300)
    checks, tails = [], []
    start = time.monotonic()
    with log_odds_scale(config["last_year"], checks, tails):
        manifest = study.execute_analysis(argparse.Namespace(**config))
    shutil.copyfile(__file__, output / "source/concept_groups_logit.py")
    pd.DataFrame(tails).to_csv(output / "low_risk_tails.csv", index=False)
    logit_manifest = {
        "status": manifest["status"], "scale": "log-odds: cached dp/dx divided by p(1-p) per row, no clipping",
        "wrapper_sha256": study.file_sha256(__file__), "study_source_sha256": study.file_sha256(study.__file__),
        "original_run": str(ORIGINAL), "original_config_sha256": original["config_sha256"],
        "config": config, "elapsed_seconds": time.monotonic() - start, "build_checks": checks,
    }
    study.atomic_json(output / "logit-manifest.json", logit_manifest)
    return logit_manifest


# ------------------------------------------------------------------ comparison

def gradient_free_differences(original: Path, logit: Path) -> dict:
    """Largest absolute differences of measures that never involve the gradient."""
    import pandas as pd
    metrics = {
        "annual": {"support", "contribution_norm", "cancellation", "brier", "mortality", "mean_risk",
                   "active_brier", "active_mortality", "active_mean_risk", "positive_direction_fraction",
                   "common_active_support", "common_active_brier", "common_active_mortality",
                   "common_active_mean_risk"},
        "members": {"activation"}, "representation": None, "summary": None,
        "families": {"contribution_norm", "cancellation", "brier", "mortality", "mean_risk",
                     "common_active_brier", "common_active_mortality", "common_active_mean_risk"},
    }
    output = {}
    for name, kept in metrics.items():
        left, right = pd.read_csv(original / f"{name}.csv"), pd.read_csv(logit / f"{name}.csv")
        if kept is not None:
            left, right = left[left.metric.isin(kept)], right[right.metric.isin(kept)]
        keys = [column for column in left.columns if column not in ("value", "low", "high", "defined_draws",
                                                                   "undefined_draw_fraction")]
        joined = left.merge(right, on=keys, how="outer", suffixes=("_p", "_l"), indicator=True)
        assert (joined["_merge"] == "both").all(), f"{name}: rows differ between runs"
        output[name] = {"rows": len(joined), **{
            column: float(np.nanmax(np.abs(joined[f"{column}_p"] - joined[f"{column}_l"]).to_numpy(), initial=0))
            for column in ("value", "low", "high")},
            "definedness_equal": bool((joined.value_p.isna() == joined.value_l.isna()).all())}
    return output


def associations_joined(original: Path, logit: Path):
    """Probability vs log-odds associations on identical keys (corrected logic on both)."""
    import pandas as pd
    columns = ["value", "low", "high", "defined_draws"]
    left = pd.read_csv(study.association_table_path(original))
    right = pd.read_csv(logit / "associations.csv")
    for table in (left, right):
        table["variant"] = table["variant"].fillna("")
    shared = ["transitions", "families", "variants"]
    joined = left[ASSOCIATION_KEYS + shared + columns].merge(
        right[ASSOCIATION_KEYS + columns], on=ASSOCIATION_KEYS, how="outer", validate="one_to_one",
        suffixes=("_probability", "_log_odds"), indicator=True)
    assert (joined["_merge"] == "both").all(), "association rows differ between scales"
    return joined.drop(columns="_merge")


def figure(joined, variant: str, path: Path) -> None:
    """Pooled probability vs log-odds associations with 95% intervals, one panel per subgroup."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = joined[(joined.level == "pooled_changes") & (joined.variant == variant)]
    order = [(feature, kind, adjustment)
             for feature in ("absolute_loading", "common_active_absolute_direction")
             for kind in ("selected", "singleton")
             for adjustment in ("none", "family_and_year", "family_year_mortality_and_support",
                                "family_year_mortality_risk_and_support")]
    names = {"absolute_loading": "|loading| vs Brier", "common_active_absolute_direction": "|direction| vs Brier (common active)",
             "selected": "group", "singleton": "anchor", "none": "unadjusted", "family_and_year": "concept+year",
             "family_year_mortality_and_support": "+mortality+support",
             "family_year_mortality_risk_and_support": "+mortality+risk+support"}
    styles = (("probability", "#2a78d6", "o", -0.17), ("log_odds", "#eb6834", "s", 0.17))
    fig, axes = plt.subplots(1, 3, figsize=(17, 9), sharey=True)
    for axis, subgroup in zip(axes, ("all", "A", "R")):
        panel = rows[rows.subgroup == subgroup].set_index(["feature", "kind", "adjustment"])
        for position, key in enumerate(order):
            if key not in panel.index:
                continue
            row = panel.loc[key]
            for scale, colour, marker, offset in styles:
                y = position + offset
                if np.isfinite(row[f"low_{scale}"]):
                    axis.hlines(y, row[f"low_{scale}"], row[f"high_{scale}"], color=colour, linewidth=2)
                if np.isfinite(row[f"value_{scale}"]):
                    axis.plot(row[f"value_{scale}"], y, marker, color=colour, markersize=8,
                              label=scale.replace("_", "-") if position == 0 else None)
        for boundary in (3.5, 7.5, 11.5):
            axis.axhline(boundary, color="#cccccc", linewidth=0.8)
        axis.axvline(0, color="grey", linestyle="--", linewidth=1)
        axis.set_xlim(-1.05, 1.05)
        axis.set(title={"all": "All evaluation patients", "A": "Original activation group",
                        "R": "Original rule group"}[subgroup], xlabel="Rank correlation with concurrent Brier change")
    axes[0].set_yticks(range(len(order)), [", ".join(names[part] for part in key) for key in order], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].legend(title="Gradient scale", loc="lower left", fontsize=9)
    label = {"last_3": "last 3 years (primary)", "all_history": "all previous years (auxiliary)"}[variant]
    fig.suptitle(f"Pooled year-to-year associations, {label}: probability vs log-odds gradient (95% patient-bootstrap intervals)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def tail_summary(tails):
    """Distribution over group-year cells of the lowest-risk-decile share of summed |loading|."""
    import pandas as pd
    tails = tails[tails.rows >= 10].copy()
    tails["variant"] = tails["variants"].str.split(",")
    tails = tails.explode("variant")
    records = []
    for (variant, kind, subgroup), cells in tails.groupby(["variant", "kind", "subgroup"]):
        record = {"variant": variant, "kind": kind, "subgroup": subgroup, "cells": len(cells)}
        for column in ("low_risk_decile_share_log_odds", "low_risk_decile_share_probability",
                       "largest_row_share_log_odds", "largest_row_share_probability"):
            values = cells[column].dropna()
            record.update({f"{column}_median": values.median(), f"{column}_q90": values.quantile(0.9),
                           f"{column}_max": values.max()})
        records.append(record)
    return pd.DataFrame(records)


def compare() -> dict:
    """Write joined associations, key results, tail summary, identity checks and figures."""
    import pandas as pd
    logit = EXTENSION / "run"
    read = [ORIGINAL / name for name in ("annual.csv", "members.csv", "representation.csv", "summary.csv",
                                         "families.csv", "associations.csv", "run-manifest.json",
                                         "association-correction/associations.csv",
                                         "association-correction/correction-manifest.json")]
    before = {str(path): study.file_sha256(path) for path in read}
    joined = associations_joined(ORIGINAL, logit)
    joined.to_csv(EXTENSION / "associations_probability_vs_log_odds.csv", index=False)
    key = joined[(joined.level == "pooled_changes") & joined.kind.isin(["selected", "singleton"])]
    key = key.sort_values(["variant", "feature", "kind", "subgroup", "adjustment"], ascending=[False, True, False, True, True])
    key.to_csv(EXTENSION / "key_results_pooled.csv", index=False)
    for variant in ("last_3", "all_history"):
        figure(joined, variant, EXTENSION / f"pooled_associations_{variant}.png")
    tails = pd.read_csv(logit / "low_risk_tails.csv")
    tail_summary(tails).to_csv(EXTENSION / "low_risk_tail_summary.csv", index=False)
    checks = {"gradient_free_max_abs_difference": gradient_free_differences(ORIGINAL, logit)}
    transitions = pd.read_csv(logit / "transitions.csv", usecols=["concept", "pair", "kind", "side", "start",
                                                                   "subgroup", "transition", "metric", "value"])
    wide = transitions.set_index(["concept", "pair", "kind", "side", "start", "subgroup", "transition",
                                  "metric"])["value"].unstack("metric")
    residual = wide.loading_change - wide.contribution_change - wide.gradient_change
    checks["midpoint_decomposition_max_abs_residual"] = float(np.nanmax(np.abs(residual.to_numpy())))
    checks["midpoint_decomposition_max_abs_loading_change"] = float(np.nanmax(np.abs(wide.loading_change.to_numpy())))
    checks["frozen_inputs_unchanged"] = before == {path: study.file_sha256(path) for path in before}
    study.atomic_json(EXTENSION / "checks.json", checks)
    return checks


def main() -> None:
    import logging
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("evaluate", "smoke", "compare"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = compare() if arguments.command == "compare" else run(arguments.command)
    print(json.dumps(result, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
