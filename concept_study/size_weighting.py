"""Size weighting and minimum group size: are the headline estimates driven by tiny cells?

Held-out groups are small (last_3, group R: median 29 patients per concept-year, a tenth at 8 or
fewer). The frozen study uses every non-empty cell, and its rank correlations weight a 5-patient
concept-year like a 100-patient one. Its intervals already carry the sampling noise; the open
questions are the point estimates (attenuation by noisy cells, regression to the mean in changes)
and power. This module redoes the headline estimates (a) with each cell weighted by its size and
(b) restricted to cells with at least 1 (all), 5, 10, 20 and 30 patients. Nothing is refitted.

Estimates (group R, last_3 primary, all_history auxiliary; definitions as in answers.csv):
- agreement vs absolute sensitivity, probability and log-odds scale, within concept and full
  adjustment (concept + predictor-year + mortality + log n): agreement_vs_influence.csv, level rows;
- absolute sensitivity vs Brier and log-odds absolute sensitivity vs Brier, full and full + mean
  risk: same_year_performance.csv (years <= 2014);
- agreement per step: mean over concept-pairs t -> t+1 of the agreement change, its returning part
  and its composition part (answers.csv "composition split"; last_3 only, as frozen);
- input drift: rule-feature PSI vs agreement change since the reference year, within concept and
  years since reference (checks/input_drift; concept-cluster bootstrap as frozen).
All four headline correlations are on group R, so feature and target share one cell and one n.

Weight. Every measure is a mean over the cell's patients (agreement is a share; sensitivity, Brier
and risk are patient means), so its sampling variance is (per-patient variance)/n. The per-patient
variance differs by measure and is unknown for sensitivity, so the common part, n, is the weight
for a level. The inverse binomial variance n/(p(1-p)) is not used: it depends on the agreement value
itself, so it would up-weight cells for being near 0 or 1 and couple the weight to the ranked value.
A change between two cells has variance proportional to 1/n0 + 1/n1, so its weight is
1/(1/n0 + 1/n1) (half the harmonic mean; zero when either cell is empty). For the composition split
the same total-R weight is used for the total, returning and composition parts, so the weighted
parts still add up to the weighted total. Weighting by n changes the estimand from "the typical
concept-year" towards "the typical patient": concepts with large groups count more.

Weighted rank partial correlation (reduces exactly to the frozen one with unit weights):
1. weighted mid-ranks of x, y and each numeric control: weight of smaller values + half the weight
   of ties (unit weights: average rank - 1/2, a shift absorbed by the intercept);
2. weighted least squares of the two rank vectors on the same design as the frozen code (intercept,
   ranked numeric controls, concept and predictor-year indicators);
3. weighted Pearson correlation of the two residual vectors.
With integer weights this equals the unweighted estimate on rows repeated that many times.

Minimum n. The n of the group the measure is computed on: R for every estimate here (agreement is
the share of R patients activating the factor). For a change, the smaller of the two years' R
counts. The curve is a sensitivity curve, not a filter recommendation; the row set is fixed by the
observed n, like the frozen code fixes rows on the point estimate. "n >= 1" is the frozen row set: an
empty cell has no value (the frozen agreement-step mean keeps its five empty pairs as undefined rows,
and they stay in so that its concept draws are the frozen ones). Rows and concepts count defined rows.

Intervals reuse the cached joint bootstrap: patient replicate b of every value and of every cell n
(the weight is recomputed per replicate from that replicate's n), combined with the same seeded
concept draw (bootstrap.concept_resamples). With all cells the row set and the draws equal the
frozen ones, so the weighted and frozen intervals are paired. A restricted row set has fewer
concepts, so its concept draws differ. Input drift has no patient replicates (frozen design): its
weights are fixed and its interval is input_drift.concept_bootstrap with the frozen seed.

    .venv/bin/python -m concept_study.size_weighting
"""
import numpy as np
import pandas as pd

from concept_study import associations, bootstrap, features, input_drift, measures, paths, tables

OUTPUT = paths.OUTPUT / "extensions" / "size_weighting"
MIN_N = (1, 5, 10, 20, 30)
SETTINGS = [("none", m) for m in MIN_N] + [("n", 1)]
ROW_CONTROLS = ("mortality", "log_n", "mean_risk")     # frozen row set: complete over every control of the comparison
CORRELATIONS = (  # label, feature, target, adjustment, frozen table
    ("agreement vs sensitivity | within concept", "agreement", "absolute_sensitivity", "within_concept", "agreement_vs_influence"),
    ("agreement vs log-odds sensitivity | within concept", "agreement", "absolute_sensitivity_logit", "within_concept", "agreement_vs_influence"),
    ("agreement vs sensitivity | full", "agreement", "absolute_sensitivity", "full", "agreement_vs_influence"),
    ("agreement vs log-odds sensitivity | full", "agreement", "absolute_sensitivity_logit", "full", "agreement_vs_influence"),
    ("sensitivity vs Brier | full", "absolute_sensitivity", "brier", "full", "same_year_performance"),
    ("sensitivity vs Brier | full + risk", "absolute_sensitivity", "brier", "full_and_risk", "same_year_performance"),
    ("log-odds sensitivity vs Brier | full", "absolute_sensitivity_logit", "brier", "full", "same_year_performance"),
    ("log-odds sensitivity vs Brier | full + risk", "absolute_sensitivity_logit", "brier", "full_and_risk", "same_year_performance"),
)
STEP_PARTS = ("total change", "returning part", "composition part")
DRIFT = "rule-feature PSI vs agreement change | within concept + years"


# ---------------------------------------------------------------- estimators

def weighted_ranks(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mid-ranks: weight of smaller values plus half the weight of ties."""
    _, codes = np.unique(values, return_inverse=True)
    ties = np.bincount(codes, weights=weights)
    return (np.cumsum(ties) - ties / 2)[codes]


def weighted_partial_rank(x, y, w, numeric=(), indicators=()) -> float:
    """Weighted rank correlation of x and y after weighted least squares on ranked numeric controls
    and indicator columns. Unit weights give associations._partial_rank (same guards)."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    for value in numeric:
        ok &= np.isfinite(value)
    if ok.sum() < 4:
        return np.nan
    w = w[ok] / w[ok].mean()
    design = [np.ones((ok.sum(), 1))]
    design += [weighted_ranks(value[ok], w)[:, None] for value in numeric]
    design += [associations._indicators(np.asarray(labels)[ok]) for labels in indicators]
    design = np.concatenate(design, axis=1)
    targets = np.column_stack([weighted_ranks(x[ok], w), weighted_ranks(y[ok], w)])
    root = np.sqrt(w)[:, None]
    coef, _, rank, _ = np.linalg.lstsq(design * root, targets * root, rcond=None)
    residual = targets - design @ coef
    spread = w @ (targets - w @ targets / w.sum()) ** 2
    if ok.sum() - rank < 3 or np.min(spread) < 1e-9:          # too few rows, or x or y constant
        return np.nan
    if np.min(w @ residual ** 2 / spread) < 1e-12:            # x or y fully explained by the controls
        return np.nan
    e = residual - w @ residual / w.sum()
    return float(w @ (e[:, 0] * e[:, 1]) / np.sqrt(np.prod(w @ e ** 2)))


def correlation(x, y, w, concepts, numeric=(), indicators=(), concept_indicators=False) -> np.ndarray:
    """associations.correlation with weights w (replicates, rows): same concept draws and seed."""
    draws = bootstrap.concept_resamples(concepts, len(x) - 1)
    out = np.full(len(x), np.nan)
    for b, (rows, labels) in enumerate(draws):
        groups = ([labels] if concept_indicators else []) + [np.asarray(i)[rows] for i in indicators]
        out[b] = weighted_partial_rank(x[b, rows], y[b, rows], w[b, rows], [v[b, rows] for v in numeric], groups)
    return out


def weighted_mean(values, w, concepts) -> np.ndarray:
    """Weighted mean over rows per replicate, concepts resampled as in tables.mean_with_interval."""
    out = []
    for b, (rows, _) in enumerate(bootstrap.concept_resamples(concepts, len(values) - 1)):
        v, u = values[b, rows], w[b, rows]
        ok = np.isfinite(v) & np.isfinite(u) & (u > 0)
        out.append((u[ok] @ v[ok]) / u[ok].sum() if ok.any() else np.nan)
    return np.array(out)


def change_weight(n0, n1):
    """1 / (1/n0 + 1/n1): inverse of the variance factor of a difference of two cell means; 0 if a cell is empty."""
    with np.errstate(divide="ignore"):
        return 1 / (1 / np.asarray(n0, float) + 1 / np.asarray(n1, float))


# ---------------------------------------------------------------- estimates

def load_panel() -> tuple:
    """(records with CAV fits, R-group panel point table, replicate cube)."""
    records = [r for r in measures.load_index()["concepts"] if r["valid_fits"]]
    point, cube = associations.build_panel(records, groups=("R",))
    return records, point, cube


def _correlation_cell(label, variant, feature, target, adjustment, weighting, min_n, point, sub):
    controls = associations.controls_for(adjustment, feature, target)
    numeric = [sub[c] for c in controls if c in sub]
    indicators = [point[c].to_numpy() for c in controls if c not in sub and c != "concept"]
    w = sub["n"] if weighting == "n" else np.ones_like(sub["n"])
    values = correlation(sub[feature], sub[target], w, point.concept.to_numpy(), numeric, indicators,
                         "concept" in controls)
    sizes = point.concept.value_counts()
    return {"estimate": label, "variant": variant, "weighting": weighting, "min_n": min_n, "rows": len(point),
            "concepts": len(sizes), "concepts_2plus_rows": int((sizes >= 2).sum()),
            **associations.summarize(values)}


def correlation_cells(point, cube) -> list:
    """Delayed tasks: every headline correlation x variant x setting."""
    from joblib import delayed
    tasks = []
    for variant in paths.VARIANTS:
        base = associations.in_variant(point, variant) & (point.year <= paths.LAST_OUTCOME_YEAR).to_numpy()
        for label, feature, target, adjustment, _ in CORRELATIONS:
            keep = associations.complete_rows(cube, base, (feature, target) + ROW_CONTROLS)
            names = {feature, target, "n"} | set(ROW_CONTROLS)
            for weighting, min_n in SETTINGS:
                rows = keep & (cube["n"][0] >= min_n)
                sub = {name: cube[name][:, rows] for name in names}
                tasks.append(delayed(_correlation_cell)(label, variant, feature, target, adjustment, weighting, min_n,
                                                        point[rows].reset_index(drop=True), sub))
    return tasks


def step_table(records, point, cube) -> pd.DataFrame:
    """Agreement change per step t -> t+1 (last_3): total, returning and composition parts."""
    rows, values = tables.split_replicates([r for r in records if "last_3" in r["variants"]], "concept|agreement")
    position = {key: i for i, key in enumerate(zip(point.concept, point.year))}
    n0 = cube["n"][:, [position[k] for k in zip(rows.concept, rows.start)]]
    n1 = cube["n"][:, [position[k] for k in zip(rows.concept, rows.end)]]
    weight, smaller = change_weight(n0, n1), np.minimum(n0[0], n1[0])
    parts = dict(zip(STEP_PARTS, (values[..., 0], values[..., 1], values[..., 0] - values[..., 1])))
    out = []
    for part, value in parts.items():
        for weighting, min_n in SETTINGS:
            keep = smaller >= min_n if min_n > 1 else np.ones(len(rows), bool)   # all: the frozen rows and draws
            w = weight[:, keep] if weighting == "n" else np.ones_like(weight[:, keep])
            concepts = rows.concept.to_numpy()[keep]
            sizes = pd.Series(rows.concept.to_numpy()[keep & np.isfinite(value[0])]).value_counts()
            out.append({"estimate": f"agreement per step | {part}", "variant": "last_3", "weighting": weighting,
                        "min_n": min_n, "rows": int(sizes.sum()), "concepts": len(sizes),
                        "concepts_2plus_rows": int((sizes >= 2).sum()),
                        **associations.summarize(weighted_mean(value[:, keep], w, concepts))})
    return pd.DataFrame(out)


def drift_frame() -> pd.DataFrame:
    """Frozen concept_drift.csv with the R counts at the reference year and at the year."""
    frame = pd.read_csv(input_drift.OUTPUT / "concept_drift.csv")
    r = measures.GROUPS.index("R")
    counts = {c["concept"]: features.load_features(c)["count"][r] for c in measures.load_index()["concepts"]}
    frame["n0"] = [counts[c][measures.YEARS.index(y)] for c, y in zip(frame.concept, frame.reference)]
    frame["n1"] = [counts[c][measures.YEARS.index(y)] for c, y in zip(frame.concept, frame.year)]
    frame["weight"] = change_weight(frame.n0, frame.n1)
    frame["smaller_n"] = np.minimum(frame.n0, frame.n1)
    return frame


def drift_statistic(frame, weighted: bool) -> float:
    w = frame.weight.to_numpy(float) if weighted else np.ones(len(frame))
    return weighted_partial_rank(frame.rule_feature_psi.to_numpy(), frame.agreement_change.to_numpy(), w,
                                 [frame.years_since_reference.to_numpy(float)], [frame.concept.to_numpy()])


def _drift_cell(variant, weighting, min_n, part):
    value, low, high = input_drift.concept_bootstrap(part, lambda f: drift_statistic(f, weighting == "n"))
    return {"estimate": DRIFT, "variant": variant, "weighting": weighting, "min_n": min_n, "rows": len(part),
            "concepts": part.concept.nunique(), "concepts_2plus_rows": int((part.concept.value_counts() >= 2).sum()),
            "r": value, "low": low, "high": high}


def drift_cells(frame) -> list:
    from joblib import delayed
    tasks = []
    for variant in paths.VARIANTS:
        k = frame[input_drift.in_variant(frame, variant)].dropna(subset=["rule_feature_psi"])
        for weighting, min_n in SETTINGS:
            tasks.append(delayed(_drift_cell)(variant, weighting, min_n, k[k.smaller_n >= min_n].reset_index(drop=True)))
    return tasks


# ---------------------------------------------------------------- frozen values, figure

def frozen_table() -> pd.DataFrame:
    """Frozen point and interval of every estimate, from the frozen CSVs."""
    out = []
    for label, feature, target, adjustment, table in CORRELATIONS:
        frame = pd.read_csv(paths.OUTPUT / f"{table}.csv")
        part = frame[(frame.group == "R") & (frame.feature == feature) & (frame.target == target)
                     & (frame.adjustment == adjustment) & (frame.get("form", "level") == "level")]
        out += [{"estimate": label, "variant": row.variant, "frozen_r": row.r, "frozen_low": row.low,
                 "frozen_high": row.high, "frozen_rows": row.rows} for row in part.itertuples()]
    answers = pd.read_csv(paths.OUTPUT / "answers.csv")
    for part_name in STEP_PARTS:
        row = answers[answers.cell == f"agreement concept: {part_name}"].iloc[0]
        out.append({"estimate": f"agreement per step | {part_name}", "variant": "last_3", "frozen_r": row.value,
                    "frozen_low": row.low, "frozen_high": row.high, "frozen_rows": row.rows})
    drift = pd.read_csv(input_drift.OUTPUT / "input_drift_associations.csv")
    drift = drift[(drift.x == "rule_feature_psi") & (drift.y == "agreement_change")
                  & (drift.adjustment == "within_concept_and_years")]
    out += [{"estimate": DRIFT, "variant": row.variant, "frozen_r": row.spearman, "frozen_low": row.low,
             "frozen_high": row.high, "frozen_rows": row.n} for row in drift.itertuples()]
    return pd.DataFrame(out)


def figure(table: pd.DataFrame, path) -> None:
    """Estimate and 95% interval vs minimum n (last_3 filled, all_history hollow), n-weighted at the right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, grey = "#2a78d6", "#eb6834", "#6b6b6b"
    estimates = list(dict.fromkeys(table.estimate))
    columns = 3
    fig, axes = plt.subplots(int(np.ceil(len(estimates) / columns)), columns, figsize=(5.2 * columns, 3.3 * np.ceil(len(estimates) / columns)))
    slots = [f"n>={m}" for m in MIN_N] + ["weighted"]
    for ax, estimate in zip(axes.flat, estimates):
        for k, (variant, fill) in enumerate((("last_3", True), ("all_history", False))):
            part = table[(table.estimate == estimate) & (table.variant == variant)]
            if part.empty:
                continue
            for x, (weighting, min_n) in enumerate(SETTINGS):
                row = part[(part.weighting == weighting) & (part.min_n == min_n)].iloc[0]
                color = orange if weighting == "n" else blue
                shift = -0.12 + 0.24 * k if table[table.estimate == estimate].variant.nunique() > 1 else 0
                ax.errorbar(x + shift, row.r, [[row.r - row.low], [row.high - row.r]], fmt="D" if weighting == "n" else "o",
                            color=color, mfc=color if fill else "white", ms=6, lw=1.4, capsize=2)
        frozen = table[(table.estimate == estimate) & (table.variant == "last_3")].frozen_r.iloc[0]
        ax.axhline(frozen, color=blue, lw=0.8, ls=":")
        ax.axhline(0, color=grey, lw=0.6)
        counts = table[(table.estimate == estimate) & (table.variant == "last_3")].set_index(["weighting", "min_n"])
        labels = [f"{slot}\n{counts.loc[setting].rows}/{counts.loc[setting].concepts}" for slot, setting in zip(slots, SETTINGS)]
        ax.set_xticks(range(len(slots)), labels, fontsize=7.5)
        ax.set_title(estimate, fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(estimates):]:
        ax.axis("off")
    fig.suptitle("Headline estimates (group R) vs minimum R count per cell (change: smaller of the two years), 95% intervals\n"
                 "blue circles: unweighted, cells with n >= minimum; orange diamond: all cells weighted by n (change: "
                 "1/(1/n0+1/n1)); filled: last_3, hollow: all_history; dotted: frozen last_3; tick counts: last_3 rows/concepts",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    import time
    from joblib import Parallel, parallel_config
    start = time.time()
    records, point, cube = load_panel()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        rows = Parallel(n_jobs=16)(correlation_cells(point, cube) + drift_cells(drift_frame()))
    table = pd.concat([pd.DataFrame(rows), step_table(records, point, cube)], ignore_index=True)
    table = table.merge(frozen_table(), on=["estimate", "variant"], how="left")
    anchor = table[(table.weighting == "none") & (table.min_n == 1)]
    differences = anchor[["r", "low", "high"]].to_numpy(float) - anchor[["frozen_r", "frozen_low", "frozen_high"]].to_numpy(float)
    correlations = anchor[anchor.estimate.isin([c[0] for c in CORRELATIONS + ((DRIFT,),)])]
    print(f"unweighted, all cells vs frozen: max |difference| of point and interval {np.nanmax(np.abs(differences)):.1e}; "
          f"correlation row counts equal: {bool((correlations.rows == correlations.frozen_rows).all())}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "size_weighting.csv", index=False)
    figure(table, OUTPUT / "size_weighting.png")
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(table.drop(columns=["frozen_rows"]).round(3).to_string())
    print(f"runtime {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
