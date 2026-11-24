"""Input-drift baseline: what do the concept measures add to plain input-feature drift?

Input drift of year t against the reference year is the population stability index (PSI) of
each input feature among held-out evaluation patients. Features are zero-inflated counts, so
the bins are "zero" plus the quintiles of the reference year's positive values.

Two comparisons:
1. Whole cohort, per build and year: mean PSI over all 500 features against the change in
   Brier, AUROC and calibration (mean predicted risk minus mortality) since the reference year.
   Descriptive only (no interval): the evaluation rows of different builds overlap.
2. Per concept and year: mean PSI of the concept's own rule features against the change in
   concept agreement since the reference year, pooled, within concept, and within concept with
   the years since the reference removed (rank partial correlations, concept-cluster bootstrap).
   Alongside: mean PSI of all non-rule features, and of non-rule features matched one-to-one to
   the rule features by prevalence (share above zero) at the reference year.

A build or concept shared by several variants (r1 2007, all_history 2008/2009) counts in each.

    .venv/bin/python -m concept_study.input_drift
"""
import numpy as np
import pandas as pd

from concept_study import measures, paths

OUTPUT = paths.OUTPUT / "checks" / "input_drift"
LAST = paths.LAST_OUTCOME_YEAR
BOOT = 1000


def shares(x, edges):
    """Bin shares (+0.5 per bin): exact zero, then the positive values cut at the edges."""
    counts = [np.sum(x <= 0)]
    bins = np.digitize(x[x > 0], edges)
    counts += [np.sum(bins == k) for k in range(len(edges) + 1)]
    counts = np.asarray(counts, dtype=float) + 0.5
    return counts / counts.sum()


def psi(reference, current):
    """PSI per column: bins are exact zero plus quintiles of the reference's positive values."""
    values = np.zeros(reference.shape[1])
    for j in range(reference.shape[1]):
        ref, cur = reference[:, j], current[:, j]
        positive = ref[ref > 0]
        edges = np.unique(np.quantile(positive, [0.2, 0.4, 0.6, 0.8])) if len(positive) else np.array([])
        p, q = shares(ref, edges), shares(cur, edges)
        values[j] = np.sum((q - p) * np.log(q / p))
    return values


def matched_columns(prevalence, columns):
    """Non-rule columns matched one-to-one to the rule columns: nearest prevalence, none reused."""
    free = [j for j in range(len(prevalence)) if j not in set(columns)]
    matched = []
    for j in columns:
        best = min(free, key=lambda k, j=j: abs(prevalence[k] - prevalence[j]))
        matched.append(best)
        free.remove(best)
    return matched


def in_variant(frame, variant):
    """Rows whose comma-separated `variants` include this variant (a row can belong to several)."""
    return frame["variants"].str.split(",").apply(lambda names: variant in names).to_numpy()


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def auroc(p, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def ranks(x):
    return pd.Series(x).rank().to_numpy()


def partial_spearman(x, y, groups=None, covariate=None):
    """Spearman of x and y after removing concept means (groups) and a covariate, on ranks."""
    rx, ry = ranks(x), ranks(y)
    design = [np.ones(len(x))]
    if groups is not None:
        design += [(groups == g).astype(float) for g in np.unique(groups)[1:]]
    if covariate is not None:
        design.append(ranks(covariate))
    design = np.column_stack(design)
    ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    denominator = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
    return float((ex * ey).sum() / denominator) if denominator > 0 else np.nan


def concept_bootstrap(frame, statistic, seed=20261110):
    """Point estimate and 95% interval resampling whole concepts."""
    concepts = frame["concept"].unique()
    rng = np.random.default_rng(seed)
    point = statistic(frame)
    draws = []
    for _ in range(BOOT):
        chosen = rng.choice(concepts, size=len(concepts), replace=True)
        sample = pd.concat([frame[frame["concept"] == c].assign(concept=f"{c}#{k}") for k, c in enumerate(chosen)])
        draws.append(statistic(sample))
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return point, low, high


def main():
    index = measures.load_index()
    cohort, concept_rows, top = [], [], []
    for record in index["builds"]:
        build, reference = record["build"], record["reference"]
        table = measures.load_build(record)
        names = list(table["feature_names"])
        now = table["year"] == reference
        concepts = [c for c in index["concepts"] if c["build"] == build and c["reference"] == reference]
        variants = ",".join(v for v in paths.VARIANTS if any(v in c["variants"] for c in concepts))
        prevalence = (table["features"][now] > 0).mean(axis=0)
        for year in range(reference + 1, LAST + 1):
            then = table["year"] == year
            if not then.any():
                continue
            values = psi(table["features"][now], table["features"][then])
            p0, y0 = table["probability"][now], table["label"][now]
            p1, y1 = table["probability"][then], table["label"][then]
            cohort.append({"build": build, "reference": reference, "year": year, "variants": variants,
                           "years_since_reference": year - reference, "mean_psi": float(values.mean()),
                           "median_psi": float(np.median(values)), "max_psi": float(values.max()),
                           "brier_change": brier(p1, y1) - brier(p0, y0), "auroc_change": auroc(p1, y1) - auroc(p0, y0),
                           "calibration_gap_change": float((p1.mean() - y1.mean()) - (p0.mean() - y0.mean()))})
            order = np.argsort(values)[::-1][:5]
            top.append({"build": build, "reference": reference, "year": year,
                        "top_features": "; ".join(f"{names[j]} {values[j]:.2f}" for j in order)})
            for concept in concepts:
                data = measures.load_concept(concept)
                a, r = data["in_A"], data["in_R"]
                if r[now].sum() == 0 or r[then].sum() == 0:
                    continue
                columns = [names.index(f) for f in concept["rule_features"] if f in names]
                concept_rows.append({
                    "concept": concept["concept"], "variants": ",".join(concept["variants"]),
                    "reference": reference, "year": year, "years_since_reference": year - reference,
                    "rule_feature_psi": float(values[columns].mean()) if columns else np.nan,
                    "other_feature_psi": float(np.delete(values, columns).mean()),
                    "matched_feature_psi": float(values[matched_columns(prevalence, columns)].mean()) if columns else np.nan,
                    "agreement_change": float(a[then & r].mean() - a[now & r].mean()),
                    "activation_rate_change": float(a[then].mean() - a[now].mean()),
                    "rule_rate_change": float(r[then].mean() - r[now].mean()),
                })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.DataFrame(cohort)
    concepts = pd.DataFrame(concept_rows)
    cohort.to_csv(OUTPUT / "cohort_drift.csv", index=False)
    concepts.to_csv(OUTPUT / "concept_drift.csv", index=False)
    pd.DataFrame(top).to_csv(OUTPUT / "top_drifting_features.csv", index=False)

    results = []
    for variant in ("last_3", "all_history"):
        c = cohort[in_variant(cohort, variant)]
        for outcome in ("brier_change", "auroc_change", "calibration_gap_change"):
            results.append({"level": "cohort", "variant": variant, "x": "mean_psi", "y": outcome, "n": len(c),
                            "spearman": float(c["mean_psi"].corr(c[outcome], method="spearman")),
                            "partial_years": partial_spearman(c["mean_psi"].to_numpy(), c[outcome].to_numpy(),
                                                              covariate=c["years_since_reference"].to_numpy()),
                            "low": np.nan, "high": np.nan, "interval": "none: descriptive, rows overlap across builds"})
        k = concepts[in_variant(concepts, variant)].dropna(subset=["rule_feature_psi"])
        for target in ("agreement_change", "activation_rate_change", "rule_rate_change"):
            for x in ("rule_feature_psi", "matched_feature_psi", "other_feature_psi"):
                for label, fn in (
                        ("pooled", lambda f, x=x, t=target: partial_spearman(f[x].to_numpy(), f[t].to_numpy())),
                        ("within_concept", lambda f, x=x, t=target: partial_spearman(f[x].to_numpy(), f[t].to_numpy(), f["concept"].to_numpy())),
                        ("within_concept_and_years", lambda f, x=x, t=target: partial_spearman(
                            f[x].to_numpy(), f[t].to_numpy(), f["concept"].to_numpy(), f["years_since_reference"].to_numpy())),
                ):
                    point, low, high = concept_bootstrap(k, fn)
                    results.append({"level": "concept", "variant": variant, "x": x, "y": target, "adjustment": label,
                                    "n": len(k), "concepts": k["concept"].nunique(), "spearman": point, "low": low, "high": high,
                                    "interval": "concept-cluster bootstrap 95%"})
    pd.DataFrame(results).to_csv(OUTPUT / "input_drift_associations.csv", index=False)
    print(pd.DataFrame(results).round(3).to_string())


if __name__ == "__main__":
    main()
