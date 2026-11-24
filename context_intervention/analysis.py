"""Descriptive analysis of the existing, frozen patient tables."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from concept_study import measures


GROUPS = ("true_positive", "false_negative", "false_positive", "true_negative")


def primary_builds(index):
    return [item for item in index["builds"]
            if "last_3" in item["variants"] and 2008 <= item["reference"] <= 2013]


def prediction_groups(labels, probabilities, threshold):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all():
        raise ValueError("Groups require observed binary labels and finite probabilities")
    positive = probabilities >= threshold
    return np.select([(labels == 1) & positive, (labels == 1) & ~positive,
                      (labels == 0) & positive], GROUPS[:3], default=GROUPS[3])


def performance(labels, probabilities, threshold, reference_mortality):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if len(y) == 0 or len(y) != len(p) or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("Performance requires aligned, observed labels and probabilities")
    predicted = p >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    brier = np.mean((p - y) ** 2)
    reference_brier = np.mean((reference_mortality - y) ** 2)
    return {"n": len(y), "deaths": int(y.sum()), "true_positives": tp,
            "false_positives": fp, "false_negatives": fn,
            "death_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "average_precision": average_precision_score(y, p) if y.sum() else np.nan,
            "auroc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
            "brier": brier,
            "brier_skill": 1 - brier / reference_brier if reference_brier else np.nan}


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.std(x[keep]) == 0 or np.std(y[keep]) == 0:
        return np.nan
    return float(np.corrcoef(x[keep], y[keep])[0, 1])


def describe_cached_predictions():
    """Return performance, feature and concept tables for the frozen evaluation rows.

    The same patient-year may appear for multiple references. Rows after 2014 are
    explicitly excluded; the frozen concept cutoffs and rules are never refitted.
    """
    index = measures.load_index()
    performance_rows, feature_rows, concept_rows, association_rows = [], [], [], []
    for record in primary_builds(index):
        data = measures.load_build(record)
        reference = record["reference"]
        threshold = float(record["death_threshold"]["last_3"])
        valid = (data["year"] >= reference) & (data["year"] <= 2014)
        if not np.isfinite(data["label"][valid]).all():
            raise ValueError(f"Missing outcome in {record['build']} {reference}")
        identity = list(zip(data["patient_id"][valid].astype(str), data["year"][valid]))
        if len(identity) != len(set(identity)):
            raise ValueError(f"Duplicate patient-year in {record['build']} {reference}")
        groups = np.full(len(valid), "unavailable", dtype=object)
        groups[valid] = prediction_groups(data["label"][valid], data["probability"][valid], threshold)
        concepts = [c for c in index["concepts"] if c["build"] == record["build"] and c["reference"] == reference]
        for year in range(reference, 2015):
            year_mask = valid & (data["year"] == year)
            if not year_mask.any():
                continue
            common = {"build": record["build"], "reference": reference, "year": year}
            performance_rows.append({**common, **performance(data["label"][year_mask], data["probability"][year_mask],
                                                             threshold, float(data["reference_mortality"]))})
            for group in GROUPS:
                mask = year_mask & (groups == group)
                other = year_mask & (groups != group)
                if not mask.any():
                    continue
                x, xo = data["features"][mask].astype(float), data["features"][other].astype(float)
                mean = x.mean(axis=0)
                other_mean = xo.mean(axis=0) if len(xo) else np.full(x.shape[1], np.nan)
                pooled = np.sqrt((x.var(axis=0) + xo.var(axis=0)) / 2) if len(xo) else np.full(x.shape[1], np.nan)
                smd = np.divide(mean - other_mean, pooled, out=np.full_like(mean, np.nan), where=pooled > 0)
                for name, value, spread, difference in zip(data["feature_names"], mean, x.std(axis=0), smd):
                    feature_rows.append({**common, "group": group, "n": int(mask.sum()), "feature": str(name),
                                         "mean": value, "std": spread, "smd_vs_other_groups": difference})
                for record_concept in concepts:
                    concept = measures.load_concept(record_concept)
                    a, active, rule = concept["activation"], concept["in_A"], concept["in_R"]
                    in_rule, out_rule = mask & rule, mask & ~rule
                    row = {**common, "group": group, "concept": record_concept["concept"],
                           "n": int(mask.sum()), "n_rule": int(in_rule.sum()),
                           "n_active": int((mask & active).sum()),
                           "mean_activation": float(np.mean(a[mask])),
                           "mean_activation_in_rule": float(np.mean(a[in_rule])) if in_rule.any() else np.nan,
                           "activation_rate": float(np.mean(active[mask])),
                           "activation_rate_in_rule": float(np.mean(active[in_rule])) if in_rule.any() else np.nan,
                           "activation_rate_outside_rule": float(np.mean(active[out_rule])) if out_rule.any() else np.nan,
                           "rule_rate": float(np.mean(rule[mask])),
                           "agreement_rule_active": float(np.mean(active[in_rule])) if in_rule.any() else np.nan,
                           "active_in_rule": int((mask & rule & active).sum()),
                           "rule_not_active": int((mask & rule & ~active).sum()),
                           "active_not_rule": int((mask & ~rule & active).sum())}
                    sens = concept.get("sensitivity_all_negatives")
                    if sens is not None:
                        values = sens[mask].astype(float)
                        for scale in ("probability", "log_odds"):
                            scaled = values
                            if scale == "log_odds":
                                p = data["probability"][mask].astype(float)
                                denominator = p * (1 - p)
                                scaled = np.divide(values, denominator[:, None], out=np.full_like(values, np.nan),
                                                   where=denominator[:, None] > 0)
                            finite = np.isfinite(scaled).all(axis=0)
                            if finite.any():
                                z = scaled[:, finite]
                                row[f"{scale}_signed_sensitivity"] = float(np.median(z.mean(axis=0)))
                                row[f"{scale}_absolute_sensitivity"] = float(np.median(np.abs(z).mean(axis=0)))
                                row[f"{scale}_tcav"] = float(np.median((z > 0).mean(axis=0)))
                    concept_rows.append(row)
                    for i, name in enumerate(data["feature_names"]):
                        association_rows.append({**common, "group": group, "concept": record_concept["concept"],
                                                 "feature": str(name), "n": int(mask.sum()),
                                                 "correlation": correlation(a[mask], data["features"][mask, i])})
    return tuple(pd.DataFrame(rows) for rows in (performance_rows, feature_rows, concept_rows, association_rows))


def save_cached_analysis(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tables = describe_cached_predictions()
    names = ("baseline_performance", "feature_groups", "concept_groups", "concept_feature_correlations")
    differences = {
        "feature_groups": (["build", "reference", "group", "feature"], ["mean", "std", "smd_vs_other_groups"]),
        "concept_groups": (["build", "reference", "group", "concept"],
                           ["mean_activation", "mean_activation_in_rule", "activation_rate",
                            "activation_rate_in_rule", "activation_rate_outside_rule", "rule_rate",
                            "agreement_rule_active", "probability_signed_sensitivity",
                            "probability_absolute_sensitivity", "probability_tcav",
                            "log_odds_signed_sensitivity", "log_odds_absolute_sensitivity", "log_odds_tcav"]),
        "concept_feature_correlations": (["build", "reference", "group", "concept", "feature"], ["correlation"]),
    }
    for name, table in zip(names, tables):
        if name in differences:
            keys, columns = differences[name]
            columns = [column for column in columns if column in table]
            reference_rows = table.loc[table["year"] == table["reference"], keys + columns]
            reference_rows = reference_rows.rename(columns={column: f"{column}_reference" for column in columns})
            table = table.merge(reference_rows, on=keys, how="left", validate="many_to_one")
            for column in columns:
                table[f"{column}_change"] = table[column] - table[f"{column}_reference"]
        table.to_csv(output / f"{name}.csv.gz", index=False, compression="gzip")
        tables = tuple(table if name == key else old for key, old in zip(names, tables))
    timing_path = Path("stats/concept_agreement_study/extensions/error_analysis/rows.csv")
    if timing_path.exists():
        timing = pd.read_csv(timing_path, dtype={"patient_id": str})
        timing = timing[timing["set"] == "evaluation"].copy()
        groups = []
        for record in primary_builds(measures.load_index()):
            part = timing[(timing["build"] == record["build"]) & (timing["reference"] == record["reference"]) &
                          (timing["year"] <= 2014)].copy()
            part["group"] = prediction_groups(part["label"], part["model"], record["death_threshold"]["last_3"])
            part["first_registry_year"] = part["year"] == part["first_year"]
            groups.append(part)
        timing = pd.concat(groups, ignore_index=True)
        grouped = timing.groupby(["build", "reference", "year", "group"], as_index=False)
        timing_summary = grouped.agg(n=("patient_id", "size"), active_months_median=("active_months", "median"),
                                     first_registry_year_rate=("first_registry_year", "mean"),
                                     death_month_median=("death_month", "median"))
        timing_summary.to_csv(output / "record_timing_groups.csv.gz", index=False, compression="gzip")
    index_path = Path("stats/concept_agreement_study/patient_table/index.json")
    provenance = {"patient_table_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                  "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "record_timing_source": str(timing_path) if timing_path.exists() else None,
                  "record_timing_scope": "Cached future-year evaluation rows; reference-year timing unavailable"}
    (output / "baseline_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return dict(zip(names, tables))
