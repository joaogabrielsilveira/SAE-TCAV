"""Planted-shift check: do the concept measures react to a known change in the data?

The natural years showed concept drift without any link to performance. To check that the
measures can see a real change at all, the first follow-up year (reference + 1) of every
last_3 build is scored again by the frozen predictor after a known recording change:

- hd_stop_half:  haemodialysis (EVENT_C1DIALISE_HD) stops being recorded for a random half of its patients
- hd_stop_all:   haemodialysis stops being recorded for everyone
- n180_stop_all: CKD stage 5 (DIAGN_N180) stops being recorded for everyone
- placebo_stop:  the feature used by no concept rule whose prevalence (share of follow-up rows above
                 zero) is closest to haemodialysis prevalence in that build stops being recorded

For each concept the module compares the change caused by the planted shift with the natural
change from the reference year to the next year, both for concepts whose rule uses the changed
feature and for the others. Whole-cohort performance (Brier, AUROC, Death-F1 at the frozen
threshold, mean risk) is compared in the same way. R_brier_change compares the same patients
(the rule's patients before the shift).

    .venv/bin/python -m concept_study.planted_shift <build> <reference> [threads]   # one build, saves npz
    .venv/bin/python -m concept_study.planted_shift summarize                      # tables
"""
import sys
import time

import numpy as np
import pandas as pd
import torch

from concept_study import loading, measures, paths, year_input

OUTPUT = paths.OUTPUT / "checks" / "planted_shift"
BUILDS = (("last_3", 2010), ("last_3", 2011), ("last_3", 2012), ("last_3", 2013))
SHIFTS = ("hd_stop_half", "hd_stop_all", "n180_stop_all", "placebo_stop")
HD, N180 = "EVENT_C1DIALISE_HD", "DIAGN_N180"


def placebo_feature(index, features, names):
    """Feature used by no concept rule in the study whose prevalence (share of rows above zero)
    is closest to haemodialysis prevalence in the same rows."""
    used = {name for concept in index["concepts"] for name in concept["rule_features"]}
    share = (features > 0).mean(axis=0)
    target = share[names.index(HD)]
    candidates = [(abs(share[j] - target), name) for j, name in enumerate(names) if name not in used]
    return min(candidates)[1]


def shifted_features(features, names, shift, placebo, rng):
    """Copy of the feature matrix with one planted recording change."""
    x = features.copy()
    column = {"hd_stop_half": HD, "hd_stop_all": HD, "n180_stop_all": N180, "placebo_stop": placebo}[shift]
    j = names.index(column)
    rows = np.flatnonzero(x[:, j] > 0)
    if shift == "hd_stop_half":
        rows = rng.choice(rows, size=len(rows) // 2, replace=False)
    x[rows, j] = 0
    return x, column, len(rows)


def run_build(build, reference, threads=16):
    """Score every planted shift of one build; an existing npz keeps its shifts and only a missing
    shift, or a placebo shift on a different feature, is scored again."""
    import concept_subgroup_drift_pilot as pilot
    target = OUTPUT / f"build_{build}_{reference}.npz"
    index = measures.load_index()
    record = next(b for b in index["builds"] if b["build"] == build and b["reference"] == reference)
    table = measures.load_build(record)
    later = np.flatnonzero(table["year"] == reference + 1)
    names = list(table["feature_names"])
    placebo = placebo_feature(index, table["features"][later], names)
    stored = {k: v for k, v in np.load(target, allow_pickle=True).items() if k != "seconds"} if target.exists() else {}
    todo = [shift for shift in SHIFTS if f"{shift}|probability" not in stored
            or (shift == "placebo_stop" and str(stored["placebo"]) != placebo)]
    if not todo:
        return
    torch.set_num_threads(threads)
    started = time.monotonic()
    inputs = loading.open_inputs("cpu")
    population, cache = inputs["population"], inputs["caches"][build]
    predictor = cache.load(f"P/{reference}")
    device = "cpu" if predictor.get("device") == "cpu" else "cuda"
    roles = pilot.reference_year_roles(population, reference, seed=cache.config["split_seed"])
    system = year_input.restore_frozen_context(population, roles, predictor, device, int(cache.config["split_seed"]))

    rows = year_input.population_rows(population, table["patient_id"][later], table["year"][later])
    columns = pilot._feature_columns(population["X"].shape[1])
    frame = pilot._prediction_frame(population, rows, columns)
    if not np.allclose(frame[columns].to_numpy(), table["features"][later], atol=1e-5):
        raise ValueError("Population features differ from the patient table")
    replayed, _ = year_input.predict(system, frame.iloc[:32], device)
    replay_error = float(np.max(np.abs(replayed - table["probability"][later[:32]])))
    if replay_error > 1e-4:
        raise ValueError(f"Replay differs from cached probabilities ({replay_error})")

    dictionaries = pilot.load_reference_dictionaries(cache, reference, "cpu")
    arrays = {**stored, "positions": later, "replay_error": replay_error, "placebo": placebo}
    rng = np.random.default_rng([20261110, reference])
    for shift in SHIFTS:
        x, column, changed = shifted_features(table["features"][later], names, shift, placebo, rng)
        if shift not in todo:
            continue
        shifted = frame.copy()
        shifted[columns] = x
        probability, embeddings = year_input.predict(system, shifted, device)
        arrays[f"{shift}|probability"] = probability.astype(np.float32)
        arrays[f"{shift}|features"] = x.astype(np.float32)
        arrays[f"{shift}|changed_rows"] = changed
        for seed, dictionary in dictionaries.items():
            arrays[f"{shift}|codes_{seed}"] = pilot.encode_with_reference_sae(dictionary, embeddings, "cpu", 256).astype(np.float16)
        print(f"{build} {reference} {shift}: {changed} rows changed ({column}), {time.monotonic() - started:.0f} s", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez(target, seconds=time.monotonic() - started, **arrays)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def auroc(p, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def death_f1(p, y, threshold):
    predicted = p >= threshold
    tp = float((predicted & (y == 1)).sum())
    return 2 * tp / (predicted.sum() + y.sum()) if predicted.sum() + y.sum() else np.nan


def agreement(a, r):
    return float(a[r].mean()) if r.sum() else np.nan


def summarize():
    """Per-concept and whole-cohort tables comparing planted and natural changes."""
    inputs = loading.open_inputs("cpu")
    frozen = {loading.concept_key(i["build"], i["reference"], i["family"]): i["frozen"] for i in inputs["inventory"]}
    index = measures.load_index()
    concept_rows, cohort_rows = [], []
    for build, reference in BUILDS:
        path = OUTPUT / f"build_{build}_{reference}.npz"
        if not path.exists():
            continue
        stored = dict(np.load(path, allow_pickle=True))
        record = next(b for b in index["builds"] if b["build"] == build and b["reference"] == reference)
        table = measures.load_build(record)
        later = stored["positions"]
        now = np.flatnonzero(table["year"] == reference)
        y = table["label"][later]
        p0 = table["probability"][later]
        threshold = measures.death_threshold(record)
        base = {"brier": brier(p0, y), "auroc": auroc(p0, y), "death_f1": death_f1(p0, y, threshold),
                "mean_risk": float(p0.mean()), "mortality": float(y.mean())}
        natural = {"brier": brier(table["probability"][now], table["label"][now]),
                   "auroc": auroc(table["probability"][now], table["label"][now]),
                   "death_f1": death_f1(table["probability"][now], table["label"][now], threshold),
                   "mean_risk": float(table["probability"][now].mean())}
        for shift in SHIFTS:
            p1 = stored[f"{shift}|probability"].astype(np.float64)
            cohort_rows.append({"build": build, "reference": reference, "shift": shift,
                                "changed_rows": int(stored[f"{shift}|changed_rows"]), "rows": len(later),
                                **{f"{k}_change": (brier(p1, y) if k == "brier" else auroc(p1, y) if k == "auroc"
                                                   else death_f1(p1, y, threshold) if k == "death_f1" else float(p1.mean())) - base[k]
                                   for k in ("brier", "auroc", "death_f1", "mean_risk")},
                                **{f"{k}_natural_change": base[k] - natural[k] for k in ("brier", "auroc", "death_f1", "mean_risk")}})
        for concept in index["concepts"]:
            if concept["build"] != build or concept["reference"] != reference:
                continue
            values = measures.load_concept(concept)
            rule = frozen[concept["concept"]]["rule"]
            a0, r0 = values["in_A"][later], values["in_R"][later]
            natural_agreement = agreement(values["in_A"][later], values["in_R"][later]) - agreement(values["in_A"][now], values["in_R"][now])
            natural_rate = float(values["in_A"][later].mean() - values["in_A"][now].mean())
            for shift in SHIFTS:
                activation = stored[f"{shift}|codes_{concept['seed']}"][:, concept["factor"]].astype(np.float32)
                a1 = activation > concept["cutoff"]
                r1 = rule.mask(stored[f"{shift}|features"])
                column = {"hd_stop_half": HD, "hd_stop_all": HD, "n180_stop_all": N180,
                          "placebo_stop": str(stored["placebo"])}[shift]
                concept_rows.append({
                    "concept": concept["concept"], "build": build, "reference": reference, "shift": shift,
                    "rule_uses_feature": column in concept["rule_features"],
                    "agreement_change": agreement(a1, r1) - agreement(a0, r0),
                    "agreement_change_rule_fixed": agreement(a1, r0) - agreement(a0, r0),
                    "activation_rate_change": float(a1.mean() - a0.mean()),
                    "rule_rate_change": float(r1.mean() - r0.mean()),
                    "mean_activation_in_R_change": float(activation[r0].mean() - values["activation"][later][r0].mean()) if r0.sum() else np.nan,
                    "R_brier_change": (brier(stored[f"{shift}|probability"][r0].astype(float), y[r0]) - brier(p0[r0], y[r0])) if r0.sum() else np.nan,
                    "natural_agreement_change": natural_agreement,
                    "natural_activation_rate_change": natural_rate,
                })
    pd.DataFrame(concept_rows).to_csv(OUTPUT / "planted_shift_concepts.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(OUTPUT / "planted_shift_cohort.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1] == "summarize":
        summarize()
    else:
        run_build(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 16)
