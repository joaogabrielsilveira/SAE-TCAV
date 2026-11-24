"""Error analysis of the frozen DR-TabPFN predictors (post-freeze extension, 10 Nov 2026).

New file; nothing in the frozen study is changed. It asks why the predictors get right what they
get right and where they fail, in four steps:

  1. how good the model is against simple baselines trained on the model's own context rows;
  2. label timing: inputs are event counts of the same calendar year as the death label, so a
     patient who dies mid-year has a shorter record. How much of the performance is that?
  3. error slices at patient level, pooled over later years (risk fifths, the frozen decision
     threshold, the frozen clinical rules, the patient's own rows in the context);
  4. concept layer: inside a rule, does factor on / off separate deaths beyond predicted risk?

Patients. "evaluation" = the held-out evaluation patients of the frozen study. "discovery" =
patients who entered after the reference year and were hashed to the 'fitting' role. No frozen
object (context, SAE, rules, CAVs, thresholds) saw them. Only years after the reference year,
up to 2014, are used.

Variants. last_3 (primary) = predictors all_history 2008-2009 (their window is all history) and
last_3 2010-2013; all_history (auxiliary) = all_history 2008-2013. The one-year 2007 predictor is
left out (its year-input problem is documented in the frozen study).

Intervals are 95% percentile intervals of a patient bootstrap (1,000 draws; a patient's draw count
weights all of its rows, in every predictor and year). Pooled concept statistics also resample concepts.

Run on Fiji, from ~/SAE-TCAV:
    .venv/bin/python -m concept_study.error_analysis prepare
    .venv/bin/python -m concept_study.error_analysis lagged <build> <reference> [threads]
    .venv/bin/python -m concept_study.error_analysis analyse
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from concept_study import paths

OUTPUT = paths.OUTPUT / "extensions" / "error_analysis"
EVENTS = paths.CODE_ROOT / "tidy_event_data.feather"
VARIANTS = {"last_3": [("all_history", 2008), ("all_history", 2009)] + [("last_3", r) for r in range(2010, 2014)],
            "all_history": [("all_history", r) for r in range(2008, 2014)]}
BUILDS = sorted(set(VARIANTS["last_3"] + VARIANTS["all_history"]))
REPLICATES = 1000
SEED = paths.BOOTSTRAP_SEED
BASELINES = ("logistic", "boosting", "count")


# ---------------------------------------------------------------- prepare

def record_timing(events: pd.DataFrame, patients: set) -> pd.DataFrame:
    """Per patient-year: number of care events, months with any care event, first / last such month, death month.

    Events are monthly records. Care events are all events except DEATH (all event types, not only
    the 500 model inputs).
    """
    e = events[events.patient_id.astype(str).isin(patients)].copy()
    e["patient_id"] = e.patient_id.astype(str)
    date = pd.to_datetime(e.date)
    e["year"], e["month"] = date.dt.year, date.dt.month
    care = e[e.event != "DEATH"].groupby(["patient_id", "year"]).month
    table = pd.DataFrame({"events_all": care.size(), "active_months": care.nunique(),
                          "first_month": care.min(), "last_month": care.max()})
    death = e[e.event == "DEATH"].groupby(["patient_id", "year"]).month.min().rename("death_month")
    return table.join(death, how="outer")


def context_rows(population: dict, roles: dict, predictor: dict) -> np.ndarray:
    """Population rows of the frozen predictor's context (same selection as year_input.restore_frozen_context)."""
    reference, year = roles["reference_year"], population["year"]
    rows = (year == reference) & (roles["row_role"] == "tabpfn_context") & ~roles["is_entrant"]
    if predictor.get("context_window", "reference_only") != "reference_only":
        first = min(predictor["context_years"])
        rows |= (year >= first) & (year < reference) & ~population["post_death"]
    rows = np.flatnonzero(rows)
    if len(rows) != predictor["context_rows"]:
        raise ValueError(f"Context has {len(rows)} rows, the cached predictor has {predictor['context_rows']}")
    return rows


def previous_year_rows(patient_ids: np.ndarray, years: np.ndarray) -> np.ndarray:
    """For every row, the row of the same patient one year earlier (-1 when absent)."""
    lookup = {(p, int(y)): i for i, (p, y) in enumerate(zip(patient_ids, years))}
    return np.array([lookup.get((p, int(y) - 1), -1) for p, y in zip(patient_ids, years)])


def baseline_models() -> dict:
    """Untuned baselines (library defaults, fixed seed). Inputs are log(1 + count)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    return {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "boosting": lambda: XGBClassifier(random_state=42, n_jobs=8),
        "count": lambda: LogisticRegression(max_iter=2000),      # one input: total count over the 500 features
    }


def baseline_scores(X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, lag: np.ndarray) -> dict:
    """Death probabilities of each baseline on the test rows.

    Same-year: trained on the context rows' own features. Lagged ("_lag"): trained on the context
    rows' previous-year features (rows without a previous year are dropped) and scored on the test
    rows' previous-year features (NaN when the test row has no previous year).
    """
    logged = np.log1p(X)
    total = np.log1p(X.sum(1, keepdims=True))
    out = {}
    train_lag, test_lag = train[lag[train] >= 0], lag[test] >= 0
    for name, make in baseline_models().items():
        inputs = total if name == "count" else logged
        out[name] = make().fit(inputs[train], y[train]).predict_proba(inputs[test])[:, 1]
        lagged = np.full(len(test), np.nan)
        model = make().fit(inputs[lag[train_lag]], y[train_lag])
        lagged[test_lag] = model.predict_proba(inputs[lag[test[test_lag]]])[:, 1]
        out[name + "_lag"] = lagged
    return out


def study_rows(population: dict, roles: dict) -> np.ndarray:
    """Later-year rows (up to 2014) of evaluation patients and of post-reference entrants in the 'fitting' role."""
    year, role = population["year"], roles["row_role"]
    keep = (year > roles["reference_year"]) & (year <= paths.LAST_OUTCOME_YEAR)
    return np.flatnonzero(keep & ((role == "evaluation") | ((role == "fitting") & roles["is_entrant"])))


def build_rows(inputs: dict, build: str, reference: int, timing: pd.DataFrame, lag: np.ndarray) -> pd.DataFrame:
    """Rows of one build with model and baseline scores; also writes the build's concept memberships."""
    import concept_subgroup_drift_pilot as pilot
    population, cache = inputs["population"], inputs["caches"][build]
    predictor = cache.load(f"P/{reference}")
    roles = pilot.reference_year_roles(population, reference, seed=cache.config["split_seed"])
    rows = study_rows(population, roles)
    position = np.searchsorted(predictor["row_index"], rows)
    if not np.array_equal(predictor["row_index"][position], rows):
        raise ValueError("Rows missing from the cached predictor archive")
    context = context_rows(population, roles, predictor)
    patient = population["patient_id"].astype(str)
    in_context = pd.Series(patient[context]).value_counts()
    X, y, year = population["X"], population["y"], population["year"]
    table = pd.DataFrame({
        "build": build, "reference": reference, "row": rows, "patient_id": patient[rows], "year": year[rows],
        "set": np.where(roles["row_role"][rows] == "evaluation", "evaluation", "discovery"),
        "entrant": roles["is_entrant"][rows], "post_death": population["post_death"][rows],
        "label": y[rows], "model": predictor["probabilities"][position, 1],
        "has_previous_year": lag[rows] >= 0,
        "patient_rows_in_context": in_context.reindex(patient[rows]).fillna(0).to_numpy(int),
        "first_year": [inputs["first_year"][p] for p in patient[rows]],
        "feature_events": X[rows].sum(1),
    })
    for name, values in baseline_scores(X, y, context, rows, lag).items():
        table[name] = values
    table["context_rows"], table["context_mortality"] = len(context), y[context].mean()
    save_concepts(inputs, build, reference, rows, predictor["embeddings"][position])
    return table.join(timing, on=["patient_id", "year"])


def save_concepts(inputs: dict, build: str, reference: int, rows: np.ndarray, embeddings: np.ndarray) -> None:
    """Frozen rule membership and medoid-factor activation of every concept of the build, for the study rows."""
    import concept_subgroup_drift_pilot as pilot
    items = [i for i in inputs["inventory"] if i["build"] == build and i["reference"] == reference]
    dictionaries = pilot.load_reference_dictionaries(inputs["caches"][build], reference, "cpu")
    codes = {seed: pilot.encode_with_reference_sae(d, embeddings, "cpu", 256) for seed, d in dictionaries.items()}
    X = inputs["population"]["X"][rows]
    activation = np.column_stack([codes[int(i["frozen"]["medoid"][0])][:, int(i["frozen"]["medoid"][1])]
                                  for i in items]).astype(np.float32)
    in_rule = np.column_stack([np.asarray(i["frozen"]["rule"].mask(X), dtype=bool) for i in items])
    np.savez(OUTPUT / f"concepts_{build}_{reference}.npz", row=rows, activation=activation, in_R=in_rule,
             key=np.array([f"{build}/{reference}/{i['family']}" for i in items]),
             variants=np.array([",".join(i["variants"]) for i in items]),
             cutoff=np.array([i["frozen"]["cutoff"] for i in items], dtype=float))


def check_against_patient_table(rows: pd.DataFrame) -> dict:
    """Evaluation rows, probabilities and concept memberships must match the frozen patient table."""
    from concept_study import measures
    index = json.loads((paths.PATIENT_TABLE / "index.json").read_text())
    worst, concepts, rules_differing, flags_differing = 0.0, 0, 0, 0
    for (build, reference), part in rows[rows.set == "evaluation"].groupby(["build", "reference"]):
        frozen = measures.load_build({"file": f"build_{build}_{reference}.npz"})
        keys = pd.MultiIndex.from_arrays([frozen["patient_id"].astype(str), frozen["year"]])
        mine = pd.MultiIndex.from_arrays([part.patient_id.astype(str), part.year])
        position = keys.get_indexer(mine)
        later = (frozen["year"] > reference) & (frozen["year"] <= paths.LAST_OUTCOME_YEAR)
        if (position < 0).any() or len(part) != later.sum():
            raise ValueError(f"{build} {reference}: evaluation rows differ from the frozen patient table")
        worst = max(worst, float(np.abs(part.model.to_numpy() - frozen["probability"][position]).max()))
        stored = np.load(OUTPUT / f"concepts_{build}_{reference}.npz")
        at = np.searchsorted(stored["row"], part.row.to_numpy())
        for record in [c for c in index["concepts"] if c["build"] == build and c["reference"] == reference]:
            column = int(np.flatnonzero(stored["key"] == record["concept"])[0])
            concept = measures.load_concept(record)
            concepts += 1
            rules_differing += int((stored["in_R"][at, column] != concept["in_R"][position]).sum())
            flags_differing += int(((stored["activation"][at, column] > record["cutoff"]) != concept["in_A"][position]).sum())
    if worst > 1e-5 or rules_differing:
        raise ValueError(f"Mismatch with the frozen patient table: probabilities {worst}, rule flags {rules_differing}")
    return {"max_probability_difference": worst, "concepts_checked": concepts,
            "rule_flags_differing": rules_differing, "factor_flags_differing": flags_differing}


def prepare() -> None:
    from concept_study import loading
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inputs = loading.open_inputs("cpu")
    population = inputs["population"]
    patient = population["patient_id"].astype(str)
    timing = record_timing(pd.read_feather(EVENTS), set(patient))
    print(f"record timing ready, {time.monotonic() - started:.0f} s", flush=True)
    lag = previous_year_rows(patient, population["year"])
    parts = []
    for build, reference in BUILDS:
        parts.append(build_rows(inputs, build, reference, timing, lag))
        print(f"{build} {reference}: {len(parts[-1])} rows, {time.monotonic() - started:.0f} s", flush=True)
    rows = pd.concat(parts, ignore_index=True)
    check = check_against_patient_table(rows)
    rows.to_csv(OUTPUT / "rows.csv", index=False)
    (OUTPUT / "prepare.json").write_text(json.dumps({**check, "rows": len(rows),
                                                     "patients": int(rows.patient_id.nunique())}, indent=1))
    print(f"{len(rows)} rows, {rows.patient_id.nunique()} patients, {check}, {time.monotonic() - started:.0f} s",
          flush=True)


# ---------------------------------------------------------------- frozen model on previous-year inputs

def lagged(build: str, reference: str, threads: str = "8") -> None:
    """Score later evaluation patient-years with the frozen model, giving it the patient's previous-year
    features (the year input stays the true year). Rows without a previous year are left out."""
    import torch
    import concept_subgroup_drift_pilot as pilot
    from concept_study import loading, year_input
    reference, started = int(reference), time.monotonic()
    target = OUTPUT / f"lagged_{build}_{reference}.npz"
    if target.exists():
        return
    torch.set_num_threads(int(threads))
    inputs = loading.open_inputs("cpu")
    population, cache = inputs["population"], inputs["caches"][build]
    predictor = cache.load(f"P/{reference}")
    roles = pilot.reference_year_roles(population, reference, seed=cache.config["split_seed"])
    system = year_input.restore_frozen_context(population, roles, predictor, "cpu", int(cache.config["split_seed"]))
    lag = previous_year_rows(population["patient_id"].astype(str), population["year"])
    rows = study_rows(population, roles)
    rows = rows[(roles["row_role"][rows] == "evaluation") & (lag[rows] >= 0)]
    columns = pilot._feature_columns(population["X"].shape[1])
    true = pilot._prediction_frame(population, rows[:16], columns)
    replayed, _ = year_input.predict(system, true, "cpu")
    cached = predictor["probabilities"][np.searchsorted(predictor["row_index"], rows[:16]), 1]
    replay_error = float(np.max(np.abs(replayed - cached)))
    if replay_error > 1e-4:
        raise ValueError(f"True-input re-prediction differs from cached probabilities ({replay_error})")
    frame = pilot._prediction_frame(population, lag[rows], columns)
    frame["year"] = population["year"][rows]
    probability, _ = year_input.predict(system, frame, "cpu")
    np.savez(target, row=rows, probability=probability.astype(np.float32), replay_error=replay_error,
             seconds=time.monotonic() - started)
    print(f"{build} {reference}: {len(rows)} rows, replay error {replay_error:.1e}, "
          f"{time.monotonic() - started:.0f} s", flush=True)


# ---------------------------------------------------------------- statistics

def patient_weights(patient_ids: np.ndarray, replicates: int = REPLICATES, seed: int = SEED) -> np.ndarray:
    """Row weights (1 + replicates, rows): row 0 is all ones, the others are patient bootstrap draw counts."""
    unique, inverse = np.unique(np.asarray(patient_ids).astype(str), return_inverse=True)
    draws = np.random.default_rng(seed).multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=replicates)
    return np.vstack([np.ones((1, len(unique))), draws]).astype(np.float32)[:, inverse]


def auroc_parts(label: np.ndarray, score: np.ndarray, weights: np.ndarray) -> tuple:
    """Weighted count of (death, survivor) pairs ranked correctly (ties 1/2) and of all such pairs, per replicate."""
    if not len(label):
        return np.zeros(len(weights)), np.zeros(len(weights))
    order = np.argsort(score, kind="stable")
    ordered = score[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])          # one group per distinct score
    dead = (label[order] == 1).astype(np.float32)
    deaths = np.add.reduceat(weights[:, order] * dead, starts, axis=1).astype(float)
    survivors = np.add.reduceat(weights[:, order] * (1 - dead), starts, axis=1).astype(float)
    below = np.cumsum(survivors, axis=1) - survivors
    return (deaths * (below + 0.5 * survivors)).sum(1), deaths.sum(1) * survivors.sum(1)


def auroc(label: np.ndarray, score: np.ndarray, weights: np.ndarray, strata: np.ndarray = None) -> np.ndarray:
    """Weighted AUROC per replicate. With strata, only pairs inside the same stratum are compared."""
    strata = np.zeros(len(label), int) if strata is None else strata
    correct, pairs = np.zeros(len(weights)), np.zeros(len(weights))
    for value in np.unique(strata):
        at = strata == value
        c, p = auroc_parts(label[at], score[at], weights[:, at])
        correct, pairs = correct + c, pairs + p
    return np.where(pairs > 0, correct / np.where(pairs > 0, pairs, 1), np.nan)


def mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean per replicate (NaN when no weight)."""
    total = weights.sum(1)
    return np.where(total > 0, weights @ values / np.where(total > 0, total, 1), np.nan)


def estimate(name: str, replicates: np.ndarray) -> dict:
    """Point estimate (row 0) and 95% percentile interval of the bootstrap replicates."""
    draws = replicates[1:][np.isfinite(replicates[1:])]
    low, high = np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan)
    return {name: replicates[0], f"{name}_low": low, f"{name}_high": high}


def scores_table(rows: pd.DataFrame, weights: np.ndarray, scores: list, reference_score: str = "model") -> list:
    """AUROC, average precision (point), Brier and Brier skill of each score; differences from the reference score.

    Brier skill compares with always predicting the death rate of the model's own context rows.
    Rows where any listed score is missing are dropped, so all scores share the same rows.
    """
    from sklearn.metrics import average_precision_score
    keep = rows[scores].notna().all(1).to_numpy()
    part, w = rows[keep], weights[:, keep]
    y = part.label.to_numpy(float)
    constant = mean((part.context_mortality.to_numpy() - y) ** 2, w)
    base = {"rows": len(part), "patients": part.patient_id.nunique(), "deaths": int(y.sum())}
    values = {s: {"auroc": auroc(y, part[s].to_numpy(), w), "brier": mean((part[s].to_numpy() - y) ** 2, w)}
              for s in scores}
    out = []
    for s in scores:
        record = {"score": s, **base, **estimate("auroc", values[s]["auroc"]),
                  "average_precision": average_precision_score(y, part[s]) if 0 < y.sum() < len(y) else np.nan,
                  **estimate("brier", values[s]["brier"]),
                  **estimate("brier_skill", 1 - values[s]["brier"] / constant),
                  "mean_risk": part[s].mean(), "mortality": y.mean()}
        if s != reference_score and reference_score in values:
            record.update(estimate(f"auroc_minus_{reference_score}", values[s]["auroc"] - values[reference_score]["auroc"]))
            record.update(estimate(f"brier_minus_{reference_score}", values[s]["brier"] - values[reference_score]["brier"]))
        out.append(record)
    return out


def subsets(rows: pd.DataFrame) -> dict:
    """Row masks of the patient sets reported throughout."""
    return {"evaluation": (rows.set == "evaluation").to_numpy(), "discovery": (rows.set == "discovery").to_numpy(),
            "both": np.ones(len(rows), bool)}


# ---------------------------------------------------------------- step 1: baselines

def step1_baselines(rows: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    out = []
    for name, at in subsets(rows).items():
        for record in scores_table(rows[at], weights[:, at], ["model", *BASELINES]):
            out.append({"set": name, **record})
    return pd.DataFrame(out)


def by_build_year(rows: pd.DataFrame) -> pd.DataFrame:
    """Point estimates per predictor, year and set (no intervals)."""
    from sklearn.metrics import roc_auc_score
    out = []
    for (build, reference, year, name), part in rows.groupby(["build", "reference", "year", "set"]):
        y = part.label.to_numpy()
        record = {"build": build, "reference": reference, "year": year, "set": name, "rows": len(part),
                  "deaths": int(y.sum()), "mortality": y.mean(), "context_mortality": part.context_mortality.iloc[0]}
        for s in ("model", *BASELINES):
            record[f"auroc_{s}"] = roc_auc_score(y, part[s]) if 0 < y.sum() < len(y) else np.nan
            record[f"brier_{s}"] = ((part[s] - y) ** 2).mean()
        out.append(record)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- step 2: label timing

def patient_years(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per patient-year (a patient-year is scored by several predictors); model risk is averaged."""
    first = rows.drop_duplicates(["patient_id", "year"]).set_index(["patient_id", "year"])
    first["model"] = rows.groupby(["patient_id", "year"]).model.mean()
    return first.reset_index()


def step2_record_length(rows: pd.DataFrame) -> pd.DataFrame:
    """Mortality and mean predicted risk by the number of months with any care event in the year."""
    out = []
    for name, at in subsets(rows).items():
        unique = patient_years(rows[at])
        unique["months"] = unique.active_months.fillna(0).astype(int)
        for months, part in unique.groupby("months"):
            out.append({"set": name, "months_with_care_events": months, "patient_years": len(part),
                        "deaths": int(part.label.sum()), "mortality": part.label.mean(), "mean_risk": part.model.mean(),
                        "median_feature_events": part.feature_events.median(),
                        "share_first_year_in_registry": (part.first_year == part.year).mean(),
                        "share_rows_after_death_year": part.post_death.mean()})
    return pd.DataFrame(out)


def step2_death_month(rows: pd.DataFrame) -> pd.DataFrame:
    """Deaths by calendar month of death: record length and how the model ranks them against all survivors."""
    out = []
    for name, at in subsets(rows).items():
        part = rows[at]
        y, p = part.label.to_numpy(), part.model.to_numpy()
        survivors = np.sort(p[y == 0])
        rank = (np.searchsorted(survivors, p, "left") + np.searchsorted(survivors, p, "right")) / 2 / len(survivors)
        deaths = part.assign(rank=rank)[y == 1]
        for month, group in deaths.groupby(deaths.death_month.fillna(0).astype(int)):
            unique = group.drop_duplicates(["patient_id", "year"])
            out.append({"set": name, "death_month": month, "deaths": len(unique),
                        "median_months_with_care_events": unique.active_months.median(),
                        "median_feature_events": unique.feature_events.median(),
                        "mean_risk": group.model.mean(), "auroc_against_all_survivors": group["rank"].mean()})
    return pd.DataFrame(out)


def step2_timing_scores(rows: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """How well record length alone ranks deaths, and what the model adds within the same record length."""
    out = []
    for name, at in subsets(rows).items():
        part, w = rows[at], weights[:, at]
        y = part.label.to_numpy(float)
        months = part.active_months.fillna(0).to_numpy()
        last = part.last_month.fillna(0).to_numpy()
        quarter = np.ceil(part.death_month.to_numpy() / 3)
        candidates = {
            "model": (part.model.to_numpy(), None, None),
            "fewer months with care events": (-months, None, None),
            "earlier last month with care events": (-last, None, None),
            "fewer events over the 500 inputs": (-part.feature_events.to_numpy(), None, None),
            "fewer events per month with care": (-part.feature_events.to_numpy() / np.maximum(months, 1), None, None),
            "model, pairs with the same number of months": (part.model.to_numpy(), months, None),
            "model, pairs with the same last month": (part.model.to_numpy(), last, None),
            "boosting, pairs with the same number of months": (part.boosting.to_numpy(), months, None),
            "model, patients with care events in all 12 months": (part.model.to_numpy(), None, months == 12),
            "model, patients with care events in under 12 months": (part.model.to_numpy(), None, months < 12),
        }
        for q in (1, 2, 3, 4):
            candidates[f"model, deaths of quarter {q} against all survivors"] = (
                part.model.to_numpy(), None, (y == 0) | (quarter == q))
        for label, (score, strata, keep) in candidates.items():
            keep = np.ones(len(part), bool) if keep is None else keep
            out.append({"set": name, "score": label, "rows": int(keep.sum()), "deaths": int(y[keep].sum()),
                        **estimate("auroc", auroc(y[keep], score[keep], w[:, keep],
                                                  None if strata is None else strata[keep]))})
    return pd.DataFrame(out)


def step2_previous_year(rows: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """Same rows, inputs of the same year against inputs of the previous year."""
    scores = ["model", *BASELINES, *[b + "_lag" for b in BASELINES]]
    if "model_previous_year_inputs" in rows and rows.model_previous_year_inputs.notna().any():
        scores.insert(1, "model_previous_year_inputs")
    out = []
    for name, at in subsets(rows).items():
        if rows[at][scores].notna().all(1).sum() == 0:
            continue
        for record in scores_table(rows[at], weights[:, at], scores):
            out.append({"set": name, **record})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- step 3: error slices

def step3_risk_fifths(rows: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """Observed mortality against mean predicted risk in fifths of predicted risk (fifths cut on all rows of the variant)."""
    fifth = pd.qcut(rows.model, 5, labels=False).to_numpy() + 1
    out = []
    for name, at in subsets(rows).items():
        for k in range(1, 6):
            cell = at & (fifth == k)
            part, w = rows[cell], weights[:, cell]
            months = part.active_months.fillna(0)
            out.append({"set": name, "risk_fifth": k, "rows": len(part), "deaths": int(part.label.sum()),
                        **estimate("mortality", mean(part.label.to_numpy(float), w)),
                        "mean_risk": part.model.mean(), "median_months_with_care_events": months.median(),
                        "share_all_12_months": (months == 12).mean()})
    return pd.DataFrame(out)


def frozen_thresholds(variant: str) -> dict:
    """{(build, reference): frozen Death-F1 threshold of the variant} from the frozen patient-table index."""
    index = json.loads((paths.PATIENT_TABLE / "index.json").read_text())
    return {(b["build"], b["reference"]): b["death_threshold"][variant] for b in index["builds"]
            if variant in b["death_threshold"]}


def step3_threshold_groups(rows: pd.DataFrame, weights: np.ndarray, variant: str) -> tuple:
    """Deaths caught / missed and survivors flagged / not flagged at the frozen decision threshold."""
    thresholds = frozen_thresholds(variant)
    cut = np.array([thresholds[(b, r)] for b, r in zip(rows.build, rows.reference)])
    flagged, y = (rows.model.to_numpy() >= cut), rows.label.to_numpy(float)
    names = np.select([(y == 1) & flagged, (y == 1) & ~flagged, (y == 0) & flagged],
                      ["death, flagged", "death, not flagged", "survivor, flagged"], "survivor, not flagged")
    groups, summary = [], []
    for name, at in subsets(rows).items():
        w = weights[:, at]
        caught, deaths, flags = (w @ (flagged & (y == 1))[at]), (w @ y[at]), (w @ flagged[at].astype(float))
        summary.append({"set": name, "rows": int(at.sum()), **estimate("recall", caught / deaths),
                        **estimate("precision", caught / flags), **estimate("death_f1", 2 * caught / (flags + deaths)),
                        **estimate("share_flagged", flags / w.sum(1))})
        for group in ("death, flagged", "death, not flagged", "survivor, flagged", "survivor, not flagged"):
            part = rows[at & (names == group)]
            months = part.active_months.fillna(0)
            groups.append({"set": name, "group": group, "rows": len(part),
                           "patient_years": len(part.drop_duplicates(["patient_id", "year"])),
                           "mean_risk": part.model.mean(), "median_months_with_care_events": months.median(),
                           "share_all_12_months": (months == 12).mean(),
                           "median_last_month_with_care_events": part.last_month.median(),
                           "median_death_month": part.death_month.median(),
                           "median_feature_events": part.feature_events.median(),
                           "share_first_year_in_registry": (part.first_year == part.year).mean(),
                           "share_rows_after_death_year": part.post_death.mean(),
                           "share_with_previous_year": part.has_previous_year.mean()})
    return pd.DataFrame(summary), pd.DataFrame(groups)


def step3_feature_links(rows: pd.DataFrame, X: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Among deaths and among survivors: rank correlation of predicted risk with each input and record covariate.

    Shown separately for discovery and evaluation patients, to see which links repeat.
    """
    from scipy.stats import rankdata
    extra = {"months with care events": rows.active_months.fillna(0).to_numpy(),
             "last month with care events": rows.last_month.fillna(0).to_numpy(),
             "events over the 500 inputs": rows.feature_events.to_numpy(),
             "first year in registry": (rows.first_year == rows.year).to_numpy(float),
             "death month": rows.death_month.to_numpy(float)}
    values = np.column_stack([X[rows.row.to_numpy()], *extra.values()])
    names = list(feature_names) + list(extra)
    columns = {}
    for outcome, label in (("deaths", 1), ("survivors", 0)):
        for name, at in subsets(rows).items():
            cell = at & (rows.label.to_numpy() == label)
            v, p = values[cell], rows.model.to_numpy()[cell]
            rho = np.full(len(names), np.nan)
            for j in range(len(names)):
                ok = np.isfinite(v[:, j])
                if ok.sum() > 2 and np.ptp(v[ok, j]) > 0:
                    rho[j] = np.corrcoef(rankdata(v[ok, j]), rankdata(p[ok]))[0, 1]
            columns[f"{outcome}_{name}"] = rho
            columns[f"{outcome}_{name}_share_nonzero"] = np.nanmean(np.where(np.isfinite(v), v != 0, np.nan), axis=0)
    return pd.DataFrame({"input": names, **columns})


def load_concepts(rows: pd.DataFrame, variant: str) -> list:
    """Per concept of the variant: positions in `rows`, rule membership and factor-on flag."""
    out = []
    for (build, reference), part in rows.groupby(["build", "reference"]):
        stored = np.load(OUTPUT / f"concepts_{build}_{reference}.npz")
        at = np.searchsorted(stored["row"], part.row.to_numpy())
        for column, key in enumerate(stored["key"]):
            if variant in str(stored["variants"][column]).split(","):
                out.append({"concept": str(key), "positions": np.flatnonzero((rows.build == build)
                                                                             & (rows.reference == reference)),
                            "in_R": stored["in_R"][at, column],
                            "in_A": stored["activation"][at, column] > stored["cutoff"][column]})
    return out


def step3_rule_slices(rows: pd.DataFrame, weights: np.ndarray, concepts: list) -> pd.DataFrame:
    """Calibration and discrimination inside each frozen clinical rule, pooled over later years."""
    sets = subsets(rows)
    y, p = rows.label.to_numpy(float), rows.model.to_numpy()
    out = []
    for concept in concepts:
        record = {"concept": concept["concept"]}
        for name in ("evaluation", "discovery", "both"):
            at = concept["positions"][concept["in_R"] & sets[name][concept["positions"]]]
            w = weights[:, at]
            observed, expected = w @ y[at], w @ p[at]
            record.update({f"{name}_rows": len(at), f"{name}_deaths": int(y[at].sum()),
                           f"{name}_mortality": y[at].mean() if len(at) else np.nan,
                           f"{name}_mean_risk": p[at].mean() if len(at) else np.nan,
                           **estimate(f"{name}_mortality_minus_risk", mean(y[at] - p[at], w)),
                           **estimate(f"{name}_observed_over_expected",
                                      np.where(expected > 0, observed / np.where(expected > 0, expected, 1), np.nan)),
                           **estimate(f"{name}_auroc", auroc(y[at], p[at], w))})
        out.append(record)
    return pd.DataFrame(out)


def step3_own_history(rows: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """Model performance by how many of the patient's own earlier rows sit in the model's context."""
    out = []
    for name, at in subsets(rows).items():
        for count in sorted(rows.patient_rows_in_context[at].unique()):
            cell = at & (rows.patient_rows_in_context == count).to_numpy()
            part, w = rows[cell], weights[:, cell]
            y = part.label.to_numpy(float)
            out.append({"set": name, "patient_rows_in_context": count, "rows": len(part), "deaths": int(y.sum()),
                        "mortality": y.mean(), "mean_risk": part.model.mean(),
                        "median_months_with_care_events": part.active_months.fillna(0).median(),
                        **estimate("auroc", auroc(y, part.model.to_numpy(), w))})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- step 4: concept layer

def recalibrated(rows: pd.DataFrame) -> np.ndarray:
    """Predicted risk after one logistic recalibration (intercept and slope on the log-odds), fitted on all rows given."""
    from sklearn.linear_model import LogisticRegression
    logit = np.log(rows.model / (1 - rows.model)).to_numpy()[:, None]
    return LogisticRegression(C=1e6, max_iter=1000).fit(logit, rows.label).predict_proba(logit)[:, 1]


def concept_contrast(concepts: list, y: np.ndarray, p: np.ndarray, weights: np.ndarray, keep: np.ndarray) -> tuple:
    """Per concept and replicate, inside the rule: sums needed for the factor-on against factor-off comparison.

    Returns arrays (concepts, replicates) of: weight, deaths and expected deaths with the factor on and off.
    """
    names = ("n_on", "deaths_on", "expected_on", "n_off", "deaths_off", "expected_off")
    sums = {name: np.zeros((len(concepts), len(weights))) for name in names}
    for k, concept in enumerate(concepts):
        inside = concept["in_R"] & keep[concept["positions"]]
        for side, flag in (("on", concept["in_A"]), ("off", ~concept["in_A"])):
            at = concept["positions"][inside & flag]
            w = weights[:, at]
            sums[f"n_{side}"][k], sums[f"deaths_{side}"][k], sums[f"expected_{side}"][k] = w.sum(1), w @ y[at], w @ p[at]
    return sums


def pooled_contrast(sums: dict, draws: np.ndarray) -> dict:
    """Pooled over concepts (replicate b uses concept draw b and patient replicate b).

    'residual difference' is within concept: (mortality - mean risk) with the factor on minus the same with
    the factor off, averaged over concepts with weight n_on * n_off / (n_on + n_off).
    """
    replicate = np.arange(draws.shape[0])[:, None]
    s = {name: values[draws, replicate] for name, values in sums.items()}                 # (replicates, concepts)
    with np.errstate(all="ignore"):
        residual_on = (s["deaths_on"] - s["expected_on"]) / s["n_on"]
        residual_off = (s["deaths_off"] - s["expected_off"]) / s["n_off"]
        weight = np.nan_to_num(s["n_on"] * s["n_off"] / (s["n_on"] + s["n_off"]))
        difference = np.nansum(weight * (residual_on - residual_off), 1) / weight.sum(1)
        ratio_on = s["deaths_on"].sum(1) / s["expected_on"].sum(1)
        ratio_off = s["deaths_off"].sum(1) / s["expected_off"].sum(1)
    return {**estimate("residual_difference_on_minus_off", difference),
            **estimate("observed_over_expected_on", ratio_on), **estimate("observed_over_expected_off", ratio_off),
            **estimate("ratio_on_over_off", ratio_on / ratio_off)}


def step4_concept_layer(rows: pd.DataFrame, weights: np.ndarray, concepts: list) -> tuple:
    """Inside each frozen rule: do patients with the factor on die more or less than their predicted risk says,
    compared with rule patients with the factor off?"""
    y = rows.label.to_numpy(float)
    rng = np.random.default_rng(SEED)
    draws = np.vstack([np.arange(len(concepts)), rng.integers(0, len(concepts), (REPLICATES, len(concepts)))])
    risks = {"predicted risk": rows.model.to_numpy(), "recalibrated risk": recalibrated(rows)}
    pooled, per_concept = [], []
    for name, at in subsets(rows).items():
        for label, p in risks.items():
            sums = concept_contrast(concepts, y, p, weights, at)
            point = {k: v[:, 0] for k, v in sums.items()}
            pooled.append({"set": name, "risk": label, "concepts": len(concepts),
                           "rule_rows_factor_on": int(point["n_on"].sum()), "deaths_factor_on": int(point["deaths_on"].sum()),
                           "mean_risk_factor_on": point["expected_on"].sum() / point["n_on"].sum(),
                           "rule_rows_factor_off": int(point["n_off"].sum()), "deaths_factor_off": int(point["deaths_off"].sum()),
                           "mean_risk_factor_off": point["expected_off"].sum() / point["n_off"].sum(),
                           **pooled_contrast(sums, draws)})
            if label == "predicted risk":
                for k, concept in enumerate(concepts):
                    per_concept.append({"set": name, "concept": concept["concept"],
                                        **{key: value[k] for key, value in point.items()}})
    return pd.DataFrame(pooled), pd.DataFrame(per_concept)


# ---------------------------------------------------------------- run

def load_rows(variant: str) -> pd.DataFrame:
    rows = pd.read_csv(OUTPUT / "rows.csv", dtype={"patient_id": str})
    keep = [(b, r) in VARIANTS[variant] for b, r in zip(rows.build, rows.reference)]
    rows = rows[keep].reset_index(drop=True)
    rows["model_previous_year_inputs"] = np.nan
    for build, reference in VARIANTS[variant]:
        path = OUTPUT / f"lagged_{build}_{reference}.npz"
        if path.exists():
            stored = np.load(path)
            at = (rows.build == build) & (rows.reference == reference)
            scores = pd.Series(stored["probability"], index=stored["row"])
            rows.loc[at, "model_previous_year_inputs"] = scores.reindex(rows.row[at]).to_numpy()
    return rows


def analyse() -> None:
    from concept_study import loading
    started = time.monotonic()
    population = loading.open_inputs("cpu")["population"]
    X, feature_names = population["X"], list(population["feature_names"])
    tables = {}
    for variant in VARIANTS:
        rows = load_rows(variant)
        weights = patient_weights(rows.patient_id.to_numpy())
        concepts = load_concepts(rows, variant)
        summary, groups = step3_threshold_groups(rows, weights, variant)
        pooled, per_concept = step4_concept_layer(rows, weights, concepts)
        results = {
            "step1_baselines": step1_baselines(rows, weights),
            "step1_by_predictor_year": by_build_year(rows),
            "step2_record_length": step2_record_length(rows),
            "step2_death_month": step2_death_month(rows),
            "step2_timing_scores": step2_timing_scores(rows, weights),
            "step2_previous_year_inputs": step2_previous_year(rows, weights),
            "step3_risk_fifths": step3_risk_fifths(rows, weights),
            "step3_threshold_summary": summary, "step3_threshold_groups": groups,
            "step3_feature_links": step3_feature_links(rows, X, feature_names),
            "step3_rule_slices": step3_rule_slices(rows, weights, concepts),
            "step3_own_history": step3_own_history(rows, weights),
            "step4_concept_layer": pooled, "step4_per_concept": per_concept,
        }
        for name, table in results.items():
            tables.setdefault(name, []).append(table.assign(variant=variant))
        print(f"{variant}: {len(rows)} rows, {len(concepts)} concepts, {time.monotonic() - started:.0f} s", flush=True)
    for name, parts in tables.items():
        table = pd.concat(parts, ignore_index=True)
        table.insert(0, "variant", table.pop("variant"))
        table.to_csv(OUTPUT / f"{name}.csv", index=False)
    (OUTPUT / "run.json").write_text(json.dumps({"replicates": REPLICATES, "seed": SEED, "variants": VARIANTS,
                                                 "seconds": time.monotonic() - started}, indent=1))


if __name__ == "__main__":
    {"prepare": prepare, "lagged": lagged, "analyse": analyse}[sys.argv[1]](*sys.argv[2:])
