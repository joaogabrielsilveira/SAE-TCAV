"""Aggregates for the error explorer page that the saved notebook outputs lack.

Same frozen inputs, checks and 1,000 shared patient draws as error_profiles_over_time.ipynb
(error_profiles.load_study, SEED 20261106). Writes group summaries only, to stats/error_explorer/:

  performance.csv            four score versions: true year / year input held at the reference
                             year, each frozen and with the death-rate shift; every metric, every
                             year, reference->year, consecutive and reference->later changes
  aggregate.csv              the same, averaged over predictors by years after the reference
  cohort_errors.csv          missed-death and false-alarm rates for newcomers vs returning patients
  concept_year_input.csv     concept activation with the true year vs the year held, and observed
                             vs feature-implied (notebook ridge, refitted identically)
  concept_features.csv       inside each concept rule: missed vs caught deaths and false alarms vs
                             correct survivors on all 500 features, plus risk and mortality
  concept_context_features.csv  the same features among context deaths/survivors in the rule
  context_by_year.csv        context presence beside each later year
  frozen_concept_measures.csv   frozen-study TCAV, sensitivity, agreement, profile shift per year
  factor_link_changes.csv    factor error-link changes for every year pair (plain, adjusted)

Each block is checked against the saved notebook outputs where they overlap.
Run from ~/SAE-TCAV:  .venv/bin/python -u -B error_explorer_export.py
"""
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import error_profiles as ep

SAVED = Path("stats/error_profiles_over_time")
OUTPUT = Path("stats/error_explorer")
SEED, DRAWS = 20261106, 1000
TOLERANCES = {"activation": {"rtol": float(np.finfo(np.float16).eps), "atol": 1e-5},
              "float32": {"rtol": 1e-4, "atol": 1e-5}}
LAST_YEAR = 2014  # 2015 outcomes are never used
RIDGE_FOLDS, RIDGE_ALPHAS = 5, (.1, 1., 10., 100., 1000.)  # as in the notebook
FROZEN_KEYS = {"tcav": "influence|all_negatives|tcav",
               "signed_sensitivity": "influence|all_negatives|signed_sensitivity",
               "absolute_sensitivity": "influence|all_negatives|absolute_sensitivity",
               "share_fits_positive": "influence|all_negatives|share_fits_positive",
               "agreement": "concept|agreement", "overlap": "concept|overlap", "rule_recall": "concept|rule_recall",
               "rule_rate": "concept|rule_rate", "activation_rate": "concept|activation_rate",
               "profile_shift": "profile|shift", "mean_activation": "group|mean_activation",
               "mean_risk": "group|mean_risk", "calibration_gap": "group|calibration_gap",
               "removal_risk_change": "group|removal_risk_change"}


# ---------------------------------------------------------------- small pure helpers
def periods_of(years, reference):
    """Each evaluation year plus later years pooled."""
    years = np.asarray(years)
    periods = {str(y): years == y for y in sorted(np.unique(years))}
    periods["later"] = years > reference
    return periods


def change_pairs(year_names, reference):
    """Reference -> each later year and later pooled, then every consecutive pair (no duplicates)."""
    years = [y for y in year_names if y != "later"]
    first = [(str(reference), y) for y in years if y != str(reference)] + [(str(reference), "later")]
    return list(dict.fromkeys(first + list(zip(years[:-1], years[1:]))))


def rule_from_key(key):
    """Inverse of error_profiles.rule_key."""
    return [(int(j), op, float.fromhex(value)) for j, op, value in json.loads(key)]


def summarize_columns(draws, **columns):
    """summarize_matrix plus constant columns."""
    return ep.summarize_matrix(draws).assign(**columns)


def mean_over_predictors(arrays):
    """Per-draw mean over predictors; a draw undefined for any predictor stays undefined."""
    return np.mean(np.vstack(arrays), axis=0)


# ---------------------------------------------------------------- blocks
def performance_draws(frame, weights, threshold, reference, shifted):
    """{(period, version, metric): draws} with the notebook's per-draw metrics."""
    y, p, years = (frame[c].to_numpy() for c in ("label", "probability", "year"))
    metrics = list(ep.baseline_values(y, p, threshold))
    draws = {}
    for period, at in periods_of(years, reference).items():
        for version in ("frozen", "shifted"):
            estimates = []
            for d, w in enumerate(weights[:, at]):
                score = p[at] if version == "frozen" else shifted[d, at]
                estimates.append(ep.baseline_values(y[at], score, threshold, w) if np.isfinite(score).all() else None)
            for metric in metrics:
                draws[(period, version, metric)] = np.array([e[metric] if e else np.nan for e in estimates])
    return draws


def performance_rows(draws, reference, source):
    periods = list(dict.fromkeys(k[0] for k in draws))
    rows = []
    for (period, version, metric), values in draws.items():
        rows.append({"reference": reference, "input": source, "version": version, "metric": metric,
                     "kind": "level", "start": period, "end": period, **ep.summarize_draws(values)})
    for start, end in change_pairs(periods, reference):
        for version in ("frozen", "shifted"):
            for metric in {k[2] for k in draws}:
                change = draws[(end, version, metric)] - draws[(start, version, metric)]
                rows.append({"reference": reference, "input": source, "version": version, "metric": metric,
                             "kind": "change", "start": start, "end": end, **ep.summarize_draws(change)})
    return rows


def check_performance(rows, reference):
    """True-year levels and reference->year changes must equal the notebook's file."""
    mine = pd.DataFrame(rows).query("input == 'true_year'")
    mine["period"] = np.where(mine["kind"] == "level", mine["end"], mine["start"] + "→" + mine["end"])
    saved = pd.read_csv(SAVED / f"{reference}_performance.csv", dtype={"period": str})
    both = mine.merge(saved, on=["period", "version", "metric"], suffixes=("", "_saved"), validate="one_to_one")
    if len(both) != len(saved):
        raise ValueError(f"{reference}: {len(saved) - len(both)} saved performance rows not reproduced")
    for column in ("value", "low", "high"):
        if not np.array_equal(both[column].isna(), both[f"{column}_saved"].isna()):
            raise ValueError(f"{reference}: performance {column} undefined in different places than the notebook")
        gap = np.nanmax(np.abs(both[column] - both[f"{column}_saved"]))
        if not (gap <= 1e-12 or np.isnan(gap)):
            raise ValueError(f"{reference}: performance {column} differs from the notebook by {gap}")
    return float(np.nanmax(np.abs(both["value"] - both["value_saved"])))


def cohort_errors(frame, weights, threshold, reference):
    """Missed-death and false-alarm rates for newcomers (first year) vs returning patients."""
    y, p, years, first = (frame[c].to_numpy() for c in ("label", "probability", "year", "first_year"))
    cohorts = {"all": np.ones(len(y), bool), "newcomers": first == years, "returning": first < years}
    rows = []
    for period, at in periods_of(years, reference).items():
        for cohort, keep in cohorts.items():
            for side, label, error in (("deaths", 1, p < threshold), ("survivors", 0, p >= threshold)):
                base = at & keep & (y == label)
                denominator, numerator = weights[:, base].sum(axis=1), weights[:, base & error].sum(axis=1)
                rate = np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)
                rows.append({"reference": reference, "period": period, "cohort": cohort, "side": side,
                             "rows": int(base.sum()), "errors": int((base & error).sum()), **ep.summarize_draws(rate)})
    return rows


def held_lookup(build, reference):
    stored = np.load(f"stats/concept_agreement_study/year_input/build_{build}_{reference}.npz", allow_pickle=True)
    keys = list(zip(stored["patient_id"].astype(str), stored["year"].astype(int)))
    return {"position": {k: i for i, k in enumerate(keys)}, "probability": stored["probability_swapped"].astype(float),
            "codes": {s: np.asarray(stored[f"codes_{s}"], float) for s in (42, 135, 215)}}


def year_held_inputs(frame, codes, held, reference):
    """Probabilities and SAE codes with the year input held at the reference year (later rows only)."""
    keys = list(zip(frame["patient_id"].astype(str), frame["year"].astype(int)))
    later = frame["year"].to_numpy() > reference
    missing = [k for k, l in zip(keys, later) if l and k not in held["position"]]
    if missing:
        raise ValueError(f"{reference}: {len(missing)} later evaluation rows lack a year-held score")
    position = np.array([held["position"].get(k, -1) for k in keys])
    probability = frame["probability"].to_numpy(float).copy()
    probability[later] = held["probability"][position[later]]
    held_codes = {s: np.asarray(codes[s], float).copy() for s in codes}
    for s in held_codes:
        held_codes[s][later] = held["codes"][s][position[later]]
    # At the reference year the held year is the true year: stored values must agree where present.
    same = (~later) & (position >= 0)
    check = {"reference_rows_compared": int(same.sum())}
    if same.any():
        check["max_probability_gap"] = float(np.max(np.abs(held["probability"][position[same]] - frame["probability"].to_numpy()[same])))
        check["max_code_gap"] = float(max(np.max(np.abs(held["codes"][s][position[same]] - np.asarray(codes[s], float)[same])) for s in codes))
    return probability, held_codes, check


def within_concept_features(frame, member, weights, threshold, reference, quantities):
    """Inside one rule: error minus correct, per period and per change pair, every quantity."""
    sub, w = frame[member].reset_index(drop=True), weights[:, member]
    names = list(quantities)
    profile, draws = ep.year_profile(sub, {q: np.asarray(v)[member] for q, v in quantities.items()}, w, threshold, reference)
    periods = list(dict.fromkeys(profile["period"]))
    rows = []
    views = {"deaths": ("missed_deaths", "caught_deaths"), "survivors": ("false_alarms", "correct_survivors"),
             "all": ("all", None)}
    for view, (error, correct) in views.items():
        difference = {p: np.column_stack([draws[(p, error, q)] - (draws[(p, correct, q)] if correct else 0.) for q in names])
                      for p in periods}
        for p in periods:
            rows.append(summarize_columns(difference[p], quantity=names, view=view, kind="difference", start=p, end=p))
        for start, end in change_pairs(periods, reference):
            rows.append(summarize_columns(difference[end] - difference[start], quantity=names, view=view,
                                          kind="change", start=start, end=end))
    counts = profile[profile["quantity"] == names[0]][["period", "group", "rows", "patients", "deaths", "survivors"]]
    return pd.concat(rows, ignore_index=True), counts


def factor_link_draws(values, targets, weights, frame, reference):
    """The notebook's factor_links loop, returning its draws for every period."""
    draws = {}
    for period, at in periods_of(frame["year"].to_numpy(), reference).items():
        layout = ep.tie_layout(values[at])
        local = {name: np.broadcast_to(t, weights.shape)[:, at] for name, t in targets.items()}
        for name in targets:
            draws[(period, name)] = np.full((len(weights), values.shape[1]), np.nan)
        for d, full in enumerate(weights[:, at].astype(float)):
            cached = {}
            for name, t in local.items():
                ok = np.isfinite(t[d])
                w = np.where(ok, full, 0.)
                if w.sum() == 0:
                    continue
                key = ok.tobytes()
                if key not in cached:
                    cached[key] = ep.weighted_rank_matrix(layout, w)
                draws[(period, name)][d] = ep.weighted_pearson(cached[key], ep.weighted_ranks(np.where(ok, t[d], 0.), w), w)
    return draws


def frozen_measure_rows(record, case_reference):
    """Frozen-study replicate arrays -> levels and changes (own bootstrap, groups R / A / all)."""
    from concept_study import measures, tables
    data = measures.load_replicates(record)
    years = np.array(measures.YEARS)
    keep = [i for i, y in enumerate(years) if case_reference <= y <= LAST_YEAR]
    names = [str(years[i]) for i in keep]
    rows = []
    for name, key in FROZEN_KEYS.items():
        if key not in data:
            continue
        values = data[key]
        if key.startswith("concept|"):
            groups = {"concept": values}
        elif key.startswith("profile|"):
            groups = {g: values[:, measures.PROFILE_GROUPS.index(g)] for g in ("A", "R", "all")}
        else:
            groups = {g: values[:, measures.GROUPS.index(g)] for g in ("A", "R", "all")}
        for group, value in groups.items():
            value = value[:, keep].astype(float)
            point, low, high = tables._interval(value)
            for j, year in enumerate(names):
                rows.append({"measure": name, "group": group, "kind": "level", "start": year, "end": year,
                             "value": point[j], "low": low[j], "high": high[j]})
            for start, end in change_pairs(names, case_reference):
                if end == "later":
                    continue
                change = value[:, names.index(end)] - value[:, names.index(start)]
                point, low, high = tables._interval(change)
                rows.append({"measure": name, "group": group, "kind": "change", "start": start, "end": end,
                             "value": point, "low": low, "high": high})
    return rows


# ---------------------------------------------------------------- run
def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started, log = time.perf_counter(), {}
    stamp = lambda name: log.__setitem__(name, round(time.perf_counter() - started, 1)) or print(name, log[name], "s", flush=True)
    study = ep.load_study(TOLERANCES)
    cases = study["cases"]
    patient_sets = [view["frame"]["patient_id"].to_numpy() for case in cases
                    for name, view in case["views"].items() if name in ("evaluation", "entrants")]
    bootstrap = ep.patient_bootstrap(patient_sets, DRAWS, SEED)
    stamp("load")

    performance, cohorts, year_input, concept_rows, concept_counts, context_features = [], [], [], [], [], []
    context_rows, frozen_rows, checks, change_draws = [], [], [], {}
    for case in cases:
        ref, build, threshold = case["reference"], case["build"], case["threshold"]
        evaluation = case["views"]["evaluation"]
        frame, weights = evaluation["frame"], ep.row_weights(evaluation["frame"]["patient_id"], bootstrap)
        held = held_lookup(build, ref)
        held_probability, held_codes, held_check = year_held_inputs(frame, evaluation["codes"], held, ref)
        checks.append({"reference": ref, "check": "year-held reference rows", **held_check})

        # 1. four score versions
        case_rows = []
        for source, probability in (("true_year", frame["probability"].to_numpy()), ("year_held", held_probability)):
            version_frame = frame.assign(probability=probability)
            risks, shifted, _ = ep.annual_benchmarks(version_frame, weights, ref)
            draws = performance_draws(version_frame, weights, threshold, ref, shifted)
            case_rows += performance_rows(draws, ref, source)
            for (period, version, metric), values in draws.items():
                change_draws[(ref, source, version, metric, period)] = values
            if source == "true_year":
                true_risks = risks
        checks.append({"reference": ref, "check": "performance vs notebook", "max_value_gap": check_performance(case_rows, ref)})
        performance += case_rows
        cohorts += cohort_errors(frame, weights, threshold, ref)
        stamp(f"{ref} performance")

        # 2. concepts: year input, inside-rule features, model risk vs mortality
        quantities = {str(n).strip(): evaluation["features"][:, j] for j, n in enumerate(case["feature_names"])}
        if len(quantities) != evaluation["features"].shape[1]:
            raise ValueError("Feature names collide after stripping spaces")
        y, p = frame["label"].to_numpy(float), frame["probability"].to_numpy(float)
        quantities |= {"model_risk": p, "model_risk_year_held": held_probability, "death": y,
                       "risk_minus_death": p - y, "risk_minus_death_year_held": held_probability - y,
                       # undefined (not zero) for the other outcome, so the mean is a rate over deaths / survivors
                       "missed": np.where(y == 1, (p < threshold).astype(float), np.nan),
                       "false_alarm": np.where(y == 0, (p >= threshold).astype(float), np.nan),
                       "months_with_care": frame["active_months"].to_numpy(float)}
        context = case["context"]
        saved_concepts = pd.read_csv(SAVED / f"{ref}_concept_profiles.csv", dtype={"period": str})
        for concept in case["concepts"]:
            key, record, arrays = concept["record"]["concept"], concept["record"], concept["arrays"]
            true_activation = arrays["activation"].astype(float)
            held_activation = held_codes[record["seed"]][:, record["factor"]]
            ridge = ep.crossfit_ridge(evaluation["features"], arrays["activation"], frame["patient_id"].to_numpy(),
                                      frame["year"].to_numpy(), ref, SEED, RIDGE_FOLDS, RIDGE_ALPHAS)["prediction"]
            values = {"activation": true_activation, "activation_year_held": held_activation,
                      "year_input_effect": true_activation - held_activation,
                      "ridge_activation": ridge, "activation_minus_ridge": true_activation - ridge,
                      "activated": (true_activation > record["cutoff"]).astype(float),
                      "activated_year_held": (held_activation > record["cutoff"]).astype(float)}
            profile, draws = ep.year_profile(frame, values, weights, threshold, ref)
            saved_ridge = saved_concepts.query("concept == @key and quantity == 'ridge_activation'")
            mine_ridge = profile.query("quantity == 'ridge_activation'").merge(saved_ridge, on=["period", "group"], suffixes=("", "_saved"))
            ridge_gap = float(np.nanmax(np.abs(mine_ridge["value"] - mine_ridge["value_saved"])))
            if len(mine_ridge) != len(saved_ridge) or ridge_gap > 1e-9:
                raise ValueError(f"{key}: ridge activation differs from the notebook by {ridge_gap}")
            year_input.append(pd.concat([profile.assign(kind="profile"), ep.profile_contrasts(profile, draws, ref)],
                                        ignore_index=True).assign(concept=key))
            member = arrays["in_R"].astype(bool)
            if member.any():
                within, counts = within_concept_features(frame, member, weights, threshold, ref, quantities)
                concept_rows.append(within.assign(reference=ref, concept=key))
                concept_counts.append(counts.assign(reference=ref, concept=key))
            inside = ep.rule_mask(concept["rule"], context["features"])
            for label, side in ((1, "context_deaths"), (0, "context_survivors")):
                rows = inside & (context["labels"] == label)
                context_features.append(pd.DataFrame({
                    "reference": ref, "concept": key, "side": side, "rows": int(rows.sum()),
                    "quantity": [str(n).strip() for n in case["feature_names"]],
                    "value": context["features"][rows].mean(axis=0) if rows.any() else np.nan}))
            frozen_rows += [dict(reference=ref, concept=key, **row) for row in frozen_measure_rows(record, ref)]
        stamp(f"{ref} concepts")

        # 3. context presence per later year (existing concepts, factor rules, slices)
        groups = [("existing", c["record"]["concept"], c["rule"]) for c in case["concepts"]]
        described = pd.read_csv(SAVED / f"{ref}_factor_descriptions.csv").dropna(subset=["rule"])
        groups += [("factor_rule", key, rule_from_key(key)) for key in described["rule"].unique()]
        slices = pd.read_csv(SAVED / f"{ref}_slices.csv")
        groups += [("slice", row.slice, rule_from_key(row.rule)) for row in slices.itertuples()]
        years = frame["year"].to_numpy()
        for kind, key, rule in groups:
            context_member, member = ep.rule_mask(rule, context["features"]), ep.rule_mask(rule, evaluation["features"])
            group_weights = weights[:1] if kind == "factor_rule" else weights
            for year in sorted(set(years) - {ref}):
                at = years == year
                share = ep.context_shares(context_member, context["labels"], member[at], frame[at].reset_index(drop=True),
                                          group_weights[:, at], threshold, ref)
                context_rows.append({"reference": ref, "kind": kind, "group": key, "year": int(year), **share})
        stamp(f"{ref} context by year")

        # 4. factor error-link changes for every pair (evaluation, plain and adjusted targets)
        keys, codes = ep.factor_columns(evaluation["codes"])
        names = [f"{seed}:{k}" for seed, k in keys]
        targets = {k: v for k, v in ep.choosing_targets(frame, true_risks, weights).items() if k in ("plain", "adjusted")}
        draws = factor_link_draws(codes, targets, weights, frame, ref)
        saved = pd.read_csv(SAVED / f"{ref}_factor_links.csv", dtype={"period": str}).query("set == 'evaluation' and kind == 'link'")
        gap = 0.
        for (period, target), matrix in draws.items():
            mine = ep.summarize_matrix(matrix)["value"].to_numpy()
            theirs = saved.query("period == @period and target == @target").set_index("factor").loc[names, "value"].to_numpy()
            gap = max(gap, float(np.nanmax(np.abs(mine - theirs))))
        if gap > 1e-10:
            raise ValueError(f"{ref}: factor links differ from the notebook by {gap}")
        checks.append({"reference": ref, "check": "factor links vs notebook", "max_value_gap": gap})
        periods = list(dict.fromkeys(p for p, _ in draws))
        links = [summarize_columns(draws[(p, t)], factor=names, kind="link", start=p, end=p, target=t, reference=ref)
                 for p in periods for t in targets]
        links += [summarize_columns(draws[(e, t)] - draws[(s, t)], factor=names, kind="change", start=s, end=e, target=t, reference=ref)
                  for s, e in change_pairs(periods, ref) for t in targets]
        pd.concat(links, ignore_index=True).to_csv(OUTPUT / f"{ref}_factor_link_changes.csv", index=False)
        del draws, links
        stamp(f"{ref} factor links")

    # 5. average over predictors by years after the reference (shared draws)
    aggregate = []
    references = [case["reference"] for case in cases]
    for source in ("true_year", "year_held"):
        for version in ("frozen", "shifted"):
            for metric in {k[3] for k in change_draws}:
                for distance in range(0, LAST_YEAR - min(references) + 1):
                    used = [r for r in references if r + distance <= LAST_YEAR]
                    level = mean_over_predictors([change_draws[(r, source, version, metric, str(r + distance))] for r in used])
                    entry = {"input": source, "version": version, "metric": metric, "distance": distance,
                             "predictors": " ".join(map(str, used)), "n_predictors": len(used)}
                    aggregate.append({**entry, "kind": "level", **ep.summarize_draws(level)})
                    if distance == 0:
                        continue
                    for kind, start in (("vs_reference", lambda r: r), ("vs_previous", lambda r: r + distance - 1)):
                        change = mean_over_predictors([change_draws[(r, source, version, metric, str(r + distance))] -
                                                       change_draws[(r, source, version, metric, str(start(r)))] for r in used])
                        aggregate.append({**entry, "kind": kind, **ep.summarize_draws(change)})

    for name, table in (("performance", pd.DataFrame(performance)), ("aggregate", pd.DataFrame(aggregate)),
                        ("cohort_errors", pd.DataFrame(cohorts)), ("concept_year_input", pd.concat(year_input, ignore_index=True)),
                        ("concept_features", pd.concat(concept_rows, ignore_index=True)),
                        ("concept_feature_counts", pd.concat(concept_counts, ignore_index=True)),
                        ("concept_context_features", pd.concat(context_features, ignore_index=True)),
                        ("context_by_year", pd.DataFrame(context_rows)), ("frozen_concept_measures", pd.DataFrame(frozen_rows)),
                        ("checks", pd.DataFrame(checks))):
        table.to_csv(OUTPUT / f"{name}.csv", index=False)
    stamp("written")
    (OUTPUT / "run.json").write_text(json.dumps({"seconds": log, "seed": SEED, "draws": DRAWS,
                                                  "helper_sha256": __import__("hashlib").sha256(Path(ep.__file__).read_bytes()).hexdigest(),
                                                  "threads": len(os.sched_getaffinity(0))}, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
