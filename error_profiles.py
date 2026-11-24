"""Frozen-cache diagnostics for error profiles over time."""

import numpy as np
import pandas as pd
import warnings
import hashlib
import json
import time
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from scipy.special import expit, logit
from scipy.optimize import brentq
from sklearn.tree import DecisionTreeRegressor

# Every frozen predictor serving the last-three-years context, by reference year:
# (saved build name, number of frozen concepts). Data begin in 2007, so the
# 2007-2009 contexts hold every available year; those builds carry other names.
PREDICTORS = {2007: ("r1", 4), 2008: ("all_history", 6), 2009: ("all_history", 10),
              2010: ("last_3", 7), 2011: ("last_3", 14), 2012: ("last_3", 7), 2013: ("last_3", 14)}
FIRST_DATA_YEAR = 2007


def last_three_years(reference):
    """Calendar years a last-three-years context may hold at this reference year."""
    return set(range(max(FIRST_DATA_YEAR, reference - 2), reference + 1))


def population_rows(cached, population):
    """Locate cached patient-years in the frozen population."""
    def keys(data):
        values = list(zip(np.asarray(data["patient_id"]).astype(str), data["year"]))
        if len(values) != len(set(values)):
            raise ValueError("Duplicate patient/year identity")
        return values

    lookup = {key: i for i, key in enumerate(keys(population))}
    wanted = keys(cached)
    if any(key not in lookup for key in wanted):
        raise ValueError("Cached patient/year absent from population")
    rows = np.array([lookup[key] for key in wanted], dtype=int)
    if not np.array_equal(cached["feature_names"], population["feature_names"]):
        raise ValueError("Cached feature names/order disagree")
    if not np.array_equal(cached["features"], population["X"][rows]):
        raise ValueError("Cached features disagree after identity join")
    if not np.array_equal(cached["label"], population["y"][rows]):
        raise ValueError("Cached labels disagree after identity join")
    return rows


def select_rows(population, roles, reference):
    """Keep saved evaluation patients; only first-ever post-reference entrants choose."""
    year, role = np.asarray(population["year"]), np.asarray(roles["row_role"])
    patients = np.asarray(population["patient_id"]).astype(str)
    unique, inverse = np.unique(patients, return_inverse=True)
    first = np.full(len(unique), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, inverse, year)
    newly_joined = first[inverse] > reference
    legal = (year >= reference) & (year <= 2014)
    selected = {"evaluation": np.flatnonzero(legal & (role == "evaluation")),
                "entrants": np.flatnonzero(legal & (year > reference) & (role == "fitting") &
                                            roles["is_entrant"] & newly_joined),
                "direction_fit": np.flatnonzero((year == reference) & (role == "cav_fit"))}
    if set(patients[selected["evaluation"]]) & set(patients[selected["entrants"]]):
        raise ValueError("Evaluation and choosing entrant patients overlap")
    return selected


def join_timing(rows, timing):
    if not timing.index.is_unique:
        raise ValueError("Duplicate patient/year timing")
    timing = timing.assign(timing_row_present=True)
    joined = rows.join(timing, on=["patient_id", "year"])
    joined["timing_row_present"] = joined["timing_row_present"].eq(True)
    death_only = (joined["timing_row_present"] & joined["active_months"].isna() &
                  joined.get("death_month", pd.Series(np.nan, index=joined.index)).notna() &
                  joined.get("events_all", pd.Series(np.nan, index=joined.index)).isna())
    # record_timing outer-joins deaths onto care groups: an existing death-only
    # row represents an observed empty care-event set, unlike an absent row.
    joined.loc[death_only, "active_months"] = 0.
    if "events_all" in joined:
        joined.loc[death_only, "events_all"] = 0.
    joined["death_only_timing"] = death_only
    joined["timing_observed"] = joined["active_months"].notna()
    return joined


def fit_directions(embeddings, membership, patients, seeds):
    """November all-negative fitting recipe, on the caller's explicit contrast pool.

    Existing concepts pass rule positives plus minimum-activation negatives.
    New rules pass every fitting patient, splitting rule against its complement.
    Seeds, stratified resampling, fixed class mass, solver and sign are unchanged.
    """
    X, labels, patients = np.asarray(embeddings, float), np.asarray(membership, int), np.asarray(patients)
    if X.ndim != 2 or len(X) != len(labels) or len(X) != len(patients) or not np.isfinite(X).all():
        raise ValueError("Direction fitting embeddings/labels/patients must be finite and aligned")
    support = [len(np.unique(patients[labels == k])) for k in (0, 1)]
    mass = 2 * min(support)
    directions, states, draw_support = np.full((len(seeds), X.shape[1]), np.nan), [], []
    for position, seed in enumerate(seeds):
        if min(support) < 2:
            states.append("not fittable at the reference year")
            draw_support.append(support)
            continue
        rng = np.random.default_rng(seed)
        pools = [np.flatnonzero(labels == k) for k in (0, 1)]
        draws = [rng.choice(pool, len(pool), replace=True) for pool in pools]
        multiplicity = np.bincount(np.concatenate(draws), minlength=len(labels))
        weights = np.zeros(len(labels), float)
        for k in (0, 1):
            at = labels == k
            weights[at] = multiplicity[at] * (mass / 2) / multiplicity[at].sum()
        resampled = [len(np.unique(patients[(labels == k) & (weights > 0)])) for k in (0, 1)]
        draw_support.append(resampled)
        if min(resampled) < 2:
            states.append("insufficient resampled distinct patients")
            continue
        # pragmatic: the frozen logistic recipe is CPU-only and these matrices are small.
        model = LogisticRegression(C=.1, penalty="l2", solver="liblinear", class_weight=None,
                                   max_iter=1000, random_state=42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, labels, sample_weight=weights)
        norm = np.linalg.norm(model.coef_[0])
        if any(issubclass(w.category, ConvergenceWarning) for w in caught) or not np.isfinite(norm) or norm <= 0:
            states.append("convergence or nonfinite/zero coefficient")
        else:
            directions[position] = model.coef_[0] / norm
            states.append("measured")
    return {"directions": directions, "matching": support[1], "nonmatching": support[0],
            "fit_status": states, "draw_support": draw_support, "class_mass": mass,
            "seeds": list(seeds)}


def baseline_values(labels, probabilities, threshold, weights=None):
    """Frozen-threshold metrics; empty denominators stay non-estimable."""
    y, p = np.asarray(labels), np.asarray(probabilities, float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    death, flagged = y == 1, p >= threshold
    n, deaths, survivors = w.sum(), w[death].sum(), w[~death].sum()
    tp, fp, fn = w[death & flagged].sum(), w[~death & flagged].sum(), w[death & ~flagged].sum()
    ratio = lambda a, b: a / b if b > 0 else np.nan
    both = deaths > 0 and survivors > 0
    return {"n": n, "deaths": deaths, "survivors": survivors, "true_positives": tp,
            "false_positives": fp, "false_negatives": fn, "recall": ratio(tp, deaths),
            "false_alarm_rate": ratio(fp, survivors), "share_flagged": ratio(tp + fp, n),
            "mortality": ratio(deaths, n), "mean_score": ratio(w @ p, n),
            "precision": ratio(tp, tp + fp), "death_f1": ratio(2 * tp, 2 * tp + fp + fn),
            "brier": ratio(w @ (p - y) ** 2, n),
            "auroc": roc_auc_score(y, p, sample_weight=w) if both else np.nan,
            "average_precision": average_precision_score(y, p, sample_weight=w) if deaths > 0 else np.nan,
            "one_death": ratio(1., deaths), "one_survivor": ratio(1., survivors), "one_patient": ratio(1., n)}


def load_study(tolerances):
    """Read frozen caches only; verify baseline/rules/directions/encoding before use.

    Timing missingness is returned explicitly. A missing reference timing value
    disables timing-dependent analyses; unrelated plain descriptions remain valid.
    No predictor head is restored, replayed, fitted or called here.
    """
    import cmr_common
    import concept_subgroup_drift_pilot as pilot
    from concept_study import loading, measures, paths, error_analysis

    started = time.perf_counter()
    inputs, index = loading.open_inputs("cpu"), measures.load_index()
    population, inventory = inputs["population"], inputs["inventory"]
    directions, direction_counts = loading.load_cav_directions()
    baseline = pd.read_csv(paths.CODE_ROOT / "stats/context_intervention/baseline_performance.csv.gz")
    wanted_ids = set(np.asarray(population["patient_id"]).astype(str))
    events = pd.read_feather(error_analysis.EVENTS)
    input_paths = {error_analysis.EVENTS, paths.PATIENT_TABLE / "index.json", paths.CAV_DIRECTIONS,
                   paths.CODE_ROOT / "stats/context_intervention/baseline_performance.csv.gz"}
    # Read calendar dates before selecting outcomes: 2015 outcomes never enter timing.
    event_year = pd.to_datetime(events["date"]).dt.year
    timing = error_analysis.record_timing(events.loc[event_year <= 2014], wanted_ids)
    del events
    cases, checks, coverage, composition = [], [], [], []
    serving = sorted((b["reference"], b["build"]) for b in index["builds"] if "last_3" in b["variants"])
    if serving != sorted((reference, build) for reference, (build, _) in PREDICTORS.items()):
        raise ValueError(f"Saved builds serving last_3 differ from PREDICTORS: {serving}")
    for reference, (build, expected_concepts) in PREDICTORS.items():
        records = [b for b in index["builds"] if b["build"] == build and b["reference"] == reference
                   and "last_3" in b["variants"]]
        if len(records) != 1:
            raise ValueError(f"Expected one {build}/{reference} saved build")
        record = records[0]
        threshold = float(record["death_threshold"]["last_3"])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid frozen threshold")
        entries = [i for i in inventory if i["build"] == build and i["reference"] == reference]
        if not entries:
            raise ValueError(f"Missing frozen inventory {build}/{reference}")
        roles, cache = entries[0]["roles"], inputs["caches"][build]
        if any(i["roles"]["identity"] != roles["identity"] for i in entries):
            raise ValueError("Concepts disagree on role identity")
        # The frozen study projects away context/discovery roles. Reconstruct the
        # original map with its saved seed (only reference-year labels are read).
        original_roles = pilot.reference_year_roles(population, reference, seed=cache.config["split_seed"])
        if original_roles["identity"] != roles["identity"]:
            raise ValueError("Reconstructed original role identity disagrees with frozen predictor")
        for name in ("evaluation", "cav_fit", "rule_discovery", "concept_selection"):
            np.testing.assert_array_equal(original_roles["row_role"] == name, roles["row_role"] == name)
        roles = original_roles
        chosen = select_rows(population, roles, reference)
        meta, archive, archive_rows = cmr_common.verified_predictor_archive(population, cache, reference, roles)
        input_paths.update((archive, archive.with_name("meta.json")))
        if meta["reference_year"] != reference:
            raise ValueError("Predictor reference year mismatch")
        context = error_analysis.context_rows(population, roles, meta)
        # The window label differs by build; the context's actual years are what must match.
        if set(np.unique(population["year"][context]).tolist()) != last_three_years(reference):
            raise ValueError(f"Context years at {reference} are not the last three available years")
        context_patients = population["patient_id"][context]
        if np.intersect1d(population["patient_id"][chosen["entrants"]], context_patients).size:
            raise ValueError("Choosing entrants were used in the predictor context")
        dictionaries = pilot.load_reference_dictionaries(cache, reference, "cpu")
        for seed in dictionaries:
            input_paths.update(p for p in cache._unit_path(f"S/{seed}/{reference}").iterdir() if p.is_file())
        if set(dictionaries) != {42, 135, 215}:
            raise ValueError("Frozen SAE seeds differ")
        views = {}
        for name, rows in chosen.items():
            position = np.searchsorted(archive_rows, rows)
            if np.any(position >= len(archive_rows)) or not np.array_equal(archive_rows[position], rows):
                raise ValueError("Selected patient rows missing from predictor archive")
            frame = pd.DataFrame({"patient_id": np.asarray(population["patient_id"])[rows].astype(str),
                                  "year": population["year"][rows], "label": population["y"][rows],
                                  "entrant": roles["is_entrant"][rows],
                                  "first_year": [inputs["first_year"][str(p)] for p in population["patient_id"][rows]]})
            frame = join_timing(frame, timing)
            if not np.isin(frame["label"], [0, 1]).all():
                raise ValueError("Missing/nonbinary outcome in selected years")
            values = {key: cmr_common.read_array_rows(archive, key, position)
                      for key in ("embeddings", "gradients", "probabilities")}
            if values["embeddings"].shape != (len(rows), 192) or values["gradients"].shape != (len(rows), 192):
                raise ValueError("Cached embedding/gradient dimensions differ")
            if not all(np.isfinite(a).all() for a in values.values()):
                raise ValueError("Nonfinite saved representation")
            p = values.pop("probabilities")[:, 1]
            if np.any((p < 0) | (p > 1)):
                raise ValueError("Probability outside [0,1]")
            frame["probability"] = p
            codes = {s: pilot.encode_with_reference_sae(dictionaries[s], values["embeddings"], "cpu", 256)
                     for s in (42, 135, 215)}
            if any(v.shape != (len(rows), 288) for v in codes.values()):
                raise ValueError("Frozen factor count differs from 864")
            views[name] = {"rows": rows, "frame": frame, "features": population["X"][rows],
                           "codes": codes, **values}
            for year in sorted(frame["year"].unique()):
                part = frame[frame["year"] == year]
                coverage.append({"reference": reference, "set": name, "year": year, "rows": len(part),
                                 "timing_rows": int(part["timing_observed"].sum()),
                                 "no_timing": int((~part["timing_observed"]).sum()),
                                 "absent_timing_rows": int((~part["timing_row_present"]).sum()),
                                 "death_only_zero_care_rows": int(part["death_only_timing"].sum()),
                                 "zero_months": int((part["active_months"] == 0).sum())})
                composition.append({"reference": reference, "set": name, "year": year, "rows": len(part),
                                    "patients": part["patient_id"].nunique(),
                                    "first_year_rows": int((part["first_year"] == year).sum()),
                                    "entrant_rows": int(part["entrant"].sum()),
                                    "mean_active_months": part["active_months"].mean(), "mortality": part["label"].mean()})
                if name == "entrants":
                    broad = ((population["year"] == year) & (roles["row_role"] == "fitting") &
                             roles["is_entrant"])
                    original_rows = np.flatnonzero(broad)
                    gap = np.array([inputs["first_year"][str(p)] <= reference
                                    for p in population["patient_id"][original_rows]])
                    composition[-1].update(
                        original_fitting_entrant_rows=len(original_rows),
                        excluded_gapreturner_rows=int(gap.sum()),
                        excluded_context_exposed_rows=int(np.isin(population["patient_id"][original_rows],
                                                                  context_patients).sum()),
                        chosen_context_exposed_rows=0)
        build_path = paths.PATIENT_TABLE / record["file"]
        input_paths.add(build_path)
        saved_year = cmr_common.read_array_rows(build_path, "year")
        valid = np.flatnonzero((saved_year >= reference) & (saved_year <= 2014))
        saved = {k: cmr_common.read_array_rows(build_path, k, valid)
                 for k in ("patient_id", "year", "label", "probability", "features")}
        saved["feature_names"] = cmr_common.read_array_rows(build_path, "feature_names")
        if saved["features"].shape[1] != 500 or len(set(saved["feature_names"])) != 500:
            raise ValueError("Expected 500 distinct frozen features")
        eval_rows = population_rows(saved, population)
        evaluation, fitting = views["evaluation"], views["direction_fit"]
        if not np.array_equal(eval_rows, evaluation["rows"]):
            raise ValueError("Saved evaluation identities/order differ from frozen roles")
        np.testing.assert_array_equal(saved["probability"], evaluation["frame"]["probability"])
        for seed in (42, 135, 215):
            stored_codes = cmr_common.read_array_rows(build_path, f"codes_{seed}", valid)
            encoded = evaluation["codes"][seed]
            np.testing.assert_allclose(encoded, stored_codes, **tolerances["activation"])
            checks.append({"reference": reference, "check": "864-factor encoding", "seed": seed,
                           "factors": encoded.shape[1], "rows": len(encoded),
                           "different_values": int(np.count_nonzero(encoded != stored_codes)),
                           "different_after_float16_rounding": int(np.count_nonzero(encoded.astype(np.float16) != stored_codes)),
                           "rtol": tolerances["activation"]["rtol"], "atol": tolerances["activation"]["atol"],
                           "max_abs_error": float(np.max(np.abs(encoded - stored_codes)))})
        # A's saved baseline holds 2008-2013 only; 2007 is checked against its own build file above.
        saved_baseline = baseline[(baseline["build"] == build) & (baseline["reference"] == reference)]
        for year in range(reference, 2015):
            part = evaluation["frame"].query("year == @year")
            computed = baseline_values(part["label"].to_numpy(), part["probability"].to_numpy(), threshold)
            reference_mortality = float(cmr_common.read_array_rows(build_path, "reference_mortality"))
            reference_brier = np.mean((reference_mortality - saved["label"][saved["year"] == year]) ** 2)
            computed["brier_skill"] = 1 - computed["brier"] / reference_brier if reference_brier else np.nan
            if saved_baseline.empty:
                checks.append({"reference": reference, "year": year, "check": "baseline (no saved copy)", **computed})
                continue
            truth = saved_baseline[saved_baseline["year"] == year]
            if len(truth) != 1:
                raise ValueError("Expected one saved baseline per predictor/year")
            for key in ("n", "deaths", "true_positives", "false_positives", "false_negatives",
                        "recall", "precision", "death_f1", "average_precision", "auroc", "brier", "brier_skill"):
                np.testing.assert_allclose(computed[key], truth.iloc[0][key], rtol=0, atol=1e-12, equal_nan=True,
                                           err_msg=f"Baseline {reference}/{year}/{key}")
            checks.append({"reference": reference, "year": year, "check": "baseline", **computed})
        concepts = []
        selected = sorted([c for c in index["concepts"] if c["build"] == build and c["reference"] == reference],
                          key=lambda c: c["concept"])
        if len(selected) != expected_concepts:
            raise ValueError(f"Expected {expected_concepts} concepts at {reference}")
        for concept in selected:
            matches = [i for i in entries if i["family"] == concept["family"]]
            if len(matches) != 1:
                raise ValueError("Frozen rule inventory identity mismatch")
            rule = matches[0]["frozen"]["rule"]
            concept_path = paths.PATIENT_TABLE / "concepts" / concept["file"]
            input_paths.add(concept_path)
            with np.load(concept_path) as stored:
                arrays = {k: stored[k][valid] for k in stored.files}
            np.testing.assert_array_equal(rule.mask(evaluation["features"]), arrays["in_R"])
            np.testing.assert_array_equal(arrays["in_A"], arrays["activation"] > concept["cutoff"])
            encoded = evaluation["codes"][concept["seed"]][:, concept["factor"]]
            np.testing.assert_allclose(encoded, arrays["activation"], **tolerances["float32"])
            activation = fitting["codes"][concept["seed"]][:, concept["factor"]]
            cutoff = float(activation[np.isfinite(activation)].min())
            membership = rule.mask(fitting["features"])
            contrast = membership | (~membership & np.isfinite(activation) & (activation <= cutoff))
            fit = fit_directions(fitting["embeddings"][contrast], membership[contrast],
                                 fitting["frame"]["patient_id"].to_numpy()[contrast], range(1001, 1051))
            key = ("all_negatives", concept["concept"])
            saved_direction = directions.get(key)
            if saved_direction is None:
                if np.isfinite(fit["directions"]).any() or direction_counts.get(key, (0, 0))[1] != 0:
                    raise ValueError(f"Missing saved direction for fittable {concept['concept']}")
            else:
                np.testing.assert_allclose(fit["directions"], saved_direction,
                                           **tolerances["float32"], equal_nan=True)
            checks.append({"reference": reference, "concept": concept["concept"], "check": "rule/direction",
                           "matching_fit_patients": fit["matching"], "nonmatching_fit_patients": fit["nonmatching"],
                           "measured_fits": sum(s == "measured" for s in fit["fit_status"]),
                           "max_abs_error": (float(np.nanmax(np.abs(fit["directions"] - saved_direction)))
                                             if saved_direction is not None and np.isfinite(saved_direction).any() else np.nan),
                           "rtol": tolerances["float32"]["rtol"], "atol": tolerances["float32"]["atol"],
                           "negative_pool": "minimum activation among reference fitting patients", "negative_cutoff": cutoff})
            concepts.append({"record": concept, "rule": rule, "arrays": arrays, "direction_fit": fit})
        reference_frame = evaluation["frame"].query("year == @reference")
        timing_ready = reference_frame["timing_observed"].all()
        cases.append({"reference": reference, "build": build, "threshold": threshold, "views": views, "concepts": concepts,
                      "context": {"rows": context, "features": population["X"][context],
                                  "labels": population["y"][context],
                                  "patients": population["patient_id"][context], "years": population["year"][context]},
                      "feature_names": saved["feature_names"], "timing_ready": bool(timing_ready),
                      "families": [{"family": i["family"],
                                    "members": sorted((int(s), int(k)) for s, k in i["family_record"]["members"])}
                                   for i in entries],
                      "source_identity": {"population": meta["population_identity"], "roles": roles["identity"],
                                          "context": meta["context_identity"],
                                          "sae": {str(s): dictionaries[s]["unit_identity"] for s in dictionaries}}})
    sources = {name: hashlib.sha256((paths.CODE_ROOT / name).read_bytes()).hexdigest()
               for name in ("cmr_common.py", "concept_subgroup_drift_pilot.py", "concept_study/loading.py",
                            "concept_study/error_analysis.py", "concept_measurement_reliability.ipynb")}
    return {"cases": cases, "parity": pd.DataFrame(checks), "timing_coverage": pd.DataFrame(coverage),
            "composition": pd.DataFrame(composition), "source_hashes": sources,
            "input_hashes": {**inputs["ctx"]["input_files"],
                             **{str(path): cmr_common.file_hash(path) for path in sorted(input_paths)}},
            "load_seconds": time.perf_counter() - started}


def prediction_groups(labels, probabilities, threshold):
    y, p = np.asarray(labels), np.asarray(probabilities)
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("Prediction groups require observed outcomes and finite scores")
    flagged = p >= threshold
    return np.select([(y == 1) & ~flagged, (y == 1) & flagged, (y == 0) & flagged],
                     ["missed_deaths", "caught_deaths", "false_alarms"], default="correct_survivors")


def patient_bootstrap(patient_sets, draws, seed):
    ids = np.unique(np.concatenate(patient_sets).astype(str))
    if len(ids) == 0:
        raise ValueError("Patient bootstrap needs patients")
    sampled = np.random.default_rng(seed).multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=draws)
    return ids, np.vstack([np.ones(len(ids), dtype=np.int64), sampled])


def row_weights(patients, bootstrap):
    return bootstrap[1][:, np.searchsorted(bootstrap[0], np.asarray(patients).astype(str))]


def mix_within(n0, errors0, n1, errors1):
    """Symmetric exact accounting; a one-ended group contributes entirely to mix."""
    n0, e0, n1, e1 = np.broadcast_arrays(n0, errors0, n1, errors1)
    divide = lambda a, b: np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b > 0)
    r0, r1 = divide(e0, n0), divide(e1, n1)
    r0, r1 = np.where(n0 > 0, r0, r1), np.where(n1 > 0, r1, r0)
    w0, w1 = divide(n0, n0.sum(axis=-1, keepdims=True)), divide(n1, n1.sum(axis=-1, keepdims=True))
    present = n0 + n1 > 0
    mix = np.sum(np.where(present, (w1 - w0) * (r0 + r1) / 2, 0.), axis=-1)
    within = np.sum(np.where(present, (w1 + w0) * (r1 - r0) / 2, 0.), axis=-1)
    change = divide(e1.sum(axis=-1), n1.sum(axis=-1)) - divide(e0.sum(axis=-1), n0.sum(axis=-1))
    return {"change": change, "mix": mix, "within": within,
            "empty_group_patients": np.sum(np.where((n0 == 0) | (n1 == 0), n0 + n1, 0), axis=-1)}


def record_length_groups(months, labels):
    months, labels = np.asarray(months, float), np.asarray(labels)
    return np.where(labels == 1, np.ceil(months / 3), months)


def shift_scores(labels, probabilities, years, reference, weights):
    """Set each annual mean-score minus mortality gap to its reference value."""
    y, p, years, w = map(np.asarray, (labels, probabilities, years, weights))
    z, eps = score_logits(p)
    result, diagnostics = np.full(len(p), np.nan), {}
    ref = years == reference
    gap = np.sum(w[ref] * (p[ref] - y[ref])) / w[ref].sum() if w[ref].sum() else np.nan
    for year in np.unique(years):
        at = years == year
        mass = w[at].sum()
        target = np.sum(w[at] * y[at]) / mass + gap if mass and np.isfinite(gap) else np.nan
        state, offset = "fitted", np.nan
        if not np.isfinite(target):
            state = "no reference or annual patients"
        elif target < 0 or target > 1:
            state = "impossible calibration target"
        elif target in (0, 1):
            offset = -np.inf if target == 0 else np.inf
            result[at] = target
        else:
            bound = 2 * np.log(np.finfo(float).max)
            offset = brentq(lambda delta: np.sum(w[at] * expit(z[at] + delta)) / mass - target,
                            -bound, bound)
            result[at] = expit(z[at] + offset)
        diagnostics[int(year)] = {"status": state, "offset": offset, "target_mean_score": target,
                                  "endpoint_count": int(np.count_nonzero((p[at] == 0) | (p[at] == 1))),
                                  "logit_clip_epsilon": eps, "weighted_rows": mass}
    return result, diagnostics


def score_logits(probabilities):
    """Numerical endpoint policy only; frozen scores and flags are never changed."""
    p = np.asarray(probabilities)
    dtype = p.dtype if np.issubdtype(p.dtype, np.floating) else np.dtype(float)
    eps = float(np.finfo(dtype).eps)
    return logit(np.clip(p.astype(float), eps, 1 - eps)), eps


def score_risk(labels, probabilities, weights=None):
    """Annual two-parameter score benchmark; never substitute a pooled fit."""
    y, p = np.asarray(labels), np.asarray(probabilities)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    z, eps = score_logits(p)
    result = {"risk": np.full(len(y), np.nan), "status": "fitted", "intercept": np.nan, "slope": np.nan,
              "endpoint_count": int(np.count_nonzero((p == 0) | (p == 1))), "logit_clip_epsilon": eps,
              "rows": len(y), "weighted_rows": float(w.sum()), "weighted_deaths": float(w @ y)}
    if len(np.unique(y[w > 0])) < 2:
        result["status"] = "single outcome class"
        return result
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(z[:, None], y, sample_weight=w)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            result["status"] = "score fit did not converge"
        elif not np.isfinite(np.r_[model.intercept_, model.coef_.ravel()]).all():
            result["status"] = "nonfinite score fit"
        else:
            result.update(risk=model.predict_proba(z[:, None])[:, 1], intercept=float(model.intercept_[0]),
                          slope=float(model.coef_[0, 0]))
    except (ValueError, FloatingPointError) as error:
        result["status"] = f"score fit failed: {type(error).__name__}"
    return result


def expected_errors(risk, flagged, weights):
    risk, flagged, w = np.asarray(risk), np.asarray(flagged), np.asarray(weights)
    deaths, survivors = np.sum(w * risk), np.sum(w * (1 - risk))
    missed, alarms = np.sum(w * risk * ~flagged), np.sum(w * (1 - risk) * flagged)
    return {"expected_deaths": deaths, "expected_survivors": survivors,
            "expected_missed_deaths": missed, "expected_false_alarms": alarms,
            "missed_death_rate": missed / deaths if deaths > 0 else np.nan,
            "false_alarm_rate": alarms / survivors if survivors > 0 else np.nan}


def summarize_draws(values):
    """Point estimate plus patient percentiles, retaining undefined and degenerate draws."""
    a = np.asarray(values, float)
    valid = np.isfinite(a[1:])
    finite = a[1:][valid]
    low, high = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
    return {"value": float(a[0]), "low": float(low), "high": float(high),
            "valid_draws": int(valid.sum()), "undefined_draws": int((~valid).sum()),
            "degenerate_interval": bool(len(finite) and low == high),
            "estimate_status": "estimated" if np.isfinite(a[0]) else "not estimable"}


def year_profile(frame, quantities, weights, threshold, reference):
    """One annual/four-group profile engine for every patient-level quantity."""
    names = list(quantities)
    values = np.column_stack([np.asarray(quantities[name], float) for name in names])
    finite = np.isfinite(values)
    groups = prediction_groups(frame["label"], frame["probability"], threshold)
    periods = {str(year): frame["year"].to_numpy() == year for year in sorted(frame["year"].unique())}
    periods["later"] = frame["year"].to_numpy() > reference
    rows, draws = [], {}
    for period, period_mask in periods.items():
        for group in ("all", "missed_deaths", "caught_deaths", "false_alarms", "correct_survivors"):
            mask = period_mask & ((groups == group) if group != "all" else True)
            w = weights[:, mask]
            selected_finite = finite[mask]
            defined_rows = selected_finite.sum(axis=0)
            counts = {"rows": int(mask.sum()), "patients": frame.loc[mask, "patient_id"].nunique(),
                      "deaths": int(frame.loc[mask, "label"].sum()),
                      "survivors": int(mask.sum() - frame.loc[mask, "label"].sum())}
            # Zero is used only for the numerator of a missing value; its weight
            # is also removed from the denominator, so no value is imputed.
            numerator = w @ np.where(selected_finite, values[mask], 0.)
            denominator = w @ selected_finite.astype(float)
            estimates = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
            for j, quantity in enumerate(names):
                draws[(period, group, quantity)] = estimates[:, j]
                rows.append({"reference": reference, "period": period, "group": group, "quantity": quantity,
                             **counts, "defined_rows": int(defined_rows[j]), "numerator": numerator[0, j],
                             "denominator": denominator[0, j], **summarize_draws(estimates[:, j])})
    return pd.DataFrame(rows), draws


def profile_contrasts(profile, draws, reference):
    """Both feature views, annual differences and every consecutive/reference change."""
    views = {"same_outcome_deaths": ("missed_deaths", "caught_deaths"),
             "same_outcome_survivors": ("false_alarms", "correct_survivors"),
             "same_prediction_flagged": ("false_alarms", "caught_deaths"),
             "same_prediction_unflagged": ("missed_deaths", "correct_survivors"), "all": ("all", None)}
    periods = list(dict.fromkeys(profile["period"]))
    years = sorted(p for p in periods if p != "later")
    pairs = (list(dict.fromkeys([(str(reference), p) for p in periods if p != str(reference)] +
                                list(zip(years[:-1], years[1:])))) if str(reference) in periods else [])
    lookup = {(row["period"], row["group"], row["quantity"]): row for row in profile.to_dict("records")}
    rows = []
    for quantity in profile["quantity"].unique():
        for view, (error, correct) in views.items():
            differences = {p: draws[(p, error, quantity)] - (draws[(p, correct, quantity)] if correct else 0.)
                           for p in periods}
            for period in periods:
                counts = {f"{label}_{field}": lookup[(period, group, quantity)][field]
                          for label, group in (("error", error), ("correct", correct)) if group
                          for field in ("rows", "patients", "deaths", "survivors", "defined_rows")}
                rows.append({"reference": reference, "quantity": quantity, "view": view, "kind": "difference",
                             "start": period, "end": period, **counts, **summarize_draws(differences[period])})
            for start, end in pairs:
                counts = {f"{endpoint}_{label}_{field}": lookup[(period, group, quantity)][field]
                          for endpoint, period in (("start", start), ("end", end))
                          for label, group in (("error", error), ("correct", correct)) if group
                          for field in ("rows", "patients", "deaths", "survivors", "defined_rows")}
                rows.append({"reference": reference, "quantity": quantity, "view": view, "kind": "change",
                             "start": start, "end": end, **counts,
                             **summarize_draws(differences[end] - differences[start])})
    return pd.DataFrame(rows)


def crossfit_ridge(features, activations, patients, years, reference, seed, folds, alphas):
    """Each patient's annual predictions use the reference model that excluded them."""
    X, y, patients, years = map(np.asarray, (features, activations, patients, years))
    unique, inverse = np.unique(patients.astype(str), return_inverse=True)
    folds = min(folds, len(np.unique(patients[years == reference])))
    if folds < 3:
        raise ValueError("Nested patient cross-fitting needs at least three reference patients")
    assignment = np.empty(len(unique), int)
    assignment[np.random.default_rng(seed).permutation(len(unique))] = np.arange(len(unique)) % folds
    row_fold = assignment[inverse]
    prediction, audit = np.full(y.shape, np.nan, dtype=float), []
    for fold in range(folds):
        train = (years == reference) & (row_fold != fold)
        test = row_fold == fold
        inner_folds = min(folds, len(np.unique(patients[train])))
        inner = list(GroupKFold(inner_folds).split(X[train], y[train], groups=patients[train]))
        pipeline = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
        model = GridSearchCV(pipeline, {"ridge__alpha": list(alphas)}, cv=inner,
                             scoring="neg_mean_squared_error", n_jobs=1)
        model.fit(X[train], y[train])
        prediction[test] = model.predict(X[test])
        audit.append({"fold": fold, "training_rows": int(train.sum()), "predicted_rows": int(test.sum()),
                      "training_patients": len(np.unique(patients[train])),
                      "patient_overlap": len(np.intersect1d(patients[train], patients[test])),
                      "alpha": model.best_params_["ridge__alpha"], "inner_score": model.best_score_})
    at_reference = years == reference
    heldout_error = np.sqrt(np.mean((prediction[at_reference] - y[at_reference]) ** 2, axis=0))
    return {"prediction": prediction, "folds": row_fold, "audit": pd.DataFrame(audit),
            "reference_rmse": heldout_error}


def annual_benchmarks(frame, weights, reference):
    """Refit annual score risks and calibration shifts in every patient draw."""
    y, p, years = (frame[name].to_numpy() for name in ("label", "probability", "year"))
    risks, shifted = np.full(weights.shape, np.nan), np.full(weights.shape, np.nan)
    diagnostics = []
    for draw, w in enumerate(weights):
        shifted[draw], offsets = shift_scores(y, p, years, reference, w)
        for year in sorted(np.unique(years)):
            at = years == year
            fit = score_risk(y[at], p[at], w[at])
            risks[draw, at] = fit.pop("risk")
            diagnostics.append({"reference": reference, "draw": draw, "year": year,
                                **fit, **{f"shift_{k}": v for k, v in offsets[int(year)].items()}})
    return risks, shifted, pd.DataFrame(diagnostics)


def performance_table(frame, weights, threshold, reference, shifted):
    """Annual and reference/later pooled performance with joint patient intervals."""
    years, y, p = (frame[name].to_numpy() for name in ("year", "label", "probability"))
    periods = {str(year): years == year for year in sorted(np.unique(years))}
    periods["later"] = years > reference
    rows, draws = [], {}
    for period, at in periods.items():
        for version in ("frozen", "shifted"):
            estimates = []
            for draw, w in enumerate(weights[:, at]):
                score = p[at] if version == "frozen" else shifted[draw, at]
                if np.isfinite(score).all():
                    estimates.append(baseline_values(y[at], score, threshold, w))
                else:
                    estimates.append({key: np.nan for key in baseline_values(y[at], p[at], threshold, w)})
            for metric in estimates[0]:
                values = np.array([estimate[metric] for estimate in estimates])
                point = estimates[0]
                n, d, s = point["n"], point["deaths"], point["survivors"]
                tp, fp, fn = point["true_positives"], point["false_positives"], point["false_negatives"]
                counts = {"recall": (tp, d), "false_alarm_rate": (fp, s), "share_flagged": (tp + fp, n),
                          "mortality": (d, n), "precision": (tp, tp + fp), "death_f1": (2 * tp, 2 * tp + fp + fn),
                          "mean_score": (point["mean_score"] * n, n), "brier": (point["brier"] * n, n),
                          "one_death": (1., d), "one_survivor": (1., s), "one_patient": (1., n)}
                numerator, denominator = counts.get(metric, (values[0], 1.) if metric in
                                                     ("n", "deaths", "survivors", "true_positives", "false_positives", "false_negatives")
                                                     else (np.nan, np.nan))
                draws[(period, version, metric)] = values
                rows.append({"reference": reference, "period": period, "version": version, "metric": metric,
                             "rows": int(at.sum()), "patients": frame.loc[at, "patient_id"].nunique(),
                             "deaths": int(y[at].sum()), "survivors": int(at.sum() - y[at].sum()),
                             "numerator": numerator, "denominator": denominator,
                             "zero_observed_events": bool(np.isfinite(values[0]) and values[0] == 0),
                             **summarize_draws(values)})
    for version in ("frozen", "shifted"):
        for metric in estimates[0]:
            for end in periods:
                if end == str(reference):
                    continue
                change = draws[(end, version, metric)] - draws[(str(reference), version, metric)]
                end_mask = periods[end]
                denominator = int(y[end_mask].sum()) if metric == "recall" else int((1 - y[end_mask]).sum())
                rows.append({"reference": reference, "period": f"{reference}→{end}", "version": version,
                             "metric": metric, "rows": int(end_mask.sum()),
                             "patients": frame.loc[end_mask, "patient_id"].nunique(),
                             "deaths": int(y[end_mask].sum()), "survivors": int((1 - y[end_mask]).sum()),
                             "patient_equivalent": float(change[0] * denominator) if metric in ("recall", "false_alarm_rate") else np.nan,
                             **summarize_draws(change)})
    return pd.DataFrame(rows)


def rate_splits(frame, strata, weights, threshold, reference, score_risks=None):
    """Exact observed and score-alone rate splits, with each missing draw retained."""
    years, y, p = (frame[name].to_numpy() for name in ("year", "label", "probability"))
    strata = np.asarray(strata)
    groups = np.unique(strata[np.isfinite(strata)])
    periods = {str(year): years == year for year in sorted(np.unique(years))}
    periods["later"] = years > reference
    year_names = [p for p in periods if p != "later"]
    pairs = list(dict.fromkeys(list(zip(year_names[:-1], year_names[1:])) +
                               [(str(reference), end) for end in periods if end != str(reference)]))
    rows = []
    for side, label, error in (("deaths", 1, p < threshold), ("survivors", 0, p >= threshold)):
        totals, unstratified = {}, {}
        for period, at in periods.items():
            unstratified[period] = int(np.sum(at & (y == label) & ~np.isfinite(strata.astype(float))))
            denominator = np.stack([weights[:, at & (y == label) & (strata == group)].sum(axis=1) for group in groups], axis=1)
            numerator = np.stack([weights[:, at & (y == label) & error & (strata == group)].sum(axis=1) for group in groups], axis=1)
            totals[(period, "observed")] = denominator, numerator
            if score_risks is not None:
                risk = score_risks if label == 1 else 1 - score_risks
                expected_n = np.stack([(weights[:, at & (strata == group)] * risk[:, at & (strata == group)]).sum(axis=1)
                                       for group in groups], axis=1)
                expected_e = np.stack([(weights[:, at & error & (strata == group)] * risk[:, at & error & (strata == group)]).sum(axis=1)
                                       for group in groups], axis=1)
                totals[(period, "score_alone")] = expected_n, expected_e
        for start, end in pairs:
            versions = ("observed", "score_alone") if score_risks is not None else ("observed",)
            for version in versions:
                n0, e0 = totals[(start, version)]; n1, e1 = totals[(end, version)]
                result = mix_within(n0, e0, n1, e1)
                for component in ("change", "mix", "within"):
                    rows.append({"reference": reference, "side": side, "start": start, "end": end,
                                 "version": version, "component": component,
                                 "start_denominator": n0[0].sum(), "end_denominator": n1[0].sum(),
                                 "start_errors": e0[0].sum(), "end_errors": e1[0].sum(),
                                 "empty_group_patients": result["empty_group_patients"][0],
                                 "start_rows_without_stratum": unstratified[start], "end_rows_without_stratum": unstratified[end],
                                 "patient_equivalent": result[component][0] * n1[0].sum(),
                                 "zero_observed_events": bool(e0[0].sum() + e1[0].sum() == 0),
                                 **summarize_draws(result[component])})
    return pd.DataFrame(rows)


def log_odds_sensitivity(sensitivity, probabilities):
    """Saved probability gradients become undefined, never zero, at score endpoints."""
    p = np.asarray(probabilities, float)
    denominator = p * (1 - p)
    values = np.divide(sensitivity, denominator[:, None], out=np.full_like(sensitivity, np.nan, dtype=float),
                       where=denominator[:, None] > 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values, axis=1)


def rule_error_rates(frame, membership, weights, threshold, reference, score_risks):
    """Inside/outside rates use actual or expected outcome denominators explicitly."""
    years, y, p = (frame[name].to_numpy() for name in ("year", "label", "probability"))
    periods = {str(year): years == year for year in sorted(np.unique(years))}
    periods["later"] = years > reference
    membership, rows = np.asarray(membership, bool), []
    for period, at in periods.items():
        for group, mask in (("inside", membership), ("outside", ~membership)):
            selected = at & mask
            for side, label, error in (("deaths", 1, p < threshold), ("survivors", 0, p >= threshold)):
                observed_n = weights[:, selected & (y == label)].sum(axis=1)
                observed_e = weights[:, selected & (y == label) & error].sum(axis=1)
                risk = score_risks if label == 1 else 1 - score_risks
                expected_n = np.sum(weights[:, selected] * risk[:, selected], axis=1)
                expected_e = np.sum(weights[:, selected & error] * risk[:, selected & error], axis=1)
                for version, numerator, denominator in (("observed", observed_e, observed_n),
                                                         ("score_alone", expected_e, expected_n)):
                    value = np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)
                    rows.append({"reference": reference, "period": period, "group": group, "side": side,
                                 "version": version, "rows": int(selected.sum()),
                                 "patients": frame.loc[selected, "patient_id"].nunique(),
                                 "deaths": int(y[selected].sum()), "survivors": int((1 - y[selected]).sum()),
                                 "numerator": numerator[0], "denominator": denominator[0],
                                 "zero_observed_events": bool(observed_e[0] == 0), **summarize_draws(value)})
    return pd.DataFrame(rows)


def weighted_spearman(values, targets, weights, layouts=None):
    """Exact multiplicity-weighted midranks, equivalent to expanding each draw."""
    X, Y, w = np.asarray(values, float), np.asarray(targets, float), np.asarray(weights, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if layouts is None:
        layouts = rank_layouts(X)
    if w.sum() == 0:
        return np.full((X.shape[1], Y.shape[1]), np.nan)
    xr = np.column_stack([weighted_ranks(X[:, j], w, layout) for j, layout in enumerate(layouts)])
    yr = np.column_stack([weighted_ranks(Y[:, j], w) for j in range(Y.shape[1])])
    xr -= np.sum(w[:, None] * xr, axis=0) / w.sum()
    yr -= np.sum(w[:, None] * yr, axis=0) / w.sum()
    numerator = xr.T @ (w[:, None] * yr)
    denominator = np.sqrt(np.sum(w[:, None] * xr ** 2, axis=0)[:, None] *
                          np.sum(w[:, None] * yr ** 2, axis=0)[None, :])
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def rank_layouts(values):
    result = []
    for column in np.asarray(values).T:
        order = np.argsort(column, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(column[order]) != 0) + 1]
        result.append((order, starts))
    return result


def weighted_ranks(values, weights, layout=None):
    if np.any(~np.isfinite(values) & (weights > 0)):
        return np.full(len(values), np.nan)
    if layout is None:
        layout = rank_layouts(np.where(np.isfinite(values), values, 0.)[:, None])[0]
    order, starts = layout
    masses = np.add.reduceat(weights[order], starts)
    midranks = np.cumsum(masses) - masses / 2
    ranks = np.empty(len(values), float)
    ranks[order] = np.repeat(midranks, np.diff(np.r_[starts, len(values)]))
    return ranks


def training_target(training, other):
    """Score risk and exact-month means fitted only on training patients."""
    train_target, other_target, diagnostics = np.full(len(training), np.nan), np.full(len(other), np.nan), []
    for year in sorted(set(training["year"]) | set(other["year"])):
        train_at, other_at = training["year"].to_numpy() == year, other["year"].to_numpy() == year
        train, held = training.loc[train_at], other.loc[other_at]
        fit = score_risk(train["label"].to_numpy(), train["probability"].to_numpy())
        train_residual = train["label"].to_numpy() - fit.pop("risk")
        held_logit, _ = score_logits(held["probability"].to_numpy())
        held_residual = held["label"].to_numpy() - expit(fit["intercept"] + fit["slope"] * held_logit)
        means = pd.DataFrame({"month": train["active_months"].to_numpy(), "residual": train_residual}).groupby("month")["residual"].mean()
        train_values = train_residual - means.reindex(train["active_months"]).to_numpy()
        held_values = held_residual - means.reindex(held["active_months"]).to_numpy()
        train_target[train_at], other_target[other_at] = train_values, held_values
        diagnostics.append({"year": year, **fit, "training_rows": len(train), "heldout_rows": len(held),
                            "undefined_training_rows": int((~np.isfinite(train_values)).sum()),
                            "undefined_heldout_rows": int((~np.isfinite(held_values)).sum()),
                            "unseen_month_rows": int((~held["active_months"].isin(means.index)).sum())})
    return train_target, other_target, pd.DataFrame(diagnostics)


def fit_tree(features, target, patients, names, folds, seed, frame=None):
    """One grouped-CV pruning function for factor descriptions and residual slices.

    Candidate leaf budgets 1..8 exhaust a depth-three tree. Each fitting fold
    derives its own pruning path; no held-out target determines that path.
    The fullest training-only pruned tree within a budget predicts its fold.
    """
    X, patients = np.asarray(features), np.asarray(patients)
    folds = min(folds, len(np.unique(patients)))
    diagnostics, losses, fits, nuisance_fits = [], {k: [] for k in range(1, 9)}, 0, 0
    result = {"status": "not estimable: fewer than two patient folds", "rules": [], "fit_count": 0,
              "nuisance_fit_count": 0, "cv_diagnostics": pd.DataFrame(), "tree": None}
    if folds < 2:
        return result
    for fold, (train, held) in enumerate(GroupKFold(folds).split(X, groups=patients)):
        if frame is None:
            yt, yv = np.asarray(target)[train], np.asarray(target)[held]
        else:
            yt, yv, nuisance = training_target(frame.iloc[train], frame.iloc[held])
            nuisance_fits += len(nuisance)
            diagnostics.extend(nuisance.assign(fold=fold).to_dict("records"))
        # A row whose (year, month) cell is absent from training has no target:
        # it is left out of this fold's fit or loss and counted, never filled.
        train_ok, held_ok = np.isfinite(yt), np.isfinite(yv)
        diagnostics.append({"fold": fold, "status": "fold rows", "training_rows": len(train), "heldout_rows": len(held),
                            "undefined_training_rows": int((~train_ok).sum()), "undefined_heldout_rows": int((~held_ok).sum())})
        if not train_ok.any() or not held_ok.any():
            continue
        models, count = pruning_models(X[train][train_ok], yt[train_ok], seed)
        fits += count
        for budget in losses:
            model = max((m for m in models if m.get_n_leaves() <= budget), key=lambda m: m.get_n_leaves())
            losses[budget].append(float(np.mean((model.predict(X[held][held_ok]) - yv[held_ok]) ** 2)))
    candidates = [(np.mean(losses[k]), k) for k in losses if len(losses[k]) == folds]
    if not candidates:
        result.update(status="not estimable: CV choosing targets undefined", fit_count=fits,
                      nuisance_fit_count=nuisance_fits, cv_diagnostics=pd.DataFrame(diagnostics))
        return result
    loss, budget = min(candidates)
    if frame is not None:
        target, _, nuisance = training_target(frame, frame)
        nuisance_fits += len(nuisance)
        diagnostics.extend(nuisance.assign(fold="full").to_dict("records"))
    target = np.asarray(target, float)
    ok = np.isfinite(target)
    result["undefined_target_rows"] = int((~ok).sum())
    if not ok.any():
        result.update(status="not estimable: full choosing target undefined", fit_count=fits,
                      nuisance_fit_count=nuisance_fits, cv_diagnostics=pd.DataFrame(diagnostics))
        return result
    models, count = pruning_models(X[ok], target[ok], seed)
    fits += count
    model = max((m for m in models if m.get_n_leaves() <= budget), key=lambda m: m.get_n_leaves())
    leaves = {}
    def visit(node, conditions):
        feature = model.tree_.feature[node]
        if feature < 0:
            leaves[node] = tuple(conditions)
        else:
            threshold = float(model.tree_.threshold[node])
            visit(model.tree_.children_left[node], conditions + [(int(feature), "<=", threshold)])
            visit(model.tree_.children_right[node], conditions + [(int(feature), ">", threshold)])
    visit(0, [])
    highest = max(leaves, key=lambda node: model.tree_.value[node].ravel()[0])
    rules = list(leaves.values()) if model.get_n_leaves() > 1 else []
    result.update(status="described" if rules else "no feature description", rules=rules,
                  factor_rule=leaves[highest] if rules else None, tree=model, fit_count=fits,
                  nuisance_fit_count=nuisance_fits, cv_diagnostics=pd.DataFrame(diagnostics),
                  selected_leaf_budget=budget, leaves=model.get_n_leaves(), cv_loss=loss,
                  feature_names=list(names), pruning_candidates=[k for _, k in candidates])
    return result


def pruning_models(features, target, seed):
    base = DecisionTreeRegressor(max_depth=3, random_state=seed)
    alphas = np.unique(base.cost_complexity_pruning_path(features, target).ccp_alphas)
    models = [DecisionTreeRegressor(max_depth=3, random_state=seed, ccp_alpha=float(alpha)).fit(features, target)
              for alpha in alphas]
    return models, 1 + len(models)


def rule_key(rule):
    return json.dumps(sorted((int(j), str(op), float(value).hex()) for j, op, value in rule), separators=(",", ":"))


def rule_mask(rule, features):
    if hasattr(rule, "mask"):
        return np.asarray(rule.mask(features), bool)
    mask = np.ones(len(features), bool)
    for j, op, value in rule:
        mask &= features[:, j] <= value if op == "<=" else features[:, j] > value
    return mask


def context_presence(membership, labels):
    membership, labels = np.asarray(membership, bool), np.asarray(labels)
    n, deaths = int(membership.sum()), int(np.sum(membership & (labels == 1)))
    return {"absence": "no matching context rows" if n == 0 else
            ("matching context rows but no deaths" if deaths == 0 else "context deaths present"),
            "matching_rows": n, "matching_deaths": deaths, "matching_survivors": n - deaths,
            "context_rows": len(labels), "context_deaths": int(np.sum(labels == 1)),
            "context_survivors": int(np.sum(labels == 0))}


def effective_dimension(codes):
    centered = np.asarray(codes, float) - np.mean(codes, axis=0)
    covariance = centered.T @ centered
    denominator = np.sum(covariance ** 2)
    return float(np.trace(covariance) ** 2 / denominator) if denominator > 0 else np.nan


def summarize_matrix(draws):
    """Vectorized summarize_draws over columns: row 0 is the point estimate."""
    a = np.asarray(draws, float)
    valid = np.isfinite(a[1:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanquantile(a[1:], [.025, .975], axis=0) if len(a) > 1 else (np.full(a.shape[1], np.nan),) * 2
    return pd.DataFrame({"value": a[0], "low": low, "high": high, "valid_draws": valid.sum(axis=0),
                         "undefined_draws": (~valid).sum(axis=0),
                         "degenerate_interval": valid.any(axis=0) & (low == high),
                         "estimate_status": np.where(np.isfinite(a[0]), "estimated", "not estimable")})


def factor_columns(codes):
    """All frozen factors in fixed seed/factor order."""
    keys = [(int(seed), k) for seed in sorted(codes) for k in range(codes[seed].shape[1])]
    return keys, np.column_stack([np.asarray(codes[seed], float) for seed in sorted(codes)])


def tie_layout(values):
    """Sort order and tie-block bounds of every column, reused across draws."""
    X = np.asarray(values, float)
    order = np.argsort(X, axis=0, kind="stable")
    s, n, index = np.take_along_axis(X, order, axis=0), len(X), np.arange(len(X))[:, None]
    differs = s[1:] != s[:-1]
    first = np.vstack([np.ones((1, X.shape[1]), bool), differs])
    last = np.vstack([differs, np.ones((1, X.shape[1]), bool)])
    start = np.maximum.accumulate(np.where(first, index, 0), axis=0)
    end = np.minimum.accumulate(np.where(last, index, n - 1)[::-1], axis=0)[::-1]
    return order, start, end


def weighted_rank_matrix(layout, weights):
    """Exact multiplicity-weighted midranks of every column (same as weighted_ranks)."""
    order, start, end = layout
    w = np.asarray(weights, float)[order]
    total = np.cumsum(w, axis=0)
    midrank = (np.take_along_axis(total - w, start, axis=0) + np.take_along_axis(total, end, axis=0)) / 2
    ranks = np.empty_like(total)
    np.put_along_axis(ranks, order, midrank, axis=0)
    return ranks


def weighted_pearson(x, y, w):
    mass = w.sum()
    xc, yc = x - (w @ x) / mass, y - (w @ y) / mass
    denominator = np.sqrt((w @ xc ** 2) * (w @ yc ** 2))
    return np.divide((w * yc) @ xc, denominator, out=np.full(x.shape[1], np.nan), where=denominator > 0)


def choosing_targets(frame, risks, weights):
    """Plain, score-alone and record-length-adjusted residuals, refitted in every draw.

    Adjusted = outcome minus score-alone risk, minus its mean within the same year
    and exact months with care (one grouping for deaths and survivors).
    """
    y, p = frame["label"].to_numpy(float), frame["probability"].to_numpy(float)
    score = y[None, :] - risks
    months = frame["active_months"].to_numpy(float)
    known = np.isfinite(months)
    cells = pd.factorize(pd.Series(list(zip(frame["year"], months))).where(known))[0]
    onehot = np.zeros((len(y), max(cells.max() + 1, 1)))
    onehot[known, cells[known]] = 1.
    w = np.where(np.isfinite(score), weights, 0.)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = ((w * np.nan_to_num(score)) @ onehot) / (w @ onehot)
    adjusted = np.full_like(score, np.nan)
    adjusted[:, known] = score[:, known] - means[:, cells[known]]
    return {"plain": y - p, "score_alone": score, "adjusted": adjusted}


def factor_links(values, names, targets, weights, frame, reference):
    """Rank correlation of every factor with every target in each year, later years
    pooled and (if the reference year exists) the reference-to-later change.

    A row with an undefined target gets no weight for that target in that draw;
    undefined rows are counted (point draw), never imputed.
    """
    years = frame["year"].to_numpy()
    periods = {str(y): years == y for y in sorted(np.unique(years))}
    periods["later"] = years > reference
    draws, rows = {}, []
    for period, at in periods.items():
        layout = tie_layout(values[at])
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
                    cached[key] = weighted_rank_matrix(layout, w)
                ranks = weighted_ranks(np.where(ok, t[d], 0.), w)
                draws[(period, name)][d] = weighted_pearson(cached[key], ranks, w)
        for name, t in local.items():
            rows.append(summarize_matrix(draws[(period, name)]).assign(
                factor=names, period=period, kind="link", target=name, rows=int(at.sum()),
                undefined_target_rows=int((~np.isfinite(t[0])).sum())))
    if str(reference) in periods:
        for target in targets:
            change = draws[("later", target)] - draws[(str(reference), target)]
            rows.append(summarize_matrix(change).assign(factor=names, period=f"{reference}→later",
                                                        kind="change", target=target))
    return pd.DataFrame(pd.concat(rows, ignore_index=True))


def _describe(features, target, patients, folds, seed):
    fit = fit_tree(features, target, patients, [], folds, seed)
    fit.pop("cv_diagnostics")
    return fit


def describe_factors(features, codes, patients, folds, seed, workers):
    """One grouped-CV tree per factor (activation value as target) on later entrants."""
    from joblib import Parallel, delayed
    return Parallel(n_jobs=workers)(delayed(_describe)(features, codes[:, j], patients, folds, seed)
                                    for j in range(codes.shape[1]))


def heldout_r2(prediction, observed):
    """Share of activation variance explained; undefined for a constant activation."""
    total = np.sum((observed - observed.mean()) ** 2)
    return float(1 - np.sum((prediction - observed) ** 2) / total) if total > 0 else np.nan


def rule_directions(embeddings, features, patients, rules, seeds, workers):
    """Copied direction recipe: each new rule against every other fitting patient."""
    from joblib import Parallel, delayed
    return Parallel(n_jobs=workers)(delayed(fit_directions)(embeddings, rule_mask(rule, features), patients, seeds)
                                    for rule in rules)


def projected_sensitivity(gradients, probabilities, directions):
    """Per-fit log-odds sensitivity (rows x fits); endpoints and failed fits undefined."""
    p = np.asarray(probabilities, float)
    denominator = (p * (1 - p))[:, None]
    raw = np.asarray(gradients, float) @ np.asarray(directions, float).T
    return np.divide(raw, denominator, out=np.full(raw.shape, np.nan), where=denominator > 0)


def fit_spread(per_fit, member, years, reference):
    """Spread over direction fits of the matching patients' mean sensitivity.
    A distribution over fits, not a patient interval."""
    out = {}
    for period, at in ((str(reference), years == reference), ("later", years > reference)):
        rows = per_fit[member & at]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(rows, axis=0) if len(rows) else np.full(per_fit.shape[1], np.nan)
        finite = means[np.isfinite(means)]
        out.update({f"fit_spread_p05_{period}": np.quantile(finite, .05) if len(finite) else np.nan,
                    f"fit_spread_p95_{period}": np.quantile(finite, .95) if len(finite) else np.nan,
                    f"fit_spread_fits_{period}": len(finite)})
    return out


def context_shares(context_member, context_labels, member, frame, weights, threshold, reference):
    """Context shares (no interval) beside later evaluation deaths, missed deaths,
    survivors and false alarms (patient intervals when weights carry draws)."""
    row = context_presence(context_member, context_labels)
    ratio = lambda a, b: a / b if b else np.nan
    row.update(context_death_share=ratio(row["matching_deaths"], row["context_deaths"]),
               context_survivor_share=ratio(row["matching_survivors"], row["context_survivors"]))
    y, p, later = frame["label"].to_numpy(), frame["probability"].to_numpy(), frame["year"].to_numpy() > reference
    for name, mask in (("later_deaths", later & (y == 1)), ("later_missed_deaths", later & (y == 1) & (p < threshold)),
                       ("later_survivors", later & (y == 0)), ("later_false_alarms", later & (y == 0) & (p >= threshold))):
        total, inside = weights[:, mask].sum(axis=1), weights[:, mask & member].sum(axis=1)
        share = np.divide(inside, total, out=np.full(len(weights), np.nan), where=total > 0)
        summary = summarize_draws(share)
        row.update({f"{name}_share": summary.pop("value"), **{f"{name}_{k}": v for k, v in summary.items()},
                    f"{name}_matching": int(np.sum(mask & member)), f"{name}_rows": int(mask.sum())})
    return row


def slice_report(frame, member, weights, risks, threshold, reference):
    """Per year (reference included) and later pooled, for all, newcomers and returning
    patients: share, observed minus predicted and minus score-alone mortality, counts."""
    y, p, years = (frame[c].to_numpy() for c in ("label", "probability", "year"))
    first, months = frame["first_year"].to_numpy(), frame["active_months"].to_numpy(float)
    periods = {str(year): years == year for year in sorted(np.unique(years))}
    periods["later"] = years > reference
    cohorts = {"all": np.ones(len(y), bool), "newcomers": first == years, "returning": first < years}
    ratio = lambda a, b: np.divide(a, b, out=np.full(len(b), np.nan), where=b > 0)
    rows = []
    for period, at in periods.items():
        for cohort, keep in cohorts.items():
            base, inside = at & keep, at & keep & member
            mass = weights[:, inside].sum(axis=1)
            values = {"share": ratio(mass, weights[:, base].sum(axis=1)),
                      "observed_minus_predicted": ratio(weights[:, inside] @ (y - p)[inside], mass),
                      "observed_minus_score_alone": ratio(np.sum(weights[:, inside] * (y[inside] - risks[:, inside]), axis=1), mass)}
            counts = {"reference": reference, "period": period, "cohort": cohort, "rows": int(inside.sum()),
                      "base_rows": int(base.sum()), "patients": frame.loc[inside, "patient_id"].nunique(),
                      "deaths": int(y[inside].sum()), "missed_deaths": int(np.sum(inside & (y == 1) & (p < threshold))),
                      "false_alarms": int(np.sum(inside & (y == 0) & (p >= threshold))),
                      "mean_months_with_care": float(np.nanmean(months[inside])) if np.isfinite(months[inside]).any() else np.nan}
            for quantity, draws in values.items():
                rows.append({**counts, "quantity": quantity, **summarize_draws(draws)})
    return pd.DataFrame(rows)


def rule_text(rule, names):
    return " and ".join(f"{names[j]} {op} {value:.4g}" for j, op, value in rule) if rule else "everyone"
