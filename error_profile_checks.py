"""Planted-change checks of the error profile (error_profiles.py).

A check is a before/after pair of evaluation rows. "Before" is year 0 and "after" is year 1
of a two-year frame whose reference year is 0, so every error_profiles function runs
unchanged on the pair. The profile functions that read scores only through the frozen flag
(rate_splits, rule_error_rates, year_profile) receive the flag itself as the score with
threshold 0.5: for one predictor this gives identical results, and it lets predictors with
different frozen thresholds be stacked into one pooled frame.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

import error_profiles as ep

SEED, DRAWS = 20261106, 1000
SHIFTS = ("hd_stop_half", "hd_stop_all", "n180_stop_all", "placebo_stop")
STOPPED = {"hd_stop_half": "EVENT_C1DIALISE_HD", "hd_stop_all": "EVENT_C1DIALISE_HD", "n180_stop_all": "DIAGN_N180"}
METRICS = ("recall", "false_alarm_rate", "share_flagged", "death_f1", "auroc", "brier", "mean_score")
COUNTS = ("n", "deaths", "survivors", "true_positives", "false_positives", "false_negatives")
KEY = ["level", "predictor", "measure", "item"]
# Rule membership is the profile's own concept quantity; "factor active" (factor code above the
# concept's saved cutoff) is an added quantity the original profile does not compute.
SOURCE = {"rule": "profile quantity: rule membership",
          "factor active": "new quantity: factor active (not computed by the original profile)"}


def aligned_rows(view, table, positions, year, tolerance):
    """Evaluation-view row of every planted position; ValueError on any disagreement."""
    frame = view["frame"]
    at = np.flatnonzero(frame["year"].to_numpy() == year)
    ids = frame["patient_id"].to_numpy()[at].astype(str)
    planted = np.asarray(table["patient_id"])[positions].astype(str)
    if (np.any(np.asarray(table["year"])[positions] != year) or len(set(ids)) != len(ids)
            or sorted(ids) != sorted(planted)):
        raise ValueError("Planted rows and evaluation rows hold different patients")
    rows = at[pd.Index(ids).get_indexer(planted)]
    if not np.array_equal(frame["label"].to_numpy()[rows], np.asarray(table["label"])[positions]):
        raise ValueError("Planted rows and evaluation rows have different outcomes")
    if not np.allclose(frame["probability"].to_numpy()[rows], np.asarray(table["probability"])[positions], **tolerance):
        raise ValueError("Planted rows and evaluation rows have different scores")
    if not np.array_equal(view["features"][rows], np.asarray(table["features"])[positions]):
        raise ValueError("Planted rows and evaluation rows have different features")
    return rows


def make_side(frame, features, codes, concepts, weight=None):
    """One set of rows with every concept's membership recomputed on these rows: the rule on
    these features, and factor active = these factor codes above the concept's saved cutoff."""
    members = {}
    for concept in concepts:
        record = concept["record"]
        members[(record["concept"], "rule")] = np.asarray(concept["rule"].mask(features), bool)
        activation = np.asarray(codes[record["seed"]][:, record["factor"]], np.float32)
        members[(record["concept"], "factor active")] = activation > record["cutoff"]
    frame = frame[["patient_id", "label", "probability", "active_months"]].reset_index(drop=True)
    return {"frame": frame, "features": np.asarray(features), "members": members,
            "weight": np.ones(len(frame)) if weight is None else np.asarray(weight, float)}


def take(side, rows):
    return {"frame": side["frame"].iloc[rows].reset_index(drop=True), "features": side["features"][rows],
            "members": {k: v[rows] for k, v in side["members"].items()}, "weight": side["weight"][rows]}


def shifted_side(side, offset):
    """P1: the same rows with `offset` added to the log-odds of every score."""
    z, _ = ep.score_logits(side["frame"]["probability"].to_numpy())
    return {**side, "frame": side["frame"].assign(probability=expit(z + offset))}


def tilt_weights(months, labels, drop):
    """P2: survivor weights exp(-lambda * months), lambda chosen so that the survivors' mean
    months with care falls by `drop`; survivors' total weight and every death (weight 1) are
    unchanged. Rows without timing keep weight 1 and are never given a month value."""
    m, survivor = np.asarray(months, float), np.asarray(labels) == 0
    s = survivor & np.isfinite(m)
    target = m[s].mean() - drop
    mean_at = lambda lam: np.average(m[s], weights=np.exp(-lam * (m[s] - m[s].min())))
    lam = brentq(lambda value: mean_at(value) - target, 0., 50.)
    w = np.ones(len(m))
    w[s] = np.exp(-lam * (m[s] - m[s].min()))
    w[s] *= s.sum() / w[s].sum()
    return w, lam


def split_halves(side, seed):
    """P5: the rows split at random into two halves (reproducible by seed)."""
    order = np.random.default_rng(seed).permutation(len(side["frame"]))
    half = len(order) // 2
    return take(side, np.sort(order[:half])), take(side, np.sort(order[half:]))


def pair(before, after, threshold, predictor, bootstrap):
    """Stack before (year 0) and after (year 1). Attach patient-bootstrap weights (times any
    planted row weight), the annual score fits and score shift refitted in every draw
    (annual_benchmarks, anchored to before) and baseline_values per draw for both scores."""
    frame = pd.concat([before["frame"].assign(year=0), after["frame"].assign(year=1)], ignore_index=True)
    frame["flagged"] = frame["probability"].to_numpy() >= threshold
    weights = ep.row_weights(frame["patient_id"], bootstrap) * np.r_[before["weight"], after["weight"]]
    risks, shifted, fits = ep.annual_benchmarks(frame, weights, 0)
    y, p, year = frame["label"].to_numpy(), frame["probability"].to_numpy(), frame["year"].to_numpy()
    metrics = {}
    for version, scores in (("frozen", np.broadcast_to(p, weights.shape)), ("shifted", shifted)):
        for period in (0, 1):
            at = year == period
            empty = {k: np.nan for k in ep.baseline_values(y[at], p[at], threshold)}
            metrics[(version, period)] = pd.DataFrame([
                ep.baseline_values(y[at], s[at], threshold, w[at]) if np.isfinite(s[at]).all() else empty
                for s, w in zip(scores, weights)])
    return {"predictor": predictor, "frame": frame, "threshold": threshold, "weights": weights, "risks": risks,
            "shifted": shifted, "fits": fits, "metrics": metrics,
            "features": np.vstack([before["features"], after["features"]]),
            "members": {k: np.r_[before["members"][k], after["members"][k]] for k in before["members"]}}


def pooled_metrics(tables):
    """Frozen-threshold metrics from baseline_values counts summed over predictors (for one
    predictor these are baseline_values' own values). AUROC does not add up over predictors
    with different scores: the pooled AUROC is the mean of the predictors' AUROCs."""
    n, deaths, survivors, tp, fp, fn = sum(t[list(COUNTS)].to_numpy(float) for t in tables).T
    div = lambda a, b: np.divide(a, b, out=np.full(len(a), np.nan), where=b > 0)
    weighted = lambda name: sum(t[name].to_numpy(float) * t["n"].to_numpy(float) for t in tables)
    return {"recall": div(tp, deaths), "false_alarm_rate": div(fp, survivors), "share_flagged": div(tp + fp, n),
            "death_f1": div(2 * tp, 2 * tp + fp + fn), "brier": div(weighted("brier"), n),
            "mean_score": div(weighted("mean_score"), n),
            "auroc": np.mean([t["auroc"].to_numpy(float) for t in tables], axis=0),
            "deaths": deaths, "survivors": survivors}


def what_moved(parts):
    """After minus before for every metric, with frozen scores and after the score shift."""
    rows = []
    for version, level in (("frozen", "what moved"), ("shifted", "after score shift")):
        before, after = (pooled_metrics([p["metrics"][(version, period)] for p in parts]) for period in (0, 1))
        for metric in METRICS:
            change = after[metric] - before[metric]
            per = {"recall": after["deaths"][0], "false_alarm_rate": after["survivors"][0]}.get(metric, np.nan)
            rows.append({"level": level, "measure": metric, "item": "", "before": before[metric][0],
                         "after": after[metric][0], "change_in_patients": change[0] * per, **ep.summarize_draws(change)})
    return pd.DataFrame(rows)


def flag_flips(before, after, threshold, had_feature=None):
    """Paired plants: patients whose frozen flag changed; optionally restricted checks on the
    patients who did not have the stopped feature."""
    y = before["frame"]["label"].to_numpy()
    p0, p1 = before["frame"]["probability"].to_numpy(float), after["frame"]["probability"].to_numpy(float)
    f0, f1 = p0 >= threshold, p1 >= threshold
    out = {"deaths_newly_flagged": int(np.sum((y == 1) & ~f0 & f1)), "deaths_no_longer_flagged": int(np.sum((y == 1) & f0 & ~f1)),
           "survivors_newly_flagged": int(np.sum((y == 0) & ~f0 & f1)), "survivors_no_longer_flagged": int(np.sum((y == 0) & f0 & ~f1))}
    if had_feature is not None:
        out.update(patients_with_feature=int(had_feature.sum()), flips_without_feature=int(np.sum((f0 != f1) & ~had_feature)),
                   max_score_change_without_feature=float(np.max(np.abs(p1 - p0)[~had_feature], initial=0.)),
                   max_score_change_with_feature=float(np.max(np.abs(p1 - p0)[had_feature], initial=0.)))
    return out


def flag_frame(parts):
    """Stacked rows with the frozen flag as the score (threshold 0.5) for flag-only functions."""
    frame = pd.concat([p["frame"] for p in parts], ignore_index=True)
    return frame.assign(probability=frame["flagged"].astype(float))


def record_length(parts):
    """rate_splits over months with care (survivors exact months, deaths quarters), mix/within.
    Pooled strata are predictor x length group, so a predictor's own mix stays its own."""
    strata = np.concatenate([i * 100 + ep.record_length_groups(p["frame"]["active_months"], p["frame"]["label"])
                             for i, p in enumerate(parts)])
    split = ep.rate_splits(flag_frame(parts), strata, np.hstack([p["weights"] for p in parts]), .5, 0)
    split = split[split["end"] == "1"]
    return split.assign(level="record length", measure=split["side"] + "|" + split["component"], item="")


def feature_contrasts(parts, names):
    """year_profile + profile_contrasts on all features; after-minus-before change of every view,
    ranked by absolute change and by absolute change over interval width (1 = largest)."""
    X = np.vstack([p["features"] for p in parts])
    profile, draws = ep.year_profile(flag_frame(parts), {str(n): X[:, j] for j, n in enumerate(names)},
                                     np.hstack([p["weights"] for p in parts]), .5, 0)
    table = ep.profile_contrasts(profile, draws, 0)
    table = table[(table["kind"] == "change") & (table["end"] == "1")].copy()
    size = table["value"].abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        signal = size / (table["high"] - table["low"])
    table["rank_by_size"] = size.groupby(table["view"]).rank(ascending=False, method="min")
    table["rank_by_size_over_interval"] = signal.groupby(table["view"]).rank(ascending=False, method="min")
    return table.assign(level="features", measure=table["view"], item=table["quantity"])


def group_rates(frame, member, weights, risks):
    """Per draw and period: membership share and inside/outside missed-death and false-alarm
    rates, observed and score-alone (the numerators and denominators of rule_error_rates)."""
    y, flagged, year = frame["label"].to_numpy(), frame["flagged"].to_numpy(), frame["year"].to_numpy()
    div = lambda a, b: np.divide(a, b, out=np.full(len(a), np.nan), where=b > 0)
    out = {}
    for period in (0, 1):
        at = year == period
        out[(period, "share")] = div(weights[:, at & member].sum(axis=1), weights[:, at].sum(axis=1))
        for group, mask in (("inside", member), ("outside", ~member)):
            chosen = at & mask
            for side, label, error in (("deaths", 1, ~flagged), ("survivors", 0, flagged)):
                risk = risks if label == 1 else 1 - risks
                out[(period, group, side, "observed")] = div(weights[:, chosen & (y == label) & error].sum(axis=1),
                                                             weights[:, chosen & (y == label)].sum(axis=1))
                out[(period, group, side, "score_alone")] = div(np.sum(weights[:, chosen & error] * risk[:, chosen & error], axis=1),
                                                                np.sum(weights[:, chosen] * risk[:, chosen], axis=1))
    return out


def concept_changes(part):
    """Every concept membership (rule; factor active): share change, inside/outside error-rate
    change observed, score-alone, and observed minus score-alone; rate_splits and
    rule_error_rates as the profile notebook calls them."""
    frame, weights, risks = part["frame"], part["weights"], part["risks"]
    y, year = frame["label"].to_numpy(), frame["year"].to_numpy()
    rows, splits, rates_tables = [], [], []
    for (concept, definition), member in part["members"].items():
        rates = group_rates(frame, member, weights, risks)
        source = SOURCE[definition]
        base = {"level": "concepts", "item": concept, "definition": definition, "source": source}
        rows.append({**base, "measure": f"{definition}|share", "before": rates[(0, "share")][0],
                     "after": rates[(1, "share")][0], **ep.summarize_draws(rates[(1, "share")] - rates[(0, "share")])})
        for group in ("inside", "outside"):
            for side in ("deaths", "survivors"):
                observed = rates[(1, group, side, "observed")] - rates[(0, group, side, "observed")]
                alone = rates[(1, group, side, "score_alone")] - rates[(0, group, side, "score_alone")]
                mask = member if group == "inside" else ~member
                counts = {f"{kind}_in_group_{when}": int(np.sum(mask & (year == period) & (y == label)))
                          for kind, label in (("deaths", 1), ("survivors", 0)) for when, period in (("before", 0), ("after", 1))}
                for version, change in (("observed", observed), ("score_alone", alone),
                                        ("observed_minus_score_alone", observed - alone)):
                    rows.append({**base, "measure": f"{definition}|{group}|{side}|{version}", "group": group,
                                 "side": side, "version": version, "before": rates[(0, group, side, "observed")][0],
                                 "after": rates[(1, group, side, "observed")][0], **counts, **ep.summarize_draws(change)})
        split = ep.rate_splits(flag_frame([part]), member, weights, .5, 0, risks)
        split = split[split["end"] == "1"]
        splits.append(split.assign(level="concept splits", item=concept, definition=definition, source=source,
                                   measure=definition + "|" + split["side"] + "|" + split["component"] + "|" + split["version"]))
        error_rates = ep.rule_error_rates(flag_frame([part]), member, weights, .5, 0, risks)
        rates_tables.append(error_rates[error_rates["period"] != "later"].assign(concept=concept, definition=definition, source=source))
    predictor = str(part["predictor"])
    return {"concepts": pd.DataFrame(rows).assign(predictor=predictor),
            "concept_splits": pd.concat(splits, ignore_index=True).assign(predictor=predictor),
            "concept_error_rates": pd.concat(rates_tables, ignore_index=True).assign(predictor=predictor)}


def levels(parts, names):
    """Every profile level for one plant, per predictor and pooled over the predictors given.
    Concept-level numbers are per predictor (concepts differ between predictors)."""
    sets = [(str(p["predictor"]), [p]) for p in parts] + ([("pooled", parts)] if len(parts) > 1 else [])
    out = {"what_moved": [], "record_length": [], "features": []}
    for name, chosen in sets:
        out["what_moved"].append(what_moved(chosen).assign(predictor=name))
        out["record_length"].append(record_length(chosen).assign(predictor=name))
        out["features"].append(feature_contrasts(chosen, names).assign(predictor=name))
    out = {k: pd.concat(v, ignore_index=True) for k, v in out.items()}
    concepts = [concept_changes(p) for p in parts]
    return out | {k: pd.concat([c[k] for c in concepts], ignore_index=True) for k in concepts[0]}


def stack(result):
    """One long table of every estimate (key columns + value/low/high) across levels."""
    names = ("what_moved", "record_length", "features", "concepts", "concept_splits")
    return pd.concat([result[k][KEY + ["value", "low", "high"]] for k in names], ignore_index=True)


def concept_ranks(concepts, uses):
    """Rank (1 = largest absolute change) of every concept's inside change within each predictor
    and jointly over all predictors ('pooled'), per membership/side/version."""
    table = concepts[concepts["group"] == "inside"].copy()
    table["uses_stopped_feature"] = [uses.get((p, c), False) for p, c in zip(table["predictor"], table["item"])]
    size = table["value"].abs()
    table["rank_in_predictor"] = size.groupby([table["predictor"], table["measure"]]).rank(ascending=False, method="min")
    table["rank_pooled"] = size.groupby(table["measure"]).rank(ascending=False, method="min")
    table["concepts_in_predictor"] = table.groupby(["predictor", "measure"])["item"].transform("size")
    return table


def no_change(sides, names, bootstrap, seed):
    """P5, one split: each predictor's rows split at random (seed [split, predictor position]);
    one half is before, the other after."""
    parts = []
    for i, (predictor, side, threshold) in enumerate(sides):
        first, second = split_halves(side, [seed, i])
        parts.append(pair(first, second, threshold, predictor, bootstrap))
    return stack(levels(parts, names))


def spread(values, keys):
    """Spread over splits of every estimate's difference between two random halves of different
    patients (point estimates; rows = splits). A scale reference only: for a paired plant (same
    patients before and after) the value when nothing is planted is exactly 0."""
    values = np.asarray(values, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanquantile(values, [.025, .975], axis=0)
        mean, sd = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
    return keys.assign(random_halves_mean=mean, random_halves_sd=sd, random_halves_low=low, random_halves_high=high,
                       random_halves_splits=np.isfinite(values).sum(axis=0))


def interval_exclusion(table):
    """Share of defined, non-degenerate patient intervals that exclude 0, per level."""
    defined = np.isfinite(table["low"]) & np.isfinite(table["high"]) & (table["low"] < table["high"])
    excludes = defined & ((table["low"] > 0) | (table["high"] < 0))
    return (table.assign(defined=defined, excludes_zero=excludes).groupby("level")[["defined", "excludes_zero"]].sum()
            .assign(share_excluding_zero=lambda t: t["excludes_zero"] / t["defined"]).reset_index())
