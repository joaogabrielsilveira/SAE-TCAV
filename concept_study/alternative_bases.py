"""Does the SAE earn its place? The frozen concept pipeline run on cheaper bases of the same embeddings.

For every last_3 reference build, the study's concept pipeline is run unchanged on factors from five
bases of the same 192-dim reference embeddings, standardized with the frozen reference scaler, on the
same rows. Nothing of the frozen study is refit or rewritten.

- sae:      the three frozen dictionaries, re-encoded here. Families, candidates, rules, cutoffs and
            agreement must reproduce the frozen study (correctness anchor, written to anchor.csv).
- pca:      principal components of the SAE training rows (sae_discovery). Each component gives two
            nonnegative factors, ReLU(+score) and ReLU(-score). Leading components are taken until the
            factor count equals the SAE candidate count of that reference (9-17 factors, 5-9 components;
            the top 5 of 192 components already hold 85-93% of the variance).
- kmeans:   k-means (10 starts) on the SAE training rows, k = the SAE candidate count, one fit per SAE
            seed. Factor j is the margin by which a patient is closer to centre j than to the next
            centre (squared distance) and zero outside cluster j: support = membership, size = how
            firmly the patient belongs, so the percentile-of-positives targets keep their meaning.
- random:   ReLU of 288 random unit directions per seed (as many as SAE latents) on the standardized
            embeddings, no bias (the data are centred on the SAE training rows). Three draws of three seeds.
- rule_cav: no factor search. For each frozen SAE concept, a CAV fitted directly on the raw embeddings
            of the reference matching rows, rule patients against all other patients
            (temporal_cav.train_temporal_cav, mean unit direction of 10 balanced draws). The factor is the
            CAV projection; its cutoff gives the same share of rule-discovery rows above it as the SAE
            concept's own target. The rule is the SAE concept's frozen rule.

sae, kmeans and random go through the same family step (Pearson mutual best match over three seeds,
full-family medoids ranked by positive count); every basis is capped at the SAE candidate count of the
reference. Rules come from the frozen semantic sweep (pilot.run_threshold_sweep: rule-discovery rows,
patient 50/50 fit/tune split, A = top 10..50% of positives, F2 selection). A concept is a frozen rule
and cutoff; A = factor > cutoff and R = rule on held-out evaluation rows of every year. Agreement uses
the study's joint patient bootstrap (+ concept resampling for means); the rule-input drift relation
reuses input_drift (years up to 2014, concept-cluster bootstrap). CAV sensitivity is not repeated.

    .venv/bin/python -m concept_study.alternative_bases
"""
import json
import time

import numpy as np
import pandas as pd

from concept_study import bootstrap, family_controls, input_drift, loading, measures, paths, tables

SEEDS = paths.SAE_SEEDS
DEVICE = paths.DEVICE
RANDOM_DRAWS = 5
RANDOM_LATENTS = 288
CAV_SEEDS = tuple(range(1001, 1011))
WORKERS = 24
BASES = ("sae", "pca", "kmeans", "random", "rule_cav")
COLORS = {"sae": "#2a78d6", "pca": "#eb6834", "kmeans": "#1baf7a", "random": "#eda100", "rule_cav": "#6250d6"}
YEARS = measures.YEARS
OUTPUT = paths.OUTPUT / "extensions" / "alternative_bases"


# ---------------------------------------------------------------- factor bases (standardized rows in, factors out)

def pca_encoder(fit, count):
    """ReLU(+score), ReLU(-score) of the leading principal components of `fit`, `count` factors in all."""
    mean = fit.mean(0)
    vectors = np.linalg.svd(fit - mean, full_matrices=False)[2][:(count + 1) // 2]

    def encode(z):
        scores = (z - mean) @ vectors.T
        factors = np.empty((len(z), 2 * scores.shape[1]), np.float32)
        factors[:, 0::2], factors[:, 1::2] = np.maximum(scores, 0), np.maximum(-scores, 0)
        return factors[:, :count]
    return encode


def kmeans_encoder(fit, k, seed):
    """Factor j = (squared distance to the second-closest centre) - (to centre j) for members of cluster j, else 0."""
    from sklearn.cluster import KMeans
    centres = KMeans(k, n_init=10, random_state=seed).fit(fit).cluster_centers_

    def encode(z):
        distance = ((z[:, None, :] - centres[None]) ** 2).sum(-1)
        order = np.argsort(distance, axis=1)
        rows = np.arange(len(z))
        factors = np.zeros_like(distance, dtype=np.float32)
        factors[rows, order[:, 0]] = distance[rows, order[:, 1]] - distance[rows, order[:, 0]]
        return factors
    return encode


def random_encoder(dimension, latents, seed):
    """ReLU of random unit directions (no bias)."""
    directions = np.random.default_rng(seed).standard_normal((latents, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return lambda z: np.maximum(z @ directions.T, 0).astype(np.float32)


def cav_direction(embeddings, features, rule, patients):
    """Mean unit CAV over balanced draws: rule rows against every other row (no activation is used)."""
    from temporal_cav import train_temporal_cav
    fits = [train_temporal_cav(embeddings=embeddings, features=features, activations=np.zeros(len(embeddings)),
                               rule=rule, activation_target=0.1, rule_source="rule_direct", patient_ids=patients,
                               minimum_positive=2, minimum_negative=2, seed=seed) for seed in CAV_SEEDS]
    directions = [fit["cav"] for fit in fits if fit["valid"]]
    if not directions:
        return None
    mean = np.mean(directions, axis=0)
    return mean / np.linalg.norm(mean)


def matched_cutoff(projection, share):
    """Cutoff with `share` of the rows strictly above it."""
    return float(np.quantile(projection, 1 - share)) if share > 0 else float(np.max(projection))


# ---------------------------------------------------------------- per-concept measures

def rule_quality(target, in_rule):
    """Precision, recall and F1 of the rule for the factor target."""
    both = float((target & in_rule).sum())
    precision = both / in_rule.sum() if in_rule.any() else np.nan
    recall = both / target.sum() if target.any() else np.nan
    f1 = 2 * both / (target.sum() + in_rule.sum()) if (target.any() or in_rule.any()) else np.nan
    return precision, recall, f1


def agreement_replicates(in_a, in_r, years, weights):
    """(replicates, YEARS) share of rule rows that activate, under the joint patient bootstrap weights."""
    masks = np.concatenate([np.stack([in_r & (years == y) for y in YEARS], 1),
                            np.stack([in_r & in_a & (years == y) for y in YEARS], 1)], axis=1)
    sums = bootstrap.weighted_sums(weights, masks, np.ones((len(years), 1), np.float32))[..., 0].cpu().numpy()
    return measures.ratio(sums[:, len(YEARS):], sums[:, :len(YEARS)])


def steps(agreement, reference):
    """(replicates, steps) agreement change of every consecutive pair (t, t+1) from the reference year."""
    at = {year: k for k, year in enumerate(YEARS)}
    return np.stack([agreement[:, at[t + 1]] - agreement[:, at[t]] for t in range(reference, paths.LAST_YEAR)], 1)


def jaccard(a, b):
    union = len(a | b) if isinstance(a, set) else int((a | b).sum())
    inter = len(a & b) if isinstance(a, set) else int((a & b).sum())
    return inter / union if union else np.nan


# ---------------------------------------------------------------- one reference build

def reference_data(inputs, record):
    """Matching, rule, SAE-training and evaluation rows of one build, with checks against the frozen objects."""
    import cmr_common
    import concept_subgroup_drift_pilot as pilot
    build, reference = record["build"], record["reference"]
    cache, population = inputs["caches"][build], inputs["population"]
    roles = next(i["roles"] for i in inputs["inventory"] if (i["build"], i["reference"]) == (build, reference))
    families, match_emb, _ = family_controls.matching_inputs(cache, reference)
    match_rows = families["row_index"]
    rule_pos = np.flatnonzero(roles["row_role"][match_rows] == "rule_discovery")
    _, archive, rows = cmr_common.verified_predictor_archive(population, cache, reference, roles)

    split = pilot.reference_year_roles(population, reference, cache.config["split_seed"])
    if split["identity"] != roles["identity"]:
        raise ValueError(f"{build} {reference}: recomputed roles differ from the frozen roles")
    chosen = np.flatnonzero((population["year"][rows] == reference) & (split["row_role"][rows] == "sae_discovery")
                            & ~split["is_entrant"][rows])
    if pilot._identity_hash({"row_index": rows[chosen]}) != cache.load(f"S/{SEEDS[0]}/{reference}")["discovery_rows_identity"]:
        raise ValueError(f"{build} {reference}: SAE training rows differ from the frozen ones")
    evaluation = np.flatnonzero(roles["row_role"][rows] == "evaluation")
    table = measures.load_build(record)
    if not (np.array_equal(population["patient_id"][rows[evaluation]].astype(str), table["patient_id"])
            and np.array_equal(population["year"][rows[evaluation]], table["year"])):
        raise ValueError(f"{build} {reference}: evaluation rows differ from the patient table")
    dictionaries = pilot.load_reference_dictionaries(cache, reference, DEVICE)
    scaler = dictionaries[SEEDS[0]]["scaler"]
    fit_emb = cmr_common.read_array_rows(archive, "embeddings", chosen)
    if not np.allclose(fit_emb.astype(np.float64).mean(0), scaler.mean_, atol=1e-5):
        raise ValueError(f"{build} {reference}: scaler was not fitted on these SAE training rows")
    eval_emb = cmr_common.read_array_rows(archive, "embeddings", evaluation)
    standardize = lambda e: np.asarray(scaler.transform(e), dtype=np.float32)
    return {"build": build, "reference": reference, "cache": cache, "families": families,
            "dictionaries": dictionaries, "scaler": scaler, "standardize": standardize,
            "match_emb": match_emb, "eval_emb": eval_emb, "fit_z": standardize(fit_emb),
            "rule_pos": rule_pos,
            "X_match": population["X"][match_rows].astype(np.float64),
            "X_rule": population["X"][match_rows[rule_pos]].astype(np.float64),
            "rule_patients": population["patient_id"][match_rows[rule_pos]],
            "match_patients": population["patient_id"][match_rows].astype(str),
            "X_eval": population["X"][rows[evaluation]].astype(np.float64),
            "feature_names": list(population["feature_names"]), "table": table}


def family_candidates(codes, prefix, cap):
    """Families over the seeds on the matching rows; full-family medoids ranked by positive count, capped."""
    import concept_subgroup_drift_pilot as pilot
    result = pilot.build_factor_families([codes[s] for s in SEEDS], list(SEEDS), device=DEVICE, id_prefix=prefix)
    medoids = {f["id"]: codes[f["medoid"][0]][:, f["medoid"][1]] for f in result["families"] if f["kind"] == "full"}
    latents = codes[SEEDS[0]].shape[1]
    selection = pilot.family_candidates(result["families"], medoids, latents, 1.0)
    return result["counts"], selection["candidates"][:cap]


def rule_sweep(data, candidates, match_codes):
    """The frozen semantic threshold sweep on the rule-discovery rows of the candidate medoids."""
    import concept_subgroup_drift_pilot as pilot
    config = data["cache"].config
    activations = np.stack([match_codes[c["medoid"][0]][data["rule_pos"], c["medoid"][1]] for c in candidates])
    return pilot.run_threshold_sweep(data["X_rule"], data["rule_patients"], data["feature_names"], candidates,
                                     activations.astype(np.float64), config["threshold_sweep"], config["rule"],
                                     reference_year=data["reference"], split_seed=config["split_seed"],
                                     workers=WORKERS, rule_source=config["rule"]["rule_source"],
                                     recurrence=config["rule"]["recurrence"])


def concept_record(basis, draw, data, family_id, rule, cutoff, act_rule, act_eval, entry=None):
    """Reference-year description plus A and R on the evaluation rows of one concept."""
    from concept_study.patient_table import rule_features
    target, in_rule = act_rule > cutoff, rule.mask(data["X_rule"])
    precision, recall, f1 = rule_quality(target, in_rule)
    prefix = data["build"] if basis == "sae" else f"{basis}{draw}/{data['build']}"
    return {"basis": basis, "draw": draw, "build": data["build"], "reference": data["reference"],
            "concept": f"{prefix}/{data['reference']}/{family_id}", "family_id": family_id, "cutoff": float(cutoff),
            "activation_target": entry["activation_target"] if entry else np.nan,
            "rule_count": len(rule.rules), "condition_count": sum(len(r.conditions) for r in rule.rules),
            "rule_features": rule_features(rule.to_dict()), "rule_dict": rule.to_dict(), "rule_identity": entry["rule_identity"] if entry else None,
            "tune_f2": entry["tune_metrics"]["f2"] if entry else np.nan,
            "rule_precision": precision, "rule_recall": recall, "rule_f1": f1,
            "target_share_rule_rows": float(target.mean()), "R_rule": in_rule,
            "in_A": act_eval > cutoff, "in_R": rule.mask(data["X_eval"])}


def basis_concepts(basis, draw, data, encoders, cap, prefix):
    """Candidates, frozen rules and concepts of one basis (encoders: {seed: f(raw embeddings) -> factors})."""
    from semantic_rules import RuleSet
    match = {s: e(data["match_emb"]) for s, e in encoders.items()}
    if len(encoders) == 1:
        (seed, codes), = match.items()
        counts = None
        candidates = [{"rank": j + 1, "family_id": f"{prefix}/{j // 2 + 1}{'+-'[j % 2]}", "medoid": [seed, j],
                       "positive_count": int((codes[:, j] > 0).sum())} for j in range(codes.shape[1])]
    else:
        counts, candidates = family_candidates(match, prefix, cap)
    sweep = rule_sweep(data, candidates, match)
    evaluation = {}
    concepts = []
    for entry in sweep["frozen"]:
        seed, factor = entry["medoid"]
        if seed not in evaluation:
            evaluation[seed] = encoders[seed](data["eval_emb"])
        concepts.append(concept_record(basis, draw, data, entry["family_id"], RuleSet.from_dict(entry["rule_set"]),
                                       entry["cutoff"], match[seed][data["rule_pos"], factor],
                                       evaluation[seed][:, factor], entry))
    summary = {"basis": basis, "draw": draw, "build": data["build"], "reference": data["reference"],
               "candidates": len(candidates), "rules": len(concepts),
               "full_families": counts["full"] if counts else np.nan}
    return concepts, summary, sweep


def sae_encoders(data):
    """The frozen dictionaries through the pilot's own encoder (frozen scaler inside)."""
    import concept_subgroup_drift_pilot as pilot
    return {s: (lambda e, s=s: pilot.encode_with_reference_sae(data["dictionaries"][s], e, DEVICE, 256)) for s in SEEDS}


def on_raw(encoder, data):
    """Encoder of standardized rows -> encoder of raw embeddings (frozen reference scaler)."""
    return lambda e: encoder(data["standardize"](e))


def sae_anchor(data, concepts, summary, sweep, index):
    """Compare the re-run SAE pipeline with the frozen families, rules and patient-table concepts."""
    cache, reference = data["cache"], data["reference"]
    cached_families = cache.load(f"D/{reference}/families")
    cached_rules = {e["family_id"]: e for e in cache.load(f"D/{reference}/rules")["frozen"]}
    frozen_records = {c["concept"]: c for c in index["concepts"]}
    mine = {c["family_id"]: c for c in concepts}
    activation_gap, mask_mismatch, agreement_gap = 0.0, 0, 0.0
    missing = 0
    for c in concepts:
        record = frozen_records.get(c["concept"])
        if record is None:
            missing += 1
            continue
        stored = measures.load_concept(record)
        mask_mismatch += int((stored["in_A"] != c["in_A"]).sum() + (stored["in_R"] != c["in_R"]).sum())
        frozen = measures.load_replicates(record)["concept|agreement"]
        agreement_gap = max(agreement_gap, float(np.nanmax(np.abs(frozen - c["agreement"]))) if np.isfinite(frozen).any() else 0)
        if not np.array_equal(np.isnan(frozen), np.isnan(c["agreement"])):
            agreement_gap = np.inf
    return {"build": data["build"], "reference": reference,
            "family_counts_equal": summary["full_families"] == cached_families["construction"]["counts"]["full"],
            "candidates_equal": summary["candidates"] == len(cached_families["selection"]["candidates"]),
            "frozen_rules": len(mine), "frozen_rules_cached": len(cached_rules),
            "rule_identities_equal": mine.keys() == cached_rules.keys() and all(
                mine[k]["rule_identity"] == cached_rules[k]["rule_identity"]
                and np.isclose(mine[k]["cutoff"], cached_rules[k]["cutoff"]) for k in mine),
            "concepts_missing_from_index": missing, "A_or_R_row_mismatches": mask_mismatch, "agreement_max_abs_gap": agreement_gap,
            "index_concepts": sum(c["build"] == data["build"] and c["reference"] == reference for c in index["concepts"])}


def rule_cav_concepts(data, sae_concepts):
    """Rule-direct CAV factor for each SAE concept: same rule, CAV projection, matched target share."""
    out = []
    for c in sae_concepts:
        from semantic_rules import RuleSet
        rule = RuleSet.from_dict(c["rule_dict"])
        direction = cav_direction(data["match_emb"], data["X_match"], rule, data["match_patients"])
        if direction is None:
            continue
        projection_rule = data["match_emb"][data["rule_pos"]] @ direction
        cutoff = matched_cutoff(projection_rule, c["target_share_rule_rows"])
        record = concept_record("rule_cav", 0, data, c["family_id"], rule, cutoff, projection_rule,
                                data["eval_emb"] @ direction)
        record.update(activation_target=c["activation_target"], rule_identity=c["rule_identity"],
                      paired_sae=c["concept"], cav=direction)
        out.append(record)
    return out


def run_reference(inputs, record, index, weights):
    """Every basis on one build: concept records with agreement replicates, summaries and the SAE anchor."""
    started = time.time()
    data = reference_data(inputs, record)
    reference, build = data["reference"], data["build"]
    prefix = f"{reference}/{data['cache'].config['split_seed']}"
    import concept_subgroup_drift_pilot as pilot
    sae_cap = pilot.candidate_cap(RANDOM_LATENTS, data["cache"].config["candidate_cap_fraction"])
    sae, sae_summary, sae_sweep = basis_concepts("sae", 0, data, sae_encoders(data), sae_cap, prefix)
    cap = sae_summary["candidates"]
    concepts, summaries = list(sae), [sae_summary]
    pca, s, _ = basis_concepts("pca", 0, data, {0: on_raw(pca_encoder(data["fit_z"], cap), data)}, cap, f"pca/{prefix}")
    concepts += pca
    summaries.append({**s, "variance_share": float(pca_variance(data["fit_z"], (cap + 1) // 2))})
    kmeans = {s_: on_raw(kmeans_encoder(data["fit_z"], cap, s_), data) for s_ in SEEDS}
    k, s, _ = basis_concepts("kmeans", 0, data, kmeans, cap, f"kmeans/{prefix}")
    concepts += k
    summaries.append(s)
    for draw in range(1, RANDOM_DRAWS + 1):
        encoders = {s_: on_raw(random_encoder(data["fit_z"].shape[1], RANDOM_LATENTS, [20261110, reference, draw, s_]), data)
                    for s_ in SEEDS}
        r, s, _ = basis_concepts("random", draw, data, encoders, cap, f"random{draw}/{prefix}")
        concepts += r
        summaries.append(s)
    cav = rule_cav_concepts(data, sae)
    concepts += cav
    summaries.append({"basis": "rule_cav", "draw": 0, "build": build, "reference": reference,
                      "candidates": len(sae), "rules": len(cav), "full_families": np.nan})

    row_weights = bootstrap.row_weights(data["table"]["patient_id"], *weights)
    sae_rules = [c for c in sae]
    sae_features = set().union(*[set(c["rule_features"]) for c in sae_rules]) if sae_rules else set()
    for c in concepts:
        c["agreement"] = agreement_replicates(c["in_A"], c["in_R"], data["table"]["year"], row_weights)
        c["rule_overlap_with_sae"] = max([jaccard(c["R_rule"], s["R_rule"]) for s in sae_rules] or [np.nan])
        c["feature_overlap_with_sae"] = max([jaccard(set(c["rule_features"]), set(s["rule_features"])) for s in sae_rules] or [np.nan])
        c["same_rule_as_sae"] = c["rule_identity"] in {s["rule_identity"] for s in sae_rules}
    for s in summaries:
        own = [c for c in concepts if c["basis"] == s["basis"] and c["draw"] == s["draw"]]
        s["rule_feature_jaccard_with_sae"] = jaccard(set().union(*[set(c["rule_features"]) for c in own]) if own else set(), sae_features)
    drift = drift_rows(concepts, data)
    anchor = sae_anchor(data, sae, sae_summary, sae_sweep, index)
    print(f"{build} {reference}: cap {cap}; rules " + ", ".join(f"{s['basis']}{s['draw'] or ''} {s['rules']}" for s in summaries)
          + f"; anchor {anchor}; {time.time() - started:.0f}s", flush=True)
    for c in concepts:   # keep only what the summaries need
        for name in ("in_A", "in_R", "R_rule"):
            c.pop(name)
    return concepts, summaries, drift, anchor


def pca_variance(fit, components):
    values = np.linalg.svd(fit - fit.mean(0), compute_uv=False) ** 2
    return values[:components].sum() / values.sum()


def drift_rows(concepts, data):
    """input_drift's concept rows: rule-feature PSI and agreement change since the reference, years up to 2014."""
    table, reference = data["table"], data["reference"]
    names, now = list(table["feature_names"]), table["year"] == reference
    psi = {year: input_drift.psi(table["features"][now], table["features"][table["year"] == year])
           for year in range(reference + 1, input_drift.LAST + 1) if (table["year"] == year).any()}
    rows = []
    for c in concepts:
        a, r = c["in_A"], c["in_R"]
        columns = [names.index(f) for f in c["rule_features"] if f in names]
        for year, values in psi.items():
            then = table["year"] == year
            if r[now].sum() == 0 or r[then].sum() == 0 or not columns:
                continue
            rows.append({"basis": c["basis"], "draw": c["draw"], "concept": c["concept"], "reference": reference,
                         "year": year, "years_since_reference": year - reference,
                         "rule_feature_psi": float(values[columns].mean()),
                         "agreement_change": float(a[then & r].mean() - a[now & r].mean())})
    return rows


# ---------------------------------------------------------------- summaries

def within_concept_drift(frame):
    """input_drift's within-concept-and-years rank partial correlation with its concept-cluster bootstrap.

    Rows are sorted by concept key first: the bootstrap draws follow the order in which concepts appear,
    and this is the order of input_drift (index order), so the SAE reproduces its interval exactly."""
    frame = frame.sort_values("concept", kind="stable")
    statistic = lambda f: input_drift.partial_spearman(f["rule_feature_psi"].to_numpy(), f["agreement_change"].to_numpy(),
                                                       f["concept"].to_numpy(), f["years_since_reference"].to_numpy())
    return input_drift.concept_bootstrap(frame, statistic)


def summarize(concepts, summaries, drift):
    """One row per basis (and random draw): reference-year description, agreement over time, drift relation."""
    reference = pd.DataFrame(summaries)
    drift = pd.DataFrame(drift)
    out = []
    for (basis, draw), part in reference.groupby(["basis", "draw"], sort=False):
        own = [c for c in concepts if c["basis"] == basis and c["draw"] == draw]
        labels = np.array([c["concept"] for c in own])
        row = {"basis": basis, "draw": draw, "builds": len(part), "candidates": int(part["candidates"].sum()),
               "concepts": int(part["rules"].sum()), "share_with_rule": part["rules"].sum() / part["candidates"].sum(),
               "full_families": part["full_families"].sum() if part["full_families"].notna().all() else np.nan,
               "rule_feature_jaccard_with_sae_mean": part["rule_feature_jaccard_with_sae"].mean(),
               "pca_variance_share_min": part.get("variance_share", pd.Series(dtype=float)).min(),
               "pca_variance_share_max": part.get("variance_share", pd.Series(dtype=float)).max()}
        frame = pd.DataFrame([{k: c[k] for k in ("condition_count", "rule_count", "rule_precision", "rule_recall",
                                                 "rule_f1", "tune_f2", "rule_overlap_with_sae", "feature_overlap_with_sae",
                                                 "same_rule_as_sae")} for c in own])
        for name in frame.columns:
            values = frame[name].astype(float)
            row[f"{name}_median"], row[f"{name}_q25"], row[f"{name}_q75"] = values.median(), values.quantile(.25), values.quantile(.75)
        at_reference = np.stack([c["agreement"][:, YEARS.index(c["reference"])] for c in own], 1)
        row["reference_agreement"], row["reference_agreement_low"], row["reference_agreement_high"] = \
            tables.mean_with_interval(at_reference, labels)
        changes = [steps(c["agreement"], c["reference"]) for c in own]
        step_labels = np.concatenate([[c["concept"]] * x.shape[1] for c, x in zip(own, changes)])
        stacked = np.concatenate(changes, 1)
        row["concept_steps"] = int(np.isfinite(stacked[0]).sum())
        row["step_change"], row["step_change_low"], row["step_change_high"] = tables.mean_with_interval(stacked, step_labels)
        d = drift[(drift["basis"] == basis) & (drift["draw"] == draw)]
        row["drift_rows"], row["drift_concepts"] = len(d), d["concept"].nunique()
        row["psi_within_concept_years"], row["psi_low"], row["psi_high"] = within_concept_drift(d)
        out.append(row)
    return pd.DataFrame(out)


def paired_cav_difference(concepts):
    """rule_cav minus its own SAE concept: reference agreement and mean step change (same concepts, same weights)."""
    sae = {c["concept"]: c for c in concepts if c["basis"] == "sae"}
    pairs = [(c, sae[c["paired_sae"]]) for c in concepts if c["basis"] == "rule_cav"]
    labels = np.array([s["concept"] for _, s in pairs])
    level = np.stack([c["agreement"][:, YEARS.index(c["reference"])] - s["agreement"][:, YEARS.index(s["reference"])]
                      for c, s in pairs], 1)
    change = [steps(c["agreement"], c["reference"]) - steps(s["agreement"], s["reference"]) for c, s in pairs]
    step_labels = np.concatenate([[s["concept"]] * x.shape[1] for (_, s), x in zip(pairs, change)])
    rows = []
    for name, values, lab in (("reference_agreement", level, labels), ("step_change", np.concatenate(change, 1), step_labels)):
        point, low, high = tables.mean_with_interval(values, lab)
        rows.append({"measure": f"rule_cav minus sae: {name}", "point": point, "low": low, "high": high,
                     "concepts": len(pairs)})
    return pd.DataFrame(rows)


def trajectory(concepts):
    """Mean agreement by years since the reference, per basis and draw, with the joint interval."""
    rows = []
    keys = sorted({(c["basis"], c["draw"]) for c in concepts}, key=lambda k: (BASES.index(k[0]), k[1]))
    for basis, draw in keys:
        own = [c for c in concepts if c["basis"] == basis and c["draw"] == draw]
        for offset in range(len(YEARS)):
            values = [(c["agreement"][:, YEARS.index(c["reference"]) + offset], c["concept"]) for c in own
                      if c["reference"] + offset <= paths.LAST_YEAR]
            if len(values) < 2:
                continue
            point, low, high = tables.mean_with_interval(np.stack([v for v, _ in values], 1), np.array([l for _, l in values]))
            rows.append({"basis": basis, "draw": draw, "years_since_reference": offset, "concepts": len(values),
                         "agreement": point, "low": low, "high": high})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def label_of(basis, draw):
    return {"sae": "SAE", "pca": "PCA", "kmeans": "k-means", "random": f"random {draw}", "rule_cav": "rule CAV"}[basis]


def plot_reference(summary, concepts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(summary.itertuples())
    labels = [label_of(r.basis, r.draw) for r in rows]
    colors = [COLORS[r.basis] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), sharey=True)
    ax = axes[0]
    ax.barh(y, [r.share_with_rule for r in rows], color=colors, height=0.6)
    for yi, r in zip(y, rows):
        ax.text(r.share_with_rule + 0.01, yi, f"{r.concepts}/{r.candidates}", va="center", fontsize=8, color="#52514e")
    ax.set(xlim=(0, 1.15), xlabel="share of candidates with a frozen rule", title="Candidates that yield a rule")
    for ax, name, title in ((axes[1], "condition_count", "Rule conditions per concept"),
                            (axes[2], "rule_overlap_with_sae", "Rule patients: best Jaccard\nwith an SAE concept rule")):
        for yi, r, color in zip(y, rows, colors):
            values = np.array([c[name] for c in concepts if c["basis"] == r.basis and c["draw"] == r.draw], float)
            jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(values))
            ax.scatter(values, yi + jitter, s=10, alpha=0.5, color=color)
            ax.plot([np.nanmedian(values)] * 2, [yi - 0.3, yi + 0.3], color="#0b0b0b", lw=2)
        ax.set(title=title, xlabel="per concept (bar = median)")
    ax = axes[3]
    for yi, r, color in zip(y, rows, colors):
        ax.errorbar(r.reference_agreement, yi, xerr=[[r.reference_agreement - r.reference_agreement_low],
                                                     [r.reference_agreement_high - r.reference_agreement]],
                    fmt="o", color=color, ms=7, capsize=3)
    ax.set(xlabel="mean agreement P(A | R), held-out rows", title="Agreement in the reference year")
    axes[0].set_yticks(y, labels)
    for ax in axes:
        ax.grid(axis="x", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("last_3 reference builds 2007-2013: concepts from each basis at the reference year", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_over_time(summary, curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    ax = axes[0]
    for (basis, draw), part in curves.groupby(["basis", "draw"], sort=False):
        part = part[part["concepts"] >= 4]
        style = dict(color=COLORS[basis], lw=2 if basis != "random" else 1.2, alpha=1 if basis != "random" else 0.8)
        label = label_of(basis, draw) if basis != "random" else (f"random ({RANDOM_DRAWS} draws)" if draw == 1 else None)
        ax.plot(part["years_since_reference"], part["agreement"], marker="o", ms=4, label=label, **style)
        if basis != "random":
            ax.fill_between(part["years_since_reference"], part["low"], part["high"], color=COLORS[basis], alpha=0.12, lw=0)
    ax.set(xlabel="years since the reference year", ylabel="mean agreement P(A | R)",
           title="Agreement over time (frozen rules and cutoffs)\nbands: joint patient + concept 95% interval")
    ax.legend(frameon=False, fontsize=8)
    rows = list(summary.itertuples())
    y = np.arange(len(rows))[::-1]
    for ax, (point, low, high), title in (
            (axes[1], ("step_change", "step_change_low", "step_change_high"), "Mean agreement change per year step"),
            (axes[2], ("psi_within_concept_years", "psi_low", "psi_high"),
             "Agreement change vs own rule-input PSI\n(rank r within concept and years, to 2014)")):
        for yi, r in zip(y, rows):
            p, lo, hi = getattr(r, point), getattr(r, low), getattr(r, high)
            ax.errorbar(p, yi, xerr=[[p - lo], [hi - p]], fmt="o", color=COLORS[r.basis], ms=7, capsize=3)
        ax.axvline(0, color="#8a8984", lw=0.8)
        ax.set_yticks(y, [label_of(r.basis, r.draw) for r in rows])
        ax.set(title=title, xlabel="point and 95% interval")
    for ax in axes:
        ax.grid(color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- run

def main():
    started = time.time()
    index = measures.load_index()
    inputs = loading.open_inputs(DEVICE)
    weights = bootstrap.patient_weights(measures.all_patient_ids(index))
    concepts, summaries, drift, anchors = [], [], [], []
    for record in index["builds"]:
        if paths.PRIMARY_VARIANT not in record["variants"]:
            continue
        c, s, d, a = run_reference(inputs, record, index, weights)
        concepts += c
        summaries += s
        drift += d
        anchors.append(a)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "figures").mkdir(exist_ok=True)
    summary = summarize(concepts, summaries, drift)
    paired = paired_cav_difference(concepts)
    curves = trajectory(concepts)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    paired.to_csv(OUTPUT / "rule_cav_vs_sae.csv", index=False)
    curves.to_csv(OUTPUT / "agreement_by_years_since_reference.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUTPUT / "reference_builds.csv", index=False)
    pd.DataFrame(drift).to_csv(OUTPUT / "concept_drift.csv", index=False)
    pd.DataFrame(anchors).to_csv(OUTPUT / "anchor.csv", index=False)
    keep = ("basis", "draw", "build", "reference", "concept", "activation_target", "cutoff", "rule_count",
            "condition_count", "tune_f2", "rule_precision", "rule_recall", "rule_f1", "target_share_rule_rows",
            "rule_overlap_with_sae", "feature_overlap_with_sae", "same_rule_as_sae")
    frame = pd.DataFrame([{**{k: c[k] for k in keep}, "rule_features": ";".join(c["rule_features"]),
                           **{f"agreement_{y}": c["agreement"][0, k] for k, y in enumerate(YEARS)}} for c in concepts])
    frame.to_csv(OUTPUT / "concepts.csv", index=False)
    plot_reference(summary, concepts, OUTPUT / "figures" / "reference_year.png")
    plot_over_time(summary, curves, OUTPUT / "figures" / "over_time.png")
    (OUTPUT / "run.json").write_text(json.dumps({"seconds": time.time() - started, "workers": WORKERS,
                                                 "random_draws": RANDOM_DRAWS, "variant": paths.PRIMARY_VARIANT}))
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(summary.round(3).T.to_string())
        print(paired.round(4).to_string())
        print(pd.DataFrame(anchors).to_string())


if __name__ == "__main__":
    main()
