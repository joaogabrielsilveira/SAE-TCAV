"""Candidate sources for the concept-candidate study: group score, PCA, matched-count pool, candidate table.

Rows: scales and PCA are fitted on SAE-training rows (role sae_discovery); candidates are ranked and
checked on the matching rows (rule_discovery + cav_fit). Evaluation, concept_selection, later-year and
entrant rows never enter a fit (check_fit_rows).
"""
import math
import warnings

import numpy as np

POOLED_SOURCES = ("sae", "group", "pca")
LEAF_COUNTS = (3, 5, 8)
TIE = 1e-6
MIN_PRECISION = 0.5
MIN_LIFT = 1.5
SEED = 42
CAV_C = 0.1
MIN_CLASS = 2


def check_fit_rows(rows, row_role, year, is_entrant, reference, allowed_roles):
    """Raise if any fitting row has a disallowed role, is not in the reference year, or is an entrant."""
    rows = np.asarray(rows, dtype=int)
    bad_role = ~np.isin(row_role[rows], allowed_roles)
    if bad_role.any():
        raise ValueError(f"fit rows with roles {sorted(set(row_role[rows][bad_role]))}, allowed {list(allowed_roles)}")
    if (year[rows] > reference).any():
        raise ValueError("fit rows from a later year than the reference")
    if (year[rows] != reference).any():
        raise ValueError("fit rows outside the reference year")
    if is_entrant[rows].any():
        raise ValueError("fit rows of entrant patients")


def group_score(match_members, fit_members):
    """Mean of members divided by their SD on the SAE-training rows; SD-0 members are left out and counted."""
    scale = np.asarray(fit_members, dtype=np.float64).std(0)
    used = scale > 0
    score = None
    if used.any():
        score = (np.asarray(match_members, dtype=np.float64)[:, used] / scale[used]).mean(1)
    return {"score": score, "scale": scale, "used": used, "left_out": int((~used).sum())}


def member_correlations(match_members, score, used):
    """Pearson correlation of each used member with the group score and with each other; negative ones listed."""
    members = np.asarray(match_members, dtype=np.float64)[:, used]
    kept = np.flatnonzero(used)
    with_score = np.array([np.corrcoef(members[:, j], score)[0, 1] for j in range(members.shape[1])])
    between = np.corrcoef(members.T) if members.shape[1] > 1 else np.ones((1, 1))
    negative_pairs = [(int(kept[i]), int(kept[j])) for i in range(len(kept)) for j in range(i + 1, len(kept))
                      if between[i, j] < 0]
    return {"members": kept.tolist(), "with_score": with_score, "between": between,
            "negative_with_score": [int(m) for m, r in zip(kept, with_score) if r < 0],
            "negative_pairs": negative_pairs}


def pca_candidates(fit_z, count):
    """Encoder giving ReLU(+score) and ReLU(-score) of the leading components of the SAE-training rows."""
    from concept_study.alternative_bases import pca_encoder
    return pca_encoder(fit_z, count)


def matched_count_pool(candidates, cap):
    """SAE, group and PCA candidates ranked by positive count on the matching rows, first `cap` kept. Random is never pooled."""
    pooled = [c for c in candidates if c["source"] in POOLED_SOURCES]
    return sorted(pooled, key=lambda c: -c["positive_count"])[:cap]


def candidate_row(build, reference, source, seed, draw, definition, scale, orientation,
                  activations, patients, member_checks=None):
    """One candidate-table row: identity, definition, scale, orientation, support per row set, group-score checks."""
    row = {"build": build, "reference": reference, "source": source, "seed": seed, "draw": draw,
           **definition, "scale": scale, "orientation": orientation}
    for name, activation in activations.items():
        positive = np.asarray(activation) > 0
        row[f"positive_rows_{name}"] = int(positive.sum())
        row[f"positive_patients_{name}"] = int(len(set(np.asarray(patients[name])[positive])))
    row.update(member_checks or {})
    return row


def choose_leaves(attempts):
    """Leaf count with the lowest tune MAE; scores within TIE of the best go to the fewer leaves."""
    best = min(a["tune_mae"] for a in attempts)
    return min(a["leaves"] for a in attempts if a["tune_mae"] <= best + TIE)


def retained_reduction(mae_constant, mae_tree, mae_boosting):
    """(MAE_constant - MAE_tree) / (MAE_constant - MAE_boosting); NaN when boosting does not beat the constant."""
    if not mae_constant > mae_boosting:
        return float("nan")
    return float((mae_constant - mae_tree) / (mae_constant - mae_boosting))


def tree_panel(X_fit, y_fit, X_tune, y_tune):
    """Trees of 3/5/8 leaves and the constant-median, ridge and boosting benchmarks, fitted on fit rows, scored on tune rows."""
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    fitted = {}

    def tune_mae(model, name=None):
        fitted[name] = model.fit(X_fit, y_fit)
        return float(np.abs(y_tune - fitted[name].predict(X_tune)).mean())

    min_leaf = max(10, math.ceil(0.1 * len(y_fit)))
    trees = {leaves: DecisionTreeRegressor(criterion="squared_error", max_leaf_nodes=leaves, max_depth=3,
                                           min_samples_leaf=min_leaf, random_state=SEED) for leaves in LEAF_COUNTS}
    attempts = [{"leaves": leaves, "tune_mae": tune_mae(tree, ("tree", leaves))} for leaves, tree in trees.items()]
    leaves = choose_leaves(attempts)
    mae = {"tree": next(a["tune_mae"] for a in attempts if a["leaves"] == leaves),
           "constant": tune_mae(DummyRegressor(strategy="median"), "constant"),
           "ridge": tune_mae(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), "ridge"),
           "boosting": tune_mae(HistGradientBoostingRegressor(max_iter=50, max_leaf_nodes=7, min_samples_leaf=10,
                                                              l2_regularization=1.0, early_stopping=False,
                                                              random_state=SEED), "boosting")}
    return {"tree": trees[leaves], "leaves": leaves, "attempts": attempts, "tune_mae": mae,
            "models": {"tree": trees[leaves], **{k: fitted[k] for k in ("constant", "ridge", "boosting")}},
            "retained_reduction": retained_reduction(mae["constant"], mae["tree"], mae["boosting"])}


def leaf_table(tree, feature_names):
    """One row per leaf: the conditions on its path, mean score and fit rows."""
    t = tree.tree_
    rows = []

    def walk(node, conditions):
        if t.children_left[node] == -1:
            rows.append({"conditions": " and ".join(conditions) or "all rows",
                         "mean": float(t.value[node][0][0]), "n": int(t.n_node_samples[node])})
            return
        name, cut = feature_names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conditions + [f"{name} <= {cut:.6g}"])
        walk(t.children_right[node], conditions + [f"{name} > {cut:.6g}"])

    walk(0, [])
    return rows


def f_scores(tp, fp, fn, n):
    """Precision, recall, F1, F2 and coverage from counts; precision is None with nothing selected."""
    selected, positives = tp + fp, tp + fn
    f = lambda beta2: (1 + beta2) * tp / ((1 + beta2) * tp + beta2 * fn + fp) if (1 + beta2) * tp + beta2 * fn + fp else 0.0
    return {"precision": tp / selected if selected else None, "recall": tp / positives if positives else None,
            "f1": f(1), "f2": f(4), "coverage": selected / n if n else None}


def tune_threshold(target, predicted):
    """Cutoff (strict >) maximizing F2 with precision >= 0.5 and lift >= 1.5; status 'none' when nothing is feasible."""
    target, predicted = np.asarray(target, bool), np.asarray(predicted, float)
    n, positives = len(target), int(target.sum())
    best = None
    if positives:
        unique = np.unique(predicted)
        for cut in np.r_[np.nextafter(unique[0], -np.inf), unique, np.nextafter(unique[-1], np.inf)]:
            chosen = predicted > cut
            tp = int((target & chosen).sum())
            scores = f_scores(tp, int(chosen.sum()) - tp, positives - tp, n)
            if scores["precision"] is None or scores["precision"] < MIN_PRECISION \
                    or scores["precision"] / (positives / n) < MIN_LIFT:
                continue
            if best is None or scores["f2"] > best["f2"] + TIE or (
                    abs(scores["f2"] - best["f2"]) <= TIE and (scores["precision"], cut) > (best["precision"], best["threshold"])):
                best = {**scores, "threshold": float(cut)}
    if best is None:
        return {"status": "none", "threshold": None, "f2": None}
    return {"status": "measured", **best}


# ---------------------------------------------------------------- CAV negative pools (all arrays are cav_fit rows)

def class_weights(labels, multiplicity, mass):
    labels, multiplicity = np.asarray(labels), np.asarray(multiplicity, float)
    weights = np.zeros(len(labels), float)
    for label in (0, 1):
        at = labels == label
        total = multiplicity[at].sum()
        if total:
            weights[at] = multiplicity[at] * (mass / 2) / total
    return weights


def draw_multiplicity(labels, seed, method):
    labels = np.asarray(labels, int)
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(labels == label) for label in (0, 1)]
    draws = [rng.choice(pool, len(pool), replace=True) for pool in pools]
    if method == "legacy":
        count = min(map(len, pools))
        draws = [rng.choice(draw, count, replace=False) if len(draw) > count else draw for draw in draws]
    return np.bincount(np.concatenate(draws), minlength=len(labels))


def fit_classifier(X, labels, multiplicity, mass, patients):
    """Weighted liblinear logistic fit; unit direction, raw coef and intercept, status and distinct-patient support."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    labels, patients = np.asarray(labels, int), np.asarray(patients)
    weights = class_weights(labels, multiplicity, mass)
    support = [len(np.unique(patients[(labels == k) & (weights > 0)])) for k in (0, 1)]
    if min(support) < MIN_CLASS:
        return {"status": "unsupported", "support": support}
    model = LogisticRegression(C=CAV_C, penalty="l2", solver="liblinear", class_weight=None, max_iter=1000,
                               random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, labels, sample_weight=weights)
    norm = np.linalg.norm(model.coef_[0])
    if any(issubclass(w.category, ConvergenceWarning) for w in caught) or not np.isfinite(norm) or norm <= 0:
        return {"status": "failed", "support": support}
    return {"status": "measured", "support": support, "direction": model.coef_[0] / norm,
            "coef": model.coef_[0].copy(), "intercept": float(model.intercept_[0]),
            "class_mass": [float(weights[labels == k].sum()) for k in (0, 1)]}


def fit_direction(X, labels, multiplicity, mass, patients):
    """Unit direction of fit_classifier, NaN when the fit is unsupported or failed."""
    fit = fit_classifier(X, labels, multiplicity, mass, patients)
    return fit["direction"] if fit["status"] == "measured" else np.full(X.shape[1], np.nan)


def pool_rows(rule, activation):
    """Rows of the eligible pool (rule rows + negatives at the candidate's minimum activation) and the literal pool (every row)."""
    rule, activation = np.asarray(rule, bool), np.asarray(activation, float)
    finite = activation[np.isfinite(activation)]
    cutoff = finite.min() if len(finite) else np.nan
    eligible = rule | (np.isfinite(activation) & (activation <= cutoff))
    return {"eligible": eligible, "literal": np.ones(len(rule), bool)}


def pool_multiplicities(rule, activation, seed):
    """Per-row multiplicities of legacy, eligible and literal fits for one seed (zero outside the pool).
    The three share the eligible pool's positive draw; literal negatives come from default_rng([seed, 1])."""
    rule = np.asarray(rule, bool)
    eligible = pool_rows(rule, activation)["eligible"]
    out = {}
    for method, name in (("legacy", "legacy"), ("all_negatives", "eligible")):
        out[name] = np.zeros(len(rule), int)
        out[name][eligible] = draw_multiplicity(rule[eligible].astype(int), seed, method)
    negatives = np.flatnonzero(~rule)
    drawn = np.random.default_rng([seed, 1]).choice(negatives, len(negatives), replace=True)
    out["literal"] = np.where(rule, out["eligible"], np.bincount(drawn, minlength=len(rule)))
    return out


def pool_mass(rule, patients, rows):
    """Total class mass: twice the smaller distinct-patient count of the two classes in the pool."""
    return 2 * min(len(np.unique(np.asarray(patients)[rows & (np.asarray(rule, bool) == k)])) for k in (False, True))


def fit_pool(X, rule, activation, patients, seed, pool):
    """One bootstrap fit of a pool ('legacy', 'eligible' or 'literal') on the cav_fit rows."""
    rule, patients = np.asarray(rule, bool), np.asarray(patients)
    rows = pool_rows(rule, activation)["literal" if pool == "literal" else "eligible"]
    multiplicity = pool_multiplicities(rule, activation, seed)[pool]
    return fit_classifier(X[rows], rule[rows].astype(int), multiplicity[rows], pool_mass(rule, patients, rows),
                          patients[rows])


def declared_fit(X, rule, activation, patients, pool):
    """The unresampled classifier of a pool: every row once, same weights rule and C."""
    rule, patients = np.asarray(rule, bool), np.asarray(patients)
    rows = pool_rows(rule, activation)["literal" if pool == "literal" else "eligible"]
    return fit_classifier(X[rows], rule[rows].astype(int), np.ones(rows.sum()), pool_mass(rule, patients, rows),
                          patients[rows])


# ---------------------------------------------------------------- scoring populations and readouts

def scoring_populations(rule, activation, patients, cutoff):
    """Rows scored for every pool: all rule-labelled rows, and the restricted stratum (rule rows + minimum-activation negatives)."""
    rule, activation, patients = np.asarray(rule, bool), np.asarray(activation, float), np.asarray(patients)
    rows = {"all": np.arange(len(rule)), "restricted": np.flatnonzero(rule | (np.isfinite(activation) & (activation <= cutoff)))}
    return {name: {"rows": r, "positive_patients": len(np.unique(patients[r][rule[r]])),
                   "negative_patients": len(np.unique(patients[r][~rule[r]]))} for name, r in rows.items()}


def auroc(labels, scores):
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels, int)
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None


def score_populations(populations, rule, scores):
    """AUROC and patient counts of one score vector on each population."""
    rule = np.asarray(rule, bool)
    return {name: {"auroc": auroc(rule[p["rows"]], np.asarray(scores)[p["rows"]]),
                   "positive_patients": p["positive_patients"], "negative_patients": p["negative_patients"]}
            for name, p in populations.items()}


BATCH_LOG = []


def device():
    """cuda when available, else cpu (the batched readouts run on either)."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_size(bytes_per_item, total):
    """Items per batch whose estimated bytes stay within half of the free GPU memory (all items on cpu); at least 1."""
    import torch
    size = total
    if torch.cuda.is_available():
        size = min(total, int(torch.cuda.mem_get_info()[0] // 2 // bytes_per_item))
    size = max(1, size)
    BATCH_LOG.append({"bytes_per_item": int(bytes_per_item), "total": int(total), "size": int(size)})
    return size


def weighted_auroc(labels, scores, weights):
    """Weighted AUROC (fits, draws) of score rows under weight rows, ties counted half, as sklearn does; NaN when a class has no weight."""
    import torch
    dev, f64 = device(), torch.float64
    y = torch.as_tensor(np.asarray(labels, float), dtype=f64, device=dev)
    S = torch.as_tensor(np.asarray(scores, float), dtype=f64, device=dev)
    W = torch.as_tensor(np.asarray(weights, float), dtype=f64, device=dev)
    positive, negative = W * y, W * (1 - y)
    denominator = positive.sum(1) * negative.sum(1)
    n, fits, draws = S.shape[1], S.shape[0], W.shape[0]
    out = torch.empty((fits, draws), dtype=f64, device=dev)
    step = batch_size(8 * (3 * n * n + 2 * draws * n), fits)
    for a in range(0, fits, step):
        s = S[a:a + step]
        wins = (s[:, :, None] > s[:, None, :]).to(f64) + 0.5 * (s[:, :, None] == s[:, None, :]).to(f64)
        out[a:a + step] = torch.einsum("dn,fnm,dm->fd", positive, wins, negative) / denominator
    return out.cpu().numpy()


def evaluation_angles(gradients, directions):
    """Boundary angle (degrees) of each fit to the mean unit log-odds gradient of the given rows."""
    import torch
    from tcav_geometry import coherence
    dev = device()
    mean = torch.as_tensor(coherence(gradients)["mean_direction"], dtype=torch.float64, device=dev)
    d = torch.as_tensor(np.asarray(directions, float), dtype=torch.float64, device=dev)
    return torch.rad2deg(torch.asin(torch.clamp(d @ mean / d.norm(dim=1), -1, 1))).cpu().numpy()


def mean_sensitivity(gradients, directions):
    """Mean over finite rows of gradient . direction, per fit."""
    import torch
    gradients = np.asarray(gradients, float)
    g = torch.as_tensor(gradients[np.isfinite(gradients).all(axis=1)], dtype=torch.float64, device=device())
    d = torch.as_tensor(np.asarray(directions, float), dtype=torch.float64, device=device())
    return (g @ d.T).mean(0).cpu().numpy()


def fit_cosines(directions):
    """Cosines between every pair of successful fits (unit directions)."""
    import torch
    valid = np.asarray(directions, float)
    valid = torch.as_tensor(valid[np.isfinite(valid).all(axis=1)], dtype=torch.float64, device=device())
    return (valid @ valid.T)[torch.triu_indices(len(valid), len(valid), 1, device=valid.device).unbind()].cpu().numpy()


def join_saved(seeds, new, saved_seeds, saved):
    """Cosine of each new fit with the saved fit of the same seed; failed (NaN) fits are kept, as NaN."""
    position = {s: i for i, s in enumerate(saved_seeds)}
    new, saved = np.asarray(new, float), np.asarray(saved, float)
    cosine, failed_new, failed_saved = [], [], []
    for i, seed in enumerate(seeds):
        a, b = new[i], saved[position[seed]]
        failed_new.append(not np.isfinite(a).all())
        failed_saved.append(not np.isfinite(b).all())
        ok = not (failed_new[-1] or failed_saved[-1])
        cosine.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) if ok else np.nan)
    return {"seed": np.array(seeds), "cosine": np.array(cosine), "failed_new": np.array(failed_new),
            "failed_saved": np.array(failed_saved)}


# ---------------------------------------------------------------- source comparison

BOOTSTRAP_SEEDS = tuple(range(3001, 3501))


def merge_duplicates(candidates):
    """Candidates with identical rule patient sets become one row that keeps every id and source label."""
    merged = {}
    for c in candidates:
        if c["rule_patients"] is None:
            continue
        row = merged.setdefault(frozenset(c["rule_patients"]), {"patients": set(c["rule_patients"]), "ids": [], "sources": []})
        row["ids"].append(c["id"])
        row["sources"].append(c["source"])
    return list(merged.values())


def jaccard(a, b):
    union = len(a | b)
    return len(a & b) / union if union else float("nan")


def best_jaccard(candidates):
    """Per candidate id, the highest Jaccard of its rule patient set with a candidate of another source (NaN if none)."""
    ruled = [c for c in candidates if c["rule_patients"] is not None]
    best = {c["id"]: float("nan") for c in candidates}
    for c in ruled:
        values = [jaccard(set(c["rule_patients"]), set(o["rule_patients"])) for o in ruled if o["source"] != c["source"]]
        values = [v for v in values if not np.isnan(v)]
        if values:
            best[c["id"]] = max(values)
    return best


def source_yield(results, metric):
    """Per source: candidates, those with a rule, distinct rule patient sets, and median / 5% / 95% of `metric`."""
    rows = []
    for source in dict.fromkeys(r["source"] for r in results):
        own = [r for r in results if r["source"] == source]
        ruled = [r for r in own if r["rule_patients"] is not None]
        values = np.array([r[metric] for r in ruled], float)
        values = values[np.isfinite(values)]
        q = np.quantile(values, [0.5, 0.05, 0.95]) if len(values) else [float("nan")] * 3
        rows.append({"source": source, "candidates": len(own), "with_rule": len(ruled),
                     "distinct_rule_sets": len({frozenset(r["rule_patients"]) for r in ruled}),
                     "median": float(q[0]), "q05": float(q[1]), "q95": float(q[2])})
    return rows


def patient_weights(patient_ids, seeds=BOOTSTRAP_SEEDS):
    """Row weights (draws, rows) of patient-cluster bootstrap draws: each patient's draw count, one draw per seed."""
    unique, position = np.unique(np.asarray(patient_ids).astype(str), return_inverse=True)
    counts = np.stack([np.bincount(np.random.default_rng(s).integers(0, len(unique), len(unique)), minlength=len(unique))
                       for s in seeds])
    return counts[:, position]


def bootstrap_auroc(labels, scores_by_method, weights):
    """Weighted AUROC of every method under the same patient draws (batched torch), with 2.5-97.5% intervals and paired differences."""
    methods = list(scores_by_method)
    scores = np.stack([np.asarray(scores_by_method[m], float) for m in methods])
    values = weighted_auroc(labels, scores, np.vstack([np.ones((1, scores.shape[1])), np.asarray(weights, float)]))
    draws = {m: values[i, 1:] for i, m in enumerate(methods)}
    difference = {(a, b): draws[a] - draws[b] for i, a in enumerate(methods) for b in methods[i + 1:]}
    interval = {k: tuple(float(x) for x in np.nanquantile(v, [0.025, 0.975])) for k, v in {**draws, **difference}.items()}
    point = {m: None if np.isnan(values[i, 0]) else float(values[i, 0]) for i, m in enumerate(methods)}
    return {"draws": draws, "difference": difference, "interval": interval, "point": point}


# ---------------------------------------------------------------- loading and anchors

def reference_roles(population, cache, reference, frozen_roles):
    """Original reference-year roles rebuilt with the build's split seed, checked against the frozen role identity."""
    import concept_subgroup_drift_pilot as pilot
    roles = pilot.reference_year_roles(population, reference, seed=cache.config["split_seed"])
    if roles["identity"] != frozen_roles["identity"]:
        raise ValueError("Reconstructed role identity disagrees with the frozen one")
    for name in ("evaluation", "cav_fit", "rule_discovery", "concept_selection"):
        np.testing.assert_array_equal(roles["row_role"] == name, frozen_roles["row_role"] == name)
    return roles


def load_build_two_modes(inputs, build, reference, roles):
    """The build loaded in develop mode (rule_discovery, cav_fit, concept_selection rows) and in evaluate mode (evaluation rows)."""
    from concept_study import loading
    develop_inputs = {**inputs, "ctx": {**inputs["ctx"], "mode": "develop"}}
    return {"develop": loading.load_build(develop_inputs, build, reference, roles),
            "evaluate": loading.load_build(inputs, build, reference, roles), "develop_inputs": develop_inputs}


def replay_gap(fits, saved):
    """Largest absolute gap to the saved directions on fits present in both, whether the failed-fit rows agree, and the pass at rtol 1e-4 / atol 1e-5."""
    fits, saved = np.asarray(fits, float), np.asarray(saved, float)
    failed_fits, failed_saved = np.isnan(fits).all(1), np.isnan(saved).all(1)
    same = bool(np.array_equal(failed_fits, failed_saved))
    both = ~failed_fits & ~failed_saved
    gap = float(np.abs(fits[both] - saved[both]).max()) if both.any() else 0.0
    return {"max_abs_gap": gap, "nan_pattern_equal": same,
            "passed": same and bool(np.allclose(fits[both], saved[both], rtol=1e-4, atol=1e-5))}


def sae_rule_anchor(data, index):
    """Re-run the frozen SAE path on the reference data and compare with the frozen families, rules and masks.
    Returns the anchor row with the SAE concepts, summary and sweep for the caller."""
    import concept_subgroup_drift_pilot as pilot
    from concept_study import alternative_bases as ab, bootstrap, measures
    cap = pilot.candidate_cap(ab.RANDOM_LATENTS, data["cache"].config["candidate_cap_fraction"])
    prefix = f"{data['reference']}/{data['cache'].config['split_seed']}"
    concepts, summary, sweep = ab.basis_concepts("sae", 0, data, ab.sae_encoders(data), cap, prefix)
    row_weights = bootstrap.row_weights(data["table"]["patient_id"], *bootstrap.patient_weights(measures.all_patient_ids(index)))
    for c in concepts:
        c["agreement"] = ab.agreement_replicates(c["in_A"], c["in_R"], data["table"]["year"], row_weights)
    return ab.sae_anchor(data, concepts, summary, sweep, index), concepts, summary, sweep


def tree_anchor(new, selected_trees_csv):
    """Descriptive: how many new tree selections (concept -> model name) equal the earlier selected-trees.csv."""
    import csv
    with open(selected_trees_csv, newline="") as handle:
        old = {row["concept"]: row["model"] for row in csv.DictReader(handle)}
    compared = [c for c in new if c in old]
    equal = sum(new[c] == old[c] for c in compared)
    return {"compared": len(compared), "equal": equal, "different": len(compared) - equal,
            "missing_in_selected": len(new) - len(compared)}


def freeze_selection_aurocs(unit, estimator, C=CAV_C):
    """Selection-row AUROC of each saved fit of one concept, keyed by seed (None for fits without one)."""
    fits = unit["joint"][(estimator, C)]
    discrimination = unit["scores"][(estimator, C)]["discrimination"]
    return {fit["seed"]: d["auroc"] if d["status"] == "measured" else None for fit, d in zip(fits, discrimination)}


# ---------------------------------------------------------------- candidate sources (notebook helpers)

def family_members(families):
    """{family id: [(seed, factor), ...]} of the full families."""
    return {f["id"]: [(int(s), int(k)) for s, k in f["members"]] for f in families if f["kind"] == "full"}


def group_encoder(encoders, fit_codes, members):
    """Encoder of raw rows to one group score per family (members scaled by their SD on the SAE-training rows).
    Families without a usable member are dropped and returned by id; SD-0 members are left out of the mean."""
    info, dropped = [], []
    for family, pairs in members.items():
        fit = np.stack([fit_codes[s][:, f] for s, f in pairs], axis=1)
        result = group_score(fit, fit)
        if result["score"] is None:
            dropped.append(family)
        else:
            info.append({"id": family, "members": pairs, "scale": result["scale"], "used": result["used"],
                         "left_out": result["left_out"]})

    def encode(raw):
        codes = {s: encoders[s](raw) for s in encoders}
        return np.stack([(np.stack([codes[s][:, f] for s, f in i["members"]], 1)[:, i["used"]] / i["scale"][i["used"]]).mean(1)
                         for i in info], axis=1)
    return encode, info, dropped


def top_columns(matrix, cap):
    """Indices of the `cap` columns with most positive rows (ties keep column order)."""
    return np.argsort(-(np.asarray(matrix) > 0).sum(0), kind="stable")[:cap]


def columns_encoder(source_encoders, picks):
    """Encoder stacking chosen (source name, column) pairs; each source is encoded once per call."""
    def encode(raw):
        encoded = {name: source_encoders[name](raw) for name in dict.fromkeys(n for n, _ in picks)}
        return np.stack([encoded[name][:, column] for name, column in picks], axis=1)
    return encode


def pca_axes(fit_z, count):
    """Mean and leading axes behind pca_candidates (for export)."""
    mean = np.asarray(fit_z).mean(0)
    return mean, np.linalg.svd(fit_z - mean, full_matrices=False)[2][:(count + 1) // 2]


def random_directions(dimension, latents, seed):
    """The random unit directions behind alternative_bases.random_encoder (for export)."""
    directions = np.random.default_rng(seed).standard_normal((latents, dimension))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def factor_concepts(basis, draw, data, encode, ids):
    """Rules of a single-encoder source (every column a candidate) through the frozen sweep, as basis_concepts does."""
    from semantic_rules import RuleSet
    from concept_study import alternative_bases as ab
    codes = encode(data["match_emb"])
    candidates = [{"rank": j + 1, "family_id": ids[j], "medoid": [0, j], "positive_count": int((codes[:, j] > 0).sum())}
                  for j in range(codes.shape[1])]
    sweep = ab.rule_sweep(data, candidates, {0: codes})
    evaluation = encode(data["eval_emb"])
    concepts = [ab.concept_record(basis, draw, data, e["family_id"], RuleSet.from_dict(e["rule_set"]), e["cutoff"],
                                  codes[data["rule_pos"], e["medoid"][1]], evaluation[:, e["medoid"][1]], e)
                for e in sweep["frozen"]]
    attach_sweep(concepts, sweep)
    return concepts, candidates, sweep


def pool_concepts(pool_meta, concepts_by_source):
    """Pool candidates and the concepts of those with a rule, taken from the sources' own sweeps.
    The sweep treats every candidate alone (own inner split, own backend seed), so a pool member keeps its source's rule, cutoff, masks and split."""
    candidates, concepts = [], []
    for j, m in enumerate(pool_meta):
        pooled = f"{m['source']}:{m['id']}"
        candidates.append({"rank": j + 1, "family_id": pooled, "medoid": [0, j], "positive_count": m["positive_count"]})
        source = concepts_by_source[m["source"]].get(m["id"])
        if source is not None:
            concepts.append({**source, "basis": "pool", "draw": 0, "family_id": pooled, "medoid": [0, j],
                             "concept": f"pool0/{source['build']}/{source['reference']}/{pooled}"})
    return concepts, candidates


def attach_sweep(concepts, sweep):
    """Record each concept's medoid (seed, factor) and inner patient split from its frozen sweep entry."""
    if len(concepts) != len(sweep["frozen"]):
        raise ValueError("Concepts and frozen sweep entries differ in number")
    for c, e in zip(concepts, sweep["frozen"]):
        c["medoid"], c["inner_split"] = e["medoid"], e["inner_split"]


# ---------------------------------------------------------------- fits and anchors

def _fit_task(X, rule, activation, patients, pool, seeds):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return [declared_fit(X, rule, activation, patients, pool) if s is None else fit_pool(X, rule, activation, patients, s, pool)
                for s in seeds]


def fit_candidates(X, candidates, seeds, workers):
    """Every liblinear fit of a build in one joblib call (`workers` processes, one BLAS thread each).
    candidates: [{"rule", "activation", "patients"}]; returns per candidate {pool: fits for the seeds, then the declared fit}."""
    from joblib import Parallel, delayed
    seeds = list(seeds) + [None]
    halves = [[seeds[i] for i in part] for part in np.array_split(np.arange(len(seeds)), 2)]
    jobs = [(i, pool, half) for i in range(len(candidates)) for pool in POOLS for half in halves]
    parts = Parallel(n_jobs=workers)(delayed(_fit_task)(X, candidates[i]["rule"], candidates[i]["activation"],
                                                        candidates[i]["patients"], pool, half) for i, pool, half in jobs)
    out = [{pool: [] for pool in POOLS} for _ in candidates]
    for (i, pool, _), part in zip(jobs, parts):
        out[i][pool] += part
    return out


def replay_concepts(X, candidates, saved, seeds, workers):
    """Per candidate, replay_gap of the legacy / eligible refits against its saved directions ({pool: (seeds, d)})."""
    d = X.shape[1]
    fitted = fit_candidates(X, candidates, seeds, workers)
    return [{pool: replay_gap(np.stack([f["direction"] if f["status"] == "measured" else np.full(d, np.nan) for f in fits[pool][:-1]]), matrix)
             for pool, matrix in own.items()} for fits, own in zip(fitted, saved)]


def anchors_pass(anchor, replays):
    """True when the SAE rules, masks, agreement and every direction replay match exactly."""
    return bool(anchor["rule_identities_equal"] and anchor["family_counts_equal"] and anchor["candidates_equal"]
                and anchor["concepts_missing_from_index"] == 0 and anchor["A_or_R_row_mismatches"] == 0
                and anchor["agreement_max_abs_gap"] == 0 and all(r["passed"] for rep in replays for r in rep.values()))


# ---------------------------------------------------------------- descriptions

def _scores(high, decision, prefix, suffix=""):
    high, decision = np.asarray(high, bool), np.asarray(decision, bool)
    tp = int((high & decision).sum())
    fp, fn = int((~high & decision).sum()), int((high & ~decision).sum())
    scores = f_scores(tp, fp, fn, len(high))
    row = {f"{prefix}{k}{suffix}": float("nan") if v is None else v for k, v in scores.items()}
    return {f"{prefix}tp{suffix}": tp, f"{prefix}fp{suffix}": fp, f"{prefix}fn{suffix}": fn, **row}


def describe_candidate(X_fit, y_fit, X_tune, y_tune, X_eval, y_eval, cutoff, in_rule):
    """Graded tree (fit/tune rows) scored on evaluation rows next to the rule, as continuous and binary fidelity.
    Returns (row, panel); an unsupported candidate has panel None."""
    if len(y_fit) < 2 or len(y_tune) < 1:
        return {"status": "unsupported"}, None
    panel = tree_panel(X_fit, y_fit, X_tune, y_tune)
    predicted = {name: model.predict(X_eval) for name, model in panel["models"].items()}
    high = np.asarray(y_eval) > cutoff
    mae = {name: float(np.abs(y_eval - p).mean()) for name, p in predicted.items()}
    tree_high = float(np.abs(y_eval[high] - predicted["tree"][high]).mean()) if high.any() else float("nan")
    tuned = tune_threshold(np.asarray(y_tune) > cutoff, panel["tree"].predict(X_tune))
    row = {"status": "measured", "leaves": panel["leaves"], "threshold_status": tuned["status"],
           "tuned_threshold": float("nan") if tuned["threshold"] is None else tuned["threshold"],
           "retained_reduction_tune": panel["retained_reduction"],
           **{f"tune_mae_{k}": v for k, v in panel["tune_mae"].items()},
           **{f"{k}_mae": v for k, v in mae.items()}, "tree_mae_high": tree_high,
           "retained_reduction": retained_reduction(mae["constant"], mae["tree"], mae["boosting"]),
           **_scores(high, in_rule, "rule_"), **_scores(high, predicted["tree"] > cutoff, "tree_", "_cutoff")}
    if tuned["threshold"] is not None:
        row.update(_scores(high, predicted["tree"] > tuned["threshold"], "tree_", "_tuned"))
    else:
        row.update({k.replace("_cutoff", "_tuned"): float("nan") for k in row if k.startswith("tree_") and k.endswith("_cutoff")})
    return row, panel


def interval(values):
    finite = np.asarray(values, float)[np.isfinite(values)]
    low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) else (float("nan"),) * 2
    return {"low": float(low), "high": float(high), "undefined_fraction": float(1 - len(finite) / len(values))}


def fidelity_intervals(weights, truth, tree, constant, boosting, in_A, in_rule, cutoff):
    """Patient-bootstrap intervals (draws paired) of rule F1, tree F1 at the cutoff, their difference and the retained reduction."""
    A = np.asarray(in_A, bool)

    def f1(decision):
        d = np.asarray(decision, bool)
        tp, fp, fn = weights @ (A & d), weights @ (~A & d), weights @ (A & ~d)
        denominator = 2 * tp + fp + fn
        return np.divide(2 * tp, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)

    def mae(prediction):
        return weights @ np.abs(truth - prediction) / weights.sum(1)

    rule, tree_f1 = f1(in_rule), f1(np.asarray(tree) > cutoff)
    const, boost = mae(constant), mae(boosting)
    gain = const - boost
    retained = np.divide(const - mae(tree), gain, out=np.full(len(weights), np.nan), where=gain > 0)
    return {"rule_f1": interval(rule), "tree_f1": interval(tree_f1), "f1_difference": interval(rule - tree_f1),
            "retained_reduction": interval(retained)}


# ---------------------------------------------------------------- CAV pools over candidates

POOLS = ("legacy", "eligible", "literal")


def pool_results(fitted, seeds, X, rule, activation, patients, sets):
    """Readouts of one candidate's fits ({pool: fits for `seeds`, declared last}) on each scoring set
    ({name: X, rule, activation, patients, gradients}): AUROC on both populations, boundary angle and R sensitivity, batched over the fits."""
    import torch
    finite = np.asarray(activation, float)[np.isfinite(activation)]
    cutoff = float(finite.min())
    populations = {name: scoring_populations(s["rule"], s["activation"], s["patients"], cutoff) for name, s in sets.items()}
    flat = [(pool, seed, fit) for pool in POOLS for seed, fit in zip([*seeds, None], fitted[pool])]
    measured = [i for i, (_, _, fit) in enumerate(flat) if fit["status"] == "measured"]
    rows = [{"status": fit["status"], "support_0": fit["support"][0], "support_1": fit["support"][1]} for _, _, fit in flat]
    if measured:
        coef = np.stack([flat[i][2]["coef"] for i in measured])
        intercept = np.array([flat[i][2]["intercept"] for i in measured])
        direction = np.stack([flat[i][2]["direction"] for i in measured])
    for name, s in sets.items():
        in_rule = np.asarray(s["rule"], bool)
        values = {key: np.full(len(flat), np.nan) for key in ("auroc_all", "auroc_restricted", "angle", "sensitivity")}
        if measured:
            dev = device()
            scores = (torch.as_tensor(coef, device=dev) @ torch.as_tensor(np.asarray(s["X"], float), device=dev).T
                      + torch.as_tensor(intercept, device=dev)[:, None]).cpu().numpy()
            masks = np.stack([np.isin(np.arange(len(in_rule)), populations[name][p]["rows"]) for p in ("all", "restricted")]).astype(float)
            auc = weighted_auroc(in_rule, scores, masks)
            values["auroc_all"][measured], values["auroc_restricted"][measured] = auc[:, 0], auc[:, 1]
            values["angle"][measured] = evaluation_angles(s["gradients"], direction)
            if in_rule.any():
                values["sensitivity"][measured] = mean_sensitivity(s["gradients"][in_rule], direction)
        for key, column in values.items():
            for row, value in zip(rows, column):
                row[f"{key}_{name}"] = float(value)
    d = X.shape[1]
    out = {"populations": populations, "cutoff": cutoff, "directions": {}, "coefs": {}, "intercepts": {}, "statuses": {},
           "cosines": {}, "declared": {}, "fits": [{"pool": pool, "seed": seed, **row} for (pool, seed, _), row in zip(flat, rows) if seed is not None]}
    for pool in POOLS:
        fits = fitted[pool][:-1]
        out["directions"][pool] = np.stack([f["direction"] if f["status"] == "measured" else np.full(d, np.nan) for f in fits])
        out["coefs"][pool] = np.stack([f["coef"] if f["status"] == "measured" else np.full(d, np.nan) for f in fits])
        out["intercepts"][pool] = np.array([f["intercept"] if f["status"] == "measured" else np.nan for f in fits])
        out["statuses"][pool] = [f["status"] for f in fits]
        out["cosines"][pool] = fit_cosines(out["directions"][pool])
        declared = next(row for (p, seed, _), row in zip(flat, rows) if p == pool and seed is None)
        out["declared"][pool] = {"fit": fitted[pool][-1], "readouts": declared}
    return out


def run_pools(X, rule, activation, patients, sets, seeds, workers):
    """One candidate end to end: its fits (one joblib call) and their readouts."""
    seeds = list(seeds)
    fitted = fit_candidates(X, [{"rule": rule, "activation": activation, "patients": patients}], seeds, workers)[0]
    return pool_results(fitted, seeds, X, rule, activation, patients, sets)


def source_pool_summary(per_concept, cosines=None):
    """Per panel, source and pool: concept count and medians over concepts of the per-concept median-fit AUROC (both populations),
    boundary angle and fit cosine. Each random draw keeps its own rows, with one more pooled row ('random (all draws)') as a summary;
    matched-count pool members (copies of their sources) are left out."""
    import pandas as pd
    frame = per_concept[per_concept.source != "pool"]
    pooled = frame[frame.source.str.startswith("random")].assign(source="random (all draws)")
    frame = pd.concat([frame, pooled], ignore_index=True)
    if cosines is not None:
        frame = frame.merge(cosines[["concept", "pool", "cosine_median"]], on=["concept", "pool"], how="left")
    columns = {"concepts": ("concept", "nunique"), "median_auroc_all": ("auroc_all", "median"),
               "median_auroc_restricted": ("auroc_restricted", "median"), "median_angle": ("angle_median", "median"),
               "median_angle_p05": ("angle_p05", "median"), "median_angle_p95": ("angle_p95", "median"),
               "median_measured_share": ("measured_share", "median")}
    if cosines is not None:
        columns["median_fit_cosine"] = ("cosine_median", "median")
    return frame.groupby(["panel", "source", "pool"]).agg(**columns).reset_index()


def cosine_table(classifiers):
    """Per concept and pool, cosines between the measured bootstrap fits: pair count, median, 10% quantile, minimum.
    classifiers: {concept: {'<pool>|coefs': (fits, d), '<pool>|statuses': [...]}}."""
    import pandas as pd
    rows = []
    for concept, item in classifiers.items():
        for pool in POOLS:
            coefs = np.asarray(item[f"{pool}|coefs"], float)[np.asarray(item[f"{pool}|statuses"]) == "measured"]
            cos = fit_cosines(coefs / np.linalg.norm(coefs, axis=1, keepdims=True)) if len(coefs) else np.array([])
            rows.append({"concept": concept, "pool": pool, "pairs": len(cos),
                         "cosine_median": float(np.median(cos)) if len(cos) else np.nan,
                         "cosine_q10": float(np.quantile(cos, 0.1)) if len(cos) else np.nan,
                         "cosine_min": float(np.min(cos)) if len(cos) else np.nan})
    return pd.DataFrame(rows)


def classifiers_from_npz(path):
    """The bootstrap coefficients and statuses of cav_classifiers.npz, as cosine_table reads them."""
    out = {}
    with np.load(path) as z:
        for name in z.files:
            if name.endswith("|coef"):
                concept, pool, _ = name.split("|")
                item = out.setdefault(concept.replace("~", "/"), {})
                item[f"{pool}|coefs"], item[f"{pool}|statuses"] = z[name], z[name[:-4] + "status"]
    return out


def write_source_tables(directory, classifiers):
    """Write fit_cosines_by_concept.csv and pool_summary_by_source_{development,evaluation}.csv next to the per-concept tables in `directory`."""
    import pandas as pd
    from pathlib import Path
    directory = Path(directory)
    cosines = cosine_table(classifiers)
    cosines.to_csv(directory / "fit_cosines_by_concept.csv", index=False)
    for name in ("development", "evaluation"):
        source_pool_summary(pd.read_csv(directory / f"{name}_by_concept.csv"), cosines).to_csv(
            directory / f"pool_summary_by_source_{name}.csv", index=False)
