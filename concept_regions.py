"""Linear and region probes for the concept-region comparison: fits, the choice of settings on
selection rows, and a file hash for frozen inputs.

Probes are fitted on cav_fit rows of the reference year. Pools, class weights and support rules are
the ones in concept_candidates (cc); nothing here reads evaluation or later-year rows.
"""
import hashlib
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

import concept_candidates as cc

PREPROCESSING = ("raw", "standardised")
C_GRID = (0.01, 0.1, 1, 10, 100)
WIDTH_MULTIPLIERS = (0.25, 0.5, 1, 2, 4)
FIT_ROLES = ("cav_fit",)
REGION_ANCHOR = (PREPROCESSING.index("raw"), C_GRID.index(1), WIDTH_MULTIPLIERS.index(1))
LINEAR_ANCHOR = (PREPROCESSING.index("raw"), C_GRID.index(cc.CAV_C))


def sha256(path):
    """SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gamma_scale(X):
    """sklearn's gamma="scale": 1 / (number of features x variance of all entries)."""
    variance = np.asarray(X, float).var()
    if variance == 0:
        raise ValueError("gamma='scale' is undefined for input with zero variance")
    return 1.0 / (X.shape[1] * variance)


def preprocessed(X, preprocessing):
    """Rows as the classifier sees them. The scaler is fitted on the rows it is given."""
    return StandardScaler().fit_transform(X) if preprocessing == "standardised" else X


def probe_pipeline(preprocessing, classifier):
    """The standardiser (standardised only) followed by the classifier."""
    steps = [("scale", StandardScaler())] if preprocessing == "standardised" else []
    return Pipeline([*steps, ("classifier", classifier)])


def pool_inputs(X, rule, activation, patients, pool):
    """Rows of one pool, their labels, class weights, and distinct-patient support per class (negative, positive)."""
    rule, patients = np.asarray(rule, bool), np.asarray(patients)
    rows = cc.pool_rows(rule, activation)[pool]
    labels = rule[rows].astype(int)
    weights = cc.class_weights(labels, np.ones(len(labels)), cc.pool_mass(rule, patients, rows))
    support = [len(np.unique(patients[rows][labels == k])) for k in (0, 1)]
    return {"X": np.asarray(X)[rows], "labels": labels, "weights": weights, "support": support}


def selection_auroc(probe, X, rule):
    """AUROC of a probe's decision scores on rule-labelled rows; NaN when a class is absent."""
    value = cc.auroc(rule, probe.decision_function(X))
    return np.nan if value is None else value


def fit_checked(model, inputs):
    """Weighted fit of the pool; any warning marks the fit failed. Returns the status."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(inputs["X"], inputs["labels"], classifier__sample_weight=inputs["weights"])
    return "failed" if caught else "measured"


def fit_region(X, rule, activation, patients, pool, preprocessing, C, m):
    """RBF support vector classifier on one pool, with class weights and gamma = m x gamma_scale of the preprocessed rows."""
    if preprocessing not in PREPROCESSING:
        raise ValueError(f"preprocessing must be one of {PREPROCESSING}, got {preprocessing!r}")
    inputs = pool_inputs(X, rule, activation, patients, pool)
    if min(inputs["support"]) < cc.MIN_CLASS:
        return {**inputs, "status": "unsupported", "pipeline": None, "support_vectors": None}
    gamma = m * gamma_scale(preprocessed(inputs["X"], preprocessing))
    model = probe_pipeline(preprocessing, SVC(kernel="rbf", C=C, gamma=gamma))
    status = fit_checked(model, inputs)
    if status != "measured":
        return {**inputs, "status": status, "pipeline": None, "support_vectors": None}
    return {**inputs, "status": status, "pipeline": model,
            "support_vectors": int(model.named_steps["classifier"].n_support_.sum())}


def fit_linear(X, rule, activation, patients, pool, preprocessing, C):
    """Logistic probe with the Work 3 liblinear settings and the same class weights, at regularisation C."""
    if preprocessing not in PREPROCESSING:
        raise ValueError(f"preprocessing must be one of {PREPROCESSING}, got {preprocessing!r}")
    inputs = pool_inputs(X, rule, activation, patients, pool)
    if min(inputs["support"]) < cc.MIN_CLASS:
        return {**inputs, "status": "unsupported", "pipeline": None, "support_vectors": None}
    classifier = LogisticRegression(C=C, penalty="l2", solver="liblinear", max_iter=1000, random_state=cc.SEED)
    model = probe_pipeline(preprocessing, classifier)
    status = fit_checked(model, inputs)
    if status != "measured":
        return {**inputs, "status": status, "pipeline": None, "support_vectors": None}
    return {**inputs, "status": status, "pipeline": model, "support_vectors": None}


def fit_concept_task(*arguments):
    """fit_concept with one BLAS thread, for one joblib worker."""
    with threadpool_limits(1):
        return fit_concept(*arguments)


def grid_distance(position, anchor):
    """Steps in C and width from the anchor, plus one when the preprocessing differs from the anchor's."""
    return sum(abs(a - b) for a, b in zip(position[1:], anchor[1:])) + (position[0] != anchor[0])


def choose_setting(settings, anchor):
    """Setting with the highest selection AUROC among the measured ones. An exact tie goes to the anchor, else to the
    nearest grid setting, then raw before standardised, smaller C and smaller width. When no selection AUROC is
    defined, the anchor is used and labelled. Failed settings are excluded and counted."""
    measured = [s for s in settings if s["status"] == "measured"]
    excluded = sum(s["status"] == "failed" for s in settings)
    defined = [s for s in measured if not np.isnan(s["selection_auroc"])]
    if not defined:
        default = next((s for s in measured if s["position"] == anchor), None)
        if not measured:
            rule = "no measured setting"
        elif default is None:
            rule = "no default: anchor not measured"
        else:
            rule = "default: selection undefined"
        return {"setting": default, "rule": rule, "excluded": excluded, "tied": 0}
    best = max(s["selection_auroc"] for s in defined)
    tied = [s for s in defined if s["selection_auroc"] == best]
    anchor_setting = next((s for s in tied if s["position"] == anchor), None)
    if len(tied) == 1:
        chosen, rule = tied[0], "max"
    elif anchor_setting is not None:
        chosen, rule = anchor_setting, "tie: anchor"
    else:
        chosen = min(tied, key=lambda s: (grid_distance(s["position"], anchor), s["position"]))
        rule = "tie: nearest grid setting"
    return {"setting": chosen, "rule": rule, "excluded": excluded, "tied": len(tied)}


def setting_row(probe, position, preprocessing, C, width, fit, X_selection, rule_selection):
    """One setting's record: fit status, support, zero-variance features, fit and selection AUROC. The pipeline is
    kept for the choice; it is None unless the fit was measured."""
    pipeline = fit["pipeline"]
    measured = pipeline is not None
    return {"probe": probe, "position": position, "preprocessing": preprocessing, "C": C, "width_multiplier": width,
            "status": fit["status"], "support_negative": fit["support"][0],
            "support_positive": fit["support"][1], "zero_variance_features": int((fit["X"].var(0) == 0).sum()),
            "support_vectors": fit["support_vectors"],
            "fit_auroc": selection_auroc(pipeline, fit["X"], fit["labels"]) if measured else np.nan,
            "selection_auroc": selection_auroc(pipeline, X_selection, rule_selection) if measured else np.nan,
            "pipeline": pipeline}


def tuning_row(setting, chosen_region, chosen_linear):
    """A setting's row for the tuning table, without the fitted pipeline."""
    row = {key: value for key, value in setting.items() if key != "pipeline"}
    row["chosen"] = setting is chosen_region or setting is chosen_linear
    return row


def fit_concept(X_fit, rule_fit, activation_fit, patients_fit, X_selection, rule_selection, pool):
    """All region and linear settings of one concept and pool: fitted on cav_fit rows, scored on
    selection rows, one setting chosen per probe. The fixed linear probe is the raw setting at the Work 3 C."""
    settings = {"region": [], "linear": []}
    for preprocessing_index, preprocessing in enumerate(PREPROCESSING):
        for c_index, C in enumerate(C_GRID):
            linear_fit = fit_linear(X_fit, rule_fit, activation_fit, patients_fit, pool, preprocessing, C)
            settings["linear"].append(setting_row("linear", (preprocessing_index, c_index), preprocessing, C,
                                                  None, linear_fit, X_selection, rule_selection))
            for m_index, m in enumerate(WIDTH_MULTIPLIERS):
                region_fit = fit_region(X_fit, rule_fit, activation_fit, patients_fit, pool, preprocessing, C, m)
                settings["region"].append(setting_row("region", (preprocessing_index, c_index, m_index), preprocessing,
                                                      C, m, region_fit, X_selection, rule_selection))
    chosen_region = choose_setting(settings["region"], REGION_ANCHOR)
    chosen_linear = choose_setting(settings["linear"], LINEAR_ANCHOR)
    tuning = [tuning_row(s, chosen_region["setting"], chosen_linear["setting"])
              for s in settings["region"] + settings["linear"]]
    fixed = next(s for s in settings["linear"] if s["position"] == LINEAR_ANCHOR)
    return {"tuning": tuning, "region": chosen_region, "linear": chosen_linear, "fixed_linear": fixed["pipeline"]}


QUANTILE_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
POSITIONS = ("above 0", "below 0", "spans 0", "identical", "undefined")
CHANCE_SIDE = 0.025


def ratio(numerator, denominator):
    """Elementwise ratio, NaN where the denominator is zero: an empty set is undefined, not zero."""
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    return np.divide(numerator, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)


def decision_scores(probe, X):
    """Decision scores of a frozen probe: X @ coef + intercept for a saved linear probe (coef, intercept),
    otherwise the fitted pipeline's decision_function."""
    if isinstance(probe, tuple):
        coef, intercept = probe
        return np.asarray(X, float) @ coef + intercept
    return probe.decision_function(X)


def draw_counts_for_rows(count_matrix, population_patient_ids, row_patient_ids):
    """Patient draw counts of each row: count_matrix (draws x sorted unique population patients) read by patient position."""
    patients = np.unique(np.asarray(population_patient_ids).astype(str))
    count_matrix = np.asarray(count_matrix)
    if count_matrix.shape[1] != len(patients):
        raise ValueError("count matrix columns do not match the population's patients")
    rows = np.asarray(row_patient_ids).astype(str)
    position = np.searchsorted(patients, rows)
    found = patients[np.minimum(position, len(patients) - 1)] == rows
    if not found.all():
        raise ValueError("row patients are not in the population")
    return count_matrix[:, position]


def recall_by_draw(labels, scores, weights):
    """Weighted recall at score 0 for each draw: weight of rule rows scored above 0 over the weight of rule rows."""
    labels, member = np.asarray(labels, bool), np.asarray(scores, float) > 0
    weights = np.asarray(weights, float)
    return ratio(weights @ (member & labels).astype(float), weights @ labels.astype(float))


def membership(labels, scores, weights, in_A):
    """Weighted members (score above 0), recall, precision P(rule | member) and P(A | member) for each draw."""
    labels, in_A, member = np.asarray(labels, bool), np.asarray(in_A, bool), np.asarray(scores, float) > 0
    weights = np.asarray(weights, float)
    members = weights @ member.astype(float)
    return {"members": members,
            "recall": recall_by_draw(labels, scores, weights),
            "precision": ratio(weights @ (member & labels).astype(float), members),
            "p_a_given_member": ratio(weights @ (member & in_A).astype(float), members)}


def members_jaccard(scores_a, scores_b, weights):
    """Weighted Jaccard of the two probes' members (score above 0) for each draw; NaN when neither has members."""
    member_a, member_b = np.asarray(scores_a, float) > 0, np.asarray(scores_b, float) > 0
    weights = np.asarray(weights, float)
    return ratio(weights @ (member_a & member_b).astype(float), weights @ (member_a | member_b).astype(float))


def score_quantiles(scores, labels):
    """Unweighted score quantiles (5, 25, 50, 75, 95%) of rule rows and of other rows; NaN for an empty group."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    groups = (("in rule", scores[labels]), ("not in rule", scores[~labels]))
    return {name: np.quantile(group, QUANTILE_LEVELS) if len(group) else np.full(len(QUANTILE_LEVELS), np.nan)
            for name, group in groups}


def concept_positions(differences):
    """Position of each concept's 2.5-97.5% interval of region minus linear: above 0, below 0 or spans 0.
    A value identical in every draw is 'identical'; no defined draw is 'undefined'."""
    positions = []
    for row in np.asarray(differences, float):
        finite = row[np.isfinite(row)]
        if not len(finite):
            positions.append("undefined")
        elif np.ptp(finite) == 0:
            positions.append("identical")
        else:
            bounds = cc.interval(row)
            if bounds["low"] > 0:
                positions.append("above 0")
            elif bounds["high"] < 0:
                positions.append("below 0")
            else:
                positions.append("spans 0")
    return positions


def position_counts(differences):
    """Concepts in each position of concept_positions, and the count chance alone gives on each side of 0: CHANCE_SIDE
    of the non-degenerate concepts (those that are neither identical nor undefined)."""
    positions = concept_positions(differences)
    counts = {position: positions.count(position) for position in POSITIONS}
    nondegenerate = counts["above 0"] + counts["below 0"] + counts["spans 0"]
    return {**counts, "chance each side": CHANCE_SIDE * nondegenerate}


def changes_from_reference(reference, later):
    """Change from the reference year per probe, and the region change minus the linear change, draw by draw."""
    change = {probe: np.asarray(later[probe], float) - np.asarray(reference[probe], float) for probe in reference}
    change["region - linear"] = change["region"] - change["linear"]
    return change


def coverage_row(build, year, concept, pool, population_name, population, probe_status):
    """Coverage of one concept, pool, year and scoring population (from cc.scoring_populations), with each probe's
    fit status. An unsupported concept keeps its row, with its status and no AUROC."""
    return {"build": build, "year": year, "concept": concept, "pool": pool, "population": population_name,
            "rows": len(population["rows"]), "positive_patients": population["positive_patients"],
            "negative_patients": population["negative_patients"],
            **{f"{probe}_status": status for probe, status in probe_status.items()}}


def scoring_rows(row_role, year, years):
    """Rows scored in the given years: evaluation-role rows only. Rule-discovery and fitting rows never score."""
    return (np.asarray(row_role) == "evaluation") & np.isin(np.asarray(year), years)


def row_mask(rows, group="sae", build=None, reference=None, panel=None, years_since=None, stage=None):
    """Rows of the concept table (a DataFrame) that match every filter given. Build names repeat across references,
    so one build is picked by build and reference together; without reference, all references of that name match."""
    mask = (rows["group"] == group).to_numpy()
    filters = {"build": build, "reference": reference, "panel": panel,
               "years_since_reference": years_since, "stage": stage}
    for column, value in filters.items():
        if value is not None:
            mask &= (rows[column] == value).to_numpy()
    return mask


def agreement_given_rule(labels, in_A, weights):
    """Weighted P(A | R) for each draw: weight of rule rows also in A over the weight of rule rows; NaN when R is empty."""
    labels, in_A = np.asarray(labels, bool), np.asarray(in_A, bool)
    weights = np.asarray(weights, float)
    return ratio(weights @ (labels & in_A).astype(float), weights @ labels.astype(float))


def restricted_cutoff(activation):
    """Minimum finite activation of the cav_fit rows: negatives at or below it form the restricted stratum. NaN when none is finite."""
    finite = np.asarray(activation, float)
    finite = finite[np.isfinite(finite)]
    return float(finite.min()) if len(finite) else np.nan


def declared_matches(declared, saved):
    """True when a refit of a declared classifier reproduces the saved status, support and, if measured, coef and intercept exactly."""
    if declared["status"] != saved["status"] or list(declared["support"]) != list(saved["support"]):
        return False
    if saved["status"] != "measured":
        return True
    return bool(np.array_equal(declared["coef"], saved["coef"]) and declared["intercept"] == saved["intercept"])


def median_by_draw(values):
    """Median over concepts (rows) of each draw (columns), skipping undefined values; returns the median, the defined
    count and the undefined fraction of each draw. A draw with no defined value has a NaN median."""
    values = np.asarray(values, float)
    defined = np.isfinite(values)
    median = np.full(values.shape[1], np.nan)
    has_value = defined.any(axis=0)
    median[has_value] = np.nanmedian(values[:, has_value], axis=0)
    return {"median": median, "defined": defined.sum(axis=0), "undefined_fraction": 1 - defined.mean(axis=0)}

