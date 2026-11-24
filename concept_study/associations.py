"""Concept-year panel, rank correlations (raw and partial) and exact-horizon joins.

Every function works on a replicate axis: row 0 is the point estimate, rows 1..B the
joint bootstrap replicates. Pooled correlations (correlation, associate, grid) also resample
concepts in each replicate; within-concept correlations (spearman, within_concept) resample patients only.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from concept_study import bootstrap, features, measures, paths

PERFORMANCE = ("brier", "brier_skill", "death_f1", "mortality", "n", "deaths", "mean_risk", "calibration_gap")
AGREEMENT = ("agreement", "rule_f2", "overlap", "rule_recall")
REPRESENTATION = ("activation_rate", "mean_activation", "profile_shift", "reconstruction_error")
INFLUENCE = ("signed_sensitivity", "absolute_sensitivity", "tcav", "share_fits_positive")
REMOVAL = ("removal_brier_change", "removal_risk_change", "removal_brier_outcome_free", "removal_brier_death_part")
ADJUSTMENTS = {
    "none": [],
    "case_mix": ["mortality", "log_n"],
    "within_concept": ["concept"],
    "within_predictor_year": ["predictor_year"],
    "full": ["concept", "predictor_year", "mortality", "log_n"],
    "full_and_risk": ["concept", "predictor_year", "mortality", "log_n", "mean_risk"],
    "within_concept_and_risk": ["concept", "mean_risk"],
}


# ---------------------------------------------------------------- panel

def _influence_name(name: str, estimator: str) -> str:
    if estimator == paths.PRIMARY_ESTIMATOR:
        return name
    return f"{name}_{'logit' if estimator == paths.LOGIT_ESTIMATOR else estimator}"


def build_panel(records: list, groups: tuple = ("A", "R")) -> tuple:
    """Return (point DataFrame, {column: (replicates, rows) array}) for every concept-year-group.

    Rows exist from the reference year to 2015; performance columns are NaN after 2014.
    """
    frames, cubes = [], {}
    for record in records:
        data = measures.load_replicates(record)
        for group in groups:
            g = measures.GROUPS.index(group)
            columns = {name: data[f"group|{name}"][:, g] for name in PERFORMANCE + REMOVAL}
            columns.update(mean_activation=data["group|mean_activation"][:, g],
                           reconstruction_error=data["group|reconstruction_error"][:, g])
            columns.update({name: data[f"concept|{name}"] for name in AGREEMENT + ("activation_rate",)})
            columns["profile_shift"] = data["profile|shift"][:, measures.PROFILE_GROUPS.index(group)]
            for estimator in paths.INFLUENCE_ESTIMATORS:
                for name in INFLUENCE:
                    key = f"influence|{estimator}|{name}"
                    value = data[key][:, g] if key in data else np.full_like(columns["brier"], np.nan)
                    columns[_influence_name(name, estimator)] = value
            shifts = features.load_features(record)
            for name in ("feature_shift_all", "feature_shift_rule"):
                columns[name] = shifts[name][:, g]
            years = np.array(measures.YEARS)
            keep = years >= record["reference"]
            frames.append(pd.DataFrame({
                "concept": record["concept"], "variants": ",".join(record["variants"]),
                "reference": record["reference"], "year": years[keep], "group": group,
                "build": record["build"]}))
            for name, value in columns.items():
                cubes.setdefault(name, []).append(value[:, keep])
    point = pd.concat(frames, ignore_index=True)
    cube = {name: np.concatenate(parts, axis=1).astype(np.float64) for name, parts in cubes.items()}
    cube["log_n"] = np.log(np.where(cube["n"] > 0, cube["n"], np.nan))
    for name, value in cube.items():
        point[name] = value[0]
    point["years_since_reference"] = point.year - point.reference
    point["predictor_year"] = point.build + "/" + point.reference.astype(str) + "/" + point.year.astype(str)
    return point, cube


def select(point: pd.DataFrame, cube: dict, mask) -> tuple:
    """Rows of the panel selected by a boolean mask, for the point table and every replicate."""
    mask = np.asarray(mask, dtype=bool)
    return point[mask].reset_index(drop=True), {name: value[:, mask] for name, value in cube.items()}


def in_variant(point: pd.DataFrame, variant: str) -> np.ndarray:
    return point.variants.str.split(",").apply(lambda names: variant in names).to_numpy()


def add_changes(point: pd.DataFrame, cube: dict, columns: list) -> None:
    """Add '<column>_change' = value at year minus value at year - 1 (same concept and group)."""
    previous = lag_index(point, -1)
    for name in columns:
        value = cube[name]
        change = np.full_like(value, np.nan)
        ok = previous >= 0
        change[:, ok] = value[:, ok] - value[:, previous[ok]]
        cube[f"{name}_change"] = change
        point[f"{name}_change"] = change[0]


def lag_index(point: pd.DataFrame, shift: int) -> np.ndarray:
    """For each row, position of the same concept-group row at year + shift, or -1."""
    position = {key: i for i, key in enumerate(zip(point.concept, point.group, point.year))}
    return np.array([position.get((c, g, y + shift), -1) for c, g, y in zip(point.concept, point.group, point.year)])


# ---------------------------------------------------------------- correlations

def _ranks(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Average ranks along the last axis among valid entries; invalid entries get NaN."""
    return rankdata(np.where(valid, values, np.nan), axis=-1, nan_policy="omit")


def _pearson(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n = valid.sum(-1)
    x = np.where(valid, x, 0.0)
    y = np.where(valid, y, 0.0)
    with np.errstate(all="ignore"):
        mx, my = x.sum(-1) / n, y.sum(-1) / n
        dx = np.where(valid, x - mx[..., None], 0.0)
        dy = np.where(valid, y - my[..., None], 0.0)
        r = (dx * dy).sum(-1) / np.sqrt((dx ** 2).sum(-1) * (dy ** 2).sum(-1))
    r[(n < 3) | ~np.isfinite(r)] = np.nan
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation per replicate. x, y: (replicates, rows) -> (replicates,)."""
    valid = np.isfinite(x) & np.isfinite(y)
    return _pearson(_ranks(x, valid), _ranks(y, valid), valid)


def _indicators(labels: np.ndarray) -> np.ndarray:
    """Indicator columns of a label array, first level dropped."""
    _, codes = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    columns = np.zeros((len(codes), codes.max() + 1 if len(codes) else 1))
    columns[np.arange(len(codes)), codes] = 1
    return columns[:, 1:]


def _partial_rank(x: np.ndarray, y: np.ndarray, numeric: list, indicators: list) -> float:
    """One replicate: rank correlation of x and y after removing (least squares) ranked numeric
    covariates and indicator columns (plain Spearman without controls). NaN rows are left out."""
    ok = np.isfinite(x) & np.isfinite(y)
    for value in numeric:
        ok &= np.isfinite(value)
    if ok.sum() < 4:
        return np.nan
    design = [np.ones((ok.sum(), 1))]
    design += [rankdata(value[ok])[:, None] for value in numeric]
    design += [_indicators(labels[ok]) for labels in indicators]
    design = np.concatenate(design, axis=1)
    targets = np.column_stack([rankdata(x[ok]), rankdata(y[ok])])
    coef, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    residual = targets - design @ coef
    spread = ((targets - targets.mean(0)) ** 2).sum(0)
    if ok.sum() - rank < 3 or np.min(spread) < 1e-9:          # too few rows, or x or y constant
        return np.nan
    if np.min((residual ** 2).sum(0) / spread) < 1e-12:        # x or y fully explained by the controls
        return np.nan
    return np.corrcoef(residual.T)[0, 1]


def correlation(x: np.ndarray, y: np.ndarray, concepts: np.ndarray = None, numeric: list = (),
                indicators: list = (), concept_indicators: bool = False,
                seed: int = paths.BOOTSTRAP_SEED) -> np.ndarray:
    """Rank (partial) correlation for the point estimate (row 0) and every bootstrap replicate.

    x, y and numeric covariates: (replicates, rows) over a fixed row set; indicators: (rows,) label
    arrays (e.g. predictor-year). Replicate b >= 1 combines patient replicate b of every value with a
    draw of the concepts with replacement (bootstrap.concept_resamples), so the interval reflects which
    concepts are in the study as well as which patients; with concept_indicators=True, concept
    indicator columns are added and a concept drawn twice gets two sets of them. concepts=None
    resamples patients only. Rows undefined in a replicate (a small group whose patients were all
    drawn zero times) are left out of that replicate only. Same construction as the November
    same-year analysis for the point estimate.
    """
    if concepts is None:
        concepts = np.arange(x.shape[1])
        draws = [(np.arange(x.shape[1]), concepts)] * len(x)
    else:
        draws = bootstrap.concept_resamples(concepts, len(x) - 1, seed)
    out = np.full(len(x), np.nan)
    for b, (rows, labels) in enumerate(draws):
        groups = ([labels] if concept_indicators else []) + [np.asarray(i)[rows] for i in indicators]
        out[b] = _partial_rank(x[b, rows], y[b, rows], [v[b, rows] for v in numeric], groups)
    return out


def controls_for(controls, feature: str, target: str) -> list:
    """Control names of an ADJUSTMENTS entry or list. For year-to-year changes (feature or target
    ending in '_change', other than the removal measures, which are levels) the risk control is the
    change in mean risk, not its level."""
    names = ADJUSTMENTS[controls] if isinstance(controls, str) else list(controls)
    if any(name.endswith("_change") and name not in REMOVAL for name in (feature, target)):
        names = ["mean_risk_change" if name == "mean_risk" else name for name in names]
    return names


def associate(point: pd.DataFrame, cube: dict, feature: str, target: str, controls) -> np.ndarray:
    """Correlation of feature and target over all given panel rows, for the point and each replicate.

    controls: an ADJUSTMENTS name or a list of column names. Columns present in the cube are
    ranked numeric covariates; 'concept' and other point columns (predictor_year) become indicator
    columns. point must have a 'concept' column (used for concept resampling).
    Measures saved by earlier runs have only a point estimate (one replicate row).
    """
    controls = controls_for(controls, feature, target)
    numeric_names = [c for c in controls if c in cube]
    replicates = min([len(cube[feature]), len(cube[target])] + [len(cube[c]) for c in numeric_names])
    indicators = [point[c].to_numpy() for c in controls if c not in cube and c != "concept"]
    return correlation(cube[feature][:replicates], cube[target][:replicates], point["concept"].to_numpy(),
                       [cube[c][:replicates] for c in numeric_names], indicators, "concept" in controls)


def complete_rows(cube: dict, mask: np.ndarray, names) -> np.ndarray:
    """Rows of mask where every named column is defined in the point estimate."""
    rows = np.asarray(mask, dtype=bool).copy()
    for name in names:
        rows &= np.isfinite(cube[name][0])
    return rows


def _run_cell(labels: dict, point: pd.DataFrame, cube: dict, feature: str, target: str, controls) -> dict:
    values = associate(point, cube, feature, target, controls)
    adjustment = controls if isinstance(controls, str) else ("+".join(controls) or "none")
    return {**labels, "adjustment": adjustment, "rows": len(point), **summarize(values)}


def grid(point: pd.DataFrame, cube: dict, cells: list, jobs: int = 32) -> pd.DataFrame:
    """Evaluate many associations in parallel.

    cells: list of dicts with keys feature, target, controls (name or list), mask (row selection)
    and any labels to carry into the output table. Cells with the same feature, target and mask form
    one comparison and use one fixed row set: complete cases (point estimate) over the feature, the
    target and every numeric control used by any adjustment of that comparison, so adjustments differ
    only in the controls. 'rows' counts that set. Each worker receives only its own columns.
    """
    from joblib import Parallel, delayed, parallel_config
    shared = {}
    for cell in cells:
        key = (cell["feature"], cell["target"], np.asarray(cell["mask"], dtype=bool).tobytes())
        names = shared.setdefault(key, {cell["feature"], cell["target"]})
        names.update(c for c in controls_for(cell["controls"], cell["feature"], cell["target"]) if c in cube)
    row_sets = {key: complete_rows(cube, np.frombuffer(key[2], dtype=bool), names) for key, names in shared.items()}
    tasks = []
    for cell in cells:
        rows = row_sets[(cell["feature"], cell["target"], np.asarray(cell["mask"], dtype=bool).tobytes())]
        controls = controls_for(cell["controls"], cell["feature"], cell["target"])
        names = {cell["feature"], cell["target"]} | {c for c in controls if c in cube}
        sub_cube = {name: cube[name][:, rows] for name in names}
        sub_point = point.loc[rows, ["concept"] + [c for c in controls if c not in cube and c != "concept"]]
        labels = {k: v for k, v in cell.items() if k not in ("mask", "controls")}
        tasks.append(delayed(_run_cell)(labels, sub_point.reset_index(drop=True), sub_cube, cell["feature"],
                                        cell["target"], cell["controls"]))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return pd.DataFrame(Parallel(n_jobs=jobs)(tasks))


def summarize(values: np.ndarray) -> dict:
    """Point estimate (replicate 0), median of the replicates and 95% percentile interval over them.
    undefined_share: share of replicates without a value (NaN when there are no replicates)."""
    values = np.asarray(values, dtype=float)
    rest = values[1:][np.isfinite(values[1:])]
    return {"r": values[0], "replicate_median": np.median(rest) if len(rest) else np.nan,
            "low": np.percentile(rest, 2.5) if len(rest) else np.nan,
            "high": np.percentile(rest, 97.5) if len(rest) else np.nan,
            "undefined_share": 1 - len(rest) / (len(values) - 1) if len(values) > 1 else np.nan}


# ---------------------------------------------------------------- horizons

def horizon_panel(point: pd.DataFrame, cube: dict, horizon: int, targets: list) -> tuple:
    """Rows at year t joined to the same concept-group at t + horizon (outcomes up to 2014 only).

    Adds 'future_<target>' and 'future_<target>_change' (future minus current) columns.
    """
    future = lag_index(point, horizon)
    keep = (future >= 0) & (point.year.to_numpy() + horizon <= paths.LAST_OUTCOME_YEAR)
    rows, ahead = np.flatnonzero(keep), future[keep]
    joined = point.iloc[rows].reset_index(drop=True).copy()
    joined["future_year"] = point.year.to_numpy()[ahead]
    joined["horizon"] = horizon
    joined_cube = {name: value[:, rows] for name, value in cube.items()}
    for target in targets:
        joined_cube[f"future_{target}"] = cube[target][:, ahead]
        joined_cube[f"future_{target}_change"] = cube[target][:, ahead] - cube[target][:, rows]
        joined[f"future_{target}"] = joined_cube[f"future_{target}"][0]
        joined[f"future_{target}_change"] = joined_cube[f"future_{target}_change"][0]
    if not (joined.future_year - joined.year).eq(horizon).all() or joined.future_year.max() > paths.LAST_OUTCOME_YEAR:
        raise AssertionError("Horizon join paired the wrong years")
    return joined, joined_cube


# ---------------------------------------------------------------- within-concept sign count

def within_concept(point: pd.DataFrame, cube: dict, feature: str, target: str) -> pd.DataFrame:
    """Spearman correlation over the years of each concept-group, for the point and each replicate."""
    out = []
    for (concept, group), rows in point.groupby(["concept", "group"]).indices.items():
        r = spearman(cube[feature][:, rows], cube[target][:, rows])
        out.append({"concept": concept, "group": group, "years": len(rows), "values": r})
    return pd.DataFrame(out)


def sign_count(table: pd.DataFrame) -> np.ndarray:
    """Number of concepts with a positive within-concept correlation, per replicate."""
    matrix = np.stack(table["values"].to_numpy())             # concepts x replicates
    return (matrix > 0).sum(0)


def shuffled_sign_counts(point: pd.DataFrame, cube: dict, feature: str, target: str,
                         shuffles: int = paths.SHUFFLES, seed: int = paths.BOOTSTRAP_SEED) -> np.ndarray:
    """Reference distribution: shuffle the feature's years within each concept-group, count positives."""
    rng = np.random.default_rng(seed)
    groups = list(point.groupby(["concept", "group"]).indices.values())
    x, y = cube[feature][0], cube[target][0]
    stacked = []
    for rows in groups:
        perms = np.stack([rng.permutation(rows) for _ in range(shuffles)])
        stacked.append(spearman(x[perms], np.broadcast_to(y[rows], perms.shape)))
    counts = (np.stack(stacked) > 0).sum(0)
    return counts
