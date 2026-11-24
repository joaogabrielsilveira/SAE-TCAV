"""Tidy tables (point estimate + joint-bootstrap 95% interval) built from the cached replicates."""
import numpy as np
import pandas as pd

from concept_study import bootstrap, measures, paths


def _interval(values: np.ndarray) -> tuple:
    """values: (replicates, ...) with replicate 0 the point estimate."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        low, high = bootstrap.percentile_interval(values[1:], axis=0)
    return values[0], low, high


def _base(record: dict) -> dict:
    return {"concept": record["concept"], "variants": ",".join(record["variants"]),
            "reference": record["reference"], "build": record["build"]}


def _check_split_key(key: str) -> None:
    """The returning subset at the start year excludes patients who die that year: no outcome splits."""
    if key.startswith("group|") and key.split("|")[-1] in measures.OUTCOME_MEASURES:
        raise ValueError(f"{key} uses outcomes; the composition split is defined only for outcome-free measures")


def split_table(records: list, keys: dict, groups=("A", "R", "all")) -> pd.DataFrame:
    """Composition split of each measure for every concept and comparison pair.

    keys: {display name: replicate key}, e.g. {"agreement": "concept|agreement",
    "absolute sensitivity": "influence|all_negatives|absolute_sensitivity"}.
    Concept-level keys (no group axis) are reported once with group 'concept'. Outcome measures are refused.
    """
    for key in keys.values():
        _check_split_key(key)
    rows = []
    for record in records:
        data = measures.load_replicates(record)
        pairs = data["pairs"].astype(int)
        for name, key in keys.items():
            full = f"split|{key}"
            if full not in data:
                continue
            values = data[full]
            by_group = {"concept": values} if key.startswith("concept|") else {
                g: values[:, measures.GROUPS.index(g)] for g in groups}
            for group, value in by_group.items():
                total, total_low, total_high = _interval(value[..., 0])
                same, same_low, same_high = _interval(value[..., 1])
                mixed, mixed_low, mixed_high = _interval(value[..., 0] - value[..., 1])
                for p, (start, end) in enumerate(pairs):
                    rows.append({**_base(record), "measure": name, "group": group, "start": start, "end": end,
                                 "consecutive": end == start + 1,
                                 "total": total[p], "total_low": total_low[p], "total_high": total_high[p],
                                 "returning": same[p], "returning_low": same_low[p], "returning_high": same_high[p],
                                 "composition": mixed[p], "composition_low": mixed_low[p],
                                 "composition_high": mixed_high[p]})
    return pd.DataFrame(rows)


def split_replicates(records: list, key: str, group: str = None, consecutive_only: bool = True) -> tuple:
    """Stack the split replicates of one measure over concepts and pairs.

    Returns (row table, array (replicates, rows, 2) with [total, returning]).
    """
    _check_split_key(key)
    tables, arrays = [], []
    for record in records:
        data = measures.load_replicates(record)
        full = f"split|{key}"
        if full not in data:
            continue
        value = data[full] if group is None else data[full][:, measures.GROUPS.index(group)]
        pairs = data["pairs"].astype(int)
        keep = (pairs[:, 1] == pairs[:, 0] + 1) if consecutive_only else np.ones(len(pairs), bool)
        tables.append(pd.DataFrame({**_base(record), "start": pairs[keep, 0], "end": pairs[keep, 1]}))
        arrays.append(value[:, keep])
    return pd.concat(tables, ignore_index=True), np.concatenate(arrays, axis=1)


def transition_table(records: list, estimator: str = paths.PRIMARY_ESTIMATOR,
                     replicate_name: str = "absolute_difference") -> tuple:
    """Rule patients who stopped vs kept activating: per concept and consecutive pair.

    Returns (table with point and interval, replicate array (replicates, rows) of `replicate_name`).
    """
    rows, replicate_rows = [], []
    for record in records:
        data = measures.load_replicates(record)
        prefix = f"transition|{estimator}|"
        if prefix + "signed_difference" not in data:
            continue
        pairs = data["pairs"].astype(int)
        for p, (start, end) in enumerate(pairs):
            if end != start + 1 or data[prefix + "stopped_n"][p] == 0 or data[prefix + "kept_n"][p] == 0:
                continue
            row = {**_base(record), "start": start, "end": end,
                   "stopped_n": int(data[prefix + "stopped_n"][p]), "kept_n": int(data[prefix + "kept_n"][p])}
            for name in ("signed_difference", "absolute_difference", "activation_difference", "risk_difference"):
                value, low, high = _interval(data[prefix + name][:, p])
                row.update({name: value, f"{name}_low": low, f"{name}_high": high})
            rows.append(row)
            replicate_rows.append(data[prefix + replicate_name][:, p])
    return pd.DataFrame(rows), np.stack(replicate_rows, axis=1)


def per_fit_table(records: list, estimator: str) -> pd.DataFrame:
    """Per-fit (no bootstrap) TCAV, signed and absolute sensitivity in A and R by year."""
    with np.load(paths.CAV_DIRECTIONS) as directions:
        seeds_of = {name: directions[name] for name in directions.files if name.startswith(f"seeds|{estimator}|")}
        valid_of = {name: np.isfinite(directions[name]).all(axis=1) for name in directions.files
                    if name.startswith(f"{estimator}|")}
    rows = []
    for record in records:
        data = measures.load_replicates(record)
        key = f"per_fit|{estimator}|tcav"
        if key not in data:
            continue
        seeds = seeds_of[f"seeds|{estimator}|{record['concept']}"][valid_of[f"{estimator}|{record['concept']}"]]
        for group in ("A", "R"):
            g = measures.GROUPS.index(group)
            for y, year in enumerate(measures.YEARS):
                if year < record["reference"]:
                    continue
                for metric in ("tcav", "signed_sensitivity", "absolute_sensitivity"):
                    values = data[f"per_fit|{estimator}|{metric}"][g, y]
                    rows.append(pd.DataFrame({**_base(record), "group": group, "year": year, "metric": metric,
                                              "draw": seeds, "value": values}))
    return pd.concat(rows, ignore_index=True)


def level_table(records: list, keys: dict, groups=("A", "R")) -> pd.DataFrame:
    """Point and interval of selected measures per concept, group and year."""
    rows = []
    for record in records:
        data = measures.load_replicates(record)
        for name, key in keys.items():
            if key not in data:
                continue
            values = data[key]
            by_group = {"concept": values} if key.startswith("concept|") else {
                g: values[:, (measures.PROFILE_GROUPS if key.startswith("profile|") else measures.GROUPS).index(g)]
                for g in groups}
            for group, value in by_group.items():
                point, low, high = _interval(value)
                for y, year in enumerate(measures.YEARS):
                    if year >= record["reference"]:
                        rows.append({**_base(record), "measure": name, "group": group, "year": year,
                                     "value": point[y], "low": low[y], "high": high[y]})
    return pd.DataFrame(rows)


def stacked_measure(records: list, key: str, group: str = None) -> tuple:
    """(row table of concept-years, replicate array (replicates, rows)) for one measure."""
    tables, arrays = [], []
    for record in records:
        data = measures.load_replicates(record)
        if key not in data:
            continue
        value = data[key]
        if group is not None:
            value = value[:, measures.GROUPS.index(group)]
        years = np.array(measures.YEARS)
        keep = years >= record["reference"]
        tables.append(pd.DataFrame({**_base(record), "year": years[keep]}))
        arrays.append(value[:, keep])
    return pd.concat(tables, ignore_index=True), np.concatenate(arrays, axis=1)


def median_with_interval(values: np.ndarray, concepts: np.ndarray = None) -> tuple:
    """Median across rows (axis 1) per replicate -> (point, low, high). See _summary_with_interval."""
    return _summary_with_interval(values, np.nanmedian, concepts)


def mean_with_interval(values: np.ndarray, concepts: np.ndarray = None) -> tuple:
    """Mean across rows (axis 1) per replicate -> (point, low, high). See _summary_with_interval."""
    return _summary_with_interval(values, np.nanmean, concepts)


def _summary_with_interval(values: np.ndarray, summary, concepts: np.ndarray = None) -> tuple:
    """Summary over rows per replicate. With concepts (one label per row), replicate b also resamples
    concepts with replacement (bootstrap.concept_resamples) on top of patient replicate b."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if concepts is None:
            per_replicate = summary(values, axis=1)
        else:
            draws = bootstrap.concept_resamples(concepts, len(values) - 1)
            per_replicate = np.array([summary(values[b, rows]) for b, (rows, _) in enumerate(draws)])
    point, low, high = _interval(per_replicate)
    return float(point), float(low), float(high)
