"""Composition versus same-patient change, and joint rule x activation transitions.

Split of a measure's change between a start year s and an end year e:

    total change      = M(all patients at e)       - M(all patients at s)
    returning part    = M(patients at e seen at s) - M(patients at s seen at e)
    composition part  = total change - returning part

The returning part is the change among patients observed in both years; the composition
part is what is left, i.e. the effect of patients leaving and entering. The split is exact.
"""
import numpy as np


def comparison_pairs(reference: int, years: tuple, last_year: int) -> list:
    """Consecutive pairs (t, t+1) from the reference year, then (reference, t) for t >= reference + 2."""
    consecutive = [(t, t + 1) for t in range(reference, last_year)]
    from_reference = [(reference, t) for t in range(reference + 2, last_year + 1)]
    return [pair for pair in consecutive + from_reference if pair[0] in years and pair[1] in years]


def split_change(values: np.ndarray, years: tuple, pairs: list) -> np.ndarray:
    """(..., years, subsets) measure -> (..., pairs, 2) with [total change, returning part].

    Subset 0 holds every patient; subset 1 + k holds patients also present in years[k].
    """
    out = np.full(values.shape[:-2] + (len(pairs), 2), np.nan)
    position = {year: k for k, year in enumerate(years)}
    for p, (start, end) in enumerate(pairs):
        s, e = position[start], position[end]
        out[..., p, 0] = values[..., e, 0] - values[..., s, 0]
        out[..., p, 1] = values[..., e, 1 + s] - values[..., s, 1 + e]
    return out


def transition_table(patients: np.ndarray, years: np.ndarray, in_a: np.ndarray, in_r: np.ndarray,
                     start: int, end: int) -> dict:
    """Row positions of patients present in both years, with their start/end A and R memberships."""
    at_start = {p: i for i, p in enumerate(patients) if years[i] == start}
    at_end = {p: i for i, p in enumerate(patients) if years[i] == end}
    common = sorted(at_start.keys() & at_end.keys())
    first = np.array([at_start[p] for p in common], dtype=int)
    second = np.array([at_end[p] for p in common], dtype=int)
    return {"patients": np.array(common), "start_rows": first, "end_rows": second,
            "A_start": in_a[first], "A_end": in_a[second], "R_start": in_r[first], "R_end": in_r[second]}
