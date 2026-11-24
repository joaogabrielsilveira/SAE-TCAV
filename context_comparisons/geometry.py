"""Check whether cosine retrieval can distinguish rows inside its strata."""

import numpy as np

from context_intervention import sampling
from context_intervention.analysis import GROUPS, prediction_groups


TOLERANCE = 1e-8


def retrieval_diagnostics(context, patients, years, labels, outcomes, probabilities, threshold, size=250):
    """Measure the actual representative scores, without changing normalization.

    A stratum discriminates only if it has a partial retrieval quota, nonzero
    directional spread, and a nonconstant maximum cosine score. Exact and near
    ties are recorded even when another stratum provides useful discrimination.
    """
    context, patients = np.asarray(context, float), np.asarray(patients, float)
    if (context.ndim != 2 or patients.ndim != 2 or context.shape[1] != patients.shape[1]
            or not np.isfinite(context).all() or not np.isfinite(patients).all()):
        raise ValueError("Retrieval diagnostics require aligned finite matrices")
    norms = np.linalg.norm(context, axis=1)
    normalized = np.divide(context, norms[:, None], out=np.zeros_like(context), where=norms[:, None] > 0)
    keys = np.asarray(sampling.strata(years, labels))
    quotas = sampling.quotas(keys, min(size, len(context)))
    groups = prediction_groups(outcomes, probabilities, threshold)
    records = []
    for group in GROUPS:
        rows = np.flatnonzero(groups == group)
        if not len(rows):
            continue
        representatives = patients[rows[sampling.representatives(patients[rows])]]
        query_norms = np.linalg.norm(representatives, axis=1)
        directions = np.divide(representatives, query_norms[:, None], out=np.zeros_like(representatives),
                               where=query_norms[:, None] > 0)
        scores = (normalized @ directions.T).max(axis=1)
        for key, quota in quotas.items():
            members = np.flatnonzero(np.all(keys == key, axis=1))
            values = scores[members]
            spread = float(np.max(np.linalg.norm(normalized[members] - normalized[members[0]], axis=1)))
            score_spread = float(np.ptp(values))
            partial = 0 < quota < len(members)
            boundary_ties = 0
            if partial:
                ordered = np.sort(values)[::-1]
                if abs(ordered[quota - 1] - ordered[quota]) <= TOLERANCE:
                    boundary_ties = int(np.isclose(values, ordered[quota - 1], atol=TOLERANCE, rtol=0).sum())
            records.append({"group": group, "year": int(key[0]), "outcome": int(key[1]),
                            "rows": len(members), "quota": quota,
                            "zero_norm_rows": int((norms[members] == 0).sum()),
                            "zero_norm_representatives": int((query_norms == 0).sum()),
                            "normalized_direction_spread": spread, "similarity_spread": score_spread,
                            "boundary_tied_rows": boundary_ties,
                            "discriminates": bool(partial and spread > TOLERANCE and score_spread > TOLERANCE)})
    return {"tolerance": TOLERANCE, "strata": records,
            "any_discriminating_stratum": any(row["discriminates"] for row in records),
            "fallback_rule": "original full context when no partially retrieved stratum discriminates"}
