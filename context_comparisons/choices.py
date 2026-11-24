"""Compare validation objectives using the same candidate predictions."""

import numpy as np

from context_intervention.analysis import performance


OBJECTIVES = ("death_f1", "brier_skill", "average_precision")
FORMS = ("direct", "hard_sampling", "graded_sampling")


def options(form):
    if form == "direct":
        return [("original", 0.), (form, .25), (form, .5)]
    if form == "hard_sampling":
        return [("uniform", 0.), (form, .25), (form, .5)]
    if form == "graded_sampling":
        return [("uniform", 0.), (form, float(np.log(2))), (form, float(np.log(4)))]
    raise ValueError(f"Unknown intervention form: {form}")


def score_option(labels, predictions, threshold, reference_mortality, form, strength):
    predictions = np.asarray(predictions)
    if predictions.ndim != 2 or predictions.shape[1] != len(labels):
        raise ValueError("Expected sampling-seed by patient probabilities")
    seed_metrics = [performance(labels, p, threshold, reference_mortality) for p in predictions]
    return {"form": form, "strength": float(strength),
            **{name: float(np.mean([m[name] for m in seed_metrics])) for name in OBJECTIVES}}


def choose(records, objective):
    """Keep the original F1 tie rule; use F1 as the tie breaker for other objectives."""
    order = {"death_f1": ("death_f1", "brier_skill"),
             "brier_skill": ("brier_skill", "death_f1"),
             "average_precision": ("average_precision", "death_f1", "brier_skill")}
    if objective not in order:
        raise ValueError("Unknown selection objective")
    if not records:
        return {"form": "original", "strength": 0., "reason": "insufficient development outcomes"}

    def rank(record):
        values = tuple(record[name] if np.isfinite(record[name]) else -np.inf for name in order[objective])
        return (*values, -record["strength"])

    return dict(max(records, key=rank))
