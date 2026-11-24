"""Restore the existing predictor and score explicitly supplied contexts."""

import numpy as np
import torch

from concept_study import error_analysis, loading, year_input


def open_reference(build, reference, device="cuda"):
    """Load a frozen predictor's population, roles and original context rows."""
    import concept_subgroup_drift_pilot as pilot

    inputs = loading.open_inputs(device)
    cache = inputs["caches"][build]
    population = inputs["population"]
    predictor = cache.load(f"P/{reference}")
    roles = pilot.reference_year_roles(population, reference, seed=cache.config["split_seed"])
    rows = error_analysis.context_rows(population, roles, predictor)
    columns = pilot._feature_columns(population["X"].shape[1])
    return {"build": build, "inputs": inputs, "cache": cache, "population": population, "predictor": predictor,
            "roles": roles, "context_rows": rows, "columns": columns, "reference": reference,
            "origin": min(predictor.get("context_years", [reference])),
            "seed": int(cache.config["split_seed"])}


def frame_for_rows(reference, rows, include_labels=False):
    import concept_subgroup_drift_pilot as pilot

    rows = np.asarray(rows, dtype=int)
    frame = pilot._prediction_frame(reference["population"], rows, reference["columns"])
    if include_labels:
        frame["y"] = reference["population"]["y"][rows]
    return frame


def restore(reference, device="cpu"):
    """Restore and verify the exact original context against the frozen identity."""
    return year_input.restore_frozen_context(reference["population"], reference["roles"],
                                             reference["predictor"], device, reference["seed"])


def fit_context(reference, selected_rows, device="cuda"):
    """Set up the frozen model on selected original context rows, preserving domain origin."""
    import concept_subgroup_drift_pilot as pilot

    selected_rows = np.asarray(selected_rows, dtype=int)
    if len(np.unique(selected_rows)) != len(selected_rows):
        raise ValueError("Context rows must not be duplicated")
    if not np.isin(selected_rows, reference["context_rows"]).all():
        raise ValueError("Every selected row must belong to the original context")
    frame = frame_for_rows(reference, selected_rows, include_labels=True)
    try:
        system = pilot.fit_frozen_predictor(frame, reference["columns"], device,
                                             seed=reference["seed"], domain_origin=reference["origin"])
    except torch.cuda.OutOfMemoryError:
        if device != "cuda":
            raise
        torch.cuda.empty_cache()
        device = "cpu"
        system = pilot.fit_frozen_predictor(frame, reference["columns"], device,
                                             seed=reference["seed"], domain_origin=reference["origin"])
    # The original reference year governs query validity even if every row from
    # that year was removed. The original calendar domain origin also stays fixed.
    system["reference_year"] = reference["reference"]
    if system["checkpoint_sha256"] != reference["predictor"]["checkpoint_sha256"]:
        raise ValueError("Modified context used a different checkpoint")
    return system, device


def predict_rows(reference, system, rows, device, batch=64, historical_context=False):
    rows = np.asarray(rows, dtype=int)
    if np.any(reference["population"]["year"][rows] > 2014):
        raise ValueError("This outcome study excludes years after 2014")
    if historical_context and not np.isin(rows, reference["context_rows"]).all():
        raise ValueError("Historical queries are limited to original context rows")
    if not len(rows):
        return np.array([], dtype=float), np.empty((0, 192), dtype=float)
    if not historical_context:
        return year_input.predict(system, frame_for_rows(reference, rows), device, batch=batch)
    # The capture helper normally forbids earlier queries. For descriptive
    # context representations only, run those rows in the same query position
    # and with their true calendar domain. These queries can include the same
    # row in the context and therefore are not leave-one-out explanations.
    original = system["reference_year"]
    try:
        system["reference_year"] = reference["origin"]
        return year_input.predict(system, frame_for_rows(reference, rows), device, batch=batch)
    finally:
        system["reference_year"] = original


def context_predictions(reference, selected_rows, query_rows, device="cuda", batch=64):
    """Predict with a selected context, falling back to CPU only on GPU OOM."""
    system, actual_device = fit_context(reference, selected_rows, device)
    try:
        probabilities, embeddings = predict_rows(reference, system, query_rows, actual_device, batch)
    except torch.cuda.OutOfMemoryError:
        if actual_device != "cuda":
            raise
        del system
        torch.cuda.empty_cache()
        system, actual_device = fit_context(reference, selected_rows, "cpu")
        probabilities, embeddings = predict_rows(reference, system, query_rows, actual_device, batch)
    return probabilities, embeddings, actual_device
