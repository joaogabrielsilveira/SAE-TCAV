"""Fixed feature and concept representations for development and context rows."""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from context_intervention import data, predictor


def original_context_embeddings(reference, output, device="cuda"):
    """Original-context query embeddings, including historical context years.

    Each context row is queried in the model's query position while the original
    context remains present. A queried context row can therefore also be present
    as an example; these representations do not isolate that row's influence.
    """
    rows = reference["context_rows"]
    identity = hashlib.sha256()
    identity.update(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
    for value in (reference["predictor"]["context_identity"], reference["predictor"]["checkpoint_sha256"],
                  reference["population"]["fingerprint"], reference["origin"]):
        identity.update(str(value).encode())
    identity.update(Path(__file__).read_bytes())
    identity.update(Path(predictor.__file__).read_bytes())
    key = identity.hexdigest()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"context_embeddings_{key}.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as stored:
            if str(stored["identity"]) != key or not np.array_equal(stored["rows"], rows):
                raise ValueError("Context embedding cache identity mismatch")
            return stored["embeddings"].copy(), str(stored["device"])
    system = None
    try:
        system = predictor.restore(reference, device)
        _, embeddings = predictor.predict_rows(reference, system, rows, device, batch=32, historical_context=True)
    except torch.cuda.OutOfMemoryError:
        if device != "cuda":
            raise
        del system
        torch.cuda.empty_cache()
        device = "cpu"
        system = predictor.restore(reference, device)
        _, embeddings = predictor.predict_rows(reference, system, rows, device, batch=32, historical_context=True)
    temporary = target.with_name(target.stem + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, embeddings=embeddings, rows=rows, device=device, identity=key)
    os.replace(temporary, target)
    return embeddings, device


def representations(reference, development_rows, concept_records, output, sae_device="cuda"):
    """Return standardized context/development features and frozen concept factors."""
    population = reference["population"]
    context_rows = reference["context_rows"]
    context_features = np.log1p(population["X"][context_rows].astype(float))
    development_features = np.log1p(population["X"][development_rows].astype(float))
    if np.any(~np.isfinite(context_features)) or np.any(~np.isfinite(development_features)):
        raise ValueError("Log-transformed counts must be finite")
    features = (data.standardized(context_features, context_features),
                data.standardized(development_features, context_features))
    context_embeddings, actual_device = original_context_embeddings(reference, output)
    _, development_embeddings = data.cached_predictor_rows(reference, development_rows)
    both = np.vstack((context_embeddings, development_embeddings))
    activations = data.factor_activations(reference, both, concept_records, sae_device)
    context_concepts = activations[:len(context_rows)]
    development_concepts = activations[len(context_rows):]
    reference_mask = population["year"][development_rows] == reference["reference"]
    if not reference_mask.any():
        raise ValueError("Reference-year development rows are required for concept standardization")
    reference_concepts = development_concepts[reference_mask]
    concepts = (data.standardized(context_concepts, reference_concepts),
                data.standardized(development_concepts, reference_concepts))
    return {"features": features, "concepts": concepts, "context_embedding_device": actual_device}
