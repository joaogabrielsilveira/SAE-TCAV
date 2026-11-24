"""Observe labeled context states during fitting, without querying context inputs."""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from context_intervention import predictor


SEMANTICS = {"role": "labeled context", "layer": "final transformer_encoder output before decoder",
             "token": "last feature/label token", "axes": "output[batch=0, context_row, token=-1, embedding]",
             "meaning": "descriptive endpoint; not the per-layer cached attention keys or values"}


def implementation_identity():
    from tabpfn.model import transformer

    directory = Path(transformer.__file__).parent
    return {"torch_version": str(torch.__version__), "semantics": SEMANTICS,
            "installed_model_sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted(directory.glob("*.py"))}}


def validate_states(states, count):
    if states.shape != (count, 192) or not np.isfinite(states).all():
        raise ValueError("Unexpected native state shape or nonfinite values")
    return states


@contextmanager
def observe_native_states():
    """Capture the installed model's native train-embedding coordinates unchanged.

    The installed model defines train_embeddings as the last feature/label token
    of transformer_encoder output, with sequence and batch axes transposed.
    Global hooks exist only inside this context manager and observe tensors;
    they neither replace outputs nor alter the installed library's source.
    """
    from tabpfn.model.transformer import PerFeatureTransformer

    captured = []
    handles = []

    def before_model(module, args):
        if not isinstance(module, PerFeatureTransformer):
            return

        def after_encoder(encoder, inputs, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 4 or output.shape[0] != 1:
                raise ValueError("Native context extraction requires one configuration and a four-dimensional encoder output")
            captured.append(output[0, :, -1, :].detach().cpu().numpy().copy())

        handles.append(module.transformer_encoder.register_forward_hook(after_encoder))

    global_handle = torch.nn.modules.module.register_module_forward_pre_hook(before_model)
    try:
        yield captured
    finally:
        global_handle.remove()
        for handle in handles:
            handle.remove()


def ordered_states(system, captured, count):
    if len(captured) != 1:
        raise ValueError(f"Expected one context forward pass, observed {len(captured)}")
    model = system["classifier"]
    configurations = model.ensemble_configurations_
    if len(configurations) != 1:
        raise ValueError("Native state comparison requires one frozen configuration")
    indices = np.asarray(configurations[0].subsample_samples_configuration, dtype=int)
    if not np.array_equal(np.sort(indices), np.arange(count)):
        raise ValueError("Native context rows were subsampled or duplicated")
    states = validate_states(captured[0], count)
    return states[np.argsort(indices)]


def extract(reference, directory):
    """Restore the frozen context once and retain its native labeled states."""
    rows = reference["context_rows"]
    digest = hashlib.sha256()
    for value in (reference["predictor"]["context_identity"], reference["predictor"]["checkpoint_sha256"],
                  reference["population"]["fingerprint"]):
        digest.update(str(value).encode())
    digest.update(Path(__file__).read_bytes())
    digest.update(Path(predictor.__file__).read_bytes())
    for name in ("concept_subgroup_drift_pilot.py", "tabpfn_model.py"):
        digest.update(Path(name).read_bytes())
    digest.update(str((reference["origin"], reference["seed"])).encode())
    implementation = implementation_identity()
    digest.update(json.dumps(implementation, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(rows).tobytes())
    identity = digest.hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"native_{reference['build']}_{reference['reference']}.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as stored:
            if str(stored["identity"]) != identity or not np.array_equal(stored["rows"], rows):
                raise ValueError("Native state cache differs from its source context")
            return validate_states(stored["states"].copy(), len(rows))
    with observe_native_states() as captured:
        system = predictor.restore(reference, "cpu")
    states = ordered_states(system, captured, len(rows))
    del system
    temporary = target.with_name(target.stem + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, rows=rows, states=states, identity=identity,
                        role="labeled context", device="cpu", implementation=json.dumps(implementation, sort_keys=True))
    os.replace(temporary, target)
    return states
