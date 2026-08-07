"""Versioned, content-addressed artifacts for semantic experiments."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Iterable

import numpy as np

from artifact_storage import (
    ARTIFACT_SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_jsonl_gzip,
    describe_json,
    descriptor_for_file,
    read_artifact,
    validate_descriptor,
)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def stable_hash(*objects: Any) -> str:
    payload = json.dumps(
        objects, default=_json_default, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_fingerprint(array: np.ndarray) -> str:
    """Hash shape, dtype, and contiguous bytes without JSON expansion."""

    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    if value.dtype.kind in {"O", "U", "S"}:
        digest.update(
            json.dumps(value.tolist(), default=_json_default, separators=(",", ":")).encode()
        )
    else:
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def derive_seed(base_seed: int, *identifiers: Any) -> int:
    digest = stable_hash(base_seed, identifiers)
    return int(digest[:8], 16)


class SemanticArtifactStore:
    def __init__(self, root: str | Path, experiment_hash: str):
        self.root = Path(root) / experiment_hash
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, value: Any) -> Path:
        path = self.root / name
        return atomic_write_json(path, value, compact=False)

    def write_jsonl(self, name: str, rows: Iterable[Any]) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, default=_json_default, sort_keys=True, allow_nan=False))
                    handle.write("\n")
            with temporary.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        json.loads(line)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def write_jsonl_gzip(self, name: str, rows: Iterable[Any]) -> dict[str, Any]:
        descriptor = atomic_write_jsonl_gzip(self.root / name, rows)
        descriptor["path"] = name
        return descriptor

    def write_npz(self, name: str, **arrays: np.ndarray) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            with np.load(temporary, allow_pickle=False) as bundle:
                if set(bundle.files) != set(arrays):
                    raise ValueError("NPZ validation changed array members")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def read_json(self, name: str) -> Any:
        with (self.root / name).open(encoding="utf-8") as handle:
            return json.load(handle)

    def read_jsonl(self, name: str) -> list[Any]:
        return read_artifact(self.root, name)

    def exists(self, name: str) -> bool:
        return (self.root / name).exists()


def build_semantic_result_index(
    *,
    store: SemanticArtifactStore,
    scientific_schema_version: str,
    experiment_hash: str,
    manifest_descriptor: dict[str, Any],
    semantic_models_descriptor: dict[str, Any],
    pair_results_descriptor: dict[str, Any],
) -> dict[str, Any]:
    """Build the compact completion marker written after every other payload."""

    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "scientific_schema_version": scientific_schema_version,
        # Retain the established name for lightweight consumers.
        "schema_version": scientific_schema_version,
        "experiment_hash": experiment_hash,
        "complete": True,
        "artifact_dir": str(store.root),
        "artifacts": {
            "manifest": manifest_descriptor,
            "semantic_models": semantic_models_descriptor,
            "pair_results": pair_results_descriptor,
        },
    }


def write_semantic_result_index(store: SemanticArtifactStore, index: dict[str, Any]) -> Path:
    """Validate all descriptors and atomically publish the completion index last."""

    validate_semantic_result_index(index, store.root)
    return atomic_write_json(store.root / "result.json", index)


def validate_semantic_result_index(
    index: dict[str, Any], artifact_dir: str | Path
) -> None:
    if index.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported semantic artifact schema")
    if index.get("complete") is not True:
        raise ValueError("semantic result index is incomplete")
    if not index.get("experiment_hash"):
        raise ValueError("semantic result index has no experiment hash")
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("semantic result index has no artifacts")
    for name in ("manifest", "semantic_models", "pair_results"):
        descriptor = artifacts.get(name)
        if not isinstance(descriptor, dict):
            raise ValueError(f"semantic result index has no {name} descriptor")
        validate_descriptor(artifact_dir, descriptor)


def load_semantic_result(
    store: SemanticArtifactStore, *, expected_experiment_hash: str | None = None
) -> dict[str, Any]:
    """Load a v2 bundle into the unchanged Python API, or return a legacy v1 result."""

    index = store.read_json("result.json")
    if index.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        required = {"manifest", "semantic_models", "pair_results"}
        if not required.issubset(index):
            raise ValueError("legacy semantic result is incomplete")
        return index
    validate_semantic_result_index(index, store.root)
    if (
        expected_experiment_hash is not None
        and index.get("experiment_hash") != expected_experiment_hash
    ):
        raise ValueError("semantic result experiment identity mismatch")
    artifacts = index["artifacts"]
    return {
        "schema_version": index["scientific_schema_version"],
        "experiment_hash": index["experiment_hash"],
        "artifact_dir": str(store.root),
        "cache_hit": False,
        "manifest": read_artifact(store.root, artifacts["manifest"]),
        "semantic_models": read_artifact(store.root, artifacts["semantic_models"]),
        "pair_results": read_artifact(store.root, artifacts["pair_results"]),
    }


def semantic_bundle_descriptors(
    store: SemanticArtifactStore,
    *,
    semantic_models_descriptor: dict[str, Any],
    pair_results_descriptor: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = describe_json(store.root / "manifest.json", relative_to=store.root)
    return manifest, semantic_models_descriptor, pair_results_descriptor


def environment_manifest() -> dict[str, Any]:
    versions: dict[str, str] = {"python": platform.python_version(), "numpy": np.__version__}
    for package in ("sklearn", "scipy", "pandas", "torch"):
        try:
            module = __import__(package)
            versions[package] = str(module.__version__)
        except (ImportError, AttributeError):
            versions[package] = "unavailable"
    return versions
