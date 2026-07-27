"""Versioned, content-addressed artifacts for semantic experiments."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Iterable

import numpy as np


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
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, default=_json_default, sort_keys=True, indent=2, allow_nan=False)
        return path

    def write_jsonl(self, name: str, rows: Iterable[Any]) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=_json_default, sort_keys=True, allow_nan=False))
                handle.write("\n")
        return path

    def write_npz(self, name: str, **arrays: np.ndarray) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return path

    def read_json(self, name: str) -> Any:
        with (self.root / name).open(encoding="utf-8") as handle:
            return json.load(handle)

    def read_jsonl(self, name: str) -> list[Any]:
        with (self.root / name).open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def exists(self, name: str) -> bool:
        return (self.root / name).exists()


def environment_manifest() -> dict[str, Any]:
    versions: dict[str, str] = {"python": platform.python_version(), "numpy": np.__version__}
    for package in ("sklearn", "scipy", "pandas", "torch"):
        try:
            module = __import__(package)
            versions[package] = str(module.__version__)
        except (ImportError, AttributeError):
            versions[package] = "unavailable"
    return versions
