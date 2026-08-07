"""Lossless, deterministic, and atomic storage helpers for experiment artifacts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Mapping

import numpy as np


ARTIFACT_SCHEMA_VERSION = 2
JSONL_GZIP_FORMAT = "jsonl+gzip"
# Audited after the storage-only implementation is complete.  These constants
# live outside the scientific source sets so recording them cannot change the
# fingerprints they guard.
STORAGE_EQUIVALENT_SEMANTIC_FINGERPRINT = (
    "345d47a4717f7ad3d9b714e3453f02bc839372736610b6eaa1955d2660274a21"
)
STORAGE_EQUIVALENT_RUNNER_FINGERPRINT = (
    "d65eff6d322da5d99a5291d962aa39f5a4ec5a5c924604a3d2fed3060cc0ebc5"
)


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(jsonable(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    return Path(name)


def _publish(temporary: Path, destination: Path) -> None:
    os.replace(temporary, destination)


def atomic_write_json(path: str | Path, value: Any, *, compact: bool = True) -> Path:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        text = (
            canonical_json(value)
            if compact
            else json.dumps(jsonable(value), sort_keys=True, indent=2, allow_nan=False)
        )
        temporary.write_text(text + "\n", encoding="utf-8")
        # Validate the only copy that can be published.
        with temporary.open(encoding="utf-8") as handle:
            json.load(handle)
        _publish(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _open_deterministic_gzip(path: Path):
    raw = path.open("wb")
    zipped = gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0)
    return raw, zipped


def atomic_write_jsonl_gzip(
    path: str | Path, rows: Iterable[Any]
) -> dict[str, Any]:
    destination = Path(path)
    if not destination.name.endswith(".jsonl.gz"):
        raise ValueError("gzip JSONL artifacts must end in .jsonl.gz")
    temporary = _temporary_path(destination)
    row_count = 0
    try:
        raw, zipped = _open_deterministic_gzip(temporary)
        try:
            for row in rows:
                zipped.write(canonical_json(row).encode("utf-8"))
                zipped.write(b"\n")
                row_count += 1
        finally:
            zipped.close()
            raw.close()
        # A complete decompression and JSON parse catches truncated output.
        verified = 0
        with gzip.open(temporary, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    json.loads(line)
                    verified += 1
        if verified != row_count:
            raise ValueError("gzip JSONL row count changed during validation")
        _publish(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return descriptor_for_file(destination, format=JSONL_GZIP_FORMAT, row_count=row_count)


def atomic_write_json_gzip(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    if not destination.name.endswith(".json.gz"):
        raise ValueError("gzip JSON artifacts must end in .json.gz")
    temporary = _temporary_path(destination)
    try:
        raw, zipped = _open_deterministic_gzip(temporary)
        try:
            zipped.write(canonical_json(value).encode("utf-8"))
            zipped.write(b"\n")
        finally:
            zipped.close()
            raw.close()
        with gzip.open(temporary, "rt", encoding="utf-8") as handle:
            json.load(handle)
        _publish(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_json_file(path: str | Path) -> Any:
    source = Path(path)
    if source.name.endswith(".gz"):
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with source.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_gzip_copy(path: str | Path, source: str | Path) -> Path:
    """Deterministically gzip source bytes and validate an exact round trip."""

    destination = Path(path)
    source_path = Path(source)
    temporary = _temporary_path(destination)
    source_digest = file_sha256(source_path)
    try:
        raw, zipped = _open_deterministic_gzip(temporary)
        try:
            with source_path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    zipped.write(chunk)
        finally:
            zipped.close()
            raw.close()
        digest = hashlib.sha256()
        with gzip.open(temporary, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != source_digest:
            raise ValueError(f"gzip round-trip mismatch for {source_path}")
        _publish(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def descriptor_for_file(
    path: str | Path,
    *,
    format: str | None = None,
    row_count: int | None = None,
    relative_to: str | Path | None = None,
) -> dict[str, Any]:
    artifact = Path(path)
    if format is None:
        format = infer_format(artifact.name)
    recorded_path = artifact
    if relative_to is not None:
        recorded_path = artifact.relative_to(Path(relative_to))
    descriptor: dict[str, Any] = {
        "path": str(recorded_path),
        "format": format,
        "sha256": file_sha256(artifact),
        "size_bytes": artifact.stat().st_size,
    }
    if row_count is not None:
        descriptor["row_count"] = int(row_count)
    return descriptor


def infer_format(name: str) -> str:
    if name.endswith(".jsonl.gz"):
        return JSONL_GZIP_FORMAT
    if name.endswith(".json.gz"):
        return "json+gzip"
    if name.endswith(".jsonl"):
        return "jsonl"
    if name.endswith(".json"):
        return "json"
    if name.endswith(".csv"):
        return "csv"
    if name.endswith(".npz"):
        return "npz"
    return "binary"


def resolve_artifact_path(base: str | Path, entry: str | Mapping[str, Any]) -> Path:
    raw = entry if isinstance(entry, str) else entry.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("artifact descriptor has no path")
    relative = Path(raw)
    if relative.is_absolute():
        return relative
    if ".." in relative.parts:
        raise ValueError(f"unsafe artifact path: {raw}")
    return Path(base) / relative


def validate_descriptor(
    base: str | Path, entry: Mapping[str, Any], *, parse: bool = True
) -> Path:
    path = resolve_artifact_path(base, entry)
    if not path.is_file():
        raise ValueError(f"missing artifact: {path}")
    if path.stat().st_size != int(entry["size_bytes"]):
        raise ValueError(f"artifact size mismatch: {path}")
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(f"artifact checksum mismatch: {path}")
    if parse:
        format = str(entry.get("format", infer_format(path.name)))
        if format in {JSONL_GZIP_FORMAT, "jsonl"}:
            actual = sum(1 for _ in iter_artifact_rows(base, entry, validate=False))
        else:
            value = read_artifact(base, entry, validate=False)
            actual = len(value) if isinstance(value, list) else 1
        if "row_count" in entry and actual != int(entry["row_count"]):
            raise ValueError(f"artifact row-count mismatch: {path}")
    return path


def iter_artifact_rows(
    base: str | Path,
    entry: str | Mapping[str, Any],
    *,
    validate: bool = True,
) -> Iterator[Any]:
    if isinstance(entry, Mapping) and validate:
        validate_descriptor(base, entry, parse=False)
    path = resolve_artifact_path(base, entry)
    format = infer_format(path.name) if isinstance(entry, str) else str(entry.get("format", infer_format(path.name)))
    if format == JSONL_GZIP_FORMAT:
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    elif format == "jsonl":
        opener = lambda: path.open(encoding="utf-8")
    else:
        value = read_artifact(base, entry, validate=validate)
        if not isinstance(value, list):
            raise ValueError(f"artifact is not a row table: {path}")
        yield from value
        return
    count = 0
    with opener() as handle:
        for line in handle:
            if line.strip():
                count += 1
                yield json.loads(line)
    if isinstance(entry, Mapping) and "row_count" in entry and count != int(entry["row_count"]):
        raise ValueError(f"artifact row-count mismatch: {path}")


def read_artifact(
    base: str | Path,
    entry: str | Mapping[str, Any],
    *,
    validate: bool = True,
) -> Any:
    if isinstance(entry, Mapping) and validate:
        validate_descriptor(base, entry, parse=False)
    path = resolve_artifact_path(base, entry)
    format = infer_format(path.name) if isinstance(entry, str) else str(entry.get("format", infer_format(path.name)))
    if format == JSONL_GZIP_FORMAT:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = [json.loads(line) for line in handle if line.strip()]
    elif format == "jsonl":
        with path.open(encoding="utf-8") as handle:
            value = [json.loads(line) for line in handle if line.strip()]
    elif format == "json+gzip":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
    elif format == "json":
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    elif format == "csv":
        with path.open(newline="", encoding="utf-8") as handle:
            value = list(csv.DictReader(handle))
    else:
        raise ValueError(f"unsupported readable artifact format {format!r}")
    if isinstance(entry, Mapping) and "row_count" in entry:
        actual = len(value) if isinstance(value, list) else 1
        if actual != int(entry["row_count"]):
            raise ValueError(f"artifact row-count mismatch: {path}")
    return value


def describe_json(path: str | Path, *, relative_to: str | Path | None = None) -> dict[str, Any]:
    artifact = Path(path)
    with artifact.open(encoding="utf-8") as handle:
        value = json.load(handle)
    row_count = len(value) if isinstance(value, list) else 1
    return descriptor_for_file(
        artifact,
        format="json",
        row_count=row_count,
        relative_to=relative_to,
    )
