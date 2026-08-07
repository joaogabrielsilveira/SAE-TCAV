"""Shared, stage-aware cache for comparison experiments.

The module deliberately exposes one main operation, :meth:`ComparisonCache.resolve`.
Key construction, locking, validation, atomic publication, quarantine, and
reporting remain implementation details behind that seam.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Generic, Literal, Mapping, Sequence, TypeVar
import uuid

import numpy as np

from semantic_artifacts import stable_hash
from artifact_storage import (
    atomic_write_json,
    atomic_write_json_gzip,
    read_json_file,
)


T = TypeVar("T")
CACHE_SCHEMA_VERSION = 2

_FORCE_GROUPS = {
    "prepared": {"prepared"},
    "splits": {"splits"},
    "tabpfn": {
        "tabpfn_fit",
        "walkforward",
        "embeddings_raw",
        "embeddings_scaled",
        "tcav_gradients",
        "tcav_factor",
    },
    "embeddings": {"embeddings_raw", "embeddings_scaled"},
    "sae": {"sae_model"},
    "activations": {"sae_activations"},
    "matching": {"matching_pair"},
    "functional": {
        "high_precision_rule",
        "forced_rule",
        "cav",
        "tcav_gradients",
        "tcav_factor",
    },
    "semantic": {
        "semantic_bootstrap",
        "semantic_families",
        "semantic_selection",
    },
}
FORCE_STAGE_CHOICES = tuple(_FORCE_GROUPS)


@dataclass(frozen=True)
class CacheResult(Generic[T]):
    value: T
    stage: str
    item: str
    key: str
    artifact_path: Path | None
    status: Literal["hit", "miss", "forced", "disabled"]
    reason: str
    output_fingerprints: dict[str, str]


@dataclass(frozen=True)
class CacheEvent:
    stage: str
    item: str
    key: str
    status: str
    reason: str
    artifact_path: str | None
    output_fingerprints: dict[str, str]


class ComparisonCache:
    """Resolve immutable stage results from a shared local cache."""

    def __init__(
        self,
        root: str | Path,
        *,
        enabled: bool = True,
        verification: Literal["manifest", "checksum"] = "checksum",
        forced_stages: Sequence[str] = (),
    ) -> None:
        if verification not in {"manifest", "checksum"}:
            raise ValueError("cache verification must be 'manifest' or 'checksum'")
        unknown = set(forced_stages) - set(FORCE_STAGE_CHOICES)
        if unknown:
            raise ValueError(f"Unknown forced cache stages: {sorted(unknown)}")
        self.root = Path(root)
        self.enabled = bool(enabled)
        self.verification = verification
        self.forced_stages = frozenset(forced_stages)
        self.events: list[CacheEvent] = []
        self.invalid_entries = 0

    def resolve(
        self,
        *,
        stage: str,
        item: str,
        dependencies: Mapping[str, Any],
        source_fingerprint: str,
        load: Callable[[Path], T],
        compute: Callable[[], T],
        store: Callable[[Path, T], None],
        validate: Callable[[T], None] | None = None,
        fingerprint: Callable[[T], Mapping[str, str]] | None = None,
        environment_fingerprint: str | None = None,
        stage_schema_version: int = 1,
        ignore_store_errors: bool = False,
    ) -> CacheResult[T]:
        """Load or compute one stage value.

        Callbacks operate only on the entry directory. ``store`` writes payloads;
        this module writes and validates the manifest.
        """

        identity = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "stage_schema_version": int(stage_schema_version),
            "stage": stage,
            "dependencies": dependencies,
            "source_fingerprint": source_fingerprint,
            "environment_fingerprint": environment_fingerprint,
        }
        key = stable_hash(identity)
        forced = self._stage_forced(stage)
        if not self.enabled:
            value = compute()
            self._validate(value, validate)
            return self._finish(
                value,
                stage,
                item,
                key,
                None,
                "disabled",
                "cache_disabled",
                self._fingerprints(value, fingerprint),
            )

        entry = self.root / stage / key
        lock_path = self.root / ".locks" / stage / f"{key}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            manifest = self._read_valid_manifest(entry, stage, key)
            if manifest is not None and not forced:
                try:
                    value = load(entry)
                    self._validate(value, validate)
                except Exception as error:
                    self.invalid_entries += 1
                    self._quarantine(entry, f"load-{type(error).__name__}")
                else:
                    return self._finish(
                        value,
                        stage,
                        item,
                        key,
                        entry,
                        "hit",
                        "valid_entry",
                        dict(manifest.get("output_fingerprints", {})),
                    )

            value = compute()
            self._validate(value, validate)
            output_fingerprints = self._fingerprints(value, fingerprint)

            if forced and manifest is not None:
                canonical = dict(manifest.get("output_fingerprints", {}))
                reason = (
                    "forced_output_matches_canonical"
                    if canonical == output_fingerprints
                    else "forced_output_differs_from_canonical"
                )
                return self._finish(
                    value,
                    stage,
                    item,
                    key,
                    None,
                    "forced",
                    reason,
                    output_fingerprints,
                )

            entry.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{key}.", dir=str(entry.parent))
            )
            try:
                try:
                    store(temporary, value)
                except Exception:
                    if not ignore_store_errors:
                        raise
                    return self._finish(
                        value,
                        stage,
                        item,
                        key,
                        None,
                        "forced" if forced else "miss",
                        "store_unsupported",
                        output_fingerprints,
                    )
                payloads = self._payload_manifest(temporary)
                manifest_value = {
                    **identity,
                    "item": item,
                    "key": key,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "payloads": payloads,
                    "output_fingerprints": output_fingerprints,
                    "complete": True,
                }
                atomic_write_json_gzip(
                    temporary / "manifest.json.gz", manifest_value
                )
                if entry.exists():
                    # Defensive only; lock normally makes this impossible.
                    shutil.rmtree(temporary)
                else:
                    os.replace(temporary, entry)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary, ignore_errors=True)

            return self._finish(
                value,
                stage,
                item,
                key,
                entry,
                "forced" if forced else "miss",
                "forced_new_entry" if forced else "entry_missing",
                output_fingerprints,
            )

    def summary(self) -> dict[str, Any]:
        counts = Counter(event.status for event in self.events)
        by_stage: dict[str, Counter[str]] = defaultdict(Counter)
        for event in self.events:
            by_stage[event.stage][event.status] += 1
        return {
            "root": str(self.root),
            "hits": counts["hit"],
            "misses": counts["miss"],
            "forced": counts["forced"],
            "disabled": counts["disabled"],
            "invalid": self.invalid_entries,
            "by_stage": {
                stage: dict(sorted(values.items()))
                for stage, values in sorted(by_stage.items())
            },
        }

    def write_refs(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.name.endswith(".json"):
            destination = destination.with_name(destination.name + ".gz")
        elif not destination.name.endswith(".json.gz"):
            raise ValueError("cache reference files must end in .json or .json.gz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json_gzip(
            destination,
            {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "cache_root": str(self.root.resolve()),
                "entries": [asdict(event) for event in self.events],
            },
        )
        return destination

    def _stage_forced(self, stage: str) -> bool:
        return any(stage in _FORCE_GROUPS[group] for group in self.forced_stages)

    def _read_valid_manifest(
        self, entry: Path, stage: str, key: str
    ) -> dict[str, Any] | None:
        manifest_path = _manifest_path(entry)
        if manifest_path is None:
            return None
        try:
            manifest = read_json_file(manifest_path)
            if (
                manifest.get("cache_schema_version") != CACHE_SCHEMA_VERSION
                or manifest.get("stage") != stage
                or manifest.get("key") != key
                or manifest.get("complete") is not True
            ):
                raise ValueError("manifest identity mismatch")
            manifest_identity = {
                name: manifest.get(name)
                for name in (
                    "cache_schema_version",
                    "stage_schema_version",
                    "stage",
                    "dependencies",
                    "source_fingerprint",
                    "environment_fingerprint",
                )
            }
            if stable_hash(manifest_identity) != key:
                raise ValueError("manifest dependencies do not match key")
            for relative, metadata in manifest.get("payloads", {}).items():
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(f"unsafe payload path {relative}")
                payload = entry / relative_path
                if not payload.is_file():
                    raise ValueError(f"missing payload {relative}")
                if payload.stat().st_size != int(metadata["size_bytes"]):
                    raise ValueError(f"payload size mismatch {relative}")
                if (
                    self.verification == "checksum"
                    and _file_sha256(payload) != metadata["sha256"]
                ):
                    raise ValueError(f"payload checksum mismatch {relative}")
            return manifest
        except Exception as error:
            self.invalid_entries += 1
            self._quarantine(entry, f"invalid-{type(error).__name__}")
            return None

    def _quarantine(self, entry: Path, reason: str) -> None:
        if not entry.exists():
            return
        quarantine = self.root / "quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        target = quarantine / (
            f"{entry.parent.name}-{entry.name}-{reason}-{uuid.uuid4().hex[:8]}"
        )
        os.replace(entry, target)

    @staticmethod
    def _validate(value: T, validate: Callable[[T], None] | None) -> None:
        if validate is not None:
            validate(value)

    @staticmethod
    def _fingerprints(
        value: T, fingerprint: Callable[[T], Mapping[str, str]] | None
    ) -> dict[str, str]:
        return dict(fingerprint(value)) if fingerprint is not None else {}

    def _finish(
        self,
        value: T,
        stage: str,
        item: str,
        key: str,
        artifact_path: Path | None,
        status: Literal["hit", "miss", "forced", "disabled"],
        reason: str,
        output_fingerprints: dict[str, str],
    ) -> CacheResult[T]:
        event = CacheEvent(
            stage=stage,
            item=item,
            key=key,
            status=status,
            reason=reason,
            artifact_path=str(artifact_path) if artifact_path is not None else None,
            output_fingerprints=output_fingerprints,
        )
        self.events.append(event)
        return CacheResult(
            value=value,
            stage=stage,
            item=item,
            key=key,
            artifact_path=artifact_path,
            status=status,
            reason=reason,
            output_fingerprints=output_fingerprints,
        )

    @staticmethod
    def _payload_manifest(directory: Path) -> dict[str, dict[str, Any]]:
        return {
            str(path.relative_to(directory)): {
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name not in {"manifest.json", "manifest.json.gz"}
        }

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        atomic_write_json(path, value, compact=False)


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_entries(root: Path):
    for stage in sorted(root.iterdir()) if root.exists() else ():
        if not stage.is_dir() or stage.name in {".locks", "quarantine"}:
            continue
        for entry in sorted(stage.iterdir()):
            if entry.is_dir() and _manifest_path(entry) is not None:
                yield stage.name, entry


def _manifest_path(entry: Path) -> Path | None:
    for name in ("manifest.json.gz", "manifest.json"):
        path = entry / name
        if path.is_file():
            return path
    return None


def _inspect(root: Path) -> dict[str, Any]:
    stages: dict[str, dict[str, int]] = defaultdict(
        lambda: {"entries": 0, "size_bytes": 0}
    )
    for stage, entry in _cache_entries(root):
        stages[stage]["entries"] += 1
        stages[stage]["size_bytes"] += sum(
            path.stat().st_size for path in entry.rglob("*") if path.is_file()
        )
    health = _reference_health(root)
    return {
        "root": str(root),
        "entries": sum(value["entries"] for value in stages.values()),
        "size_bytes": sum(value["size_bytes"] for value in stages.values()),
        "stages": dict(sorted(stages.items())),
        "references": health,
    }


def _reference_search_roots(root: Path) -> tuple[Path, ...]:
    candidates = {root.parent}
    if root.name == "v2" or root.parent.name in {"_cache", "cache"}:
        candidates.add(root.parent.parent)
    return tuple(sorted(candidates, key=lambda value: str(value)))


def _reference_health(root: Path) -> dict[str, Any]:
    referenced: set[tuple[str, str]] = set()
    valid_files: list[str] = []
    ignored_files: list[str] = []
    refs_paths: set[Path] = set()
    for search_root in _reference_search_roots(root):
        if not search_root.exists():
            continue
        refs_paths.update(search_root.rglob("cache_refs.json"))
        refs_paths.update(search_root.rglob("cache_refs.json.gz"))
    resolved_root = root.resolve()
    for refs_path in sorted(refs_paths):
        try:
            refs = read_json_file(refs_path)
            declared = Path(str(refs["cache_root"])).resolve()
            if declared != resolved_root:
                ignored_files.append(str(refs_path))
                continue
            for entry in refs.get("entries", []):
                referenced.add((str(entry["stage"]), str(entry["key"])))
            valid_files.append(str(refs_path))
        except (OSError, ValueError, KeyError, TypeError):
            ignored_files.append(str(refs_path))
            continue
    existing = {(stage, entry.name) for stage, entry in _cache_entries(root)}
    missing = referenced - existing
    unreferenced = existing - referenced
    return {
        "valid_reference_file_count": len(valid_files),
        "valid_reference_files": valid_files,
        "ignored_reference_file_count": len(ignored_files),
        "referenced_key_count": len(referenced),
        "missing_referenced_target_count": len(missing),
        "missing_referenced_targets": [
            {"stage": stage, "key": key} for stage, key in sorted(missing)
        ],
        "unreferenced_key_count": len(unreferenced),
        "unreferenced_keys": [
            {"stage": stage, "key": key} for stage, key in sorted(unreferenced)
        ],
    }


def _referenced_keys(root: Path) -> set[tuple[str, str]]:
    health = _reference_health(root)
    missing = {
        (row["stage"], row["key"])
        for row in health["missing_referenced_targets"]
    }
    unreferenced = {
        (row["stage"], row["key"])
        for row in health["unreferenced_keys"]
    }
    existing = {(stage, entry.name) for stage, entry in _cache_entries(root)}
    return (existing - unreferenced) | missing


def _prune(
    root: Path,
    *,
    unreferenced: bool,
    older_than_days: int | None,
    apply: bool,
    stages: Sequence[str] = (),
    unsafe_no_references: bool = False,
    final_only: bool = False,
    parent_manifest: str | Path | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    health = _reference_health(root)
    referenced = _referenced_keys(root) if unreferenced else set()
    entries = list(_cache_entries(root))
    if (
        apply
        and unreferenced
        and entries
        and health["valid_reference_file_count"] == 0
        and not unsafe_no_references
    ):
        raise RuntimeError(
            "refusing unreferenced cache pruning: nonempty cache has no valid reference files"
        )
    if final_only:
        if parent_manifest is None:
            raise ValueError("final-only pruning requires a parent manifest")
        _validate_completed_parent(Path(parent_manifest))
    selected: list[Path] = []
    size_bytes = 0
    selected_stages = set(stages)
    for stage, entry in entries:
        if selected_stages and stage not in selected_stages:
            continue
        if unreferenced and (stage, entry.name) in referenced:
            continue
        if older_than_days is not None:
            manifest_path = _manifest_path(entry)
            if manifest_path is None:
                continue
            created = datetime.fromisoformat(
                read_json_file(manifest_path)["created_at"]
            )
            if (now - created).days < older_than_days:
                continue
        selected.append(entry)
        size_bytes += sum(
            path.stat().st_size for path in entry.rglob("*") if path.is_file()
        )
    if apply:
        for entry in selected:
            shutil.rmtree(entry)
    return {
        "root": str(root),
        "dry_run": not apply,
        "entries": len(selected),
        "size_bytes": size_bytes,
        "valid_reference_file_count": health["valid_reference_file_count"],
    }


def _validate_completed_parent(path: Path) -> None:
    from artifact_storage import validate_descriptor

    parent = read_json_file(path)
    if parent.get("complete") is not True:
        raise ValueError("parent manifest is not complete")
    for descriptor in parent.get("aggregate_artifacts", {}).values():
        if not isinstance(descriptor, Mapping):
            raise ValueError("parent uses an unvalidated legacy aggregate artifact")
        validate_descriptor(path.parent, descriptor)
    for entry in parent.get("successful_experiments", []):
        manifest_path = Path(entry["manifest"])
        if not manifest_path.is_absolute():
            manifest_path = path.parent / manifest_path
        manifest = read_json_file(manifest_path)
        if manifest.get("complete") is not True:
            raise ValueError(f"split manifest is not complete: {manifest_path}")
        for descriptor in manifest.get("artifacts", {}).values():
            if not isinstance(descriptor, Mapping):
                raise ValueError(f"split uses an unvalidated legacy artifact: {manifest_path}")
            validate_descriptor(manifest_path.parent, descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--root", required=True)
    prune_parser = subparsers.add_parser("prune")
    prune_parser.add_argument("--root", required=True)
    prune_parser.add_argument("--unreferenced", action="store_true")
    prune_parser.add_argument("--older-than-days", type=int)
    prune_parser.add_argument("--stage", action="append", default=[])
    prune_parser.add_argument("--final-only", action="store_true")
    prune_parser.add_argument("--parent")
    prune_parser.add_argument("--unsafe-no-references", action="store_true")
    prune_parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.command == "inspect":
        print(json.dumps(_inspect(root), sort_keys=True, indent=2))
        return 0
    if not args.unreferenced and args.older_than_days is None and not args.stage and not args.final_only:
        parser.error("prune requires a retention policy")
    if args.older_than_days is not None and args.older_than_days < 0:
        parser.error("--older-than-days must be non-negative")
    print(
        json.dumps(
            _prune(
                root,
                unreferenced=args.unreferenced,
                older_than_days=args.older_than_days,
                apply=args.apply,
                stages=args.stage,
                unsafe_no_references=args.unsafe_no_references,
                final_only=args.final_only,
                parent_manifest=args.parent,
            ),
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
