"""Inspect, migrate, and export compact experiment artifacts.

Migration is dry-run by default.  ``--apply`` publishes one validated file or
bundle at a time, making repeated invocations safe after interruption.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

from artifact_storage import (
    ARTIFACT_SCHEMA_VERSION,
    atomic_gzip_copy,
    atomic_write_json,
    atomic_write_json_gzip,
    atomic_write_jsonl_gzip,
    canonical_json,
    describe_json,
    descriptor_for_file,
    file_sha256,
    infer_format,
    iter_artifact_rows,
    read_artifact,
    read_json_file,
    validate_descriptor,
)
from semantic_artifacts import (
    SemanticArtifactStore,
    build_semantic_result_index,
    load_semantic_result,
    semantic_bundle_descriptors,
    validate_semantic_result_index,
    write_semantic_result_index,
)


def _iter_files(root: Path):
    for directory, _, names in os.walk(root):
        base = Path(directory)
        yield base, set(names)


def _count_jsonl(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def _iter_jsonl(path: Path) -> Iterator[Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _iter_json_array(path: Path) -> Iterator[Any]:
    """Incrementally decode a top-level JSON array with bounded buffering."""

    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def fill() -> bool:
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = handle.read(1024 * 1024)
            if not chunk:
                eof = True
                return False
            buffer += chunk
            return True

        fill()
        while True:
            while position >= len(buffer) and fill():
                pass
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError(f"not a JSON array: {path}")
        position += 1
        first = True
        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if not fill():
                    raise ValueError(f"truncated JSON array: {path}")
            if buffer[position] == "]":
                return
            if not first:
                if buffer[position] != ",":
                    raise ValueError(f"invalid JSON array separator: {path}")
                position += 1
                while True:
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if position < len(buffer):
                        break
                    if not fill():
                        raise ValueError(f"truncated JSON array: {path}")
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    break
                except json.JSONDecodeError:
                    if eof or not fill():
                        raise ValueError(f"truncated JSON value: {path}")
            first = False
            yield value


def _canonical_rows(rows: Iterable[Any]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        digest.update(canonical_json(row).encode("utf-8"))
        digest.update(b"\n")
        count += 1
    return digest.hexdigest(), count


def _is_v2_semantic(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8") as handle:
            prefix = handle.read(1024 * 1024)
        if '"artifact_schema_version"' not in prefix:
            return False
        value = read_json_file(path)
        if value.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            return False
        validate_semantic_result_index(value, path.parent)
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _migrate_semantic_bundle(result_path: Path, *, apply: bool) -> dict[str, Any]:
    if _is_v2_semantic(result_path):
        index = read_json_file(result_path)
        missing_counts = [
            name for name, descriptor in index["artifacts"].items()
            if "row_count" not in descriptor
        ]
        if not missing_counts:
            return {"status": "already_v2", "path": str(result_path), "reclaimable_bytes": 0}
        if not apply:
            return {"status": "would_migrate", "path": str(result_path), "reclaimable_bytes": 0}
        updated = dict(index)
        updated["artifacts"] = {
            name: dict(descriptor)
            for name, descriptor in index["artifacts"].items()
        }
        for name in missing_counts:
            descriptor = updated["artifacts"][name]
            if descriptor.get("format") in {"jsonl", "jsonl+gzip"}:
                count = sum(
                    1 for _ in iter_artifact_rows(result_path.parent, descriptor)
                )
            else:
                value = read_artifact(result_path.parent, descriptor)
                count = len(value) if isinstance(value, list) else 1
            descriptor["row_count"] = count
        validate_semantic_result_index(updated, result_path.parent)
        atomic_write_json(result_path, updated)
        return {"status": "migrated", "path": str(result_path), "reclaimable_bytes": 0}
    rules = result_path.parent / "semantic_rules.jsonl"
    pairs = result_path.parent / "pair_results.jsonl"
    manifest_path = result_path.parent / "manifest.json"
    missing = [str(path) for path in (rules, pairs, manifest_path) if not path.is_file()]
    if missing:
        return {"status": "blocked", "path": str(result_path), "reason": f"missing {missing}"}
    reclaimable = result_path.stat().st_size + rules.stat().st_size + pairs.stat().st_size
    if not apply:
        return {"status": "would_migrate", "path": str(result_path), "reclaimable_bytes": reclaimable}

    rules_gz = rules.with_name(rules.name + ".gz")
    pairs_gz = pairs.with_name(pairs.name + ".gz")
    atomic_gzip_copy(rules_gz, rules)
    atomic_gzip_copy(pairs_gz, pairs)
    original_rules_digest, original_rules_count = _canonical_rows(
        _iter_jsonl(rules)
    )
    original_pairs_digest, original_pairs_count = _canonical_rows(
        _iter_jsonl(pairs)
    )
    rules_descriptor = descriptor_for_file(
        rules_gz,
        format="jsonl+gzip",
        row_count=original_rules_count,
        relative_to=result_path.parent,
    )
    pairs_descriptor = descriptor_for_file(
        pairs_gz,
        format="jsonl+gzip",
        row_count=original_pairs_count,
        relative_to=result_path.parent,
    )
    store = SemanticArtifactStore(result_path.parent.parent, result_path.parent.name)
    manifest_descriptor, rules_descriptor, pairs_descriptor = semantic_bundle_descriptors(
        store,
        semantic_models_descriptor=rules_descriptor,
        pair_results_descriptor=pairs_descriptor,
    )
    manifest_value = read_json_file(manifest_path)
    experiment_hash = str(manifest_value.get("experiment_hash", result_path.parent.name))
    index = build_semantic_result_index(
        store=store,
        scientific_schema_version=str(manifest_value.get("schema_version", "unknown")),
        experiment_hash=experiment_hash,
        manifest_descriptor=manifest_descriptor,
        semantic_models_descriptor=rules_descriptor,
        pair_results_descriptor=pairs_descriptor,
    )
    write_semantic_result_index(store, index)
    published_index = read_json_file(result_path)
    validate_semantic_result_index(published_index, result_path.parent)
    if _canonical_rows(iter_artifact_rows(
        result_path.parent, published_index["artifacts"]["semantic_models"]
    )) != (
        original_rules_digest, original_rules_count
    ):
        raise ValueError(f"semantic model rows changed while migrating {result_path}")
    if _canonical_rows(iter_artifact_rows(
        result_path.parent, published_index["artifacts"]["pair_results"]
    )) != (
        original_pairs_digest, original_pairs_count
    ):
        raise ValueError(f"semantic pair rows changed while migrating {result_path}")
    rules.unlink()
    pairs.unlink()
    return {"status": "migrated", "path": str(result_path), "reclaimable_bytes": reclaimable}


def _inferred_temporal_identity(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        from temporal_config import TemporalRobustnessConfig
        from temporal_production import ProductionTemporalAdapter
        from temporal_robustness import (
            _dependent_config_fingerprints,
            _scientific_config,
            _temporal_source_fingerprints,
        )
        from semantic_artifacts import stable_hash

        config_path = Path("temporal_robustness.example.json")
        if not config_path.is_file():
            return None
        config = TemporalRobustnessConfig.from_json(config_path)
        return {
            "scientific_config_fingerprint": stable_hash(_scientific_config(config)),
            "legacy_config_fingerprint": manifest.get("config_fingerprint"),
            "population_fingerprints": manifest.get("population_fingerprints"),
            "reference_year": int(manifest["reference_year"]),
            "patient_split_seed": int(manifest["patient_split_seed"]),
            "requested_patient_split_seed": int(manifest["requested_patient_split_seed"]),
            "role_fingerprint": manifest.get("role_fingerprint"),
            "dependent_config_fingerprints": _dependent_config_fingerprints(config),
            "source_fingerprints": _temporal_source_fingerprints(ProductionTemporalAdapter()),
        }
    except (ImportError, OSError, TypeError, ValueError, KeyError):
        return None


def _migrate_temporal_manifest(path: Path, *, apply: bool) -> dict[str, Any]:
    try:
        manifest = read_json_file(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"status": "blocked", "path": str(path), "reason": str(error)}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not {
        "reference_year", "patient_split_seed", "role_fingerprint"
    }.issubset(manifest):
        return {"status": "not_temporal", "path": str(path)}
    if (
        manifest.get("artifact_schema_version") == ARTIFACT_SCHEMA_VERSION
        and manifest.get("complete") is True
        and all(isinstance(value, Mapping) for value in artifacts.values())
    ):
        try:
            for descriptor in artifacts.values():
                validate_descriptor(path.parent, descriptor)
            missing_counts = [
                name for name, descriptor in artifacts.items()
                if "row_count" not in descriptor
            ]
            if not missing_counts:
                return {"status": "already_v2", "path": str(path), "reclaimable_bytes": 0}
            if not apply:
                return {"status": "would_migrate", "path": str(path), "reclaimable_bytes": 0}
            completed_artifacts = {
                name: dict(descriptor) for name, descriptor in artifacts.items()
            }
            for name in missing_counts:
                descriptor = completed_artifacts[name]
                if descriptor.get("format") in {"jsonl", "jsonl+gzip"}:
                    row_count = sum(
                        1 for _ in iter_artifact_rows(path.parent, descriptor)
                    )
                else:
                    value = read_artifact(path.parent, descriptor)
                    row_count = len(value) if isinstance(value, list) else 1
                descriptor["row_count"] = row_count
            updated = dict(manifest)
            updated["artifacts"] = completed_artifacts
            atomic_write_json(path, updated)
            for descriptor in completed_artifacts.values():
                validate_descriptor(path.parent, descriptor)
            return {"status": "migrated", "path": str(path), "reclaimable_bytes": 0}
        except (OSError, ValueError, KeyError, TypeError) as error:
            return {"status": "blocked", "path": str(path), "reason": str(error)}

    candidates: dict[str, str | Mapping[str, Any]] = dict(artifacts)
    for name in ("reference_roles", "temporal_domain_map"):
        candidate = path.parent / f"{name}.json"
        if candidate.is_file():
            candidates.setdefault(name, candidate.name)
    reclaimable = 0
    for name, entry in candidates.items():
        source = path.parent / entry if isinstance(entry, str) else None
        if source is not None and source.is_file():
            csv_copy = path.parent / f"{name}.csv"
            reclaimable += source.stat().st_size
            if csv_copy.is_file():
                reclaimable += csv_copy.stat().st_size
    if not apply:
        return {"status": "would_migrate", "path": str(path), "reclaimable_bytes": reclaimable}

    migrated: dict[str, dict[str, Any]] = {}
    remove_after_publish: list[Path] = []
    for name, entry in sorted(candidates.items()):
        if isinstance(entry, Mapping):
            validate_descriptor(path.parent, entry)
            migrated[name] = dict(entry)
            continue
        source = path.parent / entry
        if not source.is_file():
            raise ValueError(f"missing temporal artifact {source}")
        with source.open(encoding="utf-8") as handle:
            first = ""
            while not first:
                character = handle.read(1)
                if not character:
                    break
                if not character.isspace():
                    first = character
        if first == "[":
            probe = _iter_json_array(source)
            try:
                first_row = next(probe)
            except StopIteration:
                first_row = None
                is_row_table = True
            else:
                is_row_table = isinstance(first_row, Mapping)
            if not is_row_table:
                # Artifact v2 reserves JSONL for row tables.  A top-level list
                # of scalars remains an ordinary JSON value.
                migrated[name] = describe_json(source, relative_to=path.parent)
                csv_copy = path.parent / f"{name}.csv"
                if csv_copy.is_file():
                    remove_after_publish.append(csv_copy)
                continue

            def mapping_rows() -> Iterator[Mapping[str, Any]]:
                if first_row is not None:
                    yield first_row
                for row in probe:
                    if not isinstance(row, Mapping):
                        raise ValueError(f"mixed non-mapping row in {source}")
                    yield row

            original_digest, original_count = _canonical_rows(_iter_json_array(source))
            destination = path.parent / f"{name}.jsonl.gz"
            descriptor = atomic_write_jsonl_gzip(destination, mapping_rows())
            descriptor["path"] = destination.name
            new_digest, new_count = _canonical_rows(
                iter_artifact_rows(path.parent, descriptor)
            )
            if (new_digest, new_count) != (original_digest, original_count):
                raise ValueError(f"typed rows changed while migrating {source}")
            migrated[name] = descriptor
            remove_after_publish.append(source)
            csv_copy = path.parent / f"{name}.csv"
            if csv_copy.is_file():
                remove_after_publish.append(csv_copy)
        else:
            migrated[name] = describe_json(source, relative_to=path.parent)

    updated = dict(manifest)
    updated.update(
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "cache_schema_version": 2,
            "complete": True,
            "artifacts": migrated,
        }
    )
    identity = _inferred_temporal_identity(updated)
    if identity is not None:
        updated["scientific_config_fingerprint"] = identity["scientific_config_fingerprint"]
        updated["dependent_config_fingerprints"] = identity["dependent_config_fingerprints"]
        updated["source_fingerprints"] = identity["source_fingerprints"]
        updated["scientific_identity"] = identity
    atomic_write_json(path, updated)
    published = read_json_file(path)
    for descriptor in published["artifacts"].values():
        validate_descriptor(path.parent, descriptor)
    for source in remove_after_publish:
        source.unlink(missing_ok=True)
    return {"status": "migrated", "path": str(path), "reclaimable_bytes": reclaimable}


def _validate_cache_manifest(entry: Path, manifest: Mapping[str, Any]) -> None:
    from semantic_artifacts import stable_hash

    identity = {
        name: manifest.get(name)
        for name in (
            "cache_schema_version", "stage_schema_version", "stage", "dependencies",
            "source_fingerprint", "environment_fingerprint",
        )
    }
    if manifest.get("complete") is not True or stable_hash(identity) != manifest.get("key"):
        raise ValueError(f"cache manifest identity mismatch: {entry}")
    for relative, metadata in manifest.get("payloads", {}).items():
        payload = entry / relative
        if not payload.is_file() or payload.stat().st_size != int(metadata["size_bytes"]):
            raise ValueError(f"cache payload mismatch: {payload}")
        if file_sha256(payload) != metadata["sha256"]:
            raise ValueError(f"cache payload checksum mismatch: {payload}")


def _migrate_cache_manifest(
    path: Path, *, apply: bool, manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    try:
        if manifest is None:
            manifest = read_json_file(path)
        if "key" not in manifest or "payloads" not in manifest:
            return {"status": "not_cache", "path": str(path)}
        _validate_cache_manifest(path.parent, manifest)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return {"status": "blocked", "path": str(path), "reason": str(error)}
    if path.name.endswith(".gz"):
        return {"status": "already_v2", "path": str(path), "reclaimable_bytes": 0}
    if not apply:
        return {"status": "would_migrate", "path": str(path), "reclaimable_bytes": path.stat().st_size}
    size = path.stat().st_size
    destination = path.with_name("manifest.json.gz")
    atomic_write_json_gzip(destination, manifest)
    reloaded = read_json_file(destination)
    if reloaded != manifest:
        raise ValueError(f"cache manifest changed while migrating {path}")
    path.unlink()
    return {"status": "migrated", "path": str(path), "reclaimable_bytes": size}


def _migrate_refs(path: Path, *, apply: bool) -> dict[str, Any]:
    try:
        refs = read_json_file(path)
        if "cache_root" not in refs or "entries" not in refs:
            return {"status": "not_refs", "path": str(path)}
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"status": "blocked", "path": str(path), "reason": str(error)}
    if path.name.endswith(".gz"):
        return {"status": "already_v2", "path": str(path), "reclaimable_bytes": 0}
    size = path.stat().st_size
    if not apply:
        return {"status": "would_migrate", "path": str(path), "reclaimable_bytes": size}
    destination = path.with_name(path.name + ".gz")
    atomic_write_json_gzip(destination, refs)
    if read_json_file(destination) != refs:
        raise ValueError(f"cache refs changed while migrating {path}")
    path.unlink()
    return {"status": "migrated", "path": str(path), "reclaimable_bytes": size}


def inspect(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    report: dict[str, Any] = {
        "root": str(base),
        "semantic_v1": 0,
        "semantic_v2": 0,
        "temporal_incomplete_or_legacy": 0,
        "temporal_complete_v2": 0,
        "cache_manifests_plain": 0,
        "cache_manifests_gzip": 0,
        "cache_refs_plain": 0,
        "cache_refs_gzip": 0,
        "cache_reference_health": [],
        "expected_reclaimable_bytes": 0,
        "migration_blockers": [],
    }
    cache_roots: set[Path] = set()
    for directory, names in _iter_files(base):
        if "result.json" in names:
            path = directory / "result.json"
            if _is_v2_semantic(path):
                report["semantic_v2"] += 1
            else:
                report["semantic_v1"] += 1
                report["expected_reclaimable_bytes"] += sum(
                    candidate.stat().st_size
                    for candidate in (
                        path,
                        directory / "semantic_rules.jsonl",
                        directory / "pair_results.jsonl",
                    )
                    if candidate.is_file()
                )
        if "manifest.json" in names:
            path = directory / "manifest.json"
            try:
                value = read_json_file(path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                report["migration_blockers"].append({"path": str(path), "reason": str(error)})
                continue
            if "key" in value and "payloads" in value:
                report["cache_manifests_plain"] += 1
                report["expected_reclaimable_bytes"] += path.stat().st_size
                cache_roots.add(directory.parent.parent.resolve())
            elif "reference_year" in value and "artifacts" in value:
                if (
                    value.get("artifact_schema_version") == 2
                    and value.get("complete") is True
                    and all(
                        isinstance(descriptor, Mapping)
                        and "row_count" in descriptor
                        for descriptor in value["artifacts"].values()
                    )
                ):
                    try:
                        for descriptor in value["artifacts"].values():
                            if not isinstance(descriptor, Mapping):
                                raise ValueError("legacy artifact entry in v2 manifest")
                            validate_descriptor(directory, descriptor)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        report["temporal_incomplete_or_legacy"] += 1
                        report["migration_blockers"].append(
                            {"path": str(path), "reason": str(error)}
                        )
                    else:
                        report["temporal_complete_v2"] += 1
                else:
                    report["temporal_incomplete_or_legacy"] += 1
                    legacy_paths: set[Path] = set()
                    for name, entry in value.get("artifacts", {}).items():
                        if isinstance(entry, str):
                            legacy_paths.add(directory / entry)
                            legacy_paths.add(directory / f"{name}.csv")
                    for name in ("reference_roles", "temporal_domain_map"):
                        legacy_paths.add(directory / f"{name}.json")
                        legacy_paths.add(directory / f"{name}.csv")
                    report["expected_reclaimable_bytes"] += sum(
                        candidate.stat().st_size
                        for candidate in legacy_paths
                        if candidate.is_file()
                    )
        if "manifest.json.gz" in names:
            report["cache_manifests_gzip"] += 1
            cache_roots.add(directory.parent.parent.resolve())
        if "cache_refs.json" in names:
            report["cache_refs_plain"] += 1
            refs_path = directory / "cache_refs.json"
            report["expected_reclaimable_bytes"] += refs_path.stat().st_size
            try:
                cache_roots.add(Path(str(read_json_file(refs_path)["cache_root"])).resolve())
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                report["migration_blockers"].append(
                    {"path": str(refs_path), "reason": str(error)}
                )
        if "cache_refs.json.gz" in names:
            report["cache_refs_gzip"] += 1
            refs_path = directory / "cache_refs.json.gz"
            try:
                cache_roots.add(Path(str(read_json_file(refs_path)["cache_root"])).resolve())
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                report["migration_blockers"].append(
                    {"path": str(refs_path), "reason": str(error)}
                )
    from comparison_cache import _reference_health

    for cache_root in sorted(cache_roots, key=str):
        if not cache_root.is_dir():
            report["migration_blockers"].append(
                {"path": str(cache_root), "reason": "declared cache root is missing"}
            )
            continue
        health = _reference_health(cache_root)
        report["cache_reference_health"].append(
            {
                "cache_root": str(cache_root),
                "valid_reference_file_count": health["valid_reference_file_count"],
                "ignored_reference_file_count": health["ignored_reference_file_count"],
                "referenced_key_count": health["referenced_key_count"],
                "missing_referenced_target_count": health[
                    "missing_referenced_target_count"
                ],
                "unreferenced_key_count": health["unreferenced_key_count"],
            }
        )
    return report


def migrate(root: str | Path, *, apply: bool = False) -> dict[str, Any]:
    base = Path(root)
    counts: dict[str, int] = {}
    reclaimable = 0
    blockers = []
    for directory, names in _iter_files(base):
        operations = []
        if "result.json" in names:
            operations.append((_migrate_semantic_bundle, directory / "result.json"))
        if "manifest.json" in names:
            manifest_path = directory / "manifest.json"
            try:
                value = read_json_file(manifest_path)
            except (OSError, ValueError, json.JSONDecodeError):
                value = {}
            if "key" in value and "payloads" in value:
                operations.append((
                    lambda current, *, apply, cached=value: _migrate_cache_manifest(
                        current, apply=apply, manifest=cached
                    ),
                    manifest_path,
                ))
            elif "reference_year" in value and "artifacts" in value:
                operations.append((_migrate_temporal_manifest, manifest_path))
        if "cache_refs.json" in names:
            operations.append((_migrate_refs, directory / "cache_refs.json"))
        for operation, path in operations:
            try:
                result = operation(path, apply=apply)
            except Exception as error:
                result = {"status": "blocked", "path": str(path), "reason": str(error)}
            status = result["status"]
            counts[status] = counts.get(status, 0) + 1
            reclaimable += int(result.get("reclaimable_bytes", 0))
            if status == "blocked":
                blockers.append({"path": result["path"], "reason": result["reason"]})
    return {
        "root": str(base),
        "dry_run": not apply,
        "counts": dict(sorted(counts.items())),
        "source_bytes_eligible_for_reclaim": reclaimable,
        "migration_blockers": blockers,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def _export_rows(path: Path, row_factory) -> int:
    fieldnames: set[str] = set()
    count = 0
    for row in row_factory():
        if not isinstance(row, Mapping):
            raise ValueError(f"CSV export requires mapping rows: {path}")
        fieldnames.update(str(key) for key in row)
        count += 1
    ordered = sorted(fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ordered)
            if ordered:
                writer.writeheader()
                for row in row_factory():
                    writer.writerow({key: _csv_value(row.get(key)) for key in ordered})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def export_csv(parent_manifest: str | Path, output: str | Path) -> dict[str, Any]:
    parent_path = Path(parent_manifest)
    parent = read_json_file(parent_path)
    output_root = Path(output)
    files = []
    for name, descriptor in sorted(parent.get("aggregate_artifacts", {}).items()):
        try:
            row_factory = lambda d=descriptor: iter_artifact_rows(parent_path.parent, d)
            count = _export_rows(output_root / "aggregate" / f"{name}.csv", row_factory)
        except ValueError:
            continue
        files.append({"path": str(output_root / "aggregate" / f"{name}.csv"), "row_count": count})
    for experiment in parent.get("successful_experiments", []):
        split_manifest_path = Path(experiment["manifest"])
        if not split_manifest_path.is_absolute():
            split_manifest_path = parent_path.parent / split_manifest_path
        split = read_json_file(split_manifest_path)
        relative = Path(f"reference_{split['reference_year']}") / f"split_{split['patient_split_seed']}"
        for name, descriptor in sorted(split.get("artifacts", {}).items()):
            try:
                row_factory = lambda d=descriptor, b=split_manifest_path.parent: iter_artifact_rows(b, d)
                target = output_root / relative / f"{name}.csv"
                count = _export_rows(target, row_factory)
            except ValueError:
                continue
            files.append({"path": str(target), "row_count": count})
    return {"parent": str(parent_path), "output": str(output_root), "files": files}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--root", required=True)
    migrate_parser = commands.add_parser("migrate")
    migrate_parser.add_argument("--root", required=True)
    migrate_parser.add_argument("--apply", action="store_true")
    export_parser = commands.add_parser("export-csv")
    export_parser.add_argument("--parent", required=True)
    export_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        result = inspect(args.root)
    elif args.command == "migrate":
        result = migrate(args.root, apply=args.apply)
    else:
        result = export_csv(args.parent, args.output)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
