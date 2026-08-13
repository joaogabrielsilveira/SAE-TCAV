import csv
import json
from pathlib import Path

from artifact_maintenance import export_csv, inspect, migrate
from artifact_storage import (
    atomic_write_json_gzip,
    read_artifact,
    read_json_file,
)
from semantic_artifacts import load_semantic_result, SemanticArtifactStore


def test_semantic_v1_migration_is_dry_run_idempotent_and_reloadable(tmp_path):
    root = tmp_path / "semantic"
    bundle = root / "abc123"
    bundle.mkdir(parents=True)
    models = [{"factor": 1, "nested": [True, None]}]
    pairs = [{"pair": 2, "score": 0.5}]
    (bundle / "manifest.json").write_text(
        json.dumps({"schema_version": "2.0", "experiment_hash": "abc123"}),
        encoding="utf-8",
    )
    (bundle / "semantic_rules.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in models),
        encoding="utf-8",
    )
    (bundle / "pair_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in pairs),
        encoding="utf-8",
    )
    (bundle / "result.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "experiment_hash": "abc123",
                "manifest": {"schema_version": "2.0", "experiment_hash": "abc123"},
                "semantic_models": models,
                "pair_results": pairs,
            }
        ),
        encoding="utf-8",
    )

    dry = migrate(root)
    assert dry["dry_run"] is True
    assert (bundle / "semantic_rules.jsonl").is_file()

    applied = migrate(root, apply=True)
    assert applied["counts"]["migrated"] == 1
    assert not (bundle / "semantic_rules.jsonl").exists()
    assert (bundle / "semantic_rules.jsonl.gz").is_file()
    result_index = read_json_file(bundle / "result.json")
    assert result_index["artifact_schema_version"] == 2
    assert "semantic_models" not in result_index
    reloaded = load_semantic_result(SemanticArtifactStore(root, "abc123"))
    assert reloaded["semantic_models"] == models
    assert reloaded["pair_results"] == pairs
    assert migrate(root, apply=True)["counts"]["already_v2"] == 1


def test_temporal_migration_and_csv_export_preserve_nested_json(tmp_path):
    run = tmp_path / "run"
    split = run / "reference_2007" / "split_42"
    split.mkdir(parents=True)
    rows = [{"flag": True, "nested": [1, None], "mapping": {"a": 2}}]
    (split / "performance.json").write_text(json.dumps(rows), encoding="utf-8")
    (split / "performance.csv").write_text("flag\nTrue\n", encoding="utf-8")
    (split / "reference_roles.json").write_text(json.dumps([]), encoding="utf-8")
    (split / "temporal_domain_map.json").write_text(json.dumps({"2007": 0}), encoding="utf-8")
    (split / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "reference_year": 2007,
                "patient_split_seed": 42,
                "requested_patient_split_seed": 42,
                "role_fingerprint": "role",
                "config_fingerprint": "config",
                "population_fingerprints": {},
                "artifacts": {"performance": "performance.json"},
            }
        ),
        encoding="utf-8",
    )
    result = migrate(run, apply=True)
    assert result["counts"]["migrated"] == 1
    manifest = read_json_file(split / "manifest.json")
    assert manifest["complete"] is True
    assert read_artifact(split, manifest["artifacts"]["performance"]) == rows
    assert not (split / "performance.csv").exists()

    aggregate = run / "aggregate"
    aggregate.mkdir()
    aggregate_descriptor = dict(manifest["artifacts"]["performance"])
    source = split / aggregate_descriptor["path"]
    target = aggregate / source.name
    target.write_bytes(source.read_bytes())
    aggregate_descriptor["path"] = f"aggregate/{target.name}"
    parent = {
        "complete": True,
        "successful_experiments": [{"manifest": str(split / "manifest.json")}],
        "aggregate_artifacts": {"performance": aggregate_descriptor},
    }
    parent_path = run / "parent_manifest.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    exported = export_csv(parent_path, tmp_path / "csv")
    assert exported["files"]
    with (tmp_path / "csv" / "aggregate" / "performance.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["nested"]) == [1, None]
    assert json.loads(row["mapping"]) == {"a": 2}


def test_temporal_migration_keeps_non_table_arrays_as_json(tmp_path):
    split = tmp_path / "run" / "reference_2007" / "split_42"
    split.mkdir(parents=True)
    values = [1, True, None]
    (split / "values.json").write_text(json.dumps(values), encoding="utf-8")
    (split / "values.csv").write_text("value\n1\n", encoding="utf-8")
    (split / "manifest.json").write_text(
        json.dumps(
            {
                "reference_year": 2007,
                "patient_split_seed": 42,
                "requested_patient_split_seed": 42,
                "role_fingerprint": "role",
                "artifacts": {"values": "values.json"},
            }
        ),
        encoding="utf-8",
    )

    assert migrate(tmp_path / "run", apply=True)["counts"]["migrated"] == 1
    manifest = read_json_file(split / "manifest.json")
    descriptor = manifest["artifacts"]["values"]
    assert descriptor["format"] == "json"
    assert descriptor["row_count"] == len(values)
    assert read_artifact(split, descriptor) == values
    assert not (split / "values.csv").exists()


def test_inspect_reports_exact_cache_reference_health(tmp_path):
    root = tmp_path / "artifacts"
    cache = root / "split" / "cache"
    entry = cache / "semantic_selection" / "key-1"
    entry.mkdir(parents=True)
    atomic_write_json_gzip(
        entry / "manifest.json.gz",
        {"key": "key-1", "payloads": {}, "complete": True},
    )
    refs = root / "split" / "semantic" / "cache_refs.json.gz"
    refs.parent.mkdir(parents=True)
    atomic_write_json_gzip(
        refs,
        {
            "cache_root": str(cache.resolve()),
            "entries": [{"stage": "semantic_selection", "key": "key-1"}],
        },
    )

    report = inspect(root)
    assert report["cache_manifests_gzip"] == 1
    assert report["cache_refs_gzip"] == 1
    assert report["cache_reference_health"] == [
        {
            "cache_root": str(cache.resolve()),
            "valid_reference_file_count": 1,
            "ignored_reference_file_count": 0,
            "referenced_key_count": 1,
            "missing_referenced_target_count": 0,
            "unreferenced_key_count": 0,
        }
    ]
