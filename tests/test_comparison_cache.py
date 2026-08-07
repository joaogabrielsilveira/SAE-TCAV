import json
import multiprocessing
from pathlib import Path
import time

import pytest

from comparison_cache import ComparisonCache, _inspect, _prune
from artifact_storage import atomic_write_json_gzip, read_json_file


def _concurrent_cache_worker(root, marker):
    cache = ComparisonCache(root)

    def compute():
        with Path(marker).open("a", encoding="utf-8") as handle:
            handle.write("computed\n")
        time.sleep(0.1)
        return "shared"

    cache.resolve(
        stage="prepared",
        item="concurrent",
        dependencies={"value": 1},
        source_fingerprint="source",
        load=lambda directory: (directory / "value.txt").read_text(
            encoding="utf-8"
        ),
        compute=compute,
        store=lambda directory, value: (directory / "value.txt").write_text(
            value, encoding="utf-8"
        ),
        fingerprint=lambda value: {"value": value},
    )


def _resolve(
    cache,
    calls,
    *,
    dependency=1,
    stage="prepared",
    item="item",
    value="value",
):
    def compute():
        calls.append(("compute", value))
        return value

    def store(directory, result):
        (directory / "value.txt").write_text(result, encoding="utf-8")

    return cache.resolve(
        stage=stage,
        item=item,
        dependencies={"dependency": dependency},
        source_fingerprint="source",
        load=lambda directory: (directory / "value.txt").read_text(
            encoding="utf-8"
        ),
        compute=compute,
        store=store,
        validate=lambda result: (
            None
            if isinstance(result, str)
            else (_ for _ in ()).throw(ValueError("not text"))
        ),
        fingerprint=lambda result: {"value": result},
    )


def test_cache_resolves_miss_then_hit_and_dependency_change(tmp_path):
    calls = []
    cache = ComparisonCache(tmp_path)

    first = _resolve(cache, calls)
    second = _resolve(cache, calls)
    changed = _resolve(cache, calls, dependency=2)

    assert first.status == "miss"
    assert second.status == "hit"
    assert changed.status == "miss"
    assert calls == [("compute", "value"), ("compute", "value")]
    assert first.key == second.key
    assert changed.key != first.key
    assert cache.summary()["hits"] == 1
    assert cache.summary()["misses"] == 2


def test_item_label_does_not_change_scientific_cache_identity(tmp_path):
    calls = []
    cache = ComparisonCache(tmp_path)

    first = _resolve(cache, calls, item="run-index:0")
    second = _resolve(cache, calls, item="run-index:3")

    assert first.key == second.key
    assert second.status == "hit"
    assert calls == [("compute", "value")]


def test_checksum_mismatch_quarantines_and_recomputes(tmp_path):
    calls = []
    cache = ComparisonCache(tmp_path)
    first = _resolve(cache, calls)
    (first.artifact_path / "value.txt").write_text("corrupt", encoding="utf-8")

    second = _resolve(cache, calls)

    assert second.status == "miss"
    assert second.value == "value"
    assert len(list((tmp_path / "quarantine").iterdir())) == 1
    assert calls == [("compute", "value"), ("compute", "value")]
    assert cache.summary()["invalid"] == 1


def test_forced_stage_does_not_replace_valid_canonical_entry(tmp_path):
    calls = []
    _resolve(ComparisonCache(tmp_path), calls, value="canonical")
    forced = _resolve(
        ComparisonCache(tmp_path, forced_stages=("prepared",)),
        calls,
        value="fresh",
    )
    normal = _resolve(ComparisonCache(tmp_path), calls, value="ignored")

    assert forced.status == "forced"
    assert forced.reason == "forced_output_differs_from_canonical"
    assert forced.artifact_path is None
    assert normal.status == "hit"
    assert normal.value == "canonical"


def test_forced_equal_output_verifies_canonical_entry(tmp_path):
    calls = []
    _resolve(ComparisonCache(tmp_path), calls, value="same")

    forced = _resolve(
        ComparisonCache(tmp_path, forced_stages=("prepared",)),
        calls,
        value="same",
    )

    assert forced.status == "forced"
    assert forced.reason == "forced_output_matches_canonical"


def test_disabled_cache_reads_and_writes_nothing(tmp_path):
    calls = []
    result = _resolve(ComparisonCache(tmp_path, enabled=False), calls)

    assert result.status == "disabled"
    assert result.artifact_path is None
    assert list(tmp_path.iterdir()) == []


def test_incomplete_manifest_recomputes(tmp_path):
    calls = []
    cache = ComparisonCache(tmp_path)
    first = _resolve(cache, calls)
    manifest_path = first.artifact_path / "manifest.json.gz"
    manifest = read_json_file(manifest_path)
    manifest["complete"] = False
    atomic_write_json_gzip(manifest_path, manifest)

    second = _resolve(cache, calls)

    assert second.status == "miss"
    assert second.value == "value"


def test_failed_store_publishes_no_entry(tmp_path):
    cache = ComparisonCache(tmp_path)

    def fail_store(directory, value):
        (directory / "partial.txt").write_text(value, encoding="utf-8")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        cache.resolve(
            stage="prepared",
            item="item",
            dependencies={"dependency": 1},
            source_fingerprint="source",
            load=lambda directory: "unused",
            compute=lambda: "value",
            store=fail_store,
            fingerprint=lambda value: {"value": value},
        )

    assert _inspect(tmp_path)["entries"] == 0


def test_cache_refs_are_serializable(tmp_path):
    calls = []
    cache = ComparisonCache(tmp_path / "cache")
    _resolve(cache, calls)

    refs = cache.write_refs(tmp_path / "run" / "cache_refs.json")
    payload = read_json_file(refs)

    assert payload["cache_schema_version"] == 2
    assert payload["entries"][0]["stage"] == "prepared"
    assert refs.name == "cache_refs.json.gz"


def test_unknown_force_stage_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown forced"):
        ComparisonCache(tmp_path, forced_stages=("unknown",))


def test_inspect_and_prune_are_dry_run_by_default(tmp_path):
    calls = []
    root = tmp_path / "cache"
    cache = ComparisonCache(root)
    result = _resolve(cache, calls)

    report = _inspect(root)
    dry_run = _prune(
        root,
        unreferenced=False,
        older_than_days=0,
        apply=False,
    )

    assert report["entries"] == 1
    assert dry_run["entries"] == 1
    assert dry_run["dry_run"] is True
    assert result.artifact_path.exists()

    applied = _prune(
        root,
        unreferenced=False,
        older_than_days=0,
        apply=True,
    )
    assert applied["dry_run"] is False
    assert not result.artifact_path.exists()


def test_nested_references_protect_entries_and_wrong_roots_are_ignored(tmp_path):
    root = tmp_path / "cache"
    cache = ComparisonCache(root)
    result = _resolve(cache, [])
    valid_refs = cache.write_refs(
        tmp_path / "runs" / "reference_2009" / "split_45" / "cache_refs.json"
    )
    wrong_refs = tmp_path / "runs" / "other" / "cache_refs.json.gz"
    wrong_refs.parent.mkdir(parents=True)
    atomic_write_json_gzip(
        wrong_refs,
        {
            "cache_root": str((tmp_path / "different-cache").resolve()),
            "entries": [{"stage": result.stage, "key": result.key}],
        },
    )

    health = _inspect(root)["references"]
    assert health["valid_reference_file_count"] == 1
    assert health["ignored_reference_file_count"] == 1
    assert health["referenced_key_count"] == 1
    assert health["missing_referenced_target_count"] == 0
    assert health["unreferenced_key_count"] == 0
    assert _prune(root, unreferenced=True, older_than_days=None, apply=False)[
        "entries"
    ] == 0

    valid_refs.unlink()
    with pytest.raises(RuntimeError, match="no valid reference files"):
        _prune(root, unreferenced=True, older_than_days=None, apply=True)
    assert result.artifact_path.exists()


def test_plain_and_gzip_cache_manifests_load_identically(tmp_path):
    root = tmp_path / "cache"
    first = _resolve(ComparisonCache(root), [])
    compressed = first.artifact_path / "manifest.json.gz"
    manifest = read_json_file(compressed)
    (first.artifact_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    compressed.unlink()

    second = _resolve(ComparisonCache(root), [])
    assert second.status == "hit"
    assert second.value == first.value


def test_concurrent_resolvers_publish_one_entry(tmp_path):
    root = tmp_path / "cache"
    marker = tmp_path / "computations.txt"
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(
            target=_concurrent_cache_worker,
            args=(str(root), str(marker)),
        )
        for _ in range(2)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert marker.read_text(encoding="utf-8").splitlines() == ["computed"]
    assert _inspect(root)["entries"] == 1
