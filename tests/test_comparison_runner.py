import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

import comparison_runner
from comparison_runner import (
    AcceleratorRunnerConfig,
    ComparisonRunnerConfig,
    FunctionalRunnerConfig,
    MatchingRunnerConfig,
    SAERunnerConfig,
    _EmbeddingData,
    _PreparedData,
    _SAEData,
    _runtime_estimate,
    run_comparison,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "renal.feather"
    dataset.write_bytes(b"fake-feather-content")
    semantic = tmp_path / "semantic.json"
    semantic.write_text(
        json.dumps(
            {
                "activation_targets": {
                    "positive_fractions": [0.1, 0.5],
                },
                "discovery": {
                    "n_bootstraps": 2,
                    "trees_per_bootstrap": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    return dataset, semantic


def _config(tmp_path: Path) -> ComparisonRunnerConfig:
    dataset, semantic = _write_inputs(tmp_path)
    return ComparisonRunnerConfig.from_dict(
        {
            "dataset_path": str(dataset),
            "semantic_config_path": str(semantic),
            "artifact_dir": str(tmp_path / "artifacts"),
            "functional": {"enabled": False},
        }
    )


def _prepared_data() -> _PreparedData:
    n_train = 8
    n_test = 12
    X_train = np.arange(n_train * 2, dtype=np.float32).reshape(n_train, 2)
    X_test = np.arange(n_test * 2, dtype=np.float32).reshape(n_test, 2)
    patient_ids = np.asarray([f"patient-{index}" for index in range(n_test)])
    years = np.asarray([2006 + index % 3 for index in range(n_test)])
    return _PreparedData(
        train_rows=None,
        test_rows=None,
        feature_names=("feature_a", "feature_b"),
        X_train=X_train,
        y_train=np.arange(n_train) % 2,
        years_train=np.asarray([2000 + index % 3 for index in range(n_train)]),
        X_test=X_test,
        y_test=np.arange(n_test) % 2,
        years_test=years,
        patient_ids=patient_ids,
        record_keys=np.asarray(
            [
                f"patient:{patient}|year:{year}"
                for patient, year in zip(patient_ids, years)
            ]
        ),
    )


def _embedding_data(prepared: _PreparedData) -> _EmbeddingData:
    train = np.column_stack(
        (prepared.X_train, np.ones(len(prepared.X_train), dtype=np.float32))
    )
    test = np.column_stack(
        (prepared.X_test, np.ones(len(prepared.X_test), dtype=np.float32))
    )
    return _EmbeddingData(
        model=object(),
        model_device="cpu",
        example_add_shape=None,
        year_to_domain={
            int(year): index
            for index, year in enumerate(
                sorted(
                    set(prepared.years_train.tolist())
                    | set(prepared.years_test.tolist())
                )
            )
        },
        train_raw=train,
        test_raw=test,
        train_scaled=train,
        test_scaled=test,
        scaler=object(),
        fit_time_seconds=0.01,
        walkforward_metrics=[],
    )


class _FakeAdapter:
    def __init__(self) -> None:
        self.calls = []
        self.prepared = _prepared_data()
        self.embedded = _embedding_data(self.prepared)
        n = len(self.prepared.X_test)
        self.sae_data = _SAEData(
            runs=[{"idx": 0}, {"idx": 1}],
            activations={
                0: np.column_stack(
                    (
                        np.arange(n, dtype=float),
                        np.arange(n, dtype=float) + 100,
                    )
                ),
                1: np.column_stack(
                    (
                        np.arange(n, dtype=float) + 200,
                        np.arange(n, dtype=float) + 300,
                    )
                ),
            },
        )
        self.all_matches = [
            {
                "sae_i_idx": 0,
                "sae_j_idx": 1,
                "original_concept": 0,
                "best_pair": 1,
                "cos_sim": 0.93,
                "overlap": 0.62,
            },
            {
                "sae_i_idx": 0,
                "sae_j_idx": 1,
                "original_concept": 1,
                "best_pair": 0,
                "cos_sim": 0.41,
                "overlap": 0.22,
            },
        ]
        self.selected_matches = [self.all_matches[0]]
        self.semantic_payload = None

    def prepare(self, config, workspace, *, force):
        self.calls.append(("prepare", force))
        return self.prepared

    def embeddings(self, prepared, config, workspace, *, force):
        self.calls.append(("embeddings", force))
        assert prepared is self.prepared
        return self.embedded

    def train_saes(
        self, prepared, embeddings, splits, config, workspace, *, force
    ):
        self.calls.append(("train_saes", force))
        assert prepared is self.prepared
        assert embeddings is self.embedded
        split_rows = np.concatenate(
            [
                splits["idx_semantic_fit"],
                splits["idx_semantic_select"],
                splits["idx_tcav_eval"],
                splits["idx_semantic_final"],
            ]
        )
        assert sorted(split_rows.tolist()) == list(range(len(prepared.X_test)))
        return self.sae_data

    def match(self, sae_data, config, workspace):
        self.calls.append(("match", None))
        assert sae_data is self.sae_data
        return self.all_matches, self.selected_matches

    def functional(
        self,
        prepared,
        embeddings,
        sae_data,
        splits,
        matches,
        config,
        workspace,
        *,
        force,
    ):
        self.calls.append(("functional", force))
        assert list(matches) == self.selected_matches
        return {}, [{"status": "disabled"}]

    def semantic(
        self,
        prepared,
        sae_data,
        matches,
        functional,
        config,
        workspace,
        *,
        force,
    ):
        self.calls.append(("semantic", force))
        assert list(matches) == self.selected_matches
        assert functional == {}
        self.semantic_payload = {
            "X": prepared.X_test.copy(),
            "patient_ids": prepared.patient_ids.copy(),
            "record_keys": prepared.record_keys.copy(),
            "activations": {
                run_id: matrix.copy()
                for run_id, matrix in sae_data.activations.items()
            },
        }
        return {
            "artifact_dir": str(workspace / "semantic" / "fake"),
            "experiment_hash": "fake-semantic-hash",
            "manifest": {
                "config": {
                    "class_analysis": {
                        "enabled": True,
                    }
                }
            },
        }


class _FailingAdapter:
    def __getattr__(self, name):
        raise AssertionError(f"Complete-result cache unexpectedly called {name}")


def test_config_defaults_and_strict_nested_validation():
    config = ComparisonRunnerConfig.from_dict({})

    assert config.sae.seeds == (42, 135)
    assert config.accelerator == AcceleratorRunnerConfig()
    assert config.matching == MatchingRunnerConfig()
    assert config.functional == FunctionalRunnerConfig()
    assert config.functional.enabled is True

    with pytest.raises(ValueError, match="Unknown comparison config fields"):
        ComparisonRunnerConfig.from_dict({"unexpected": True})
    with pytest.raises(ValueError, match="Unknown sae fields"):
        ComparisonRunnerConfig.from_dict({"sae": {"surprise": 1}})
    with pytest.raises(ValueError, match="accelerator.device"):
        ComparisonRunnerConfig.from_dict(
            {"accelerator": {"device": "quantum"}}
        )
    with pytest.raises(ValueError, match="at least two unique"):
        SAERunnerConfig(seeds=(42, 42))
    with pytest.raises(ValueError, match="must be a boolean"):
        ComparisonRunnerConfig.from_dict({"functional": {"enabled": 1}})


def test_config_json_resolves_all_paths_relative_to_config(tmp_path):
    config_dir = tmp_path / "configuration"
    config_dir.mkdir()
    config_path = config_dir / "comparison.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_path": "../data/input.feather",
                "semantic_config_path": "semantic.json",
                "artifact_dir": "../artifacts",
            }
        ),
        encoding="utf-8",
    )

    config = ComparisonRunnerConfig.from_json(config_path)

    assert Path(config.dataset_path) == (tmp_path / "data/input.feather").resolve()
    assert Path(config.semantic_config_path) == (
        config_dir / "semantic.json"
    ).resolve()
    assert Path(config.artifact_dir) == (tmp_path / "artifacts").resolve()


def test_two_run_orchestration_preserves_alignment_and_selected_matches(
    monkeypatch, tmp_path
):
    config = _config(tmp_path)
    adapter = _FakeAdapter()
    monkeypatch.setattr(
        comparison_runner,
        "_runner_source_fingerprint",
        lambda: "fixed-source",
    )
    monkeypatch.setattr(
        "semantic_artifacts.environment_manifest",
        lambda: {"python": "test"},
    )

    summary = run_comparison(config, adapter=adapter)

    assert [call[0] for call in adapter.calls] == [
        "prepare",
        "embeddings",
        "train_saes",
        "match",
        "functional",
        "semantic",
    ]
    assert summary["n_all_matches"] == 2
    assert summary["n_selected_matches"] == 1
    assert summary["class_analysis_enabled"] is True
    payload = adapter.semantic_payload
    np.testing.assert_array_equal(payload["X"], adapter.prepared.X_test)
    np.testing.assert_array_equal(
        payload["patient_ids"], adapter.prepared.patient_ids
    )
    np.testing.assert_array_equal(
        payload["record_keys"], adapter.prepared.record_keys
    )
    for run_id, expected in adapter.sae_data.activations.items():
        np.testing.assert_array_equal(payload["activations"][run_id], expected)

    artifact_dir = Path(summary["artifact_dir"])
    bundle_path = artifact_dir / "semantic_inputs.npz"
    with np.load(bundle_path, allow_pickle=False) as bundle:
        np.testing.assert_array_equal(bundle["X"], adapter.prepared.X_test)
        np.testing.assert_array_equal(
            bundle["patient_ids"], adapter.prepared.patient_ids.astype(str)
        )
        np.testing.assert_array_equal(
            bundle["record_keys"], adapter.prepared.record_keys.astype(str)
        )
        np.testing.assert_array_equal(
            bundle["activations_run_0"], adapter.sae_data.activations[0]
        )
        np.testing.assert_array_equal(
            bundle["activations_run_1"], adapter.sae_data.activations[1]
        )
    stage_metrics = json.loads(
        (artifact_dir / "stage_metrics.json").read_text(encoding="utf-8")
    )
    assert list(stage_metrics) == sorted(
        [
            "geometric_matching",
            "high_precision_cav_tcav",
            "prepare",
            "sae_training_and_encoding",
            "semantic_bundle",
            "split",
            "stable_semantic_comparison",
            "tabpfn_and_embeddings",
        ]
    )
    assert all(row["seconds"] >= 0 for row in stage_metrics.values())
    assert all(row["status"] == "completed" for row in stage_metrics.values())
    assert summary["resolved_device"] in {"cpu", "cuda"}
    assert summary["total_timed_seconds"] == pytest.approx(
        sum(row["seconds"] for row in stage_metrics.values())
    )
    runner_manifest = json.loads(
        (artifact_dir / "runner_manifest.json").read_text(encoding="utf-8")
    )
    assert runner_manifest["accelerator"]["resolved_device"] in {"cpu", "cuda"}
    assert runner_manifest["stage_metrics"] == stage_metrics


def test_runtime_estimate_counts_selected_unique_factors(tmp_path):
    config = _config(tmp_path)
    matches = [
        {
            "sae_i_idx": 0,
            "sae_j_idx": 1,
            "original_concept": 4,
            "best_pair": 7,
        },
        {
            "sae_i_idx": 0,
            "sae_j_idx": 1,
            "original_concept": 4,
            "best_pair": 8,
        },
    ]

    estimate = _runtime_estimate(config, matches)

    assert estimate == {
        "selected_pairs": 2,
        "unique_factors": 3,
        "activation_thresholds": 2,
        "tree_fits": 36,
    }


def test_complete_result_cache_avoids_all_adapter_stages(monkeypatch, tmp_path):
    config = _config(tmp_path)
    adapter = _FakeAdapter()
    monkeypatch.setattr(
        comparison_runner,
        "_runner_source_fingerprint",
        lambda: "fixed-source",
    )
    monkeypatch.setattr(
        "semantic_artifacts.environment_manifest",
        lambda: {"python": "test"},
    )
    first = run_comparison(config, adapter=adapter)

    cached = run_comparison(config, adapter=_FailingAdapter())

    assert first["cache_hit"] is False
    assert cached == {**first, "cache_hit": True}


def test_semantic_config_content_invalidates_complete_result_cache(
    monkeypatch, tmp_path
):
    config = _config(tmp_path)
    monkeypatch.setattr(
        comparison_runner,
        "_runner_source_fingerprint",
        lambda: "fixed-source",
    )
    monkeypatch.setattr(
        "semantic_artifacts.environment_manifest",
        lambda: {"python": "test"},
    )
    first_adapter = _FakeAdapter()
    first = run_comparison(config, adapter=first_adapter)

    Path(config.semantic_config_path).write_text(
        json.dumps(
            {
                "activation_targets": {
                    "positive_fractions": [0.1, 0.25, 0.5],
                },
                "discovery": {
                    "n_bootstraps": 2,
                    "trees_per_bootstrap": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    second_adapter = _FakeAdapter()
    second = run_comparison(config, adapter=second_adapter)

    assert second["cache_hit"] is False
    assert second["runner_hash"] != first["runner_hash"]
    assert second_adapter.calls


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("dataset", "Renal Feather file not found"),
        ("semantic", "Semantic configuration not found"),
    ],
)
def test_run_comparison_reports_missing_required_files(
    tmp_path, missing, message
):
    dataset = tmp_path / "data.feather"
    semantic = tmp_path / "semantic.json"
    if missing != "dataset":
        dataset.write_bytes(b"dataset")
    if missing != "semantic":
        semantic.write_text("{}", encoding="utf-8")
    config = ComparisonRunnerConfig(
        dataset_path=str(dataset),
        semantic_config_path=str(semantic),
        artifact_dir=str(tmp_path / "artifacts"),
    )

    with pytest.raises(FileNotFoundError, match=message):
        run_comparison(config, adapter=_FailingAdapter())


def test_main_comparison_help_does_not_load_pipeline(monkeypatch, capsys):
    launcher = Path(comparison_runner.__file__).with_name("main-comparison.py")
    monkeypatch.setattr(sys, "argv", [str(launcher), "--help"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(launcher), run_name="__main__")

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "complete renal cross-run sae semantic comparison" in output.lower()
    assert "--device" in output
    assert "--skip-functional" in output
