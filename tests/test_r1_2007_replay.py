import numpy as np
import pytest
import json

import r1_2007_replay as replay


def test_query_batch_preserves_archive_order_and_companions():
    years = np.array([2007, 2008, 2015, 2007, 2015, 2008])
    patients = np.array(["a", "b", "c", "d", "e", "f"])
    population = {"patient_id": patients, "year": years}
    archived_rows = np.flatnonzero(years >= 2007)

    selected = replay.select_query_batch(population, archived_rows, 2007, batch_size=4)

    assert selected.tolist() == [0, 1, 2, 3]
    assert years[selected].tolist() == [2007, 2008, 2015, 2007]
    assert patients[selected].tolist() == ["a", "b", "c", "d"]


def test_query_batch_rejects_archive_order_or_identity_gap():
    population = {"patient_id": np.array(["a", "b", "c"]),
                  "year": np.array([2007, 2015, 2008])}

    with pytest.raises(ValueError, match="archive row identity"):
        replay.select_query_batch(population, np.array([0, 2]), 2007, batch_size=2)


def test_probe_rows_resolve_by_patient_year_identity():
    population = {"patient_id": np.array(["p1", "p2", "p3", "p4"]),
                  "year": np.array([2007, 2007, 2008, 2009])}
    batch_rows = np.array([2, 0, 1, 3])
    actual = {"patient_id": np.array(["p2", "p1", "p3", "p4"]),
              "year": np.array([2007, 2007, 2008, 2009])}

    probes = replay.select_probe_rows(population, batch_rows, actual, 2007,
                                      probe_years=(2007, 2008, 2009))

    assert probes == {
        2007: {"identity": ("p2", 2007), "actual_table_row": 0,
               "population_row": 1, "batch_position": 2},
        2008: {"identity": ("p3", 2008), "actual_table_row": 2,
               "population_row": 2, "batch_position": 0},
        2009: {"identity": ("p4", 2009), "actual_table_row": 3,
               "population_row": 3, "batch_position": 3},
    }


def test_held_batch_preserves_candidate_order_and_precheck():
    held = {"patient_id": np.array(["h1", "h2", "h3"]),
            "year": np.array([2008, 2009, 2010]), "positions": np.array([1, 2, 3])}
    actual = {"patient_id": np.array(["a1", "h1", "h2", "h3", "a4"]),
              "year": np.array([2007, 2008, 2009, 2010, 2015])}

    selection = replay.select_held_batch(held, actual, 2007, batch_size=2, precheck_size=2)

    assert selection["held_rows"].tolist() == [0, 1]
    assert selection["precheck_rows"].tolist() == [1, 2]
    assert selection["held_identities"] == [("h1", 2008), ("h2", 2009)]
    assert selection["held_actual_rows"].tolist() == [1, 2]


def test_held_batch_rejects_unmapped_identity_without_positions():
    held = {"patient_id": np.array(["unknown"]), "year": np.array([2008])}
    actual = {"patient_id": np.array(["a1", "a2"]), "year": np.array([2007, 2008])}

    with pytest.raises(ValueError, match="absent from actual archive"):
        replay.select_held_batch(held, actual, 2007, batch_size=1, precheck_size=1)


def test_isolated_source_checks_hash_and_registers_dataclasses_module(tmp_path):
    import hashlib
    import sys

    source = tmp_path / "snapshot.py"
    source.write_text("from dataclasses import dataclass\n@dataclass\nclass Marker:\n    value: int\n")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    loaded = replay.load_isolated_source(source, expected, module_name="_test_replay_snapshot")

    assert loaded.Marker(3).value == 3
    assert sys.modules["_test_replay_snapshot"] is loaded
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        replay.load_isolated_source(source, "0" * 64, module_name="_test_wrong_snapshot")


def test_provenance_verification_rejects_changed_consumed_file(tmp_path):
    source = tmp_path / "input.npz"
    source.write_bytes(b"approved bytes")
    expected = {"input": {"sha256": replay.file_sha256(source), "status": "available"}}
    assert replay.verify_provenance_hashes({"input": source}, expected)["input"] == expected["input"]["sha256"]

    source.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="provenance SHA-256 mismatch"):
        replay.verify_provenance_hashes({"input": source}, expected)


def test_preparation_rejects_changed_context_and_population_identities(monkeypatch):
    population = _synthetic_population()
    actual, held = _synthetic_archives(population)
    roles = _synthetic_roles(len(population["year"]))
    meta = _synthetic_meta(population, roles, actual, held)
    pilot = _synthetic_pilot()
    def no_inference(*args, **kwargs):
        raise AssertionError("metadata preparation must not instantiate or call a predictor")
    monkeypatch.setattr(pilot, "fit_frozen_predictor", no_inference, raising=False)
    monkeypatch.setattr(pilot, "capture_frozen_predictions", no_inference, raising=False)

    prepared = replay.prepare_r1_input_metadata(
        pilot, population, roles, actual, held, np.arange(8), meta, {"split_seed": 42},
        reference_year=2007, batch_size=4, held_batch_size=2, precheck_size=2,
        probe_years=(2007, 2008, 2009, 2010),
    )
    assert prepared["query_rows"].tolist() == [0, 1, 2, 3]
    assert prepared["probes"][2007]["identity"] == ("p0", 2007)
    assert prepared["probes"][2007]["batch_position"] == 0
    assert prepared["actual_probe_domains"][2008] == 1
    assert prepared["held_probe_domain"] == 0
    with pytest.raises(ValueError, match="only saved reference_only context"):
        replay.prepare_r1_input_metadata(
            pilot, population, roles, actual, held, np.arange(8), meta,
            {"split_seed": 42, "context_window": "last_3"},
            reference_year=2007, batch_size=4, held_batch_size=2, precheck_size=2,
            probe_years=(2007, 2008, 2009, 2010),
        )

    changed_roles = dict(roles)
    changed_roles["row_role"] = roles["row_role"].copy()
    changed_roles["row_role"][0] = "other"
    with pytest.raises(ValueError, match="context row identity mismatch"):
        replay.prepare_r1_input_metadata(
            pilot, population, changed_roles, actual, held, np.arange(8), meta, {"split_seed": 42},
            reference_year=2007, batch_size=4, held_batch_size=2, precheck_size=2,
            probe_years=(2007, 2008, 2009, 2010),
        )

    changed_population = dict(population)
    changed_population["X"] = population["X"].copy()
    changed_population["X"][0, 0] += 1
    with pytest.raises(ValueError, match="population identity mismatch"):
        replay.prepare_r1_input_metadata(
            pilot, changed_population, roles, actual, held, np.arange(8), meta, {"split_seed": 42},
            reference_year=2007, batch_size=4, held_batch_size=2, precheck_size=2,
            probe_years=(2007, 2008, 2009, 2010),
        )


def test_feasibility_executes_reference_gates_then_one_full_batch(monkeypatch, tmp_path):
    inputs, trace = _runner_inputs()
    monkeypatch.setattr(replay, "_load_r1_inputs", lambda: inputs, raising=False)
    monkeypatch.setattr(replay, "_require_replay_runtime", lambda inputs=None: {"device": "cpu", "threads": 1,
        "precision": "fp32", "seed": 42}, raising=False)
    monkeypatch.setattr(replay, "_seed_replay_runtime", lambda seed: None, raising=False)

    output = tmp_path / "feasibility"
    (output / "run").mkdir(parents=True)
    (output / "run" / "run.json").write_text("{\"job_id\": \"synthetic\"}")
    result = replay.run_feasibility(output)

    assert result["status"] == "completed"
    assert trace[0] == ("precision", "recovered", "cpu")
    assert trace[1][0] == "fit" and trace[1][1] == "recovered"
    assert [entry[0] for entry in trace[2:9]] == ["capture"] * 7
    assert [entry[1] for entry in trace[2:5]] == [["p0", "p1", "p2", "p3"]] * 2 + [["p3", "p2", "p1", "p0"]]
    assert [entry[1] for entry in trace[5:9]] == [["p0"], ["p1"], ["p2"], ["p3"]]
    assert trace[9][0:2] == ("capture", [f"p{i}" for i in range(16)])
    assert result["runtime"]["seed"] == 42 and result["runtime"]["precision"] == "fp32"
    saved = np.load(output / "recovered_actual_probes.npz", allow_pickle=False)
    assert saved["years"].tolist() == list(range(2007, 2015))
    assert json_load(output / "completion.json")["status"] == "completed"
    assert json_load(output / "run.json")["launcher"]["job_id"] == "synthetic"


def test_comparison_reuses_matching_feasibility_and_runs_isolated_arms(monkeypatch, tmp_path):
    inputs, trace = _runner_inputs()
    monkeypatch.setattr(replay, "_load_r1_inputs", lambda: inputs, raising=False)
    monkeypatch.setattr(replay, "_require_replay_runtime", lambda inputs=None: {"device": "cpu", "threads": 1,
        "precision": "fp32", "seed": 42}, raising=False)
    monkeypatch.setattr(replay, "_seed_replay_runtime", lambda seed: None, raising=False)
    feasibility_dir = tmp_path / "feasibility"
    feasibility = replay.run_feasibility(feasibility_dir)
    feasibility_fits = sum(entry[0] == "fit" for entry in trace)
    assert feasibility_fits == 1

    inputs["fingerprint_facts"]["actual_archive_sha256"] = "changed"
    with pytest.raises(ValueError, match="feasibility fingerprint mismatch"):
        replay.run_comparison(feasibility_dir, tmp_path / "rejected")
    assert sum(entry[0] == "fit" for entry in trace) == feasibility_fits

    inputs["fingerprint_facts"]["actual_archive_sha256"] = "actual-fixture"
    result = replay.run_comparison(feasibility_dir, tmp_path / "comparison")
    assert result["status"] == "completed"
    new_fits = [entry for entry in trace if entry[0] == "fit"][feasibility_fits:]
    assert [entry[1] for entry in new_fits] == ["current", "current", "recovered", "current"]
    assert len({entry[2] for entry in new_fits}) == 4
    held_calls = [entry for entry in trace if entry[0] == "capture" and entry[1] == [f"p{i}" for i in range(4, 12)]]
    assert len(held_calls) == 2
    for held_call in held_calls:
        assert trace[trace.index(held_call) - 1][1] == ["p4", "p5", "p6", "p7"]
    held_archive = np.load(tmp_path / "comparison" / "recovered_held_candidate.npz", allow_pickle=False)
    assert not held_archive["target_embeddings_available"].item()
    assert "target_embeddings" not in held_archive.files
    assert result["feasibility_reused"] is True
    run_path = feasibility_dir / "run.json"
    run_path.write_text(run_path.read_text() + " ")
    with pytest.raises(ValueError, match="run-record identity mismatch"):
        replay._load_feasibility(feasibility_dir, result["fingerprint"])


@pytest.mark.parametrize("mismatch, message", [
    ("weight", "different learned parameters"),
    ("config", "different fitted input/config identities"),
])
def test_comparison_stops_when_source_fit_identity_changes(monkeypatch, tmp_path, mismatch, message):
    inputs, trace = _runner_inputs(source_mismatch=mismatch)
    monkeypatch.setattr(replay, "_load_r1_inputs", lambda: inputs, raising=False)
    monkeypatch.setattr(replay, "_require_replay_runtime", lambda inputs=None: {"device": "cpu", "threads": 1,
        "precision": "fp32", "seed": 42}, raising=False)
    monkeypatch.setattr(replay, "_seed_replay_runtime", lambda seed: None, raising=False)
    feasibility = tmp_path / "feasibility"
    replay.run_feasibility(feasibility)
    output = tmp_path / "comparison"

    with pytest.raises(ValueError, match=message):
        replay.run_comparison(feasibility, output)

    comparison_fits = [entry for entry in trace if entry[0] == "fit"][1:]
    assert [entry[1] for entry in comparison_fits] == ["current"]
    assert (output / "current_actual_batch.npz").is_file()
    assert json_load(output / "completion.json")["status"] == "failed"
    report = replay.build_saved_report(feasibility, output)
    source_pair = next(row for row in report["comparisons"]
                       if row["comparison"] == "recovered_vs_current_actual_batch")
    assert source_pair["probability_status"] in ("unavailable", "identity_mismatch")
    assert source_pair.get("max_abs_class1_probability_error") is None
    archive_diagnostic = next(row for row in report["comparisons"]
                              if row["comparison"] == "current_actual_batch_vs_archive")
    assert archive_diagnostic["probability_status"] == "available"
    (output / "current_actual_batch.npz").write_bytes(b"tampered partial output")
    corrupted = replay.build_saved_report(feasibility, output)
    assert corrupted["comparison_status"] == "invalid_output_checksum"
    assert corrupted["comparisons"] == []


def test_held_precheck_mismatch_is_reported_and_does_not_block_candidate_capture(monkeypatch, tmp_path):
    inputs, trace = _runner_inputs()
    inputs["actual"]["probability"][4] += np.float32(0.5)
    monkeypatch.setattr(replay, "_load_r1_inputs", lambda: inputs, raising=False)
    monkeypatch.setattr(replay, "_require_replay_runtime", lambda inputs=None: {"device": "cpu", "threads": 1,
        "precision": "fp32", "seed": 42}, raising=False)
    monkeypatch.setattr(replay, "_seed_replay_runtime", lambda seed: None, raising=False)
    feasibility = tmp_path / "feasibility"
    replay.run_feasibility(feasibility)
    comparison = tmp_path / "comparison"
    result = replay.run_comparison(feasibility, comparison)

    assert result["status"] == "completed"
    run_record = json_load(comparison / "run.json")
    for name in ("recovered_held_candidate", "current_held_candidate"):
        diagnostic = run_record["conditions"][name]["precheck_diagnostic"]
        assert diagnostic["status"] == "mismatch"
        assert diagnostic["count"] == 4
        assert diagnostic["max_abs_class1_probability_error"] >= 0.5
        assert diagnostic["within_combined_tolerance"] is False
    assert sum(entry[0] == "capture" and entry[1] == [f"p{i}" for i in range(4, 12)]
               for entry in trace) == 2
    report = json_load(comparison / "report.json")
    assert report["held_precheck_diagnostics"]["recovered_held_candidate"]["status"] == "mismatch"
    assert "max_abs_class1_probability_error" in report["held_precheck_diagnostics"]["current_held_candidate"]


def test_failed_capture_records_failure_without_completion(monkeypatch, tmp_path):
    inputs, trace = _runner_inputs(fail_capture=True)
    monkeypatch.setattr(replay, "_load_r1_inputs", lambda: inputs, raising=False)
    monkeypatch.setattr(replay, "_require_replay_runtime", lambda inputs=None: {"device": "cpu", "threads": 1,
        "precision": "fp32", "seed": 42}, raising=False)
    monkeypatch.setattr(replay, "_seed_replay_runtime", lambda seed: None, raising=False)

    with pytest.raises(RuntimeError, match="synthetic capture failure"):
        replay.run_feasibility(tmp_path / "failed")

    record = json_load(tmp_path / "failed" / "completion.json")
    assert record["status"] == "failed"
    assert record["error"]["type"] == "RuntimeError"


def test_invalid_state_identity_fails_before_reference_gates_or_batch_capture(monkeypatch, tmp_path):
    inputs, trace = _runner_inputs()
    monkeypatch.setattr(replay, "_load_r1_inputs", lambda: inputs, raising=False)
    monkeypatch.setattr(replay, "_require_replay_runtime", lambda inputs=None: {"device": "cpu", "threads": 1,
        "precision": "fp32", "seed": 42}, raising=False)
    monkeypatch.setattr(replay, "_seed_replay_runtime", lambda seed: None, raising=False)

    def reject_state(pilot, system):
        raise RuntimeError("synthetic invalid fitted-state identity")

    monkeypatch.setattr(replay, "_state_report", reject_state)
    with pytest.raises(RuntimeError, match="synthetic invalid fitted-state identity"):
        replay.run_feasibility(tmp_path / "invalid-state")

    assert [entry[1] for entry in trace if entry[0] == "fit"] == ["recovered"]
    assert not any(entry[0] == "capture" for entry in trace)


def test_comparison_failure_preserves_completed_arm_evidence(monkeypatch, tmp_path):
    inputs, _ = _runner_inputs()
    monkeypatch.setattr(replay, "_load_r1_inputs", lambda: inputs, raising=False)
    monkeypatch.setattr(replay, "_require_replay_runtime", lambda inputs=None: {"device": "cpu", "threads": 1,
        "precision": "fp32", "seed": 42}, raising=False)
    monkeypatch.setattr(replay, "_seed_replay_runtime", lambda seed: None, raising=False)
    feasibility = tmp_path / "feasibility"
    replay.run_feasibility(feasibility)
    original = replay._capture_condition

    def fail_after_first_arm(inputs, source, mode):
        if mode == "actual_singletons":
            raise RuntimeError("synthetic later-arm failure")
        return original(inputs, source, mode)

    monkeypatch.setattr(replay, "_capture_condition", fail_after_first_arm)
    output = tmp_path / "comparison-failed"
    with pytest.raises(RuntimeError, match="later-arm failure"):
        replay.run_comparison(feasibility, output)
    assert (output / "current_actual_batch.npz").is_file()
    run_record = json_load(output / "run.json")
    assert run_record["status"] == "failed"
    assert "current_actual_batch.npz" in run_record["partial_outputs"]
    assert json_load(output / "completion.json")["status"] == "failed"


def test_launcher_output_directory_is_reused_but_existing_results_are_rejected(tmp_path):
    output = tmp_path / "launched"
    (output / "run").mkdir(parents=True)
    (output / "run" / "run.json").write_text("{}")
    assert replay._create_analysis_output(output) == output

    (output / "completion.json").write_text("{}")
    with pytest.raises(FileExistsError, match="refusing to overwrite replay output"):
        replay._create_analysis_output(output)


def test_runtime_guard_rejects_wrong_threads_before_any_fit(monkeypatch):
    inputs, trace = _runner_inputs()
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(RuntimeError, match="OMP_NUM_THREADS=1"):
        replay._require_replay_runtime(inputs)
    assert not any(entry[0] == "fit" for entry in trace)


def test_capture_validation_rejects_nonfinite_companion_rows():
    captured = {"probabilities": np.array([[0.5, 0.5], [np.nan, 0.5]], dtype=np.float32),
                "embeddings": np.zeros((2, 3), dtype=np.float32)}
    with pytest.raises(RuntimeError, match="nonfinite predictor outputs"):
        replay._checked_capture(captured, 2, "synthetic full batch")


def test_real_current_and_recovered_predictors_expose_stable_state_identity_on_synthetic_data():
    from pathlib import Path
    import pandas as pd
    import concept_subgroup_drift_pilot as current_pilot

    recovered_path = (Path(__file__).resolve().parents[1] / "stats/concept_agreement_study/extensions"
                      / "r1_2007_replay/recovered/concept_subgroup_drift_pilot.py")
    recovered_pilot = replay.load_isolated_source(
        recovered_path, replay.RECOVERED_SOURCE_SHA256, module_name="_test_recovered_pilot_state")
    generator = np.random.default_rng(42)
    features = ["f0", "f1", "f2"]
    context = pd.DataFrame(generator.normal(size=(32, 3)).astype(np.float32), columns=features)
    context["patient_id"] = [f"synthetic-context-{index}" for index in range(32)]
    context["year"] = 2007
    context["y"] = np.arange(32, dtype=np.int64) % 2
    query = context.iloc[:2].drop(columns="y").copy()
    query["patient_id"] = ["synthetic-query-0", "synthetic-query-1"]
    query["year"] = 2008

    for pilot in (current_pilot, recovered_pilot):
        system = pilot.fit_frozen_predictor(context, features, "cpu", seed=42)
        before = replay._state_report(pilot, system)
        prediction = pilot.capture_frozen_predictions(system, query, "cpu")
        assert np.asarray(prediction["probabilities"].detach().cpu()).shape == (2, 2)
        after = replay._state_report(pilot, system)
        assert before["learned_parameter_identity"] == replay.CANONICAL_MODEL_PARAMETER_SHA256
        assert before["learned_parameter_identity"] == after["learned_parameter_identity"]
        assert before["identity"] == after["identity"]
        assert before["learned_parameter_count"] > 0
        parameter = next(system["classifier"].cached_models_[0].named_parameters())[1]
        original = parameter.detach().clone()
        import torch
        with torch.inference_mode():
            parameter.add_(1)
        with pytest.raises(RuntimeError, match="canonical checkpoint identity"):
            replay._state_report(pilot, system)
        with torch.inference_mode():
            parameter.copy_(original)


def test_state_report_rejects_unsupported_cached_model_container():
    from types import SimpleNamespace

    class Pilot:
        @staticmethod
        def predictor_identity_report(classifier):
            return {"identity": "synthetic", "field_digests": {}}

    system = {"classifier": SimpleNamespace(
        model=None, cached_models_=[object()], pilot_checkpoint_sha256_=replay.CHECKPOINT_SHA256)}
    with pytest.raises(RuntimeError, match="lacks named parameters"):
        replay._state_report(Pilot, system)
    system["classifier"].cached_models_.append(object())
    with pytest.raises(RuntimeError, match="exactly one cached learned model"):
        replay._state_report(Pilot, system)


def test_saved_probe_comparison_reports_class1_raw_embedding_and_missing_held_embedding():
    left = {"years": np.array([2007, 2008]),
            "probabilities": np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32),
            "embeddings": np.array([[0.0, 0.00001], [0.5, 0.6]], dtype=np.float32),
            "target_probabilities": np.array([[0.0, 0.800005], [0.0, 0.5]], dtype=np.float32),
            "target_embeddings": np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float32),
            "target_embeddings_available": np.asarray(True)}

    actual = replay.compare_probe_arrays(left)

    assert actual[0]["max_abs_class1_probability_error"] == pytest.approx(5e-6, abs=1e-8)
    assert actual[0]["class1_probability_within_combined_tolerance"] is True
    assert actual[0]["max_abs_raw_embedding_error"] == pytest.approx(1e-5, abs=1e-8)
    assert actual[0]["raw_embedding_within_combined_tolerance"] is True
    assert actual[1]["class1_probability_within_combined_tolerance"] is False
    assert actual[1]["raw_embedding_within_combined_tolerance"] is False

    held = dict(left)
    held["target_probabilities"] = left["target_probabilities"]
    held["target_embeddings_available"] = np.asarray(False)
    held.pop("target_embeddings")
    held_result = replay.compare_probe_arrays(held)
    assert held_result[0]["raw_embedding_status"] == "unavailable"
    assert held_result[0]["max_abs_raw_embedding_error"] is None
    left["target_probabilities"] = np.array([0.800005, 0.5], dtype=np.float32)
    assert replay.compare_probe_arrays(left)[0]["class1_probability_within_combined_tolerance"] is True


def test_saved_report_rejects_corrupt_outputs_and_does_not_fallback_missing_pair_reference(tmp_path):
    feasibility = tmp_path / "feasibility"
    comparison = tmp_path / "comparison"
    _saved_attempt(feasibility, "completed", {})
    payload = _probe_payload()
    _saved_attempt(comparison, "completed", {"current_actual_batch.npz": payload},
                   identity_statuses=("actual_source",))
    report = replay.build_saved_report(feasibility, comparison)
    pair = next(row for row in report["comparisons"]
                if row["comparison"] == "recovered_vs_current_actual_batch")
    assert pair["probability_status"] == "unavailable"
    assert pair["qualification"] == "missing verified comparison reference"

    (comparison / "current_actual_batch.npz").write_bytes(b"corrupt")
    invalid = replay.build_saved_report(feasibility, comparison)
    assert invalid["comparison_status"] == "invalid_output_checksum"
    assert all(row.get("year") is None for row in invalid["comparisons"]
               if row["comparison"] == "current_actual_batch_vs_archive")


def test_saved_report_requires_consistent_completion_and_external_write_preserves_evidence(tmp_path):
    attempt = tmp_path / "attempt"
    _saved_attempt(attempt, "completed", {"recovered_actual_probes.npz": _probe_payload()})
    run_before = (attempt / "run.json").read_bytes()
    completion_before = (attempt / "completion.json").read_bytes()
    report_dir = tmp_path / "fresh-report"
    replay.write_saved_report(attempt, output=report_dir)
    assert (attempt / "run.json").read_bytes() == run_before
    assert (attempt / "completion.json").read_bytes() == completion_before

    completion = json_load(attempt / "completion.json")
    completion["status"] = "running"
    (attempt / "completion.json").write_text(json.dumps(completion))
    invalid = replay.build_saved_report(attempt)
    assert invalid["status"] == "incomplete_or_invalid"
    assert invalid["comparisons"] == []


def test_saved_report_requires_matching_verified_attempt_fingerprints_and_exposes_limits(tmp_path):
    import shutil

    feasibility = tmp_path / "feasibility"
    comparison = tmp_path / "comparison"
    payload = _probe_payload()
    _saved_attempt(feasibility, "completed", {"recovered_actual_probes.npz": payload}, fingerprint="f1")
    _saved_attempt(comparison, "completed", {"current_actual_batch.npz": payload,
                                                "recovered_held_candidate.npz": payload,
                                                "current_held_candidate.npz": payload}, fingerprint="f2")
    mismatch = replay.build_saved_report(feasibility, comparison)
    assert mismatch["status"] == "fingerprint_mismatch"
    assert mismatch["comparisons"] == []

    (feasibility / "recovered_actual_probes.npz").write_bytes(b"changed")
    invalid = replay.build_saved_report(feasibility, comparison)
    assert invalid["status"] == "invalid_feasibility"
    assert invalid["comparisons"] == []

    shutil.rmtree(feasibility)
    _saved_attempt(feasibility, "completed", {"recovered_actual_probes.npz": payload}, fingerprint="f1")
    shutil.rmtree(comparison)
    _saved_attempt(comparison, "completed", {"recovered_actual_probes.npz": payload,
                                                "current_actual_batch.npz": payload,
                                                "current_actual_singletons.npz": payload,
                                                "recovered_held_candidate.npz": payload,
                                                "current_held_candidate.npz": payload}, fingerprint="f1",
                   identity_statuses=("actual_source", "current_schedule", "held_source"))
    report = replay.build_saved_report(feasibility, comparison)
    held_rows = [row for row in report["comparisons"] if row["input"] == "held"]
    assert held_rows
    assert all("writer/schedule" in row["qualification"] and "64 rows" in row["qualification"]
               and "CPU replay vs original CUDA" in row["qualification"] for row in held_rows)
    assert report["limitations"]["historical_dependency_identity"].startswith("unknown")
    assert "finite-difference gradient checks" in report["limitations"]["omitted_non_inference_continuation"]
    markdown_dir = tmp_path / "report"
    replay.write_saved_report(feasibility, comparison, output=markdown_dir)
    markdown = (markdown_dir / "report.md").read_text()
    assert "Historical dependency identity: unknown" in markdown
    assert "Held writer and schedule: unknown" in markdown
    assert "finite-difference gradient checks" in markdown
    assert "original source runtime was CUDA" in markdown
    assert "continuation embedding export/rebuild/replay" in markdown

    shutil.rmtree(feasibility)
    _saved_attempt(feasibility, "failed", {}, fingerprint="f1")
    failed_base = replay.build_saved_report(feasibility, comparison)
    assert failed_base["status"] == "feasibility_failed"


def _probe_payload():
    return {"years": np.array([2007, 2008]),
            "probabilities": np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32),
            "embeddings": np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
            "target_probabilities": np.array([0.8, 0.5], dtype=np.float32),
            "target_embeddings_available": np.asarray(False)}


def _saved_attempt(path, status, payloads, error_message=None, fingerprint="synthetic-fingerprint",
                   identity_statuses=()):
    path.mkdir()
    outputs = {}
    for name, payload in payloads.items():
        np.savez_compressed(path / name, **payload)
        outputs[name] = replay.file_sha256(path / name)
    run = {"status": status, "fingerprint": fingerprint, "outputs": outputs,
           "partial_outputs": outputs if status == "failed" else {},
           "conditions": {"identity_comparisons": {
               name: {"status": "matched", "learned_parameters_match": True,
                      "compared_field_identities": []} for name in identity_statuses}},
           "input_fingerprints": {}, "runtime": {}}
    if error_message:
        run["error"] = {"type": "RuntimeError", "message": error_message}
    (path / "run.json").write_text(json.dumps(run))
    completion = {"status": status, "fingerprint": run["fingerprint"],
                  "outputs": outputs if status == "completed" else {},
                  "partial_outputs": outputs if status == "failed" else {},
                  "run_sha256": replay.file_sha256(path / "run.json")}
    (path / "completion.json").write_text(json.dumps(completion))


def test_saved_report_is_aggregate_only_and_marks_failed_or_partial_attempts(tmp_path):
    feasibility = tmp_path / "feasibility"
    _saved_attempt(feasibility, "failed", {}, error_message="private p0 value")
    report = replay.build_saved_report(feasibility)
    serialized = json.dumps(report)
    assert report["status"] == "failed"
    assert report["comparisons"]
    assert all(row["probability_status"] == "unavailable" for row in report["comparisons"])
    assert "private p0 value" not in serialized
    assert "p0" not in serialized


def json_load(path):
    return json.loads(path.read_text())


def _runner_inputs(fail_capture=False, source_mismatch=None):
    import pandas as pd
    from types import SimpleNamespace

    class Trace(list):
        pass

    trace = Trace()
    trace.fitted_models = []
    population = {"patient_id": np.array([f"p{i}" for i in range(16)]),
                  "year": np.array([2007] * 4 + list(range(2008, 2016)) + list(range(2008, 2012))),
                  "y": np.array([0, 1] * 8),
                  "X": np.arange(16, dtype=np.float32).reshape(-1, 1)}
    actual = {"patient_id": population["patient_id"], "year": population["year"],
              "probability": np.arange(16, dtype=np.float32) / 100}
    held_rows = np.arange(4, 12)
    held = {"patient_id": actual["patient_id"][held_rows], "year": actual["year"][held_rows],
            "positions": held_rows}

    def make_pilot(name):
        class Pilot:
            FIRST_CONTEXT_YEAR = 1900
            PREDICTOR_GATE_ROWS = 4

            @staticmethod
            def _feature_columns(count):
                return [f"f{i}" for i in range(count)]

            @staticmethod
            def _identity_hash(value):
                import hashlib
                return hashlib.sha256(repr(value).encode()).hexdigest()

            @staticmethod
            def _prediction_frame(population_data, rows, columns):
                return pd.DataFrame({"f0": population_data["X"][rows, 0],
                                     "patient_id": population_data["patient_id"][rows],
                                     "year": population_data["year"][rows]})

            @staticmethod
            def configure_predictor_precision(device):
                import torch
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                trace.append(("precision", name, device))
                return {"amp": "off", "tf32": False}

            @staticmethod
            def fit_frozen_predictor(context, columns, device, seed=42):
                weight = np.array([2 if name == "current" and source_mismatch == "weight" else 1], dtype=np.float32)
                decoder = type("Decoder", (), {"state_dict": lambda self: {"weight": weight}})()
                cached = SimpleNamespace(decoder_dict={"standard": decoder},
                                         named_parameters=lambda: [("weight", weight)])
                model = SimpleNamespace(named_parameters=lambda: [("weight", weight)])
                fitted = SimpleNamespace(source=name, serial=sum(item[0] == "fit" for item in trace),
                                         model=model, cached_models_=[cached])
                trace.fitted_models.append(fitted)
                trace.append(("fit", name, id(fitted), device, seed, context[["f0"]].to_numpy().copy()))
                return {"classifier": fitted, "context_identity": "shared-context",
                        "context_rows": "same-rows", "reference_year": 2007,
                        "feature_names": tuple(columns), "checkpoint_sha256": replay.CHECKPOINT_SHA256,
                        "settings": {"precision": {"amp": "off"}, "seed": 42,
                                     "model_type": "single_fast", "fit_at_predict_time": False,
                                     "fp16_inference": False, "sklearn_compatible_precision": False,
                                     "domain_policy": "calendar_year_minus_reference_year"}}

            @staticmethod
            def capture_frozen_predictions(system, frame, device):
                ids = frame.patient_id.tolist()
                trace.append(("capture", ids, frame.year.tolist(), device, id(system["classifier"])))
                if fail_capture:
                    raise RuntimeError("synthetic capture failure")
                values = frame.f0.to_numpy(dtype=np.float32)
                return {"probabilities": np.column_stack((1 - values / 100, values / 100)),
                        "embeddings": np.column_stack((values, values + 1))}

            @staticmethod
            def predictor_identity_report(classifier):
                identity_names = ("X_", "y_", "_categorical_features", "additional_x_", "categorical_features",
                                  "classes_", "c_processed_", "cached_preprocessors_", "cached_shuffles_",
                                  "data_mean_", "data_std_", "average_logits", "softmax_temperature",
                                  "ensemble_configurations_", "transformer_predict_kwargs", "preprocess_transforms",
                                  "feature_shift_decoder", "normalize_with_test", "normalize_to_ranking",
                                  "normalize_to_clip", "remove_outliers", "device_", "fit_at_predict_time",
                                  "fp16_inference", "sklearn_compatible_precision", "pilot_checkpoint_sha256_",
                                  "pilot_precision_", "pilot_domain_policy_")
                fields = {key: "same-config" for key in identity_names}
                if source_mismatch == "config" and classifier.source == "current":
                    fields["X_"] = "changed-config"
                return {"identity": f"identity-{classifier.source}-{classifier.serial}",
                        "field_digests": fields}
        return Pilot

    probes = {year: {"identity": (f"p{0 if year == 2007 else year - 2004}", year),
                     "population_row": 0 if year == 2007 else year - 2004,
                     "batch_position": 0 if year == 2007 else year - 2004}
              for year in range(2007, 2015)}
    held_selection = {"held_rows": np.arange(8), "held_actual_rows": held_rows,
                      "precheck_rows": np.arange(4, 8),
                      "held_identities": [(f"p{i}", int(actual["year"][i])) for i in held_rows]}
    inputs = {
        "population": population,
        "actual": actual,
        "held": held,
        "prepared": {"context_rows": np.array([0, 1]), "query_rows": np.arange(16),
                     "probes": probes, "actual_probe_domains": {year: year - 2007 for year in probes},
                     "held_selection": held_selection, "context_identity": "same-rows",
                     "held_probe_rows": {year: year - 2008 for year in range(2008, 2015)},
                     "held_probe_domains": {year: 0 for year in range(2008, 2015)}},
        "recovered_pilot": make_pilot("recovered"),
        "current_pilot": make_pilot("current"),
        "held_probes": {year: {"identity": (f"p{year - 2004}", year),
                               "archive_row": year - 2004,
                               "batch_position": year - 2008}
                        for year in range(2008, 2015)},
        "config": {"split_seed": 42},
        "saved_meta": {"batch_rows": 4},
        "fingerprint_facts": {"population_sha256": "pop", "actual_archive_sha256": "actual-fixture",
                              "checkpoint_sha256": "checkpoint", "dependency_sha256": "dep"},
        "source_hashes": {"recovered": replay.RECOVERED_SOURCE_SHA256, "current": replay.CURRENT_SOURCE_SHA256},
        "dependency_identity": {"method": "fake-tree", "sha256": "dep"},
        "runtime": {"device": "cpu", "threads": 1, "precision": "fp32", "seed": 42},
        "gate_rows": np.arange(4),
        "query_batch_size": 16,
        "held_batch_size": 8,
        "precheck_size": 4,
        "targets": {"actual_probabilities": np.arange(8, dtype=np.float32),
                    "actual_embeddings": np.zeros((8, 2), dtype=np.float32),
                    "held_probabilities": np.arange(7, dtype=np.float32)},
    }
    return inputs, trace


def _synthetic_population():
    return {
        "X": np.arange(16, dtype=np.float32).reshape(8, 2),
        "y": np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        "year": np.array([2007, 2008, 2009, 2010, 2015, 2007, 2008, 2015]),
        "patient_id": np.array([f"p{i}" for i in range(8)]),
        "record_keys": np.array([f"r{i}" for i in range(8)]),
        "feature_names": ("x0", "x1"),
        "post_death": np.zeros(8, dtype=bool),
    }


def _synthetic_roles(rows):
    row_role = np.array(["tabpfn_context", "heldout", "heldout", "heldout",
                         "entrant", "tabpfn_context", "heldout", "entrant"])
    is_entrant = np.array([False, False, False, False, True, False, False, True])
    return {"reference_year": 2007, "identity": "roles-v1", "row_role": row_role[:rows],
            "is_entrant": is_entrant[:rows]}


def _synthetic_archives(population):
    actual_positions = np.array([0, 1, 2, 3, 5, 6, 4, 7])
    actual_years = population["year"][actual_positions]
    actual_labels = population["y"][actual_positions].astype(np.float32)
    actual_labels[actual_years > 2014] = np.nan
    actual = {"patient_id": population["patient_id"][actual_positions], "year": actual_years,
              "features": population["X"][actual_positions], "label": actual_labels,
              "feature_names": np.array(population["feature_names"])}
    held_positions = np.array([1, 2, 3, 5, 6, 7])
    held = {"patient_id": actual["patient_id"][held_positions],
            "year": actual["year"][held_positions], "positions": held_positions}
    return actual, held


def _synthetic_pilot():
    class Pilot:
        @staticmethod
        def _identity_hash(value):
            def normalized(item):
                if isinstance(item, np.generic):
                    return item.item()
                if isinstance(item, np.ndarray):
                    return item.tolist()
                if isinstance(item, dict):
                    return {key: normalized(val) for key, val in item.items()}
                if isinstance(item, (tuple, list)):
                    return [normalized(val) for val in item]
                return item
            import hashlib
            import json
            return hashlib.sha256(json.dumps(normalized(value), sort_keys=True).encode()).hexdigest()

        @staticmethod
        def _feature_columns(count):
            return [f"f{i}" for i in range(count)]

        @staticmethod
        def _prediction_frame(population, rows, columns):
            return type("Frame", (), {"patient_id": population["patient_id"][rows],
                                      "year": population["year"][rows]})()

    return Pilot


def _synthetic_meta(population, roles, actual, held):
    pilot = _synthetic_pilot()
    context_rows = np.array([0, 5])
    frame = pilot._prediction_frame(population, context_rows, ["f0", "f1"])
    return {
        "population_identity": pilot._identity_hash({"X": population["X"], "y": population["y"],
            "year": population["year"], "patient_id": population["patient_id"],
            "record_keys": population["record_keys"], "feature_names": list(population["feature_names"])}),
        "roles_identity": roles["identity"],
        "context_rows_identity": pilot._identity_hash(list(zip(frame.patient_id, frame.year))),
        "batch_rows": 4, "start_batch_rows": 4, "oom_backoffs": [], "_rows": 8,
        "inference_chunks": 2, "context_rows": len(context_rows),
        "feature_count": 2, "reference_year": 2007,
        "settings": {"domain_policy": "calendar_year_minus_reference_year", "seed": 42},
        "output_row_identity": pilot._identity_hash({"row_index": np.arange(8)}),
    }

