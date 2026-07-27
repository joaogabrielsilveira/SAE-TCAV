import json

import pytest

from runtime_acceleration import StageTelemetry, resolve_torch_device


def test_cpu_device_resolution_is_explicit():
    assert resolve_torch_device("cpu") == "cpu"


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_torch_device("tpu")


def test_stage_telemetry_persists_success_and_failure(tmp_path):
    output = tmp_path / "stage_metrics.json"
    telemetry = StageTelemetry(output, requested_device="cpu")

    with telemetry.measure("success"):
        pass
    with pytest.raises(RuntimeError, match="expected"):
        with telemetry.measure("failure"):
            raise RuntimeError("expected")

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["success"]["status"] == "completed"
    assert persisted["failure"]["status"] == "failed"
    assert persisted["failure"]["error_type"] == "RuntimeError"
    assert all(row["seconds"] >= 0 for row in persisted.values())
