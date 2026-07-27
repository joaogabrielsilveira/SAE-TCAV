"""Small accelerator and runtime-telemetry interface for comparison runs."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Iterator


SUPPORTED_DEVICES = {"auto", "cpu", "cuda"}


def resolve_torch_device(requested: str = "auto") -> str:
    """Resolve ``auto`` and fail early when explicitly requested CUDA is absent."""

    if requested not in SUPPORTED_DEVICES:
        raise ValueError(
            f"device must be one of {sorted(SUPPORTED_DEVICES)}, got {requested!r}"
        )
    try:
        import torch
    except ImportError as error:
        if requested == "cuda":
            raise RuntimeError("CUDA was requested, but PyTorch is unavailable") from error
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def accelerator_manifest(requested: str = "auto") -> dict[str, object]:
    """Describe the accelerator selected for the current process."""

    resolved = resolve_torch_device(requested)
    result: dict[str, object] = {
        "requested_device": requested,
        "resolved_device": resolved,
        "cuda_available": False,
    }
    try:
        import torch
    except ImportError:
        result["torch_version"] = "unavailable"
        return result

    result["torch_version"] = str(torch.__version__)
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["cuda_version"] = torch.version.cuda
    if resolved == "cuda":
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "cuda_device_index": int(index),
                "cuda_device_name": torch.cuda.get_device_name(index),
                "cuda_compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
                "cuda_total_memory_bytes": int(properties.total_memory),
            }
        )
    return result


class StageTelemetry:
    """Persist stage duration and peak CUDA memory after every completed stage."""

    def __init__(self, output_path: str | Path, requested_device: str = "auto"):
        self.output_path = Path(output_path)
        self.device = resolve_torch_device(requested_device)
        self.records: dict[str, dict[str, object]] = {}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        if not stage or stage in self.records:
            raise ValueError(f"Stage name must be unique and non-empty: {stage!r}")

        torch = _torch_for_cuda(self.device)
        if torch is not None:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        status = "completed"
        error_type = None
        try:
            yield
        except BaseException as error:
            status = "failed"
            error_type = type(error).__name__
            raise
        finally:
            if torch is not None:
                torch.cuda.synchronize()
            record: dict[str, object] = {
                "seconds": float(time.perf_counter() - started),
                "status": status,
                "device": self.device,
            }
            if error_type is not None:
                record["error_type"] = error_type
            if torch is not None:
                record.update(
                    {
                        "cuda_peak_allocated_bytes": int(
                            torch.cuda.max_memory_allocated()
                        ),
                        "cuda_peak_reserved_bytes": int(
                            torch.cuda.max_memory_reserved()
                        ),
                    }
                )
            self.records[stage] = record
            self._persist()

    def _persist(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.records, handle, sort_keys=True, indent=2)
        temporary.replace(self.output_path)


def _torch_for_cuda(device: str):
    if device != "cuda":
        return None
    import torch

    return torch


__all__ = [
    "StageTelemetry",
    "accelerator_manifest",
    "resolve_torch_device",
]
