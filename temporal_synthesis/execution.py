"""Deterministic search jobs, runtime validation, and resumable checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence, TypeVar

from .config import MetricSynthesisRuntimeConfig

T = TypeVar("T")
_EXECUTOR_CACHE: dict[tuple[str, int], Any] = {}


def _cpu_worker_initialization() -> None:
    import torch
    torch.set_num_threads(1)


@dataclass(frozen=True, order=True)
class SearchJob:
    outer_year: int
    inner_year: int
    candidate_index: int
    validation_seed: int

    @property
    def seed(self) -> int:
        payload = f"{self.outer_year}:{self.inner_year}:{self.candidate_index}:{self.validation_seed}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7fff_ffff

    @property
    def key(self) -> str:
        return f"outer-{self.outer_year}.inner-{self.inner_year}.candidate-{self.candidate_index}.seed-{self.validation_seed}"


@dataclass(frozen=True)
class ExecutorSelection:
    device: str
    executor: str
    workers: int
    selection_reason: str


def representative_benchmark_jobs(reference_years: Sequence[int], candidate_count: int = 216) -> list[SearchJob]:
    """Return the fixed, schedule-independent 28-job benchmark sample."""
    references = tuple(sorted({int(year) for year in reference_years}))
    if len(references) < 2 or candidate_count < 1:
        raise ValueError("benchmark needs at least two references and one candidate")
    jobs: list[SearchJob] = []
    for index in range(28):
        outer = references[index % len(references)]
        inner = references[(index + 1) % len(references)]
        jobs.append(SearchJob(outer, inner, 1 + (index * max(1, candidate_count // 28)) % candidate_count, 42))
    return jobs


def select_fastest_benchmark(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select the valid benchmark with greatest throughput deterministically."""
    valid = [row for row in records if row.get("valid") is True and float(row.get("jobs_per_second", 0.)) > 0]
    if not valid:
        raise RuntimeError("no valid executor benchmark")
    return max(valid, key=lambda row: (float(row["jobs_per_second"]), -int(row.get("workers", 1)), str(row.get("executor", ""))))


def run_search_jobs(jobs: Sequence[SearchJob], function: Callable[[SearchJob], T],
                    selection: ExecutorSelection) -> list[T]:
    """Execute jobs concurrently while returning stable job-key order."""
    ordered = sorted(jobs)
    if selection.executor == "serial":
        return [function(job) for job in ordered]
    executor_type = ThreadPoolExecutor if selection.executor == "thread" else ProcessPoolExecutor
    kwargs: dict[str, Any] = {"max_workers": selection.workers}
    if executor_type is ProcessPoolExecutor:
        kwargs["initializer"] = _cpu_worker_initialization
    cache_key = (selection.executor, selection.workers)
    executor = _EXECUTOR_CACHE.get(cache_key)
    if executor is None:
        executor = executor_type(**kwargs)
        _EXECUTOR_CACHE[cache_key] = executor
    futures = {job: executor.submit(function, job) for job in ordered}
    return [futures[job].result() for job in ordered]


def shutdown_search_executors() -> None:
    """Release persistent pools after a synthesis run."""
    for executor in _EXECUTOR_CACHE.values():
        executor.shutdown(wait=True)
    _EXECUTOR_CACHE.clear()


def resolve_executor(runtime: MetricSynthesisRuntimeConfig) -> ExecutorSelection:
    """Resolve strict overrides; auto uses safe deterministic platform defaults."""
    import os
    import torch

    device = runtime.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    executor = runtime.executor
    if executor == "auto":
        executor = "thread" if device.startswith("cuda") else "process"
    if device.startswith("cuda") and executor == "process":
        raise ValueError("CUDA search requires the thread or serial executor")
    if device == "cpu" and executor == "thread":
        raise ValueError("CPU search requires the process or serial executor")
    available = max(1, os.cpu_count() or 1)
    workers = min(28, available) if runtime.workers == "auto" else int(runtime.workers)
    if executor == "serial":
        if runtime.workers != "auto" and workers != 1:
            raise ValueError("serial executor requires exactly one worker")
        workers = 1
    return ExecutorSelection(device, executor, workers, "validated_override" if "auto" not in (runtime.device, runtime.executor, runtime.workers) else "automatic_resource_resolution")


class ResumeStore:
    """Atomically persist completed groups under a validated identity."""

    def __init__(self, root: Path, identity: Mapping[str, Any], enabled: bool = True):
        self.root = root
        self.identity = dict(identity)
        self.enabled = enabled
        if enabled:
            root.mkdir(parents=True, exist_ok=True)
            identity_path = root / "identity.json"
            if identity_path.exists() and json.loads(identity_path.read_text()) != self.identity:
                raise RuntimeError("resume work directory identity does not match this run")
            if not identity_path.exists():
                self._write(identity_path, self.identity)

    @staticmethod
    def _write(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as output:
                json.dump(value, output, sort_keys=True, separators=(",", ":"))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_name, path)
        finally:
            Path(temporary_name).unlink(missing_ok=True)

    def load(self, group: str) -> Any | None:
        path = self.root / f"{group}.json"
        return json.loads(path.read_text()) if self.enabled and path.exists() else None

    def save(self, group: str, value: Any) -> None:
        if self.enabled:
            self._write(self.root / f"{group}.json", value)
