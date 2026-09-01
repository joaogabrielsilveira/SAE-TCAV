"""Scientific and runtime configuration with separate identity semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .profiles import METRIC_PROFILES


@dataclass(frozen=True)
class MetricSynthesisConfig:
    """Settings that can change a scientific result and therefore its hash."""

    profiles: tuple[str, ...] = ("core", "p50_tcav_extended")
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-3, 1e-2, .1, 1., 10., 100.)
    parallel_repetitions: int = 1000
    bootstrap_repetitions: int = 1000
    seed: int = 42
    dae_profile: str | None = "core"
    dae_latent_dimensions: int | None = 2
    dae_epochs: int = 250
    dae_early_stopping_patience: int = 30
    dae_validation_seeds: tuple[int, ...] = (42, 43, 44)
    material_degradation_noise_quantile: float = .95

    def __post_init__(self) -> None:
        if not set(self.profiles).issubset(METRIC_PROFILES):
            raise ValueError("unknown metric profile")
        if self.parallel_repetitions < 10:
            raise ValueError("parallel analysis needs at least 10 repetitions")
        if not self.ridge_alphas or any(alpha <= 0 for alpha in self.ridge_alphas):
            raise ValueError("ridge alphas must be nonempty and positive")
        if self.bootstrap_repetitions < 0:
            raise ValueError("bootstrap repetitions cannot be negative")
        if (self.dae_profile is None) != (self.dae_latent_dimensions is None):
            raise ValueError("dae_profile and dae_latent_dimensions must be supplied together")
        if self.dae_profile is not None and self.dae_profile not in self.profiles:
            raise ValueError("DAE profile must be analyzed")
        if self.dae_latent_dimensions is not None and self.dae_latent_dimensions < 1:
            raise ValueError("DAE dimensions must be positive")
        if self.dae_epochs < 1 or self.dae_early_stopping_patience < 1:
            raise ValueError("DAE epochs and patience must be positive")
        if not self.dae_validation_seeds:
            raise ValueError("at least one DAE seed is required")
        if not 0 < self.material_degradation_noise_quantile < 1:
            raise ValueError("noise quantile must lie strictly between zero and one")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricSynthesisRuntimeConfig:
    """Execution settings excluded from scientific artifact identity."""

    device: str = "auto"
    executor: str = "auto"
    workers: int | str = "auto"
    progress: bool = True
    resume: bool = True
    log_file: str | None = "temporal-metric-synthesis.log"

    def __post_init__(self) -> None:
        if self.device != "auto" and self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be auto, cpu, or a CUDA device")
        if self.executor not in {"auto", "serial", "thread", "process"}:
            raise ValueError("executor must be auto, serial, thread, or process")
        if self.workers != "auto" and (not isinstance(self.workers, int) or self.workers < 1):
            raise ValueError("workers must be auto or a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
