"""Outcome-free denoising representation used by temporal metric synthesis.

This module has no performance/outcome argument.  It operates only on the
reference-oriented utility matrix and supports grouped validation in the
caller.  Scaling is deliberately fitted on the training matrix of each split.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import logging
import threading
from typing import Any, Literal

import numpy as np

LOGGER = logging.getLogger(__name__)
_INITIALIZATION_LOCK = threading.Lock()


def _epoch_progress(epochs: int):
    if not LOGGER.isEnabledFor(logging.INFO):
        return range(epochs)
    from tqdm.auto import trange
    return trange(epochs, desc="DAE training", unit="epoch", leave=False)


@dataclass(frozen=True)
class UtilityPreprocessor:
    """A train-only utility transform with an exact inverse to utility scale."""
    mode: Literal["raw_sigmoid", "standard_linear", "logit_linear"]
    location: np.ndarray
    scale: np.ndarray
    logit_epsilon: float = 1e-4

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if self.mode == "raw_sigmoid":
            return np.clip(values, 0., 1.)
        if self.mode == "logit_linear":
            values = np.clip(values, self.logit_epsilon, 1. - self.logit_epsilon)
            values = np.log(values / (1. - values))
        return ((values - self.location) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if self.mode == "raw_sigmoid":
            return np.clip(values, 0., 1.)
        values = values * self.scale + self.location
        if self.mode == "logit_linear":
            values = 1. / (1. + np.exp(-np.clip(values, -30., 30.)))
        return np.clip(values, 0., 1.).astype(np.float32)

    def checkpoint_arrays(self) -> dict[str, np.ndarray]:
        return {"preprocessor_location": self.location, "preprocessor_scale": self.scale,
                "preprocessor_mode": np.asarray([self.mode], dtype="U"),
                "preprocessor_logit_epsilon": np.asarray([self.logit_epsilon], dtype=np.float32)}


def fit_utility_preprocessor(values: np.ndarray, mode: str) -> UtilityPreprocessor:
    """Fit a transform *only* on a training matrix."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("utility values must be a finite two-dimensional matrix")
    if mode not in {"raw_sigmoid", "standard_linear", "logit_linear"}:
        raise ValueError(f"unknown utility preprocessing mode: {mode}")
    if mode == "raw_sigmoid":
        return UtilityPreprocessor(mode=mode, location=np.zeros(values.shape[1], dtype=np.float32), scale=np.ones(values.shape[1], dtype=np.float32))
    work = values
    if mode == "logit_linear":
        work = np.clip(work, 1e-4, 1. - 1e-4)
        work = np.log(work / (1. - work))
    location = work.mean(axis=0).astype(np.float32)
    scale = np.maximum(work.std(axis=0), 1e-6).astype(np.float32)
    return UtilityPreprocessor(mode=mode, location=location, scale=scale)


@dataclass(frozen=True)
class DenoisingAutoencoderConfig:
    latent_dimensions: int
    hidden_dimensions: int = 8
    dropout_probability: float = .05
    noise_std: float = .01
    epochs: int = 250
    early_stopping_patience: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    output_activation: Literal["linear", "sigmoid"] = "linear"
    metric_loss_weights: tuple[float, ...] | None = None
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.latent_dimensions < 1 or self.hidden_dimensions < 1:
            raise ValueError("network dimensions must be positive")
        if not 0 <= self.dropout_probability < 1 or self.noise_std < 0:
            raise ValueError("corruption parameters are invalid")
        if self.epochs < 1 or self.early_stopping_patience < 1:
            raise ValueError("epochs and patience must be positive")
        if self.output_activation not in {"linear", "sigmoid"}:
            raise ValueError("output activation must be linear or sigmoid")
        if self.metric_loss_weights is not None and any(x <= 0 for x in self.metric_loss_weights):
            raise ValueError("metric loss weights must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_device(requested: str = "auto") -> str:
    """Return a deterministic device choice without requiring CUDA."""
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def corrupt_utilities(values: np.ndarray, config: DenoisingAutoencoderConfig,
                      rng: np.random.Generator, replacement: np.ndarray | None = None) -> np.ndarray:
    """Drop features to their train-set centre and add noise.

    ``replacement`` is zero in standardized/logit space and the per-feature
    training mean in raw utility space.  This avoids an artificial .5
    intervention for boundary-clustered utilities.
    """
    clean = np.asarray(values, dtype=np.float32)
    centre = np.zeros(clean.shape[1], dtype=np.float32) if replacement is None else np.asarray(replacement, dtype=np.float32)
    if centre.shape != (clean.shape[1],):
        raise ValueError("replacement must have one value per utility")
    keep = rng.random(clean.shape) >= config.dropout_probability
    noisy = np.where(keep, clean, centre[None, :])
    if config.noise_std:
        noisy = noisy + rng.normal(0, config.noise_std, clean.shape).astype(np.float32)
    if config.output_activation == "sigmoid":
        noisy = np.clip(noisy, 0., 1.)
    return noisy.astype(np.float32, copy=False)


def corrupt_utilities_device(values, config: DenoisingAutoencoderConfig, generator, replacement=None):
    """Generate deterministic corruption directly on a Torch tensor's device."""
    import torch
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional tensor")
    centre = torch.zeros(values.shape[1], dtype=values.dtype, device=values.device) if replacement is None else replacement
    if tuple(centre.shape) != (values.shape[1],):
        raise ValueError("replacement must have one value per utility")
    keep = torch.rand(values.shape, dtype=values.dtype, device=values.device, generator=generator) >= config.dropout_probability
    noisy = torch.where(keep, values, centre.unsqueeze(0))
    if config.noise_std:
        noise = torch.randn(values.shape, dtype=values.dtype, device=values.device, generator=generator)
        noisy = noisy + noise * config.noise_std
    return noisy.clamp(0., 1.) if config.output_activation == "sigmoid" else noisy


def fit_denoising_autoencoder(values: np.ndarray, config: DenoisingAutoencoderConfig,
                              validation_values: np.ndarray | None = None, *, progress: bool = True,
                              search_mode: bool = False) -> dict[str, Any]:
    """Fit a tiny DAE with clean grouped-validation early stopping.

    Values have already been transformed by a train-only preprocessor.
    ``validation_values`` is never used for fitting or corruption, only for
    checkpoint selection.
    """
    import torch
    from torch import nn
    clean = np.asarray(values, dtype=np.float32)
    validation = None if validation_values is None else np.asarray(validation_values, dtype=np.float32)
    if clean.ndim != 2 or clean.shape[0] < 2 or not np.isfinite(clean).all():
        raise ValueError("values must be a finite matrix with at least two rows")
    if validation is not None and (validation.ndim != 2 or validation.shape[1] != clean.shape[1] or not np.isfinite(validation).all()):
        raise ValueError("validation values must be a finite compatible matrix")
    if config.latent_dimensions > clean.shape[1]:
        raise ValueError("latent dimensions exceed input dimensions")
    if config.metric_loss_weights is not None and len(config.metric_loss_weights) != clean.shape[1]:
        raise ValueError("metric loss weights must match utility columns")
    device = select_device(config.device)
    if progress:
        LOGGER.info("Starting outcome-free DAE: %d train × %d utilities, %s validation, k=%d, device=%s", clean.shape[0], clean.shape[1], 0 if validation is None else validation.shape[0], config.latent_dimensions, device)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.monotone = config.latent_dimensions == 1
            self.first = nn.Linear(clean.shape[1], config.hidden_dimensions, bias=True)
            self.latent = nn.Linear(config.hidden_dimensions, config.latent_dimensions, bias=True)
            decoder = [nn.Linear(config.latent_dimensions, config.hidden_dimensions), nn.ReLU(), nn.Linear(config.hidden_dimensions, clean.shape[1])]
            if config.output_activation == "sigmoid": decoder.append(nn.Sigmoid())
            self.decoder = nn.Sequential(*decoder)
        def encode(self, x):
            if not self.monotone:
                return self.latent(torch.relu(self.first(x)))
            first = torch.nn.functional.linear(x, torch.nn.functional.softplus(self.first.weight), self.first.bias)
            hidden = torch.nn.functional.softplus(first)
            return torch.nn.functional.linear(hidden, torch.nn.functional.softplus(self.latent.weight), self.latent.bias)
        def forward(self, x):
            z = self.encode(x)
            return z, self.decoder(z)

    # Torch initialization uses global RNG state.  Isolate it so concurrent
    # CUDA thread schedules cannot alter model weights or the caller's RNG.
    cuda_devices = [torch.device(device).index or 0] if device.startswith("cuda") else []
    with _INITIALIZATION_LOCK, torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(config.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(config.seed)
        model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    target = torch.as_tensor(clean, device=device)
    validation_target = None if validation is None else torch.as_tensor(validation, device=device)
    weights = torch.ones(clean.shape[1], dtype=torch.float32, device=device) if config.metric_loss_weights is None else torch.as_tensor(config.metric_loss_weights, dtype=torch.float32, device=device)
    weights = weights / weights.mean()
    replacement = clean.mean(axis=0) if config.output_activation == "sigmoid" else np.zeros(clean.shape[1], dtype=np.float32)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    history: list[dict[str, float | int | None]] = []
    best_state, best_epoch, best_validation = None, 0, np.inf
    stale_epochs = 0
    for epoch in (_epoch_progress(config.epochs) if progress else range(config.epochs)):
        model.train()
        corrupted = corrupt_utilities_device(target, config, generator,
                                             torch.as_tensor(replacement, device=device))
        _, reconstruction = model(corrupted)
        loss = torch.mean((reconstruction - target).square() * weights)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            train_mse = None
            if validation_target is None or not search_mode:
                _, clean_reconstruction = model(target)
                train_mse = float(torch.mean((clean_reconstruction - target).square() * weights).cpu())
            validation_mse = None if validation_target is None else float(torch.mean((model(validation_target)[1] - validation_target).square() * weights).cpu())
        monitored = train_mse if validation_mse is None else validation_mse
        history.append({"epoch": epoch + 1, "training_mse": train_mse, "validation_mse": validation_mse})
        if monitored < best_validation - 1e-10:
            best_state, best_epoch, best_validation = deepcopy(model.state_dict()), epoch + 1, monitored
            stale_epochs = 0
        else:
            stale_epochs += 1
            if validation is not None and stale_epochs >= config.early_stopping_patience:
                if progress: LOGGER.info("DAE early stopped at epoch %d; best grouped-validation epoch=%d", epoch + 1, best_epoch)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        latent, reconstruction = model(target)
        validation_reconstruction = None if validation_target is None else model(validation_target)[1]
    if progress:
        LOGGER.info("Completed DAE training on %s; selected epoch=%d, monitored MSE=%.6g", device, best_epoch, best_validation)
    return {"model": model, "latent": latent.cpu().numpy(), "reconstruction": reconstruction.cpu().numpy(),
            "validation_reconstruction": None if validation_reconstruction is None else validation_reconstruction.cpu().numpy(),
            "history": history, "device": device, "torch_version": torch.__version__, "seed": config.seed,
            "monotone_score": config.latent_dimensions == 1, "best_epoch": best_epoch,
            "best_validation_mse": float(best_validation)}
