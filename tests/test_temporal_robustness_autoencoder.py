import numpy as np
import pytest

torch = pytest.importorskip("torch")

from temporal_robustness_autoencoder import (
    DenoisingAutoencoderConfig, corrupt_utilities, corrupt_utilities_device, fit_denoising_autoencoder,
    fit_utility_preprocessor, select_device,
)


def test_corruption_is_bounded_and_clean_target_is_not_modified():
    values=np.asarray([[0., .2], [.8, 1.]], dtype=np.float32)
    result=corrupt_utilities(values,DenoisingAutoencoderConfig(1, dropout_probability=.5, noise_std=.1, output_activation="sigmoid"),np.random.default_rng(4))
    assert result.shape == values.shape
    assert np.all((result >= 0) & (result <= 1))
    assert values[0, 0] == 0


def test_train_only_standard_and_logit_transforms_round_trip_to_utility_scale():
    values=np.asarray([[.01, .2], [.3, .8], [.99, .6]], dtype=np.float32)
    for mode in ("standard_linear", "logit_linear"):
        preprocessor=fit_utility_preprocessor(values[:2], mode)
        recovered=preprocessor.inverse_transform(preprocessor.transform(values))
        assert np.allclose(recovered, values, atol=2e-5)
    transformed=fit_utility_preprocessor(values[:2], "standard_linear").transform(values)
    corrupted=corrupt_utilities(transformed,DenoisingAutoencoderConfig(1, noise_std=.1),np.random.default_rng(1))
    assert np.isfinite(corrupted).all()  # transformed values are intentionally not clipped to [0, 1]


def test_one_dimensional_encoder_is_coordinatewise_monotone():
    rng=np.random.default_rng(3); values=rng.uniform(0,1,(24,4)).astype(np.float32)
    fitted=fit_denoising_autoencoder(values,DenoisingAutoencoderConfig(1, hidden_dimensions=5, epochs=4, seed=7))
    model=fitted["model"]
    left=torch.as_tensor(values[:8]); right=torch.clamp(left+.05,max=1)
    with torch.no_grad():
        assert torch.all(model.encode(right) >= model.encode(left)-1e-7)
    assert fitted["latent"].shape == (24,1)
    assert fitted["monotone_score"] is True


def test_validation_is_used_only_for_early_stopping_checkpoint_selection():
    rng=np.random.default_rng(8); train=rng.normal(size=(20,3)).astype(np.float32); validation=rng.normal(size=(7,3)).astype(np.float32)
    fitted=fit_denoising_autoencoder(train,DenoisingAutoencoderConfig(2, hidden_dimensions=4, epochs=6, early_stopping_patience=2, seed=9),validation,progress=False)
    assert fitted["best_epoch"] >= 1
    assert fitted["validation_reconstruction"].shape == validation.shape
    assert all("validation_mse" in row for row in fitted["history"])


def test_device_auto_is_cpu_or_cuda():
    assert select_device() in {"cpu", "cuda"}


def test_device_native_corruption_is_deterministic():
    values = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12
    config = DenoisingAutoencoderConfig(2, dropout_probability=.2, noise_std=.1)
    left = torch.Generator(device=values.device).manual_seed(19)
    right = torch.Generator(device=values.device).manual_seed(19)
    assert torch.equal(corrupt_utilities_device(values, config, left),
                       corrupt_utilities_device(values, config, right))
