import importlib.util
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch

import decision_tree

if importlib.util.find_spec("tabpfn") is None:
    tabpfn_stub = types.ModuleType("tabpfn")
    tabpfn_stub.__path__ = []
    tabpfn_stub.TabPFNClassifier = type("TabPFNClassifier", (), {})
    best_models_stub = types.ModuleType("tabpfn.best_models")
    best_models_stub.get_best_tabpfn = lambda **_: None
    best_models_stub.TabPFNModelPathsConfig = type(
        "TabPFNModelPathsConfig",
        (),
        {"__init__": lambda self, **_: None},
    )
    sys.modules["tabpfn"] = tabpfn_stub
    sys.modules["tabpfn.best_models"] = best_models_stub

import sae_compare
import tcav
from sae import SAE, train_sae_model


def test_train_sae_model_infers_input_dimension():
    model = train_sae_model(
        torch.zeros((4, 7)),
        epochs=0,
        save_data=False,
        scaling_factor=2.0,
    )

    assert model.encoder.in_features == 7
    assert model.num_latents == 14


def test_train_sae_model_rejects_mismatched_explicit_dimension():
    with pytest.raises(ValueError, match="data_dimension"):
        train_sae_model(
            torch.zeros((4, 7)),
            epochs=0,
            save_data=False,
            data_dimension=8,
        )


def test_train_all_saes_forwards_explicit_training_controls(monkeypatch):
    calls = []

    def fake_train_sae_model(**kwargs):
        calls.append(kwargs)
        return SAE(
            data_dimension=kwargs["inputs"].shape[1],
            scaling_factor=kwargs["scaling_factor"],
            use_decoder_bias=True,
            type=kwargs["type"],
            k=kwargs["k"],
            k_aux=kwargs["k_aux"],
        )

    monkeypatch.setattr(sae_compare, "train_sae_model", fake_train_sae_model)
    runs = sae_compare.train_all_saes(
        2,
        np.ones((5, 3), dtype=np.float32),
        scaling_factor=1.0,
        seeds=[11, 22],
        epochs=3,
        learning_rate=0.02,
        weight_decay=0.04,
        device="cpu",
        encoding_batch_size=2,
        show_progress=False,
    )

    assert [run["seed"] for run in runs] == [11, 22]
    assert [call["rng_seed"] for call in calls] == [11, 22]
    assert all(call["epochs"] == 3 for call in calls)
    assert all(call["learning_rate"] == 0.02 for call in calls)
    assert all(call["weight_decay"] == 0.04 for call in calls)
    assert all(call["device"] == "cpu" for call in calls)
    assert all(call["show_progress"] is False for call in calls)


def test_train_all_saes_preserves_default_seed_schedule(monkeypatch):
    calls = []

    def fake_train_sae_model(**kwargs):
        calls.append(kwargs)
        return SAE(
            data_dimension=kwargs["inputs"].shape[1],
            scaling_factor=kwargs["scaling_factor"],
            use_decoder_bias=True,
            type=kwargs["type"],
            k=kwargs["k"],
            k_aux=kwargs["k_aux"],
        )

    monkeypatch.setattr(sae_compare, "train_sae_model", fake_train_sae_model)
    sae_compare.train_all_saes(
        2,
        np.ones((5, 3), dtype=np.float32),
        scaling_factor=1.0,
    )

    assert [call["rng_seed"] for call in calls] == [42, 135]
    assert all(call["epochs"] == 1000 for call in calls)
    assert all(call["learning_rate"] == 1e-3 for call in calls)
    assert all(call["weight_decay"] == 0.0 for call in calls)


def test_train_all_saes_requires_one_seed_per_run():
    with pytest.raises(ValueError, match="exactly num_models"):
        sae_compare.train_all_saes(
            2,
            np.ones((5, 3), dtype=np.float32),
            seeds=[11],
        )


def test_forced_rule_graph_export_accepts_path_output_directory(tmp_path):
    activations = np.arange(1, 9, dtype=np.float32).reshape(-1, 1)
    features = np.arange(8, dtype=np.float32).reshape(-1, 1)
    empty_rules = pd.DataFrame(
        columns=[
            "Factor",
            "Rule",
            "Class",
            "Precision",
            "Recall",
            "Patients",
            "Patients_concept",
        ]
    )

    decision_tree.get_rules_forced(
        train_activations=activations,
        X=features,
        surviving_concepts=np.asarray([0]),
        tree_rules_df=empty_rules,
        perc=50,
        feature_names=["feature"],
        graph_output_dir=tmp_path / "graphs",
    )

    assert (tmp_path / "graphs" / "0.dot").is_file()


def test_encode_sae_batches_without_changing_outputs_or_model_device():
    model = SAE(
        data_dimension=3,
        scaling_factor=1.0,
        use_decoder_bias=True,
        type="ReLU",
    )
    values = np.arange(21, dtype=np.float32).reshape(7, 3)
    with torch.inference_mode():
        expected = model.encode(torch.as_tensor(values)).numpy()

    actual = sae_compare.encode_sae(
        {"model": model, "model_type": "ReLU"},
        values,
        device="cpu",
        batch_size=2,
    )

    np.testing.assert_allclose(actual, expected)
    assert next(model.parameters()).device.type == "cpu"


def test_tcav_gradient_batch_size_controls_chunking(tmp_path):
    class FakeModel:
        def __init__(self):
            decoder = torch.nn.Linear(3, 2, bias=False)
            self.model_processed_ = types.SimpleNamespace(
                decoder_dict={"standard": decoder}
            )
            self.batch_sizes = []

        def get_embeddings(self, values, additional_x):
            self.batch_sizes.append(len(values))
            return torch.as_tensor(values, dtype=torch.float32)

    model = FakeModel()
    values = np.arange(15, dtype=np.float32).reshape(5, 3)
    gradients = tcav.get_model_gradients(
        model,
        np.arange(5),
        values,
        cache_file=tmp_path / "gradients.pkl",
        batch_size=2,
        device="cpu",
    )

    assert model.batch_sizes == [2, 2, 1]
    assert gradients.shape == values.shape


def test_tcav_gradient_batch_size_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        tcav.get_model_gradients(
            object(),
            np.arange(1),
            np.zeros((1, 3), dtype=np.float32),
            cache_file=tmp_path / "gradients.pkl",
            batch_size=0,
            device="cpu",
        )


def test_stable_concepts_uses_year_specific_gradient_cache(monkeypatch, tmp_path):
    gradient_calls = []

    monkeypatch.setattr(sae_compare, "GRADS_FILE", tmp_path / "grads.pkl")
    monkeypatch.setattr(
        sae_compare,
        "train_binary_trees",
        lambda *_args, **_kwargs: {10: []},
    )
    monkeypatch.setattr(
        sae_compare,
        "get_model_gradients",
        lambda **kwargs: gradient_calls.append(kwargs) or np.zeros((2, 3)),
    )
    monkeypatch.setattr(
        sae_compare,
        "train_cavs_from_rules",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        sae_compare,
        "get_tcav_scores",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        sae_compare,
        "get_significant_concepts",
        lambda **_kwargs: ([], pd.DataFrame()),
    )

    sae_compare.get_stable_concepts_year(
        year=2007,
        X_df_scaled=pd.DataFrame(np.zeros((2, 3))),
        X_np_scaled=np.zeros((2, 3)),
        tabpfn=object(),
        embs=np.zeros((2, 3)),
        codes=np.zeros((2, 3)),
        y_np=np.zeros(2),
        dist_np=np.zeros(2),
        embs_scaler=object(),
        feature_cols=["a", "b", "c"],
    )

    assert len(gradient_calls) == 1
    assert gradient_calls[0]["cache_file"] == tmp_path / "grads_2007.pkl"
    assert "year" not in gradient_calls[0]


def test_plot_run_results_creates_output_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    full_results = {
        (1.5, 3, 0.1): {
            "relevant_pairs_fraction": 0.5,
            "mean_cos_sim": 0.8,
            "mean_overlap": 0.4,
        }
    }
    model_results = {
        (1.5, 0.1): {
            "mean_mse": 0.01,
            "mean_sparsity": 0.1,
            "mean_dead_neurons": 2,
        }
    }

    sae_compare.plot_run_results(
        full_results=full_results,
        model_results=model_results,
        hyper_param=0.1,
        scaling_factors=[1.5],
        model_n=[3],
        model_type="ReLU",
    )

    assert (
        tmp_path
        / "stats"
        / "sae"
        / "ReLU"
        / "good_pairs_cos_sim[Alpha=0.1,scale=1.5].png"
    ).is_file()
    sae_compare.plt.close("all")


def test_train_binary_trees_restricts_factor_ids_and_preserves_ids():
    X = np.concatenate([np.zeros(20), np.ones(20)]).reshape(-1, 1)
    activations = np.zeros((40, 3), dtype=float)
    activations[20:, 1] = 1.0

    rules = decision_tree.train_binary_trees(
        activations,
        X,
        ["feature"],
        factor_ids=[1],
        min_positive_samples=5,
        max_depth=2,
    )

    assert all(
        rule["Factor"] == 1
        for percentile_rules in rules.values()
        for rule in percentile_rules
    )
    assert any(rules.values())


def test_train_binary_trees_honors_minimum_positive_samples():
    X = np.concatenate([np.zeros(36), np.ones(4)]).reshape(-1, 1)
    activations = np.zeros((40, 2), dtype=float)
    activations[-4:, 1] = 1.0

    rules = decision_tree.train_binary_trees(
        activations,
        X,
        ["feature"],
        factor_ids=[1],
        min_positive_samples=5,
        max_depth=2,
    )

    assert all(not percentile_rules for percentile_rules in rules.values())
