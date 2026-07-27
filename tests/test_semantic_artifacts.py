import numpy as np

from semantic_artifacts import SemanticArtifactStore, derive_seed, stable_hash


def test_hash_and_seed_are_deterministic_and_order_stable():
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
    assert derive_seed(42, "run", 1) == derive_seed(42, "run", 1)
    assert derive_seed(42, "run", 1) != derive_seed(42, "run", 2)


def test_artifact_store_round_trip(tmp_path):
    store = SemanticArtifactStore(tmp_path, "experiment")
    store.write_json("manifest.json", {"array": np.array([1, 2])})
    store.write_npz("masks.npz", selected=np.array([True, False]))
    assert store.read_json("manifest.json") == {"array": [1, 2]}
    assert store.exists("masks.npz")
