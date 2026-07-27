import json

import pytest

from semantic_config import SemanticExperimentConfig, load_clinical_groups


def test_config_round_trip_and_unknown_field(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(SemanticExperimentConfig().to_dict()))
    loaded = SemanticExperimentConfig.from_json(path)
    assert loaded.activation_targets.positive_fractions == [0.10, 0.25, 0.50] or loaded.activation_targets.positive_fractions == (0.10, 0.25, 0.50)
    with pytest.raises(ValueError, match="Unknown"):
        SemanticExperimentConfig.from_dict({"typo": True})


def test_clinical_groups_are_external_and_many_to_many(tmp_path):
    path = tmp_path / "groups.json"
    path.write_text(json.dumps({"creatinine": ["renal", "laboratory"], "age": "demographic"}))
    assert load_clinical_groups(path) == {
        "age": ("demographic",),
        "creatinine": ("laboratory", "renal"),
    }
