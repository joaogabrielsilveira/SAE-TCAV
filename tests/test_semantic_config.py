import json

import pytest

from semantic_config import ClassAnalysisConfig, SemanticExperimentConfig, load_clinical_groups


def test_config_round_trip_and_unknown_field(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(SemanticExperimentConfig().to_dict()))
    loaded = SemanticExperimentConfig.from_json(path)
    assert loaded.activation_targets.positive_fractions == [0.1, 0.2, 0.3, 0.4, 0.5] or loaded.activation_targets.positive_fractions == (0.1, 0.2, 0.3, 0.4, 0.5)
    assert loaded.class_analysis == ClassAnalysisConfig(enabled=True)
    with pytest.raises(ValueError, match="Unknown"):
        SemanticExperimentConfig.from_dict({"typo": True})


def test_class_analysis_defaults_enabled_and_can_be_disabled():
    assert SemanticExperimentConfig.from_dict({}).class_analysis.enabled is True

    config = SemanticExperimentConfig.from_dict({"class_analysis": {"enabled": False}})

    assert config.class_analysis.enabled is False
    assert config.to_dict()["class_analysis"] == {"enabled": False}


def test_class_analysis_enabled_is_strictly_boolean():
    with pytest.raises(ValueError, match="must be a boolean"):
        SemanticExperimentConfig.from_dict({"class_analysis": {"enabled": 1}})

    with pytest.raises(ValueError, match="Unknown class_analysis fields"):
        SemanticExperimentConfig.from_dict({"class_analysis": {"unknown": True}})

    with pytest.raises(ValueError, match="must be a JSON object"):
        SemanticExperimentConfig.from_dict({"class_analysis": False})


def test_runtime_progress_defaults_enabled_and_is_strictly_boolean():
    assert SemanticExperimentConfig.from_dict({}).runtime.show_progress is True
    assert (
        SemanticExperimentConfig.from_dict(
            {"runtime": {"show_progress": False}}
        ).runtime.show_progress
        is False
    )

    with pytest.raises(ValueError, match="runtime.cache and runtime.show_progress"):
        SemanticExperimentConfig.from_dict({"runtime": {"show_progress": 1}})


def test_clinical_groups_are_external_and_many_to_many(tmp_path):
    path = tmp_path / "groups.json"
    path.write_text(json.dumps({"creatinine": ["renal", "laboratory"], "age": "demographic"}))
    assert load_clinical_groups(path) == {
        "age": ("demographic",),
        "creatinine": ("laboratory", "renal"),
    }
