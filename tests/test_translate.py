import pytest

from a2b_lightvpr.config import DataConfig, ExperimentConfig
from a2b_lightvpr.generator.translate import validate_translation_checkpoint


def test_full_translation_requires_matching_final_checkpoint():
    config = ExperimentConfig(DataConfig("robot.csv", "protocol.json"))
    state = {
        "kind": "generator_inference",
        "step": config.generator.max_steps,
        "objective": config.generator.objective,
        "config_fingerprint": config.fingerprint(),
    }
    validate_translation_checkpoint(state, config, None)
    with pytest.raises(ValueError, match="final configured generator step"):
        validate_translation_checkpoint({**state, "step": 1}, config, None)


def test_smoke_translation_rejects_wrong_objective():
    config = ExperimentConfig(DataConfig("robot.csv", "protocol.json"))
    state = {
        "kind": "generator_inference",
        "step": 1,
        "objective": config.generator.objective,
        "config_fingerprint": config.fingerprint(),
    }
    validate_translation_checkpoint(state, config, 1)
    with pytest.raises(ValueError, match="objective"):
        validate_translation_checkpoint({**state, "objective": "unsupported"}, config, 1)
