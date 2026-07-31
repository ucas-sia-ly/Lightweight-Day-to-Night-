from dataclasses import replace

import pytest

from a2b_lightvpr.config import DataConfig, ExperimentConfig
from a2b_lightvpr.generator.translate import (
    load_translated_database,
    validate_translation_checkpoint,
)


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


def test_smoke_translation_allows_nonsemantic_config_changes_but_not_wrong_objective():
    config = ExperimentConfig(DataConfig("robot.csv", "protocol.json"))
    state = {
        "kind": "generator_inference",
        "step": 1,
        "objective": config.generator.objective,
        "config_fingerprint": config.fingerprint(),
    }
    validate_translation_checkpoint(state, config, 1)
    changed = replace(config, generator=replace(config.generator, seed=9))
    validate_translation_checkpoint(state, changed, 1)
    with pytest.raises(ValueError, match="objective"):
        validate_translation_checkpoint({**state, "objective": "qsattn"}, config, 1)


def test_generated_database_requires_exact_source_set(tmp_path):
    from a2b_lightvpr.data import RobotCarRow

    root = tmp_path / "databases"
    dataset_root = root / "robotcar"
    dataset_root.mkdir(parents=True)
    source = tmp_path / "day.jpg"
    generated = tmp_path / "night.png"
    source.write_bytes(b"day")
    generated.write_bytes(b"night")
    (dataset_root / "manifest.csv").write_text(
        "dataset,source_path,generated_path,generator_sha256\n"
        f"robotcar,{source},{generated},fixture\n",
        encoding="utf-8",
    )
    config = ExperimentConfig(
        DataConfig(
            "robot.csv", "protocol.json", generated_database_root=str(root)
        )
    )
    row = RobotCarRow(
        "test", "database", "day", "route", "1", 0.0, 0.0, 0.0,
        str(source), "",
    )
    translated, digest = load_translated_database(config, "robotcar", [row])
    assert translated[0].source_path == str(generated.resolve())
    assert len(digest) == 64

    other = replace(row, source_path=str(tmp_path / "other.jpg"))
    with pytest.raises(ValueError, match="missing=1"):
        load_translated_database(config, "robotcar", [other])
