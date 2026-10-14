import csv
from dataclasses import replace

import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from a2b_lightvpr.checkpointing import data_fingerprints
from a2b_lightvpr.config import DataConfig, ExperimentConfig
from a2b_lightvpr.transforms import VPRTransform
from a2b_lightvpr.vpr.trainer import _evaluate_validation


class ColorDescriptor(nn.Module):
    descriptor_dim = 3

    def encode(self, images):
        return F.normalize(images.mean(dim=(2, 3)), dim=1)


def public_validation_fixture(tmp_path):
    root = tmp_path / "MSLS"
    for city, x, color in (("cph", 0, (255, 0, 0)), ("sf", 100, (0, 255, 0))):
        for role in ("database", "query"):
            folder = root / "train_val" / city / role
            (folder / "images").mkdir(parents=True)
            Image.new("RGB", (8, 8), color).save(folder / "images/day.jpg")
            with (folder / "postprocessed.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["key", "easting", "northing", "night"])
                writer.writeheader()
                writer.writerows([
                    {"key": "day", "easting": x, "northing": 0, "night": "False"},
                    # Its image is deliberately absent: validation must not use
                    # public night evaluation queries to select checkpoints.
                    {"key": "night", "easting": x, "northing": 0, "night": "True"},
                ])
    sources = tmp_path / "source.csv"
    sources.write_text("unchanged training source identities")
    config = ExperimentConfig(DataConfig(msls_root=str(root), vpr_source_manifest=str(sources)))
    return replace(config, vpr=replace(config.vpr, workers=0))


def test_training_validation_uses_public_day_split_and_ignores_fake_test_database(tmp_path):
    config = public_validation_fixture(tmp_path)
    config = replace(config, evaluation=replace(
        config.evaluation, database_domain="fake_night", fake_database_manifest="missing-test-fakes.csv"))
    metrics = _evaluate_validation(ColorDescriptor(), config, VPRTransform(8, 8), torch.device("cpu"), None)
    assert metrics["recalls"]["1"] == 1.0
    assert metrics["valid_queries"] == metrics["total_queries"] == 2
    assert metrics["database_domain"] == "day"
    assert metrics["positive_radius_m"] == 25.0 and metrics["positive_yaw_deg"] is None
    assert set(metrics["datasets"]) == {"msls_day_val"}


def test_validation_metadata_changes_checkpoint_data_identity(tmp_path):
    config = public_validation_fixture(tmp_path)
    before = data_fingerprints(config)
    assert set(before) == {"vpr_source_manifest", "validation_msls_cph_database",
                           "validation_msls_cph_query", "validation_msls_sf_database", "validation_msls_sf_query"}
    table = tmp_path / "MSLS/train_val/cph/query/postprocessed.csv"
    table.write_text(table.read_text().replace("day,0,0,False", "day,10,0,False"))
    after = data_fingerprints(config)
    assert before["validation_msls_cph_query"] != after["validation_msls_cph_query"]
    assert before["vpr_source_manifest"] == after["vpr_source_manifest"]


def test_public_test_queries_cannot_enter_training_validation(tmp_path):
    config = public_validation_fixture(tmp_path)
    config = replace(config, evaluation=replace(config.evaluation, datasets=("robotcar_seasons_night",)))
    with pytest.raises(ValueError, match="public day validation"):
        _evaluate_validation(ColorDescriptor(), config, VPRTransform(8, 8), torch.device("cpu"), None)
