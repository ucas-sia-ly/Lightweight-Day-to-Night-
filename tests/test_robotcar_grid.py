from dataclasses import replace
import json

import numpy as np
import torch

from a2b_lightvpr.config import DataConfig, ExperimentConfig
from a2b_lightvpr.reference_models import SuiteRow
from a2b_lightvpr import robotcar_grid


def test_holm_adjustment_is_step_down_monotonic():
    adjusted = robotcar_grid.holm_adjust({"a": 0.001, "b": 0.02, "c": 0.04})
    assert adjusted == {"a": 0.003, "b": 0.04, "c": 0.04}


def test_robotcar_grid_uses_circular_yaw_and_none_disables_filter():
    database = [SuiteRow("db.jpg", 0.0, 0.0, 359.0)]
    query = [SuiteRow("query.jpg", 0.0, 0.0, 1.0)]
    assert robotcar_grid._positives(database, query, 5.0, 10.0)[0].tolist() == [0]
    opposite = [SuiteRow("query.jpg", 0.0, 0.0, 180.0)]
    assert robotcar_grid._positives(database, opposite, 5.0, 10.0)[0].tolist() == []
    assert robotcar_grid._positives(database, opposite, 5.0, None)[0].tolist() == [0]


def test_robotcar_grid_extracts_descriptors_once_per_condition(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    config = ExperimentConfig(DataConfig("robot.csv", "protocol.json"))
    config = replace(config, vpr=replace(config.vpr, output_dir=str(tmp_path), workers=0))

    class Model(torch.nn.Module):
        descriptor_dim = 2

    rows = [
        SuiteRow("a.jpg", 0.0, 0.0, 359.0),
        SuiteRow("b.jpg", 20.0, 0.0, 180.0),
    ]
    calls = []
    monkeypatch.setattr(robotcar_grid, "load_vpr_model", lambda *args: (Model(), {"epoch": 2}, checkpoint))
    monkeypatch.setattr(robotcar_grid, "suite_rows", lambda *args: (rows, rows))
    def extract(model, selected, *args):
        calls.append(tuple(row.source_path for row in selected))
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(robotcar_grid, "extract_descriptors", extract)

    output = robotcar_grid.evaluate_robotcar_grid(
        config, checkpoint, conditions=("night",), distances=(5.0, 10.0, 25.0),
        yaw_degrees=("10", "30", "none"),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(calls) == 2
    assert len(payload["conditions"]["night"]["cells"]) == 9
    cell = payload["conditions"]["night"]["cells"]["distance_5m_yaw_10deg"]
    assert cell["positive_count_by_query"] == [1, 1]
    assert cell["excluded_queries"] == 0
