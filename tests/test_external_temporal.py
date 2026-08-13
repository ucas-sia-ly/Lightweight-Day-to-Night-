import json

from a2b_lightvpr.cli import _parser
from a2b_lightvpr import external_temporal as external


def test_gardens_manifest_maps_each_sequence_to_0_through_199(tmp_path):
    for offset, suffix in ((0, "day_left"), (200, "day_right"), (400, "night_right")):
        for index in range(200):
            (tmp_path / f"{offset + index:03d}_{suffix}.jpg").touch()
    groups = external.gardens_rows(tmp_path)
    assert set(groups) == {"day_left", "day_right", "night_right"}
    assert all([row.sequence_index for row in rows] == list(range(200)) for rows in groups.values())


def test_gardens_reference_cli_uses_adapted_database_by_default():
    parsed = _parser().parse_args([
        "evaluate-gardens-reference", "--config", "c.yaml", "--registry", "r.yaml",
        "--model", "salad", "--root", "gardens", "--output-dir", "out",
    ])
    assert parsed.database_encoder == "adapted"


def test_gardens_summary_requires_v1_four_window_results(tmp_path):
    root = tmp_path / "models" / "model-a"
    root.mkdir(parents=True)
    recalls = {"1": .5, "5": .7, "10": .8}
    grid = {"frame_tolerance_0": {"recalls": recalls}}
    payload = {
        "schema": "gardens_reference_evaluation_v1", "temporal_method": "mgtr_v1",
        "model": "model-a", "descriptor_dim": 512, "database_encoder": "adapted",
        "night_right": {"single_frame": grid, "temporal": grid},
        "window_sweep": {
            str(window): {"night_right": grid} for window in (3, 4, 5, 6)
        },
    }
    (root / "evaluation.json").write_text(json.dumps(payload))
    output = tmp_path / "summary.json"
    external.summarize_gardens_models(tmp_path / "models", output)
    summary = json.loads(output.read_text())
    assert summary["temporal_method"] == "mgtr_v1"
    assert summary["primary_window_size"] == 4
    assert summary["models"][0]["temporal"] == recalls
    assert set(summary["models"][0]["window_sweep"]) == {"3", "4", "5", "6"}


def test_gardens_summary_accepts_ragtr_results(tmp_path):
    root = tmp_path / "models" / "ragtr"
    root.mkdir(parents=True)
    recalls = {"1": .6, "5": .8, "10": .9}
    grid = {"frame_tolerance_0": {"recalls": recalls}}
    payload = {
        "schema": "gardens_ragtr_evaluation_v1", "temporal_method": "ragtr",
        "model": "mixvpr-4096-cdnd", "descriptor_dim": 4096,
        "primary_window_size": 5,
        "night_right": {"single_frame": grid, "temporal": grid},
        "window_sweep": {
            str(window): {"night_right": grid} for window in (3, 4, 5, 6)
        },
    }
    (root / "evaluation.json").write_text(json.dumps(payload))
    output = tmp_path / "summary.json"
    external.summarize_gardens_models(tmp_path / "models", output)
    summary = json.loads(output.read_text())
    assert summary["temporal_method"] == "ragtr"
    assert summary["models"][0]["primary_window_size"] == 5


def test_gardens_uses_only_v1_parameters():
    parameters = external.TemporalParameters(
        window_size=4, top_k=10, transition_radius_m=5,
        motion_weight=.5, motion_sigma_m=1, nms_radius_m=0,
    )
    assert set(external.asdict(parameters)) == {
        "window_size", "top_k", "transition_radius_m", "motion_weight",
        "raw_top_k", "nms_radius_m", "nms_yaw_deg", "motion_sigma_m",
    }


def test_gardens_cli_exposes_ragtr_without_changing_v1_default():
    default = _parser().parse_args([
        "evaluate-gardens-temporal", "--config", "c.yaml", "--checkpoint", "m.pt",
        "--root", "gardens", "--output-dir", "out",
    ])
    ragtr = _parser().parse_args([
        "evaluate-gardens-temporal", "--config", "c.yaml", "--checkpoint", "m.pt",
        "--root", "gardens", "--output-dir", "out", "--method", "ragtr",
    ])
    assert default.method == "mgtr-v1"
    assert ragtr.method == "ragtr"
