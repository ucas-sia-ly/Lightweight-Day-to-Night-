from a2b_lightvpr.config import DataConfig, ExperimentConfig
from a2b_lightvpr.tracking import TrainingTracker


def test_disabled_tracker_is_noop(tmp_path):
    config = ExperimentConfig(DataConfig("robot.csv", "protocol.json", "external.csv"))
    tracker = TrainingTracker(
        config, job_type="test", default_name="disabled", output_dir=tmp_path
    )
    tracker.log({"loss": 1.0}, step=1)
    tracker.summary({"done": True})
    tracker.finish()
    assert tracker.enabled is False
