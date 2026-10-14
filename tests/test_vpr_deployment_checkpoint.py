import torch
import pytest

from a2b_lightvpr.config import DataConfig, ExperimentConfig
from a2b_lightvpr.runtime import load_vpr_model
from a2b_lightvpr.vpr.trainer import export_vpr_inference_checkpoint


def test_training_checkpoint_conversion_is_model_exact_and_strips_training_assets(tmp_path):
    config = ExperimentConfig(DataConfig(generator_manifest="external.csv"))
    model_state = {"encoder.weight": torch.randn(3, 4)}
    training = tmp_path / "training.pt"
    torch.save(
        {
            "kind": "vpr_training_state",
            "epoch": 4,
            "best_val_recall1": 0.8,
            "epoch_zero_recall1": 0.5,
            "config_fingerprint": config.fingerprint(),
            "data_fingerprints": {"manifest": "hash"},
            "model": model_state,
            "classifier": {"weight": torch.randn(2, 3)},
            "optimizer": {"state": {}},
            "relation_queue_state": {"entries": "large"},
            "sampler_state": {"cursor": 7},
            "teacher_cache_fingerprint": "teacher",
        },
        training,
    )
    inference = tmp_path / "inference.pt"
    export_vpr_inference_checkpoint(config, training, inference)
    payload = torch.load(inference, map_location="cpu", weights_only=True)
    assert payload["kind"] == "vpr_inference"
    assert payload["descriptor_dim"] == 512
    assert payload["random_seed"] == 0
    assert torch.equal(payload["model"]["encoder.weight"], model_state["encoder.weight"])
    forbidden = {
        "classifier", "optimizer", "relation_queue_state", "sampler_state",
        "teacher_cache_fingerprint",
    }
    assert forbidden.isdisjoint(payload)


def test_deployment_loader_rejects_checkpoint_dimension_mismatch(tmp_path):
    config = ExperimentConfig(DataConfig(generator_manifest="external.csv"))
    checkpoint = tmp_path / "wrong-dimension.pt"
    torch.save({"kind": "vpr_inference", "descriptor_dim": 128, "model": {}}, checkpoint)
    with pytest.raises(ValueError, match="descriptor dimension"):
        load_vpr_model(config, checkpoint)


def test_deployment_loader_rejects_checkpoint_seed_mismatch(tmp_path):
    config = ExperimentConfig(DataConfig(generator_manifest="external.csv"))
    checkpoint = tmp_path / "wrong-seed.pt"
    torch.save({
        "kind": "vpr_inference",
        "descriptor_dim": 512,
        "random_seed": 3,
        "model": {},
    }, checkpoint)
    with pytest.raises(ValueError, match="random seed"):
        load_vpr_model(config, checkpoint)
