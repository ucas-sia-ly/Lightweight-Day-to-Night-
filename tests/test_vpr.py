import pytest
import torch
import importlib.util
import hashlib
from pathlib import Path

from a2b_lightvpr.vpr.losses import (
    MarginCosineClassifier,
    day_fake_classification_loss,
)
from a2b_lightvpr.vpr.model import (
    NetVLADPool,
    MixVPR,
    OfficialMixVPRModel,
    build_descriptor_model,
    eigenplaces_checkpoint,
)


@pytest.mark.parametrize("aggregator", ["gem", "convap", "mixvpr"])
def test_resnet50_aggregators_honor_512_contract(aggregator):
    model = build_descriptor_model(
        aggregator, image_height=64, image_width=96, pretrained=False
    ).eval()
    with torch.no_grad():
        descriptor = model.encode(torch.randn(1, 3, 64, 96))
    assert descriptor.shape == (1, 512)
    assert torch.allclose(descriptor.norm(dim=1), torch.ones(1), atol=1e-5)


@pytest.mark.parametrize("descriptor_dim,channels,rows", [(128, 64, 2), (512, 256, 2)])
def test_mixvpr_matches_official_implementation(descriptor_dim, channels, rows):
    path = Path("/home/admin123/diffusion_vpr_test/reference_code/MixVPR/models/aggregators/mixvpr.py")
    spec = importlib.util.spec_from_file_location("official_mixvpr", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    local = MixVPR(32, (4, 6), descriptor_dim).eval()
    official = module.MixVPR(
        in_channels=32, in_h=4, in_w=6, out_channels=channels,
        mix_depth=4, mlp_ratio=1, out_rows=rows,
    ).eval()
    official.load_state_dict(local.state_dict(), strict=True)
    value = torch.randn(2, 32, 4, 6)
    assert torch.allclose(
        torch.nn.functional.normalize(local(value), dim=1), official(value), atol=1e-6
    )

@pytest.mark.parametrize("descriptor_dim", [128, 512, 4096])
def test_mixvpr_descriptor_dimensions(descriptor_dim):
    model = MixVPR(32, (4, 6), descriptor_dim).eval()
    with torch.no_grad():
        value = model(torch.randn(2, 32, 4, 6))
    assert value.shape == (2, descriptor_dim)


@pytest.mark.parametrize(
    "descriptor_dim,filename,expected_sha256",
    [
        (128, "resnet50_MixVPR_128_channels(64)_rows(2).ckpt", "d9dc62449ebfbfca0ac138e3062469458f957c5dd8d61157fe07007d1da54b6f"),
        (512, "resnet50_MixVPR_512_channels(256)_rows(2).ckpt", "7e1180f77d04ece604f2d12f8b6b9b323a99d6e43863c879028bfa17b34807cf"),
        (4096, "resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt", "97528606773e9920e93ca4d211daeabf1c7312480f38b45379ee5551a1b4dd24"),
    ],
)
def test_mixvpr_initialization_checkpoints_load_strictly(
    descriptor_dim, filename, expected_sha256
):
    checkpoint = Path(
        "/home/admin123/diffusion_vpr_test/CycleGAN-turbo_NetVLAD/checkpoints/"
        + filename
    )
    if not checkpoint.is_file():
        pytest.skip("Official MixVPR initialization checkpoint unavailable")
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected_sha256
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    OfficialMixVPRModel(descriptor_dim).load_state_dict(state, strict=True)


def test_netvlad_pool_output_contract():
    pool = NetVLADPool(num_clusters=4, dim=8)
    output = pool(torch.randn(2, 8, 5, 7))
    assert output.shape == (2, 32)
    assert torch.allclose(output.norm(dim=1), torch.ones(2), atol=1e-5)


def test_eigenplaces_download_records_official_metadata(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    def fake_download(url, map_location, file_name):
        assert url.startswith(
            "https://github.com/gmberton/EigenPlaces/releases/download/v1.0/"
        )
        torch.save({"backbone.weight": torch.ones(1)}, checkpoint_dir / file_name)
        return {"backbone.weight": torch.ones(1)}

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fake_download)
    checkpoint = eigenplaces_checkpoint(512)
    metadata = __import__("json").loads(
        checkpoint.with_suffix(checkpoint.suffix + ".json").read_text()
    )
    assert metadata["url"].endswith("ResNet50_512_eigenplaces.pth")
    assert metadata["size_bytes"] == checkpoint.stat().st_size
    assert len(metadata["sha256"]) == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload bridge")
def test_classifier_cpu_offload_supports_amp_optimizer_step():
    feature = torch.nn.Linear(8, 16).cuda()
    classifier = MarginCosineClassifier(16, 32).cpu()
    optimizer = torch.optim.AdamW([
        {"params": feature.parameters()},
        {"params": classifier.parameters()},
    ])
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    labels = torch.arange(4, device="cuda")
    with torch.autocast("cuda"):
        descriptors = torch.nn.functional.normalize(
            feature(torch.randn(4, 8, device="cuda")), dim=1
        )
        loss = torch.nn.functional.cross_entropy(classifier(descriptors, labels), labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    classifier_state = optimizer.state[classifier.weight]
    assert classifier.weight.device.type == "cpu"
    assert classifier_state["exp_avg"].device.type == "cpu"
    assert feature.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_4096_classifier_amp_accumulation_resume_stays_on_gpu(tmp_path):
    classifier = MarginCosineClassifier(4096, 64).cuda()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)

    optimizer.zero_grad(set_to_none=True)
    for _ in range(4):
        descriptors = torch.nn.functional.normalize(
            torch.randn(2, 4096, device="cuda"), dim=1
        )
        labels = torch.arange(2, device="cuda")
        with torch.autocast("cuda"):
            loss = torch.nn.functional.cross_entropy(
                classifier(descriptors, labels), labels
            ) / 4
        scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    checkpoint = tmp_path / "classifier_4096.pt"
    torch.save(
        {
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        checkpoint,
    )
    resumed_classifier = MarginCosineClassifier(4096, 64).cuda()
    resumed_optimizer = torch.optim.AdamW(resumed_classifier.parameters(), lr=1e-4)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resumed_classifier.load_state_dict(state["classifier"])
    resumed_optimizer.load_state_dict(state["optimizer"])

    assert resumed_classifier.weight.device.type == "cuda"
    optimizer_state = resumed_optimizer.state[resumed_classifier.weight]
    assert optimizer_state["exp_avg"].device.type == "cuda"
    assert optimizer_state["exp_avg_sq"].device.type == "cuda"


def test_day_fake_classification_loss_is_finite():
    day = torch.nn.functional.normalize(torch.randn(4, 512), dim=1)
    fake = torch.nn.functional.normalize(torch.randn(4, 512), dim=1)
    labels = torch.tensor([0, 1, 2, 3])
    classifier = MarginCosineClassifier(512, 4)
    loss, components = day_fake_classification_loss(day, fake, labels, classifier)
    assert torch.isfinite(loss)
    assert set(components) == {"classification_day", "classification_fake"}
