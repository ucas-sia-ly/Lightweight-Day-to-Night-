import csv
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from a2b_lightvpr.references.models import audit_reference_models
from a2b_lightvpr.references.training import gsv_rows, select_gsv_images
from a2b_lightvpr.vpr.anyloc import AnyLocSmallModel
from a2b_lightvpr.vpr.boq import BoQSmallModel, PCAProjectedReference


class FakeDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 384
        self.blocks = nn.ModuleList(nn.Linear(1, 1) for _ in range(12))
        self.norm = nn.LayerNorm(384)
        self.marker = nn.Parameter(torch.ones(1))

    def forward_features(self, images):
        batch, _, height, width = images.shape
        patches = (height // 14) * (width // 14)
        # This contract excludes CLS and register tokens, exactly as DINOv2 does.
        values = torch.arange(
            batch * patches * self.embed_dim, dtype=images.dtype, device=images.device
        ).reshape(batch, patches, self.embed_dim)
        return {
            "x_norm_patchtokens": torch.nn.functional.normalize(values + 1, dim=-1),
            "x_norm_clstoken": torch.ones(batch, self.embed_dim, device=images.device),
        }

    def get_intermediate_layers(self, images, n, norm=False):
        return (self.forward_features(images)["x_norm_patchtokens"],)


def _fake_dino_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "dino.pt"
    torch.save(FakeDINO().state_dict(), checkpoint)
    return checkpoint


def test_boq_small_excludes_special_tokens_and_outputs_12288(tmp_path, monkeypatch):
    checkpoint = _fake_dino_checkpoint(tmp_path)
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: FakeDINO())
    (tmp_path / "hubconf.py").write_text("", encoding="utf-8")
    model = BoQSmallModel(str(tmp_path), "dinov2_vits14_reg", str(checkpoint), 2)
    output = model.encode(torch.randn(1, 3, 28, 28))
    assert output.shape == (1, 12_288)
    assert torch.allclose(output.norm(dim=1), torch.ones(1), atol=1e-5)


def test_anyloc_small_requires_32_by_384_vocabulary_and_normalizes(tmp_path, monkeypatch):
    checkpoint = _fake_dino_checkpoint(tmp_path)
    vocabulary = tmp_path / "vocabulary.pt"
    torch.save({
        "centers": torch.randn(32, 384),
        "metadata": {
            "layer": 11, "facet": "token", "num_clusters": 32,
            "patch_dim": 384, "descriptor_dim": 12_288,
            "official_checkpoint": False,
        },
    }, vocabulary)
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: FakeDINO())
    (tmp_path / "hubconf.py").write_text("", encoding="utf-8")
    model = AnyLocSmallModel(
        str(tmp_path), "dinov2_vits14_reg", str(checkpoint), vocabulary
    )
    patches = model.patch_descriptors(torch.randn(2, 3, 28, 28))
    output = model.encode_patches(patches)
    assert patches.shape == (2, 4, 384)
    assert output.shape == (2, 12_288)
    assert torch.allclose(output.norm(dim=1), torch.ones(2), atol=1e-5)


def test_pca_wrapper_accepts_any_12288d_small_reference(tmp_path):
    class FakeSmall(nn.Module):
        descriptor_dim = 12_288

        def encode(self, images):
            return torch.nn.functional.normalize(images, dim=1)

    projection = tmp_path / "pca.pt"
    components = torch.zeros(12_288, 512)
    components[:512] = torch.eye(512)
    torch.save({
        "kind": "reference_pca",
        "input_dim": 12_288,
        "output_dim": 512,
        "mean": torch.zeros(12_288),
        "components": components,
        "metadata": {"whiten": False},
    }, projection)
    model = PCAProjectedReference(FakeSmall(), projection, 512)
    output = model.encode(torch.randn(2, 12_288))
    assert output.shape == (2, 512)
    assert torch.allclose(output.norm(dim=1), torch.ones(2), atol=1e-5)


def test_gsv_selection_is_balanced_and_reproducible(tmp_path):
    (tmp_path / "Dataframes").mkdir()
    (tmp_path / "Images").mkdir()
    header = ["place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid"]
    for city in ("A", "B"):
        (tmp_path / "Images" / city).mkdir()
        with (tmp_path / "Dataframes" / f"{city}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            for place in range(2):
                for view in range(4):
                    row = {
                        "place_id": place, "year": 2020, "month": 1,
                        "northdeg": view, "city_id": city, "lat": 1.0,
                        "lon": 2.0, "panoid": f"{place}-{view}",
                    }
                    writer.writerow(row)
                    name = (
                        f"{city}_{str(place).zfill(7)}_2020_01_{str(view).zfill(3)}_"
                        f"1.0_2.0_{place}-{view}.jpg"
                    )
                    Image.new("RGB", (8, 8)).save(tmp_path / "Images" / city / name)
    assert len(gsv_rows(tmp_path)) == 4
    first = select_gsv_images(tmp_path, 8, 7)
    second = select_gsv_images(tmp_path, 8, 7)
    assert first == second
    assert {row["dataset"] for row in first} == {"A", "B"}
    assert len({row["sample_index"] for row in first}) == 8


def test_small_registry_reports_training_provenance(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    registry = tmp_path / "models.yaml"
    registry.write_text(
        "reference_models:\n"
        "  small:\n"
        "    loader: fixture\n"
        "    backbone: DINOv2-S/14-Reg4\n"
        "    descriptor_dim: 12288\n"
        f"    checkpoint: {checkpoint}\n"
        "    official_checkpoint: false\n"
        "    training_status: re-trained\n",
        encoding="utf-8",
    )
    result = audit_reference_models(registry)["models"]["small"]
    assert result["ready"]
    assert result["backbone"] == "DINOv2-S/14-Reg4"
    assert result["official_checkpoint"] is False
    assert result["training_status"] == "re-trained"
