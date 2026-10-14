"""Explicit aliases for resources moved from the original research workspace.

Do not rewrite historical manifests/checkpoints or search by basename. An alias
is used only when the original path is absent and its exact replacement exists.
Missing dates and weights therefore remain missing.
"""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATASETS_ROOT = WORKSPACE_ROOT / "datasets"
TORCH_HUB_ROOT = Path.home() / ".cache/torch/hub"
LEGACY_ROOT = Path("/home/admin123/diffusion_vpr_test")
LEGACY_DATASETS_ROOT = LEGACY_ROOT / "CycleGAN-turbo_NetVLAD/dataset"


def _relocated_candidates(path: Path):
    datasets = Path(os.environ.get("A2B_DATASETS_ROOT", str(DATASETS_ROOT))).expanduser()
    try:
        relative = path.relative_to(LEGACY_DATASETS_ROOT)
    except ValueError:
        pass
    else:
        if relative.parts and relative.parts[0] == "gsv-cities":
            yield WORKSPACE_ROOT / "WORKSPACE/dataset" / relative
        else:
            if relative.parts and relative.parts[0] == "RobotCar_RGB":
                relative = Path("RobotCar", *relative.parts[1:])
            yield datasets / relative
    for old_project in ("A2B-LightVPR", "Lightweight-Day-to-Night"):
        try:
            yield PROJECT_ROOT / path.relative_to(LEGACY_ROOT / old_project)
        except ValueError:
            pass
    try:
        relative = path.relative_to(LEGACY_ROOT / "reference_code")
    except ValueError:
        pass
    else:
        if path.name in {
            "dinov2_vits14_reg4_pretrain.pth", "dinov2_vits14_pretrain.pth",
            "dino_salad_512_32.ckpt", "dinov2_12288.pth",
            "ResNet50_512_eigenplaces.pth", "ResNet50_2048_eigenplaces.pth",
        }:
            yield TORCH_HUB_ROOT / "checkpoints" / path.name
        repositories = {
            "dinov2": "facebookresearch_dinov2_main",
            "salad": "serizba_salad_main",
            "CosPlace": "gmberton_cosplace_main",
        }
        if relative.parts and relative.parts[0] in repositories:
            yield TORCH_HUB_ROOT / repositories[relative.parts[0]] / Path(*relative.parts[1:])
        yield PROJECT_ROOT / "third_party" / relative
        yield PROJECT_ROOT / "third_party/weights" / path.name
    try:
        relative = path.relative_to(LEGACY_ROOT / "CycleGAN-turbo_NetVLAD/checkpoints")
    except ValueError:
        pass
    else:
        yield PROJECT_ROOT / "third_party/weights" / relative


def resolve_resource_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.exists() or not path.is_absolute():
        return path
    for candidate in _relocated_candidates(path):
        if candidate.exists():
            return candidate.resolve()
    return path


def resolve_manifest_image_path(value: str, manifest: str | Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(resolve_resource_path(path))
    # Existing generated/source CSVs use paths relative to the invocation root.
    if path.exists():
        return str(path.resolve())
    manifest = Path(manifest).expanduser().resolve()
    candidate = manifest.parent / path
    if candidate.is_file():
        return str(candidate.resolve())
    return value


def resource_paths_equal(first: str | Path, second: str | Path) -> bool:
    return resolve_resource_path(first).resolve() == resolve_resource_path(second).resolve()
