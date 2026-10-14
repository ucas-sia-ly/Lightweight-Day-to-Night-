from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from ..checkpointing import file_sha256, write_json
from ..config import ExperimentConfig
from ..data import read_generated_manifest
from ..paths import resolve_resource_path, resource_paths_equal
from ..transforms import denormalize_generator, generator_inference_tensor
from .networks import ResnetGenerator


GENERATED_FIELDS = (
    "image_id",
    "source_id",
    "source_path",
    "generated_path",
    "split",
    "domain",
    "role",
    "traverse",
    "utm_e",
    "utm_n",
    "yaw",
    "place_id",
    "generator_sha256",
    "dataset",
)


def validate_translation_checkpoint(state: dict, config: ExperimentConfig, limit: int | None) -> None:
    if state.get("kind") != "generator_inference":
        raise ValueError("translate-train requires an inference-only generator checkpoint")
    if state.get("objective") != config.generator.objective:
        raise ValueError("Generator checkpoint objective does not match the translation config")
    if limit is None and int(state.get("step", -1)) != int(config.generator.max_steps):
        raise ValueError(
            "Full translation requires the final configured generator step; "
            "use --limit only for an isolated smoke test"
        )


def translation_metadata_matches(existing: dict, expected: dict) -> bool:
    if existing.keys() != expected.keys():
        return False
    return all(
        resource_paths_equal(existing[key], value)
        if key in {"checkpoint", "source_manifest"} else existing[key] == value
        for key, value in expected.items()
    )


def translate_training_set(
    config: ExperimentConfig,
    checkpoint: str | Path | None = None,
    *,
    limit: int | None = None,
) -> Path:
    checkpoint = resolve_resource_path(checkpoint or Path(config.generator.output_dir) / "generator_final.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_translation_checkpoint(state, config, limit)
    generator = ResnetGenerator()
    generator.load_state_dict(state["generator"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.eval().to(device)
    checkpoint_hash = file_sha256(checkpoint)

    output_root = Path(config.data.generated_root)
    if limit is not None:
        output_root = output_root / "_smoke"
    image_root = output_root / "images"
    metadata_path = output_root / "translation_metadata.json"
    rows = [
        row for row in read_generated_manifest(config.data.vpr_source_manifest)
        if row["split"] == "train"
    ]
    if limit is not None:
        rows = rows[: int(limit)]
    expected_metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": int(state.get("step", 0)),
        "checkpoint_config_fingerprint": state.get("config_fingerprint", ""),
        "source_manifest": str(Path(config.data.vpr_source_manifest).resolve()),
        "source_manifest_sha256": file_sha256(config.data.vpr_source_manifest),
        "output_size": [config.data.output_height, config.data.output_width],
        "datasets": sorted({row["dataset"] for row in rows}),
        "smoke_limit": int(limit) if limit is not None else None,
    }
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not translation_metadata_matches(existing, expected_metadata):
            raise ValueError(
                f"Translation output fingerprint mismatch at {metadata_path}; "
                "use a new generated_root"
            )
    elif image_root.exists() and any(image_root.rglob("*.png")):
        raise ValueError(
            f"Translation images exist without fingerprint metadata at {image_root}"
        )
    image_root.mkdir(parents=True, exist_ok=True)
    if not metadata_path.is_file():
        write_json(expected_metadata, metadata_path)
    output_rows = []
    for row in tqdm(rows, desc="translate multi-dataset VPR train"):
        output_path = image_root / row["dataset"] / f'{row["image_id"]}.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_path.is_file():
            with Image.open(row["source_path"]) as image:
                tensor = generator_inference_tensor(
                    image, config.data.output_height, config.data.output_width
                ).unsqueeze(0)
            with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
                generated = generator(tensor.to(device))[0].cpu()
            TF.to_pil_image(denormalize_generator(generated)).save(output_path, compress_level=3)
        output_rows.append(
            {
                "image_id": row["image_id"],
                "source_id": row["image_id"],
                "source_path": row["source_path"],
                "generated_path": str(output_path.resolve()),
                "split": "train",
                "domain": "fake_night",
                "role": row.get("role", "database"),
                "traverse": row.get("route", ""),
                "utm_e": row.get("utm_e", ""),
                "utm_n": row.get("utm_n", ""),
                "yaw": row.get("yaw", ""),
                "place_id": row["place_id"],
                "generator_sha256": checkpoint_hash,
                "dataset": row["dataset"],
            }
        )
    manifest = output_root / "manifest.csv"
    if manifest.is_file() and read_generated_manifest(manifest) == output_rows:
        # Relocation alone must not rewrite the manifest hash used by teacher
        # caches and historical checkpoints.
        return manifest
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GENERATED_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(manifest)
    return manifest
