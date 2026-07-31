from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from ..checkpointing import file_sha256, write_json
from ..config import ExperimentConfig
from ..data import read_generated_manifest
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

SUITE_DATABASE_FIELDS = (
    "dataset",
    "source_path",
    "generated_path",
    "generator_sha256",
)


def _suite_database_name(dataset: str) -> str:
    return "tokyo" if dataset in {"tokyo_night", "tokyo_sunset"} else dataset


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


def translate_training_set(
    config: ExperimentConfig,
    checkpoint: str | Path | None = None,
    *,
    limit: int | None = None,
) -> Path:
    checkpoint = Path(checkpoint or Path(config.generator.output_dir) / "generator_final.pt")
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
        if existing != expected_metadata:
            raise ValueError(
                f"Translation output fingerprint mismatch at {metadata_path}; "
                "use a new generated_root"
            )
    elif image_root.exists() and any(image_root.rglob("*.png")):
        raise ValueError(
            f"Translation images exist without fingerprint metadata at {image_root}"
        )
    image_root.mkdir(parents=True, exist_ok=True)
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
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GENERATED_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(manifest)
    return manifest


def translate_suite_databases(
    config: ExperimentConfig,
    datasets: tuple[str, ...],
    checkpoint: str | Path | None = None,
    *,
    limit: int | None = None,
) -> list[Path]:
    """Translate fixed daytime databases once for night-to-night retrieval."""
    from ..reference_models import suite_rows

    checkpoint = Path(checkpoint or Path(config.generator.output_dir) / "generator_final.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_translation_checkpoint(state, config, limit)
    generator = ResnetGenerator()
    generator.load_state_dict(state["generator"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.eval().to(device)
    checkpoint_hash = file_sha256(checkpoint)
    manifests = []
    completed = set()
    for dataset in datasets:
        database_name = _suite_database_name(dataset)
        if database_name in completed:
            continue
        split = "val" if dataset == "msls_night" else "test"
        database, _ = suite_rows(config, dataset, split)
        if limit is not None:
            database = database[: int(limit)]
        output_root = Path(config.data.generated_database_root) / database_name
        if limit is not None:
            output_root = output_root / "_smoke"
        image_root = output_root / "images"
        metadata_path = output_root / "translation_metadata.json"
        expected_metadata = {
            "database": database_name,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_step": int(state.get("step", 0)),
            "source_count": len(database),
            "output_size": [config.data.output_height, config.data.output_width],
            "smoke_limit": int(limit) if limit is not None else None,
        }
        if metadata_path.is_file():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing != expected_metadata:
                raise ValueError(
                    f"Suite translation fingerprint mismatch at {metadata_path}; use a new root"
                )
        elif image_root.exists() and any(image_root.glob("*.png")):
            raise ValueError(f"Suite images exist without metadata at {image_root}")
        image_root.mkdir(parents=True, exist_ok=True)
        write_json(expected_metadata, metadata_path)
        output_rows = []
        for row in tqdm(database, desc=f"translate {dataset} database"):
            source_path = str(Path(row.source_path).resolve())
            image_id = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:20]
            output_path = image_root / f"{image_id}.png"
            if not output_path.is_file():
                with Image.open(source_path) as image:
                    tensor = generator_inference_tensor(
                        image, config.data.output_height, config.data.output_width
                    ).unsqueeze(0)
                with torch.inference_mode(), torch.autocast(
                    device.type, enabled=device.type == "cuda"
                ):
                    generated = generator(tensor.to(device))[0].cpu()
                TF.to_pil_image(denormalize_generator(generated)).save(
                    output_path, compress_level=3
                )
            output_rows.append({
                "dataset": dataset,
                "source_path": source_path,
                "generated_path": str(output_path.resolve()),
                "generator_sha256": checkpoint_hash,
            })
        manifest = output_root / "manifest.csv"
        temporary = manifest.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUITE_DATABASE_FIELDS)
            writer.writeheader()
            writer.writerows(output_rows)
        temporary.replace(manifest)
        manifests.append(manifest)
        completed.add(database_name)
    return manifests


def load_translated_database(
    config: ExperimentConfig, dataset: str, database: list
) -> tuple[list, str]:
    manifest = (
        Path(config.data.generated_database_root)
        / _suite_database_name(dataset)
        / "manifest.csv"
    )
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Missing generated-night database for {dataset}: {manifest}. "
            "Run translate-suite-database first."
        )
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {
        str(Path(row["source_path"]).resolve()): row["generated_path"] for row in rows
    }
    expected = [str(Path(row.source_path).resolve()) for row in database]
    missing = [path for path in expected if path not in mapping]
    extras = sorted(set(mapping) - set(expected))
    if missing or extras:
        raise ValueError(
            f"Generated database does not match {dataset}: "
            f"missing={len(missing)}, extras={len(extras)}"
        )
    translated = [
        replace(row, source_path=str(Path(mapping[path]).resolve()))
        for row, path in zip(database, expected)
    ]
    return translated, file_sha256(manifest)
