from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm.auto import tqdm

from ..checkpointing import (
    atomic_torch_save,
    file_sha256,
    restore_rng_state,
    rng_state,
    write_json,
)
from ..transforms import IMAGENET_MEAN, IMAGENET_STD
from ..vpr.anyloc import AnyLocSmallModel
from ..vpr.boq import BoQReferenceModel, BoQSmallModel


def _gsv_image_name(city: str, place_id: int, row: dict[str, str]) -> str:
    return (
        f"{city}_{str(place_id).zfill(7)}_{str(row['year']).zfill(4)}_"
        f"{str(row['month']).zfill(2)}_{str(row['northdeg']).zfill(3)}_"
        f"{row['lat']}_{row['lon']}_{row['panoid']}.jpg"
    )


@lru_cache(maxsize=4)
def _gsv_rows_cached(root_value: str) -> tuple[dict[str, object], ...]:
    root = Path(root_value)
    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for table in sorted((root / "Dataframes").glob("*.csv")):
        city = table.stem
        with table.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                place_id = int(row["place_id"])
                image = root / "Images" / city / _gsv_image_name(city, place_id, row)
                groups[(city, place_id)].append(str(image))
    rows = []
    for (city, place_id), paths in sorted(groups.items()):
        if len(paths) >= 4:
            rows.append({"city": city, "place_id": place_id, "paths": sorted(paths)})
    if not rows:
        raise ValueError(f"No four-view GSV-Cities places found under {root}")
    return tuple(rows)


def gsv_rows(root: str | Path) -> list[dict[str, object]]:
    root_value = str(Path(root).expanduser().resolve())
    return list(_gsv_rows_cached(root_value))


def gsv_manifest_fingerprint(root: str | Path) -> str:
    root = Path(root).expanduser().resolve()
    digest = hashlib.sha256()
    for table in sorted((root / "Dataframes").glob("*.csv")):
        digest.update(table.name.encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(table)))
    return digest.hexdigest()


def select_gsv_images(root: str | Path, count: int, seed: int) -> list[dict[str, object]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in gsv_rows(root):
        grouped[str(row["city"])].extend(row["paths"])
    cities = sorted(grouped)
    base, remainder = divmod(int(count), len(cities))
    all_paths = sorted(path for paths in grouped.values() for path in paths)
    path_to_index = {path: index for index, path in enumerate(all_paths)}
    selected = []
    for position, city in enumerate(cities):
        quota = base + int(position < remainder)
        candidates = sorted(grouped[city])
        if quota > len(candidates):
            raise ValueError(f"GSV-Cities fitting quota exceeds {city} images")
        rng = random.Random(int(seed) + position * 1_000_003)
        selected.extend(
            {"dataset": city, "source_path": path, "sample_index": path_to_index[path]}
            for path in rng.sample(candidates, quota)
        )
    random.Random(int(seed) + 97).shuffle(selected)
    return selected


class GSVPlaceDataset(Dataset):
    def __init__(self, root: str | Path, seed: int, image_size: int = 224):
        self.rows = gsv_rows(root)
        self.seed = int(seed)
        self.epoch = 0
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((image_size, image_size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.RandAugment(num_ops=3, magnitude=15, interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        paths = rng.sample(row["paths"], 4)
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(self.transform(image.convert("RGB")))
        return torch.stack(images), torch.full((4,), index, dtype=torch.long)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_boq_small(config_path: str | Path, *, smoke_steps: int = 0) -> Path:
    import yaml
    from pytorch_metric_learning import losses, miners

    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["boq_small"]
    if int(config["descriptor_dim"]) != 12_288:
        raise ValueError("BoQ-S training requires a 12,288-D descriptor")
    if int(config["images_per_place"]) != 4:
        raise ValueError("BoQ-S training requires four images per place")
    if (
        int(config["places_per_batch"]) * int(config["gradient_accumulation_steps"])
        != int(config["effective_places_per_batch"])
    ):
        raise ValueError("BoQ-S physical and effective batch settings are inconsistent")
    if int(config["train_image_size"]) % 14:
        raise ValueError("BoQ-S training image size must be divisible by the patch size")
    if not config.get("teacher_checkpoint") or not config.get("teacher_repo"):
        raise ValueError("BoQ-S distillation requires a BoQ-B teacher checkpoint and repo")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "checkpoints" / "latest.pt"
    final = output / "model_best.pt"
    seed = int(config["random_seed"])
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("BoQ-S training requires CUDA")
    dataset = GSVPlaceDataset(config["gsv_cities_root"], seed, int(config["train_image_size"]))
    manifest_sha = gsv_manifest_fingerprint(config["gsv_cities_root"])
    backbone_sha = file_sha256(config["backbone_checkpoint"])
    teacher_sha = file_sha256(config["teacher_checkpoint"])
    model = BoQSmallModel(
        config["dinov2_repo"],
        config["backbone_name"],
        config["backbone_checkpoint"],
        int(config["trainable_blocks"]),
        config.get("boq_repo", ""),
    ).to(device)
    teacher = BoQReferenceModel(
        config["teacher_repo"], config["teacher_checkpoint"],
        backbone_name=config.get("teacher_backbone_name", "dinov2_vitb14"),
        descriptor_dim=12_288,
    ).eval().to(device)
    teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": float(config["learning_rate"])},
            {"params": model.aggregator.parameters(), "lr": float(config["learning_rate"])},
        ],
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(config["milestones"]), gamma=float(config["lr_multiplier"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    start_epoch, optimizer_step = 0, 0
    if latest.is_file():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state.get("config") != config:
            raise RuntimeError("Existing BoQ-S checkpoint configuration does not match")
        if state.get("gsv_manifest_sha256") != manifest_sha:
            raise RuntimeError("Existing BoQ-S checkpoint GSV-Cities manifest does not match")
        if state.get("backbone_checkpoint_sha256") != backbone_sha:
            raise RuntimeError("Existing BoQ-S checkpoint backbone does not match")
        if state.get("teacher_checkpoint_sha256") != teacher_sha:
            raise RuntimeError("Existing BoQ-S checkpoint teacher does not match")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_rng_state(state["rng"])
        start_epoch = int(state["epoch"])
        optimizer_step = int(state["optimizer_step"])
    write_json({
        "schema": "boq_small_training_v1",
        "config": config,
        "gsv_manifest_sha256": manifest_sha,
        "backbone_checkpoint_sha256": backbone_sha,
        "teacher_checkpoint_sha256": teacher_sha,
        "teacher_backbone": config.get("teacher_backbone_name", "dinov2_vitb14"),
        "distillation_loss": config.get("distillation_loss", "scaled_mse"),
        "distillation_weight": float(config.get("distillation_weight", 1.0)),
        "official_checkpoint": False,
        "training_status": "re-trained-and-distilled",
        "selection_rule": "fixed_budget_last_epoch",
    }, output / "resolved_config.json")
    epochs = int(config["epochs"])
    accumulation = int(config["gradient_accumulation_steps"])
    stop_after = int(smoke_steps)
    metric_loss = losses.MultiSimilarityLoss(alpha=1, beta=50, base=0.0)
    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    for epoch in range(start_epoch, epochs):
        dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=int(config["places_per_batch"]),
            shuffle=True,
            generator=generator,
            num_workers=int(config["workers"]),
            pin_memory=True,
            persistent_workers=False,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(loader, desc=f"boq-small epoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        completed_batches = 0
        for batch_index, (images, labels) in enumerate(progress):
            images = images.flatten(0, 1).to(device, non_blocking=True)
            labels = labels.flatten().to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                descriptors = model.encode(images)
                hard_pairs = miner(descriptors.float(), labels)
                retrieval_loss = metric_loss(descriptors.float(), labels, hard_pairs)
                with torch.no_grad():
                    teacher_descriptors = teacher.encode(images)
                distill_loss = F.mse_loss(
                    descriptors.float(), teacher_descriptors.float(), reduction="mean"
                ) * descriptors.shape[1]
                loss = retrieval_loss + float(config.get("distillation_weight", 1.0)) * distill_loss
            group_start = (batch_index // accumulation) * accumulation
            divisor = min(accumulation, len(loader) - group_start)
            scaler.scale(loss / divisor).backward()
            update = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            if update:
                if epoch < int(config["warmup_epochs"]):
                    steps_per_epoch = max(1, (len(loader) + accumulation - 1) // accumulation)
                    warmup_steps = int(config["warmup_epochs"]) * steps_per_epoch
                    scale = min(1.0, (optimizer_step + 1) / warmup_steps)
                    for group in optimizer.param_groups:
                        group["lr"] = float(config["learning_rate"]) * scale
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
            epoch_loss += float(loss.item())
            completed_batches += 1
            progress.set_postfix(loss=f"{loss.item():.3f}")
            if stop_after and completed_batches >= stop_after:
                break
        if stop_after:
            break
        scheduler.step()
        state = {
            "kind": "boq_small_training",
            "epoch": epoch + 1,
            "optimizer_step": optimizer_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "mean_training_loss": epoch_loss / max(completed_batches, 1),
            "random_seed": seed,
            "descriptor_dim": 12_288,
            "input_size": [322, 322],
            "backbone_checkpoint_sha256": backbone_sha,
            "gsv_manifest_sha256": manifest_sha,
            "teacher_checkpoint_sha256": teacher_sha,
            "teacher_backbone": config.get("teacher_backbone_name", "dinov2_vitb14"),
            "distillation_loss": config.get("distillation_loss", "scaled_mse"),
            "distillation_weight": float(config.get("distillation_weight", 1.0)),
            "official_checkpoint": False,
            "training_status": "re-trained-and-distilled",
            "selection_rule": "fixed_budget_last_epoch",
            "rng": rng_state(),
        }
        atomic_torch_save(state, latest)
        atomic_torch_save({
            key: value for key, value in state.items()
            if key not in {"optimizer", "scheduler", "scaler"}
        }, final)
    return final


class _ImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_size: int):
        self.rows = rows
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((image_size, image_size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.rows[index]["source_path"]) as image:
            return self.transform(image.convert("RGB")), index


def fit_anyloc_small_vocabulary(
    registry_path: str | Path,
    model_name: str,
    output: str | Path,
    *,
    samples: int = 20_000,
    patches_per_image: int = 32,
    random_seed: int = 0,
    batch_size: int = 16,
    workers: int = 4,
) -> Path:
    import yaml
    import joblib
    from sklearn.cluster import MiniBatchKMeans

    registry = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8"))["reference_models"]
    spec = registry[model_name]
    rows = select_gsv_images(spec["fit_root"], samples, random_seed)
    output = Path(output)
    metadata = {
        "schema": "anyloc_small_vocabulary_v2",
        "feature_extraction": "dinov2_block_output_pre_norm",
        "source_model": model_name,
        "backbone_checkpoint_sha256": file_sha256(spec["checkpoint"]),
        "gsv_manifest_sha256": gsv_manifest_fingerprint(spec["fit_root"]),
        "sample_count": int(samples),
        "sample_paths": [row["source_path"] for row in rows],
        "sample_indices": [row["sample_index"] for row in rows],
        "patches_per_image": int(patches_per_image),
        "random_seed": int(random_seed),
        "layer": int(spec["layer"]),
        "facet": spec["facet"],
        "num_clusters": int(spec["num_clusters"]),
        "patch_dim": 384,
        "descriptor_dim": 12_288,
        "normalization": "l2-before-cosine-kmeans",
        "official_checkpoint": False,
        "training_status": "re-fitted",
    }
    if output.is_file():
        state = torch.load(output, map_location="cpu", weights_only=True)
        if state.get("metadata") == metadata:
            return output
        raise RuntimeError("Existing AnyLoc-S vocabulary metadata does not match")
    partial = output.with_suffix(".partial.joblib")
    partial_metadata = output.with_suffix(".partial.json")
    completed_batches = 0
    if partial.is_file() or partial_metadata.is_file():
        if not partial.is_file() or not partial_metadata.is_file():
            raise RuntimeError("AnyLoc-S vocabulary partial checkpoint is incomplete")
        stored = json.loads(partial_metadata.read_text(encoding="utf-8"))
        if stored.get("identity") != metadata:
            raise RuntimeError("AnyLoc-S vocabulary partial checkpoint metadata does not match")
        state = joblib.load(partial)
        kmeans = state["kmeans"]
        completed_batches = int(state["completed_batches"])
    else:
        kmeans = MiniBatchKMeans(
            n_clusters=int(spec["num_clusters"]), random_state=int(random_seed),
            batch_size=max(4096, int(batch_size) * int(patches_per_image)),
            n_init=3, reassignment_ratio=0.0,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("AnyLoc-S vocabulary fitting requires CUDA")
    # A temporary identity vocabulary is sufficient to instantiate the feature extractor.
    temporary = output.with_suffix(".bootstrap.pt")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({"centers": torch.eye(32, 384), "metadata": {}}, temporary)
    model = AnyLocSmallModel(
        spec["repo"], spec["backbone_name"], spec["checkpoint"], temporary,
        layer=int(spec["layer"]), facet=spec["facet"], num_clusters=int(spec["num_clusters"]),
    ).eval().to(device)
    completed_images = min(len(rows), completed_batches * int(batch_size))
    loader = DataLoader(
        _ImageDataset(rows[completed_images:], int(spec["native_size"][0])),
        batch_size=int(batch_size), num_workers=int(workers), pin_memory=True,
    )
    with torch.inference_mode():
        for local_batch, (images, _) in enumerate(tqdm(loader, desc="anyloc-small/vocabulary")):
            patches = model.patch_descriptors(images.to(device, non_blocking=True)).cpu()
            global_start = completed_images + local_batch * int(batch_size)
            selected = torch.stack([
                value[torch.randperm(
                    value.shape[0],
                    generator=torch.Generator().manual_seed(
                        int(random_seed) + int(rows[global_start + offset]["sample_index"])
                    ),
                )[:patches_per_image]]
                for offset, value in enumerate(patches)
            ]).flatten(0, 1).numpy()
            kmeans.partial_fit(selected)
            completed_batches += 1
            if completed_batches % 100 == 0:
                temporary_partial = partial.with_suffix(partial.suffix + ".tmp")
                joblib.dump({
                    "kmeans": kmeans,
                    "completed_batches": completed_batches,
                }, temporary_partial)
                os.replace(temporary_partial, partial)
                write_json({
                    "identity": metadata,
                    "completed_batches": completed_batches,
                }, partial_metadata)
    centers = F.normalize(torch.from_numpy(kmeans.cluster_centers_).float(), dim=1)
    atomic_torch_save({"kind": "anyloc_vocabulary", "centers": centers, "metadata": metadata}, output)
    write_json(metadata, output.with_suffix(".json"))
    temporary.unlink(missing_ok=True)
    partial.unlink(missing_ok=True)
    partial_metadata.unlink(missing_ok=True)
    return output
