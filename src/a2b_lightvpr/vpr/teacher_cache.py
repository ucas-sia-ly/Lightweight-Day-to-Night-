from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..checkpointing import file_sha256, write_json
from ..config import ExperimentConfig
from ..data import read_generated_manifest
from ..transforms import IMAGENET_MEAN, IMAGENET_STD
from .teachers import build_relation_teacher


CACHE_SCHEMA_VERSION = 1


def _cache_paths(path: str | Path) -> tuple[Path, Path, Path]:
    data = Path(path).expanduser().resolve()
    if data.suffix != ".npy":
        raise ValueError("relation_teacher_cache must end in .npy")
    return data, data.with_suffix(".meta.json"), data.with_suffix(".ids.json")


def _ids_sha256(image_ids: list[str]) -> str:
    payload = json.dumps(image_ids, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _expected_metadata(config: ExperimentConfig, rows: list[dict], descriptor_dim: int) -> dict:
    cfg = config.vpr
    manifest = Path(config.data.generated_root) / "manifest.csv"
    image_ids = [row["image_id"] for row in rows]
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "teacher": cfg.relation_teacher,
        "manifest_sha256": file_sha256(manifest),
        "checkpoint_sha256": file_sha256(cfg.relation_teacher_checkpoint),
        "image_ids_sha256": _ids_sha256(image_ids),
        "input_size": int(cfg.relation_teacher_input_size),
        "descriptor_dim": int(descriptor_dim),
        "rows": len(rows),
        "dtype": "float16",
    }


class _TeacherImages(Dataset):
    def __init__(self, rows: list[dict], input_size: int, start: int):
        self.rows = rows
        self.input_size = int(input_size)
        self.start = int(start)

    def __len__(self) -> int:
        return len(self.rows) - self.start

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row_index = self.start + index
        with Image.open(self.rows[row_index]["source_path"]) as image:
            tensor = TF.to_tensor(image.convert("RGB"))
        tensor = TF.resize(
            tensor,
            [self.input_size, self.input_size],
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True,
        )
        return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD), row_index


def cache_relation_teacher(config: ExperimentConfig) -> Path:
    cfg = config.vpr
    if cfg.relation_teacher not in {"salad", "boq"}:
        raise ValueError("cache-vpr-teacher requires relation_teacher=salad or boq")
    manifest = Path(config.data.generated_root) / "manifest.csv"
    rows = read_generated_manifest(manifest)
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("Teacher cache accepts train rows only")
    image_ids = [row["image_id"] for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Teacher cache requires unique image_id values")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_relation_teacher(
        cfg.relation_teacher, cfg.relation_teacher_repo, cfg.relation_teacher_checkpoint
    ).to(device)
    descriptor_dim = int(model.output_dim)
    expected = _expected_metadata(config, rows, descriptor_dim)
    data_path, meta_path, ids_path = _cache_paths(cfg.relation_teacher_cache)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    if meta_path.is_file() or data_path.is_file() or ids_path.is_file():
        if not (meta_path.is_file() and data_path.is_file() and ids_path.is_file()):
            raise RuntimeError("Teacher cache is partial but lacks data, metadata, or ID file")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"Teacher cache metadata mismatch for {key}")
        stored_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if stored_ids != image_ids:
            raise RuntimeError("Teacher cache image IDs do not match the manifest")
        completed = int(metadata.get("completed_rows", 0))
        descriptors = np.lib.format.open_memmap(data_path, mode="r+")
        if descriptors.shape != (len(rows), descriptor_dim) or descriptors.dtype != np.float16:
            raise RuntimeError("Teacher cache array shape or dtype is invalid")
    else:
        write_json(image_ids, ids_path)
        descriptors = np.lib.format.open_memmap(
            data_path, mode="w+", dtype=np.float16, shape=(len(rows), descriptor_dim)
        )
        metadata = {**expected, "completed_rows": 0, "complete": False}
        write_json(metadata, meta_path)

    if completed >= len(rows):
        metadata.update(completed_rows=len(rows), complete=True)
        write_json(metadata, meta_path)
        return data_path

    loader = DataLoader(
        _TeacherImages(rows, cfg.relation_teacher_input_size, completed),
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        pin_memory=device.type == "cuda",
        shuffle=False,
    )
    use_amp = bool(cfg.amp and device.type == "cuda")
    with torch.inference_mode():
        for images, indices in tqdm(loader, desc=f"cache/{cfg.relation_teacher}"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                output = model(images)
            output = torch.nn.functional.normalize(output.float(), dim=1).cpu().numpy().astype(np.float16)
            first, last = int(indices[0]), int(indices[-1]) + 1
            if last - first != output.shape[0]:
                raise RuntimeError("Teacher cache loader returned non-contiguous indices")
            descriptors[first:last] = output
            descriptors.flush()
            completed = last
            metadata.update(completed_rows=completed, complete=False)
            write_json(metadata, meta_path)
    metadata.update(completed_rows=len(rows), complete=True)
    write_json(metadata, meta_path)
    return data_path


class TeacherDescriptorCache:
    def __init__(self, config: ExperimentConfig, manifest: str | Path):
        cfg = config.vpr
        data_path, meta_path, ids_path = _cache_paths(cfg.relation_teacher_cache)
        if not (data_path.is_file() and meta_path.is_file() and ids_path.is_file()):
            raise FileNotFoundError("Strong teacher cache is missing; run cache-vpr-teacher first")
        rows = read_generated_manifest(manifest)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        descriptor_dim = int(metadata.get("descriptor_dim", 0))
        expected = _expected_metadata(config, rows, descriptor_dim)
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"Teacher cache metadata mismatch for {key}")
        if not metadata.get("complete") or metadata.get("completed_rows") != len(rows):
            raise RuntimeError("Strong teacher cache is incomplete")
        self.image_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        expected_ids = [row["image_id"] for row in rows]
        if self.image_ids != expected_ids:
            raise RuntimeError("Teacher cache image IDs do not match the manifest")
        self.path = data_path
        self.shape = (len(rows), descriptor_dim)
        self._descriptors = None
        self.fingerprint = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _array(self):
        if self._descriptors is None:
            self._descriptors = np.load(self.path, mmap_mode="r")
            if self._descriptors.shape != self.shape or self._descriptors.dtype != np.float16:
                raise RuntimeError("Teacher cache array shape or dtype is invalid")
        return self._descriptors

    def descriptor(self, index: int, image_id: str) -> torch.Tensor:
        if self.image_ids[index] != image_id:
            raise RuntimeError(f"Teacher cache image ID mismatch at row {index}")
        return torch.from_numpy(np.array(self._array()[index], dtype=np.float32, copy=True))
