from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..checkpointing import file_sha256, write_json
from ..data import read_generated_manifest
from ..evaluation import extract_descriptors
from ..runtime import load_vpr_model
from ..transforms import VPRTransform
from ..robotcar_temporal import _top_scores


ROUTE_MINING_TOP_K = 100


def _ids_sha256(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f'{row["manifest_index"]}\0{row["image_id"]}\n'.encode())
    return digest.hexdigest()


def mine_robotcar_route_candidates(
    config,
    checkpoint: str | Path,
    output: str | Path | None = None,
) -> Path:
    if config.vpr.descriptor_dim != 4096:
        raise ValueError("Route mining requires a 4096-dimensional checkpoint")
    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output or config.vpr.route_mining_cache).expanduser().resolve()
    if not str(output):
        raise ValueError("Route mining output path is required")
    manifest = Path(config.data.generated_root) / "manifest.csv"
    all_rows = read_generated_manifest(manifest)
    rows = []
    for manifest_index, row in enumerate(all_rows):
        if row.get("dataset") != "robotcar" or row.get("split") != "train":
            continue
        if not row.get("utm_e") or not row.get("utm_n") or not row.get("yaw"):
            continue
        rows.append({**row, "manifest_index": manifest_index})
    if not rows:
        raise ValueError("No metric RobotCar day rows are available for route mining")
    descriptors_path = output.with_suffix(".descriptors.npy")
    metadata_path = output.with_suffix(".json")
    expected = {
        "schema": "robotcar_route_mining_v1",
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest_sha256": file_sha256(manifest),
        "descriptor_dim": 4096,
        "top_k": min(ROUTE_MINING_TOP_K, len(rows)),
        "rows": len(rows),
        "image_ids_sha256": _ids_sha256(rows),
        "positive_radius_m": config.vpr.route_positive_radius_m,
        "positive_yaw_deg": config.vpr.route_positive_yaw_deg,
        "negative_radius_m": config.vpr.route_negative_radius_m,
    }
    if output.is_file() and descriptors_path.is_file() and metadata_path.is_file():
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(actual.get(key) == value for key, value in expected.items()):
            if (
                actual.get("cache_sha256") == file_sha256(output)
                and actual.get("descriptors_sha256") == file_sha256(descriptors_path)
            ):
                return output

    model, state, _ = load_vpr_model(config, checkpoint)
    if int(getattr(model, "descriptor_dim", 0)) != 4096:
        raise ValueError("Route mining model descriptor dimension must be 4096")
    if int(state.get("descriptor_dim", 4096)) != 4096:
        raise ValueError("Route mining checkpoint descriptor dimension must be 4096")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    transform = VPRTransform(config.vpr.image_height, config.vpr.image_width)
    descriptor_rows = [SimpleNamespace(source_path=row["source_path"]) for row in rows]
    descriptors = extract_descriptors(
        model, descriptor_rows, transform, device,
        max(1, config.vpr.batch_size * 2), config.vpr.workers,
    ).astype(np.float32)
    candidate_indices, _ = _top_scores(descriptors, descriptors, expected["top_k"])
    coordinates = np.asarray(
        [(float(row["utm_e"]), float(row["utm_n"])) for row in rows], dtype=np.float64
    )
    yaw = np.asarray([float(row["yaw"]) for row in rows], dtype=np.float32)
    image_ids = np.asarray([row["image_id"] for row in rows], dtype=object)
    positives = np.zeros(candidate_indices.shape, dtype=np.bool_)
    negatives = np.zeros(candidate_indices.shape, dtype=np.bool_)
    for anchor in range(len(rows)):
        candidates = candidate_indices[anchor]
        distances = np.linalg.norm(coordinates[candidates] - coordinates[anchor], axis=1)
        yaw_delta = np.abs((yaw[candidates] - yaw[anchor] + 180.0) % 360.0 - 180.0)
        different = image_ids[candidates] != image_ids[anchor]
        positives[anchor] = (
            different
            & (distances <= config.vpr.route_positive_radius_m)
            & (yaw_delta <= config.vpr.route_positive_yaw_deg)
        )
        negatives[anchor] = different & (distances > config.vpr.route_negative_radius_m)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            manifest_indices=np.asarray([row["manifest_index"] for row in rows], dtype=np.int64),
            candidate_indices=candidate_indices.astype(np.int32),
            positive_mask=positives,
            negative_mask=negatives,
        )
    os.replace(temporary, output)
    descriptor_temp = descriptors_path.with_suffix(descriptors_path.suffix + ".tmp")
    with descriptor_temp.open("wb") as handle:
        np.save(handle, descriptors.astype(np.float16))
    os.replace(descriptor_temp, descriptors_path)
    write_json({
        **expected,
        "cache_sha256": file_sha256(output),
        "descriptors_sha256": file_sha256(descriptors_path),
        "image_ids": [row["image_id"] for row in rows],
        "valid_positive_anchors": int(positives.any(axis=1).sum()),
        "valid_negative_anchors": int(negatives.any(axis=1).sum()),
    }, metadata_path)
    return output


class RouteMiningCache:
    def __init__(self, config, generated_manifest: str | Path):
        path = Path(config.vpr.route_mining_cache).expanduser().resolve()
        descriptors_path = path.with_suffix(".descriptors.npy")
        metadata_path = path.with_suffix(".json")
        if not path.is_file() or not descriptors_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("Route mining cache is incomplete; run mine-robotcar-route first")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "schema": "robotcar_route_mining_v1",
            "manifest_sha256": file_sha256(generated_manifest),
            "descriptor_dim": config.vpr.descriptor_dim,
            "checkpoint_sha256": file_sha256(config.vpr.finetune_checkpoint),
            "positive_radius_m": config.vpr.route_positive_radius_m,
            "positive_yaw_deg": config.vpr.route_positive_yaw_deg,
            "negative_radius_m": config.vpr.route_negative_radius_m,
        }
        mismatches = {
            key: (self.metadata.get(key), value)
            for key, value in expected.items() if self.metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Route mining cache metadata mismatch: {mismatches}")
        if self.metadata.get("cache_sha256") != file_sha256(path):
            raise RuntimeError("Route mining cache SHA256 mismatch")
        if self.metadata.get("descriptors_sha256") != file_sha256(descriptors_path):
            raise RuntimeError("Route mining descriptor SHA256 mismatch")
        with np.load(path) as arrays:
            self.manifest_indices = arrays["manifest_indices"].copy()
            self.candidate_indices = arrays["candidate_indices"].copy()
            self.positive_mask = arrays["positive_mask"].copy()
            self.negative_mask = arrays["negative_mask"].copy()
        self.descriptors = np.load(descriptors_path, mmap_mode="r")
        self.by_manifest_index = {
            int(value): index for index, value in enumerate(self.manifest_indices)
        }

    @property
    def fingerprint(self) -> str:
        return str(self.metadata["cache_sha256"])

    def candidates(self, manifest_indices: torch.Tensor, device: torch.device):
        local = np.asarray([
            self.by_manifest_index.get(int(value), -1)
            for value in manifest_indices.detach().cpu().tolist()
        ], dtype=np.int64)
        batch = len(local)
        top_k = self.candidate_indices.shape[1]
        descriptors = np.zeros((batch, top_k, self.descriptors.shape[1]), dtype=np.float16)
        positives = np.zeros((batch, top_k), dtype=np.bool_)
        negatives = np.zeros((batch, top_k), dtype=np.bool_)
        valid = local >= 0
        if valid.any():
            candidates = self.candidate_indices[local[valid]]
            descriptors[valid] = self.descriptors[candidates]
            positives[valid] = self.positive_mask[local[valid]]
            negatives[valid] = self.negative_mask[local[valid]]
        return (
            torch.from_numpy(descriptors).to(device=device, dtype=torch.float32),
            torch.from_numpy(positives).to(device=device),
            torch.from_numpy(negatives).to(device=device),
        )
