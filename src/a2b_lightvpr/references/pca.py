from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from ..checkpointing import atomic_torch_save, file_sha256, write_json
from ..data import ImagePathDataset, read_generated_manifest
from .models import FixedResizeTransform, _registry, build_reference_model, suite_rows
from .training import gsv_manifest_fingerprint, select_gsv_images
from ..trained_suite import SUITE_PROTOCOLS


def _stratified_indices(rows: list[dict], count: int, seed: int) -> list[int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row.get("dataset", "unknown")].append(index)
    datasets = sorted(grouped)
    if not datasets:
        raise ValueError("PCA fitting manifest is empty")
    base, remainder = divmod(int(count), len(datasets))
    rng = random.Random(int(seed))
    selected = []
    for position, dataset in enumerate(datasets):
        quota = base + int(position < remainder)
        if len(grouped[dataset]) < quota:
            raise ValueError(f"PCA sample quota exceeds {dataset} rows")
        selected.extend(rng.sample(grouped[dataset], quota))
    rng.shuffle(selected)
    return selected


def _assert_no_evaluation_overlap(config, selected_rows: list[dict]) -> None:
    training_paths = {
        str(Path(row["source_path"]).expanduser().resolve())
        for row in selected_rows
    }
    evaluation_paths = set()
    for dataset, split, _, _, _ in SUITE_PROTOCOLS.values():
        database, queries = suite_rows(config, dataset, split)
        evaluation_paths.update(
            str(Path(row.source_path).expanduser().resolve())
            for row in (*database, *queries)
        )
    overlap = training_paths & evaluation_paths
    if overlap:
        raise ValueError(f"PCA fitting data overlaps evaluation data: {next(iter(overlap))}")


def _cache_descriptors(
    model,
    rows,
    transform,
    device: torch.device,
    batch_size: int,
    workers: int,
    output: Path,
    metadata: dict,
) -> np.ndarray:
    metadata_path = output.with_suffix(".meta.json")
    completed_rows = 0
    if output.is_file() and metadata_path.is_file():
        stored = json.loads(metadata_path.read_text(encoding="utf-8"))
        if stored == metadata:
            descriptors = np.load(output, mmap_mode="r")
            expected = (len(rows), int(model.descriptor_dim))
            if descriptors.shape != expected or descriptors.dtype != np.float16:
                raise RuntimeError("Existing PCA descriptor cache has the wrong shape or dtype")
            return descriptors
        if stored.get("identity") != metadata:
            raise RuntimeError("Existing PCA descriptor cache metadata does not match")
        completed_rows = int(stored.get("completed_rows", 0))
        descriptors = np.load(output, mmap_mode="r+")
        expected = (len(rows), int(model.descriptor_dim))
        if descriptors.shape != expected or descriptors.dtype != np.float16:
            raise RuntimeError("Existing PCA descriptor cache has the wrong shape or dtype")
        if completed_rows == len(rows) and stored.get("complete"):
            return np.load(output, mmap_mode="r")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptors = np.lib.format.open_memmap(
            output,
            mode="w+",
            dtype=np.float16,
            shape=(len(rows), int(model.descriptor_dim)),
        )
        write_json({"identity": metadata, "completed_rows": 0, "complete": False}, metadata_path)
    if not 0 <= completed_rows <= len(rows):
        raise RuntimeError("PCA descriptor cache has an invalid resume cursor")
    dataset = ImagePathDataset(rows, transform)
    loader = DataLoader(
        Subset(dataset, range(completed_rows, len(dataset))),
        batch_size=int(batch_size),
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
    )
    model.eval().to(device)
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for batch_index, (images, indices) in enumerate(
            tqdm(loader, desc="pca/train-descriptors")
        ):
            with torch.autocast(device.type, enabled=use_amp):
                values = model.encode(images.to(device, non_blocking=True))
            descriptors[indices.numpy()] = values.float().cpu().numpy().astype(np.float16)
            completed_rows = int(indices.max().item()) + 1
            if (batch_index + 1) % 10 == 0:
                descriptors.flush()
                write_json({
                    "identity": metadata,
                    "completed_rows": completed_rows,
                    "complete": False,
                }, metadata_path)
    descriptors.flush()
    write_json({
        "identity": metadata,
        "completed_rows": len(rows),
        "complete": True,
    }, metadata_path)
    return np.load(output, mmap_mode="r")


def fit_reference_pca(
    config,
    registry_path: str | Path,
    model_name: str,
    output: str | Path,
    descriptor_cache: str | Path,
    *,
    components: int = 512,
    samples: int = 20_000,
    random_seed: int = 0,
    batch_size: int = 16,
    workers: int = 4,
) -> Path:
    registry = _registry(registry_path)
    if model_name not in registry:
        raise ValueError(f"Unknown PCA source model: {model_name}")
    spec = registry[model_name]
    if spec.get("loader") not in {"boq_reference", "boq_small", "anyloc_small", "cricavpr_small"}:
        raise ValueError("PCA fitting requires a raw BoQ, AnyLoc or CricaVPR-S reference")
    if spec.get("batch_dependent") and int(batch_size) != 1:
        raise ValueError("Batch-dependent reference PCA requires batch-size=1 to match single-image evaluation")
    if not 0 < int(components) < int(spec["descriptor_dim"]):
        raise ValueError("PCA components must be below the source descriptor dimension")
    if int(samples) < int(components) + 32:
        raise ValueError("PCA sample count must exceed components plus oversampling")

    if spec.get("fit_source") == "gsv_cities":
        selected_rows = select_gsv_images(spec["fit_root"], int(samples), int(random_seed))
        indices = [int(row["sample_index"]) for row in selected_rows]
        manifest_sha256 = gsv_manifest_fingerprint(spec["fit_root"])
        fitting_source = str(Path(spec["fit_root"]).expanduser().resolve())
    else:
        manifest = Path(config.data.generated_root) / "manifest.csv"
        rows = read_generated_manifest(manifest)
        indices = _stratified_indices(rows, int(samples), int(random_seed))
        selected_rows = [rows[index] for index in indices]
        manifest_sha256 = file_sha256(manifest)
        fitting_source = str(manifest.resolve())
    _assert_no_evaluation_overlap(config, selected_rows)
    artifact_hashes = {
        str(Path(path).expanduser().resolve()): file_sha256(path)
        for path in spec.get("required_files", [])
        if Path(path).expanduser().is_file()
    }
    metadata = {
        "schema": "reference_pca_fit_v1",
        "source_model": model_name,
        "source_checkpoint_sha256": file_sha256(spec["checkpoint"]),
        "manifest_sha256": manifest_sha256,
        "fitting_source": fitting_source,
        "input_dim": int(spec["descriptor_dim"]),
        "output_dim": int(components),
        "sample_count": int(samples),
        "sample_indices": indices,
        "random_seed": int(random_seed),
        "q": int(components) + 32,
        "niter": 2,
        "whiten": False,
        "datasets": sorted({row.get("dataset", "unknown") for row in selected_rows}),
        "source_artifact_sha256": artifact_hashes,
    }
    output = Path(output)
    if spec.get("batch_dependent"):
        metadata["extraction_batch_size"] = int(batch_size)
        metadata["input_size"] = list(spec["native_size"])
    if output.is_file():
        stored = torch.load(output, map_location="cpu", weights_only=True)
        if stored.get("kind") == "reference_pca" and stored.get("metadata") == metadata:
            return output
        raise RuntimeError("Existing PCA checkpoint metadata does not match")

    model = build_reference_model(spec)
    height, width = (int(value) for value in spec["native_size"])
    transform = FixedResizeTransform(height, width)
    image_rows = [SimpleNamespace(source_path=row["source_path"]) for row in selected_rows]
    cache_metadata = {
        key: metadata[key]
        for key in (
            "schema", "source_model", "source_checkpoint_sha256", "manifest_sha256",
            "sample_count", "sample_indices", "random_seed", "source_artifact_sha256",
        )
    }
    if spec.get("batch_dependent"):
        cache_metadata.update({
            "extraction_batch_size": int(batch_size),
            "input_size": list(spec["native_size"]),
        })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Reference PCA fitting requires CUDA")
    descriptors = _cache_descriptors(
        model,
        image_rows,
        transform,
        device,
        int(batch_size),
        int(workers),
        Path(descriptor_cache),
        cache_metadata,
    )
    model.cpu()
    del model
    torch.cuda.empty_cache()

    values = torch.from_numpy(np.asarray(descriptors, dtype=np.float32)).to(device)
    mean = values.mean(dim=0)
    centered = values - mean
    torch.manual_seed(int(random_seed))
    _, singular_values, vectors = torch.pca_lowrank(
        centered,
        q=int(components) + 32,
        center=False,
        niter=2,
    )
    state = {
        "kind": "reference_pca",
        "input_dim": int(spec["descriptor_dim"]),
        "output_dim": int(components),
        "mean": mean.cpu(),
        "components": vectors[:, : int(components)].cpu(),
        "singular_values": singular_values[: int(components)].cpu(),
        "metadata": metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(state, output)
    return output
