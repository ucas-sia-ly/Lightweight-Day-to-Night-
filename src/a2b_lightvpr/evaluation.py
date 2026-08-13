from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpointing import write_json
from .config import EvalConfig
from .data import ImagePathDataset, RobotCarRow, benchmark_rows
from .transforms import VPRTransform


def circular_yaw_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = np.abs(first - second) % 360.0
    return np.minimum(difference, 360.0 - difference)


def positive_indices(
    database: list[RobotCarRow],
    queries: list[RobotCarRow],
    radius_m: float,
    yaw_deg: float,
) -> list[np.ndarray]:
    db_xy = np.asarray([(row.easting, row.northing) for row in database], dtype=np.float64)
    db_yaw = np.asarray([row.yaw_deg for row in database], dtype=np.float64)
    positives = []
    for query in queries:
        distances = np.linalg.norm(db_xy - np.asarray([query.easting, query.northing]), axis=1)
        yaw_difference = circular_yaw_difference(db_yaw, np.asarray(query.yaw_deg))
        positives.append(np.flatnonzero((distances <= radius_m) & (yaw_difference <= yaw_deg)))
    return positives


def retrieval_metrics(
    database_descriptors: np.ndarray,
    query_descriptors: np.ndarray,
    positives: list[np.ndarray],
    recall_values: tuple[int, ...],
) -> dict:
    database_descriptors = np.asarray(database_descriptors, dtype=np.float32)
    query_descriptors = np.asarray(query_descriptors, dtype=np.float32)
    maximum = min(max(recall_values), len(database_descriptors))
    try:
        import faiss

        index = faiss.IndexFlatIP(database_descriptors.shape[1])
        index.add(database_descriptors)
        _, predictions = index.search(query_descriptors, maximum)
    except ImportError:
        scores = query_descriptors @ database_descriptors.T
        predictions = np.argpartition(-scores, maximum - 1, axis=1)[:, :maximum]
        order = np.take_along_axis(scores, predictions, axis=1).argsort(axis=1)[:, ::-1]
        predictions = np.take_along_axis(predictions, order, axis=1)

    valid = np.asarray([len(items) > 0 for items in positives], dtype=bool)
    success_by_k = {}
    recalls = {}
    first_ranks = np.full(len(positives), np.inf)
    for query_index, positive in enumerate(positives):
        if not valid[query_index]:
            continue
        matches = np.flatnonzero(np.isin(predictions[query_index], positive))
        if len(matches):
            first_ranks[query_index] = int(matches[0]) + 1
    for value in recall_values:
        success = valid & (first_ranks <= value)
        success_by_k[str(value)] = success.astype(np.uint8).tolist()
        recalls[str(value)] = float(success.sum() / valid.sum())
    reciprocal = np.where(np.isfinite(first_ranks), 1.0 / first_ranks, 0.0)
    return {
        "recalls": recalls,
        "mrr": float(reciprocal[valid].mean()),
        "valid_queries": int(valid.sum()),
        "total_queries": len(positives),
        "valid_by_query": valid.astype(np.uint8).tolist(),
        "success_by_k": success_by_k,
    }


def extract_descriptors(model, rows, transform, device, batch_size=64, workers=4):
    dataset = ImagePathDataset(rows, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda")
    output = np.empty((len(dataset), model.descriptor_dim), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for images, indices in tqdm(loader, desc="descriptors", leave=False):
            descriptors = model.encode(images.to(device, non_blocking=True)).float().cpu().numpy()
            output[indices.numpy()] = descriptors
    return output


def evaluate_model(
    model,
    robotcar_manifest: str,
    split: str,
    transform: VPRTransform,
    evaluation: EvalConfig,
    device: torch.device,
    *,
    batch_size: int = 64,
    workers: int = 4,
    output_path: str | Path | None = None,
) -> dict:
    database, queries = benchmark_rows(robotcar_manifest, split)
    database_descriptors = extract_descriptors(model, database, transform, device, batch_size, workers)
    query_descriptors = extract_descriptors(model, queries, transform, device, batch_size, workers)
    positives = positive_indices(
        database, queries, evaluation.positive_radius_m, evaluation.positive_yaw_deg
    )
    metrics = retrieval_metrics(database_descriptors, query_descriptors, positives, evaluation.recalls)
    metrics.update(
        split=split,
        database=len(database),
        queries=len(queries),
        positive_radius_m=evaluation.positive_radius_m,
        positive_yaw_deg=evaluation.positive_yaw_deg,
        descriptor_dim=model.descriptor_dim,
    )
    if output_path:
        write_json(metrics, output_path)
    return metrics
