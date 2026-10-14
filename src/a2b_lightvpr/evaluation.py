from __future__ import annotations

import csv
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpointing import file_sha256, write_json
from .config import EvalConfig
from .data import ImagePathDataset
from .paths import resolve_resource_path


def resolve_database_domain(database, evaluation: EvalConfig, split: str, queries=()):
    """Replace DB image paths only; preserve order, poses and positive protocol.

    A fake manifest explicitly maps original source_path to generated_path. It
    can contain several benchmark splits, but every selected DB source must have
    an unambiguous mapping for this split. Query paths are checked for leakage;
    Query rows are never replaced.
    """
    if evaluation.database_domain == "day":
        return database, ""
    if evaluation.database_domain != "fake_night" or not evaluation.fake_database_manifest:
        raise ValueError("Fake database evaluation requires an explicit mapping manifest")
    manifest = Path(evaluation.fake_database_manifest).expanduser().resolve()
    def canonical(value):
        path = Path(value).expanduser()
        return str(resolve_resource_path(path if path.is_absolute() else manifest.parent / path).resolve())
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"source_path", "generated_path"}.issubset(reader.fieldnames or []):
            raise ValueError("Fake database manifest needs source_path and generated_path")
        mapping = {}
        for row in reader:
            if row.get("split") and row["split"] != split:
                continue
            if not row.get("source_path") or not row.get("generated_path"):
                raise ValueError("Fake database mapping paths cannot be empty")
            source, generated = canonical(row["source_path"]), canonical(row["generated_path"])
            if source in mapping and mapping[source] != generated:
                raise ValueError(f"Ambiguous fake database mapping: {source}")
            if source == generated:
                raise ValueError("Fake database mapping must point to a distinct generated image")
            mapping[source] = generated
    real_paths = {str(Path(row.source_path).expanduser().resolve()) for row in (*database, *queries)}
    output = []
    used_outputs = {}
    for row in database:
        source = str(Path(row.source_path).expanduser().resolve())
        if source not in mapping:
            raise ValueError(f"Fake database mapping does not cover {split} DB source: {source}")
        if not Path(mapping[source]).is_file():
            raise FileNotFoundError(f"Fake database image missing: {mapping[source]}")
        if mapping[source] in real_paths:
            raise ValueError("Fake database output cannot be a real Database or Query image")
        if mapping[source] in used_outputs and used_outputs[mapping[source]] != source:
            raise ValueError("Distinct database sources must map to distinct generated images")
        used_outputs[mapping[source]] = source
        output.append(replace(row, source_path=mapping[source]))
    return output, file_sha256(manifest)


def circular_yaw_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = np.abs(first - second) % 360.0
    return np.minimum(difference, 360.0 - difference)


def positive_indices(
    database: list,
    queries: list,
    radius_m: float,
    yaw_deg: float,
) -> list[np.ndarray]:
    if any(not getattr(row, "ground_truth_available", True) for row in (*database, *queries)):
        raise ValueError("Positive indices require measured ground truth; public test poses are unavailable")
    db_xy = np.asarray([(row.easting, row.northing) for row in database], dtype=np.float64)
    db_yaw = np.asarray([row.yaw_deg for row in database], dtype=np.float64)
    positives = []
    for query in queries:
        distances = np.linalg.norm(db_xy - np.asarray([query.easting, query.northing]), axis=1)
        yaw_difference = circular_yaw_difference(db_yaw, np.asarray(query.yaw_deg))
        positives.append(np.flatnonzero((distances <= radius_m) & (yaw_difference <= yaw_deg)))
    return positives


def search_topk_descriptors(database_descriptors, query_descriptors, top_k: int, *, database_groups=None):
    """Exact inner-product retrieval, also usable when Query ground truth is private."""
    database_descriptors = np.asarray(database_descriptors, dtype=np.float32)
    query_descriptors = np.asarray(query_descriptors, dtype=np.float32)
    if (
        database_descriptors.ndim != 2 or query_descriptors.ndim != 2
        or database_descriptors.shape[1] != query_descriptors.shape[1]
        or not len(database_descriptors) or not len(query_descriptors)
        or database_descriptors.shape[1] == 0
    ):
        raise ValueError("Retrieval needs non-empty, matching descriptor matrices")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not np.isfinite(database_descriptors).all() or not np.isfinite(query_descriptors).all():
        raise ValueError("Retrieval descriptors must be finite")
    maximum = min(top_k, len(database_descriptors))
    grouped_maximum = None
    if database_groups is not None:
        if len(database_groups) != len(database_descriptors):
            raise ValueError("Retrieval groups must match the database rows")
        if any(group is None or group == "" for group in database_groups):
            raise ValueError("Retrieval group IDs cannot be empty")
        try:
            counts = Counter(database_groups)
        except TypeError as error:
            raise ValueError("Retrieval group IDs must be hashable") from error
        grouped_maximum = min(top_k, len(counts))
        # At most max_group_size candidates can belong to one panorama. This
        # guarantees enough unique groups without a full-database rank matrix.
        maximum = min(grouped_maximum * max(counts.values()), len(database_descriptors))
    try:
        import faiss

        index = faiss.IndexFlatIP(database_descriptors.shape[1])
        index.add(database_descriptors)
        scores, predictions = index.search(query_descriptors, maximum)
    except ImportError:
        # Bound memory instead of materializing the entire Query x Database matrix.
        scores = np.empty((len(query_descriptors), maximum), dtype=np.float32)
        predictions = np.empty((len(query_descriptors), maximum), dtype=np.int64)
        for start in range(0, len(query_descriptors), 128):
            similarities = query_descriptors[start:start + 128] @ database_descriptors.T
            candidates = np.argpartition(-similarities, maximum - 1, axis=1)[:, :maximum]
            candidate_scores = np.take_along_axis(similarities, candidates, axis=1)
            order = candidate_scores.argsort(axis=1)[:, ::-1]
            end = start + len(similarities)
            predictions[start:end] = np.take_along_axis(candidates, order, axis=1)
            scores[start:end] = np.take_along_axis(candidate_scores, order, axis=1)
    if grouped_maximum is not None:
        grouped_scores = np.empty((len(query_descriptors), grouped_maximum), dtype=np.float32)
        grouped_predictions = np.empty((len(query_descriptors), grouped_maximum), dtype=np.int64)
        for query_index, (query_scores, query_predictions) in enumerate(zip(scores, predictions)):
            seen = set()
            rank = 0
            for score, prediction in zip(query_scores, query_predictions):
                group = database_groups[int(prediction)]
                if group in seen:
                    continue
                seen.add(group)
                grouped_scores[query_index, rank] = score
                grouped_predictions[query_index, rank] = prediction
                rank += 1
                if rank == grouped_maximum:
                    break
            if rank != grouped_maximum:
                raise RuntimeError("Retrieval candidate budget did not cover the requested unique groups")
        return grouped_scores, grouped_predictions
    return scores, predictions


def retrieval_metrics(
    database_descriptors: np.ndarray,
    query_descriptors: np.ndarray,
    positives: list[np.ndarray],
    recall_values: tuple[int, ...],
    database_groups=None,
) -> dict:
    if database_groups is None:
        _, predictions = search_topk_descriptors(database_descriptors, query_descriptors, max(recall_values))
    else:
        _, predictions = search_topk_descriptors(
            database_descriptors, query_descriptors, max(recall_values), database_groups=database_groups,
        )

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
