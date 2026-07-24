from __future__ import annotations

import csv
import hashlib
import importlib
import json
import resource
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from .checkpointing import file_sha256, write_json
from .config import ExperimentConfig
from .data import benchmark_rows
from .evaluation import extract_descriptors, positive_indices, retrieval_metrics
from .tracking import TrainingTracker
from .transforms import IMAGENET_MEAN, IMAGENET_STD, VPRTransform
from .vpr.model import build_descriptor_model, eigenplaces_checkpoint, load_initial_weights
from .vpr.teachers import build_relation_teacher


@dataclass(frozen=True)
class SuiteRow:
    source_path: str
    easting: float
    northing: float
    yaw_deg: float = 0.0
    condition: str = "night"


def protocol_fingerprint(database, queries, radius: float, yaw: float | None) -> str:
    digest = hashlib.sha256()
    digest.update(f"radius={radius}\0yaw={yaw}\n".encode("utf-8"))
    for role, rows in (("database", database), ("queries", queries)):
        for row in rows:
            digest.update(
                "\0".join(
                    (
                        role,
                        str(getattr(row, "source_path", "")),
                        str(getattr(row, "easting", "")),
                        str(getattr(row, "northing", "")),
                        str(getattr(row, "yaw_deg", "")),
                    )
                ).encode("utf-8")
            )
            digest.update(b"\n")
    return digest.hexdigest()


class FixedResizeTransform:
    def __init__(self, height: int, width: int):
        self.height, self.width = int(height), int(width)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        tensor = TF.to_tensor(TF.resize(
            image.convert("RGB"), [self.height, self.width],
            interpolation=TF.InterpolationMode.BICUBIC, antialias=True,
        ))
        return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


class ReferenceAdapter(torch.nn.Module):
    def __init__(self, model, descriptor_dim: int):
        super().__init__()
        self.model = model
        self.descriptor_dim = int(descriptor_dim)

    def encode(self, images):
        output = self.model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]
        output = output.flatten(1)
        if output.shape[1] != self.descriptor_dim:
            raise RuntimeError(
                f"Reference descriptor mismatch: expected {self.descriptor_dim}, got {output.shape[1]}"
            )
        return F.normalize(output, dim=1)

    def forward(self, images):
        return self.encode(images)


def _registry(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return raw.get("reference_models", {})


def audit_reference_models(path: str | Path, output: str | Path | None = None) -> dict:
    results = {}
    for name, spec in _registry(path).items():
        missing = []
        for field in ("repo", "checkpoint", "database_checkpoint"):
            value = spec.get(field, "")
            if value and not Path(value).expanduser().exists():
                missing.append({"field": field, "path": value})
        for value in spec.get("required_files", []):
            if not Path(value).expanduser().exists():
                missing.append({"field": "required_files", "path": value})
        supported = bool(spec.get("supported", True))
        results[name] = {
            "ready": supported and not missing,
            "supported": supported,
            "missing": missing,
            "note": spec.get("note", ""),
            "download_url": spec.get("download_url", ""),
            "loader": spec.get("loader", ""),
            "descriptor_dim": spec.get("descriptor_dim"),
        }
    payload = {
        "registry": str(Path(path).resolve()),
        "ready": all(value["ready"] for value in results.values()),
        "models": results,
    }
    if output:
        write_json(payload, output)
    return payload


def _strict_state(model, checkpoint: str | Path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"Strict checkpoint mismatch: {result}")


@contextmanager
def _isolated_repo_import(repo: str | Path, module_prefixes: tuple[str, ...]):
    """Import one reference repository without leaking common module names."""
    repo = str(Path(repo).expanduser().resolve())
    saved = {
        name: module for name, module in tuple(sys.modules.items())
        if name in module_prefixes or name.startswith(tuple(f"{p}." for p in module_prefixes))
    }
    for name in saved:
        sys.modules.pop(name, None)
    sys.path.insert(0, repo)
    original_download = torch.hub.download_url_to_file
    torch.hub.download_url_to_file = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("Reference evaluation is offline; download the file listed by the audit first")
    )
    try:
        yield
    finally:
        torch.hub.download_url_to_file = original_download
        sys.path.remove(repo)
        for name in tuple(sys.modules):
            if name in module_prefixes or name.startswith(tuple(f"{p}." for p in module_prefixes)):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


def build_reference_model(spec: dict) -> ReferenceAdapter:
    loader = spec["loader"]
    descriptor_dim = int(spec["descriptor_dim"])
    if loader == "a2b_gem":
        model = build_descriptor_model(
            spec["backbone"], "gem", image_height=spec["native_size"][0],
            image_width=spec["native_size"][1], pretrained=False,
        )
        result = load_initial_weights(model, spec["checkpoint"], "full")
        if result["missing"] or result["unexpected"]:
            raise ValueError(f"Reference checkpoint mismatch: {result}")
        return ReferenceAdapter(model, descriptor_dim)
    if loader == "eigenplaces_hub":
        checkpoint = eigenplaces_checkpoint(descriptor_dim)
        model = build_descriptor_model(
            "resnet50", "gem", image_height=spec["native_size"][0],
            image_width=spec["native_size"][1], descriptor_dim=descriptor_dim,
            pretrained=False, init_checkpoint=str(checkpoint), init_checkpoint_scope="full",
            model_family="eigenplaces",
        )
        spec["_resolved_checkpoint"] = str(checkpoint)
        return ReferenceAdapter(model, descriptor_dim)
    if loader == "nocplace_r50":
        with _isolated_repo_import(spec["repo"], ("nocplace_models",)):
            module = importlib.import_module("nocplace_models.cosplace_network")
            model = module.GeoLocalizationNet("ResNet50", descriptor_dim, train_all_layers=True)
            _strict_state(model, spec["checkpoint"])
        return ReferenceAdapter(model, descriptor_dim)
    if loader == "nocplace_factory":
        with _isolated_repo_import(spec["repo"], ("nocplace_models",)):
            factory = importlib.import_module("nocplace_models.model_factory")
            model = factory.build_vpr_model(spec["backbone"], descriptor_dim)
            factory.load_checkpoint(model, spec["checkpoint"], "cpu", strict=True)
        return ReferenceAdapter(model, descriptor_dim)
    if loader == "relation_teacher":
        model = build_relation_teacher(
            spec["teacher"], spec["dinov2_repo"], spec["checkpoint"]
        )
        return ReferenceAdapter(model, descriptor_dim)
    if loader == "convap_local":
        with _isolated_repo_import(spec["repo"], ("vpr_models",)):
            module = importlib.import_module("vpr_models.convap")
            model = module.ConvAPModel({"in_channels": 2048, "out_channels": 128})
            _strict_state(model, spec["checkpoint"])
        return ReferenceAdapter(model, descriptor_dim)
    if loader == "cricavpr":
        repo = Path(spec["repo"]).expanduser().resolve()
        with _isolated_repo_import(repo, ("network", "backbone")):
            module = importlib.import_module("network")
            model = torch.nn.DataParallel(module.CricaVPRNet())
        state = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state)
        model.load_state_dict(state, strict=True)
        return ReferenceAdapter(model, descriptor_dim)
    if loader == "vpr_methods":
        with _isolated_repo_import(spec["repo"], ("vpr_models",)):
            module = importlib.import_module("vpr_models")
            model = module.get_model(
                spec["method"], spec.get("backbone"), descriptor_dim
            )
        return ReferenceAdapter(model, descriptor_dim)
    raise ValueError(f"Unknown reference loader: {loader}")


def _svox_rows(root: Path):
    base = root / "images" / "test"
    outputs = []
    for folder in ("gallery", "queries_night"):
        rows = []
        for path in sorted((base / folder).glob("*")):
            parts = path.name.split("@")
            if len(parts) >= 4:
                rows.append(SuiteRow(str(path.resolve()), float(parts[1]), float(parts[2])))
        outputs.append(rows)
    return outputs[0], outputs[1]


def _msls_rows(root: Path):
    database, queries = [], []
    for city in ("cph", "sf"):
        for role in ("database", "query"):
            table = root / "train_val" / city / role / "postprocessed.csv"
            with table.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    night = row.get("night", "False").lower() == "true"
                    path = table.parent / "images" / f'{row["key"]}.jpg'
                    item = SuiteRow(str(path.resolve()), float(row["easting"]), float(row["northing"]))
                    if role == "database" and not night:
                        database.append(item)
                    elif role == "query" and night:
                        queries.append(item)
    return database, queries


def _tokyo_rows(root: Path, condition: str):
    database = []
    for path in sorted((root / "database_gsv_vga").glob("*/*/*.png")):
        easting, northing = path.parent.name.split("_")[:2]
        database.append(SuiteRow(str(path.resolve()), float(easting), float(northing)))
    condition_code = {"tokyo_night": "N", "tokyo_sunset": "S"}[condition]
    queries = []
    for metadata in sorted((root / "queries").glob("*.csv")):
        with metadata.open(newline="", encoding="utf-8") as handle:
            row = next(csv.reader(handle))
        if len(row) >= 9 and row[5] == condition_code:
            queries.append(SuiteRow(
                str((root / "queries" / row[0]).resolve()), float(row[7]), float(row[8]),
                condition=condition,
            ))
    return database, queries


def suite_rows(config: ExperimentConfig, dataset: str, split: str = "test"):
    if dataset == "robotcar":
        return benchmark_rows(config.data.robotcar_manifest, split)
    if dataset == "svox_night":
        return _svox_rows(Path(config.data.svox_root))
    if dataset == "msls_night":
        return _msls_rows(Path(config.data.msls_root))
    if dataset in {"tokyo_night", "tokyo_sunset"}:
        return _tokyo_rows(Path(config.data.tokyo_root), dataset)
    raise ValueError(f"Unsupported suite dataset: {dataset}")


def evaluate_dual_encoder(
    database_model, query_model, database, queries, transform, radius: float,
    yaw: float | None, recalls: tuple[int, ...], device, batch_size: int, workers: int,
):
    database_descriptors = extract_descriptors(database_model, database, transform, device, batch_size, workers)
    query_descriptors = extract_descriptors(query_model, queries, transform, device, batch_size, workers)
    if yaw is None:
        db_xy = np.asarray([(row.easting, row.northing) for row in database])
        positives = [
            np.flatnonzero(np.linalg.norm(db_xy - (row.easting, row.northing), axis=1) <= radius)
            for row in queries
        ]
    else:
        positives = positive_indices(database, queries, radius, yaw)
    return retrieval_metrics(database_descriptors, query_descriptors, positives, recalls)


def evaluate_vpr_suite(
    config: ExperimentConfig, registry_path: str | Path, *, model_name: str,
    dataset: str, transform_name: str = "native", protocol: str = "paper",
    split: str = "test", output: str | Path | None = None,
) -> Path:
    registry = _registry(registry_path)
    if model_name not in registry:
        raise ValueError(f"Unknown reference model: {model_name}")
    audit = audit_reference_models(registry_path)
    if not audit["models"][model_name]["ready"]:
        raise FileNotFoundError(json.dumps(audit["models"][model_name], indent=2))
    spec = registry[model_name]
    query_model = build_reference_model(spec)
    use_original_database = (
        config.vpr.database_encoder == "original" and bool(spec.get("database_checkpoint"))
    )
    database_model = (
        build_reference_model({**spec, "checkpoint": spec["database_checkpoint"]})
        if use_original_database else query_model
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_model.eval().to(device)
    database_model.eval().to(device)
    database, queries = suite_rows(config, dataset, split)
    if transform_name == "native":
        height, width = spec["native_size"]
        transform = FixedResizeTransform(height, width)
    else:
        height, width = 320, 480
        transform = VPRTransform(height, width)
    radius = 10.0 if protocol == "project" else 25.0
    yaw = 30.0 if protocol == "project" and dataset == "robotcar" else None
    batch_size = int(spec.get("native_batch_size", 1)) if transform_name == "native" else 16
    metrics = evaluate_dual_encoder(
        database_model, query_model, database, queries, transform, radius, yaw,
        config.evaluation.recalls, device, batch_size,
        config.vpr.workers,
    )
    sample = torch.randn(1, 3, height, width, device=device)
    timing = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(10):
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                query_model.encode(sample)
        for _ in range(int(spec.get("benchmark_iterations", 100))):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                query_model.encode(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timing.append((time.perf_counter() - start) * 1000.0)
    metrics.update({
        "model": model_name, "dataset": dataset, "split": split,
        "database_encoder": "original" if use_original_database else "adapted",
        "protocol": protocol, "positive_radius_m": radius, "positive_yaw_deg": yaw,
        "transform": transform_name, "input_size": [height, width],
        "descriptor_dim": query_model.descriptor_dim,
        "parameters": sum(parameter.numel() for parameter in query_model.parameters()),
        "batch1_p50_ms": float(np.percentile(timing, 50)),
        "batch1_p95_ms": float(np.percentile(timing, 95)),
        "benchmark_precision": "fp16" if device.type == "cuda" else "fp32",
        "peak_cuda_mib": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda" else None
        ),
        "process_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "query_checkpoint_sha256": (
            file_sha256(spec.get("_resolved_checkpoint", spec.get("checkpoint")))
            if spec.get("_resolved_checkpoint") or spec.get("checkpoint") else ""
        ),
        "database_checkpoint_sha256": (
            file_sha256(spec["database_checkpoint"]) if spec.get("database_checkpoint") else ""
        ),
    })
    output = Path(output or Path("runs/reference") / model_name / dataset / protocol / transform_name / "metrics.json")
    write_json(metrics, output)
    tracker = TrainingTracker(config, job_type="evaluate-reference-vpr",
        default_name=f"{model_name}-{dataset}-{protocol}-{transform_name}",
        output_dir=output.parent,
        run_key=f"{model_name}:{dataset}:{protocol}:{transform_name}:{metrics['database_encoder']}")
    tracker.summary({
        "reference/model": model_name, "reference/dataset": dataset,
        "reference/recall_at_1": metrics["recalls"]["1"],
        "reference/recall_at_5": metrics["recalls"].get("5"),
        "reference/recall_at_10": metrics["recalls"].get("10"),
        "reference/mrr": metrics["mrr"],
    })
    tracker.finish()
    return output
