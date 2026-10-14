from __future__ import annotations

import csv
import hashlib
import importlib
import json
import resource
import os
import platform
import sys
import time
import urllib.request
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms import functional as TF

from ..checkpointing import file_sha256, write_json
from ..config import ExperimentConfig
from ..evaluation import extract_descriptors, positive_indices, retrieval_metrics, search_topk_descriptors
from ..transforms import IMAGENET_MEAN, IMAGENET_STD, VPRTransform
from ..vpr.model import build_descriptor_model, eigenplaces_checkpoint
from ..vpr.anyloc import AnyLocSmallModel
from ..vpr.boq import BoQReferenceModel, BoQSmallModel, PCAProjectedReference
from ..vpr.salad import SALADStudentModel
from ..vpr.teachers import build_relation_teacher


@dataclass(frozen=True)
class SuiteRow:
    source_path: str
    easting: float
    northing: float
    yaw_deg: float = 0.0
    condition: str = "night"


def query_protocol_fingerprint(
    database, queries, radius: float | None, yaw: float | None
) -> str:
    """Hash retrieval rows and the positive-set definition, but not preprocessing."""
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
                        str(getattr(row, "ground_truth_available", True)),
                    )
                ).encode("utf-8")
            )
            digest.update(b"\n")
    return digest.hexdigest()


def evaluation_fingerprint(
    query_fingerprint: str,
    *,
    transform: str,
    height: int,
    width: int,
    interpolation: str = "bicubic",
    padding: str = "none",
    normalization: str = "imagenet",
) -> str:
    """Hash all conditions that can change descriptors for a fixed query protocol."""
    payload = {
        "query_protocol_fingerprint": str(query_fingerprint),
        "transform": str(transform),
        "input_size": [int(height), int(width)],
        "interpolation": str(interpolation),
        "padding": str(padding),
        "normalization": str(normalization),
    }
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# Compatibility for historical callers and JSON migration. New code should use the
# explicit query/evaluation names above.
protocol_fingerprint = query_protocol_fingerprint


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
        for field in (
            "repo", "aggregator_repo", "checkpoint", "database_checkpoint",
            "backbone_checkpoint", "teacher_checkpoint", "vocabulary",
            "projection_checkpoint", "fit_root",
        ):
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
            "backbone": spec.get("backbone", ""),
            "official_checkpoint": bool(spec.get("official_checkpoint", True)),
            "training_status": spec.get("training_status", "official"),
            "required_artifacts": {
                str(Path(path).expanduser().resolve()): {
                    "bytes": Path(path).expanduser().stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in spec.get("required_files", [])
                if Path(path).expanduser().is_file()
            },
        }
    payload = {
        "registry": str(Path(path).resolve()),
        "ready": all(value["ready"] for value in results.values()),
        "models": results,
    }
    if output:
        write_json(payload, output)
    return payload


def _download_with_resume(url: str, destination: Path, chunk_size: int = 8 * 1024 * 1024) -> None:
    """Download to a partial file, resuming only when the server honors Range."""
    existing = destination.stat().st_size if destination.is_file() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", response.getcode())
        append = existing > 0 and status == 206
        if append:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {existing}-"):
                raise RuntimeError(
                    f"Server returned an invalid resume range: {content_range!r}"
                )
        else:
            existing = 0

        content_length = response.headers.get("Content-Length")
        total = existing + int(content_length) if content_length else None
        downloaded = existing
        next_report = downloaded + 64 * 1024 * 1024
        mode = "ab" if append else "wb"
        with destination.open(mode) as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    if total:
                        print(
                            f"Downloaded {downloaded / 1024**2:.1f}/{total / 1024**2:.1f} MiB "
                            f"({100.0 * downloaded / total:.1f}%)",
                            flush=True,
                        )
                    else:
                        print(f"Downloaded {downloaded / 1024**2:.1f} MiB", flush=True)
                    next_report = downloaded + 64 * 1024 * 1024
        if total is not None and downloaded != total:
            raise RuntimeError(
                f"Incomplete checkpoint download: expected {total} bytes, got {downloaded}"
            )


def fetch_reference_model(path: str | Path, model_name: str) -> dict:
    registry = _registry(path)
    if model_name not in registry:
        raise ValueError(f"Unknown reference model: {model_name}")
    spec = registry[model_name]
    url = str(spec.get("download_url", ""))
    checkpoint_value = str(spec.get("checkpoint", ""))
    if not url or not checkpoint_value:
        raise ValueError(f"{model_name} does not declare a downloadable checkpoint")
    checkpoint = Path(checkpoint_value).expanduser()
    allowed = {
        "huggingface.co": ("/TobiasRobotics/Patch-NetVLAD/resolve/main/",),
        "github.com": ("/gmberton/EigenPlaces/releases/download/v1.0/",),
    }
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc not in allowed or not any(
        parsed.path.startswith(prefix) for prefix in allowed[parsed.netloc]
    ):
        raise ValueError(f"Refusing unapproved reference-model URL: {url}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        temporary = checkpoint.with_suffix(checkpoint.suffix + ".part")
        _download_with_resume(url, temporary)
        if temporary.stat().st_size == 0:
            raise RuntimeError("Downloaded checkpoint is empty")
        os.replace(temporary, checkpoint)
    metadata = {
        "model": model_name,
        "url": url,
        "file": str(checkpoint.resolve()),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": file_sha256(checkpoint),
    }
    write_json(metadata, checkpoint.with_suffix(checkpoint.suffix + ".json"))
    return metadata


def _strict_state(model, checkpoint: str | Path):
    # NumPy 1.26 has a compatibility _core package that may not export
    # multiarray until its submodule is imported; older NumPy only has core.
    try:
        numpy_multiarray = importlib.import_module("numpy._core.multiarray")
    except ModuleNotFoundError:
        numpy_multiarray = importlib.import_module("numpy.core.multiarray")
    safe_numpy_globals = (
        numpy_multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    )
    with torch.serialization.safe_globals(safe_numpy_globals):
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


def build_reference_model(spec: dict) -> torch.nn.Module:
    loader = spec["loader"]
    descriptor_dim = int(spec["descriptor_dim"])
    if loader == "cricavpr_small":
        from ..vpr.crica import build_crica_small

        model = build_crica_small(spec["repo"])
        _strict_state(model, spec["checkpoint"])
        return ReferenceAdapter(model, descriptor_dim)
    if loader == "boq_reference":
        return BoQReferenceModel(
            spec["repo"], spec["checkpoint"],
            backbone_name=spec.get("backbone_name", "dinov2_vitb14"),
            descriptor_dim=descriptor_dim,
            backbone_checkpoint=spec.get("backbone_checkpoint", ""),
        )
    if loader == "boq_small":
        model = BoQSmallModel(
            spec["repo"], spec["backbone_name"], spec["backbone_checkpoint"],
            int(spec.get("trainable_blocks", 2)),
            spec.get("aggregator_repo", ""),
        )
        _strict_state(model, spec["checkpoint"])
        return model
    if loader == "anyloc_small":
        return AnyLocSmallModel(
            spec["repo"], spec["backbone_name"], spec["checkpoint"],
            spec["vocabulary"], layer=int(spec.get("layer", 11)),
            facet=spec.get("facet", "token"),
            num_clusters=int(spec.get("num_clusters", 32)),
        )
    if loader == "salad_student":
        model = SALADStudentModel(
            spec["repo"], spec["backbone_name"], spec["backbone_checkpoint"],
            int(spec.get("trainable_blocks", 2)),
        )
        _strict_state(model, spec["checkpoint"])
        return model
    if loader == "boq_pca":
        source_spec = dict(spec)
        source_spec["loader"] = spec.get("source_loader", "boq_reference")
        source_spec["descriptor_dim"] = int(spec["source_descriptor_dim"])
        base = build_reference_model(source_spec)
        return PCAProjectedReference(
            base, spec["projection_checkpoint"], descriptor_dim
        )
    if loader == "official_mixvpr":
        model = build_descriptor_model(
            "mixvpr",
            image_height=int(spec["native_size"][0]),
            image_width=int(spec["native_size"][1]),
            descriptor_dim=descriptor_dim,
            pretrained=False,
            init_checkpoint=spec["checkpoint"],
            model_family="mixvpr",
        )
        return ReferenceAdapter(model, descriptor_dim)
    if loader == "eigenplaces_hub":
        checkpoint = eigenplaces_checkpoint(descriptor_dim)
        model = build_descriptor_model(
            "gem", image_height=spec["native_size"][0],
            image_width=spec["native_size"][1], descriptor_dim=descriptor_dim,
            pretrained=False, init_checkpoint=str(checkpoint),
            model_family="eigenplaces",
        )
        spec["_resolved_checkpoint"] = str(checkpoint)
        return ReferenceAdapter(model, descriptor_dim)
    if loader == "cosplace_checkpoint":
        model = build_descriptor_model(
            "gem", image_height=320, image_width=320, descriptor_dim=descriptor_dim,
            pretrained=False, init_checkpoint=spec["checkpoint"], model_family="cosplace",
        )
        return ReferenceAdapter(model, descriptor_dim)
    if loader == "netvlad_pytorch":
        model = build_descriptor_model(
            "netvlad", image_height=spec["native_size"][0],
            image_width=spec["native_size"][1], descriptor_dim=descriptor_dim,
            pretrained=False, init_checkpoint=spec["checkpoint"],
            model_family="netvlad", num_clusters=int(spec.get("num_clusters", 64)),
        )
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


def _reference_checkpoint(spec: dict) -> Path:
    value = spec.get("_resolved_checkpoint") or spec.get("checkpoint")
    if not value:
        raise ValueError("Reference model did not resolve a checkpoint path")
    return Path(value).expanduser().resolve()


def benchmark_reference_models(
    config: ExperimentConfig,
    registry_path: str | Path,
    model_names: tuple[str, ...] = ("all",),
    *,
    warmup: int = 20,
    iterations: int = 100,
    output: str | Path = "runs/reference/resource_benchmark.json",
) -> Path:
    registry = _registry(registry_path)
    selected = tuple(registry) if "all" in model_names else model_names
    unknown = sorted(set(selected) - set(registry))
    if unknown:
        raise ValueError(f"Unknown reference models: {unknown}")
    audit = audit_reference_models(registry_path)
    unavailable = {name: audit["models"][name] for name in selected if not audit["models"][name]["ready"]}
    if unavailable:
        raise FileNotFoundError(json.dumps(unavailable, indent=2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for name in selected:
        spec = registry[name]
        model = build_reference_model(spec).eval().to(device)
        height, width = (int(item) for item in spec["native_size"])
        sample = torch.randn(1, 3, height, width, device=device)
        use_amp = device.type == "cuda"
        times = []
        with torch.inference_mode():
            for _ in range(warmup):
                with torch.autocast(device.type, enabled=use_amp):
                    model.encode(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            for _ in range(iterations):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                tick = time.perf_counter()
                with torch.autocast(device.type, enabled=use_amp):
                    model.encode(sample)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                times.append((time.perf_counter() - tick) * 1000.0)
            elapsed = time.perf_counter() - started
            batch_dependence = None
            if spec.get("batch_dependent"):
                with torch.autocast(device.type, enabled=use_amp):
                    alone = model.encode(sample)
                    mixed = model.encode(torch.cat((sample, torch.flip(sample, (-1,))), dim=0))[:1]
                batch_dependence = {
                    "cosine": float(F.cosine_similarity(alone.float(), mixed.float()).item()),
                    "max_abs": float((alone.float() - mixed.float()).abs().max().item()),
                }
        descriptor_dim = int(spec["descriptor_dim"])
        parameters = sum(parameter.numel() for parameter in model.parameters())
        checkpoint = _reference_checkpoint(spec)
        database_memory = {}
        for count in (1_000, 100_000, 1_000_000):
            database_memory[str(count)] = {
                "fp16_gib": count * descriptor_dim * 2 / 1024**3,
                "fp32_gib": count * descriptor_dim * 4 / 1024**3,
            }
        results[name] = {
            "descriptor_dim": descriptor_dim,
            "input_size": [height, width],
            "parameters": parameters,
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": file_sha256(checkpoint),
            "batch1_p50_ms": float(np.percentile(times, 50)),
            "batch1_p95_ms": float(np.percentile(times, 95)),
            "batch1_images_per_second": float(iterations / max(elapsed, 1e-9)),
            "peak_cuda_mib": (
                float(torch.cuda.max_memory_allocated(device) / 1024**2)
                if device.type == "cuda" else None
            ),
            "process_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
            "database_descriptor_memory": database_memory,
            "batch_dependence": batch_dependence,
            "backbone": spec.get("backbone", ""),
            "official_checkpoint": bool(spec.get("official_checkpoint", True)),
            "training_status": spec.get("training_status", "official"),
            "required_artifacts": {
                str(Path(path).expanduser().resolve()): {
                    "bytes": Path(path).expanduser().stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in spec.get("required_files", [])
                if Path(path).expanduser().is_file()
            },
        }
        if isinstance(model, PCAProjectedReference):
            projection_sample = torch.randn(
                1, int(model.model.descriptor_dim), device=device
            )
            projection_times = []
            with torch.inference_mode():
                for _ in range(warmup):
                    model.project(projection_sample)
                for _ in range(iterations):
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    tick = time.perf_counter()
                    model.project(projection_sample)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    projection_times.append((time.perf_counter() - tick) * 1000.0)
            results[name]["projection_p50_ms"] = float(np.percentile(projection_times, 50))
            results[name]["projection_p95_ms"] = float(np.percentile(projection_times, 95))
            encoder_times = []
            with torch.inference_mode():
                for _ in range(warmup):
                    with torch.autocast(device.type, enabled=use_amp):
                        model.model.encode(sample)
                for _ in range(iterations):
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    tick = time.perf_counter()
                    with torch.autocast(device.type, enabled=use_amp):
                        model.model.encode(sample)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    encoder_times.append((time.perf_counter() - tick) * 1000.0)
            results[name]["encoder_only_p50_ms"] = float(np.percentile(encoder_times, 50))
            results[name]["encoder_only_p95_ms"] = float(np.percentile(encoder_times, 95))
        if spec.get("projection_checkpoint"):
            projection = Path(spec["projection_checkpoint"]).expanduser().resolve()
            results[name]["projection_checkpoint"] = str(projection)
            results[name]["projection_checkpoint_bytes"] = projection.stat().st_size
            results[name]["projection_checkpoint_sha256"] = file_sha256(projection)
        del model, sample
        if device.type == "cuda":
            torch.cuda.empty_cache()
    baseline = results.get("mixvpr-r50-512")
    if baseline is not None:
        for metrics in results.values():
            metrics["relative_to_mixvpr_r50_512"] = {
                "mixvpr_speedup_x": metrics["batch1_p95_ms"] / baseline["batch1_p95_ms"],
                "descriptor_storage_saved_fraction": 1.0 - 512.0 / metrics["descriptor_dim"],
                "peak_cuda_saved_fraction": (
                    1.0 - baseline["peak_cuda_mib"] / metrics["peak_cuda_mib"]
                    if metrics["peak_cuda_mib"] else None
                ),
            }
    payload = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "precision": "fp16" if device.type == "cuda" else "fp32",
        "warmup": int(warmup),
        "iterations": int(iterations),
        "models": results,
    }
    output = Path(output)
    write_json(payload, output)
    return output


def validate_reference_resources(
    registry_path: str | Path,
    resources_path: str | Path,
    model_names: tuple[str, ...],
) -> bool:
    registry = _registry(registry_path)
    resources_file = Path(resources_path)
    if not resources_file.is_file():
        return False
    stored = json.loads(resources_file.read_text(encoding="utf-8")).get("models", {})
    for name in model_names:
        if name not in registry or name not in stored:
            return False
        spec, result = registry[name], stored[name]
        checkpoint = Path(spec.get("checkpoint", "")).expanduser()
        if not checkpoint.is_file() or result.get("checkpoint_sha256") != file_sha256(checkpoint):
            return False
        projection_value = spec.get("projection_checkpoint")
        if projection_value:
            projection = Path(projection_value).expanduser()
            if (
                not projection.is_file()
                or result.get("projection_checkpoint_sha256") != file_sha256(projection)
            ):
                return False
        artifacts = result.get("required_artifacts", {})
        for value in spec.get("required_files", []):
            path = Path(value).expanduser()
            resolved = str(path.resolve())
            if (
                not path.is_file()
                or resolved not in artifacts
                or artifacts[resolved].get("sha256") != file_sha256(path)
            ):
                return False
    return True


def _svox_rows(root: Path, split: str = "test", condition: str = "night"):
    base = root / "images" / split
    query_folder = {
        "night": "queries_night",
        "day": "queries",
        "sun": "queries_sun",
    }[condition]
    outputs = []
    for folder in ("gallery", query_folder):
        rows = []
        for path in sorted((base / folder).glob("*")):
            parts = path.name.split("@")
            if len(parts) >= 4:
                rows.append(SuiteRow(str(path.resolve()), float(parts[1]), float(parts[2])))
        outputs.append(rows)
    return outputs[0], outputs[1]


def _msls_rows(root: Path, condition: str = "night"):
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
                    elif role == "query" and night == (condition == "night"):
                        queries.append(item)
    return database, queries


def _tokyo_rows(root: Path, condition: str):
    database = []
    for path in sorted((root / "database_gsv_vga").glob("*/*/*.png")):
        easting, northing = path.parent.name.split("_")[:2]
        database.append(SuiteRow(str(path.resolve()), float(easting), float(northing)))
    condition_code = {
        "tokyo_night": "N", "tokyo_sunset": "S", "tokyo_day": "D"
    }[condition]
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
    if dataset in {"robotcar_seasons_night", "robotcar_seasons_day"}:
        if split != "test":
            raise ValueError("RobotCar Seasons public test cannot be used for training validation")
        from ..robotcar_seasons import DAY_CONDITIONS, NIGHT_CONDITIONS, robotcar_seasons_rows
        conditions = NIGHT_CONDITIONS if dataset.endswith("night") else DAY_CONDITIONS
        return robotcar_seasons_rows(config.data.robotcar_seasons_root, conditions=conditions, protocol="v2")
    if dataset == "svox_night":
        return _svox_rows(Path(config.data.svox_root), split, "night")
    if dataset == "svox_day_val":
        return _svox_rows(Path(config.data.svox_root), "val", "day")
    if dataset == "svox_day":
        return _svox_rows(Path(config.data.svox_root), "test", "day")
    if dataset == "svox_sun":
        return _svox_rows(Path(config.data.svox_root), "test", "sun")
    if dataset == "msls_night":
        return _msls_rows(Path(config.data.msls_root), "night")
    if dataset == "msls_day_val":
        return _msls_rows(Path(config.data.msls_root), "day")
    if dataset in {"tokyo_night", "tokyo_sunset", "tokyo_day"}:
        return _tokyo_rows(Path(config.data.tokyo_root), dataset)
    raise ValueError(f"Unsupported suite dataset: {dataset}")


def suite_protocol_metadata(config, dataset):
    if dataset in {"robotcar_seasons_night", "robotcar_seasons_day"}:
        from ..robotcar_seasons import V2_README, _metadata_hashes
        return {
            "public_protocol": "robotcar_seasons_v2",
            "camera": "rear",
            "official_readme": V2_README,
            "split_metadata_sha256": _metadata_hashes(Path(config.data.robotcar_seasons_root), "v2"),
            "ground_truth_status": "official_test_query_poses_not_public",
        }
    return {}


def evaluate_dual_encoder(
    database_model, query_model, database, queries, transform, radius: float | None,
    yaw: float | None, recalls: tuple[int, ...], device, batch_size: int, workers: int,
):
    database_descriptors = extract_descriptors(database_model, database, transform, device, batch_size, workers)
    query_descriptors = extract_descriptors(query_model, queries, transform, device, batch_size, workers)
    if any(not getattr(row, "ground_truth_available", True) for row in (*database, *queries)):
        scores, indices = search_topk_descriptors(database_descriptors, query_descriptors, max(recalls))
        predictions = []
        for query, query_scores, query_indices in zip(queries, scores, indices):
            predictions.append({
                "query_id": getattr(query, "image_id", query.source_path),
                "query_path": query.source_path,
                "condition": getattr(query, "condition", ""),
                "camera": getattr(query, "camera", ""),
                "matches": [{"database_id": getattr(database[int(index)], "image_id", database[int(index)].source_path),
                             "database_path": database[int(index)].source_path,
                             "score": float(score)} for score, index in zip(query_scores, query_indices)],
            })
        return {"recalls": None, "mrr": None, "valid_queries": None,
                "total_queries": len(queries), "ground_truth_available": False,
                "evaluation_status": "retrieval_only_no_public_test_gt",
                "retrieval_top_k": int(indices.shape[1]), "retrieval_predictions": predictions}
    if radius is None:
        raise ValueError("Measured Recall requires an explicit positive radius")
    if yaw is None:
        db_xy = np.asarray([(row.easting, row.northing) for row in database])
        positives = [
            np.flatnonzero(np.linalg.norm(db_xy - (row.easting, row.northing), axis=1) <= radius)
            for row in queries
        ]
    else:
        positives = positive_indices(database, queries, radius, yaw)
    metrics = retrieval_metrics(database_descriptors, query_descriptors, positives, recalls)
    metrics.update(ground_truth_available=True, evaluation_status="measured")
    return metrics
