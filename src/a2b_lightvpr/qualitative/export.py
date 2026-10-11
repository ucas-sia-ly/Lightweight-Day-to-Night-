from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ..checkpointing import file_sha256, write_json
from ..config import ExperimentConfig
from ..evaluation import extract_descriptors, positive_indices
from ..references.models import (
    FixedResizeTransform,
    ReferenceAdapter,
    _isolated_repo_import,
    _strict_state,
    query_protocol_fingerprint,
    suite_rows,
)
from ..runtime import load_vpr_model
from ..transforms import VPRTransform
from ..vpr.model import build_descriptor_model


@dataclass(frozen=True)
class QualitativeDataset:
    protocol: str
    dataset: str
    split: str
    radius_m: float
    yaw_deg: float | None


DATASETS = {
    "robotcar_project": QualitativeDataset("robotcar_project", "robotcar", "test", 10.0, 30.0),
    "svox_night": QualitativeDataset("svox_night", "svox_night", "test", 25.0, None),
    "msls_night": QualitativeDataset("msls_night", "msls_night", "val", 25.0, None),
    "tokyo_night": QualitativeDataset("tokyo_night", "tokyo_night", "test", 25.0, None),
}


MODEL_LABELS = {
    "baseline": "Day+fake top-1",
    "cdnd": "CDND top-1",
    "cosplace": "CosPlace top-1",
    "eigenplaces": "EigenPlaces top-1",
    "official_mixvpr512": "Official MixVPR-512 top-1",
    "anyloc_pca": "AnyLoc-PCA top-1",
    "crica_pca": "CricaVPR-PCA top-1",
    "salad": "SALAD top-1",
    "boq_pca": "BoQ-PCA top-1",
    "nocplace": "NocPlace top-1",
}


def _local_gem_model(checkpoint: Path) -> ReferenceAdapter:
    repo = Path("/home/admin123/diffusion_vpr_test/reference_code/CosPlace")
    with _isolated_repo_import(repo, ("cosplace_model",)):
        import torchvision
        from cosplace_model.cosplace_network import GeoLocalizationNet
        import cosplace_model.cosplace_network as network

        network.get_pretrained_torchvision_model = lambda name: torchvision.models.resnet50(
            weights=None
        )
        model = GeoLocalizationNet("ResNet50", 512, train_all_layers=True)
        _strict_state(model, checkpoint)
    return ReferenceAdapter(model, 512)


def _models(config: ExperimentConfig, selected_models: tuple[str, ...]):
    from ..references.models import _registry, build_reference_model

    registry = _registry("configs/references/day_completion.yaml")
    mapping = {
        "cosplace": "cosplace-r50-512-320",
        "eigenplaces": "eigenplaces-r50-512-320",
        "anyloc_pca": "anyloc-dinov2-s-gsv-pca512",
        "crica_pca": "cricavpr-dinov2-s-gsv322-pca512",
        "salad": "salad-dinov2-s-544-gsv",
        "boq_pca": "boq-dinov2-s-pca512",
    }
    unknown = set(selected_models) - set(MODEL_LABELS)
    if unknown:
        raise ValueError(f"Unsupported qualitative models: {unknown}")
    for name in selected_models:
        spec = dict(registry[mapping[name]]) if name in mapping else {}
        if name in ("baseline", "cdnd"):
            run = "day_fake_pk_seed0" if name == "baseline" else "salad_no_gate_seed0"
            checkpoint = Path("runs/vpr/cdnd_512") / run / "model_best.pt"
            model = load_vpr_model(config, checkpoint)[0]
            transform = VPRTransform(320, 320)
        elif name == "nocplace":
            checkpoint = Path("runs/reference/nocplace_protocol_clean5k/best_model.pth")
            # Same R50-GeM-FC architecture and strict state keys as CosPlace;
            # avoids the nonexistent NocPlace model_factory and network downloads.
            model = _local_gem_model(checkpoint)
            transform = VPRTransform(320, 320)
        elif name == "official_mixvpr512":
            checkpoint = Path("/home/admin123/diffusion_vpr_test/CycleGAN-turbo_NetVLAD/checkpoints/resnet50_MixVPR_512_channels(256)_rows(2).ckpt")
            model = build_descriptor_model(
                "mixvpr", image_height=320, image_width=320, descriptor_dim=512,
                pretrained=False, init_checkpoint=str(checkpoint), model_family="mixvpr",
            )
            transform = FixedResizeTransform(320, 320)
        else:
            checkpoint = Path(spec["checkpoint"])
            if name == "eigenplaces":
                # Load the explicitly selected local weights, not a second hub copy.
                model = _local_gem_model(checkpoint)
            else:
                model = build_reference_model(spec)
            transform = FixedResizeTransform(*spec["native_size"])
        artifacts = [str(checkpoint)] + [spec[k] for k in
            ("projection_checkpoint", "vocabulary", "backbone_checkpoint") if spec.get(k)]
        model.qualitative_artifacts = {str(Path(p).resolve()): file_sha256(p) for p in artifacts}
        model.qualitative_batch_size = 1 if name == "crica_pca" else None
        model.qualitative_spec = spec
        yield name, model, transform, checkpoint


def _cache_descriptors(
    model,
    rows,
    transform,
    device,
    batch_size: int,
    workers: int,
    path: Path,
    identity: dict,
) -> np.ndarray:
    metadata_path = path.with_suffix(".json")
    if path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata == identity:
            descriptors = np.load(path, mmap_mode="r")
            if descriptors.shape == (len(rows), int(model.descriptor_dim)):
                return descriptors
    descriptors = extract_descriptors(model, rows, transform, device, batch_size, workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, descriptors)
    temporary.replace(path)
    write_json(identity, metadata_path)
    return np.load(path, mmap_mode="r")


def _top_indices(database: np.ndarray, query: np.ndarray, count: int = 10) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(database, dtype=np.float32) @ np.asarray(query, dtype=np.float32)
    count = min(int(count), len(scores))
    indices = np.argpartition(-scores, count - 1)[:count]
    indices = indices[np.argsort(scores[indices])[::-1]]
    return indices.astype(np.int64), scores[indices]


def _positives(database, query, spec: QualitativeDataset) -> np.ndarray:
    if spec.yaw_deg is None:
        xy = np.asarray([(row.easting, row.northing) for row in database])
        return np.flatnonzero(
            np.linalg.norm(xy - (query.easting, query.northing), axis=1) <= spec.radius_m
        )
    return positive_indices(database, [query], spec.radius_m, spec.yaw_deg)[0]


def _copy_image(source: str | Path, destination: Path) -> str:
    source = Path(source)
    suffix = source.suffix.lower() or ".jpg"
    output = destination.with_suffix(suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    return output.name


def _contact_sheet(
    directory: Path,
    entries: list[tuple[str, Path, bool | None]],
) -> None:
    cell_width, cell_height, title_height = 400, 370, 78
    width = cell_width * len(entries)
    canvas = Image.new("RGB", (width, cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 23) if Path(font_path).is_file() else ImageFont.load_default()
    for column, (label, path, correct) in enumerate(entries):
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell_width - 30, 275))
        x = column * cell_width + (cell_width - image.width) // 2
        y = title_height + (275 - image.height) // 2
        canvas.paste(image, (x, y))
        if correct is not None:
            color = "#009E73" if correct else "#D55E00"
            draw.rectangle((x - 4, y - 4, x + image.width + 4, y + image.height + 4), outline=color, width=4)
        draw.multiline_text(
            (column * cell_width + 10, 8), label, fill="black", font=font, spacing=3
        )
    canvas.save(directory / "comparison.jpg", quality=94)


def _combined_contact_sheet(output_root: Path, datasets: list[str]) -> None:
    rows = []
    for dataset in datasets:
        path = output_root / dataset / "comparison.jpg"
        with Image.open(path) as image:
            rows.append((dataset, image.convert("RGB")))
    width = max(image.width for _, image in rows)
    label_height = 34
    canvas = Image.new(
        "RGB", (width, sum(image.height + label_height for _, image in rows)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    y = 0
    for dataset, image in rows:
        draw.text((8, y + 8), dataset.replace("_", " ").title(), fill="black")
        y += label_height
        canvas.paste(image, (0, y))
        y += image.height
    canvas.save(output_root / "comparison_all.jpg", quality=95)


def export_qualitative_top1(
    config: ExperimentConfig,
    selection_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    *,
    batch_size: int = 32,
    workers: int = 4,
    models: tuple[str, ...] = ("baseline", "cdnd"),
) -> Path:
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    selected = {item["dataset"]: item for item in selection["examples"]}
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unsupported qualitative datasets: {unknown}")
    output_root, cache_root = Path(output_dir), Path(cache_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {
        "schema": "qualitative_top1_v1",
        "selection": str(Path(selection_path).resolve()),
        "device": str(device),
        "model_order": list(models),
        "examples": {},
    }

    rows_by_dataset = {}
    for name, item in selected.items():
        spec = DATASETS[name]
        database, queries = suite_rows(config, spec.dataset, spec.split)
        query_index = int(item["query_index"])
        if not 0 <= query_index < len(queries):
            raise IndexError(f"Query index outside {name}: {query_index}")
        query = queries[query_index]
        positives = _positives(database, query, spec)
        if not len(positives):
            raise RuntimeError(f"Selected query has no positive: {name}/{query_index}")
        preferred_path = item.get("positive_reference_path")
        if preferred_path:
            matching = [
                int(index) for index in positives
                if Path(database[int(index)].source_path).resolve() == Path(preferred_path).resolve()
            ]
            if not matching:
                raise ValueError(
                    f"Preferred positive is outside the frozen protocol: {name}/{preferred_path}"
                )
            closest = matching[0]
        else:
            closest = min(positives, key=lambda index: (
                database[int(index)].easting - query.easting
            ) ** 2 + (database[int(index)].northing - query.northing) ** 2)
        directory = output_root / name
        query_file = directory / _copy_image(query.source_path, directory / "query")
        positive_file = directory / _copy_image(
            database[int(closest)].source_path, directory / "ground_truth_positive"
        )
        rows_by_dataset[name] = (spec, database, query, positives, query_file, positive_file)
        results["examples"][name] = {
            "query_index": query_index,
            "query_path": query.source_path,
            "ground_truth_positive_index": int(closest),
            "ground_truth_positive_path": database[int(closest)].source_path,
            "models": {},
        }

    for model_name, model, transform, checkpoint in _models(config, models):
        model.eval().to(device)
        checkpoint_sha256 = file_sha256(checkpoint)
        for name, (spec, database, query, positives, query_file, positive_file) in rows_by_dataset.items():
            protocol = query_protocol_fingerprint(
                database, [query], spec.radius_m, spec.yaw_deg
            )
            identity = {
                "schema": "qualitative_descriptor_cache_v2",
                "model": model_name,
                "checkpoint_sha256": checkpoint_sha256,
                "protocol": name,
                "database_rows": len(database),
                "query_protocol_fingerprint": protocol,
                "descriptor_dim": int(model.descriptor_dim),
                "transform": type(transform).__name__,
                "input_size": [transform.height, transform.width],
                "artifacts": model.qualitative_artifacts,
                "batch_size": model.qualitative_batch_size or batch_size,
            }
            database_descriptors = _cache_descriptors(
                model, database, transform, device, model.qualitative_batch_size or batch_size, workers,
                cache_root / model_name / f"{name}.npy", identity,
            )
            query_descriptor = extract_descriptors(
                model, [query], transform, device, 1, workers
            )[0]
            indices, scores = _top_indices(database_descriptors, query_descriptor, 10)
            success = [int(np.isin(indices[:value], positives).any()) for value in (1, 5, 10)]
            expected = selected[name].get(model_name)
            if expected is not None and success != list(expected):
                raise RuntimeError(
                    f"{model_name}/{name} reproduced {success}, expected {expected}; "
                    "checkpoint or preprocessing does not match the selection"
                )
            top1 = int(indices[0])
            top1_row = database[top1]
            distance_m = float(np.hypot(
                top1_row.easting - query.easting,
                top1_row.northing - query.northing,
            ))
            yaw_error_deg = None
            if spec.yaw_deg is not None:
                yaw_error_deg = float(
                    abs((top1_row.yaw_deg - query.yaw_deg + 180.0) % 360.0 - 180.0)
                )
            directory = output_root / name
            output_name = _copy_image(top1_row.source_path, directory / f"{model_name}_top1")
            results["examples"][name]["models"][model_name] = {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "top1_index": top1,
                "top1_path": database[top1].source_path,
                "top1_score": float(scores[0]),
                "top1_distance_m": distance_m,
                "top1_yaw_error_deg": yaw_error_deg,
                "success_at_1_5_10": success,
                "output_file": output_name,
            }
        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for name, example in results["examples"].items():
        directory = output_root / name
        entries = [("Night query", next(directory.glob("query.*")), None)]
        for model_name in models:
            model_result = example["models"][model_name]
            correct = bool(model_result["success_at_1_5_10"][0])
            geometry = f"{model_result['top1_distance_m']:.1f} m"
            if model_result["top1_yaw_error_deg"] is not None:
                geometry += f", {model_result['top1_yaw_error_deg']:.1f} deg"
            entries.append((
                f"{MODEL_LABELS[model_name]}\n"
                f"{'CORRECT' if correct else 'INCORRECT'}; {geometry}",
                directory / model_result["output_file"],
                correct,
            ))
        entries.append((
            "Valid positive",
            next(directory.glob("ground_truth_positive.*")),
            True,
        ))
        _contact_sheet(directory, entries)
    _combined_contact_sheet(output_root, list(results["examples"]))
    output = output_root / "top1_results.json"
    write_json(results, output)
    return output
