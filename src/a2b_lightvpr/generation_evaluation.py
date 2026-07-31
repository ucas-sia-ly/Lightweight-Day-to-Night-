from __future__ import annotations

import csv
import re
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from .checkpointing import file_sha256, write_json
from .config import ExperimentConfig
from .generator.networks import ResnetGenerator
from .tracking import TrainingTracker
from .transforms import denormalize_generator, nocplace_paper_crop


PAPER_TARGETS = {"fid": 62.21, "l2": 98.12, "psnr": 13.98, "ssim": 0.75}


def tokyo_condition_rows(root: str | Path) -> tuple[list[Path], list[Path]]:
    query_root = Path(root) / "queries"
    conditions = {"D": [], "N": []}
    for metadata in sorted(query_root.glob("*.csv")):
        with metadata.open(newline="", encoding="utf-8") as handle:
            row = next(csv.reader(handle))
        if len(row) >= 6 and row[5] in conditions:
            path = query_root / row[0]
            if not path.is_file():
                raise FileNotFoundError(path)
            conditions[row[5]].append(path)
    if len(conditions["D"]) != 375 or len(conditions["N"]) != 375:
        raise ValueError("Tokyo protocol requires exactly 375 day and 375 night images")
    return conditions["D"], conditions["N"]


def aachen_condition_rows(root: str | Path) -> tuple[list[Path], list[Path]]:
    root = Path(root)
    image_root = root / "images" / "images_upright"
    query_root = root / "queries"
    outputs = []
    for name in ("day_time_queries_with_intrinsics.txt", "night_time_queries_with_intrinsics.txt"):
        paths = []
        for line in (query_root / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                path = image_root / line.split()[0]
                if not path.is_file():
                    raise FileNotFoundError(path)
                paths.append(path)
        outputs.append(paths)
    return outputs[0], outputs[1]


def bootstrap_summary(values: list[float], samples: int = 2000, seed: int = 0) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("Cannot summarize empty metric values")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        batch = min(256, samples - start)
        indices = rng.integers(0, len(array), size=(batch, len(array)))
        means[start:start + batch] = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "count": int(len(array)),
    }


def polynomial_mmd(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or len(second) < 2:
        raise ValueError("KID requires at least two samples per distribution")
    # The unbiased estimator can be negative for finite samples.  When callers
    # pass the exact same observations, however, the population discrepancy is
    # known to be zero and avoiding two differently weighted diagonal terms is
    # both exact and less surprising for diagnostics.
    if first.shape == second.shape and np.array_equal(first, second):
        return 0.0
    dimension = first.shape[1]
    kernel_x = (first @ first.T / dimension + 1.0) ** 3
    kernel_y = (second @ second.T / dimension + 1.0) ** 3
    kernel_xy = (first @ second.T / dimension + 1.0) ** 3
    x_term = (kernel_x.sum() - np.trace(kernel_x)) / (len(first) * (len(first) - 1))
    y_term = (kernel_y.sum() - np.trace(kernel_y)) / (len(second) * (len(second) - 1))
    return float(x_term + y_term - 2.0 * kernel_xy.mean())


def _distribution_metrics(generated: Path, real: Path, batch_size: int, workers: int):
    from pytorch_fid.fid_score import calculate_frechet_distance, get_activations
    from pytorch_fid.inception import InceptionV3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block]).to(device)
    generated_files = sorted(str(path) for path in generated.glob("*.png"))
    real_files = sorted(str(path) for path in real.glob("*.png"))
    fake_features = get_activations(generated_files, model, batch_size, 2048, device, workers)
    real_features = get_activations(real_files, model, batch_size, 2048, device, workers)
    fid = calculate_frechet_distance(
        fake_features.mean(0), np.cov(fake_features, rowvar=False),
        real_features.mean(0), np.cov(real_features, rowvar=False),
    )
    rng = np.random.default_rng(0)
    subset_size = min(100, len(fake_features), len(real_features))
    kid_values = [
        polynomial_mmd(
            fake_features[rng.choice(len(fake_features), subset_size, replace=False)],
            real_features[rng.choice(len(real_features), subset_size, replace=False)],
        )
        for _ in range(100)
    ]
    return float(fid), bootstrap_summary(kid_values, samples=2000)


def _load_generator(checkpoint: Path, device: torch.device, config: ExperimentConfig):
    if checkpoint.name.endswith("_net_G.pth"):
        from .nocplace_repro import load_official_negcut_generator

        model = load_official_negcut_generator(config, checkpoint, device)
        epoch = checkpoint.name.removesuffix("_net_G.pth")
        return model.requires_grad_(False), {
            "kind": "official_negcut_generator",
            "step": int(epoch) * 5000 if epoch.isdigit() else 0,
        }
    checkpoint = checkpoint.resolve()
    training_checkpoint_dir = (Path(config.generator.output_dir) / "checkpoints").resolve()
    is_project_training_state = (
        checkpoint.parent == training_checkpoint_dir
        and re.fullmatch(r"state_\d{7}\.pt", checkpoint.name) is not None
    )
    # Project training states contain NumPy/Python RNG state, which cannot be
    # decoded by PyTorch 2.6's weights-only unpickler. Full unpickling is only
    # permitted for the project's own, conventionally named checkpoint path.
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=not is_project_training_state,
    )
    if not isinstance(state, dict):
        raise ValueError("Generator checkpoint must contain a state dictionary")
    if state.get("kind") not in {"generator_inference", "generator_training_state"}:
        raise ValueError("Expected a generator inference or training-state checkpoint")
    if not isinstance(state.get("generator"), dict):
        raise ValueError("Generator checkpoint is missing the generator state dictionary")
    model = ResnetGenerator()
    model.load_state_dict(state["generator"], strict=True)
    return model.eval().requires_grad_(False).to(device), state


def evaluate_generation(
    config: ExperimentConfig,
    checkpoint: str | Path,
    *,
    dataset: str = "tokyo",
    data_root: str = "",
    output_dir: str | Path | None = None,
    limit: int = 0,
    crop_size: int = 512,
    short_side: int = 640,
    fid_batch_size: int = 32,
    workers: int = 4,
    timing_iterations: int = 200,
) -> Path:
    checkpoint = Path(checkpoint).resolve()
    root = data_root or (config.data.tokyo_root if dataset == "tokyo" else config.data.aachen_root)
    if not root:
        raise ValueError(f"No data root configured for {dataset}")
    day_paths, night_paths = (
        tokyo_condition_rows(root) if dataset == "tokyo" else aachen_condition_rows(root)
    )
    if limit:
        day_paths, night_paths = day_paths[:limit], night_paths[:limit]
    checkpoint_hash = file_sha256(checkpoint)
    output = Path(output_dir or Path(config.generator.output_dir) / "generation_eval" / dataset / checkpoint.stem)
    generated_dir, real_dir = output / "generated", output / "real_night"
    generated_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state = _load_generator(checkpoint, device, config)
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    values = {"l2": [], "psnr": [], "ssim": []}
    for index, path in enumerate(tqdm(day_paths, desc=f"evaluate generation/{dataset}")):
        with Image.open(path) as image:
            source_image = nocplace_paper_crop(image, short_side, crop_size)
        source = TF.normalize(TF.to_tensor(source_image), (0.5,) * 3, (0.5,) * 3).unsqueeze(0)
        with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
            generated = model(source.to(device))[0].cpu()
        generated_image = TF.to_pil_image(denormalize_generator(generated))
        generated_image.save(generated_dir / f"{index:04d}.png", compress_level=3)
        source_array = np.asarray(source_image, dtype=np.float32)
        generated_array = np.asarray(generated_image, dtype=np.float32)
        values["l2"].append(float(np.linalg.norm(source_array - generated_array, axis=2).mean()))
        values["psnr"].append(float(peak_signal_noise_ratio(source_array, generated_array, data_range=255.0)))
        values["ssim"].append(float(structural_similarity(
            source_array, generated_array, channel_axis=2, data_range=255.0
        )))
    for index, path in enumerate(tqdm(night_paths, desc="prepare real night")):
        with Image.open(path) as image:
            nocplace_paper_crop(image, short_side, crop_size).save(real_dir / f"{index:04d}.png")

    distribution = (None, None) if limit else _distribution_metrics(
        generated_dir, real_dir, fid_batch_size, workers
    )
    sample = torch.randn(1, 3, crop_size, crop_size, device=device)
    timings = {}
    for precision in ("fp32", "fp16"):
        enabled = precision == "fp16" and device.type == "cuda"
        measured = []
        with torch.inference_mode():
            for _ in range(20):
                with torch.autocast(device.type, enabled=enabled): model(sample)
            for _ in range(timing_iterations):
                if device.type == "cuda": torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.autocast(device.type, enabled=enabled): model(sample)
                if device.type == "cuda": torch.cuda.synchronize()
                measured.append((time.perf_counter() - start) * 1000.0)
        timings[precision] = {
            "mean_ms": float(statistics.fmean(measured)),
            "p50_ms": float(np.percentile(measured, 50)),
            "p95_ms": float(np.percentile(measured, 95)),
        }
    metrics = {
        "dataset": dataset,
        "protocol": "Tokyo-375" if dataset == "tokyo" else "Aachen-local-non-paper-compatible",
        # NocPlace reports the combined NightStreets protocol, not Tokyo alone.
        "paper_comparable": False,
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "step": int(state.get("step", 0)), "day_images": len(day_paths),
        "night_images": len(night_paths), "crop_size": crop_size, "short_side": short_side,
        "fid": distribution[0], "kid": distribution[1],
        "l2": bootstrap_summary(values["l2"]),
        "psnr": bootstrap_summary(values["psnr"]),
        "ssim": bootstrap_summary(values["ssim"]),
        "per_image": values, "timing": timings,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
        "paper_nightcities_targets": PAPER_TARGETS,
    }
    write_json(metrics, output / "metrics.json")
    tracker = TrainingTracker(config, job_type="evaluate-generation",
        default_name=f"generation-{dataset}-{checkpoint.stem}", output_dir=output)
    tracker.summary({
        "generation/fid": metrics["fid"],
        "generation/kid_mean": metrics["kid"]["mean"] if metrics["kid"] else None,
        "generation/l2": metrics["l2"]["mean"],
        "generation/psnr": metrics["psnr"]["mean"],
        "generation/ssim": metrics["ssim"]["mean"],
        "generation/checkpoint_sha256": checkpoint_hash,
    })
    tracker.finish()
    return output / "metrics.json"
