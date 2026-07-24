#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from a2b_lightvpr.checkpointing import file_sha256, write_json
from a2b_lightvpr.generator.networks import ResnetGenerator
from a2b_lightvpr.transforms import denormalize_generator, nocplace_paper_crop


PAPER_TARGETS = {
    "fid_max_exclusive": 62.21,
    "l2_max_exclusive": 98.12,
    "psnr_min_exclusive": 13.98,
    "ssim_min_exclusive": 0.75,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an A2B generator with a reconstructed NocPlace protocol."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokyo-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--load-short-side", type=int, default=640)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Smoke only; FID is skipped.")
    parser.add_argument("--timing-iterations", type=int, default=200)
    return parser.parse_args()


def tokyo_condition_rows(root: str | Path) -> tuple[list[Path], list[Path]]:
    query_root = Path(root) / "queries"
    conditions: dict[str, list[Path]] = {"D": [], "N": []}
    for metadata_path in sorted(query_root.glob("*.csv")):
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.reader(handle))
        if len(row) < 6 or row[5] not in conditions:
            continue
        image_path = query_root / row[0]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        conditions[row[5]].append(image_path)
    if len(conditions["D"]) != 375 or len(conditions["N"]) != 375:
        raise ValueError(
            "Tokyo 24/7 reconstruction requires exactly 375 day and 375 night images; "
            f"found day={len(conditions['D'])}, night={len(conditions['N'])}"
        )
    return conditions["D"], conditions["N"]


def generator_tensor(image: Image.Image) -> torch.Tensor:
    return TF.normalize(TF.to_tensor(image), (0.5,) * 3, (0.5,) * 3)


def load_generator(checkpoint: str | Path, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("kind") != "generator_inference":
        raise ValueError("Expected a generator_inference checkpoint")
    model = ResnetGenerator()
    model.load_state_dict(state["generator"])
    return model.eval().requires_grad_(False).to(device), state


def save_real_night(
    paths: list[Path], output_dir: Path, short_side: int, crop_size: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(tqdm(paths, desc="prepare real night")):
        output = output_dir / f"{index:04d}.png"
        if output.is_file():
            continue
        with Image.open(path) as image:
            nocplace_paper_crop(image, short_side, crop_size).save(output, compress_level=3)


def generate_and_score(
    model,
    paths: list[Path],
    output_dir: Path,
    device: torch.device,
    short_side: int,
    crop_size: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    l2_values, psnr_values, ssim_values = [], [], []
    for index, path in enumerate(tqdm(paths, desc="generate and score")):
        with Image.open(path) as image:
            source_image = nocplace_paper_crop(image, short_side, crop_size)
        source = generator_tensor(source_image).unsqueeze(0).to(device)
        with torch.inference_mode():
            generated = model(source)[0].cpu()
        generated_image = TF.to_pil_image(denormalize_generator(generated))
        generated_image.save(output_dir / f"{index:04d}.png", compress_level=3)

        source_array = np.asarray(source_image, dtype=np.float32)
        generated_array = np.asarray(generated_image, dtype=np.float32)
        difference = source_array - generated_array
        l2_values.append(float(np.linalg.norm(difference, axis=2).mean()))
        psnr_values.append(
            float(peak_signal_noise_ratio(source_array, generated_array, data_range=255.0))
        )
        ssim_values.append(
            float(
                structural_similarity(
                    source_array,
                    generated_array,
                    channel_axis=2,
                    data_range=255.0,
                )
            )
        )
    return {
        "l2": float(np.mean(l2_values)),
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
        "per_image": {
            "l2": l2_values,
            "psnr": psnr_values,
            "ssim": ssim_values,
        },
    }


def calculate_fid(generated_dir: Path, real_dir: Path, batch_size: int, workers: int) -> float:
    from pytorch_fid.fid_score import calculate_fid_given_paths

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return float(
        calculate_fid_given_paths(
            [str(generated_dir), str(real_dir)],
            batch_size,
            device,
            2048,
            workers,
        )
    )


def benchmark(model, device: torch.device, crop_size: int, iterations: int) -> dict:
    sample = torch.randn(1, 3, crop_size, crop_size, device=device)
    output = {}
    for precision in ("fp32", "fp16"):
        enabled = precision == "fp16" and device.type == "cuda"
        with torch.inference_mode():
            for _ in range(20):
                with torch.autocast(device.type, enabled=enabled):
                    model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings = []
            for _ in range(iterations):
                start = time.perf_counter()
                with torch.autocast(device.type, enabled=enabled):
                    model(sample)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000.0)
        output[precision] = {
            "mean_ms": float(statistics.fmean(timings)),
            "p50_ms": float(np.percentile(timings, 50)),
            "p95_ms": float(np.percentile(timings, 95)),
        }
    return output


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    generated_dir = output_dir / "generated"
    real_dir = output_dir.parent / "reference_night"
    day_paths, night_paths = tokyo_condition_rows(args.tokyo_root)
    if args.limit:
        day_paths = day_paths[: args.limit]
        night_paths = night_paths[: args.limit]

    model, state = load_generator(checkpoint, device)
    save_real_night(
        night_paths, real_dir, args.load_short_side, args.crop_size
    )
    metrics = generate_and_score(
        model,
        day_paths,
        generated_dir,
        device,
        args.load_short_side,
        args.crop_size,
    )
    metrics["fid"] = (
        None
        if args.limit
        else calculate_fid(generated_dir, real_dir, args.fid_batch_size, args.workers)
    )
    metrics["timing"] = benchmark(model, device, args.crop_size, args.timing_iterations)
    metrics.update(
        protocol="Tokyo-375 reconstructed from NocPlace NightStreets",
        comparable_to_full_nightstreets=False,
        reason="The local Aachen version does not match the paper's 234 day / 196 night split.",
        day_images=len(day_paths),
        night_images=len(night_paths),
        load_short_side=args.load_short_side,
        crop_size=args.crop_size,
        checkpoint=str(checkpoint),
        checkpoint_sha256=file_sha256(checkpoint),
        objective=state.get("objective"),
        step=state.get("step"),
        paper_nightcities_targets=PAPER_TARGETS,
    )
    if metrics["fid"] is not None:
        metrics["beats_paper_targets"] = {
            "fid": metrics["fid"] < PAPER_TARGETS["fid_max_exclusive"],
            "l2": metrics["l2"] < PAPER_TARGETS["l2_max_exclusive"],
            "psnr": metrics["psnr"] > PAPER_TARGETS["psnr_min_exclusive"],
            "ssim": metrics["ssim"] > PAPER_TARGETS["ssim_min_exclusive"],
        }
        metrics["beats_all_paper_targets"] = all(metrics["beats_paper_targets"].values())
    write_json(metrics, output_dir / "metrics.json")
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
