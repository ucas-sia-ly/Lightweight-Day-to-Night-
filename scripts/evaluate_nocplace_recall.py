#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from a2b_lightvpr.config import EvalConfig
from a2b_lightvpr.evaluation import evaluate_model, paired_bootstrap
from a2b_lightvpr.transforms import VPRTransform


class NocPlaceAdapter(torch.nn.Module):
    descriptor_dim = 512

    def __init__(self, repository: Path, checkpoint: Path):
        super().__init__()
        sys.path.insert(0, str(repository))
        from nocplace_models.cosplace_network import GeoLocalizationNet

        self.model = GeoLocalizationNet("ResNet50", 512, train_all_layers=True)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        result = self.model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                f"NocPlace checkpoint mismatch: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the local NocPlace checkpoint with the exact A2B-LightVPR protocol."
    )
    parser.add_argument("--nocplace-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--robotcar-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--resolutions", default="256x384,320x480,512x512",
        help="Comma-separated HxW fit-and-pad resolutions.",
    )
    parser.add_argument("--candidate-json", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NocPlaceAdapter(Path(args.nocplace_repo), Path(args.checkpoint)).eval().to(device)
    evaluation = EvalConfig()
    results = {}
    for item in args.resolutions.split(","):
        height, width = (int(value) for value in item.lower().split("x", 1))
        metrics = evaluate_model(
            model,
            args.robotcar_manifest,
            args.split,
            VPRTransform(height, width),
            evaluation,
            device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        results[item] = metrics
    if args.candidate_json:
        candidate = json.loads(Path(args.candidate_json).read_text(encoding="utf-8"))
        results["paired_comparison"] = {}
        for resolution, metrics in results.copy().items():
            if not isinstance(metrics, dict) or "success_by_k" not in metrics:
                continue
            results["paired_comparison"][resolution] = paired_bootstrap(
                candidate["success_by_k"]["1"],
                metrics["success_by_k"]["1"],
                evaluation.bootstrap_samples,
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        resolution: {
            "recalls": metrics["recalls"],
            "mrr": metrics["mrr"],
            "valid_queries": metrics["valid_queries"],
        }
        for resolution, metrics in results.items()
        if isinstance(metrics, dict) and "recalls" in metrics
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
