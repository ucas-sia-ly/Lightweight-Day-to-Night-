"""Train CricaVPR-S on GSV-Cities, reusing official data, loss and forward code.

Fixed-budget last-epoch selection; resumes at completed epoch boundaries.
The official third-party checkout is never modified.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from a2b_lightvpr.checkpointing import (
    atomic_torch_save, file_sha256, restore_rng_state, rng_state, write_json,
)
from a2b_lightvpr.references.models import _isolated_repo_import
from a2b_lightvpr.references.training import _set_seed, gsv_manifest_fingerprint
from a2b_lightvpr.vpr.crica import build_crica_small, initialize_crica_training


def official_dataset(repo: Path, root: Path, image_size: int):
    """Replace only the official dataloader's hardcoded root at import time."""
    name = "dataloaders.train.GSVCitiesDataset"
    path = repo / "dataloaders/train/GSVCitiesDataset.py"
    source = path.read_text()
    old = "BASE_PATH = '/home/lufeng/data/VPR/datasets_vg/datasets/gsv_cities/'"
    if source.count(old) != 1:
        raise RuntimeError("Official GSV loader changed; review its dataset-root assignment")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source.replace(old, f"BASE_PATH = {str(root) + '/'!r}"), str(path), "exec"), module.__dict__)
    loader = importlib.import_module("dataloaders.GSVCities")
    original = loader.get_GSVCities
    def get_dataset():
        dataset = original()
        dataset.transform.transforms[0] = dataset.transform.transforms[0].__class__((image_size, image_size))
        return dataset
    loader.get_GSVCities = get_dataset
    return get_dataset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="/home/admin123/github/Lightweight-Day-to-Night/third_party/CricaVPR")
    parser.add_argument("--gsv-root", default="/home/admin123/github/WORKSPACE/dataset/gsv-cities")
    parser.add_argument("--checkpoint", default="/home/admin123/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
    parser.add_argument("--output", default="runs/reference/cricavpr-dinov2-s-gsv-5376")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32, help="Places per batch; four images per place")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224, choices=(224, 322))
    parser.add_argument("--smoke-test", action="store_true", help="One real two-place batch; no training artifacts saved")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0 or args.lr <= 0:
        parser.error("Require epochs>=1, batch-size>=2, workers>=0 and lr>0")
    if not torch.cuda.is_available():
        raise RuntimeError("CricaVPR training requires CUDA")
    torch.set_num_threads(4)
    repo, root = Path(args.repo).resolve(), Path(args.gsv_root).resolve()
    if not (root / "Images").is_dir() or not (root / "Dataframes").is_dir():
        raise FileNotFoundError(f"Incomplete GSV-Cities root: {root}")
    _set_seed(args.seed)
    model = build_crica_small(str(repo), args.checkpoint)
    initialize_crica_training(model)
    model.cuda()
    output = Path(args.output)
    identity = {
        "schema": "cricavpr_small_gsv_v1",
        "backbone": "dinov2_vits14", "descriptor_dim": 5376,
        "foundation_sha256": file_sha256(args.checkpoint),
        "gsv_manifest_sha256": gsv_manifest_fingerprint(root),
        "gsv_root": str(root), "epochs": args.epochs, "seed": args.seed,
        "places_per_batch": args.batch_size, "images_per_place": 4,
        "workers": args.workers, "lr": args.lr, "image_size": [args.image_size, args.image_size],
        "optimizer": "Adam", "scheduler": "StepLR: halve every 3 epochs",
        "precision": "float32", "selection_rule": "fixed_budget_last_epoch",
        "official_checkpoint": False, "third_party_source_modified": False,
        "adapter_channels": [384, 192, 96, 12, 48, 12, 48],
        "source_sha256": {
            str(p.relative_to(repo)): file_sha256(p) for p in sorted(repo.rglob("*.py"))
        },
        "wrapper_sha256": file_sha256(__file__),
        "model_adapter_sha256": file_sha256(Path(__file__).resolve().parents[2] / "src/a2b_lightvpr/vpr/crica.py"),
    }
    latest = output / "checkpoints/latest.pt"
    with _isolated_repo_import(repo, ("dataloaders", "loss")):
        dataset = official_dataset(repo, root, args.image_size)
        loss_function = importlib.import_module("loss").loss_function
        loader = DataLoader(
            dataset, batch_size=2 if args.smoke_test else args.batch_size,
            num_workers=0 if args.smoke_test else args.workers,
            shuffle=False, drop_last=False, pin_memory=True,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(loader) * 3, gamma=0.5)
        start_epoch = 0
        if latest.is_file() and not args.smoke_test:
            state = torch.load(latest, map_location="cpu", weights_only=False)
            if state["identity"] != identity:
                raise RuntimeError("Existing checkpoint settings/source differ; use a new output directory")
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            restore_rng_state(state["rng"])
            start_epoch = int(state["epoch"])
            print(f"Resuming after epoch {start_epoch}", flush=True)
        if not args.smoke_test:
            write_json(identity, output / "training_metadata.json")
        print(f"GSV: {len(dataset)} places; parameters={sum(p.numel() for p in model.parameters()):,}; "
              f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
        for epoch in range(start_epoch, args.epochs):
            model.train()
            total_loss, batches = 0.0, 0
            for images, labels in tqdm(loader, desc=f"crica-small/epoch-{epoch + 1}"):
                images = images.flatten(0, 1).cuda(non_blocking=True)
                labels = labels.flatten().cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                descriptors = model(images)
                loss = loss_function(descriptors, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite CricaVPR loss")
                loss.backward()
                if args.smoke_test:
                    assert descriptors.shape == (8, 5376)
                    assert all(p.grad is None for p in model.backbone.parameters() if not p.requires_grad)
                    grad = model.backbone.blocks[-1].adapter.D_fc2.weight.grad
                    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                batches += 1
                if args.smoke_test:
                    model.eval()
                    with torch.inference_mode():
                        descriptor = model(images[:1])
                    assert descriptor.shape == (1, 5376) and torch.isfinite(descriptor).all()
                    assert torch.allclose(descriptor.norm(dim=1), torch.ones(1, device="cuda"), atol=1e-5)
                    print(f"Smoke passed: real GSV batch, loss={loss.item():.6f}, adapter gradients, frozen backbone, unit descriptor", flush=True)
                    return
            print(f"Epoch {epoch + 1}/{args.epochs}: mean_loss={total_loss / batches:.6f}", flush=True)
            atomic_torch_save({
                "identity": identity, "epoch": epoch + 1,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "rng": rng_state(),
            }, latest)
        final = output / "model.pt"
        # Avoid changing the final checkpoint hash when a completed run is repeated.
        if start_epoch != args.epochs or not final.is_file():
            atomic_torch_save({"model": model.state_dict(), "metadata": identity}, final)
        print(f"Complete: {final}", flush=True)


if __name__ == "__main__":
    main()
