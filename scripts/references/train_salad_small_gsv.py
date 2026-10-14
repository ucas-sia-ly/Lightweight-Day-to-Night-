"""Run the official SALAD training code with a local GSV-Cities checkout.

This wrapper does not modify reference_code/salad. It only patches the official
module's hard-coded dataset root and DINOv2 hub call so that local weights are
used without a network download.
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import torch
from a2b_lightvpr.checkpointing import file_sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--salad-repo", default="/home/admin123/.cache/torch/hub/serizba_salad_main")
    parser.add_argument("--gsv-root", default="/home/admin123/github/WORKSPACE/dataset/gsv-cities")
    parser.add_argument("--dinov2-repo", default="/home/admin123/.cache/torch/hub/facebookresearch_dinov2_main")
    parser.add_argument("--dinov2-checkpoint", default="/home/admin123/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
    parser.add_argument("--output", default="runs/reference/salad-dinov2-s-544-gsv")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    salad_repo = Path(args.salad_repo).expanduser().resolve()
    gsv_root = Path(args.gsv_root).expanduser().resolve()
    dinov2_repo = Path(args.dinov2_repo).expanduser().resolve()
    dinov2_checkpoint = Path(args.dinov2_checkpoint).expanduser().resolve()
    if not (salad_repo / "main.py").is_file():
        raise FileNotFoundError(f"SALAD repository is incomplete: {salad_repo}")
    if not (gsv_root / "Dataframes").is_dir() or not (gsv_root / "Images").is_dir():
        raise FileNotFoundError(f"GSV-Cities root is incomplete: {gsv_root}")
    if not dinov2_checkpoint.is_file():
        raise FileNotFoundError(f"DINOv2-S checkpoint not found: {dinov2_checkpoint}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # The official dataloader uses ../data/GSVCities as a relative path.
    # Create a non-destructive symlink outside the repository source tree.
    official_data_root = salad_repo.parent / "data"
    official_data_root.mkdir(parents=True, exist_ok=True)
    official_gsv = official_data_root / "GSVCities"
    if not official_gsv.exists():
        official_gsv.symlink_to(gsv_root, target_is_directory=True)

    original_cwd = Path.cwd()
    os.chdir(salad_repo)
    sys.path.insert(0, str(salad_repo))
    # Validation modules are imported eagerly by the official datamodule.
    # Stub only these unused modules so GSV-Cities-only training does not
    # require unrelated Pittsburgh/MSLS downloads.
    pitts_stub = types.ModuleType("dataloaders.PittsburgDataset")
    mapillary_stub = types.ModuleType("dataloaders.MapillaryDataset")
    pitts_stub.get_whole_val_set = lambda **kwargs: None
    mapillary_stub.MSLS = lambda **kwargs: None
    sys.modules["dataloaders.PittsburgDataset"] = pitts_stub
    sys.modules["dataloaders.MapillaryDataset"] = mapillary_stub
    import dataloaders.GSVCitiesDataset as gsv_dataset
    import dataloaders.GSVCitiesDataloader as gsv_loader

    original_hub_load = torch.hub.load

    def local_dinov2(repo_or_dir, model, *hub_args, **hub_kwargs):
        if str(repo_or_dir) == "facebookresearch/dinov2":
            network = original_hub_load(
                str(dinov2_repo), "dinov2_vits14", source="local", pretrained=False
            )
            state = torch.load(dinov2_checkpoint, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            result = network.load_state_dict(state, strict=True)
            if result.missing_keys or result.unexpected_keys:
                raise RuntimeError(f"DINOv2-S checkpoint mismatch: {result}")
            return network
        return original_hub_load(repo_or_dir, model, *hub_args, **hub_kwargs)

    torch.hub.load = local_dinov2
    try:
        from main import VPRModel
        from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
        import pytorch_lightning as pl

        class GSVOnlyDataModule(GSVCitiesDataModule):
            def setup(self, stage=None):
                if stage in (None, "fit"):
                    self.reload()

            def val_dataloader(self):
                return []

        datamodule = GSVOnlyDataModule(
            batch_size=args.batch_size,
            img_per_place=4,
            min_img_per_place=4,
            shuffle_all=False,
            random_sample_from_each_place=True,
            image_size=(224, 224),
            num_workers=args.workers,
            show_data_stats=True,
            val_set_names=["pitts30k_val", "msls_val"],
        )
        model = VPRModel(
            backbone_arch="dinov2_vits14",
            backbone_config={"num_trainable_blocks": 4, "return_token": True, "norm_layer": True},
            agg_arch="SALAD",
            # Compact SALAD head: 16*32 local dimensions + 32 global-token dimensions.
            agg_config={"num_channels": 384, "num_clusters": 16, "cluster_dim": 32, "token_dim": 32},
            lr=6e-5,
            optimizer="adamw",
            weight_decay=9.5e-9,
            momentum=0.9,
            lr_sched="linear",
            lr_sched_args={"start_factor": 1, "end_factor": 0.2, "total_iters": 4000},
            loss_name="MultiSimilarityLoss",
            miner_name="MultiSimilarityMiner",
            miner_margin=0.1,
            faiss_gpu=False,
        )
        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=str(Path(args.output).expanduser().resolve()),
            filename="salad-small-{epoch:02d}",
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=0,
            save_last=True,
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            default_root_dir=str(Path(args.output).expanduser().resolve()),
            num_nodes=1,
            num_sanity_val_steps=0,
            precision="16-mixed",
            max_epochs=args.epochs,
            check_val_every_n_epoch=1,
            callbacks=[checkpoint_cb],
            reload_dataloaders_every_n_epochs=1,
            log_every_n_steps=20,
        )
        trainer.fit(model=model, datamodule=datamodule)
        output_dir = Path(args.output).expanduser().resolve()
        write_json(
            {
                "schema": "official_salad_small_gsv_training_v1",
                "training_method": "reference_code/salad/main.py",
                "backbone": "dinov2_vits14",
                "backbone_checkpoint": str(dinov2_checkpoint),
                "backbone_checkpoint_sha256": file_sha256(dinov2_checkpoint),
                "gsv_cities_root": str(gsv_root),
                "image_size": [224, 224],
                "batch_size_places": args.batch_size,
                "images_per_place": 4,
                "aggregator": "SALAD",
                "aggregator_config": {"num_channels": 384, "num_clusters": 16, "cluster_dim": 32, "token_dim": 32},
                "descriptor_dim": 544,
                "epochs": args.epochs,
                "seed": args.seed,
                "official_source": str(salad_repo),
                "third_party_source_modified": False,
            },
            output_dir / "training_metadata.json",
        )
    finally:
        os.chdir(original_cwd)
        torch.hub.load = original_hub_load
        sys.modules.pop("dataloaders.PittsburgDataset", None)
        sys.modules.pop("dataloaders.MapillaryDataset", None)
        sys.path.remove(str(salad_repo))


if __name__ == "__main__":
    main()
