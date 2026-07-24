from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import functional as TF

from .checkpointing import file_sha256, write_json
from .config import ExperimentConfig
from .tracking import TrainingTracker
from .transforms import IMAGENET_MEAN, IMAGENET_STD, nocplace_paper_crop
from .vpr.losses import MarginCosineClassifier, ikt_loss


DATASETS = ("darkdriving", "robotcar", "bdd100k", "msls", "svox")
MANIFEST_FIELDS = (
    "selection_id", "dataset", "domain", "route", "source_path", "source_sha256",
    "export_path", "selection_reason",
)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: list[dict], fields) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _route_balanced(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Round-robin over routes, with deterministic shuffles inside each route."""
    pools: dict[str, list[dict]] = {}
    seen = set()
    for row in rows:
        path = str(Path(row["source_path"]).expanduser().resolve())
        if path in seen or not Path(path).is_file():
            continue
        seen.add(path)
        item = dict(row)
        item["source_path"] = path
        pools.setdefault(row.get("route") or "unknown", []).append(item)
    rng = random.Random(seed)
    for route, pool in pools.items():
        random.Random(f"{seed}:{route}").shuffle(pool)
    routes = sorted(pools)
    rng.shuffle(routes)
    selected = []
    while routes and len(selected) < count:
        remaining = []
        for route in routes:
            if pools[route] and len(selected) < count:
                selected.append(pools[route].pop())
            if pools[route]:
                remaining.append(route)
        routes = remaining
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} unique usable images are available; need {count}")
    return selected


def _domain_candidates(rows: list[dict], dataset: str, domain: str) -> list[dict]:
    candidates = []
    field = "source_path" if domain == "day" else "target_path"
    for row in rows:
        if row.get("dataset") != dataset or row.get("split") != "train":
            continue
        value = row.get(field, "")
        if value:
            candidates.append({"source_path": value, "route": row.get("route", "unknown")})
    return candidates


def prepare_nocplace_repro(config: ExperimentConfig) -> dict:
    repro = config.nocplace_repro
    source_rows = _read_csv(config.data.generator_manifest)
    root = Path(repro.data_root).expanduser().resolve()
    dataset_root = root / "negcut_5k"
    manifest_rows = []
    used_paths = set()
    expected_exports = set()
    for dataset_index, dataset in enumerate(DATASETS):
        for domain_index, domain in enumerate(("day", "night")):
            selected = _route_balanced(
                _domain_candidates(source_rows, dataset, domain),
                repro.samples_per_dataset_domain,
                repro.seed + dataset_index * 101 + domain_index,
            )
            export_dir = dataset_root / ("trainA" if domain == "day" else "trainB")
            export_dir.mkdir(parents=True, exist_ok=True)
            for index, row in enumerate(selected):
                source = Path(row["source_path"])
                if str(source) in used_paths:
                    raise ValueError(f"Duplicate source selected across domains: {source}")
                used_paths.add(str(source))
                suffix = source.suffix.lower() if source.suffix else ".jpg"
                export = export_dir / f"{dataset}_{index:04d}_{hashlib.sha1(str(source).encode()).hexdigest()[:12]}{suffix}"
                if export.is_symlink() or export.exists():
                    if export.resolve() != source.resolve():
                        raise FileExistsError(f"Refusing to replace unrelated export: {export}")
                else:
                    export.symlink_to(source.resolve())
                expected_exports.add(export.absolute())
                manifest_rows.append({
                    "selection_id": hashlib.sha1(f"{dataset}:{domain}:{source}".encode()).hexdigest(),
                    "dataset": dataset,
                    "domain": domain,
                    "route": row.get("route", "unknown"),
                    "source_path": str(source.resolve()),
                    "source_sha256": file_sha256(source),
                    "export_path": str(export.absolute()),
                    "selection_reason": "deterministic route-balanced clean-night sample",
                })
    for folder in (dataset_root / "trainA", dataset_root / "trainB"):
        for path in folder.iterdir():
            if path.absolute() in expected_exports:
                continue
            if not path.is_symlink():
                raise FileExistsError(f"Refusing to delete an unmanaged NEG-CUT export: {path}")
            path.unlink()
        if len(list(folder.iterdir())) != 2500:
            raise RuntimeError(f"Expected exactly 2,500 images in {folder}")
    manifest = root / "negcut_5k_manifest.csv"
    _write_csv(manifest, manifest_rows, MANIFEST_FIELDS)
    counts = {
        f"{dataset}/{domain}": sum(
            row["dataset"] == dataset and row["domain"] == domain for row in manifest_rows
        )
        for dataset in DATASETS for domain in ("day", "night")
    }
    result = {
        "manifest": str(manifest), "dataset_root": str(dataset_root),
        "rows": len(manifest_rows), "counts": counts,
        "manifest_sha256": file_sha256(manifest),
    }
    write_json(result, root / "prepare_summary.json")
    return result


def build_negcut_train_command(config: ExperimentConfig) -> list[str]:
    repro = config.nocplace_repro
    repo = Path(repro.negcut_repo).expanduser().resolve()
    data_root = Path(repro.data_root).expanduser().resolve() / "negcut_5k"
    if not (repo / "train.py").is_file():
        raise FileNotFoundError(f"Official NEG-CUT train.py is missing: {repo / 'train.py'}")
    if not (data_root / "trainA").is_dir() or not (data_root / "trainB").is_dir():
        raise FileNotFoundError("Run prepare-nocplace-repro before NEG-CUT training")
    return [
        sys.executable, "train.py", "--dataroot", str(data_root), "--name", repro.negcut_name,
        "--checkpoints_dir", str(Path(repro.negcut_checkpoints_dir).expanduser().resolve()),
        "--model", "negcut", "--NEGCUT_mode", "negcut", "--dataset_mode", "unaligned",
        "--direction", "AtoB", "--netG", "resnet_9blocks", "--gan_mode", "lsgan",
        "--preprocess", "resize_and_crop", "--load_size", str(repro.load_size),
        "--crop_size", str(repro.crop_size), "--batch_size", str(repro.negcut_batch_size),
        "--lr", str(repro.negcut_lr), "--n_epochs", str(repro.negcut_epochs),
        "--n_epochs_decay", str(repro.negcut_decay_epochs), "--epoch_count", "1",
        "--save_epoch_freq", "100", "--save_latest_freq", "5000",
        "--no_html", "--display_id", "-1",
    ]


def train_negcut_reference(
    config: ExperimentConfig, *, dry_run: bool = False, smoke_iterations: int | None = None
) -> dict:
    command = build_negcut_train_command(config)
    repro = config.nocplace_repro
    run_name = repro.negcut_name
    if smoke_iterations:
        run_name = f"{repro.negcut_name}_smoke"
        command.extend([
            "--name", run_name, "--n_epochs", "1", "--n_epochs_decay", "0",
            "--max_dataset_size", str(int(smoke_iterations)), "--save_epoch_freq", "1",
            "--save_latest_freq", str(int(smoke_iterations)),
        ])
    output = Path(repro.output_dir).expanduser().resolve() / "negcut"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": shlex.join(command), "cwd": str(Path(repro.negcut_repo).resolve()),
        "official_defaults": {"lambda_GAN": 1.0, "lambda_NCE": 1.0, "lambda_MS_neg": 1.0},
        "environment": {"MKL_THREADING_LAYER": "GNU"},
        "dry_run": bool(dry_run),
    }
    write_json(payload, output / "resolved_command.json")
    if not dry_run:
        tracker = TrainingTracker(
            config, job_type="train-negcut-reference", default_name="negcut-official-clean5k",
            output_dir=output, run_key=file_sha256(Path(repro.data_root) / "negcut_5k_manifest.csv"),
        )
        environment = os.environ.copy()
        environment["MKL_THREADING_LAYER"] = "GNU"
        process = subprocess.Popen(
            command, cwd=repro.negcut_repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=environment, bufsize=1,
        )
        loss_pattern = re.compile(r"\(epoch: (\d+), iters: (\d+),.*?\) (.*)")
        value_pattern = re.compile(r"([A-Za-z_]+):\s+([-+0-9.eE]+)")
        learning_rate = repro.negcut_lr
        with (output / "train.log").open("a", encoding="utf-8") as log:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                match = loss_pattern.search(line)
                if match:
                    epoch, iteration = int(match.group(1)), int(match.group(2))
                    values = {key: float(value) for key, value in value_pattern.findall(match.group(3))}
                    # UnalignedDataset length is max(2,500 day, 2,500 night) per epoch.
                    step = (epoch - 1) * 2500 + iteration
                    tracker.log({
                        "train/gan_generator": values.get("G_GAN"),
                        "train/negcut": values.get("NCE"),
                        # Official loss N combines hard-negative adversarial and diversity terms.
                        "train/negative_generator": values.get("N"),
                        "train/discriminator_real": values.get("D_real"),
                        "train/discriminator_fake": values.get("D_fake"),
                        "train/lr": learning_rate,
                        "train/epoch": epoch,
                    }, step=step)
                lr_match = re.search(r"learning rate .* -> ([-+0-9.eE]+)", line)
                if lr_match:
                    learning_rate = float(lr_match.group(1))
        returncode = process.wait()
        payload["returncode"] = returncode
        if returncode == 0 and tracker.enabled:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data_root = Path(repro.data_root).expanduser().resolve() / "negcut_5k"
            source_paths = sorted((data_root / "trainA").iterdir())[:4]
            target_paths = sorted((data_root / "trainB").iterdir())[:4]
            source = torch.stack([
                _generator_tensor(Image.open(path), repro.load_size, repro.crop_size)
                for path in source_paths
            ]).to(device)
            target = torch.stack([
                _generator_tensor(Image.open(path), repro.load_size, repro.crop_size)
                for path in target_paths
            ]).to(device)
            checkpoint_root = Path(repro.negcut_checkpoints_dir).expanduser().resolve() / run_name
            snapshot_epochs = (1,) if smoke_iterations else (100, 200, 300, 400)
            for epoch in snapshot_epochs:
                checkpoint = checkpoint_root / f"{epoch}_net_G.pth"
                if not checkpoint.is_file():
                    continue
                generator = load_official_negcut_generator(config, checkpoint, device)
                with torch.inference_mode():
                    fake = generator(source)
                tracker.log_generator_images(source, fake, target, step=epoch * 2500)
                del generator, fake
        tracker.summary({
            "negcut/epochs": 1 if smoke_iterations else 400, "negcut/lambda_gan": 1.0,
            "negcut/lambda_nce": 1.0, "negcut/lambda_negative_diversity": 1.0,
        })
        tracker.finish(exit_code=returncode)
        if returncode:
            raise RuntimeError(f"Official NEG-CUT exited with code {returncode}; see {output / 'train.log'}")
    return payload


@contextmanager
def _repo_import(repo: str | Path):
    repo = str(Path(repo).expanduser().resolve())
    sys.path.insert(0, repo)
    try:
        yield
    finally:
        sys.path.remove(repo)


def load_official_negcut_generator(config: ExperimentConfig, checkpoint: str | Path, device):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if checkpoint.name == "generator_final.pt" or not checkpoint.name.endswith("_net_G.pth"):
        raise ValueError("Expected an official NEG-CUT *_net_G.pth checkpoint, not a project generator checkpoint")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    with _repo_import(config.nocplace_repro.negcut_repo):
        from models.generator import define_G

        model = define_G(
            3, 3, 64, "resnet_9blocks", norm="instance", use_dropout=False,
            init_type="xavier", init_gain=0.02, no_antialias=False,
            no_antialias_up=False, gpu_ids=[], opt=None,
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("Official NEG-CUT checkpoint must contain a raw state dict")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"Strict official NEG-CUT checkpoint mismatch: {result}")
    return model.eval().to(device)


def _generator_tensor(image: Image.Image, load_size: int, crop_size: int) -> torch.Tensor:
    image = nocplace_paper_crop(image.convert("RGB"), load_size, crop_size)
    return TF.normalize(TF.to_tensor(image), [0.5] * 3, [0.5] * 3)


def translate_nocplace_train(
    config: ExperimentConfig, checkpoint: str | Path, *, limit: int | None = None
) -> dict:
    repro = config.nocplace_repro
    rows = _read_csv(config.data.vpr_source_manifest)
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("NocPlace translation manifest must contain train rows only")
    if not limit and len(rows) != 118_313:
        raise ValueError(f"Formal NocPlace translation requires 118,313 source rows, got {len(rows)}")
    if limit:
        rows = rows[: int(limit)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_official_negcut_generator(config, checkpoint, device)
    base_output_root = Path(repro.generated_root).expanduser().resolve()
    output_root = base_output_root / f"smoke_limit_{int(limit)}" if limit else base_output_root
    vpr_export_root = (
        output_root / "vpr" if limit
        else Path(repro.data_root).expanduser().resolve() / "vpr"
    )
    image_root = output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = file_sha256(checkpoint)
    translated = []
    partial = output_root / "manifest.partial.csv"
    complete_manifest = output_root / "manifest.csv"
    resume_manifest = complete_manifest if complete_manifest.is_file() else partial
    resumed = {}
    if resume_manifest.is_file():
        resumed = {
            row["image_id"]: row for row in _read_csv(resume_manifest)
            if row.get("generator_sha256") == checkpoint_hash
        }
    batch_size = repro.translation_batch_size
    last_partial = 0
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            items = []
            pending_tensors, pending_positions = [], []
            for row in rows[start:start + batch_size]:
                source = Path(row["source_path"])
                yaw = row.get("yaw") or "0"
                legacy_name = (
                    f'@{row.get("utm_e", "0")}@{row.get("utm_n", "0")}@{row["dataset"]}'
                    f'@{row.get("route", "unknown")}@{hashlib.sha1(row["image_id"].encode()).hexdigest()[:16]}'
                    f'@0@0@0@{yaw}@.jpg'
                ).replace("/", "_")
                output = image_root / row["dataset"] / legacy_name
                output.parent.mkdir(parents=True, exist_ok=True)
                day_export = vpr_export_root / "train" / legacy_name
                night_export = vpr_export_root / "train_night" / legacy_name
                item = (row, source, output, day_export, night_export)
                items.append(item)
                cached = resumed.get(row["image_id"])
                valid_cached = bool(cached and Path(cached.get("generated_path", "")) == output)
                if valid_cached:
                    try:
                        with Image.open(output) as image:
                            valid_cached = image.size == (512, 512)
                            image.verify()
                    except (OSError, ValueError):
                        valid_cached = False
                if not valid_cached:
                    with Image.open(source) as image:
                        pending_tensors.append(_generator_tensor(
                            image, repro.load_size, repro.crop_size
                        ))
                    pending_positions.append(len(items) - 1)
            if pending_tensors:
                fake_batch = model(torch.stack(pending_tensors).to(device)).clamp(-1, 1).cpu()
                for fake, position in zip(fake_batch, pending_positions):
                    output = items[position][2]
                    TF.to_pil_image(fake.add(1).mul(127.5).byte()).save(output, quality=95)
            for row, source, output, day_export, night_export in items:
                day_export.parent.mkdir(parents=True, exist_ok=True)
                night_export.parent.mkdir(parents=True, exist_ok=True)
                for export, target in ((day_export, source), (night_export, output)):
                    if export.is_symlink() or export.exists():
                        if export.resolve() != target.resolve():
                            raise FileExistsError(f"Refusing to replace unrelated VPR export: {export}")
                    else:
                        export.symlink_to(target.resolve())
                translated.append({
                    **row, "generated_path": str(output.resolve()),
                    "day_export_path": str(day_export.absolute()),
                    "night_export_path": str(night_export.absolute()),
                    "generator_sha256": checkpoint_hash,
                    "output_width": 512, "output_height": 512,
                })
            if len(translated) - last_partial >= 1000:
                _write_csv(partial, translated, translated[0].keys())
                last_partial = len(translated)
    manifest = complete_manifest
    fields = list(rows[0].keys()) + [
        "generated_path", "day_export_path", "night_export_path", "generator_sha256",
        "output_width", "output_height"
    ] if rows else []
    _write_csv(manifest, translated, fields)
    result = {
        "manifest": str(manifest), "rows": len(translated), "expected_full_rows": 118_313,
        "complete": len(translated) == 118_313 and not limit,
        "generator_sha256": checkpoint_hash,
    }
    write_json(result, output_root / "translation_summary.json")
    return result


def _build_nocplace_model(config: ExperimentConfig, checkpoint: str | Path, device):
    repo = config.nocplace_repro.nocplace_repo
    with _repo_import(repo):
        from nocplace_models import model_factory

        model = model_factory.build_vpr_model("ResNet50", 512, train_all_layers=True)
        result = model_factory.load_checkpoint(model, str(checkpoint), "cpu", strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"Strict CosPlace checkpoint mismatch: {result}")
    return model.to(device)


class _NocPlaceDataset(Dataset):
    def __init__(self, rows: list[dict], teacher_cache: np.ndarray | None = None):
        self.rows = rows
        places = sorted({f'{row["dataset"]}:{row["place_id"]}' for row in rows})
        labels = {place: index for index, place in enumerate(places)}
        self.labels = [labels[f'{row["dataset"]}:{row["place_id"]}'] for row in rows]
        self.place_indices: dict[int, list[int]] = {}
        for index, label in enumerate(self.labels):
            self.place_indices.setdefault(label, []).append(index)
        self.teacher_cache = teacher_cache

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def transform(image: Image.Image) -> torch.Tensor:
        tensor = TF.to_tensor(TF.resize(
            image.convert("RGB"), [512, 512],
            interpolation=TF.InterpolationMode.BICUBIC, antialias=True,
        ))
        return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["generated_path"]) as image:
            fake = self.transform(image)
        sample = {"fake": fake, "label": self.labels[index], "image_id": row["image_id"]}
        if self.teacher_cache is not None:
            sample["teacher"] = torch.from_numpy(np.asarray(self.teacher_cache[index]).copy()).float()
        return sample


class PlaceBalancedSampler(Sampler[int]):
    def __init__(self, dataset: _NocPlaceDataset, samples: int, seed: int = 0):
        self.dataset, self.samples, self.seed = dataset, int(samples), int(seed)

    def __len__(self):
        return self.samples

    def __iter__(self):
        rng = random.Random(self.seed)
        places = list(self.dataset.place_indices)
        rng.shuffle(places)
        offsets = {place: 0 for place in places}
        pools = {place: list(indices) for place, indices in self.dataset.place_indices.items()}
        for pool in pools.values():
            rng.shuffle(pool)
        for sample_index in range(self.samples):
            place = places[sample_index % len(places)]
            pool = pools[place]
            offset = offsets[place]
            if offset == len(pool):
                rng.shuffle(pool)
                offset = 0
            yield pool[offset]
            offsets[place] = offset + 1


def _cache_teacher(
    config: ExperimentConfig, rows: list[dict], device, manifest: Path
) -> tuple[np.memmap, dict]:
    repro = config.nocplace_repro
    formal_manifest = Path(repro.generated_root).expanduser().resolve() / "manifest.csv"
    root = (
        Path(repro.data_root).expanduser().resolve() / "vpr" / "train_feat"
        if manifest == formal_manifest else manifest.parent / "vpr" / "train_feat"
    )
    root.mkdir(parents=True, exist_ok=True)
    cache_path, metadata_path = root / "cosplace_day_fp16.npy", root / "metadata.json"
    manifest_hash = file_sha256(manifest)
    checkpoint_hash = file_sha256(repro.cosplace_checkpoint)
    expected = {
        "rows": len(rows), "descriptor_dim": 512, "manifest_sha256": manifest_hash,
        "checkpoint_sha256": checkpoint_hash,
        "image_ids_sha256": hashlib.sha256("\n".join(row["image_id"] for row in rows).encode()).hexdigest(),
        "normalized": True, "complete": True,
    }
    if metadata_path.is_file() and cache_path.is_file():
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        if actual == expected:
            return np.load(cache_path, mmap_mode="r"), expected
    teacher = _build_nocplace_model(config, repro.cosplace_checkpoint, device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float16, shape=(len(rows), 512))
    with torch.inference_mode():
        for start in range(0, len(rows), repro.vpr_batch_size):
            batch = []
            for row in rows[start:start + repro.vpr_batch_size]:
                with Image.open(row["source_path"]) as image:
                    batch.append(_NocPlaceDataset.transform(image))
            descriptors = F.normalize(teacher(torch.stack(batch).to(device)), dim=1)
            cache[start:start + len(batch)] = descriptors.cpu().numpy().astype(np.float16)
    cache.flush()
    write_json(expected, metadata_path)
    return np.load(cache_path, mmap_mode="r"), expected


def nocplace_reproduction_loss(
    fake: torch.Tensor, labels: torch.Tensor, teacher: torch.Tensor,
    classifier: MarginCosineClassifier, temperature: float = 4.0, weight: float = 30.0,
):
    lmc = F.cross_entropy(classifier(fake, labels), labels)
    ikt = ikt_loss(teacher, fake, classifier, temperature)
    return lmc + weight * ikt, {"lmc": lmc, "ikt": ikt}


def train_nocplace_reference(
    config: ExperimentConfig, *, smoke_iterations: int | None = None
) -> dict:
    repro = config.nocplace_repro
    manifest = Path(repro.generated_root).expanduser().resolve() / "manifest.csv"
    if smoke_iterations and not manifest.is_file():
        smoke_manifests = sorted(
            Path(repro.generated_root).expanduser().resolve().glob("smoke_limit_*/manifest.csv")
        )
        if smoke_manifests:
            manifest = smoke_manifests[-1]
    rows = _read_csv(manifest)
    if not smoke_iterations and len(rows) != 118_313:
        raise ValueError(f"Formal NocPlace training requires 118,313 translated rows, got {len(rows)}")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("NocPlace training refuses val/test rows")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache, cache_metadata = _cache_teacher(config, rows, device, manifest)
    dataset = _NocPlaceDataset(rows, cache)
    if not smoke_iterations and len(dataset.place_indices) != 29_860:
        raise ValueError(
            f"Formal NocPlace training requires 29,860 place classes, got {len(dataset.place_indices)}"
        )
    iterations = int(smoke_iterations or repro.vpr_iterations)
    sampler = PlaceBalancedSampler(dataset, iterations * repro.vpr_batch_size, repro.seed)
    loader = DataLoader(
        dataset, batch_size=repro.vpr_batch_size, sampler=sampler, num_workers=repro.workers,
        pin_memory=device.type == "cuda", drop_last=True,
    )
    student = _build_nocplace_model(config, repro.cosplace_checkpoint, device).train()
    classifier = MarginCosineClassifier(512, len(dataset.place_indices)).to(device)
    optimizer = torch.optim.Adam([
        {"params": student.parameters(), "lr": repro.model_lr},
        {"params": classifier.parameters(), "lr": repro.classifier_lr},
    ])
    output = Path(repro.output_dir).expanduser().resolve()
    if smoke_iterations:
        output = output / f"smoke_iterations_{iterations}"
    output.mkdir(parents=True, exist_ok=True)
    tracker = TrainingTracker(
        config, job_type="train-nocplace-reference", default_name="nocplace-r50-ikt-clean5k",
        output_dir=output, run_key=file_sha256(manifest),
    )
    started = time.perf_counter()
    latest = {}
    for step, batch in enumerate(loader, 1):
        fake = batch["fake"].to(device)
        labels = batch["label"].to(device)
        teacher = F.normalize(batch["teacher"].to(device), dim=1)
        optimizer.zero_grad(set_to_none=True)
        descriptors = F.normalize(student(fake), dim=1)
        loss, components = nocplace_reproduction_loss(
            descriptors, labels, teacher, classifier,
            repro.ikt_temperature, repro.ikt_weight,
        )
        loss.backward()
        optimizer.step()
        latest = {key: float(value.detach()) for key, value in components.items()}
        if step == 1 or step % config.tracking.log_every_steps == 0:
            elapsed = max(time.perf_counter() - started, 1e-9)
            tracker.log({
                "train/lmc": latest["lmc"], "train/ikt": latest["ikt"],
                "train/total": float(loss.detach()), "train/model_lr": repro.model_lr,
                "train/classifier_lr": repro.classifier_lr,
                "train/samples": step * repro.vpr_batch_size,
                "train/samples_per_second": step * repro.vpr_batch_size / elapsed,
                "train/teacher_entropy": float((-(F.softmax(classifier(teacher), 1) * F.log_softmax(classifier(teacher), 1)).sum(1).mean()).detach()),
                "train/student_entropy": float((-(F.softmax(classifier(descriptors), 1) * F.log_softmax(classifier(descriptors), 1)).sum(1).mean()).detach()),
            }, step=step)
        if step >= iterations:
            break
    inference_path = output / "best_model.pth"
    state_path = output / "training_state.pt"
    torch.save(student.state_dict(), inference_path)
    torch.save({
        "model_state_dict": student.state_dict(), "classifier_state_dict": classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "iterations": iterations,
        "teacher_cache": cache_metadata, "config_fingerprint": config.fingerprint(),
    }, state_path)
    result = {
        "checkpoint": str(inference_path), "training_state": str(state_path),
        "iterations": iterations, "samples": iterations * repro.vpr_batch_size,
        "classes": len(dataset.place_indices), "loss": latest,
        "checkpoint_sha256": file_sha256(inference_path), "teacher_cache": cache_metadata,
        "protocol_name": "NocPlace-protocol-clean5k-local",
    }
    write_json(result, output / "training_summary.json")
    tracker.summary({"nocplace/iterations": iterations, "nocplace/classes": len(dataset.place_indices)})
    tracker.finish()
    return result
