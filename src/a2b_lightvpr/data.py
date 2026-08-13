from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .config import DataConfig
from .transforms import GeneratorTrainTransform, VPRTransform

FIXED_RANDOM_STATE = 0


@dataclass(frozen=True)
class RobotCarRow:
    split: str
    role: str
    domain: str
    traverse: str
    timestamp: str
    easting: float
    northing: float
    yaw_deg: float
    source_path: str
    output_path: str

    @property
    def image_id(self) -> str:
        key = f"{self.traverse}:{self.timestamp}:{self.role}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]

    @property
    def place_id(self) -> str:
        x = math.floor(self.easting / 10.0)
        y = math.floor(self.northing / 10.0)
        heading = math.floor((self.yaw_deg % 360.0) / 30.0)
        return f"{x}:{y}:{heading}"


def read_robotcar_manifest(path: str | Path) -> list[RobotCarRow]:
    manifest = Path(path).expanduser().resolve()
    repository_root = manifest.parents[3]
    def resolve_source(value: str) -> str:
        source = Path(value)
        return str(source if source.is_absolute() else (repository_root / source).resolve())
    with manifest.open(newline="", encoding="utf-8") as handle:
        return [
            RobotCarRow(
                split=row["split"],
                role=row["role"],
                domain=row["domain"],
                traverse=row["traverse"],
                timestamp=row["timestamp"],
                easting=float(row["easting"]),
                northing=float(row["northing"]),
                yaw_deg=float(row["yaw_deg"]),
                source_path=resolve_source(row["source_path"]),
                output_path=row["output_path"],
            )
            for row in csv.DictReader(handle)
        ]


def read_generator_manifest(path: str | Path, split: str = "train") -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    audit_path = Path(path).with_suffix(".audit.json")
    if audit_path.is_file():
        invalid = set(json.loads(audit_path.read_text(encoding="utf-8")).get("invalid_ids", []))
        rows = [row for row in rows if row.get("sample_id", row.get("pair_id")) not in invalid]
    rows = [row for row in rows if row.get("valid", "1").lower() not in {"0", "false", "no"}]
    return rows


def audit_data(config: DataConfig, *, decode: bool = False) -> dict:
    rows = read_robotcar_manifest(config.robotcar_manifest)
    protocol = json.loads(Path(config.robotcar_protocol).read_text(encoding="utf-8"))
    traverses: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        traverses[split] = {row.traverse for row in rows if row.split == split}
    overlaps = {
        f"{a}_{b}": sorted(traverses[a] & traverses[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(overlaps.values()):
        raise ValueError(f"RobotCar traversal leakage detected: {overlaps}")
    counts = Counter((row.split, row.role, row.domain) for row in rows)
    missing: list[str] = []
    checked = 0
    if decode:
        for row in rows:
            path = Path(row.source_path)
            if not path.is_file():
                missing.append(str(path))
                continue
            with Image.open(path) as image:
                image.verify()
            checked += 1
    generator_rows = read_generator_manifest(config.generator_manifest)
    result = {
        "robotcar_counts": {"/".join(key): value for key, value in sorted(counts.items())},
        "generator_train_rows": len(generator_rows),
        "generator_datasets": dict(Counter(row["dataset"] for row in generator_rows)),
        "traverses": {key: sorted(value) for key, value in traverses.items()},
        "overlaps": overlaps,
        "decoded": checked,
        "missing": missing,
        "protocol_dataset": protocol.get("dataset"),
    }
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} RobotCar images; first={missing[0]}")
    return result


class BalancedGeneratorDataset(Dataset):
    """Dataset-balanced, unpaired day/night sampling for formal QS-Attn training."""

    def __init__(self, config: DataConfig, length: int):
        rows = read_generator_manifest(config.generator_manifest)
        grouped: dict[str, dict[str, list]] = {}
        for row in rows:
            pools = grouped.setdefault(row["dataset"], {"day": [], "night": []})
            if row.get("source_path"):
                pools["day"].append(row["source_path"])
            if row.get("target_path"):
                pools["night"].append(row["target_path"])
        self.grouped = {
            name: pools for name, pools in grouped.items() if pools["day"] and pools["night"]
        }
        if not self.grouped:
            raise ValueError("Generator manifest has no dataset with both day and night images")
        configured = config.dataset_weights or {name: 1.0 for name in self.grouped}
        self.datasets = [name for name in sorted(self.grouped) if name in configured]
        self.weights = [float(configured[name]) for name in self.datasets]
        if not self.datasets:
            raise ValueError("No configured dataset weights match the generator manifest")
        self.length = int(length)
        self.transform = GeneratorTrainTransform(config.load_short_side, config.crop_size)

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _open(path: str) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def __getitem__(self, index: int) -> dict:
        rng = random.Random(FIXED_RANDOM_STATE + int(index) * 1_000_003)
        dataset = rng.choices(self.datasets, weights=self.weights, k=1)[0]
        pools = self.grouped[dataset]
        day_path, night_path = rng.choice(pools["day"]), rng.choice(pools["night"])
        source = self.transform(self._open(day_path), rng=rng)
        target = self.transform(self._open(night_path), rng=rng)
        return {
            "source": source,
            "target": target,
            "source_path": day_path,
            "target_path": night_path,
            "dataset": dataset,
        }


def read_generated_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class PairedVPRDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        transform: VPRTransform,
        min_class_images: int = 4,
        min_class_images_by_dataset: dict[str, int] | None = None,
        dataset_sampling_weights: dict[str, float] | None = None,
        datasets: tuple[str, ...] | None = None,
        teacher_cache=None,
    ):
        rows = read_generated_manifest(manifest)
        for manifest_index, row in enumerate(rows):
            row["manifest_index"] = manifest_index
        included = set(datasets or ())
        if included:
            rows = [row for row in rows if row.get("dataset") in included]
        invalid_splits = sorted({row.get("split", "") for row in rows} - {"train"})
        if invalid_splits:
            raise ValueError(
                f"Generated VPR manifest contains non-train splits: {invalid_splits}"
            )
        for row in rows:
            row["global_place_id"] = f'{row.get("dataset", "unknown")}:{row["place_id"]}'
        counts = Counter(row["global_place_id"] for row in rows)
        minimums = min_class_images_by_dataset or {}
        self.rows = [
            row for row in rows
            if counts[row["global_place_id"]] >= minimums.get(
                row.get("dataset", "unknown"), min_class_images
            )
        ]
        places = sorted({row["global_place_id"] for row in self.rows})
        self.place_to_label = {place: index for index, place in enumerate(places)}
        self.transform = transform
        self.teacher_cache = teacher_cache
        self.dataset_sampling_weights = dict(dataset_sampling_weights or {})
        if not self.rows:
            raise ValueError("No generated VPR rows survive the place-class filter")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with Image.open(row["source_path"]) as day_file:
            day = self.transform(day_file)
        with Image.open(row["generated_path"]) as fake_file:
            fake = self.transform(fake_file)
        sample = {
            "day": day,
            "fake": fake,
            "label": torch.tensor(self.place_to_label[row["global_place_id"]], dtype=torch.long),
            "image_id": row["image_id"],
            "image_key": torch.tensor(
                int(
                    hashlib.sha1(
                        f'{row.get("dataset", "unknown")}:{row["image_id"]}'.encode("utf-8")
                    ).hexdigest()[:15],
                    16,
                ),
                dtype=torch.long,
            ),
            "dataset": row.get("dataset", "unknown"),
            "manifest_index": torch.tensor(row["manifest_index"], dtype=torch.long),
            "utm_xy": torch.tensor(
                [float(row.get("utm_e") or "nan"), float(row.get("utm_n") or "nan")],
                dtype=torch.float32,
            ),
            "yaw_deg": torch.tensor(float(row.get("yaw") or "nan"), dtype=torch.float32),
        }
        if self.teacher_cache is not None:
            sample["teacher_descriptor"] = self.teacher_cache.descriptor(
                row["manifest_index"], row["image_id"]
            )
        return sample

    def balanced_weights(self) -> torch.Tensor:
        """Equal total probability per dataset, then per place within a dataset."""
        dataset_places: dict[str, set[str]] = {}
        place_counts = Counter()
        for row in self.rows:
            dataset_places.setdefault(row.get("dataset", "unknown"), set()).add(row["global_place_id"])
            place_counts[row["global_place_id"]] += 1
        if self.dataset_sampling_weights:
            if set(self.dataset_sampling_weights) != set(dataset_places):
                raise ValueError("VPR dataset sampling weights must name every available dataset")
            total = sum(self.dataset_sampling_weights.values())
            dataset_weights = {
                name: self.dataset_sampling_weights[name] / total for name in dataset_places
            }
        else:
            dataset_weights = {name: 1.0 / len(dataset_places) for name in dataset_places}
        weights = []
        for row in self.rows:
            dataset = row.get("dataset", "unknown")
            weights.append(
                dataset_weights[dataset]
                / (len(dataset_places[dataset]) * place_counts[row["global_place_id"]])
            )
        return torch.tensor(weights, dtype=torch.double)


class PKBalancedBatchSampler(Sampler[list[int]]):
    """Deterministic single-dataset P-place/K-image batches with resumable cursor."""

    def __init__(
        self,
        rows: list[dict],
        quotas: dict[str, int],
        places_per_batch: int = 4,
        images_per_place: int = 2,
    ):
        self.rows = rows
        self.quotas = {str(key): int(value) for key, value in quotas.items()}
        self.places_per_batch = int(places_per_batch)
        self.images_per_place = int(images_per_place)
        self.batch_size = self.places_per_batch * self.images_per_place
        self.epoch = 0
        self.cursor = 0
        self.by_place: dict[str, dict[str, list[int]]] = {}
        for index, row in enumerate(rows):
            dataset = row.get("dataset", "unknown")
            place = row["global_place_id"]
            self.by_place.setdefault(dataset, {}).setdefault(place, []).append(index)
        if set(self.quotas) != set(self.by_place):
            raise ValueError("PK quotas must name every available dataset")
        if self.batch_size < 1:
            raise ValueError("PK batch size must be positive")
        for dataset, places in self.by_place.items():
            if len(places) < self.places_per_batch:
                raise ValueError(
                    f"PK sampler needs {self.places_per_batch} places in {dataset}, got {len(places)}"
                )
            too_small = [place for place, values in places.items() if len(values) < self.images_per_place]
            if too_small:
                raise ValueError(
                    f"PK sampler needs {self.images_per_place} distinct images per place; "
                    f"{dataset} has {len(too_small)} undersized places"
                )
            if self.quotas[dataset] <= 0 or self.quotas[dataset] % self.batch_size:
                raise ValueError("Every PK quota must be positive and divisible by P*K")
        self._batches: list[list[int]] | None = None

    def __len__(self) -> int:
        return sum(self.quotas.values()) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch != self.epoch:
            self.epoch = epoch
            self.cursor = 0
            self._batches = None

    def _epoch_batches(self) -> list[list[int]]:
        if self._batches is not None:
            return self._batches
        batches: list[list[int]] = []
        for dataset in sorted(self.quotas):
            salt = int(hashlib.sha1(dataset.encode("utf-8")).hexdigest()[:8], 16)
            rng = random.Random(FIXED_RANDOM_STATE + self.epoch * 1_000_003 + salt)
            places = sorted(self.by_place[dataset])
            count = self.quotas[dataset] // self.batch_size
            place_cycle: list[str] = []
            for _ in range(count):
                if len(place_cycle) < self.places_per_batch:
                    refill = list(places)
                    rng.shuffle(refill)
                    place_cycle.extend(refill)
                selected = place_cycle[: self.places_per_batch]
                del place_cycle[: self.places_per_batch]
                # A cycle boundary can duplicate a place; repair deterministically.
                if len(set(selected)) != self.places_per_batch:
                    available = [place for place in places if place not in selected]
                    for position in range(self.places_per_batch):
                        if selected.count(selected[position]) > 1:
                            selected[position] = available.pop(0)
                batch = []
                for place in selected:
                    batch.extend(rng.sample(self.by_place[dataset][place], self.images_per_place))
                batches.append(batch)
        random.Random(FIXED_RANDOM_STATE + self.epoch * 1_000_003 + 97).shuffle(batches)
        self._batches = batches
        return batches

    def __iter__(self):
        batches = self._epoch_batches()
        while self.cursor < len(batches):
            batch = batches[self.cursor]
            self.cursor += 1
            yield list(batch)

    def state_dict(self) -> dict:
        return {
            "kind": "pk_balanced",
            "epoch": self.epoch,
            "cursor": self.cursor,
            "fixed_random_state": FIXED_RANDOM_STATE,
            "quotas": dict(self.quotas),
            "places_per_batch": self.places_per_batch,
            "images_per_place": self.images_per_place,
        }

    def load_state_dict(self, state: dict) -> None:
        expected = {
            "kind": "pk_balanced",
            "fixed_random_state": FIXED_RANDOM_STATE,
            "quotas": self.quotas,
            "places_per_batch": self.places_per_batch,
            "images_per_place": self.images_per_place,
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if (
                state.get(key, state.get("seed") if key == "fixed_random_state" else None)
                != value
            )
        }
        if mismatches:
            raise ValueError(f"PK sampler state does not match: {mismatches}")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self._batches = None
        if not 0 <= self.cursor <= len(self):
            raise ValueError("PK sampler cursor is outside the epoch")

    def statistics(self) -> dict[str, dict[str, float | int]]:
        return {
            dataset: {
                "quota": quota,
                "batches": quota // self.batch_size,
                "places": len(self.by_place[dataset]),
                "places_per_batch": self.places_per_batch,
                "images_per_place": self.images_per_place,
            }
            for dataset, quota in self.quotas.items()
        }


class RollingBalancedSampler(Sampler[int]):
    """Take deterministic rolling windows from place-balanced dataset cycles."""

    def __init__(self, rows: list[dict], quotas: dict[str, int]):
        self.rows = rows
        self.quotas = {str(key): int(value) for key, value in quotas.items()}
        self.epoch = 0
        self.by_dataset: dict[str, list[int]] = {}
        self.by_place: dict[str, dict[str, list[int]]] = {}
        for index, row in enumerate(rows):
            dataset = row.get("dataset", "unknown")
            place = row["global_place_id"]
            self.by_dataset.setdefault(dataset, []).append(index)
            self.by_place.setdefault(dataset, {}).setdefault(place, []).append(index)
        if set(self.quotas) != set(self.by_dataset):
            raise ValueError("Rolling quotas must name every available dataset")
        self._last_epoch_indices: dict[str, list[int]] = {}
        self._cycles: dict[str, list[list[int]]] = {
            dataset: [] for dataset in self.quotas
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return sum(self.quotas.values())

    def _new_cycle(self, dataset: str, cycle: int) -> list[int]:
        salt = int(hashlib.sha1(dataset.encode()).hexdigest()[:8], 16)
        rng = random.Random(FIXED_RANDOM_STATE + salt + cycle * 1_000_003)
        places = sorted(self.by_place[dataset])
        rng.shuffle(places)
        images = {place: list(self.by_place[dataset][place]) for place in places}
        for values in images.values():
            rng.shuffle(values)
        offsets = {place: 0 for place in places}
        output = []
        while len(output) < len(self.by_dataset[dataset]):
            for place in places:
                offset = offsets[place]
                if offset < len(images[place]):
                    output.append(images[place][offset])
                    offsets[place] = offset + 1
        return output

    def _cycle(self, dataset: str, cycle: int) -> list[int]:
        """Return a cached deterministic full-dataset permutation."""
        cached = self._cycles[dataset]
        while len(cached) <= cycle:
            cycle_index = len(cached)
            cached.append(self._new_cycle(dataset, cycle_index))
        return cached[cycle]

    def _window(self, dataset: str, epoch: int) -> list[int]:
        row_count = len(self.by_dataset[dataset])
        quota = self.quotas[dataset]
        offset = epoch * quota
        cycle, position = divmod(offset, row_count)
        output = []
        seen: set[int] = set()
        while len(output) < quota:
            values = self._cycle(dataset, cycle)
            for value in values[position:]:
                if value in seen:
                    continue
                output.append(value)
                seen.add(value)
                if len(output) == quota:
                    break
                if len(seen) == row_count:
                    # Oversubscribed datasets repeat only after full coverage.
                    seen.clear()
            cycle += 1
            position = 0
        if quota <= row_count and len(set(output)) != len(output):
            raise RuntimeError(f"Rolling sampler repeated a {dataset} image within one epoch")
        return output

    def __iter__(self):
        by_dataset = {
            dataset: self._window(dataset, self.epoch) for dataset in sorted(self.quotas)
        }
        self._last_epoch_indices = by_dataset
        output = [index for values in by_dataset.values() for index in values]
        random.Random(FIXED_RANDOM_STATE + self.epoch * 1_000_003 + 97).shuffle(output)
        return iter(output)

    def statistics(self) -> dict[str, dict[str, float | int]]:
        statistics = {}
        for dataset, quota in self.quotas.items():
            cumulative = set()
            for epoch in range(self.epoch + 1):
                cumulative.update(self._window(dataset, epoch))
            current = self._last_epoch_indices.get(dataset, self._window(dataset, self.epoch))
            total = (self.epoch + 1) * quota
            statistics[dataset] = {
                "quota": quota,
                "epoch_unique_images": len(set(current)),
                "cumulative_unique_images": len(cumulative),
                "image_coverage_ratio": len(cumulative) / len(self.by_dataset[dataset]),
                "place_coverage_ratio": len({self.rows[i]["global_place_id"] for i in cumulative})
                / len(self.by_place[dataset]),
                "repeat_factor": total / max(1, len(cumulative)),
            }
        return statistics

class CoverageBalancedSampler(Sampler[int]):
    """Deterministic full coverage followed by place-balanced quota filling."""

    def __init__(self, rows: list[dict], quotas: dict[str, int]):
        self.rows = rows
        self.quotas = {str(key): int(value) for key, value in quotas.items()}
        self.epoch = 0
        available = {row.get("dataset", "unknown") for row in rows}
        if set(self.quotas) != available:
            raise ValueError("Coverage quotas must name every available dataset")
        self.by_dataset: dict[str, list[int]] = {}
        self.by_place: dict[str, dict[str, list[int]]] = {}
        for index, row in enumerate(rows):
            dataset = row.get("dataset", "unknown")
            place = row["global_place_id"]
            self.by_dataset.setdefault(dataset, []).append(index)
            self.by_place.setdefault(dataset, {}).setdefault(place, []).append(index)
        for dataset, indices in self.by_dataset.items():
            if self.quotas[dataset] < len(indices):
                raise ValueError(
                    f"Coverage quota for {dataset} is {self.quotas[dataset]}, "
                    f"below its {len(indices)} rows"
                )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return sum(self.quotas.values())

    def _dataset_indices(self, dataset: str) -> list[int]:
        salt = int(hashlib.sha1(dataset.encode()).hexdigest()[:8], 16)
        rng = random.Random(FIXED_RANDOM_STATE + self.epoch * 1_000_003 + salt)
        covered = list(self.by_dataset[dataset])
        rng.shuffle(covered)
        output = covered
        remaining = self.quotas[dataset] - len(output)
        places = sorted(self.by_place[dataset])
        rng.shuffle(places)
        image_cycles = {place: list(self.by_place[dataset][place]) for place in places}
        image_offsets = {place: 0 for place in places}
        for values in image_cycles.values():
            rng.shuffle(values)
        place_offset = 0
        while remaining:
            place = places[place_offset % len(places)]
            values = image_cycles[place]
            offset = image_offsets[place]
            if offset and offset % len(values) == 0:
                rng.shuffle(values)
            output.append(values[offset % len(values)])
            image_offsets[place] = offset + 1
            place_offset += 1
            remaining -= 1
        return output

    def __iter__(self):
        indices = []
        for dataset in sorted(self.quotas):
            indices.extend(self._dataset_indices(dataset))
        rng = random.Random(FIXED_RANDOM_STATE + self.epoch * 1_000_003 + 97)
        rng.shuffle(indices)
        return iter(indices)

    def statistics(self) -> dict[str, dict[str, float | int]]:
        return {
            dataset: {
                "quota": quota,
                "unique_images": len(self.by_dataset[dataset]),
                "coverage_ratio": 1.0,
                "repeat_factor": quota / len(self.by_dataset[dataset]),
            }
            for dataset, quota in self.quotas.items()
        }


def benchmark_rows(manifest: str | Path, split: str) -> tuple[list[RobotCarRow], list[RobotCarRow]]:
    if split not in {"val", "test"}:
        raise ValueError("Benchmark evaluation is restricted to val or test")
    rows = read_robotcar_manifest(manifest)
    database = [row for row in rows if row.split == split and row.role == "database"]
    queries = [row for row in rows if row.split == split and row.role == "queries"]
    if not database or not queries:
        raise ValueError(f"Incomplete benchmark split: {split}")
    return database, queries


class ImagePathDataset(Dataset):
    def __init__(self, rows: Iterable[RobotCarRow], transform: VPRTransform):
        self.rows = list(rows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row.source_path) as image:
            return self.transform(image), index
