from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .config import DataConfig
from .paths import resolve_manifest_image_path, resolve_resource_path
from .transforms import PhotometricVPR, VPRTransform, patch_valid_mask

FIXED_RANDOM_STATE = 0


def optional_float(value) -> float:
    """Parse optional pose metadata without rejecting datasets that do not provide it."""
    try:
        return float(value) if value not in (None, "") else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def read_generator_manifest(path: str | Path, split: str = "train") -> list[dict[str, str]]:
    path = resolve_resource_path(path)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    audit_path = Path(path).with_suffix(".audit.json")
    if audit_path.is_file():
        invalid = set(json.loads(audit_path.read_text(encoding="utf-8")).get("invalid_ids", []))
        rows = [row for row in rows if row.get("sample_id", row.get("pair_id")) not in invalid]
    rows = [row for row in rows if row.get("valid", "1").lower() not in {"0", "false", "no"}]
    for row in rows:
        for key in ("source_path", "target_path"):
            if row.get(key):
                row[key] = resolve_manifest_image_path(row[key], path)
    return rows


def audit_data(config: DataConfig, *, decode: bool = False) -> dict:
    """Audit training manifests independently of retired custom test protocols."""
    generator_rows = read_generator_manifest(config.generator_manifest) if config.generator_manifest else []
    source_manifest = resolve_resource_path(config.vpr_source_manifest)
    source_rows = read_generated_manifest(source_manifest) if source_manifest.is_file() else []
    if any(row.get("split") != "train" for row in source_rows):
        raise ValueError("VPR source manifest contains non-train rows")
    image_paths = {row[key] for row in generator_rows for key in ("source_path", "target_path") if row.get(key)}
    image_paths.update(row["source_path"] for row in source_rows if row.get("source_path"))
    checked, missing = 0, []
    if decode:
        for value in sorted(image_paths):
            path = Path(value)
            if not path.is_file():
                missing.append(str(path))
                continue
            with Image.open(path) as image:
                image.verify()
            checked += 1
    invalid_pose_fields: dict[str, dict[str, int]] = {}
    generated_manifest = resolve_resource_path(config.generated_root) / "manifest.csv"
    if generated_manifest.is_file():
        for row in read_generated_manifest(generated_manifest):
            dataset = row.get("dataset", "unknown")
            for field in ("utm_e", "utm_n", "yaw"):
                raw = row.get(field)
                if raw in (None, ""):
                    continue
                try:
                    float(raw)
                except (TypeError, ValueError):
                    invalid_pose_fields.setdefault(dataset, {}).setdefault(field, 0)
                    invalid_pose_fields[dataset][field] += 1
    seasons_root = resolve_resource_path(config.robotcar_seasons_root)
    result = {
        "generator_train_rows": len(generator_rows),
        "generator_datasets": dict(Counter(row["dataset"] for row in generator_rows)),
        "vpr_train_rows": len(source_rows),
        "vpr_datasets": dict(Counter(row.get("dataset", "unknown") for row in source_rows)),
        "robotcar_seasons_root": str(seasons_root),
        "robotcar_seasons_available": (seasons_root / "images").is_dir(),
        "decoded": checked,
        "missing": missing,
        "invalid_pose_fields": invalid_pose_fields,
    }
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} training images; first={missing[0]}")
    return result


def read_generated_manifest(path: str | Path) -> list[dict[str, str]]:
    path = resolve_resource_path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("source_path", "generated_path"):
            if row.get(key):
                row[key] = resolve_manifest_image_path(row[key], path)
    return rows


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
        include_fake: bool = True,
    ):
        rows = self._read_manifest(manifest)
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
        self.include_fake = bool(include_fake)
        self.dataset_sampling_weights = dict(dataset_sampling_weights or {})
        if not self.rows:
            raise ValueError("No generated VPR rows survive the place-class filter")

    def __len__(self) -> int:
        return len(self.rows)

    def _read_manifest(self, manifest: str | Path) -> list[dict[str, str]]:
        return read_generated_manifest(manifest)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with Image.open(row["source_path"]) as day_file:
            day = self.transform(day_file)
        sample = {"day": day, **self._sample_metadata(index)}
        if self.include_fake:
            with Image.open(row["generated_path"]) as fake_file:
                sample["fake"] = self.transform(fake_file)
        if self.teacher_cache is not None:
            sample["teacher_descriptor"] = self.teacher_cache.descriptor(
                row["manifest_index"], row["image_id"]
            )
        return sample

    def _sample_metadata(self, index: int) -> dict:
        row = self.rows[index]
        return {
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
                [optional_float(row.get("utm_e")), optional_float(row.get("utm_n"))],
                dtype=torch.float32,
            ),
            "yaw_deg": torch.tensor(optional_float(row.get("yaw")), dtype=torch.float32),
        }

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


class StableNightVPRDataset(PairedVPRDataset):
    """Source-distinct place supervision with optional appearance-only views.

    Source-only modes consume the original VPR source manifest and never access
    generated images. Fake modes consume a generated manifest. Duplicate source
    images (including outputs from multiple generators) count only once before
    place filtering; the first compatible row supplies the fake view. Geometry
    masks mark padding, and deliberately make no Day/Fake alignment claim.

    Augmentation depends only on the configured seed, canonical source and epoch.
    Repeated source images within an epoch share one augmented view. The epoch
    tensor is shared with persistent workers so resumed runs reproduce these views.
    """

    MODES = {"day_only", "day_aug", "day_fake", "day_aug_fake"}

    def __init__(
        self,
        manifest: str | Path,
        transform: VPRTransform,
        min_class_images: int = 4,
        min_class_images_by_dataset: dict[str, int] | None = None,
        dataset_sampling_weights: dict[str, float] | None = None,
        datasets: tuple[str, ...] | None = None,
        train_domains: str = "day_only",
        random_seed: int = FIXED_RANDOM_STATE,
    ):
        if train_domains not in self.MODES:
            raise ValueError(f"StableNight train_domains must be one of {sorted(self.MODES)}")
        if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
            raise ValueError("StableNight random_seed must be a non-negative integer")
        self.train_domains = train_domains
        self.random_seed = random_seed
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self._stable_datasets = set(datasets or ())
        self.augmentation = PhotometricVPR()
        super().__init__(
            manifest,
            transform,
            min_class_images=min_class_images,
            min_class_images_by_dataset=min_class_images_by_dataset,
            dataset_sampling_weights=dataset_sampling_weights,
            datasets=datasets,
            include_fake=train_domains in {"day_fake", "day_aug_fake"},
        )

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("StableNight epoch must be a non-negative integer")
        self._epoch.fill_(epoch)

    def _augmentation_seed(self, source_key: int) -> int:
        identity = f"{self.random_seed}:{source_key}:{int(self._epoch.item())}"
        return int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)

    def _read_manifest(self, manifest: str | Path) -> list[dict[str, str]]:
        rows = read_generated_manifest(manifest)
        unique: dict[tuple[str, str], dict] = {}
        generated: dict[str, tuple[tuple[str, str], str]] = {}
        fake_mode = self.train_domains in {"day_fake", "day_aug_fake"}
        for row in rows:
            dataset = row.get("dataset", "unknown")
            if self._stable_datasets and dataset not in self._stable_datasets:
                continue
            missing = [key for key in ("source_path", "place_id", "image_id", "split") if not row.get(key)]
            if missing:
                raise ValueError(f"StableNight manifest row is missing {', '.join(missing)}")
            # Resolve aliases/symlinks using the same working-directory convention
            # as the original dataset. The generator's image ID is not identity.
            row["source_path"] = str(Path(row["source_path"]).expanduser().resolve())
            source_key = (dataset, row["source_path"])
            if fake_mode:
                if not row.get("generated_path"):
                    raise ValueError("StableNight fake modes require generated_path")
                row["generated_path"] = str(Path(row["generated_path"]).expanduser().resolve())
                fake_path = row["generated_path"]
                generator_hash = row.get("generator_sha256", "")
                if fake_path in generated:
                    previous_source, previous_hash = generated[fake_path]
                    if previous_source != source_key:
                        raise ValueError("A generated image cannot represent multiple source images")
                    if previous_hash and generator_hash and previous_hash != generator_hash:
                        raise ValueError("Conflicting generator hashes for the same generated image")
                    generator_hash = generator_hash or previous_hash
                generated[fake_path] = (source_key, generator_hash)
            if source_key in unique:
                previous = unique[source_key]
                if any(previous[key] != row[key] for key in ("place_id", "split")):
                    raise ValueError("The same source image has conflicting place labels or splits")
                continue
            unique[source_key] = row
        return list(unique.values())

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with Image.open(row["source_path"]) as day_file:
            day, day_valid_pixels = self.transform.transform_with_mask(day_file, patch_size=None)
        sample = {"day": day, **self._sample_metadata(index)}
        sample["image_key"] = torch.tensor(
            int(hashlib.sha1(
                f'{row.get("dataset", "unknown")}:{row["source_path"]}'.encode("utf-8")
            ).hexdigest()[:15], 16),
            dtype=torch.long,
        )
        sample["day_valid_mask"] = patch_valid_mask(day_valid_pixels)
        if self.train_domains in {"day_aug", "day_aug_fake"}:
            augmentation_seed = self._augmentation_seed(int(sample["image_key"].item()))
            sample["aug"] = self.augmentation(
                day, day_valid_pixels,
                rng=random.Random(augmentation_seed),
                torch_generator=torch.Generator(device=day.device).manual_seed(augmentation_seed),
            )
        if self.include_fake:
            with Image.open(row["generated_path"]) as fake_file:
                fake, fake_valid_mask = self.transform.transform_with_mask(fake_file)
            sample["fake"] = fake
            sample["fake_valid_mask"] = fake_valid_mask
        return sample


class PKBalancedBatchSampler(Sampler[list[int]]):
    """Deterministic single-dataset P-place/K-image batches with resumable cursor."""

    def __init__(
        self,
        rows: list[dict],
        quotas: dict[str, int],
        places_per_batch: int = 4,
        images_per_place: int = 2,
        random_seed: int = FIXED_RANDOM_STATE,
    ):
        self.rows = rows
        self.quotas = {str(key): int(value) for key, value in quotas.items()}
        self.places_per_batch = int(places_per_batch)
        self.images_per_place = int(images_per_place)
        self.random_seed = int(random_seed)
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
            rng = random.Random(self.random_seed + self.epoch * 1_000_003 + salt)
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
        random.Random(self.random_seed + self.epoch * 1_000_003 + 97).shuffle(batches)
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
            "random_seed": self.random_seed,
            "quotas": dict(self.quotas),
            "places_per_batch": self.places_per_batch,
            "images_per_place": self.images_per_place,
        }

    def load_state_dict(self, state: dict) -> None:
        expected = {
            "kind": "pk_balanced",
            "random_seed": self.random_seed,
            "quotas": self.quotas,
            "places_per_batch": self.places_per_batch,
            "images_per_place": self.images_per_place,
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(
                key,
                state.get("fixed_random_state", state.get("seed"))
                if key == "random_seed"
                else None,
            ) != value
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

    def __init__(
        self,
        rows: list[dict],
        quotas: dict[str, int],
        random_seed: int = FIXED_RANDOM_STATE,
    ):
        self.rows = rows
        self.quotas = {str(key): int(value) for key, value in quotas.items()}
        self.epoch = 0
        self.random_seed = int(random_seed)
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
        rng = random.Random(self.random_seed + salt + cycle * 1_000_003)
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
        random.Random(self.random_seed + self.epoch * 1_000_003 + 97).shuffle(output)
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

    def __init__(
        self,
        rows: list[dict],
        quotas: dict[str, int],
        random_seed: int = FIXED_RANDOM_STATE,
    ):
        self.rows = rows
        self.quotas = {str(key): int(value) for key, value in quotas.items()}
        self.epoch = 0
        self.random_seed = int(random_seed)
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
        rng = random.Random(self.random_seed + self.epoch * 1_000_003 + salt)
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
        rng = random.Random(self.random_seed + self.epoch * 1_000_003 + 97)
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


class ImagePathDataset(Dataset):
    def __init__(self, rows: Iterable, transform: VPRTransform):
        self.rows = list(rows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row.source_path) as image:
            return self.transform(image), index
