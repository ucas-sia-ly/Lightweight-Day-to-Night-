"""Source-distinct GSV-Cities data for standard place metric learning."""
from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image

from .checkpointing import file_sha256, write_json
from .data import PKBalancedBatchSampler, PairedVPRDataset
from .paths import resolve_manifest_image_path, resolve_resource_path
from .transforms import PhotometricVPR, VPRTransform


MANIFEST_FIELDS = (
    "image_id", "dataset", "split", "domain", "place_id", "route", "source_path",
    "latitude", "longitude", "yaw", "year", "month", "panoid", "role",
)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def prepare_gsv_metric(
    root: str | Path,
    output_dir: str | Path = "data/manifests_gsv_metric",
    *,
    min_class_images: int = 4,
) -> dict:
    """Write an audited training manifest from the original GSV place IDs.

    Reuses the existing reference-training reader and naming convention. Only
    GSV Dataframes/Images are read; generated images and benchmark splits never
    participate. Missing eligible images or conflicting source ownership abort
    preparation before the output manifest is written.
    """
    from .references.training import _gsv_image_name, _gsv_rows_cached, gsv_rows

    _positive_integer(min_class_images, "min_class_images")
    if min_class_images < 4:
        raise ValueError("GSV metric training requires at least four distinct sources per place")
    root = Path(root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.is_relative_to(root):
        raise ValueError("Metric manifests cannot overwrite the GSV source dataset")
    tables = sorted((root / "Dataframes").glob("*.csv"))
    if not tables:
        raise FileNotFoundError(f"GSV metadata CSVs are missing: {root / 'Dataframes'}")
    # Preparation audits current CSV bytes, rather than a prior reader cache
    # produced before a dataset update in this process.
    _gsv_rows_cached.cache_clear()
    grouped = gsv_rows(root)
    eligible_groups = {(str(item["city"]), int(item["place_id"])) for item in grouped}
    # Read original metadata so latitude/longitude and capture identity are
    # preserved rather than reconstructed from ambiguous underscore filenames.
    metadata = {}
    raw_rows, raw_places = 0, set()
    for table in tables:
        city = table.stem
        with table.open(newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                place_id = int(record["place_id"])
                key = (city, place_id)
                raw_rows += 1
                raw_places.add(key)
                if key not in eligible_groups:
                    continue
                image = root / "Images" / city / _gsv_image_name(city, place_id, record)
                metadata[str(image)] = record
    retained = {}
    place_sources = defaultdict(set)
    missing, duplicate_rows = [], 0
    for group in grouped:
        city, place_id = str(group["city"]), int(group["place_id"])
        label = f"{city}:{place_id}"
        for value in group["paths"]:
            path = Path(value).resolve()
            canonical = str(path)
            if not path.is_file():
                missing.append(canonical)
                continue
            if canonical in retained:
                if retained[canonical]["place_id"] != label:
                    raise ValueError(f"A GSV source image has conflicting place labels: {path}")
                duplicate_rows += 1
                continue
            original = metadata[str(value)]
            # Relative identity remains stable when the complete GSV dataset is
            # moved. A source symlink outside its root retains canonical identity.
            identity = path.relative_to(root).as_posix() if path.is_relative_to(root) else canonical
            image_id = hashlib.sha256(identity.encode()).hexdigest()[:32]
            latitude, longitude = original.get("lat", ""), original.get("lon", "")
            for field, coordinate, limit in (("latitude", latitude, 90), ("longitude", longitude, 180)):
                if coordinate and (not math.isfinite(float(coordinate)) or abs(float(coordinate)) > limit):
                    raise ValueError(f"Invalid GSV {field} in {path}")
            yaw = original.get("northdeg", "")
            if yaw:
                yaw = str(float(yaw) % 360.0)
            retained[canonical] = {
                "image_id": image_id, "dataset": "gsv_cities", "split": "train", "domain": "day",
                "place_id": label, "route": city, "source_path": canonical,
                "latitude": latitude, "longitude": longitude, "yaw": yaw,
                "year": original.get("year", ""), "month": original.get("month", ""),
                "panoid": original.get("panoid", ""), "role": "database",
            }
            place_sources[label].add(canonical)
    if missing:
        raise FileNotFoundError(f"Missing {len(set(missing))} GSV training sources; examples={sorted(set(missing))[:3]}")
    eligible = {label for label, paths in place_sources.items() if len(paths) >= min_class_images}
    rows = sorted((row for row in retained.values() if row["place_id"] in eligible),
                  key=lambda row: (row["place_id"], row["source_path"]))
    if not rows:
        raise ValueError("No GSV places have four distinct accessible training sources")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("GSV source image identity collision")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "vpr_train.csv"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=output,
                                         prefix=".vpr_train.", delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, manifest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    from .references import training as training_module
    counts = Counter(row["place_id"] for row in rows)
    city_places = Counter(label.rsplit(":", 1)[0] for label in counts)
    audit = {
        "dataset": "gsv_cities", "split": "train", "source_root": str(root),
        "manifest": str(manifest), "manifest_sha256": file_sha256(manifest),
        "raw_metadata_rows": raw_rows, "raw_places": len(raw_places),
        "rows": len(rows), "places": len(counts), "distinct_sources": len(rows),
        "min_class_images": min_class_images, "duplicate_source_rows_removed": duplicate_rows,
        "excluded_places": len(raw_places) - len(counts),
        "images_per_place_histogram": dict(sorted(Counter(counts.values()).items())),
        "images_by_city": dict(sorted(Counter(row["route"] for row in rows).items())),
        "places_by_city": dict(sorted(city_places.items())),
        "minimum_samples_for_city_pk_32x4": sum(math.ceil(value / 32) * 128 for value in city_places.values()),
        "metadata_csv_sha256": {table.name: file_sha256(table) for table in tables},
        "reader": "a2b_lightvpr.references.training.gsv_rows",
        "reader_code_sha256": file_sha256(training_module.__file__),
        "preparation_code_sha256": file_sha256(__file__),
        "upstream_dataset": "https://github.com/amaralibey/gsv-cities",
        "label_definition": "original GSV-Cities city and place_id, distinct source images",
        "geometry": "GPS latitude/longitude retained; no invented UTM or benchmark labels",
        "benchmark_splits_read": [], "generated_images_read": False,
        "image_checks": "file existence; image decoding is deferred to training",
    }
    write_json(audit, output / "audit.json")
    return audit


class MetricVPRDataset(PairedVPRDataset):
    """One original or augmented image per distinct source, with place labels.

    Keeps the existing PK-sampler metadata contract. Source duplicates are
    removed before class-size filtering, so four CSV aliases of one photograph
    cannot supply four positives. Appearance RNGs depend only on seed, epoch and
    canonical source identity and are shared consistently with worker processes.
    """

    def __init__(
        self, manifest: str | Path, transform: VPRTransform, min_class_images: int = 4,
        min_class_images_by_dataset: dict[str, int] | None = None,
        dataset_sampling_weights: dict[str, float] | None = None,
        datasets: tuple[str, ...] | None = None, *, train_domains: str = "day_aug",
        random_seed: int = 0, augmentation_probability: float = 0.5,
    ):
        if train_domains not in {"day_only", "day_aug"}:
            raise ValueError("Metric VPR train_domains must be day_only or day_aug")
        if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
            raise ValueError("Metric VPR random_seed must be a non-negative integer")
        if isinstance(augmentation_probability, bool) or not isinstance(augmentation_probability, (float, int)) or not 0 <= augmentation_probability <= 1:
            raise ValueError("Metric augmentation probability must be in [0, 1]")
        _positive_integer(min_class_images, "min_class_images")
        if min_class_images < 4 or any(value < 4 for value in (min_class_images_by_dataset or {}).values()):
            raise ValueError("Metric VPR requires at least four distinct sources per place")
        self.train_domains = train_domains
        self.random_seed = random_seed
        self.augmentation_probability = float(augmentation_probability)
        self._metric_datasets = set(datasets or ())
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.augmentation = PhotometricVPR(exposure_stops=(-1.5, 0.3))
        try:
            super().__init__(manifest, transform, min_class_images, min_class_images_by_dataset,
                             dataset_sampling_weights, datasets, include_fake=False)
        except ValueError as error:
            if str(error) == "No generated VPR rows survive the place-class filter":
                raise ValueError("No metric places survive the four-distinct-source class filter") from error
            raise
        for row in self.rows:
            row["global_place_id"] = sys.intern(row["global_place_id"])
        self.source_keys = [int.from_bytes(hashlib.sha256(
            f'{row.get("dataset", "unknown")}:{row["source_path"]}'.encode()).digest()[:8], "big") % (2**63 - 1)
            for row in self.rows]

    def _read_manifest(self, manifest):
        unique, ids = {}, {}
        manifest = resolve_resource_path(manifest)
        with Path(manifest).open(newline="", encoding="utf-8") as handle:
            return self._read_metric_rows(csv.DictReader(handle), manifest, unique, ids)

    def _read_metric_rows(self, records, manifest, unique, ids):
        # Keep the full capture/GPS provenance in the CSV, but retain only
        # fields used by training. Stream records instead of materializing a
        # second list of 500k full metadata dictionaries during initialization.
        fields = ("image_id", "dataset", "split", "place_id", "route", "source_path", "yaw")
        for original in records:
            row = {key: original[key] for key in fields if key in original}
            dataset = row.get("dataset", "unknown")
            if self._metric_datasets and dataset not in self._metric_datasets:
                continue
            required = ("source_path", "place_id", "image_id", "split")
            if any(not row.get(key) for key in required):
                raise ValueError("Metric manifest requires source_path, place_id, image_id and split")
            if row["split"] != "train":
                raise ValueError("Metric source manifest contains non-train rows")
            row["source_path"] = str(Path(resolve_manifest_image_path(row["source_path"], manifest)).expanduser().resolve())
            for name in ("dataset", "split", "place_id", "route"):
                if name in row:
                    row[name] = sys.intern(row[name])
            source = row["source_path"]
            if source in unique:
                if (unique[source].get("dataset"), unique[source]["place_id"]) != (dataset, row["place_id"]):
                    raise ValueError("A metric source image has conflicting place labels")
                continue
            key = (dataset, row["image_id"])
            if key in ids and ids[key] != source:
                raise ValueError("A metric image_id refers to multiple source images")
            ids[key] = source
            unique[source] = row
        return list(unique.values())

    def set_epoch(self, epoch: int):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Metric dataset epoch must be a non-negative integer")
        self._epoch.fill_(epoch)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["source_path"]) as image:
            tensor, valid_pixels = self.transform.transform_with_mask(image, patch_size=None)
        identity = f"{self.random_seed}:{self.source_keys[index]}:{int(self._epoch.item())}"
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") % (2**63 - 1)
        rng = random.Random(seed)
        augmented = self.train_domains == "day_aug" and rng.random() < self.augmentation_probability
        if augmented:
            generator = torch.Generator().manual_seed(seed)
            tensor = self.augmentation(tensor, valid_pixels, rng=rng, torch_generator=generator)
        metadata = self._sample_metadata(index)
        metadata["image_key"] = torch.tensor(self.source_keys[index], dtype=torch.long)
        return {"image": tensor, "is_augmented": torch.tensor(augmented), **metadata}


class MetricCityBatchSampler(PKBalancedBatchSampler):
    """Existing PK sampling within cities, with full place coverage each epoch.

    Each city first receives enough batches to visit every place. Remaining
    batches are allocated by its share of place labels, using deterministic
    largest remainders. Every shuffled city cycle is partitioned into P-place
    chunks; its final short chunk is padded from previously visited places.
    The existing PK sampler supplies metadata and checkpoint lifecycle support.
    """

    def __init__(self, rows, quotas, places_per_batch=32, images_per_place=4, random_seed=0):
        _positive_integer(places_per_batch, "places_per_batch")
        _positive_integer(images_per_place, "images_per_place")
        for value in quotas.values():
            _positive_integer(value, "city quota")
        if set(quotas) != {"gsv_cities"}:
            raise ValueError("Within-city metric batches require a single gsv_cities quota")
        super().__init__(rows, quotas, places_per_batch, images_per_place, random_seed)
        self.by_city = {}
        place_city, seen_sources = {}, set()
        partition_digest = hashlib.sha256()
        for index, row in enumerate(rows):
            city = row.get("route")
            source = row.get("source_path")
            if not city or not source:
                raise ValueError("Within-city metric sampling requires route(city) and source_path")
            if source in seen_sources:
                raise ValueError("Within-city metric sampling requires distinct canonical source rows")
            seen_sources.add(source)
            place = row["global_place_id"]
            if place in place_city and place_city[place] != city:
                raise ValueError("A metric place label belongs to multiple cities")
            place_city[place] = city
            self.by_city.setdefault(city, {}).setdefault(place, []).append(index)
            partition_digest.update(f"{index}\0{city}\0{place}\0{source}\n".encode())
        if any(len(places) < self.places_per_batch for places in self.by_city.values()):
            raise ValueError("Every metric city must contain P distinct places")
        self.city_partition_fingerprint = partition_digest.hexdigest()
        minimum = {city: math.ceil(len(places) / self.places_per_batch)
                   for city, places in self.by_city.items()}
        self.minimum_coverage_samples = sum(minimum.values()) * self.batch_size
        batches = sum(self.quotas.values()) // self.batch_size
        extra = batches - sum(minimum.values())
        if extra < 0:
            raise ValueError(f"Within-city full place coverage needs at least {self.minimum_coverage_samples} samples per epoch")
        total_places = sum(len(places) for places in self.by_city.values())
        share = {city: extra * len(places) / total_places for city, places in self.by_city.items()}
        allocation = {city: math.floor(value) for city, value in share.items()}
        remaining = extra - sum(allocation.values())
        for city in sorted(share, key=lambda key: (-(share[key] - allocation[key]), key))[:remaining]:
            allocation[city] += 1
        self.city_quotas = {city: (minimum[city] + allocation[city]) * self.batch_size
                            for city in sorted(minimum)}

    def _epoch_batches(self):
        if self._batches is not None:
            return self._batches
        batches = []
        for city in sorted(self.city_quotas):
            salt = int(hashlib.sha1(city.encode()).hexdigest()[:8], 16)
            rng = random.Random(self.random_seed + self.epoch * 1_000_003 + salt)
            places = sorted(self.by_city[city])
            target = self.city_quotas[city] // self.batch_size
            city_batches = []
            while len(city_batches) < target:
                cycle = list(places)
                rng.shuffle(cycle)
                for start in range(0, len(cycle), self.places_per_batch):
                    selected = cycle[start:start + self.places_per_batch]
                    if len(selected) < self.places_per_batch:
                        selected += rng.sample(cycle[:start], self.places_per_batch - len(selected))
                    batch = []
                    for place in selected:
                        batch.extend(rng.sample(self.by_city[city][place], self.images_per_place))
                    city_batches.append(batch)
                    if len(city_batches) == target:
                        break
            batches.extend(city_batches)
        random.Random(self.random_seed + self.epoch * 1_000_003 + 97).shuffle(batches)
        self._batches = batches
        return batches

    def state_dict(self):
        state = super().state_dict()
        state.update(kind="metric_city_pk", city_partition_fingerprint=self.city_partition_fingerprint,
                     city_quotas=dict(self.city_quotas), minimum_coverage_samples=self.minimum_coverage_samples)
        return state

    def load_state_dict(self, state):
        expected = {
            "kind": "metric_city_pk", "city_partition_fingerprint": self.city_partition_fingerprint,
            "city_quotas": self.city_quotas, "minimum_coverage_samples": self.minimum_coverage_samples,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("Metric city sampler state does not match the city/source partition")
        state = dict(state)
        state["kind"] = "pk_balanced"
        super().load_state_dict(state)

    def statistics(self):
        values = super().statistics()
        values["gsv_cities"].update(city_quotas=dict(self.city_quotas),
                                    city_places={city: len(places) for city, places in self.by_city.items()},
                                    minimum_coverage_samples=self.minimum_coverage_samples,
                                    place_coverage_ratio=1.0)
        return values
