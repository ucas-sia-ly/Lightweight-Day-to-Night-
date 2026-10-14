"""Read-only inventory of configured resources and source-distinct VPR sampling."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from .checkpointing import validation_resource_paths, write_json
from .data import read_generated_manifest
from .paths import resolve_resource_path


def audit_resources(config, output_path=None, scan_images: bool = True) -> dict:
    """Check paths/CSV labels without decoding images, loading weights or downloading.

    Skipping image scans reports unknown image missing counts. In that mode,
    ``training_ready`` covers assets and manifest/sampler structure only.
    Historical manifest bytes and cache provenance are never rewritten.
    """
    cfg, data = config.vpr, config.data
    stable = cfg.model_family == "stable_night_salad"
    source_only = stable and cfg.train_domains in {"day_only", "day_aug"}
    needs_fake = cfg.train_domains in {"day_fake", "day_aug_fake"}
    uses_dino = cfg.model_family in {"salad", "stable_night_salad", "dino_mix", "boq"}
    needs_validation = cfg.evaluate_epoch_zero or cfg.selection_rule == "validation_best"
    assets = {}
    errors = []

    def asset(name, value, required_for="optional", *, directory=False, entrypoint=""):
        configured = str(value) if value else ""
        path = resolve_resource_path(value).resolve() if value else None
        available = bool(path and (path.is_dir() if directory else path.is_file()))
        if available and entrypoint:
            available = (path / entrypoint).is_file()
        assets[name] = {
            "configured_path": configured,
            "path": str(path) if path else "",
            "available": available,
            "required_for": required_for,
        }
        if entrypoint:
            assets[name]["entrypoint"] = entrypoint
        if required_for == "training" and not available:
            errors.append(f"Missing required asset: {name}")
        return path

    asset("backbone_repo", cfg.backbone_repo, "training" if uses_dino else "optional",
          directory=True, entrypoint="hubconf.py" if uses_dino else "")
    asset("backbone_weights", cfg.backbone_checkpoint, "training" if uses_dino else "optional")
    initial = cfg.init_checkpoint
    if initial == "eigenplaces://resnet50":
        # Match the existing model's cache locator without invoking its downloader.
        initial = Path.home() / ".cache/torch/hub/checkpoints" / f"ResNet50_{cfg.descriptor_dim}_eigenplaces.pth"
    asset("init_weights", initial, "training" if cfg.init_checkpoint or cfg.model_family in {"mixvpr", "netvlad"} else "optional")
    assets["init_weights"]["configured_path"] = cfg.init_checkpoint
    asset("resume_weights", cfg.resume, "training" if cfg.resume else "optional")
    asset("aggregator_repo", cfg.aggregator_repo,
          "training" if cfg.model_family == "boq" and cfg.aggregator_repo else "optional",
          directory=True, entrypoint="src/boq.py" if cfg.model_family == "boq" and cfg.aggregator_repo else "")
    if cfg.cross_domain_aux == "cross_domain_relation":
        asset("relation_teacher_repo", cfg.relation_teacher_repo, "training", directory=True, entrypoint="hubconf.py")
        asset("relation_teacher_weights", cfg.relation_teacher_checkpoint, "training")
        cache = asset("relation_teacher_cache", cfg.relation_teacher_cache, "training")
        if cache and cache.suffix != ".npy":
            errors.append("Relation teacher cache must end in .npy")
        asset("relation_teacher_cache_metadata", cache.with_suffix(".meta.json") if cache else "", "training")
        asset("relation_teacher_cache_ids", cache.with_suffix(".ids.json") if cache else "", "training")
    training_path = data.vpr_source_manifest if source_only else Path(data.generated_root) / "manifest.csv"
    manifest = asset("training_manifest", training_path, "training")
    asset("generator_manifest", data.generator_manifest, "generator")
    seasons = asset("robotcar_seasons_root", data.robotcar_seasons_root, "evaluation", directory=True)
    asset("robotcar_seasons_images", seasons / "images" if seasons else "", "evaluation", directory=True)
    for split in ("train", "test"):
        asset(f"robotcar_seasons_{split}_list", seasons / f"robotcar_v2_{split}.txt" if seasons else "", "evaluation")
    for name in ("msls_root", "svox_root", "tokyo_root"):
        uses_validation_root = (
            needs_validation and
            ((name == "msls_root" and "msls_day_val" in config.evaluation.datasets)
             or (name == "svox_root" and "svox_day_val" in config.evaluation.datasets))
        )
        asset(name, getattr(data, name), "training" if uses_validation_root else "evaluation", directory=True)
    if needs_validation:
        for name, path in validation_resource_paths(config).items():
            is_directory = name.startswith("validation_svox_")
            resolved = asset(name, path, "training", directory=is_directory)
            if is_directory and assets[name]["available"] and not any(item.is_file() for item in resolved.iterdir()):
                assets[name]["available"] = False
                errors.append(f"Empty required public validation folder: {name}")
    if config.evaluation.database_domain == "fake_night":
        asset("fake_database_manifest", config.evaluation.fake_database_manifest, "evaluation")

    rows = []
    if assets["training_manifest"]["available"]:
        try:
            rows = read_generated_manifest(manifest)
        except (OSError, UnicodeError, ValueError, csv.Error) as error:
            errors.append(f"Cannot read training manifest: {error}")
    active_datasets = set(cfg.datasets)
    rows = [row for row in rows if not active_datasets or row.get("dataset", "unknown") in active_datasets]
    invalid_splits = dict(Counter(row.get("split", "") for row in rows if row.get("split") != "train"))
    if invalid_splits:
        errors.append("Training manifest contains missing or non-train splits")
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row.get("dataset", "unknown")].append(row)
    datasets = {}
    inspected = {}
    referenced_images = set()
    generated_metadata = {}
    invalid_rows = 0
    required_fields = ("source_path", "image_id", "place_id", "split") + (("generated_path",) if needs_fake else ())

    def canonical(value):
        return str(Path(value).expanduser().resolve()) if value else ""

    def exists(path):
        if path not in inspected:
            inspected[path] = Path(path).is_file()
        return inspected[path]

    for dataset in sorted(active_datasets or by_dataset):
        selected = by_dataset[dataset]
        sources, fakes = set(), set()
        source_rows, fake_rows = Counter(), Counter()
        places = defaultdict(list)
        source_labels = defaultdict(set)
        dataset_invalid_rows = 0
        fake_ownership_conflicts = set()
        generator_hash_conflicts = set()
        for row in selected:
            if any(not row.get(key) for key in required_fields):
                dataset_invalid_rows += 1
            source, fake = canonical(row.get("source_path", "")), canonical(row.get("generated_path", ""))
            if source:
                sources.add(source)
                source_rows[source] += 1
                source_labels[source].add(row.get("place_id", ""))
            if fake and needs_fake:
                fakes.add(fake)
                fake_rows[fake] += 1
                if stable:
                    identity = (dataset, source)
                    generator_hash = row.get("generator_sha256", "")
                    if fake in generated_metadata:
                        previous_identity, previous_hash = generated_metadata[fake]
                        if previous_identity != identity:
                            fake_ownership_conflicts.add(fake)
                        if previous_hash and generator_hash and previous_hash != generator_hash:
                            generator_hash_conflicts.add(fake)
                        # Keep the first owner and retain any known hash rather
                        # than letting an empty duplicate erase its provenance.
                        generated_metadata[fake] = (previous_identity, previous_hash or generator_hash)
                    else:
                        generated_metadata[fake] = (identity, generator_hash)
            places[row.get("place_id", "")].append(source)
        invalid_rows += dataset_invalid_rows
        conflicts = sum(len(labels) > 1 for labels in source_labels.values())
        if stable and conflicts:
            errors.append(f"{dataset}: a canonical source has conflicting place labels")
        if fake_ownership_conflicts:
            errors.append(f"{dataset}: one generated image represents different canonical source identities")
        if generator_hash_conflicts:
            errors.append(f"{dataset}: a generated image has conflicting generator hashes")
        referenced_images.update(sources | fakes)
        minimum = cfg.min_class_images_by_dataset.get(dataset, cfg.min_class_images)
        eligible = {
            place: paths for place, paths in places.items()
            if place and (len({path for path in paths if path}) if stable else len(paths)) >= minimum
        }
        eligible_rows = sum(len({path for path in paths if path}) if stable else len(paths) for paths in eligible.values())
        undersized = sorted(place for place, paths in eligible.items()
                            if len({path for path in paths if path}) < cfg.pk_images_per_place)
        pk_ready = cfg.sampler != "pk_balanced" or (
            len(eligible) >= cfg.pk_places_per_batch and not undersized
        )
        if not eligible:
            errors.append(f"{dataset}: no eligible place classes survive configured minimums")
        if not pk_ready:
            errors.append(f"{dataset}: PK sampling lacks P places or K distinct source images")
        missing_sources = sorted(path for path in sources if not exists(path)) if scan_images else None
        missing_fakes = sorted(path for path in fakes if not exists(path)) if scan_images else None
        if missing_sources or missing_fakes:
            errors.append(f"{dataset}: selected training images are missing")
        datasets[dataset] = {
            "selected_rows": len(selected),
            "unique_sources": len(sources),
            "duplicate_source_rows": sum(count - 1 for count in source_rows.values()),
            "unique_fake_paths": len(fakes),
            "source_missing": len(missing_sources) if scan_images else None,
            "fake_missing": len(missing_fakes) if scan_images else None,
            "source_missing_rows": sum(source_rows[path] for path in missing_sources) if scan_images else None,
            "fake_missing_rows": sum(fake_rows[path] for path in missing_fakes) if scan_images else None,
            "missing_examples": {"source": (missing_sources or [])[:5], "fake": (missing_fakes or [])[:5]},
            "invalid_rows": dataset_invalid_rows,
            "conflicting_source_labels": conflicts,
            "conflicting_fake_ownership": len(fake_ownership_conflicts),
            "conflicting_generator_hashes": len(generator_hash_conflicts),
            "fake_metadata_conflict_examples": sorted(fake_ownership_conflicts | generator_hash_conflicts)[:5],
            "eligible_places": len(eligible),
            "eligible_rows": eligible_rows,
            "class_minimum": minimum,
            "pk_ready": pk_ready,
            "pk_undersized_places": len(undersized),
            "pk_undersized_examples": undersized[:5],
        }
    if invalid_rows:
        errors.append("Training manifest rows lack required paths, IDs, labels or splits")
    if not rows:
        errors.append("No selected training rows")
    available_datasets = {name for name, info in datasets.items() if info["eligible_places"] > 0}
    if cfg.dataset_sampling_weights and set(cfg.dataset_sampling_weights) != available_datasets:
        errors.append("Dataset sampling weights must name exactly the eligible training datasets")
    report = {
        "model_family": cfg.model_family,
        "train_domains": cfg.train_domains,
        "assets": assets,
        "datasets": datasets,
        "selected_rows": len(rows),
        "invalid_splits": invalid_splits,
        "images_scanned": bool(scan_images),
        "readiness_scope": "assets_manifest_sampling_and_image_existence" if scan_images else "assets_manifest_and_sampling_only",
        "training_ready": not errors,
        "training_errors": errors,
        "evaluation_gaps": [name for name, value in assets.items() if value["required_for"] == "evaluation" and not value["available"]],
        "limitations": ["Weight contents, teacher-cache contents, image decoding and runtime dependencies are not verified."],
    }
    if output_path is not None:
        output = Path(output_path).expanduser().resolve()
        protected = {value["path"] for value in assets.values() if value["path"]} | referenced_images
        if str(output) in protected:
            raise ValueError("Resource audit output cannot overwrite an input resource")
        write_json(report, output)
    return report
