"""Public RobotCar Seasons image retrieval; private test poses are never guessed."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .checkpointing import file_sha256, write_json
from .paths import DATASETS_ROOT


NIGHT_CONDITIONS = ("night", "night-rain")
DAY_CONDITIONS = ("dawn", "dusk", "overcast-summer", "overcast-winter", "rain", "snow", "sun")
CAMERAS = ("left", "rear", "right")
DATASET_README = "https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/README_RobotCar-Seasons.md"
V2_README = "https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/README_RobotCar_v2.md"


def default_robotcar_seasons_root() -> Path:
    return Path(os.environ.get("A2B_DATASETS_ROOT", str(DATASETS_ROOT))).expanduser() / "RobotCar-Seasons"


@dataclass(frozen=True)
class RobotCarSeasonsRow:
    source_path: str
    image_id: str  # Relative to images/, including condition and camera.
    condition: str
    camera: str

    @property
    def ground_truth_available(self):
        return False

    @property
    def easting(self):
        return None

    @property
    def northing(self):
        return None

    @property
    def yaw_deg(self):
        return None


def _selection(protocol, cameras):
    if protocol not in {"original", "v2"}:
        raise ValueError("RobotCar Seasons protocol must be original or v2")
    return tuple(cameras) if cameras is not None else (("rear",) if protocol == "v2" else CAMERAS)


def _v2_queries(root):
    """Resolve official camera/timestamp PNG names to the released JPG images.

    Require exact, unique camera/timestamp matches and exclude the public train
    poses. Never turn those training images into test queries.
    """
    index = {}
    images = root / "images"
    for path in images.glob("*/*/*.jpg"):
        if path.is_file():
            index.setdefault((path.parent.name, path.stem), []).append(path)
    train_ids = set()
    with (root / "robotcar_v2_train.txt").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.split()
            path = Path(fields[0])
            if len(fields) != 17 or len(path.parts) != 3 or path.is_absolute() or ".." in path.parts:
                raise ValueError("Invalid robotcar_v2_train.txt row")
            train_ids.add(path.with_suffix(".jpg").as_posix())
    if not train_ids:
        raise ValueError("Official RobotCar v2 train list cannot be empty")
    rows, seen = [], set()
    with (root / "robotcar_v2_test.txt").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            path = Path(line.strip())
            if len(path.parts) != 2 or path.parent.name not in CAMERAS or path.suffix not in {".png", ".jpg"}:
                raise ValueError("Invalid robotcar_v2_test.txt image name")
            matches = index.get((path.parent.name, path.stem), [])
            if len(matches) != 1:
                raise ValueError(f"Official v2 query needs one exact camera/timestamp image: {path} ({len(matches)} matches)")
            image = matches[0]
            image_id = image.relative_to(images).as_posix()
            if image_id in train_ids or image_id in seen:
                raise ValueError(f"Duplicate or train/test overlapping RobotCar v2 query: {image_id}")
            seen.add(image_id)
            rows.append(RobotCarSeasonsRow(str(image.resolve()), image_id, image.parent.parent.name, image.parent.name))
    if not rows:
        raise ValueError("Official RobotCar v2 test list cannot be empty")
    return sorted(rows, key=lambda row: row.image_id)


def robotcar_seasons_rows(root, *, conditions=NIGHT_CONDITIONS, cameras=None, protocol="v2"):
    """Read public images with an explicit original/v2 Query split and no guessed poses."""
    cameras = _selection(protocol, cameras)
    for name, values, allowed in (("conditions", conditions, NIGHT_CONDITIONS + DAY_CONDITIONS), ("cameras", cameras, CAMERAS)):
        if not values or len(set(values)) != len(values) or not set(values).issubset(allowed):
            raise ValueError(f"Invalid RobotCar Seasons {name}: {values}")
    root = Path(root).expanduser().resolve()
    images = root / "images"
    outputs = []
    for selected_conditions in (("overcast-reference",), conditions):
        rows = []
        for condition in selected_conditions:
            for camera in cameras:
                folder = images / condition / camera
                paths = sorted(path for path in folder.glob("*.jpg") if path.is_file())
                if not paths:
                    raise FileNotFoundError(f"RobotCar Seasons images missing or empty: {folder}")
                rows.extend(RobotCarSeasonsRow(
                    str(path.resolve()), path.relative_to(images).as_posix(), condition, camera
                ) for path in paths)
        outputs.append(rows)
    if protocol == "v2":
        queries = [row for row in _v2_queries(root) if row.condition in conditions and row.camera in cameras]
        if not queries:
            raise ValueError("No official RobotCar v2 test queries match the selected conditions/cameras")
        return outputs[0], queries
    return outputs[0], outputs[1]


def _metadata_hashes(root, protocol):
    return {name: file_sha256(root / name) for name in ("robotcar_v2_train.txt", "robotcar_v2_test.txt")} if protocol == "v2" else {}


def _inventory_fingerprint(database, queries):
    payload = [(role, row.image_id, row.source_path)
               for role, rows in (("database", database), ("queries", queries)) for row in rows]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _check_output(output, root, protected=()):
    output = Path(output).expanduser().resolve()
    if output.is_relative_to(Path(root).expanduser().resolve()) or output in {
        Path(path).expanduser().resolve() for path in protected if path
    }:
        raise ValueError("RobotCar Seasons output cannot overwrite dataset or model inputs")
    return output


def audit_robotcar_seasons(root=None, *, output=None, conditions=NIGHT_CONDITIONS, cameras=None, protocol="v2"):
    root = Path(root if root is not None else default_robotcar_seasons_root()).expanduser().resolve()
    cameras = _selection(protocol, cameras)
    try:
        database, queries = robotcar_seasons_rows(root, conditions=conditions, cameras=cameras, protocol=protocol)
        error = ""
    except (FileNotFoundError, ValueError) as failure:
        database, queries, error = [], [], str(failure)
    groups = {}
    for condition in ("overcast-reference", *conditions):
        groups[condition] = {
            camera: sum(path.is_file() for path in (root / "images" / condition / camera).glob("*.jpg"))
            for camera in cameras
        }
    report = {
        "dataset": "robotcar_seasons", "root": str(root),
        "protocol": protocol, "split_metadata_sha256": _metadata_hashes(root, protocol) if not error else {},
        "layout": "original_condition_camera", "conditions": list(conditions), "cameras": list(cameras),
        "image_retrieval_ready": not error, "local_recall_ready": False,
        "ground_truth_status": "official_test_query_poses_not_public",
        "database": len(database), "queries": len(queries), "images_by_condition_camera": groups,
        "inventory_fingerprint": _inventory_fingerprint(database, queries) if not error else "",
        "scope": "image_paths_only_no_decoding_or_reference_model_validation",
        "official_readme": V2_README if protocol == "v2" else DATASET_README, "error": error,
    }
    if output:
        write_json(report, _check_output(output, root))
    return report

