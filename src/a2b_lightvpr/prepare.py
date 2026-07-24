from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .clean_weather import filter_clear_weather


GENERATOR_FIELDS = (
    "sample_id", "dataset", "split", "route", "paired", "source_path",
    "target_path", "valid", "exclusion_reason",
)
VPR_FIELDS = (
    "image_id", "dataset", "split", "place_id", "route", "source_path",
    "utm_e", "utm_n", "yaw", "role",
)
VPR_MINIMUMS = {"msls": 4, "robotcar": 2, "svox": 2}
EXPECTED_VPR_ELIGIBLE = {
    "msls": {"places": 18960, "rows": 75840},
    "robotcar": {"places": 1464, "rows": 8947},
    "svox": {"places": 9436, "rows": 33526},
}


def _write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _image_problem(path: Path) -> str:
    try:
        with Image.open(path) as image:
            image.load()
            if image.width < 320 or image.height < 240:
                return "unexpected_dimensions"
            sample = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32)
        if float(sample.std()) < 2.0:
            return "near_constant"
        if float((sample < 2).all(axis=2).mean()) > 0.98:
            return "near_black"
        if float((sample > 253).all(axis=2).mean()) > 0.98:
            return "near_white"
    except (OSError, ValueError) as exc:
        return f"decode_error:{type(exc).__name__}"
    return ""


def _paired_rows(dataset: str, day: list[Path], night: list[Path], route: str = ""):
    if not day or not night:
        return []
    count = max(len(day), len(night))
    return [
        {
            "sample_id": f"{dataset}:{index}", "dataset": dataset, "split": "train",
            "route": route, "paired": "0", "source_path": str(day[index % len(day)].resolve()),
            "target_path": str(night[index % len(night)].resolve()), "valid": "1",
            "exclusion_reason": "",
        }
        for index in range(count)
    ]


def _even_sample(values: list[Path], limit: int) -> list[Path]:
    if len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, limit, dtype=np.int64)
    return [values[int(index)] for index in indices]


def _cap_views_per_place(rows: list[dict[str, str]], limit: int = 4):
    counts = Counter()
    retained = []
    for row in rows:
        key = (row["dataset"], row["place_id"])
        if counts[key] < limit:
            retained.append(row)
            counts[key] += 1
    return retained


def _parse_svox_train_path(path: Path) -> tuple[float, float, str] | None:
    parts = path.name.split("@")
    if len(parts) < 7 or parts[-2] not in {"F", "B"}:
        return None
    try:
        return float(parts[1]), float(parts[2]), parts[-2]
    except ValueError:
        return None


def build_manifests(
    data_root: str | Path, output_dir: str | Path, *, decode_darkdriving: bool = True,
    clip_checkpoint: str | Path = "/home/admin123/.cache/clip/ViT-B-32.pt",
):
    root, output = Path(data_root), Path(output_dir)
    generator_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []

    # DarkDriving is the only aligned source. A bad image invalidates the complete pair.
    dark = root / "DarkDriving_Base"
    previous_problems = {}
    previous_audit = output / "darkdriving_audit.csv"
    if not decode_darkdriving and previous_audit.is_file():
        with previous_audit.open(newline="", encoding="utf-8") as handle:
            previous_problems = {
                row["sample_id"]: row.get("exclusion_reason", "")
                for row in csv.DictReader(handle) if row.get("valid") == "0"
            }
    day_by_name = {p.name: p for p in (dark / "Day").glob("*") if p.is_file()}
    night_by_name = {p.name: p for p in (dark / "Night").glob("*") if p.is_file()}
    for name in sorted(day_by_name.keys() & night_by_name.keys()):
        day, night = day_by_name[name], night_by_name[name]
        problems = [problem for problem in previous_problems.get(f"darkdriving:{name}", "").split(";") if problem]
        if decode_darkdriving:
            for domain, path in (("day", day), ("night", night)):
                problem = _image_problem(path)
                if problem:
                    problems.append(f"{domain}:{problem}")
        valid = not problems
        row = {
            "sample_id": f"darkdriving:{name}", "dataset": "darkdriving", "split": "train",
            "route": "aligned", "paired": "1", "source_path": str(day.resolve()),
            "target_path": str(night.resolve()), "valid": str(int(valid)),
            "exclusion_reason": ";".join(problems),
        }
        audit_rows.append(row)
        if valid:
            generator_rows.append(row)

    robot_manifest = root / "dvg_benchmark_paper/robotcar_nightday/manifest.csv"
    robot_train = []
    if robot_manifest.is_file():
        robot_repository = robot_manifest.parents[3]
        with robot_manifest.open(newline="", encoding="utf-8") as handle:
            robot_train = [row for row in csv.DictReader(handle) if row["split"] == "train"]
        def robot_path(row):
            path = Path(row["source_path"])
            return path if path.is_absolute() else (robot_repository / path).resolve()
        robot_day = [robot_path(row) for row in robot_train if row["domain"] == "day"]
        robot_night = [robot_path(row) for row in robot_train if row["domain"] == "night"]
        generator_rows.extend(_paired_rows("robotcar", robot_day, robot_night))

    # BDD100K contributes diverse road day/night appearance, never VPR labels.
    labels = root / "bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
    images = root / "bdd100k/bdd100k/bdd100k/images/100k/train"
    if labels.is_file():
        bdd_day, bdd_night = [], []
        for item in json.loads(labels.read_text(encoding="utf-8")):
            path = images / item["name"]
            if not path.is_file():
                matches = list(images.glob(f'*/{item["name"]}'))
                path = matches[0] if matches else path
            if path.is_file() and item.get("attributes", {}).get("weather") == "clear":
                time_of_day = item.get("attributes", {}).get("timeofday")
                if time_of_day == "daytime":
                    bdd_day.append(path)
                elif time_of_day == "night":
                    bdd_night.append(path)
        generator_rows.extend(_paired_rows("bdd100k", bdd_day, bdd_night))

    # MSLS training cities provide both appearance diversity and place supervision.
    vpr_rows: list[dict[str, str]] = []
    for row in robot_train:
        if row["domain"] != "day" or row.get("role") != "database":
            continue
        easting, northing = float(row["easting"]), float(row["northing"])
        yaw = float(row.get("yaw_deg", 0.0))
        place = f"{int(easting // 10)}:{int(northing // 10)}:{int((yaw % 360) // 30)}"
        image_id = hashlib.sha1(
            f'{row["traverse"]}:{row["timestamp"]}:{row["role"]}'.encode()
        ).hexdigest()[:20]
        vpr_rows.append({
            "image_id": image_id, "dataset": "robotcar", "split": "train",
            "place_id": place, "route": row["traverse"],
            "source_path": str(robot_path(row)),
            "utm_e": row["easting"], "utm_n": row["northing"], "yaw": str(yaw),
            "role": row["role"],
        })
    msls = root / "MSLS/train_val"
    heldout_cities = {"cph", "sf"}
    msls_pools: dict[str, tuple[list[Path], list[Path]]] = {}
    for table in sorted(msls.glob("*/*/postprocessed.csv")):
        city, role = table.parents[1].name, table.parent.name
        if city in heldout_cities:
            continue
        day, night = [], []
        with table.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                path = table.parent / "images" / f'{row["key"]}.jpg'
                if not path.is_file():
                    continue
                is_night = row.get("night", "False").lower() == "true"
                (night if is_night else day).append(path)
                if not is_night:
                    vpr_rows.append({
                        "image_id": f'msls:{city}:{row["key"]}', "dataset": "msls",
                        "split": "train", "place_id": f'{city}:{row["unique_cluster"]}',
                        "route": city, "source_path": str(path.resolve()),
                        "utm_e": row["easting"], "utm_n": row["northing"], "yaw": "", "role": role,
                    })
        msls_pools[city] = (_even_sample(day, 5000), _even_sample(night, 5000))

    weather_items = [
        (path, city, domain)
        for city, pools in msls_pools.items()
        for domain, values in zip(("day", "night"), pools)
        for path in values
    ]
    accepted_weather = filter_clear_weather(
        weather_items, output, clip_checkpoint
    ) if weather_items else set()
    for city, (day, night) in msls_pools.items():
        generator_rows.extend(_paired_rows(
            "msls",
            [path for path in day if path.resolve() in accepted_weather],
            [path for path in night if path.resolve() in accepted_weather],
            city,
        ))

    # SVOX official train locations are disjoint from its test split.
    svox_train = root / "svox/images/train"
    svox_day = sorted((svox_train / "gallery").glob("*")) + sorted((svox_train / "queries").glob("*"))
    svox_night = sorted((svox_train / "queries_night").glob("*"))
    generator_rows.extend(_paired_rows("svox", svox_day, svox_night, "train"))
    for role in ("gallery", "queries"):
        for path in sorted((svox_train / role).glob("*")):
            parsed = _parse_svox_train_path(path)
            if parsed is None:
                continue
            easting, northing, direction = parsed
            # Keep both camera directions in one 10 m place class. Splitting F/B
            # would discard single-direction cells and change the frozen
            # 9,436-place/33,526-image protocol.
            place = f"{int(easting // 10)}:{int(northing // 10)}"
            vpr_rows.append({
                "image_id": f"svox:{role}:{path.stem}", "dataset": "svox",
                "split": "train", "place_id": place, "route": "train",
                "source_path": str(path.resolve()), "utm_e": str(easting),
                "utm_n": str(northing), "yaw": direction, "role": role,
            })

    capped = _cap_views_per_place(
        [row for row in vpr_rows if row["dataset"] == "msls"], limit=4
    )
    vpr_rows = capped + [
        row for row in vpr_rows if row["dataset"] in {"robotcar", "svox"}
    ]
    eligible = {}
    eligible_places: dict[str, set[str]] = {}
    for dataset, minimum in VPR_MINIMUMS.items():
        subset = [row for row in vpr_rows if row["dataset"] == dataset]
        counts = Counter(row["place_id"] for row in subset)
        eligible_places[dataset] = {
            place for place, count in counts.items() if count >= minimum
        }
        eligible[dataset] = {
            "places": len(eligible_places[dataset]),
            "rows": sum(counts[place] for place in eligible_places[dataset]),
        }
    if all(any(row["dataset"] == name for row in vpr_rows) for name in EXPECTED_VPR_ELIGIBLE):
        if eligible != EXPECTED_VPR_ELIGIBLE:
            raise RuntimeError(
                f"VPR place statistics changed; audit and update the golden values: {eligible}"
            )
    vpr_rows = [
        row for row in vpr_rows
        if row["place_id"] in eligible_places.get(row["dataset"], set())
    ]
    _write_csv(output / "generator_train.csv", GENERATOR_FIELDS, generator_rows)
    _write_csv(output / "darkdriving_audit.csv", GENERATOR_FIELDS, audit_rows)
    _write_csv(output / "vpr_train.csv", VPR_FIELDS, vpr_rows)
    summary = {
        "generator": dict(Counter(row["dataset"] for row in generator_rows)),
        "vpr": dict(Counter(row["dataset"] for row in vpr_rows)),
        "vpr_eligible": eligible,
        "darkdriving_valid_pairs": sum(row["valid"] == "1" for row in audit_rows),
        "darkdriving_invalid_pairs": sum(row["valid"] == "0" for row in audit_rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
