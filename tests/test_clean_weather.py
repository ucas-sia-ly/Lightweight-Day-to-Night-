import csv
import json
from pathlib import Path

from PIL import Image

from a2b_lightvpr.checkpointing import file_sha256
from a2b_lightvpr.clean_weather import (
    AUDIT_FIELDS,
    _candidate_hash,
    _image_problem,
    filter_clear_weather,
)


def test_weather_audit_rejects_corrupt_image_before_clip(tmp_path):
    image = tmp_path / "broken.jpg"
    image.write_bytes(b"not-a-jpeg")
    assert _image_problem(image).startswith("decode_error")


def test_weather_audit_reuses_matching_cache_without_loading_clip(tmp_path):
    image = tmp_path / "road.jpg"
    Image.new("RGB", (640, 480), (80, 90, 100)).save(image)
    checkpoint = tmp_path / "clip.pt"
    checkpoint.write_bytes(b"fixture")
    items = [(image, "city", "night")]
    checkpoint_hash = file_sha256(checkpoint)
    with (tmp_path / "weather_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerow({
            "path": str(image.resolve()), "city": "city", "domain": "night",
            "clear": "0.8", "rain": "0.1", "snow": "0.05", "fog": "0.05",
            "keep": "1", "reason": "", "checkpoint_sha256": checkpoint_hash,
        })
    (tmp_path / "weather_audit_summary.json").write_text(json.dumps({
        "audit_schema": 2,
        "checkpoint_sha256": checkpoint_hash,
        "candidate_sha256": _candidate_hash(items),
        "clear_threshold": 0.55,
        "margin": 0.15,
        "candidates": 1,
    }), encoding="utf-8")
    assert filter_clear_weather(items, tmp_path, checkpoint) == {image.resolve()}
