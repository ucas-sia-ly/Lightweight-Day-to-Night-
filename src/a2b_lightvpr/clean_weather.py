from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from .checkpointing import file_sha256, write_json


AUDIT_FIELDS = (
    "path", "city", "domain", "clear", "rain", "snow", "fog", "keep", "reason",
    "checkpoint_sha256",
)


def _candidate_hash(items: list[tuple[Path, str, str]]) -> str:
    payload = "\n".join(f"{path.resolve()}|{city}|{domain}" for path, city, domain in items)
    return hashlib.sha256(payload.encode()).hexdigest()


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


def filter_clear_weather(
    items: list[tuple[Path, str, str]], output: Path, checkpoint: str | Path,
    *, clear_threshold: float = 0.55, margin: float = 0.15, batch_size: int = 64,
) -> set[Path]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CLIP checkpoint not found: {checkpoint}")
    audit_path = output / "weather_audit.csv"
    summary_path = output / "weather_audit_summary.json"
    identity = {
        "audit_schema": 2,
        "checkpoint_sha256": file_sha256(checkpoint),
        "candidate_sha256": _candidate_hash(items),
        "clear_threshold": clear_threshold,
        "margin": margin,
        "candidates": len(items),
    }
    if audit_path.is_file() and summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if all(summary.get(key) == value for key, value in identity.items()):
            with audit_path.open(newline="", encoding="utf-8") as handle:
                return {Path(row["path"]) for row in csv.DictReader(handle) if row["keep"] == "1"}

    try:
        import clip
    except ImportError as exc:
        raise RuntimeError("Clean-night preparation requires the installed OpenAI CLIP package") from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(str(checkpoint), device=device)
    prompts = {
        domain: [
            f"a clear dry road at {domain}", f"a rainy wet road at {domain}",
            f"a snowy road at {domain}", f"a foggy road at {domain}",
        ] for domain in {item[2] for item in items}
    }
    text = {}
    with torch.inference_mode():
        for domain, values in prompts.items():
            encoded = model.encode_text(clip.tokenize(values).to(device)).float()
            text[domain] = torch.nn.functional.normalize(encoded, dim=1)

    rows = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        problems = [_image_problem(path) for path, _, _ in batch]
        valid_indices = [index for index, problem in enumerate(problems) if not problem]
        encoded_by_index = {}
        if valid_indices:
            prepared = []
            for index in valid_indices:
                with Image.open(batch[index][0]) as image:
                    prepared.append(preprocess(image.convert("RGB")))
            images = torch.stack(prepared).to(device)
            with torch.inference_mode():
                encoded = torch.nn.functional.normalize(model.encode_image(images).float(), dim=1)
            encoded_by_index = {
                batch_index: encoded[value_index]
                for value_index, batch_index in enumerate(valid_indices)
            }
        for index, (path, city, domain) in enumerate(batch):
            if index in encoded_by_index:
                probabilities = (
                    100 * encoded_by_index[index].unsqueeze(0) @ text[domain].T
                ).softmax(1)[0].cpu()
            else:
                probabilities = torch.zeros(4)
            clear = float(probabilities[0])
            adverse = float(probabilities[1:].max())
            problem = problems[index]
            keep = not problem and clear >= clear_threshold and clear - adverse >= margin
            reason = problem or ("" if keep else "weather_not_clear")
            rows.append({
                "path": str(path.resolve()), "city": city, "domain": domain,
                "clear": clear, "rain": float(probabilities[1]),
                "snow": float(probabilities[2]), "fog": float(probabilities[3]),
                "keep": str(int(keep)), "reason": reason,
                "checkpoint_sha256": identity["checkpoint_sha256"],
            })
    output.mkdir(parents=True, exist_ok=True)
    temporary = audit_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(audit_path)
    for decision in ("accepted", "rejected"):
        directory = output / "audit_samples" / decision
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)
        selected = [row for row in rows if (row["keep"] == "1") == (decision == "accepted")][:100]
        for index, row in enumerate(selected):
            source = Path(row["path"])
            shutil.copy2(source, directory / f"{index:03d}_{source.name}")
    write_json({
        **identity,
        "accepted": sum(row["keep"] == "1" for row in rows),
        "rejected": sum(row["keep"] == "0" for row in rows),
    }, summary_path)
    return {Path(row["path"]) for row in rows if row["keep"] == "1"}
