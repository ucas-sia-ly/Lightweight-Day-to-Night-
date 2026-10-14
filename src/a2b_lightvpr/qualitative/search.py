"""Exhaustive, checkpoint-specific nighttime top-1 selection and figure export."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..checkpointing import file_sha256, write_json
from .export import (
    DATASETS, _cache_descriptors, _copy_image, _models, _positives,
)
from ..references.models import query_protocol_fingerprint, suite_rows


ORDER = ("cosplace", "eigenplaces", "official_mixvpr512", "nocplace",
         "anyloc_pca", "crica_pca", "salad", "boq_pca", "cdnd")
LABELS = ("CosPlace", "EigenPlaces", "MixVPR", "NocPlace", "AnyLoc-PCA",
          "CricaVPR-PCA", "SALAD", "BoQ-PCA", "CDND (Ours)")
PROTOCOLS = ("msls_night", "svox_night", "tokyo_night")
EXCLUDED_PROTOCOLS = {
    "robotcar_seasons_night": {
        "status": "retrieval_only_no_public_test_gt",
        "reason": "Correctness-based automatic example selection requires public ground truth; export an explicit Query Top-K instead.",
    },
}


def rank_candidates(correct: dict[str, np.ndarray], valid: np.ndarray) -> list[int]:
    """Maximize failures; ties prefer successes confined to BoQ/SALAD."""
    candidates = np.flatnonzero(valid & correct["cdnd"])
    others = [name for name in correct if name != "cdnd"]
    strict = [name for name in others if name not in ("boq_pca", "salad")]
    return sorted(map(int, candidates), key=lambda i: (
        sum(bool(correct[n][i]) for n in others),
        sum(bool(correct[n][i]) for n in strict), i,
    ))


def render_figure(results: dict, output: Path) -> None:
    cell_w, cell_h, header, left, gap = 260, 195, 65, 75, 6
    canvas = Image.new("RGB", (left + 11 * (cell_w + gap), header + len(PROTOCOLS) * (cell_h + gap)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    for col, label in enumerate(("Query", *LABELS, "Ground truth")):
        x = left + col * (cell_w + gap)
        draw.text((x + cell_w / 2, 25), label, font=font, fill="black", anchor="mm")
    for row, protocol in enumerate(PROTOCOLS):
        example = results["examples"][protocol]
        directory = output / protocol
        y = header + row * (cell_h + gap)
        label = {"msls_night": "MSLS", "svox_night": "SVOX", "tokyo_night": "Tokyo"}[protocol]
        strip = Image.new("RGB", (cell_h, left), "white")
        ImageDraw.Draw(strip).text((cell_h / 2, left / 2), label, anchor="mm", fill="black", font=font)
        canvas.paste(strip.rotate(90, expand=True), (0, y))
        entries = [(example["query_file"], None)]
        entries.extend((example["models"][n]["output_file"], example["models"][n]["correct_at_1"]) for n in ORDER)
        entries.append((example["ground_truth_file"], True))
        for col, (filename, correct) in enumerate(entries):
            x = left + col * (cell_w + gap)
            with Image.open(directory / filename) as image:
                # Letterbox only for display; do not crop away retrieval evidence.
                tile = ImageOps.pad(image.convert("RGB"), (cell_w, cell_h), color="white")
            canvas.paste(tile, (x, y))
            if correct is not None:
                draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1),
                               outline="#00a040" if correct else "#ef2020", width=4)
    canvas.save(output / "comparison_all.png")
    canvas.save(output / "comparison_all.jpg", quality=96)
    canvas.save(output / "comparison_all.pdf", resolution=300)


def search_qualitative_top1(config, selection_path, output_dir, cache_dir, *, batch_size=32, workers=4):
    import faiss

    faiss.omp_set_num_threads(4)
    torch.set_num_threads(4)
    output, cache = Path(output_dir), Path(cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, positives, protocols = {}, {}, {}
    for name in PROTOCOLS:
        spec = DATASETS[name]
        db, queries = suite_rows(config, spec.dataset, spec.split)
        if not db or not queries:
            raise ValueError(f"Empty dataset: {name}")
        rows[name] = (db, queries)
        positives[name] = [_positives(db, q, spec) for q in queries]
        protocols[name] = query_protocol_fingerprint(db, queries, spec.radius_m, spec.yaw_deg)

    predictions = {name: {} for name in PROTOCOLS}
    identities = {}
    for model_name, model, transform, checkpoint in _models(config, ORDER):
        effective_batch = model.qualitative_batch_size or batch_size
        identity = {
            "schema": "night_qualitative_search_v1", "model": model_name,
            "checkpoint": str(checkpoint.resolve()),
            "artifacts": model.qualitative_artifacts,
            "spec": model.qualitative_spec,
            "input_size": [transform.height, transform.width],
            "transform": type(transform).__name__, "batch_size": effective_batch,
            "precision": "fp32", "torch": str(torch.__version__),
            "export_code_sha256": file_sha256(__file__),
            "loader_code_sha256": file_sha256(Path(__file__).with_name("export.py")),
            "crica_code_sha256": file_sha256(Path(__file__).resolve().parents[1] / "vpr/crica.py"),
        }
        identities[model_name] = identity
        for name in PROTOCOLS:
            db, queries = rows[name]
            identity_ds = {**identity, "protocol": protocols[name], "dataset": name}
            pred_path = cache / model_name / f"{name}_top1.json"
            if pred_path.is_file():
                saved = json.loads(pred_path.read_text())
                if saved.get("identity") == identity_ds and len(saved.get("indices", [])) == len(queries):
                    predictions[name][model_name] = saved
                    print(f"[cached] {model_name}/{name}", flush=True)
                    continue
            print(f"[extract] {model_name}/{name}, batch={effective_batch}", flush=True)
            model.eval().to(device)
            descriptors = []
            for role, image_rows in (("database", db), ("queries", queries)):
                descriptors.append(_cache_descriptors(
                    model, image_rows, transform, device, effective_batch, workers,
                    cache / model_name / f"{name}_{role}.npy", {**identity_ds, "role": role},
                ))
            index = faiss.IndexFlatIP(model.descriptor_dim)
            index.add(np.asarray(descriptors[0], dtype=np.float32))
            scores, indices = index.search(np.asarray(descriptors[1], dtype=np.float32), 1)
            indices = indices[:, 0]
            correct = [bool(int(i) in pos) for i, pos in zip(indices, positives[name])]
            saved = {"identity": identity_ds, "indices": indices.tolist(),
                     "scores": scores[:, 0].tolist(), "correct": correct}
            write_json(saved, pred_path)
            predictions[name][model_name] = saved
            del descriptors, index
        model.cpu()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = {"schema": "night_qualitative_selection_v1", "model_order": list(ORDER),
               "selection_rule": "CDND correct; minimize other successes; tie prefers BoQ/SALAD successes; then query index",
               "note": "Selected qualitative examples, not an estimate of overall recall. NocPlace uses the user-approved substitute-data reproduction.",
               "model_identities": identities, "excluded": EXCLUDED_PROTOCOLS, "examples": {}}
    for name in PROTOCOLS:
        db, queries = rows[name]
        correct = {n: np.asarray(predictions[name][n]["correct"], dtype=bool) for n in ORDER}
        valid = np.asarray([len(p) > 0 for p in positives[name]])
        ranking = rank_candidates(correct, valid)
        if not ranking:
            raise RuntimeError(f"No valid CDND top-1 success in {name}; no fabricated example exported")
        q_index = ranking[0]
        q = queries[q_index]
        gt = min(map(int, positives[name][q_index]), key=lambda i:
                 (db[i].easting - q.easting)**2 + (db[i].northing - q.northing)**2)
        directory = output / name
        example = {
            "dataset": name, "query_index": q_index, "query_path": q.source_path,
            "query_file": _copy_image(q.source_path, directory / "query"),
            "ground_truth_positive_index": gt, "ground_truth_positive_path": db[gt].source_path,
            "ground_truth_file": _copy_image(db[gt].source_path, directory / "ground_truth_positive"),
            "valid_query_count": int(valid.sum()), "cdnd_correct_candidates": len(ranking),
            "ranked_candidates": [{"query_index": i, "correct_at_1": {n: bool(correct[n][i]) for n in ORDER}}
                                  for i in ranking[:20]], "models": {},
        }
        for n in ORDER:
            top1 = predictions[name][n]["indices"][q_index]
            example["models"][n] = {
                "top1_index": top1, "top1_path": db[top1].source_path,
                "top1_score": predictions[name][n]["scores"][q_index],
                "correct_at_1": bool(correct[n][q_index]),
                "output_file": _copy_image(db[top1].source_path, directory / f"{n}_top1"),
            }
        results["examples"][name] = example
        print(name, q_index, {n: bool(correct[n][q_index]) for n in ORDER}, flush=True)
    write_json(results, output / "top1_results.json")
    write_json({"selection_rule": results["selection_rule"], "examples": [
        {"dataset": n, "query_index": e["query_index"]} for n, e in results["examples"].items()
    ]}, selection_path)
    render_figure(results, output)
    return output / "top1_results.json"
