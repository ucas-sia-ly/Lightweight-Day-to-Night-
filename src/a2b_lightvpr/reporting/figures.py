from __future__ import annotations

import json
from pathlib import Path

from ..checkpointing import write_json
from .seed0_summary import NIGHT_MACRO


REFERENCE_MODELS = (
    ("CosPlace", "cosplace-r50-512", "final_suite/native/adapted/summary.json", "cnn"),
    ("EigenPlaces", "eigenplaces-r50-512", "final_suite/native/adapted/summary.json", "cnn"),
    ("MixVPR-4096", "mixvpr-r50-4096", "final_suite/native/adapted/summary.json", "cnn"),
)

SMALL_REFERENCE_MODELS = (
    ("SALAD-S + CDND", "salad-small-544"),
)


def _night_macro(payload: dict) -> float:
    return sum(
        float(payload["datasets"][name]["recalls"]["1"]) for name in NIGHT_MACRO
    ) / len(NIGHT_MACRO)


def _student_result(payload: dict) -> tuple[float, dict, str]:
    if payload.get("schema") == "cdnd_paper_experiments_v1":
        accuracy = float(payload["main"]["cdnd"]["night_macro_recall1"]["mean"])
        seed_zero_path = payload["inputs"]["cdnd"].get("0")
        if not seed_zero_path:
            raise ValueError("Multi-seed summary does not identify the seed-0 CDND suite")
        seed_zero = json.loads(Path(seed_zero_path).read_text(encoding="utf-8"))
        return accuracy, seed_zero, "three-seed mean"
    return _night_macro(payload), payload, "seed-0 result"


def _point(label: str, family: str, accuracy: float, resources: dict) -> dict:
    dimension = int(resources["descriptor_dim"])
    return {
        "label": label,
        "family": family,
        "night_macro_recall1": accuracy,
        "parameters_million": float(resources["parameters"]) / 1_000_000,
        "descriptor_dim": dimension,
        "storage_per_million_fp16_gib": 1_000_000 * dimension * 2 / 1024**3,
    }


def plot_accuracy_efficiency(
    student_summary: str | Path,
    reference_root: str | Path,
    resource_summaries: list[str | Path],
    output: str | Path,
    compact_summary: str | Path | None = None,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install the paper extra to generate manuscript plots") from exc

    student_report = json.loads(Path(student_summary).read_text(encoding="utf-8"))
    student_accuracy, student, student_statistic = _student_result(student_report)
    resources = {}
    gpu = None
    for resource_summary in resource_summaries:
        payload = json.loads(Path(resource_summary).read_text(encoding="utf-8"))
        resources.update(payload["models"])
        gpu = gpu or payload.get("gpu")
    points = [_point("CDND-512", "ours", student_accuracy, {
        "descriptor_dim": student["descriptor_dim"],
        "parameters": student["benchmark"]["parameters"],
    })]
    reference_root = Path(reference_root)
    for label, resource_key, relative_summary, family in REFERENCE_MODELS:
        summary_path = reference_root / resource_key / relative_summary
        if not summary_path.is_file() or resource_key not in resources:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        points.append(
            _point(label, family, _night_macro(summary), resources[resource_key])
        )

    if compact_summary:
        compact = json.loads(Path(compact_summary).read_text(encoding="utf-8"))
        compact_models = compact.get("models", {})
        for label, model_key in SMALL_REFERENCE_MODELS:
            model = compact_models.get(model_key)
            if model is None or model.get("resources") is None:
                continue
            points.append(
                _point(label, "small", float(model["night_macro_recall1"]), model["resources"])
            )

    colors = {
        "ours": "#009E73",
        "cnn": "#0072B2",
        "small": "#E69F00",
    }
    markers = {"ours": "*", "cnn": "o", "small": "D"}
    offsets = {
        "CDND-512": ((5, 5), (5, -10)),
        "CosPlace": ((5, 5), (5, -9)),
        "EigenPlaces": ((5, 5), (5, 5)),
        "MixVPR-4096": ((5, 5), (5, 5)),
        "SALAD-S + CDND": ((5, -10), (5, -11)),
    }
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), constrained_layout=True)
    x_fields = (
        ("parameters_million", "Parameters (M)", False),
        ("storage_per_million_fp16_gib", "Storage per 1M images, FP16 (GiB)", True),
    )
    for axis_index, (axis, (field, label, log_scale)) in enumerate(zip(axes, x_fields)):
        for point in points:
            family = point["family"]
            axis.scatter(
                point[field], 100 * point["night_macro_recall1"],
                color=colors[family], marker=markers[family],
                s=80 if family == "ours" else 28, linewidths=0.8,
                edgecolors="black" if family == "ours" else colors[family], zorder=3,
            )
            dx, dy = offsets[point["label"]][axis_index]
            axis.annotate(
                point["label"], (point[field], 100 * point["night_macro_recall1"]),
                xytext=(dx, dy), textcoords="offset points", fontsize=8.5,
            )
        if log_scale:
            axis.set_xscale("log")
        axis.set_xlabel(label)
        axis.set_ylim(45, 98)
        axis.grid(True, linestyle=":", linewidth=0.75, alpha=0.65)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Three-protocol mean Recall@1 (%)")
    axes[0].set_title("(a) Model capacity")
    axes[1].set_title("(b) Descriptor storage")
    axes[1].legend(
        handles=[
            plt.Line2D([], [], color=colors["ours"], marker=markers["ours"], linestyle="", label="CDND"),
            plt.Line2D([], [], color=colors["cnn"], marker=markers["cnn"], linestyle="", label="CNN"),
            plt.Line2D([], [], color=colors["small"], marker=markers["small"], linestyle="", label="Retrained DINOv2-S"),
        ],
        loc="lower right", frameon=False,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    write_json(
        {
            "schema": "paper_accuracy_efficiency_v1",
            "night_macro_protocols": list(NIGHT_MACRO),
            "gpu": gpu,
            "student_statistic": student_statistic,
            "points": points,
        },
        output.with_suffix(".json"),
    )
    return output
