#!/usr/bin/env python3
"""Generate LaTeX result tables from the experiment JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ("robotcar_project", "RC-10m/30$^\\circ$"),
    ("svox_night", "SVOX-N"),
    ("msls_night", "MSLS-N (val)"),
    ("tokyo_night", "Tokyo-N"),
    ("tokyo_sunset", "Tokyo-S (transition)"),
)
INDEPENDENT_NIGHT_TESTS = ("robotcar_project", "svox_night", "tokyo_night")


def read_json(relative: str) -> dict | None:
    path = ROOT / relative
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_suite(relative: str) -> dict | None:
    """Read per-dataset metrics.json as the authoritative result source."""
    payload = read_json(relative)
    if payload is None:
        return None
    suite_root = (ROOT / relative).parent
    datasets = dict(payload.get("datasets", {}))
    for metrics_path in suite_root.glob("*/metrics.json"):
        name = metrics_path.parent.name
        if metrics_path.is_file():
            datasets[name] = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {**payload, "datasets": datasets}


def pct(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.2f}"


def metric(dataset: dict, key: str) -> str:
    if key.startswith("recall"):
        return pct(dataset["recalls"][key.rsplit("_", 1)[-1]])
    return pct(dataset[key])


def recalls_r1(
    payload: dict | None, fallback: dict[str, float] | None = None
) -> list[float | None]:
    fallback = fallback or {}
    if payload is None:
        return [fallback.get(name) for name, _ in DATASETS]
    return [
        fallback.get(name) if name not in payload.get("datasets", {})
        else payload["datasets"][name]["recalls"]["1"]
        for name, _ in DATASETS
    ]


def mean_pct(values: list[float | None]) -> str:
    present = [value for value in values if value is not None]
    return "--" if not present else pct(sum(present) / len(present))


def independent_night_macro(payload: dict | None) -> str:
    if payload is None:
        return "--"
    values = []
    for name in INDEPENDENT_NIGHT_TESTS:
        item = payload.get("datasets", {}).get(name)
        if item is None:
            return "--"
        values.append(float(item["recalls"]["1"]))
    return pct(sum(values) / len(values))


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = content.rstrip().replace(r"\\", r"\tabularnewline")
    path.write_text(rows + "\n\\bottomrule\n", encoding="utf-8")


def generation_table(output: Path) -> None:
    rows = []
    for step in (50_000, 100_000, 125_000):
        payload = read_json(f"runs/paper/generation/qsattn_official_tokyo_step_{step}/metrics.json")
        if payload is None:
            rows.append(f"官方 QS-Attn 目标 {step // 1000}k & 待补 & -- & -- & -- & -- & --\\\\")
        else:
            rows.append(
                f"官方 QS-Attn 目标 {step // 1000}k & Tokyo-375 & "
                f"{payload['fid']:.2f} & {payload['kid']['mean']:.4f} & "
                f"{payload['l2']['mean']:.2f} & {payload['psnr']['mean']:.2f} & "
                f"{payload['ssim']['mean']:.3f}\\\\"
            )
    final = read_json("runs/paper/generation/qsattn_official_tokyo/metrics.json")
    if final:
        rows.append(
            f"官方 QS-Attn 目标 150k & Tokyo-375 & {final['fid']:.2f} & "
            f"{final['kid']['mean']:.4f} & {final['l2']['mean']:.2f} & "
            f"{final['psnr']['mean']:.2f} & {final['ssim']['mean']:.3f}\\\\"
        )
    aachen = read_json("runs/paper/generation/qsattn_official_aachen/metrics.json")
    if aachen:
        rows.append(
            f"官方 QS-Attn 目标 150k & Aachen-local & {aachen['fid']:.2f} & "
            f"{aachen['kid']['mean']:.4f} & {aachen['l2']['mean']:.2f} & "
            f"{aachen['psnr']['mean']:.2f} & {aachen['ssim']['mean']:.3f}\\\\"
        )
    write(output / "generation.tex", "\n".join(rows))


def loss_tables(output: Path) -> None:
    selection = read_json("runs/comparisons/official_loss_preserve_vs_day_fake_selection.json")
    if selection is None:
        selection_rows = (
            "Day + fake & -- & -- & -- & 0.00 & --\\\\\n"
            "Day + fake + preserve & -- & -- & -- & -- & --\\\\"
        )
    else:
        validation = selection["validation"]
        macro = selection["macro_recall_at_1"]
        bootstrap = selection["stratified_paired_bootstrap_r1"]
        p_one_sided = bootstrap.get("one_sided_p_non_positive")
        def side(name, which):
            return pct(validation[name][f"{which}_recall_at_1"])
        selection_rows = (
            f"{selection['baseline']} & {side('robotcar_val','baseline')} & "
            f"{side('msls_val','baseline')} & {pct(macro['baseline'])} & 0.00 & --\\\\\n"
            f"{selection['candidate']} & {side('robotcar_val','candidate')} & "
            f"{side('msls_val','candidate')} & {pct(macro['candidate'])} & "
            f"{100*macro['delta']:+.2f} & "
            f"{('--' if p_one_sided is None else f'{p_one_sided:.3f}')}\\\\"
        )
    write(output / "loss_selection.tex", selection_rows)

    configs = (
        ("Day only CM", "runs/vpr/loss_ablation_official_compute_matched/day_only_seed0/final_suite/summary.json"),
        ("Fake only CM", "runs/vpr/loss_ablation_official_compute_matched/fake_only_seed0/final_suite/summary.json"),
        ("Day + fake", "runs/vpr/loss_ablation_official/day_fake_seed0/final_suite/summary.json"),
        ("Day + fake + preserve", "runs/vpr/qsattn_official/mixvpr_r50_512_seed0/final_suite/summary.json"),
    )
    rows = []
    for label, path in configs:
        payload = read_suite(path)
        raw_values = recalls_r1(payload)
        values = [pct(value) for value in raw_values]
        rows.append(
            f"{label} & " + " & ".join(values)
            + f" & {independent_night_macro(payload)}\\\\"
        )
    write(output / "loss_suite.tex", "\n".join(rows))

    day_protocols = (
        ("robotcar_day", "RC-day"), ("svox_day", "SVOX-day"),
        ("svox_sun", "SVOX-sun"), ("tokyo_day", "Tokyo-day"),
    )
    day_models = (
        ("MixVPR official", "runs/reference/mixvpr-r50-512/final_suite/native/adapted/summary.json"),
        ("Day only CM", "runs/vpr/loss_ablation_official_compute_matched/day_only_seed0/final_suite/summary.json"),
        ("Fake only CM", "runs/vpr/loss_ablation_official_compute_matched/fake_only_seed0/final_suite/summary.json"),
        ("Day + fake", "runs/vpr/loss_ablation_official/day_fake_seed0/final_suite/summary.json"),
        ("Day + fake + preserve", "runs/vpr/qsattn_official/mixvpr_r50_512_seed0/final_suite/summary.json"),
    )
    day_rows = []
    for label, path in day_models:
        payload = read_suite(path)
        values = [
            None if payload is None or name not in payload.get("datasets", {})
            else payload["datasets"][name]["recalls"]["1"]
            for name, _ in day_protocols
        ]
        day_rows.append(f"{label} & " + " & ".join(pct(value) for value in values)
                        + f" & {mean_pct(values)}\\\\")
    write(output / "day_ability.tex", "\n".join(day_rows))

    compute_rows = (
        "Day only CM & 120,000 & 120,000 & 0 & 4 & 3,750\\\\\n"
        "Fake only CM & 120,000 & 120,000 & 0 & 4 & 3,750\\\\\n"
        "Day + fake & 60,000 & 120,000 & 0 & 2 & 3,750\\\\\n"
        "Day + fake + preserve & 60,000 & 120,000 & 60,000 & 2 & 3,750\\\\"
    )
    write(output / "compute_matching.tex", compute_rows)


def main_r1_table(output: Path) -> None:
    selection = read_json("runs/comparisons/official_loss_preserve_vs_day_fake_selection.json")
    if selection and selection.get("selected") == selection.get("baseline"):
        a2b_label = "A2B-LightVPR, seed 0（day+fake）"
        a2b_path = "runs/vpr/loss_ablation_official/day_fake_seed0/final_suite/summary.json"
    else:
        a2b_label = "A2B-LightVPR, seed 0（day+fake+preserve）"
        a2b_path = "runs/vpr/qsattn_official/mixvpr_r50_512_seed0/final_suite/summary.json"
    models = (
        (a2b_label, a2b_path),
        ("NetVLAD-4096（官方 QS 伪图适配）", "runs/vpr/netvlad_safe/netvlad_vgg16_4096_seed0/final_suite/summary.json"),
        ("MixVPR-R50-512（官方零样本）", "runs/reference/mixvpr-r50-512/final_suite/native/adapted/summary.json"),
        ("MixVPR-R50-4096（官方零样本）", "runs/reference/mixvpr-r50-4096/final_suite/native/adapted/summary.json"),
        ("CosPlace-R50-512（官方零样本）", "runs/reference/cosplace-r50-512/final_suite/native/adapted/summary.json"),
        ("EigenPlaces-R50-512（官方零样本）", "runs/reference/eigenplaces-r50-512/final_suite/native/adapted/summary.json"),
        ("EigenPlaces-R50-2048（官方零样本）", "runs/reference/eigenplaces-r50-2048/final_suite/native/adapted/summary.json"),
        ("SALAD-DINOv2-544（官方零样本）", "runs/reference/salad-dinov2-544/final_suite/native/adapted/summary.json"),
        ("CricaVPR-DINOv2-10752（官方零样本）", "runs/reference/cricavpr-dinov2-b-10752/final_suite/native/adapted/summary.json"),
        ("BoQ-DINOv2-12288（官方零样本）", "runs/reference/boq-dinov2-12288/final_suite/native/adapted/summary.json"),
        ("NocPlace OD=true（协议复现）", "runs/reference/nocplace-r50-ikt-clean5k/final_suite/unified/original/summary.json"),
        ("NocPlace OD=false（协议复现）", "runs/reference/nocplace-r50-ikt-clean5k/final_suite/unified/adapted/summary.json"),
    )
    rows = []
    for label, path in models:
        payload = read_suite(path)
        raw_values = recalls_r1(payload)
        values = [pct(value) for value in raw_values]
        rows.append(
            f"{label} & " + " & ".join(values)
            + f" & {independent_night_macro(payload)}\\\\"
        )
    write(output / "main_r1.tex", "\n".join(rows))


def adaptation_gain_table(output: Path) -> None:
    selection = read_json("runs/comparisons/official_loss_preserve_vs_day_fake_selection.json")
    adapted_path = (
        "runs/vpr/qsattn_official/mixvpr_r50_512_seed0/final_suite/summary.json"
        if selection and selection.get("candidate_selected")
        else "runs/vpr/loss_ablation_official/day_fake_seed0/final_suite/summary.json"
    )
    adapted = read_suite(adapted_path)
    initial = read_suite(
        "runs/reference/mixvpr-r50-512/final_suite/fitpad_320x320/adapted/summary.json"
    )
    rows = []
    if adapted is not None and initial is not None:
        for dataset, short in DATASETS:
            before_item = initial.get("datasets", {}).get(dataset)
            after_item = adapted.get("datasets", {}).get(dataset)
            if before_item is None or after_item is None:
                rows.append(f"{short} & -- & -- & --\\\\")
                continue
            if not (
                before_item.get("evaluation_fingerprint")
                and before_item.get("evaluation_fingerprint")
                == after_item.get("evaluation_fingerprint")
            ):
                rows.append(f"{short} & -- & -- & --\\\\")
                continue
            before = before_item["recalls"]["1"]
            after = after_item["recalls"]["1"]
            rows.append(f"{short} & {pct(before)} & {pct(after)} & {100*(after-before):+.2f}\\\\")
    else:
        rows.append("待运行 & -- & -- & --\\\\")
    write(output / "adaptation_gain.tex", "\n".join(rows))


def controlled_preprocessing_table(output: Path) -> None:
    models = (
        ("A2B-LightVPR", "runs/vpr/qsattn_official/mixvpr_r50_512_seed0/final_suite/summary.json"),
        ("MixVPR official", "runs/reference/mixvpr-r50-512/final_suite/fitpad_320x320/adapted/summary.json"),
        ("NocPlace OD=true", "runs/reference/nocplace-r50-ikt-clean5k/final_suite/fitpad_320x320/original/summary.json"),
        ("NocPlace OD=false", "runs/reference/nocplace-r50-ikt-clean5k/final_suite/fitpad_320x320/adapted/summary.json"),
    )
    rows = []
    for label, path in models:
        payload = read_suite(path)
        values = recalls_r1(payload)
        rows.append(
            f"{label} & " + " & ".join(pct(value) for value in values)
            + f" & {independent_night_macro(payload)}\\\\"
        )
    write(output / "controlled_preprocessing.tex", "\n".join(rows))


def resources_table(output: Path) -> None:
    payload = read_json("runs/reference/resource_benchmark.json")
    wanted = (
        "mixvpr-r50-512",
        "mixvpr-r50-4096",
        "cosplace-r50-512",
        "eigenplaces-r50-512",
        "eigenplaces-r50-2048",
        "salad-dinov2-544",
        "cricavpr-dinov2-b-10752",
        "boq-dinov2-12288",
        "netvlad-vgg16-pitts4096",
    )
    rows = []
    for name in wanted:
        item = None if payload is None else payload["models"].get(name)
        if item is None:
            rows.append(f"{name} & -- & -- & -- & -- & -- & -- & --\\\\")
            continue
        checkpoint_mib = item["checkpoint_bytes"] / (1024**2)
        memory = item["database_descriptor_memory"]["1000000"]["fp32_gib"]
        rows.append(
            f"{name} & {item['parameters']/1e6:.2f} & {checkpoint_mib:.1f} & "
            f"{item['descriptor_dim']} & {memory:.2f} & {item['peak_cuda_mib']:.1f} & "
            f"{item['batch1_p50_ms']:.2f} & {item['batch1_p95_ms']:.2f}\\\\"
        )
    write(output / "resources.tex", "\n".join(rows))


def appendix_metrics(output: Path) -> None:
    selection = read_json("runs/comparisons/official_loss_preserve_vs_day_fake_selection.json")
    if selection and selection.get("selected") == selection.get("baseline"):
        a2b_path = "runs/vpr/loss_ablation_official/day_fake_seed0/final_suite/summary.json"
    else:
        a2b_path = "runs/vpr/qsattn_official/mixvpr_r50_512_seed0/final_suite/summary.json"
    models = (
        ("A2B-LightVPR", a2b_path),
        ("NetVLAD-adapted", "runs/vpr/netvlad_safe/netvlad_vgg16_4096_seed0/final_suite/summary.json"),
        ("MixVPR-512", "runs/reference/mixvpr-r50-512/final_suite/native/adapted/summary.json"),
        ("MixVPR-4096", "runs/reference/mixvpr-r50-4096/final_suite/native/adapted/summary.json"),
        ("CosPlace-512", "runs/reference/cosplace-r50-512/final_suite/native/adapted/summary.json"),
        ("EigenPlaces-512", "runs/reference/eigenplaces-r50-512/final_suite/native/adapted/summary.json"),
        ("EigenPlaces-2048", "runs/reference/eigenplaces-r50-2048/final_suite/native/adapted/summary.json"),
        ("SALAD", "runs/reference/salad-dinov2-544/final_suite/native/adapted/summary.json"),
        ("CricaVPR", "runs/reference/cricavpr-dinov2-b-10752/final_suite/native/adapted/summary.json"),
        ("BoQ", "runs/reference/boq-dinov2-12288/final_suite/native/adapted/summary.json"),
        ("NocPlace-OD-true", "runs/reference/nocplace-r50-ikt-clean5k/final_suite/unified/original/summary.json"),
        ("NocPlace-OD-false", "runs/reference/nocplace-r50-ikt-clean5k/final_suite/unified/adapted/summary.json"),
    )
    rows = []
    fingerprints = []
    for label, path in models:
        payload = read_suite(path)
        if payload is None:
            continue
        for dataset, short in DATASETS:
            item = payload.get("datasets", {}).get(dataset)
            if item is None:
                continue
            rows.append(
                f"{label} & {short} & {metric(item, 'recall_1')} & {metric(item, 'recall_5')} & "
                f"{metric(item, 'recall_10')} & {metric(item, 'mrr')} & "
                f"{item['database']} & {item['queries']} & {item['valid_queries']}\\\\"
            )
            query_fp = item.get(
                "query_protocol_fingerprint", item.get("protocol_fingerprint", "--")
            )
            eval_fp = item.get("evaluation_fingerprint", "legacy/pending")
            fingerprints.append(
                f"{label} / {short} & \\texttt{{\\seqsplit{{{query_fp}}}}} & "
                f"\\texttt{{\\seqsplit{{{eval_fp}}}}}\\\\"
            )
    write(output / "appendix_metrics.tex", "\n".join(rows))
    unique = list(dict.fromkeys(fingerprints))
    write(output / "protocol_fingerprints.tex", "\n".join(unique))


def nocplace_bootstrap_table(output: Path) -> None:
    selection = read_json("runs/comparisons/official_loss_preserve_vs_day_fake_selection.json")
    prefix = "official_preserve" if selection and selection.get("candidate_selected") else "official_day_fake"
    comparisons = (
        ("OD=true", f"runs/comparisons/{prefix}_vs_nocplace_original.json"),
        ("OD=false", f"runs/comparisons/{prefix}_vs_nocplace_adapted.json"),
    )
    rows = []
    for label, path in comparisons:
        payload = read_json(path)
        if payload is None:
            continue
        for dataset, short in DATASETS:
            item = payload["datasets"][dataset]
            bootstrap = item["paired_bootstrap_r1"]
            lower, upper = bootstrap["ci95"]
            rows.append(
                f"{label} & {short} & {pct(item['candidate_recall_at_1'])} & "
                f"{pct(item['baseline_recall_at_1'])} & {100*bootstrap['delta']:.2f} & "
                f"[{100*lower:.2f}, {100*upper:.2f}]\\\\"
            )
    if not rows:
        rows.append("待运行 & -- & -- & -- & -- & --\\\\")
    write(output / "nocplace_bootstrap.tex", "\n".join(rows))


def seed_stability_table(output: Path) -> None:
    rows = []
    for seed in (0, 1, 2):
        payload = read_suite(
            f"runs/vpr/qsattn_official/mixvpr_r50_512_seed{seed}/final_suite/summary.json"
        )
        values = recalls_r1(payload)
        rows.append(
            f"{seed} & " + " & ".join(pct(value) for value in values)
            + f" & {independent_night_macro(payload)}\\\\"
        )
    write(output / "seed_stability.tex", "\n".join(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="paper/generated_tables")
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    generation_table(output)
    loss_tables(output)
    main_r1_table(output)
    adaptation_gain_table(output)
    controlled_preprocessing_table(output)
    resources_table(output)
    appendix_metrics(output)
    nocplace_bootstrap_table(output)
    seed_stability_table(output)
    print(output)


if __name__ == "__main__":
    main()
