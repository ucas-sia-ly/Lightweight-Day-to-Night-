from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import apply_overrides, load_config
from .data import audit_data
from .evaluation import evaluate_model
from .generator.trainer import train_generator
from .generation_evaluation import evaluate_generation
from .loss_selection import compare_loss_selection, compare_universal_loss_selection
from .prepare import build_manifests
from .generator.translate import translate_suite_databases, translate_training_set
from .runtime import benchmark_model, export_onnx, load_vpr_model
from .reference_models import (
    audit_reference_models,
    benchmark_reference_models,
    evaluate_vpr_suite,
    fetch_reference_model,
)
from .reference_suite import compare_suite_bootstrap, evaluate_reference_suite
from .nocplace_repro import (
    prepare_nocplace_repro,
    train_negcut_reference,
    train_nocplace_reference,
    translate_nocplace_train,
)
from .transforms import VPRTransform
from .trained_suite import SUITE_PROTOCOLS, evaluate_trained_suite
from .robotcar_grid import (
    compare_robotcar_grids, evaluate_reference_robotcar_grid, evaluate_robotcar_grid,
)
from .vpr.trainer import train_vpr
from .vpr.teacher_cache import cache_relation_teacher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="a2b-lightvpr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(name: str):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="SECTION.FIELD=VALUE",
            help="Repeatable typed YAML configuration override.",
        )
        return command

    audit = common("audit-data")
    audit.add_argument("--decode", action="store_true")

    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--output-dir", default="data/manifests")
    prepare.add_argument("--skip-darkdriving-decode", action="store_true")
    prepare.add_argument("--clip-checkpoint", default="/home/admin123/.cache/clip/ViT-B-32.pt")

    generator = common("train-generator")
    generator.add_argument("--smoke-steps", type=int, default=0)

    translate = common("translate-train")
    translate.add_argument("--checkpoint", default="")
    translate.add_argument("--limit", type=int, default=0, help="Smoke-test only; 0 translates all train rows.")

    translate_suite = common("translate-suite-database")
    translate_suite.add_argument("--checkpoint", default="")
    translate_suite.add_argument(
        "--datasets", nargs="+", required=True,
        choices=["robotcar", "svox_night", "msls_night", "tokyo_night"],
    )
    translate_suite.add_argument("--limit", type=int, default=0)

    generation_eval = common("evaluate-generation")
    generation_eval.add_argument("--checkpoint", required=True)
    generation_eval.add_argument("--dataset", choices=["tokyo", "aachen"], default="tokyo")
    generation_eval.add_argument("--data-root", default="")
    generation_eval.add_argument("--output-dir", default="")
    generation_eval.add_argument("--limit", type=int, default=0)
    generation_eval.add_argument("--timing-iterations", type=int, default=200)

    vpr = common("train-vpr")
    vpr.add_argument("--smoke-epochs", type=int, default=0)

    common("cache-vpr-teacher")

    reference_audit = subparsers.add_parser("audit-reference-models")
    reference_audit.add_argument("--config", required=True)
    reference_audit.add_argument("--output", default="runs/reference/audit.json")

    reference_fetch = subparsers.add_parser("fetch-reference-model")
    reference_fetch.add_argument("--config", required=True)
    reference_fetch.add_argument("--model", required=True)

    reference_benchmark = common("benchmark-reference-models")
    reference_benchmark.add_argument("--models", nargs="+", default=["all"])
    reference_benchmark.add_argument("--warmup", type=int, default=20)
    reference_benchmark.add_argument("--iterations", type=int, default=100)
    reference_benchmark.add_argument("--output", default="runs/reference/resource_benchmark.json")

    reference_eval = common("evaluate-vpr-suite")
    reference_eval.add_argument("--model", required=True)
    reference_eval.add_argument("--dataset", required=True,
        choices=["robotcar", "svox_night", "msls_night", "tokyo_night", "tokyo_sunset"])
    reference_eval.add_argument("--transform", choices=["native", "unified"], default="native")
    reference_eval.add_argument("--protocol", choices=["project", "paper"], default="paper")
    reference_eval.add_argument("--split", choices=["val", "test"], default="test")
    reference_eval.add_argument("--output", default="")

    reference_suite = common("evaluate-reference-suite")
    reference_suite.add_argument("--model", required=True)
    reference_suite.add_argument(
        "--database-encoder", choices=["original", "adapted"], default="original"
    )
    reference_suite.add_argument(
        "--transform", choices=["native", "unified"], default="unified"
    )
    reference_suite.add_argument(
        "--datasets", nargs="+", default=["all"],
        choices=["all", "all-day", "all-conditions", *SUITE_PROTOCOLS]
    )
    reference_suite.add_argument("--output-dir", default="")
    reference_suite.add_argument("--batch-size", type=int, default=0)
    reference_suite.add_argument("--allow-batch-dependent", action="store_true")

    compare_suite = subparsers.add_parser("compare-suite-bootstrap")
    compare_suite.add_argument("--candidate", required=True)
    compare_suite.add_argument("--baseline", required=True)
    compare_suite.add_argument("--output", required=True)
    compare_suite.add_argument("--samples", type=int, default=10_000)

    def add_loss_comparison_arguments(command):
        command.add_argument("--candidate-validation", required=True)
        command.add_argument("--baseline-validation", required=True)
        command.add_argument("--candidate-suite", required=True)
        command.add_argument("--baseline-suite", required=True)
        command.add_argument("--candidate-name", default="preserve")
        command.add_argument("--baseline-name", default="no-preserve")
        command.add_argument("--output", required=True)
        command.add_argument("--samples", type=int, default=10_000)
        command.add_argument("--seed", type=int, default=0)

    compare_loss = subparsers.add_parser("compare-loss-selection")
    add_loss_comparison_arguments(compare_loss)
    compare_universal = subparsers.add_parser("compare-universal-loss-selection")
    add_loss_comparison_arguments(compare_universal)

    trained_suite = common("evaluate-trained-suite")
    trained_suite.add_argument("--checkpoint", required=True)
    trained_suite.add_argument(
        "--datasets", nargs="+", default=["all"],
        choices=["all", "all-day", "all-conditions", *SUITE_PROTOCOLS],
    )

    robotcar_grid = common("evaluate-robotcar-grid")
    robotcar_grid.add_argument("--checkpoint", required=True)
    robotcar_grid.add_argument("--conditions", nargs="+", default=["night", "day"], choices=["night", "day"])
    robotcar_grid.add_argument("--distances", nargs="+", type=float, default=[5, 10, 25])
    robotcar_grid.add_argument("--yaw-degrees", nargs="+", default=["10", "30", "none"])
    robotcar_grid.add_argument("--output", default="")

    reference_grid = common("evaluate-reference-robotcar-grid")
    reference_grid.add_argument("--model", required=True)
    reference_grid.add_argument("--database-encoder", choices=["original", "adapted"], default="adapted")
    reference_grid.add_argument("--transform", choices=["native", "unified"], default="unified")
    reference_grid.add_argument("--conditions", nargs="+", default=["night", "day"], choices=["night", "day"])
    reference_grid.add_argument("--distances", nargs="+", type=float, default=[5, 10, 25])
    reference_grid.add_argument("--yaw-degrees", nargs="+", default=["10", "30", "none"])
    reference_grid.add_argument("--output", default="")

    compare_grid = subparsers.add_parser("compare-robotcar-grid")
    compare_grid.add_argument("--candidate", required=True)
    compare_grid.add_argument("--baseline", required=True)
    compare_grid.add_argument("--output", required=True)
    compare_grid.add_argument("--samples", type=int, default=10_000)

    common("prepare-nocplace-repro")
    negcut_reference = common("train-negcut-reference")
    negcut_reference.add_argument(
        "--dry-run", action="store_true", help="Validate and print the exact official command only."
    )
    negcut_reference.add_argument(
        "--smoke-iterations", type=int, default=0,
        help="Run a separate one-epoch official NEG-CUT smoke experiment on N images.",
    )
    nocplace_translate = common("translate-nocplace-train")
    nocplace_translate.add_argument("--checkpoint", required=True)
    nocplace_translate.add_argument(
        "--limit", type=int, default=0, help="Smoke-test only; formal translation requires 118,313 rows."
    )
    nocplace_train = common("train-nocplace-reference")
    nocplace_train.add_argument(
        "--smoke-iterations", type=int, default=0,
        help="Smoke-test only; formal training always uses 10,000 iterations.",
    )

    evaluate = common("evaluate")
    evaluate.add_argument("--checkpoint", default="")
    evaluate.add_argument("--split", choices=["val", "test"], default="val")
    evaluate.add_argument("--baseline-json", default="")

    benchmark = common("benchmark")
    benchmark.add_argument("--checkpoint", default="")
    benchmark.add_argument("--warmup", type=int, default=100)
    benchmark.add_argument("--iterations", type=int, default=1000)

    export = common("export-onnx")
    export.add_argument("--checkpoint", default="")
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare-data":
        print(json.dumps(build_manifests(
            args.data_root, args.output_dir,
            decode_darkdriving=not args.skip_darkdriving_decode,
            clip_checkpoint=args.clip_checkpoint,
        ), indent=2, sort_keys=True))
        return
    if args.command == "audit-reference-models":
        result = audit_reference_models(args.config, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "fetch-reference-model":
        print(json.dumps(fetch_reference_model(args.config, args.model), indent=2, sort_keys=True))
        return
    if args.command == "compare-suite-bootstrap":
        print(compare_suite_bootstrap(
            args.candidate, args.baseline, args.output, samples=args.samples,
        ))
        return
    if args.command == "compare-loss-selection":
        print(compare_loss_selection(
            args.candidate_validation,
            args.baseline_validation,
            args.candidate_suite,
            args.baseline_suite,
            args.output,
            candidate_name=args.candidate_name,
            baseline_name=args.baseline_name,
            samples=args.samples,
            seed=args.seed,
        ))
        return
    if args.command == "compare-universal-loss-selection":
        print(compare_universal_loss_selection(
            args.candidate_validation, args.baseline_validation,
            args.candidate_suite, args.baseline_suite, args.output,
            candidate_name=args.candidate_name, baseline_name=args.baseline_name,
            samples=args.samples, seed=args.seed,
        ))
        return
    if args.command == "compare-robotcar-grid":
        print(compare_robotcar_grids(
            args.candidate, args.baseline, args.output, samples=args.samples,
        ))
        return
    config = apply_overrides(load_config(args.config), args.set)
    if args.command == "audit-data":
        print(json.dumps(audit_data(config.data, decode=args.decode), indent=2, sort_keys=True))
        return
    if args.command == "train-generator":
        print(train_generator(config, smoke_steps=args.smoke_steps or None))
        return
    if args.command == "translate-train":
        print(
            translate_training_set(
                config, args.checkpoint or None, limit=args.limit or None
            )
        )
        return
    if args.command == "translate-suite-database":
        print("\n".join(str(path) for path in translate_suite_databases(
            config, tuple(args.datasets), args.checkpoint or None,
            limit=args.limit or None,
        )))
        return
    if args.command == "evaluate-generation":
        print(evaluate_generation(
            config, args.checkpoint, dataset=args.dataset, data_root=args.data_root,
            output_dir=args.output_dir or None, limit=args.limit,
            timing_iterations=args.timing_iterations,
        ))
        return
    if args.command == "train-vpr":
        print(train_vpr(config, smoke_epochs=args.smoke_epochs or None))
        return
    if args.command == "cache-vpr-teacher":
        print(cache_relation_teacher(config))
        return
    if args.command == "benchmark-reference-models":
        print(benchmark_reference_models(
            config, args.config, tuple(args.models), warmup=args.warmup,
            iterations=args.iterations, output=args.output,
        ))
        return
    if args.command == "evaluate-vpr-suite":
        print(evaluate_vpr_suite(
            config, args.config, model_name=args.model, dataset=args.dataset,
            transform_name=args.transform, protocol=args.protocol, split=args.split,
            output=args.output or None,
        ))
        return
    if args.command == "evaluate-reference-suite":
        print(evaluate_reference_suite(
            config,
            args.config,
            model_name=args.model,
            database_encoder=args.database_encoder,
            transform_name=args.transform,
            datasets=tuple(args.datasets),
            output_dir=args.output_dir or None,
            batch_size_override=args.batch_size or None,
            allow_batch_dependent=args.allow_batch_dependent,
        ))
        return
    if args.command == "evaluate-trained-suite":
        print(evaluate_trained_suite(config, args.checkpoint, tuple(args.datasets)))
        return
    if args.command == "evaluate-robotcar-grid":
        print(evaluate_robotcar_grid(
            config, args.checkpoint, conditions=tuple(args.conditions),
            distances=tuple(args.distances), yaw_degrees=tuple(args.yaw_degrees),
            output=args.output or None,
        ))
        return
    if args.command == "evaluate-reference-robotcar-grid":
        print(evaluate_reference_robotcar_grid(
            config, args.config, model_name=args.model,
            database_encoder=args.database_encoder, transform_name=args.transform,
            conditions=tuple(args.conditions), distances=tuple(args.distances),
            yaw_degrees=tuple(args.yaw_degrees), output=args.output or None,
        ))
        return
    if args.command == "prepare-nocplace-repro":
        print(json.dumps(prepare_nocplace_repro(config), indent=2, sort_keys=True))
        return
    if args.command == "train-negcut-reference":
        print(json.dumps(train_negcut_reference(
            config, dry_run=args.dry_run, smoke_iterations=args.smoke_iterations or None,
        ), indent=2, sort_keys=True))
        return
    if args.command == "translate-nocplace-train":
        print(json.dumps(translate_nocplace_train(
            config, args.checkpoint, limit=args.limit or None,
        ), indent=2, sort_keys=True))
        return
    if args.command == "train-nocplace-reference":
        print(json.dumps(train_nocplace_reference(
            config, smoke_iterations=args.smoke_iterations or None,
        ), indent=2, sort_keys=True))
        return
    if args.command == "evaluate":
        model, _, _ = load_vpr_model(config, args.checkpoint or None)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        output = Path(config.vpr.output_dir) / f"{args.split}_metrics.json"
        metrics = evaluate_model(
            model,
            config.data.robotcar_manifest,
            args.split,
            VPRTransform(config.vpr.image_height, config.vpr.image_width),
            config.evaluation,
            device,
            batch_size=max(1, config.vpr.batch_size * 2),
            workers=config.vpr.workers,
            output_path=output,
            baseline_path=args.baseline_json or None,
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return
    if args.command == "benchmark":
        print(
            json.dumps(
                benchmark_model(
                    config,
                    args.checkpoint or None,
                    warmup=args.warmup,
                    iterations=args.iterations,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "export-onnx":
        print(export_onnx(config, args.checkpoint or None))
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
