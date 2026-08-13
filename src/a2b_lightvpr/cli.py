from __future__ import annotations

import argparse
import json

from .config import apply_overrides, load_config
from .data import audit_data
from .external_temporal import (
    evaluate_gardens_reference,
    evaluate_gardens_temporal,
    summarize_gardens_models,
)
from .generator.trainer import train_generator
from .generator.translate import translate_training_set
from .prepare import build_manifests
from .reference_models import (
    audit_reference_models,
    benchmark_reference_models,
    fetch_reference_model,
)
from .reference_suite import evaluate_reference_suite
from .ragtr import (
    build_ragtr_geometry_cache,
    evaluate_robotcar_ragtr,
    select_robotcar_ragtr,
    summarize_robotcar_ragtr,
)
from .robotcar_temporal import (
    evaluate_robotcar_temporal,
    select_robotcar_temporal,
    summarize_robotcar_temporal,
)
from .runtime import benchmark_model
from .single_run_summary import summarize_cdnd_seed0
from .trained_suite import SUITE_PROTOCOLS, evaluate_trained_suite
from .vpr.teacher_cache import cache_relation_teacher
from .vpr.route_mining import mine_robotcar_route_candidates
from .vpr.trainer import export_vpr_inference_checkpoint, train_vpr


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="a2b-lightvpr")
    commands = parser.add_subparsers(dest="command", required=True)

    def common(name: str):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument(
            "--set", action="append", default=[], metavar="SECTION.FIELD=VALUE"
        )
        return command

    audit = common("audit-data")
    audit.add_argument("--decode", action="store_true")

    prepare = commands.add_parser("prepare-data")
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--output-dir", default="data/manifests_clean_night")
    prepare.add_argument("--skip-darkdriving-decode", action="store_true")
    prepare.add_argument("--clip-checkpoint", default="/home/admin123/.cache/clip/ViT-B-32.pt")

    generator = common("train-generator")
    generator.add_argument("--smoke-steps", type=int, default=0)
    translate = common("translate-train")
    translate.add_argument("--checkpoint", default="")
    translate.add_argument("--limit", type=int, default=0)

    vpr = common("train-vpr")
    vpr.add_argument("--smoke-epochs", type=int, default=0)
    common("cache-vpr-teacher")

    export = common("export-vpr-inference")
    export.add_argument("--training-checkpoint", required=True)
    export.add_argument("--output", required=True)

    trained = common("evaluate-trained-suite")
    trained.add_argument("--checkpoint", required=True)
    trained.add_argument(
        "--datasets", nargs="+", default=["all"],
        choices=["all", "all-day", "all-conditions", *SUITE_PROTOCOLS],
    )

    benchmark = common("benchmark")
    benchmark.add_argument("--checkpoint", default="")
    benchmark.add_argument("--warmup", type=int, default=100)
    benchmark.add_argument("--iterations", type=int, default=1000)

    summary = commands.add_parser("summarize-cdnd-seed0")
    summary.add_argument("--baseline", action="append", required=True, metavar="DIM=JSON")
    summary.add_argument("--candidate", action="append", required=True, metavar="DIM=JSON")
    summary.add_argument("--output", required=True)

    audit_reference = commands.add_parser("audit-reference-models")
    audit_reference.add_argument("--config", required=True)
    audit_reference.add_argument("--output", default="runs/reference/audit.json")
    fetch = commands.add_parser("fetch-reference-model")
    fetch.add_argument("--config", required=True)
    fetch.add_argument("--model", required=True)

    reference_benchmark = common("benchmark-reference-models")
    reference_benchmark.add_argument("--models", nargs="+", default=["all"])
    reference_benchmark.add_argument("--warmup", type=int, default=20)
    reference_benchmark.add_argument("--iterations", type=int, default=100)
    reference_benchmark.add_argument(
        "--output", default="runs/reference/resource_benchmark.json"
    )

    reference = common("evaluate-reference-suite")
    reference.add_argument("--model", required=True)
    reference.add_argument(
        "--database-encoder", choices=["original", "adapted"], default="original"
    )
    reference.add_argument("--transform", choices=["native", "unified"], default="unified")
    reference.add_argument(
        "--datasets", nargs="+", default=["all"],
        choices=["all", "all-day", "all-conditions", *SUITE_PROTOCOLS],
    )
    reference.add_argument("--output-dir", default="")
    reference.add_argument("--batch-size", type=int, default=0)
    reference.add_argument("--allow-batch-dependent", action="store_true")

    temporal_select = common("select-robotcar-temporal")
    temporal_select.add_argument("--checkpoint", required=True)
    temporal_select.add_argument("--cache", required=True)
    temporal_select.add_argument("--output", required=True)
    temporal_select.add_argument("--geometry-cache", default="")
    temporal_select.add_argument("--method", choices=["mgtr-v1", "ragtr"], default="mgtr-v1")
    temporal_evaluate = common("evaluate-robotcar-temporal")
    temporal_evaluate.add_argument("--checkpoint", required=True)
    temporal_evaluate.add_argument("--selection", required=True)
    temporal_evaluate.add_argument("--condition", choices=["night", "day"], required=True)
    temporal_evaluate.add_argument("--cache", required=True)
    temporal_evaluate.add_argument("--output", required=True)
    temporal_evaluate.add_argument("--ablations-output", default="")
    temporal_evaluate.add_argument("--geometry-cache", default="")
    temporal_evaluate.add_argument(
        "--method", choices=["mgtr-v1", "ragtr"], default="mgtr-v1"
    )
    temporal_summary = commands.add_parser("summarize-robotcar-temporal")
    for name in ("night", "day", "crica", "salad", "boq", "reference-resources", "output"):
        temporal_summary.add_argument(f"--{name}", required=True)
    temporal_summary.add_argument(
        "--method", choices=["mgtr-v1", "ragtr"], default="mgtr-v1"
    )

    geometry = common("cache-robotcar-geometry")
    geometry.add_argument("--checkpoint", required=True)
    geometry.add_argument("--condition", choices=["val", "night", "day"], required=True)
    geometry.add_argument("--cache", required=True)
    geometry.add_argument("--geometry-cache", required=True)

    mining = common("mine-robotcar-route")
    mining.add_argument("--checkpoint", required=True)
    mining.add_argument("--output", default="")

    gardens = common("evaluate-gardens-temporal")
    gardens.add_argument("--checkpoint", required=True)
    gardens.add_argument("--root", required=True)
    gardens.add_argument("--output-dir", required=True)
    gardens.add_argument("--method", choices=["mgtr-v1", "ragtr"], default="mgtr-v1")
    gardens_reference = common("evaluate-gardens-reference")
    gardens_reference.add_argument("--registry", required=True)
    gardens_reference.add_argument("--model", required=True)
    gardens_reference.add_argument("--root", required=True)
    gardens_reference.add_argument("--output-dir", required=True)
    gardens_reference.add_argument(
        "--database-encoder", choices=["adapted", "original"], default="adapted"
    )
    gardens_summary = commands.add_parser("summarize-gardens-models")
    gardens_summary.add_argument("--results-root", required=True)
    gardens_summary.add_argument("--primary-result", default="")
    gardens_summary.add_argument("--output", required=True)
    return parser


def _dimensions(expressions: list[str]) -> dict[int, str]:
    result = {}
    for expression in expressions:
        if "=" not in expression:
            raise ValueError("CDND result must use DIM=SUMMARY_JSON")
        dimension, path = expression.split("=", 1)
        dimension = int(dimension)
        if dimension in result:
            raise ValueError(f"Duplicate descriptor dimension: {dimension}")
        result[dimension] = path
    return result


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare-data":
        result = build_manifests(
            args.data_root, args.output_dir,
            decode_darkdriving=not args.skip_darkdriving_decode,
            clip_checkpoint=args.clip_checkpoint,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "summarize-cdnd-seed0":
        print(summarize_cdnd_seed0(
            _dimensions(args.baseline), _dimensions(args.candidate), args.output
        ))
        return
    if args.command == "audit-reference-models":
        print(json.dumps(
            audit_reference_models(args.config, args.output), indent=2, sort_keys=True
        ))
        return
    if args.command == "fetch-reference-model":
        print(json.dumps(
            fetch_reference_model(args.config, args.model), indent=2, sort_keys=True
        ))
        return
    if args.command == "summarize-robotcar-temporal":
        if args.method == "ragtr":
            print(summarize_robotcar_ragtr(
                args.night, args.day, args.salad, args.boq, args.output
            ))
        else:
            print(summarize_robotcar_temporal(
                args.night, args.day, args.crica, args.salad, args.boq, args.output,
                args.reference_resources,
            ))
        return
    if args.command == "summarize-gardens-models":
        print(summarize_gardens_models(
            args.results_root, args.output, args.primary_result or None
        ))
        return

    config = apply_overrides(load_config(args.config), args.set)
    if args.command == "audit-data":
        result = audit_data(config.data, decode=args.decode)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "train-generator":
        print(train_generator(config, smoke_steps=args.smoke_steps or None))
    elif args.command == "translate-train":
        print(translate_training_set(config, args.checkpoint or None, limit=args.limit or None))
    elif args.command == "train-vpr":
        print(train_vpr(config, smoke_epochs=args.smoke_epochs or None))
    elif args.command == "cache-vpr-teacher":
        print(cache_relation_teacher(config))
    elif args.command == "export-vpr-inference":
        print(export_vpr_inference_checkpoint(
            config, args.training_checkpoint, args.output
        ))
    elif args.command == "evaluate-trained-suite":
        print(evaluate_trained_suite(config, args.checkpoint, tuple(args.datasets)))
    elif args.command == "benchmark":
        print(json.dumps(benchmark_model(
            config, args.checkpoint or None,
            warmup=args.warmup, iterations=args.iterations,
        ), indent=2, sort_keys=True))
    elif args.command == "benchmark-reference-models":
        print(benchmark_reference_models(
            config, args.config, tuple(args.models), warmup=args.warmup,
            iterations=args.iterations, output=args.output,
        ))
    elif args.command == "evaluate-reference-suite":
        print(evaluate_reference_suite(
            config, args.config, model_name=args.model,
            database_encoder=args.database_encoder, transform_name=args.transform,
            datasets=tuple(args.datasets), output_dir=args.output_dir or None,
            batch_size_override=args.batch_size or None,
            allow_batch_dependent=args.allow_batch_dependent,
        ))
    elif args.command == "select-robotcar-temporal":
        if args.method == "ragtr":
            if not args.geometry_cache:
                raise ValueError("RA-GTR selection requires --geometry-cache")
            print(select_robotcar_ragtr(
                config, args.checkpoint, args.output, args.cache, args.geometry_cache
            ))
        else:
            print(select_robotcar_temporal(config, args.checkpoint, args.output, args.cache))
    elif args.command == "evaluate-robotcar-temporal":
        if args.method == "ragtr":
            if not args.geometry_cache:
                raise ValueError("RA-GTR evaluation requires --geometry-cache")
            print(evaluate_robotcar_ragtr(
                config, args.checkpoint, args.selection, args.condition,
                args.output, args.cache, args.geometry_cache,
            ))
        else:
            print(evaluate_robotcar_temporal(
                config, args.checkpoint, args.selection, args.condition,
                args.output, args.cache, args.ablations_output or None,
            ))
    elif args.command == "cache-robotcar-geometry":
        print(build_ragtr_geometry_cache(
            config, args.checkpoint, args.condition, args.cache, args.geometry_cache
        ))
    elif args.command == "mine-robotcar-route":
        print(mine_robotcar_route_candidates(
            config, args.checkpoint, args.output or None
        ))
    elif args.command == "evaluate-gardens-temporal":
        print(evaluate_gardens_temporal(
            config, args.checkpoint, args.root, args.output_dir, method=args.method
        ))
    elif args.command == "evaluate-gardens-reference":
        print(evaluate_gardens_reference(
            config, args.registry, args.model, args.root, args.output_dir,
            args.database_encoder,
        ))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
