from __future__ import annotations

import argparse
import json

from .config import apply_overrides, load_config
from .data import audit_data
from .generator.trainer import train_generator
from .generator.translate import translate_training_set
from .prepare import build_manifests
from .reporting.multiseed_summary import summarize_cdnd_multiseed
from .reporting.figures import plot_accuracy_efficiency
from .reporting.paired_summary import summarize_paired_vpr
from .qualitative.export import export_qualitative_top1
from .references.models import (
    audit_reference_models,
    benchmark_reference_models,
    fetch_reference_model,
    validate_reference_resources,
)
from .references.coverage import audit_reference_coverage
from .references.pca import fit_reference_pca
from .references.suite import evaluate_reference_suite
from .robotcar_temporal import (
    evaluate_robotcar_temporal,
    select_robotcar_temporal,
    summarize_robotcar_temporal,
)
from .runtime import benchmark_model
from .reporting.seed0_summary import summarize_cdnd_seed0
from .references.training import fit_anyloc_small_vocabulary, train_boq_small
from .references.summary import summarize_small_references
from .trained_suite import SUITE_PROTOCOLS, evaluate_trained_suite
from .vpr.teacher_cache import cache_relation_teacher
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

    multi_summary = commands.add_parser("summarize-cdnd-multiseed")
    multi_summary.add_argument("--baseline", action="append", required=True, metavar="SEED=JSON")
    multi_summary.add_argument("--candidate", action="append", required=True, metavar="SEED=JSON")
    multi_summary.add_argument("--ablation", action="append", required=True, metavar="NAME=JSON")
    multi_summary.add_argument("--output", required=True)

    paired_summary = commands.add_parser("summarize-paired-vpr")
    paired_summary.add_argument("--baseline", required=True)
    paired_summary.add_argument("--candidate", required=True)
    paired_summary.add_argument("--label", required=True)
    paired_summary.add_argument("--output", required=True)

    paper_plot = commands.add_parser("plot-paper-efficiency")
    paper_plot.add_argument("--student-summary", required=True)
    paper_plot.add_argument("--reference-root", default="runs/reference")
    paper_plot.add_argument("--resource-summary", action="append", required=True)
    paper_plot.add_argument("--compact-summary")
    paper_plot.add_argument("--output", required=True)

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

    fit_pca = common("fit-reference-pca")
    fit_pca.add_argument("--registry", required=True)
    fit_pca.add_argument("--model", required=True)
    fit_pca.add_argument("--output", required=True)
    fit_pca.add_argument("--cache", required=True)
    fit_pca.add_argument("--components", type=int, default=512)
    fit_pca.add_argument("--samples", type=int, default=20_000)
    fit_pca.add_argument("--seed", type=int, default=0)
    fit_pca.add_argument("--batch-size", type=int, default=16)
    fit_pca.add_argument("--workers", type=int, default=4)

    boq_small = commands.add_parser("train-boq-small")
    boq_small.add_argument("--config", required=True)
    boq_small.add_argument("--smoke-steps", type=int, default=0)

    anyloc_vocab = commands.add_parser("fit-anyloc-small-vocabulary")
    anyloc_vocab.add_argument("--registry", required=True)
    anyloc_vocab.add_argument("--model", required=True)
    anyloc_vocab.add_argument("--output", required=True)
    anyloc_vocab.add_argument("--samples", type=int, default=20_000)
    anyloc_vocab.add_argument("--patches-per-image", type=int, default=32)
    anyloc_vocab.add_argument("--seed", type=int, default=0)
    anyloc_vocab.add_argument("--batch-size", type=int, default=16)
    anyloc_vocab.add_argument("--workers", type=int, default=4)

    small_summary = commands.add_parser("summarize-small-references")
    small_summary.add_argument("--registry", required=True)
    small_summary.add_argument("--reference-root", default="runs/reference")
    small_summary.add_argument("--resources", action="append", required=True)
    small_summary.add_argument("--models", nargs="+", default=["all"])
    small_summary.add_argument("--paired-summary", default="")
    small_summary.add_argument("--output", required=True)

    validate_resources = commands.add_parser("validate-reference-resources")
    validate_resources.add_argument("--registry", required=True)
    validate_resources.add_argument("--resources", required=True)
    validate_resources.add_argument("--models", nargs="+", required=True)

    coverage = commands.add_parser("audit-reference-coverage")
    coverage.add_argument("--registry", required=True)
    coverage.add_argument("--reference-root", default="runs/reference")
    coverage.add_argument("--resources", action="append", required=True)
    coverage.add_argument("--output", required=True)

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
    temporal_select.add_argument("--method", choices=["clustered-mgtr"], default="clustered-mgtr")
    temporal_evaluate = common("evaluate-robotcar-temporal")
    temporal_evaluate.add_argument("--checkpoint", required=True)
    temporal_evaluate.add_argument("--selection", required=True)
    temporal_evaluate.add_argument("--condition", choices=["night", "day"], required=True)
    temporal_evaluate.add_argument("--cache", required=True)
    temporal_evaluate.add_argument("--output", required=True)
    temporal_evaluate.add_argument(
        "--method", choices=["clustered-mgtr"], default="clustered-mgtr"
    )
    temporal_summary = commands.add_parser("summarize-robotcar-temporal")
    for name in ("night", "day", "crica", "salad", "boq", "reference-resources", "output"):
        temporal_summary.add_argument(f"--{name}", required=True)
    temporal_summary.add_argument(
        "--method", choices=["clustered-mgtr"], default="clustered-mgtr"
    )
    qualitative = common("export-qualitative-top1")
    qualitative.add_argument("--selection", required=True)
    qualitative.add_argument("--search", action="store_true", help="Search all four night query sets and select examples automatically")
    qualitative.add_argument("--output-dir", required=True)
    qualitative.add_argument("--cache-dir", required=True)
    qualitative.add_argument("--batch-size", type=int, default=32)
    qualitative.add_argument("--workers", type=int, default=4)
    qualitative.add_argument(
        "--models", nargs="+",
        default=["cdnd", "cosplace", "eigenplaces", "official_mixvpr512"],
        choices=[
            "baseline", "cdnd", "cosplace", "eigenplaces", "official_mixvpr512",
            "nocplace", "anyloc_pca", "crica_pca", "salad", "boq_pca",
        ],
    )
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


def _keyed_paths(expressions: list[str], *, integer_keys: bool = False) -> dict:
    result = {}
    for expression in expressions:
        if "=" not in expression:
            raise ValueError(f"Result mapping must use KEY=JSON: {expression!r}")
        raw_key, path = expression.split("=", 1)
        key = int(raw_key) if integer_keys else raw_key
        if key in result:
            raise ValueError(f"Duplicate result key: {key}")
        result[key] = path
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
    if args.command == "summarize-cdnd-multiseed":
        print(summarize_cdnd_multiseed(
            _keyed_paths(args.baseline, integer_keys=True),
            _keyed_paths(args.candidate, integer_keys=True),
            _keyed_paths(args.ablation),
            args.output,
        ))
        return
    if args.command == "summarize-paired-vpr":
        print(summarize_paired_vpr(
            args.baseline,
            args.candidate,
            args.label,
            args.output,
        ))
        return
    if args.command == "plot-paper-efficiency":
        print(plot_accuracy_efficiency(
            args.student_summary,
            args.reference_root,
            args.resource_summary,
            args.output,
            args.compact_summary,
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
    if args.command == "audit-reference-coverage":
        print(audit_reference_coverage(
            args.registry,
            args.reference_root,
            args.resources,
            args.output,
        ))
        return
    if args.command == "train-boq-small":
        print(train_boq_small(args.config, smoke_steps=args.smoke_steps))
        return
    if args.command == "fit-anyloc-small-vocabulary":
        print(fit_anyloc_small_vocabulary(
            args.registry, args.model, args.output,
            samples=args.samples, patches_per_image=args.patches_per_image,
            random_seed=args.seed, batch_size=args.batch_size, workers=args.workers,
        ))
        return
    if args.command == "summarize-small-references":
        print(summarize_small_references(
            args.registry, args.reference_root, args.resources, args.output,
            model_names=tuple(args.models),
            paired_summary=args.paired_summary or None,
        ))
        return
    if args.command == "validate-reference-resources":
        if not validate_reference_resources(
            args.registry, args.resources, tuple(args.models)
        ):
            raise SystemExit(1)
        print(args.resources)
        return
    if args.command == "summarize-robotcar-temporal":
        print(summarize_robotcar_temporal(
                args.night, args.day, args.crica, args.salad, args.boq, args.output,
                args.reference_resources,
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
    elif args.command == "fit-reference-pca":
        print(fit_reference_pca(
            config,
            args.registry,
            args.model,
            args.output,
            args.cache,
            components=args.components,
            samples=args.samples,
            random_seed=args.seed,
            batch_size=args.batch_size,
            workers=args.workers,
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
        print(select_robotcar_temporal(config, args.checkpoint, args.output, args.cache, args.method))
    elif args.command == "evaluate-robotcar-temporal":
        print(evaluate_robotcar_temporal(
                config, args.checkpoint, args.selection, args.condition,
                args.output, args.cache, None, args.method,
            ))
    elif args.command == "export-qualitative-top1":
        if args.search:
            from .qualitative.search import search_qualitative_top1
            print(search_qualitative_top1(
                config, args.selection, args.output_dir, args.cache_dir,
                batch_size=args.batch_size, workers=args.workers,
            ))
        else:
            print(export_qualitative_top1(
                config, args.selection, args.output_dir, args.cache_dir,
                batch_size=args.batch_size, workers=args.workers,
                models=tuple(args.models),
            ))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
