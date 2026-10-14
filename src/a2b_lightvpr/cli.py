from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from .config import apply_overrides, load_config
from .data import audit_data
from .resource_audit import audit_resources
from .robotcar_seasons import (
    CAMERAS, DAY_CONDITIONS, NIGHT_CONDITIONS, audit_robotcar_seasons,
    default_robotcar_seasons_root,
)
from .references.models import (
    audit_reference_models,
    benchmark_reference_models,
    fetch_reference_model,
    validate_reference_resources,
)
from .references.coverage import audit_reference_coverage
from .references.suite import evaluate_reference_suite
from .runtime import benchmark_model
from .trained_suite import SUITE_PROTOCOLS, evaluate_trained_suite
from .vpr.trainer import export_vpr_initial_checkpoint, export_vpr_inference_checkpoint, train_vpr


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="a2b-lightvpr")
    commands = parser.add_subparsers(dest="command", required=True)

    def common(name: str):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE")
        return command

    audit = common("audit-data")
    audit.add_argument("--decode", action="store_true")
    resources = common("audit-resources")
    resources.add_argument("--output", default="")
    resources.add_argument("--skip-images", action="store_true")

    seasons_audit = commands.add_parser("audit-robotcar-seasons")
    seasons_audit.add_argument("--data-root", default=str(default_robotcar_seasons_root()))
    seasons_audit.add_argument("--conditions", nargs="+", choices=NIGHT_CONDITIONS + DAY_CONDITIONS, default=list(NIGHT_CONDITIONS))
    seasons_audit.add_argument("--protocol", choices=["original", "v2"], default="v2")
    seasons_audit.add_argument("--cameras", nargs="+", choices=CAMERAS, default=None)
    seasons_audit.add_argument("--output", default="")

    gsv = commands.add_parser("prepare-gsv-metric")
    gsv.add_argument("--data-root", required=True)
    gsv.add_argument("--output-dir", default="data/manifests_gsv_metric")
    gsv.add_argument("--min-class-images", type=int, default=4)

    vpr = common("train-vpr")
    vpr.add_argument("--smoke-epochs", type=int, default=0)

    export = common("export-vpr-inference")
    export.add_argument("--training-checkpoint", required=True)
    export.add_argument("--output", required=True)
    initial_export = common("export-vpr-initial")
    initial_export.add_argument("--output", required=True)

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
    reference_benchmark.add_argument("--output", default="runs/reference/resource_benchmark.json")

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
    reference.add_argument("--database-encoder", choices=["original", "adapted"], default="original")
    reference.add_argument("--transform", choices=["native", "unified"], default="unified")
    reference.add_argument(
        "--datasets", nargs="+", default=["all"],
        choices=["all", "all-day", "all-conditions", *SUITE_PROTOCOLS],
    )
    reference.add_argument("--output-dir", default="")
    reference.add_argument("--batch-size", type=int, default=0)
    reference.add_argument("--allow-batch-dependent", action="store_true")
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    # With workers=0, small per-image transforms run in this process. Using
    # every CPU core for each tiny tensor operation can severely slow them.
    # Keep explicit user thread settings; otherwise use a modest default.
    cpu_threads = int(os.environ.get("OMP_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "2")))
    if cpu_threads < 1:
        raise ValueError("OMP_NUM_THREADS/MKL_NUM_THREADS must be positive")
    torch.set_num_threads(cpu_threads)
    if args.command == "train-vpr":
        print(f"CPU intra-op threads: {torch.get_num_threads()}", file=sys.stderr, flush=True)
    if args.command == "audit-robotcar-seasons":
        print(json.dumps(audit_robotcar_seasons(
            args.data_root, output=args.output or None,
            conditions=tuple(args.conditions), cameras=args.cameras, protocol=args.protocol,
        ), indent=2, sort_keys=True))
        return
    if args.command == "prepare-gsv-metric":
        from .gsv_metric import prepare_gsv_metric
        print(json.dumps(prepare_gsv_metric(
            args.data_root, args.output_dir, min_class_images=args.min_class_images,
        ), indent=2, sort_keys=True))
        return
    if args.command == "audit-reference-models":
        print(json.dumps(audit_reference_models(args.config, args.output), indent=2, sort_keys=True))
        return
    if args.command == "fetch-reference-model":
        print(json.dumps(fetch_reference_model(args.config, args.model), indent=2, sort_keys=True))
        return
    if args.command == "audit-reference-coverage":
        print(audit_reference_coverage(args.registry, args.reference_root, args.resources, args.output))
        return
    if args.command == "validate-reference-resources":
        if not validate_reference_resources(args.registry, args.resources, tuple(args.models)):
            raise SystemExit(1)
        print(args.resources)
        return

    config = apply_overrides(load_config(args.config), args.set)
    if args.command == "audit-data":
        print(json.dumps(audit_data(config.data, decode=args.decode), indent=2, sort_keys=True))
    elif args.command == "audit-resources":
        print(json.dumps(audit_resources(
            config, output_path=args.output or None, scan_images=not args.skip_images,
        ), indent=2, sort_keys=True))
    elif args.command == "train-vpr":
        print(train_vpr(config, smoke_epochs=args.smoke_epochs or None))
    elif args.command == "export-vpr-inference":
        print(export_vpr_inference_checkpoint(config, args.training_checkpoint, args.output))
    elif args.command == "export-vpr-initial":
        print(export_vpr_initial_checkpoint(config, args.output))
    elif args.command == "evaluate-trained-suite":
        print(evaluate_trained_suite(config, args.checkpoint, tuple(args.datasets)))
    elif args.command == "benchmark":
        print(json.dumps(benchmark_model(
            config, args.checkpoint or None, warmup=args.warmup, iterations=args.iterations,
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
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
