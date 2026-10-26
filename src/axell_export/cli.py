"""Inspect configurations and export a caller-owned checkpoint to a new directory."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

from .config import get_preset, list_presets
from .registry import list_models


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="axell-export")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list-models", help="Show the eight registered architectures")
    commands.add_parser(
        "list-presets", help="Show export configurations without importing GPU libraries"
    )
    describe = commands.add_parser("describe-preset")
    describe.add_argument("--preset", required=True)
    for command in ("plan", "export"):
        child = commands.add_parser(command)
        child.add_argument(
            "--weights", required=True, type=Path, help="An existing trusted .pt file"
        )
        child.add_argument("--preset", required=True)
        child.add_argument(
            "--output-dir", required=True, type=Path, help="A new private run directory"
        )
        child.add_argument(
            "--calibration-data", type=Path, help="INT8 only: local training-data YAML"
        )
        child.add_argument("--device", type=int, choices=(0,), default=0, help="CUDA device 0")
        child.add_argument("--fraction", type=float, help="INT8 calibration fraction, default 0.30")
        child.add_argument(
            "--calibration-selection",
            choices=("seeded-random", "sorted-first"),
            help="INT8 sampling: seeded-random (default), or historical sorted-first",
        )
        child.add_argument("--calibration-seed", type=int, help="INT8 sampling seed, default 42")
        child.add_argument("--conf", type=float, help="NMS export confidence, default 0.25")
        child.add_argument("--iou", type=float, help="NMS export IoU threshold, default 0.7")
        child.add_argument("--max-det", type=int, help="NMS export detection limit, default 300")
    return parser


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.command == "list-models":
            result = [asdict(model) for model in list_models()]
        elif args.command == "list-presets":
            result = [config.to_dict() for config in list_presets()]
        elif args.command == "describe-preset":
            result = get_preset(args.preset).to_dict()
        else:
            from .exporting import export_engine, plan_export

            config = get_preset(args.preset)
            overrides = {}
            for argument, field in (
                ("fraction", "calibration_fraction"),
                ("calibration_selection", "calibration_selection"),
                ("calibration_seed", "calibration_seed"),
            ):
                if getattr(args, argument) is None:
                    continue
                if config.precision != "int8":
                    raise ValueError(
                        f"--{argument.replace('_', '-')} applies only to INT8 calibration"
                    )
                overrides[field] = getattr(args, argument)
            for argument in ("conf", "iou", "max_det"):
                if getattr(args, argument) is None:
                    continue
                if not config.nms:
                    raise ValueError(f"--{argument.replace('_', '-')} requires an NMS preset")
                overrides[argument] = getattr(args, argument)
            config = replace(config, **overrides).validate()
            operation = plan_export if args.command == "plan" else export_engine
            result = operation(
                args.weights,
                args.output_dir,
                config,
                calibration_data=args.calibration_data,
                device=args.device,
            )
            if isinstance(result, Path):
                result = {"manifest": str(result)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
