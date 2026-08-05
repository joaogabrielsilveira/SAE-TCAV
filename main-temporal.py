"""Run leakage-safe reference-year temporal robustness experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from temporal_config import TemporalRobustnessConfig
from temporal_robustness import run_temporal_robustness


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="temporal_robustness.example.json",
        help="Temporal robustness JSON configuration",
    )
    parser.add_argument("--data", help="Override input Feather path")
    parser.add_argument("--artifact-dir", help="Override artifact root")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--development",
        action="store_true",
        help="Use first two patient-split and SAE seeds",
    )
    parser.add_argument(
        "--reference-year",
        action="append",
        type=int,
        help="Run only this reference year; repeat for multiple years",
    )
    parser.add_argument("--force", action="store_true", help="Ignore stage caches")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)

    config = TemporalRobustnessConfig.from_json(args.config)
    if args.development:
        config = config.development_profile()
    if args.reference_year:
        config = replace(config, reference_years=tuple(args.reference_year))
    if args.data:
        config = replace(config, dataset_path=str(Path(args.data).resolve()))
    if args.artifact_dir:
        config = replace(config, artifact_dir=str(Path(args.artifact_dir).resolve()))
    if args.device:
        config = replace(config, device=args.device)
    if args.force:
        config = replace(config, force=True)
    if args.no_cache:
        config = replace(config, use_cache=False)
    if args.no_progress:
        config = replace(config, show_progress=False)

    result = run_temporal_robustness(config, fail_fast=args.fail_fast)
    print(result["artifact_dir"])
    if result["failed_experiments"]:
        print(f"Failed experiments: {len(result['failed_experiments'])}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
