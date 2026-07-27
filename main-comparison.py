"""Run the complete renal cross-run SAE semantic comparison."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from comparison_runner import ComparisonRunnerConfig, run_comparison


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="comparison_runner.example.json",
        help="Complete comparison JSON configuration",
    )
    parser.add_argument("--data", help="Override renal Feather path")
    parser.add_argument("--artifact-dir", help="Override artifact root")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        help="Override accelerator device; 'cuda' fails early if unavailable",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Disable geometry filtering; can increase semantic runtime greatly",
    )
    parser.add_argument(
        "--skip-functional",
        action="store_true",
        help="Skip high-precision rule, CAV, and TCAV stages",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore complete and stage caches",
    )
    args = parser.parse_args(argv)

    config = ComparisonRunnerConfig.from_json(args.config)
    if args.data:
        config = replace(config, dataset_path=str(Path(args.data).resolve()))
    if args.artifact_dir:
        config = replace(
            config, artifact_dir=str(Path(args.artifact_dir).resolve())
        )
    if args.device:
        config = replace(
            config,
            accelerator=replace(config.accelerator, device=args.device),
        )
    if args.all_pairs:
        config = replace(
            config,
            matching=replace(config.matching, minimum_score=None),
        )
    if args.skip_functional:
        config = replace(
            config,
            functional=replace(config.functional, enabled=False),
        )

    summary = run_comparison(config, force=args.force)
    print(summary["artifact_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
