"""Build MAUT-inspired temporal Conceptual Robustness Index artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from temporal_cri import CRIAnalysisConfig, build_cri_analysis
from temporal_unified_analysis import UnifiedAnalysisConfig
from temporal_unified_enrichment import build_unified_enrichment


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir", default="stats/temporal_robustness",
        help="Temporal robustness artifact root",
    )
    parser.add_argument(
        "--parent-hash", default="5fd57eb7b61700cda81e",
        help="Completed temporal parent identity",
    )
    parser.add_argument(
        "--enrichment-manifest", type=Path,
        help="Reuse this completed enrichment manifest instead of locating/building it",
    )
    args = parser.parse_args(argv)

    unified = UnifiedAnalysisConfig(parent_hash=args.parent_hash)
    enrichment = args.enrichment_manifest
    if enrichment is None:
        enrichment = build_unified_enrichment(args.artifact_dir, unified)
    print(build_cri_analysis(enrichment, unified, CRIAnalysisConfig()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
