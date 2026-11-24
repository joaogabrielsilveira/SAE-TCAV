"""Run cheap cached choices now, or queue additional comparisons after the original job."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


ORIGINAL = Path("stats/context_intervention")
OUTPUT = Path("stats/context_comparisons")


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def source_snapshot():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}


def original_running(pid):
    try:
        os.kill(pid, 0)
        command = open(f"/proc/{pid}/cmdline", "rb").read()
        return b"context_intervention.study" in command
    except (ProcessLookupError, FileNotFoundError):
        return False


def wait_for_original():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pid = int((ORIGINAL / "study.pid").read_text())
    atomic_json(OUTPUT / "status.json", {"phase": "waiting for original study", "original_pid": pid})
    while original_running(pid):
        time.sleep(60)
    # Load experiment modules only after waiting. A long-lived waiter must never
    # execute old imported modules while fingerprinting newer files on disk.
    from context_comparisons.provenance import validate_completed_study
    validate_completed_study(ORIGINAL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--wait-for-original", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    snapshot = source_snapshot()
    if args.metrics_only:
        from context_comparisons.study import run_metrics_only
        run_metrics_only()
        from context_comparisons.reporting import summarize_choices
        summarize_choices(OUTPUT)
        return
    if args.wait_for_original:
        wait_for_original()
    if source_snapshot() != snapshot:
        raise RuntimeError("Comparison source changed while waiting; restart only the comparison worker")
    from context_comparisons.study import run_all
    atomic_json(OUTPUT / "status.json", {"phase": "running latest-year and native-context comparisons"})
    run_all()
    from context_comparisons.reporting import summarize
    result = summarize(OUTPUT)
    atomic_json(OUTPUT / "status.json", {"phase": "complete", **result,
                                         "metric_comparison": "cached validation decisions; test results only where cached"})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUTPUT / "status.json", {"phase": "stopped with an explicit limitation", "error": str(error)})
        raise
