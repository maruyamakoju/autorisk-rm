"""Backward-compatible wrapper for multi-source inference runs.

Deprecated:
    Use `python scripts/run_all_inference.py ...` directly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper. Prefer scripts/run_all_inference.py.",
    )
    parser.add_argument("--only", default=None, help="Run only one source")
    parser.add_argument("--skip-mining", action="store_true", help="No-op (kept for compatibility)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip report generation")
    parser.add_argument("--dry-run", action="store_true", help="Preview commands only")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cmd = [sys.executable, "scripts/run_all_inference.py"]
    if args.only:
        cmd.extend(["--only", args.only])
    if args.skip_eval:
        cmd.append("--skip-report")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.fail_fast:
        cmd.append("--fail-fast")
    if args.skip_mining:
        print("[run_multi_video] --skip-mining is ignored in this wrapper.")

    completed = subprocess.run(cmd, cwd=str(ROOT))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

