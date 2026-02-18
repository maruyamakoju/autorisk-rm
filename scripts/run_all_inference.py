"""Run inference + analysis post-processing across configured video sources."""

from __future__ import annotations

import argparse
from pathlib import Path

from autorisk.multi_video.runner import RunAllOptions, run_all_sources

ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-source inference workflow.")
    parser.add_argument(
        "--only",
        default=None,
        help="Run only one source name (public/japan/winter/us_highway)",
    )
    parser.add_argument("--skip-supplement", action="store_true", help="Skip supplement step")
    parser.add_argument("--skip-ttc", action="store_true", help="Skip TTC step")
    parser.add_argument("--skip-grounding", action="store_true", help="Skip grounding step")
    parser.add_argument("--skip-report", action="store_true", help="Skip report step")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands only")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume infer from existing results (default)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Always run infer regardless of existing progress",
    )
    parser.add_argument(
        "--fail-fast",
        dest="fail_fast",
        action="store_true",
        default=False,
        help="Stop immediately on first failed step",
    )
    parser.add_argument(
        "--keep-going",
        dest="fail_fast",
        action="store_false",
        help="Continue other sources even when a step fails (default)",
    )
    parser.add_argument(
        "--summary-path",
        default="outputs/multi_video/run_summary.json",
        help="Path to write machine-readable run summary JSON",
    )
    parser.add_argument(
        "--validate-summary",
        action="store_true",
        default=True,
        help="Validate summary JSON after run (default: on)",
    )
    parser.add_argument(
        "--no-validate-summary",
        dest="validate_summary",
        action="store_false",
        help="Skip summary validation",
    )
    parser.add_argument("--schema-dir", default=None, help="Optional schema directory override")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    options = RunAllOptions(
        only=args.only,
        skip_supplement=bool(args.skip_supplement),
        skip_ttc=bool(args.skip_ttc),
        skip_grounding=bool(args.skip_grounding),
        skip_report=bool(args.skip_report),
        dry_run=bool(args.dry_run),
        resume=bool(args.resume),
        fail_fast=bool(args.fail_fast),
        summary_path=str(args.summary_path),
    )
    summary = run_all_sources(repo_root=ROOT, options=options)
    if args.validate_summary:
        from autorisk.multi_video.validate import validate_multi_video_run_summary

        summary_path = (ROOT / options.summary_path).resolve()
        validate_res = validate_multi_video_run_summary(summary_path, schema_dir=args.schema_dir)
        print(
            f"Validated run summary: ok={validate_res.ok} issues={len(validate_res.issues)} source={validate_res.source}"
        )
        if not validate_res.ok:
            for issue in validate_res.issues[:20]:
                print(f"- {issue.kind}: {issue.path} {issue.detail}")
            raise SystemExit(2)
    if not bool(summary.get("ok", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
