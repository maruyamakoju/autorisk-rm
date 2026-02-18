"""Generate submission-ready metrics summary across configured video sources."""

from __future__ import annotations

import argparse
from pathlib import Path

from autorisk.multi_video.submission_metrics import (
    build_submission_metrics,
    write_submission_metrics,
)

ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compact submission metrics.")
    parser.add_argument("--only", default=None, help="Only include one source key")
    parser.add_argument(
        "--out",
        default="outputs/multi_video/submission_metrics.json",
        help="Output path for JSON summary",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate output JSON after write (default: on)",
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip output validation",
    )
    parser.add_argument("--schema-dir", default=None, help="Optional schema directory override")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_submission_metrics(repo_root=ROOT, only=args.only)
    out_path = write_submission_metrics(payload, output_path=ROOT / str(args.out))
    print(f"Wrote: {out_path}")
    if args.validate:
        from autorisk.multi_video.validate import validate_submission_metrics

        validate_res = validate_submission_metrics(out_path, schema_dir=args.schema_dir)
        print(
            f"Validated submission metrics: ok={validate_res.ok} issues={len(validate_res.issues)} source={validate_res.source}"
        )
        if not validate_res.ok:
            for issue in validate_res.issues[:20]:
                print(f"- {issue.kind}: {issue.path} {issue.detail}")
            raise SystemExit(2)


if __name__ == "__main__":
    main()
