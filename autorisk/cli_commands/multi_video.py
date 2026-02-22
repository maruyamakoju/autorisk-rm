"""Multi-video CLI command registrations."""

from __future__ import annotations

from pathlib import Path

import click


@click.command("multi-run")
@click.option(
    "--repo-root",
    default=".",
    show_default=True,
    help="Repository root containing configs/ and outputs/",
)
@click.option(
    "--only",
    default=None,
    help="Run only one source name (public/japan/winter/us_highway)",
)
@click.option("--skip-supplement", is_flag=True, help="Skip supplement step")
@click.option("--skip-ttc", is_flag=True, help="Skip TTC step")
@click.option("--skip-grounding", is_flag=True, help="Skip grounding step")
@click.option("--skip-report", is_flag=True, help="Skip report step")
@click.option("--dry-run", is_flag=True, help="Print planned commands only")
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Resume inference from existing results",
)
@click.option(
    "--fail-fast/--keep-going",
    default=False,
    show_default=True,
    help="Stop after first failed source",
)
@click.option(
    "--summary-path",
    default="outputs/multi_video/run_summary.json",
    show_default=True,
    help="Relative path for run summary JSON",
)
@click.option(
    "--validate-summary/--no-validate-summary",
    default=True,
    show_default=True,
    help="Validate run summary against schema + semantic checks",
)
@click.option("--schema-dir", default=None, help="Optional schema directory override")
def multi_run(
    repo_root: str,
    only: str | None,
    skip_supplement: bool,
    skip_ttc: bool,
    skip_grounding: bool,
    skip_report: bool,
    dry_run: bool,
    resume: bool,
    fail_fast: bool,
    summary_path: str,
    validate_summary: bool,
    schema_dir: str | None,
) -> None:
    """Run multi-source inference + analysis pipeline."""
    from autorisk.multi_video.runner import RunAllOptions, run_all_sources
    from autorisk.multi_video.validate import validate_multi_video_run_summary

    root = Path(repo_root).expanduser().resolve()
    options = RunAllOptions(
        only=only,
        skip_supplement=bool(skip_supplement),
        skip_ttc=bool(skip_ttc),
        skip_grounding=bool(skip_grounding),
        skip_report=bool(skip_report),
        dry_run=bool(dry_run),
        resume=bool(resume),
        fail_fast=bool(fail_fast),
        summary_path=str(summary_path),
    )
    summary = run_all_sources(repo_root=root, options=options)
    if validate_summary:
        summary_abs = (root / options.summary_path).resolve()
        validate_res = validate_multi_video_run_summary(
            summary_abs, schema_dir=schema_dir
        )
        click.echo(
            f"Validated run summary: ok={validate_res.ok} issues={len(validate_res.issues)} source={validate_res.source}"
        )
        if not validate_res.ok:
            for issue in validate_res.issues[:20]:
                click.echo(
                    f"- {issue.kind}: {issue.path} {issue.detail}".rstrip(), err=True
                )
            raise SystemExit(2)
    if not bool(summary.get("ok", False)):
        raise SystemExit(1)


@click.command("submission-metrics")
@click.option(
    "--repo-root",
    default=".",
    show_default=True,
    help="Repository root containing outputs/",
)
@click.option("--only", default=None, help="Only include one source name")
@click.option(
    "--out",
    "output_path",
    default="outputs/multi_video/submission_metrics.json",
    show_default=True,
    help="Output JSON path",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    show_default=True,
    help="Validate metrics JSON against schema + semantic checks",
)
@click.option("--schema-dir", default=None, help="Optional schema directory override")
def submission_metrics(
    repo_root: str,
    only: str | None,
    output_path: str,
    validate: bool,
    schema_dir: str | None,
) -> None:
    """Generate compact submission metrics across configured sources."""
    from autorisk.multi_video.submission_metrics import (
        build_submission_metrics,
        write_submission_metrics,
    )
    from autorisk.multi_video.validate import validate_submission_metrics

    root = Path(repo_root).expanduser().resolve()
    payload = build_submission_metrics(repo_root=root, only=only)
    out_path = write_submission_metrics(payload, output_path=root / output_path)
    click.echo(f"Wrote: {out_path}")
    if validate:
        validate_res = validate_submission_metrics(out_path, schema_dir=schema_dir)
        click.echo(
            f"Validated submission metrics: ok={validate_res.ok} issues={len(validate_res.issues)} source={validate_res.source}"
        )
        if not validate_res.ok:
            for issue in validate_res.issues[:20]:
                click.echo(
                    f"- {issue.kind}: {issue.path} {issue.detail}".rstrip(), err=True
                )
            raise SystemExit(2)


@click.command("multi-validate")
@click.option("--run-summary", required=True, help="Path to multi-run summary JSON")
@click.option(
    "--submission-metrics",
    "submission_metrics_path",
    default=None,
    help="Optional path to submission metrics JSON",
)
@click.option("--schema-dir", default=None, help="Optional schema directory override")
@click.option(
    "--enforce/--no-enforce",
    default=True,
    show_default=True,
    help="Exit non-zero on validation issues",
)
def multi_validate(
    run_summary: str,
    submission_metrics_path: str | None,
    schema_dir: str | None,
    enforce: bool,
) -> None:
    """Validate multi-video summary/metrics artifacts."""
    from autorisk.multi_video.validate import (
        validate_multi_video_run_summary,
        validate_submission_metrics,
    )

    summary_res = validate_multi_video_run_summary(run_summary, schema_dir=schema_dir)
    click.echo(
        f"run_summary: ok={summary_res.ok} issues={len(summary_res.issues)} source={summary_res.source}"
    )
    issue_count = len(summary_res.issues)
    for issue in summary_res.issues[:20]:
        click.echo(f"- {issue.kind}: {issue.path} {issue.detail}".rstrip(), err=True)

    metrics_res = None
    if (
        submission_metrics_path is not None
        and str(submission_metrics_path).strip() != ""
    ):
        metrics_res = validate_submission_metrics(
            submission_metrics_path, schema_dir=schema_dir
        )
        click.echo(
            f"submission_metrics: ok={metrics_res.ok} issues={len(metrics_res.issues)} source={metrics_res.source}"
        )
        issue_count += len(metrics_res.issues)
        for issue in metrics_res.issues[:20]:
            click.echo(
                f"- {issue.kind}: {issue.path} {issue.detail}".rstrip(), err=True
            )

    if enforce and issue_count > 0:
        raise SystemExit(2)


def register_multi_video_commands(cli: click.Group) -> None:
    """Register multi-run/submission validation commands."""
    cli.add_command(multi_run)
    cli.add_command(submission_metrics)
    cli.add_command(multi_validate)
