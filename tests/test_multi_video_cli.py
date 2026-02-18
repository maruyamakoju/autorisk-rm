from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from autorisk.cli import cli


def test_multi_run_cli_passes_options(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_all_sources(*, repo_root, options):  # type: ignore[no-untyped-def]
        captured["repo_root"] = repo_root
        captured["options"] = options
        return {"ok": True}

    monkeypatch.setattr("autorisk.multi_video.runner.run_all_sources", fake_run_all_sources)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "multi-run",
            "--repo-root",
            str(tmp_path),
            "--only",
            "japan",
            "--skip-ttc",
            "--dry-run",
            "--fail-fast",
            "--no-validate-summary",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["repo_root"] == tmp_path.resolve()
    options = captured["options"]
    assert options.only == "japan"
    assert options.skip_ttc is True
    assert options.dry_run is True
    assert options.fail_fast is True


def test_multi_run_cli_returns_nonzero_on_failure(monkeypatch) -> None:
    def fake_run_all_sources(*, repo_root, options):  # type: ignore[no-untyped-def,unused-argument]
        return {"ok": False}

    monkeypatch.setattr("autorisk.multi_video.runner.run_all_sources", fake_run_all_sources)
    runner = CliRunner()
    result = runner.invoke(cli, ["multi-run", "--no-validate-summary"])
    assert result.exit_code == 1


def test_submission_metrics_cli_writes_output(monkeypatch, tmp_path: Path) -> None:
    def fake_build(*, repo_root, only):  # type: ignore[no-untyped-def]
        assert repo_root == tmp_path.resolve()
        assert only == "public"
        return {"schema_version": 1}

    def fake_write(payload, *, output_path):  # type: ignore[no-untyped-def]
        assert payload == {"schema_version": 1}
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")
        return out

    monkeypatch.setattr("autorisk.multi_video.submission_metrics.build_submission_metrics", fake_build)
    monkeypatch.setattr("autorisk.multi_video.submission_metrics.write_submission_metrics", fake_write)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "submission-metrics",
            "--repo-root",
            str(tmp_path),
            "--only",
            "public",
            "--out",
            "outputs/metrics.json",
            "--no-validate",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Wrote:" in result.output


def test_multi_validate_cli_reports_success(tmp_path: Path) -> None:
    run_summary = tmp_path / "run_summary.json"
    run_summary.write_text(
        """
{
  "schema_version": 1,
  "started_at_utc": "2026-02-18T00:00:00+00:00",
  "finished_at_utc": "2026-02-18T00:00:01+00:00",
  "elapsed_sec": 1.0,
  "dry_run": true,
  "resume": true,
  "fail_fast": false,
  "skip": {"supplement": true, "ttc": true, "grounding": true, "report": true},
  "sources": [],
  "ok": true,
  "failed_sources": 0
}
        """.strip(),
        encoding="utf-8",
    )
    submission_metrics = tmp_path / "submission_metrics.json"
    submission_metrics.write_text(
        """
{
  "schema_version": 1,
  "generated_at_utc": "2026-02-18T00:00:01+00:00",
  "sources_total": 0,
  "sources_available": 0,
  "clips_total": 0,
  "sources": []
}
        """.strip(),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "multi-validate",
            "--run-summary",
            str(run_summary),
            "--submission-metrics",
            str(submission_metrics),
            "--enforce",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "run_summary: ok=True" in result.output
    assert "submission_metrics: ok=True" in result.output
