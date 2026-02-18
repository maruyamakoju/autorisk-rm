from __future__ import annotations

import json
from pathlib import Path

from autorisk.multi_video.runner import RunAllOptions, run_all_sources


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_run_all_sources_resume_skips_completed_infer(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "public_run"
    clips_dir = run_dir / "clips"
    _touch(clips_dir / "candidate_001.mp4")
    _touch(clips_dir / "candidate_002.mp4")
    (run_dir / "cosmos_results.json").write_text(
        json.dumps([{"id": 1}, {"id": 2}], ensure_ascii=False),
        encoding="utf-8",
    )

    options = RunAllOptions(
        only="public",
        resume=True,
        skip_supplement=True,
        skip_ttc=True,
        skip_grounding=True,
        skip_report=True,
        summary_path="outputs/summary.json",
    )

    calls: list[list[str]] = []

    def fake_exec(cmd: list[str], cwd: Path) -> int:  # noqa: ARG001
        calls.append(cmd)
        return 0

    summary = run_all_sources(
        repo_root=tmp_path,
        options=options,
        sources=[
            {
                "name": "public",
                "config_path": str(tmp_path / "configs/public.yaml"),
                "output_dir": "outputs/public_run",
                "default_video_path": "data/sample.mp4",
            }
        ],
        executor=fake_exec,
        log=lambda msg: None,
    )
    assert summary["ok"] is True
    assert summary["failed_sources"] == 0
    assert calls == []  # infer skipped by resume, others disabled
    infer_step = summary["sources"][0]["steps"][0]
    assert infer_step["skipped"] is True
    assert "resume: already complete" in infer_step["reason"]


def test_run_all_sources_fail_fast_stops_remaining_sources(tmp_path: Path) -> None:
    first_dir = tmp_path / "outputs" / "first_run" / "clips"
    second_dir = tmp_path / "outputs" / "second_run" / "clips"
    _touch(first_dir / "candidate_001.mp4")
    _touch(second_dir / "candidate_001.mp4")

    options = RunAllOptions(
        resume=False,
        fail_fast=True,
        skip_supplement=True,
        skip_ttc=True,
        skip_grounding=True,
        skip_report=True,
        summary_path="outputs/summary_fail_fast.json",
    )

    call_count = 0

    def fake_exec(cmd: list[str], cwd: Path) -> int:  # noqa: ARG001
        nonlocal call_count
        call_count += 1
        return 1

    summary = run_all_sources(
        repo_root=tmp_path,
        options=options,
        sources=[
            {
                "name": "first",
                "config_path": str(tmp_path / "configs/first.yaml"),
                "output_dir": "outputs/first_run",
                "default_video_path": "",
            },
            {
                "name": "second",
                "config_path": str(tmp_path / "configs/second.yaml"),
                "output_dir": "outputs/second_run",
                "default_video_path": "",
            },
        ],
        executor=fake_exec,
        log=lambda msg: None,
    )
    assert summary["ok"] is False
    assert summary["failed_sources"] == 1
    assert len(summary["sources"]) == 1
    assert call_count == 1


def test_run_all_sources_dry_run_executes_all_steps_without_results(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "public_run" / "clips"
    _touch(run_dir / "candidate_001.mp4")

    options = RunAllOptions(
        dry_run=True,
        resume=False,
        skip_supplement=False,
        skip_ttc=False,
        skip_grounding=False,
        skip_report=False,
        summary_path="outputs/summary_dry_run.json",
    )

    summary = run_all_sources(
        repo_root=tmp_path,
        options=options,
        sources=[
            {
                "name": "public",
                "config_path": str(tmp_path / "configs/public.yaml"),
                "output_dir": "outputs/public_run",
                "default_video_path": "",
            }
        ],
        log=lambda msg: None,
    )
    assert summary["ok"] is True
    assert summary["failed_sources"] == 0
    steps = summary["sources"][0]["steps"]
    labels = [s["label"] for s in steps]
    assert labels == [
        "public/infer",
        "public/supplement",
        "public/ttc",
        "public/grounding",
        "public/report",
    ]
    assert all(step.get("dry_run") is True for step in steps)

