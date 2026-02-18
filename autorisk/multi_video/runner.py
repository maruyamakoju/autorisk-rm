"""Orchestration helpers for unattended multi-source inference runs."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from autorisk.utils.video_sources import list_video_sources

CommandExecutor = Callable[[list[str], Path], int]
LoggerFn = Callable[[str], None]


@dataclass(frozen=True)
class RunAllOptions:
    only: str | None = None
    skip_supplement: bool = False
    skip_ttc: bool = False
    skip_grounding: bool = False
    skip_report: bool = False
    dry_run: bool = False
    resume: bool = True
    fail_fast: bool = False
    summary_path: str = "outputs/multi_video/run_summary.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_log(message: str) -> None:
    print(message, flush=True)


def _default_executor(cmd: list[str], cwd: Path) -> int:
    result = subprocess.run(cmd, cwd=str(cwd))
    return int(result.returncode)


def _count_done(output_dir: Path) -> tuple[int, int]:
    clips_dir = output_dir / "clips"
    results_path = output_dir / "cosmos_results.json"
    total = len(list(clips_dir.glob("candidate_*.mp4"))) if clips_dir.exists() else 0
    done = 0
    if results_path.exists():
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            done = len(payload) if isinstance(payload, list) else 0
        except Exception:
            done = 0
    return done, total


def _step_skip(label: str, reason: str) -> dict[str, Any]:
    return {
        "label": label,
        "ok": True,
        "skipped": True,
        "reason": reason,
        "returncode": 0,
        "elapsed_sec": 0.0,
    }


def _run_step(
    *,
    cmd: list[str],
    label: str,
    dry_run: bool,
    repo_root: Path,
    executor: CommandExecutor,
    log: LoggerFn,
) -> dict[str, Any]:
    log("")
    log("=" * 60)
    log(f"[{label}] {'DRY-RUN' if dry_run else 'Running'}: {' '.join(cmd)}")
    log("=" * 60)
    if dry_run:
        return {
            "label": label,
            "cmd": cmd,
            "ok": True,
            "dry_run": True,
            "returncode": 0,
            "elapsed_sec": 0.0,
        }

    t0 = time.time()
    rc = executor(cmd, repo_root)
    elapsed = time.time() - t0
    ok = rc == 0
    log(f"[{label}] {'OK' if ok else 'FAILED'} in {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    return {
        "label": label,
        "cmd": cmd,
        "ok": ok,
        "dry_run": False,
        "returncode": int(rc),
        "elapsed_sec": round(elapsed, 3),
    }


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_all_sources(
    *,
    repo_root: str | Path,
    options: RunAllOptions,
    sources: list[dict[str, Any]] | None = None,
    executor: CommandExecutor | None = None,
    log: LoggerFn | None = None,
) -> dict[str, Any]:
    """Run configured inference/analysis stages for all selected sources."""
    root = Path(repo_root).resolve()
    logger = log or _default_log
    execute = executor or _default_executor

    run_sources = sources if sources is not None else list_video_sources(repo_root=root, only=options.only)
    started_at = _utc_now_iso()
    t0 = time.time()

    summary: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": started_at,
        "finished_at_utc": "",
        "elapsed_sec": 0.0,
        "dry_run": bool(options.dry_run),
        "resume": bool(options.resume),
        "fail_fast": bool(options.fail_fast),
        "skip": {
            "supplement": bool(options.skip_supplement),
            "ttc": bool(options.skip_ttc),
            "grounding": bool(options.skip_grounding),
            "report": bool(options.skip_report),
        },
        "sources": [],
        "ok": True,
        "failed_sources": 0,
    }

    abort_all = False
    for source in run_sources:
        name = str(source["name"])
        config = str(source["config_path"])
        output = str(source["output_dir"])
        output_dir = (root / output).resolve()
        clips_dir = output_dir / "clips"
        results_path = output_dir / "cosmos_results.json"

        source_entry: dict[str, Any] = {
            "name": name,
            "config_path": config,
            "output_dir": str(output_dir),
            "default_video_path": str(source.get("default_video_path", "")),
            "started_at_utc": _utc_now_iso(),
            "finished_at_utc": "",
            "elapsed_sec": 0.0,
            "ok": True,
            "steps": [],
        }
        source_t0 = time.time()

        done_before, total_before = _count_done(output_dir)
        source_entry["clips_total"] = total_before
        source_entry["results_done_before"] = done_before

        infer_label = f"{name}/infer"
        if not clips_dir.exists():
            source_entry["steps"].append(_step_skip(infer_label, "missing clips directory"))
        elif options.resume and total_before > 0 and done_before >= total_before:
            source_entry["steps"].append(
                _step_skip(infer_label, f"resume: already complete ({done_before}/{total_before})")
            )
        else:
            infer_res = _run_step(
                cmd=[
                    sys.executable,
                    "-m",
                    "autorisk.cli",
                    "-c",
                    config,
                    "infer",
                    "--clips-dir",
                    str(clips_dir),
                    "-o",
                    str(output_dir),
                ],
                label=infer_label,
                dry_run=options.dry_run,
                repo_root=root,
                executor=execute,
                log=logger,
            )
            source_entry["steps"].append(infer_res)
            if not infer_res["ok"]:
                source_entry["ok"] = False
                if options.fail_fast:
                    abort_all = True

        done_after, total_after = _count_done(output_dir)
        source_entry["results_done_after_infer"] = done_after
        source_entry["clips_total_after_infer"] = total_after
        results_available = options.dry_run or results_path.exists()

        def run_or_skip(
            step: str,
            cmd: list[str],
            *,
            disabled: bool,
            reason_if_disabled: str,
        ) -> None:
            if disabled:
                source_entry["steps"].append(_step_skip(step, reason_if_disabled))
                return
            if not results_available:
                source_entry["steps"].append(_step_skip(step, "missing cosmos_results.json"))
                return
            res = _run_step(
                cmd=cmd,
                label=step,
                dry_run=options.dry_run,
                repo_root=root,
                executor=execute,
                log=logger,
            )
            source_entry["steps"].append(res)
            if not res["ok"]:
                source_entry["ok"] = False

        run_or_skip(
            f"{name}/supplement",
            [
                sys.executable,
                "-m",
                "autorisk.cli",
                "-c",
                config,
                "supplement",
                "-r",
                str(results_path),
                "-o",
                str(output_dir),
            ],
            disabled=options.skip_supplement,
            reason_if_disabled="--skip-supplement",
        )
        if options.fail_fast and source_entry["ok"] is False:
            abort_all = True

        run_or_skip(
            f"{name}/ttc",
            [
                sys.executable,
                "-m",
                "autorisk.cli",
                "-c",
                config,
                "ttc",
                "--clips-dir",
                str(clips_dir),
                "-o",
                str(output_dir),
            ],
            disabled=options.skip_ttc,
            reason_if_disabled="--skip-ttc",
        )
        if options.fail_fast and source_entry["ok"] is False:
            abort_all = True

        run_or_skip(
            f"{name}/grounding",
            [
                sys.executable,
                "-m",
                "autorisk.cli",
                "-c",
                config,
                "grounding",
                "-r",
                str(results_path),
                "-o",
                str(output_dir),
            ],
            disabled=options.skip_grounding,
            reason_if_disabled="--skip-grounding",
        )
        if options.fail_fast and source_entry["ok"] is False:
            abort_all = True

        run_or_skip(
            f"{name}/report",
            [
                sys.executable,
                "-m",
                "autorisk.cli",
                "-c",
                config,
                "report",
                "-r",
                str(results_path),
                "-o",
                str(output_dir),
                "-f",
                "html",
            ],
            disabled=options.skip_report,
            reason_if_disabled="--skip-report",
        )
        if options.fail_fast and source_entry["ok"] is False:
            abort_all = True

        source_entry["elapsed_sec"] = round(time.time() - source_t0, 3)
        source_entry["finished_at_utc"] = _utc_now_iso()
        summary["sources"].append(source_entry)

        if abort_all:
            logger(f"[{name}] fail-fast triggered; aborting remaining sources.")
            break

    failed_sources = sum(1 for source_entry in summary["sources"] if not bool(source_entry.get("ok", False)))
    summary["failed_sources"] = failed_sources
    summary["ok"] = failed_sources == 0
    summary["elapsed_sec"] = round(time.time() - t0, 3)
    summary["finished_at_utc"] = _utc_now_iso()

    summary_path = (root / options.summary_path).resolve()
    _write_summary(summary_path, summary)
    logger("")
    logger(f"Summary: {summary_path}")
    logger(f"Overall: {'OK' if summary['ok'] else 'FAILED'} (failed_sources={failed_sources})")

    return summary

