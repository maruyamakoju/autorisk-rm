"""Registry and validation for multi-video source configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autorisk.utils.config import load_config


@dataclass(frozen=True)
class VideoSource:
    name: str
    config_path: str
    output_dir: str
    default_video_path: str | None = None


VIDEO_SOURCES: tuple[VideoSource, ...] = (
    VideoSource(
        name="public",
        config_path="configs/public.yaml",
        output_dir="outputs/public_run",
        default_video_path="data/public_samples/uk_dashcam_compilation.mp4",
    ),
    VideoSource(
        name="japan",
        config_path="configs/japan.yaml",
        output_dir="outputs/japan_run",
        default_video_path="data/multi_video/japan_5min.mp4",
    ),
    VideoSource(
        name="winter",
        config_path="configs/winter.yaml",
        output_dir="outputs/winter_run",
        default_video_path="data/multi_video/winter_5min.mp4",
    ),
    VideoSource(
        name="us_highway",
        config_path="configs/us_highway.yaml",
        output_dir="outputs/us_highway_run",
        default_video_path="data/multi_video/us_highway_5min.mp4",
    ),
)


def _as_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{field} must be numeric, got: {value!r}") from exc


def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{field} must be int, got: {value!r}") from exc


def validate_video_source(source: VideoSource, *, repo_root: str | Path) -> dict[str, Any]:
    """Validate one source config and return normalized metadata."""
    root = Path(repo_root).resolve()
    cfg_path = (root / source.config_path).resolve()
    if not cfg_path.exists() or not cfg_path.is_file():
        raise FileNotFoundError(f"[{source.name}] missing config: {cfg_path}")

    cfg = load_config(config_path=cfg_path)
    output_dir = str(getattr(getattr(cfg, "general", {}), "output_dir", "")).strip()
    if output_dir == "":
        raise ValueError(f"[{source.name}] general.output_dir is required in {cfg_path}")
    if output_dir != source.output_dir:
        raise ValueError(
            f"[{source.name}] output_dir mismatch: config={output_dir!r} registry={source.output_dir!r}"
        )

    top_n = _as_int(getattr(getattr(cfg, "mining", {}), "top_n", 0), "mining.top_n")
    if top_n <= 0:
        raise ValueError(f"[{source.name}] mining.top_n must be > 0")

    audio_w = _as_float(getattr(getattr(getattr(cfg, "mining", {}), "audio", {}), "weight", 0.0), "mining.audio.weight")
    motion_w = _as_float(getattr(getattr(getattr(cfg, "mining", {}), "motion", {}), "weight", 0.0), "mining.motion.weight")
    proximity_w = _as_float(getattr(getattr(getattr(cfg, "mining", {}), "proximity", {}), "weight", 0.0), "mining.proximity.weight")
    if min(audio_w, motion_w, proximity_w) < 0:
        raise ValueError(f"[{source.name}] mining signal weights must be >= 0")
    total_w = audio_w + motion_w + proximity_w
    if not (0.99 <= total_w <= 1.01):
        raise ValueError(
            f"[{source.name}] mining signal weights must sum to ~1.0, got {total_w:.3f}"
        )

    backend = str(getattr(getattr(cfg, "cosmos", {}), "backend", "")).strip().lower()
    if backend not in {"local", "api"}:
        raise ValueError(f"[{source.name}] cosmos.backend must be local/api, got {backend!r}")

    local_fps = _as_int(getattr(getattr(cfg, "cosmos", {}), "local_fps", 0), "cosmos.local_fps")
    if local_fps <= 0:
        raise ValueError(f"[{source.name}] cosmos.local_fps must be > 0")

    report_fmt = str(getattr(getattr(cfg, "report", {}), "format", "")).strip().lower()
    if report_fmt not in {"html", "markdown"}:
        raise ValueError(f"[{source.name}] report.format must be html/markdown, got {report_fmt!r}")

    return {
        "name": source.name,
        "config_path": str(cfg_path),
        "output_dir": output_dir,
        "default_video_path": source.default_video_path or "",
    }


def list_video_sources(
    *,
    repo_root: str | Path,
    only: str | None = None,
) -> list[dict[str, Any]]:
    """Return validated source metadata in deterministic order."""
    selected: list[VideoSource]
    if only is None or str(only).strip() == "":
        selected = list(VIDEO_SOURCES)
    else:
        target = str(only).strip().lower()
        selected = [s for s in VIDEO_SOURCES if s.name == target]
        if not selected:
            known = ", ".join(s.name for s in VIDEO_SOURCES)
            raise ValueError(f"unknown source {only!r}; expected one of: {known}")

    return [validate_video_source(source, repo_root=repo_root) for source in selected]
