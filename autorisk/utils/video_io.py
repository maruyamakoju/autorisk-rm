"""Video I/O utilities: reading, segment extraction, frame sampling."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)


def get_video_info(path: str | Path) -> dict:
    """Return basic video metadata via OpenCV.

    Returns:
        Dict with keys: width, height, fps, frame_count, duration_sec.
    """
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_sec"] = info["frame_count"] / max(info["fps"], 1e-6)
    cap.release()
    return info


def extract_clip_ffmpeg(
    src: str | Path,
    dst: str | Path,
    start_sec: float,
    duration_sec: float,
    fps: int | None = None,
    short_edge: int | None = None,
) -> Path:
    """Extract a video clip using FFmpeg subprocess.

    Args:
        src: Source video path.
        dst: Destination clip path.
        start_sec: Start time in seconds.
        duration_sec: Duration in seconds.
        fps: Optional output FPS.
        short_edge: Optional resize to this short-edge dimension.

    Returns:
        Path to the created clip.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", str(src),
        "-t", f"{duration_sec:.3f}",
    ]

    vf_filters = []
    if short_edge is not None:
        vf_filters.append(
            f"scale='if(gt(iw,ih),-2,{short_edge})':'if(gt(iw,ih),{short_edge},-2)'"
        )
    if fps is not None:
        vf_filters.append(f"fps={fps}")

    if vf_filters:
        cmd += ["-vf", ",".join(vf_filters)]

    cmd += [
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        str(dst),
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        log.error("FFmpeg error: %s", result.stderr[-500:] if result.stderr else "")
        raise RuntimeError(f"FFmpeg failed for {src} -> {dst}")

    log.debug("Extracted clip: %s (%.1fs @ %.1fs)", dst.name, duration_sec, start_sec)
    return dst


def sample_frames(
    path: str | Path,
    n_frames: int = 16,
    resize: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """Sample N evenly-spaced frames from a video file.

    Args:
        path: Video file path.
        n_frames: Number of frames to sample.
        resize: Optional (width, height) to resize frames.

    Returns:
        List of BGR numpy arrays.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {path}")

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            if resize is not None:
                frame = cv2.resize(frame, resize)
            frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(f"Could not read any frames from {path}")

    return frames


def extract_audio_wav(
    src: str | Path,
    dst: str | Path | None = None,
    sr: int = 16000,
) -> Path:
    """Extract audio track as mono WAV using FFmpeg.

    Args:
        src: Source video path.
        dst: Destination WAV path. Auto-generated if None.
        sr: Sample rate.

    Returns:
        Path to WAV file.
    """
    if dst is None:
        dst = Path(tempfile.mktemp(suffix=".wav"))
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-ac", "1",
        str(dst),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        log.warning("Audio extraction failed (video may have no audio): %s", src)
        return dst

    return dst
