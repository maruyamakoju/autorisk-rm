"""Optical-flow based motion signal scorer using Farneback."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig

from autorisk.mining.base import SignalScorer
from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)

# Process every Nth frame for optical flow (3x-5x speedup)
_FLOW_FRAME_SKIP = 3


class MotionScorer(SignalScorer):
    """Score sudden large motions via dense optical flow.

    Computes Farneback optical flow between sampled frames (every Nth),
    then aggregates per-window mean magnitude and variance.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.motion_cfg = cfg.mining.motion

    @property
    def name(self) -> str:
        return "motion"

    def score(self, video_path: Path, fps: float) -> np.ndarray:
        window_sec = self.cfg.mining.window_sec
        frames_per_window = max(1, int(fps * window_sec))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_windows = max(1, total_frames // frames_per_window)

        resize_h = self.cfg.video.resize_short_edge

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return np.zeros(n_windows, dtype=np.float32)

        prev_gray = self._to_gray_resized(prev_frame, resize_h)

        # Per-window accumulators
        window_mags: list[list[float]] = [[] for _ in range(n_windows)]
        window_vars: list[list[float]] = [[] for _ in range(n_windows)]

        frame_idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for speed
            if frame_idx % _FLOW_FRAME_SKIP != 0:
                frame_idx += 1
                continue

            gray = self._to_gray_resized(frame, resize_h)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=self.motion_cfg.flow_scale,
                levels=self.motion_cfg.flow_levels,
                winsize=self.motion_cfg.flow_winsize,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            win_idx = min(frame_idx // frames_per_window, n_windows - 1)
            window_mags[win_idx].append(float(mag.mean()))
            window_vars[win_idx].append(float(mag.var()))

            prev_gray = gray
            frame_idx += 1

        cap.release()

        magnitudes = np.zeros(n_windows, dtype=np.float32)
        variances = np.zeros(n_windows, dtype=np.float32)

        for i in range(n_windows):
            if window_mags[i]:
                magnitudes[i] = np.mean(window_mags[i])
                variances[i] = np.mean(window_vars[i])

        mag_n = self.normalize(magnitudes)
        var_n = self.normalize(variances)
        combined = 0.6 * mag_n + 0.4 * var_n

        log.info(
            "MotionScorer: %d windows, max_mag=%.3f, max_var=%.3f (skip=%d)",
            len(combined), magnitudes.max(), variances.max(), _FLOW_FRAME_SKIP,
        )
        return combined

    @staticmethod
    def _to_gray_resized(frame: np.ndarray, short_edge: int) -> np.ndarray:
        h, w = frame.shape[:2]
        if min(h, w) > short_edge:
            scale = short_edge / min(h, w)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
