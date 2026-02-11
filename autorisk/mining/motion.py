"""Optical-flow based motion signal scorer using Farneback."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig

from autorisk.mining.base import SignalScorer
from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)


class MotionScorer(SignalScorer):
    """Score sudden large motions via dense optical flow.

    Computes Farneback optical flow between consecutive frames,
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
        magnitudes: list[float] = []
        variances: list[float] = []

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return np.zeros(n_windows, dtype=np.float32)

        prev_gray = self._to_gray_resized(prev_frame, resize_h)

        frame_mags: list[float] = []
        frame_vars: list[float] = []
        frame_idx = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

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
            frame_mags.append(float(mag.mean()))
            frame_vars.append(float(mag.var()))

            # Window boundary
            if frame_idx % frames_per_window == 0 and frame_mags:
                magnitudes.append(np.mean(frame_mags))
                variances.append(np.mean(frame_vars))
                frame_mags.clear()
                frame_vars.clear()

            prev_gray = gray
            frame_idx += 1

        cap.release()

        # Flush remaining frames
        if frame_mags:
            magnitudes.append(np.mean(frame_mags))
            variances.append(np.mean(frame_vars))

        if not magnitudes:
            return np.zeros(n_windows, dtype=np.float32)

        mag_arr = np.array(magnitudes, dtype=np.float32)
        var_arr = np.array(variances, dtype=np.float32)

        # Combine mean magnitude and variance (sudden motion = high both)
        mag_n = self.normalize(mag_arr)
        var_n = self.normalize(var_arr)
        combined = 0.6 * mag_n + 0.4 * var_n

        log.info(
            "MotionScorer: %d windows, max_mag=%.3f, max_var=%.3f",
            len(combined), mag_arr.max(), var_arr.max(),
        )
        return combined

    @staticmethod
    def _to_gray_resized(frame: np.ndarray, short_edge: int) -> np.ndarray:
        h, w = frame.shape[:2]
        if min(h, w) > short_edge:
            scale = short_edge / min(h, w)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
