"""Proximity-based danger scorer using YOLO object detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig

from autorisk.mining.base import SignalScorer
from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)


class ProximityScorer(SignalScorer):
    """Score danger based on detected object proximity and size.

    Uses YOLOv8n to detect vehicles/pedestrians, then scores each window
    by max bounding-box area (close objects) and center proximity.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.prox_cfg = cfg.mining.proximity
        self._model = None

    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.prox_cfg.model)
            log.info("Loaded YOLO model: %s", self.prox_cfg.model)
        return self._model

    @property
    def name(self) -> str:
        return "proximity"

    def score(self, video_path: Path, fps: float) -> np.ndarray:
        window_sec = self.cfg.mining.window_sec
        frames_per_window = max(1, int(fps * window_sec))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_area = max(frame_w * frame_h, 1)
        center_x, center_y = frame_w / 2, frame_h / 2

        model = self._get_model()
        classes = list(self.prox_cfg.classes)
        conf_thresh = self.prox_cfg.confidence
        center_weight = self.prox_cfg.center_weight

        window_scores: list[float] = []
        window_frame_scores: list[float] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO on a subset of frames (every 3rd) for speed
            if frame_idx % 3 == 0:
                results = model(
                    frame,
                    conf=conf_thresh,
                    classes=classes,
                    verbose=False,
                )

                max_score = 0.0
                for r in results:
                    if r.boxes is None or len(r.boxes) == 0:
                        continue
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox_area = (x2 - x1) * (y2 - y1)
                        area_score = bbox_area / frame_area if self.prox_cfg.bbox_area_norm else bbox_area

                        # Center proximity: how close is the box center to frame center
                        bx, by = (x1 + x2) / 2, (y1 + y2) / 2
                        dist = np.sqrt((bx - center_x) ** 2 + (by - center_y) ** 2)
                        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
                        center_score = 1.0 - (dist / max(max_dist, 1e-6))

                        combined = (1 - center_weight) * area_score + center_weight * center_score
                        max_score = max(max_score, combined)

                window_frame_scores.append(max_score)

            # Window boundary
            if (frame_idx + 1) % frames_per_window == 0 and window_frame_scores:
                window_scores.append(np.max(window_frame_scores))
                window_frame_scores.clear()

            frame_idx += 1

        cap.release()

        # Flush remaining
        if window_frame_scores:
            window_scores.append(np.max(window_frame_scores))

        if not window_scores:
            n_windows = max(1, total_frames // frames_per_window)
            return np.zeros(n_windows, dtype=np.float32)

        scores = np.array(window_scores, dtype=np.float32)
        log.info(
            "ProximityScorer: %d windows, max_score=%.3f, mean_score=%.3f",
            len(scores), scores.max(), scores.mean(),
        )
        return scores
