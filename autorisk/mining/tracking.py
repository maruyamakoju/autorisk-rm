"""TTC (Time-to-Collision) estimator via object tracking.

Uses YOLOv8n + ByteTrack to track objects across clip frames,
then estimates TTC from bounding-box expansion rate (tau approximation).

TTC_tau = bbox_area / (d(bbox_area)/dt)

This is the monocular "tau" metric from ecological optics: objects approaching
the camera produce expanding retinal images, and the expansion rate directly
encodes time-to-contact without requiring depth or calibration.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)

# Safety threshold: TTC below this (seconds) indicates imminent danger
TTC_CRITICAL = 1.5
TTC_WARNING = 3.0


@dataclass
class TrackTTC:
    """TTC result for a single tracked object."""
    track_id: int
    class_name: str
    min_ttc: float  # Minimum TTC observed (seconds)
    mean_ttc: float  # Mean TTC over track lifetime
    ttc_timeline: list[float] = field(default_factory=list)
    # Track metadata
    first_frame: int = 0
    last_frame: int = 0
    max_bbox_area_ratio: float = 0.0  # Max bbox area / frame area


@dataclass
class ClipTTCResult:
    """TTC analysis result for a single clip."""
    clip_path: str
    min_ttc: float  # Global minimum TTC across all tracks
    mean_min_ttc: float  # Mean of per-track minimum TTCs
    n_tracks: int
    n_critical: int  # Tracks with TTC < TTC_CRITICAL
    n_warning: int  # Tracks with TTC < TTC_WARNING
    tracks: list[TrackTTC] = field(default_factory=list)
    # Per-second TTC timeline (minimum across all active tracks)
    ttc_per_second: list[float] = field(default_factory=list)


# COCO class names for the tracked classes
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck",
}


class TTCEstimator:
    """Estimate Time-to-Collision for objects in dashcam clips."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._tracker = None
        # Process every N-th frame for speed (TTC needs dense sampling)
        self.frame_step = 2

    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLO
            model_name = self.cfg.mining.proximity.model
            self._model = YOLO(model_name)
            log.info("TTC: Loaded YOLO model: %s", model_name)
        return self._model

    def _new_tracker(self):
        import supervision as sv
        return sv.ByteTrack(
            track_activation_threshold=0.3,
            lost_track_buffer=15,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )

    def analyze_clip(self, clip_path: Path | str) -> ClipTTCResult:
        """Analyze a single clip for TTC.

        Args:
            clip_path: Path to the video clip.

        Returns:
            ClipTTCResult with per-track and per-second TTC data.
        """
        import supervision as sv

        clip_path = Path(clip_path)
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            log.warning("TTC: Cannot open %s", clip_path)
            return ClipTTCResult(
                clip_path=str(clip_path), min_ttc=float("inf"),
                mean_min_ttc=float("inf"), n_tracks=0,
                n_critical=0, n_warning=0,
            )

        fps = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(
            cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS), 1e-6
        ) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_area = max(frame_w * frame_h, 1)

        model = self._get_model()
        tracker = self._new_tracker()
        classes = list(self.cfg.mining.proximity.classes)
        conf_thresh = self.cfg.mining.proximity.confidence

        # Track areas over time: track_id -> [(frame_idx, bbox_area, class_id)]
        track_history: dict[int, list[tuple[int, float, int]]] = defaultdict(list)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_step == 0:
                results = model(
                    frame, conf=conf_thresh, classes=classes, verbose=False,
                )
                detections = sv.Detections.from_ultralytics(results[0])

                if len(detections) > 0:
                    tracked = tracker.update_with_detections(detections)

                    if tracked.tracker_id is not None:
                        for i, tid in enumerate(tracked.tracker_id):
                            x1, y1, x2, y2 = tracked.xyxy[i]
                            area = (x2 - x1) * (y2 - y1)
                            cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                            track_history[tid].append((frame_idx, float(area), cls_id))
                else:
                    # Still update tracker with empty detections
                    tracker.update_with_detections(
                        sv.Detections.empty()
                    )

            frame_idx += 1

        cap.release()

        # Compute TTC for each track
        dt = self.frame_step / fps  # Time between processed frames
        track_results: list[TrackTTC] = []

        for tid, history in track_history.items():
            if len(history) < 3:
                continue

            frames = [h[0] for h in history]
            areas = np.array([h[1] for h in history])
            cls_id = history[0][2]

            # Compute area rate of change (forward difference)
            ttc_values = []
            for i in range(1, len(areas)):
                da_dt = (areas[i] - areas[i - 1]) / dt
                if da_dt > 0:  # Object approaching (area growing)
                    ttc = areas[i] / da_dt
                    ttc = min(ttc, 30.0)  # Cap at 30s
                    ttc_values.append(ttc)
                else:
                    ttc_values.append(float("inf"))  # Object receding

            if not ttc_values:
                continue

            finite_ttcs = [t for t in ttc_values if t < float("inf")]
            if not finite_ttcs:
                continue

            track_results.append(TrackTTC(
                track_id=tid,
                class_name=COCO_NAMES.get(cls_id, f"class_{cls_id}"),
                min_ttc=min(finite_ttcs),
                mean_ttc=float(np.mean(finite_ttcs)),
                ttc_timeline=ttc_values,
                first_frame=frames[0],
                last_frame=frames[-1],
                max_bbox_area_ratio=float(areas.max() / frame_area),
            ))

        # Compute per-second TTC timeline
        total_seconds = int(frame_idx / fps) + 1
        ttc_per_second: list[float] = []

        for sec in range(total_seconds):
            sec_start = sec * fps
            sec_end = (sec + 1) * fps
            min_ttc_sec = float("inf")

            for tid, history in track_history.items():
                for frame_i, area, _ in history:
                    if sec_start <= frame_i < sec_end:
                        # Find this frame's TTC from track results
                        for tr in track_results:
                            if tr.track_id == tid:
                                idx_in_track = next(
                                    (j for j, h in enumerate(history) if h[0] == frame_i),
                                    None,
                                )
                                if idx_in_track is not None and idx_in_track > 0:
                                    ttc_idx = idx_in_track - 1
                                    if ttc_idx < len(tr.ttc_timeline):
                                        min_ttc_sec = min(
                                            min_ttc_sec, tr.ttc_timeline[ttc_idx],
                                        )
                                break

            ttc_per_second.append(min_ttc_sec if min_ttc_sec < float("inf") else -1.0)

        # Aggregate
        if track_results:
            global_min_ttc = min(t.min_ttc for t in track_results)
            mean_min_ttc = float(np.mean([t.min_ttc for t in track_results]))
        else:
            global_min_ttc = float("inf")
            mean_min_ttc = float("inf")

        n_critical = sum(1 for t in track_results if t.min_ttc < TTC_CRITICAL)
        n_warning = sum(1 for t in track_results if t.min_ttc < TTC_WARNING)

        return ClipTTCResult(
            clip_path=str(clip_path),
            min_ttc=global_min_ttc,
            mean_min_ttc=mean_min_ttc,
            n_tracks=len(track_results),
            n_critical=n_critical,
            n_warning=n_warning,
            tracks=sorted(track_results, key=lambda t: t.min_ttc),
            ttc_per_second=ttc_per_second,
        )

    def analyze_clips(
        self,
        clips_dir: Path | str,
        output_dir: Path | str | None = None,
    ) -> list[ClipTTCResult]:
        """Analyze all clips in a directory.

        Args:
            clips_dir: Directory containing clip MP4 files.
            output_dir: Optional output dir for saving results.

        Returns:
            List of ClipTTCResult, one per clip.
        """
        from tqdm import tqdm

        clips_dir = Path(clips_dir)
        clips = sorted(clips_dir.glob("candidate_*.mp4"))
        if not clips:
            log.warning("TTC: No candidate clips found in %s", clips_dir)
            return []

        log.info("TTC: Analyzing %d clips...", len(clips))
        results = []
        for clip in tqdm(clips, desc="TTC analysis"):
            result = self.analyze_clip(clip)
            results.append(result)

            min_ttc_str = f"{result.min_ttc:.2f}s" if result.min_ttc < float("inf") else "inf"
            log.info(
                "  %s: min_TTC=%s, tracks=%d, critical=%d",
                clip.name, min_ttc_str, result.n_tracks, result.n_critical,
            )

        if output_dir is not None:
            self.save_results(results, Path(output_dir))

        return results

    @staticmethod
    def save_results(results: list[ClipTTCResult], output_dir: Path) -> Path:
        """Save TTC results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "ttc_results.json"

        data = []
        for r in results:
            data.append({
                "clip_path": r.clip_path,
                "min_ttc": r.min_ttc if r.min_ttc < float("inf") else None,
                "mean_min_ttc": r.mean_min_ttc if r.mean_min_ttc < float("inf") else None,
                "n_tracks": r.n_tracks,
                "n_critical": r.n_critical,
                "n_warning": r.n_warning,
                "ttc_per_second": [float(x) for x in r.ttc_per_second],
                "tracks": [
                    {
                        "track_id": int(t.track_id),
                        "class_name": t.class_name,
                        "min_ttc": float(t.min_ttc),
                        "mean_ttc": float(t.mean_ttc),
                        "max_bbox_area_ratio": float(t.max_bbox_area_ratio),
                    }
                    for t in r.tracks[:5]  # Top 5 most critical tracks
                ],
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        log.info("TTC: Saved results to %s", path)
        return path

    @staticmethod
    def load_results(path: Path) -> list[ClipTTCResult]:
        """Load TTC results from JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        results = []
        for entry in data:
            tracks = [
                TrackTTC(
                    track_id=t["track_id"],
                    class_name=t["class_name"],
                    min_ttc=t["min_ttc"],
                    mean_ttc=t["mean_ttc"],
                    max_bbox_area_ratio=t.get("max_bbox_area_ratio", 0.0),
                )
                for t in entry.get("tracks", [])
            ]
            results.append(ClipTTCResult(
                clip_path=entry["clip_path"],
                min_ttc=entry["min_ttc"] if entry["min_ttc"] is not None else float("inf"),
                mean_min_ttc=entry["mean_min_ttc"] if entry["mean_min_ttc"] is not None else float("inf"),
                n_tracks=entry["n_tracks"],
                n_critical=entry["n_critical"],
                n_warning=entry["n_warning"],
                tracks=tracks,
                ttc_per_second=entry.get("ttc_per_second", []),
            ))
        return results


def compute_ttc_severity_correlation(
    ttc_results: list[ClipTTCResult],
    gt_labels: dict[str, str],
) -> dict:
    """Compute correlation between TTC and GT severity.

    Args:
        ttc_results: TTC analysis results per clip.
        gt_labels: Ground truth severity labels keyed by clip filename.

    Returns:
        Dict with correlation statistics.
    """
    from scipy import stats

    severity_order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

    ttc_vals = []
    sev_ords = []

    for r in ttc_results:
        clip_name = Path(r.clip_path).name
        gt_sev = gt_labels.get(clip_name)
        if gt_sev is None or r.min_ttc == float("inf"):
            continue
        ttc_vals.append(r.min_ttc)
        sev_ords.append(severity_order[gt_sev])

    if len(ttc_vals) < 3:
        return {"n_samples": len(ttc_vals), "spearman_rho": 0.0, "spearman_p": 1.0}

    # TTC should be negatively correlated with severity (lower TTC = higher danger)
    rho, p = stats.spearmanr(ttc_vals, sev_ords)

    # Mean TTC by severity
    mean_by_sev = {}
    for sev_name, sev_ord in severity_order.items():
        mask_ttcs = [
            ttc_vals[i] for i in range(len(sev_ords)) if sev_ords[i] == sev_ord
        ]
        if mask_ttcs:
            mean_by_sev[sev_name] = float(np.mean(mask_ttcs))

    return {
        "n_samples": len(ttc_vals),
        "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
        "spearman_p": float(p) if not np.isnan(p) else 1.0,
        "mean_ttc_by_severity": mean_by_sev,
    }
