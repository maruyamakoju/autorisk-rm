"""Signal fusion: weighted aggregation, peak detection, Top-N extraction."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from scipy.signal import find_peaks

from autorisk.mining.audio import AudioScorer
from autorisk.mining.base import SignalScorer
from autorisk.mining.motion import MotionScorer
from autorisk.mining.proximity import ProximityScorer
from autorisk.utils.logger import setup_logger
from autorisk.utils.video_io import extract_clip_ffmpeg, get_video_info

log = setup_logger(__name__)


@dataclass
class Candidate:
    """A danger candidate extracted from the video."""
    rank: int
    peak_time_sec: float
    start_sec: float
    end_sec: float
    fused_score: float
    clip_path: str = ""
    signal_scores: dict[str, float] = field(default_factory=dict)


class SignalFuser:
    """Fuse multiple signal scores, detect peaks, extract top-N clips."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.scorers: list[tuple[SignalScorer, float]] = []

        if cfg.mining.audio.enabled:
            self.scorers.append((AudioScorer(cfg), cfg.mining.audio.weight))
        if cfg.mining.motion.enabled:
            self.scorers.append((MotionScorer(cfg), cfg.mining.motion.weight))
        if cfg.mining.proximity.enabled:
            self.scorers.append((ProximityScorer(cfg), cfg.mining.proximity.weight))

    def compute_fused_scores(
        self, video_path: Path,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run all scorers and produce fused per-window scores.

        Returns:
            (fused_scores, per_signal_scores_dict)
        """
        info = get_video_info(video_path)
        fps = info["fps"]

        raw_signals: dict[str, np.ndarray] = {}
        weights: dict[str, float] = {}

        for scorer, w in self.scorers:
            log.info("Running %s scorer...", scorer.name)
            raw = scorer.score(video_path, fps)
            raw_signals[scorer.name] = raw
            weights[scorer.name] = w

        # Align lengths to minimum
        min_len = min(len(v) for v in raw_signals.values()) if raw_signals else 0
        if min_len == 0:
            return np.zeros(1, dtype=np.float32), {}

        normed: dict[str, np.ndarray] = {}
        for name, raw in raw_signals.items():
            trimmed = raw[:min_len]
            normed[name] = SignalScorer.normalize(trimmed)

        # Weighted sum
        total_weight = sum(weights.values())
        fused = np.zeros(min_len, dtype=np.float32)
        for name, n_scores in normed.items():
            fused += (weights[name] / total_weight) * n_scores

        return fused, normed

    def detect_peaks(self, fused_scores: np.ndarray) -> np.ndarray:
        """Find peaks in fused score array using scipy.

        Returns:
            Array of peak indices sorted by score descending.
        """
        pcfg = self.cfg.mining.peak_detection
        peaks, properties = find_peaks(
            fused_scores,
            distance=pcfg.distance,
            prominence=pcfg.prominence,
            height=pcfg.height,
        )

        if len(peaks) == 0:
            # Fallback: top-N by raw score
            log.warning("No peaks found by scipy, using top-N argmax fallback")
            n = min(self.cfg.mining.top_n, len(fused_scores))
            peaks = np.argsort(fused_scores)[-n:][::-1]
            return peaks

        # Sort by score descending
        order = np.argsort(fused_scores[peaks])[::-1]
        return peaks[order]

    def merge_nearby_peaks(
        self, peaks: np.ndarray, window_sec: float,
    ) -> np.ndarray:
        """Merge peaks within merge_gap_sec of each other (keep highest)."""
        merge_gap = self.cfg.mining.merge_gap_sec / window_sec
        if len(peaks) <= 1:
            return peaks

        merged = [peaks[0]]
        for p in peaks[1:]:
            if all(abs(p - m) >= merge_gap for m in merged):
                merged.append(p)

        return np.array(merged)

    def extract_candidates(
        self,
        video_path: Path,
        output_dir: Path,
    ) -> list[Candidate]:
        """Full extraction pipeline: score → fuse → peaks → clips.

        Args:
            video_path: Input video path.
            output_dir: Directory for output clips and CSV.

        Returns:
            List of Candidate objects sorted by fused score.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        clips_dir = output_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        info = get_video_info(video_path)
        duration = info["duration_sec"]
        window_sec = self.cfg.mining.window_sec

        log.info("Computing signal scores for %.1fs video...", duration)
        fused, per_signal = self.compute_fused_scores(video_path)

        log.info("Detecting peaks...")
        peaks = self.detect_peaks(fused)
        peaks = self.merge_nearby_peaks(peaks, window_sec)

        top_n = self.cfg.mining.top_n
        peaks = peaks[:top_n]
        log.info("Selected %d candidate peaks", len(peaks))

        padding = self.cfg.video.segment_padding_sec
        candidates: list[Candidate] = []

        for rank, peak_idx in enumerate(peaks, 1):
            peak_time = peak_idx * window_sec
            start = max(0, peak_time - padding)
            end = min(duration, peak_time + padding)
            clip_duration = end - start

            clip_name = f"candidate_{rank:03d}_t{peak_time:.1f}s.mp4"
            clip_path = clips_dir / clip_name

            try:
                extract_clip_ffmpeg(
                    video_path, clip_path,
                    start_sec=start,
                    duration_sec=clip_duration,
                    fps=self.cfg.video.clip_fps,
                    short_edge=self.cfg.video.resize_short_edge,
                )
            except RuntimeError:
                log.warning("Failed to extract clip %s, skipping", clip_name)
                continue

            sig_scores = {
                name: float(scores[peak_idx]) if peak_idx < len(scores) else 0.0
                for name, scores in per_signal.items()
            }

            candidates.append(Candidate(
                rank=rank,
                peak_time_sec=peak_time,
                start_sec=start,
                end_sec=end,
                fused_score=float(fused[peak_idx]),
                clip_path=str(clip_path),
                signal_scores=sig_scores,
            ))

        # Save candidates CSV
        csv_path = output_dir / "candidates.csv"
        self._save_csv(candidates, csv_path)
        log.info("Saved %d candidates to %s", len(candidates), csv_path)

        return candidates

    @staticmethod
    def _save_csv(candidates: list[Candidate], path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "rank", "peak_time_sec", "start_sec", "end_sec",
                "fused_score", "clip_path", "signal_scores",
            ])
            for c in candidates:
                writer.writerow([
                    c.rank, f"{c.peak_time_sec:.2f}",
                    f"{c.start_sec:.2f}", f"{c.end_sec:.2f}",
                    f"{c.fused_score:.4f}", c.clip_path,
                    str(c.signal_scores),
                ])
