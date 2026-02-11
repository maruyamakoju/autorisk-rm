"""Abstract base class for signal scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from omegaconf import DictConfig


class SignalScorer(ABC):
    """Base class for per-second danger signal scorers.

    Each scorer produces a 1-D array of scores, one per time window,
    where higher values indicate greater likelihood of a dangerous event.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable signal name."""

    @abstractmethod
    def score(self, video_path: Path, fps: float) -> np.ndarray:
        """Compute per-window danger scores for the entire video.

        Args:
            video_path: Path to the input video.
            fps: Video frame rate.

        Returns:
            1-D float array of shape (n_windows,), raw scores.
        """

    @staticmethod
    def normalize(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores to [0, 1]."""
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-9:
            return np.zeros_like(scores)
        return (scores - mn) / (mx - mn)
