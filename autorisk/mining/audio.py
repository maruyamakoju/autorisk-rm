"""Audio-based danger signal scorer: RMS, delta-RMS, spectral, horn-band."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from scipy.io import wavfile

from autorisk.mining.base import SignalScorer
from autorisk.utils.logger import setup_logger
from autorisk.utils.video_io import extract_audio_wav

log = setup_logger(__name__)


class AudioScorer(SignalScorer):
    """Score audio energy spikes that may indicate dangerous events.

    Combines:
    - RMS energy per window
    - Delta-RMS (sudden loudness change)
    - Spectral centroid shift
    - Horn-band energy (300-1000 Hz)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.audio_cfg = cfg.mining.audio

    @property
    def name(self) -> str:
        return "audio"

    def score(self, video_path: Path, fps: float) -> np.ndarray:
        window_sec = self.cfg.mining.window_sec
        wav_path = extract_audio_wav(video_path)

        try:
            sr, data = wavfile.read(str(wav_path))
        except Exception:
            log.warning("No audio track found, returning zero scores")
            duration = 0
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / max(fps, 1e-6)
                cap.release()
            except Exception:
                pass
            n_windows = max(1, int(duration / window_sec))
            return np.zeros(n_windows, dtype=np.float32)

        if data.dtype != np.float32:
            data = data.astype(np.float32) / max(np.abs(data).max(), 1e-9)

        samples_per_window = int(sr * window_sec)
        n_windows = len(data) // samples_per_window
        if n_windows == 0:
            return np.zeros(1, dtype=np.float32)

        data = data[: n_windows * samples_per_window]
        windows = data.reshape(n_windows, samples_per_window)

        # RMS per window
        rms = np.sqrt(np.mean(windows ** 2, axis=1))

        # Delta-RMS
        delta_w = self.audio_cfg.delta_window
        delta_rms = np.zeros_like(rms)
        for i in range(delta_w, len(rms)):
            delta_rms[i] = max(0, rms[i] - np.mean(rms[max(0, i - delta_w):i]))

        # Horn-band energy via FFT
        horn_lo, horn_hi = self.audio_cfg.horn_band_hz
        horn_energy = np.zeros(n_windows, dtype=np.float32)
        freqs = np.fft.rfftfreq(samples_per_window, d=1.0 / sr)
        band_mask = (freqs >= horn_lo) & (freqs <= horn_hi)

        for i in range(n_windows):
            spectrum = np.abs(np.fft.rfft(windows[i]))
            total_energy = np.sum(spectrum ** 2) + 1e-9
            horn_energy[i] = np.sum(spectrum[band_mask] ** 2) / total_energy

        # Combine sub-signals (equal weights)
        rms_n = self.normalize(rms)
        delta_n = self.normalize(delta_rms)
        horn_n = self.normalize(horn_energy)

        combined = (rms_n + delta_n + horn_n) / 3.0

        log.info(
            "AudioScorer: %d windows, max_rms=%.3f, max_delta=%.3f, max_horn=%.3f",
            n_windows, rms_n.max(), delta_n.max(), horn_n.max(),
        )
        return combined.astype(np.float32)
