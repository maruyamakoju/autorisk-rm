"""NVIDIA Build API client for Cosmos Reason 2."""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from openai import OpenAI

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)

# Fallback VLM if primary model is unavailable
_FALLBACK_MODEL = "meta/llama-3.2-11b-vision-instruct"


class CosmosAPIClient:
    """Client for NVIDIA Build API (OpenAI-compatible endpoint).

    Handles base64 video encoding, rate limiting, retries, and model fallback.
    For Cosmos models: sends video as base64 video_url.
    For fallback VLMs: extracts frames and sends as images.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        api_cfg = cfg.cosmos.api
        self.base_url = api_cfg.base_url
        self.model = api_cfg.model
        self.max_tokens = api_cfg.max_tokens
        self.temperature = api_cfg.temperature
        self.top_p = api_cfg.top_p
        self.timeout = api_cfg.timeout_sec
        self.max_retries = api_cfg.max_retries
        self.retry_delay = api_cfg.retry_delay_sec

        # Rate limiting
        self._rpm_limit = api_cfg.rate_limit_rpm
        self._request_times: list[float] = []

        # API key
        self.api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not self.api_key:
            log.warning("NVIDIA_API_KEY not set. API calls will fail.")

        self._client: OpenAI | None = None
        self._active_model: str | None = None
        self._use_frames: bool = False

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client

    def _resolve_model(self) -> str:
        """Return a working model name, falling back if primary 404s."""
        if self._active_model is not None:
            return self._active_model

        client = self._get_client()
        try:
            client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            self._active_model = self.model
            self._use_frames = False
            log.info("Using primary model: %s", self.model)
        except Exception:
            log.warning(
                "Primary model %s unavailable, falling back to %s",
                self.model, _FALLBACK_MODEL,
            )
            self._active_model = _FALLBACK_MODEL
            self._use_frames = True  # Fallback VLM needs images, not video

        return self._active_model

    def _enforce_rate_limit(self) -> None:
        """Sleep if needed to stay within RPM limit."""
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= self._rpm_limit:
            sleep_time = 60 - (now - self._request_times[0]) + 0.1
            log.info("Rate limit: sleeping %.1fs", sleep_time)
            time.sleep(sleep_time)

    @staticmethod
    def encode_video_base64(video_path: str | Path) -> str:
        """Read video file and return base64-encoded string."""
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def extract_frames_base64(
        video_path: str | Path,
        n_frames: int = 8,
    ) -> list[str]:
        """Extract N evenly-spaced frames and return as base64 JPEG strings."""
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        frames_b64 = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                frames_b64.append(b64)

        cap.release()
        return frames_b64

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        video_b64: str,
        video_path: str | Path | None = None,
    ) -> str:
        """Send a chat completion request with video/image content.

        Args:
            system_prompt: System message.
            user_prompt: User message text.
            video_b64: Base64-encoded video data (used for Cosmos).
            video_path: Original video path (used for frame extraction fallback).

        Returns:
            Raw model response text.
        """
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY not set - cannot call Cosmos API")

        self._enforce_rate_limit()

        client = self._get_client()
        model = self._resolve_model()

        # Build content based on model type
        if self._use_frames and video_path is not None:
            # Frame-based: extract middle frame as representative image
            frames_b64 = self.extract_frames_base64(video_path, n_frames=3)
            if not frames_b64:
                raise RuntimeError(f"No frames extracted from {video_path}")

            # Use middle frame as most representative
            mid_frame = frames_b64[len(frames_b64) // 2]

            content: list[dict] = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{mid_frame}"},
                },
                {
                    "type": "text",
                    "text": (
                        "This is a frame from a dashcam video clip. "
                        f"Analyze it for driving risks.\n\n{user_prompt}"
                    ),
                },
            ]
        else:
            # Video-based: send full video (Cosmos native)
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                },
                {"type": "text", "text": user_prompt},
            ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                self._request_times.append(time.time())
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    timeout=self.timeout,
                )
                return response.choices[0].message.content

            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    wait = self.retry_delay * attempt
                    log.warning("Rate limited (429), retrying in %.1fs...", wait)
                    time.sleep(wait)
                    continue

                log.error(
                    "API request failed (attempt %d/%d): %s",
                    attempt, self.max_retries, err_str[:200],
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(f"API failed after {self.max_retries} attempts")
