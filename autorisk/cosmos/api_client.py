"""NVIDIA Build API client for Cosmos Reason 2."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from openai import OpenAI

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)

_FALLBACK_MODEL = "meta/llama-3.2-11b-vision-instruct"


class CosmosAPIClient:
    """Client for NVIDIA Build API (OpenAI-compatible endpoint).

    Cosmos: video_url (base64 MP4) + extra_body media_io_kwargs
    Fallback VLM: image_url (single JPEG frame)
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
        self.cosmos_num_frames = cfg.cosmos.max_frames_per_clip

        self._rpm_limit = api_cfg.rate_limit_rpm
        self._request_times: list[float] = []

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
        """Probe primary model; fall back if unavailable."""
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
            log.info("Using primary model: %s (video mode)", self.model)
        except Exception:
            log.warning(
                "Primary model %s unavailable, falling back to %s (image mode)",
                self.model, _FALLBACK_MODEL,
            )
            self._active_model = _FALLBACK_MODEL
            self._use_frames = True

        return self._active_model

    def _enforce_rate_limit(self) -> None:
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= self._rpm_limit:
            sleep_time = 60 - (now - self._request_times[0]) + 0.1
            log.info("Rate limit: sleeping %.1fs", sleep_time)
            time.sleep(sleep_time)

    @staticmethod
    def encode_video_base64(video_path: str | Path) -> str:
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def extract_middle_frame_base64(video_path: str | Path) -> str:
        """Extract the middle frame as base64 JPEG."""
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid = max(0, total // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Cannot read frame from {video_path}")
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _build_cosmos_request(
        self, system_prompt: str, user_prompt: str, video_b64: str,
    ) -> tuple[list[dict], dict]:
        """Build messages + extra_body for Cosmos (video mode)."""
        content = [
            {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
            },
            {"type": "text", "text": user_prompt},
        ]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        extra_body = {
            "media_io_kwargs": {
                "video": {"fps": 1.0, "num_frames": self.cosmos_num_frames},
            },
        }
        return messages, extra_body

    def _build_fallback_request(
        self, system_prompt: str, user_prompt: str, video_path: str | Path,
    ) -> tuple[list[dict], dict]:
        """Build messages for fallback VLM (single image mode)."""
        frame_b64 = self.extract_middle_frame_base64(video_path)
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
            },
            {
                "type": "text",
                "text": (
                    "This is a key frame from a dashcam video clip. "
                    f"Analyze it for driving risks.\n\n{user_prompt}"
                ),
            },
        ]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return messages, {}

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        video_b64: str,
        video_path: str | Path | None = None,
    ) -> str:
        """Send chat completion with video (Cosmos) or image (fallback).

        Args:
            system_prompt: System message.
            user_prompt: User message text.
            video_b64: Base64-encoded MP4 (for Cosmos).
            video_path: Path to video file (for fallback frame extraction).

        Returns:
            Raw model response text.
        """
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY not set - cannot call API")

        self._enforce_rate_limit()

        client = self._get_client()
        model = self._resolve_model()

        if self._use_frames:
            if video_path is None:
                raise RuntimeError("video_path required for fallback frame mode")
            messages, extra_body = self._build_fallback_request(
                system_prompt, user_prompt, video_path,
            )
        else:
            messages, extra_body = self._build_cosmos_request(
                system_prompt, user_prompt, video_b64,
            )

        for attempt in range(1, self.max_retries + 1):
            try:
                self._request_times.append(time.time())

                kwargs = dict(
                    model=model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    timeout=self.timeout,
                )
                if extra_body:
                    kwargs["extra_body"] = extra_body

                response = client.chat.completions.create(**kwargs)
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
