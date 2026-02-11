"""NVIDIA Build API client for Cosmos Reason 2."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import requests
from omegaconf import DictConfig

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)


class CosmosAPIClient:
    """Client for NVIDIA Build API (OpenAI-compatible endpoint).

    Handles base64 video encoding, rate limiting, and retries.
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

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        video_b64: str,
    ) -> str:
        """Send a chat completion request with video content.

        Args:
            system_prompt: System message.
            user_prompt: User message text.
            video_b64: Base64-encoded video data.

        Returns:
            Raw model response text.
        """
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY not set - cannot call Cosmos API")

        self._enforce_rate_limit()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:video/mp4;base64,{video_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                },
            ],
        }

        url = f"{self.base_url}/chat/completions"

        for attempt in range(1, self.max_retries + 1):
            try:
                self._request_times.append(time.time())
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 429:
                    wait = self.retry_delay * attempt
                    log.warning("Rate limited (429), retrying in %.1fs...", wait)
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

            except requests.exceptions.Timeout:
                log.warning("Request timeout (attempt %d/%d)", attempt, self.max_retries)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
            except requests.exceptions.RequestException as e:
                log.error("API request failed (attempt %d/%d): %s", attempt, self.max_retries, e)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(f"Cosmos API failed after {self.max_retries} attempts")
