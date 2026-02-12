"""Local Cosmos Reason 2 inference using transformers."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from omegaconf import DictConfig

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)


class CosmosLocalClient:
    """Local inference client for Cosmos-Reason2-8B via transformers.

    Uses Qwen3VLForConditionalGeneration with video input.
    Requires ~16GB VRAM in float16.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        local_cfg = cfg.cosmos.local
        self.model_name = local_cfg.model_name
        self.max_new_tokens = local_cfg.max_new_tokens
        self.temperature = local_cfg.temperature
        self.torch_dtype = getattr(torch, local_cfg.torch_dtype, torch.float16)
        self.fps = cfg.cosmos.get("local_fps", 4)

        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        """Lazy-load model and processor on first use."""
        if self._model is not None:
            return

        import transformers

        log.info("Loading %s (this may take a minute)...", self.model_name)

        # Use HF token from env or huggingface_hub cache
        hf_token = os.environ.get("HF_TOKEN", None)

        self._model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=self.torch_dtype,
            device_map="auto",
            attn_implementation="sdpa",
            token=hf_token,
        )
        self._processor = transformers.AutoProcessor.from_pretrained(
            self.model_name, token=hf_token,
        )

        log.info(
            "Model loaded: %s (dtype=%s, device=%s)",
            self.model_name,
            self.torch_dtype,
            next(self._model.parameters()).device,
        )

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        video_b64: str = "",
        video_path: str | Path | None = None,
    ) -> str:
        """Run local inference on a video clip.

        Args:
            system_prompt: System message.
            user_prompt: User message text.
            video_b64: Unused (for API compatibility).
            video_path: Path to video file.

        Returns:
            Raw model response text.
        """
        if video_path is None:
            raise RuntimeError("video_path is required for local inference")

        self._load_model()

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Build messages in Qwen3-VL format
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path.resolve()),
                        "fps": self.fps,
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            fps=self.fps,
        )
        inputs = inputs.to(self._model.device)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]

        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]

    @staticmethod
    def encode_video_base64(video_path: str | Path) -> str:
        """No-op for API compatibility."""
        return ""
