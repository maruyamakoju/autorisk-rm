"""Cosmos Predict 2 client for future video generation.

Uses nvidia/Cosmos-Predict2-2B-Video2World (Cosmos2VideoToWorldPipeline)
to generate "what happens next" predictions from dashcam clips.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import cv2
import torch
from omegaconf import DictConfig

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)

NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed scenes, "
    "poor color balance, washed out colors, choppy sequences, jerky movements, "
    "low frame rate, artifacting, color banding, unnatural transitions, "
    "outdated special effects, fake elements, unconvincing visuals, poorly edited content, "
    "jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
)

# 480p for VRAM safety on 32GB GPUs; 720p = 1280x704 needs ~32.5GB
FRAME_WIDTH = 832
FRAME_HEIGHT = 480
NUM_FRAMES = 50  # 5 seconds at 10 FPS
OUTPUT_FPS = 10


def _extract_last_frame(video_path: Path) -> "Image.Image":
    """Extract the last frame from a video as a PIL Image, resized for Predict 2.

    Returns:
        PIL Image at (FRAME_WIDTH x FRAME_HEIGHT).
    """
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_count - 1, 0))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Cannot read last frame from: {video_path}")

    # OpenCV BGR -> RGB -> PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = img.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.LANCZOS)
    return img


def build_prompt(cosmos_result: dict) -> str:
    """Build a prediction prompt from Cosmos Reason 2 output.

    Uses causal_reasoning and short_term_prediction to create context
    for the world model prediction.
    """
    causal = cosmos_result.get("causal_reasoning", "")
    prediction = cosmos_result.get("short_term_prediction", "")
    severity = cosmos_result.get("severity", "NONE")

    parts = ["Dashcam forward-facing view."]

    if causal:
        parts.append(causal[:300])

    if prediction:
        parts.append(prediction[:200])

    if severity in ("HIGH", "MEDIUM"):
        parts.append("Show what happens next in this dangerous driving scenario.")

    return " ".join(parts)


class CosmosPredictClient:
    """Client for Cosmos-Predict2-2B-Video2World inference.

    Uses Cosmos2VideoToWorldPipeline from diffusers.
    Generates 5-second 480p prediction videos from dashcam clip last frames.
    Requires ~32.5GB VRAM in BF16; uses CPU offload as fallback.
    """

    MODEL_ID = "nvidia/Cosmos-Predict2-2B-Video2World"

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        predict_cfg = cfg.cosmos.get("predict", {})
        self.model_name = predict_cfg.get("model_name", self.MODEL_ID)
        self.enabled = predict_cfg.get("enabled", False)
        self.severity_filter = set(predict_cfg.get("severity_filter", ["HIGH", "MEDIUM"]))
        self._pipeline = None
        self._use_cpu_offload = False

    def load_model(self) -> None:
        """Load the Cosmos Predict 2 diffusion pipeline."""
        if self._pipeline is not None:
            return

        try:
            from diffusers import Cosmos2VideoToWorldPipeline
        except ImportError:
            raise ImportError(
                "diffusers >= 0.34.0 is required for Cosmos Predict 2. "
                "Install with: pip install -U diffusers"
            )

        log.info("Loading %s...", self.model_name)
        t0 = time.time()

        try:
            self._pipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
            )
            self._pipeline.to("cuda")
            log.info("Model loaded on GPU in %.1fs", time.time() - t0)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            log.warning("CUDA OOM, enabling CPU offload: %s", e)
            if self._pipeline is not None:
                del self._pipeline
            gc.collect()
            torch.cuda.empty_cache()

            self._pipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
            )
            self._pipeline.enable_model_cpu_offload()
            self._use_cpu_offload = True
            log.info("Model loaded with CPU offload in %.1fs", time.time() - t0)

    def unload(self) -> None:
        """Release model and free VRAM."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Cosmos Predict 2 model unloaded.")

    def predict_clip(
        self,
        clip_path: Path,
        prompt: str,
        output_path: Path,
    ) -> dict:
        """Generate a future prediction video from a clip's last frame.

        Args:
            clip_path: Path to input clip video.
            prompt: Text prompt for generation.
            output_path: Path for output MP4.

        Returns:
            Dict with clip info and output path.
        """
        from diffusers.utils import export_to_video

        self.load_model()

        clip_path = Path(clip_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract last frame as PIL Image
        input_image = _extract_last_frame(clip_path)

        log.info(
            "Generating prediction for %s (480p, %d frames)...",
            clip_path.name, NUM_FRAMES,
        )
        t0 = time.time()

        output = self._pipeline(
            image=input_image,
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=FRAME_HEIGHT,
            width=FRAME_WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=30,
            guidance_scale=7.0,
            generator=torch.Generator(device="cuda").manual_seed(42),
        )

        gen_time = time.time() - t0
        log.info("Generated in %.1fs", gen_time)

        # Save output video
        frames = output.frames[0] if hasattr(output, "frames") and output.frames else []
        n_frames = len(frames)

        if frames:
            export_to_video(frames, str(output_path), fps=OUTPUT_FPS)
            log.info("Saved %d frames to %s", n_frames, output_path)
        else:
            log.warning("No frames generated for %s", clip_path.name)

        return {
            "clip_name": clip_path.name,
            "prompt": prompt[:200],
            "output_path": str(output_path),
            "generation_time_sec": round(gen_time, 1),
            "n_frames": n_frames,
            "resolution": f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            "fps": OUTPUT_FPS,
            "cpu_offload": self._use_cpu_offload,
        }

    def predict_batch(
        self,
        cosmos_results: list[dict],
        clips_dir: Path,
        output_dir: Path,
        severity_filter: set[str] | None = None,
    ) -> list[dict]:
        """Generate predictions for filtered clips.

        Args:
            cosmos_results: List of Cosmos Reason 2 result dicts.
            clips_dir: Directory containing clip MP4 files.
            output_dir: Directory for output prediction videos.
            severity_filter: Set of severity levels to process.

        Returns:
            List of prediction result dicts.
        """
        if severity_filter is None:
            severity_filter = self.severity_filter

        clips_dir = Path(clips_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Filter clips by severity
        filtered = [
            r for r in cosmos_results
            if r.get("severity", "NONE") in severity_filter
        ]
        log.info(
            "Predict batch: %d/%d clips match severity filter %s",
            len(filtered), len(cosmos_results), severity_filter,
        )

        results = []
        for i, r in enumerate(filtered):
            clip_name = Path(str(r.get("clip_path", ""))).name
            clip_path = clips_dir / clip_name

            if not clip_path.exists():
                log.warning("Clip not found: %s", clip_path)
                continue

            prompt = build_prompt(r)
            out_path = output_dir / f"predict_{clip_name}"

            log.info("=== Clip %d/%d: %s ===", i + 1, len(filtered), clip_name)

            try:
                result = self.predict_clip(clip_path, prompt, out_path)
                results.append(result)
            except Exception as e:
                log.error("Failed to predict %s: %s", clip_name, e)
                results.append({
                    "clip_name": clip_name,
                    "prompt": prompt[:200],
                    "output_path": "",
                    "error": str(e),
                })

        # Save metadata
        meta_path = output_dir / "predict_results.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info("Predict metadata saved to %s", meta_path)

        return results
