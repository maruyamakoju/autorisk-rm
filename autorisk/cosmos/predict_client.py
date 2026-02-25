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


# ---------------------------------------------------------------------------
# Hazard-type language for counterfactual prompts
# ---------------------------------------------------------------------------

_COLLISION_LANGUAGE = {
    "Pedestrian": "A pedestrian is struck by the vehicle.",
    "Collision": "A collision occurs with another vehicle or object.",
    "Cut-in": "A sideswipe collision occurs with the cutting-in vehicle.",
    "Vehicles": "A multi-vehicle collision occurs.",
}

_SAFE_LANGUAGE = {
    "Pedestrian": "The pedestrian crosses safely without contact.",
    "Collision": "The vehicle avoids the collision entirely.",
    "Cut-in": "The vehicles maintain safe separation.",
    "Vehicles": "All vehicles pass without contact.",
}


def _primary_hazard_type(cosmos_result: dict) -> str:
    """Return the first hazard type, or 'Collision' as fallback."""
    hazards = cosmos_result.get("hazards", [])
    if hazards and isinstance(hazards[0], dict):
        return hazards[0].get("type", "Collision")
    return "Collision"


def _match_hazard_language(hazard_type: str, table: dict[str, str]) -> str:
    """Match hazard type to language table using substring matching."""
    ht_lower = hazard_type.lower()
    for key, value in table.items():
        if key.lower() in ht_lower:
            return value
    return table.get("Collision", "A collision occurs.")


def build_danger_prompt(cosmos_result: dict) -> str:
    """Build a DANGER counterfactual prompt (no reaction -> collision).

    Uses causal_reasoning + short_term_prediction + hazard-specific collision.
    """
    causal = cosmos_result.get("causal_reasoning", "")
    prediction = cosmos_result.get("short_term_prediction", "")
    hazard_type = _primary_hazard_type(cosmos_result)
    collision = _match_hazard_language(hazard_type, _COLLISION_LANGUAGE)

    parts = ["Dashcam forward-facing view."]
    if causal:
        parts.append(causal[:300])
    if prediction:
        parts.append(prediction[:200])
    parts.append("The driver does not react in time.")
    parts.append(collision)

    return " ".join(parts)


def build_safe_prompt(cosmos_result: dict) -> str:
    """Build a SAFE counterfactual prompt (evasive action -> danger avoided).

    Uses recommended_action (falls back to causal_reasoning if empty).
    """
    action = cosmos_result.get("recommended_action", "").strip()
    causal = cosmos_result.get("causal_reasoning", "")
    hazard_type = _primary_hazard_type(cosmos_result)
    safe_resolution = _match_hazard_language(hazard_type, _SAFE_LANGUAGE)

    parts = ["Dashcam forward-facing view. The driver takes evasive action:"]
    if action:
        parts.append(action[:250])
    elif causal:
        parts.append(f"responding to {causal[:200]}")
    parts.append(safe_resolution)
    parts.append("No collision occurs.")

    return " ".join(parts)


def build_counterfactual_prompts(cosmos_result: dict) -> tuple[str, str]:
    """Build both DANGER and SAFE prompts for a clip.

    Returns:
        (danger_prompt, safe_prompt) tuple.
    """
    return build_danger_prompt(cosmos_result), build_safe_prompt(cosmos_result)


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

    def _predict_with_seed(
        self,
        clip_path: Path,
        prompt: str,
        output_path: Path,
        seed: int = 42,
    ) -> dict:
        """Generate a prediction video with a specific seed.

        Args:
            clip_path: Path to input clip video.
            prompt: Text prompt for generation.
            output_path: Path for output MP4.
            seed: Random seed for the generator.

        Returns:
            Dict with clip info and output path.
        """
        from diffusers.utils import export_to_video

        self.load_model()

        clip_path = Path(clip_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_image = _extract_last_frame(clip_path)

        log.info(
            "Generating prediction for %s (480p, %d frames, seed=%d)...",
            clip_path.name, NUM_FRAMES, seed,
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
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )

        gen_time = time.time() - t0
        log.info("Generated in %.1fs", gen_time)

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
            "seed": seed,
        }

    def predict_clip(
        self,
        clip_path: Path,
        prompt: str,
        output_path: Path,
    ) -> dict:
        """Generate a future prediction video from a clip's last frame."""
        return self._predict_with_seed(clip_path, prompt, output_path, seed=42)

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

    def predict_counterfactual_batch(
        self,
        cosmos_results: list[dict],
        clips_dir: Path,
        output_dir: Path,
        severity_filter: set[str] | None = None,
        corrected_results: list[dict] | None = None,
        danger_seed: int = 42,
        safe_seed: int = 137,
    ) -> list[dict]:
        """Generate DANGER/SAFE counterfactual video pairs for high-severity clips.

        Args:
            cosmos_results: List of Cosmos Reason 2 result dicts.
            clips_dir: Directory containing clip MP4 files.
            output_dir: Directory for output counterfactual videos.
            severity_filter: Severity levels to process (default: {"HIGH"}).
            corrected_results: Optional signal-corrected results (overrides severity).
            danger_seed: Seed for danger scenario generation.
            safe_seed: Seed for safe scenario generation.

        Returns:
            List of counterfactual result dicts (one per clip, with danger/safe sub-dicts).
        """
        if severity_filter is None:
            severity_filter = {"HIGH"}

        clips_dir = Path(clips_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build severity lookup from corrected results if provided
        corrected_severity: dict[str, str] = {}
        if corrected_results:
            for cr in corrected_results:
                name = Path(str(cr.get("clip_path", ""))).name
                corrected_severity[name] = cr.get("severity", "NONE")

        # Filter clips: use corrected severity if available, else original
        filtered = []
        for r in cosmos_results:
            clip_name = Path(str(r.get("clip_path", ""))).name
            sev = corrected_severity.get(clip_name, r.get("severity", "NONE"))
            if sev in severity_filter:
                filtered.append(r)

        log.info(
            "Counterfactual batch: %d/%d clips match severity filter %s",
            len(filtered), len(cosmos_results), severity_filter,
        )

        results = []
        for i, r in enumerate(filtered):
            clip_name = Path(str(r.get("clip_path", ""))).name
            clip_path = clips_dir / clip_name

            if not clip_path.exists():
                log.warning("Clip not found: %s", clip_path)
                continue

            stem = clip_path.stem
            danger_prompt, safe_prompt = build_counterfactual_prompts(r)

            log.info("=== Counterfactual %d/%d: %s ===", i + 1, len(filtered), clip_name)

            entry: dict = {"clip_name": clip_name, "severity": r.get("severity", "NONE")}

            # DANGER scenario
            try:
                danger_path = output_dir / f"{stem}_danger.mp4"
                log.info("  DANGER scenario (seed=%d)...", danger_seed)
                entry["danger"] = self._predict_with_seed(
                    clip_path, danger_prompt, danger_path, seed=danger_seed,
                )
                entry["danger"]["scenario"] = "danger"
            except Exception as e:
                log.error("  DANGER failed for %s: %s", clip_name, e)
                entry["danger"] = {
                    "clip_name": clip_name, "scenario": "danger",
                    "prompt": danger_prompt[:200], "output_path": "", "error": str(e),
                }

            # SAFE scenario
            try:
                safe_path = output_dir / f"{stem}_safe.mp4"
                log.info("  SAFE scenario (seed=%d)...", safe_seed)
                entry["safe"] = self._predict_with_seed(
                    clip_path, safe_prompt, safe_path, seed=safe_seed,
                )
                entry["safe"]["scenario"] = "safe"
            except Exception as e:
                log.error("  SAFE failed for %s: %s", clip_name, e)
                entry["safe"] = {
                    "clip_name": clip_name, "scenario": "safe",
                    "prompt": safe_prompt[:200], "output_path": "", "error": str(e),
                }

            results.append(entry)

        # Save metadata
        meta_path = output_dir / "counterfactual_results.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info("Counterfactual metadata saved to %s", meta_path)

        return results
