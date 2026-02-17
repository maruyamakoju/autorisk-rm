"""Attention / saliency visualization for Cosmos Reason 2 (Qwen3-VL).

Uses gradient-based input saliency: computes the gradient of the model's
next-token logits with respect to the visual input tokens, revealing which
spatial regions most influence the model's risk assessment.

This is a standard explainability technique (Input × Gradient / Vanilla Gradient)
from the interpretable ML literature, adapted for vision-language models.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)


def _apply_heatmap(
    frame: np.ndarray,
    saliency: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay saliency heatmap on frame.

    Args:
        frame: BGR image (H, W, 3).
        saliency: 2D saliency map (H', W'), values in [0, 1].
        alpha: Blending factor.
        colormap: OpenCV colormap.

    Returns:
        Blended image (H, W, 3).
    """
    h, w = frame.shape[:2]
    # Resize saliency to frame dimensions
    sal_resized = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_CUBIC)
    # Normalize to [0, 255]
    sal_uint8 = (sal_resized * 255).astype(np.uint8)
    # Apply colormap
    heatmap = cv2.applyColorMap(sal_uint8, colormap)
    # Blend
    blended = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
    return blended


def _frame_to_b64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode frame as base64 JPEG."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("ascii")


class SaliencyExtractor:
    """Extract gradient-based saliency maps from Cosmos Reason 2 (Qwen3-VL).

    For each clip, performs a single forward pass with gradient tracking,
    then computes the gradient of the most likely next token with respect
    to visual input tokens. The gradient magnitude per spatial position
    gives a saliency map showing "where the model looks."
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._processor = None
        self.fps = cfg.cosmos.get("local_fps", 4)

    def _load_model(self) -> None:
        """Load model for saliency extraction."""
        if self._model is not None:
            return

        import transformers

        model_name = self.cfg.cosmos.local.model_name
        dtype = getattr(torch, self.cfg.cosmos.local.torch_dtype, torch.float16)
        hf_token = os.environ.get("HF_TOKEN", None)

        log.info("Saliency: Loading %s for gradient extraction...", model_name)

        self._model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto",
            # Use eager attention for gradient compatibility
            attn_implementation="eager",
            token=hf_token,
        )
        self._processor = transformers.AutoProcessor.from_pretrained(
            model_name, token=hf_token,
        )
        log.info("Saliency: Model loaded (eager attention for gradient flow)")

    def extract_saliency(
        self,
        clip_path: Path | str,
        system_prompt: str = "",
        user_prompt: str = "",
    ) -> dict:
        """Extract saliency map for a single clip.

        Args:
            clip_path: Path to video clip.
            system_prompt: System prompt (for context).
            user_prompt: User prompt.

        Returns:
            Dict with saliency data:
                - saliency_frames: list of (frame_idx, saliency_2d) tuples
                - peak_frame_idx: index of frame with highest saliency
                - heatmap_b64: base64 encoded heatmap overlay on peak frame
                - raw_frame_b64: base64 encoded raw peak frame
        """
        self._load_model()
        clip_path = Path(clip_path)

        if not system_prompt:
            from autorisk.cosmos.prompt import SYSTEM_PROMPT, USER_PROMPT
            system_prompt = SYSTEM_PROMPT
            user_prompt = USER_PROMPT

        # Build messages
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
                        "video": str(clip_path.resolve()),
                        "fps": self.fps,
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        t0 = time.time()
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            fps=self.fps,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Get grid dimensions for spatial mapping
        video_grid_thw = inputs.get("video_grid_thw", None)  # (num_videos, 3)

        # Enable gradients on video pixel values
        pixel_key = "pixel_values_videos"
        if pixel_key not in inputs or inputs[pixel_key] is None:
            log.warning("Saliency: No video pixel values for %s", clip_path.name)
            return {}

        pixel_values = inputs[pixel_key]
        pixel_values = pixel_values.detach().clone().requires_grad_(True)
        inputs[pixel_key] = pixel_values

        # Single forward pass (NOT generate)
        self._model.eval()
        outputs = self._model(**inputs)

        # Get logits for the last position (next-token prediction)
        logits = outputs.logits[0, -1, :]  # (vocab_size,)

        # Take the top-k logits and compute gradient
        top_logit = logits.max()
        top_logit.backward(retain_graph=False)

        # Get gradient w.r.t. pixel values
        grad = pixel_values.grad  # Same shape as pixel_values

        if grad is None:
            log.warning("Saliency: No gradient computed for %s", clip_path.name)
            return {}

        grad_time = time.time() - t0
        log.info("Saliency: Gradient computed in %.1fs for %s", grad_time, clip_path.name)

        # Process gradient into spatial saliency maps
        # pixel_values shape: (total_patches, channels, patch_h, patch_w)
        # or it could be (batch, channels, ...) depending on preprocessing
        grad_np = grad.detach().cpu().float().numpy()

        # Compute per-patch saliency (L2 norm across channels)
        if grad_np.ndim == 4:
            # (N_patches, C, pH, pW) → take norm across C, pH, pW
            patch_saliency = np.linalg.norm(
                grad_np.reshape(grad_np.shape[0], -1), axis=1,
            )
        elif grad_np.ndim == 3:
            # (N_patches, C, features) → norm across last dims
            patch_saliency = np.linalg.norm(
                grad_np.reshape(grad_np.shape[0], -1), axis=1,
            )
        else:
            # Flatten and use raw magnitude
            patch_saliency = np.abs(grad_np).reshape(grad_np.shape[0], -1).mean(axis=1)

        # Map patches to spatial grid using video_grid_thw
        if video_grid_thw is not None:
            thw = video_grid_thw[0].cpu().numpy()  # (T, H, W)
            t_grid, h_grid, w_grid = int(thw[0]), int(thw[1]), int(thw[2])
        else:
            # Fallback: assume square grid
            total = len(patch_saliency)
            t_grid = max(1, int(np.cbrt(total)))
            h_grid = w_grid = int(np.sqrt(total / t_grid))

        n_spatial = h_grid * w_grid
        n_temporal = min(t_grid, len(patch_saliency) // max(n_spatial, 1))

        if n_temporal == 0 or n_spatial == 0:
            log.warning("Saliency: Invalid grid dimensions for %s", clip_path.name)
            return {}

        # Reshape to (T, H, W)
        used_patches = n_temporal * n_spatial
        sal_grid = patch_saliency[:used_patches].reshape(n_temporal, h_grid, w_grid)

        # Normalize to [0, 1]
        sal_min = sal_grid.min()
        sal_max = sal_grid.max()
        if sal_max > sal_min:
            sal_grid = (sal_grid - sal_min) / (sal_max - sal_min)
        else:
            sal_grid = np.zeros_like(sal_grid)

        # Find peak frame (highest total saliency)
        frame_saliency_sums = sal_grid.sum(axis=(1, 2))
        peak_frame_idx = int(np.argmax(frame_saliency_sums))

        # Get the peak saliency map
        peak_saliency = sal_grid[peak_frame_idx]  # (H_grid, W_grid)

        # Extract the corresponding raw frame from the clip
        cap = cv2.VideoCapture(str(clip_path))
        clip_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Map peak_frame_idx (in model's temporal grid) to actual video frame
        frame_indices = np.linspace(0, total_frames - 1, n_temporal, dtype=int)
        actual_frame_idx = frame_indices[peak_frame_idx] if peak_frame_idx < len(frame_indices) else total_frames // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
        ret, raw_frame = cap.read()
        cap.release()

        if not ret:
            log.warning("Saliency: Cannot read frame %d from %s", actual_frame_idx, clip_path.name)
            return {}

        # Create heatmap overlay
        heatmap_frame = _apply_heatmap(raw_frame, peak_saliency, alpha=0.45)

        # Resize for report embedding
        target_w = 480
        h, w = raw_frame.shape[:2]
        target_h = int(h * target_w / w)
        raw_frame_resized = cv2.resize(raw_frame, (target_w, target_h))
        heatmap_resized = cv2.resize(heatmap_frame, (target_w, target_h))

        # Clean up GPU memory
        del grad, pixel_values, outputs, logits
        torch.cuda.empty_cache()

        result = {
            "clip_name": clip_path.name,
            "n_temporal_frames": n_temporal,
            "spatial_grid": (h_grid, w_grid),
            "peak_frame_idx": peak_frame_idx,
            "peak_frame_actual": int(actual_frame_idx),
            "frame_saliency_sums": frame_saliency_sums.tolist(),
            "heatmap_b64": _frame_to_b64(heatmap_resized),
            "raw_frame_b64": _frame_to_b64(raw_frame_resized),
        }

        log.info(
            "Saliency: %s — peak frame %d/%d, grid %dx%d",
            clip_path.name, peak_frame_idx, n_temporal, h_grid, w_grid,
        )

        return result

    def extract_batch(
        self,
        clips_dir: Path | str,
        output_dir: Path | str,
        severity_filter: set[str] | None = None,
        cosmos_results_path: Path | str | None = None,
        max_clips: int = 10,
    ) -> list[dict]:
        """Extract saliency for multiple clips.

        Args:
            clips_dir: Directory with clip MP4 files.
            output_dir: Output directory for results.
            severity_filter: Only process clips with these severities.
            cosmos_results_path: Path to cosmos_results.json for filtering.
            max_clips: Maximum number of clips to process.

        Returns:
            List of saliency result dicts.
        """
        clips_dir = Path(clips_dir)
        output_dir = Path(output_dir)

        # Determine which clips to process
        if cosmos_results_path and severity_filter:
            with open(cosmos_results_path, encoding="utf-8") as f:
                cosmos_results = json.load(f)

            clip_names = [
                Path(r["clip_path"]).name
                for r in cosmos_results
                if r.get("severity", "NONE") in severity_filter
                and r.get("parse_success", True)
            ]
        else:
            clip_names = [c.name for c in sorted(clips_dir.glob("candidate_*.mp4"))]

        clip_names = clip_names[:max_clips]
        log.info("Saliency: Processing %d clips...", len(clip_names))

        results = []
        for clip_name in clip_names:
            clip_path = clips_dir / clip_name
            if not clip_path.exists():
                continue

            try:
                result = self.extract_saliency(clip_path)
                if result:
                    results.append(result)
            except Exception as e:
                log.error("Saliency: Failed for %s: %s", clip_name, e)
                continue

        # Save results
        self.save_results(results, output_dir)
        return results

    @staticmethod
    def save_results(results: list[dict], output_dir: Path) -> Path:
        """Save saliency results to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "saliency_results.json"

        # Save without base64 images (too large for JSON)
        # Images are embedded in the HTML report directly
        save_data = []
        for r in results:
            save_data.append({
                "clip_name": r["clip_name"],
                "n_temporal_frames": r["n_temporal_frames"],
                "spatial_grid": r["spatial_grid"],
                "peak_frame_idx": r["peak_frame_idx"],
                "peak_frame_actual": r["peak_frame_actual"],
                "frame_saliency_sums": r["frame_saliency_sums"],
                "has_heatmap": bool(r.get("heatmap_b64")),
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        # Save base64 images separately for report
        images_path = output_dir / "saliency_images.json"
        images_data = {
            r["clip_name"]: {
                "heatmap_b64": r.get("heatmap_b64", ""),
                "raw_frame_b64": r.get("raw_frame_b64", ""),
            }
            for r in results
            if r.get("heatmap_b64")
        }

        with open(images_path, "w", encoding="utf-8") as f:
            json.dump(images_data, f, ensure_ascii=False)

        log.info("Saliency: Saved %d results to %s", len(results), path)
        return path

    @staticmethod
    def load_images(output_dir: Path) -> dict[str, dict]:
        """Load saliency images from JSON."""
        path = Path(output_dir) / "saliency_images.json"
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)
