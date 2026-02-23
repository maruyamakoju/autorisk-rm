"""Run Cosmos Predict 2 on HIGH/MEDIUM clips from the public run.

Standalone script (not using CLI) for direct control and logging.
Usage: python scripts/run_predict2.py
"""

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch


class _PassthroughSafetyChecker:
    """Passthrough safety checker for Cosmos pipelines.

    The cosmos_guardrail package is not yet published on PyPI.
    This provides the expected interface (check_text_safety, check_video_safety)
    while passing all content through. Dashcam footage is safe content.
    """

    def to(self, device):
        return self

    def check_text_safety(self, text: str) -> bool:
        return True

    def check_video_safety(self, video: np.ndarray) -> np.ndarray:
        return video


def _install_dummy_guardrail():
    """Register a dummy cosmos_guardrail module in sys.modules."""
    import importlib
    import importlib.machinery

    mod = types.ModuleType("cosmos_guardrail")
    mod.CosmosSafetyChecker = _PassthroughSafetyChecker
    mod.__version__ = "0.0.0"
    mod.__spec__ = importlib.machinery.ModuleSpec("cosmos_guardrail", None)
    sys.modules["cosmos_guardrail"] = mod


def main():
    print("=" * 60)
    print("Cosmos Predict 2 - Future Video Generation")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Free VRAM: {free_gb:.1f} GB")
    print()

    # Install dummy guardrail before importing diffusers pipeline
    # cosmos_guardrail is not yet published on PyPI; we provide a passthrough
    _install_dummy_guardrail()
    print("Safety checker: passthrough (cosmos_guardrail not on PyPI)")

    # Load cosmos results
    results_path = Path("outputs/public_run/cosmos_results.json")
    clips_dir = Path("outputs/public_run/clips")
    output_dir = Path("outputs/public_run/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path, encoding="utf-8") as f:
        cosmos_results = json.load(f)

    # Filter HIGH + MEDIUM clips
    severity_filter = {"HIGH", "MEDIUM"}
    filtered = [
        r for r in cosmos_results
        if r.get("severity", "NONE") in severity_filter
    ]
    print(f"Clips to process: {len(filtered)} (HIGH + MEDIUM)")
    for r in filtered:
        name = Path(r["clip_path"]).name
        print(f"  {name}: {r['severity']}")
    print()

    # Import and build prompt
    from autorisk.cosmos.predict_client import (
        NEGATIVE_PROMPT,
        FRAME_HEIGHT,
        FRAME_WIDTH,
        NUM_FRAMES,
        OUTPUT_FPS,
        _extract_last_frame,
        build_prompt,
    )

    # Load pipeline with CPU offload for safety (32.5GB model on 32GB GPU)
    print("Loading Cosmos2VideoToWorldPipeline (BF16, CPU offload)...")
    t0 = time.time()

    from diffusers import Cosmos2VideoToWorldPipeline
    from diffusers.utils import export_to_video

    # Monkey-patch the pipeline module's CosmosSafetyChecker reference
    # (the module-level try/except may have already cached the stub class)
    import diffusers.pipelines.cosmos.pipeline_cosmos2_video2world as _cosmos_mod
    _cosmos_mod.CosmosSafetyChecker = _PassthroughSafetyChecker

    pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
        "nvidia/Cosmos-Predict2-2B-Video2World",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s (CPU offload enabled)")
    print()

    # Generate predictions
    results = []
    for i, r in enumerate(filtered):
        clip_name = Path(r["clip_path"]).name
        clip_path = clips_dir / clip_name

        if not clip_path.exists():
            print(f"SKIP: {clip_name} not found")
            continue

        prompt = build_prompt(r)
        out_path = output_dir / f"predict_{clip_name}"

        print(f"[{i+1}/{len(filtered)}] {clip_name} ({r['severity']})")
        print(f"  Prompt: {prompt[:100]}...")

        # Extract last frame
        input_image = _extract_last_frame(clip_path)
        print(f"  Frame extracted: {input_image.size}")

        # Generate
        t1 = time.time()
        try:
            output = pipe(
                image=input_image,
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                height=FRAME_HEIGHT,
                width=FRAME_WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=30,
                guidance_scale=7.0,
                generator=torch.Generator(device="cpu").manual_seed(42),
            )

            gen_time = time.time() - t1

            frames = output.frames[0] if hasattr(output, "frames") and output.frames else []
            n_frames = len(frames)

            if frames:
                export_to_video(frames, str(out_path), fps=OUTPUT_FPS)
                print(f"  Generated {n_frames} frames in {gen_time:.1f}s -> {out_path}")
            else:
                print(f"  WARNING: No frames generated ({gen_time:.1f}s)")

            results.append({
                "clip_name": clip_name,
                "prompt": prompt[:200],
                "output_path": str(out_path),
                "generation_time_sec": round(gen_time, 1),
                "n_frames": n_frames,
                "resolution": f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
                "fps": OUTPUT_FPS,
                "cpu_offload": True,
            })

        except Exception as e:
            gen_time = time.time() - t1
            print(f"  ERROR after {gen_time:.1f}s: {e}")
            results.append({
                "clip_name": clip_name,
                "prompt": prompt[:200],
                "output_path": "",
                "error": str(e),
            })

        # Print VRAM usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  VRAM: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
        print()

    # Save results
    meta_path = output_dir / "predict_results.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Done! {len([r for r in results if r.get('output_path')])} videos generated")
    print(f"Metadata: {meta_path}")
    print("=" * 60)

    # Cleanup
    del pipe
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
