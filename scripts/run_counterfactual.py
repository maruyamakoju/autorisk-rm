"""Generate DANGER/SAFE counterfactual videos for HIGH clips.

Uses subprocess isolation: each video is generated in its own Python process
to prevent CUDA error cascading (one OOM/error won't kill the rest).

Usage: python scripts/run_counterfactual.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Worker script executed as subprocess for a single video generation
# ---------------------------------------------------------------------------

WORKER_SCRIPT = r'''
import gc, json, sys, time, types
from pathlib import Path
import numpy as np, torch

class _PassthroughSafetyChecker:
    def to(self, device): return self
    def check_text_safety(self, text): return True
    def check_video_safety(self, video): return video

def _install_dummy_guardrail():
    import importlib, importlib.machinery
    mod = types.ModuleType("cosmos_guardrail")
    mod.CosmosSafetyChecker = _PassthroughSafetyChecker
    mod.__version__ = "0.0.0"
    mod.__spec__ = importlib.machinery.ModuleSpec("cosmos_guardrail", None)
    sys.modules["cosmos_guardrail"] = mod

def main():
    args = json.loads(sys.argv[1])
    clip_path = Path(args["clip_path"])
    prompt = args["prompt"]
    output_path = Path(args["output_path"])
    seed = args["seed"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    _install_dummy_guardrail()

    from autorisk.cosmos.predict_client import (
        FRAME_HEIGHT, FRAME_WIDTH, NEGATIVE_PROMPT, NUM_FRAMES, OUTPUT_FPS,
        _extract_last_frame,
    )

    from diffusers import Cosmos2VideoToWorldPipeline
    from diffusers.utils import export_to_video
    import diffusers.pipelines.cosmos.pipeline_cosmos2_video2world as _cosmos_mod
    _cosmos_mod.CosmosSafetyChecker = _PassthroughSafetyChecker

    pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
        "nvidia/Cosmos-Predict2-2B-Video2World", torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    input_image = _extract_last_frame(clip_path)

    t0 = time.time()
    output = pipe(
        image=input_image, prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
        height=FRAME_HEIGHT, width=FRAME_WIDTH, num_frames=NUM_FRAMES,
        num_inference_steps=30, guidance_scale=7.0,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    gen_time = time.time() - t0

    frames = output.frames[0] if hasattr(output, "frames") and output.frames else []
    if frames:
        export_to_video(frames, str(output_path), fps=OUTPUT_FPS)

    result = {
        "output_path": str(output_path) if frames else "",
        "generation_time_sec": round(gen_time, 1),
        "n_frames": len(frames),
        "seed": seed,
    }
    print("__RESULT__" + json.dumps(result))

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
'''


def _run_single_video(clip_path: str, prompt: str, output_path: str,
                      seed: int, label: str, timeout: int = 600) -> dict:
    """Run a single video generation in a subprocess."""
    args = json.dumps({
        "clip_path": clip_path,
        "prompt": prompt,
        "output_path": output_path,
        "seed": seed,
    })

    print(f"  {label} (seed={seed}): spawning subprocess...")
    t0 = time.time()

    try:
        result = subprocess.run(
            [sys.executable, "-c", WORKER_SCRIPT, args],
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )

        wall_time = time.time() - t0

        # Extract result from stdout
        for line in result.stdout.split("\n"):
            if line.startswith("__RESULT__"):
                data = json.loads(line[len("__RESULT__"):])
                print(f"  {label}: {data.get('n_frames', 0)} frames in {data.get('generation_time_sec', 0)}s "
                      f"(wall {wall_time:.0f}s)")
                return data

        # No result line found - check stderr
        err_msg = result.stderr[-500:] if result.stderr else "no output"
        print(f"  {label} FAILED (exit {result.returncode}, wall {wall_time:.0f}s): {err_msg[:200]}")
        return {"output_path": "", "error": err_msg[:500], "seed": seed}

    except subprocess.TimeoutExpired:
        wall_time = time.time() - t0
        print(f"  {label} TIMEOUT after {wall_time:.0f}s")
        return {"output_path": "", "error": f"timeout after {timeout}s", "seed": seed}
    except Exception as e:
        print(f"  {label} ERROR: {e}")
        return {"output_path": "", "error": str(e), "seed": seed}


def main():
    print("=" * 60)
    print("Counterfactual Safety Videos - Subprocess Isolation")
    print("=" * 60)

    # Load cosmos results
    results_path = Path("outputs/public_run/cosmos_results.json")
    clips_dir = Path("outputs/public_run/clips")
    output_dir = Path("outputs/public_run/counterfactuals")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path, encoding="utf-8") as f:
        cosmos_results = json.load(f)

    # Load corrected results if available
    corrected_path = Path("outputs/enhanced_correction/corrected_results.json")
    corrected_severity = {}
    if corrected_path.exists():
        with open(corrected_path, encoding="utf-8") as f:
            for cr in json.load(f):
                name = Path(str(cr.get("clip_path", ""))).name
                corrected_severity[name] = cr.get("severity", "NONE")
        print(f"Loaded corrected severity from {corrected_path}")

    # Filter HIGH clips
    filtered = []
    for r in cosmos_results:
        clip_name = Path(str(r.get("clip_path", ""))).name
        sev = corrected_severity.get(clip_name, r.get("severity", "NONE"))
        if sev == "HIGH":
            filtered.append(r)

    print(f"HIGH clips: {len(filtered)}")
    for r in filtered:
        name = Path(r["clip_path"]).name
        orig = r.get("severity", "NONE")
        corr = corrected_severity.get(name, orig)
        tag = f" (corrected from {orig})" if corr != orig else ""
        print(f"  {name}: {corr}{tag}")
    print()

    # Import prompt builder (lightweight, no GPU)
    from autorisk.cosmos.predict_client import build_counterfactual_prompts

    # Load existing results to skip already-generated videos
    meta_path = output_dir / "counterfactual_results.json"
    existing = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for entry in json.load(f):
                cn = entry.get("clip_name", "")
                if cn:
                    existing[cn] = entry

    DANGER_SEED = 42
    SAFE_SEED = 137

    results = []
    total_start = time.time()

    for i, r in enumerate(filtered):
        clip_name = Path(r["clip_path"]).name
        clip_path = clips_dir / clip_name
        stem = clip_path.stem

        if not clip_path.exists():
            print(f"SKIP: {clip_name} not found")
            continue

        danger_prompt, safe_prompt = build_counterfactual_prompts(r)

        print(f"\n[{i+1}/{len(filtered)}] {clip_name}")

        entry = {"clip_name": clip_name, "severity": r.get("severity", "NONE")}

        # Check if DANGER already exists
        prev = existing.get(clip_name, {})
        danger_existing = prev.get("danger", {}).get("output_path", "")
        if danger_existing and Path(danger_existing).exists():
            print(f"  DANGER: already exists, skipping")
            entry["danger"] = prev["danger"]
        else:
            danger_path = str(output_dir / f"{stem}_danger.mp4")
            data = _run_single_video(
                str(clip_path), danger_prompt, danger_path, DANGER_SEED, "DANGER",
            )
            entry["danger"] = {
                "clip_name": clip_name, "scenario": "danger",
                "prompt": danger_prompt[:200], **data,
            }

        # Check if SAFE already exists
        safe_existing = prev.get("safe", {}).get("output_path", "")
        if safe_existing and Path(safe_existing).exists():
            print(f"  SAFE: already exists, skipping")
            entry["safe"] = prev["safe"]
        else:
            safe_path = str(output_dir / f"{stem}_safe.mp4")
            data = _run_single_video(
                str(clip_path), safe_prompt, safe_path, SAFE_SEED, "SAFE",
            )
            entry["safe"] = {
                "clip_name": clip_name, "scenario": "safe",
                "prompt": safe_prompt[:200], **data,
            }

        results.append(entry)

        # Incremental save
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    total_time = time.time() - total_start
    n_videos = sum(
        (1 if r.get("danger", {}).get("output_path") else 0)
        + (1 if r.get("safe", {}).get("output_path") else 0)
        for r in results
    )

    print()
    print("=" * 60)
    print(f"Done! {n_videos} videos generated ({len(results)} clips) in {total_time:.0f}s")
    print(f"Metadata: {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
