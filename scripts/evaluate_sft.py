"""Evaluate LoRA fine-tuned Cosmos-Reason2-8B vs base model on SFT val set.

Runs MCQ inference on the validation split (or train/all), computes accuracy
for each question type (severity_mcq, high_detection, evasive_action), and
prints a before/after comparison table.

Usage:
    python scripts/evaluate_sft.py
    python scripts/evaluate_sft.py --checkpoint outputs/sft_lora/best_checkpoint
    python scripts/evaluate_sft.py --split all --checkpoint outputs/sft_lora/final
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="outputs/sft_lora/best_checkpoint",
        help="Path to LoRA adapter checkpoint (saved via model.save_pretrained())",
    )
    p.add_argument("--data-dir", default="data/sft")
    p.add_argument(
        "--split", choices=["val", "train", "all"], default="val",
        help="Which data split to evaluate on (default: val)"
    )
    p.add_argument("--nframes", type=int, default=4, help="Video frames to sample (default 4 for 8B on 32GB)")
    p.add_argument("--max-new-tokens", type=int, default=5,
                   help="Max tokens to generate per answer (MCQ answer is 1 letter)")
    p.add_argument(
        "--no-base", action="store_true",
        help="Skip base model evaluation (saves time if you only want LoRA results)"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def extract_answer(generated_text: str) -> str:
    """Extract single letter answer (A/B/C/D) from generated text."""
    for char in generated_text.strip():
        if char in "ABCD":
            return char
    return "?"


def run_inference(
    model,
    processor,
    samples: list[dict],
    nframes: int,
    max_new_tokens: int,
    device: torch.device,
    tag: str,
) -> list[dict]:
    """Run MCQ inference on samples; return list of result dicts."""
    from qwen_vl_utils import process_vision_info

    results = []
    model.eval()

    for i, sample in enumerate(samples):
        video_path = sample["video"]
        conv = sample["conversations"]
        human_text = conv[0]["value"]
        gt_answer = conv[1]["value"]

        # Strip <video>\n tag — we pass video via content type instead
        question_text = human_text.replace("<video>\n", "").replace("<video>", "").strip()

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "nframes": nframes},
                    {"type": "text", "text": question_text},
                ],
            },
        ]

        try:
            if not Path(video_path).exists():
                print(f"  [{tag}] SKIP (video not found): {Path(video_path).name}")
                continue

            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            _, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[prompt_text],
                videos=video_inputs,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            # Decode only newly generated tokens
            new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            generated = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
            pred = extract_answer(generated)

            results.append({
                "id": sample["id"],
                "type": sample["type"],
                "gt_severity": sample.get("gt_severity", ""),
                "gt": gt_answer,
                "pred": pred,
                "correct": pred == gt_answer,
                "generated": generated.strip(),
            })

        except Exception as e:
            print(f"  [{tag}] ERROR on {sample.get('id', '?')}: {e}")
            continue

        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            n_done = len(results)
            acc = sum(r["correct"] for r in results) / max(n_done, 1)
            print(f"  [{tag}] {i + 1}/{len(samples)} (done={n_done}) acc={acc:.3f}")

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy overall and per question type."""
    by_type: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r["correct"])

    metrics = {
        "overall": sum(r["correct"] for r in results) / max(len(results), 1),
        "n": len(results),
    }
    for q_type, corrects in sorted(by_type.items()):
        metrics[q_type] = sum(corrects) / max(len(corrects), 1)

    return metrics


def print_comparison(base_metrics: dict, lora_metrics: dict):
    """Print before/after accuracy comparison table."""
    print()
    print("=" * 62)
    print("  LoRA Fine-Tuning Results  (Cosmos-Reason2-2B)")
    print("=" * 62)
    print(f"  {'Metric':<26} {'Base':>8} {'LoRA':>8} {'Delta':>8}")
    print("  " + "-" * 55)

    all_keys = sorted(
        k for k in set(list(base_metrics) + list(lora_metrics))
        if k != "n"
    )
    for key in all_keys:
        base_val = base_metrics.get(key, float("nan"))
        lora_val = lora_metrics.get(key, float("nan"))
        if lora_metrics:
            delta = lora_val - base_val
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.3f}"
        else:
            delta_str = "  N/A"
        bstr = f"{base_val:.3f}" if base_val == base_val else " N/A"
        lstr = f"{lora_val:.3f}" if (lora_metrics and lora_val == lora_val) else " N/A"
        print(f"  {key:<26} {bstr:>8} {lstr:>8} {delta_str:>8}")

    print("  " + "-" * 55)
    n_base = base_metrics.get("n", 0)
    n_lora = lora_metrics.get("n", 0) if lora_metrics else 0
    print(f"  {'n_samples':<26} {n_base:>8} {n_lora:>8}")
    print("=" * 62)


def print_errors(results: list[dict], tag: str, top_n: int = 5):
    """Print top-N incorrect predictions for inspection."""
    errors = [r for r in results if not r["correct"]]
    if not errors:
        print(f"\n[{tag}] All {len(results)} predictions correct!")
        return
    print(f"\n[{tag}] {len(errors)}/{len(results)} incorrect predictions (showing up to {top_n}):")
    for r in errors[:top_n]:
        print(f"  {r['id']}: gt={r['gt']} pred={r['pred']} generated='{r['generated']}'")


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_base_model(model_name: str, hf_token: str | None):
    import transformers
    processor = transformers.AutoProcessor.from_pretrained(model_name, token=hf_token)
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        token=hf_token,
    )
    return model, processor


def load_lora_model(model_name: str, checkpoint_path: Path, hf_token: str | None):
    import transformers
    from peft import PeftModel

    processor = transformers.AutoProcessor.from_pretrained(model_name, token=hf_token)
    base = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        token=hf_token,
    )
    model = PeftModel.from_pretrained(base, str(checkpoint_path))
    # Merge LoRA weights into base for faster inference
    model = model.merge_and_unload()
    return model, processor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint)

    # Select split
    split_file = {"val": "sft_val.json", "train": "sft_train.json", "all": "sft_all.json"}[args.split]
    data_path = data_dir / split_file
    if not data_path.exists():
        print(f"ERROR: Data not found at {data_path}")
        print("Run: python scripts/prepare_sft_data.py first")
        return

    samples = load_dataset(data_path)
    print(f"Loaded {len(samples)} samples from {data_path} (split={args.split})")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    device = torch.device("cuda")
    MODEL_NAME = "nvidia/Cosmos-Reason2-8B"  # 2B not cached; use 8B
    hf_token = __import__("os").environ.get("HF_TOKEN", None)

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -----------------------------------------------------------------------
    # Base model
    # -----------------------------------------------------------------------
    base_metrics: dict = {}
    base_results: list[dict] = []

    if not args.no_base:
        print(f"\nLoading base model ({MODEL_NAME})...")
        t0 = time.time()
        base_model, processor = load_base_model(MODEL_NAME, hf_token)
        print(f"Base model loaded in {time.time() - t0:.1f}s")

        print(f"\nRunning base model inference ({len(samples)} samples)...")
        base_results = run_inference(
            base_model, processor, samples, args.nframes, args.max_new_tokens, device, "BASE"
        )
        base_metrics = compute_metrics(base_results)
        print(f"\nBase accuracy: {base_metrics['overall']:.3f} ({base_metrics['n']} samples)")
        print_errors(base_results, "BASE")

        del base_model
        gc.collect()
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # LoRA model
    # -----------------------------------------------------------------------
    lora_metrics: dict = {}
    lora_results: list[dict] = []

    if not checkpoint_path.exists():
        print(f"\nWARNING: Checkpoint not found at {checkpoint_path}")
        print("Skipping LoRA evaluation. Run training first:")
        print("  python scripts/run_sft_lora.py --epochs 3")
    else:
        print(f"\nLoading LoRA model from {checkpoint_path}...")
        t0 = time.time()
        lora_model, processor = load_lora_model(MODEL_NAME, checkpoint_path, hf_token)
        print(f"LoRA model loaded in {time.time() - t0:.1f}s")

        print(f"\nRunning LoRA model inference ({len(samples)} samples)...")
        lora_results = run_inference(
            lora_model, processor, samples, args.nframes, args.max_new_tokens, device, "LORA"
        )
        lora_metrics = compute_metrics(lora_results)
        print(f"\nLoRA accuracy: {lora_metrics['overall']:.3f} ({lora_metrics['n']} samples)")
        print_errors(lora_results, "LORA")

        del lora_model
        gc.collect()
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------------
    print_comparison(base_metrics, lora_metrics)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out_dir = checkpoint_path.parent if checkpoint_path.exists() else Path("outputs/sft_lora")
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_out = {
        "model": MODEL_NAME,
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "nframes": args.nframes,
        "base_metrics": base_metrics,
        "lora_metrics": lora_metrics,
        "base_results": base_results,
        "lora_results": lora_results,
    }
    out_path = out_dir / f"eval_{args.split}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved evaluation results → {out_path}")


if __name__ == "__main__":
    main()
