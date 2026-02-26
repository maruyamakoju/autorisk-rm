"""LoRA fine-tuning of Cosmos-Reason2-8B on dashcam severity classification.

Single-GPU training on RTX 5090 (32GB) using PEFT LoRA.
Follows the Cosmos Cookbook Intelligent Transportation recipe approach,
adapted for single-GPU with LoRA instead of full SFT on 8x A100.

Uses Cosmos-Reason2-8B (already cached) with nframes=4 to fit 32GB VRAM.
Base model weights are frozen; only LoRA adapter params are trained.

Usage: python scripts/run_sft_lora.py [--epochs 3] [--lr 2e-4] [--nframes 4]
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "nvidia/Cosmos-Reason2-8B"  # 2B not cached; 8B works with LoRA + grad checkpointing
OUTPUT_DIR = Path("outputs/sft_lora")
DATA_DIR = Path("data/sft")

# LoRA config (matches Qwen3-VL target layers)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training config
NFRAMES = 2           # 2 frames (very conservative for 8B on 32GB; increase if VRAM allows)
MAX_SEQ_LEN = 4096
BATCH_SIZE = 1        # per-step batch (single GPU)
GRAD_ACCUM = 8        # effective batch = 8
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--nframes", type=int, default=NFRAMES,
                   help="Video frames per clip (default 4; reduce if OOM)")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--resume-from", default=None, help="Path to checkpoint dir to resume")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Video processing helpers
# ---------------------------------------------------------------------------

def build_messages_with_response(sample: dict, nframes: int) -> tuple[list[dict], list[dict]]:
    """Build Qwen3-VL messages for a sample.

    Returns:
        (prompt_messages, full_messages) — prompt-only and prompt+response
    """
    video_path = sample["video"]
    conv = sample["conversations"]
    human_text = conv[0]["value"]  # "<video>\n[question]"
    gpt_text = conv[1]["value"]    # "A", "B", "C", or "D"

    # Remove leading <video>\n tag from human text (handled via content type)
    question_text = human_text.replace("<video>\n", "").replace("<video>", "").strip()

    prompt_messages = [
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

    full_messages = prompt_messages + [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": gpt_text}],
        }
    ]

    return prompt_messages, full_messages


def find_response_start(input_ids: torch.Tensor, im_start_id: int, assistant_ids: list[int]) -> int:
    """Find token index where assistant response starts.

    Searches backward for the last <|im_start|>assistant\\n sequence.
    Returns the index AFTER that sequence (where the actual response begins).
    """
    ids_list = input_ids.tolist()
    marker = [im_start_id] + assistant_ids
    marker_len = len(marker)

    for i in range(len(ids_list) - marker_len, -1, -1):
        if ids_list[i:i + marker_len] == marker:
            return i + marker_len

    # Fallback: try to find just im_start from the end
    for i in range(len(ids_list) - 1, -1, -1):
        if ids_list[i] == im_start_id:
            return i + 1

    return 0  # last resort: compute loss on everything


def prepare_batch(
    samples: list[dict],
    processor,
    nframes: int,
    max_seq_len: int,
    device: torch.device,
    im_start_id: int,
    assistant_token_ids: list[int],
) -> dict | None:
    """Process a list of samples into model inputs with labels.

    Returns None if processing fails (e.g., missing video file).
    """
    from qwen_vl_utils import process_vision_info

    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    all_pixel_values_videos = []
    all_video_grid_thw = []

    for sample in samples:
        try:
            _, full_messages = build_messages_with_response(sample, nframes)

            # Check video exists
            video_path = sample["video"]
            if not Path(video_path).exists():
                print(f"  SKIP (video not found): {video_path}")
                continue

            # Apply chat template to full conversation (prompt + response)
            full_text = processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Process vision info
            _, video_inputs = process_vision_info(full_messages)

            if not video_inputs:
                print(f"  SKIP (no video inputs): {sample['id']}")
                continue

            # Tokenize
            inputs = processor(
                text=[full_text],
                videos=video_inputs,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_seq_len,
            )

            input_ids = inputs["input_ids"][0]  # [seq_len]
            attention_mask = inputs["attention_mask"][0]

            # Create labels: -100 for prompt, actual ids for response
            labels = input_ids.clone()
            response_start = find_response_start(
                input_ids, im_start_id, assistant_token_ids
            )
            labels[:response_start] = -100  # mask prompt tokens

            # Ensure at least some response tokens are being trained on
            n_response_tokens = (labels != -100).sum().item()
            if n_response_tokens == 0:
                print(f"  WARN: No response tokens found for {sample['id']}, using full loss")
                labels = input_ids.clone()

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)

            # Collect video tensors
            if "pixel_values_videos" in inputs:
                all_pixel_values_videos.append(inputs["pixel_values_videos"])
            if "video_grid_thw" in inputs:
                all_video_grid_thw.append(inputs["video_grid_thw"])

        except Exception as e:
            print(f"  ERROR processing {sample.get('id', '?')}: {e}")
            continue

    if not all_input_ids:
        return None

    # Pad to same length within batch
    max_len = max(ids.shape[0] for ids in all_input_ids)
    pad_id = processor.tokenizer.pad_token_id or 0

    padded_ids = torch.full((len(all_input_ids), max_len), pad_id, dtype=torch.long)
    padded_masks = torch.zeros(len(all_input_ids), max_len, dtype=torch.long)
    padded_labels = torch.full((len(all_input_ids), max_len), -100, dtype=torch.long)

    for i, (ids, mask, lbls) in enumerate(zip(all_input_ids, all_attention_masks, all_labels)):
        padded_ids[i, :ids.shape[0]] = ids
        padded_masks[i, :mask.shape[0]] = mask
        padded_labels[i, :lbls.shape[0]] = lbls

    batch = {
        "input_ids": padded_ids.to(device),
        "attention_mask": padded_masks.to(device),
        "labels": padded_labels.to(device),
    }

    # Add video tensors if present
    if all_pixel_values_videos:
        try:
            batch["pixel_values_videos"] = torch.cat(all_pixel_values_videos, dim=0).to(device)
        except Exception:
            # If shapes differ, use the first one only (batch_size=1 typical)
            batch["pixel_values_videos"] = all_pixel_values_videos[0].to(device)

    if all_video_grid_thw:
        try:
            batch["video_grid_thw"] = torch.cat(all_video_grid_thw, dim=0).to(device)
        except Exception:
            batch["video_grid_thw"] = all_video_grid_thw[0].to(device)

    return batch


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("Cosmos Reason 2 LoRA Fine-Tuning")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, nframes: {args.nframes}")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    device = torch.device("cuda")
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)}, Free VRAM: {free_gb:.1f}GB\n")

    # Load data
    train_path = data_dir / "sft_train.json"
    val_path = data_dir / "sft_val.json"
    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        print("Run: python scripts/prepare_sft_data.py first")
        return

    train_data = load_dataset(train_path)
    val_data = load_dataset(val_path) if val_path.exists() else []
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}\n")

    # Import ML libraries
    import transformers
    from peft import LoraConfig, TaskType, get_peft_model

    # Load model and processor
    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()

    hf_token = __import__("os").environ.get("HF_TOKEN", None)
    processor = transformers.AutoProcessor.from_pretrained(MODEL_NAME, token=hf_token)

    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        token=hf_token,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"VRAM after load: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")

    # Enable gradient checkpointing BEFORE applying LoRA (saves ~30% activation memory)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Apply LoRA
    print(f"\nApplying LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    # Required for gradient flow with gradient checkpointing + PEFT
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Get special token IDs for label masking
    tokenizer = processor.tokenizer
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_token_ids = tokenizer.encode("assistant\n", add_special_tokens=False)
    print(f"im_start_id={im_start_id}, assistant_token_ids={assistant_token_ids}")

    # Optimizer (only LoRA parameters need updating)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    # LR scheduler (cosine decay)
    total_steps = (len(train_data) * args.epochs + GRAD_ACCUM - 1) // GRAD_ACCUM
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )

    print(f"\nTotal optimizer steps: {total_steps} ({args.epochs} epochs)\n")

    # Training loop
    best_val_loss = float("inf")
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch = time.time()

        # Shuffle training data
        import random
        rng = random.Random(epoch * 42)
        shuffled = list(train_data)
        rng.shuffle(shuffled)

        for step, sample in enumerate(shuffled):
            batch = prepare_batch(
                [sample],
                processor,
                nframes=args.nframes,
                max_seq_len=MAX_SEQ_LEN,
                device=device,
                im_start_id=im_start_id,
                assistant_token_ids=assistant_token_ids,
            )
            if batch is None:
                continue

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [E{epoch} S{step}] NaN/Inf loss, skipping")
                optimizer.zero_grad()
                continue

            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM
            epoch_steps += 1

            # Gradient accumulation step
            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(shuffled):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = epoch_loss / max(epoch_steps, 1)
                lr_now = scheduler.get_last_lr()[0]
                vram = torch.cuda.memory_allocated() / 1024**3
                print(
                    f"  [E{epoch} step {global_step:4d}] "
                    f"loss={avg_loss:.4f} lr={lr_now:.2e} "
                    f"VRAM={vram:.1f}GB"
                )

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        epoch_time = time.time() - t_epoch
        print(f"\nEpoch {epoch} done: avg_loss={avg_epoch_loss:.4f}, time={epoch_time:.0f}s")

        # Validation
        if val_data:
            val_loss = evaluate_loss(
                model, processor, val_data, args.nframes,
                MAX_SEQ_LEN, device, im_start_id, assistant_token_ids,
            )
            print(f"Epoch {epoch} val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = output_dir / "best_checkpoint"
                model.save_pretrained(str(ckpt_path))
                processor.save_pretrained(str(ckpt_path))
                print(f"  -> Saved best checkpoint (val_loss={val_loss:.4f})")

        # Save epoch checkpoint
        ckpt_path = output_dir / f"epoch_{epoch}"
        model.save_pretrained(str(ckpt_path))
        processor.save_pretrained(str(ckpt_path))
        print(f"Saved epoch {epoch} checkpoint -> {ckpt_path}\n")

    # Save final
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))
    print(f"\nTraining complete! Final model -> {final_path}")
    print(f"Best val_loss: {best_val_loss:.4f}")

    # Save training summary
    summary = {
        "model": MODEL_NAME,
        "epochs": args.epochs,
        "lr": args.lr,
        "nframes": args.nframes,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "target_modules": LORA_TARGET_MODULES,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "total_steps": global_step,
        "best_val_loss": best_val_loss,
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_loss(
    model, processor, val_data: list[dict],
    nframes: int, max_seq_len: int, device: torch.device,
    im_start_id: int, assistant_token_ids: list[int],
) -> float:
    """Compute average loss on validation set."""
    model.eval()
    total_loss = 0.0
    n = 0

    for sample in val_data:
        batch = prepare_batch(
            [sample], processor, nframes, max_seq_len,
            device, im_start_id, assistant_token_ids,
        )
        if batch is None:
            continue
        outputs = model(**batch)
        if not torch.isnan(outputs.loss):
            total_loss += outputs.loss.item()
            n += 1

    model.train()
    return total_loss / max(n, 1)


if __name__ == "__main__":
    main()
