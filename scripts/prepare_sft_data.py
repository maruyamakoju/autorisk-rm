"""Prepare SFT dataset from AutoRisk-RM GT-labeled dashcam clips.

Converts our 20 blind-labeled clips + cosmos_results.json into
LLaVA JSON format for Cosmos Reason 2 LoRA fine-tuning.

Generates 3 question types per clip:
  1. Severity MCQ (4-choice: NONE/LOW/MEDIUM/HIGH)
  2. HIGH detection binary (requires immediate action?)
  3. Evasive action required binary (MEDIUM or HIGH?)

Output: data/sft/sft_train.json, data/sft/sft_val.json

Usage: python scripts/prepare_sft_data.py
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

# Reproducible split
RANDOM_SEED = 42
VAL_FRACTION = 0.25  # 5 clips val, 15 clips train

SEVERITY_DESCRIPTIONS = {
    "NONE": "NONE - Normal driving, no hazard detected",
    "LOW": "LOW - Attention needed but safe margin exists",
    "MEDIUM": "MEDIUM - Evasive action warranted (decelerate or yield)",
    "HIGH": "HIGH - Collision risk, immediate emergency response needed",
}

SEVERITY_TO_LETTER = {"NONE": "A", "LOW": "B", "MEDIUM": "C", "HIGH": "D"}


def make_severity_mcq(video_path: str, severity: str, entry_id: str) -> dict:
    """Create a 4-choice severity classification question."""
    question = (
        "Watch this dashcam clip carefully. What is the danger severity level?\n"
        f"A: {SEVERITY_DESCRIPTIONS['NONE']}\n"
        f"B: {SEVERITY_DESCRIPTIONS['LOW']}\n"
        f"C: {SEVERITY_DESCRIPTIONS['MEDIUM']}\n"
        f"D: {SEVERITY_DESCRIPTIONS['HIGH']}\n"
        "Answer with the option's letter from the given choices directly."
    )
    answer = SEVERITY_TO_LETTER[severity]
    return {
        "id": f"{entry_id}_severity_mcq",
        "video": video_path,
        "type": "severity_mcq",
        "gt_severity": severity,
        "conversations": [
            {"from": "human", "value": f"<video>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }


def make_high_detection(video_path: str, severity: str, entry_id: str) -> dict:
    """Create a binary HIGH severity detection question."""
    question = (
        "Watch this dashcam clip. Does it show a HIGH severity danger "
        "requiring immediate emergency action from the driver?\n"
        "A: Yes\n"
        "B: No\n"
        "Answer with the option's letter from the given choices directly."
    )
    answer = "A" if severity == "HIGH" else "B"
    return {
        "id": f"{entry_id}_high_detection",
        "video": video_path,
        "type": "high_detection",
        "gt_severity": severity,
        "conversations": [
            {"from": "human", "value": f"<video>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }


def make_evasive_action(video_path: str, severity: str, entry_id: str) -> dict:
    """Create a binary evasive action required question."""
    question = (
        "Watch this dashcam clip. Does the driver need to take evasive action "
        "(decelerate, brake hard, or change lane) to avoid a potential hazard?\n"
        "A: Yes\n"
        "B: No\n"
        "Answer with the option's letter from the given choices directly."
    )
    answer = "A" if severity in ("MEDIUM", "HIGH") else "B"
    return {
        "id": f"{entry_id}_evasive_action",
        "video": video_path,
        "type": "evasive_action",
        "gt_severity": severity,
        "conversations": [
            {"from": "human", "value": f"<video>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }


def main():
    project_root = Path(__file__).parent.parent
    gt_path = project_root / "data" / "annotations" / "gt_labels.csv"
    clips_base = project_root / "outputs" / "public_run" / "clips"
    output_dir = project_root / "data" / "sft"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load GT labels
    gt_labels: list[tuple[str, str]] = []
    with open(gt_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clip_name = Path(row["clip_path"]).name
            clip_path = clips_base / clip_name
            severity = row["severity"].strip().upper()
            if clip_path.exists() and severity in SEVERITY_TO_LETTER:
                gt_labels.append((str(clip_path), severity, clip_path.stem))
            else:
                print(f"  SKIP: {clip_name} (exists={clip_path.exists()}, sev={severity})")

    print(f"Loaded {len(gt_labels)} GT-labeled clips")

    # Show distribution
    from collections import Counter
    dist = Counter(sev for _, sev, _ in gt_labels)
    print(f"Distribution: {dict(dist)}")

    # Build all samples
    all_samples: list[dict] = []
    for video_path, severity, stem in gt_labels:
        all_samples.append(make_severity_mcq(video_path, severity, stem))
        all_samples.append(make_high_detection(video_path, severity, stem))
        all_samples.append(make_evasive_action(video_path, severity, stem))

    print(f"Total samples generated: {len(all_samples)}")

    # Train/val split at clip level (not sample level, to avoid data leakage)
    rng = random.Random(RANDOM_SEED)
    clips = list({(vp, sev, stem) for vp, sev, stem in gt_labels})
    rng.shuffle(clips)
    n_val = max(1, int(len(clips) * VAL_FRACTION))
    val_clips = {stem for _, _, stem in clips[:n_val]}
    train_clips = {stem for _, _, stem in clips[n_val:]}

    train_samples = [s for s in all_samples if any(stem in s["id"] for stem in train_clips)]
    val_samples = [s for s in all_samples if any(stem in s["id"] for stem in val_clips)]

    print(f"Train: {len(train_samples)} samples ({len(train_clips)} clips)")
    print(f"Val:   {len(val_samples)} samples ({len(val_clips)} clips)")

    # Show val clips
    print(f"Val clip stems: {sorted(val_clips)}")

    # Save
    train_path = output_dir / "sft_train.json"
    val_path = output_dir / "sft_val.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to:")
    print(f"  {train_path} ({len(train_samples)} samples)")
    print(f"  {val_path}   ({len(val_samples)} samples)")

    # Also save a combined file for easy loading
    all_path = output_dir / "sft_all.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    print(f"  {all_path}   ({len(all_samples)} samples, train+val combined)")


if __name__ == "__main__":
    main()
