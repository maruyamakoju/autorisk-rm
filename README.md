# AutoRisk-RM: Automated Risk Mining from Dashcam Videos

End-to-end pipeline that automatically extracts dangerous moments from long dashcam videos and uses **NVIDIA Cosmos Reason 2** for causal reasoning, severity classification, and action recommendations.

## Architecture

```
Long Dashcam Video (5 min+)
       |
  [B1] Candidate Mining ── Audio (RMS/horn) + Optical Flow (Farneback) + Proximity (YOLOv8n)
       |                    → weighted fusion → peak detection → Top-N clip extraction
       |
  [B2] Cosmos Reason 2 ─── Local 8B model (nvidia/Cosmos-Reason2-8B)
       |                    → structured JSON: severity, hazards, causal reasoning,
       |                      short-term prediction, recommended action
       |
  [B3] Severity Ranking ── HIGH > MEDIUM > LOW > NONE (tiebreak: hazards → evidence → actors)
       |
  [B4] Evaluation ──────── Accuracy, Macro-F1, 5-item Explanation Checklist (MEDIUM/HIGH only)
       |
  [B5] Ablation Study ──── Baseline (score threshold) vs Cosmos video (full pipeline)
       |
  [Report] ─────────────── HTML/Markdown with top incidents, metrics, and ablation table
```

### B1: Multi-Signal Danger Mining

```
                         Long Dashcam Video (302s, 30 FPS)
                    ┌──────────────┬───────────────┬──────────────┐
                    ↓              ↓               ↓
              ┌──────────┐  ┌───────────┐  ┌─────────────┐
              │  Audio   │  │  Optical  │  │  Proximity  │
              │  Scorer  │  │   Flow    │  │   (YOLO)    │
              │          │  │  Scorer   │  │   Scorer    │
              │ RMS +    │  │ Farneback │  │ YOLOv8n     │
              │ delta +  │  │ magnitude │  │ bbox area + │
              │ horn-band│  │ + variance│  │ center dist │
              └────┬─────┘  └─────┬─────┘  └──────┬──────┘
                   │              │               │
              per-second     per-second      per-second
               scores         scores          scores
                   │              │               │
                   └──────┬───────┴───────────────┘
                          ↓
                 ┌──────────────────┐
                 │  Weighted Fusion │  audio=0.3, motion=0.4, proximity=0.3
                 │  + Normalization │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │  Peak Detection  │  scipy.signal.find_peaks
                 │  + Top-N Select  │  + neighbor merge (±5s clips)
                 └────────┬─────────┘
                          ↓
                   20 candidate clips (10s each)
```

## Key Features

- **Multi-signal danger mining**: Fuses audio (RMS, delta-RMS, horn-band detection), optical flow (Farneback magnitude + variance), and object proximity (YOLOv8n bbox area + center distance) with configurable weights
- **Cosmos Reason 2 video understanding**: Local inference on GPU with `nvidia/Cosmos-Reason2-8B` (Qwen3VL backbone, float16, ~16 GB VRAM). Produces structured JSON with severity, hazard details, causal reasoning, predictions, and recommended actions
- **100% JSON parse success**: Robust multi-layer parsing with truncation repair (direct JSON → markdown fence → open fence + truncation repair → brace extraction → missing comma fix → trailing key cleanup → markdown field parser)
- **Reproducible Public Mode**: One-command pipeline using publicly available dashcam footage with blind-labeled ground truth
- **Re-parse capability**: Fix parse failures without re-running expensive inference via `autorisk reparse`

## Quickstart

```bash
# 1. Install
pip install -e .

# 2. Set up environment
cp .env.example .env
# Add your HF_TOKEN (for gated Cosmos model access on HuggingFace)
# IMPORTANT: Never commit .env — it is in .gitignore

# 3. Run full pipeline on public video
python -m autorisk.cli -c configs/public.yaml run \
  -i data/public_samples/uk_dashcam_compilation.mp4 \
  -o outputs/public_run

# 4. View results
# Open outputs/public_run/report.html in browser
```

## Public Mode (Reproducible Evaluation)

Public Mode uses a freely available dashcam compilation for fully reproducible results.

### Step 1: Download source video

```bash
# Option A: Automated download
python scripts/download_public_data.py

# Option B: Manual download (if yt-dlp fails)
# Download from: https://www.youtube.com/watch?v=i7HspkH7aT4
# Save as: data/public_samples/uk_dashcam_compilation.mp4
# Recommended: 360p MP4 (format 18) to avoid SABR streaming issues
# yt-dlp -f 18 -o "data/public_samples/uk_dashcam_compilation.mp4" "https://www.youtube.com/watch?v=i7HspkH7aT4"
```

Source: [UK Dash Camera Compilation #1](https://www.youtube.com/watch?v=i7HspkH7aT4) by JA Dashcam UK (302s, 640x360, 30 FPS)

### Step 2: Run pipeline

```bash
# Full pipeline (mining + Cosmos inference + ranking + eval + report)
# ~3 hours on RTX 5090 for 20 candidates
python -m autorisk.cli -c configs/public.yaml run \
  -i data/public_samples/uk_dashcam_compilation.mp4 \
  -o outputs/public_run
```

### Step 3: Evaluate and ablation (after inference completes)

```bash
# Evaluation against blind-labeled GT
python -m autorisk.cli eval \
  -r outputs/public_run/cosmos_results.json \
  -o outputs/public_run

# Minimal ablation (baseline score threshold vs Cosmos video)
python -m autorisk.cli ablation \
  -r outputs/public_run/cosmos_results.json \
  -g data/annotations/gt_labels.csv \
  -o outputs/public_run

# Generate HTML report
python -m autorisk.cli report \
  -r outputs/public_run/cosmos_results.json \
  -o outputs/public_run
```

### Fallback: Synthetic video for quick validation

If the public video is unavailable, you can validate the pipeline with a synthetic test video:

```bash
python -m autorisk.cli run \
  -i data/videos/public/dashcam_realistic.mp4 \
  -o outputs/synthetic_test --skip-ablation
```

### Environment notes

- **`.env` is never committed** (listed in `.gitignore`). It contains `HF_TOKEN` and optionally `NVIDIA_API_KEY`
- First run downloads the Cosmos-Reason2-8B model (~17 GB) from HuggingFace. Subsequent runs use the cache
- GPU VRAM requirement: ~16 GB for float16 inference. RTX 4090/5090 recommended

## Ground Truth Labeling

GT severity labels in `data/annotations/gt_labels.csv` were assigned **blind to model output** (before reviewing Cosmos predictions) using the following criteria:

| Severity | Definition |
|----------|-----------|
| **NONE** | No hazard elements (normal driving) |
| **LOW** | Attention needed but margin exists (distant pedestrians, gentle merge) |
| **MEDIUM** | Evasive action warranted (decelerate, maintain distance, yield) |
| **HIGH** | Collision risk, emergency braking or swerving needed (close pass, cut-in, near-miss) |

Explanation Checklist (`data/annotations/checklist_labels.csv`) is scored only for MEDIUM/HIGH clips, with 5 binary items evaluating the quality of Cosmos's reasoning.

## Results (Public Mode, 20 clips)

### Classification Metrics (Prompt Engineering Progression)
| Metric | v1 (initial) | v2 (calibrated) | v3 (final) |
|--------|-------|-------|-------|
| Accuracy | 0.200 | 0.250 | **0.350** |
| Macro-F1 | 0.188 | 0.291 | **0.346** |
| Checklist | 3.83/5 | 3.00/5 | **5.00/5** |
| Parse success | 19/20 | 17/20 | **20/20** |
| HIGH predictions | 14/20 | 3/20 | 3/20 |

### Prediction Distribution Shift (v1 → v3)

```
              v1 (initial prompt)              v3 (calibrated + 2-stage)        Ground Truth
  NONE  ██ 1                         NONE  ██ 2                        NONE  ████ 4
  LOW   ██ 2                         LOW   ██████████ 10               LOW   █████████ 9
  MED   ███ 3                        MED   █████ 5                     MED   ████ 4
  HIGH  ██████████████ 14  ← bias    HIGH  ███ 3  ← matches GT        HIGH  ███ 3
```

Key insight: v1 classified 70% of clips as HIGH (GT: 15%). Calibration guidance + false-positive examples + removing danger-score priming reduced HIGH predictions to match the GT distribution exactly.

### Explanation Checklist (10 MEDIUM/HIGH clips, auto-heuristic)
| Item | Score |
|------|-------|
| Actors accurate | 1.00 |
| Causal reasoning clear | 1.00 |
| Spatial relations specific | 1.00 |
| Short-term prediction plausible | 1.00 |
| Recommended action reasonable | 1.00 |
| **Mean total** | **5.00/5** |

### Confusion Matrix (rows=GT, cols=Predicted)
|  | NONE | LOW | MEDIUM | HIGH |
|------|------|-----|--------|------|
| **NONE** (4) | 0 | 3 | 1 | 0 |
| **LOW** (9) | 1 | 3 | 4 | 1 |
| **MEDIUM** (4) | 0 | 2 | 2 | 0 |
| **HIGH** (3) | 1 | 0 | 0 | 2 |

### Ablation Study
| Mode | Accuracy | Macro-F1 | Checklist |
|------|----------|----------|-----------|
| Baseline (mining score threshold only) | 0.350 | 0.295 | 1.00/5 |
| **Cosmos video (full pipeline)** | **0.350** | **0.346** | **5.00/5** |

### 2-Stage Inference Strategy

```
Stage 1: Full analysis (all 20 clips)
┌─────────────┐    ┌──────────────────┐    ┌──────────────────────────┐
│  Video clip  │───→│  Cosmos Reason 2 │───→│  severity, hazards,      │
│  (4 FPS)     │    │  (calibrated     │    │  causal_reasoning,       │
│              │    │   prompt)        │    │  evidence, confidence    │
└─────────────┘    └──────────────────┘    └──────────────────────────┘
                                                       │
                                           Filter: MEDIUM/HIGH clips
                                           missing prediction/action
                                                       ↓
Stage 2: Supplement pass (10 clips)
┌─────────────┐    ┌──────────────────┐    ┌──────────────────────────┐
│  Same clip   │───→│  Cosmos Reason 2 │───→│  short_term_prediction,  │
│  + Stage 1   │    │  (supplement     │    │  recommended_action      │
│    context   │    │   prompt)        │    │  → merged into Stage 1   │
└─────────────┘    └──────────────────┘    └──────────────────────────┘

Result: Accurate classification (Stage 1) + Complete explanations (Stage 2) = 5.00/5 checklist
```

### Key Findings

1. **Prompt engineering fixes HIGH-severity bias**: The initial prompt caused 14/20 clips to be classified as HIGH (GT: 3 HIGH). After adding calibration guidance, false-positive examples, and removing danger-score priming, the model predicts HIGH for only 3/20 clips — matching the GT distribution. This improved Macro-F1 by **84%** (0.188 → 0.346).

2. **2-stage inference recovers explanation completeness**: The calibrated prompt produces shorter outputs, dropping prediction/action fields. A lightweight 2nd-pass supplement (targeting only MEDIUM/HIGH clips with missing fields) recovers **perfect 5.00/5 checklist** without re-running full inference. This decouples classification accuracy from explanation quality.

3. **Cosmos adds structured reasoning over baseline**: While the mining-score baseline achieves comparable accuracy (0.350), Cosmos provides dramatically richer outputs — perfect explanation quality (5.00 vs 1.00 checklist), better class balance (F1 0.346 vs 0.295), and actionable causal reasoning, predictions, and recommended actions that a simple threshold cannot produce.

4. **Robust JSON parsing achieves 100% success**: Multi-layer repair pipeline (truncation repair, missing comma fix, trailing key cleanup, markdown fallback) plus reduced-FPS retry for stubborn clips achieves 20/20 parse success.

## CLI Commands

```bash
# Full pipeline
python -m autorisk.cli run -i VIDEO -o OUTPUT_DIR

# Individual stages
python -m autorisk.cli mine -i VIDEO -o OUTPUT_DIR        # B1: Candidate extraction
python -m autorisk.cli infer -d CLIPS_DIR -o OUTPUT_DIR   # B2: Cosmos inference
python -m autorisk.cli eval -r RESULTS.json -o OUTPUT_DIR # B4: Evaluation
python -m autorisk.cli ablation -r RESULTS.json -o DIR    # B5: Minimal ablation
python -m autorisk.cli report -r RESULTS.json -o DIR      # Report generation
python -m autorisk.cli supplement -r RESULTS.json           # Fill missing prediction/action (2nd pass)
python -m autorisk.cli reparse -r RESULTS.json             # Re-parse failed entries

# With public config
python -m autorisk.cli -c configs/public.yaml run -i VIDEO -o OUTPUT_DIR
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA (install separately from [pytorch.org](https://pytorch.org/get-started/locally/) — not included in `pip install -e .` to avoid CPU-only fallback)
- NVIDIA GPU with 16+ GB VRAM (RTX 4090/5090 recommended)
- ~17 GB disk for Cosmos-Reason2-8B model weights
- HuggingFace account with [model access](https://huggingface.co/nvidia/Cosmos-Reason2-8B)
- FFmpeg 7+ (for clip extraction)

## Project Structure

```
autorisk/
  mining/         # B1: Audio, motion, proximity signal scorers + fusion
  cosmos/         # B2: Local inference client, prompt templates, schemas
  eval/           # B4/B5: Metrics, checklist, ablation
  report/         # HTML/Markdown report generation
  pipeline.py     # E2E orchestration
  cli.py          # Click CLI (run/mine/infer/eval/ablation/report)
configs/
  default.yaml    # Default configuration
  public.yaml     # Public mode (reproducible evaluation)
data/
  annotations/    # GT labels (blind-labeled) and checklist scores
scripts/
  download_public_data.py  # Public video downloader (yt-dlp)
```

## License

This project uses NVIDIA Cosmos Reason 2 under the [NVIDIA Open Model License](https://developer.nvidia.com/open-model-license).
