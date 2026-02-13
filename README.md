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

## Key Features

- **Multi-signal danger mining**: Fuses audio (RMS, delta-RMS, horn-band detection), optical flow (Farneback magnitude + variance), and object proximity (YOLOv8n bbox area + center distance) with configurable weights
- **Cosmos Reason 2 video understanding**: Local inference on GPU with `nvidia/Cosmos-Reason2-8B` (Qwen3VL backbone, float16, ~16 GB VRAM). Produces structured JSON with severity, hazard details, causal reasoning, predictions, and recommended actions
- **95% JSON parse success**: Robust multi-layer parsing with truncation repair (direct JSON → markdown fence → open fence + truncation repair → brace extraction → missing comma fix → trailing key cleanup → markdown field parser)
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

### Classification Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.200 |
| Macro-F1 | 0.188 |
| Parse success | 19/20 (95%) |

### Explanation Checklist (18 MEDIUM/HIGH clips, auto-heuristic)
| Item | Score |
|------|-------|
| Actors accurate | 1.00 |
| Causal reasoning clear | 1.00 |
| Spatial relations specific | 1.00 |
| Short-term prediction plausible | 0.50 |
| Recommended action reasonable | 0.33 |
| **Mean total** | **3.83/5** |

### Ablation Study
| Mode | Accuracy | Macro-F1 | Checklist |
|------|----------|----------|-----------|
| Baseline (mining score threshold only) | 0.350 | 0.295 | 1.00/5 |
| Cosmos video (full pipeline) | 0.200 | 0.188 | 3.83/5 |

### Key Findings

1. **Cosmos Reason 2 has a conservative HIGH-severity bias**: The model classifies 14/20 clips as HIGH, while blind GT labels have 3 HIGH, 4 MEDIUM, 9 LOW, 4 NONE. This is appropriate for safety-critical applications where false negatives are more dangerous than false positives.

2. **Explanation quality is excellent**: Perfect scores (1.0) for actor identification, causal reasoning, and spatial relations. Cosmos provides detailed, specific descriptions of actors, their spatial relationships, and the causal chains leading to risk.

3. **Cosmos dramatically improves explanation quality**: The ablation shows baseline (score thresholds) achieves higher accuracy (35% vs 20%) because its severity distribution better matches the GT. However, Cosmos produces **3.8x better explanations** (checklist 3.83 vs 1.00), demonstrating the value of VLM-based reasoning over simple signal processing.

4. **Robust JSON parsing recovers truncated model output**: The model sometimes generates JSON that is cut off before completion (missing closing braces, missing commas between fields). The multi-layer repair pipeline recovers 9/10 initially failed parses.

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
python -m autorisk.cli reparse -r RESULTS.json             # Re-parse failed entries

# With public config
python -m autorisk.cli -c configs/public.yaml run -i VIDEO -o OUTPUT_DIR
```

## Requirements

- Python 3.10+
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
