# AutoRisk-RM: Automated Risk Mining from Dashcam Videos

End-to-end pipeline that automatically extracts dangerous moments from long dashcam videos and uses **NVIDIA Cosmos Reason 2** for causal reasoning, severity classification, and action recommendations.

## Architecture

```
Long Dashcam Video
       |
  [B1] Candidate Mining (Audio + Optical Flow + YOLO Proximity)
       |
  [B2] Cosmos Reason 2 Inference (Local 8B model, video understanding)
       |
  [B3] Severity Ranking (HIGH > MEDIUM > LOW > NONE)
       |
  [B4] Evaluation (Accuracy, Macro-F1, 5-item Explanation Checklist)
       |
  [B5] Ablation Study (Mining-only vs Cosmos-1frame vs Cosmos-video)
       |
  [Report] HTML/Markdown with top incidents, metrics, and analysis
```

## Key Features

- **Multi-signal danger mining**: Fuses audio (RMS, horn detection), optical flow (Farneback), and object proximity (YOLOv8n) to find candidate dangerous moments
- **Cosmos Reason 2 video understanding**: Local inference with `nvidia/Cosmos-Reason2-8B` for structured JSON output including severity, hazards, causal reasoning, predictions, and recommended actions
- **100% JSON parse success**: Clean structured output from Cosmos, no markdown fallback needed
- **Reproducible Public Mode**: One-command pipeline using publicly available dashcam footage

## Quickstart

```bash
# 1. Install
pip install -e .

# 2. Set up environment
cp .env.example .env
# Add your HF_TOKEN (for Cosmos model access)

# 3. Run Public Mode (reproducible evaluation)
python -m autorisk.cli -c configs/public.yaml run \
  -i data/public_samples/uk_dashcam_compilation.mp4 \
  -o outputs/public_run --skip-ablation

# 4. View results
# Open outputs/public_run/report.html in browser
```

## Public Mode

For reproducible evaluation, download the public dashcam video:

```bash
# Download source video (UK Dash Camera Compilation #1)
python scripts/download_public_data.py

# Run full pipeline
python -m autorisk.cli -c configs/public.yaml run \
  -i data/public_samples/uk_dashcam_compilation.mp4 \
  -o outputs/public_run
```

Source: [UK Dash Camera Compilation #1](https://www.youtube.com/watch?v=i7HspkH7aT4) by JA Dashcam UK

## Evaluation

### Classification Metrics
| Metric | Value |
|--------|-------|
| Accuracy | TBD |
| Macro-F1 | TBD |

### Explanation Checklist (MEDIUM/HIGH clips only)
| Item | Score |
|------|-------|
| Actors accurate | TBD |
| Causal reasoning clear | TBD |
| Spatial relations specific | TBD |
| Short-term prediction plausible | TBD |
| Recommended action reasonable | TBD |
| **Mean total** | **TBD/5** |

### Ablation Study
| Mode | Accuracy | Macro-F1 | Checklist |
|------|----------|----------|-----------|
| Mining score only (no Cosmos) | TBD | TBD | N/A |
| Cosmos 1-frame | TBD | TBD | TBD |
| Cosmos video (full) | TBD | TBD | TBD |

## CLI Commands

```bash
# Full pipeline
autorisk run -i VIDEO -o OUTPUT_DIR

# Individual stages
autorisk mine -i VIDEO -o OUTPUT_DIR      # B1: Candidate extraction
autorisk infer -d CLIPS_DIR -o OUTPUT_DIR  # B2: Cosmos inference
autorisk eval -r RESULTS.json -o OUTPUT_DIR # B4: Evaluation
autorisk report -r RESULTS.json -o OUTPUT_DIR # Report generation
```

## Requirements

- Python 3.10+
- NVIDIA GPU with 16+ GB VRAM (RTX 4090/5090 recommended)
- ~17 GB disk for Cosmos-Reason2-8B model weights
- HuggingFace account with [model access](https://huggingface.co/nvidia/Cosmos-Reason2-8B)

## Project Structure

```
autorisk/
  mining/         # B1: Audio, motion, proximity signal scorers + fusion
  cosmos/         # B2: Local inference client, prompt templates, schemas
  eval/           # B4/B5: Metrics, checklist, ablation
  report/         # HTML/Markdown report generation
  pipeline.py     # E2E orchestration
  cli.py          # Click CLI
configs/
  default.yaml    # Default configuration
  public.yaml     # Public mode (reproducible evaluation)
data/
  annotations/    # GT labels and checklist scores
```

## License

This project uses NVIDIA Cosmos Reason 2 under the [NVIDIA Open Model License](https://developer.nvidia.com/open-model-license).
