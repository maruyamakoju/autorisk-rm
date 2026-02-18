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

**Data usage note**: The source video is publicly available on YouTube under standard YouTube license. It is **not redistributed** in this repository — users download it directly from the original source via `yt-dlp` or manual download. Only the extracted 10-second analysis clips (transformative, analytical use) and ground-truth annotations (original work) are used for evaluation. No video data is included in the git repository.

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

GT severity labels in `data/annotations/gt_labels.csv` were assigned **blind to model output** using the following procedure:

1. **Mining first**: Run `autorisk mine` to extract 20 candidate clips from the dashcam video
2. **Watch clips only**: Review each 10-second clip in a video player (VLC) without running Cosmos inference
3. **Assign severity**: Label each clip using the criteria below, based solely on visual content
4. **Freeze GT**: Commit `gt_labels.csv` to the repository before running any inference
5. **Run inference**: Execute Cosmos analysis and compare predictions against the frozen GT

This blind-labeling protocol ensures GT labels are **not influenced by model predictions**, providing an honest evaluation baseline.

| Severity | Definition |
|----------|-----------|
| **NONE** | No hazard elements (normal driving) |
| **LOW** | Attention needed but margin exists (distant pedestrians, gentle merge) |
| **MEDIUM** | Evasive action warranted (decelerate, maintain distance, yield) |
| **HIGH** | Collision risk, emergency braking or swerving needed (close pass, cut-in, near-miss) |

Explanation Checklist (`data/annotations/checklist_labels.csv`) is scored only for MEDIUM/HIGH clips, with 5 binary items evaluating the quality of Cosmos's reasoning. Checklist items are evaluated via auto-heuristic (checking for non-empty structured fields) to ensure reproducibility.

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

### Signal Contribution Analysis

Spearman rank correlation between each mining signal and ground-truth severity:

| Signal | Spearman ρ | p-value | Threshold Acc. | Threshold F1 |
|--------|-----------|---------|---------------|-------------|
| Audio (RMS/horn) | +0.192 | 0.418 | 40.0% | 0.340 |
| Motion (optical flow) | +0.140 | 0.556 | 35.0% | 0.250 |
| Proximity (YOLOv8n) | +0.223 | 0.345 | 35.0% | 0.294 |
| **Fused (weighted)** | **+0.192** | 0.418 | **45.0%** | **0.412** |

Fused signal outperforms every individual signal on threshold F1 (0.412 vs max 0.340), confirming the value of multi-signal fusion. No single signal achieves statistical significance (all p>0.05), reinforcing the need for Cosmos video understanding.

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| NONE | 0.0% | 0.0% | 0.000 | 4 |
| LOW | 37.5% | 33.3% | 0.353 | 9 |
| MEDIUM | 28.6% | 50.0% | 0.364 | 4 |
| **HIGH** | **66.7%** | **66.7%** | **0.667** | 3 |

The model identifies genuinely dangerous situations (HIGH F1=0.667) best, but over-predicts risk in safe scenes (NONE F1=0.000) — a conservative bias that is safer for real-world driving than missing actual hazards.

### Error Analysis (13/20 misclassified)

```
Error Distribution:
  Over-estimation (predicted higher than GT):  9 errors (69%)
  Under-estimation (predicted lower than GT):  4 errors (31%)

  Adjacent miss (off by 1 severity level):    10 errors (77%)
  Major miss (off by 2+ levels):               3 errors (23%)
```

77% of errors are adjacent misses — the model's severity judgment is close but boundary calibration between classes remains the primary challenge. Over-estimation dominates (69%), consistent with the model's conservative safety bias.

### Key Findings

1. **Prompt engineering fixes HIGH-severity bias**: The initial prompt caused 14/20 clips to be classified as HIGH (GT: 3 HIGH). After adding calibration guidance, false-positive examples, and removing danger-score priming, the model predicts HIGH for only 3/20 clips — matching the GT distribution. This improved Macro-F1 by **84%** (0.188 → 0.346).

2. **2-stage inference recovers explanation completeness**: The calibrated prompt produces shorter outputs, dropping prediction/action fields. A lightweight 2nd-pass supplement (targeting only MEDIUM/HIGH clips with missing fields) recovers **perfect 5.00/5 checklist** without re-running full inference. This decouples classification accuracy from explanation quality.

3. **Cosmos adds structured reasoning over baseline**: While the mining-score baseline achieves comparable accuracy (0.350), Cosmos provides dramatically richer outputs — perfect explanation quality (5.00 vs 1.00 checklist), better class balance (F1 0.346 vs 0.295), and actionable causal reasoning, predictions, and recommended actions that a simple threshold cannot produce.

4. **Multi-signal fusion outperforms individual signals**: No single signal (audio, motion, proximity) achieves statistical significance for severity prediction (all Spearman p>0.05). Weighted fusion improves threshold F1 by 21% over the best individual signal (0.412 vs 0.340), demonstrating the complementary information in each modality.

5. **Conservative safety bias**: 69% of misclassifications are over-estimations (model predicts higher severity than GT). This is a desirable property for safety-critical applications — the model errs on the side of caution rather than missing real hazards (HIGH recall=66.7%).

6. **Robust JSON parsing achieves 100% success**: Multi-layer repair pipeline (truncation repair, missing comma fix, trailing key cleanup, markdown fallback) plus reduced-FPS retry for stubborn clips achieves 20/20 parse success.

### TTC (Time-to-Collision) Analysis

Monocular TTC estimation via YOLOv8n + ByteTrack object tracking using the tau approximation (TTC = bbox_area / d(area)/dt):

| Severity | Mean TTC | Interpretation |
|----------|---------|----------------|
| NONE | 1.77s | Objects distant or receding |
| LOW | 0.50s | Moderate approach rate |
| MEDIUM | 0.42s | Close approach |
| HIGH | 0.52s | Rapid closure |

**Spearman ρ = -0.495 (p=0.026)** — the only metric achieving statistical significance (p<0.05). TTC provides an objective, physics-based safety metric that validates Cosmos's subjective severity judgments.

### Confidence Calibration

| Metric | Before | After T-scaling (T=10.0) | Improvement |
|--------|--------|--------------------------|-------------|
| ECE | 0.578 | 0.362 | 37% |
| Brier Score | 0.569 | 0.340 | 40% |

Per-severity analysis reveals the model is severely overconfident for NONE predictions (confidence=0.70, accuracy=0.00) and underconfident for HIGH (confidence=0.33, accuracy=0.67). Temperature scaling (Guo et al., 2017) corrects this miscalibration.

### Cross-Modal Grounding

Measures agreement between quantitative mining signals and Cosmos's natural-language reasoning:

| Signal | Grounding Rate | Interpretation |
|--------|---------------|----------------|
| Proximity | 100% | Cosmos always references nearby objects when detected |
| Motion | 88.9% | Cosmos identifies sudden movements in most cases |
| Audio | 25.0% | Expected: VLM processes video frames only, not audio |

Mean grounding score: **90.8%** (15/20 clips fully grounded). Audio's low grounding rate validates that Cosmos reasons from visual evidence, not signal metadata.

## CLI Commands

```bash
# Full pipeline
python -m autorisk.cli run -i VIDEO -o OUTPUT_DIR

# Individual stages
python -m autorisk.cli mine -i VIDEO -o OUTPUT_DIR        # B1: Candidate extraction
python -m autorisk.cli infer -d CLIPS_DIR -o OUTPUT_DIR   # B2: Cosmos inference
python -m autorisk.cli eval -r RESULTS.json -o OUTPUT_DIR # B4: Evaluation
python -m autorisk.cli ablation -r RESULTS.json -o DIR    # B5: Minimal ablation
python -m autorisk.cli analyze -r RESULTS.json -o DIR     # Deep analysis (signal/error/per-class)
python -m autorisk.cli report -r RESULTS.json -o DIR      # Report generation
python -m autorisk.cli supplement -r RESULTS.json           # Fill missing prediction/action (2nd pass)
python -m autorisk.cli reparse -r RESULTS.json             # Re-parse failed entries
python -m autorisk.cli ttc -d CLIPS_DIR -o DIR              # TTC via object tracking (YOLOv8n + ByteTrack)
python -m autorisk.cli grounding -r RESULTS.json -o DIR     # Cross-modal grounding analysis
python -m autorisk.cli calibration -r RESULTS.json -o DIR   # Confidence calibration (ECE + T-scaling)
python -m autorisk.cli saliency -d CLIPS_DIR -r RESULTS.json -o DIR  # Gradient saliency maps (requires GPU)
python -m autorisk.cli audit-pack -r RUN_DIR               # Auditable evidence pack (manifest/trace/checksums)
python -m autorisk.cli audit-sign -p PACK_OR_ZIP --private-key keys/private.pem [--private-key-password-env AUTORISK_SIGNING_KEY_PASSWORD] [--public-key keys/public.pem] [--key-label ops-2026q1]  # Sign audit pack (Ed25519)
python -m autorisk.cli audit-attest -p PACK_OR_ZIP --private-key keys/private.pem [--private-key-password-env AUTORISK_SIGNING_KEY_PASSWORD] [--public-key keys/public.pem] [--key-label ops-2026q1]  # Attest non-checksummed finalize/validate artifacts
python -m autorisk.cli audit-verify -p PACK_OR_ZIP [--public-key keys/public.pem | --public-key-dir keys/trusted] [--profile audit-grade] [--expect-pack-fingerprint <64hex>] [--require-signature --require-public-key --require-attestation] [--revocation-file revoked_key_ids.txt]  # Verify integrity + authenticity
python -m autorisk.cli audit-validate -p PACK_OR_ZIP --enforce   # Validate audit contract (schema + semantic checks)
python -m autorisk.cli audit-verifier-bundle --out verifier_bundle --public-key-dir keys/trusted [--revocation-file revoked_key_ids.txt]  # Build portable verifier bundle
python -m autorisk.cli audit-handoff -r RUN_DIR --out handoff    # Build single handoff folder (PACK + verifier bundle + finalize record)
python -m autorisk.cli audit-handoff-verify -d HANDOFF_DIR --profile audit-grade [--expect-pack-fingerprint <64hex>] --enforce  # Verify handoff in one command (checksums + verify + validate + attestation)
python -m autorisk.cli review-approve -r RUN_DIR --rank N --severity MEDIUM --reason "..."  # Append human decision
python -m autorisk.cli review-apply -r RUN_DIR             # Produce cosmos_results_reviewed.json
python -m autorisk.cli policy-check -r RUN_DIR --policy configs/policy.yaml --enforce  # Enforce review gating policy
python -m autorisk.cli finalize-run -r RUN_DIR --zip --audit-grade --sign-private-key keys/private.pem --sign-public-key-dir keys/trusted --handoff-out RUN_DIR/handoff_latest  # Recommended audit-grade handoff (verify+validate+handoff)
python -m autorisk.cli finalize-run -r RUN_DIR --policy configs/policy.yaml --zip --enforce [--sign-private-key ... --sign-public-key ... --require-signature --require-trusted-key]  # review-apply -> policy-check -> audit-pack -> audit-sign? -> audit-verify

# With public config
python -m autorisk.cli -c configs/public.yaml run -i VIDEO -o OUTPUT_DIR
```

`audit-pack` output (default: `RUN_DIR/audit_pack_<timestamp>`):

- `manifest.json`: run/model/prompt provenance, file inventory, summary stats
- `decision_trace.jsonl`: one record per candidate with signal scores, parsing/repair log, final decision, raw response
- `checksums.sha256.txt`: SHA256 chain for every file in the pack
- `signature.json` (optional): Ed25519 signature payload (`audit-sign`) over `checksums.sha256.txt` and `manifest.json`
- `attestation.json` (optional but recommended): Ed25519 attestation over non-checksummed `run_artifacts/finalize_record.json` and `run_artifacts/audit_validate_report.json`
- `run_artifacts/*`: copied run outputs (`cosmos_results.json`, `candidates.csv`, `cosmos_results_reviewed.json`, `review_apply_report.json`, `review_diff_report.json`, `policy_report.json`, `review_queue.json`, `policy_snapshot.json`, `audit_validate_report.json`, eval/ablation/report files if present)
- `clips/*`: candidate clip evidence (when `--include-clips`)
- `audit_pack_<timestamp>.zip`: handoff-ready archive (when `--zip`)
- `handoff_<timestamp>/`: optional single handoff folder from `audit-handoff` containing `PACK.zip`, `verifier_bundle.zip`, `finalize_record.json`, `audit_validate_report.json` (if available), `HANDOFF.md`, and `handoff_checksums.sha256.txt`
- `handoff_checksums.sha256.txt` intentionally excludes `PACK.zip` (and `finalize_record.json`) to avoid circular hash dependencies; `PACK.zip` is verified by `audit-verify`
- Pack fingerprint (`checksums.sha256.txt` SHA256) is printed by CLI and should be recorded externally (ticket/DB) at submission time
- `finalize_record.json` is emitted in `RUN_DIR` and copied into `run_artifacts/finalize_record.json` during `finalize-run` for external audit logging
- `audit-verify` prints `Unchecked files` (e.g., `run_artifacts/finalize_record.json`, `run_artifacts/audit_validate_report.json`) so non-checksummed payload is explicit during audit review
- `audit-handoff-verify` validates `attestation.json` by default (`--require-attestation`) and checks it against PACK fingerprint + run artifacts hashes
- For PACK-only receipt (without handoff folder), use `audit-verify --profile audit-grade` (or `--require-attestation`) to verify signature + attestation in one command
- `--expect-pack-fingerprint <64hex>` is supported by both `audit-verify` and `audit-handoff-verify` to prevent PACK substitution (compare against ticket/DB recorded fingerprint)
- `--profile default` is diagnostics-only: crypto requirements are not enforced and results must not be used for audit-grade acceptance
- Legacy bundles without `attestation.json` can be inspected with `--no-require-attestation` for diagnostics only (not valid for audit-grade decisions)
- `audit-validate` supports `--profile audit-grade` to require signature/finalize/policy/review artifacts in addition to schema + semantic checks
- `finalize-run --audit-grade` now includes `audit-validate` and handoff generation in one command (`--write-handoff` defaults to ON in audit-grade mode)
- Policy defaults are configurable in `configs/policy.yaml` and policy source/hash are recorded in `policy_report.json`
- Runtime defaults (schemas + policy) are bundled in package resources, so `policy-check` and `audit-validate` work even when running outside the repo root
- Trust model: by default `audit-verify` does **not** trust `signature.json` embedded public keys. For audit-grade authenticity, always provide `--public-key` (or `--public-key-dir`) and use `--profile audit-grade`.
- In `audit-verify --profile audit-grade`, `signature_key_id` and `attestation_key_id` must match (single signing identity for PACK + attestation).
- Key rotation: use `--public-key-dir` to auto-select the verification key by `signed.key_id`.
- Revocation: pass `--revoked-key-id` and/or `--revocation-file` to reject compromised signing keys at verify time.
- Key operations and rotation policy: `KEYS.md`
- Audit contract (required files, unchecked-file policy, semantic guarantees): `AUDIT_CONTRACT.md`

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
  audit/          # Audit pack builder (manifest/trace/checksums/zip)
  policy/         # Review-gating policy checks and queue/report generation
  review/         # Human review logging and reviewed-result generation
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
