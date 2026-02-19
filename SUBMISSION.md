# Cosmos Cookoff Submission — AutoRisk-RM

## Text Description (copy-paste for submission form)

AutoRisk-RM is an end-to-end, fully automated pipeline that mines dangerous moments from long dashcam recordings and analyzes them with NVIDIA Cosmos Reason 2. It fuses three real-time signals — audio spikes (RMS/horn-band), optical-flow motion changes (Farneback), and object proximity (YOLOv8n) — to extract top-N candidate clips from hours of footage. Each clip is then analyzed locally on GPU by Cosmos Reason 2 (8B; official guidance: >=32 GB VRAM, BF16 recommended), which produces structured JSON output: calibrated severity classification, identified hazard actors with spatial relations, causal reasoning chains, short-term predictions, and recommended defensive actions.

Key technical contributions: (1) prompt calibration that eliminated a 70% HIGH false-positive rate, matching ground-truth distribution exactly; (2) a 2-stage inference strategy where a lightweight supplement pass recovers missing prediction/action fields for medium/high-risk clips, achieving a perfect 5/5 explanation checklist without re-running full inference; (3) a multi-layer JSON repair pipeline (truncation repair, missing-comma fix, brace extraction) achieving 100% parse success on model outputs; (4) physics-based TTC (time-to-collision) via YOLOv8n+ByteTrack tracking with statistically significant correlation to severity (Spearman rho=-0.495, p=0.026); (5) gradient saliency maps showing which video regions influence Cosmos's reasoning; (6) cross-modal grounding analysis (90.8% agreement between mining signals and VLM reasoning); (7) multi-video generalization across 4 diverse driving conditions (UK urban, Japan, winter/snow, US highway).

An interactive Streamlit dashboard with 6 pages (Overview, Clip Explorer, Evaluation, Signal Analysis, Technical Depth, Cross-Run Comparison) provides real-time exploration of results including video playback, attention heatmaps, TTC timelines, and cross-source comparison. Multi-video evaluation demonstrates generalization: 65 clips across 4 diverse conditions (Japan: 87.5% grounding, Winter: 100% parse/90% grounding, US Highway: 97.6% grounding) with severity distributions that correctly differentiate driving conditions. Public Mode provides full reproducibility: one command processes a rights-cleared public dashcam video, runs the pipeline, evaluates against blind-labeled ground truth (20 clips), and generates results — all within ~3 hours on a single RTX 5090.

## Submission Checklist

- [ ] Text description (above, ~180 words)
- [ ] Demo video (< 3 minutes) — upload to YouTube/Google Drive/Loom and paste link
- [ ] Code repository: https://github.com/maruyamakoju/autorisk-rm
- [ ] README with deployment instructions: included in repo

## Demo Video Script (3 min)

0:00-0:15  Problem: Manual review of hours of dashcam footage is slow and expensive
0:15-0:30  Architecture overview: B1 Mining (3 signals) -> B2 Cosmos Reason 2 -> Ranking -> Analysis
0:30-0:50  Streamlit Dashboard: Overview page — KPIs, severity distribution, pipeline diagram
0:50-1:20  Clip Explorer: pick a HIGH clip, show video + Cosmos output (hazards, causal, prediction, action)
1:20-1:40  Gradient saliency: show raw frame vs attention heatmap, explain what the model "sees"
1:40-2:00  Technical Depth: TTC box plots, cross-modal grounding (90.8%), calibration
2:00-2:20  Cross-Run Comparison: switch between UK/Japan/Winter/Highway, show severity distribution differences
2:20-2:40  Signal Analysis: Spearman correlations, ablation (Cosmos VLM F1 +17% over baseline)
2:40-3:00  Wrap-up: 4 diverse sources, 65 clips analyzed, applicable to fleet safety, ADAS, insurance
