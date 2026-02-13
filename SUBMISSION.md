# Cosmos Cookoff Submission — AutoRisk-RM

## Text Description (copy-paste for submission form)

AutoRisk-RM is an end-to-end, fully automated pipeline that mines dangerous moments from long dashcam recordings and analyzes them with NVIDIA Cosmos Reason 2. It fuses three real-time signals — audio spikes (RMS/horn-band), optical-flow motion changes (Farneback), and object proximity (YOLOv8n) — to extract top-N candidate clips from hours of footage. Each clip is then analyzed locally on GPU by Cosmos Reason 2 (8B, float16), which produces structured JSON output: calibrated severity classification, identified hazard actors with spatial relations, causal reasoning chains, short-term predictions, and recommended defensive actions.

Key technical contributions: (1) prompt calibration that eliminated a 70% HIGH false-positive rate, matching ground-truth distribution exactly; (2) a 2-stage inference strategy where a lightweight supplement pass recovers missing prediction/action fields for medium/high-risk clips, achieving a perfect 5/5 explanation checklist without re-running full inference; (3) a multi-layer JSON repair pipeline (truncation repair, missing-comma fix, brace extraction) achieving 100% parse success on model outputs.

Public Mode provides full reproducibility: one command downloads a public dashcam video, runs the pipeline, evaluates against blind-labeled ground truth (20 clips), and generates an interactive HTML report — all within ~3 hours on a single RTX 5090.

## Submission Checklist

- [ ] Text description (above, ~180 words)
- [ ] Demo video (< 3 minutes) — upload to YouTube/Google Drive/Loom and paste link
- [ ] Code repository: https://github.com/maruyamakoju/autorisk-rm
- [ ] README with deployment instructions: included in repo

## Demo Video Script (3 min)

0:00–0:20  Problem: Manual review of hours of dashcam footage is slow and expensive
0:20–0:45  Show README pipeline diagram: B1 Mining → B2 Cosmos Reason 2 → Eval → Report
0:45–1:20  Terminal: show candidates.csv + clips/ folder (fully automatic extraction)
1:20–2:20  Open report.html: scroll through Top incidents, show structured Cosmos output
           (hazards, causal reasoning, prediction, action) — point out the green "Cosmos Reason 2" tags
2:20–2:50  Show metrics: accuracy/F1/checklist + ablation "Why Cosmos Matters" side-by-side
2:50–3:00  Wrap-up: applicable to fleet safety, ADAS debugging, insurance claims
