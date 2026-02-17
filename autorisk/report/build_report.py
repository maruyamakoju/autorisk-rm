"""Report builder: Jinja2 rendering for HTML/Markdown reports."""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

import cv2
from jinja2 import Environment, FileSystemLoader
from omegaconf import DictConfig

from autorisk.cosmos.schema import CosmosResponse
from autorisk.eval.ablation import AblationResult
from autorisk.eval.analysis import AnalysisReport
from autorisk.eval.evaluator import EvalReport
from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)


def _extract_thumbnail_b64(clip_path: Path, width: int = 320) -> str:
    """Extract middle frame from clip and return as base64 JPEG."""
    try:
        cap = cv2.VideoCapture(str(clip_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return ""
        h, w = frame.shape[:2]
        new_h = int(h * width / w)
        frame = cv2.resize(frame, (width, new_h))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode("ascii")
    except Exception:
        return ""


class ReportBuilder:
    """Build HTML or Markdown reports from pipeline results."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True,
        )

    def build(
        self,
        responses: list[CosmosResponse],
        eval_report: EvalReport | None = None,
        ablation_results: list[AblationResult] | None = None,
        analysis_report: AnalysisReport | None = None,
        ttc_results=None,
        calibration_report=None,
        grounding_report=None,
        saliency_images: dict | None = None,
        output_dir: Path | str = "outputs",
        video_name: str = "",
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        top_n = self.cfg.report.top_n_examples
        top_responses = responses[:top_n]

        # Prepare template context
        examples = []
        for i, r in enumerate(top_responses, 1):
            a = r.assessment
            clip_abs = Path(r.request.clip_path) if r.request.clip_path else None
            clip_rel = ""
            thumb_b64 = ""
            if clip_abs and clip_abs.exists():
                try:
                    clip_rel = clip_abs.relative_to(output_dir).as_posix()
                except ValueError:
                    clip_rel = clip_abs.as_posix()
                thumb_b64 = _extract_thumbnail_b64(clip_abs)

            # Attach TTC data for this clip
            clip_ttc = None
            if ttc_results:
                clip_name = clip_abs.name if clip_abs else ""
                for tr in ttc_results:
                    if Path(tr.clip_path).name == clip_name:
                        clip_ttc = {
                            "min_ttc": tr.min_ttc if tr.min_ttc < float("inf") else None,
                            "n_tracks": tr.n_tracks,
                            "n_critical": tr.n_critical,
                            "top_track": {
                                "class_name": tr.tracks[0].class_name,
                                "min_ttc": tr.tracks[0].min_ttc,
                            } if tr.tracks else None,
                        }
                        break

            # Attach saliency heatmap if available
            clip_saliency_b64 = ""
            if saliency_images and clip_abs:
                sal_data = saliency_images.get(clip_abs.name, {})
                clip_saliency_b64 = sal_data.get("heatmap_b64", "")

            examples.append({
                "rank": i,
                "clip_path": r.request.clip_path,
                "clip_rel": clip_rel,
                "clip_name": clip_abs.name if clip_abs else "",
                "thumbnail_b64": thumb_b64,
                "saliency_b64": clip_saliency_b64,
                "peak_time": r.request.peak_time_sec,
                "fused_score": r.request.fused_score,
                "severity": a.severity,
                "hazards": [
                    {
                        "type": h.type,
                        "actors": ", ".join(h.actors),
                        "spatial": h.spatial_relation,
                    }
                    for h in a.hazards
                ],
                "causal_reasoning": a.causal_reasoning,
                "prediction": a.short_term_prediction,
                "action": a.recommended_action,
                "evidence": a.evidence,
                "confidence": a.confidence,
                "ttc": clip_ttc,
            })

        # Timeline data (all responses, not just top-N)
        timeline = []
        for r in responses:
            timeline.append({
                "time": r.request.peak_time_sec,
                "severity": r.assessment.severity,
            })
        timeline.sort(key=lambda x: x["time"])

        context = {
            "title": "AutoRisk-RM Analysis Report",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_name": video_name,
            "total_candidates": len(responses),
            "examples": examples,
            "severity_summary": self._severity_summary(responses),
            "timeline": timeline,
        }

        if eval_report is not None:
            context["eval"] = {
                "accuracy": eval_report.accuracy,
                "macro_f1": eval_report.macro_f1,
                "checklist": eval_report.checklist_means,
                "n_samples": eval_report.n_samples,
                "confusion": eval_report.confusion,
                "n_failures": len(eval_report.failures),
            }

        if ablation_results:
            context["ablation"] = [
                {
                    "mode": r.mode,
                    "description": r.description,
                    "accuracy": r.accuracy,
                    "macro_f1": r.macro_f1,
                    "checklist_mean": r.checklist_mean,
                    "n_candidates": r.n_candidates,
                }
                for r in ablation_results
            ]

        if analysis_report is not None:
            context["analysis"] = {
                "signal_analysis": [
                    {
                        "signal_name": s.signal_name,
                        "spearman_rho": s.spearman_rho,
                        "spearman_p": s.spearman_p,
                        "mean_score_by_severity": s.mean_score_by_severity,
                        "threshold_accuracy": s.threshold_accuracy,
                        "threshold_f1": s.threshold_f1,
                    }
                    for s in analysis_report.signal_analysis
                ],
                "signal_heatmap": analysis_report.signal_heatmap,
                "per_class_metrics": [
                    {
                        "label": m.label,
                        "precision": m.precision,
                        "recall": m.recall,
                        "f1": m.f1,
                        "support": m.support,
                    }
                    for m in analysis_report.per_class_metrics
                ],
                "error_summary": analysis_report.error_summary,
                "error_details": [
                    {
                        "clip_name": e.clip_name,
                        "gt_severity": e.gt_severity,
                        "pred_severity": e.pred_severity,
                        "error_type": e.error_type,
                        "severity_gap": e.severity_gap,
                        "fused_score": e.fused_score,
                        "reasoning_excerpt": e.reasoning_excerpt,
                    }
                    for e in analysis_report.error_details
                ],
            }

        # TTC analysis context
        if ttc_results:
            from autorisk.mining.tracking import TTC_CRITICAL, TTC_WARNING
            ttc_data = []
            for tr in ttc_results:
                ttc_data.append({
                    "clip_name": Path(tr.clip_path).name,
                    "min_ttc": tr.min_ttc if tr.min_ttc < float("inf") else None,
                    "n_tracks": tr.n_tracks,
                    "n_critical": tr.n_critical,
                    "n_warning": tr.n_warning,
                })
            context["ttc"] = {
                "clips": ttc_data,
                "critical_threshold": TTC_CRITICAL,
                "warning_threshold": TTC_WARNING,
            }

        # Calibration context
        if calibration_report:
            context["calibration"] = {
                "ece": calibration_report.ece,
                "ece_after": calibration_report.ece_after,
                "mce": calibration_report.mce,
                "optimal_temperature": calibration_report.optimal_temperature,
                "brier_score": calibration_report.brier_score,
                "brier_score_after": calibration_report.brier_score_after,
                "n_samples": calibration_report.n_samples,
                "confidence_by_severity": calibration_report.confidence_by_severity,
                "bins": [
                    {
                        "bin_lower": b.bin_lower,
                        "bin_upper": b.bin_upper,
                        "avg_confidence": b.avg_confidence,
                        "avg_accuracy": b.avg_accuracy,
                        "count": b.count,
                        "gap": b.gap,
                    }
                    for b in calibration_report.bins
                ],
                "bins_after": [
                    {
                        "bin_lower": b.bin_lower,
                        "bin_upper": b.bin_upper,
                        "avg_confidence": b.avg_confidence,
                        "avg_accuracy": b.avg_accuracy,
                        "count": b.count,
                        "gap": b.gap,
                    }
                    for b in calibration_report.bins_after
                ],
            }

        # Grounding context
        if grounding_report:
            context["grounding"] = {
                "mean_score": grounding_report.mean_grounding_score,
                "n_clips": grounding_report.n_clips,
                "n_fully_grounded": grounding_report.n_fully_grounded,
                "n_has_hallucination": grounding_report.n_has_hallucination,
                "n_has_ungrounded": grounding_report.n_has_ungrounded,
                "signal_rates": grounding_report.signal_grounding_rates,
                "by_severity": grounding_report.grounding_by_severity,
                "details": [
                    {
                        "clip_name": d.clip_name,
                        "grounding_score": d.grounding_score,
                        "active_signals": d.active_signals,
                        "mentioned_signals": d.mentioned_signals,
                        "ungrounded": d.ungrounded_signals,
                        "hallucinated": d.hallucinated_signals,
                    }
                    for d in grounding_report.details
                ],
            }

        fmt = self.cfg.report.format
        if fmt == "html":
            return self._render_html(context, output_dir)
        else:
            return self._render_markdown(context, output_dir)

    def _render_html(self, context: dict, output_dir: Path) -> Path:
        template = self.env.get_template("report.html.jinja2")
        html = template.render(**context)
        path = output_dir / "report.html"
        path.write_text(html, encoding="utf-8")
        log.info("Generated HTML report: %s", path)
        return path

    def _render_markdown(self, context: dict, output_dir: Path) -> Path:
        lines = [
            f"# {context['title']}",
            f"Generated: {context['generated_at']}",
            f"Video: {context['video_name']}",
            f"Total candidates: {context['total_candidates']}",
            "",
            "## Severity Summary",
        ]

        for sev, count in context["severity_summary"].items():
            lines.append(f"- **{sev}**: {count}")

        lines.append("")
        lines.append(f"## Top {len(context['examples'])} Risk Events")
        lines.append("")

        for ex in context["examples"]:
            lines.append(f"### #{ex['rank']} — {ex['severity']} (confidence: {ex['confidence']:.2f})")
            lines.append(f"- **Clip**: {ex['clip_name']}")
            lines.append(f"- **Time**: {ex['peak_time']:.1f}s (score: {ex['fused_score']:.3f})")
            lines.append(f"- **Causal reasoning**: {ex['causal_reasoning']}")
            lines.append(f"- **Prediction**: {ex['prediction']}")
            lines.append(f"- **Action**: {ex['action']}")
            if ex["hazards"]:
                lines.append("- **Hazards**:")
                for h in ex["hazards"]:
                    lines.append(f"  - {h['type']}: {h['actors']} ({h['spatial']})")
            lines.append("")

        if "eval" in context:
            ev = context["eval"]
            lines.append("## Evaluation Metrics")
            lines.append(f"- Accuracy: {ev['accuracy']:.3f}")
            lines.append(f"- Macro-F1: {ev['macro_f1']:.3f}")
            lines.append(f"- Checklist mean: {ev['checklist'].get('mean_total', 0):.2f}/5")
            lines.append("")

        if "ablation" in context:
            lines.append("## Ablation Results")
            lines.append("| Mode | Candidates | Accuracy | F1 | Checklist |")
            lines.append("|------|-----------|----------|-----|-----------|")
            for ab in context["ablation"]:
                lines.append(
                    f"| {ab['mode']} | {ab['n_candidates']} | "
                    f"{ab['accuracy']:.3f} | {ab['macro_f1']:.3f} | "
                    f"{ab['checklist_mean']:.2f} |"
                )
            lines.append("")

        path = output_dir / "report.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        log.info("Generated Markdown report: %s", path)
        return path

    @staticmethod
    def _severity_summary(responses: list[CosmosResponse]) -> dict[str, int]:
        summary: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}
        for r in responses:
            sev = r.assessment.severity.upper()
            summary[sev] = summary.get(sev, 0) + 1
        return summary
