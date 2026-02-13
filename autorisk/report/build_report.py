"""Report builder: Jinja2 rendering for HTML/Markdown reports."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from omegaconf import DictConfig, OmegaConf

from autorisk.cosmos.schema import CosmosResponse
from autorisk.eval.ablation import AblationResult
from autorisk.eval.evaluator import EvalReport
from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)


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
        output_dir: Path | str = "outputs",
        video_name: str = "",
    ) -> Path:
        """Render the full report.

        Args:
            responses: Ranked Cosmos responses.
            eval_report: Optional evaluation report.
            ablation_results: Optional ablation results.
            output_dir: Output directory.
            video_name: Source video filename.

        Returns:
            Path to generated report.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        top_n = self.cfg.report.top_n_examples
        top_responses = responses[:top_n]

        # Prepare template context
        examples = []
        for i, r in enumerate(top_responses, 1):
            a = r.assessment
            examples.append({
                "rank": i,
                "clip_path": r.request.clip_path,
                "clip_name": Path(r.request.clip_path).name if r.request.clip_path else "",
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
            })

        context = {
            "title": "AutoRisk-RM Analysis Report",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_name": video_name,
            "total_candidates": len(responses),
            "examples": examples,
            "severity_summary": self._severity_summary(responses),
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
