"""CLI interface for AutoRisk-RM using Click."""

from __future__ import annotations

from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()


@click.group()
@click.option("--config", "-c", default=None, help="Path to custom YAML config")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, config: str | None, verbose: bool) -> None:
    """AutoRisk-RM: Automated Risk Mining from Dashcam Videos."""
    from autorisk.utils.config import load_config

    overrides = {}
    if verbose:
        overrides["general.log_level"] = "DEBUG"

    cfg = load_config(overrides=overrides or None, config_path=config)
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = cfg

    from autorisk.utils.logger import setup_logger
    setup_logger("autorisk", level=cfg.general.log_level)


@cli.command()
@click.option("--input", "-i", "video_path", required=True, help="Input video path")
@click.option("--out", "-o", "output_dir", default=None, help="Output directory")
@click.option("--skip-eval", is_flag=True, help="Skip evaluation stage")
@click.option("--skip-ablation", is_flag=True, help="Skip ablation study")
@click.option("--skip-report", is_flag=True, help="Skip report generation")
@click.option("--mode", default="default", help="Run mode: default or public")
@click.pass_context
def run(
    ctx: click.Context,
    video_path: str,
    output_dir: str | None,
    skip_eval: bool,
    skip_ablation: bool,
    skip_report: bool,
    mode: str,
) -> None:
    """Run the full AutoRisk-RM pipeline (B1 -> B2 -> B3 -> B4 -> B5 -> Report)."""
    import json

    from autorisk.pipeline import Pipeline

    cfg = ctx.obj["cfg"]

    if mode == "public":
        from autorisk.utils.config import load_config
        cfg = load_config(config_path="configs/public.yaml")

    pipeline = Pipeline(cfg)

    if mode == "public" and not Path(video_path).exists():
        click.echo("Public mode: downloading sample videos...")
        from scripts.download_public_data import download_public_videos
        videos = download_public_videos(cfg)
        if videos:
            video_path = str(videos[0])
        else:
            click.echo("Error: No videos downloaded", err=True)
            raise SystemExit(1)

    results = pipeline.run(
        video_path=video_path,
        output_dir=output_dir,
        skip_eval=skip_eval,
        skip_ablation=skip_ablation,
        skip_report=skip_report,
    )

    click.echo(json.dumps(results, indent=2, ensure_ascii=False))


@cli.command()
@click.option("--input", "-i", "video_path", required=True, help="Input video path")
@click.option("--out", "-o", "output_dir", default="outputs/mining", help="Output directory")
@click.pass_context
def mine(ctx: click.Context, video_path: str, output_dir: str) -> None:
    """Run only the candidate mining stage (B1)."""
    from autorisk.pipeline import Pipeline

    cfg = ctx.obj["cfg"]
    pipeline = Pipeline(cfg)
    candidates = pipeline.mine_only(video_path, output_dir)
    click.echo(f"Extracted {len(candidates)} candidates to {output_dir}")


@cli.command()
@click.option("--clips-dir", "-d", required=True, help="Directory containing clip MP4s")
@click.option("--out", "-o", "output_dir", default="outputs/inference", help="Output directory")
@click.pass_context
def infer(ctx: click.Context, clips_dir: str, output_dir: str) -> None:
    """Run Cosmos inference on pre-extracted clips (B2)."""
    from autorisk.pipeline import Pipeline

    cfg = ctx.obj["cfg"]
    pipeline = Pipeline(cfg)
    responses = pipeline.infer_only(clips_dir, output_dir)
    click.echo(f"Processed {len(responses)} clips, results saved to {output_dir}")


@cli.command("eval")
@click.option("--results", "-r", required=True, help="Path to cosmos_results.json")
@click.option("--gt", "-g", default=None, help="Path to GT labels CSV")
@click.option("--out", "-o", "output_dir", default="outputs/eval", help="Output directory")
@click.pass_context
def evaluate(
    ctx: click.Context,
    results: str,
    gt: str | None,
    output_dir: str,
) -> None:
    """Evaluate Cosmos results against ground truth (B4)."""
    import json

    from autorisk.cosmos.schema import (
        CosmosRequest,
        CosmosResponse,
        RiskAssessment,
    )
    from autorisk.eval.evaluator import Evaluator

    cfg = ctx.obj["cfg"]

    # Load results JSON
    with open(results, encoding="utf-8") as f:
        raw_results = json.load(f)

    responses: list[CosmosResponse] = []
    for entry in raw_results:
        from autorisk.cosmos.schema import HazardDetail
        assessment = RiskAssessment(
            severity=entry.get("severity", "NONE"),
            hazards=[HazardDetail(**h) for h in entry.get("hazards", [])],
            causal_reasoning=entry.get("causal_reasoning", ""),
            short_term_prediction=entry.get("short_term_prediction", ""),
            recommended_action=entry.get("recommended_action", ""),
            evidence=entry.get("evidence", []),
            confidence=entry.get("confidence", 0.0),
        )
        request = CosmosRequest(
            clip_path=entry.get("clip_path", ""),
            candidate_rank=entry.get("candidate_rank", 0),
            peak_time_sec=entry.get("peak_time_sec", 0.0),
            fused_score=entry.get("fused_score", 0.0),
        )
        responses.append(CosmosResponse(
            request=request,
            assessment=assessment,
            parse_success=entry.get("parse_success", True),
        ))

    evaluator = Evaluator(cfg.eval.severity_labels)
    gt_labels = evaluator.load_gt_labels(gt or cfg.eval.gt_path)
    eval_report = evaluator.evaluate(responses, gt_labels or None)
    evaluator.save_report(eval_report, Path(output_dir))

    click.echo(f"Accuracy: {eval_report.accuracy:.3f}")
    click.echo(f"Macro-F1: {eval_report.macro_f1:.3f}")
    click.echo(f"Checklist mean: {eval_report.checklist_means.get('mean_total', 0):.2f}/5")


@cli.command()
@click.option("--results", "-r", required=True, help="Path to cosmos_results.json")
@click.option("--out", "-o", "output_dir", default="outputs", help="Output directory")
@click.option("--format", "-f", "fmt", default="html", type=click.Choice(["html", "markdown"]))
@click.pass_context
def report(
    ctx: click.Context,
    results: str,
    output_dir: str,
    fmt: str,
) -> None:
    """Generate analysis report from Cosmos results."""
    import json

    from omegaconf import OmegaConf

    from autorisk.cosmos.schema import (
        CosmosRequest,
        CosmosResponse,
        HazardDetail,
        RiskAssessment,
    )
    from autorisk.ranking import rank_responses
    from autorisk.report.build_report import ReportBuilder

    cfg = ctx.obj["cfg"]
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"report": {"format": fmt}}))

    with open(results, encoding="utf-8") as f:
        raw_results = json.load(f)

    responses: list[CosmosResponse] = []
    for entry in raw_results:
        assessment = RiskAssessment(
            severity=entry.get("severity", "NONE"),
            hazards=[HazardDetail(**h) for h in entry.get("hazards", [])],
            causal_reasoning=entry.get("causal_reasoning", ""),
            short_term_prediction=entry.get("short_term_prediction", ""),
            recommended_action=entry.get("recommended_action", ""),
            evidence=entry.get("evidence", []),
            confidence=entry.get("confidence", 0.0),
        )
        request = CosmosRequest(
            clip_path=entry.get("clip_path", ""),
            candidate_rank=entry.get("candidate_rank", 0),
            peak_time_sec=entry.get("peak_time_sec", 0.0),
            fused_score=entry.get("fused_score", 0.0),
        )
        responses.append(CosmosResponse(request=request, assessment=assessment))

    ranked = rank_responses(responses)
    builder = ReportBuilder(cfg)
    report_path = builder.build(ranked, output_dir=Path(output_dir))
    click.echo(f"Report generated: {report_path}")


if __name__ == "__main__":
    cli()
