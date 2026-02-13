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

    from autorisk.cosmos.schema import CosmosResponse
    from autorisk.eval.evaluator import Evaluator

    cfg = ctx.obj["cfg"]

    # Load results JSON
    with open(results, encoding="utf-8") as f:
        raw_results = json.load(f)

    responses = [CosmosResponse.from_dict(entry) for entry in raw_results]

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

    from autorisk.cosmos.schema import CosmosResponse
    from autorisk.ranking import rank_responses
    from autorisk.report.build_report import ReportBuilder

    cfg = ctx.obj["cfg"]
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"report": {"format": fmt}}))

    with open(results, encoding="utf-8") as f:
        raw_results = json.load(f)

    responses = [CosmosResponse.from_dict(entry) for entry in raw_results]

    ranked = rank_responses(responses)

    # Load eval and ablation results if available in the same directory
    results_dir = Path(results).parent
    eval_report = None
    ablation_results = None

    eval_path = results_dir / "eval_report.json"
    if eval_path.exists():
        from autorisk.eval.evaluator import EvalReport
        with open(eval_path, encoding="utf-8") as ef:
            eval_data = json.load(ef)
        eval_report = EvalReport(
            n_samples=eval_data.get("n_samples", 0),
            accuracy=eval_data.get("accuracy", 0.0),
            macro_f1=eval_data.get("macro_f1", 0.0),
            checklist_means=eval_data.get("checklist_means", {}),
            confusion=eval_data.get("confusion_matrix", {}),
            failures=eval_data.get("failures", []),
        )

    ablation_path = results_dir / "ablation_results.json"
    if ablation_path.exists():
        from autorisk.eval.ablation import AblationResult
        with open(ablation_path, encoding="utf-8") as af:
            abl_data = json.load(af)
        ablation_results = [
            AblationResult(
                mode=a["mode"],
                description=a["description"],
                accuracy=a.get("accuracy", 0.0),
                macro_f1=a.get("macro_f1", 0.0),
                checklist_mean=a.get("checklist_mean", 0.0),
                n_candidates=a.get("n_candidates", 0),
            )
            for a in abl_data
        ]

    # Load analysis report if available
    analysis_report = None
    analysis_path = results_dir / "analysis_report.json"
    if analysis_path.exists():
        from autorisk.eval.analysis import (
            AnalysisReport,
            ErrorDetail,
            PerClassMetrics,
            SignalAnalysisResult,
        )
        with open(analysis_path, encoding="utf-8") as anf:
            an_data = json.load(anf)
        analysis_report = AnalysisReport(
            signal_analysis=[
                SignalAnalysisResult(**s) for s in an_data.get("signal_analysis", [])
            ],
            signal_heatmap=an_data.get("signal_heatmap", {}),
            per_class_metrics=[
                PerClassMetrics(**m) for m in an_data.get("per_class_metrics", [])
            ],
            error_details=[
                ErrorDetail(**e) for e in an_data.get("error_details", [])
            ],
            error_summary=an_data.get("error_summary", {}),
        )

    builder = ReportBuilder(cfg)
    report_path = builder.build(
        ranked,
        eval_report=eval_report,
        ablation_results=ablation_results,
        analysis_report=analysis_report,
        output_dir=Path(output_dir),
    )
    click.echo(f"Report generated: {report_path}")


@cli.command()
@click.option("--results", "-r", required=True, help="Path to cosmos_results.json")
@click.option("--gt", "-g", default=None, help="Path to GT labels CSV")
@click.option("--out", "-o", "output_dir", default="outputs/ablation", help="Output directory")
@click.pass_context
def ablation(
    ctx: click.Context,
    results: str,
    gt: str | None,
    output_dir: str,
) -> None:
    """Run minimal ablation study using existing Cosmos results (B5)."""
    from autorisk.eval.ablation import AblationRunner
    from autorisk.eval.evaluator import Evaluator

    cfg = ctx.obj["cfg"]

    evaluator = Evaluator(cfg.eval.severity_labels)
    gt_path = gt or cfg.eval.gt_path
    gt_labels = evaluator.load_gt_labels(gt_path)

    if not gt_labels:
        click.echo("Error: No GT labels found. Fill in data/annotations/gt_labels.csv first.", err=True)
        raise SystemExit(1)

    runner = AblationRunner(cfg)
    abl_results = runner.run_minimal_ablation(
        cosmos_results_path=Path(results),
        gt_labels=gt_labels,
        output_dir=Path(output_dir),
    )

    AblationRunner.save_results(abl_results, Path(output_dir))

    click.echo("\nAblation Results:")
    click.echo(f"{'Mode':<20} {'Accuracy':>10} {'Macro-F1':>10} {'Checklist':>10}")
    click.echo("-" * 52)
    for r in abl_results:
        click.echo(
            f"{r.mode:<20} {r.accuracy:>10.3f} {r.macro_f1:>10.3f} "
            f"{r.checklist_mean:>10.2f}"
        )


@cli.command()
@click.option("--results", "-r", required=True, help="Path to cosmos_results.json")
@click.option("--candidates", "-c", "candidates_csv", default=None, help="Path to candidates.csv")
@click.option("--gt", "-g", default=None, help="Path to GT labels CSV")
@click.option("--out", "-o", "output_dir", default=None, help="Output directory (default: same as results)")
@click.pass_context
def analyze(
    ctx: click.Context,
    results: str,
    candidates_csv: str | None,
    gt: str | None,
    output_dir: str | None,
) -> None:
    """Run deep analysis: signal contribution, error analysis, per-class metrics."""
    from autorisk.eval.analysis import AnalysisEngine

    cfg = ctx.obj["cfg"]
    results_path = Path(results)
    results_dir = results_path.parent
    save_dir = Path(output_dir) if output_dir else results_dir

    # Auto-discover candidates.csv and GT if not specified
    cand_path = Path(candidates_csv) if candidates_csv else results_dir / "candidates.csv"
    gt_path = Path(gt) if gt else Path(cfg.eval.gt_path)

    if not cand_path.exists():
        click.echo(f"Error: candidates.csv not found at {cand_path}", err=True)
        raise SystemExit(1)
    if not gt_path.exists():
        click.echo(f"Error: GT labels not found at {gt_path}", err=True)
        raise SystemExit(1)

    engine = AnalysisEngine()
    report = engine.run(
        cosmos_results_path=results_path,
        candidates_csv_path=cand_path,
        gt_labels_path=gt_path,
    )
    engine.save(report, save_dir)

    # Print summary
    click.echo("\n=== Signal Contribution ===")
    for s in report.signal_analysis:
        click.echo(
            f"  {s.signal_name:12s}  rho={s.spearman_rho:+.3f}  "
            f"thresh_acc={s.threshold_accuracy:.3f}  thresh_f1={s.threshold_f1:.3f}"
        )

    click.echo("\n=== Per-Class Metrics ===")
    click.echo(f"  {'Class':<8s} {'Precision':>9s} {'Recall':>8s} {'F1':>8s} {'Support':>8s}")
    for m in report.per_class_metrics:
        click.echo(
            f"  {m.label:<8s} {m.precision:>9.3f} {m.recall:>8.3f} {m.f1:>8.3f} {m.support:>8d}"
        )

    click.echo(f"\n=== Error Analysis ({report.error_summary.get('total_errors', 0)} errors) ===")
    es = report.error_summary
    click.echo(f"  Over-estimation: {es.get('over_estimation', 0)} ({es.get('over_estimation_pct', 0):.0f}%)")
    click.echo(f"  Under-estimation: {es.get('under_estimation', 0)}")
    click.echo(f"  Adjacent miss (off by 1): {es.get('adjacent_miss', 0)} ({es.get('adjacent_miss_pct', 0):.0f}%)")
    click.echo(f"  Major miss (off by 2+): {es.get('major_miss', 0)}")

    click.echo(f"\nSaved to: {save_dir / 'analysis_report.json'}")


@cli.command()
@click.option("--results", "-r", required=True, help="Path to cosmos_results.json")
@click.option("--out", "-o", "output_dir", default=None, help="Output directory (default: same as input)")
@click.pass_context
def supplement(ctx: click.Context, results: str, output_dir: str | None) -> None:
    """Fill missing prediction/action fields via 2nd-pass inference."""
    from autorisk.cosmos.infer import CosmosInferenceEngine

    results_path = Path(results)
    engine = CosmosInferenceEngine(ctx.obj["cfg"])
    responses = engine.supplement_results(results_path)

    save_dir = Path(output_dir) if output_dir else results_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    engine.save_results(responses, save_dir)

    n_supplemented = sum(
        1 for r in responses
        if r.assessment.severity in ("MEDIUM", "HIGH")
        and r.assessment.short_term_prediction.strip()
        and r.assessment.recommended_action.strip()
    )
    n_target = sum(
        1 for r in responses if r.assessment.severity in ("MEDIUM", "HIGH")
    )
    click.echo(f"Supplement complete: {n_supplemented}/{n_target} MEDIUM/HIGH have prediction+action")
    click.echo(f"Saved to: {save_dir / 'cosmos_results.json'}")


@cli.command()
@click.option("--results", "-r", required=True, help="Path to cosmos_results.json")
@click.option("--out", "-o", "output_dir", default=None, help="Output directory (default: same as input)")
@click.pass_context
def reparse(ctx: click.Context, results: str, output_dir: str | None) -> None:
    """Re-parse failed entries in cosmos_results.json with improved parser."""
    from autorisk.cosmos.infer import CosmosInferenceEngine

    results_path = Path(results)
    responses = CosmosInferenceEngine.reparse_results(results_path)

    # Save to output dir (or same dir as input)
    save_dir = Path(output_dir) if output_dir else results_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # Use the engine's save method
    engine = CosmosInferenceEngine(ctx.obj["cfg"])
    engine.save_results(responses, save_dir)

    n_success = sum(1 for r in responses if r.parse_success)
    click.echo(f"Re-parsed: {n_success}/{len(responses)} parse success")
    click.echo(f"Saved to: {save_dir / 'cosmos_results.json'}")


if __name__ == "__main__":
    cli()
