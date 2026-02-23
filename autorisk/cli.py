"""CLI interface for AutoRisk-RM using Click."""

from __future__ import annotations

from pathlib import Path

import click
from dotenv import load_dotenv

from autorisk.cli_commands import register_audit_commands, register_multi_video_commands

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
@click.option(
    "--allow-public-download",
    is_flag=True,
    help="Allow mode=public to auto-download third-party video when --input path is missing",
)
@click.pass_context
def run(
    ctx: click.Context,
    video_path: str,
    output_dir: str | None,
    skip_eval: bool,
    skip_ablation: bool,
    skip_report: bool,
    mode: str,
    allow_public_download: bool,
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
        if not allow_public_download:
            click.echo(
                (
                    "Error: public mode input is missing and automatic third-party download is disabled. "
                    "Provide a licensed local video path, or re-run with --allow-public-download "
                    "after confirming rights to the source."
                ),
                err=True,
            )
            click.echo(
                "Hint: python scripts/download_public_data.py --ack-data-rights --config configs/public.yaml",
                err=True,
            )
            raise SystemExit(2)
        click.echo("Public mode: downloading source video(s)...")
        from scripts.download_public_data import download_public_videos
        videos = download_public_videos(cfg, allow_third_party=True)
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

    # Load TTC results if available
    ttc_results = None
    ttc_path = results_dir / "ttc_results.json"
    if ttc_path.exists():
        from autorisk.mining.tracking import TTCEstimator
        ttc_results = TTCEstimator.load_results(ttc_path)

    # Load calibration report if available
    calibration_report = None
    cal_path = results_dir / "calibration_report.json"
    if cal_path.exists():
        from autorisk.eval.calibration import CalibrationAnalyzer
        calibration_report = CalibrationAnalyzer.load(cal_path)

    # Load grounding report if available
    grounding_report = None
    grd_path = results_dir / "grounding_report.json"
    if grd_path.exists():
        from autorisk.cosmos.grounding import GroundingAnalyzer
        grounding_report = GroundingAnalyzer.load(grd_path)

    # Load saliency images if available
    saliency_images = None
    sal_path = results_dir / "saliency_images.json"
    if sal_path.exists():
        with open(sal_path, encoding="utf-8") as sf:
            saliency_images = json.load(sf)

    builder = ReportBuilder(cfg)
    report_path = builder.build(
        ranked,
        eval_report=eval_report,
        ablation_results=ablation_results,
        analysis_report=analysis_report,
        ttc_results=ttc_results,
        calibration_report=calibration_report,
        grounding_report=grounding_report,
        saliency_images=saliency_images,
        output_dir=Path(output_dir),
    )
    click.echo(f"Report generated: {report_path}")


@cli.command("narrative")
@click.option("--results", "-r", required=True, help="Path to cosmos_results.json")
@click.option("--out", "-o", "output_path", default=None, help="Output markdown file path")
@click.pass_context
def narrative(
    ctx: click.Context,
    results: str,
    output_path: str | None,
) -> None:
    """Generate human-readable safety narrative from cosmos results."""
    from autorisk.report.safety_narrative import generate_from_json

    results_path = Path(results)
    if not results_path.exists():
        click.echo(f"Error: Results file not found: {results}", err=True)
        raise SystemExit(1)

    if output_path is None:
        output_path = results_path.parent / "safety_narrative.md"
    else:
        output_path = Path(output_path)

    saved_path = generate_from_json(results_path, output_path)
    click.echo(f"Safety narrative generated: {saved_path}")


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


@cli.command()
@click.option("--clips-dir", "-d", required=True, help="Path to clips directory")
@click.option("--out", "-o", "output_dir", default=None, help="Output directory (default: same as clips parent)")
@click.option("--gt", "-g", default=None, help="Path to GT labels CSV (for correlation)")
@click.pass_context
def ttc(
    ctx: click.Context,
    clips_dir: str,
    output_dir: str | None,
    gt: str | None,
) -> None:
    """Compute Time-to-Collision (TTC) via object tracking for all clips."""
    from autorisk.mining.tracking import TTCEstimator, compute_ttc_severity_correlation

    cfg = ctx.obj["cfg"]
    clips_path = Path(clips_dir)
    save_dir = Path(output_dir) if output_dir else clips_path.parent

    estimator = TTCEstimator(cfg)
    results = estimator.analyze_clips(clips_path, save_dir)

    click.echo(f"\n=== TTC Analysis ({len(results)} clips) ===")
    for r in results:
        clip = Path(r.clip_path).name
        min_ttc = f"{r.min_ttc:.2f}s" if r.min_ttc < float("inf") else "inf"
        click.echo(f"  {clip}: TTC={min_ttc}, tracks={r.n_tracks}, critical={r.n_critical}")

    # Correlation with GT if available
    gt_path = Path(gt) if gt else Path(cfg.eval.gt_path)
    if gt_path.exists():
        import csv as csv_mod
        gt_labels = {}
        with open(gt_path, encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                clip = row.get("clip_path", "").strip()
                sev = row.get("severity", "NONE").strip().upper()
                if clip:
                    gt_labels[Path(clip).name] = sev

        corr = compute_ttc_severity_correlation(results, gt_labels)
        click.echo(f"\nTTC vs Severity: rho={corr['spearman_rho']:.3f} (p={corr['spearman_p']:.4f})")
        for sev, ttc_val in corr.get("mean_ttc_by_severity", {}).items():
            click.echo(f"  {sev}: mean TTC = {ttc_val:.2f}s")

    click.echo(f"\nSaved to: {save_dir / 'ttc_results.json'}")


@cli.command()
@click.option("--results", "-r", required=True, help="Path to cosmos_results.json")
@click.option("--candidates", "-c", "candidates_csv", default=None, help="Path to candidates.csv")
@click.option("--out", "-o", "output_dir", default=None, help="Output directory")
@click.pass_context
def grounding(
    ctx: click.Context,
    results: str,
    candidates_csv: str | None,
    output_dir: str | None,
) -> None:
    """Analyze cross-modal grounding between mining signals and Cosmos reasoning."""
    from autorisk.cosmos.grounding import GroundingAnalyzer

    results_path = Path(results)
    results_dir = results_path.parent
    save_dir = Path(output_dir) if output_dir else results_dir

    cand_path = Path(candidates_csv) if candidates_csv else results_dir / "candidates.csv"
    if not cand_path.exists():
        click.echo(f"Error: candidates.csv not found at {cand_path}", err=True)
        raise SystemExit(1)

    analyzer = GroundingAnalyzer()
    report = analyzer.run(results_path, cand_path)
    analyzer.save(report, save_dir)

    click.echo(f"\n=== Cross-Modal Grounding ({report.n_clips} clips) ===")
    click.echo(f"  Mean grounding score: {report.mean_grounding_score:.3f}")
    click.echo(f"  Fully grounded: {report.n_fully_grounded}/{report.n_clips}")
    click.echo(f"  Has hallucination: {report.n_has_hallucination}")
    click.echo(f"\n  Signal grounding rates:")
    for sig, rate in report.signal_grounding_rates.items():
        click.echo(f"    {sig}: {rate:.3f}")
    click.echo(f"\nSaved to: {save_dir / 'grounding_report.json'}")


@cli.command()
@click.option("--results", "-r", required=True, help="Path to cosmos_results.json")
@click.option("--gt", "-g", default=None, help="Path to GT labels CSV")
@click.option("--out", "-o", "output_dir", default=None, help="Output directory")
@click.pass_context
def calibration(
    ctx: click.Context,
    results: str,
    gt: str | None,
    output_dir: str | None,
) -> None:
    """Analyze confidence calibration: ECE, temperature scaling, reliability diagram."""
    from autorisk.eval.calibration import CalibrationAnalyzer

    cfg = ctx.obj["cfg"]
    results_path = Path(results)
    save_dir = Path(output_dir) if output_dir else results_path.parent
    gt_path = Path(gt) if gt else Path(cfg.eval.gt_path)

    if not gt_path.exists():
        click.echo(f"Error: GT labels not found at {gt_path}", err=True)
        raise SystemExit(1)

    analyzer = CalibrationAnalyzer()
    report = analyzer.run(results_path, gt_path)
    analyzer.save(report, save_dir)

    click.echo(f"\n=== Confidence Calibration ({report.n_samples} samples) ===")
    click.echo(f"  ECE: {report.ece:.4f} -> {report.ece_after:.4f} (T={report.optimal_temperature:.2f})")
    click.echo(f"  MCE: {report.mce:.4f}")
    click.echo(f"  Brier: {report.brier_score:.4f} -> {report.brier_score_after:.4f}")
    click.echo(f"\n  Per-severity confidence:")
    for sev, stats in report.confidence_by_severity.items():
        oc = "OVERCONFIDENT" if stats["overconfident"] else "underconfident"
        click.echo(f"    {sev}: conf={stats['mean_confidence']:.3f}, acc={stats['accuracy']:.3f} ({oc})")
    click.echo(f"\nSaved to: {save_dir / 'calibration_report.json'}")


@cli.command()
@click.option("--clips-dir", "-d", required=True, help="Path to clips directory")
@click.option("--results", "-r", default=None, help="Path to cosmos_results.json (for severity filtering)")
@click.option("--out", "-o", "output_dir", default=None, help="Output directory")
@click.option("--max-clips", default=10, help="Maximum clips to process")
@click.option("--severity", "-s", multiple=True, default=("MEDIUM", "HIGH"), help="Severity filter")
@click.pass_context
def saliency(
    ctx: click.Context,
    clips_dir: str,
    results: str | None,
    output_dir: str | None,
    max_clips: int,
    severity: tuple[str, ...],
) -> None:
    """Extract gradient-based saliency maps (requires GPU + model loading)."""
    from autorisk.viz.attention import SaliencyExtractor

    cfg = ctx.obj["cfg"]
    clips_path = Path(clips_dir)
    save_dir = Path(output_dir) if output_dir else clips_path.parent
    results_path = Path(results) if results else None

    extractor = SaliencyExtractor(cfg)
    saliency_results = extractor.extract_batch(
        clips_path,
        save_dir,
        severity_filter=set(severity),
        cosmos_results_path=results_path,
        max_clips=max_clips,
    )

    click.echo(f"\n=== Saliency Extraction ({len(saliency_results)} clips) ===")
    for r in saliency_results:
        click.echo(
            f"  {r['clip_name']}: peak_frame={r['peak_frame_idx']}/{r['n_temporal_frames']}, "
            f"grid={r['spatial_grid']}"
        )
    click.echo(f"\nSaved to: {save_dir}")



@cli.command()
@click.option("--results", "-r", required=True, help="Path to cosmos_results.json")
@click.option("--gt", "-g", default=None, help="Path to GT labels CSV")
@click.option("--params", "-p", "params_path", default=None, help="Path to optimal_params.json (skip search)")
@click.option("--search", is_flag=True, help="Run random search for optimal parameters")
@click.option("--loocv", is_flag=True, help="Run Leave-One-Out CV grid search")
@click.option("--out", "-o", "output_dir", default="outputs/enhanced_correction", help="Output directory")
@click.pass_context
def correct(
    ctx: click.Context,
    results: str,
    gt: str | None,
    params_path: str | None,
    search: bool,
    loocv: bool,
    output_dir: str,
) -> None:
    """Apply signal-based severity correction using TTC + fused scores."""
    import json

    from autorisk.eval.enhanced_correction import (
        EnhancedCorrector,
        evaluate_enhanced,
        grid_search_enhanced,
        load_gt_labels,
        loocv_grid_search,
        save_correction_outputs,
    )

    cfg = ctx.obj["cfg"]
    results_path = Path(results)
    results_dir = results_path.parent
    save_dir = Path(output_dir)

    with open(results_path, encoding="utf-8") as f:
        cosmos_results = json.load(f)

    # Load TTC data
    ttc_path = results_dir / "ttc_results.json"
    ttc_data = []
    if ttc_path.exists():
        with open(ttc_path, encoding="utf-8") as f:
            ttc_data = json.load(f)

    # Load GT
    gt_path = Path(gt) if gt else Path(cfg.eval.gt_path)
    gt_labels = load_gt_labels(gt_path)

    # Load or search for params
    loocv_report = None
    if params_path:
        with open(params_path, encoding="utf-8") as f:
            params = json.load(f)
    elif search or loocv:
        params, _ = grid_search_enhanced(cosmos_results, ttc_data, gt_labels)
        click.echo(f"Best params accuracy: {evaluate_enhanced(EnhancedCorrector(params).correct_batch(cosmos_results, ttc_data), gt_labels).accuracy:.3f}")
    else:
        # Try loading from existing output or use defaults
        default_params_path = save_dir / "optimal_params.json"
        if default_params_path.exists():
            with open(default_params_path, encoding="utf-8") as f:
                params = json.load(f)
        else:
            from autorisk.eval.enhanced_correction import DEFAULT_PARAMS
            params = dict(DEFAULT_PARAMS)

    # Run LOOCV if requested
    if loocv:
        loocv_report = loocv_grid_search(cosmos_results, ttc_data, gt_labels)
        click.echo(f"LOOCV Accuracy: {loocv_report['loocv_accuracy']:.3f}")
        click.echo(f"LOOCV Macro-F1: {loocv_report['loocv_f1']:.3f}")

    # Apply correction
    corrector = EnhancedCorrector(params)
    corrected = corrector.correct_batch(cosmos_results, ttc_data)
    report = evaluate_enhanced(corrected, gt_labels)

    save_correction_outputs(corrected, report, params, save_dir, loocv_report)

    click.echo(f"\nAccuracy: {report.accuracy:.3f} (was 0.350)")
    click.echo(f"Macro-F1: {report.macro_f1:.3f}")
    click.echo(f"Correct: {report.n_correct}/{report.n_total}")
    click.echo(f"Saved to: {save_dir}")


@cli.command("predict")
@click.option("--results", "-r", required=True, help="Path to cosmos_results.json")
@click.option("--clips-dir", "-d", default=None, help="Path to clips directory")
@click.option("--out", "-o", "output_dir", default=None, help="Output directory")
@click.option("--severity", "-s", multiple=True, default=("HIGH", "MEDIUM"), help="Severity filter")
@click.pass_context
def predict(
    ctx: click.Context,
    results: str,
    clips_dir: str | None,
    output_dir: str | None,
    severity: tuple[str, ...],
) -> None:
    """Generate future prediction videos using Cosmos Predict 2."""
    import json

    from autorisk.cosmos.predict_client import CosmosPredictClient

    cfg = ctx.obj["cfg"]
    results_path = Path(results)
    results_dir = results_path.parent
    clips = Path(clips_dir) if clips_dir else results_dir / "clips"
    save_dir = Path(output_dir) if output_dir else results_dir / "predictions"

    with open(results_path, encoding="utf-8") as f:
        cosmos_results = json.load(f)

    # Unload Reason 2 if loaded to free VRAM
    click.echo("Initializing Cosmos Predict 2...")
    client = CosmosPredictClient(cfg)

    predict_results = client.predict_batch(
        cosmos_results=cosmos_results,
        clips_dir=clips,
        output_dir=save_dir,
        severity_filter=set(severity),
    )

    client.unload()

    click.echo(f"\nGenerated {len(predict_results)} prediction videos")
    for pr in predict_results:
        click.echo(f"  {pr['clip_name']}: {pr['output_path']}")
    click.echo(f"Saved to: {save_dir}")

    # Save predict metadata
    with open(save_dir / "predict_results.json", "w", encoding="utf-8") as f:
        json.dump(predict_results, f, indent=2, ensure_ascii=False)


# Register extracted command groups.
register_audit_commands(cli)
register_multi_video_commands(cli)


@cli.command()
@click.option("--run-dir", "-r", default=None, help="Run output directory")
@click.option("--port", "-p", default=8501, help="Streamlit server port")
def dashboard(run_dir: str | None, port: int) -> None:
    """Launch interactive Streamlit dashboard."""
    import subprocess

    app_path = Path(__file__).parent / "dashboard" / "app.py"
    if not app_path.exists():
        click.echo(f"Error: dashboard app not found at {app_path}", err=True)
        raise SystemExit(1)

    cmd = ["streamlit", "run", str(app_path), "--server.port", str(port)]
    click.echo(f"Launching dashboard at http://localhost:{port}")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    cli()
