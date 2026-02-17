"""CLI interface for AutoRisk-RM using Click."""

from __future__ import annotations

import os
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import zipfile

import click
from dotenv import load_dotenv

load_dotenv()


def _resolve_private_key_password(
    *,
    private_key_password: str | None,
    private_key_password_env: str | None,
) -> str | None:
    if private_key_password is not None and private_key_password_env is not None:
        raise click.UsageError("Use either --private-key-password or --private-key-password-env, not both.")
    if private_key_password_env is not None:
        env_name = str(private_key_password_env).strip()
        if env_name == "":
            raise click.UsageError("--private-key-password-env cannot be empty.")
        value = os.environ.get(env_name)
        if value is None:
            raise click.UsageError(f"Environment variable not set: {env_name}")
        return value
    if private_key_password is None:
        return None
    return str(private_key_password)


def _load_revoked_key_ids(
    *,
    revoked_key_ids: tuple[str, ...],
    revocation_file: str | None,
) -> set[str]:
    out = {str(v).strip().lower() for v in revoked_key_ids if str(v).strip() != ""}
    if revocation_file is None or str(revocation_file).strip() == "":
        return out
    path = Path(revocation_file).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise click.UsageError(f"Revocation file not found: {path}")
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line == "" or line.startswith("#"):
            continue
        out.add(line.split()[0].lower())
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _autorisk_version() -> str:
    try:
        return str(importlib.metadata.version("autorisk-rm"))
    except Exception:
        try:
            from autorisk import __version__

            return str(__version__)
        except Exception:
            return "unknown"


def _upsert_zip_member(zip_path: Path, member_name: str, payload: bytes) -> None:
    with tempfile.TemporaryDirectory(prefix="autorisk-finalize-") as tmp_dir:
        tmp_zip = Path(tmp_dir) / zip_path.name
        with zipfile.ZipFile(zip_path, "r") as src, zipfile.ZipFile(
            tmp_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as dst:
            for info in src.infolist():
                if info.filename == member_name:
                    continue
                if info.is_dir():
                    dst.writestr(info, b"")
                    continue
                dst.writestr(info, src.read(info.filename))
            dst.writestr(member_name, payload)
        tmp_zip.replace(zip_path)


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


@cli.command("audit-pack")
@click.option("--run-dir", "-r", required=True, help="Run directory containing cosmos_results.json and candidates.csv")
@click.option("--input-video", "-i", default=None, help="Optional source input video path (for SHA256 provenance)")
@click.option("--review-log", default=None, help="Optional human review log jsonl to include in the pack")
@click.option("--out", "-o", "output_dir", default=None, help="Output directory (default: RUN_DIR/audit_pack_<timestamp>)")
@click.option("--include-clips/--no-include-clips", default=True, show_default=True, help="Copy candidate clips into the pack")
@click.option("--zip/--no-zip", "create_zip", default=True, show_default=True, help="Create zip bundle for handoff")
@click.pass_context
def audit_pack(
    ctx: click.Context,
    run_dir: str,
    input_video: str | None,
    review_log: str | None,
    output_dir: str | None,
    include_clips: bool,
    create_zip: bool,
) -> None:
    """Build an auditable evidence pack (manifest + decision trace + checksums)."""
    from autorisk.audit.pack import build_audit_pack

    cfg = (ctx.obj or {}).get("cfg")
    result = build_audit_pack(
        run_dir=run_dir,
        cfg=cfg,
        output_dir=output_dir,
        input_video=input_video,
        review_log=review_log,
        include_clips=include_clips,
        create_zip=create_zip,
    )

    click.echo(f"Audit pack directory: {result.output_dir}")
    click.echo(f"Records: {result.records}")
    click.echo(f"Manifest: {result.manifest_path}")
    click.echo(f"Decision trace: {result.decision_trace_path}")
    click.echo(f"Checksums: {result.checksums_path}")
    click.echo(f"Pack fingerprint (checksums SHA256): {result.checksums_sha256}")
    if result.zip_path is not None:
        click.echo(f"Zip: {result.zip_path}")


@cli.command("audit-sign")
@click.option("--pack", "-p", required=True, help="Audit pack directory or zip bundle")
@click.option("--private-key", required=True, help="Ed25519 private key PEM path")
@click.option("--private-key-password", default=None, help="Private key password (less secure; prefer --private-key-password-env)")
@click.option("--private-key-password-env", default=None, help="Environment variable containing private key password")
@click.option("--public-key", default=None, help="Optional Ed25519 public key PEM path")
@click.option("--key-label", default=None, help="Optional key label metadata (for audit logs)")
@click.option(
    "--embed-public-key/--no-embed-public-key",
    default=False,
    show_default=True,
    help="Embed public key PEM in signature.json",
)
def audit_sign(
    pack: str,
    private_key: str,
    private_key_password: str | None,
    private_key_password_env: str | None,
    public_key: str | None,
    key_label: str | None,
    embed_public_key: bool,
) -> None:
    """Sign an audit pack using Ed25519."""
    from autorisk.audit.sign import sign_audit_pack

    resolved_password = _resolve_private_key_password(
        private_key_password=private_key_password,
        private_key_password_env=private_key_password_env,
    )
    res = sign_audit_pack(
        pack,
        private_key_path=private_key,
        private_key_password=resolved_password,
        public_key_path=public_key,
        key_label=key_label,
        include_public_key=embed_public_key,
    )
    click.echo(f"Source: {res.source}")
    click.echo(f"Mode: {res.mode}")
    click.echo(f"Signature: {res.signature_path}")
    click.echo(f"Key ID: {res.key_id}")
    click.echo(f"Checksums SHA256: {res.checksums_sha256}")
    click.echo(f"Manifest SHA256: {res.manifest_sha256}")


@cli.command("audit-attest")
@click.option("--pack", "-p", required=True, help="Audit pack directory or zip bundle")
@click.option("--private-key", required=True, help="Ed25519 private key PEM path")
@click.option("--private-key-password", default=None, help="Private key password (less secure; prefer --private-key-password-env)")
@click.option("--private-key-password-env", default=None, help="Environment variable containing private key password")
@click.option("--public-key", default=None, help="Optional Ed25519 public key PEM path")
@click.option("--key-label", default=None, help="Optional human-readable key label (metadata only)")
@click.option(
    "--embed-public-key/--no-embed-public-key",
    default=False,
    show_default=True,
    help="Embed public key PEM in attestation.json",
)
def audit_attest(
    pack: str,
    private_key: str,
    private_key_password: str | None,
    private_key_password_env: str | None,
    public_key: str | None,
    key_label: str | None,
    embed_public_key: bool,
) -> None:
    """Generate attestation.json over non-checksummed run artifacts."""
    from autorisk.audit.attestation import attest_audit_pack

    resolved_password = _resolve_private_key_password(
        private_key_password=private_key_password,
        private_key_password_env=private_key_password_env,
    )
    res = attest_audit_pack(
        pack,
        private_key_path=private_key,
        private_key_password=resolved_password,
        public_key_path=public_key,
        key_label=key_label,
        include_public_key=embed_public_key,
    )
    click.echo(f"Source: {res.source}")
    click.echo(f"Mode: {res.mode}")
    click.echo(f"Attestation: {res.attestation_path}")
    click.echo(f"Key ID: {res.key_id}")
    click.echo(f"Pack fingerprint: {res.pack_fingerprint}")
    click.echo(f"Finalize record SHA256: {res.finalize_record_sha256}")
    click.echo(f"Validate report SHA256: {res.audit_validate_report_sha256}")


@cli.command("audit-verify")
@click.option("--pack", "-p", required=True, help="Audit pack directory or zip bundle")
@click.option("--strict/--no-strict", default=True, show_default=True, help="Strict verification (fail on any issue)")
@click.option("--public-key", default=None, help="Optional Ed25519 public key PEM path for signature verification")
@click.option("--public-key-dir", default=None, help="Optional trusted public key directory (.pem). Key is selected by signature key_id.")
@click.option("--require-signature/--no-require-signature", default=False, show_default=True, help="Fail if signature.json is missing")
@click.option("--require-public-key/--no-require-public-key", default=False, show_default=True, help="Fail unless --public-key or --public-key-dir is provided")
@click.option(
    "--require-attestation/--no-require-attestation",
    default=False,
    show_default=True,
    help="Fail if attestation.json is missing or unverifiable (recommended for audit decisions)",
)
@click.option(
    "--trust-embedded-public-key/--no-trust-embedded-public-key",
    default=False,
    show_default=True,
    help="Allow verification against public_key_pem embedded in signature.json/attestation.json",
)
@click.option("--revoked-key-id", "revoked_key_ids", multiple=True, help="Revoked signature key_id (repeatable)")
@click.option("--revocation-file", default=None, help="Path to revoked key IDs file (one key_id per line)")
@click.option("--json-out", default=None, help="Optional path to write verification result JSON")
def audit_verify(
    pack: str,
    strict: bool,
    public_key: str | None,
    public_key_dir: str | None,
    require_signature: bool,
    require_public_key: bool,
    require_attestation: bool,
    trust_embedded_public_key: bool,
    revoked_key_ids: tuple[str, ...],
    revocation_file: str | None,
    json_out: str | None,
) -> None:
    """Verify an audit pack using checksums.sha256.txt."""
    from autorisk.audit.verify import verify_audit_pack

    revoked = _load_revoked_key_ids(
        revoked_key_ids=revoked_key_ids,
        revocation_file=revocation_file,
    )
    result = verify_audit_pack(
        pack,
        strict=strict,
        public_key=public_key,
        public_key_dir=public_key_dir,
        require_signature=require_signature,
        require_public_key=require_public_key,
        require_attestation=require_attestation,
        trust_embedded_public_key=trust_embedded_public_key,
        revoked_key_ids=revoked,
    )

    click.echo(f"Source: {result.source}")
    click.echo(f"Mode: {result.mode}")
    click.echo(f"Pack root: {result.pack_root}")
    click.echo(f"Checksums: {result.checksums_path}")
    click.echo(f"Checksums SHA256: {result.checksums_sha256}")
    click.echo(f"Expected files: {result.expected_files}")
    click.echo(f"Verified files: {result.verified_files}")
    click.echo(f"Signature present: {result.signature_present}")
    if result.signature_present:
        click.echo(f"Signature path: {result.signature_path}")
        click.echo(f"Signature key id: {result.signature_key_id}")
        click.echo(f"Signature key source: {result.signature_key_source or 'none'}")
        click.echo(f"Signature verified: {result.signature_verified}")
    click.echo(f"Attestation present: {result.attestation_present}")
    if result.attestation_present:
        click.echo(f"Attestation path: {result.attestation_path}")
        click.echo(f"Attestation key id: {result.attestation_key_id}")
        click.echo(f"Attestation key source: {result.attestation_key_source or 'none'}")
    click.echo(f"Attestation verified: {result.attestation_verified}")
    unchecked_files = list(result.unchecked_files or [])
    click.echo(f"Unchecked files: {len(unchecked_files)}")
    for rel in unchecked_files[:20]:
        click.echo(f"  - {rel}")
    if len(unchecked_files) > 20:
        click.echo(f"  ... ({len(unchecked_files) - 20} more)")
    click.echo(f"Issues: {len(result.issues)}")

    if json_out is not None and str(json_out).strip() != "":
        out_path = Path(json_out).expanduser().resolve()
        out_path.write_text(result.to_json(), encoding="utf-8")
        click.echo(f"Wrote: {out_path}")

    if result.issues:
        for issue in result.issues[:50]:
            click.echo(f"- {issue.kind}: {issue.path} {issue.detail}".rstrip())
        if len(result.issues) > 50:
            click.echo(f"... ({len(result.issues) - 50} more)")
        if strict:
            raise SystemExit(2)


@cli.command("audit-verifier-bundle")
@click.option("--out", required=True, help="Output directory for verifier bundle")
@click.option("--public-key", default=None, help="Trusted public key PEM path")
@click.option("--public-key-dir", default=None, help="Trusted public key directory (.pem)")
@click.option("--revoked-key-id", "revoked_key_ids", multiple=True, help="Revoked signature key_id (repeatable)")
@click.option("--revocation-file", default=None, help="Path to revoked key IDs file (one key_id per line)")
@click.option("--pack-ref", default="PACK_OR_ZIP", help="Pack path placeholder shown in VERIFY.md")
def audit_verifier_bundle(
    out: str,
    public_key: str | None,
    public_key_dir: str | None,
    revoked_key_ids: tuple[str, ...],
    revocation_file: str | None,
    pack_ref: str,
) -> None:
    """Build verifier bundle (trusted keys + revocation file + VERIFY.md)."""
    from autorisk.audit.verifier_bundle import build_verifier_bundle

    revoked = _load_revoked_key_ids(
        revoked_key_ids=revoked_key_ids,
        revocation_file=revocation_file,
    )
    res = build_verifier_bundle(
        output_dir=out,
        public_key=public_key,
        public_key_dir=public_key_dir,
        revoked_key_ids=revoked,
        revocation_file=revocation_file,
        verify_pack_reference=pack_ref,
    )
    click.echo(f"Verifier bundle: {res.output_dir}")
    click.echo(f"Trusted keys dir: {res.trusted_keys_dir}")
    click.echo(f"Trusted keys: {len(res.key_files)}")
    click.echo(f"Revocation file: {res.revocation_path}")
    click.echo(f"Revoked key IDs: {len(res.revoked_key_ids)}")
    click.echo(f"VERIFY: {res.verify_md_path}")


@cli.command("audit-validate")
@click.option("--pack", "-p", required=True, help="Audit pack directory or zip bundle")
@click.option("--schema-dir", default=None, help="Schema directory (default: repo/schemas)")
@click.option(
    "--profile",
    type=click.Choice(["default", "audit-grade"], case_sensitive=False),
    default="default",
    show_default=True,
    help="Validation profile (audit-grade enforces required handoff artifacts)",
)
@click.option("--require-signature/--no-require-signature", default=False, show_default=True, help="Require signature.json")
@click.option("--require-finalize-record/--no-require-finalize-record", default=False, show_default=True, help="Require run_artifacts/finalize_record.json")
@click.option("--require-validate-report/--no-require-validate-report", default=False, show_default=True, help="Require run_artifacts/audit_validate_report.json")
@click.option("--require-policy-snapshot/--no-require-policy-snapshot", default=False, show_default=True, help="Require run_artifacts/policy_snapshot.json")
@click.option("--require-review-artifacts/--no-require-review-artifacts", default=False, show_default=True, help="Require review artifacts (review_apply/review_diff/reviewed_results)")
@click.option("--semantic/--no-semantic", "semantic_checks", default=True, show_default=True, help="Run semantic consistency checks in addition to JSON Schema validation")
@click.option("--enforce/--no-enforce", default=False, show_default=True, help="Exit non-zero when any validation issue is found")
@click.option("--json-out", default=None, help="Optional path to write validation result JSON")
def audit_validate(
    pack: str,
    schema_dir: str | None,
    profile: str,
    require_signature: bool,
    require_finalize_record: bool,
    require_validate_report: bool,
    require_policy_snapshot: bool,
    require_review_artifacts: bool,
    semantic_checks: bool,
    enforce: bool,
    json_out: str | None,
) -> None:
    """Validate audit-pack contract (schema + semantics)."""
    from autorisk.audit.validate import validate_audit_pack

    result = validate_audit_pack(
        pack,
        schema_dir=schema_dir,
        semantic_checks=semantic_checks,
        profile=profile,
        require_signature=require_signature,
        require_finalize_record=require_finalize_record,
        require_validate_report=require_validate_report,
        require_policy_snapshot=require_policy_snapshot,
        require_review_artifacts=require_review_artifacts,
    )

    click.echo(f"Source: {result.source}")
    click.echo(f"Mode: {result.mode}")
    click.echo(f"Pack root: {result.pack_root}")
    click.echo(f"Schema dir: {result.schema_dir}")
    click.echo(f"Files validated: {result.files_validated}")
    click.echo(f"Records validated: {result.records_validated}")
    click.echo(f"Issues: {len(result.issues)}")

    if json_out is not None and str(json_out).strip() != "":
        out_path = Path(json_out).expanduser().resolve()
        out_path.write_text(result.to_json(), encoding="utf-8")
        click.echo(f"Wrote: {out_path}")

    if result.issues:
        for issue in result.issues[:50]:
            line_text = f":{issue.line}" if issue.line is not None else ""
            click.echo(f"- {issue.kind}: {issue.path}{line_text} {issue.detail}".rstrip())
        if len(result.issues) > 50:
            click.echo(f"... ({len(result.issues) - 50} more)")
        if enforce:
            raise SystemExit(2)


@cli.command("audit-handoff")
@click.option("--run-dir", "-r", required=True, help="Run directory containing finalize-run outputs")
@click.option("--out", "-o", "output_dir", default=None, help="Output handoff directory (default: RUN_DIR/handoff_<timestamp>)")
@click.option("--pack-zip", default=None, help="Optional audit pack zip path (default: latest RUN_DIR/audit_pack_*.zip)")
@click.option("--verifier-bundle-dir", default=None, help="Optional verifier bundle directory (default: RUN_DIR/verifier_bundle)")
@click.option("--finalize-record", default=None, help="Optional finalize_record.json path (default: RUN_DIR/finalize_record.json)")
def audit_handoff(
    run_dir: str,
    output_dir: str | None,
    pack_zip: str | None,
    verifier_bundle_dir: str | None,
    finalize_record: str | None,
) -> None:
    """Build single handoff artifact set (PACK.zip + verifier bundle + metadata)."""
    from autorisk.audit.handoff import build_audit_handoff

    result = build_audit_handoff(
        run_dir=run_dir,
        output_dir=output_dir,
        pack_zip=pack_zip,
        verifier_bundle_dir=verifier_bundle_dir,
        finalize_record=finalize_record,
    )

    click.echo(f"Handoff directory: {result.output_dir}")
    click.echo(f"Pack: {result.pack_zip_path}")
    click.echo(f"Verifier bundle zip: {result.verifier_bundle_zip_path}")
    click.echo(f"Finalize record: {result.finalize_record_path}")
    if result.validate_report_path is not None:
        click.echo(f"Validate report: {result.validate_report_path}")
    click.echo(f"Guide: {result.handoff_guide_path}")
    click.echo(f"Checksums: {result.checksums_path}")


@cli.command("audit-handoff-verify")
@click.option("--handoff", "-d", "handoff_dir", required=True, help="Handoff directory containing PACK.zip and verifier_bundle.zip")
@click.option("--strict/--no-strict", default=True, show_default=True, help="Strict checksum verification for bundled PACK.zip")
@click.option("--require-signature/--no-require-signature", default=True, show_default=True, help="Require signature.json in bundled PACK.zip")
@click.option("--require-public-key/--no-require-public-key", default=True, show_default=True, help="Require trusted key anchor from verifier bundle")
@click.option(
    "--require-attestation/--no-require-attestation",
    default=True,
    show_default=True,
    help="Require attestation.json in bundled PACK.zip (--no-require-attestation is diagnostics only)",
)
@click.option(
    "--validate-profile",
    type=click.Choice(["default", "audit-grade"], case_sensitive=False),
    default="audit-grade",
    show_default=True,
    help="Validation profile for bundled PACK.zip",
)
@click.option(
    "--compare-bundled-validate-report/--no-compare-bundled-validate-report",
    default=True,
    show_default=True,
    help="Compare bundled audit_validate_report.json with recomputed validation result",
)
@click.option("--enforce/--no-enforce", default=True, show_default=True, help="Exit non-zero when any issue is found")
@click.option("--json-out", default=None, help="Optional path to write handoff verification result JSON")
def audit_handoff_verify(
    handoff_dir: str,
    strict: bool,
    require_signature: bool,
    require_public_key: bool,
    require_attestation: bool,
    validate_profile: str,
    compare_bundled_validate_report: bool,
    enforce: bool,
    json_out: str | None,
) -> None:
    """Verify handoff folder end-to-end (checksums + audit-verify + attestation + audit-validate)."""
    from autorisk.audit.handoff_verify import verify_audit_handoff

    result = verify_audit_handoff(
        handoff_dir,
        strict=strict,
        require_signature=require_signature,
        require_public_key=require_public_key,
        require_attestation=require_attestation,
        validate_profile=validate_profile,
        compare_bundled_validate_report=compare_bundled_validate_report,
    )

    click.echo(f"Handoff dir: {result.handoff_dir}")
    click.echo(f"Checksums: {result.checksums_path}")
    click.echo(f"Listed files: {result.listed_files}")
    click.echo(f"Verified files: {result.verified_files}")
    click.echo(f"Pack: {result.pack_path}")
    click.echo(f"Verifier bundle: {result.verifier_bundle_zip_path}")
    click.echo(f"Finalize record: {result.finalize_record_path}")
    if result.validate_report_path is not None:
        click.echo(f"Bundled validate report: {result.validate_report_path}")
    click.echo(f"audit-verify ok: {result.audit_verify_ok}")
    click.echo(f"audit-validate ok: {result.audit_validate_ok}")
    click.echo(f"Attestation present: {result.attestation_present}")
    click.echo(f"Attestation verified: {result.attestation_verified}")
    if result.attestation_present:
        click.echo(f"Attestation key id: {result.attestation_key_id}")
        click.echo(f"Attestation key source: {result.attestation_key_source or 'none'}")
    if result.bundled_validate_report_match is not None:
        click.echo(f"validate report match: {result.bundled_validate_report_match}")
    click.echo(f"Issues: {len(result.issues)}")

    if json_out is not None and str(json_out).strip() != "":
        out_path = Path(json_out).expanduser().resolve()
        out_path.write_text(result.to_json(), encoding="utf-8")
        click.echo(f"Wrote: {out_path}")

    if result.issues:
        for issue in result.issues[:50]:
            click.echo(f"- {issue.kind}: {issue.path} {issue.detail}".rstrip())
        if len(result.issues) > 50:
            click.echo(f"... ({len(result.issues) - 50} more)")
        if enforce:
            raise SystemExit(2)


@cli.command("review-approve")
@click.option("--run-dir", "-r", required=True, help="Run directory containing cosmos_results.json")
@click.option("--rank", type=int, required=True, help="candidate_rank to approve/override")
@click.option(
    "--severity",
    type=click.Choice(["NONE", "LOW", "MEDIUM", "HIGH"], case_sensitive=False),
    required=True,
    help="Final severity after human review",
)
@click.option("--reason", required=True, help="Human reviewer reason (kept for audit)")
@click.option("--evidence", multiple=True, help="Optional evidence references (repeatable)")
@click.option("--operator", "operator_user", default=None, help="Override operator username (default: env)")
@click.option("--log", "log_path", default=None, help="Optional review log path (default: RUN_DIR/review_log.jsonl)")
def review_approve(
    run_dir: str,
    rank: int,
    severity: str,
    reason: str,
    evidence: tuple[str, ...],
    operator_user: str | None,
    log_path: str | None,
) -> None:
    """Append one human review decision into review_log.jsonl."""
    from autorisk.review.log import append_review_decision

    path, record, rec_sha = append_review_decision(
        run_dir=run_dir,
        candidate_rank=rank,
        severity_after=severity,
        reason=reason,
        evidence_refs=list(evidence),
        operator_user=operator_user,
        log_path=log_path,
    )

    click.echo(f"Review log: {path}")
    click.echo(f"Recorded: rank={rank} {record['decision_before']['severity']} -> {record['decision_after']['severity']}")
    click.echo(f"Review record sha256: {rec_sha}")


@cli.command("review-apply")
@click.option("--run-dir", "-r", required=True, help="Run directory containing cosmos_results.json")
@click.option("--log", "log_path", default=None, help="Optional review log path (default: RUN_DIR/review_log.jsonl)")
@click.option("--out", "output_path", default=None, help="Output reviewed results path (default: RUN_DIR/cosmos_results_reviewed.json)")
@click.option("--allow-stale/--no-allow-stale", default=False, show_default=True, help="Apply reviews even if results_sha256 differs")
def review_apply(run_dir: str, log_path: str | None, output_path: str | None, allow_stale: bool) -> None:
    """Apply review_log.jsonl decisions into cosmos_results_reviewed.json (non-destructive)."""
    from autorisk.review.log import apply_review_overrides

    res = apply_review_overrides(
        run_dir=run_dir,
        log_path=log_path,
        output_path=output_path,
        allow_stale=allow_stale,
        write_report=True,
    )

    click.echo(f"Input: {res.input_results}")
    click.echo(f"Review log: {res.log_path}")
    click.echo(f"Output: {res.output_results}")
    click.echo(f"Diff report: {res.diff_report_path}")
    click.echo(f"Applied: {res.applied}")
    click.echo(f"Skipped stale: {res.skipped_stale}")
    click.echo(f"Skipped missing rank: {res.skipped_missing}")


@cli.command("policy-check")
@click.option("--run-dir", "-r", required=True, help="Run directory containing cosmos_results.json")
@click.option("--policy", default=None, help="Policy YAML path (default: configs/policy.yaml if present)")
@click.option("--review-log", default=None, help="Optional review log path (default: RUN_DIR/review_log.jsonl)")
@click.option("--report-out", default=None, help="Optional policy report path (default: RUN_DIR/policy_report.json)")
@click.option("--queue-out", default=None, help="Optional review queue path (default: RUN_DIR/review_queue.json)")
@click.option("--snapshot-out", default=None, help="Optional policy snapshot path (default: RUN_DIR/policy_snapshot.json)")
@click.option("--allow-stale", "allow_stale", flag_value=True, default=None, help="Override policy to allow stale review logs")
@click.option("--no-allow-stale", "allow_stale", flag_value=False, help="Override policy to disallow stale review logs")
@click.option("--enforce/--no-enforce", default=False, show_default=True, help="Exit non-zero if policy check fails")
def policy_check(
    run_dir: str,
    policy: str | None,
    review_log: str | None,
    report_out: str | None,
    queue_out: str | None,
    snapshot_out: str | None,
    allow_stale: bool | None,
    enforce: bool,
) -> None:
    """Enforce review gating policy and emit policy_report/review_queue."""
    from autorisk.policy.check import run_policy_check

    res = run_policy_check(
        run_dir=run_dir,
        policy_path=policy,
        review_log=review_log,
        report_path=report_out,
        queue_path=queue_out,
        snapshot_path=snapshot_out,
        allow_stale=allow_stale,
        write_outputs=True,
    )

    click.echo(f"Policy report: {res.report_path}")
    click.echo(f"Review queue: {res.queue_path}")
    click.echo(f"Policy snapshot: {res.snapshot_path}")
    click.echo(f"Policy source: {res.policy_source.get('policy_path') or res.policy_source.get('source_type')}")
    click.echo(f"Passed: {res.passed}")
    click.echo(f"Required review: {res.required_review_count}")
    click.echo(f"Reviewed (valid): {res.reviewed_count_valid}")
    click.echo(f"Reviewed (stale): {res.reviewed_count_stale}")
    click.echo(f"Missing review: {res.missing_review_count}")

    if res.violations:
        for v in res.violations[:50]:
            click.echo(
                f"- rank={v.get('candidate_rank')} sev={v.get('severity')} reasons={','.join(v.get('violation_reasons', []))}"
            )
        if len(res.violations) > 50:
            click.echo(f"... ({len(res.violations) - 50} more)")
        if enforce:
            raise SystemExit(2)


@cli.command("finalize-run")
@click.option("--run-dir", "-r", required=True, help="Run directory containing cosmos_results.json")
@click.option("--policy", default=None, help="Policy YAML path (default: configs/policy.yaml if present)")
@click.option("--review-log", default=None, help="Optional review log path (default: RUN_DIR/review_log.jsonl)")
@click.option("--input-video", "-i", default=None, help="Optional source input video path (for SHA256 provenance)")
@click.option("--out", "-o", "output_dir", default=None, help="Audit pack output directory")
@click.option("--include-clips/--no-include-clips", default=True, show_default=True, help="Copy candidate clips into audit pack")
@click.option("--zip/--no-zip", "create_zip", default=True, show_default=True, help="Create zip bundle for handoff")
@click.option("--allow-stale", "allow_stale", flag_value=True, default=None, help="Override policy to allow stale review logs")
@click.option("--no-allow-stale", "allow_stale", flag_value=False, help="Override policy to disallow stale review logs")
@click.option("--enforce/--no-enforce", default=True, show_default=True, help="Fail finalization on policy or integrity violations")
@click.option("--audit-grade/--no-audit-grade", default=False, show_default=True, help="Enable strict audit mode (signature/trusted key/enforce)")
@click.option("--sign-private-key", default=None, help="Optional Ed25519 private key PEM for audit-sign")
@click.option("--sign-private-key-password", default=None, help="Private key password (less secure; prefer --sign-private-key-password-env)")
@click.option("--sign-private-key-password-env", default=None, help="Environment variable containing private key password")
@click.option("--sign-public-key", default=None, help="Optional Ed25519 public key PEM for signature verification")
@click.option("--sign-public-key-dir", default=None, help="Optional trusted public key directory for verification")
@click.option("--require-signature/--no-require-signature", default=False, show_default=True, help="Require signature.json during final verify")
@click.option("--require-trusted-key/--no-require-trusted-key", default=False, show_default=True, help="Require explicit trusted key anchor (--sign-public-key or --sign-public-key-dir)")
@click.option("--verifier-bundle-out", default=None, help="Optional output directory for verifier bundle")
@click.option("--write-verifier-bundle/--no-write-verifier-bundle", default=None, help="Generate verifier bundle (keys/trusted + revoked_key_ids + VERIFY.md)")
@click.option("--handoff-out", default=None, help="Optional output directory for audit handoff folder")
@click.option("--write-handoff/--no-write-handoff", default=None, help="Generate handoff folder (PACK.zip + verifier bundle + finalize record + HANDOFF.md)")
@click.option(
    "--embed-public-key/--no-embed-public-key",
    default=False,
    show_default=True,
    help="Embed public key PEM in signature.json when signing",
)
@click.option(
    "--trust-embedded-public-key/--no-trust-embedded-public-key",
    default=False,
    show_default=True,
    help="Allow fallback to embedded public key when --sign-public-key is not provided",
)
@click.option("--revoked-key-id", "revoked_key_ids", multiple=True, help="Revoked signature key_id (repeatable)")
@click.option("--revocation-file", default=None, help="Path to revoked key IDs file (one key_id per line)")
@click.pass_context
def finalize_run(
    ctx: click.Context,
    run_dir: str,
    policy: str | None,
    review_log: str | None,
    input_video: str | None,
    output_dir: str | None,
    include_clips: bool,
    create_zip: bool,
    allow_stale: bool | None,
    enforce: bool,
    audit_grade: bool,
    sign_private_key: str | None,
    sign_private_key_password: str | None,
    sign_private_key_password_env: str | None,
    sign_public_key: str | None,
    sign_public_key_dir: str | None,
    require_signature: bool,
    require_trusted_key: bool,
    verifier_bundle_out: str | None,
    write_verifier_bundle: bool | None,
    handoff_out: str | None,
    write_handoff: bool | None,
    embed_public_key: bool,
    trust_embedded_public_key: bool,
    revoked_key_ids: tuple[str, ...],
    revocation_file: str | None,
) -> None:
    """Apply review, enforce policy, build audit pack, and verify integrity."""
    from autorisk.audit.attestation import attest_audit_pack
    from autorisk.audit.pack import build_audit_pack
    from autorisk.audit.sign import sign_audit_pack
    from autorisk.audit.validate import validate_audit_pack
    from autorisk.audit.verify import verify_audit_pack
    from autorisk.audit.handoff import build_audit_handoff
    from autorisk.audit.verifier_bundle import build_verifier_bundle
    from autorisk.policy.check import resolve_policy, run_policy_check
    from autorisk.review.log import apply_review_overrides

    if audit_grade:
        enforce = True
        require_signature = True
        require_trusted_key = True
        trust_embedded_public_key = False
        click.echo("[audit-grade] enforce=true require_signature=true require_trusted_key=true trust_embedded_public_key=false")

    if audit_grade and (sign_private_key is None or str(sign_private_key).strip() == ""):
        click.echo("Error: --audit-grade requires --sign-private-key", err=True)
        raise SystemExit(2)

    resolved_sign_password = _resolve_private_key_password(
        private_key_password=sign_private_key_password,
        private_key_password_env=sign_private_key_password_env,
    )
    revoked = _load_revoked_key_ids(
        revoked_key_ids=revoked_key_ids,
        revocation_file=revocation_file,
    )
    write_verifier_bundle_effective = bool(write_verifier_bundle) if write_verifier_bundle is not None else bool(audit_grade)
    write_handoff_effective = bool(write_handoff) if write_handoff is not None else bool(audit_grade)
    if write_handoff_effective:
        write_verifier_bundle_effective = True
    if write_handoff_effective and not create_zip:
        click.echo("Error: --write-handoff requires --zip (handoff requires PACK.zip)", err=True)
        raise SystemExit(2)

    effective_policy, policy_source = resolve_policy(
        policy_path=policy,
        required_review_severities=None,
        require_parse_failure_review=None,
        require_error_review=None,
        allow_stale=allow_stale,
    )
    allow_stale_effective = bool(effective_policy["allow_stale"])
    click.echo(
        f"[0/4] policy: source={policy_source.get('policy_path') or policy_source.get('source_type')} allow_stale={allow_stale_effective}"
    )

    apply_res = apply_review_overrides(
        run_dir=run_dir,
        log_path=review_log,
        output_path=None,
        allow_stale=allow_stale_effective,
        write_report=True,
    )
    click.echo(f"[1/4] review-apply: {apply_res.output_results}")
    click.echo(f"        applied={apply_res.applied} stale={apply_res.skipped_stale} missing={apply_res.skipped_missing}")

    policy_res = run_policy_check(
        run_dir=run_dir,
        policy_path=policy,
        review_log=review_log,
        allow_stale=allow_stale_effective,
        write_outputs=True,
    )
    click.echo(f"[2/4] policy-check: passed={policy_res.passed} missing={policy_res.missing_review_count}")
    click.echo(f"        report={policy_res.report_path}")
    click.echo(f"        queue={policy_res.queue_path}")
    if enforce and not policy_res.passed:
        raise SystemExit(2)

    cfg = (ctx.obj or {}).get("cfg")
    pack_res = build_audit_pack(
        run_dir=run_dir,
        cfg=cfg,
        output_dir=output_dir,
        input_video=input_video,
        review_log=review_log,
        include_clips=include_clips,
        create_zip=create_zip,
    )
    click.echo(f"[3/4] audit-pack: {pack_res.output_dir}")
    click.echo(f"        fingerprint={pack_res.checksums_sha256}")
    if pack_res.zip_path is not None:
        click.echo(f"        zip={pack_res.zip_path}")

    verify_target = pack_res.zip_path if pack_res.zip_path is not None else pack_res.output_dir
    if sign_private_key is not None and str(sign_private_key).strip() != "":
        has_sign_public_key = sign_public_key is not None and str(sign_public_key).strip() != ""
        has_sign_public_key_dir = sign_public_key_dir is not None and str(sign_public_key_dir).strip() != ""
        if not has_sign_public_key and not has_sign_public_key_dir:
            warning_message = (
                "--sign-private-key was provided without --sign-public-key/--sign-public-key-dir. "
                "Authenticity should be anchored to an external trusted key."
            )
            if audit_grade or require_trusted_key:
                click.echo(f"Error: {warning_message}", err=True)
                raise SystemExit(2)
            click.echo(f"Warning: {warning_message}", err=True)
        sign_res = sign_audit_pack(
            verify_target,
            private_key_path=sign_private_key,
            private_key_password=resolved_sign_password,
            public_key_path=sign_public_key,
            include_public_key=embed_public_key,
        )
        click.echo(f"[3.5/4] audit-sign: {sign_res.signature_path}")
        click.echo(f"        key_id={sign_res.key_id}")

    has_trusted_anchor = (
        (sign_public_key is not None and str(sign_public_key).strip() != "")
        or (sign_public_key_dir is not None and str(sign_public_key_dir).strip() != "")
    )
    if require_trusted_key and not has_trusted_anchor:
        click.echo("Error: --require-trusted-key requires --sign-public-key or --sign-public-key-dir", err=True)
        raise SystemExit(2)

    verify_res = verify_audit_pack(
        verify_target,
        strict=True,
        public_key=sign_public_key,
        public_key_dir=sign_public_key_dir,
        require_signature=require_signature,
        require_public_key=require_trusted_key,
        trust_embedded_public_key=trust_embedded_public_key,
        revoked_key_ids=revoked,
    )
    click.echo(
        f"[4/4] audit-verify: issues={len(verify_res.issues)} expected={verify_res.expected_files} verified={verify_res.verified_files}"
    )
    if verify_res.signature_present:
        click.echo(
            f"        signature_verified={verify_res.signature_verified} key_id={verify_res.signature_key_id}"
        )

    verifier_bundle_path = ""
    if write_verifier_bundle_effective:
        bundle_out = (
            Path(verifier_bundle_out).expanduser().resolve()
            if verifier_bundle_out is not None and str(verifier_bundle_out).strip() != ""
            else (Path(run_dir).resolve() / "verifier_bundle")
        )
        bundle_res = build_verifier_bundle(
            output_dir=bundle_out,
            public_key=sign_public_key,
            public_key_dir=sign_public_key_dir,
            revoked_key_ids=revoked,
            revocation_file=revocation_file,
            verify_pack_reference=(pack_res.zip_path.name if pack_res.zip_path is not None else pack_res.output_dir.name),
        )
        verifier_bundle_path = str(bundle_res.output_dir)
        click.echo(f"[4.5/4] verifier-bundle: {bundle_res.output_dir}")

    run_dir_path = Path(run_dir).resolve()
    validate_report_path = run_dir_path / "audit_validate_report.json"
    pack_finalize_record_path = pack_res.output_dir / "run_artifacts" / "finalize_record.json"
    pack_validate_report_path = pack_res.output_dir / "run_artifacts" / "audit_validate_report.json"

    # Stage 1: baseline validation before finalize metadata injection.
    validate_res = validate_audit_pack(
        verify_target,
        semantic_checks=True,
        profile="default",
    )
    validate_report_payload = validate_res.to_json()
    validate_report_path.write_text(validate_report_payload, encoding="utf-8")
    validate_report_sha256 = _sha256_file(validate_report_path)

    handoff_path = ""
    handoff_checksums_sha256 = ""
    handoff_pack_zip_sha256 = ""
    handoff_verifier_bundle_zip_sha256 = ""
    revocation_file_sha256 = ""
    revocation_path_resolved = ""
    if revocation_file is not None and str(revocation_file).strip() != "":
        rev_path = Path(revocation_file).expanduser().resolve()
        if rev_path.exists() and rev_path.is_file():
            revocation_path_resolved = str(rev_path)
            revocation_file_sha256 = _sha256_file(rev_path)

    finalize_record: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": _utc_now_iso(),
        "run_dir": str(run_dir_path),
        "pack_dir": str(pack_res.output_dir),
        "zip_path": str(pack_res.zip_path) if pack_res.zip_path is not None else "",
        "autorisk_version": _autorisk_version(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pack_fingerprint": pack_res.checksums_sha256,
        "signature_present": bool(verify_res.signature_present),
        "signature_verified": verify_res.signature_verified,
        "signature_key_id": verify_res.signature_key_id,
        "signature_key_source": verify_res.signature_key_source,
        "policy_source": policy_res.policy_source,
        "policy_sha256": str(policy_res.policy_source.get("policy_sha256", "")),
        "revocation_file": revocation_path_resolved,
        "revocation_file_sha256": revocation_file_sha256,
        "revoked_key_ids_count": len(revoked),
        "verification_issues": len(verify_res.issues),
        "enforce": bool(enforce),
        "audit_grade": bool(audit_grade),
        "require_signature": bool(require_signature),
        "require_trusted_key": bool(require_trusted_key),
        "trust_embedded_public_key": bool(trust_embedded_public_key),
        "verifier_bundle_path": verifier_bundle_path,
        "validate_ok": bool(validate_res.ok),
        "validate_issues_count": len(validate_res.issues),
        "validate_report_path": "run_artifacts/audit_validate_report.json",
        "validate_report_sha256": validate_report_sha256,
        "handoff_path": handoff_path,
        "handoff_checksums_sha256": handoff_checksums_sha256,
        "handoff_pack_zip_sha256": handoff_pack_zip_sha256,
        "handoff_verifier_bundle_zip_sha256": handoff_verifier_bundle_zip_sha256,
    }
    finalize_record_path = run_dir_path / "finalize_record.json"

    def _sync_run_finalize(record: dict[str, object]) -> None:
        _write_json(finalize_record_path, record)

    def _sync_pack_metadata(record: dict[str, object], report_payload: str) -> None:
        _write_json(pack_finalize_record_path, record)
        pack_validate_report_path.write_text(report_payload, encoding="utf-8")
        if pack_res.zip_path is not None:
            _upsert_zip_member(
                pack_res.zip_path,
                "run_artifacts/finalize_record.json",
                json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"),
            )
            _upsert_zip_member(
                pack_res.zip_path,
                "run_artifacts/audit_validate_report.json",
                report_payload.encode("utf-8"),
            )

    _sync_run_finalize(finalize_record)
    _sync_pack_metadata(finalize_record, validate_report_payload)

    # Stage 2: final validation with requested profile after finalize artifacts are present.
    validate_profile = "audit-grade" if audit_grade else "default"
    validate_res = validate_audit_pack(
        verify_target,
        semantic_checks=True,
        profile=validate_profile,
    )
    validate_report_payload = validate_res.to_json()
    validate_report_path.write_text(validate_report_payload, encoding="utf-8")
    validate_report_sha256 = _sha256_file(validate_report_path)
    finalize_record["validate_ok"] = bool(validate_res.ok)
    finalize_record["validate_issues_count"] = len(validate_res.issues)
    finalize_record["validate_report_sha256"] = validate_report_sha256
    _sync_run_finalize(finalize_record)
    _sync_pack_metadata(finalize_record, validate_report_payload)
    click.echo(f"[4.6/4] audit-validate: issues={len(validate_res.issues)} profile={validate_profile}")

    handoff_res = None
    if write_handoff_effective:
        if pack_res.zip_path is None:
            click.echo("Error: handoff requires a zip pack, but zip is missing", err=True)
            raise SystemExit(2)
        if verifier_bundle_path == "":
            click.echo("Error: handoff requires verifier bundle, but it was not generated", err=True)
            raise SystemExit(2)
        handoff_output_dir = (
            Path(handoff_out).expanduser().resolve()
            if handoff_out is not None and str(handoff_out).strip() != ""
            else (run_dir_path / "handoff_latest")
        )
        handoff_res = build_audit_handoff(
            run_dir=run_dir_path,
            output_dir=handoff_output_dir,
            pack_zip=pack_res.zip_path,
            verifier_bundle_dir=Path(verifier_bundle_path),
            finalize_record=finalize_record_path,
        )
        handoff_path = str(handoff_res.output_dir)
        handoff_checksums_sha256 = _sha256_file(handoff_res.checksums_path)
        handoff_verifier_bundle_zip_sha256 = _sha256_file(handoff_res.verifier_bundle_zip_path)
        finalize_record["handoff_path"] = handoff_path
        finalize_record["handoff_checksums_sha256"] = handoff_checksums_sha256
        finalize_record["handoff_verifier_bundle_zip_sha256"] = handoff_verifier_bundle_zip_sha256
        _sync_run_finalize(finalize_record)
        pack_finalize_record = dict(finalize_record)
        pack_finalize_record["handoff_path"] = ""
        pack_finalize_record["handoff_checksums_sha256"] = ""
        pack_finalize_record["handoff_pack_zip_sha256"] = ""
        pack_finalize_record["handoff_verifier_bundle_zip_sha256"] = ""
        pack_finalize_record["handoff_anchor_checksums_sha256"] = handoff_checksums_sha256
        pack_finalize_record["handoff_anchor_verifier_bundle_zip_sha256"] = handoff_verifier_bundle_zip_sha256
        _sync_pack_metadata(pack_finalize_record, validate_report_payload)
    else:
        pack_finalize_record = dict(finalize_record)

    if sign_private_key is not None and str(sign_private_key).strip() != "":
        attest_res = attest_audit_pack(
            verify_target,
            private_key_path=sign_private_key,
            private_key_password=resolved_sign_password,
            public_key_path=sign_public_key,
            include_public_key=embed_public_key,
        )
        click.echo(f"[4.65/4] audit-attest: {attest_res.attestation_path}")
        click.echo(f"        key_id={attest_res.key_id}")

    final_verify_res = verify_res
    if audit_grade:
        final_verify_res = verify_audit_pack(
            verify_target,
            strict=True,
            public_key=sign_public_key,
            public_key_dir=sign_public_key_dir,
            require_signature=True,
            require_public_key=True,
            require_attestation=True,
            trust_embedded_public_key=False,
            revoked_key_ids=revoked,
        )
        click.echo(
            f"[4.66/4] audit-verify(final): issues={len(final_verify_res.issues)} "
            f"signature={final_verify_res.signature_verified} attestation={final_verify_res.attestation_verified}"
        )

    if enforce and (verify_res.issues or validate_res.issues or final_verify_res.issues):
        raise SystemExit(2)

    if handoff_res is not None and pack_res.zip_path is not None:
        shutil.copy2(pack_res.zip_path, handoff_res.pack_zip_path)
        handoff_pack_zip_sha256 = _sha256_file(handoff_res.pack_zip_path)
        finalize_record["handoff_pack_zip_sha256"] = handoff_pack_zip_sha256
        _sync_run_finalize(finalize_record)
        handoff_finalize_record_path = Path(handoff_path) / "finalize_record.json"
        _write_json(handoff_finalize_record_path, finalize_record)
        click.echo(f"[4.7/4] audit-handoff: {handoff_res.output_dir}")

    click.echo(f"[4.8/4] finalize-record: {finalize_record_path}")

if __name__ == "__main__":
    cli()
