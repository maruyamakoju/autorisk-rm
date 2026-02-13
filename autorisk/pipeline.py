"""End-to-end pipeline orchestration: B1 → B2 → B3 → B4 → B5 → Report."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig

from autorisk.cosmos.infer import CosmosInferenceEngine
from autorisk.cosmos.schema import CosmosResponse
from autorisk.eval.ablation import AblationResult, AblationRunner
from autorisk.eval.evaluator import EvalReport, Evaluator
from autorisk.mining.fuse import Candidate, SignalFuser
from autorisk.ranking import rank_responses
from autorisk.report.build_report import ReportBuilder
from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)


class Pipeline:
    """AutoRisk-RM end-to-end pipeline."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def run(
        self,
        video_path: str | Path,
        output_dir: str | Path | None = None,
        skip_eval: bool = False,
        skip_ablation: bool = False,
        skip_report: bool = False,
    ) -> dict:
        """Execute the full pipeline.

        Args:
            video_path: Path to input video.
            output_dir: Output directory (overrides config).
            skip_eval: Skip evaluation stage.
            skip_ablation: Skip ablation stage.
            skip_report: Skip report generation.

        Returns:
            Dict with all pipeline outputs.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if output_dir is None:
            output_dir = Path(self.cfg.general.output_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: dict = {"video": str(video_path), "output_dir": str(output_dir)}

        # B1: Candidate extraction
        log.info("=== B1: Candidate Mining ===")
        fuser = SignalFuser(self.cfg)
        candidates = fuser.extract_candidates(video_path, output_dir)
        results["n_candidates"] = len(candidates)
        log.info("Extracted %d candidates", len(candidates))

        if not candidates:
            log.warning("No candidates found, stopping pipeline")
            results["status"] = "no_candidates"
            return results

        # B2: Cosmos inference
        log.info("=== B2: Cosmos Inference ===")
        engine = CosmosInferenceEngine(self.cfg)
        responses = engine.infer_batch(candidates, output_dir=output_dir)
        engine.save_results(responses, output_dir)
        results["n_responses"] = len(responses)

        # B3: Ranking
        log.info("=== B3: Risk Ranking ===")
        ranked = rank_responses(responses)
        results["top_severity"] = ranked[0].assessment.severity if ranked else "N/A"

        # B4: Evaluation (optional, requires GT)
        eval_report: EvalReport | None = None
        if not skip_eval:
            log.info("=== B4: Evaluation ===")
            evaluator = Evaluator(self.cfg.eval.severity_labels)

            gt_labels = evaluator.load_gt_labels(self.cfg.eval.gt_path)
            checklist_gt = evaluator.load_checklist_gt(self.cfg.eval.checklist_path)

            if gt_labels:
                eval_report = evaluator.evaluate(ranked, gt_labels, checklist_gt)
                evaluator.save_report(eval_report, output_dir)
                results["accuracy"] = eval_report.accuracy
                results["macro_f1"] = eval_report.macro_f1
            else:
                # Still run checklist (auto-heuristic)
                eval_report = evaluator.evaluate(ranked)
                evaluator.save_report(eval_report, output_dir)

            results["checklist_mean"] = eval_report.checklist_means.get("mean_total", 0)

        # B5: Ablation (optional)
        ablation_results: list[AblationResult] = []
        if not skip_ablation:
            log.info("=== B5: Ablation Study ===")
            runner = AblationRunner(self.cfg)

            # Load GT independently (evaluator may not exist if skip_eval)
            gt_labels_abl: dict[str, str] = {}
            try:
                abl_evaluator = Evaluator()
                gt_labels_abl = abl_evaluator.load_gt_labels(self.cfg.eval.gt_path)
            except Exception:
                pass

            signal_abl = runner.run_signal_ablation(
                video_path, output_dir, gt_labels_abl or None,
            )
            ablation_results.extend(signal_abl)
            AblationRunner.save_results(ablation_results, output_dir)

        # Report
        if not skip_report:
            log.info("=== Report Generation ===")
            builder = ReportBuilder(self.cfg)
            report_path = builder.build(
                ranked,
                eval_report=eval_report,
                ablation_results=ablation_results or None,
                output_dir=output_dir,
                video_name=video_path.name,
            )
            results["report_path"] = str(report_path)

        results["status"] = "completed"
        log.info("=== Pipeline Complete ===")
        return results

    def mine_only(
        self, video_path: str | Path, output_dir: str | Path,
    ) -> list[Candidate]:
        """Run only the mining stage (B1)."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fuser = SignalFuser(self.cfg)
        return fuser.extract_candidates(video_path, output_dir)

    def infer_only(
        self, clips_dir: str | Path, output_dir: str | Path,
    ) -> list[CosmosResponse]:
        """Run only Cosmos inference (B2) on pre-extracted clips."""
        clips_dir = Path(clips_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        clip_files = sorted(clips_dir.glob("*.mp4"))
        if not clip_files:
            log.warning("No MP4 clips found in %s", clips_dir)
            return []

        candidates = [
            Candidate(
                rank=i,
                peak_time_sec=0,
                start_sec=0,
                end_sec=0,
                fused_score=0,
                clip_path=str(p),
            )
            for i, p in enumerate(clip_files, 1)
        ]

        engine = CosmosInferenceEngine(self.cfg)
        responses = engine.infer_batch(candidates, output_dir=output_dir)
        engine.save_results(responses, output_dir)
        return responses
