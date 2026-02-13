"""Ablation study module (B5): pipeline and signal ablation experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""
    mode: str
    description: str
    accuracy: float = 0.0
    macro_f1: float = 0.0
    checklist_mean: float = 0.0
    n_candidates: int = 0
    details: dict = field(default_factory=dict)


class AblationRunner:
    """Run pipeline and signal ablation experiments.

    Pipeline ablation modes:
        - mining_only: Only run B1 candidate extraction, no Cosmos
        - cosmos_only: Skip B1, use all segments for Cosmos
        - full: Complete pipeline (B1 + B2 + B3)

    Signal ablation modes:
        - audio_only: Only audio scorer
        - motion_only: Only motion scorer
        - proximity_only: Only proximity scorer
        - all_signals: All scorers combined (default)
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def run_signal_ablation(
        self,
        video_path: Path,
        output_dir: Path,
        gt_labels: dict[str, str] | None = None,
    ) -> list[AblationResult]:
        """Run signal ablation: each signal solo vs all combined.

        For each mode, modifies config to enable only specific signals,
        runs mining, then evaluates if GT is available.
        """
        from autorisk.eval.evaluator import Evaluator
        from autorisk.mining.fuse import SignalFuser

        signal_configs = {
            "audio_only": {"audio": True, "motion": False, "proximity": False},
            "motion_only": {"audio": False, "motion": True, "proximity": False},
            "proximity_only": {"audio": False, "motion": False, "proximity": True},
            "all_signals": {"audio": True, "motion": True, "proximity": True},
        }

        results: list[AblationResult] = []
        evaluator = Evaluator()

        for mode, signals in signal_configs.items():
            log.info("Signal ablation: %s", mode)
            mode_dir = output_dir / "ablation" / mode
            mode_dir.mkdir(parents=True, exist_ok=True)

            # Create modified config
            overrides = {
                "mining": {
                    "audio": {"enabled": signals["audio"]},
                    "motion": {"enabled": signals["motion"]},
                    "proximity": {"enabled": signals["proximity"]},
                }
            }
            mode_cfg = OmegaConf.merge(self.cfg, OmegaConf.create(overrides))

            fuser = SignalFuser(mode_cfg)
            candidates = fuser.extract_candidates(video_path, mode_dir)

            result = AblationResult(
                mode=mode,
                description=f"Signals: {signals}",
                n_candidates=len(candidates),
            )

            # If GT available and Cosmos results exist, evaluate
            if gt_labels and candidates:
                try:
                    from autorisk.cosmos.infer import CosmosInferenceEngine
                    from autorisk.ranking import rank_responses

                    engine = CosmosInferenceEngine(mode_cfg)
                    responses = engine.infer_batch(candidates)
                    responses = rank_responses(responses)

                    eval_report = evaluator.evaluate(responses, gt_labels)
                    result.accuracy = eval_report.accuracy
                    result.macro_f1 = eval_report.macro_f1
                    result.checklist_mean = eval_report.checklist_means.get("mean_total", 0)
                except Exception as e:
                    log.warning("Evaluation failed for %s: %s", mode, e)

            results.append(result)
            log.info(
                "  %s: %d candidates, acc=%.3f, f1=%.3f",
                mode, result.n_candidates, result.accuracy, result.macro_f1,
            )

        return results

    def run_pipeline_ablation(
        self,
        video_path: Path,
        output_dir: Path,
        gt_labels: dict[str, str] | None = None,
    ) -> list[AblationResult]:
        """Run pipeline ablation: mining_only / cosmos_only / full."""
        from autorisk.cosmos.infer import CosmosInferenceEngine
        from autorisk.eval.evaluator import Evaluator
        from autorisk.mining.fuse import SignalFuser
        from autorisk.ranking import rank_responses

        evaluator = Evaluator()
        results: list[AblationResult] = []

        # 1. mining_only: extract candidates without Cosmos
        log.info("Pipeline ablation: mining_only")
        mode_dir = output_dir / "ablation" / "mining_only"
        mode_dir.mkdir(parents=True, exist_ok=True)

        fuser = SignalFuser(self.cfg)
        candidates = fuser.extract_candidates(video_path, mode_dir)
        results.append(AblationResult(
            mode="mining_only",
            description="Candidate extraction only, no Cosmos inference",
            n_candidates=len(candidates),
        ))

        # 2. full: complete pipeline
        log.info("Pipeline ablation: full")
        mode_dir = output_dir / "ablation" / "full"
        mode_dir.mkdir(parents=True, exist_ok=True)

        candidates_full = fuser.extract_candidates(video_path, mode_dir)
        if candidates_full:
            try:
                engine = CosmosInferenceEngine(self.cfg)
                responses = engine.infer_batch(candidates_full)
                ranked = rank_responses(responses)

                full_result = AblationResult(
                    mode="full",
                    description="Full pipeline: B1 + B2 + B3",
                    n_candidates=len(candidates_full),
                )

                if gt_labels:
                    eval_report = evaluator.evaluate(ranked, gt_labels)
                    full_result.accuracy = eval_report.accuracy
                    full_result.macro_f1 = eval_report.macro_f1
                    full_result.checklist_mean = eval_report.checklist_means.get(
                        "mean_total", 0,
                    )

                results.append(full_result)
            except Exception as e:
                log.warning("Full pipeline ablation failed: %s", e)
                results.append(AblationResult(
                    mode="full",
                    description=f"Failed: {e}",
                    n_candidates=len(candidates_full),
                ))

        return results

    def run_minimal_ablation(
        self,
        cosmos_results_path: Path,
        gt_labels: dict[str, str],
        output_dir: Path,
    ) -> list[AblationResult]:
        """Run minimal 3-mode ablation using existing Cosmos results.

        Modes:
            1. baseline: Map fused_score to severity via thresholds (no Cosmos)
            2. cosmos_video: Use existing Cosmos video results (main pipeline)

        This avoids re-running expensive Cosmos inference.
        Cosmos-1frame can be added later if time allows.

        Args:
            cosmos_results_path: Path to cosmos_results.json from main run.
            gt_labels: Ground truth severity labels.
            output_dir: Output directory for ablation results.

        Returns:
            List of AblationResult.
        """
        from autorisk.cosmos.schema import (
            CosmosRequest,
            CosmosResponse,
            RiskAssessment,
        )
        from autorisk.eval.evaluator import Evaluator

        evaluator = Evaluator()
        results: list[AblationResult] = []

        # Load existing Cosmos results
        with open(cosmos_results_path, encoding="utf-8") as f:
            raw_results = json.load(f)

        # --- Mode 1: Baseline (fused_score threshold, no Cosmos) ---
        log.info("Minimal ablation: baseline (score threshold)")
        baseline_responses = []
        for entry in raw_results:
            score = entry.get("fused_score", 0.0)
            if score >= 0.7:
                sev = "HIGH"
            elif score >= 0.5:
                sev = "MEDIUM"
            elif score >= 0.3:
                sev = "LOW"
            else:
                sev = "NONE"

            request = CosmosRequest(
                clip_path=entry.get("clip_path", ""),
                candidate_rank=entry.get("candidate_rank", 0),
                peak_time_sec=entry.get("peak_time_sec", 0.0),
                fused_score=score,
            )
            assessment = RiskAssessment(
                severity=sev,
                causal_reasoning=f"Fused score {score:.3f} maps to {sev}",
                confidence=score,
            )
            baseline_responses.append(CosmosResponse(
                request=request, assessment=assessment, parse_success=True,
            ))

        baseline_eval = evaluator.evaluate(baseline_responses, gt_labels)
        results.append(AblationResult(
            mode="baseline",
            description="Mining score threshold only (no Cosmos)",
            accuracy=baseline_eval.accuracy,
            macro_f1=baseline_eval.macro_f1,
            checklist_mean=baseline_eval.checklist_means.get("mean_total", 0),
            n_candidates=len(baseline_responses),
        ))

        # --- Mode 2: Cosmos video (existing results) ---
        log.info("Minimal ablation: cosmos_video (existing results)")
        video_responses = [CosmosResponse.from_dict(entry) for entry in raw_results]

        video_eval = evaluator.evaluate(video_responses, gt_labels)
        results.append(AblationResult(
            mode="cosmos_video",
            description="Full Cosmos Reason 2 video inference",
            accuracy=video_eval.accuracy,
            macro_f1=video_eval.macro_f1,
            checklist_mean=video_eval.checklist_means.get("mean_total", 0),
            n_candidates=len(video_responses),
        ))

        # Log summary
        for r in results:
            log.info(
                "  %s: acc=%.3f, f1=%.3f, checklist=%.2f",
                r.mode, r.accuracy, r.macro_f1, r.checklist_mean,
            )

        return results

    @staticmethod
    def save_results(
        results: list[AblationResult],
        output_dir: Path,
    ) -> Path:
        """Save ablation results to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "ablation_results.json"

        data = [
            {
                "mode": r.mode,
                "description": r.description,
                "accuracy": r.accuracy,
                "macro_f1": r.macro_f1,
                "checklist_mean": r.checklist_mean,
                "n_candidates": r.n_candidates,
                "details": r.details,
            }
            for r in results
        ]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        log.info("Saved ablation results to %s", path)
        return path
