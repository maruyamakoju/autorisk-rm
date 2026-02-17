"""Cross-Modal Grounding: mutual validation between mining signals and Cosmos reasoning.

Evaluates whether Cosmos's natural-language reasoning is grounded in the
quantitative mining signals. Computes per-clip grounding scores measuring
agreement between signal spikes and VLM explanations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)


# Keywords that map signal types to Cosmos reasoning concepts
AUDIO_KEYWORDS = [
    "horn", "honk", "brake", "screech", "loud", "sound", "noise",
    "siren", "alarm", "crash", "impact", "squeal", "tyre", "tire",
    "audio", "beep", "skid",
]

MOTION_KEYWORDS = [
    "sudden", "swerve", "braking", "accelerat", "decelerat", "speed",
    "fast", "rapid", "erratic", "aggressive", "sharp", "turn",
    "motion", "movement", "maneuver", "manoeuvre", "abrupt", "quick",
    "cut", "merge", "drift", "lane change", "overtake",
]

PROXIMITY_KEYWORDS = [
    "close", "near", "proximity", "distance", "follow", "tailgat",
    "adjacent", "encroach", "crowd", "congest", "tight", "narrow",
    "gap", "space", "behind", "ahead", "front", "approaching",
    "pedestrian", "cyclist", "vehicle", "car", "truck", "bus",
    "person", "walker", "motorcycle", "bicycle",
]


@dataclass
class GroundingDetail:
    """Grounding analysis for a single clip."""
    clip_name: str
    # Signal activation levels (normalized 0-1)
    audio_activation: float = 0.0
    motion_activation: float = 0.0
    proximity_activation: float = 0.0
    # Keyword matches in Cosmos reasoning
    audio_keyword_hits: int = 0
    motion_keyword_hits: int = 0
    proximity_keyword_hits: int = 0
    # Agreement score: do active signals have matching keywords?
    grounding_score: float = 0.0
    # Which signals are "active" (above threshold)
    active_signals: list[str] = field(default_factory=list)
    # Which signal types are mentioned in reasoning
    mentioned_signals: list[str] = field(default_factory=list)
    # Ungrounded: active signal with no keyword match
    ungrounded_signals: list[str] = field(default_factory=list)
    # Hallucinated: keyword match with no active signal
    hallucinated_signals: list[str] = field(default_factory=list)


@dataclass
class GroundingReport:
    """Aggregate grounding analysis."""
    mean_grounding_score: float = 0.0
    n_clips: int = 0
    n_fully_grounded: int = 0  # All active signals mentioned
    n_has_hallucination: int = 0  # Mentions signals not active
    n_has_ungrounded: int = 0  # Active signals not mentioned
    # Per-signal grounding rates
    signal_grounding_rates: dict[str, float] = field(default_factory=dict)
    # Details per clip
    details: list[GroundingDetail] = field(default_factory=list)
    # Grounding score vs severity
    grounding_by_severity: dict[str, float] = field(default_factory=dict)


class GroundingAnalyzer:
    """Analyze cross-modal grounding between mining signals and Cosmos reasoning."""

    def __init__(self, activation_threshold: float = 0.4) -> None:
        self.activation_threshold = activation_threshold

    def run(
        self,
        cosmos_results_path: Path,
        candidates_csv_path: Path,
    ) -> GroundingReport:
        """Run grounding analysis.

        Args:
            cosmos_results_path: Path to cosmos_results.json.
            candidates_csv_path: Path to candidates.csv with signal scores.

        Returns:
            GroundingReport with per-clip and aggregate grounding metrics.
        """
        import ast
        import csv

        # Load cosmos results
        with open(cosmos_results_path, encoding="utf-8") as f:
            cosmos_results = json.load(f)

        # Load candidates with signal scores
        signal_scores = {}
        with open(candidates_csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clip_name = Path(row.get("clip_path", "")).name
                try:
                    scores = ast.literal_eval(row.get("signal_scores", "{}"))
                except (ValueError, SyntaxError):
                    scores = {}
                signal_scores[clip_name] = scores

        # Analyze each clip
        details: list[GroundingDetail] = []

        for entry in cosmos_results:
            if not entry.get("parse_success", True):
                continue

            clip_name = Path(entry.get("clip_path", "")).name
            scores = signal_scores.get(clip_name, {})

            # Get signal activations
            audio_act = scores.get("audio", 0.0)
            motion_act = scores.get("motion", 0.0)
            proximity_act = scores.get("proximity", 0.0)

            # Combine all text fields for keyword search
            reasoning_text = " ".join([
                entry.get("causal_reasoning", ""),
                entry.get("short_term_prediction", ""),
                entry.get("recommended_action", ""),
                " ".join(entry.get("evidence", [])),
                " ".join(
                    h.get("type", "") + " " + " ".join(h.get("actors", []))
                    for h in entry.get("hazards", [])
                ),
            ]).lower()

            # Count keyword hits per signal type
            audio_hits = sum(
                1 for kw in AUDIO_KEYWORDS if kw in reasoning_text
            )
            motion_hits = sum(
                1 for kw in MOTION_KEYWORDS if kw in reasoning_text
            )
            proximity_hits = sum(
                1 for kw in PROXIMITY_KEYWORDS if kw in reasoning_text
            )

            # Determine active signals
            active = []
            if audio_act >= self.activation_threshold:
                active.append("audio")
            if motion_act >= self.activation_threshold:
                active.append("motion")
            if proximity_act >= self.activation_threshold:
                active.append("proximity")

            # Determine mentioned signals
            mentioned = []
            if audio_hits >= 1:
                mentioned.append("audio")
            if motion_hits >= 2:  # Higher threshold — motion words are common
                mentioned.append("motion")
            if proximity_hits >= 2:  # Higher threshold — proximity words are common
                mentioned.append("proximity")

            # Compute grounding score
            ungrounded = [s for s in active if s not in mentioned]
            hallucinated = [s for s in mentioned if s not in active]

            # Score: proportion of active signals that are grounded
            if active:
                grounding = len([s for s in active if s in mentioned]) / len(active)
            elif mentioned:
                # No active signals but model mentions some — partial hallucination
                grounding = 0.5
            else:
                # Neither active nor mentioned — vacuously grounded
                grounding = 1.0

            details.append(GroundingDetail(
                clip_name=clip_name,
                audio_activation=audio_act,
                motion_activation=motion_act,
                proximity_activation=proximity_act,
                audio_keyword_hits=audio_hits,
                motion_keyword_hits=motion_hits,
                proximity_keyword_hits=proximity_hits,
                grounding_score=grounding,
                active_signals=active,
                mentioned_signals=mentioned,
                ungrounded_signals=ungrounded,
                hallucinated_signals=hallucinated,
            ))

        if not details:
            return GroundingReport()

        # Aggregate stats
        mean_score = float(np.mean([d.grounding_score for d in details]))
        n_fully = sum(1 for d in details if not d.ungrounded_signals)
        n_halluc = sum(1 for d in details if d.hallucinated_signals)
        n_ungrounded = sum(1 for d in details if d.ungrounded_signals)

        # Per-signal grounding rate
        signal_rates = {}
        for sig in ["audio", "motion", "proximity"]:
            active_count = sum(1 for d in details if sig in d.active_signals)
            grounded_count = sum(
                1 for d in details
                if sig in d.active_signals and sig in d.mentioned_signals
            )
            signal_rates[sig] = (
                grounded_count / active_count if active_count > 0 else 0.0
            )

        # Grounding by severity
        grounding_by_sev = {}
        for entry, detail in zip(cosmos_results, details):
            sev = entry.get("severity", "NONE")
            if sev not in grounding_by_sev:
                grounding_by_sev[sev] = []
            grounding_by_sev[sev].append(detail.grounding_score)

        grounding_by_sev_mean = {
            sev: float(np.mean(scores))
            for sev, scores in grounding_by_sev.items()
        }

        report = GroundingReport(
            mean_grounding_score=mean_score,
            n_clips=len(details),
            n_fully_grounded=n_fully,
            n_has_hallucination=n_halluc,
            n_has_ungrounded=n_ungrounded,
            signal_grounding_rates=signal_rates,
            details=details,
            grounding_by_severity=grounding_by_sev_mean,
        )

        log.info(
            "Grounding: mean=%.3f, fully_grounded=%d/%d, hallucinated=%d",
            mean_score, n_fully, len(details), n_halluc,
        )

        return report

    @staticmethod
    def save(report: GroundingReport, output_dir: Path) -> Path:
        """Save grounding report to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "grounding_report.json"

        data = {
            "mean_grounding_score": report.mean_grounding_score,
            "n_clips": report.n_clips,
            "n_fully_grounded": report.n_fully_grounded,
            "n_has_hallucination": report.n_has_hallucination,
            "n_has_ungrounded": report.n_has_ungrounded,
            "signal_grounding_rates": report.signal_grounding_rates,
            "grounding_by_severity": report.grounding_by_severity,
            "details": [
                {
                    "clip_name": d.clip_name,
                    "audio_activation": d.audio_activation,
                    "motion_activation": d.motion_activation,
                    "proximity_activation": d.proximity_activation,
                    "audio_keyword_hits": d.audio_keyword_hits,
                    "motion_keyword_hits": d.motion_keyword_hits,
                    "proximity_keyword_hits": d.proximity_keyword_hits,
                    "grounding_score": d.grounding_score,
                    "active_signals": d.active_signals,
                    "mentioned_signals": d.mentioned_signals,
                    "ungrounded_signals": d.ungrounded_signals,
                    "hallucinated_signals": d.hallucinated_signals,
                }
                for d in report.details
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        log.info("Grounding: Saved report to %s", path)
        return path

    @staticmethod
    def load(path: Path) -> GroundingReport:
        """Load grounding report from JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        details = [
            GroundingDetail(**d) for d in data.get("details", [])
        ]

        return GroundingReport(
            mean_grounding_score=data["mean_grounding_score"],
            n_clips=data["n_clips"],
            n_fully_grounded=data["n_fully_grounded"],
            n_has_hallucination=data["n_has_hallucination"],
            n_has_ungrounded=data["n_has_ungrounded"],
            signal_grounding_rates=data.get("signal_grounding_rates", {}),
            details=details,
            grounding_by_severity=data.get("grounding_by_severity", {}),
        )
