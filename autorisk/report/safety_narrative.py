"""Generate human-readable safety narrative from cosmos results."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

SEVERITY_ORDER: tuple[str, str, str, str] = ("HIGH", "MEDIUM", "LOW", "NONE")


def _normalize_severity(raw: Any) -> str:
    value = str(raw or "").strip().upper()
    if value in {"HIGH", "MEDIUM", "LOW", "NONE"}:
        return value
    return "NONE"


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def generate_safety_narrative(cosmos_results: list[dict]) -> str:
    """Generate a markdown narrative report from parsed cosmos results."""
    if not cosmos_results:
        return "No analysis results available."

    total = len(cosmos_results)
    severity_counter: Counter[str] = Counter(
        _normalize_severity(item.get("severity")) for item in cosmos_results if isinstance(item, dict)
    )
    n_high = severity_counter.get("HIGH", 0)
    n_medium = severity_counter.get("MEDIUM", 0)
    n_low = severity_counter.get("LOW", 0)
    n_none = severity_counter.get("NONE", 0)

    all_hazards: list[str] = []
    for item in cosmos_results:
        if not isinstance(item, dict):
            continue
        for hazard in item.get("hazards", []):
            if not isinstance(hazard, dict):
                continue
            hazard_type = str(hazard.get("type", "")).strip()
            if hazard_type:
                all_hazards.append(hazard_type.lower())
    top_hazards = Counter(all_hazards).most_common(5)

    critical = [
        item
        for item in cosmos_results
        if isinstance(item, dict) and _normalize_severity(item.get("severity")) == "HIGH"
    ]

    lines: list[str] = []
    lines.append("# Safety Narrative Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"Analyzed **{total}** dashcam segments. Identified **{n_high + n_medium}** segments "
        f"requiring attention ({n_high} critical, {n_medium} moderate)."
    )
    lines.append("")

    lines.append("## Severity Distribution")
    lines.append("")
    lines.append(f"- **HIGH** (Critical): {n_high} ({_pct(n_high, total):.1f}%)")
    lines.append(f"- **MEDIUM** (Moderate): {n_medium} ({_pct(n_medium, total):.1f}%)")
    lines.append(f"- **LOW** (Minor): {n_low} ({_pct(n_low, total):.1f}%)")
    lines.append(f"- **NONE** (Routine): {n_none} ({_pct(n_none, total):.1f}%)")
    lines.append("")

    lines.append("## Risk Assessment")
    lines.append("")
    if n_high > 0:
        lines.append(
            f"- {n_high} critical incident(s) detected. These involve potential collision risk or emergency maneuvers."
        )
    else:
        lines.append("- No critical incidents detected.")
    if n_medium > 0:
        lines.append(
            f"- {n_medium} moderate-risk situation(s) detected. Defensive actions (braking, lane adjustment, vigilance) are recommended."
        )
    lines.append("")

    if top_hazards:
        lines.append("## Common Hazard Patterns")
        lines.append("")
        for hazard_type, count in top_hazards:
            lines.append(f"- **{hazard_type.title()}**: {count} occurrence(s)")
        lines.append("")

    if critical:
        lines.append("## Critical Incidents (HIGH)")
        lines.append("")
        for idx, incident in enumerate(critical[:10], 1):
            clip_path = str(incident.get("clip_path", "")).strip()
            clip_name = Path(clip_path).name if clip_path else f"Incident {idx}"
            confidence = float(incident.get("confidence", 0) or 0)
            reasoning = str(incident.get("causal_reasoning", "No reasoning available")).strip()

            lines.append(f"### {idx}. {clip_name} (confidence: {confidence:.0%})")
            lines.append("")
            lines.append(f"**Reasoning:** {reasoning}")
            lines.append("")

            hazards = incident.get("hazards", [])
            if isinstance(hazards, list) and hazards:
                lines.append("**Hazards:**")
                for hazard in hazards:
                    if not isinstance(hazard, dict):
                        continue
                    hazard_type = str(hazard.get("type", "unknown")).strip().title()
                    actors = ", ".join(str(a) for a in hazard.get("actors", []))
                    if actors:
                        lines.append(f"- {hazard_type} involving {actors}")
                    else:
                        lines.append(f"- {hazard_type}")
                lines.append("")

            action = str(incident.get("recommended_action", "")).strip()
            if action:
                lines.append(f"**Recommended Action:** {action}")
                lines.append("")

        if len(critical) > 10:
            lines.append(f"*...and {len(critical) - 10} more critical incidents.*")
            lines.append("")

    parse_success_count = sum(
        1
        for item in cosmos_results
        if isinstance(item, dict) and bool(item.get("parse_success", False))
    )
    avg_confidence = sum(
        float(item.get("confidence", 0) or 0) for item in cosmos_results if isinstance(item, dict)
    ) / total

    lines.append("## Analysis Quality")
    lines.append("")
    lines.append(
        f"- **Parse Success Rate**: {parse_success_count}/{total} ({_pct(parse_success_count, total):.1f}%)"
    )
    lines.append(f"- **Average Confidence**: {avg_confidence:.1%}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by AutoRisk-RM using NVIDIA Cosmos Reason 2*")
    lines.append("")
    return "\n".join(lines)


def save_narrative(cosmos_results: list[dict], output_path: Path) -> Path:
    """Generate and save narrative markdown to `output_path`."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generate_safety_narrative(cosmos_results), encoding="utf-8")
    return output


def generate_from_json(results_path: Path, output_path: Path | None = None) -> Path:
    """Load cosmos results JSON and write narrative markdown."""
    src = Path(results_path)
    with src.open(encoding="utf-8") as f:
        cosmos_results = json.load(f)
    if not isinstance(cosmos_results, list):
        raise ValueError("cosmos_results.json must contain a list")
    out = Path(output_path) if output_path is not None else (src.parent / "safety_narrative.md")
    return save_narrative(cosmos_results, out)

