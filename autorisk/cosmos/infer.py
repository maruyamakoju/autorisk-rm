"""Cosmos Reason 2 inference engine: batch processing, parsing, fallback."""

from __future__ import annotations

import json
import re
from pathlib import Path

from omegaconf import DictConfig
from tqdm import tqdm

from autorisk.cosmos.prompt import SYSTEM_PROMPT, USER_PROMPT
from autorisk.cosmos.schema import (
    CosmosRequest,
    CosmosResponse,
    HazardDetail,
    RiskAssessment,
)
from autorisk.mining.fuse import Candidate
from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)


def _parse_think_answer(text: str) -> tuple[str, str]:
    """Extract <think> and <answer> blocks from Cosmos response."""
    think = ""
    answer = text

    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        think = think_match.group(1).strip()

    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()

    return think, answer


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from text, handling markdown fences."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try markdown code fence
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return None


def _clean_md(text: str) -> str:
    """Strip markdown bold markers and surrounding whitespace."""
    return text.strip().strip("*").strip()


def _parse_markdown_response(text: str) -> dict | None:
    """Fallback parser for markdown-formatted responses (non-JSON).

    Extracts severity, hazard type, actors, causal reasoning, etc.
    from markdown text like '**Severity:** HIGH'.
    """
    result: dict = {}

    # Severity - key fix: \s* after closing ** before the severity word
    sev_match = re.search(
        r"\*{0,2}Severity\*{0,2}:\s*\*{0,2}\s*(\w+)", text, re.IGNORECASE,
    )
    if sev_match:
        sev = sev_match.group(1).upper()
        if sev in ("HIGH", "MEDIUM", "LOW", "NONE"):
            result["severity"] = sev

    if "severity" not in result:
        return None

    # Hazard type / actors / spatial - use pattern that handles ** around values
    _field_re = r"\*{{0,2}}{label}\*{{0,2}}:\s*\*{{0,2}}\s*(.+?)(?:\n|$)"

    type_match = re.search(
        _field_re.format(label="Type"), text, re.IGNORECASE,
    )
    actors_match = re.search(
        _field_re.format(label=r"Actors?"), text, re.IGNORECASE,
    )
    spatial_match = re.search(
        _field_re.format(label=r"Spatial[_ ]?Relat\w*"), text, re.IGNORECASE,
    )

    hazard: dict = {}
    if type_match:
        hazard["type"] = _clean_md(type_match.group(1))
    if actors_match:
        raw = _clean_md(actors_match.group(1))
        hazard["actors"] = [a.strip() for a in raw.split(",")]
    if spatial_match:
        hazard["spatial_relation"] = _clean_md(spatial_match.group(1))

    if hazard:
        result["hazards"] = [hazard]

    # Section-based extraction: split on **Header:** patterns
    sections = re.split(r"\n\s*\*{1,2}([^*\n]+)\*{0,2}:\s*\*{0,2}\s*", text)

    section_map: dict[str, str] = {}
    for i in range(1, len(sections) - 1, 2):
        key = sections[i].strip().lower()
        val = sections[i + 1].strip() if i + 1 < len(sections) else ""
        section_map[key] = val

    # Causal reasoning
    for key in ("causal reasoning", "causal_reasoning", "causal reason"):
        if key in section_map:
            result["causal_reasoning"] = _clean_md(section_map[key])
            break

    # Short-term prediction
    for key in section_map:
        if "short" in key and "predict" in key:
            result["short_term_prediction"] = _clean_md(section_map[key])
            break

    # Recommended action
    for key in section_map:
        if "recommend" in key and "action" in key:
            result["recommended_action"] = _clean_md(section_map[key])
            break

    # Fallback regex if section parsing missed causal reasoning
    if "causal_reasoning" not in result:
        causal_match = re.search(
            r"\*{0,2}Causal[_ ]?Reason\w*\*{0,2}:?\s*\n?(.*?)(?:\n\s*\*{1,2}\w|$)",
            text, re.IGNORECASE | re.DOTALL,
        )
        if causal_match:
            result["causal_reasoning"] = _clean_md(causal_match.group(1))

    # Confidence (default 0.7 if we got severity from text)
    result.setdefault("confidence", 0.7)
    result.setdefault("evidence", [])

    return result


def _dict_to_assessment(data: dict) -> RiskAssessment:
    """Convert raw dict to RiskAssessment, handling missing/malformed fields."""
    severity = str(data.get("severity", "NONE")).upper()
    if severity not in ("HIGH", "MEDIUM", "LOW", "NONE"):
        severity = "NONE"

    hazards = []
    for h in data.get("hazards", []):
        if isinstance(h, dict):
            hazards.append(HazardDetail(
                type=str(h.get("type", "unknown")),
                actors=[str(a) for a in h.get("actors", [])],
                spatial_relation=str(h.get("spatial_relation", "")),
            ))

    return RiskAssessment(
        severity=severity,
        hazards=hazards,
        causal_reasoning=str(data.get("causal_reasoning", "")),
        short_term_prediction=str(data.get("short_term_prediction", "")),
        recommended_action=str(data.get("recommended_action", "")),
        evidence=[str(e) for e in data.get("evidence", [])],
        confidence=float(data.get("confidence", 0.0)),
    )


def _fallback_assessment() -> RiskAssessment:
    """Return a minimal fallback assessment when parsing fails."""
    return RiskAssessment(
        severity="NONE",
        causal_reasoning="Unable to parse model response",
        confidence=0.0,
    )


def _create_client(cfg: DictConfig):
    """Create the appropriate client based on config backend setting."""
    backend = cfg.cosmos.get("backend", "api")
    if backend == "local":
        from autorisk.cosmos.local_client import CosmosLocalClient
        log.info("Using LOCAL backend (transformers)")
        return CosmosLocalClient(cfg)
    else:
        from autorisk.cosmos.api_client import CosmosAPIClient
        log.info("Using API backend (NVIDIA Build)")
        return CosmosAPIClient(cfg)


class CosmosInferenceEngine:
    """Orchestrates Cosmos Reason 2 inference across candidate clips."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.client = _create_client(cfg)

    def infer_single(self, request: CosmosRequest) -> CosmosResponse:
        """Run inference on a single clip.

        Args:
            request: Inference request with clip path.

        Returns:
            CosmosResponse with parsed assessment.
        """
        clip_path = Path(request.clip_path)
        if not clip_path.exists():
            return CosmosResponse(
                request=request,
                assessment=_fallback_assessment(),
                parse_success=False,
                error=f"Clip not found: {clip_path}",
            )

        user_prompt = USER_PROMPT.format(fused_score=request.fused_score)

        try:
            video_b64 = self.client.encode_video_base64(clip_path)
            raw_response = self.client.chat_completion(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                video_b64=video_b64,
                video_path=clip_path,
            )
        except Exception as e:
            log.error("Inference failed for %s: %s", clip_path.name, e)
            return CosmosResponse(
                request=request,
                assessment=_fallback_assessment(),
                parse_success=False,
                error=str(e),
            )

        think, answer = _parse_think_answer(raw_response)
        json_data = _extract_json(answer)

        if json_data is None:
            # Try extracting from full response
            json_data = _extract_json(raw_response)

        if json_data is not None:
            assessment = _dict_to_assessment(json_data)
            parse_success = True
        else:
            # Try markdown fallback parser
            md_data = _parse_markdown_response(raw_response)
            if md_data is not None:
                log.info("Parsed markdown response for %s", clip_path.name)
                assessment = _dict_to_assessment(md_data)
                parse_success = True
            else:
                log.warning("Failed to parse response for %s", clip_path.name)
                assessment = _fallback_assessment()
                assessment.causal_reasoning = answer[:500] if answer else raw_response[:500]
                parse_success = False

        return CosmosResponse(
            request=request,
            assessment=assessment,
            raw_thinking=think,
            raw_answer=answer,
            parse_success=parse_success,
        )

    def infer_batch(
        self,
        candidates: list[Candidate],
    ) -> list[CosmosResponse]:
        """Run inference on a batch of candidates.

        Args:
            candidates: List of mining candidates with clip paths.

        Returns:
            List of CosmosResponse objects.
        """
        responses: list[CosmosResponse] = []

        for cand in tqdm(candidates, desc="Cosmos inference"):
            request = CosmosRequest(
                clip_path=cand.clip_path,
                candidate_rank=cand.rank,
                peak_time_sec=cand.peak_time_sec,
                fused_score=cand.fused_score,
            )
            response = self.infer_single(request)
            responses.append(response)

            log.info(
                "Clip %03d | severity=%s | confidence=%.2f | parse=%s",
                cand.rank,
                response.assessment.severity,
                response.assessment.confidence,
                response.parse_success,
            )

        return responses

    def save_results(
        self,
        responses: list[CosmosResponse],
        output_dir: Path,
    ) -> Path:
        """Save inference results as JSON.

        Args:
            responses: List of Cosmos responses.
            output_dir: Output directory.

        Returns:
            Path to saved JSON file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "cosmos_results.json"

        results = []
        for r in responses:
            results.append({
                "candidate_rank": r.request.candidate_rank,
                "clip_path": r.request.clip_path,
                "peak_time_sec": r.request.peak_time_sec,
                "fused_score": r.request.fused_score,
                "severity": r.assessment.severity,
                "hazards": [h.model_dump() for h in r.assessment.hazards],
                "causal_reasoning": r.assessment.causal_reasoning,
                "short_term_prediction": r.assessment.short_term_prediction,
                "recommended_action": r.assessment.recommended_action,
                "evidence": r.assessment.evidence,
                "confidence": r.assessment.confidence,
                "parse_success": r.parse_success,
                "error": r.error,
                "raw_answer": r.raw_answer[:1000] if r.raw_answer else "",
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        log.info("Saved %d inference results to %s", len(results), out_path)
        return out_path
