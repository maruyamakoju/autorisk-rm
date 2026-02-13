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


def _close_json(text: str) -> str:
    """Analyze JSON nesting state and append closing characters."""
    in_string = False
    escape_next = False
    stack: list[str] = []

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if in_string:
            if ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()

    suffix = ""
    if in_string:
        suffix += '"'
    suffix += "".join(reversed(stack))
    return text + suffix


def _fix_missing_commas(text: str) -> str:
    """Fix missing commas between JSON object fields.

    LLMs sometimes output JSON with missing commas:
        "field1": "value1"
        "field2": "value2"
    This inserts the missing comma.
    """
    # Pattern: end of a JSON value followed by whitespace then a new key
    # Matches: "value"\n  "key" or number\n  "key" or ]\n  "key" or }\n  "key"
    return re.sub(
        r'("|\d|true|false|null|\]|\})\s*\n(\s*")',
        r"\1,\n\2",
        text,
    )


def _repair_truncated_json(text: str) -> str:
    """Attempt to repair truncated JSON by closing open braces/brackets/strings.

    The model sometimes generates JSON that gets cut off before completing,
    leaving unclosed brackets, braces, or strings. This function adds the
    minimum necessary closing characters. Also fixes missing commas.
    """
    # Fix missing commas first
    text = _fix_missing_commas(text)

    repaired = _close_json(text)

    # If valid, return directly
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        pass

    # Try removing trailing incomplete key-value pair from the REPAIRED text.
    # Handles: ..."value",\n  "truncated_key"}  →  ..."value"}
    # Or:      ..."value",\n  "key": "truncated_val"}  →  ..."value"}
    for pattern in [
        r',\s*"[^"]*"\s*[}\]]*\s*$',       # orphan key without value
        r',\s*"[^"]*"\s*:\s*"[^"]*"\s*[}\]]*\s*$',  # truncated key-value
    ]:
        cleaned = re.sub(pattern, "", repaired)
        if cleaned != repaired:
            candidate = _close_json(cleaned)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

    return repaired


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from text, handling markdown fences and truncation."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip leading "json " prefix (model sometimes outputs "json {..." without fence)
    stripped = text.strip()
    if stripped.startswith("json"):
        stripped = stripped[4:].strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Try markdown code fence (greedy match for content between fences)
    fence_match = re.search(r"```(?:json)?\s*\n?(.*)\n?```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try fence without closing ``` (model truncated before closing fence)
    open_fence = re.search(r"```(?:json)?\s*\n?(.*)", text, re.DOTALL)
    if open_fence:
        inner = open_fence.group(1).strip()
        # Try parsing the inner content directly
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass
        # Try repairing truncated JSON inside the fence
        repaired = _repair_truncated_json(inner)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block (greedy: outermost braces)
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    # Try finding { without matching } (truncated JSON)
    brace_start = re.search(r"\{.*", text, re.DOTALL)
    if brace_start:
        repaired = _repair_truncated_json(brace_start.group())
        try:
            return json.loads(repaired)
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
        output_dir: Path | None = None,
    ) -> list[CosmosResponse]:
        """Run inference on a batch of candidates.

        Args:
            candidates: List of mining candidates with clip paths.
            output_dir: Optional dir for incremental saving after each clip.

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

            # Save incrementally to prevent data loss on crash
            if output_dir is not None:
                self.save_results(responses, output_dir)

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
                "raw_answer": r.raw_answer or "",
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        log.info("Saved %d inference results to %s", len(results), out_path)
        return out_path

    @staticmethod
    def reparse_results(results_path: Path) -> list[CosmosResponse]:
        """Re-parse saved results using the improved JSON parser.

        Reads cosmos_results.json and re-attempts parsing on entries that
        previously failed (parse_success=False). Useful after fixing the
        parser without re-running expensive inference.

        Args:
            results_path: Path to cosmos_results.json.

        Returns:
            List of CosmosResponse with re-parsed assessments.
        """
        with open(results_path, encoding="utf-8") as f:
            raw_results = json.load(f)

        responses: list[CosmosResponse] = []
        reparsed_count = 0

        for entry in raw_results:
            request = CosmosRequest(
                clip_path=entry.get("clip_path", ""),
                candidate_rank=entry.get("candidate_rank", 0),
                peak_time_sec=entry.get("peak_time_sec", 0.0),
                fused_score=entry.get("fused_score", 0.0),
            )

            raw_answer = entry.get("raw_answer", "")
            was_failed = not entry.get("parse_success", True)

            if was_failed and raw_answer:
                # Re-attempt parsing with improved parser
                json_data = _extract_json(raw_answer)

                if json_data is None:
                    # Try markdown fallback
                    md_data = _parse_markdown_response(raw_answer)
                    if md_data is not None:
                        json_data = md_data

                if json_data is not None:
                    assessment = _dict_to_assessment(json_data)
                    log.info(
                        "Re-parsed %s: severity=%s",
                        Path(entry["clip_path"]).name,
                        assessment.severity,
                    )
                    responses.append(CosmosResponse(
                        request=request,
                        assessment=assessment,
                        raw_answer=raw_answer,
                        parse_success=True,
                    ))
                    reparsed_count += 1
                    continue

            # Keep original entry (either success or still-failed)
            responses.append(CosmosResponse.from_dict(entry))

        log.info(
            "Re-parsed %d/%d previously failed entries",
            reparsed_count,
            sum(1 for e in raw_results if not e.get("parse_success", True)),
        )
        return responses
