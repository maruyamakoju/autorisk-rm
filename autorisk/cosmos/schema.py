"""Pydantic schemas for Cosmos Reason 2 request/response."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HazardDetail(BaseModel):
    """A single identified hazard."""
    type: str = Field(description="Hazard type (e.g. 'collision_risk', 'pedestrian_crossing')")
    actors: list[str] = Field(default_factory=list, description="Involved actors")
    spatial_relation: str = Field(default="", description="Spatial relationship description")


class RiskAssessment(BaseModel):
    """Structured risk assessment from Cosmos Reason 2."""
    severity: str = Field(description="Risk severity: HIGH, MEDIUM, LOW, or NONE")
    hazards: list[HazardDetail] = Field(default_factory=list, description="Identified hazards")
    causal_reasoning: str = Field(default="", description="Causal chain explanation")
    short_term_prediction: str = Field(default="", description="What would happen next without intervention")
    recommended_action: str = Field(default="", description="Recommended driving action")
    evidence: list[str] = Field(default_factory=list, description="Visual evidence supporting assessment")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Model confidence")


class CosmosRequest(BaseModel):
    """Request payload for Cosmos inference."""
    clip_path: str
    candidate_rank: int = 0
    peak_time_sec: float = 0.0
    fused_score: float = 0.0


class CosmosResponse(BaseModel):
    """Full response wrapper from Cosmos inference."""
    request: CosmosRequest
    assessment: RiskAssessment
    raw_thinking: str = Field(default="", description="Raw <think> block content")
    raw_answer: str = Field(default="", description="Raw <answer> block content")
    parse_success: bool = True
    error: str = ""

    @classmethod
    def from_dict(cls, entry: dict) -> "CosmosResponse":
        """Create a CosmosResponse from a raw dict (as stored in cosmos_results.json).

        This is the single canonical way to deserialize saved results.
        """
        request = CosmosRequest(
            clip_path=entry.get("clip_path", ""),
            candidate_rank=entry.get("candidate_rank", 0),
            peak_time_sec=entry.get("peak_time_sec", 0.0),
            fused_score=entry.get("fused_score", 0.0),
        )
        assessment = RiskAssessment(
            severity=entry.get("severity", "NONE"),
            hazards=[HazardDetail(**h) for h in entry.get("hazards", [])],
            causal_reasoning=entry.get("causal_reasoning", ""),
            short_term_prediction=entry.get("short_term_prediction", ""),
            recommended_action=entry.get("recommended_action", ""),
            evidence=entry.get("evidence", []),
            confidence=entry.get("confidence", 0.0),
        )
        return cls(
            request=request,
            assessment=assessment,
            raw_answer=entry.get("raw_answer", ""),
            parse_success=entry.get("parse_success", True),
            error=entry.get("error", ""),
        )
