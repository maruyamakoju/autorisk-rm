"""Search helpers for dashboard pages.

This module is intentionally streamlit-free so logic can be unit-tested
under lightweight CI environments.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchHit:
    clip: dict
    relevance: float
    matched_fields: tuple[str, ...]


def parse_query_terms(query: str) -> list[str]:
    """Normalize a free-form query into distinct lowercase terms."""
    terms: list[str] = []
    seen: set[str] = set()
    for raw in str(query or "").split():
        term = raw.strip().lower()
        if term == "" or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def highlight_terms(text: str, terms: list[str]) -> str:
    """Escape text and wrap matched terms in <mark> tags."""
    escaped = html.escape(str(text or ""))
    if escaped == "" or not terms:
        return escaped

    out = escaped
    for term in terms:
        if term == "":
            continue
        pattern = re.compile(f"({re.escape(term)})", re.IGNORECASE)
        out = pattern.sub(r"<mark>\1</mark>", out)
    return out


def search_clips(cosmos_results: list[dict], query: str) -> list[SearchHit]:
    """Search clips by relevance over severity, hazards, evidence, and reasoning."""
    terms = parse_query_terms(query)
    if not terms:
        return []

    hits: list[SearchHit] = []
    for clip in cosmos_results:
        if not isinstance(clip, dict):
            continue

        score = 0.0
        fields: set[str] = set()

        severity = str(clip.get("severity", "")).strip().lower()
        if severity in terms:
            score += 5.0
            fields.add("severity")

        reasoning = str(clip.get("causal_reasoning", "")).lower()
        if any(t in reasoning for t in terms):
            score += 3.0
            fields.add("causal_reasoning")

        evidence = " ".join(str(x) for x in clip.get("evidence", [])).lower()
        if any(t in evidence for t in terms):
            score += 2.0
            fields.add("evidence")

        hazard_hit = False
        for hazard in clip.get("hazards", []):
            if not isinstance(hazard, dict):
                continue
            h_type = str(hazard.get("type", "")).lower()
            actors = " ".join(str(x) for x in hazard.get("actors", [])).lower()
            if any((t in h_type) or (t in actors) for t in terms):
                hazard_hit = True
                break
        if hazard_hit:
            score += 2.0
            fields.add("hazards")

        if score <= 0:
            continue
        rank = int(clip.get("candidate_rank", 10**9))
        # Stable order across identical scores.
        hits.append(
            SearchHit(
                clip={**clip, "_search_rank": rank},
                relevance=score,
                matched_fields=tuple(sorted(fields)),
            )
        )

    hits.sort(key=lambda h: (-h.relevance, int(h.clip.get("_search_rank", 10**9))))
    for hit in hits:
        hit.clip.pop("_search_rank", None)
    return hits

