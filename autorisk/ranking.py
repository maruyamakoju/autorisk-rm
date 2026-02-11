"""Risk ranking module (B3): sort candidates by severity and tiebreakers."""

from __future__ import annotations

from autorisk.cosmos.schema import CosmosResponse
from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)

SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}


def rank_responses(responses: list[CosmosResponse]) -> list[CosmosResponse]:
    """Rank Cosmos responses by severity, then tiebreak by hazard/evidence/actor count.

    Sorting priority:
        1. Severity: HIGH > MEDIUM > LOW > NONE
        2. Number of hazards (desc)
        3. Number of evidence items (desc)
        4. Number of unique actors across hazards (desc)

    Args:
        responses: Unordered list of CosmosResponse.

    Returns:
        Sorted list (highest risk first).
    """

    def sort_key(r: CosmosResponse) -> tuple:
        a = r.assessment
        sev = SEVERITY_ORDER.get(a.severity.upper(), 3)
        n_hazards = len(a.hazards)
        n_evidence = len(a.evidence)
        n_actors = len({
            actor
            for h in a.hazards
            for actor in h.actors
        })
        # Negate counts for descending sort
        return (sev, -n_hazards, -n_evidence, -n_actors)

    ranked = sorted(responses, key=sort_key)

    for i, r in enumerate(ranked, 1):
        log.debug(
            "Rank %d: severity=%s, hazards=%d, clip=%s",
            i, r.assessment.severity, len(r.assessment.hazards),
            r.request.clip_path,
        )

    log.info("Ranked %d responses", len(ranked))
    return ranked
