"""Explanation Checklist: 5-item binary scoring for explanation quality."""

from __future__ import annotations

from dataclasses import dataclass

from autorisk.cosmos.schema import CosmosResponse


CHECKLIST_ITEMS = [
    "actors_accurate",
    "causal_clear",
    "spatial_specific",
    "prediction_plausible",
    "action_reasonable",
]

CHECKLIST_DESCRIPTIONS = {
    "actors_accurate": "All relevant actors are correctly identified",
    "causal_clear": "Causal chain is clearly and logically explained",
    "spatial_specific": "Spatial relationships are specific (distances, lanes, positions)",
    "prediction_plausible": "Short-term prediction is plausible and consistent",
    "action_reasonable": "Recommended action is reasonable and actionable",
}


@dataclass
class ChecklistResult:
    """Result of checklist evaluation for a single response."""
    clip_id: str
    scores: dict[str, int]  # item_name -> 0 or 1
    total: int  # Sum of scores (0-5)
    notes: str = ""


class ExplanationChecklist:
    """Evaluate explanation quality against 5-item checklist.

    Each item is scored 0 (absent/poor) or 1 (present/adequate).
    Total score ranges from 0 to 5.
    """

    @staticmethod
    def evaluate_single(
        response: CosmosResponse,
        gt_scores: dict[str, int] | None = None,
    ) -> ChecklistResult:
        """Auto-evaluate a single response against the checklist.

        If gt_scores is provided, uses those. Otherwise, applies heuristic checks.

        Args:
            response: Cosmos inference response.
            gt_scores: Optional ground-truth scores per item.

        Returns:
            ChecklistResult with per-item and total scores.
        """
        if gt_scores is not None:
            total = sum(gt_scores.get(item, 0) for item in CHECKLIST_ITEMS)
            return ChecklistResult(
                clip_id=response.request.clip_path,
                scores={item: gt_scores.get(item, 0) for item in CHECKLIST_ITEMS},
                total=total,
            )

        # Heuristic auto-evaluation
        a = response.assessment
        scores: dict[str, int] = {}

        # 1. Actors accurate: at least one actor identified
        all_actors = [actor for h in a.hazards for actor in h.actors]
        scores["actors_accurate"] = 1 if len(all_actors) > 0 else 0

        # 2. Causal clear: causal reasoning is non-empty and substantive
        scores["causal_clear"] = 1 if len(a.causal_reasoning) > 20 else 0

        # 3. Spatial specific: spatial relations present and descriptive
        has_spatial = any(len(h.spatial_relation) > 10 for h in a.hazards)
        scores["spatial_specific"] = 1 if has_spatial else 0

        # 4. Prediction plausible: prediction is non-empty
        scores["prediction_plausible"] = 1 if len(a.short_term_prediction) > 10 else 0

        # 5. Action reasonable: action recommendation is non-empty
        scores["action_reasonable"] = 1 if len(a.recommended_action) > 10 else 0

        total = sum(scores.values())
        return ChecklistResult(
            clip_id=response.request.clip_path,
            scores=scores,
            total=total,
        )

    def evaluate_batch(
        self,
        responses: list[CosmosResponse],
        gt_map: dict[str, dict[str, int]] | None = None,
    ) -> list[ChecklistResult]:
        """Evaluate a batch of responses.

        Args:
            responses: List of Cosmos responses.
            gt_map: Optional mapping of clip_path -> per-item GT scores.

        Returns:
            List of ChecklistResult.
        """
        results = []
        for r in responses:
            gt = gt_map.get(r.request.clip_path) if gt_map else None
            results.append(self.evaluate_single(r, gt))
        return results

    @staticmethod
    def aggregate(results: list[ChecklistResult]) -> dict[str, float]:
        """Compute mean scores across all results.

        Returns:
            Dict with per-item means and overall mean total.
        """
        if not results:
            return {item: 0.0 for item in CHECKLIST_ITEMS + ["mean_total"]}

        agg: dict[str, float] = {}
        for item in CHECKLIST_ITEMS:
            agg[item] = sum(r.scores.get(item, 0) for r in results) / len(results)
        agg["mean_total"] = sum(r.total for r in results) / len(results)
        return agg
