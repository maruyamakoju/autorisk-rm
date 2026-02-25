"""Tests for counterfactual prompt builders and batch generation (mock-based, no GPU)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autorisk.cosmos.predict_client import (
    CosmosPredictClient,
    build_counterfactual_prompts,
    build_danger_prompt,
    build_safe_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pedestrian_result():
    return {
        "clip_path": "clips/candidate_002_t145.0s.mp4",
        "severity": "HIGH",
        "hazards": [{"type": "Pedestrian", "actors": ["Person in blue shirt"]}],
        "causal_reasoning": "A pedestrian is crossing the road unexpectedly.",
        "short_term_prediction": "The pedestrian may step into the lane.",
        "recommended_action": "Slow down and stop completely to avoid the pedestrian.",
    }


@pytest.fixture()
def vehicle_result():
    return {
        "clip_path": "clips/candidate_008_t269.0s.mp4",
        "severity": "HIGH",
        "hazards": [{"type": "Cut-in of a vehicle", "actors": ["White van"]}],
        "causal_reasoning": "A white van is merging aggressively into our lane.",
        "short_term_prediction": "The van will cut in front causing a near-miss.",
        "recommended_action": "Brake firmly to maintain a safe distance.",
    }


@pytest.fixture()
def empty_result():
    return {
        "clip_path": "clips/candidate_007_t46.0s.mp4",
        "severity": "HIGH",
        "hazards": [],
        "causal_reasoning": "The ego vehicle is driving on a multi-lane road.",
        "short_term_prediction": "",
        "recommended_action": "",
    }


@pytest.fixture()
def client():
    cfg = MagicMock()
    cfg.cosmos.get.return_value = {
        "enabled": True,
        "model_name": "nvidia/Cosmos-Predict2-2B-Video2World",
        "severity_filter": ["HIGH", "MEDIUM"],
    }
    return CosmosPredictClient(cfg)


# ---------------------------------------------------------------------------
# TestBuildDangerPrompt
# ---------------------------------------------------------------------------

class TestBuildDangerPrompt:
    def test_pedestrian_collision_language(self, pedestrian_result):
        prompt = build_danger_prompt(pedestrian_result)
        assert "Dashcam" in prompt
        assert "does not react in time" in prompt
        assert "struck" in prompt.lower() or "pedestrian" in prompt.lower()

    def test_vehicle_cutin_collision_language(self, vehicle_result):
        prompt = build_danger_prompt(vehicle_result)
        assert "does not react in time" in prompt
        assert "sideswipe" in prompt.lower()

    def test_empty_result_still_works(self, empty_result):
        prompt = build_danger_prompt(empty_result)
        assert "Dashcam" in prompt
        assert "does not react in time" in prompt
        # Falls back to "Collision" hazard type
        assert "collision" in prompt.lower()

    def test_truncation(self):
        result = {
            "severity": "HIGH",
            "hazards": [{"type": "Pedestrian"}],
            "causal_reasoning": "x" * 500,
            "short_term_prediction": "y" * 400,
            "recommended_action": "z" * 300,
        }
        prompt = build_danger_prompt(result)
        assert len(prompt) < 700

    def test_includes_causal_and_prediction(self, pedestrian_result):
        prompt = build_danger_prompt(pedestrian_result)
        assert "crossing" in prompt
        assert "step" in prompt


# ---------------------------------------------------------------------------
# TestBuildSafePrompt
# ---------------------------------------------------------------------------

class TestBuildSafePrompt:
    def test_uses_recommended_action(self, pedestrian_result):
        prompt = build_safe_prompt(pedestrian_result)
        assert "evasive action" in prompt.lower()
        assert "slow down" in prompt.lower()
        assert "No collision occurs" in prompt

    def test_pedestrian_safe_resolution(self, pedestrian_result):
        prompt = build_safe_prompt(pedestrian_result)
        assert "safely" in prompt.lower() or "without contact" in prompt.lower()

    def test_fallback_when_action_empty(self, empty_result):
        prompt = build_safe_prompt(empty_result)
        assert "Dashcam" in prompt
        assert "evasive action" in prompt.lower()
        # Should fall back to causal_reasoning
        assert "responding to" in prompt
        assert "No collision occurs" in prompt

    def test_vehicle_safe_resolution(self, vehicle_result):
        prompt = build_safe_prompt(vehicle_result)
        assert "safe separation" in prompt.lower() or "without contact" in prompt.lower()

    def test_truncation(self):
        result = {
            "severity": "HIGH",
            "hazards": [{"type": "Pedestrian"}],
            "causal_reasoning": "a" * 500,
            "short_term_prediction": "b" * 400,
            "recommended_action": "c" * 400,
        }
        prompt = build_safe_prompt(result)
        assert len(prompt) < 500


# ---------------------------------------------------------------------------
# TestBuildCounterfactualPrompts
# ---------------------------------------------------------------------------

class TestBuildCounterfactualPrompts:
    def test_returns_tuple(self, pedestrian_result):
        danger, safe = build_counterfactual_prompts(pedestrian_result)
        assert isinstance(danger, str)
        assert isinstance(safe, str)

    def test_prompts_are_different(self, pedestrian_result):
        danger, safe = build_counterfactual_prompts(pedestrian_result)
        assert danger != safe
        assert "does not react" in danger
        assert "evasive action" in safe.lower()


# ---------------------------------------------------------------------------
# TestCounterfactualBatch
# ---------------------------------------------------------------------------

class TestCounterfactualBatch:
    def test_filters_high_only(self, client, tmp_path):
        results = [
            {"clip_path": "clips/a.mp4", "severity": "HIGH", "causal_reasoning": "danger",
             "hazards": [{"type": "Pedestrian"}], "recommended_action": "stop"},
            {"clip_path": "clips/b.mp4", "severity": "LOW", "causal_reasoning": "safe",
             "hazards": [], "recommended_action": ""},
            {"clip_path": "clips/c.mp4", "severity": "MEDIUM", "causal_reasoning": "caution",
             "hazards": [], "recommended_action": "slow"},
        ]

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        for name in ["a.mp4", "b.mp4", "c.mp4"]:
            (clips_dir / name).write_bytes(b"\x00" * 100)

        out_dir = tmp_path / "counterfactuals"

        with patch.object(client, "_predict_with_seed") as mock_pred:
            mock_pred.return_value = {
                "clip_name": "test.mp4", "prompt": "test",
                "output_path": "test.mp4", "generation_time_sec": 1.0,
                "n_frames": 10, "seed": 42,
            }

            batch = client.predict_counterfactual_batch(
                cosmos_results=results,
                clips_dir=clips_dir,
                output_dir=out_dir,
                severity_filter={"HIGH"},
            )

        # Only HIGH clip a.mp4 should be processed (danger + safe = 2 calls)
        assert mock_pred.call_count == 2
        assert len(batch) == 1
        assert batch[0]["clip_name"] == "a.mp4"

    def test_uses_corrected_severity(self, client, tmp_path):
        results = [
            {"clip_path": "clips/a.mp4", "severity": "MEDIUM", "causal_reasoning": "x",
             "hazards": [], "recommended_action": ""},
        ]
        corrected = [
            {"clip_path": "clips/a.mp4", "severity": "HIGH"},
        ]

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        (clips_dir / "a.mp4").write_bytes(b"\x00" * 100)
        out_dir = tmp_path / "counterfactuals"

        with patch.object(client, "_predict_with_seed") as mock_pred:
            mock_pred.return_value = {
                "clip_name": "a.mp4", "prompt": "test",
                "output_path": "test.mp4", "generation_time_sec": 1.0,
                "n_frames": 10, "seed": 42,
            }

            batch = client.predict_counterfactual_batch(
                cosmos_results=results,
                clips_dir=clips_dir,
                output_dir=out_dir,
                severity_filter={"HIGH"},
                corrected_results=corrected,
            )

        # MEDIUM clip promoted to HIGH via corrected_results
        assert mock_pred.call_count == 2
        assert len(batch) == 1

    def test_different_seeds_for_danger_safe(self, client, tmp_path):
        results = [
            {"clip_path": "clips/a.mp4", "severity": "HIGH", "causal_reasoning": "x",
             "hazards": [{"type": "Collision"}], "recommended_action": "stop"},
        ]

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        (clips_dir / "a.mp4").write_bytes(b"\x00" * 100)
        out_dir = tmp_path / "counterfactuals"

        with patch.object(client, "_predict_with_seed") as mock_pred:
            mock_pred.return_value = {
                "clip_name": "a.mp4", "prompt": "test",
                "output_path": "test.mp4", "generation_time_sec": 1.0,
                "n_frames": 10, "seed": 42,
            }

            client.predict_counterfactual_batch(
                cosmos_results=results,
                clips_dir=clips_dir,
                output_dir=out_dir,
                danger_seed=42,
                safe_seed=137,
            )

        # First call = danger (seed=42), second = safe (seed=137)
        calls = mock_pred.call_args_list
        assert calls[0].kwargs.get("seed", calls[0][1][3] if len(calls[0][1]) > 3 else None) == 42 or "danger" in str(calls[0])
        assert calls[1].kwargs.get("seed", calls[1][1][3] if len(calls[1][1]) > 3 else None) == 137 or "safe" in str(calls[1])

    def test_saves_metadata_json(self, client, tmp_path):
        results = [
            {"clip_path": "clips/a.mp4", "severity": "HIGH", "causal_reasoning": "x",
             "hazards": [], "recommended_action": "stop"},
        ]

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        (clips_dir / "a.mp4").write_bytes(b"\x00" * 100)
        out_dir = tmp_path / "counterfactuals"

        with patch.object(client, "_predict_with_seed") as mock_pred:
            mock_pred.return_value = {
                "clip_name": "a.mp4", "prompt": "test",
                "output_path": "test.mp4", "generation_time_sec": 1.0,
                "n_frames": 10, "seed": 42,
            }

            client.predict_counterfactual_batch(
                cosmos_results=results,
                clips_dir=clips_dir,
                output_dir=out_dir,
            )

        meta_path = out_dir / "counterfactual_results.json"
        assert meta_path.exists()

    def test_handles_predict_error(self, client, tmp_path):
        results = [
            {"clip_path": "clips/a.mp4", "severity": "HIGH", "causal_reasoning": "x",
             "hazards": [], "recommended_action": "stop"},
        ]

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        (clips_dir / "a.mp4").write_bytes(b"\x00" * 100)
        out_dir = tmp_path / "counterfactuals"

        with patch.object(client, "_predict_with_seed") as mock_pred:
            mock_pred.side_effect = RuntimeError("CUDA OOM")

            batch = client.predict_counterfactual_batch(
                cosmos_results=results,
                clips_dir=clips_dir,
                output_dir=out_dir,
            )

        assert len(batch) == 1
        assert "error" in batch[0]["danger"]
        assert "error" in batch[0]["safe"]
