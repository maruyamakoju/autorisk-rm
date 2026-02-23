"""Tests for Cosmos Predict 2 client (mock-based, no GPU required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autorisk.cosmos.predict_client import (
    NEGATIVE_PROMPT,
    CosmosPredictClient,
    build_prompt,
)


# --- Prompt Construction ---

class TestBuildPrompt:
    def test_basic_prompt(self):
        result = {
            "severity": "HIGH",
            "causal_reasoning": "A car is swerving into our lane at high speed.",
            "short_term_prediction": "Collision is imminent within 2 seconds.",
        }
        prompt = build_prompt(result)
        assert "Dashcam" in prompt
        assert "swerving" in prompt
        assert "imminent" in prompt
        assert "dangerous" in prompt.lower()

    def test_empty_result(self):
        prompt = build_prompt({})
        assert "Dashcam" in prompt

    def test_low_severity_no_danger_suffix(self):
        result = {
            "severity": "LOW",
            "causal_reasoning": "Normal traffic flow.",
        }
        prompt = build_prompt(result)
        assert "dangerous" not in prompt.lower()

    def test_truncates_long_text(self):
        result = {
            "severity": "HIGH",
            "causal_reasoning": "x" * 500,
            "short_term_prediction": "y" * 400,
        }
        prompt = build_prompt(result)
        # Should truncate both fields
        assert len(prompt) < 700


# --- Batch Filtering ---

class TestBatchFiltering:
    @pytest.fixture()
    def client(self):
        cfg = MagicMock()
        cfg.cosmos.get.return_value = {
            "enabled": True,
            "model_name": "nvidia/Cosmos-Predict2-2B-Video2World",
            "severity_filter": ["HIGH", "MEDIUM"],
        }
        return CosmosPredictClient(cfg)

    def test_filters_by_severity(self, client, tmp_path):
        results = [
            {"clip_path": "clips/a.mp4", "severity": "HIGH", "causal_reasoning": "danger"},
            {"clip_path": "clips/b.mp4", "severity": "LOW", "causal_reasoning": "safe"},
            {"clip_path": "clips/c.mp4", "severity": "MEDIUM", "causal_reasoning": "caution"},
            {"clip_path": "clips/d.mp4", "severity": "NONE", "causal_reasoning": "clear"},
        ]

        # Create dummy clip files
        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        for name in ["a.mp4", "b.mp4", "c.mp4", "d.mp4"]:
            (clips_dir / name).write_bytes(b"\x00" * 100)

        out_dir = tmp_path / "predictions"

        # Mock the predict_clip to avoid GPU usage
        with patch.object(client, "predict_clip") as mock_predict:
            mock_predict.return_value = {
                "clip_name": "test.mp4",
                "prompt": "test",
                "output_path": "test.mp4",
                "generation_time_sec": 1.0,
                "n_frames": 10,
            }

            batch_results = client.predict_batch(
                cosmos_results=results,
                clips_dir=clips_dir,
                output_dir=out_dir,
                severity_filter={"HIGH", "MEDIUM"},
            )

        # Should only process HIGH and MEDIUM (a.mp4, c.mp4)
        assert mock_predict.call_count == 2

    def test_handles_missing_clips(self, client, tmp_path):
        results = [
            {"clip_path": "clips/missing.mp4", "severity": "HIGH", "causal_reasoning": "x"},
        ]
        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        out_dir = tmp_path / "predictions"

        with patch.object(client, "predict_clip") as mock_predict:
            batch_results = client.predict_batch(
                cosmos_results=results,
                clips_dir=clips_dir,
                output_dir=out_dir,
            )

        # Should not call predict_clip for missing files
        mock_predict.assert_not_called()


# --- Unload ---

class TestUnload:
    def test_unload_clears_pipeline(self):
        cfg = MagicMock()
        cfg.cosmos.get.return_value = {}
        client = CosmosPredictClient(cfg)
        client._pipeline = MagicMock()
        client.unload()
        assert client._pipeline is None


# --- Negative prompt ---

class TestNegativePrompt:
    def test_negative_prompt_exists(self):
        assert len(NEGATIVE_PROMPT) > 20
        assert "static" in NEGATIVE_PROMPT.lower() or "monotonous" in NEGATIVE_PROMPT.lower()
