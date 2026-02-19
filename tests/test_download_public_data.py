from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from scripts.download_public_data import _unique_non_empty, download_public_videos


def test_unique_non_empty_filters_blank_and_duplicates() -> None:
    values = ["", "  ", "https://a", "https://a", "https://b"]
    assert _unique_non_empty(values) == ["https://a", "https://b"]


def test_download_public_videos_requires_explicit_ack(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "public": {
                "download_dir": str(tmp_path / "public_samples"),
                "max_videos": 1,
                "max_duration_sec": 60,
                "source_url": "https://example.com/video",
            }
        }
    )

    with pytest.raises(ValueError, match="allow_third_party=True"):
        download_public_videos(
            cfg,
            urls=["https://example.com/video"],
            allow_third_party=False,
        )
