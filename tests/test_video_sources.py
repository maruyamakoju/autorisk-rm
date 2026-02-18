from __future__ import annotations

from pathlib import Path

import pytest

from autorisk.utils.video_sources import VideoSource, list_video_sources, validate_video_source


def test_list_video_sources_valid_for_repo_configs() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    sources = list_video_sources(repo_root=repo_root)
    names = [s["name"] for s in sources]
    assert names == ["public", "japan", "winter", "us_highway"]
    for source in sources:
        assert Path(source["config_path"]).exists()
        assert source["output_dir"].startswith("outputs/")


def test_validate_video_source_rejects_invalid_weights(tmp_path: Path) -> None:
    cfg_path = tmp_path / "configs" / "bad.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        "\n".join(
            [
                "general:",
                "  output_dir: \"outputs/bad_run\"",
                "mining:",
                "  top_n: 10",
                "  audio:",
                "    weight: 0.8",
                "  motion:",
                "    weight: 0.8",
                "  proximity:",
                "    weight: 0.1",
                "cosmos:",
                "  backend: \"local\"",
                "  local_fps: 4",
                "report:",
                "  format: \"html\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    source = VideoSource(
        name="bad",
        config_path="configs/bad.yaml",
        output_dir="outputs/bad_run",
    )
    with pytest.raises(ValueError, match="weights must sum"):
        validate_video_source(source, repo_root=tmp_path)


def test_list_video_sources_unknown_source_raises() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    with pytest.raises(ValueError, match="unknown source"):
        list_video_sources(repo_root=repo_root, only="does_not_exist")
