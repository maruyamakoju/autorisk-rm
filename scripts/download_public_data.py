"""Download third-party source videos for public-mode testing via yt-dlp.

Safety note:
- This script does not grant any redistribution right.
- Users must confirm they have rights/permission before download and use.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)


def _unique_non_empty(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if value == "" or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def download_public_videos(
    cfg: DictConfig,
    urls: list[str] | None = None,
    *,
    allow_third_party: bool = False,
) -> list[Path]:
    """Download public-mode videos using yt-dlp Python API.

    Args:
        cfg: Configuration with public mode settings.
        urls: Optional list of URLs to download. If omitted, uses cfg.public.source_url.
        allow_third_party: Explicit acknowledgement that source rights were reviewed.

    Returns:
        List of paths to downloaded videos.
    """
    if not allow_third_party:
        raise ValueError(
            "Refusing third-party download without explicit acknowledgement. "
            "Pass allow_third_party=True (CLI: --ack-data-rights) after confirming rights."
        )

    import yt_dlp

    download_dir = Path(cfg.public.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    max_videos = cfg.public.max_videos
    max_duration = cfg.public.max_duration_sec
    cfg_source_url = str(getattr(getattr(cfg, "public", {}), "source_url", "")).strip()
    resolved_urls = _unique_non_empty([*(urls or []), cfg_source_url])
    if not resolved_urls:
        raise ValueError(
            "No source URL provided. Set public.source_url in config or pass --url explicitly."
        )
    urls = resolved_urls[:max_videos]

    downloaded: list[Path] = []

    for i, url in enumerate(urls, 1):
        output_template = str(download_dir / f"public_{i:03d}.%(ext)s")

        ydl_opts = {
            "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
            "merge_output_format": "mp4",
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }

        if max_duration:
            ydl_opts["download_ranges"] = (
                lambda info, ydl, d=max_duration: [{"start_time": 0, "end_time": d}]
            )
            ydl_opts["force_keyframes_at_cuts"] = True

        log.info("Downloading video %d/%d: %s", i, len(urls), url)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Find the downloaded file
            for f in download_dir.glob(f"public_{i:03d}.*"):
                if f.suffix == ".mp4":
                    downloaded.append(f)
                    log.info("Downloaded: %s", f.name)
                    break
        except Exception as e:
            log.warning("Download failed for %s: %s", url, e)
            continue

    log.info("Downloaded %d/%d videos to %s", len(downloaded), len(urls), download_dir)
    return downloaded


if __name__ == "__main__":
    from autorisk.utils.config import load_config

    parser = argparse.ArgumentParser(
        description=(
            "Download public-mode source video(s). "
            "You must review source rights and pass --ack-data-rights explicitly."
        )
    )
    parser.add_argument("--config", default="configs/public.yaml", help="Config path")
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Source URL to download (repeatable). If omitted, uses public.source_url from config.",
    )
    parser.add_argument(
        "--ack-data-rights",
        action="store_true",
        help="Required acknowledgement: you reviewed source rights/license and can use this media.",
    )
    args = parser.parse_args()

    if not args.ack_data_rights:
        parser.error("--ack-data-rights is required for third-party downloads")

    cfg = load_config(config_path=args.config)
    videos = download_public_videos(
        cfg,
        urls=list(args.url),
        allow_third_party=bool(args.ack_data_rights),
    )
    for v in videos:
        print(f"  {v}")
