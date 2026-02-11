"""Download public dashcam videos for testing using yt-dlp Python API."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)

# Curated list of Creative Commons / public domain dashcam compilations
PUBLIC_SOURCES = [
    "https://www.youtube.com/watch?v=iHzzSao6ypE",
    "https://www.youtube.com/watch?v=5GJGDw1mEsc",
    "https://www.youtube.com/watch?v=3yMR4MF8PJc",
]


def download_public_videos(
    cfg: DictConfig,
    urls: list[str] | None = None,
) -> list[Path]:
    """Download public dashcam videos using yt-dlp Python API.

    Args:
        cfg: Configuration with public mode settings.
        urls: Optional list of URLs to download. Defaults to PUBLIC_SOURCES.

    Returns:
        List of paths to downloaded videos.
    """
    import yt_dlp

    download_dir = Path(cfg.public.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    max_videos = cfg.public.max_videos
    max_duration = cfg.public.max_duration_sec
    urls = (urls or PUBLIC_SOURCES)[:max_videos]

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

    cfg = load_config(config_path="configs/public.yaml")
    videos = download_public_videos(cfg)
    for v in videos:
        print(f"  {v}")
