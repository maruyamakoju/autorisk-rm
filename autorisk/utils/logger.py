"""Logging configuration for AutoRisk-RM."""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "autorisk",
    level: str = "INFO",
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        name: Logger name.
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for log output.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.encoding = "utf-8"
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
