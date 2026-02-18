"""Report generation module."""

from __future__ import annotations

from typing import Any

__all__ = ["ReportBuilder"]


def __getattr__(name: str) -> Any:
    if name == "ReportBuilder":
        from autorisk.report.build_report import ReportBuilder

        return ReportBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
