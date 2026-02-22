"""CLI command modules for AutoRisk-RM."""

from autorisk.cli_commands.audit import register_audit_commands
from autorisk.cli_commands.multi_video import register_multi_video_commands

__all__ = [
    "register_audit_commands",
    "register_multi_video_commands",
]
