from __future__ import annotations

from autorisk.cli import cli


def test_cli_registers_extracted_audit_and_multi_commands() -> None:
    expected = {
        "audit-pack",
        "audit-sign",
        "audit-attest",
        "audit-verify",
        "audit-verifier-bundle",
        "audit-validate",
        "audit-handoff",
        "audit-handoff-verify",
        "review-approve",
        "review-apply",
        "policy-check",
        "finalize-run",
        "multi-run",
        "submission-metrics",
        "multi-validate",
    }
    missing = sorted(name for name in expected if name not in cli.commands)
    assert missing == []
