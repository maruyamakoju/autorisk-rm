from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from autorisk.cli import cli


def test_narrative_cli_writes_output(tmp_path: Path) -> None:
    results_path = tmp_path / "cosmos_results.json"
    results_path.write_text(
        json.dumps(
            [
                {
                    "candidate_rank": 1,
                    "severity": "LOW",
                    "causal_reasoning": "steady driving",
                    "parse_success": True,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "out.md"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["narrative", "--results", str(results_path), "--out", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert "Safety narrative generated:" in result.output

