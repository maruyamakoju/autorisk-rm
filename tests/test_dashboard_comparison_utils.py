from __future__ import annotations

from pathlib import Path

from autorisk.dashboard.comparison_utils import (
    compute_run_kpis,
    discover_runs,
    mean_signal_scores,
    positive_min_ttc_values,
    severity_counts,
    severity_ratios,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")


def test_discover_runs_loads_valid_and_reports_skipped(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    _touch(outputs / "public_run" / "cosmos_results.json")
    _touch(outputs / "broken_run" / "cosmos_results.json")
    _touch(outputs / "no_results_run" / "candidates.csv")

    def fake_loader(path: str) -> dict:
        run_name = Path(path).name
        if run_name == "broken_run":
            raise RuntimeError("bad json payload")
        return {"run_name": run_name}

    runs, skipped = discover_runs(outputs_root=outputs, loader=fake_loader)
    assert set(runs.keys()) == {"UK Urban"}  # mapped from public_run
    assert runs["UK Urban"]["run_name"] == "public_run"
    assert len(skipped) == 1
    assert "broken_run" in skipped[0]


def test_compute_helpers_return_stable_values() -> None:
    run_data = {
        "cosmos_results": [
            {"severity": "LOW", "parse_success": True, "confidence": 0.3},
            {"severity": "HIGH", "parse_success": False, "confidence": "0.7"},
            {"severity": "LOW", "parse_success": True, "confidence": None},
        ],
        "eval_report": {
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "checklist_means": {"mean_total": 4.25},
        },
    }
    kpi = compute_run_kpis(run_data)
    assert kpi["clips"] == 3
    assert kpi["parse_ok"] == 2
    assert kpi["accuracy"] == 0.5
    assert kpi["macro_f1"] == 0.4
    assert kpi["checklist_mean_total"] == 4.25

    counts = severity_counts(run_data["cosmos_results"])
    assert counts["LOW"] == 2
    assert counts["HIGH"] == 1

    ratios = severity_ratios(run_data["cosmos_results"])
    assert ratios["LOW"] == 2 / 3
    assert ratios["HIGH"] == 1 / 3

    means = mean_signal_scores(
        [
            {"signal_scores": {"audio": 0.2, "motion": 0.4, "proximity": 0.6}},
            {"signal_scores": {"audio": "0.4", "motion": 0.2, "proximity": 0.8}},
            {"signal_scores": "invalid-row"},
        ]
    )
    assert round(means["audio"], 5) == 0.3
    assert round(means["motion"], 5) == 0.3
    assert round(means["proximity"], 5) == 0.7

    ttc = positive_min_ttc_values(
        [
            {"min_ttc": 1.2},
            {"min_ttc": -1},
            {"min_ttc": "0.8"},
            {"min_ttc": "bad"},
        ]
    )
    assert ttc == [1.2, 0.8]

