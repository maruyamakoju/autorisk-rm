#!/usr/bin/env python3
"""Evaluate v4 prompt outputs against existing baseline artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], desc: str) -> bool:
    print("\n" + "=" * 60)
    print(f"Running: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] {desc}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[FAIL] {desc} (exit={exc.returncode})")
        return False


def load_results(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {
        "total": len(data),
        "parse_success": sum(1 for row in data if row.get("parse_success", False)),
        "severities": {
            sev: sum(1 for row in data if row.get("severity") == sev)
            for sev in ("HIGH", "MEDIUM", "LOW", "NONE")
        },
    }


def main() -> None:
    print("AutoRisk-RM v4 Evaluation Pipeline")
    print("=" * 60)

    v3_results = Path("outputs/public_run/cosmos_results.json")
    v4_results = Path("outputs/public_run_v4/cosmos_results.json")
    if not v4_results.exists():
        print("ERROR: v4 results not found.")
        raise SystemExit(1)

    v3_data = load_results(v3_results)
    v4_data = load_results(v4_results)

    print("\n" + "=" * 60)
    print("Quick Stats Comparison")
    print("=" * 60)
    print(f"{'Metric':<20} {'v3':<15} {'v4':<15}")
    print("-" * 60)
    print(f"{'Total clips':<20} {v3_data['total']:<15} {v4_data['total']:<15}")
    print(f"{'Parse success':<20} {v3_data['parse_success']:<15} {v4_data['parse_success']:<15}")
    for sev in ("HIGH", "MEDIUM", "LOW", "NONE"):
        print(f"{sev:<20} {v3_data['severities'][sev]:<15} {v4_data['severities'][sev]:<15}")

    if not run_cmd(
        [sys.executable, "-m", "autorisk.cli", "eval", "-r", str(v4_results), "-o", "outputs/public_run_v4"],
        "v4 Evaluation",
    ):
        raise SystemExit(1)

    run_cmd(
        [sys.executable, "-m", "autorisk.cli", "report", "-r", str(v4_results), "-o", "outputs/public_run_v4"],
        "v4 Report Generation",
    )
    run_cmd(
        [
            sys.executable,
            "-m",
            "autorisk.cli",
            "narrative",
            "-r",
            str(v4_results),
            "-o",
            "outputs/public_run_v4/safety_narrative.md",
        ],
        "v4 Safety Narrative",
    )

    v3_eval = Path("outputs/public_run/eval_report.json")
    v4_eval = Path("outputs/public_run_v4/eval_report.json")
    if v3_eval.exists() and v4_eval.exists():
        with v3_eval.open(encoding="utf-8") as f:
            v3_metrics = json.load(f)
        with v4_eval.open(encoding="utf-8") as f:
            v4_metrics = json.load(f)

        print("\n" + "=" * 60)
        print("Evaluation Comparison")
        print("=" * 60)
        print(f"{'Metric':<20} {'v3':<15} {'v4':<15} {'Delta':<15}")
        print("-" * 60)

        acc_v3 = float(v3_metrics.get("accuracy", 0))
        acc_v4 = float(v4_metrics.get("accuracy", 0))
        print(f"{'Accuracy':<20} {acc_v3:<15.1%} {acc_v4:<15.1%} {acc_v4 - acc_v3:+.1%}")

        f1_v3 = float(v3_metrics.get("macro_f1", 0))
        f1_v4 = float(v4_metrics.get("macro_f1", 0))
        print(f"{'Macro-F1':<20} {f1_v3:<15.3f} {f1_v4:<15.3f} {f1_v4 - f1_v3:+.3f}")

        cl_v3 = float(v3_metrics.get("checklist_means", {}).get("mean_total", 0))
        cl_v4 = float(v4_metrics.get("checklist_means", {}).get("mean_total", 0))
        print(f"{'Checklist (0-5)':<20} {cl_v3:<15.2f} {cl_v4:<15.2f} {cl_v4 - cl_v3:+.2f}")

    print("\n[DONE] Evaluation pipeline complete.")
    print("Results: outputs/public_run_v4/")


if __name__ == "__main__":
    main()

