#!/usr/bin/env python3
"""Run prepared budget checkpoints from the neutral base environment."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
PROTOCOL = REPO / "paper_experiments/task_free_calibration/budget_ablation_protocol.generated.json"
RESULTS = REPO / "results/task_free_calibration/budget_ablation"


def evaluate(budget: int) -> None:
    run = RESULTS / f"budget_{budget}__model-seed-0__n-20"
    if (run / "results.json").exists():
        return
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(REPO / "paper_experiments/calvin_baselines/run_protocol.py"),
            "--protocol",
            str(PROTOCOL),
            "--workspace",
            str(WORKSPACE),
            "--method",
            f"budget_{budget}",
            "--model-seed",
            "0",
            "--num-sequences",
            "20",
            "--sequence-seed",
            "1729",
            "--skip-memory",
            "--overwrite",
        ],
        cwd=REPO,
        check=True,
    )


def read_result(path: Path) -> dict[str, float]:
    value = next(iter(json.loads((path / "results.json").read_text()).values()))
    return {
        "avg_seq_len": float(value["avg_seq_len"]),
        "depth_1": float(value["chain_sr"]["1"]),
        "depth_2": float(value["chain_sr"]["2"]),
    }


def summarize() -> None:
    rows = {
        str(budget): read_result(RESULTS / f"budget_{budget}__model-seed-0__n-20")
        for budget in (64, 320, 640)
    }
    rows["1280"] = read_result(
        REPO
        / "results/task_free_calibration/controlled_interface"
        / "native_task_free_calibrated__model-seed-0__n-20"
    )
    payload = {
        "status": "complete",
        "evaluation_sequences": 20,
        "sequence_seed": 1729,
        "budgets": rows,
    }
    (RESULTS / "budget_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Task-free target-interaction budget pilot",
        "",
        "| Target transitions | Average chain | Depth-1 | Depth-2 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for budget in (64, 320, 640, 1280):
        value = rows[str(budget)]
        lines.append(
            f"| {budget} | {value['avg_seq_len']:.3f} | "
            f"{100 * value['depth_1']:.1f}% | {100 * value['depth_2']:.1f}% |"
        )
    (RESULTS / "BUDGET_ABLATION.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    for budget in (64, 320, 640):
        evaluate(budget)
    summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
