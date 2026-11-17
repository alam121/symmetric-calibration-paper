#!/usr/bin/env python3
"""Prepare and evaluate task-free target-interaction budget ablations."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
PACKAGE = REPO / "paper_experiments/task_free_calibration"
DATA = REPO / "data/task_free_calibration"
SOURCE = DATA / "calvin_seed101"
TARGET = DATA / "calvin_axis_swap_seed201"
RESULTS = REPO / "results/task_free_calibration/budget_ablation"
BASE_PROTOCOL = PACKAGE / "controlled_interface_protocol.json"
BUDGETS = (64, 320, 640)


def subset_directory(budget: int) -> Path:
    destination = DATA / f"calvin_axis_swap_seed201_budget{budget}"
    destination.mkdir(parents=True, exist_ok=True)
    shards = sorted(TARGET.glob("calibration_episode_*.npz"))
    selected_count = max(2, math.ceil(budget / 64))
    selected = shards[:selected_count]
    base, remainder = divmod(budget, len(selected))
    counts = [base + int(index < remainder) for index in range(len(selected))]
    for index, (source, count) in enumerate(zip(selected, counts)):
        output = destination / f"calibration_episode_{index:05d}.npz"
        with np.load(source, allow_pickle=False) as archive:
            arrays = {key: archive[key][:count] for key in archive.files}
        np.savez_compressed(output, **arrays)
    manifest = json.loads((TARGET / "manifest.json").read_text())
    manifest.update(
        {
            "kind": "task_free_calibration_budget_subset",
            "source_directory": str(TARGET.resolve()),
            "transition_budget": budget,
            "episodes_complete": len(selected),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def fit(budget: int, target: Path) -> Path:
    output = DATA / f"calibrator_calvin_axis_swap_budget{budget}_seed201.json"
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "openvla-oft",
        "python",
        "-u",
        str(PACKAGE / "fit_calibrator.py"),
        "--source",
        str(SOURCE),
        "--target",
        str(target),
        "--output",
        str(output),
        "--source-orientation",
        "euler_xyz",
        "--target-orientation",
        "euler_xyz",
        "--source-gripper-width",
        "1",
        "--target-gripper-width",
        "1",
        "--split-seed",
        "1729",
    ]
    subprocess.run(command, cwd=REPO, check=True)
    return output


def make_protocol(calibrators: dict[int, Path]) -> Path:
    protocol = json.loads(BASE_PROTOCOL.read_text())
    template = protocol["methods"]["native_task_free_calibrated"]
    methods = {}
    for budget, calibrator in calibrators.items():
        method = json.loads(json.dumps(template))
        method["description"] = (
            f"Frozen VLA calibrated from {budget} task-free target transitions."
        )
        method["environment"]["CALVIN_TASK_FREE_CALIBRATION_PATH"] = str(
            calibrator.resolve()
        )
        methods[f"budget_{budget}"] = method
    protocol["methods"] = methods
    protocol["paths"]["results_root"] = (
        "vla-deployment-benchmark/results/task_free_calibration/budget_ablation"
    )
    output = PACKAGE / "budget_ablation_protocol.generated.json"
    output.write_text(json.dumps(protocol, indent=2) + "\n")
    return output


def evaluate(protocol: Path, budget: int) -> None:
    run = RESULTS / f"budget_{budget}__model-seed-0__n-20"
    if (run / "results.json").exists():
        return
    command = [
        "/home/alam/miniconda3/bin/python",
        "-u",
        str(REPO / "paper_experiments/calvin_baselines/run_protocol.py"),
        "--protocol",
        str(protocol),
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
    ]
    environment = os.environ.copy()
    for key in (
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_PROMPT_MODIFIER",
        "_CE_CONDA",
        "_CE_M",
    ):
        environment.pop(key, None)
    subprocess.run(command, cwd=REPO, env=environment, check=True)


def result(run: Path) -> dict[str, float]:
    value = next(iter(json.loads((run / "results.json").read_text()).values()))
    return {
        "avg_seq_len": float(value["avg_seq_len"]),
        "depth_1": float(value["chain_sr"]["1"]),
        "depth_2": float(value["chain_sr"]["2"]),
    }


def summarize() -> None:
    rows = {}
    for budget in BUDGETS:
        rows[str(budget)] = result(
            RESULTS / f"budget_{budget}__model-seed-0__n-20"
        )
    rows["1280"] = result(
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
    RESULTS.mkdir(parents=True, exist_ok=True)
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
    calibrators = {budget: fit(budget, subset_directory(budget)) for budget in BUDGETS}
    protocol = make_protocol(calibrators)
    for budget in BUDGETS:
        evaluate(protocol, budget)
    summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
