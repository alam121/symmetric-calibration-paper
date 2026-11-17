#!/usr/bin/env python3
"""Finish the matched 100-sequence controlled-interface confirmation."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
RESULTS = REPO / "results/task_free_calibration/controlled_interface"
PROTOCOL = REPO / "paper_experiments/task_free_calibration/controlled_interface_protocol.json"
STATE = RESULTS / "confirmation_state.json"
CALIBRATED = RESULTS / "native_task_free_calibrated__model-seed-0__n-100"
UNCALIBRATED = RESULTS / "native_hidden_interface__model-seed-0__n-100"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_state(stage: str, **extra: object) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"status": "running", "stage": stage, "updated_utc": utc_now(), **extra}
    STATE.write_text(json.dumps(payload, indent=2) + "\n")


def completed(run: Path) -> bool:
    manifest = run / "manifest.json"
    result = run / "results.json"
    if not manifest.exists() or not result.exists():
        return False
    return json.loads(manifest.read_text()).get("status") == "complete"


def wait_for(run: Path) -> None:
    while not completed(run):
        write_state("waiting_for_calibrated_n100", run=str(run))
        time.sleep(15)


def run_hidden_baseline() -> None:
    if completed(UNCALIBRATED):
        return
    command = [
        sys.executable,
        "-u",
        str(REPO / "paper_experiments/calvin_baselines/run_protocol.py"),
        "--protocol",
        str(PROTOCOL),
        "--workspace",
        str(WORKSPACE),
        "--method",
        "native_hidden_interface",
        "--model-seed",
        "0",
        "--num-sequences",
        "100",
        "--sequence-seed",
        "1729",
        "--skip-memory",
    ]
    write_state("running_hidden_interface_n100", command=command)
    completed_process = subprocess.run(command, cwd=REPO, check=False)
    if completed_process.returncode:
        raise RuntimeError(
            f"hidden-interface evaluation failed with code {completed_process.returncode}"
        )


def metrics(run: Path) -> dict[str, object]:
    payload = json.loads((run / "results.json").read_text())
    value = next(iter(payload.values()))
    return {
        "avg_seq_len": float(value["avg_seq_len"]),
        "depth_1": float(value["chain_sr"]["1"]),
        "depth_2": float(value["chain_sr"]["2"]),
        "depth_3": float(value["chain_sr"]["3"]),
        "depth_4": float(value["chain_sr"]["4"]),
        "depth_5": float(value["chain_sr"]["5"]),
    }


def wilson(success_rate: float, n: int, z: float = 1.959963984540054) -> list[float]:
    center = (success_rate + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * (
        success_rate * (1 - success_rate) / n + z * z / (4 * n * n)
    ) ** 0.5 / (1 + z * z / n)
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize() -> None:
    calibrated = metrics(CALIBRATED)
    uncalibrated = metrics(UNCALIBRATED)
    d1_gain = float(calibrated["depth_1"]) - float(uncalibrated["depth_1"])
    chain_gain = float(calibrated["avg_seq_len"]) - float(uncalibrated["avg_seq_len"])
    payload = {
        "status": "complete",
        "updated_utc": utc_now(),
        "num_sequences": 100,
        "sequence_seed": 1729,
        "uncalibrated": uncalibrated,
        "task_free_calibrated": calibrated,
        "absolute_depth_1_gain": d1_gain,
        "absolute_average_chain_gain": chain_gain,
        "depth_1_wilson_95": {
            "uncalibrated": wilson(float(uncalibrated["depth_1"]), 100),
            "task_free_calibrated": wilson(float(calibrated["depth_1"]), 100),
        },
        "development_gate": {
            "requires_depth_1_gain": 0.10,
            "passed": d1_gain >= 0.10 and chain_gain > 0.0,
        },
        "supervision": {
            "task_demonstrations": False,
            "language": False,
            "rewards_or_success": False,
            "expert_actions": False,
            "vla_gradients": False,
        },
    }
    (RESULTS / "confirmation_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    percent = lambda value: f"{100 * float(value):.1f}%"
    report = f"""# Controlled task-free calibration: 100-sequence confirmation

| Condition | Average chain | Depth-1 | Depth-2 | Depth-5 |
| --- | ---: | ---: | ---: | ---: |
| Hidden interface | {uncalibrated['avg_seq_len']:.3f} | {percent(uncalibrated['depth_1'])} | {percent(uncalibrated['depth_2'])} | {percent(uncalibrated['depth_5'])} |
| Task-free calibrated | **{calibrated['avg_seq_len']:.3f}** | **{percent(calibrated['depth_1'])}** | **{percent(calibrated['depth_2'])}** | **{percent(calibrated['depth_5'])}** |

Absolute depth-1 gain: **{100 * d1_gain:.1f} percentage points**.
Absolute average-chain gain: **{chain_gain:.3f}**.
Development gate passed: **{payload['development_gate']['passed']}**.

Calibration used no task demonstrations, language, rewards, success labels,
expert actions, or VLA gradients.
"""
    (RESULTS / "CONFIRMATION.md").write_text(report)
    STATE.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    wait_for(CALIBRATED)
    run_hidden_baseline()
    summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
