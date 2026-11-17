#!/usr/bin/env python3
"""Autonomously collect, fit, and test the first task-free interface calibrator."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parent
DEFAULT_STATE = REPO / "results/task_free_calibration/first_calibration_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-result", type=Path, default=REPO / "results/embodiment_bridge/closed_loop/dynamics_aware_bridge__model-seed-0__n-1000/results.json")
    parser.add_argument("--paused-queue-pid", type=int, default=3438540)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps-per-episode", type=int, default=64)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--diagnostic-sequences", type=int, default=20)
    parser.add_argument("--development-sequences", type=int, default=100)
    parser.add_argument("--development-sequence-seed", type=int, default=1729)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    return parser.parse_args()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_utc"] = now()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def run(command: list[str], cwd: Path, state: dict[str, Any], state_path: Path, stage: str) -> None:
    state["status"] = "running"
    state["stage"] = stage
    state["command"] = command
    write_state(state_path, state)
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode:
        raise RuntimeError(f"stage {stage} failed with return code {completed.returncode}")


def result_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return next(iter(payload.values())) if payload else None


def gpu_processes() -> list[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=0",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def collection_complete(path: Path, episodes: int) -> bool:
    manifest = path / "manifest.json"
    if not manifest.exists():
        return False
    try:
        metadata = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    shards = list(path.glob("calibration_episode_*.npz"))
    return int(metadata.get("episodes_complete", 0)) >= episodes and len(shards) >= episodes


def main() -> int:
    args = parse_args()
    args.state = args.state.expanduser().resolve()
    state: dict[str, Any] = {
        "status": "waiting_for_gpu",
        "stage": "finish_supervised_baseline_seed0",
        "created_utc": now(),
        "supervision_boundary": {
            "language": False,
            "task_ids": False,
            "expert_actions": False,
            "rewards_or_success": False,
        },
    }
    write_state(args.state, state)
    while not args.wait_for_result.exists():
        time.sleep(30)

    # The old bridge queue was deliberately stopped while its current child
    # finished. Remove only that paused parent so it cannot launch five more
    # expensive baseline runs before calibration development.
    try:
        os.kill(args.paused_queue_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    while gpu_processes():
        state["gpu_processes"] = gpu_processes()
        write_state(args.state, state)
        time.sleep(5)
    state.pop("gpu_processes", None)

    source = REPO / f"data/task_free_calibration/libero_seed{args.seed}"
    target = REPO / f"data/task_free_calibration/calvin_seed{args.seed}"
    calibrator = REPO / f"data/task_free_calibration/calibrator_seed{args.seed}.json"
    if not collection_complete(source, args.episodes):
        run(
            [
                "conda", "run", "--no-capture-output", "-n", "openvla-oft", "python", "-u",
                str(HERE / "collect_libero_calibration.py"),
                "--openvla-root", str(WORKSPACE / "openvla-oft"),
                "--output", str(source), "--episodes", str(args.episodes),
                "--steps-per-episode", str(args.steps_per_episode), "--seed", str(args.seed),
            ],
            REPO,
            state,
            args.state,
            "collect_source_task_free_transitions",
        )
    if not collection_complete(target, args.episodes):
        run(
            [
                "conda", "run", "--no-capture-output", "-n", "calvin_venv", "python", "-u",
                str(HERE / "collect_calibration.py"),
                "--dataset-path", str(WORKSPACE / "calvin/dataset/task_D_D"),
                "--output", str(target), "--episodes", str(args.episodes),
                "--steps-per-episode", str(args.steps_per_episode), "--seed", str(args.seed),
            ],
            WORKSPACE / "calvin",
            state,
            args.state,
            "collect_target_task_free_transitions",
        )
    for domain, directory in (("source", source), ("target", target)):
        run(
            [
                "conda", "run", "--no-capture-output", "-n", "openvla-oft", "python",
                str(HERE / "audit_calibration.py"),
                "--data", str(directory),
                "--output", str(directory / "audit.json"),
            ],
            REPO,
            state,
            args.state,
            f"audit_{domain}_supervision_boundary",
        )
    run(
        [
            "conda", "run", "--no-capture-output", "-n", "openvla-oft", "python",
            str(HERE / "fit_calibrator.py"),
            "--source", str(source), "--target", str(target), "--output", str(calibrator),
        ],
        REPO,
        state,
        args.state,
        "fit_task_free_calibrator",
    )

    protocol = HERE / "closed_loop_protocol.json"
    results_root = REPO / "results/task_free_calibration/closed_loop"
    evaluations = (
        ("direct_source", "official"),
        ("task_free_action_only", str(args.seed)),
        ("task_free_proprio_only", str(args.seed)),
        ("task_free_calibrated", str(args.seed)),
        ("task_free_full_interface", str(args.seed)),
    )

    def evaluate(method: str, model_seed: str, sequences: int) -> dict[str, Any]:
        output = results_root / f"{method}__model-seed-{model_seed}__n-{sequences}" / "results.json"
        if result_metrics(output) is not None:
            return result_metrics(output)  # type: ignore[return-value]
        run(
            [
                sys.executable,
                str(REPO / "paper_experiments/calvin_baselines/run_protocol.py"),
                "--protocol", str(protocol), "--workspace", str(WORKSPACE),
                "--method", method, "--model-seed", model_seed,
                "--num-sequences", str(sequences),
                "--sequence-seed", str(args.development_sequence_seed),
                "--skip-memory",
            ],
            REPO,
            state,
            args.state,
            f"closed_loop_{method}_n{sequences}",
        )
        metrics = result_metrics(output)
        if metrics is None:
            raise RuntimeError(f"evaluation completed without results: {output}")
        return metrics

    diagnostic = {
        method: evaluate(method, model_seed, args.diagnostic_sequences)
        for method, model_seed in evaluations
    }
    direct = diagnostic["direct_source"]
    calibrated_conditions = {key: value for key, value in diagnostic.items() if key != "direct_source"}
    best_method, calibrated = max(
        calibrated_conditions.items(),
        key=lambda item: (item[1]["avg_seq_len"], item[1]["chain_sr"]["1"]),
    )
    improved = (
        calibrated["avg_seq_len"] > direct["avg_seq_len"]
        and calibrated["chain_sr"]["1"] > direct["chain_sr"]["1"]
    )
    development = None
    development_pass = False
    if improved:
        direct_100 = evaluate("direct_source", "official", args.development_sequences)
        calibrated_100 = evaluate(best_method, str(args.seed), args.development_sequences)
        nonzero_tasks = sum(
            int(values.get("success", 0) > 0)
            for values in calibrated_100.get("task_info", {}).values()
        )
        depth1_gain = calibrated_100["chain_sr"]["1"] - direct_100["chain_sr"]["1"]
        development_pass = depth1_gain >= 0.10 and nonzero_tasks >= 2
        development = {
            "direct": direct_100,
            "best_calibrated": calibrated_100,
            "depth1_absolute_gain": depth1_gain,
            "nonzero_success_tasks": nonzero_tasks,
            "passed": development_pass,
        }
    state.update(
        {
            "status": (
                "development_closed_loop_pass"
                if development_pass
                else "first_closed_loop_pass" if improved else "first_closed_loop_iteration_required"
            ),
            "stage": "first_calibration_complete",
            "diagnostic": diagnostic,
            "best_calibration_condition": best_method,
            "strictly_improved": improved,
            "development": development,
            "calibrator": str(calibrator),
        }
    )
    write_state(args.state, state)
    return 0 if development_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
