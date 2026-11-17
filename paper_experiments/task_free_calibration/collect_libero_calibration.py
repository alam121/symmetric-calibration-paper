#!/usr/bin/env python3
"""Collect task-free random-excitation transitions in the LIBERO source setting."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from paper_experiments.task_free_calibration.collect_calibration import SmoothExcitation


SCHEMA = (
    "step",
    "rgb_static_t",
    "rgb_gripper_t",
    "robot_obs_t",
    "action_t",
    "rgb_static_tp1",
    "rgb_gripper_tp1",
    "robot_obs_tp1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openvla-root", type=Path, default=Path("/home/alam/vla_thesis/openvla-oft"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps-per-episode", type=int, default=64)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--action-smoothing", type=float, default=0.85)
    parser.add_argument("--translation-std", type=float, default=0.28)
    parser.add_argument("--rotation-std", type=float, default=0.18)
    parser.add_argument("--gripper-period", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=256)
    return parser.parse_args()


def quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).copy()
    quaternion /= max(np.linalg.norm(quaternion), 1e-12)
    w = float(np.clip(quaternion[3], -1.0, 1.0))
    denominator = np.sqrt(max(1.0 - w * w, 0.0))
    if denominator < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (quaternion[:3] * (2.0 * np.arccos(w) / denominator)).astype(np.float32)


def compact_robot_obs(obs: dict[str, Any]) -> np.ndarray:
    """The exact 8-D proprioception consumed by LIBERO OpenVLA-OFT."""
    position = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    orientation = quaternion_to_axis_angle(obs["robot0_eef_quat"])
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    return np.concatenate((position, orientation, gripper)).astype(np.float32)


def image(obs: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(obs[key][::-1, ::-1], dtype=np.uint8)


def append_transition(
    columns: dict[str, list[np.ndarray | int]],
    step: int,
    obs: dict[str, Any],
    action: np.ndarray,
    next_obs: dict[str, Any],
) -> None:
    columns["step"].append(step)
    columns["rgb_static_t"].append(image(obs, "agentview_image"))
    columns["rgb_gripper_t"].append(image(obs, "robot0_eye_in_hand_image"))
    columns["robot_obs_t"].append(compact_robot_obs(obs))
    columns["action_t"].append(action.astype(np.float32))
    columns["rgb_static_tp1"].append(image(next_obs, "agentview_image"))
    columns["rgb_gripper_tp1"].append(image(next_obs, "robot0_eye_in_hand_image"))
    columns["robot_obs_tp1"].append(compact_robot_obs(next_obs))


def write_manifest(output: Path, args: argparse.Namespace, completed: int) -> None:
    manifest = {
        "kind": "task_free_calibration_transitions",
        "domain": "source_libero",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "suite": args.suite,
        "seed": args.seed,
        "episodes_requested": args.episodes,
        "episodes_complete": completed,
        "steps_per_episode": args.steps_per_episode,
        "schema": list(SCHEMA),
        "contains_language": False,
        "contains_task_ids": False,
        "contains_rewards_or_success": False,
        "contains_expert_actions": False,
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "manifest.json")


def main() -> int:
    args = parse_args()
    args.openvla_root = args.openvla_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.openvla_root))
    os.chdir(args.openvla_root)

    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite_types = benchmark.get_benchmark_dict()
    if args.suite not in suite_types:
        raise SystemExit(f"unknown LIBERO suite {args.suite!r}; choose from {sorted(suite_types)}")
    suite = suite_types[args.suite]()
    rng = np.random.default_rng(args.seed)
    assignments = np.arange(args.episodes) % suite.n_tasks
    rng.shuffle(assignments)
    completed = 0

    for task_id in range(suite.n_tasks):
        episode_ids = np.flatnonzero(assignments == task_id)
        if not len(episode_ids):
            continue
        task = suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=args.resolution,
            camera_widths=args.resolution,
        )
        env.seed(args.seed + task_id)
        initial_states = suite.get_task_init_states(task_id)
        try:
            for local_index, episode_id in enumerate(episode_ids):
                output_path = args.output / f"calibration_episode_{int(episode_id):05d}.npz"
                if output_path.exists():
                    completed += 1
                    continue
                env.reset()
                obs = env.set_init_state(initial_states[local_index % len(initial_states)])
                for _ in range(args.settle_steps):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                excitation = SmoothExcitation(
                    rng,
                    args.action_smoothing,
                    args.translation_std,
                    args.rotation_std,
                    args.gripper_period,
                    workspace_low=None,
                    workspace_high=None,
                )
                columns: dict[str, list[np.ndarray | int]] = {key: [] for key in SCHEMA}
                for step in range(args.steps_per_episode):
                    action = excitation(compact_robot_obs(obs), step)
                    next_obs, _, _, _ = env.step(action.tolist())
                    append_transition(columns, step, obs, action, next_obs)
                    obs = next_obs
                arrays = {key: np.asarray(values) for key, values in columns.items()}
                np.savez_compressed(output_path, **arrays)
                completed += 1
                write_manifest(args.output, args, completed)
                print(f"[{completed}/{args.episodes}] {output_path}", flush=True)
        finally:
            env.close()
    write_manifest(args.output, args, completed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
