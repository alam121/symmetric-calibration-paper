#!/usr/bin/env python3
"""Collect smooth, bounded, task-free transitions from the CALVIN environment."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


FORBIDDEN_KEYS = {
    "language",
    "instruction",
    "task",
    "task_id",
    "reward",
    "success",
    "expert_action",
}
WORKSPACE_LOW = np.asarray([-0.25, -0.40, 0.48], dtype=np.float32)
WORKSPACE_HIGH = np.asarray([0.30, 0.05, 0.75], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps-per-episode", type=int, default=64)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--action-smoothing", type=float, default=0.85)
    parser.add_argument("--translation-std", type=float, default=0.28)
    parser.add_argument("--rotation-std", type=float, default=0.18)
    parser.add_argument("--gripper-period", type=int, default=16)
    parser.add_argument(
        "--action-matrix",
        type=json.loads,
        default=None,
        help="Optional 6x6 JSON matrix mapping issued commands to physical actions.",
    )
    parser.add_argument(
        "--gripper-polarity",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
    )
    return parser.parse_args()


def deployment_action(
    action: np.ndarray,
    action_matrix: list[list[float]] | None,
    gripper_polarity: float,
) -> np.ndarray:
    """Map an issued command through a hidden actuator interface."""
    physical = np.asarray(action, dtype=np.float32).copy()
    if action_matrix is not None:
        matrix = np.asarray(action_matrix, dtype=np.float32)
        if matrix.shape != (6, 6):
            raise ValueError(f"action matrix must be 6x6, got {matrix.shape}")
        physical[:6] = physical[:6] @ matrix
    physical[6] *= gripper_polarity
    physical[:6] = np.clip(physical[:6], -1.0, 1.0)
    physical[6] = 1.0 if physical[6] >= 0.0 else -1.0
    return physical


def valid_initial_conditions() -> list[dict[str, Any]]:
    """Return scene configurations without attaching any task sequence."""
    dimensions = {
        "led": [0, 1],
        "lightbulb": [0, 1],
        "slider": ["right", "left"],
        "drawer": ["closed", "open"],
        "red_block": ["table", "slider_right", "slider_left"],
        "blue_block": ["table", "slider_right", "slider_left"],
        "pink_block": ["table", "slider_right", "slider_left"],
        "grasped": [0],
    }
    conditions = []
    for values in product(*dimensions.values()):
        condition = dict(zip(dimensions, values))
        locations = [condition[color] for color in ("red_block", "blue_block", "pink_block")]
        if locations.count("table") not in (1, 2):
            continue
        if locations.count("slider_right") >= 2 or locations.count("slider_left") >= 2:
            continue
        conditions.append(condition)
    return conditions


class SmoothExcitation:
    """Low-pass random relative actions with conservative workspace reflection."""

    def __init__(
        self,
        rng: np.random.Generator,
        smoothing: float,
        translation_std: float,
        rotation_std: float,
        gripper_period: int,
        workspace_low: np.ndarray | None = WORKSPACE_LOW,
        workspace_high: np.ndarray | None = WORKSPACE_HIGH,
    ) -> None:
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("action smoothing must be in [0, 1)")
        if gripper_period < 1:
            raise ValueError("gripper period must be positive")
        self.rng = rng
        self.smoothing = smoothing
        self.translation_std = translation_std
        self.rotation_std = rotation_std
        self.gripper_period = gripper_period
        if (workspace_low is None) != (workspace_high is None):
            raise ValueError("workspace bounds must both be set or both be None")
        self.workspace_low = None if workspace_low is None else np.asarray(workspace_low, dtype=np.float32)
        self.workspace_high = None if workspace_high is None else np.asarray(workspace_high, dtype=np.float32)
        self.motion = np.zeros(6, dtype=np.float32)
        self.gripper = 1.0

    def __call__(self, robot_obs: np.ndarray, step: int) -> np.ndarray:
        noise = np.concatenate(
            (
                self.rng.normal(0.0, self.translation_std, 3),
                self.rng.normal(0.0, self.rotation_std, 3),
            )
        ).astype(np.float32)
        self.motion = self.smoothing * self.motion + (1.0 - self.smoothing) * noise
        self.motion = np.clip(self.motion, -0.65, 0.65)

        if self.workspace_low is not None and self.workspace_high is not None:
            position = np.asarray(robot_obs[:3], dtype=np.float32)
            margin = 0.025
            for axis in range(3):
                if position[axis] < self.workspace_low[axis] + margin:
                    self.motion[axis] = abs(self.motion[axis])
                elif position[axis] > self.workspace_high[axis] - margin:
                    self.motion[axis] = -abs(self.motion[axis])

        if step and step % self.gripper_period == 0:
            self.gripper *= -1.0
        return np.concatenate((self.motion, np.asarray([self.gripper], dtype=np.float32)))


def episode_columns() -> dict[str, list[np.ndarray | int]]:
    return {
        "step": [],
        "rgb_static_t": [],
        "rgb_gripper_t": [],
        "robot_obs_t": [],
        "action_t": [],
        "rgb_static_tp1": [],
        "rgb_gripper_tp1": [],
        "robot_obs_tp1": [],
    }


def append_transition(
    columns: dict[str, list[np.ndarray | int]],
    step: int,
    obs: dict[str, Any],
    action: np.ndarray,
    next_obs: dict[str, Any],
) -> None:
    columns["step"].append(step)
    columns["rgb_static_t"].append(obs["rgb_obs"]["rgb_static"].astype(np.uint8))
    columns["rgb_gripper_t"].append(obs["rgb_obs"]["rgb_gripper"].astype(np.uint8))
    columns["robot_obs_t"].append(obs["robot_obs"].astype(np.float32))
    columns["action_t"].append(action.astype(np.float32))
    columns["rgb_static_tp1"].append(next_obs["rgb_obs"]["rgb_static"].astype(np.uint8))
    columns["rgb_gripper_tp1"].append(next_obs["rgb_obs"]["rgb_gripper"].astype(np.uint8))
    columns["robot_obs_tp1"].append(next_obs["robot_obs"].astype(np.float32))


def validate_columns(columns: dict[str, list[np.ndarray | int]]) -> None:
    forbidden = FORBIDDEN_KEYS.intersection(columns)
    if forbidden:
        raise RuntimeError(f"task supervision leaked into calibration data: {sorted(forbidden)}")
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise RuntimeError(f"invalid transition column lengths: {lengths}")


def stack_columns(columns: dict[str, list[np.ndarray | int]]) -> dict[str, np.ndarray]:
    validate_columns(columns)
    return {key: np.asarray(values) for key, values in columns.items()}


def write_manifest(output: Path, args: argparse.Namespace, completed: int) -> None:
    manifest = {
        "kind": "task_free_calibration_transitions",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_path": str(args.dataset_path.resolve()),
        "seed": args.seed,
        "episodes_requested": args.episodes,
        "episodes_complete": completed,
        "steps_per_episode": args.steps_per_episode,
        "action_smoothing": args.action_smoothing,
        "translation_std": args.translation_std,
        "rotation_std": args.rotation_std,
        "gripper_period": args.gripper_period,
        "action_matrix": args.action_matrix,
        "gripper_polarity": args.gripper_polarity,
        "schema": list(episode_columns()),
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
    if args.episodes < 1 or args.steps_per_episode < 1:
        raise SystemExit("episodes and steps per episode must be positive")
    args.dataset_path = args.dataset_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    # Imported lazily so the excitation policy and schema can be unit tested
    # without initializing PyBullet or requiring the CALVIN environment.
    from calvin_env.envs.play_table_env import get_env
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

    rng = np.random.default_rng(args.seed)
    conditions = valid_initial_conditions()
    order = rng.permutation(len(conditions))
    env = get_env(args.dataset_path / "validation", show_gui=False)
    completed = 0
    try:
        for episode in range(args.episodes):
            output_path = args.output / f"calibration_episode_{episode:05d}.npz"
            if output_path.exists():
                completed += 1
                continue
            condition = conditions[int(order[episode % len(order)])]
            robot_obs, scene_obs = get_env_state_for_initial_condition(condition)
            obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            excitation = SmoothExcitation(
                rng,
                args.action_smoothing,
                args.translation_std,
                args.rotation_std,
                args.gripper_period,
            )
            columns = episode_columns()
            for step in range(args.steps_per_episode):
                action = excitation(obs["robot_obs"], step)
                # CALVIN converts relative commands to absolute targets in
                # place. Preserve the actual normalized command for system
                # identification and pass a separate copy to the simulator.
                issued_action = action.copy()
                physical_action = deployment_action(
                    action,
                    args.action_matrix,
                    args.gripper_polarity,
                )
                next_obs, _, _, _ = env.step(physical_action.copy())
                append_transition(columns, step, obs, issued_action, next_obs)
                obs = next_obs
            np.savez_compressed(output_path, **stack_columns(columns))
            completed += 1
            write_manifest(args.output, args, completed)
            print(f"[{completed}/{args.episodes}] {output_path}", flush=True)
    finally:
        env.close()
    write_manifest(args.output, args, completed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
