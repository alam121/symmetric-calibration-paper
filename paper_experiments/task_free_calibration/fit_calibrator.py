#!/usr/bin/env python3
"""Fit a source-to-target interface solely from task-free transition shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paper_experiments.task_free_calibration.calibrator import (
    TaskFreeCalibrator,
    fit_local_action_model,
    physical_pose_delta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=1729)
    parser.add_argument(
        "--source-orientation",
        choices=("rotvec", "euler_xyz", "euler_xyz_body", "linear"),
        default="rotvec",
    )
    parser.add_argument(
        "--target-orientation",
        choices=("rotvec", "euler_xyz", "euler_xyz_body", "linear"),
        default="euler_xyz",
    )
    parser.add_argument("--source-gripper-width", type=int, choices=(1, 2), default=2)
    parser.add_argument("--target-gripper-width", type=int, choices=(1, 2), default=1)
    return parser.parse_args()


def load_directory(path: Path, validation_fraction: float, seed: int) -> tuple[np.ndarray, ...]:
    shards = sorted(path.glob("calibration_episode_*.npz"))
    if len(shards) < 2:
        raise ValueError(f"need at least two episode shards under {path}")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    validation_count = max(1, min(len(shards) - 1, round(len(shards) * validation_fraction)))
    validation_episodes = set(rng.choice(len(shards), size=validation_count, replace=False).tolist())
    actions, states, next_states, validation = [], [], [], []
    for index, shard in enumerate(shards):
        with np.load(shard, allow_pickle=False) as archive:
            required = {"action_t", "robot_obs_t", "robot_obs_tp1"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{shard} is missing {sorted(missing)}")
            action = archive["action_t"].astype(np.float64)
            state = archive["robot_obs_t"].astype(np.float64)
            next_state = archive["robot_obs_tp1"].astype(np.float64)
        actions.append(action)
        states.append(state)
        next_states.append(next_state)
        validation.append(np.full(len(action), index in validation_episodes, dtype=bool))
    return tuple(np.concatenate(values) for values in (actions, states, next_states, validation))


def image_statistics(path: Path) -> dict[str, dict[str, list[float]]]:
    result = {}
    for camera, key in (("static", "rgb_static_t"), ("wrist", "rgb_gripper_t")):
        total = np.zeros(3, dtype=np.float64)
        squared = np.zeros(3, dtype=np.float64)
        pixels = 0
        for shard in sorted(path.glob("calibration_episode_*.npz")):
            with np.load(shard, allow_pickle=False) as archive:
                images = archive[key].astype(np.float64)
            total += images.sum(axis=tuple(range(images.ndim - 1)))
            squared += np.square(images).sum(axis=tuple(range(images.ndim - 1)))
            pixels += int(np.prod(images.shape[:-1]))
        mean = total / pixels
        variance = np.maximum(squared / pixels - mean**2, 1e-6)
        result[camera] = {"mean": mean.tolist(), "std": np.sqrt(variance).tolist()}
    return result


def proprio_statistics(
    path: Path,
    orientation_kind: str,
    gripper_width: int,
) -> tuple[list[float], list[float]]:
    absolute = []
    relative_pose = []
    for shard in sorted(path.glob("calibration_episode_*.npz")):
        with np.load(shard, allow_pickle=False) as archive:
            state = archive["robot_obs_t"].astype(np.float64)
        reference = np.repeat(state[:1], len(state), axis=0)
        relative_pose.append(physical_pose_delta(reference, state, orientation_kind))
        absolute.append(state)
    absolute_array = np.concatenate(absolute)
    relative_array = np.concatenate(relative_pose)
    mean = np.concatenate((absolute_array[:, :6].mean(axis=0), absolute_array[:, 6 : 6 + gripper_width].mean(axis=0)))
    std = np.concatenate((relative_array.std(axis=0), absolute_array[:, 6 : 6 + gripper_width].std(axis=0)))
    return mean.tolist(), np.maximum(std, 1e-6).tolist()


def main() -> int:
    args = parse_args()
    source = load_directory(args.source, args.validation_fraction, args.split_seed)
    target = load_directory(args.target, args.validation_fraction, args.split_seed + 1)
    source_model = fit_local_action_model(
        *source,
        ridge=args.ridge,
        orientation_kind=args.source_orientation,
    )
    target_model = fit_local_action_model(
        *target,
        ridge=args.ridge,
        orientation_kind=args.target_orientation,
    )
    source_mean, source_std = proprio_statistics(
        args.source,
        args.source_orientation,
        args.source_gripper_width,
    )
    target_mean, target_std = proprio_statistics(
        args.target,
        args.target_orientation,
        args.target_gripper_width,
    )
    calibrator = TaskFreeCalibrator(
        source_model,
        target_model,
        args.ridge,
        source_proprio_mean=source_mean,
        source_proprio_std=source_std,
        target_proprio_mean=target_mean,
        target_proprio_std=target_std,
        source_image_stats=image_statistics(args.source),
        target_image_stats=image_statistics(args.target),
    )
    calibrator.save(args.output)
    print(json.dumps(calibrator.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
