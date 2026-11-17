#!/usr/bin/env python3
"""Audit calibration shards for supervision leakage and exploratory coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from paper_experiments.task_free_calibration.collect_calibration import FORBIDDEN_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-episodes", type=int, default=2)
    return parser.parse_args()


def audit(path: Path, minimum_episodes: int = 2) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    supervision_flags = (
        "contains_language",
        "contains_task_ids",
        "contains_rewards_or_success",
        "contains_expert_actions",
    )
    enabled = [key for key in supervision_flags if manifest.get(key) is not False]
    if enabled:
        raise ValueError(f"manifest does not certify task-free data: {enabled}")
    shards = sorted(path.glob("calibration_episode_*.npz"))
    if len(shards) < minimum_episodes:
        raise ValueError(f"only {len(shards)} shards; require at least {minimum_episodes}")

    actions = []
    steps = 0
    schema = None
    for shard in shards:
        with np.load(shard, allow_pickle=False) as archive:
            keys = set(archive.files)
            forbidden = FORBIDDEN_KEYS.intersection(keys)
            if forbidden:
                raise ValueError(f"{shard} contains forbidden keys: {sorted(forbidden)}")
            required = {"action_t", "robot_obs_t", "robot_obs_tp1"}
            if missing := required.difference(keys):
                raise ValueError(f"{shard} is missing {sorted(missing)}")
            if schema is None:
                schema = sorted(keys)
            elif schema != sorted(keys):
                raise ValueError(f"inconsistent schema in {shard}")
            action = archive["action_t"].astype(np.float64)
            if not np.isfinite(action).all() or action.ndim != 2 or action.shape[1] != 7:
                raise ValueError(f"invalid actions in {shard}: {action.shape}")
            actions.append(action)
            steps += len(action)

    action = np.concatenate(actions)
    motion_std = action[:, :6].std(axis=0)
    if np.any(motion_std < 0.005):
        raise ValueError(f"insufficient excitation; motion std={motion_std.tolist()}")
    if not set(np.unique(action[:, 6]).tolist()).issubset({-1.0, 1.0}):
        raise ValueError("gripper commands are not binary")
    if len(np.unique(action[:, 6])) != 2:
        raise ValueError("both gripper directions must be excited")
    adjacent = np.concatenate([values[1:, :6] - values[:-1, :6] for values in actions])
    report = {
        "status": "passed",
        "path": str(path.resolve()),
        "episodes": len(shards),
        "transitions": steps,
        "schema": schema,
        "forbidden_keys_present": [],
        "manifest_supervision_flags": {key: manifest[key] for key in supervision_flags},
        "motion_action_std": motion_std.tolist(),
        "mean_adjacent_motion_change": float(np.mean(np.abs(adjacent))),
        "gripper_values": np.unique(action[:, 6]).tolist(),
    }
    return report


def main() -> int:
    args = parse_args()
    report = audit(args.data.expanduser().resolve(), args.minimum_episodes)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

