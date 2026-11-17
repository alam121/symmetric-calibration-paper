#!/usr/bin/env python3
"""Export CALVIN demonstration transitions for a demonstration-supervised W fit.

Produces the (action, state_before, state_after) triples needed to fit

    d_t = phi(x_{t+1}, x_t) ~ a_t W + b

from expert demonstrations, so that a demo-fitted W can be compared against the
task-free random-interaction W used by the calibrator.

Field conventions, matching fit_local_action_model / physical_pose_delta in
calibrator.py so the two fits are directly comparable:

  action       CALVIN rel_actions, 7-D. [0:3] translation, [3:6] rotation,
               [6] gripper in {-1, +1}. Already normalised: translation is
               scaled by 1/0.02 and rotation by 1/0.05, clipped to [-1, 1].
               This is what the environment executes, so it is the native
               action, not a re-derived one.
  state        robot_obs[0:7]. [0:3] TCP position (m, world frame),
               [3:6] TCP orientation as euler xyz (rad), [6] gripper width (m).
  delta        phi(x_{t+1}, x_t) = [position difference (3),
               relative rotation as a rotation vector (3)]. The rotation term
               is (R_{t+1} R_t^-1) expressed as a rotation vector, NOT a
               component-wise subtraction of euler angles, which would be wrong
               across the +-pi wrap.

Only the 6 motion dimensions enter the linear fit; the gripper is handled
separately as a scalar slope of gripper-width change against gripper command.

Episode boundaries come from ep_start_end_ids.npy. CALVIN stores frames as one
flat sequence, so transitions must never be formed across a boundary; a pair
spanning two episodes would teleport the end effector and corrupt the fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

CALVIN = Path("/home/alam/vla_thesis/calvin/dataset/task_D_D")


def phi(state: np.ndarray, next_state: np.ndarray) -> np.ndarray:
    """Same as calibrator.physical_pose_delta(..., orientation_kind='euler_xyz')."""
    translation = next_state[:, :3] - state[:, :3]
    r0 = Rotation.from_euler("xyz", state[:, 3:6])
    r1 = Rotation.from_euler("xyz", next_state[:, 3:6])
    return np.concatenate([translation, (r1 * r0.inv()).as_rotvec()], axis=1)


def export(split: str, out: Path, max_episodes: int | None) -> dict:
    bounds = np.load(CALVIN / split / "ep_start_end_ids.npy")
    if max_episodes:
        bounds = bounds[:max_episodes]

    A, S, S1, EP, FR = [], [], [], [], []
    for ep, (lo, hi) in enumerate(bounds):
        idx = np.arange(int(lo), int(hi) + 1)
        frames = [np.load(CALVIN / split / f"episode_{i:07d}.npz") for i in idx]
        act = np.stack([f["rel_actions"] for f in frames]).astype(np.float64)
        obs = np.stack([f["robot_obs"][:7] for f in frames]).astype(np.float64)
        # transition t -> t+1, never across the episode boundary
        A.append(act[:-1]); S.append(obs[:-1]); S1.append(obs[1:])
        EP.append(np.full(len(idx) - 1, ep, dtype=np.int64))
        FR.append(idx[:-1])
        print(f"  episode {ep:3d}  frames {int(lo)}..{int(hi)}  transitions {len(idx)-1}", flush=True)

    A = np.concatenate(A); S = np.concatenate(S); S1 = np.concatenate(S1)
    EP = np.concatenate(EP); FR = np.concatenate(FR)
    D = phi(S, S1)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        action=A, state_before=S, state_after=S1, delta=D,
        episode_id=EP, frame_index=FR,
        field_notes=json.dumps({
            "action": "CALVIN rel_actions [dx,dy,dz,drx,dry,drz,gripper]; motion normalised, gripper in {-1,+1}",
            "state_before/state_after": "robot_obs[0:7] = [tcp_x,tcp_y,tcp_z, euler_x,euler_y,euler_z, gripper_width]",
            "delta": "phi(x_{t+1},x_t) = [dpos(3), relative rotation as rotvec(3)]",
            "fit": "design=[action[:, :6], 1]; solve ridge(1e-4, no penalty on bias) -> W(6x6), b(6)",
            "gripper": "slope of (gripper_width_{t+1}-gripper_width_t) on action[:,6]",
            "episode_id": "index into ep_start_end_ids.npy; transitions never cross episodes",
        }),
    )
    return {"transitions": int(len(A)), "episodes": int(len(bounds)), "path": str(out)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="training", choices=("training", "validation"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()
    info = export(args.split, args.out, args.max_episodes)
    print(f"wrote {info['path']}: {info['transitions']} transitions from {info['episodes']} episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
