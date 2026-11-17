#!/usr/bin/env python3
"""Gate before training the Calibration+FT arm: does relabelling round-trip?

The calibrated arm keeps the policy in its LIBERO action convention and lets the
task-free calibrator convert to CALVIN at deployment. For that to be sound, the
training-label conversion must be the exact inverse of the deployment map:

    deployment_calibrator( relabel( a_calvin ) )  ==  a_calvin

Deployment (calibrator.map_action) is affine, not linear, and clips:

    m_tgt = ((m_src @ S + b_S) - b_T) @ pinv(T),  clipped to [-1, 1]
    g_tgt = polarity * binarise(g_src),  polarity = sign(slope_S * slope_T)

so the inverse is

    m_src = (m_tgt @ T + b_T - b_S) @ pinv(S)
    g_src = polarity * g_tgt

Three things are checked separately because they fail in different ways:
translation, rotation, and the gripper -- the last especially, since source and
target slopes have opposite signs here and a polarity error is invisible in
aggregate motion error while destroying every manipulation.

Clipping is reported rather than hidden: it is genuinely lossy and no matrix
inverts it, so the honest question is how often it fires on real data.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from paper_experiments.task_free_calibration.calibrator import (  # noqa: E402
    TaskFreeCalibrator,
)

CALVIN = Path("/home/alam/vla_thesis/calvin/dataset/task_D_D/training")


def load(path: Path) -> TaskFreeCalibrator:
    payload = json.loads(path.read_text())
    from paper_experiments.task_free_calibration.calibrator import LocalActionModel
    return TaskFreeCalibrator(
        source=LocalActionModel.from_dict(payload["source"]),
        target=LocalActionModel.from_dict(payload["target"]),
        ridge=payload["ridge"],
        source_domain=payload.get("source_domain", "LIBERO"),
        target_domain=payload.get("target_domain", "CALVIN"),
    )


def relabel(cal: TaskFreeCalibrator, a_calvin: np.ndarray) -> np.ndarray:
    """CALVIN action -> the LIBERO-convention action that deploys to it."""
    a = np.asarray(a_calvin, dtype=np.float64).reshape(-1, 7)
    S = np.asarray(cal.source.matrix, dtype=np.float64)
    T = np.asarray(cal.target.matrix, dtype=np.float64)
    bS = np.asarray(cal.source.offset, dtype=np.float64)
    bT = np.asarray(cal.target.offset, dtype=np.float64)

    motion = (a[:, :6] @ T + bT - bS) @ np.linalg.pinv(S)

    polarity = np.sign(cal.source.gripper_slope * cal.target.gripper_slope)
    if polarity == 0:
        polarity = 1.0
    grip = polarity * np.where(a[:, 6] >= 0.0, 1.0, -1.0)

    out = np.concatenate([motion, grip[:, None]], axis=1)
    return out.reshape(np.asarray(a_calvin).shape)


def report(name: str, err: np.ndarray) -> None:
    print(f"    {name:14s} mean {err.mean():.3e}   max {err.max():.3e}   "
          f"p99 {np.percentile(err, 99):.3e}")


def check(cal: TaskFreeCalibrator, actions: np.ndarray, label: str) -> bool:
    print(f"\n  {label}  (n={len(actions)})")
    lib = relabel(cal, actions)
    back = np.asarray(cal.map_action(lib.astype(np.float32)), dtype=np.float64)

    # How often does the deployment clip bind? It is lossy and uninvertible.
    S = np.asarray(cal.source.matrix); T = np.asarray(cal.target.matrix)
    bS = np.asarray(cal.source.offset); bT = np.asarray(cal.target.offset)
    unclipped = ((lib[:, :6] @ S + bS) - bT) @ np.linalg.pinv(T)
    clipped = np.abs(unclipped) > 1.0
    print(f"    clipping fired on {100*clipped.any(axis=1).mean():.2f}% of actions, "
          f"{100*clipped.mean():.2f}% of motion dims")

    report("translation", np.abs(back[:, :3] - actions[:, :3]).max(axis=1))
    report("rotation", np.abs(back[:, 3:6] - actions[:, 3:6]).max(axis=1))

    g_in = np.where(actions[:, 6] >= 0.0, 1.0, -1.0)
    g_match = (back[:, 6] == g_in)
    print(f"    gripper        exact match {100*g_match.mean():.2f}%  "
          f"({int((~g_match).sum())} mismatches)")

    ok_motion = np.abs(back[:, :6] - actions[:, :6]).max() < 1e-3
    ok_grip = g_match.all()
    print(f"    -> motion {'OK' if ok_motion else 'FAIL'}, "
          f"gripper {'OK' if ok_grip else 'FAIL'}")
    return bool(ok_motion and ok_grip)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibrator", type=Path,
                    default=REPO / "data/task_free_calibration/calibrator_seed101.json")
    ap.add_argument("--samples", type=int, default=20000)
    args = ap.parse_args()

    cal = load(args.calibrator)
    print(f"calibrator: {args.calibrator.name}  "
          f"{cal.source_domain} -> {cal.target_domain}")
    bS = np.asarray(cal.source.offset); bT = np.asarray(cal.target.offset)
    print(f"  |offset| source {np.abs(bS).max():.3e}  target {np.abs(bT).max():.3e}  "
          f"(affine term; ignoring it would bias every label)")
    print(f"  gripper slopes: source {cal.source.gripper_slope:+.6f}  "
          f"target {cal.target.gripper_slope:+.6f}  -> polarity "
          f"{np.sign(cal.source.gripper_slope*cal.target.gripper_slope):+.0f}")

    rng = np.random.default_rng(0)
    ok = True

    synthetic = np.concatenate([
        rng.uniform(-1, 1, (args.samples, 6)),
        rng.choice([-1.0, 1.0], (args.samples, 1)),
    ], axis=1)
    ok &= check(cal, synthetic, "synthetic uniform actions")

    files = sorted(glob.glob(str(CALVIN / "episode_*.npz")))[:args.samples]
    real = np.stack([np.load(f)["rel_actions"] for f in files]).astype(np.float64)
    ok &= check(cal, real, "real CALVIN demonstration actions")

    # A gripper-only probe: open and close must survive the round trip.
    print("\n  explicit gripper probe")
    for g, name in ((1.0, "open (+1)"), (-1.0, "close (-1)")):
        a = np.zeros((1, 7)); a[0, 6] = g
        lib = relabel(cal, a)
        back = cal.map_action(lib.astype(np.float32))
        status = "OK" if back[0, 6] == g else "FAIL"
        print(f"    {name:11s} calvin {g:+.0f} -> libero {lib[0,6]:+.0f} "
              f"-> calvin {back[0,6]:+.0f}   {status}")
        ok &= bool(back[0, 6] == g)

    print(f"\n{'GATE PASSED' if ok else 'GATE FAILED'} - "
          f"{'safe to train the calibrated arm' if ok else 'do NOT train'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
