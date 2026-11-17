#!/usr/bin/env python3
"""Free-space vs contact-window error decomposition.

Tests the precision hypothesis directly. If calibration degrades precision
rather than competence, trajectories should stay close in free space and
diverge sharply in the contact window, where a grasp must be positioned to
within millimetres. If instead error is uniform across phases, the hypothesis
is wrong and the ceiling needs another explanation.

Contact windows are defined around gripper-closure events, which is where CALVIN
manipulation is precision-critical; everything else is treated as free space.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# robot_obs layout: [0:3] TCP xyz, [3:6] TCP euler, [6] gripper width,
# [7:14] joint angles, [14] gripper action
TCP_POS, TCP_ORN, GRIP_W = slice(0, 3), slice(3, 6), 6
CONTACT_HALF_WINDOW = 10  # steps either side of a closure event


def load(path: Path) -> dict[int, list[dict]]:
    episodes: dict[int, list[dict]] = {}
    for row in json.loads(Path(path).read_text()):
        episodes.setdefault(row["ep"], []).append(row)
    return episodes


def closure_steps(ep: list[dict]) -> list[int]:
    """Steps where the commanded gripper transitions toward closing."""
    grip = [r["final"][6] for r in ep]
    return [i for i in range(1, len(grip)) if grip[i] < 0 <= grip[i - 1]]


def contact_mask(ep: list[dict]) -> np.ndarray:
    mask = np.zeros(len(ep), dtype=bool)
    for c in closure_steps(ep):
        mask[max(0, c - CONTACT_HALF_WINDOW):c + CONTACT_HALF_WINDOW + 1] = True
    return mask


def pose(ep: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    robot = np.array([r["robot_obs"] for r in ep], dtype=np.float64)
    return robot[:, TCP_POS], robot[:, TCP_ORN]


def wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def compare(ref: dict[int, list[dict]], other: dict[int, list[dict]], label: str) -> None:
    free_pos, cont_pos, free_orn, cont_orn = [], [], [], []
    free_act, cont_act, clips, first_div = [], [], [], []
    skipped = 0
    for ep_id in sorted(set(ref) & set(other)):
        a, b = ref[ep_id], other[ep_id]
        n = min(len(a), len(b))
        if n < 5:
            continue
        a, b = a[:n], b[:n]
        # A subtask after the first begins wherever the previous one ended, so
        # two conditions only share a starting state when both reached it.
        # Comparing trajectories from different starts measures nothing.
        if not np.allclose(np.array(a[0]["robot_obs"][:6]),
                           np.array(b[0]["robot_obs"][:6]), atol=1e-5):
            skipped += 1
            continue
        pa, oa = pose(a)
        pb, ob = pose(b)
        dp = np.linalg.norm(pa - pb, axis=1) * 1000.0            # millimetres
        do = np.linalg.norm(wrap(oa - ob), axis=1) * 180 / np.pi  # degrees
        da = np.linalg.norm(np.array([r["final"][:6] for r in a])
                            - np.array([r["final"][:6] for r in b]), axis=1)
        mask = contact_mask(a)
        free_pos += dp[~mask].tolist(); cont_pos += dp[mask].tolist()
        free_orn += do[~mask].tolist(); cont_orn += do[mask].tolist()
        free_act += da[~mask].tolist(); cont_act += da[mask].tolist()
        clips.append(np.mean([np.any(np.abs(np.array(r["final"][:6])) >= 0.999) for r in b]))
        diverged = np.where(dp > 1.0)[0]                          # first >1 mm
        first_div.append(int(diverged[0]) if len(diverged) else n)

    def stat(v):
        return (f"{np.median(v):7.3f} {np.percentile(v, 90):9.3f}"
                if v else "      -         -")

    print(f"\n--- {label} vs clean ---")
    print(f"aligned trajectories: {len(first_div)}  "
          f"(skipped {skipped} with mismatched start state)")
    print(f"{'quantity':24s} {'free median':>12s} {'free p90':>9s} "
          f"{'contact median':>15s} {'contact p90':>12s}   ratio")
    for name, fv, cv in (("TCP position (mm)", free_pos, cont_pos),
                         ("TCP orientation (deg)", free_orn, cont_orn),
                         ("commanded action", free_act, cont_act)):
        if not fv or not cv:
            continue
        fm, cm = np.median(fv), np.median(cv)
        ratio = cm / fm if fm > 1e-9 else float("inf")
        print(f"{name:24s} {stat(fv)} {stat(cv)}   {ratio:5.2f}x")
    if clips:
        print(f"{'steps with a clip':24s} {100*np.mean(clips):.2f}%")
    if first_div:
        print(f"{'first >1mm divergence':24s} step {np.median(first_div):.0f} (median)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean", required=True)
    p.add_argument("--compare", nargs="+", required=True,
                   help="label=path pairs, e.g. calibrated=trace_cal.json")
    args = p.parse_args()

    ref = load(Path(args.clean))
    print(f"clean trace: {len(ref)} episodes, {sum(len(v) for v in ref.values())} steps")
    cm = contact_mask(next(iter(ref.values())))
    print(f"contact window = +/-{CONTACT_HALF_WINDOW} steps around gripper closure "
          f"({100*cm.mean():.0f}% of the first episode)")
    for spec in args.compare:
        label, _, path = spec.partition("=")
        compare(ref, load(Path(path)), label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
