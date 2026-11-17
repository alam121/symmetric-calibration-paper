#!/usr/bin/env python3
"""Run the Franka-trained xembody policy (manipulatorsdatepalm/isaac/sim2real/xembody.py) on the real UR10e.

The policy was trained only on a simulated Franka and sees task-frame quantities only. Here the task frame T is
the measured lab cell (origin at base_link (0.52, -1.02, -0.329), axes = base_link). Each 10 Hz tick: the real
tool0 pose -> TCP (0.245 m along tool z, the Delto F1/F3 pinch centre) in T -> the same observation as in sim ->
the policy (numpy) -> one action translated by one of

  direct      a_U = a
  matched     a_U = (a W_F + b_F - b_U) pinv(W_U), both fitted with the world convention
  mismatched  W_F world, W_U body
  analytic    the documented interface: TCP displacement moved to tool0 (pivot compensation), base_link frame
  analytic_exact  analytic, then rotated into the true command frame (--cmd_frame_yaw_deg known): the oracle

-> ur10e_interface.StreamingInterface (IK, one trajectory point per tick). Task R only (contactless): reach the
target point with the target yaw and hold it within 1 cm / 5 deg for 1 s; 20 s limit. Aborts on tilt > 25 deg
or the TCP leaving a box around the work area.

  python3 xembody_real.py --policy policy_R.npz --mode matched --franka_calib F.jsonl --real_calib U_T.jsonl \
      --target 0.07 -0.01 0.10 30 --trial t01 --out runs/ [--dry_run]
  python3 xembody_real.py --convert RAW.jsonl OUT.jsonl     # real calibration log (tool0 in base_link) -> TCP in T
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

T_ORIGIN = np.array([0.52, -1.02, -0.329])
TCP_OFFSET = 0.245
UNIT_T, UNIT_R = 0.01, 0.05
START_JOINTS = [-1.397, -0.745, 1.486, 4.001, -1.593, -2.982]   # sim-solved: TCP at T(-0.17, 0.07, 0.175), tool down
PARKED_OBJECT = np.array([0.30, 0.30, 0.031])                       # task R parks the box here in sim


def tcp_in_T(p, r):
    return p + r @ np.array([0.0, 0.0, TCP_OFFSET]) - T_ORIGIN


def convert(raw: Path, out: Path):
    """Real calibration rows (tool0 pose in base_link, extrinsic xyz Euler) -> TCP in the task frame."""
    n = 0
    with out.open("w") as f:
        for line in raw.read_text().splitlines():
            row = json.loads(line)
            if "next_state" not in row:
                continue
            for key in ("state", "next_state"):
                s = row[key]
                r = Rotation.from_euler("xyz", s[3:6]).as_matrix()
                row[key] = tcp_in_T(np.array(s[:3]), r).tolist() + list(s[3:6]) + [s[6] if len(s) > 6 else 1.0]
            f.write(json.dumps(row) + "\n")
            n += 1
    meta = raw.with_suffix("").with_suffix(".meta.json")
    if meta.exists():
        out.with_suffix("").with_suffix(".meta.json").write_text(meta.read_text())
    print(f"{raw} -> {out}: {n} transitions, TCP in the task frame")


class Policy:
    def __init__(self, path):
        d = np.load(path)
        self.mu, self.sd, self.rep = d["mu"], d["sd"], str(d["rep"])
        self.layers = [(d[f"body_{i}_weight"], d[f"body_{i}_bias"]) for i in (0, 2, 4, 6)]

    def prep(self, o):
        oyaw = np.arctan2(o[13], o[12])
        if self.rep == "tcp_full":
            return np.concatenate([o[0:12], [np.sin(oyaw), np.cos(oyaw)], o[18:25]])
        if self.rep == "relative":
            return np.concatenate([o[9:12] - o[0:3], o[18:21] - o[9:12], o[3:9], [np.sin(oyaw), np.cos(oyaw)], o[21:25]])
        raise SystemExit(f"representation {self.rep} not supported on the robot")

    def __call__(self, obs):
        x = (self.prep(obs) - self.mu) / self.sd
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < 3:
                x = np.maximum(x, 0.0)
        return np.tanh(x[:6]), bool(x[6] > 0)


def observe(tcp, r, target):
    rot6 = lambda m: np.concatenate([m[:, 0], m[:, 1]])
    return np.concatenate([tcp, rot6(r), PARKED_OBJECT, rot6(np.eye(3)), target[:3],
                           [np.sin(target[3]), np.cos(target[3])], [0.0, 3.0]]).astype(np.float32)


def make_map(args):
    if args.mode == "direct":
        return lambda a, r: a
    if args.mode in ("analytic", "analytic_exact"):
        undo = Rotation.from_euler("z", np.radians(args.cmd_frame_yaw_deg if args.mode == "analytic_exact" else 0.0)).as_matrix().T

        def analytic(a, r):
            dp, dr = a[:3] * UNIT_T, a[3:] * UNIT_R
            dp_tool0 = dp - np.cross(dr, r @ np.array([0.0, 0.0, TCP_OFFSET]))
            return np.concatenate([undo @ dp_tool0 / UNIT_T, undo @ dr / UNIT_R])
        return analytic
    from fit_compare import fit, load
    pf, pu = ("world", "world") if args.mode == "matched" else ("world", "body")
    F = fit(*load(args.franka_calib)[:3], pf)
    U = fit(*load(args.real_calib)[:3], pu)
    WF, bF, WU, bU = (np.asarray(v) for v in (F.matrix, F.bias, U.matrix, U.bias))
    pinv = np.linalg.pinv(WU)
    print(f"[XREAL] {args.mode}: M diag {np.round(np.diag(WF @ pinv), 2).tolist()}", flush=True)
    return lambda a, r: (a @ WF + bF - bU) @ pinv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--convert", nargs=2, type=Path, default=None)
    p.add_argument("--policy", type=Path)
    p.add_argument("--mode", choices=("direct", "matched", "mismatched", "analytic", "analytic_exact"))
    p.add_argument("--franka_calib", type=Path)
    p.add_argument("--real_calib", type=Path)
    p.add_argument("--target", type=float, nargs=4, help="x y z [m, task frame] yaw [deg]")
    p.add_argument("--start_joints", type=float, nargs=6, default=START_JOINTS)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--max_tilt_deg", type=float, default=25.0)
    p.add_argument("--box", type=float, nargs=6, default=[-0.30, 0.25, -0.20, 0.25, 0.05, 0.30],
                   help="TCP limits in T: x_min x_max y_min y_max z_min z_max [m]")
    p.add_argument("--trial", default="trial")
    p.add_argument("--out", type=Path, default=Path("xembody_runs"))
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--cmd_frame_yaw_deg", type=float, default=0.0,
                   help="The robot interprets commands in base_link turned this much about z (180 = UR base).")
    p.add_argument("--base_frame", default="base_link")
    p.add_argument("--tool_frame", default="tool0")
    p.add_argument("--step_time", type=float, default=0.1)
    p.add_argument("--settle_timeout", type=float, default=3.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.convert:
        return convert(*args.convert)
    import rclpy
    from ur10e_calibration_run import Chain, Runner
    from ur10e_interface import StreamingInterface
    policy, translate = Policy(args.policy), make_map(args)
    target = np.array(args.target[:3] + [np.radians(args.target[3])])
    rclpy.init()
    node = Runner(args)
    for _ in range(200):
        node.spin_for(0.05)
        if node.joints is not None and node.urdf is not None and node.tool_pose() is not None:
            break
    else:
        sys.exit("no /joint_states, /robot_description or TF")
    chain = Chain(node.urdf, args.base_frame, args.tool_frame)
    interface = StreamingInterface(node, chain, 10.0, UNIT_T, UNIT_R, frame_yaw_deg=args.cmd_frame_yaw_deg)
    if not args.dry_run:   # to the start pose; resend: the first message on a new publisher can be lost
        node.args.step_time = 6.0
        for _ in range(3):
            node.send(np.array(args.start_joints)); node.settle()
            if np.abs(node.joints - np.array(args.start_joints)).max() < 0.01:
                break
        else:
            sys.exit("could not reach the start pose")
    interface.next_tick = None
    box = np.array(args.box).reshape(3, 2)
    rows, held, outcome = [], 0, "timeout"
    try:
        for step in range(int(args.seconds * 10)):
            p, r = interface.pose()
            tcp = tcp_in_T(p, r)
            tilt = np.degrees(np.arccos(np.clip(-r[2, 2], -1, 1)))
            if tilt > args.max_tilt_deg:
                outcome = f"abort: tool tilted {tilt:.0f} deg"; break
            if np.any(tcp < box[:, 0]) or np.any(tcp > box[:, 1]):
                outcome = f"abort: TCP left the box at {np.round(tcp, 3).tolist()}"; break
            pos_err = float(np.linalg.norm(tcp - target[:3]))
            yaw = float(np.arctan2(r[1, 0], r[0, 0]))
            yaw_err = float(np.degrees(abs((yaw - target[3] + np.pi) % (2 * np.pi) - np.pi)))
            held = held + 1 if pos_err < 0.01 and yaw_err < 5.0 else 0
            if held >= 10:
                outcome = "success"; break
            a, _ = policy(observe(tcp, r, target))
            a_u = np.clip(translate(np.clip(a, -1, 1), r), -1, 1)
            sent, reason = interface.command(a_u, dry_run=args.dry_run)
            rows.append({"step": step, "tcp": tcp.tolist(), "yaw_deg": float(np.degrees(yaw)), "pos_err_mm": pos_err * 1000,
                         "yaw_err_deg": yaw_err, "tilt_deg": float(tilt), "a": a.tolist(), "a_u": a_u.tolist(),
                         "sent": sent, "reason": reason})
    except KeyboardInterrupt:
        outcome = "stopped by operator"
    if not args.dry_run:
        node.send(node.joints)
    args.out.mkdir(parents=True, exist_ok=True)
    last = rows[-1] if rows else {}
    path = args.out / f"{args.trial}_R_{args.mode}.json"
    path.write_text(json.dumps({"mode": args.mode, "cmd_frame_yaw_deg": args.cmd_frame_yaw_deg, "target": args.target, "outcome": outcome, "steps": rows}, indent=1))
    print(f"[XREAL] {args.trial} {args.mode}: {outcome}; final {last.get('pos_err_mm', float('nan')):.1f} mm, "
          f"{last.get('yaw_err_deg', float('nan')):.1f} deg -> {path}", flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
