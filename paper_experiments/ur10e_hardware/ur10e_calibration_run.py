#!/usr/bin/env python3
"""Collect one task-free calibration run on the UR10e (hardware validation, steps 2-9).

Each step draws the next action of the paper's smooth excitation (SmoothExcitation, beta =
action smoothing, seeded), turns it into a small relative TCP move, solves IK on the robot's
own /robot_description, sends it as one short joint trajectory and waits for the arm to settle.
The TCP pose (TF base_frame -> tool_frame) is logged before and after, so commanded vs realised
motion can be fitted with calibrator.fit_local_action_model.

Action convention (fixed for every run; only the calibration condition changes):
  a_t = [dx, dy, dz, drx, dry, drz] in [-1, 1] (clip 0.65, as the paper), base-frame translation
  of `unit_translation` m and a world-frame rotation vector of `unit_rotation_deg` per unit:
  R_target = exp(a_rot * unit) R_now.

Safety: per-step translation / rotation hard caps, a workspace box around the start pose (the
excitation reflects off it; a target outside is never sent), a per-step joint-change cap,
IK residual check, FK-vs-TF check at start, and --dry_run (nothing is sent; "after" is the
IK solution's FK). Ctrl+C holds the arm where it is.

  source /opt/ros/humble/setup.bash
  python3 ur10e_calibration_run.py --run_id beta085_run1 --beta 0.85 --seed 1 --steps 600 --out runs/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_experiments.task_free_calibration.collect_calibration import SmoothExcitation  # noqa: E402

ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
TRAJECTORY_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_id", required=True)
    p.add_argument("--beta", type=float, required=True, help="Excitation smoothing (paper: 0.85).")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--condition", default="", help="Free-text calibration condition, logged with every row.")
    p.add_argument("--translation_std", type=float, default=0.28, help="Excitation noise std (paper default).")
    p.add_argument("--rotation_std", type=float, default=0.18)
    p.add_argument("--match_magnitude", type=float, default=0.15,
                   help="Rescale the noise so the stationary action std is this for every beta (protocol step 9); "
                   "0 = the paper's raw noise, whose stationary std falls with beta as sqrt((1-b)/(1+b)).")
    p.add_argument("--unit_translation", type=float, default=0.02, help="m per action unit.")
    p.add_argument("--unit_rotation_deg", type=float, default=5.0, help="deg per action unit.")
    p.add_argument("--max_translation", type=float, default=0.02, help="Hard cap per step [m].")
    p.add_argument("--max_rotation_deg", type=float, default=5.0, help="Hard cap per step [deg].")
    p.add_argument("--box", type=float, nargs=3, default=[0.10, 0.10, 0.08],
                   help="Workspace half-extents around the start TCP position [m].")
    p.add_argument("--homing", action="store_true",
                   help="At the workspace / tilt edge, return to the start joints (3 s) and drop the transition instead "
                   "of reflecting the excitation per axis. Reflection assumes command axes = task axes, which is what "
                   "calibration is supposed to find out (xembody.py uses homing for both robots).")
    p.add_argument("--max_tilt_deg", type=float, default=15.0,
                   help="Orientation bound: the TCP stays within this rotation of its start orientation (the "
                   "excitation reflects off it, as off the position box).")
    p.add_argument("--max_joint_step", type=float, default=0.15, help="Per-step joint change cap [rad].")
    p.add_argument("--step_time", type=float, default=0.8, help="Trajectory duration per step [s].")
    p.add_argument("--settle_timeout", type=float, default=3.0)
    p.add_argument("--stream_hz", type=float, default=0.0,
                   help="Stream one command per 1/stream_hz without waiting to settle (the sim-to-real spec: 10 Hz, "
                   "--unit_translation 0.01 --unit_rotation_deg 2.8648). 0 = move-and-settle per step.")
    p.add_argument("--cmd_frame_yaw_deg", type=float, default=0.0,
                   help="Streamed commands are interpreted in base_link turned this much about z (180 = UR base); "
                   "the logged state stays in base_link. Requires --stream_hz.")
    p.add_argument("--base_frame", default="base_link")
    p.add_argument("--tool_frame", default="tool0")
    p.add_argument("--start_joints", type=float, nargs=6, default=None,
                   help="Move here first (slowly, one trajectory) so every run starts from the same pose.")
    p.add_argument("--start_move_time", type=float, default=8.0)
    p.add_argument("--max_condition", type=float, default=200.0,
                   help="Refuse to start if the Jacobian condition number exceeds this (near a singularity).")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def rpy_matrix(r, p, y):
    return Rotation.from_euler("xyz", [r, p, y]).as_matrix()


class Chain:
    """URDF chain base_frame -> tool_frame: FK and damped least-squares IK for the 6 arm joints."""

    def __init__(self, urdf_xml: str, base: str, tip: str):
        joints = {j.find("child").get("link"): j for j in ET.fromstring(urdf_xml).findall("joint")}
        chain, link = [], tip
        while link != base:
            if link not in joints:
                raise ValueError(f"{tip} is not below {base} in the URDF")
            chain.append(joints[link])
            link = joints[link].find("parent").get("link")
        self.chain = []
        for joint in reversed(chain):
            origin = joint.find("origin")
            xyz = np.array([float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()])
            rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
            axis = joint.find("axis")
            self.chain.append((joint.get("name"), joint.get("type"), xyz, rpy_matrix(*rpy),
                               np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else None))

    def fk(self, q: np.ndarray):
        values = dict(zip(ARM, q))
        rotation, position = np.eye(3), np.zeros(3)
        for name, kind, xyz, fixed, axis in self.chain:
            position = position + rotation @ xyz
            rotation = rotation @ fixed
            if kind == "revolute":
                rotation = rotation @ Rotation.from_rotvec(axis * values[name]).as_matrix()
        return position, rotation

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        p, r = self.fk(q)
        jac = np.zeros((6, 6))
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = 1e-6
            pd, rd = self.fk(q + dq)
            jac[:3, i] = (pd - p) / 1e-6
            jac[3:, i] = Rotation.from_matrix(rd @ r.T).as_rotvec() / 1e-6
        return jac

    def ik(self, q0: np.ndarray, position, rotation, iterations=300):
        q = q0.copy()
        for _ in range(iterations):
            p, r = self.fk(q)
            error = np.concatenate([position - p, Rotation.from_matrix(rotation @ r.T).as_rotvec()])
            if np.linalg.norm(error[:3]) < 1e-5 and np.linalg.norm(error[3:]) < 1e-4:
                break
            jacobian = np.zeros((6, 6))
            for i in range(6):
                dq = np.zeros(6)
                dq[i] = 1e-6
                pd, rd = self.fk(q + dq)
                jacobian[:3, i] = (pd - p) / 1e-6
                jacobian[3:, i] = Rotation.from_matrix(rd @ r.T).as_rotvec() / 1e-6
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-6 * np.eye(6), error)
            # Small iterations stay on the current configuration branch; unclamped, the solver
            # jumped joints by pi or 2 pi to equivalent solutions far away.
            q = q + step * min(1.0, 0.05 / max(np.abs(step).max(), 1e-12))
        p, r = self.fk(q)
        return q, float(np.linalg.norm(position - p)), float(np.linalg.norm(Rotation.from_matrix(rotation @ r.T).as_rotvec()))


class Runner(Node):
    def __init__(self, args):
        super().__init__("ur10e_calibration_run")
        self.args = args
        self.joints = None
        self.urdf = None
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.create_subscription(JointState, "/joint_states", self._joints_cb, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, "/robot_description", lambda m: setattr(self, "urdf", m.data), latched)
        self.pub = self.create_publisher(JointTrajectory, TRAJECTORY_TOPIC, 10)

    def _joints_cb(self, msg):
        positions = dict(zip(msg.name, msg.position))
        if all(n in positions for n in ARM):
            self.joints = np.array([positions[n] for n in ARM])

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.01)

    def tool_pose(self):
        try:
            t = self.tf.lookup_transform(self.args.base_frame, self.args.tool_frame, Time()).transform
        except TransformException:
            return None
        return (np.array([t.translation.x, t.translation.y, t.translation.z]),
                Rotation.from_quat([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]).as_matrix())

    def send(self, q):
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in q]
        point.velocities = [0.0] * 6
        point.time_from_start.sec = int(self.args.step_time)
        point.time_from_start.nanosec = int((self.args.step_time % 1) * 1e9)
        msg = JointTrajectory()
        msg.joint_names = ARM
        msg.points = [point]
        self.pub.publish(msg)

    def settle(self):
        """Wait the trajectory time, then until the joints stop moving."""
        self.spin_for(self.args.step_time)
        start, last, still = time.time(), self.joints.copy(), 0.0
        while time.time() - start < self.args.settle_timeout:
            self.spin_for(0.05)
            if np.abs(self.joints - last).max() < 2e-4:
                still += 0.05
                if still >= 0.2:
                    return True
            else:
                still = 0.0
            last = self.joints.copy()
        return False


def state_row(position, rotation):
    """Calibrator state: position, extrinsic xyz Euler, gripper (constant: not actuated)."""
    return position.tolist() + Rotation.from_matrix(rotation).as_euler("xyz").tolist() + [1.0]


def main():
    args = parse_args()
    rclpy.init()
    node = Runner(args)
    for _ in range(200):
        node.spin_for(0.05)
        if node.joints is not None and node.urdf is not None and node.tool_pose() is not None:
            break
    else:
        sys.exit("no /joint_states, /robot_description or TF yet")
    chain = Chain(node.urdf, args.base_frame, args.tool_frame)
    if args.start_joints is not None:
        target = np.array(args.start_joints)
        print(f"[CALIB] moving to the start pose {np.round(target, 3).tolist()} over {args.start_move_time:.0f}s",
              flush=True)
        if not args.dry_run:
            saved, args.step_time = args.step_time, args.start_move_time
            node.send(target)
            node.settle()
            args.step_time = saved
            if np.abs(node.joints - target).max() > 0.01:
                sys.exit(f"did not reach the start pose (off by {np.abs(node.joints - target).max():.3f} rad)")
    condition = float(np.linalg.cond(chain.jacobian(node.joints)))
    if condition > args.max_condition:
        sys.exit(f"start pose is near a singularity (Jacobian condition {condition:.0f} > {args.max_condition:.0f}); "
                 "choose another --start_joints")
    p_fk, r_fk = chain.fk(node.joints)
    p_tf, r_tf = node.tool_pose()
    fk_error = np.linalg.norm(p_fk - p_tf)
    if fk_error > 0.002:
        sys.exit(f"FK disagrees with TF by {fk_error * 1000:.1f} mm: wrong URDF or frames")
    start_position, start_rotation = p_tf.copy(), r_tf.copy()
    box = np.array(args.box)
    rng = np.random.default_rng(args.seed)
    stationary = np.sqrt((1 - args.beta) / (1 + args.beta))
    scale = args.match_magnitude / (stationary * args.translation_std) if args.match_magnitude > 0 else 1.0
    excitation = SmoothExcitation(rng, args.beta, args.translation_std * scale, args.rotation_std * scale,
                                  gripper_period=10 ** 9, workspace_low=None, workspace_high=None)
    unit_rot = np.radians(args.unit_rotation_deg)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.run_id}.jsonl"
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    meta.update(start_joints_measured=node.joints.tolist(), jacobian_condition=condition,
                start_position=start_position.tolist(), fk_tf_error_m=float(fk_error), noise_scale=float(scale),
                action_units={"translation_m": args.unit_translation, "rotation_deg": args.unit_rotation_deg},
                rotation_command="world: R_target = exp(a_rot * unit) R_now")
    (args.out / f"{args.run_id}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[CALIB] {args.run_id}: beta {args.beta}, seed {args.seed}, {args.steps} steps, noise x{scale:.2f}, "
          f"{'DRY RUN' if args.dry_run else 'LIVE'} -> {path}", flush=True)

    stream = None
    if args.stream_hz > 0:
        from ur10e_interface import StreamingInterface
        stream = StreamingInterface(node, chain, args.stream_hz, args.unit_translation, unit_rot, args.max_joint_step,
                                    frame_yaw_deg=args.cmd_frame_yaw_deg)
    elif args.cmd_frame_yaw_deg:
        sys.exit("--cmd_frame_yaw_deg needs --stream_hz")
    cmd_frame = Rotation.from_euler("z", np.radians(args.cmd_frame_yaw_deg)).as_matrix()
    sent = skipped = 0
    with path.open("w") as log:
        try:
            for step in range(args.steps):
                q0 = node.joints.copy()
                before = node.tool_pose()
                if before is None:
                    sys.exit("TF lost")
                p0, r0 = before
                if args.homing:
                    tilt_now = np.degrees(np.linalg.norm(Rotation.from_matrix(r0 @ start_rotation.T).as_rotvec()))
                    if np.any(np.abs(p0 - start_position) > box - 0.01) or tilt_now > args.max_tilt_deg - 2.0:
                        homes = getattr(main, "homes", 0) + 1
                        main.homes = homes
                        if not args.dry_run:
                            saved = node.args.step_time
                            node.args.step_time = 3.0
                            node.send(np.array(args.start_joints))
                            node.settle()
                            node.args.step_time = saved
                        excitation.motion[:] = 0.0
                        if stream is not None:
                            stream.next_tick = None
                        log.write(json.dumps({"run_id": args.run_id, "step": step, "homing": True}) + "\n")
                        continue
                # Workspace reflection, as SmoothExcitation does for its own bounds.
                for axis in range(3 if not args.homing else 0):
                    if p0[axis] < start_position[axis] - box[axis] + 0.01:
                        excitation.motion[axis] = abs(excitation.motion[axis])
                    elif p0[axis] > start_position[axis] + box[axis] - 0.01:
                        excitation.motion[axis] = -abs(excitation.motion[axis])
                # Orientation bound: a rotation command that would carry the tool further from its start
                # orientation than max_tilt_deg is reflected (world-frame rotation-vector components).
                tilt = Rotation.from_matrix(r0 @ start_rotation.T).as_rotvec()
                if not args.homing and np.degrees(np.linalg.norm(tilt)) > args.max_tilt_deg - 2.0:
                    for axis in range(3):
                        if excitation.motion[3 + axis] * tilt[axis] > 0:
                            excitation.motion[3 + axis] = -excitation.motion[3 + axis]
                action = excitation(np.zeros(7), step)
                action[6] = 1.0
                dp = action[:3] * args.unit_translation
                dr = action[3:6] * unit_rot
                dp *= min(1.0, args.max_translation / max(np.linalg.norm(dp), 1e-12))
                dr *= min(1.0, np.radians(args.max_rotation_deg) / max(np.linalg.norm(dr), 1e-12))
                dp, dr = cmd_frame @ dp, cmd_frame @ dr   # base_link displacement the command asks for
                p1, r1 = p0 + dp, Rotation.from_rotvec(dr).as_matrix() @ r0
                row = {"run_id": args.run_id, "condition": args.condition, "beta": args.beta, "step": step,
                       "t": time.time(), "action": action.tolist(), "commanded_dp": dp.tolist(),
                       "commanded_drotvec": dr.tolist(), "joints_before": q0.tolist(),
                       "state": state_row(p0, r0), "pose_before": {"p": p0.tolist(),
                                                                    "q_xyzw": Rotation.from_matrix(r0).as_quat().tolist()}}
                if np.degrees(np.linalg.norm(Rotation.from_matrix(r1 @ start_rotation.T).as_rotvec())) > args.max_tilt_deg:
                    row["skipped"] = "target outside the orientation bound"
                    skipped += 1
                    log.write(json.dumps(row) + "\n")
                    continue
                if np.any(np.abs(p1 - start_position) > box):
                    row["skipped"] = "target outside the workspace box"
                    skipped += 1
                    log.write(json.dumps(row) + "\n")
                    continue
                if stream is not None:
                    # The spec's streamed interface: the command's own clipping and IK, no settling; the
                    # realised motion is read at the next tick (as tasks_sim.py --calibrate does).
                    ok, reason = stream.command(action[:6], dry_run=args.dry_run)
                    if not ok:
                        row["skipped"] = reason
                        skipped += 1
                        log.write(json.dumps(row) + "\n")
                        continue
                    after = node.tool_pose()
                    if after is None:
                        sys.exit("TF lost")
                    p2, r2 = after
                    sent += 1
                    row.update(joints_after=node.joints.tolist(), settled=False, next_state=state_row(p2, r2),
                               pose_after={"p": p2.tolist(), "q_xyzw": Rotation.from_matrix(r2).as_quat().tolist()},
                               realised_dp=(p2 - p0).tolist(),
                               realised_drotvec_world=Rotation.from_matrix(r2 @ r0.T).as_rotvec().tolist(),
                               realised_drotvec_body=Rotation.from_matrix(r0.T @ r2).as_rotvec().tolist())
                    log.write(json.dumps(row) + "\n")
                    continue
                q1, pos_res, rot_res = chain.ik(q0, p1, r1)
                if pos_res > 1e-3 or rot_res > 1e-2 or np.abs(q1 - q0).max() > args.max_joint_step:
                    row["skipped"] = f"IK residual {pos_res * 1000:.2f} mm / joint step {np.abs(q1 - q0).max():.3f}"
                    skipped += 1
                    log.write(json.dumps(row) + "\n")
                    continue
                if args.dry_run:
                    p2, r2 = chain.fk(q1)
                    q2, settled = q1, True
                else:
                    node.send(q1)
                    settled = node.settle()
                    q2 = node.joints.copy()
                    after = node.tool_pose()
                    if after is None:
                        sys.exit("TF lost")
                    p2, r2 = after
                sent += 1
                row.update(joints_target=q1.tolist(), joints_after=q2.tolist(), settled=settled,
                           next_state=state_row(p2, r2),
                           pose_after={"p": p2.tolist(), "q_xyzw": Rotation.from_matrix(r2).as_quat().tolist()},
                           realised_dp=(p2 - p0).tolist(),
                           realised_drotvec_world=Rotation.from_matrix(r2 @ r0.T).as_rotvec().tolist(),
                           realised_drotvec_body=Rotation.from_matrix(r0.T @ r2).as_rotvec().tolist())
                log.write(json.dumps(row) + "\n")
                if step % 50 == 0:
                    log.flush()
                    print(f"[CALIB] step {step}: sent {sent}, skipped {skipped}, |dp| {np.linalg.norm(p2 - p0) * 1000:.1f} mm",
                          flush=True)
        except KeyboardInterrupt:
            if not args.dry_run and node.joints is not None:
                node.send(node.joints)
            print("[CALIB] interrupted: holding", flush=True)
    print(f"[CALIB] {args.run_id}: {sent} moves, {skipped} skipped -> {path}", flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
