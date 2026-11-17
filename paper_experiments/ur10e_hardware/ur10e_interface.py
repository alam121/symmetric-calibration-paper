"""The UR10e side of the relative Cartesian action spec (isaac/sim2real/spec.py), streamed.

One command per 1/rate: a_t (6-D, [-1, 1]) -> tool0 target = (p + a_xyz * unit_t, exp(a_rot * unit_r) R)
-> IK on the robot's own /robot_description -> one joint trajectory point reached in 1/rate ->
the next tick. Nothing waits for the arm: lag, speed scaling and tracking are part of the
interface, as in the sim (Isaac Lab relative differential IK at the same rate). Calibration
(ur10e_calibration_run.py --stream_hz) and deployment (sim2real_deploy.py) both go through this.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation

from ur10e_calibration_run import ARM, Chain, Runner  # noqa: F401  (same node, IK and safety checks)


class StreamingInterface:
    def __init__(self, node: Runner, chain: Chain, rate_hz: float, unit_translation: float, unit_rotation: float,
                 max_joint_step: float = 0.15, frame_yaw_deg: float = 0.0):
        self.node, self.chain = node, chain
        # Command frame = base_link turned frame_yaw_deg about z (180 = UR "base"); the pose stays in base_link.
        self.frame = Rotation.from_euler("z", np.radians(frame_yaw_deg)).as_matrix()
        self.period = 1.0 / rate_hz
        self.unit_t, self.unit_r = unit_translation, unit_rotation
        self.max_joint_step = max_joint_step
        self.next_tick = None

    def pose(self):
        return self.node.tool_pose()

    def command(self, action, dry_run=False):
        """Send one command and return at the next tick. Returns (sent, reason)."""
        action = np.clip(np.asarray(action, float), -1.0, 1.0)
        p0, r0 = self.pose()
        q0 = self.node.joints.copy()
        p1 = p0 + self.frame @ action[:3] * self.unit_t
        r1 = Rotation.from_rotvec(self.frame @ action[3:] * self.unit_r).as_matrix() @ r0
        q1, pos_res, rot_res = self.chain.ik(q0, p1, r1)
        sent, reason = True, ""
        if pos_res > 1e-3 or rot_res > 1e-2 or np.abs(q1 - q0).max() > self.max_joint_step:
            sent, reason = False, f"IK residual {pos_res * 1000:.2f} mm / joint step {np.abs(q1 - q0).max():.3f}"
        elif not dry_run:
            saved = self.node.args.step_time
            self.node.args.step_time = self.period
            self.node.send(q1)
            self.node.args.step_time = saved
        if self.next_tick is None:
            self.next_tick = time.time()
        self.next_tick += self.period
        while time.time() < self.next_tick:
            rclpy.spin_once(self.node, timeout_sec=max(0.0, min(0.005, self.next_tick - time.time())))
        return sent, reason
