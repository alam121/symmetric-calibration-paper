"""One slow joint move of the real UR10e to the xembody trial start (wrist 3 the short way). Approved 2026-10-28."""
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
EXPECTED_NOW = np.array([-1.364, -0.905, 1.853, 3.606, -1.747, 1.450])
TARGET = np.array([-1.397, -0.745, 1.486, 4.001, -1.593, 3.301])
DURATION = 10


class Mover(Node):
    def __init__(self):
        super().__init__("xembody_move_to_start")
        self.q = None
        self.create_subscription(JointState, "/joint_states", self.cb, 10)
        self.pub = self.create_publisher(JointTrajectory, "/scaled_joint_trajectory_controller/joint_trajectory", 10)

    def cb(self, m):
        d = dict(zip(m.name, m.position))
        if all(n in d for n in ARM):
            self.q = np.array([d[n] for n in ARM])

    def spin_for(self, t):
        end = time.time() + t
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def send(self):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in TARGET]
        p.velocities = [0.0] * 6
        p.time_from_start.sec = DURATION
        m = JointTrajectory()
        m.joint_names = ARM
        m.points = [p]
        self.pub.publish(m)


rclpy.init()
n = Mover()
n.spin_for(3.0)
if n.q is None:
    sys.exit("no /joint_states")
dev = np.abs(n.q - EXPECTED_NOW).max()
print("now   ", np.round(n.q, 3).tolist(), f"(max dev from measured {dev:.3f} rad)", flush=True)
if dev > 0.02:
    sys.exit("arm is not where it was measured; not moving")
for _ in range(50):
    if n.pub.get_subscription_count() > 0:
        break
    n.spin_for(0.1)
else:
    sys.exit("no subscriber on the trajectory topic; not moving")
q0 = n.q.copy()
n.send()
print("sent: target", TARGET.tolist(), f"over {DURATION} s", flush=True)
n.spin_for(2.0)
if np.abs(n.q - q0).max() < 1e-3:
    print("no motion after 2 s; resending once", flush=True)
    n.send()
t0, last, still = time.time(), n.q.copy(), 0.0
while time.time() - t0 < DURATION * 3:
    n.spin_for(0.1)
    if np.abs(n.q - last).max() < 2e-4:
        still += 0.1
        if still >= 1.0 and np.abs(n.q - TARGET).max() < 0.01:
            break
    else:
        still = 0.0
    last = n.q.copy()
print("final ", np.round(n.q, 3).tolist(), f"(max error {np.abs(n.q - TARGET).max():.4f} rad)", flush=True)
n.destroy_node()
rclpy.shutdown()
