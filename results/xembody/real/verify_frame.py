"""Low-amplitude check of the rotated command frame on the real UR10e, before any calibration in it.

From the trial start pose, send 5 ticks of 0.3-unit commands along each of +x, +y, +z (in the command frame) and
undo each with 5 ticks the other way; then +rz / -rz. Compare the measured base_link motion with what the frame
predicts (translation: R_z(theta) e; yaw sign). Stops if the TCP moves more than 3 cm or tilts more than 5 deg.

  /usr/bin/python3 verify_frame.py THETA_DEG [--dry_run]
"""
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HW = Path(__file__).resolve().parents[3] / "paper_experiments/ur10e_hardware"
sys.path.insert(0, str(HW))
sys.path.insert(0, str(HW.parents[1]))
import rclpy  # noqa: E402
from ur10e_calibration_run import Chain, Runner  # noqa: E402
from ur10e_interface import StreamingInterface  # noqa: E402

theta = float(sys.argv[1])
dry = "--dry_run" in sys.argv
START = np.array([-1.397, -0.745, 1.486, 4.001, -1.593, 3.301])
args = Namespace(base_frame="base_link", tool_frame="tool0", step_time=0.1, settle_timeout=3.0, dry_run=dry)
rclpy.init()
node = Runner(args)
for _ in range(200):
    node.spin_for(0.05)
    if node.joints is not None and node.urdf is not None and node.tool_pose() is not None:
        break
else:
    sys.exit("no /joint_states, /robot_description or TF")
chain = Chain(node.urdf, "base_link", "tool0")
iface = StreamingInterface(node, chain, 10.0, 0.01, 0.05, frame_yaw_deg=theta)
if not dry:
    node.args.step_time = 6.0
    node.send(START); node.settle()
    if np.abs(node.joints - START).max() > 0.01:
        node.send(START); node.settle()
    if np.abs(node.joints - START).max() > 0.01:
        sys.exit("could not reach the start pose")
p_start, r_start = iface.pose()
frame = Rotation.from_euler("z", np.radians(theta)).as_matrix()


def burst(a, ticks=5):
    iface.next_tick = None
    p0, r0 = iface.pose()
    for _ in range(ticks):
        sent, reason = iface.command(a, dry_run=dry)
        if not sent:
            sys.exit(f"IK refused: {reason}")
        p, r = iface.pose()
        if np.linalg.norm(p - p_start) > 0.03 or np.degrees(np.arccos(np.clip((np.trace(r @ r_start.T) - 1) / 2, -1, 1))) > 5:
            sys.exit("probe moved too far; stopped")
    node.spin_for(0.5)
    p1, r1 = iface.pose()
    return p1 - p0, Rotation.from_matrix(r1 @ r0.T).as_rotvec()


ok = True
print(f"command frame turned {theta:g} deg about z{' (DRY RUN: nothing sent, no motion expected)' if dry else ''}")
for i, name in enumerate("xyz"):
    a = np.zeros(6); a[i] = 0.3
    dp, _ = burst(a)
    burst(-a)
    want = frame[:, i]
    cos = float(dp @ want / max(np.linalg.norm(dp), 1e-9))
    print(f"+{name}_cmd: moved {np.round(dp * 1000, 1).tolist()} mm in base_link; predicted direction "
          f"{np.round(want, 2).tolist()}; cosine {cos:+.2f}")
    ok &= cos > 0.9 or dry
a = np.zeros(6); a[5] = 0.3
_, dr = burst(a)
burst(-a)
print(f"+rz_cmd: turned {np.degrees(dr[2]):+.2f} deg about base_link z (predicted sign {'+' if frame[2, 2] > 0 else '-'})")
ok &= dr[2] * frame[2, 2] > 0 or dry
if not dry:
    node.send(node.joints)
print("FRAME VERIFIED" if ok else "FRAME CHECK FAILED")
node.destroy_node()
rclpy.shutdown()
