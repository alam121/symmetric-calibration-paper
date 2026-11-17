#!/usr/bin/env python3
"""Run the sim-trained policy on the UR10e through the action interface, with no policy fine-tuning.

The policy (manipulatorsdatepalm/isaac/sim2real/policy.py, unchanged) outputs sim-interface actions;
each is mapped and streamed through the UR10e's native interface (ur10e_interface.py, the same
spec as the sim):

  direct      a_real = a_sim
  matched     a_real = a_sim  W_sim  pinv(W_real)   both fitted under the same rotation convention
  mismatched  a_real = a_sim  W_sim^world  pinv(W_real^body)   (deliberate phi mismatch)

W_sim comes from tasks_sim.py --calibrate runs, W_real from ur10e_calibration_run.py --stream_hz 10
runs, each fitted with the paper's fit_local_action_model.

Lab cell (--cell lab, measured in base_link 2026-10-13): task A brings the grasp point above the
green hexagon; task B picks the white 30 x 10 x 60 mm box standing on its end and stands it on the
hexagon's top at a target yaw (the sim's `tasks_sim.py --cell lab`, 12/12).

  python3 sim2real_deploy.py --cell lab --task B --mode direct --block 0.459 -1.014 80 --trial pose01 --dry_run

Objects (robot base frame): the receptacle and the task-A marker are measured once and passed in;
the block comes from a PoseStamped topic (e.g. an AprilTag detector) when one is publishing, else
from --block (its measured start pose), held rigidly once grasped.

  python3 sim2real_deploy.py --task B --mode matched --sim_calib SIM_RUN.jsonl --real_calib REAL_RUN.jsonl \\
      --slot 0.55 0.20 0.0 30 --block 0.55 -0.25 0.0 --trial pose01 --out deploy_runs/ [--dry_run]
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float32MultiArray, Int16MultiArray

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xembody"))
import policy as pol  # noqa: E402
import spec  # noqa: E402
from fit_compare import fit, load  # noqa: E402
from ur10e_calibration_run import Chain, Runner  # noqa: E402
from ur10e_interface import StreamingInterface  # noqa: E402

OPEN = [0.4350, -0.1130, 2.0250, -0.3700, -1.1330, 0.0000, 1.6580, -0.0930, 0.3770, 0.1710, 2.0320, -0.4400]
LAB_CLOSED = [0.3473, -0.0367, 1.9897, -0.0663, -1.0978, 0.0401, 2.0769, -0.0663, 0.3944, 0.1047, 2.1101, -0.0419]
PINCH = LAB_CLOSED[:4] + OPEN[4:8] + LAB_CLOSED[8:]  # F1 + F3 closed, F2 open (as the sim)
F2_CURLED = [-1.1330, 0.0, 0.85, 2.02]   # F2 tucked up: its tip 41 mm above the grasp point (tasks_sim.py)


# The measured cell, as tasks_sim.py's LAB: board height, hexagon (centre, top), the peg.
LAB = {"board_z": -0.329, "hexagon": ((0.586, -1.030), 0.038), "peg_half_height": 0.03, "peg_target_yaw": 30.0,
       "start_joints": [-1.3967, -0.7234, 1.4869, 3.9771, -1.5938, -0.8267],   # tool0 (0.35, -0.95, 0.071) down
       "box": [0.25, 0.75, -1.15, -0.85, -0.10, 0.20]}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=("A", "B"), required=True)
    p.add_argument("--cell", choices=("bench", "lab"), default="lab",
                   help="lab: the measured cell's hexagon / standing box, start pose and box (overridable).")
    p.add_argument("--mode", choices=("direct", "matched", "mismatched"), required=True)
    p.add_argument("--sim_calib", type=Path, help="tasks_sim.py --calibrate run (.jsonl)")
    p.add_argument("--real_calib", type=Path, help="ur10e_calibration_run.py --stream_hz run (.jsonl)")
    p.add_argument("--phi", choices=("body", "world"), default="body", help="Convention for the matched map.")
    p.add_argument("--marker", type=float, nargs=3, default=None, help="Task A target point [m].")
    p.add_argument("--slot", type=float, nargs=4, default=None, help="Receptacle floor centre x y z [m] and yaw [deg].")
    p.add_argument("--block", type=float, nargs=3, default=None, help="Block start x y [m], yaw [deg] (no tag).")
    p.add_argument("--block_topic", default="/sim2real/block_pose")
    p.add_argument("--gripper_topic", default="/gripper/target_joint")
    p.add_argument("--grasp_twist_deg", type=float, default=None,
                   help="Box yaw change in the hand when F1 + F3 close on it (lab default -57: measured on 4 pilot "
                   "grasps, -62/-50/-59/-57; the in-hand camera view was too occluded to measure it per trial).")
    p.add_argument("--inhand_capture", action="store_true",
                   help="Task B: pause ~1.5 s after the lift and save the wrist stereo pair + tool pose (box in the hand).")
    p.add_argument("--hold_force", type=int, default=50,
                   help="Delto holding mode once closed (x0.1 N on each finger's main closing joint, as the "
                   "harvest controller); 0 = position targets only (that pinch dropped the box in transfer).")
    p.add_argument("--start_joints", type=float, nargs=6, default=None,
                   help="The sim's start pose. Bench default: tool0 at (0.55, -0.05, 0.45) pointing down.")
    p.add_argument("--box", type=float, nargs=6, default=None,
                   help="tool0 workspace x_min x_max y_min y_max z_min z_max [m]; leaving it aborts.")
    p.add_argument("--max_tilt_deg", type=float, default=25.0,
                   help="Abort when tool z is further than this from pointing down (normal runs stay under ~12 deg; "
                   "the phi-mismatched map turned the hand on its side before the operator stopped it).")
    p.add_argument("--seconds", type=float, default=None, help="Default: 20 (A) / 60 (B), as the sim.")
    p.add_argument("--trial", default="trial")
    p.add_argument("--out", type=Path, default=Path("deploy_runs"))
    p.add_argument("--dry_run", action="store_true")
    # Runner fields the node expects.
    p.add_argument("--base_frame", default="base_link")
    p.add_argument("--tool_frame", default="tool0")
    p.add_argument("--step_time", type=float, default=0.1)
    p.add_argument("--settle_timeout", type=float, default=3.0)
    args = p.parse_args()
    if args.grasp_twist_deg is None:
        args.grasp_twist_deg = -57.0 if args.cell == "lab" and args.task == "B" else 0.0
    if args.cell == "lab":
        (hx, hy), hh = LAB["hexagon"]
        top = LAB["board_z"] + hh
        args.marker = args.marker or [hx, hy, top + 0.12]   # the open F2 (47 mm below) clears the standing peg
        args.slot = args.slot or [hx, hy, top, LAB["peg_target_yaw"]]
        args.block_z = LAB["board_z"] + LAB["peg_half_height"]
        args.start_joints = args.start_joints or LAB["start_joints"]
        args.box = args.box or LAB["box"]
    else:
        args.block_z = spec.BLOCK_SIZE[2] / 2
        args.start_joints = args.start_joints or [-0.4228, -1.7953, 2.1764, 4.3534, -1.5905, -2.0051]
        args.box = args.box or [0.25, 0.85, -0.55, 0.45, 0.20, 0.60]
    return args


def action_map(args) -> np.ndarray:
    if args.mode == "direct":
        return np.eye(6)
    phis = (args.phi, args.phi) if args.mode == "matched" else ("world", "body")
    ws = fit(*load(args.sim_calib)[:3], phis[0]).matrix
    wr = fit(*load(args.real_calib)[:3], phis[1]).matrix
    return ws @ np.linalg.pinv(wr)


def main():
    args = parse_args()
    M = action_map(args)
    rclpy.init()
    node = Runner(args)
    gripper_pub = node.create_publisher(Float32MultiArray, args.gripper_topic, 10)
    hold_pub = node.create_publisher(Int16MultiArray, "/gripper/holding_control", 10)
    force = {"N": None}
    node.create_subscription(Float32MultiArray, "/gripper/force", lambda m: force.update(N=list(m.data)), 10)
    tag = {"pose": None, "t": 0.0}

    def tag_cb(msg):
        q = msg.pose.orientation
        tag["pose"] = (np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
                       Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix())
        tag["t"] = time.time()

    node.create_subscription(PoseStamped, args.block_topic, tag_cb, 10)
    for _ in range(200):
        node.spin_for(0.05)
        if node.joints is not None and node.urdf is not None and node.tool_pose() is not None:
            break
    else:
        sys.exit("no /joint_states, /robot_description or TF")
    chain = Chain(node.urdf, args.base_frame, args.tool_frame)
    interface = StreamingInterface(node, chain, spec.RATE_HZ, spec.UNIT_TRANSLATION, spec.UNIT_ROTATION)

    open_posture, pinch = OPEN, PINCH
    if args.cell == "lab" and args.task == "B":   # open, F2's cap hung ~3 mm above the board at the grasp
        open_posture, pinch = OPEN[:4] + F2_CURLED + OPEN[8:], PINCH[:4] + F2_CURLED + PINCH[8:]

    # Holding mode (delto_gripper_controller.py): [on, mode 1 = normal, force, 12-joint mask], mask 0 = the
    # force-controlled joint. F1 and F3 squeeze with their main closing joint once the position close has
    # had 0.8 s; F2 (curled, not in the grasp) is held in position entirely.
    free = [max(range(f * 4, f * 4 + 4), key=lambda i: abs(pinch[i] - open_posture[i])) for f in (0, 2)]
    holding = {"on": False, "closed_ticks": 0}

    def hold(on):
        if args.dry_run or args.hold_force <= 0 or holding["on"] == on:
            return
        mask = [0 if i in free else 1 for i in range(12)]
        hold_pub.publish(Int16MultiArray(data=[1 if on else 0, 1, max(0, min(200, args.hold_force)), *mask]))
        holding["on"] = on

    def grip(command):
        if command == "close":
            holding["closed_ticks"] += 1
            if holding["closed_ticks"] == 8:
                hold(True)
        else:
            holding["closed_ticks"] = 0
            hold(False)   # release the hold before opening, as the harvest controller does
        """Policy gripper output: "close" (the F1 + F3 pinch), "open", or an opening fraction (release)."""
        if isinstance(command, str):
            posture = pinch if command == "close" else open_posture
        else:
            posture = np.array(pinch) + command * (np.array(open_posture) - np.array(pinch))
        if not args.dry_run:
            gripper_pub.publish(Float32MultiArray(data=[float(v) for v in posture]))

    print(f"[DEPLOY] task {args.task}, mode {args.mode}, map diag {np.round(np.diag(M), 3).tolist()}", flush=True)
    if not args.dry_run:
        grip("open")
        # Check the arm got there: the first message on a new publisher can be lost before DDS discovery
        # completes (one trial started where the previous one ended).
        node.args.step_time = 6.0
        for attempt in range(3):
            node.send(np.array(args.start_joints))
            node.settle()
            if np.abs(node.joints - np.array(args.start_joints)).max() < 0.01:
                break
            print(f"[DEPLOY] not at the start pose (max joint error "
                  f"{np.abs(node.joints - np.array(args.start_joints)).max():.3f} rad), resending", flush=True)
        else:
            sys.exit("[DEPLOY] could not reach the start pose; aborting")
    interface.next_tick = None

    if args.task == "A":
        task = pol.TaskA(np.array(args.marker))
    else:
        # Lab: F1 + F3 across the box's 30 mm side, 20 mm above its centre, pressed 4 mm onto the top
        # (tasks_sim.py --cell lab).
        task = pol.TaskB(half_height=LAB["peg_half_height"], grip_above=0.02, grip_along=0.0, place_above=-0.004,
                         grip_yaw_deg=90.0) if args.cell == "lab" else pol.TaskB()
        slot = (np.array(args.slot[:3]), np.radians(args.slot[3]))
        held = {"tool_to_block": None}
    camera = {}
    if args.inhand_capture:
        for side in ("left", "right"):
            node.create_subscription(CompressedImage, f"/vision/nurec/{side}/image_rect/compressed",
                                     lambda m, s=side: camera.update({s: (time.time(), m)}), qos_profile_sensor_data)
        node.create_subscription(CameraInfo, "/vision/nurec/left/camera_info",
                                 lambda m: camera.update(K=list(m.k)), qos_profile_sensor_data)

    def capture_in_hand(position, rotation):
        """Hold still, wait for a stereo pair taken after the arm stopped, save it with the tool pose."""
        start = time.time()
        while time.time() - start < 4.0:
            interface.command(np.zeros(6), dry_run=args.dry_run)
            if all(side in camera and camera[side][0] > start + 0.6 for side in ("left", "right")):
                break
        args.out.mkdir(parents=True, exist_ok=True)
        stem = args.out / f"{args.trial}_task{args.task}_{args.mode}_inhand"
        for side in ("left", "right"):
            if side in camera:
                Path(f"{stem}_{side}.jpg").write_bytes(bytes(camera[side][1].data))
        p, r = interface.pose()
        Path(f"{stem}.json").write_text(json.dumps({"K": camera.get("K"), "tool_p": p.tolist(),
                                                    "tool_q_xyzw": Rotation.from_matrix(r).as_quat().tolist()}))
        print(f"[DEPLOY] in-hand frames -> {stem}_*.jpg ({time.time() - start:.1f} s pause)", flush=True)
        interface.next_tick = None

    rows, box = [], np.array(args.box).reshape(3, 2)
    seconds = args.seconds or (20.0 if args.task == "A" else 60.0)
    outcome = "timeout"
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for step in range(int(seconds * spec.RATE_HZ)):
            position, rotation = interface.pose()
            if np.any(position < box[:, 0]) or np.any(position > box[:, 1]):
                outcome = f"abort: tool0 left the workspace box at {np.round(position, 3).tolist()}"
                break
            tilt = np.degrees(np.arccos(np.clip(-rotation[2, 2], -1.0, 1.0)))
            if tilt > args.max_tilt_deg:
                outcome = f"abort: tool tilted {tilt:.0f} deg from pointing down"
                break
            objects = None
            if args.task == "B":
                if tag["pose"] is not None and time.time() - tag["t"] < 0.5:
                    block = tag["pose"]
                elif held["tool_to_block"] is not None:  # no tag: rigidly held since the grasp
                    block = (position + rotation @ held["tool_to_block"][0], rotation @ held["tool_to_block"][1])
                else:
                    block = (np.array([args.block[0], args.block[1], args.block_z]),
                             Rotation.from_euler("z", np.radians(args.block[2])).as_matrix())
                if task.phase == "lift" and held["tool_to_block"] is None:
                    twist = Rotation.from_euler("z", np.radians(args.grasp_twist_deg)).as_matrix()
                    held["tool_to_block"] = (rotation.T @ (block[0] - position), rotation.T @ twist @ block[1])
                objects = {"block": block, "slot": slot}
            if args.inhand_capture and args.task == "B" and task.phase == "above_slot" and not camera.get("done"):
                camera["done"] = True
                capture_in_hand(position, rotation)
                position, rotation = interface.pose()
            a_sim, gripper = task.act(position, rotation, objects)
            a_real = np.clip(a_sim @ M, -1.0, 1.0)
            grip(gripper)
            sent, reason = interface.command(a_real, dry_run=args.dry_run)
            rows.append({"step": step, "t": time.time(), "phase": task.phase, "a_sim": a_sim.tolist(),
                         "a_real": a_real.tolist(), "gripper": gripper, "force_N": force["N"], "tool_p": position.tolist(),
                         "tool_q_xyzw": Rotation.from_matrix(rotation).as_quat().tolist(), "sent": sent, "reason": reason})
            if args.task == "A" and task.phase == "hold" or args.task == "B" and task.phase == "done":
                outcome = "policy finished"
                break
    except KeyboardInterrupt:   # stopped by the operator / killed: still hold, release and save the log
        outcome = "stopped by operator"

    if not args.dry_run:
        node.send(node.joints)  # hold
        hold(False)               # never leave the Delto in holding mode
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.trial}_task{args.task}_{args.mode}.json"
    path.write_text(json.dumps({"task": args.task, "mode": args.mode, "trial": args.trial, "outcome": outcome,
                                "map": M.tolist(), "phi": args.phi, "grasp_twist_deg": args.grasp_twist_deg, "sim_calib": str(args.sim_calib),
                                "real_calib": str(args.real_calib), "steps": rows}, indent=1))
    print(f"[DEPLOY] {outcome}; judge success on the table and record it in the trial sheet -> {path}", flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
