"""Sim side of the sim-to-real interface study: tasks A/B and task-free calibration in Isaac.

The UR10e + Delto on a table, a 100x40x40 mm block and a receptacle (spec.py), driven only
through the sim's native relative Cartesian interface: Isaac Lab differential IK in relative mode
(position in the base frame, R_target = exp(dr) R_now), one command per 1/RATE_HZ, streamed --
the arm is never waited for, so tracking lag belongs to the interface.

  episodes   --task A|B --episodes N: policy.py from the sim state; success from the sim
  calibrate  --calibrate RUN_ID --beta 0.85 --seed 1 --steps 300: the paper's SmoothExcitation
             (same settings as ur10e_calibration_run.py) through the same interface, logged in its
             format, for fit_compare.py / the sim-to-real adapter

  unset CONDA_SHLVL; ~/miniconda3/envs/env_isaaclab/bin/python isaac/sim2real/tasks_sim.py --task B --episodes 10 --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", choices=("A", "B"), default="B")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "results")
parser.add_argument("--calibrate", default=None, help="Run id: collect a task-free calibration run instead.")
parser.add_argument("--calib_box", type=float, nargs=3, default=[0.10, 0.10, 0.08],
                    help="Calibration: +- excursion of tool0 from the start [m] (lab: 0.05 0.05 0.05 clears the props).")
parser.add_argument("--calib_tilt_deg", type=float, default=15.0)
parser.add_argument("--beta", type=float, default=0.85)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--match_magnitude", type=float, default=0.15)
parser.add_argument("--adapter", type=Path, default=None,
                    help="JSON {'M': 6x6}: actions are sent as a @ M (to check an adapter in sim).")
parser.add_argument("--cell", choices=("bench", "lab"), default="bench",
                    help="bench: the generic table and block; lab: the measured real cell (board 0.329 m below the "
                    "base, green hexagon target, a date on a stand, a slot box).")
parser.add_argument("--lab_object", choices=("peg", "date"), default="peg",
                    help="lab task B: peg = the upright 6x3x1 cm white box, placed standing on the hexagon at a "
                    "target yaw; date = a date dropped into a slot box (the Delto could not pinch it reliably).")
parser.add_argument("--date_stand", action="store_true", help="lab date: put the date on a ridge (default: flat on the board).")
parser.add_argument("--stand_yaw_deg", type=float, default=0.0, help="lab: yaw of the ridge (and the date on it).")
parser.add_argument("--only_episode", type=int, default=None, help="Run just this episode (debugging).")
parser.add_argument("--slot_yaw_deg", type=float, default=30.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import policy as pol  # noqa: E402
from spec import (BLOCK_SIZE, CLIP, GRASP_DEPTH, RATE_HZ, RECEPTACLE_CLEARANCE, RECEPTACLE_WALL,  # noqa: E402
                  UNIT_ROTATION, UNIT_TRANSLATION)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_experiments.task_free_calibration.collect_calibration import SmoothExcitation  # noqa: E402

DT = 1.0 / 120.0
STEPS_PER_COMMAND = int(round(1.0 / (RATE_HZ * DT)))
ROBOT_USD = os.environ.get("UR10E_USD", str(Path.home() / "isaac_ws/ur10e_delto/ur10e_delto.usd"))
ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
HOME = [-1.352599, -0.582120, 1.307726, 2.136123, -1.784902, -1.059150]  # lab home
GRIP = [f"gripper_f{f}m{m}_joint" for f in (1, 2, 3) for m in (1, 2, 3, 4)]
OPEN = [0.4350, -0.1130, 2.0250, -0.3700, -1.1330, 0.0000, 1.6580, -0.0930, 0.3770, 0.1710, 2.0320, -0.4400]
LAB_CLOSED = [0.3473, -0.0367, 1.9897, -0.0663, -1.0978, 0.0401, 2.0769, -0.0663, 0.3944, 0.1047, 2.1101, -0.0419]
# Block pinch: F1 and F3 take their lab-closed posture, F2 stays open (it would push the block's end).
CLOSED = LAB_CLOSED[:4] + OPEN[4:8] + LAB_CLOSED[8:]
BLOCK_HOME = np.array([0.55, -0.25, BLOCK_SIZE[2] / 2])
SLOT = np.array([0.55, 0.20, 0.0])
MARKER_HOME = np.array([0.55, 0.0, 0.20])

# The real lab cell, measured from the wrist camera's stereo pair on 2026-10-13 (base frame).
LAB = {
    "board_z": -0.329,
    "table_centre": (0.35, -1.10), "table_size": (1.2, 0.7),
    "start_tool": np.array([0.35, -0.95, -0.329 + 0.40]),        # ~ the real look-down camera height
    "start_x": np.array([-0.552, -0.834, 0.0]),                    # tool x as on the robot, tool z down
    "hexagon": (np.array([0.586, -1.030]), 0.055, 0.038),          # centre, radius, height (model fit, 10-13)
    "date": (0.032, 0.0105),                                       # length, radius (median 32 x 21 mm)
    "date_home": np.array([0.25, -1.00]),
    # A ridge under the date (a strip on its edge): F1's closed cap passes ~10 mm from the axis down to
    # ~34 mm below the grasp point, so a round 24 mm stand was in its way; an 8 mm strip is not.
    "stand": (0.05, 0.008, 0.03),                                  # length, width, height
    "box": (np.array([0.45, -1.12]), np.radians(30.0), (0.040, 0.026), 0.05, 0.005),  # centre, yaw, opening, wall h, t
    # The white box standing on its 3x1 cm end (2026-10-13: top 62 mm above the board; centre and long-side
    # yaw from a model-silhouette fit seen from above, ~+-5 deg), to be stood on the hexagon's top.
    # The hexagon moved 3.8 cm from the first stereo estimate (that one was pulled by its side faces).
    "peg": ((0.03, 0.01, 0.06), np.array([0.459, -1.011]), np.radians(100.0), 0.015),
    "peg_target_yaw": np.radians(30.0),
}
# F2 curled inward (tip ~41 mm above the grasp point, near the axis): F1 + F3 pinch small objects
# without the open F2, the lowest point of the hand, hitting the board.
F2_CURLED = [-1.1330, 0.0, 0.85, 2.02]
# Start pose for every episode and calibration run: tool0 above the task area, tool z down.
START_TOOL = np.array([0.55, -0.05, 0.45])
START_ROTATION = np.column_stack([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def receptacle_walls(centre, yaw):
    """Four static walls around an opening of the block size plus clearance."""
    length, width = BLOCK_SIZE[0] + 2 * RECEPTACLE_CLEARANCE, BLOCK_SIZE[1] + 2 * RECEPTACLE_CLEARANCE
    thick, height = RECEPTACLE_WALL
    r = Rotation.from_euler("z", yaw)
    walls = []
    for offset, size in (((0, (width + thick) / 2), (length + 2 * thick, thick)),
                         ((0, -(width + thick) / 2), (length + 2 * thick, thick)),
                         (((length + thick) / 2, 0), (thick, width)),
                         ((-(length + thick) / 2, 0), (thick, width))):
        position = centre + r.apply([offset[0], offset[1], height / 2])
        q = r.as_quat()  # x y z w
        walls.append((tuple(position), (q[3], q[0], q[1], q[2]), (size[0], size[1], height)))
    return walls


@configclass
class SceneCfg(InteractiveSceneCfg):
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    table = AssetBaseCfg(prim_path="/World/Table", spawn=sim_utils.CuboidCfg(
        size=(1.6, 1.6, 0.05), collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.55))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.0, -0.025)))
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=ROBOT_USD, rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=ARM, stiffness=1200.0, damping=120.0,
                                       effort_limit_sim=10000.0, velocity_limit_sim=3.0),
            "gripper": ImplicitActuatorCfg(joint_names_expr=["gripper_.*"], stiffness=40.0, damping=2.0,
                                           effort_limit_sim=2.0, velocity_limit_sim=3.0),
        })
    block = RigidObjectCfg(
        prim_path="/World/Block",
        spawn=sim_utils.CuboidCfg(
            size=BLOCK_SIZE, rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1), collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.1))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(BLOCK_HOME)))


def add_receptacle(cfg, yaw):
    for i, (position, quat, size) in enumerate(receptacle_walls(SLOT, yaw)):
        setattr(cfg, f"wall{i}", AssetBaseCfg(
            prim_path=f"/World/Receptacle/wall{i}",
            spawn=sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.8))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=position, rot=quat)))


def gripper_posture(gripper):
    """Policy gripper output -> Delto joint targets: "close", "open", or an opening fraction."""
    if isinstance(gripper, str):
        return CLOSED if gripper == "close" else OPEN
    return (np.array(CLOSED) + gripper * (np.array(OPEN) - np.array(CLOSED))).tolist()


def configure_lab(cfg):
    """The measured real cell: board, hexagon, date (+ stand), slot box. Returns the date's rest height."""
    z = LAB["board_z"]
    cfg.table = AssetBaseCfg(prim_path="/World/Table", spawn=sim_utils.CuboidCfg(
        size=(*LAB["table_size"], 0.05), collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.55, 0.36))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*LAB["table_centre"], z - 0.025)))
    (hx, hr, hh) = LAB["hexagon"]
    cfg.hexagon = AssetBaseCfg(prim_path="/World/Hexagon", spawn=sim_utils.CylinderCfg(
        radius=hr, height=hh, collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.75, 0.6))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(hx[0], hx[1], z + hh / 2)))
    if args.lab_object == "peg":
        size, home, _, mass = LAB["peg"]
        cfg.block = RigidObjectCfg(prim_path="/World/Block", spawn=sim_utils.CuboidCfg(
            size=size, rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass), collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(*home, z + size[2] / 2 + 0.001)))
        return z + size[2] / 2 + 0.001   # no slot box: the target is the hexagon's top
    length, radius = LAB["date"]
    rest = z + radius + 0.002
    if args.date_stand:
        sl, sw, sh = LAB["stand"]
        q = Rotation.from_euler("z", np.radians(args.stand_yaw_deg)).as_quat()
        cfg.stand = AssetBaseCfg(prim_path="/World/Stand", spawn=sim_utils.CuboidCfg(
            size=(sl, sw, sh), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(*LAB["date_home"], z + sh / 2), rot=(q[3], q[0], q[1], q[2])))
        rest = z + sh + radius + 0.002   # dropped from 2 mm: spawned touching, PhysX shot it through the board
    cfg.block = RigidObjectCfg(prim_path="/World/Block", spawn=sim_utils.CapsuleCfg(
        radius=radius, height=length - 2 * radius, axis="X",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.5),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.008), collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.18, 0.08))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*LAB["date_home"], rest)))
    centre, yaw, (ol, ow), wall_h, t = LAB["box"]
    r = Rotation.from_euler("z", yaw)
    for i, (offset, size) in enumerate((((0, (ow + t) / 2), (ol + 2 * t, t)), ((0, -(ow + t) / 2), (ol + 2 * t, t)),
                                        (((ol + t) / 2, 0), (t, ow)), ((-(ol + t) / 2, 0), (t, ow)))):
        position = np.array([centre[0], centre[1], z]) + r.apply([offset[0], offset[1], wall_h / 2])
        q = r.as_quat()
        setattr(cfg, f"wall{i}", AssetBaseCfg(prim_path=f"/World/Box/wall{i}", spawn=sim_utils.CuboidCfg(
            size=(size[0], size[1], wall_h), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.8))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(position), rot=(q[3], q[0], q[1], q[2]))))
    return rest


def main():
    global START_TOOL, START_ROTATION, BLOCK_HOME, SLOT, MARKER_HOME, OPEN, CLOSED
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    slot_yaw = np.radians(args.slot_yaw_deg)
    lab = args.cell == "lab"
    if lab:
        rest = configure_lab(cfg)
        START_TOOL = LAB["start_tool"]
        x = LAB["start_x"] / np.linalg.norm(LAB["start_x"])
        START_ROTATION = np.column_stack([x, np.cross([0.0, 0.0, -1.0], x), [0.0, 0.0, -1.0]])
        BLOCK_HOME = np.array([*LAB["date_home"], rest])
        hx, _, hh = LAB["hexagon"]
        if args.lab_object == "peg":
            BLOCK_HOME = np.array([*LAB["peg"][1], rest])
            SLOT, slot_yaw = np.array([hx[0], hx[1], LAB["board_z"] + hh]), LAB["peg_target_yaw"]  # the hexagon top
        else:
            centre, slot_yaw, _, wall_h, _ = LAB["box"]
            SLOT = np.array([centre[0], centre[1], LAB["board_z"] + wall_h])   # the rim: the date is dropped in
        # 12 cm above the top: the open F2 hangs 47 mm below the grasp point and the straight path passes
        # ~30 mm from the standing peg (60 mm tall); at 8 cm the real hand cleared it by <1 cm.
        MARKER_HOME = np.array([hx[0], hx[1], LAB["board_z"] + hh + 0.12])
        if args.date_stand:   # F1 + F3 pinch over the ridge, F2 curled away
            OPEN = OPEN[:4] + F2_CURLED + OPEN[8:]
            CLOSED = LAB_CLOSED[:4] + F2_CURLED + LAB_CLOSED[8:]
        elif args.lab_object == "date":   # the harvest pinch: the F1 / F2 cap ends meet (8 mm gap) at 0.255 m
            CLOSED = LAB_CLOSED
        elif args.task == "B":   # peg: F1 + F3 pinch with F2 curled up (open, its cap hung ~3 mm above the board)
            OPEN = OPEN[:4] + F2_CURLED + OPEN[8:]
            CLOSED = CLOSED[:4] + F2_CURLED + CLOSED[8:]
    else:
        add_receptacle(cfg, slot_yaw)
    scene = InteractiveScene(cfg)
    sim.reset()
    robot, block = scene["robot"], scene["block"]
    dev = sim.device
    arm_ids = [robot.joint_names.index(n) for n in ARM]
    grip_ids = [robot.joint_names.index(n) for n in GRIP]
    tool = robot.body_names.index("tool0")
    limits = robot.data.joint_pos_limits.clone()
    limits[:, grip_ids, 0] = torch.clamp(limits[:, grip_ids, 0], max=-1.0)
    limits[:, grip_ids, 1] = torch.clamp(limits[:, grip_ids, 1], min=2.8)
    robot.write_joint_position_limit_to_sim(limits)
    ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True,
                                                              ik_method="dls"), num_envs=1, device=dev)
    adapter = np.array(json.loads(args.adapter.read_text())["M"]) if args.adapter else None
    state = {"grip": torch.tensor([OPEN], device=dev), "arm": torch.tensor([HOME], device=dev), "start": None}
    ik_abs = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                                                  ik_method="dls"), num_envs=1, device=dev)

    def tool_pose():
        p, q = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                         robot.data.body_pos_w[:, tool], robot.data.body_quat_w[:, tool])
        w, x, y, z = q[0].tolist()
        return p[0].cpu().numpy(), Rotation.from_quat([x, y, z, w]).as_matrix(), p, q

    def physics(n):
        for _ in range(n):
            robot.set_joint_position_target(state["arm"], joint_ids=arm_ids)
            robot.set_joint_position_target(state["grip"], joint_ids=grip_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(DT)

    def command(action):
        """The native interface: one relative pose command, then 1/RATE_HZ of physics."""
        action = np.clip(np.asarray(action, float), -CLIP, CLIP)
        if adapter is not None:
            action = np.clip(action @ adapter, -CLIP, CLIP)
        _, _, p, q = tool_pose()
        delta = torch.tensor([np.concatenate([action[:3] * UNIT_TRANSLATION, action[3:] * UNIT_ROTATION])],
                             device=dev, dtype=torch.float32)
        ik.set_command(delta, p, q)
        jacobian = robot.root_physx_view.get_jacobians()[:, tool - 1, :, arm_ids]
        state["arm"] = ik.compute(p, q, jacobian, robot.data.joint_pos[:, arm_ids])
        physics(STEPS_PER_COMMAND)

    def start_joints():
        """Solve once (teleporting, not part of any episode) for the start pose."""
        if state["start"] is None:
            joints = robot.data.default_joint_pos.clone()
            joints[:, arm_ids] = torch.tensor([HOME], device=dev)
            q = Rotation.from_matrix(START_ROTATION).as_quat()
            goal = torch.tensor([list(START_TOOL) + [q[3], q[0], q[1], q[2]]], device=dev, dtype=torch.float32)
            for _ in range(300):
                robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
                sim.step()
                scene.update(DT)
                _, _, p, quat = tool_pose()
                ik_abs.set_command(goal)
                jacobian = robot.root_physx_view.get_jacobians()[:, tool - 1, :, arm_ids]
                step = ik_abs.compute(p, quat, jacobian, joints[:, arm_ids]) - joints[:, arm_ids]
                joints[:, arm_ids] += torch.clamp(step, -0.05, 0.05)
            state["start"] = joints[:, arm_ids].clone()
            position, rotation, _, _ = tool_pose()
            print(f"[SIM2REAL] start pose: tool0 {np.round(position, 3).tolist()}, "
                  f"{np.degrees(np.linalg.norm(Rotation.from_matrix(rotation @ START_ROTATION.T).as_rotvec())):.1f} deg "
                  f"from the target orientation; joints {np.round(state['start'][0].tolist(), 4).tolist()}", flush=True)
        return state["start"]

    def reset(block_xy=None, block_yaw=0.0):
        joints = robot.data.default_joint_pos.clone()
        joints[:, arm_ids] = start_joints()
        joints[:, grip_ids] = torch.tensor([OPEN], device=dev)
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        state["arm"], state["grip"] = joints[:, arm_ids].clone(), torch.tensor([OPEN], device=dev)
        pose = torch.zeros((1, 7), device=dev)
        xy = BLOCK_HOME[:2] if block_xy is None else block_xy
        pose[0, :3] = torch.tensor([xy[0], xy[1], BLOCK_HOME[2]], device=dev)
        q = Rotation.from_euler("z", block_yaw).as_quat()
        pose[0, 3:] = torch.tensor([q[3], q[0], q[1], q[2]], device=dev)
        block.write_root_pose_to_sim(pose)
        block.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
        physics(60)

    def block_pose():
        p = block.data.root_pos_w[0].cpu().numpy()
        w, x, y, z = block.data.root_quat_w[0].tolist()
        return p, Rotation.from_quat([x, y, z, w])

    args.out.mkdir(parents=True, exist_ok=True)
    if args.calibrate:
        return calibrate(reset, tool_pose, command)
    rng = np.random.default_rng(args.seed)
    results = []
    for episode in range(args.episodes):
        if lab and args.lab_object == "peg":
            # The box where it stands, +-1 cm, its yaw +-20 deg around the measured 80 deg.
            block_xy = BLOCK_HOME[:2] + rng.uniform(-0.01, 0.01, 2)
            block_yaw = float(LAB["peg"][2] + rng.uniform(np.radians(-20), np.radians(20)))
            marker = MARKER_HOME + np.concatenate([rng.uniform(-0.015, 0.015, 2), [0.0]])
        elif lab:
            # The date lies along the ridge: +-1 cm along it, +-5 deg about it; the hexagon is fixed.
            yaw0 = np.radians(args.stand_yaw_deg)
            block_xy = BLOCK_HOME[:2] + rng.uniform(-0.01, 0.01) * np.array([np.cos(yaw0), np.sin(yaw0)])
            block_yaw = float(yaw0 + rng.uniform(np.radians(-5), np.radians(5)))
            marker = MARKER_HOME + np.concatenate([rng.uniform(-0.015, 0.015, 2), [0.0]])
        else:
            block_xy = BLOCK_HOME[:2] + rng.uniform(-0.03, 0.03, 2)
            block_yaw = float(rng.uniform(np.radians(-25), np.radians(25)))
            marker = MARKER_HOME + rng.uniform(-0.05, 0.05, 3)
        if args.only_episode is not None and episode != args.only_episode:
            continue
        reset(block_xy, block_yaw)
        if args.task == "A":
            task = pol.TaskA(marker)
        elif lab and args.lab_object == "peg":
            # F1 + F3 squeeze across the 30 mm side (tool x along the 10 mm side: +90 deg), 20 mm above the
            # centre so the open F2 clears the board; stand it 3 mm above the hexagon top, then release.
            # place_above -4 mm: the box is pressed onto the hexagon before release (on the robot, released
            # from ~8 mm up -- the arm lags the target by ~5 mm -- the light box turned as it landed).
            task = pol.TaskB(half_height=LAB["peg"][0][2] / 2, grip_above=0.02, grip_along=0.0, place_above=-0.004,
                             grip_yaw_deg=90.0)
        elif lab:
            if args.date_stand:  # F1 + F3 pinch at the date centre
                task = pol.TaskB(half_height=LAB["date"][1], grip_above=0.0, grip_along=0.0, place_above=0.015)
            else:
                # Harvest pinch on the flat date: the cap ends (0.255 m, 0.03 below the policy's 0.225 grasp
                # point) at the date centre, the F1-F2 closing line (~34 deg from tool x) across its width.
                # 8 mm above the centre: fully open, the F2 cap reaches ~17 mm below the meeting point and
                # started inside the board, so it never closed.
                task = pol.TaskB(half_height=LAB["date"][1], grip_above=0.038, grip_along=0.0, place_above=0.015,
                                 grip_yaw_deg=-(33.7 + 90.0))
        else:
            task = pol.TaskB()
        steps = int((20.0 if args.task == "A" else 60.0) * RATE_HZ)
        held = 0
        for k in range(steps):
            position, rotation, _, _ = tool_pose()
            b, b_rot = block_pose()
            action, gripper = task.act(position, rotation, {"block": (b, b_rot.as_matrix()),
                                                            "slot": (SLOT, slot_yaw)})
            posture = gripper_posture(gripper)
            state["grip"] = torch.tensor([posture], device=dev)
            if os.environ.get("SIM2REAL_DEBUG") and (k % 10 == 0 or task.phase in ("release", "retreat") or os.environ.get("SIM2REAL_DEBUG") == "2"):
                local = Rotation.from_euler("z", -slot_yaw).apply(b - SLOT)
                if task.phase in ("descend", "close", "lift"):
                    tips = []
                    for f in (1, 2):
                        i = robot.body_names.index(f"gripper_f{f}_long_tip_link")
                        w, x_, y_, z_ = robot.data.body_quat_w[0, i].tolist()
                        end = robot.data.body_pos_w[0, i].cpu().numpy() + Rotation.from_quat([x_, y_, z_, w]).apply([0.0886, 0, 0])
                        tips.append(np.round(b_rot.inv().apply(end - b) * 1000).astype(int).tolist())
                    print(f"[TIPS] phase={task.phase} F1/F2 cap ends in the date frame (mm, x = long axis): {tips}", flush=True)
                print(f"[DEBUG] t={k / RATE_HZ:4.1f}s phase={task.phase} tool={np.round(position, 3).tolist()} "
                      f"action={np.round(action, 2).tolist()} block_slot_mm={np.round(local * 1000).astype(int).tolist()} "
                      f"tilt_deg={np.degrees(np.arccos(np.clip(b_rot.apply([0, 0, 1])[2], -1, 1))):.0f}", flush=True)
            command(action)
            if args.task == "A":
                grasp = position + rotation @ np.array([0.0, 0.0, GRASP_DEPTH])
                held = held + 1 if np.linalg.norm(grasp - marker) < 0.02 else 0
                if held >= RATE_HZ:  # within 2 cm for 1 s
                    break
            elif task.phase == "done":
                break
        if args.task == "A":
            position, rotation, _, _ = tool_pose()
            error = float(np.linalg.norm(position + rotation @ np.array([0, 0, GRASP_DEPTH]) - marker))
            success, detail = held >= RATE_HZ, {"position_error_mm": error * 1000}
            if lab and args.lab_object == "peg":   # the hand must pass the standing peg without knocking it
                _, b_rot = block_pose()
                detail["peg_tilt_deg"] = float(np.degrees(np.arccos(np.clip(b_rot.apply([0, 0, 1])[2], -1, 1))))
                success = success and detail["peg_tilt_deg"] < 15
        else:
            physics(60)
            b, b_rot = block_pose()
            local = Rotation.from_euler("z", -slot_yaw).apply(b - SLOT)
            yaw_error = abs((b_rot.as_euler("xyz")[2] - slot_yaw + np.pi / 2) % np.pi - np.pi / 2)
            # Resting on the floor inside the walls (a block caught on a wall sits higher).
            if lab and args.lab_object == "peg":
                # Standing on the hexagon top: within its radius, upright, centre at top + half height.
                _, hr, _ = LAB["hexagon"]
                tilt = np.degrees(np.arccos(np.clip(b_rot.apply([0, 0, 1])[2], -1, 1)))
                inside = (np.hypot(local[0], local[1]) < hr - 0.01 and tilt < 15
                          and abs(b[2] - (SLOT[2] + LAB["peg"][0][2] / 2)) < 0.01)
            elif lab:
                # In the box: inside its opening's footprint, below the rim.
                _, _, (ol, ow), _, _ = LAB["box"]
                inside = abs(local[0]) < ol / 2 and abs(local[1]) < ow / 2 and b[2] < SLOT[2]
            else:
                # Resting on the floor inside the walls (a block caught on a wall sits higher).
                inside = abs(local[0]) < 0.02 and abs(local[1]) < 0.01 and b[2] < BLOCK_SIZE[2] / 2 + 0.005
            success = bool(inside and np.degrees(yaw_error) < 10)
            detail = {"phase": task.phase, "block_in_slot_frame_mm": np.round(local * 1000, 1).tolist(),
                      "yaw_error_deg": float(np.degrees(yaw_error))}
        results.append({"episode": episode, "success": bool(success), **detail})
        print(f"[SIM2REAL] task {args.task} episode {episode}: {'success' if success else 'failure'} {detail}", flush=True)
    rate = np.mean([r["success"] for r in results])
    name = f"task{args.task}_seed{args.seed}{'_adapted' if adapter is not None else ''}"
    (args.out / f"{name}.json").write_text(json.dumps({"task": args.task, "success_rate": rate,
                                                        "episodes": results}, indent=1))
    print(f"[SIM2REAL] task {args.task}: {rate:.0%} of {len(results)} episodes", flush=True)


def calibrate(reset, tool_pose, command):
    """The hardware runner's protocol through the sim's native interface (streamed at RATE_HZ)."""
    reset()
    rng = np.random.default_rng(args.seed)
    stationary = np.sqrt((1 - args.beta) / (1 + args.beta))
    scale = args.match_magnitude / (stationary * 0.28) if args.match_magnitude > 0 else 1.0
    excitation = SmoothExcitation(rng, args.beta, 0.28 * scale, 0.18 * scale, gripper_period=10 ** 9,
                                  workspace_low=None, workspace_high=None)
    start_position, start_rotation, _, _ = tool_pose()
    box, max_tilt = np.array(args.calib_box), np.radians(args.calib_tilt_deg)
    path = args.out / f"{args.calibrate}.jsonl"
    meta = {"run_id": args.calibrate, "beta": args.beta, "seed": args.seed, "side": "sim", "noise_scale": scale,
            "rate_hz": RATE_HZ, "units": [UNIT_TRANSLATION, UNIT_ROTATION], "start_position": start_position.tolist(),
            "box": box.tolist(), "max_tilt_deg": args.calib_tilt_deg, "cell": args.cell}
    (args.out / f"{args.calibrate}.meta.json").write_text(json.dumps(meta, indent=2))
    with path.open("w") as log:
        for step in range(args.steps):
            p0, r0, _, _ = tool_pose()
            for axis in range(3):
                if p0[axis] < start_position[axis] - box[axis] + 0.01:
                    excitation.motion[axis] = abs(excitation.motion[axis])
                elif p0[axis] > start_position[axis] + box[axis] - 0.01:
                    excitation.motion[axis] = -abs(excitation.motion[axis])
            tilt = Rotation.from_matrix(r0 @ start_rotation.T).as_rotvec()
            if np.linalg.norm(tilt) > max_tilt - np.radians(2):
                for axis in range(3):
                    if excitation.motion[3 + axis] * tilt[axis] > 0:
                        excitation.motion[3 + axis] = -excitation.motion[3 + axis]
            action = excitation(np.zeros(7), step)
            action[6] = 1.0
            command(action[:6])
            p1, r1, _, _ = tool_pose()
            row = {"run_id": args.calibrate, "beta": args.beta, "step": step, "action": action.tolist(),
                   "state": p0.tolist() + Rotation.from_matrix(r0).as_euler("xyz").tolist() + [1.0],
                   "next_state": p1.tolist() + Rotation.from_matrix(r1).as_euler("xyz").tolist() + [1.0]}
            log.write(json.dumps(row) + "\n")
    print(f"[SIM2REAL] calibration {args.calibrate}: {args.steps} streamed commands -> {path}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
