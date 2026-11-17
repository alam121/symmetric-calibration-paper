"""Cross-embodiment action-interface transfer in Isaac: a policy trained on the Franka, run on the UR10e.

Both robots work in one task frame T (axis-aligned with the robot base, its origin on the work surface)
with the same object layout and start TCP pose; the policy sees and commands only T-frame quantities:

  observation  TCP pose, object pose, target pose (all in T), last gripper command, ticks since it changed
  action       6-D in [-1, 1] at 10 Hz (1 cm / 0.05 rad per unit) + binary gripper (open / close)

Each robot executes the action through its own native interface:

  franka   Isaac Lab's Franka IK setup: the TCP (panda_hand + 0.107 m) moved in the base frame, high-PD arm,
           parallel jaws open / closed
  ur10e    tool0 (22.5 cm behind the TCP: rotations also swing the TCP) moved in UR's "base" frame, which is
           base_link turned 180 deg about z; Delto F1 + F3 pinch with F2 curled, opened over 1 s on release

Tasks:  A  pick a 30 mm cylinder, drop it into a 12 cm tray (no wrist rotation needed)
        B  pick the 30 x 10 x 60 mm box (yaw -40..80 deg), stand it on the hexagon with its long side at 30 deg
        R  contactless, orientation-critical: TCP to 10 cm above the hexagon (+-2 cm) with yaw -60..60 deg,
           held within 1 cm / 5 deg for 1 s (B's learned policy failed on the Delto grasp, not the interface)

Modes:
  collect    --robot franka: the scripted expert (policy.py on the TCP) with DART noise -> demos (.npz)
  eval       --policy expert | PATH.pt (+ --map JSON: a_U = (a W_F + b_F - b_U) pinv(W_U))
  calibrate  task-free SmoothExcitation through the native interface; TCP pose logged in T

  unset CONDA_SHLVL; ~/miniconda3/envs/env_isaaclab/bin/python isaac/sim2real/xembody.py --robot franka --task B --mode eval --policy expert --episodes 5 --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot", choices=("franka", "ur10e"), required=True)
parser.add_argument("--task", choices=("A", "B", "R"), default="B")
parser.add_argument("--mode", choices=("collect", "eval", "calibrate", "pairs"), default="eval")
parser.add_argument("--policy", default="expert", help="expert, or a trained .pt (xembody_train.py)")
parser.add_argument("--map", type=Path, default=None, help="JSON {W_F, b_F, W_U, b_U}: interface translation")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dart", type=float, default=0.3, help="collect: std of the action noise executed (not recorded)")
parser.add_argument("--run_id", default=None, help="output name (default from robot / task / mode)")
parser.add_argument("--beta", type=float, default=0.85)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--match_magnitude", type=float, default=0.15)
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "results/xembody")
parser.add_argument("--only_episode", type=int, default=None)
parser.add_argument("--calib_centre", type=float, nargs=3, default=None,
                    help="calibrate: excite around this TCP position in T (default: the start pose)")
parser.add_argument("--ur_frame_yaw_deg", type=float, default=0.0,
                    help="UR10e: undocumented extra rotation of its command frame about z (an installation / mounting "
                    "setting), on top of the documented 180 deg base frame")
parser.add_argument("--analytic", action="store_true",
                    help="UR10e: map each action with the documented interface instead of a calibration: 180 deg "
                    "base frame, pivot compensation for a tool0 -> TCP offset of --analytic_offset, ideal gains")
parser.add_argument("--analytic_offset", type=float, default=None, help="assumed tool0 -> TCP [m] (default: the true one)")
parser.add_argument("--analytic_frame_yaw_deg", type=float, default=0.0, help="assumed extra frame rotation [deg]")
parser.add_argument("--map_schedule", type=Path, default=None,
                    help='JSON [{"map": PATH, "centre": [x, y, z]}, ...]: each tick uses the map calibrated nearest the TCP')
parser.add_argument("--env_shift_xy", type=float, nargs=2, default=None, help="layout shift dx dy [m] (overrides --env_shift)")
parser.add_argument("--env_shift", type=float, default=0.0,
                    help="environment shift [m]: the whole layout (object start, tray / hexagon, R target) moved this far "
                    "along +x+y (diagonal) away from the demos' region (they covered +-2 cm)")
parser.add_argument("--table_color", type=float, nargs=3, default=(0.72, 0.55, 0.36), help="scene change: table RGB")
parser.add_argument("--obj_color", type=float, nargs=3, default=(0.95, 0.95, 0.95), help="scene change: object RGB")
parser.add_argument("--tray_color", type=float, nargs=3, default=(0.2, 0.4, 0.8), help="scene change: tray RGB")
parser.add_argument("--light_intensity", type=float, default=2500.0, help="scene change: dome light intensity")
parser.add_argument("--distractors", type=int, default=0,
                    help="scene change: N visual-only (non-colliding) distractors on the table, clear of the task path")
parser.add_argument("--distractor_seed", type=int, default=0)
parser.add_argument("--obstacle", action="store_true",
                    help="scene change: a colliding wall (1 x 10 x 15 cm) between the start pose and the object")
parser.add_argument("--camera", action="store_true",
                    help="render a 64x64 wrist camera at the same pose relative to the TCP on both robots (10 cm above, "
                    "14 cm to the side, looking just past the pinch point); collect stores the images")
parser.add_argument("--cam_shift", type=float, nargs=6, default=None,
                    help="target camera = the source mount moved by dx dy dz [m] rx ry rz [deg] in its own frame; the "
                    "policy then sees the target camera through --obs_adapter")
parser.add_argument("--obs_adapter", choices=("none", "plane", "reproject", "latent"), default="none")
parser.add_argument("--record_pairs", type=Path, default=None,
                    help="eval with --cam_shift: save both camera views (+ depth, state, pose) at every step of the rollouts")
parser.add_argument("--record_steps", type=int, default=None, help="--record_pairs: only the first K steps of each episode")
parser.add_argument("--policy_view", choices=("target", "source"), default="target",
                    help="eval with --cam_shift: which camera drives the policy (source: the policy's own view, so the "
                    "recorded pairs follow successful source-policy states)")
parser.add_argument("--latent_adapter", type=Path, default=None, help="trained latent adapter (xembody_obs_train.py)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.cam_shift is not None or args.mode == "pairs":
    args.camera = True
if args.camera:
    args.enable_cameras = True
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
from isaaclab.sensors import CameraCfg  # noqa: E402
import xembody_obs as xo  # noqa: E402
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import policy as pol  # noqa: E402
from spec import CLIP, RATE_HZ, UNIT_ROTATION, UNIT_TRANSLATION  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_experiments.task_free_calibration.collect_calibration import SmoothExcitation  # noqa: E402

pol.GRASP_DEPTH = 0.0   # the expert works on the TCP directly (policy.py adds GRASP_DEPTH to tool0)

DT = 1.0 / 120.0
STEPS_PER_COMMAND = int(round(1.0 / (RATE_HZ * DT)))

# ---- the shared task (T frame: x, y on the work surface, z up; same axes as each robot's base) ----
BOX = (0.03, 0.01, 0.06)                     # task B object, standing on its 30 x 10 end
CYL = (0.015, 0.06)                          # task A object: radius, height
BOX_HOME, BOX_YAW = np.array([-0.06, 0.01]), (np.radians(-40), np.radians(80))
HEXAGON, HEX_R, HEX_H = np.array([0.066, -0.01]), 0.055, 0.038
TARGET_YAW = np.radians(30.0)
TRAY, TRAY_IN, TRAY_WALL = np.array([0.03, 0.13]), 0.12, (0.005, 0.03)
START_TCP = np.array([-0.17, 0.07, 0.175])
START_R = np.column_stack([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])   # TCP x along +x_T, z down

# ---- the two embodiments ----
UR_USD = os.environ.get("UR10E_USD", str(Path.home() / "isaac_ws/ur10e_delto/ur10e_delto.usd"))
UR_ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
UR_HOME = [-1.352599, -0.582120, 1.307726, 2.136123, -1.784902, -1.059150]
UR_GRIP = [f"gripper_f{f}m{m}_joint" for f in (1, 2, 3) for m in (1, 2, 3, 4)]
_OPEN = [0.4350, -0.1130, 2.0250, -0.3700, -1.1330, 0.0000, 1.6580, -0.0930, 0.3770, 0.1710, 2.0320, -0.4400]
_LAB_CLOSED = [0.3473, -0.0367, 1.9897, -0.0663, -1.0978, 0.0401, 2.0769, -0.0663, 0.3944, 0.1047, 2.1101, -0.0419]
_F2_CURLED = [-1.1330, 0.0, 0.85, 2.02]
UR_OPEN = _OPEN[:4] + _F2_CURLED + _OPEN[8:]
UR_CLOSED = _LAB_CLOSED[:4] + _F2_CURLED + _LAB_CLOSED[8:]
UR_TCP = 0.245                               # tool0 -> F1 / F3 pinch centre (F1 closes ~32 mm below 0.225, F3 ~3 above)
UR_T = np.array([0.52, -1.02, -0.329])       # T origin in base_link: the measured lab cell
UR_CMD = Rotation.from_euler("z", np.pi).as_matrix()   # UR "base" frame in base_link

FR_ARM = [f"panda_joint{i}" for i in range(1, 8)]
FR_FINGERS = ["panda_finger_joint1", "panda_finger_joint2"]
FR_TCP = 0.107                               # panda_hand -> fingertip centre (Isaac Lab's IK offset)
FR_T = np.array([0.6, 0.0, 0.0])            # at 0.5 the start pose folded the elbow (j4) to its limit


def shift():
    if args.env_shift_xy is not None:
        return np.array(args.env_shift_xy)
    return args.env_shift * np.array([1.0, 1.0]) / np.sqrt(2.0)


def tray_walls(origin):
    c = origin + np.array([*(TRAY + shift()), 0.0]); t, h = TRAY_WALL; s = TRAY_IN
    return [(c + [0, (s + t) / 2, h / 2], (s + 2 * t, t, h)), (c + [0, -(s + t) / 2, h / 2], (s + 2 * t, t, h)),
            (c + [(s + t) / 2, 0, h / 2], (t, s, h)), (c + [-(s + t) / 2, 0, h / 2], (t, s, h))]


def add_scene_changes(cfg, origin):
    """Distractors: cylinders and cubes the size of the task object in random colours, at random table spots at least
    7 cm from the object's start region and clear of the tray and the start-to-object path; visual only, so they can
    change what the policy sees but not what physically happens. Obstacle: one colliding wall across that path."""
    home, tray = BOX_HOME + shift(), TRAY + shift()
    start = np.array([-0.17, 0.07])
    rng = np.random.default_rng(args.distractor_seed)
    placed = []
    while len(placed) < args.distractors:
        p = rng.uniform([-0.20, -0.10], [0.12, 0.22])
        seg = home - start
        on_path = np.linalg.norm(p - (start + np.clip((p - start) @ seg / (seg @ seg), 0, 1) * seg)) < 0.05
        if (np.linalg.norm(p - home) < 0.07 or np.abs(p - tray).max() < TRAY_IN / 2 + 0.04 or on_path
                or any(np.linalg.norm(p - q) < 0.05 for q in placed)):
            continue
        placed.append(p)
    for i, p in enumerate(placed):
        colour = tuple(float(c) for c in rng.uniform(0.05, 0.95, 3))
        shape = (sim_utils.CylinderCfg(radius=CYL[0], height=CYL[1]) if i % 2 == 0 else sim_utils.CuboidCfg(size=(0.03, 0.03, 0.03)))
        shape.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=colour)
        h = CYL[1] if i % 2 == 0 else 0.03
        setattr(cfg, f"distractor{i}", AssetBaseCfg(prim_path=f"/World/Distractor{i}", spawn=shape,
                                                    init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(origin + [*p, h / 2]))))
    if args.obstacle:
        mid = start + 0.45 * (home - start)
        yaw = np.arctan2(*(home - start)[::-1])
        q = Rotation.from_euler("z", yaw).as_quat()
        cfg.obstacle = AssetBaseCfg(prim_path="/World/Obstacle", spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.10, 0.15), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.55))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(origin + [*mid, 0.075]), rot=(q[3], q[0], q[1], q[2])))


def scene_cfg(origin):
    @configclass
    class Cfg(InteractiveSceneCfg):
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=args.light_intensity))
    cfg = Cfg(num_envs=1, env_spacing=2.0)
    cfg.table = AssetBaseCfg(prim_path="/World/Table", spawn=sim_utils.CuboidCfg(
        size=(1.2, 0.8, 0.05), collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(args.table_color))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(origin[0], origin[1], origin[2] - 0.025)))
    if args.task in ("B", "R"):
        cfg.hexagon = AssetBaseCfg(prim_path="/World/Hexagon", spawn=sim_utils.CylinderCfg(
            radius=HEX_R, height=HEX_H, collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.75, 0.6))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(origin + [*(HEXAGON + shift()), HEX_H / 2])))
        spawn = sim_utils.CuboidCfg(size=BOX)
        rest = BOX[2] / 2
    else:
        for i, (p, size) in enumerate(tray_walls(origin)):
            setattr(cfg, f"wall{i}", AssetBaseCfg(prim_path=f"/World/Tray/wall{i}", spawn=sim_utils.CuboidCfg(
                size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(args.tray_color))),
                init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(p))))
        spawn = sim_utils.CylinderCfg(radius=CYL[0], height=CYL[1])
        rest = CYL[1] / 2
    spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.5)
    spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.015)
    spawn.collision_props = sim_utils.CollisionPropertiesCfg()
    spawn.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9)
    spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(args.obj_color))
    add_scene_changes(cfg, origin)
    cfg.obj = RigidObjectCfg(prim_path="/World/Object", spawn=spawn,
                             init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(origin + [*BOX_HOME, rest + 0.001])))
    if args.robot == "ur10e":
        cfg.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=UR_USD, rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            actuators={"arm": ImplicitActuatorCfg(joint_names_expr=UR_ARM, stiffness=1200.0, damping=120.0,
                                                  effort_limit_sim=10000.0, velocity_limit_sim=3.0),
                       "gripper": ImplicitActuatorCfg(joint_names_expr=["gripper_.*"], stiffness=40.0, damping=2.0,
                                                      effort_limit_sim=2.0, velocity_limit_sim=3.0)})
    else:
        cfg.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # jaws: 8 N -- at 20-200 N the pads squeezing the box's narrow end faces squirted the light box out
        cfg.robot.actuators["panda_hand"] = cfg.robot.actuators["panda_hand"].replace(effort_limit_sim=float(os.environ.get("FR_GRIP_N", "8")))
    if args.camera:
        # Same pose relative to the TCP on both robots: 10 cm back along the tool axis and 14 cm along tool x (clear
        # of both hands), aimed 3 cm past the pinch point (ROS optical convention: z forward, y down in the image).
        off = UR_TCP if args.robot == "ur10e" else FR_TCP
        ee = "tool0" if args.robot == "ur10e" else "panda_hand"
        pos, rot = xo.source_mount(off)
        lens = sim_utils.PinholeCameraCfg(focal_length=xo.FOCAL, horizontal_aperture=xo.APERTURE, clipping_range=(0.01, 5.0))
        cfg.cam = CameraCfg(prim_path=f"{{ENV_REGEX_NS}}/Robot/{ee}/WristCam", height=64, width=64,
                            data_types=["rgb", "distance_to_image_plane"] if args.cam_shift is not None else ["rgb"],
                            spawn=lens, offset=CameraCfg.OffsetCfg(pos=tuple(pos), rot=xo.quat_wxyz(rot), convention="ros"))
        if args.cam_shift is not None:   # the target camera: same lens, moved mount, with depth for --obs_adapter reproject
            pos_t, rot_t = xo.target_mount(off, args.cam_shift)
            cfg.cam_t = CameraCfg(prim_path=f"{{ENV_REGEX_NS}}/Robot/{ee}/TargetCam", height=64, width=64,
                                  data_types=["rgb", "distance_to_image_plane"], spawn=lens,
                                  offset=CameraCfg.OffsetCfg(pos=tuple(pos_t), rot=xo.quat_wxyz(rot_t), convention="ros"))
    return cfg, rest


class Embodiment:
    """One robot: TCP pose in T, its native relative-Cartesian interface, its binary gripper."""

    def __init__(self, sim, scene):
        self.sim, self.scene = sim, scene
        self.robot = scene["robot"]
        self.dev = sim.device
        ur = args.robot == "ur10e"
        self.origin = UR_T if ur else FR_T
        self.arm = [self.robot.joint_names.index(n) for n in (UR_ARM if ur else FR_ARM)]
        self.fingers = [self.robot.joint_names.index(n) for n in (UR_GRIP if ur else FR_FINGERS)]
        self.ee = self.robot.body_names.index("tool0" if ur else "panda_hand")
        self.offset = UR_TCP if ur else FR_TCP
        if ur:
            limits = self.robot.data.joint_pos_limits.clone()
            limits[:, self.fingers, 0] = torch.clamp(limits[:, self.fingers, 0], max=-1.0)
            limits[:, self.fingers, 1] = torch.clamp(limits[:, self.fingers, 1], min=2.8)
            self.robot.write_joint_position_limit_to_sim(limits)
        self.open_q = UR_OPEN if ur else [0.04, 0.04]
        self.closed_q = UR_CLOSED if ur else [0.0, 0.0]
        self.ik_rel = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True,
                                                                           ik_method="dls"), num_envs=1, device=self.dev)
        self.ik_abs = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                                                           ik_method="dls"), num_envs=1, device=self.dev)
        home = UR_HOME if ur else self.robot.data.default_joint_pos[0, self.arm].tolist()
        self.arm_target = torch.tensor([home], device=self.dev)
        self.grip_target = torch.tensor([self.open_q], device=self.dev)
        self.start = None
        self.release, self.closed = None, False   # UR: ticks into the eased opening

    # -- state --
    def ee_pose(self):
        p, q = subtract_frame_transforms(self.robot.data.root_pos_w, self.robot.data.root_quat_w,
                                         self.robot.data.body_pos_w[:, self.ee], self.robot.data.body_quat_w[:, self.ee])
        w, x, y, z = q[0].tolist()
        return p[0].cpu().numpy(), Rotation.from_quat([x, y, z, w]).as_matrix(), p, q

    def tcp(self):
        """TCP position and rotation in T."""
        p, r, _, _ = self.ee_pose()
        return p + r @ [0.0, 0.0, self.offset] - self.origin, r

    def jacobian(self):
        return self.robot.root_physx_view.get_jacobians()[:, self.ee - 1, :, self.arm]

    def physics(self, n):
        for _ in range(n):
            self.robot.set_joint_position_target(self.arm_target, joint_ids=self.arm)
            self.robot.set_joint_position_target(self.grip_target, joint_ids=self.fingers)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(DT)

    # -- the native interface --
    def command(self, action):
        a = np.clip(np.asarray(action, float), -CLIP, CLIP)
        dp, dr = a[:3] * UNIT_TRANSLATION, a[3:] * UNIT_ROTATION
        p, r, pt, qt = self.ee_pose()
        if args.robot == "ur10e":
            # tool0, in UR's "base" frame (base_link turned 180 deg about z)
            cmd = Rotation.from_euler("z", np.pi + np.radians(args.ur_frame_yaw_deg)).as_matrix()
            delta = np.concatenate([cmd @ dp, cmd @ dr])
            self.ik_rel.set_command(torch.tensor([delta], device=self.dev, dtype=torch.float32), pt, qt)
            self.arm_target = self.ik_rel.compute(pt, qt, self.jacobian(), self.robot.data.joint_pos[:, self.arm])
        else:
            # the TCP, in the base frame: target TCP pose -> panda_hand pose -> absolute IK
            tcp = p + r @ [0.0, 0.0, FR_TCP]
            r_goal = Rotation.from_rotvec(dr).as_matrix() @ r
            p_goal = tcp + dp - r_goal @ [0.0, 0.0, FR_TCP]
            q = Rotation.from_matrix(r_goal).as_quat()
            self.ik_abs.set_command(torch.tensor([[*p_goal, q[3], q[0], q[1], q[2]]], device=self.dev, dtype=torch.float32))
            self.arm_target = self.ik_abs.compute(pt, qt, self.jacobian(), self.robot.data.joint_pos[:, self.arm])
        self.physics(STEPS_PER_COMMAND)

    def gripper(self, close):
        if close:
            self.release, self.closed, q = None, True, self.closed_q
        else:
            if args.robot == "ur10e" and self.closed:   # the Delto opens over 1 s: snapping it tipped the standing box
                self.release = 0
            self.closed = False
            if self.release is not None and self.release < 10:
                self.release += 1
                f = self.release / 10.0
                q = (np.array(self.closed_q) + f * (np.array(self.open_q) - np.array(self.closed_q))).tolist()
            else:
                q = self.open_q
        self.grip_target = torch.tensor([q], device=self.dev)

    # -- start pose (teleported, solved once) --
    def start_joints(self):
        if self.start is None:
            joints = self.robot.data.default_joint_pos.clone()
            joints[:, self.arm] = self.arm_target
            for _ in range(400):
                self.robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
                self.sim.step(); self.scene.update(DT)
                p, r, pt, qt = self.ee_pose()
                ee_goal = self.origin + START_TCP - START_R @ [0.0, 0.0, self.offset]
                q = Rotation.from_matrix(START_R).as_quat()
                self.ik_abs.set_command(torch.tensor([[*ee_goal, q[3], q[0], q[1], q[2]]], device=self.dev, dtype=torch.float32))
                step = self.ik_abs.compute(pt, qt, self.jacobian(), joints[:, self.arm]) - joints[:, self.arm]
                joints[:, self.arm] += torch.clamp(step, -0.05, 0.05)
            self.start = joints[:, self.arm].clone()
            tcp, r = self.tcp()
            print(f"[XEMB] {args.robot} start: TCP {np.round(tcp, 3).tolist()} in T (goal {START_TCP.tolist()}), "
                  f"{np.degrees(np.linalg.norm(Rotation.from_matrix(r @ START_R.T).as_rotvec())):.1f} deg off; joints "
                  f"{np.round(self.start[0].tolist(), 3).tolist()} (limits {np.round(self.robot.data.joint_pos_limits[0, self.arm].tolist(), 2).tolist()})", flush=True)
        return self.start

    def reset(self, start_noise=None):
        joints = self.robot.data.default_joint_pos.clone()
        joints[:, self.arm] = self.start_joints()
        joints[:, self.fingers] = torch.tensor([self.open_q], device=self.dev)
        self.robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        self.arm_target, self.grip_target = joints[:, self.arm].clone(), torch.tensor([self.open_q], device=self.dev)
        self.release, self.closed = None, False


def rot6(r):
    return np.concatenate([r[:, 0], r[:, 1]])


def observe(tcp, r, obj, obj_r, target, grip_cmd, since):
    """The policy's input, all in T: embodiment-free by construction."""
    return np.concatenate([tcp, rot6(r), obj, rot6(obj_r), target[:3], [np.sin(target[3]), np.cos(target[3])],
                           [float(grip_cmd), min(since, 30) / 10.0]]).astype(np.float32)


class Expert:
    """policy.py's pick-and-place on the TCP. A: no wrist rotation (cylinder, yaw-free drop into the tray).
    B: F1/F3 (or the jaws) across the box's 30 mm side at its centre (each TCP is its hand's pinch centre),
    stood on the hexagon."""

    def __init__(self, grip_above=-0.005):
        if args.task == "A":
            self.task = pol.TaskB(hover=0.10, lift=0.12, half_height=CYL[1] / 2, grip_above=0.0, grip_along=0.0,
                                  place_above=0.03, grip_yaw_deg=0.0)
        else:
            # aim 5 mm below the centre: the cloned policy closes a few mm higher than the expert, fine for the
            # Franka's tall pads, but 8 mm high the Delto's F3 passes over the box top and F1 knocks it over.
            # (A 2 mm closing tolerance instead made the demos hover and the clone never closed.)
            # Collecting: the grasp height varies (-12..0 mm) so the box's place in the hand varies and the clone must
            # place by the box's own pose -- the Delto holds it ~7 mm higher than the Franka's jaws, and a clone
            # that timed the release by TCP height pressed it onto the hexagon and never let go.
            self.task = pol.TaskB(hover=0.10, lift=0.12, half_height=BOX[2] / 2, grip_above=grip_above, grip_along=0.0,
                                  place_above=-0.004, grip_yaw_deg=90.0)

    def act(self, tcp, r, obj, obj_r, target):
        if args.task == "R":
            goal_r = Rotation.from_euler("z", target[3]).as_matrix() @ START_R
            a, reached = pol.step_towards(tcp, r, target[:3], goal_r)
            return a, False, False
        objects = {"block": (obj, obj_r), "slot": (target[:3], target[3])}
        if args.task == "A":   # the cylinder's yaw is meaningless: grasp and place with the start orientation
            objects = {"block": (obj, np.eye(3)), "slot": (target[:3], 0.0)}
        a, g = self.task.act(tcp, r, objects)
        return a, (g == "close"), self.task.phase == "done"


class Learned:
    def __init__(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        from xembody_train import REPS, Net, ImageNet
        self.image, self.mask_bottom = bool(ck.get("image")), int(ck.get("mask_bottom", 0))
        self.net = ImageNet(ck["in_dim"]) if self.image else Net(ck["in_dim"], ck.get("hidden", 256))
        self.net.load_state_dict(ck["state"]); self.net.eval()
        rep = ck.get("rep") or ("planar" if ck.get("planar") else "tcp_full" if ck.get("tcp_full") else "full")
        self.prep = REPS[rep]
        self.adapter = None
        if args.latent_adapter is not None:   # on the raw target image (--obs_adapter latent) or after reprojection
            ad = torch.load(args.latent_adapter, map_location="cpu", weights_only=False)
            self.adapter = xo.latent_adapter(); self.adapter.load_state_dict(ad["state"]); self.adapter.eval()
        self.mu, self.sd = torch.tensor(ck["mu"]), torch.tensor(ck["sd"])

    def act(self, obs, img=None):
        with torch.no_grad():
            x = (torch.tensor(self.prep(obs))[None] - self.mu) / self.sd
            if self.image:
                im = torch.tensor(img).permute(2, 0, 1)[None].float() / 255.0
                if self.mask_bottom:
                    im[:, :, -self.mask_bottom:, :] = 0.0
                if self.adapter is not None:
                    f = self.adapter(self.net.cnn(im))
                    o7 = self.net.head(torch.cat([f, x], dim=1))
                    out = torch.cat([torch.tanh(o7[:, :6]), o7[:, 6:]], dim=1)[0]
                else:
                    out = self.net(x, im)[0]
            else:
                out = self.net(x)[0]
        return out[:6].numpy(), bool(out[6] > 0)


def analytic_map(a, r_tcp):
    """The documented UR10e interface, inverted: the TCP displacement the source action intends, moved to tool0
    (subtract the swing a rotation about tool0 gives the TCP), expressed in the documented command frame."""
    off = UR_TCP if args.analytic_offset is None else args.analytic_offset
    dp, dr = a[:3] * UNIT_TRANSLATION, a[3:] * UNIT_ROTATION
    dp_tool0 = dp - np.cross(dr, r_tcp @ np.array([0.0, 0.0, off]))
    cmd = Rotation.from_euler("z", np.pi + np.radians(args.analytic_frame_yaw_deg)).as_matrix()
    return np.concatenate([cmd.T @ dp_tool0 / UNIT_TRANSLATION, cmd.T @ dr / UNIT_ROTATION])


def load_map(path):
    """a_U = (a_F W_F + b_F - b_U) pinv(W_U): both fits map an action to the TCP displacement in T."""
    m = json.loads(path.read_text())
    WF, bF, WU, bU = (np.array(m[k]) for k in ("W_F", "b_F", "W_U", "b_U"))
    pinv = np.linalg.pinv(WU)
    return lambda a: (a @ WF + bF - bU) @ pinv


def load_schedule(path):
    """Gain scheduling over local linear maps: the map whose calibration centre is nearest the TCP."""
    entries = json.loads(path.read_text())
    maps = [(np.array(e["centre"]), load_map(Path(e["map"]))) for e in entries]
    return lambda a, tcp: min(maps, key=lambda m: np.linalg.norm(tcp - m[0]))[1](a)


def episode_layout(rng):
    if args.task == "R":   # the box is parked out of the way; the target is a TCP pose
        return np.array([0.30, 0.30]), 0.0, np.array([*(HEXAGON + shift() + rng.uniform(-0.02, 0.02, 2)), 0.10,
                                                       float(rng.uniform(np.radians(-60), np.radians(60)))])
    obj = BOX_HOME + shift() + rng.uniform(-0.02, 0.02, 2)
    yaw = float(rng.uniform(*BOX_YAW)) if args.task == "B" else 0.0
    if args.task == "B":
        target = np.array([*(HEXAGON + shift() + rng.uniform(-0.01, 0.01, 2)), HEX_H, TARGET_YAW])
    else:
        target = np.array([*(TRAY + shift()), 0.0, 0.0])
    return obj, yaw, target


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device,
                                                              render_interval=int(os.environ.get("XEMB_RENDER_INTERVAL", "1"))))
    origin = UR_T if args.robot == "ur10e" else FR_T
    cfg, rest = scene_cfg(origin)
    scene = InteractiveScene(cfg)
    sim.reset()
    robot = Embodiment(sim, scene)
    obj = scene["obj"]
    dev = sim.device
    args.out.mkdir(parents=True, exist_ok=True)

    def place_object(xy, yaw):
        pose = torch.zeros((1, 7), device=dev)
        pose[0, :3] = torch.tensor([*(origin[:2] + xy), origin[2] + rest + 0.001], device=dev)
        q = Rotation.from_euler("z", yaw).as_quat()
        pose[0, 3:] = torch.tensor([q[3], q[0], q[1], q[2]], device=dev)
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))

    def object_pose():
        p = obj.data.root_pos_w[0].cpu().numpy() - origin
        w, x, y, z = obj.data.root_quat_w[0].tolist()
        return p, Rotation.from_quat([x, y, z, w]).as_matrix()

    tcp_off = UR_TCP if args.robot == "ur10e" else FR_TCP
    mounts = (xo.source_mount(tcp_off), xo.target_mount(tcp_off, args.cam_shift) if args.cam_shift is not None else None)

    def cameras_now():
        p, r, _, _ = robot.ee_pose()
        return xo.camera_in_world(p, r, mounts[0]), xo.camera_in_world(p, r, mounts[1])

    def target_view():
        """The target camera's image, through the chosen observation adapter."""
        rgb = scene["cam_t"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        if args.obs_adapter in ("none", "latent"):
            return rgb   # latent: the adapter acts inside the policy, on the encoder feature
        src, tgt = cameras_now()
        if args.obs_adapter == "plane":
            return xo.plane_warp(rgb, src, tgt, origin[2])
        depth = scene["cam_t"].data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
        return xo.reproject(rgb, depth, src, tgt)

    if args.mode == "pairs":
        return collect_pairs(robot, scene, place_object, object_pose, cameras_now, origin)

    if args.mode == "calibrate":
        if args.calib_centre is not None:
            global START_TCP
            START_TCP = np.array(args.calib_centre)
        return calibrate(robot, place_object)

    rng = np.random.default_rng(args.seed)
    learned = Learned(args.policy) if args.policy != "expert" else None
    translate = load_map(args.map) if args.map else None
    schedule = load_schedule(args.map_schedule) if args.map_schedule else None
    demos, results = {"obs": [], "act": [], "grip": [], "episode": []}, []
    pairs_rec = {}
    if args.camera:
        demos["img"] = []
    for ep in range(args.episodes):
        xy, yaw, target = episode_layout(rng)
        if args.only_episode is not None and ep != args.only_episode:
            continue
        robot.reset()
        place_object(xy, yaw)
        robot.physics(60)
        lo, hi = (float(v) for v in os.environ.get("XEMB_GRASP_RANGE", "-0.012,0.0").split(","))
        expert = Expert(float(rng.uniform(lo, hi)) if args.mode == "collect" else -0.005)
        grip_cmd, since, held_steps, ep_obs = False, 30, 0, []
        steps = int((20 if args.task == "R" else 40) * RATE_HZ)
        for k in range(steps):
            tcp, r = robot.tcp()
            o, o_r = object_pose()
            obs = observe(tcp, r, o, o_r, target, grip_cmd, since)
            img = scene["cam"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8) if args.camera else None
            if img is not None and k in (5, 40) and os.environ.get("XEMB_SAVE_CAM"):
                import cv2
                cv2.imwrite(f"{os.environ['XEMB_SAVE_CAM']}_{args.robot}_ep{ep}_k{k}.png", cv2.resize(img[..., ::-1], (256, 256), interpolation=cv2.INTER_NEAREST))
            policy_img = target_view() if args.cam_shift is not None and args.policy_view == "target" else img
            if args.record_pairs is not None and (args.record_steps is None or k < args.record_steps):
                p_ee, r_ee, _, _ = robot.ee_pose()
                for key, val in (("src", img), ("src_depth", scene["cam"].data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy().astype(np.float16)), ("tgt", scene["cam_t"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)),
                                 ("tgt_depth", scene["cam_t"].data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy().astype(np.float16)),
                                 ("obs", obs), ("ee_p", p_ee), ("ee_r", r_ee), ("episode", ep), ("step", k)):
                    pairs_rec.setdefault(key, []).append(val)
            a_exp, g_exp, done = expert.act(tcp, r, o, o_r, target)
            if learned is not None:
                a, g = learned.act(obs, policy_img)
                done = False
            else:
                a, g = a_exp, g_exp
            if args.mode == "collect":
                demos["obs"].append(obs); demos["act"].append(np.clip(a_exp, -CLIP, CLIP))
                demos["grip"].append(float(g_exp)); demos["episode"].append(ep)
                if args.camera:
                    demos["img"].append(img)
                a = np.clip(a + rng.normal(0.0, args.dart, 6), -CLIP, CLIP)   # DART: executed, not recorded
            if translate is not None:
                a = translate(np.clip(a, -CLIP, CLIP))
            if schedule is not None:
                a = schedule(np.clip(a, -CLIP, CLIP), tcp)
            if args.analytic:
                a = analytic_map(np.clip(a, -CLIP, CLIP), r)
            if os.environ.get("XEMB_ORACLE") and args.robot == "ur10e":   # feasibility check only: undo the frame flip
                a = np.concatenate([UR_CMD.T @ a[:3], UR_CMD.T @ a[3:]])
            since = 0 if g != grip_cmd else since + 1
            grip_cmd = g
            robot.gripper(g)
            robot.command(a)
            if os.environ.get("XEMB_DEBUG") == "2" and expert.task.phase in ("descend", "close"):
                names = [n for n in robot.robot.body_names if "finger" in n or "tip" in n]
                rel = {n: np.round(Rotation.from_matrix(o_r).inv().apply(
                    robot.robot.data.body_pos_w[0, robot.robot.body_names.index(n)].cpu().numpy() - origin - o) * 1000).astype(int).tolist()
                    for n in names if "franka" == args.robot[:6] or "long_tip" in n}
                print(f"[FINGERS] t={k / RATE_HZ:4.1f} {expert.task.phase} (box frame, mm) {rel}", flush=True)
            if os.environ.get("XEMB_DEBUG") and k % 10 == 0:
                print(f"[DEBUG] t={k / RATE_HZ:4.1f} phase={expert.task.phase} tcp={np.round(tcp, 3).tolist()} "
                      f"obj={np.round(o, 3).tolist()} a={np.round(a, 2).tolist()} grip={g}", flush=True)
            if args.task == "R":
                tcp1, r1 = robot.tcp()
                held_steps = held_steps + 1 if reach_error(tcp1, r1, target)[0] else 0
                if held_steps >= RATE_HZ:
                    break
                continue
            if done or (learned is not None and k > 50 and not g and since > 40 and judge(object_pose, target)[0]):
                break
        if args.task == "R":
            tcp1, r1 = robot.tcp()
            ok, pos_mm, yaw_deg = reach_error(tcp1, r1, target)
            success, detail = held_steps >= RATE_HZ, {"pos_mm": round(pos_mm, 1), "yaw_err_deg": round(yaw_deg, 1),
                                                      "tilt_deg": round(float(np.degrees(np.arccos(np.clip(-r1[2, 2], -1, 1)))), 1)}
        else:
            robot.physics(60)
            success, detail = judge(object_pose, target)
        results.append({"episode": ep, "success": success, "steps": k + 1, **detail})
        print(f"[XEMB] {args.robot} task {args.task} ep {ep}: {'success' if success else 'failure'} {detail} ({k + 1} steps)", flush=True)
    rate = float(np.mean([r["success"] for r in results])) if results else 0.0
    name = args.run_id or f"{args.robot}_task{args.task}_{args.mode}"
    if args.mode == "collect":
        np.savez(args.out / f"{name}.npz", **{k: np.array(v) for k, v in demos.items()},
                 success=np.array([r["success"] for r in results]))
    if args.record_pairs is not None:
        np.savez(args.record_pairs, **{k: np.array(v) for k, v in pairs_rec.items()}, cam_shift=np.array(args.cam_shift),
                 plane_z=np.array(origin[2]), success=np.array([r["success"] for r in results]))
    (args.out / f"{name}.json").write_text(json.dumps({"robot": args.robot, "task": args.task, "policy": args.policy,
                                                       "map": str(args.map), "success_rate": rate, "episodes": results}, indent=1))
    print(f"[XEMB] {name}: {rate:.0%} of {len(results)}", flush=True)


def collect_pairs(robot, scene, place_object, object_pose, cameras_now, origin):
    """Task-free paired observations: the object dropped at random, the hand driven through random waypoints with a
    random gripper state, both cameras recorded every tick. No task, reward or demonstration is involved."""
    rng = np.random.default_rng(args.seed)
    rec = {k: [] for k in ("src", "tgt", "tgt_depth", "obs", "ee_p", "ee_r", "episode")}
    lo, hi = np.array([-0.20, -0.06, 0.035]), np.array([0.12, 0.20, 0.20])
    for ep in range(args.episodes):
        robot.reset()
        xy = BOX_HOME + rng.uniform(-0.04, 0.04, 2)
        place_object(xy, 0.0)
        robot.physics(60)
        target = np.array([*TRAY, 0.0, 0.0])
        grip, since = False, 30
        for w in range(5):
            goal = rng.uniform(lo, hi)
            goal_r = Rotation.from_euler("z", rng.uniform(np.radians(-20), np.radians(20))).as_matrix() @ START_R
            g_new = bool(rng.random() < 0.4)
            for k in range(25):
                tcp, r = robot.tcp()
                o, o_r = object_pose()
                since = 0 if g_new != grip else since + 1
                grip = g_new
                p_ee, r_ee, _, _ = robot.ee_pose()
                rec["src"].append(scene["cam"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
                rec["tgt"].append(scene["cam_t"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
                rec["tgt_depth"].append(scene["cam_t"].data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy().astype(np.float16))
                rec["obs"].append(observe(tcp, r, o, o_r, target, grip, since))
                rec["ee_p"].append(p_ee); rec["ee_r"].append(r_ee); rec["episode"].append(ep)
                a, _ = pol.step_towards(tcp, r, goal, goal_r)
                robot.gripper(grip)
                robot.command(a)
        print(f"[XEMB] pairs ep {ep}: {len(rec['obs'])} pairs so far", flush=True)
    name = args.run_id or f"{args.robot}_pairs"
    np.savez(args.out / f"{name}.npz", **{k: np.array(v) for k, v in rec.items()}, cam_shift=np.array(args.cam_shift),
             plane_z=np.array(origin[2]))
    print(f"[XEMB] {name}: {len(rec['obs'])} paired observations", flush=True)


def reach_error(tcp, r, target):
    """Task R: TCP within 1 cm of the target point and its x axis within 5 deg of the target yaw."""
    pos = float(np.linalg.norm(tcp - target[:3]))
    yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    yaw_err = float(np.degrees(abs((yaw - target[3] + np.pi) % (2 * np.pi) - np.pi)))
    return pos < 0.01 and yaw_err < 5.0, pos * 1000, yaw_err


def judge(object_pose, target):
    p, r = object_pose()
    tilt = float(np.degrees(np.arccos(np.clip(r[2, 2], -1, 1))))
    if args.task == "A":
        # inside the tray's footprint, resting on its floor (standing: centre at CYL/2; lying: at the radius)
        tray = TRAY + shift()
        inside = bool(np.all(np.abs(p[:2] - tray) < TRAY_IN / 2 - CYL[0] + 0.005) and p[2] < CYL[1] / 2 + 0.01)
        return inside, {"obj_mm": np.round((p[:2] - tray) * 1000).tolist(), "tilt_deg": round(tilt, 1)}
    off = float(np.linalg.norm(p[:2] - target[:2]))
    yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    yaw_err = float(np.degrees(abs((yaw - target[3] + np.pi / 2) % np.pi - np.pi / 2)))
    standing = tilt < 15 and abs(p[2] - (HEX_H + BOX[2] / 2)) < 0.01
    return bool(standing and off < HEX_R - 0.01 and yaw_err < 10), {
        "off_mm": round(off * 1000, 1), "yaw_err_deg": round(yaw_err, 1), "tilt_deg": round(tilt, 1)}


def calibrate(robot, place_object):
    """Task-free: the paper's SmoothExcitation through the native interface, the TCP pose logged in T."""
    robot.reset()
    place_object(np.array([0.30, 0.30]), 0.0)   # out of the way
    robot.physics(60)
    rng = np.random.default_rng(args.seed)
    stationary = np.sqrt((1 - args.beta) / (1 + args.beta))
    scale = args.match_magnitude / (stationary * 0.28)
    exc = SmoothExcitation(rng, args.beta, 0.28 * scale, 0.18 * scale, gripper_period=10 ** 9,
                           workspace_low=None, workspace_high=None)
    p_start, r_start = robot.tcp()
    box, max_tilt = np.array([0.05, 0.05, 0.05]), np.radians(10.0)
    name = args.run_id or f"{args.robot}_calib_seed{args.seed}"
    meta = {"run_id": name, "robot": args.robot, "beta": args.beta, "seed": args.seed, "noise_scale": scale,
            "rate_hz": RATE_HZ, "units": [UNIT_TRANSLATION, UNIT_ROTATION], "frame": "T (task frame), TCP",
            "box": box.tolist(), "max_tilt_deg": 10.0}
    (args.out / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    resets = 0
    with (args.out / f"{name}.jsonl").open("w") as log:
        step = 0
        while step < args.steps:
            p0, r0 = robot.tcp()
            tilt = np.linalg.norm(Rotation.from_matrix(r0 @ r_start.T).as_rotvec())
            if np.any(np.abs(p0 - p_start) > box) or tilt > max_tilt:
                # Back to the start pose, the transition dropped. Reflecting the excitation per axis assumed the
                # command axes are the task-frame axes; on the UR10e (x, y, rx, ry reversed) that pushed it further
                # out (33 cm, 158 deg). Homing needs no knowledge of the interface being identified.
                robot.reset(); robot.physics(30)
                exc.motion[:] = 0.0
                resets += 1
                continue
            a = exc(np.zeros(7), step)
            a[6] = 1.0
            robot.command(a[:6])
            p1, r1 = robot.tcp()
            log.write(json.dumps({"run_id": name, "beta": args.beta, "step": step, "action": a.tolist(),
                                  "state": p0.tolist() + Rotation.from_matrix(r0).as_euler("xyz").tolist() + [1.0],
                                  "next_state": p1.tolist() + Rotation.from_matrix(r1).as_euler("xyz").tolist() + [1.0]}) + "\n")
            step += 1
    meta["resets"] = resets
    (args.out / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[XEMB] calibration {name}: {args.steps} commands, {resets} homing resets", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
