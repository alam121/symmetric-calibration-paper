"""State-based policy for the sim-to-real study: object pose + robot state -> relative Cartesian action.

Pure numpy: the same code drives the Isaac interface (tasks_sim.py) and the UR10e. Inputs are the
tool0 pose and the object poses (sim state, or AprilTag / jig on the robot); outputs follow
spec.py (6-D action in [-1, 1] at RATE_HZ) plus a gripper command (open / close).

  Task A (orientation tolerant): bring the Delto grasp point to a marked point (any orientation).
  Task B (orientation critical): top grasp of the 100x30x80 block, F1 + F3 pinching across its
  30 mm width (F2 stays open), lift, turn to the receptacle's yaw, lower it in, release, retreat.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from spec import BLOCK_SIZE, CLIP, GRASP_ABOVE_CENTRE, GRASP_DEPTH, UNIT_ROTATION, UNIT_TRANSLATION

GAIN = 0.6            # fraction of the remaining error commanded per step (the interface lags)
POS_TOL, ROT_TOL = 0.005, 0.03
DOWN = np.array([0.0, 0.0, -1.0])


def tool_target(grasp_point: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """tool0 position that puts the grasp point at `grasp_point` with tool orientation `rotation`."""
    return grasp_point - rotation @ np.array([0.0, 0.0, GRASP_DEPTH])


def top_grasp_rotation(yaw: float, current: np.ndarray) -> np.ndarray:
    """Tool z down, tool x along the block's LONG side: the 100x30 footprint then fits inside the
    open Delto (fingertips at F1 (6,-31), F2 (59,6), F3 (4,33) mm in tool x/y) and F1 + F3 pinch
    across the 30 mm width. Of the two equivalent yaws, the one nearer the current tool x."""
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    if np.dot(x, current[:, 0]) < 0:
        x = -x
    return np.column_stack([x, np.cross(DOWN, x), DOWN])


def step_towards(position, rotation, target_position, target_rotation, pos_tol=None):
    """Proportional relative command towards a tool0 pose; (action, reached)."""
    pos_tol = POS_TOL if pos_tol is None else pos_tol
    dp = target_position - position
    dr = Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()
    action = np.concatenate([GAIN * dp / UNIT_TRANSLATION, GAIN * dr / UNIT_ROTATION])
    return np.clip(action, -CLIP, CLIP), bool(np.linalg.norm(dp) < pos_tol and np.linalg.norm(dr) < ROT_TOL)


class TaskA:
    def __init__(self, marker: np.ndarray):
        self.marker = np.asarray(marker, float)
        self.phase = "reach"

    def act(self, position, rotation, objects=None):
        action, reached = step_towards(position, rotation, tool_target(self.marker, rotation), rotation)
        if reached:
            self.phase = "hold"
        return action, "open"


class TaskB:
    PHASES = ("above_block", "descend", "close", "lift", "above_slot", "lower", "release", "retreat", "done")

    def __init__(self, hover=0.12, lift=0.15, close_steps=20, release_steps=10, half_height=BLOCK_SIZE[2] / 2,
                 grip_above=GRASP_ABOVE_CENTRE, grip_along=0.008, place_above=0.004, grip_yaw_deg=0.0, grasp_tol=None):
        """half_height: the object's; grip_above / grip_along: grasp point above its centre and along
        tool x (towards F2); place_above: object centre goal above slot + half_height when placing
        (0.004: seat on the receptacle floor; for a drop box pass its rim as the slot height and the
        drop clearance here)."""
        self.half_height, self.grip_above, self.grip_along, self.place_above = half_height, grip_above, grip_along, place_above
        # Hand turned about its own z after aligning tool x with the object's long axis (the harvest
        # pinch closes F1 onto F2 along a line ~34 deg from tool x, so the date must lie across it).
        self.grip_yaw = np.radians(grip_yaw_deg)
        self.grasp_tol = grasp_tol   # close only within this of the grasp point (default POS_TOL)
        self.phase, self.count = "above_block", 0  # gripper out: "open", "close" or an opening fraction
        self.hover, self.lift, self.close_steps, self.release_steps = hover, lift, close_steps, release_steps
        self.grip_rotation = None
        self.grasp_xyz = None       # where the block was when the hand closed
        self.grasp_yaw = None       # and its yaw (the held block turns with the hand)
        self.retreat_target = None

    def _next(self):
        self.phase = self.PHASES[self.PHASES.index(self.phase) + 1]
        self.count = 0

    def act(self, position, rotation, objects):
        """objects: {"block": (xyz, 3x3 rotation), "slot": (xyz, yaw)} -- the block's centre and
        orientation (sim state, or an AprilTag on the robot) and the receptacle's floor centre."""
        block, block_rotation = objects["block"]
        block_rotation = np.asarray(block_rotation, float)
        block_yaw = float(np.arctan2(block_rotation[1, 0], block_rotation[0, 0]))
        slot, slot_yaw = objects["slot"]
        block, slot = np.asarray(block, float), np.asarray(slot, float)
        half_height = self.half_height
        if self.grip_rotation is None:
            self.grip_rotation = top_grasp_rotation(block_yaw, rotation) @ Rotation.from_euler("z", self.grip_yaw).as_matrix()
        # Grasp above the centre (the open F2 cap clears the table) and 8 mm towards F2 (tool +x), so
        # the open F2 (~59 mm out along tool x) ends ~67 mm from the block centre: it cleared the 50 mm
        # half-length by ~2 mm and landed on the end when the approach was 3 mm off.
        grip = np.array([0.0, 0.0, self.grip_above]) + self.grip_rotation @ np.array([self.grip_along, 0.0, 0.0])
        gripper, action, reached = "open", np.zeros(6), False
        if self.phase == "above_block":
            action, reached = step_towards(position, rotation,
                                           tool_target(block + [0, 0, self.hover], self.grip_rotation), self.grip_rotation)
        elif self.phase == "descend":
            action, reached = step_towards(position, rotation, tool_target(block + grip, self.grip_rotation),
                                           self.grip_rotation, pos_tol=self.grasp_tol)
        elif self.phase == "close":
            if self.grasp_xyz is None:
                self.grasp_xyz, self.grasp_yaw = block + grip, block_yaw
            gripper, self.count = "close", self.count + 1
            reached = self.count >= self.close_steps
        elif self.phase == "lift":
            gripper = "close"
            action, reached = step_towards(position, rotation,
                                           tool_target(self.grasp_xyz + [0, 0, self.lift], self.grip_rotation),
                                           self.grip_rotation)
        elif self.phase in ("above_slot", "lower") and np.linalg.norm(
                position + rotation @ np.array([0.0, 0.0, GRASP_DEPTH + self.grip_above]) - block) > 0.04:
            # The object is not in the hand (dropped, or never held): stop instead of chasing it.
            self.phase = "done"
        elif self.phase in ("above_slot", "lower"):
            # Place the BLOCK, not the hand: the pinch holds it slipped and tilted (~10 deg: F1 and
            # F3 grip at different heights), so the hand pose follows from the observed hand-to-block
            # transform and the block's goal -- upright, at the receptacle's yaw (the short way: it
            # is symmetric under 180 deg), centred over the receptacle.
            gripper = "close"
            difference = (slot_yaw - block_yaw + np.pi / 2) % np.pi - np.pi / 2
            block_goal_rotation = Rotation.from_euler("z", block_yaw + difference).as_matrix()
            block_goal = slot + [0, 0, half_height + (self.lift if self.phase == "above_slot" else self.place_above)]
            hand_in_block = block_rotation.T @ rotation
            tool_in_block = block_rotation.T @ (position - block)
            goal_rotation = block_goal_rotation @ hand_in_block
            goal_position = block_goal + block_goal_rotation @ tool_in_block
            action, reached = step_towards(position, rotation, goal_position, goal_rotation)
        elif self.phase == "release":
            # Open gradually (gripper = opening fraction, 0 closed .. 1 open) with the hand still: opening
            # at once flicked the standing peg ~10 deg and the rising hand knocked it down; a fixed
            # part-open posture still held it and carried it up.
            self.count += 1
            gripper = min(1.0, self.count / self.release_steps)
            reached = self.count >= self.release_steps
        elif self.phase == "retreat":
            if self.retreat_target is None:
                self.retreat_target = position + [0, 0, 0.10]
            action, reached = step_towards(position, rotation, self.retreat_target, rotation)
        if reached and self.phase != "done":
            self._next()
        return action, gripper
