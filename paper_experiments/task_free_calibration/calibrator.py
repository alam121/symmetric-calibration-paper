"""Task-free local-dynamics calibration and source-to-target action mapping."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def physical_pose_delta(
    state: np.ndarray,
    next_state: np.ndarray,
    orientation_kind: str,
) -> np.ndarray:
    """Return translation plus canonical world-frame relative rotation vector."""
    state = np.asarray(state, dtype=np.float64)
    next_state = np.asarray(next_state, dtype=np.float64)
    translation = next_state[:, :3] - state[:, :3]
    if orientation_kind == "linear":
        rotation = next_state[:, 3:6] - state[:, 3:6]
    elif orientation_kind == "rotvec":
        current_rotation = Rotation.from_rotvec(state[:, 3:6])
        next_rotation = Rotation.from_rotvec(next_state[:, 3:6])
        rotation = (next_rotation * current_rotation.inv()).as_rotvec()
    elif orientation_kind == "euler_xyz":
        current_rotation = Rotation.from_euler("xyz", state[:, 3:6])
        next_rotation = Rotation.from_euler("xyz", next_state[:, 3:6])
        rotation = (next_rotation * current_rotation.inv()).as_rotvec()
    elif orientation_kind == "euler_xyz_body":
        # Same measurement, expressed in the tool frame rather than the world
        # frame. Which of the two is used does not matter for transfer provided
        # BOTH interfaces use it: the convention is a shared distortion that
        # cancels in source @ pinv(target). Mixing the two does not cancel.
        current_rotation = Rotation.from_euler("xyz", state[:, 3:6])
        next_rotation = Rotation.from_euler("xyz", next_state[:, 3:6])
        rotation = (current_rotation.inv() * next_rotation).as_rotvec()
    else:
        raise ValueError(f"unsupported orientation kind: {orientation_kind}")
    return np.concatenate((translation, rotation), axis=1)


@dataclass
class LocalActionModel:
    """Affine one-step TCP dynamics fitted from exploratory robot transitions."""

    action_to_delta: list[list[float]]
    bias: list[float]
    gripper_slope: float
    validation_motion_mae: float
    validation_motion_r2: float
    samples: int

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.action_to_delta, dtype=np.float64)

    @property
    def offset(self) -> np.ndarray:
        return np.asarray(self.bias, dtype=np.float64)

    def predict_delta(self, motion_action: np.ndarray) -> np.ndarray:
        return np.asarray(motion_action, dtype=np.float64) @ self.matrix + self.offset

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LocalActionModel":
        return cls(**payload)


@dataclass
class TaskFreeCalibrator:
    source: LocalActionModel
    target: LocalActionModel
    ridge: float
    source_domain: str = "LIBERO"
    target_domain: str = "CALVIN"
    source_proprio_mean: list[float] | None = None
    source_proprio_std: list[float] | None = None
    target_proprio_mean: list[float] | None = None
    target_proprio_std: list[float] | None = None
    source_image_stats: dict[str, dict[str, list[float]]] | None = None
    target_image_stats: dict[str, dict[str, list[float]]] | None = None

    def map_motion(self, source_motion: np.ndarray) -> np.ndarray:
        """Match the source command's local physical effect in target action space."""
        source_motion = np.asarray(source_motion, dtype=np.float64)
        desired_delta = self.source.predict_delta(source_motion)
        target_inverse = np.linalg.pinv(self.target.matrix)
        target_motion = (desired_delta - self.target.offset) @ target_inverse
        return np.clip(target_motion, -1.0, 1.0).astype(np.float32)

    def map_action(self, source_action: np.ndarray) -> np.ndarray:
        source_action = np.asarray(source_action, dtype=np.float64)
        if source_action.shape[-1] != 7:
            raise ValueError(f"expected final action width 7, got {source_action.shape}")
        flat = source_action.reshape(-1, 7)
        mapped = np.empty_like(flat, dtype=np.float32)
        for index, action in enumerate(flat):
            mapped[index, :6] = self.map_motion(action[:6])
            polarity = np.sign(self.source.gripper_slope * self.target.gripper_slope)
            if polarity == 0:
                polarity = 1.0
            # Binarise first, then apply polarity. Testing ``polarity * g >= 0``
            # instead disagrees with the uncalibrated path at exactly g == 0,
            # because -0.0 >= 0 is true and the sign is then lost.
            binarised = 1.0 if action[6] >= 0.0 else -1.0
            mapped[index, 6] = polarity * binarised
        return mapped.reshape(source_action.shape)

    def map_proprio(
        self,
        target_robot_obs: np.ndarray,
        target_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        """Align target proprioception while preserving subtask-relative motion.

        Absolute base/world frames differ across deployments. The TCP pose is
        therefore centered at the first state of each language subtask and
        scaled using task-free motion statistics. Gripper opening remains an
        absolute physical state and is aligned by its task-free moments.
        """
        if any(
            value is None
            for value in (
                self.source_proprio_mean,
                self.source_proprio_std,
                self.target_proprio_mean,
                self.target_proprio_std,
            )
        ):
            raise RuntimeError("calibrator does not contain task-free proprioception statistics")
        target_robot_obs = np.asarray(target_robot_obs, dtype=np.float64)
        if target_robot_obs.shape[-1] < 7:
            raise ValueError("target robot observation must contain TCP pose and gripper width")
        source_mean = np.asarray(self.source_proprio_mean, dtype=np.float64)
        source_std = np.asarray(self.source_proprio_std, dtype=np.float64)
        target_mean = np.asarray(self.target_proprio_mean, dtype=np.float64)
        target_std = np.asarray(self.target_proprio_std, dtype=np.float64)
        compact_target = target_robot_obs[:7]
        normalized = (compact_target - target_mean) / target_std
        if target_reference is not None:
            target_reference = np.asarray(target_reference, dtype=np.float64)
            if target_reference.shape[-1] < 7:
                raise ValueError("target reference must contain TCP pose and gripper width")
            normalized[:3] = (compact_target[:3] - target_reference[:3]) / target_std[:3]
            current_rotation = Rotation.from_euler("xyz", compact_target[3:6])
            reference_rotation = Rotation.from_euler("xyz", target_reference[3:6])
            relative_rotation = (current_rotation * reference_rotation.inv()).as_rotvec()
            normalized[3:6] = relative_rotation / target_std[3:6]
        normalized = np.clip(normalized, -3.0, 3.0)
        source_normalized = np.concatenate((normalized[:6], np.repeat(normalized[6], 2)))
        return (source_mean + source_std * source_normalized).astype(np.float32)

    def map_image(self, target_image: np.ndarray, camera: str) -> np.ndarray:
        """Align target RGB channel moments to task-free source observations."""
        if self.source_image_stats is None or self.target_image_stats is None:
            raise RuntimeError("calibrator does not contain task-free image statistics")
        if camera not in self.source_image_stats or camera not in self.target_image_stats:
            raise KeyError(f"unknown calibrated camera: {camera}")
        image = np.asarray(target_image, dtype=np.float32)
        source = self.source_image_stats[camera]
        target = self.target_image_stats[camera]
        source_mean = np.asarray(source["mean"], dtype=np.float32)
        source_std = np.asarray(source["std"], dtype=np.float32)
        target_mean = np.asarray(target["mean"], dtype=np.float32)
        target_std = np.asarray(target["std"], dtype=np.float32)
        aligned = (image - target_mean) / target_std * source_std + source_mean
        return np.clip(aligned, 0, 255).astype(np.uint8)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "task_free_local_dynamics_interface",
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "ridge": self.ridge,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "source_proprio_mean": self.source_proprio_mean,
            "source_proprio_std": self.source_proprio_std,
            "target_proprio_mean": self.target_proprio_mean,
            "target_proprio_std": self.target_proprio_std,
            "source_image_stats": self.source_image_stats,
            "target_image_stats": self.target_image_stats,
            "uses_task_labels": False,
            "uses_language": False,
            "uses_expert_actions": False,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "TaskFreeCalibrator":
        payload = json.loads(Path(path).read_text())
        return cls(
            source=LocalActionModel.from_dict(payload["source"]),
            target=LocalActionModel.from_dict(payload["target"]),
            ridge=float(payload["ridge"]),
            source_domain=payload.get("source_domain", "LIBERO"),
            target_domain=payload.get("target_domain", "CALVIN"),
            source_proprio_mean=payload.get("source_proprio_mean"),
            source_proprio_std=payload.get("source_proprio_std"),
            target_proprio_mean=payload.get("target_proprio_mean"),
            target_proprio_std=payload.get("target_proprio_std"),
            source_image_stats=payload.get("source_image_stats"),
            target_image_stats=payload.get("target_image_stats"),
        )


def fit_local_action_model(
    action: np.ndarray,
    state: np.ndarray,
    next_state: np.ndarray,
    validation_mask: np.ndarray,
    ridge: float = 1e-4,
    orientation_kind: str = "linear",
) -> LocalActionModel:
    """Fit on the non-validation rows and report held-out transition metrics."""
    action = np.asarray(action, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    next_state = np.asarray(next_state, dtype=np.float64)
    validation_mask = np.asarray(validation_mask, dtype=bool)
    if action.ndim != 2 or action.shape[1] != 7:
        raise ValueError("actions must have shape [N,7]")
    if state.shape != next_state.shape or state.shape[0] != action.shape[0] or state.shape[1] < 7:
        raise ValueError("states must be matching [N,S>=7] arrays")
    if validation_mask.shape != (action.shape[0],) or validation_mask.all() or (~validation_mask).all():
        raise ValueError("validation mask must select a nonempty strict subset")

    delta = physical_pose_delta(state, next_state, orientation_kind)
    design = np.concatenate((action[:, :6], np.ones((len(action), 1))), axis=1)
    train = ~validation_mask
    penalty = np.eye(design.shape[1]) * ridge
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design[train].T @ design[train] + penalty, design[train].T @ delta[train])
    prediction = design[validation_mask] @ coefficients
    target = delta[validation_mask]
    residual = prediction - target
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / max(denominator, 1e-12)

    gripper_delta = next_state[:, 6] - state[:, 6]
    grip_action = action[:, 6]
    centered = grip_action[train] - grip_action[train].mean()
    gripper_slope = float(
        np.sum(centered * (gripper_delta[train] - gripper_delta[train].mean()))
        / max(np.sum(centered**2), 1e-12)
    )
    return LocalActionModel(
        action_to_delta=coefficients[:6].tolist(),
        bias=coefficients[6].tolist(),
        gripper_slope=gripper_slope,
        validation_motion_mae=mae,
        validation_motion_r2=r2,
        samples=int(action.shape[0]),
    )
