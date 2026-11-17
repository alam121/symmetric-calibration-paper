import numpy as np

from paper_experiments.task_free_calibration.calibrator import (
    TaskFreeCalibrator,
    fit_local_action_model,
    physical_pose_delta,
)


def synthetic_model(matrix: np.ndarray, gripper_slope: float, seed: int):
    rng = np.random.default_rng(seed)
    episodes, steps = 20, 32
    action = rng.uniform(-1, 1, (episodes * steps, 7))
    action[:, 6] = np.where(action[:, 6] >= 0, 1, -1)
    state = rng.normal(0, 0.1, (episodes * steps, 7))
    delta = np.zeros_like(state)
    delta[:, :6] = action[:, :6] @ matrix
    delta[:, 6] = gripper_slope * action[:, 6]
    next_state = state + delta
    validation = np.repeat(np.arange(episodes) >= 16, steps)
    return fit_local_action_model(action, state, next_state, validation, ridge=1e-8)


def test_calibrator_recovers_cross_domain_motion_and_gripper_polarity() -> None:
    source_matrix = np.diag([0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    target_matrix = np.diag([0.04, 0.02, 0.08, 0.10, 0.03, 0.14])
    source = synthetic_model(source_matrix, gripper_slope=-0.01, seed=0)
    target = synthetic_model(target_matrix, gripper_slope=0.02, seed=1)
    calibrator = TaskFreeCalibrator(source, target, ridge=1e-8)
    action = np.asarray([0.4, -0.2, 0.6, -0.1, 0.2, -0.4, 1.0])
    mapped = calibrator.map_action(action)
    np.testing.assert_allclose(target.predict_delta(mapped[:6]), source.predict_delta(action[:6]), atol=1e-6)
    assert mapped[-1] == -1.0
    assert source.validation_motion_r2 > 0.999
    assert target.validation_motion_r2 > 0.999


def test_task_free_proprio_moment_alignment() -> None:
    identity = synthetic_model(np.eye(6), gripper_slope=1.0, seed=2)
    calibrator = TaskFreeCalibrator(
        identity,
        identity,
        ridge=1e-8,
        source_proprio_mean=[1, 2, 3, 4, 5, 6, 0.02, -0.02],
        source_proprio_std=[2, 2, 2, 2, 2, 2, 0.01, 0.01],
        target_proprio_mean=[10, 20, 30, 40, 50, 60, 0.04],
        target_proprio_std=[5, 5, 5, 5, 5, 5, 0.02],
    )
    target = np.asarray([15, 15, 30, 50, 50, 55, 0.06] + [0] * 8)
    mapped = calibrator.map_proprio(target)
    np.testing.assert_allclose(mapped, [3, 0, 3, 8, 5, 4, 0.03, -0.01])
    reference = np.asarray([10, 20, 30, 40, 50, 60, 0.04] + [0] * 8)
    at_reference = calibrator.map_proprio(reference, reference)
    np.testing.assert_allclose(at_reference, [1, 2, 3, 4, 5, 6, 0.02, -0.02])


def test_physical_pose_delta_handles_euler_wraparound() -> None:
    state = np.asarray([[0, 0, 0, np.pi - 0.01, 0, 0, 0]], dtype=np.float64)
    next_state = np.asarray([[0, 0, 0, -np.pi + 0.01, 0, 0, 0]], dtype=np.float64)
    delta = physical_pose_delta(state, next_state, "euler_xyz")
    assert abs(np.linalg.norm(delta[0, 3:6]) - 0.02) < 1e-6


def test_task_free_image_moment_alignment() -> None:
    identity = synthetic_model(np.eye(6), gripper_slope=1.0, seed=3)
    calibrator = TaskFreeCalibrator(
        identity,
        identity,
        ridge=1e-8,
        source_image_stats={"static": {"mean": [100, 110, 120], "std": [20, 30, 40]}},
        target_image_stats={"static": {"mean": [50, 60, 70], "std": [10, 15, 20]}},
    )
    image = np.asarray([[[50, 75, 50], [60, 60, 90]]], dtype=np.uint8)
    aligned = calibrator.map_image(image, "static")
    np.testing.assert_array_equal(aligned, [[[100, 140, 80], [120, 110, 160]]])
