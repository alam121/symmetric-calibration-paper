import numpy as np

from paper_experiments.task_free_calibration.collect_calibration import (
    FORBIDDEN_KEYS,
    SmoothExcitation,
    deployment_action,
    episode_columns,
    stack_columns,
    valid_initial_conditions,
)


def test_deployment_action_applies_matrix_and_gripper_polarity() -> None:
    action = np.asarray([0.2, -0.4, 0.5, 0.1, -0.2, 0.3, 1.0])
    matrix = np.diag([0.5, 0.25, 2.0, -1.0, 1.0, 0.5]).tolist()
    transformed = deployment_action(action, matrix, -1.0)
    np.testing.assert_allclose(
        transformed,
        [0.1, -0.1, 1.0, -0.1, -0.2, 0.15, -1.0],
    )


def test_task_free_schema_and_initial_conditions() -> None:
    columns = episode_columns()
    assert not FORBIDDEN_KEYS.intersection(columns)
    assert len(valid_initial_conditions()) > 10

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    obs = {
        "rgb_obs": {"rgb_static": image, "rgb_gripper": image},
        "robot_obs": np.zeros(15, dtype=np.float32),
    }
    columns["step"].append(0)
    for key in ("rgb_static_t", "rgb_gripper_t", "rgb_static_tp1", "rgb_gripper_tp1"):
        columns[key].append(image)
    for key in ("robot_obs_t", "robot_obs_tp1"):
        columns[key].append(obs["robot_obs"])
    columns["action_t"].append(np.zeros(7, dtype=np.float32))
    arrays = stack_columns(columns)
    assert arrays["robot_obs_t"].shape == (1, 15)
    assert arrays["action_t"].shape == (1, 7)


def test_smooth_excitation_is_bounded_and_reflects_at_workspace_edge() -> None:
    policy = SmoothExcitation(np.random.default_rng(0), 0.85, 0.28, 0.18, 4)
    state = np.zeros(15, dtype=np.float32)
    state[:3] = [-0.25, -0.40, 0.48]
    actions = [policy(state, step) for step in range(12)]
    assert all(action.shape == (7,) for action in actions)
    assert all(np.max(np.abs(action[:6])) <= 0.65 for action in actions)
    assert all(np.all(action[:3] >= 0.0) for action in actions)
    assert {float(action[-1]) for action in actions} == {-1.0, 1.0}


def test_issued_action_can_be_preserved_from_in_place_environment_conversion() -> None:
    action = np.asarray([0.2, -0.3, 0.4, 0.1, -0.2, 0.3, 1.0], dtype=np.float32)
    issued = action.copy()
    simulator_input = action.copy()
    simulator_input[:3] *= 0.02
    simulator_input[3:6] *= 0.05
    np.testing.assert_allclose(issued, [0.2, -0.3, 0.4, 0.1, -0.2, 0.3, 1.0])
    assert not np.allclose(simulator_input, issued)
