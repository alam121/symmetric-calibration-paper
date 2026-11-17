import json

import numpy as np

from paper_experiments.task_free_calibration.audit_calibration import audit


def test_audit_accepts_only_task_free_diverse_shards(tmp_path) -> None:
    manifest = {
        "contains_language": False,
        "contains_task_ids": False,
        "contains_rewards_or_success": False,
        "contains_expert_actions": False,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    rng = np.random.default_rng(0)
    for episode in range(2):
        action = rng.uniform(-0.5, 0.5, (16, 7)).astype(np.float32)
        action[:, 6] = np.where(np.arange(16) < 8, -1, 1)
        state = rng.normal(size=(16, 8)).astype(np.float32)
        np.savez_compressed(
            tmp_path / f"calibration_episode_{episode:05d}.npz",
            action_t=action,
            robot_obs_t=state,
            robot_obs_tp1=state + 0.01,
        )
    report = audit(tmp_path)
    assert report["status"] == "passed"
    assert report["transitions"] == 32
    assert report["forbidden_keys_present"] == []
