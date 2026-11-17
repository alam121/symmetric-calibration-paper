#!/usr/bin/env python3
"""Sequential run queue for the ICRA 2027 submission.

Runs every outstanding evaluation one at a time so the single GPU is never
contended. Already-complete runs are skipped, so the queue is resumable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
PKG = REPO / "paper_experiments/task_free_calibration"
RUNNER = REPO / "paper_experiments/calvin_baselines/run_protocol.py"
PYTHON = "/home/alam/miniconda3/bin/python"
STATE = REPO / "results/task_free_calibration/icra_queue_state.json"

CONTROLLED = PKG / "controlled_interface_protocol.json"
BUDGET_PROTOCOL = PKG / "budget_ablation_protocol.generated.json"
ACTION_HEAD = PKG / "controlled_interface_action_head_protocol.generated.json"

SEQUENCE_SEED = 1729
N = 100

# (protocol, method, results_root_relative, label)
JOBS = [
    (BUDGET_PROTOCOL, "budget_64", "budget_ablation", "budget 64 @ n=100"),
    (BUDGET_PROTOCOL, "budget_320", "budget_ablation", "budget 320 @ n=100"),
    (BUDGET_PROTOCOL, "budget_640", "budget_ablation", "budget 640 @ n=100"),
    (ACTION_HEAD, "native_clean", "controlled_interface_action_head", "action-head clean"),
    (ACTION_HEAD, "native_hidden_interface", "controlled_interface_action_head", "action-head hidden"),
    (ACTION_HEAD, "native_task_free_calibrated", "controlled_interface_action_head", "action-head calibrated"),
]


def clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    # Leaving CONDA_SHLVL set while CONDA_PREFIX is unset makes conda's
    # activator dereference a None prefix and crash the policy server.
    for key in (
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_PROMPT_MODIFIER",
        "CONDA_SHLVL",
        "_CE_CONDA",
        "_CE_M",
    ):
        env.pop(key, None)
    return env


def run_dir(root: str, method: str) -> Path:
    return REPO / "results/task_free_calibration" / root / f"{method}__model-seed-0__n-{N}"


def complete(path: Path) -> bool:
    manifest = path / "manifest.json"
    if not (path / "results.json").exists() or not manifest.exists():
        return False
    return json.loads(manifest.read_text()).get("status") == "complete"


def record(**payload: object) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        json.dumps(
            {"updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **payload},
            indent=2,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pending = [job for job in JOBS if not complete(run_dir(job[2], job[1]))]
    print(f"{len(JOBS) - len(pending)} complete, {len(pending)} pending", flush=True)
    for _, method, root, label in pending:
        print(f"  pending: {label:28s} -> {root}/{method}", flush=True)
    if args.dry_run:
        return 0

    env = clean_environment()
    for index, (protocol, method, root, label) in enumerate(pending, start=1):
        print(f"\n=== [{index}/{len(pending)}] {label} ===", flush=True)
        record(stage="running", job=label, index=index, total=len(pending))
        command = [
            PYTHON, "-u", str(RUNNER),
            "--protocol", str(protocol),
            "--workspace", str(WORKSPACE),
            "--method", method,
            "--model-seed", "0",
            "--num-sequences", str(N),
            "--sequence-seed", str(SEQUENCE_SEED),
            "--skip-memory",
            "--overwrite",
        ]
        result = subprocess.run(command, cwd=REPO, env=env, check=False)
        if result.returncode:
            record(stage="failed", job=label, returncode=result.returncode)
            print(f"FAILED: {label} (code {result.returncode})", file=sys.stderr)
            return result.returncode

    record(stage="complete", jobs=[job[3] for job in JOBS])
    print("\nqueue complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
