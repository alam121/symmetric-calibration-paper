#!/usr/bin/env python3
"""Seed and distortion robustness sweep for the ICRA submission.

Answers the two sharpest remaining reviewer objections:

  1. "You report a single calibration seed."       -> vary the interaction seed.
  2. "You hand-picked one convenient distortion."  -> vary the hidden interface.

One-factor-at-a-time rather than a full factorial: seeds vary at the reference
distortion, distortions vary at the reference seed. Every stage is resumable and
skips completed work.
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
HERE = REPO / "paper_experiments/task_free_calibration"
DATA = REPO / "data/task_free_calibration"
RESULTS = REPO / "results/task_free_calibration/robustness"
RUNNER = REPO / "paper_experiments/calvin_baselines/run_protocol.py"
BASE_PROTOCOL = HERE / "controlled_interface_protocol.json"
CONDA = "/home/alam/miniconda3/bin/conda"
PYTHON = "/home/alam/miniconda3/bin/python"
STATE = RESULTS / "sweep_state.json"

SOURCE = DATA / "calvin_seed101"          # clean-interface reference interaction
EPISODES, STEPS = 20, 64
N, SEQUENCE_SEED = 100, 1729

# Reference distortion: the one already reported (axis swap + gripper flip).
REFERENCE = "axis_swap"
DISTORTIONS: dict[str, dict] = {
    "axis_swap": {
        "matrix": [[0,-0.8,0,0,0,0],[0.65,0,0,0,0,0],[0,0,0.55,0,0,0],
                   [0,0,0,0,-0.9,0],[0,0,0,0.7,0,0],[0,0,0,0,0,0.6]],
        "gripper": -1.0,
        "note": "axis permutation + sign flips + scale compression + gripper inversion",
    },
    "sign_scale": {
        # No permutation: pure sign flips and anisotropic scaling, gripper normal.
        "matrix": [[-0.75,0,0,0,0,0],[0,0.5,0,0,0,0],[0,0,-1.15,0,0,0],
                   [0,0,0,0.85,0,0],[0,0,0,0,-0.6,0],[0,0,0,0,0,1.1]],
        "gripper": 1.0,
        "note": "sign flips and anisotropic scale only, no axis permutation",
    },
    "cyclic": {
        # 3-cycle on translation, different rotation coupling, gripper flipped.
        "matrix": [[0,0,0.7,0,0,0],[0.9,0,0,0,0,0],[0,-0.45,0,0,0,0],
                   [0,0,0,0,0,0.8],[0,0,0,-1.0,0,0],[0,0,0,0,0.65,0]],
        "gripper": -1.0,
        "note": "cyclic axis permutation on translation and rotation",
    },
}
SEEDS = (201, 202, 203)


def clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PROMPT_MODIFIER",
                "CONDA_SHLVL", "_CE_CONDA", "_CE_M"):
        env.pop(key, None)
    return env


ENV = clean_environment()


def record(**payload: object) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(
        {"updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **payload},
        indent=2) + "\n")


def shell(command: list[str], cwd: Path, label: str) -> None:
    print(f"\n--- {label} ---", flush=True)
    record(stage=label)
    result = subprocess.run(command, cwd=cwd, env=ENV, check=False)
    if result.returncode:
        record(stage="failed", label=label, returncode=result.returncode)
        raise SystemExit(f"FAILED at {label} (code {result.returncode})")


def tag(distortion: str, seed: int) -> str:
    return f"{distortion}_seed{seed}"


def collect(distortion: str, seed: int) -> Path:
    out = DATA / f"calvin_{tag(distortion, seed)}"
    if (out / "manifest.json").exists() and len(list(out.glob("*.npz"))) >= EPISODES:
        return out
    spec = DISTORTIONS[distortion]
    shell([CONDA, "run", "--no-capture-output", "-n", "calvin_venv", "python", "-u",
           str(HERE / "collect_calibration.py"),
           "--dataset-path", str(WORKSPACE / "calvin/dataset/task_D_D"),
           "--output", str(out), "--episodes", str(EPISODES),
           "--steps-per-episode", str(STEPS), "--seed", str(seed),
           "--action-matrix", json.dumps(spec["matrix"]),
           "--gripper-polarity", str(spec["gripper"])],
          WORKSPACE / "calvin", f"collect {tag(distortion, seed)}")
    return out


def audit(directory: Path) -> None:
    """Prove the shard carries no task/language/reward/success/expert-action field."""
    if (directory / "audit.json").exists():
        return
    shell([CONDA, "run", "--no-capture-output", "-n", "openvla-oft", "python",
           str(HERE / "audit_calibration.py"),
           "--data", str(directory), "--output", str(directory / "audit.json")],
          REPO, f"audit {directory.name}")


def fit(distortion: str, seed: int, target: Path) -> Path:
    out = DATA / f"calibrator_{tag(distortion, seed)}.json"
    if out.exists():
        return out
    shell([CONDA, "run", "--no-capture-output", "-n", "openvla-oft", "python", "-u",
           str(HERE / "fit_calibrator.py"),
           "--source", str(SOURCE), "--target", str(target), "--output", str(out),
           "--source-orientation", "euler_xyz", "--target-orientation", "euler_xyz",
           "--source-gripper-width", "1", "--target-gripper-width", "1",
           "--split-seed", "1729"],
          REPO, f"fit {tag(distortion, seed)}")
    return out


def protocol_for(distortion: str, seed: int, calibrator: Path) -> Path:
    spec = DISTORTIONS[distortion]
    base = json.loads(BASE_PROTOCOL.read_text())
    client = {"CALVIN_DEPLOYMENT_ACTION_MATRIX": json.dumps(spec["matrix"]),
              "CALVIN_DEPLOYMENT_GRIPPER_POLARITY": str(int(spec["gripper"]))}
    for name, method in base["methods"].items():
        if name == "native_clean":
            continue
        method["client_environment"] = dict(client)
        if name == "native_task_free_calibrated":
            method["environment"]["CALVIN_TASK_FREE_CALIBRATION_PATH"] = str(calibrator)
    base["methods"]["native_hidden_interface"]["description"] = spec["note"]
    base["paths"]["results_root"] = f"vla-deployment-benchmark/results/task_free_calibration/robustness/{tag(distortion, seed)}"
    out = HERE / f"robustness_{tag(distortion, seed)}.generated.json"
    out.write_text(json.dumps(base, indent=2) + "\n")
    return out


def evaluate(protocol: Path, method: str, distortion: str, seed: int) -> None:
    run = RESULTS / tag(distortion, seed) / f"{method}__model-seed-0__n-{N}"
    manifest = run / "manifest.json"
    if (run / "results.json").exists() and manifest.exists() \
       and json.loads(manifest.read_text()).get("status") == "complete":
        return
    shell([PYTHON, "-u", str(RUNNER), "--protocol", str(protocol),
           "--workspace", str(WORKSPACE), "--method", method, "--model-seed", "0",
           "--num-sequences", str(N), "--sequence-seed", str(SEQUENCE_SEED),
           "--skip-memory", "--overwrite"],
          REPO, f"eval {method} @ {tag(distortion, seed)}")


def jobs() -> list[tuple[str, int, bool]]:
    """(distortion, seed, needs_hidden_baseline)"""
    out: list[tuple[str, int, bool]] = []
    for seed in SEEDS:                      # seed robustness at the reference distortion
        out.append((REFERENCE, seed, seed == SEEDS[0]))
    for name in DISTORTIONS:                # distortion robustness at the reference seed
        if name != REFERENCE:
            out.append((name, SEEDS[0], True))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = jobs()
    print(f"{len(plan)} (distortion, seed) cells:")
    for distortion, seed, hidden in plan:
        print(f"  {tag(distortion, seed):22s} hidden_baseline={hidden}")
    if args.dry_run:
        return 0

    for distortion, seed, needs_hidden in plan:
        target = collect(distortion, seed)
        audit(target)
        calibrator = fit(distortion, seed, target)
        protocol = protocol_for(distortion, seed, calibrator)
        if needs_hidden:
            evaluate(protocol, "native_hidden_interface", distortion, seed)
        evaluate(protocol, "native_task_free_calibrated", distortion, seed)

    record(stage="complete", cells=[tag(d, s) for d, s, _ in plan])
    print("\nrobustness sweep complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
