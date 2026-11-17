#!/usr/bin/env python3
"""Checkpoint-invariance sweep: does recovery hold a constant fraction?

The headline claim currently rests on two checkpoints (48.2% and 48.0%). Two
points is a coincidence, not a regularity. This adds two more checkpoints whose
clean ceilings span a wide range, giving four independent (ceiling, recovery)
pairs from the same distortion and the same calibrator.

The calibrator is fitted from policy-independent random interaction, so the
identical calibrator applies to every checkpoint -- no refitting needed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
HERE = REPO / "paper_experiments/task_free_calibration"
RUNNER = REPO / "paper_experiments/calvin_baselines/run_protocol.py"
BASE = HERE / "controlled_interface_protocol.json"
REFERENCE = HERE.parent / "calvin_baselines/minimum_adaptation_evaluation_protocol.generated.json"
RESULTS = REPO / "results/task_free_calibration"
PYTHON = "/home/alam/miniconda3/bin/python"
STATE = RESULTS / "checkpoint_invariance_state.json"

N, SEQUENCE_SEED = 100, 1729
MAX_SEQUENCE_SEED = (2**32 - 1) // 1_000_003

# Two additional checkpoints, chosen to widen the ceiling range beyond the
# 58%-87% already covered. Both are existing trained seeds.
CELLS = {
    "action_head_seed1": {
        "root": "invariance_action_head_seed1",
        "use_proprio": "0",
        "base_checkpoint": "moojink/openvla-7b-oft-finetuned-libero-spatial",
    },
    "native_oft_seed1": {
        "root": "invariance_native_oft_seed1",
        "use_proprio": "1",
        "base_checkpoint": None,
    },
}
METHODS = ("native_clean", "native_hidden_interface", "native_task_free_calibrated")


def clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PROMPT_MODIFIER",
                "CONDA_SHLVL", "_CE_CONDA", "_CE_M"):
        env.pop(key, None)
    return env


def record(**payload: object) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(
        {"updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **payload},
        indent=2) + "\n")


def checkpoints() -> dict[str, str]:
    ref = json.loads(REFERENCE.read_text())["methods"]
    return {
        "action_head_seed1": ref["calvin_action_head_only"]["model_seeds"]["1"],
        "native_oft_seed1": ref["calvin_native_oft"]["model_seeds"]["1"],
    }


def make_protocol(cell: str, checkpoint: str) -> Path:
    spec = CELLS[cell]
    p = json.loads(BASE.read_text())
    p["fixed_policy_settings"]["CALVIN_USE_PROPRIO"] = spec["use_proprio"]
    for method in p["methods"].values():
        method["model_seeds"] = {"0": checkpoint}
        method["environment"]["CALVIN_USE_PROPRIO"] = spec["use_proprio"]
        if spec["base_checkpoint"]:
            method["base_checkpoint"] = spec["base_checkpoint"]
    p["paths"]["results_root"] = f"vla-deployment-benchmark/results/task_free_calibration/{spec['root']}"
    out = HERE / f"invariance_{cell}.generated.json"
    out.write_text(json.dumps(p, indent=2) + "\n")
    return out


def done(run: Path) -> bool:
    manifest = run / "manifest.json"
    if not (run / "results.json").exists() or not manifest.exists():
        return False
    data = json.loads(manifest.read_text())
    used = data.get("sequence_seed_used")
    return data.get("status") == "complete" and (used is None or int(used) == SEQUENCE_SEED)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    assert 0 <= SEQUENCE_SEED <= MAX_SEQUENCE_SEED, "sequence seed overflows CALVIN's uint32 derivation"

    paths = checkpoints()
    plan = [(c, m) for c in CELLS for m in METHODS]
    print(f"{len(plan)} evaluations across {len(CELLS)} checkpoints")
    for cell, path in paths.items():
        # protocol paths may be absolute or relative to the workspace root
        exists = Path(path).exists() or (WORKSPACE / path).exists()
        print(f"  {cell:20s} {'MISSING ' if not exists else ''}{path}")
    if args.dry_run:
        return 0

    env = clean_environment()
    for cell in CELLS:
        protocol = make_protocol(cell, paths[cell])
        for method in METHODS:
            run = RESULTS / CELLS[cell]["root"] / f"{method}__model-seed-0__n-{N}"
            if done(run):
                print(f"skip (complete): {cell}/{method}", flush=True)
                continue
            label = f"{cell} / {method}"
            print(f"\n=== {label} ===", flush=True)
            record(stage="running", job=label)
            result = subprocess.run(
                [PYTHON, "-u", str(RUNNER), "--protocol", str(protocol),
                 "--workspace", str(WORKSPACE), "--method", method, "--model-seed", "0",
                 "--num-sequences", str(N), "--sequence-seed", str(SEQUENCE_SEED),
                 "--skip-memory", "--overwrite"],
                cwd=REPO, env=env, check=False)
            if result.returncode:
                record(stage="failed", job=label, returncode=result.returncode)
                raise SystemExit(f"FAILED: {label} ({result.returncode})")

    record(stage="complete")
    print("\ncheckpoint invariance sweep complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
