#!/usr/bin/env python3
"""Single sequential driver for all outstanding evaluations.

Replaces the ad-hoc shell waiters that caused three separate failures: a
broken pipe, a waiter polling for a log line printed at run *start*, and
several waiters racing for the GPU in the gap between runs. One process,
strictly sequential, resumable, and it never starts a run while another holds
the GPU.

Jobs are listed in priority order: whatever answers the open question first.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
PKG = REPO / "paper_experiments/task_free_calibration"
RUNNER = REPO / "paper_experiments/calvin_baselines/run_protocol.py"
RESULTS = REPO / "results/task_free_calibration"
PYTHON = "/home/alam/miniconda3/bin/python"
STATE = RESULTS / "driver_state.json"

# (protocol stem, method, results root, model seed, n, why)
JOBS = [
    # --- verification that the gripper fix works, before anything else -------
    ("bitexact_protocol.generated", "oracle_bitexact", "bitexact_fixed", "0", 100,
     "bit-exact oracle after the gripper fix -- should now approach clean"),
    ("controlled_interface_protocol", "native_task_free_calibrated", "controlled_fixed", "0", 100,
     "headline calibration after the fix"),
    ("oracle_sweep_protocol.generated", "oracle_sign_scale", "oracle_sweep_fixed", "0", 100, "oracle, sign+scale"),
    ("oracle_sweep_protocol.generated", "oracle_cyclic", "oracle_sweep_fixed", "0", 100, "oracle, cyclic"),
    # --- robustness -----------------------------------------------------------
    ("robustness_axis_swap_seed201.generated", "native_task_free_calibrated", "robustness_fixed/axis_swap_seed201", "0", 100, "robustness seed 201"),
    ("robustness_axis_swap_seed202.generated", "native_task_free_calibrated", "robustness_fixed/axis_swap_seed202", "0", 100, "robustness seed 202"),
    ("robustness_axis_swap_seed203.generated", "native_task_free_calibrated", "robustness_fixed/axis_swap_seed203", "0", 100, "robustness seed 203"),
    ("robustness_sign_scale_seed201.generated", "native_task_free_calibrated", "robustness_fixed/sign_scale_seed201", "0", 100, "robustness sign+scale"),
    ("robustness_cyclic_seed201.generated", "native_task_free_calibrated", "robustness_fixed/cyclic_seed201", "0", 100, "robustness cyclic"),
    # --- budget ---------------------------------------------------------------
    ("budget_ablation_protocol.generated", "budget_64", "budget_fixed", "0", 100, "budget 64"),
    ("budget_ablation_protocol.generated", "budget_320", "budget_fixed", "0", 100, "budget 320"),
    ("budget_ablation_protocol.generated", "budget_640", "budget_fixed", "0", 100, "budget 640"),
    # --- checkpoints ----------------------------------------------------------
    ("controlled_interface_action_head_protocol.generated", "native_task_free_calibrated", "ckpt_fixed/action_head_s0", "0", 100, "action head seed 0"),
    ("invariance_action_head_seed1.generated", "native_task_free_calibrated", "ckpt_fixed/action_head_s1", "0", 100, "action head seed 1"),
    ("invariance_native_oft_seed1.generated", "native_task_free_calibrated", "ckpt_fixed/lora_s1", "0", 100, "LoRA seed 1"),
    # --- second architecture + cross-domain ----------------------------------
    ("mcil_family_protocol.generated", "mcil_task_free_calibrated", "mcil_fixed", "official", 1000, "MCIL check (should stay ~100%)"),
    ("crossdomain_protocol.generated", "crossdomain_direct", "crossdomain_fixed", "official", 100, "cross-domain direct"),
    ("crossdomain_protocol.generated", "crossdomain_calibrated", "crossdomain_fixed", "official", 100, "cross-domain calibrated"),
]

SEQUENCE_SEED = 1729


def clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PROMPT_MODIFIER",
                "CONDA_SHLVL", "_CE_CONDA", "_CE_M"):
        env.pop(key, None)
    return env


def complete(root: str, method: str, seed: str, n: int) -> bool:
    run = RESULTS / root / f"{method}__model-seed-{seed}__n-{n}"
    manifest = run / "manifest.json"
    if not (run / "results.json").exists() or not manifest.exists():
        return False
    return json.loads(manifest.read_text()).get("status") == "complete"


def gpu_busy() -> bool:
    """True while any evaluation still holds the GPU."""
    found = subprocess.run(["pgrep", "-f", "run_protocol.py|evaluate_policy.py"],
                           capture_output=True, text=True)
    return bool(found.stdout.strip())


def record(**payload: object) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(
        {"updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **payload}, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pending = [j for j in JOBS if not complete(j[2], j[1], j[3], j[4])]
    print(f"{len(JOBS) - len(pending)} complete, {len(pending)} pending", flush=True)
    for _, method, root, _, n, why in pending:
        print(f"  {method:24s} n={n:<4d} {why}", flush=True)
    if args.dry_run:
        return 0

    env = clean_environment()
    while gpu_busy():
        print("waiting for the GPU to free before starting", flush=True)
        time.sleep(60)

    for index, (proto, method, root, seed, n, why) in enumerate(pending, start=1):
        if complete(root, method, seed, n):
            continue
        print(f"\n=== [{index}/{len(pending)}] {method} — {why} ===", flush=True)
        record(stage="running", job=method, index=index, total=len(pending))
        result = subprocess.run(
            [PYTHON, "-u", str(RUNNER), "--protocol", str(PKG / f"{proto}.json"),
             "--workspace", str(WORKSPACE), "--method", method, "--model-seed", seed,
             "--num-sequences", str(n), "--sequence-seed", str(SEQUENCE_SEED),
             "--skip-memory", "--overwrite"],
            cwd=REPO, env=env, check=False)
        if result.returncode:
            # A failure here is usually a protocol error, not a transient one.
            # Record it and continue so one bad job cannot strand the rest.
            print(f"FAILED {method} (code {result.returncode}) — continuing", flush=True)
            record(stage="failed_continuing", job=method, returncode=result.returncode)
        time.sleep(10)

    record(stage="complete")
    print("\ndriver complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
