#!/usr/bin/env python3
"""Phase 2: restore the reference budget run, validate on a held-out sequence
seed, then run the seed/distortion robustness sweep.

Sequence-seed guard
-------------------
CALVIN derives its per-state RNG seed as ``sequence_seed * 1_000_003 + i``
(``multistep_sequences.py``), so any ``sequence_seed`` above ``(2**32-1)//1_000_003
== 4294`` overflows numpy's seed range. The failure surfaces ~10 minutes in, as a
worker-process ``ValueError`` that leaves the policy server listening and the run
hung rather than exiting. Validate up front.
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
RUNNER = REPO / "paper_experiments/calvin_baselines/run_protocol.py"
PYTHON = "/home/alam/miniconda3/bin/python"
RESULTS = REPO / "results/task_free_calibration"
STATE = RESULTS / "phase2_state.json"

MAX_SEQUENCE_SEED = (2**32 - 1) // 1_000_003  # == 4294

REFERENCE_SEED = 1729   # every headline table uses this
HELDOUT_SEED = 2027     # independent sequence set, still under the cap
N = 100


def check_seed(seed: int) -> None:
    if not 0 <= seed <= MAX_SEQUENCE_SEED:
        raise SystemExit(
            f"sequence seed {seed} is outside CALVIN's usable range "
            f"[0, {MAX_SEQUENCE_SEED}] (seed * 1_000_003 must fit in uint32)"
        )


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


def done(run: Path, seed: int) -> bool:
    manifest = run / "manifest.json"
    if not (run / "results.json").exists() or not manifest.exists():
        return False
    data = json.loads(manifest.read_text())
    if data.get("status") != "complete":
        return False
    used = data.get("sequence_seed_used")
    return used is None or int(used) == seed


def evaluate(protocol: Path, method: str, root: str, seed: int, label: str,
             env: dict[str, str]) -> None:
    check_seed(seed)
    run = RESULTS / root / f"{method}__model-seed-0__n-{N}"
    if done(run, seed):
        print(f"skip (complete): {label}", flush=True)
        return
    print(f"\n=== {label} ===", flush=True)
    record(stage="running", job=label, seed=seed)
    result = subprocess.run(
        [PYTHON, "-u", str(RUNNER), "--protocol", str(protocol),
         "--workspace", str(WORKSPACE), "--method", method, "--model-seed", "0",
         "--num-sequences", str(N), "--sequence-seed", str(seed),
         "--skip-memory", "--overwrite"],
        cwd=REPO, env=env, check=False)
    if result.returncode:
        record(stage="failed", job=label, returncode=result.returncode)
        raise SystemExit(f"FAILED: {label} (code {result.returncode})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()

    check_seed(REFERENCE_SEED)
    check_seed(HELDOUT_SEED)
    env = clean_environment()

    # 1. Restore the reference budget_64 run clobbered by the overflowed seed.
    evaluate(HERE / "budget_ablation_protocol.generated.json", "budget_64",
             "budget_ablation", REFERENCE_SEED,
             f"restore budget_64 @ seed {REFERENCE_SEED}", env)

    # 2. Held-out sequence set: is the 64-transition result selection or signal?
    evaluate(HERE / "heldout_seed2027_protocol.generated.json", "budget_64",
             "heldout_seed2027", HELDOUT_SEED,
             f"held-out budget_64 @ seed {HELDOUT_SEED}", env)

    if args.skip_sweep:
        record(stage="complete_without_sweep")
        return 0

    # 3. Seed and distortion robustness.
    print("\n=== robustness sweep ===", flush=True)
    record(stage="robustness_sweep")
    result = subprocess.run(
        [PYTHON, "-u", str(HERE / "run_robustness_sweep.py")],
        cwd=REPO, env=env, check=False)
    if result.returncode:
        record(stage="failed", job="robustness_sweep", returncode=result.returncode)
        return result.returncode

    record(stage="complete")
    print("\nphase 2 complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
