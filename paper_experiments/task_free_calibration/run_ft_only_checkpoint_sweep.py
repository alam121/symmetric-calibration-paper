#!/usr/bin/env python3
"""Evaluate every retained checkpoint of the fine-tune-only arm.

Why this exists: `dataset.py` gates the action relabelling on ``train``, so the
validation split keeps canonical labels while this arm trains on H^-1 a*. Its
validation loss therefore *rewards the checkpoint that learned the interface
remap least*, and the selected cell is effectively arbitrary - the arm came out
non-monotonic (0 / 13 / 0 / 20 / 15) for that reason alone.

Rather than trust an inverted criterion, evaluate all four checkpoints and report
the best per budget. That is test-set selection, so it *overstates* the baseline -
which is the conservative direction for a paper claiming calibration beats it.

Every (budget, step) pair goes to its own results directory, including ones the
main driver already ran, so the whole arm shares one naming scheme and no result
depends on remembering which step a previous run happened to pick.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "paper_experiments/task_free_calibration"
RESULTS = REPO / "results/task_free_calibration/demo_efficiency_ft_only_allckpt"

_spec = importlib.util.spec_from_file_location("rde", PKG / "run_demo_efficiency.py")
rde = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rde)

STEPS = [rde.BASE_STEP + s for s in (1000, 2000, 3000, 4000)]


def protocol(n: int, step: int, run_id: str) -> Path:
    base = json.loads((PKG / "controlled_interface_protocol.json").read_text())
    m = json.loads(json.dumps(base["methods"]["native_hidden_interface"]))
    m["description"] = f"Fine-tune only, {n} demos, labels H^-1 a*, checkpoint {step}."
    m["model_seeds"] = {"0": str(rde.RUN_ROOT / f"demoeff_ft_only_N{n}--{step}_chkpt")}
    m["base_checkpoint"] = rde.BASE_CHECKPOINT
    m["environment"]["CALVIN_USE_PROPRIO"] = "1"
    m["client_environment"] = {
        "CALVIN_DEPLOYMENT_ACTION_MATRIX": json.dumps(rde.H),
        "CALVIN_DEPLOYMENT_GRIPPER_POLARITY": "-1",
    }
    base["fixed_policy_settings"]["CALVIN_USE_PROPRIO"] = "1"
    base["methods"] = {run_id: m}
    base["paths"]["results_root"] = (
        "vla-deployment-benchmark/results/task_free_calibration/"
        "demo_efficiency_ft_only_allckpt")
    out = PKG / f"{run_id}.generated.json"
    out.write_text(json.dumps(base, indent=2) + "\n")
    return out


def main() -> int:
    env = rde.clean_env()
    plan = [(n, s) for n in rde.BUDGETS for s in STEPS]
    print(f"{len(plan)} evaluations: {rde.BUDGETS} x {STEPS}", flush=True)

    for n, step in plan:
        run_id = f"ftonly_N{n}_step{step}"
        out = RESULTS / f"{run_id}__model-seed-0__n-100"
        if (out / "results.json").exists():
            man = out / "manifest.json"
            if man.exists() and json.loads(man.read_text()).get("status") == "complete":
                print(f"skip (done): {run_id}", flush=True)
                continue
        ckpt = rde.RUN_ROOT / f"demoeff_ft_only_N{n}--{step}_chkpt"
        if not ckpt.exists():
            print(f"MISSING checkpoint, skipping: {ckpt.name}", flush=True)
            continue
        proto = protocol(n, step, run_id)
        print(f"\n=== EVAL {run_id} ===", flush=True)
        subprocess.run(
            [rde.PYTHON, "-u", str(rde.RUNNER), "--protocol", str(proto),
             "--workspace", str(rde.WORKSPACE), "--method", run_id,
             "--model-seed", "0", "--num-sequences", "100",
             "--sequence-seed", "1729", "--skip-memory", "--overwrite"],
            cwd=REPO, env=env, check=False)

    print("\n=== BEST CHECKPOINT PER BUDGET ===", flush=True)
    for n in rde.BUDGETS:
        best = None
        for step in STEPS:
            f = RESULTS / f"ftonly_N{n}_step{step}__model-seed-0__n-100" / "results.json"
            if not f.exists():
                continue
            v = next(iter(json.loads(f.read_text()).values()))
            d1 = 100 * float(v["chain_sr"]["1"])
            if best is None or d1 > best[1]:
                best = (step, d1, float(v["avg_seq_len"]))
        if best:
            print(f"  N={n:2d}  best step {best[0]}  D1 {best[1]:.1f}%  avg {best[2]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
