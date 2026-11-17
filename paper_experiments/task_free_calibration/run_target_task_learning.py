#!/usr/bin/env python3
"""Target-task learning from a pretrained, pre-CALVIN policy.

Question: does task-free calibration make *learning* CALVIN under an interface
mismatch as sample-efficient as learning it with the correct interface?

This is a different experiment from run_demo_efficiency.py. There, every arm
resumed from a CALVIN-competent checkpoint, so demonstrations only ever paid for
interface adaptation. Here the base policy has never seen CALVIN, so the
demonstrations must teach the task, and the arms differ in whether the interface
is repaired *before* that learning happens.

  clean_ft     correct interface                 how many demos to learn CALVIN
  hidden_ft    distorted, no repair              cost of learning task+interface
  calib_ft     distorted, fitted calibrator      does repair remove that cost?
  oracle_ft    distorted, exact inverse          upper bound on repair

Recipe is the one that actually produced the 87% CALVIN-native policy
(final_results/scripts/run_calvin_native_oft_finetune.sh): LoRA-32 with the VLA
trainable, lr 5e-4, proprio on, image augmentation. The earlier sweeps used
``--train_vla False`` (action head only), which cannot acquire a new task at all
and is why every from-scratch cell floored.

Step budget is 20000 rather than the original 40000. The native run's validation
loss was 0.16145 at 20k and 0.16048 at 35k -- the last 15000 steps bought 0.6%,
so 20k captures essentially all the learning at half the cost.

``--merge_lora_during_training True`` so that every saved checkpoint is a
servable model; the original run left adapters only, which is why its
intermediate checkpoints cannot be evaluated.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
PKG = REPO / "paper_experiments/task_free_calibration"
FINETUNE = WORKSPACE / "openvla-oft/vla-scripts/finetune.py"
RUN_ROOT = WORKSPACE / "final_results/target_task_learning_runs"
LOGS = WORKSPACE / "final_results/target_task_learning_logs"
RUNNER = REPO / "paper_experiments/calvin_baselines/run_protocol.py"
STATE = REPO / "results/task_free_calibration/target_task_learning_state.json"
CONDA = "/home/alam/miniconda3/bin/conda"
PYTHON = "/home/alam/miniconda3/bin/python"

BASE_MODEL = "moojink/openvla-7b-oft-finetuned-libero-spatial"  # pretrained, pre-CALVIN
CALIBRATOR = REPO / "data/task_free_calibration/calibrator_seed101.json"
RELABEL_STATS = REPO / "data/task_free_calibration/dataset_statistics_libero_convention.json"
BUDGETS = [50, 100, 250, 500, 1000]
MAX_STEPS = 20000
SAVE_FREQ = 5000
VAL_FREQ = 2500
DECAY_AT = 15000


def run_id(arm: str, n: int) -> str:
    return f"ttl_{arm}_N{n}"


def record(**kw) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    hist = json.loads(STATE.read_text()) if STATE.exists() else []
    hist.append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw})
    STATE.write_text(json.dumps(hist, indent=2) + "\n")


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PROMPT_MODIFIER",
                "CONDA_SHLVL", "_CE_CONDA", "_CE_M"):
        env.pop(key, None)
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("WANDB_SILENT", "true")
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def checkpoints(arm: str, n: int) -> list[Path]:
    rid = run_id(arm, n)
    return sorted(RUN_ROOT.glob(f"{rid}--*_chkpt"),
                  key=lambda p: int(p.name.split("--")[-1].split("_")[0]))


def train(arm: str, n: int, env: dict[str, str]) -> bool:
    if checkpoints(arm, n):
        print(f"skip (trained): {run_id(arm, n)}", flush=True)
        return True
    LOGS.mkdir(parents=True, exist_ok=True)
    e = dict(env)
    e["OPENVLA_MAX_TRAIN_TRAJECTORIES"] = str(n)
    if arm == "calib_ft":
        # Keep the policy in its pretrained (LIBERO) action convention and let
        # the task-free calibrator convert at deployment, so fine-tuning only
        # has to close the visual/task gap. The statistics must accompany the
        # relabelling: the source convention is a wider action space, and
        # normalising it with CALVIN's q01/q99 saturates 38% of the targets.
        e["OPENVLA_CALIBRATOR_RELABEL_PATH"] = str(CALIBRATOR)
        e["OPENVLA_RELABEL_STATISTICS_PATH"] = str(RELABEL_STATS)
    cmd = [CONDA, "run", "--no-capture-output", "-n", "openvla-oft", "torchrun",
           "--standalone", "--nnodes=1", "--nproc-per-node=1", str(FINETUNE),
           "--vla_path", BASE_MODEL,
           "--data_root_dir", str(WORKSPACE / "datasets/rlds"),
           "--dataset_name", "calvin_task_d",
           "--run_root_dir", str(RUN_ROOT),
           "--use_l1_regression", "True", "--use_diffusion", "False",
           "--use_film", "False", "--num_images_in_input", "2",
           "--use_proprio", "True",
           "--batch_size", "1", "--grad_accumulation_steps", "8", "--seed", "0",
           "--learning_rate", "5e-4", "--lr_warmup_steps", "1000",
           "--num_steps_before_decay", str(DECAY_AT),
           "--max_steps", str(MAX_STEPS + 1),
           "--use_val_set", "True", "--val_freq", str(VAL_FREQ),
           "--val_time_limit", "180", "--save_freq", str(SAVE_FREQ),
           "--save_latest_checkpoint_only", "False", "--image_aug", "True",
           "--use_lora", "True", "--lora_rank", "32",
           # Without this the checkpoints are adapters only and cannot be served.
           "--merge_lora_during_training", "True",
           "--run_id_override", run_id(arm, n)]
    print(f"\n=== TRAIN {run_id(arm, n)}  ({n} demos, {MAX_STEPS} steps) ===", flush=True)
    record(stage="train", arm=arm, n=n)
    log = LOGS / f"{run_id(arm, n)}.log"
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=WORKSPACE / "openvla-oft", env=e,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc:
        print(f"FAILED training {run_id(arm, n)} (code {rc}) - see {log}", flush=True)
        record(stage="train_failed", arm=arm, n=n, returncode=rc)
        return False
    got = [p.name.split("--")[-1] for p in checkpoints(arm, n)]
    if not got:
        print(f"FAILED {run_id(arm, n)}: training returned 0 but wrote no checkpoint",
              flush=True)
        record(stage="no_checkpoint", arm=arm, n=n)
        return False
    print(f"  checkpoints: {' '.join(got)}", flush=True)
    return True


def protocol(arm: str, n: int, ckpt: Path, tag: str) -> Path:
    base = json.loads((PKG / "controlled_interface_protocol.json").read_text())
    m = json.loads(json.dumps(base["methods"]["native_clean"]))
    m["description"] = f"Target-task learning, {arm}, {n} demos, checkpoint {tag}."
    m["model_seeds"] = {"0": str(ckpt)}
    if arm == "calib_ft":
        # The policy was trained in the LIBERO convention, so deployment must
        # convert to CALVIN with the same calibrator used to relabel. Applied
        # client-side; the server-side variable stays unset, because setting
        # both applies the correction twice.
        m["client_environment"] = {
            "CALVIN_CLIENT_CALIBRATION_PATH": str(CALIBRATOR.resolve()),
            "CALVIN_TASK_FREE_CALIBRATION_CODE": str(REPO),
        }
        m["environment"]["CALVIN_TASK_FREE_CALIBRATION_PATH"] = ""
    base["methods"] = {f"{run_id(arm, n)}_{tag}": m}
    base["paths"]["results_root"] = (
        "vla-deployment-benchmark/results/task_free_calibration/target_task_learning")
    out = PKG / f"ttl_{arm}_N{n}_{tag}.generated.json"
    out.write_text(json.dumps(base, indent=2) + "\n")
    return out


def evaluate(arm: str, n: int, env: dict[str, str], num_sequences: int) -> None:
    for ckpt in checkpoints(arm, n):
        tag = ckpt.name.split("--")[-1].split("_")[0]
        method = f"{run_id(arm, n)}_{tag}"
        run = (REPO / "results/task_free_calibration/target_task_learning"
               / f"{method}__model-seed-0__n-{num_sequences}")
        if (run / "results.json").exists():
            man = run / "manifest.json"
            if man.exists() and json.loads(man.read_text()).get("status") == "complete":
                print(f"skip (evaluated): {method}", flush=True)
                continue
        proto = protocol(arm, n, ckpt, tag)
        print(f"\n=== EVAL {method} ===", flush=True)
        record(stage="eval", arm=arm, n=n, step=tag)
        subprocess.run([PYTHON, "-u", str(RUNNER), "--protocol", str(proto),
                        "--workspace", str(WORKSPACE), "--method", method,
                        "--model-seed", "0", "--num-sequences", str(num_sequences),
                        "--sequence-seed", "1729", "--skip-memory", "--overwrite"],
                       cwd=REPO, env=env, check=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["clean_ft"])
    ap.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    ap.add_argument("--num-sequences", type=int, default=100)
    ap.add_argument("--prune", action="store_true",
                    help="delete non-best checkpoints after evaluating a cell")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"{len(args.arms)*len(args.budgets)} cells: {args.budgets} x {args.arms}")
    print(f"base model: {BASE_MODEL}")
    print(f"{MAX_STEPS} steps/cell at ~1429 steps/h = ~{MAX_STEPS/1429:.1f} h/cell")
    if args.dry_run:
        for n in args.budgets:
            for arm in args.arms:
                print(f"  {run_id(arm,n):18s} trained={'y' if checkpoints(arm,n) else 'n'}")
        return 0

    env = clean_env()
    for n in args.budgets:
        for arm in args.arms:
            if not train(arm, n, env):
                continue
            evaluate(arm, n, env, args.num_sequences)
            if args.prune:
                keep = checkpoints(arm, n)[-1:]
                for c in checkpoints(arm, n):
                    if c not in keep:
                        shutil.rmtree(c, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
