#!/usr/bin/env python3
"""Demonstration-efficiency sweep: how many demos match zero-demo calibration?

Every arm starts from the CALVIN-competent policy (see BASE_CHECKPOINT), so the
demonstrations are spent only on adapting to the hidden interface H, never on
learning the task. That makes the arms directly comparable to the headline
zero-demonstration calibration result on the same policy.

Three arms at each budget, trained on exactly the same trajectories:

  A  fine-tune only            labels H^-1 a*   deployed under H, no calibration
                               -> the demos must teach the interface
  B  calibration + fine-tune   labels a*        deployed under H with the fitted
                               calibrator
  C  oracle inverse + fine-tune  B's checkpoint, deployed with the exact
                               analytic inverse -> ceiling on interface repair,
                               isolating the deployment-time correction

Budgets are nested prefixes of a deterministic TFDS split slice, so the arms at
a given N see identical demonstrations and only the action representation
differs. C shares B's checkpoint by construction and is asserted to.
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
FINETUNE = WORKSPACE / "openvla-oft/vla-scripts/finetune.py"
RUN_ROOT = WORKSPACE / "final_results/demo_efficiency_runs"
LOGS = WORKSPACE / "final_results/demo_efficiency_logs"
RUNNER = REPO / "paper_experiments/calvin_baselines/run_protocol.py"
STATE = REPO / "results/task_free_calibration/demo_efficiency_state.json"
CONDA = "/home/alam/miniconda3/bin/conda"
PYTHON = "/home/alam/miniconda3/bin/python"

# The hidden deployment interface for this study (power-of-two signed
# permutation, so an exact inverse exists in float32).
H = [[0, -2, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0],
     [0, 0, 0, 0, -4, 0], [0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 0, 8]]

BUDGETS = [1, 5, 10, 25, 50]
# FIXED step budget across every demonstration count.
#
# The first version of this sweep scaled steps with N (1k -> 14k), which varied
# two things at once: more demonstrations helps, more steps on a tiny dataset
# overfits. The result was non-monotonic (0 / 6 / 18 / 8 / 0 percent) and
# uninterpretable. With the budget fixed, demonstration count is the only
# independent variable.
FIXED_STEPS = 4000
STEPS = {n: FIXED_STEPS for n in BUDGETS}
# Save periodically and keep a validation set so a checkpoint can be selected on
# validation loss rather than blindly taking the last one, matching the practice
# in select_calvin_native_oft_checkpoint.py.
SAVE_EVERY = 1000

# Start from the CALVIN-competent policy, not from LIBERO.
#
# The first fixed-step sweep fine-tuned from
# ``moojink/openvla-7b-oft-finetuned-libero-spatial``, so the demonstrations had
# to teach the CALVIN task *and* the interface at once. That is hopeless at
# N <= 50: the reference policy needed 35k LoRA steps over the full dataset, and
# every arm floored at 0-5% with the oracle pinned there too, which proves the
# interface was never the binding constraint. Starting from the same checkpoint
# the headline result uses means the demonstrations only ever have to teach the
# interface, and the arms become directly comparable to the zero-demonstration
# calibration number.
BASE_CHECKPOINT = str(
    WORKSPACE / "final_results/calvin_native_oft_runs"
    / "calvin_task_d_native_oft_lora32--35000_chkpt")
# finetune.py offsets logged/saved step numbers by ``resume_step`` when resuming,
# so this run's checkpoints are named 36000..39000, not 1000..4000.
BASE_STEP = 35000

# The two repair arms must use DIFFERENT calibrators.
#
# In the first run of this sweep they did not: both branches pointed at the
# bit-exact file, so `oracle_ft` was a byte-identical duplicate of `calib_ft`.
# Their zero discordance was guaranteed by construction, not measured, and the
# sweep contained no fitted-calibration arm at all. `calib_ft` must use a
# calibrator estimated from task-free interaction (r2 < 1, no demonstrations),
# `oracle_ft` the exact analytic inverse, so the pair measures what estimation
# costs.
FITTED_CALIBRATOR = REPO / "data/task_free_calibration/calibrator_fitted_pow2_seed201.json"
ORACLE_CALIBRATOR = REPO / "data/task_free_calibration/calibrator_bitexact_pow2.json"


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PROMPT_MODIFIER",
                "CONDA_SHLVL", "_CE_CONDA", "_CE_M"):
        env.pop(key, None)
    return env


def gpu_free_mib() -> int:
    """Free GPU memory. Training needs the card to itself."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True)
    try:
        return int(out.stdout.strip().split("\n")[0])
    except Exception:
        return 0


def require_free_gpu(need_mib: int = 20000, tries: int = 20) -> bool:
    """Refuse to train while another process holds the card.

    An orphaned policy server once held 15.8 GB for an hour, and every training
    run in the queue OOM'd at startup and was recorded as a failure. Waiting is
    always better than burning the queue.
    """
    for _ in range(tries):
        free = gpu_free_mib()
        if free >= need_mib:
            return True
        holders = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader"], capture_output=True, text=True)
        print(f"GPU busy: only {free} MiB free, need {need_mib}. Holders: "
              f"{holders.stdout.strip() or 'unknown'}. Waiting...", flush=True)
        time.sleep(60)
    print("GPU never freed; aborting rather than failing every job.", flush=True)
    return False


def record(**payload: object) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(
        {"updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **payload}, indent=2) + "\n")


def run_id(arm: str, n: int) -> str:
    return f"demoeff_{arm}_N{n}"


def _validation_losses(run_file: Path) -> dict[int, float]:
    """Read {step: val loss} from an offline W&B datastore.

    Same reader as final_results/scripts/select_calvin_native_oft_checkpoint.py,
    which is the established practice in this project.
    """
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    store = DataStore()
    store.open_for_scan(str(run_file))
    losses: dict[int, float] = {}
    while True:
        blob = store.scan_data()
        if blob is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(blob)
        if not record.HasField("history"):
            continue
        row = {i.key: json.loads(i.value_json) for i in record.history.item}
        if "VLA Val/Loss" in row and "_step" in row:
            losses[int(row["_step"])] = float(row["VLA Val/Loss"])
    return losses


def _best_step(rid: str) -> int | None:
    """Step with the lowest validation loss for this run, if recorded."""
    wandb_root = WORKSPACE / "openvla-oft/wandb"
    best: tuple[float, int] | None = None
    for run_dir in sorted(wandb_root.glob("offline-run-*"),
                          key=lambda d: d.stat().st_mtime, reverse=True):
        meta = run_dir / "files" / "wandb-metadata.json"
        if not meta.exists():
            continue
        argv = json.loads(meta.read_text()).get("args", [])
        if "--run_id_override" not in argv or argv[argv.index("--run_id_override") + 1] != rid:
            continue
        # A run id can appear more than once across sweep versions. Only accept
        # a run whose max-steps matches this sweep, otherwise an older run with
        # a different budget can win and two arms that must share a checkpoint
        # end up on different ones.
        if "--max_steps" in argv:
            try:
                if int(argv[argv.index("--max_steps") + 1]) != BASE_STEP + FIXED_STEPS + 1:
                    continue
            except (ValueError, IndexError):
                pass
        # The LIBERO-based sweep used the same run ids and the same step budget,
        # so max-steps alone cannot separate it from this one. The base
        # checkpoint can.
        if "--vla_path" not in argv or argv[argv.index("--vla_path") + 1] != BASE_CHECKPOINT:
            continue
        files = list(run_dir.glob("run-*.wandb"))
        if not files:
            continue
        losses = _validation_losses(files[0])
        # Never select a checkpoint from beyond the step budget. The fixed
        # budget is the whole point of this sweep, so a run that overshot it
        # must not be rescued by picking one of its later checkpoints.
        losses = {s: v for s, v in losses.items()
                  if BASE_STEP < s <= BASE_STEP + FIXED_STEPS}
        if losses:
            step, loss = min(losses.items(), key=lambda kv: kv[1])
            best = (loss, step)
            break
    return best[1] if best else None


def checkpoint_dir(arm: str, n: int) -> Path:
    """Checkpoint to evaluate: the one with the lowest validation loss.

    A fixed step budget removes the confound between demonstration count and
    training duration, but it cannot be right for every budget at once - 4000
    steps is many more epochs over 1 demonstration than over 50. Selecting on
    validation loss lets each budget stop where it should, which is what the
    first version of this sweep lacked and why its curve was non-monotonic.
    Falls back to the final checkpoint when no validation history exists.
    """
    train_arm = "calib_ft" if arm == "oracle_ft" else arm
    rid = run_id(train_arm, n)
    step = _best_step(rid)
    if step is not None:
        selected = RUN_ROOT / f"{rid}--{step}_chkpt"
        if selected.exists():
            return selected
    return RUN_ROOT / f"{rid}--{BASE_STEP + STEPS[n]}_chkpt"


def train(arm: str, n: int, env: dict[str, str]) -> bool:
    out = checkpoint_dir(arm, n)
    if out.exists():
        print(f"skip (trained): {run_id(arm, n)}", flush=True)
        return True
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{run_id(arm, n)}.log"
    e = dict(env)
    # The original training runs logged to wandb in offline mode; without this
    # the run aborts at startup asking for an API key on a non-tty.
    e.setdefault("WANDB_MODE", "offline")
    e.setdefault("WANDB_SILENT", "true")
    e["OPENVLA_MAX_TRAIN_TRAJECTORIES"] = str(n)
    if arm == "ft_only":
        # Arm A: express the demonstrations in the target robot's convention.
        e["OPENVLA_ACTION_RELABEL_MATRIX"] = json.dumps(H)
    else:
        # calib_ft and oracle_ft share canonical labels and therefore share a
        # checkpoint; only the deployment-time correction differs.
        e.pop("OPENVLA_ACTION_RELABEL_MATRIX", None)
    if not require_free_gpu():
        return False
    steps = STEPS[n]
    cmd = [CONDA, "run", "--no-capture-output", "-n", "openvla-oft", "torchrun",
           "--standalone", "--nnodes=1", "--nproc-per-node=1", str(FINETUNE),
           "--vla_path", BASE_CHECKPOINT,
           # Loads action_head--35000 and proprio_projector--35000 from the base
           # checkpoint. Without this the backbone would be CALVIN-competent but
           # the action head randomly initialised, which floors the run.
           "--resume", "True", "--resume_step", str(BASE_STEP),
           "--data_root_dir", str(WORKSPACE / "datasets/rlds"),
           "--dataset_name", "calvin_task_d",
           "--run_root_dir", str(RUN_ROOT),
           "--use_l1_regression", "True", "--use_diffusion", "False", "--use_film", "False",
           # The CALVIN-native policy was trained with proprio (dim 15); the
           # checkpoint carries a proprio projector and the head expects it.
           "--num_images_in_input", "2", "--use_proprio", "True",
           "--train_vla", "False",              # action head only
           "--batch_size", "1", "--grad_accumulation_steps", "8", "--seed", "0",
           "--learning_rate", "5e-4", "--lr_warmup_steps", "200",
           # finetune.py stops on `log_step == max_steps`, and under --resume
           # log_step is offset by resume_step. Passing the un-offset 4001 makes
           # that equality unreachable and training never terminates: the first
           # attempt ran 22k steps before it was caught. The LR milestone is
           # counted by the scheduler itself and stays relative.
           "--num_steps_before_decay", str(steps),
           "--max_steps", str(BASE_STEP + steps + 1),
           "--use_val_set", "True", "--val_freq", str(SAVE_EVERY),
           "--val_time_limit", "120", "--save_freq", str(SAVE_EVERY),
           "--save_latest_checkpoint_only", "False", "--image_aug", "True",
           "--use_lora", "False", "--merge_lora_during_training", "False",
           "--run_id_override", run_id(arm, n)]
    print(f"\n=== TRAIN {run_id(arm, n)}  ({n} demos, {steps} steps) ===", flush=True)
    record(stage="train", arm=arm, n=n, steps=steps)
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=WORKSPACE / "openvla-oft", env=e,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc:
        print(f"FAILED training {run_id(arm, n)} (code {rc}) — see {log}", flush=True)
        record(stage="train_failed", arm=arm, n=n, returncode=rc)
        return False
    return True


def make_protocol(arm: str, n: int) -> Path:
    base = json.loads((PKG / "controlled_interface_protocol.json").read_text())
    ckpt = str(checkpoint_dir(arm, n))
    client = {"CALVIN_DEPLOYMENT_ACTION_MATRIX": json.dumps(H),
              "CALVIN_DEPLOYMENT_GRIPPER_POLARITY": "-1"}
    if arm == "ft_only":
        m = json.loads(json.dumps(base["methods"]["native_hidden_interface"]))
        m["description"] = f"Fine-tune only, {n} demos, labels H^-1 a*, no calibration."
    elif arm == "oracle_ft":
        m = json.loads(json.dumps(base["methods"]["native_task_free_calibrated"]))
        m["description"] = f"Exact analytic inverse + fine-tune, {n} demos."
        m["environment"]["CALVIN_TASK_FREE_CALIBRATION_PATH"] = str(
            ORACLE_CALIBRATOR.resolve())
    else:
        m = json.loads(json.dumps(base["methods"]["native_task_free_calibrated"]))
        m["description"] = (
            f"Task-free fitted calibration + fine-tune, {n} demos, canonical labels.")
        m["environment"]["CALVIN_TASK_FREE_CALIBRATION_PATH"] = str(
            FITTED_CALIBRATOR.resolve())
    m["model_seeds"] = {"0": ckpt}
    m["base_checkpoint"] = BASE_CHECKPOINT
    m["environment"]["CALVIN_USE_PROPRIO"] = "1"
    m["client_environment"] = client
    base["fixed_policy_settings"]["CALVIN_USE_PROPRIO"] = "1"
    base["methods"] = {run_id(arm, n): m}
    base["paths"]["results_root"] = "vla-deployment-benchmark/results/task_free_calibration/demo_efficiency"
    out = PKG / f"demoeff_{arm}_N{n}.generated.json"
    out.write_text(json.dumps(base, indent=2) + "\n")
    return out


def evaluate(arm: str, n: int, env: dict[str, str]) -> None:
    run = (REPO / "results/task_free_calibration/demo_efficiency"
           / f"{run_id(arm, n)}__model-seed-0__n-100")
    if (run / "results.json").exists():
        manifest = run / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("status") == "complete":
            print(f"skip (evaluated): {run_id(arm, n)}", flush=True)
            return
    proto = make_protocol(arm, n)
    ckpt = checkpoint_dir(arm, n).name
    if arm == "oracle_ft":
        # The oracle arm exists to isolate the deployment-time correction, so it
        # must evaluate the identical checkpoint as calib_ft at this budget.
        paired = checkpoint_dir("calib_ft", n).name
        if ckpt != paired:
            print(f"REFUSING {run_id(arm, n)}: checkpoint {ckpt} != calib_ft {paired}",
                  flush=True)
            record(stage="checkpoint_mismatch", arm=arm, n=n, got=ckpt, expected=paired)
            return
    print(f"\n=== EVAL {run_id(arm, n)}  [checkpoint {ckpt}] ===", flush=True)
    record(stage="eval", arm=arm, n=n)
    subprocess.run([PYTHON, "-u", str(RUNNER), "--protocol", str(proto),
                    "--workspace", str(WORKSPACE), "--method", run_id(arm, n),
                    "--model-seed", "0", "--num-sequences", "100",
                    "--sequence-seed", "1729", "--skip-memory", "--overwrite"],
                   cwd=REPO, env=env, check=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    ap.add_argument("--arms", nargs="+", default=["calib_ft", "ft_only", "oracle_ft"],
                    help="calib_ft = fitted calibrator, ft_only = no calibration "
                         "(labels H^-1 a*), oracle_ft = exact analytic inverse")
    args = ap.parse_args()

    plan = [(arm, n) for n in args.budgets for arm in args.arms]
    print(f"{len(plan)} cells: {args.budgets} x {args.arms}")
    total = sum(STEPS[n] for n in args.budgets) * 2
    print(f"total training steps {total:,}  (~{total/2950:.1f} h at 2,950 steps/h)")
    for arm, n in plan:
        print(f"  {run_id(arm, n):22s} steps={STEPS[n]:6d}  trained={'y' if checkpoint_dir(arm,n).exists() else 'n'}")
    if args.dry_run:
        return 0

    env = clean_env()
    for arm, n in plan:
        if not train(arm, n, env):
            print("stopping: a training failure usually means the environment is wrong, "
                  "and continuing would fail every remaining cell identically.", flush=True)
            return 1
        evaluate(arm, n, env)
        time.sleep(15)
    record(stage="complete")
    print("\ndemo-efficiency sweep complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
