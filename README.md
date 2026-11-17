# Calibrate Both Sides the Same Way

**Task-free interface calibration for moving a frozen robot policy to a new robot.**

A policy outputs commands, not motion. Two robots can accept the same kind of command and still move
differently (different frames, signs, speeds, rotation points). We let each robot make a few minutes of
random motion, fit a linear model of how its commands become motion, and translate the policy's
commands from one robot to the other, with no retraining, demonstrations or task labels.

The main finding is that **what matters is measuring both robots the same way, not measuring
each one accurately.** Errors that the measurement procedure introduces are shared, so they cancel. A
single number, the **alignment α** (the worst-case cosine between the step the policy asks for and the
step the robot takes), predicts success for any translation: calibrated, hand-written or none.

---

## Results at a glance

**Across 98 conditions** (CALVIN, simulation, real robot; all repairs and baselines):

- All 63 conditions with α ≥ 0.6 succeed in at least 82% of episodes.
- 28 of the 29 conditions with α < 0 fail completely, whatever the speed error.
- The 6 conditions in between are mixed.

See `analysis/figures/fig_alignment.pdf`.

| Prediction | Result |
|---|---|
| **P1** Measurement errors are shared and cancel | CALVIN: each single fit misses up to 51% of the speed, yet the two robots' fits agree to 3.7–8.2% |
| **P2** Feedback forgives speed errors, not direction errors | Rescaling the speed changes nothing (85% → 85%); success follows α (above) |
| **P3** Measuring the two robots differently reverses rotation and breaks rotation tasks first | CALVIN: orientation-critical tasks fall 50% → 10%, tolerant ones 98% → 74%. Sim: 0/300. Real: 0/20 |
| **P4** Calibration measures what a hand-written conversion must assume | Hidden frame rotation θ: the hand-written conversion fails from 60°, calibration gets 50/50 at every θ. Real robot with a hidden 180° frame: calibration 10/10, commands unchanged 0/10 |
| **P5** Success saturates once α clears the margin | 64 random steps per robot (6.4 s) give 48–50/50. 50 UR10e demonstrations with no map: 38/50 |
| **P6** Calibration transfers the source's closed loop, failures included | The UR10e fails on the same episodes the Franka fails. The camera (28/50) and box (21/50) gaps remain, and both are explained by what the policy sees or holds |

**CALVIN** (OpenVLA-OFT, hidden interface distortion, 100 sequences):

| Repair | 1st-task success / avg. chain length |
|---|---|
| None (hidden interface) | 2% / 0.02 |
| Task-free fit (ours) | **90% / 2.90** |
| Exact inverse (oracle) | 85% / 2.85 |
| Demo-fitted (supervised) | 85% / 2.66 |
| Matched rotation convention (body/body) | 86% / 2.84 |
| Mismatched convention (world/body) | 75% / 1.14 |
| Undistorted robot (reference) | 87% |

**Franka-trained policy on the simulated UR10e** (Isaac Lab, 50 paired episodes per cell, 3 calibration pairs):

| Translation | Task A (pick & place) | Task R (reach a pose) | α |
|---|---|---|---|
| Franka on itself (ceiling) | 50 | 50 | — |
| Commands unchanged | 0 | 0 | −1.00 |
| Hand-written conversion | 50 | 50 | 0.99 |
| Matched calibration, world/world | 50, 50, 50 | 50, 50, 50 | 1.00 |
| Matched calibration, body/body | 50, 50, 50 | 50, 50, 50 | 0.99–1.00 |
| Mismatched calibration, world/body | 0, 0, 0 | 0, 0, 0 | −1.00 |

That is 600/600 with matched calibration. On task R the UR10e ends 0.8–0.9 mm and 0.10° from the
target (median).

**Real UR10e** (task R, 10 paired targets per cell, operator at the e-stop):

| Translation | `base_link` frame | Hidden 180° frame |
|---|---|---|
| Commands unchanged | 10/10 | 0/10 (left the box) |
| Hand-written, true frame given | 10/10 | 10/10 |
| Matched calibration | 10/10 | 10/10 (1.1 mm median error) |
| Mismatched calibration | 0/10 (tilt stop) | 0/10 (tilt stop) |

---

## Repository layout

```
analysis/
  alignment.py               G, alpha, gain, direction error for every condition -> alignment.json, fig_alignment
  make_figures.py            fig_shift (layout shifts), fig_real (real UR10e trials)
  figures/                   the generated plots

paper_experiments/
  task_free_calibration/     CALVIN: excitation, calibrator (fit), evaluation drivers, analyses, protocols
  ur10e_hardware/            real UR10e: calibration run, fit_compare.py (fit and compare maps), trials
  xembody/                   Isaac Lab Franka -> UR10e cell: xembody.py (collect / calibrate / eval),
                             xembody_map.py (build a translation map), xembody_train.py (BC policy),
                             policy.py (scripted expert), runs/ (the exact scripts behind each table)

results/
  task_free_calibration/     CALVIN evaluation results (JSON; per-run logs and checkpoints not included)
  xembody/                   simulation + real-robot results
    long/                    calibration runs (3 seeds per robot) used for the main maps
    pairs/ budget/ analytic/ env/ relative/ regional/ baselink/ finetune/ scene_diag/ ...
    real/                    real UR10e: calibration (real/calib), trials, trial scripts, summaries
    *_summary.txt            human-readable summaries of each experiment
    *.pt, franka_demos_*.npz trained Franka policies and demonstrations
```

---

## How to run

### 1. Reproduce the numbers and plots from the saved results (no simulator needed)

Needs Python 3 with `numpy`, `scipy` and `matplotlib`.

```bash
python3 analysis/alignment.py      # prints gain / direction / alpha / success for every condition,
                                   # writes analysis/alignment.json and analysis/figures/fig_alignment.{pdf,png}
python3 analysis/make_figures.py   # analysis/figures/fig_shift.*, fig_real.*
```

`alignment.py` refits every robot interface from the raw calibration logs in `results/xembody`, so
this also checks the fitting code end to end. It takes a few minutes.

Fit and compare two calibration runs yourself:

```bash
python3 paper_experiments/xembody/xembody_map.py \
    --franka results/xembody/long/franka_long_seed11.jsonl \
    --ur10e  results/xembody/long/ur10e_long_seed11.jsonl --out /tmp/map.json
```

Each experiment also has a plain-text summary:

```bash
cat results/xembody/replication_summary.txt     # matched / mismatched over 3 calibration pairs
cat results/xembody/budget_summary.txt          # how much calibration data
cat results/xembody/real/trials_summary.txt     # real UR10e, base_link frame
cat results/xembody/real/trials_frame180_summary.txt   # real UR10e, hidden 180 deg frame
```

### 2. Rerun the simulation experiments (Isaac Lab)

Requirements:
- Isaac Sim with Isaac Lab, in a conda env. The scripts use `$ISAACLAB_PYTHON`, which defaults to
  `~/miniconda3/envs/env_isaaclab/bin/python`.
- The UR10e + Delto hand USD asset. Set `UR10E_USD=/path/to/ur10e_delto.usd`. The asset is not
  included, and the Franka comes with Isaac Lab.
- Run `unset CONDA_SHLVL` first, otherwise the Isaac process can start in the wrong conda env.

```bash
export ISAACLAB_PYTHON=~/miniconda3/envs/env_isaaclab/bin/python
unset CONDA_SHLVL

# one quick check: the scripted expert on the Franka
$ISAACLAB_PYTHON paper_experiments/xembody/xembody.py --robot franka --task A --mode eval --policy expert --episodes 5 --headless

# task-free calibration of each robot (random motion, no task)
$ISAACLAB_PYTHON paper_experiments/xembody/xembody.py --robot franka --task R --mode calibrate --seed 11 --steps 1500 --headless
$ISAACLAB_PYTHON paper_experiments/xembody/xembody.py --robot ur10e  --task R --mode calibrate --seed 11 --steps 1500 --headless

# Franka-trained policy on the UR10e through a calibrated map
$ISAACLAB_PYTHON paper_experiments/xembody/xembody.py --robot ur10e --task R --mode eval \
    --policy results/xembody/policy_tcpfull_R.pt --map results/xembody/maps/map_matched_world.json \
    --episodes 50 --seed 2000 --device cpu --headless
```

The exact commands behind each table are in `paper_experiments/xembody/runs/` and can be run from
anywhere. They write into `results/xembody/`, so the saved results get overwritten.

| Script | Experiment |
|---|---|
| `run_main.sh`, `run_pairs.sh` | Franka → UR10e main table, 3 calibration pairs (P2, P3) |
| `run_analytic.sh` | hand-written conversion vs calibration under a hidden frame rotation θ (P4) |
| `run_budget.sh` | how much calibration data (P5) |
| `run_finetune.sh` | baseline: UR10e demonstrations, no map (P5) |
| `run_env.sh`, `run_relative*.sh`, `run_tcpfull*.sh` | layout shifts and what the policy observes (P6) |
| `run_regional.sh`, `run_baselink.sh` | local (regional) maps, frame checks |

### 3. CALVIN experiments

These need the CALVIN benchmark and an OpenVLA-OFT CALVIN checkpoint (GPU). The pipeline is in
`paper_experiments/task_free_calibration/`, and its [README](paper_experiments/task_free_calibration/README.md)
explains the protocol:

- `collect_calibration.py`: task-free random excitation in CALVIN (no tasks, no labels).
- `calibrator.py`, `fit_calibrator.py`: fit the linear interface model and its world/body rotation conventions.
- `run_*.py`: evaluation drivers. The `*protocol*.json` files fix every condition and seed.
- `analyse_gain_vs_direction.py`, `analyse_nuisance_operator.py`, `analyse_orientation_probe.py`,
  `paired_tests.py`: the analyses behind the CALVIN tables.

Saved CALVIN results are in `results/task_free_calibration/`.

### 4. Real UR10e

`paper_experiments/ur10e_hardware/` holds the hardware calibration (`ur10e_calibration_run.py`), the fit
and comparison (`fit_compare.py`) and the paired trial sheets (`trials.py`). See its
[README](paper_experiments/ur10e_hardware/README.md) for the safety limits.

`results/xembody/real/` has the calibration logs, every trial, the run sheets and the trial scripts
(`run_trial.py`, `run_series.sh`). These scripts need ROS 2 Humble, a FastDDS profile for your own
network (`fastdds_robot.xml`, not included) and a person at the e-stop.

---

## Status

- α was computed after the experiments were run (post hoc), not pre-registered.
- Per-episode logs, videos, model checkpoints for CALVIN and the large image-policy runs are not in this
  repository (too large).
