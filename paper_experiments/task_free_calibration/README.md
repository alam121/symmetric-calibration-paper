# Task-free VLA interface calibration

This experiment asks whether an already-trained, frozen VLA can be deployed in
a new setting after a short, task-independent calibration phase. Calibration
data contain robot transitions only: there are no language instructions, task
IDs, expert actions, success labels, or demonstrations.

## Scientific separation

The existing embodiment bridge is a **supervised baseline** because it was
trained from labelled CALVIN expert trajectories. The proposed calibration
condition may observe the target cameras, robot state, issued exploratory
action, and resulting next observation/state, but never a target task label or
target expert action.

The first dataset is generated with smooth bounded random excitation. Episodes
start from diverse CALVIN scene configurations, but the collector does not load
or execute their associated benchmark tasks.

## Collection

Run inside the CALVIN environment:

```bash
conda run --no-capture-output -n calvin_venv \
  python paper_experiments/task_free_calibration/collect_calibration.py \
  --dataset-path /home/alam/vla_thesis/calvin/dataset/task_D_D \
  --output data/task_free_calibration/calvin_seed0 \
  --episodes 100 --steps-per-episode 64 --seed 101
```

A small execution smoke test should use `--episodes 2 --steps-per-episode 8`.
Each episode is stored separately so an interrupted collection remains usable.
The manifest records the exact schema and explicitly asserts the absence of
forbidden supervision.

## Required comparisons

1. Direct frozen source VLA: no target data.
2. Analytic normalization/interface mapping only.
3. Proposed task-free calibration at 1, 5, 10, and 30 minutes.
4. Supervised action/embodiment bridge using target demonstrations.
5. Target-native OpenVLA-OFT reference.

All final conditions use three independent calibration/training seeds and the
official 1,000-sequence CALVIN protocol. The primary result is the Pareto curve
between target supervision, trainable parameters, calibration duration, and
closed-loop chain success.
