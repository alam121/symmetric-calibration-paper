#!/bin/bash
# The cross-embodiment experiment: 4 conditions (+ body/body matched) x 2 tasks x 50 paired episodes.
cd "$(dirname "$0")/../../.."; unset CONDA_SHLVL
P=${ISAACLAB_PYTHON:-~/miniconda3/envs/env_isaaclab/bin/python}; X=results/xembody; M=$X/maps
run() { $P paper_experiments/xembody/xembody.py --task $1 --mode eval --policy $X/franka_policy_$1.pt --episodes 50 --seed 2000 --device cpu --headless --run_id main_$1_$2 "${@:3}" 2>&1 | grep "^\[XEMB\] main"; }
for t in A B; do
  run $t franka --robot franka &
  run $t direct --robot ur10e &
  run $t matched --robot ur10e --map $M/map_matched_world.json &
  run $t matched_body --robot ur10e --map $M/map_matched_body.json &
  run $t mismatched --robot ur10e --map $M/map_mismatched.json &
  wait
done
echo MAIN_DONE
