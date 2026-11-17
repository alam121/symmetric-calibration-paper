#!/bin/bash
# usage: run_task.sh TASK POLICY.pt TAG -- all conditions, 50 paired episodes (seed 2000)
cd "$(dirname "$0")/../../.."; unset CONDA_SHLVL
P=${ISAACLAB_PYTHON:-~/miniconda3/envs/env_isaaclab/bin/python}; X=results/xembody; M=$X/maps; t=$1; pt=$2; tag=$3
run() { $P paper_experiments/xembody/xembody.py --task $t --mode eval --policy $pt --episodes 50 --seed 2000 --device cpu --headless --run_id ${tag}_${t}_$1 "${@:2}" 2>&1 | grep "^\[XEMB\] ${tag}"; }
run franka --robot franka &
run direct --robot ur10e &
run matched --robot ur10e --map $M/map_matched_world.json &
run matched_body --robot ur10e --map $M/map_matched_body.json &
run mismatched --robot ur10e --map $M/map_mismatched.json &
wait; echo "DONE $tag $t"
