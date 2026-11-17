#!/bin/bash
# Replication over independent calibration pairs: tasks A, R x {matched_world, matched_body, mismatched} x seeds 11-13,
# 50 paired episodes each (seed 2000, the same layouts as the main table). Then the box-task diagnostic.
cd "$(dirname "$0")/../../.."; unset CONDA_SHLVL
P=${ISAACLAB_PYTHON:-~/miniconda3/envs/env_isaaclab/bin/python}; X=results/xembody; L=$X/long
run() { $P paper_experiments/xembody/xembody.py --mode eval --episodes 50 --seed 2000 --device cpu --headless --out $X/pairs "$@" 2>&1 | grep "^\[XEMB\] pair\|^\[XEMB\] box"; }
mkdir -p $X/pairs
for s in 11 12 13; do for t in A R; do for m in matched_world matched_body mismatched; do
  run --robot ur10e --task $t --policy $X/policy_planar_$t.pt --map $L/maps_$s/map_$m.json --run_id pair${s}_${t}_$m &
done; done; done; wait
# Box task (B) diagnostic with pair 11: the scripted expert vs the best cloned policy, 4 conditions
for pol in expert $X/policy_planar_B3.pt; do tag=$([ $pol = expert ] && echo expert || echo clone)
  run --robot franka --task B --policy $pol --run_id box_${tag}_franka &
  run --robot ur10e --task B --policy $pol --run_id box_${tag}_direct &
  run --robot ur10e --task B --policy $pol --map $L/maps_11/map_matched_world.json --run_id box_${tag}_matched &
  run --robot ur10e --task B --policy $pol --map $L/maps_11/map_mismatched.json --run_id box_${tag}_mismatched &
done; wait
echo PAIRS_DONE
