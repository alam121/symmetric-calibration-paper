#!/bin/bash
cd "$(dirname "$0")/../../.."; unset CONDA_SHLVL
P=${ISAACLAB_PYTHON:-~/miniconda3/envs/env_isaaclab/bin/python}; X=results/xembody; R=$X/relative
run() { $P paper_experiments/xembody/xembody.py --task A --mode eval --episodes 50 --seed 2000 --device cpu --headless --out $R "$@" 2>&1 | grep "^\[XEMB\] rel"; }
for v in V1_relative V3_relgrid; do for d in 0 3 6 9; do sh=$(echo "scale=3; $d/100" | bc)
  run --robot ur10e --policy $R/policy_A_$v.pt --env_shift $sh --run_id rel_${v}_shift${d}_direct &
  run --robot ur10e --policy $R/policy_A_$v.pt --env_shift $sh --map $X/long/maps_11/map_mismatched.json --run_id rel_${v}_shift${d}_mismatched &
done; done; wait
for d in 12 15; do sh=$(echo "scale=3; $d/100" | bc)
  run --robot franka --policy expert --env_shift $sh --run_id rel_expert_shift${d}_franka &
  for v in V1_relative V3_relgrid; do
    run --robot franka --policy $R/policy_A_$v.pt --env_shift $sh --run_id rel_${v}_shift${d}_franka &
    run --robot ur10e --policy $R/policy_A_$v.pt --env_shift $sh --map $X/long/maps_11/map_matched_world.json --run_id rel_${v}_shift${d}_matched &
  done
done; wait; echo RELATIVE2_DONE
