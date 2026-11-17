#!/bin/bash
cd "$(dirname "$0")/../../.."; unset CONDA_SHLVL
P=${ISAACLAB_PYTHON:-~/miniconda3/envs/env_isaaclab/bin/python}; X=results/xembody
run() { $P paper_experiments/xembody/xembody.py --task A --mode eval --policy $X/policy_tcpfull_A.pt --episodes 50 --seed 2000 --device cpu --headless --out $X/env "$@" 2>&1 | grep "^\[XEMB\] tcpfull"; }
for d in 0 3 6 9; do sh=$(echo "scale=3; $d/100" | bc)
  run --robot franka --env_shift $sh --run_id tcpfull_A_shift${d}_franka &
  run --robot ur10e --env_shift $sh --map $X/long/maps_11/map_matched_world.json --run_id tcpfull_A_shift${d}_matched &
  run --robot ur10e --env_shift $sh --run_id tcpfull_A_shift${d}_direct &
  run --robot ur10e --env_shift $sh --map $X/long/maps_11/map_mismatched.json --run_id tcpfull_A_shift${d}_mismatched &
done; wait; echo TCPFULL_A_DONE
