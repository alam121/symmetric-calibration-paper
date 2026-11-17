#!/bin/bash
# Environment decomposition: layout shifted 0/3/6/9 cm out of the demo region; Franka vs UR10e (matched, pair 11).
cd "$(dirname "$0")/../../.."; unset CONDA_SHLVL
P=${ISAACLAB_PYTHON:-~/miniconda3/envs/env_isaaclab/bin/python}; X=results/xembody; mkdir -p $X/env
run() { $P paper_experiments/xembody/xembody.py --mode eval --episodes 50 --seed 2000 --device cpu --headless --out $X/env "$@" 2>&1 | grep "^\[XEMB\] env"; }
for t in A R; do for d in 0 3 6 9; do sh=$(echo "scale=3; $d/100" | bc)
  run --robot franka --task $t --policy $X/policy_planar_$t.pt --env_shift $sh --run_id env_${t}_shift${d}_franka &
  run --robot ur10e --task $t --policy $X/policy_planar_$t.pt --env_shift $sh --map $X/long/maps_11/map_matched_world.json --run_id env_${t}_shift${d}_matched &
done; done; wait; echo ENV_DONE
