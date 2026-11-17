#!/bin/bash
# Item 2: calibration budget sweep. Maps from the first N transitions of each robot's long run (3 pairs).
cd "$(dirname "$0")/../../.."; unset CONDA_SHLVL
X=results/xembody; B=$X/budget
for s in 11 12 13; do for n in 16 32 64 128 300 1500; do for t in A R; do echo "$s $n $t"; done; done; done | \
xargs -P 14 -L 1 bash -c 'pol=$([ $2 = A ] && echo '$X'/relative/policy_A_V1_relative.pt || echo '$X'/policy_tcpfull_R.pt); \
  ${ISAACLAB_PYTHON:-~/miniconda3/envs/env_isaaclab/bin/python} paper_experiments/xembody/xembody.py --robot ur10e --task $2 --mode eval --policy $pol \
  --map '$B'/map_s$0_n$1.json --episodes 50 --seed 2000 --device cpu --headless --out '$B' --run_id bud_$2_s$0_n$1 2>&1 | grep "^\[XEMB\] bud_"'
echo BUDGET_DONE
