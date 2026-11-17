#!/bin/bash
# Item 3: UR10e policies from N target demos (fine-tuned from the Franka policy, or from scratch), no map.
cd "$(dirname "$0")/../../.."; unset CONDA_SHLVL
X=results/xembody; F=$X/finetune
for kind in ft scratch; do for t in A R; do for n in 1 5 10 25 50; do echo "$kind $t $n"; done; done; done | \
xargs -P 8 -L 1 bash -c '${ISAACLAB_PYTHON:-~/miniconda3/envs/env_isaaclab/bin/python} paper_experiments/xembody/xembody.py --robot ur10e --task $1 --mode eval \
  --policy '$F'/$0_$1_n$2.pt --episodes 50 --seed 2000 --device cpu --headless --out '$F' --run_id ftev_$0_$1_n$2 2>&1 | grep "^\[XEMB\] ftev_"'
echo FINETUNE_DONE
