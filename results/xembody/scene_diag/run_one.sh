#!/bin/bash
unset CONDA_SHLVL
name=$1; shift
G=/home/alam/sim2real_runs/xembody/scene_diag
cd /home/alam/manipulatorsdatepalm
XEMB_SAVE_CAM=$G/frames/$name /home/alam/miniconda3/envs/env_isaaclab/bin/python isaac/sim2real/xembody.py --robot franka --task A --mode eval \
  --policy /home/alam/sim2real_runs/xembody/image/policy_img_A.pt --episodes 20 --seed 7000 --device cpu --headless --camera \
  --out $G --run_id $name "$@" > $G/$name.log 2>&1
echo "$name exit $?" >> $G/done.txt
