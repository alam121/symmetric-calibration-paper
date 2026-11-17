#!/bin/bash
# Item 1: documented-interface (analytic) baseline vs task-free calibration, as the UR10e's command frame gets an
# undocumented rotation theta; plus an analytic map with the wrong tool offset.
cd "$(dirname "$0")/../../.."; unset CONDA_SHLVL
P=${ISAACLAB_PYTHON:-~/miniconda3/envs/env_isaaclab/bin/python}; X=results/xembody; G=$X/analytic
PA=$X/relative/policy_A_V1_relative.pt; PR=$X/policy_tcpfull_R.pt
for th in 0 15 30 60 90; do
  $P paper_experiments/xembody/xembody.py --robot ur10e --task R --mode calibrate --seed 11 --steps 1500 --ur_frame_yaw_deg $th \
     --run_id ur10e_theta$th --out $G --device cpu --headless 2>&1 | grep "calibration" &
done; wait
/usr/bin/python3 - <<'PY'
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path("paper_experiments/ur10e_hardware").resolve()))
sys.path.insert(0, str(Path(".").resolve()))
from fit_compare import fit, load
X = Path("results/xembody"; G = X / "analytic"
F = fit(*load(X / "long/franka_long_seed11.jsonl")[:3], "world")
for th in (0, 15, 30, 60, 90):
    U = fit(*load(G / f"ur10e_theta{th}.jsonl")[:3], "world")
    m = {"W_F": F.matrix, "b_F": F.bias, "W_U": U.matrix, "b_U": U.bias}
    (G / f"map_theta{th}.json").write_text(json.dumps({k: np.asarray(v).tolist() for k, v in m.items()}))
    print(f"theta {th}: UR10e R2 {U.validation_motion_r2:.3f}")
PY
run() { t=$1; shift; pol=$([ $t = A ] && echo $PA || echo $PR)
  $P paper_experiments/xembody/xembody.py --robot ur10e --task $t --mode eval --policy $pol --episodes 50 --seed 2000 --device cpu --headless --out $G "$@" 2>&1 | grep "^\[XEMB\] an_"; }
for th in 0 15 30 60 90; do for t in A R; do
  run $t --ur_frame_yaw_deg $th --map $G/map_theta$th.json --run_id an_${t}_theta${th}_matched &
  run $t --ur_frame_yaw_deg $th --analytic --run_id an_${t}_theta${th}_analytic &
  run $t --ur_frame_yaw_deg $th --run_id an_${t}_theta${th}_direct &
done; done
for t in A R; do run $t --analytic --analytic_offset 0.0 --run_id an_${t}_theta0_analytic_nooffset & done
wait; echo ANALYTIC_DONE
