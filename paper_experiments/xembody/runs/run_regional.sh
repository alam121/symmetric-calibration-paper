#!/bin/bash
# Regional calibration: calibrate both robots around the shifted task-R targets, then R at 6 / 9 cm with
# (a) the regional map only, (b) gain scheduling (start-pose map + regional map, nearest centre to the TCP).
cd "$(dirname "$0")/../../.."; unset CONDA_SHLVL
P=${ISAACLAB_PYTHON:-~/miniconda3/envs/env_isaaclab/bin/python}; X=results/xembody; G=$X/regional; mkdir -p $G
declare -A C=([6]="0.108 0.032 0.10" [9]="0.130 0.054 0.10")
for d in 6 9; do for r in franka ur10e; do
  $P paper_experiments/xembody/xembody.py --robot $r --task R --mode calibrate --seed 21 --steps 1500 --calib_centre ${C[$d]} --run_id ${r}_region$d --out $G --device cpu --headless 2>&1 | grep "calibration" &
done; done; wait
/usr/bin/python3 - <<'PY'
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path("paper_experiments/ur10e_hardware").resolve()))
sys.path.insert(0, str(Path(".").resolve()))
from fit_compare import fit, load
G = Path("results/xembody/regional"; X = G.parent
centres = {6: [0.108, 0.032, 0.10], 9: [0.130, 0.054, 0.10]}
start = [-0.17, 0.07, 0.175]
for d, c in centres.items():
    WF = fit(*load(G / f"franka_region{d}.jsonl")[:3], "world"); WU = fit(*load(G / f"ur10e_region{d}.jsonl")[:3], "world")
    m = {"W_F": WF.matrix, "b_F": WF.bias, "W_U": WU.matrix, "b_U": WU.bias, "centre": c}
    m = {k: np.asarray(v).tolist() for k, v in m.items()}
    (G / f"map_region{d}.json").write_text(json.dumps(m, indent=1))
    M = np.array(m["W_F"]) @ np.linalg.pinv(np.array(m["W_U"]))
    M0 = np.array(json.loads((X / "long/maps_11/map_matched_world.json").read_text())["M"])
    print(f"region {d} cm: R2 F {WF.validation_motion_r2:.3f} U {WU.validation_motion_r2:.3f}; ||M_region - M_start|| / ||M_start|| = "
          f"{np.linalg.norm(M - M0) / np.linalg.norm(M0):.3f}; diag {np.round(np.diag(M), 2).tolist()}")
    (G / f"schedule_region{d}.json").write_text(json.dumps([
        {"map": str(X / "long/maps_11/map_matched_world.json"), "centre": start},
        {"map": str(G / f"map_region{d}.json"), "centre": c}]))
PY
run() { $P paper_experiments/xembody/xembody.py --robot ur10e --task R --mode eval --policy $X/policy_planar_R.pt --episodes 50 --seed 2000 --device cpu --headless --out $G "$@" 2>&1 | grep "^\[XEMB\] reg"; }
for d in 6 9; do sh=0.0$d
  run --env_shift $sh --map $G/map_region$d.json --run_id reg_R_shift${d}_regional &
  run --env_shift $sh --map_schedule $G/schedule_region$d.json --run_id reg_R_shift${d}_scheduled &
done; wait; echo REGIONAL_DONE
