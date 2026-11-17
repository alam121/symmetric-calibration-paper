#!/bin/bash
# Sim prediction of the hardware run: UR10e commanded in base_link (no 180 deg flip), as the real stack.
cd "$(dirname "$0")/../../.."; unset CONDA_SHLVL
P=${ISAACLAB_PYTHON:-~/miniconda3/envs/env_isaaclab/bin/python}; X=results/xembody; G=$X/baselink; PR=$X/policy_tcpfull_R.pt
$P paper_experiments/xembody/xembody.py --robot ur10e --task R --mode calibrate --seed 11 --steps 1500 --ur_frame_yaw_deg -180 \
   --run_id ur10e_baselink --out $G --device cpu --headless 2>&1 | grep calibration
/usr/bin/python3 - <<'PY'
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path("paper_experiments/ur10e_hardware").resolve()))
sys.path.insert(0, str(Path(".").resolve()))
from fit_compare import fit, load
X = Path("results/xembody"; G = X / "baselink"
for name, (pf, pu) in {"matched": ("world", "world"), "mismatched": ("world", "body")}.items():
    F = fit(*load(X / "long/franka_long_seed11.jsonl")[:3], pf); U = fit(*load(G / "ur10e_baselink.jsonl")[:3], pu)
    (G / f"map_{name}.json").write_text(json.dumps({"W_F": np.asarray(F.matrix).tolist(), "b_F": np.asarray(F.bias).tolist(),
                                                     "W_U": np.asarray(U.matrix).tolist(), "b_U": np.asarray(U.bias).tolist()}))
    M = np.asarray(F.matrix) @ np.linalg.pinv(np.asarray(U.matrix))
    print(name, "M diag", np.round(np.diag(M), 2).tolist())
PY
run() { $P paper_experiments/xembody/xembody.py --robot ur10e --task R --mode eval --policy $PR --episodes 50 --seed 2000 --ur_frame_yaw_deg -180 \
        --device cpu --headless --out $G "$@" 2>&1 | grep "^\[XEMB\] bl_"; }
run --run_id bl_direct &
run --map $G/map_matched.json --run_id bl_matched &
run --map $G/map_mismatched.json --run_id bl_mismatched &
run --analytic --analytic_frame_yaw_deg -180 --run_id bl_analytic &
wait; echo BASELINK_DONE
