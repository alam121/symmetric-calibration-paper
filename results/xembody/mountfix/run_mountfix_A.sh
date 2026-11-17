#!/bin/bash
# Sim prediction for a contact-rich hardware run: task A (pick the cylinder, drop it in the tray), V1 task-relative
# policy (Franka-trained, frozen), UR10e with the Delto mounted as on the real robot (mountfix asset).
# Two command frames: base_link (as the real stack) and the hidden 180 deg frame (UR "base", the sim default).
# Per frame: fresh UR10e calibration, then direct, analytic (frame known), and the 2x2 convention matrix.
# Starts only after the IBSD queue (xargs PID given as $1) has exited: no Isaac sims beside it.
set -u
cd ~/manipulatorsdatepalm; unset CONDA_SHLVL
P=~/miniconda3/envs/env_isaaclab/bin/python
X=~/sim2real_runs/xembody; G=$X/mountfix; POL=$X/relative/policy_A_V1_relative.pt
USD=~/isaac_ws/ur10e_delto_mountfix/ur10e_delto.usd
COMMON="--device cpu --headless --ur_usd $USD"
if [ -n "${1:-}" ]; then
    echo "$(date '+%F %T') waiting for PID $1 to exit"
    while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi
echo "$(date '+%F %T') start"

# frame name -> extra yaw on top of UR "base" (base_link = -180)
declare -A YAW=([baselink]=-180 [hidden180]=0)

for f in baselink hidden180; do
    $P isaac/sim2real/xembody.py --robot ur10e --task R --mode calibrate --seed 11 --steps 1500 \
        --ur_frame_yaw_deg ${YAW[$f]} --run_id ur10e_mountfix_$f --out $G $COMMON > $G/calib_$f.log 2>&1 &
done
wait
grep -h calibration $G/calib_*.log

/usr/bin/python3 - <<'PY'
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "vla_thesis/vla-deployment-benchmark/paper_experiments/ur10e_hardware"))
sys.path.insert(0, str(Path.home() / "vla_thesis/vla-deployment-benchmark"))
from fit_compare import fit, load
X = Path.home() / "sim2real_runs/xembody"; G = X / "mountfix"
src = load(X / "long/franka_long_seed11.jsonl")[:3]
for f in ("baselink", "hidden180"):
    tgt = load(G / f"ur10e_mountfix_{f}.jsonl")[:3]
    for pf, pu in (("world", "world"), ("body", "body"), ("world", "body"), ("body", "world")):
        F, U = fit(*src, pf), fit(*tgt, pu)
        (G / f"map_{f}_{pf}_{pu}.json").write_text(json.dumps({
            "W_F": np.asarray(F.matrix).tolist(), "b_F": np.asarray(F.bias).tolist(),
            "W_U": np.asarray(U.matrix).tolist(), "b_U": np.asarray(U.bias).tolist()}))
        M = np.asarray(F.matrix) @ np.linalg.pinv(np.asarray(U.matrix))
        print(f, pf, pu, "R2 U %.3f" % U.validation_motion_r2, "M diag", np.round(np.diag(M), 2).tolist())
PY

ev() {   # frame run_id [args...]
    local f=$1 id=$2; shift 2
    $P isaac/sim2real/xembody.py --robot ur10e --task A --mode eval --episodes 50 --seed 2000 \
        --ur_frame_yaw_deg ${YAW[$f]} --run_id $id --out $G $COMMON "$@" > $G/$id.log 2>&1
    grep -h "^\[XEMB\] $id" $G/$id.log | tail -1
}
for f in baselink hidden180; do
    ev $f A_${f}_direct --policy $POL &
    ev $f A_${f}_analytic --policy $POL --analytic --analytic_frame_yaw_deg ${YAW[$f]} &
    for c in world_world body_body world_body body_world; do
        ev $f A_${f}_$c --policy $POL --map $G/map_${f}_$c.json &
    done
    ev $f A_${f}_expert_matched --policy expert --map $G/map_${f}_world_world.json &   # grasp check of the hand itself
done
wait
echo "$(date '+%F %T') MOUNTFIX_A_DONE"
