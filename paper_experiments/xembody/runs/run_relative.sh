#!/bin/bash
# Task A spatial generalisation: V1 relative obs | V2 absolute (TCP-full) + translation aug | V3 relative + layout-varied demos
cd "$(dirname "$0")/../../.."; unset CONDA_SHLVL
P=${ISAACLAB_PYTHON:-~/miniconda3/envs/env_isaaclab/bin/python}; X=results/xembody; R=$X/relative; mkdir -p $R
# V3 demos: the whole layout moved -10/0/+10 cm in x and y relative to the fixed start pose, 20 episodes each
k=0; for dx in -0.10 0.0 0.10; do for dy in -0.10 0.0 0.10; do k=$((k+1))
  $P paper_experiments/xembody/xembody.py --robot franka --task A --mode collect --episodes 20 --seed $((500+k)) --env_shift_xy $dx $dy \
     --run_id demosA_grid_$k --out $R --device cpu --headless 2>&1 | grep "demosA_grid" &
done; done; wait
/usr/bin/python3 - <<'PY'
import numpy as np, glob
from pathlib import Path
R = Path("results/xembody/relative"
parts = [np.load(f) for f in sorted(glob.glob(str(R / "demosA_grid_*.npz")))]
off, cat = 0, {k: [] for k in ("obs", "act", "grip", "episode", "success")}
for d in parts:
    for k in ("obs", "act", "grip"): cat[k].append(d[k])
    cat["episode"].append(d["episode"] + off); cat["success"].append(d["success"]); off += len(d["success"])
np.savez(R / "franka_demos_A_grid.npz", **{k: np.concatenate(v) for k, v in cat.items()})
print("grid demos:", off, "episodes,", int(np.concatenate(cat["success"]).sum()), "successful")
PY
T="$P paper_experiments/xembody/xembody_train.py"
$T $X/franka_demos_A.npz --obs_noise 0 --rep relative --out $R/policy_A_V1_relative.pt 2>&1 | grep "step 30000" | sed 's/^/V1 /' &
$T $X/franka_demos_A.npz --obs_noise 0 --rep tcp_full --aug_shift 0.10 --out $R/policy_A_V2_aug.pt 2>&1 | grep "step 30000" | sed 's/^/V2 /' &
$T $R/franka_demos_A_grid.npz --obs_noise 0 --rep relative --out $R/policy_A_V3_relgrid.pt 2>&1 | grep "step 30000" | sed 's/^/V3 /' &
wait
run() { $P paper_experiments/xembody/xembody.py --task A --mode eval --episodes 50 --seed 2000 --device cpu --headless --out $R "$@" 2>&1 | grep "^\[XEMB\] rel"; }
for v in V1_relative V2_aug V3_relgrid; do for d in 0 3 6 9; do sh=$(echo "scale=3; $d/100" | bc)
  run --robot franka --policy $R/policy_A_$v.pt --env_shift $sh --run_id rel_${v}_shift${d}_franka &
  run --robot ur10e --policy $R/policy_A_$v.pt --env_shift $sh --map $X/long/maps_11/map_matched_world.json --run_id rel_${v}_shift${d}_matched &
done; done; wait
echo RELATIVE_DONE
