"""Offline rehearsal of xembody_real.py: no ROS, no robot.

The real UR10e is stood in for by the world-convention affine model fitted to a held-out real calibration run
(delta = a_U W + b; delta = [TCP displacement in T, world rotation vector]). Same policy, observation, maps,
tilt/box aborts and success rule (1 cm / 5 deg held 1 s, 20 s limit) as the hardware script.
"""
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HW = Path(__file__).resolve().parents[3] / "paper_experiments/ur10e_hardware"
sys.path.insert(0, str(HW))
import xembody_real as X  # noqa: E402
from fit_compare import compare, fit, load  # noqa: E402

S = Path(sys.argv[1])
XE = Path(__file__).resolve().parents[3] / "results/xembody"
RL = Path.home() / "sim2real_runs/real_lab"
for k in (1, 2):
    X.convert(RL / f"lab_real_beta085_run{k}.jsonl", S / f"real_T_run{k}.jsonl")

a, s0, _, _ = load(S / "real_T_run1.jsonl")
print("real calib start TCP in T", np.round(s0[0, :3], 3).tolist(),
      "yaw deg", round(float(np.degrees(Rotation.from_euler("xyz", s0[0, 3:6]).as_euler("ZYX")[0])), 1))

# pre-trial convention check: the two real runs against each other, matched vs mismatched phi
fits = {(k, p): fit(*load(S / f"real_T_run{k}.jsonl")[:3], p) for k in (1, 2) for p in ("world", "body")}
for k, p in fits:
    print(f"real run{k} {p:5s}: held-out R2 {fits[k, p].validation_motion_r2:.3f}")
for pa, pb in (("world", "world"), ("body", "body"), ("world", "body")):
    c = compare(fits[1, pa].matrix, fits[2, pb].matrix)
    print(f"run1 {pa} vs run2 {pb}: E_trans {c['E_trans']:.3f}  E_rot {c['E_rot']:.3f}")

plant = fits[2, "world"]
W, b = plant.matrix, plant.offset
policy = X.Policy(XE / "real/policy_R.npz")
targets = json.loads((XE / "real/targets_R.json").read_text())
box = np.array([-0.30, 0.25, -0.20, 0.25, 0.05, 0.30]).reshape(3, 2)

for fc in ("seed1", "seed2"):
    print(f"\n== Franka calibration {fc}, real map from run1, plant = run2 world fit")
    for mode in ("direct", "matched", "mismatched", "analytic"):
        args = Namespace(mode=mode, franka_calib=XE / f"franka_calib_{fc}.jsonl", real_calib=S / "real_T_run1.jsonl")
        translate = X.make_map(args)
        res = []
        for t in targets:
            target = np.array(t[:3] + [np.radians(t[3])])
            tcp, r = X.START_TCP.copy() if hasattr(X, "START_TCP") else np.array([-0.17, 0.07, 0.175]), \
                np.column_stack([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
            held, outcome = 0, "timeout"
            for step in range(200):
                tilt = np.degrees(np.arccos(np.clip(-r[2, 2], -1, 1)))
                if tilt > 25:
                    outcome = "tilt"; break
                if np.any(tcp < box[:, 0]) or np.any(tcp > box[:, 1]):
                    outcome = "box"; break
                pe = np.linalg.norm(tcp - target[:3])
                yaw = np.arctan2(r[1, 0], r[0, 0])
                ye = np.degrees(abs((yaw - target[3] + np.pi) % (2 * np.pi) - np.pi))
                held = held + 1 if pe < 0.01 and ye < 5 else 0
                if held >= 10:
                    outcome = "ok"; break
                act, _ = policy(X.observe(tcp, r, target))
                a_u = np.clip(translate(np.clip(act, -1, 1), r), -1, 1)
                d = a_u @ W + b
                tcp = tcp + d[:3]
                r = Rotation.from_rotvec(d[3:]).as_matrix() @ r
            res.append((outcome, pe * 1000, ye, step))
        ok = [x for x in res if x[0] == "ok"]
        fails = {}
        for x in res:
            if x[0] != "ok":
                fails[x[0]] = fails.get(x[0], 0) + 1
        med = np.median([x[1] for x in ok]) if ok else float("nan")
        print(f"{mode:10s} {len(ok)}/10  median final err {med:.1f} mm  median steps "
              f"{np.median([x[3] for x in ok]) if ok else float('nan'):.0f}  fails {fails}")
