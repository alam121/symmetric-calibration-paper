"""Offline rehearsal of the unknown command-frame experiment (no ROS, no robot).

The robot interprets commands in base_link turned THETA about z. Its calibration in that frame is today's real
calibration with every logged action un-rotated (same motion, other command labels). The real arm is stood in
for by the world fit of today's run 2, fed the rotated command.

  python offline_frame.py THETA_DEG [THETA_DEG ...]
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

R = Path(__file__).resolve().parent
XE = R.parent
OUT = R / "frame_offline"
OUT.mkdir(exist_ok=True)
policy = X.Policy(R / "policy_R.npz")
targets = json.loads((R / "targets_R.json").read_text())
import os
box = np.array([float(v) for v in os.environ.get("BOX", "-0.30 0.25 -0.20 0.25 0.05 0.30").split()]).reshape(3, 2)
MAX_TILT = float(os.environ.get("MAX_TILT", "25"))
START_R = np.column_stack([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])


def relabel(src, dst, theta):
    """Calibration log as the rotated-frame robot would record it."""
    back = Rotation.from_euler("z", -theta).as_matrix()
    with dst.open("w") as f:
        for line in src.read_text().splitlines():
            row = json.loads(line)
            a = np.array(row["action"], float)
            row["action"] = list(back @ a[:3]) + list(back @ a[3:6]) + list(a[6:])
            f.write(json.dumps(row) + "\n")


for deg in map(float, sys.argv[1:]):
    th = np.radians(deg)
    for k in (1, 2):
        relabel(R / f"calib/xr_calib_seed{k}_T.jsonl", OUT / f"theta{deg:g}_seed{k}_T.jsonl", th)
    fits = {(k, p): fit(*load(OUT / f"theta{deg:g}_seed{k}_T.jsonl")[:3], p) for k in (1, 2) for p in ("world", "body")}
    cm, cx = compare(fits[1, "world"].matrix, fits[2, "world"].matrix), compare(fits[1, "world"].matrix, fits[2, "body"].matrix)
    print(f"\n== command frame turned {deg:g} deg: convention check E_rot matched {cm['E_rot']:.3f} vs mismatched "
          f"{cx['E_rot']:.3f}; E_trans {cm['E_trans']:.3f} vs {cx['E_trans']:.3f}")
    plant = fit(*load(R / "calib/xr_calib_seed2_T.jsonl")[:3], "world")
    W, b = plant.matrix, plant.offset
    frame = Rotation.from_euler("z", th).as_matrix()
    for mode in ("direct", "matched", "mismatched", "analytic", "analytic_exact"):
        args = Namespace(mode=mode, cmd_frame_yaw_deg=deg, franka_calib=XE / "franka_calib_seed1.jsonl",
                         real_calib=OUT / f"theta{deg:g}_seed1_T.jsonl")
        translate = X.make_map(args)
        res = []
        for t in targets:
            target = np.array(t[:3] + [np.radians(t[3])])
            tcp, r = np.array([-0.185, 0.083, 0.175]), START_R.copy()
            held, outcome, travel = 0, "timeout", 0.0
            for step in range(200):
                tilt = np.degrees(np.arccos(np.clip(-r[2, 2], -1, 1)))
                if tilt > MAX_TILT:
                    outcome = "tilt"; break
                if np.any(tcp < box[:, 0]) or np.any(tcp > box[:, 1]):
                    outcome = "box"; break
                pe = np.linalg.norm(tcp - target[:3])
                yaw = np.arctan2(r[1, 0], r[0, 0])
                ye = np.degrees(abs((yaw - target[3] + np.pi) % (2 * np.pi) - np.pi))
                held = held + 1 if pe < 0.01 and ye < 5 else 0
                if held >= 10:
                    outcome = "ok"; break
                a, _ = policy(X.observe(tcp, r, target))
                a_u = np.clip(translate(np.clip(a, -1, 1), r), -1, 1)
                d = np.concatenate([frame @ a_u[:3], frame @ a_u[3:]]) @ W + b
                tcp = tcp + d[:3]
                travel = max(travel, np.linalg.norm(tcp - np.array([-0.185, 0.083, 0.175])))
                r = Rotation.from_rotvec(d[3:]).as_matrix() @ r
            res.append((outcome, pe * 1000, step, travel * 100))
        ok = [x for x in res if x[0] == "ok"]
        fails = {}
        for x in res:
            if x[0] != "ok":
                fails[x[0]] = fails.get(x[0], 0) + 1
        bad = [x for x in res if x[0] != "ok"]
        fail_txt = (f"; failures {fails}, median {np.median([x[2] for x in bad]) / 10:.1f} s, max excursion "
                    f"{max(x[3] for x in bad):.0f} cm") if bad else ""
        ok_txt = f"; median {np.median([x[1] for x in ok]):.1f} mm in {np.median([x[2] for x in ok]) / 10:.1f} s" if ok else ""
        print(f"{mode:15s} {len(ok)}/10{ok_txt}{fail_txt}")
