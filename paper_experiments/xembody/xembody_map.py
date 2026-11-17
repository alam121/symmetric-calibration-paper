"""Fit the Franka and UR10e interfaces from task-free runs and write the transfer maps.

Each run (xembody.py --mode calibrate) logs the TCP pose in the task frame before and after every
command; fit_compare.fit gives d = a W + b under a rotation convention phi (world | body).

  matched     W_F, b_F and W_U, b_U fitted with the same phi
  mismatched  W_F, b_F under world, W_U, b_U under body

and the deployment translation a_U = (a_F W_F + b_F - b_U) pinv(W_U).

  /usr/bin/python3 isaac/sim2real/xembody_map.py --franka RUN.jsonl --ur10e RUN.jsonl --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paper_experiments/ur10e_hardware"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fit_compare import compare, fit, load  # noqa: E402

UNITS = np.array([0.01] * 3 + [0.05] * 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--franka", type=Path, required=True)
    p.add_argument("--ur10e", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    fits = {}
    for robot, path in (("F", a.franka), ("U", a.ur10e)):
        data = load(path)[:3]
        for phi in ("world", "body"):
            m = fit(*data, phi)
            fits[robot, phi] = (np.array(m.matrix), np.array(m.bias), float(m.validation_motion_r2))
    for robot in ("F", "U"):
        W, b, r2 = fits[robot, "world"]
        print(f"{'Franka' if robot == 'F' else 'UR10e '} (world): R2 {r2:.3f}, diag/unit {np.round(np.diag(W) / UNITS, 2).tolist()}")
    a.out.mkdir(parents=True, exist_ok=True)
    for name, (pf, pu) in {"matched_world": ("world", "world"), "matched_body": ("body", "body"),
                           "mismatched": ("world", "body")}.items():
        WF, bF, _ = fits["F", pf]
        WU, bU, _ = fits["U", pu]
        M = WF @ np.linalg.pinv(WU)
        c = compare(WF, WU)
        (a.out / f"map_{name}.json").write_text(json.dumps({
            "W_F": WF.tolist(), "b_F": bF.tolist(), "W_U": WU.tolist(), "b_U": bU.tolist(), "phi_F": pf, "phi_U": pu,
            "M": M.tolist(), "franka_run": str(a.franka), "ur10e_run": str(a.ur10e)}, indent=1))
        print(f"{name:14s} M diag {np.round(np.diag(M), 2).tolist()}  max |offdiag| {np.abs(M - np.diag(np.diag(M))).max():.2f}"
              f"  (E_trans {c['E_trans']:.2f}, E_rot {c['E_rot']:.2f} vs identity)")


if __name__ == "__main__":
    main()
