#!/usr/bin/env python3
"""Measure the nuisance operator B and test whether the two arms share it.

Model. Excitation with command autocorrelation beta does not recover the true
local action map W, but a distorted version

    W_hat(beta) = W_nominal @ B(beta)

where B is the nuisance operator induced by the excitation statistics (realised
gain falls at command reversals, so a less correlated signal under-reports the
map). In simulation W_nominal is known analytically -- CALVIN rel_actions scale
translation by 0.02 m and rotation by 0.05 rad per unit command -- so B can be
measured directly rather than only as the ratio B(b1)^-1 B(b2).

The transfer claim does not need B to be small. It needs B to be SHARED:

    W_hat_s W_hat_t^+ = (W_s B)(W_t B)^+ = W_s W_t^+   when B_s = B_t

so a large shared B cancels in the composed map while a small asymmetric one
does not. This script reports, per beta:

  ||B_s - B_t|| / ||B_s||   how nearly the two arms share the operator
  off-block energy          energy outside the translation and rotation blocks,
                            normalised -- B being block diagonal means the
                            excitation does not mix translation into rotation
  c_trans, c_rot            mean diagonal gain of each block, as a fraction of
                            nominal; their ratio tests whether B is a global
                            scalar or acts differently on the two blocks

Row convention: delta = action @ W, matching calibrator.map_action, so a
deployment distortion M composes on the left: W_nominal_distorted = M @ W_nominal.

CAVEAT on the rotation block. Normalising it requires assuming a nominal
rotation map, and that assumption is convention-dependent: c_rot comes out
positive under euler_xyz, negative under euler_xyz_body, and swings by an order
of magnitude under rotvec and linear, because the measured rotation delta and
the assumed nominal do not share a sign/axis convention. c_rot, their ratio and
the off-block energy therefore must never be quoted without naming the
orientation kind, and a claim that B is "globally scalar" or "block diagonal"
cannot rest on them as computed here. The translation block carries no such
ambiguity, so agree_trans below is the convention-free form of the shared-B
measurement and is the one to rely on.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

# fit_calibrator imports through the package path, so the repo root has to be
# importable whether this is run as a file or as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_experiments.task_free_calibration.calibrator import fit_local_action_model  # noqa: E402
from paper_experiments.task_free_calibration.fit_calibrator import load_directory  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data/task_free_calibration"
NOMINAL = np.diag([0.02, 0.02, 0.02, 0.05, 0.05, 0.05])
AXIS_SWAP = np.array([[0, -0.8, 0, 0, 0, 0], [0.65, 0, 0, 0, 0, 0], [0, 0, 0.55, 0, 0, 0],
                      [0, 0, 0, 0, -0.9, 0], [0, 0, 0, 0.7, 0, 0], [0, 0, 0, 0, 0, 0.6]], dtype=float)


def fit(shard: Path, orientation_kind: str, split_seed: int = 1729) -> np.ndarray:
    action, state, next_state, validation = load_directory(shard, 0.2, split_seed)
    model = fit_local_action_model(action, state, next_state, validation,
                                   orientation_kind=orientation_kind)
    return np.asarray(model.matrix, dtype=float)


def nuisance(W_hat: np.ndarray, distorted: bool) -> np.ndarray:
    nominal = AXIS_SWAP @ NOMINAL if distorted else NOMINAL
    return np.linalg.pinv(nominal) @ W_hat


def describe(B: np.ndarray) -> tuple[float, float, float]:
    trans, rot = B[:3, :3], B[3:, 3:]
    off = np.sqrt(np.sum(B[:3, 3:] ** 2) + np.sum(B[3:, :3] ** 2)) / np.linalg.norm(B)
    return float(np.mean(np.diag(trans))), float(np.mean(np.diag(rot))), float(off)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orientation-kind", default="euler_xyz")
    ap.add_argument("--pairs", nargs="+", default=["0.0", "0.5", "0.95"],
                    help="beta values with calvin_beta<b>_{clean,axisswap} shards")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra label:clean_dir:axisswap_dir triples")
    args = ap.parse_args()

    rows = []
    for b in args.pairs:
        rows.append((f"beta={b}", DATA / f"calvin_beta{b}_clean", DATA / f"calvin_beta{b}_axisswap"))
    for spec in args.extra:
        label, c, a = spec.split(":")
        rows.append((label, DATA / c, DATA / a))

    print("rotation-block columns are convention-dependent; see the module docstring\n")
    print(f"{'arm pair':16s} {'|Bs-Bt|/|Bs|':>13s} {'trans only':>11s} {'off-block':>10s} "
          f"{'c_trans':>8s} {'c_rot':>8s} {'c_t/c_r':>8s}")
    print("-" * 82)
    store = {}
    for label, clean, swap in rows:
        if not clean.exists() or not swap.exists():
            print(f"{label:16s} missing shards, skipped")
            continue
        Bs = nuisance(fit(clean, args.orientation_kind), distorted=False)
        Bt = nuisance(fit(swap, args.orientation_kind), distorted=True)
        store[label] = (Bs, Bt)
        agree = float(np.linalg.norm(Bs - Bt) / np.linalg.norm(Bs))
        agree_trans = float(np.linalg.norm(Bs[:3, :3] - Bt[:3, :3])
                            / np.linalg.norm(Bs[:3, :3]))
        ct, cr, off = describe(Bs)
        print(f"{label:16s} {agree:13.4f} {agree_trans:11.4f} {off:10.4f} "
              f"{ct:8.4f} {cr:8.4f} {ct/cr:8.4f}")

    # Seed-to-seed spread of the SAME arm at the same beta is the noise floor the
    # cross-arm agreement above has to beat to mean anything.
    floor = [(x, y) for x, y in itertools.combinations(store, 2)
             if x.split("=")[0] == y.split("=")[0] == "rep"]
    if floor:
        print("\nnoise floor -- same arm, same excitation, different seed:")
        print(f"  {'arm':9s} {'pair':19s} {'full B':>8s} {'trans only':>11s}")
        for x, y in floor:
            for i, arm in enumerate(("clean", "axisswap")):
                a, b = store[x][i], store[y][i]
                full = np.linalg.norm(a - b) / np.linalg.norm(a)
                tr = (np.linalg.norm(a[:3, :3] - b[:3, :3])
                      / np.linalg.norm(a[:3, :3]))
                print(f"  {arm:9s} {x + ' vs ' + y:19s} {full:8.4f} {tr:11.4f}")
        print("\nCompare these against the cross-arm columns above. If the cross-arm")
        print("discrepancy does not exceed the same-seed floor, B_s and B_t are")
        print("indistinguishable at this excitation budget -- which is the strongest")
        print("form of 'shared' the measurement can support, and also means the")
        print("cross-arm figure must not be quoted as a measured nonzero asymmetry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
