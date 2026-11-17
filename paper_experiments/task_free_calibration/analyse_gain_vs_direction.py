#!/usr/bin/env python3
"""Split composed-map error into the part feedback absorbs and the part it cannot.

Motivation: a gain-matching intervention on the demonstration-fitted calibrator
(scaling the source map by k = 0.8591) cut the composed-map error from 0.183 to
0.083 and changed closed-loop success not at all -- 85.0% before and after. That
looked like an unexplained null result. It is not.

For a composed map M and the ideal correction M* = H^-1, write the best uniform
gain onto the ideal as

    s* = <M, M*> / <M*, M*>

and split the error into

    gain      |s* - 1|                             a uniform rescale of every command
    direction ||M - s* M*||_F / ||s* M*||_F        the scale-invariant misalignment

Scaling the SOURCE map by k multiplies M by 1/k, which moves s* and leaves the
directional term algebraically untouched. The two demo-fitted calibrators
therefore have directional components equal to ~2e-16, so an identical closed-loop
score is forced, not observed. The intervention tested a quantity the closed loop
does not care about: a 14% uniform shrink is absorbed by re-observing state each
chunk and taking slightly more steps to cover the same distance, whereas a wrong
axis or sign is not recoverable by feedback at all.

Reported this way, E_rel is the wrong scalar to put next to a success rate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[2] / "data/task_free_calibration"
AXIS_SWAP = np.array([[0, -0.8, 0, 0, 0, 0], [0.65, 0, 0, 0, 0, 0], [0, 0, 0.55, 0, 0, 0],
                      [0, 0, 0, 0, -0.9, 0], [0, 0, 0, 0.7, 0, 0], [0, 0, 0, 0, 0, 0.6]], dtype=float)

ARMS = [
    # label, calibrator (None = no repair), result dir under results/task_free_calibration
    ("hidden (no repair)",          None,
     "controlled_interface/native_hidden_interface__model-seed-0__n-100"),
    ("task-free fitted",            "calibrator_calvin_axis_swap_seed201.json",
     "controlled_interface/native_task_free_calibrated__model-seed-0__n-100"),
    ("oracle (exact inverse)",      "calibrator_oracle_axis_swap.json",
     "oracle_axis_swap_rerun/native_oracle_axis_swap__model-seed-0__n-100"),
    ("demo-fitted W_t",             "calibrator_demofit_axis_swap.json",
     "demofit/native_demofit_calibrated__model-seed-0__n-100"),
    ("demo-fitted + gain match k",  "calibrator_demofit_gainmatched_axis_swap.json",
     "demofit_gainmatched/native_demofit_gainmatched__model-seed-0__n-100"),
    ("phi body_body (matched)",     "calibrator_phi_body_body_axis_swap.json",
     "phi_symmetry/body_body/native_phi_body_body__model-seed-0__n-100"),
    ("phi world_body (mismatched)", "calibrator_phi_world_body_axis_swap.json",
     "phi_symmetry/world_body/native_phi_world_body__model-seed-0__n-100"),
    ("phi body_world (mismatched)", "calibrator_phi_body_world_axis_swap.json",
     "phi_symmetry/body_world/native_phi_body_world__model-seed-0__n-100"),
]

RESULTS = Path(__file__).resolve().parents[2] / "results/task_free_calibration"


def closed_loop(rel: str) -> tuple[str, str]:
    """D1 and average chain length, or '?' when the run has not finished.

    Gated on the per-sequence file carrying a full 100 sequences: that file is
    written incrementally, so a shorter one means the evaluation is still
    running and must not be reported as a result.
    """
    ps = RESULTS / rel / "per_sequence_results.json"
    if ps.exists():
        try:
            chains = json.loads(ps.read_text())["chain_lengths"]
        except (json.JSONDecodeError, KeyError):
            chains = None
        if chains is not None:
            if len(chains) != 100:
                return f"({len(chains)})", "    ?"   # still running
            d1 = 100 * sum(1 for x in chains if x >= 1) / len(chains)
            return f"{d1:.1f}", f"{sum(chains) / len(chains):.3f}"
    # Fall back to the aggregate file. Trailing "u" marks an arm with no
    # per-sequence log, so it cannot enter a paired test -- the reason the
    # task-free arm is being re-run with logging enabled.
    agg = RESULTS / rel / "results.json"
    if not agg.exists():
        return "  ?", "    ?"
    try:
        r = next(iter(json.loads(agg.read_text()).values()))
        return f"{100 * r['chain_sr']['1']:.1f}u", f"{r['avg_seq_len']:.3f}u"
    except (json.JSONDecodeError, KeyError, StopIteration):
        return "  ?", "    ?"



def composed(name: str) -> np.ndarray:
    d = json.loads((D / name).read_text())
    Ws = np.array(d["source"]["action_to_delta"])
    Wt = np.array(d["target"]["action_to_delta"])
    return Ws @ np.linalg.pinv(Wt)


def blocks(M: np.ndarray, ideal: np.ndarray) -> tuple[float, float, float]:
    """Plain relative error per block, plus off-block leakage.

    Kept for the aggregate view, but prefer per_block() below when the arms being
    compared differ in overall gain: these figures still contain the gain term, so
    they move when only the gain was changed.
    """
    t = np.linalg.norm(M[:3, :3] - ideal[:3, :3]) / np.linalg.norm(ideal[:3, :3])
    r = np.linalg.norm(M[3:, 3:] - ideal[3:, 3:]) / np.linalg.norm(ideal[3:, 3:])
    cross = math.sqrt(float(np.sum((M[:3, 3:] - ideal[:3, 3:]) ** 2)
                            + np.sum((M[3:, :3] - ideal[3:, :3]) ** 2))) / np.linalg.norm(ideal)
    return float(t), float(r), float(cross)


def per_block(M: np.ndarray, ideal: np.ndarray) -> tuple[float, float, float, float]:
    """Best scalar and scale-invariant residual computed WITHIN each block.

    A single global scalar is the wrong normalisation for a block-localized
    defect: the best global s* is dragged down by the broken block and then
    corrupts the intact one. Fitting a scalar per block separates "this block has
    the wrong magnitude" from "this block points the wrong way".

    This is what shows that a frame-convention mismatch leaves translation
    exact (gain 1.003, residual 0.031, matching a healthy arm) while INVERTING
    rotation -- gain -0.45, i.e. every rotation command is applied in the
    opposite direction at 45% magnitude. That is why reaching and coarse tasks
    survive while orientation-dependent ones diverge and chains collapse.
    """
    out = []
    for sl in (slice(0, 3), slice(3, 6)):
        a, b = M[sl, sl], ideal[sl, sl]
        s = float(np.sum(a * b) / np.sum(b * b))
        out.append(s)
        out.append(float(np.linalg.norm(a - s * b) / np.linalg.norm(s * b)))
    return tuple(out)


def split(M: np.ndarray, ideal: np.ndarray) -> tuple[float, float, float]:
    s = float(np.sum(M * ideal) / np.sum(ideal * ideal))
    total = float(np.linalg.norm(M - ideal) / np.linalg.norm(ideal))
    direction = float(np.linalg.norm(M - s * ideal) / np.linalg.norm(s * ideal))
    return total, s, direction


def main() -> int:
    ideal = np.linalg.inv(AXIS_SWAP)
    print(f"{'arm':30s} {'E_rel':>7s} {'|s*-1|':>7s} {'E_dir':>7s} {'E_trans':>8s} "
          f"{'E_rot':>8s} {'E_cross':>8s} {'D1 %':>7s} {'avg':>7s}")
    print("-" * 104)
    for label, f, rel in ARMS:
        M = np.eye(6) if f is None else composed(f)
        total, s, direction = split(M, ideal)
        d1, avg = closed_loop(rel)
        et, er, ec = blocks(M, ideal)
        print(f"{label:30s} {total:7.4f} {abs(s - 1):7.4f} {direction:7.4f} {et:8.4f} "
              f"{er:8.4f} {ec:8.4f} {d1:>7s} {avg:>7s}")

    print(f"\n{'arm':30s} | {'TRANSLATION':^20s} | {'ROTATION':^20s}")
    print(f"{'':30s} | {'gain':>8s} {'residual':>10s} | {'gain':>8s} {'residual':>10s}")
    print("-" * 78)
    for label, f, _rel in ARMS:
        M = np.eye(6) if f is None else composed(f)
        st, dt, sr, dr = per_block(M, ideal)
        print(f"{label:30s} | {st:8.4f} {dt:10.4f} | {sr:8.4f} {dr:10.4f}")
    print("gain 1.0 = correct magnitude; a NEGATIVE gain means that block is inverted.")

    a, b = composed(ARMS[3][1]), composed(ARMS[4][1])
    na = a / (np.sum(a * ideal) / np.sum(ideal * ideal))
    nb = b / (np.sum(b * ideal) / np.sum(ideal * ideal))
    rel = np.linalg.norm(na - nb) / np.linalg.norm(na)
    print(f"\ndirectional components, plain vs gain-matched: rel diff {rel:.3e}")
    print("The gain-match changed only the gain term, so the equal closed-loop score is forced.")
    print('\nu = read from results.json; no per-sequence log, so not pairable.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
