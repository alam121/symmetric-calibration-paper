#!/usr/bin/env python3
"""Fit UR10e calibration runs and compare them (hardware validation, steps 10-14).

Every run (ur10e_calibration_run.py) is fitted with the paper's
calibrator.fit_local_action_model, d_t = a_t W + b, under both rotation conventions phi:
world (euler_xyz) and body (euler_xyz_body) -- the SAME recorded poses, only the processing
differs. Both sides are the same UR10e, so the composed map between two calibrations,

    M = W_A pinv(W_B),

should be the identity; its departure from I, split into translation / rotation / cross blocks
exactly as analyse_gain_vs_direction.blocks() and per_block() do, is the hardware form of the
paper's composed-map error. Reported:

  fits           R^2 and MAE on held-out transitions, per run and phi
  repeatability  matched beta, matched phi, independent runs: the hardware noise floor
  excitation     beta_A != beta_B, matched phi
  phi            matched beta pairs (and each run against itself) under body/body,
                 world/world, world/body, body/world
  go / no-go     step 14: world/body and body/world must break rotation, not translation

  python3 fit_compare.py runs/ [--json report.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_experiments.task_free_calibration.analyse_gain_vs_direction import blocks, per_block  # noqa: E402
from paper_experiments.task_free_calibration.calibrator import fit_local_action_model  # noqa: E402

PHI = {"world": "euler_xyz", "body": "euler_xyz_body"}


def load(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if "next_state" in r]
    meta = json.loads(path.with_suffix("").with_suffix(".meta.json").read_text()) if \
        path.with_suffix("").with_suffix(".meta.json").exists() else {}
    return (np.array([r["action"] for r in rows]), np.array([r["state"] for r in rows]),
            np.array([r["next_state"] for r in rows]), meta)


def fit(action, state, next_state, phi, seed=1729, fraction=0.2):
    mask = np.random.default_rng(seed).random(len(action)) < fraction
    model = fit_local_action_model(action, state, next_state, mask, orientation_kind=PHI[phi])
    return model


def compare(Wa, Wb):
    M = Wa @ np.linalg.pinv(Wb)
    ideal = np.eye(6)
    e_trans, e_rot, e_cross = blocks(M, ideal)
    g_trans, r_trans, g_rot, r_rot = per_block(M, ideal)
    return {"E_trans": e_trans, "E_rot": e_rot, "E_cross": e_cross,
            "gain_trans": g_trans, "dir_trans": r_trans, "gain_rot": g_rot, "dir_rot": r_rot,
            "E_rel": float(np.linalg.norm(M - ideal) / np.linalg.norm(ideal))}


def row(label, c):
    return (f"{label:34s} {c['E_rel']:7.3f} {c['E_trans']:8.3f} {c['E_rot']:8.3f} {c['E_cross']:8.3f} "
            f"{c['gain_trans']:7.3f} {c['gain_rot']:7.3f}")


HEADER = f"{'pair':34s} {'E_rel':>7s} {'E_trans':>8s} {'E_rot':>8s} {'E_cross':>8s} {'g_trans':>7s} {'g_rot':>7s}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    runs = {}
    for path in sorted(args.runs.glob("*.jsonl")):
        action, state, next_state, meta = load(path)
        if len(action) < 50:
            print(f"[FIT] {path.stem}: only {len(action)} transitions, skipped")
            continue
        runs[path.stem] = {"beta": meta.get("beta"), "n": len(action),
                           "models": {phi: fit(action, state, next_state, phi) for phi in PHI}}
    report = {"fits": {}, "repeatability": {}, "excitation": {}, "phi": {}}

    print(f"\n{'run':24s} {'beta':>5s} {'n':>5s}   {'R2 world':>8s} {'R2 body':>8s}   {'MAE world':>9s} {'MAE body':>9s}")
    for name, run in runs.items():
        m = run["models"]
        report["fits"][name] = {phi: {"r2": m[phi].validation_motion_r2, "mae": m[phi].validation_motion_mae,
                                      "W": m[phi].action_to_delta, "b": m[phi].bias} for phi in PHI}
        print(f"{name:24s} {run['beta']:5.2f} {run['n']:5d}   {m['world'].validation_motion_r2:8.3f} "
              f"{m['body'].validation_motion_r2:8.3f}   {m['world'].validation_motion_mae:9.5f} "
              f"{m['body'].validation_motion_mae:9.5f}")

    pairs = list(itertools.combinations(runs, 2))
    print("\nComposed map W_A pinv(W_B) vs identity (same robot both sides)\n" + HEADER)
    print("-- repeatability: matched beta, matched phi")
    for a, b in pairs:
        if runs[a]["beta"] == runs[b]["beta"]:
            for phi in PHI:
                c = compare(runs[a]["models"][phi].matrix, runs[b]["models"][phi].matrix)
                report["repeatability"][f"{a}|{b}|{phi}/{phi}"] = c
                print(row(f"{a} vs {b} {phi}/{phi}", c))
    print("-- excitation mismatch: different beta, matched phi")
    for a, b in pairs:
        if runs[a]["beta"] != runs[b]["beta"]:
            for phi in PHI:
                c = compare(runs[a]["models"][phi].matrix, runs[b]["models"][phi].matrix)
                report["excitation"][f"{a}|{b}|{phi}/{phi}"] = c
                print(row(f"{a}({runs[a]['beta']}) vs {b}({runs[b]['beta']}) {phi}", c))
    print("-- phi: matched beta; processing convention differs (same recorded poses)")
    same_beta = [(a, b) for a, b in pairs if runs[a]["beta"] == runs[b]["beta"]] + [(a, a) for a in runs]
    by_combo = {}
    for a, b in same_beta:
        for pa, pb in itertools.product(PHI, PHI):
            c = compare(runs[a]["models"][pa].matrix, runs[b]["models"][pb].matrix)
            report["phi"][f"{a}|{b}|{pa}/{pb}"] = c
            if a != b or pa != pb:  # a run against itself under the same phi is trivially exact
                by_combo.setdefault(f"{pa}/{pb}", []).append(c)
            if a != b or pa != pb:
                print(row(f"{a} vs {b} {pa}/{pb}", c))
    print("\nphi summary (median; matched phi over independent runs only)\n" + HEADER)
    summary = {}
    for combo, cs in by_combo.items():
        med = {k: float(np.median([c[k] for c in cs])) for k in cs[0]}
        summary[combo] = med
        print(row(combo, med))
    report["phi_summary"] = summary

    matched = max(summary["body/body"]["E_rot"], summary["world/world"]["E_rot"])
    mismatched = min(summary["world/body"]["E_rot"], summary["body/world"]["E_rot"])
    # One-sided: a phi mismatch must not make translation worse. (Mismatched pairs include same-run
    # pairs, whose E_trans is ~0, below the between-run matched floor; that is not a change.)
    trans_ok = max(summary[c]["E_trans"] - summary["body/body"]["E_trans"] for c in summary) < 0.05
    go = mismatched > 3 * max(matched, 1e-6) and trans_ok
    verdict = ("GO: mismatched phi breaks rotation (E_rot %.3f vs matched %.3f) while translation is unchanged"
               % (mismatched, matched)) if go else (
        "NO-GO: the open-loop pattern is not there (E_rot mismatched %.3f vs matched %.3f, translation unchanged: %s)"
        " -- do not run closed-loop trials yet" % (mismatched, matched, trans_ok))
    report["go_no_go"] = {"go": bool(go), "verdict": verdict}
    print("\n" + verdict)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
