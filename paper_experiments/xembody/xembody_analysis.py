"""Calibration repeatability (short vs long runs), per-pair maps, and the mechanism heatmaps.

  /usr/bin/python3 isaac/sim2real/xembody_analysis.py calib    # repeatability + maps for each seed pair
  /usr/bin/python3 isaac/sim2real/xembody_analysis.py figure   # heatmaps of W_F, W_U and the transfer maps
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paper_experiments/ur10e_hardware"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fit_compare import compare, fit, load  # noqa: E402

X = Path(__file__).resolve().parents[2] / "results/xembody"
L = X / "long"
SEEDS = (11, 12, 13)
AXES = ["x", "y", "z", "rx", "ry", "rz"]


def fit_run(path, phi, n=None):
    a, s, ns = load(path)[:3]
    if n is not None:
        a, s, ns = a[:n], s[:n], ns[:n]
    m = fit(a, s, ns, phi)
    return np.array(m.matrix), np.array(m.bias), float(m.validation_motion_r2)


def calib():
    out = {"repeatability": {}, "pairs": {}}
    print("Repeatability: composed W_i pinv(W_j) vs identity over the 3 independent runs (median of 3 pairs)")
    print(f"{'robot':7s}{'phi':6s}{'steps':>6s}{'E_rel':>8s}{'E_trans':>9s}{'E_rot':>8s}{'E_cross':>9s}{'R2':>7s}")
    for robot in ("franka", "ur10e"):
        for phi in ("world", "body"):
            for n in (300, 1500):
                fits = {s: fit_run(L / f"{robot}_long_seed{s}.jsonl", phi, n) for s in SEEDS}
                cs = [compare(fits[i][0], fits[j][0]) for i, j in itertools.combinations(SEEDS, 2)]
                row = {k: float(np.median([c[k] for c in cs])) for k in ("E_rel", "E_trans", "E_rot", "E_cross")}
                row["R2"] = float(np.median([f[2] for f in fits.values()]))
                out["repeatability"][f"{robot}/{phi}/{n}"] = row
                print(f"{robot:7s}{phi:6s}{n:6d}{row['E_rel']:8.3f}{row['E_trans']:9.3f}{row['E_rot']:8.3f}{row['E_cross']:9.3f}{row['R2']:7.3f}")
    print("\nPer calibration pair (1500 transitions each side): transfer map M = W_F pinv(W_U)")
    for s in SEEDS:
        (L / f"maps_{s}").mkdir(exist_ok=True)
        F = {phi: fit_run(L / f"franka_long_seed{s}.jsonl", phi) for phi in ("world", "body")}
        U = {phi: fit_run(L / f"ur10e_long_seed{s}.jsonl", phi) for phi in ("world", "body")}
        for name, (pf, pu) in {"matched_world": ("world", "world"), "matched_body": ("body", "body"),
                               "mismatched": ("world", "body")}.items():
            WF, bF, _ = F[pf]; WU, bU, _ = U[pu]
            M = WF @ np.linalg.pinv(WU)
            (L / f"maps_{s}" / f"map_{name}.json").write_text(json.dumps(
                {"W_F": WF.tolist(), "b_F": bF.tolist(), "W_U": WU.tolist(), "b_U": bU.tolist(), "M": M.tolist(),
                 "phi_F": pf, "phi_U": pu, "seed": s}, indent=1))
            out["pairs"][f"{s}/{name}"] = {"M_diag": np.round(np.diag(M), 3).tolist(),
                                           "max_offdiag": float(np.abs(M - np.diag(np.diag(M))).max())}
            print(f"  seed {s} {name:14s} diag {np.round(np.diag(M), 2).tolist()}  max|off| {np.abs(M - np.diag(np.diag(M))).max():.2f}")
    # map variability across pairs
    for name in ("matched_world", "matched_body", "mismatched"):
        Ms = [np.array(json.loads((L / f"maps_{s}" / f"map_{name}.json").read_text())["M"]) for s in SEEDS]
        spread = np.std(Ms, axis=0)
        rel = float(np.median([np.linalg.norm(a - b) / np.linalg.norm(a) for a, b in itertools.combinations(Ms, 2)]))
        out["pairs"][f"spread/{name}"] = {"max_entry_std": float(spread.max()), "median_rel_diff": rel}
        print(f"  {name:14s} across pairs: max entry std {spread.max():.3f}, median ||M_i - M_j|| / ||M_i|| {rel:.3f}")
    (L / "calibration_report.json").write_text(json.dumps(out, indent=1))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = SEEDS[0]
    units = np.array([0.01] * 3 + [0.05] * 3)
    WF = np.array(json.loads((L / f"maps_{s}" / "map_matched_world.json").read_text())["W_F"])
    WU = np.array(json.loads((L / f"maps_{s}" / "map_matched_world.json").read_text())["W_U"])
    Mm = np.array(json.loads((L / f"maps_{s}" / "map_matched_world.json").read_text())["M"])
    Mx = np.array(json.loads((L / f"maps_{s}" / "map_mismatched.json").read_text())["M"])
    panels = [(WF / units[None, :], "Franka W_F (world)\nrealised / commanded"),
              (WU / units[None, :], "UR10e W_U (world)\nrealised / commanded"),
              (Mm, "matched map W_F pinv(W_U)\n(world / world)"),
              (Mx, "mismatched map\n(world / body)")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, (m, title) in zip(axes, panels):
        lim = max(1.0, float(np.abs(m).max()))
        im = ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(m[i, j]) > 0.6 * lim else "black")
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels([f"d{a}" for a in AXES]); ax.set_yticklabels([f"a_{a}" for a in AXES])
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("action component (row)")
    fig.suptitle("Action-interface maps, calibration pair seed 11 (1500 transitions each).  Rows: commanded action; "
                 "columns: TCP displacement (W) or UR10e command (M).", fontsize=10)
    fig.tight_layout()
    path = X / "fig_transfer_maps.png"
    fig.savefig(path, dpi=160)
    print(f"-> {path}")


if __name__ == "__main__":
    {"calib": calib, "figure": figure}[sys.argv[1]]()
