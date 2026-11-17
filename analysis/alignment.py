"""Effective map and worst-case alignment for every translation in the paper (sim and real UR10e).

For a translation M (a_t = a M) the target robot realises the motion a M W_t, which the source robot would have
produced from the action a G with
    G = M W_t W_s^{-1}            (G = I: the target moves exactly as the source would)
W_s, W_t are "true" interfaces, fitted on calibration data that the map did not use (another seed, or the other
half of the same run). Reported per condition:
    gain   s* = <G, I> / 6                              speed error is |s* - 1|
    dir    ||G - s* I||_F / ||s* I||_F                  the paper's direction error
    align  min over unit e of  e.(eG) / |eG|            cosine between the intended and the realised step, in the
                                                         worst direction; > 0 means every error shrinks under
                                                         feedback, < 0 means some error grows
    rho_T  5th percentile over the source demonstrations of task T of cos(a, aG): the same cosine, restricted to
           the steps the policy actually takes (task alignment)
Successes are read from the evaluation files (budget, analytic) or copied from the result summaries (pairs, real).
usage: /usr/bin/python3 alignment.py   (writes alignment.json and figures/fig_alignment.pdf next to this file)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_experiments/ur10e_hardware"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fit_compare import fit, load  # noqa: E402

X = Path(__file__).resolve().parents[1] / "results/xembody"
UNITS = np.array([0.01] * 3 + [0.05] * 3)
START_R = np.column_stack([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])   # tool z down


def W(path, phi="world", half=None):
    a, s, n, _ = load(path)
    if half is not None:
        k = len(a) // 2
        a, s, n = (a[:k], s[:k], n[:k]) if half == 0 else (a[k:], s[k:], n[k:])
    return np.asarray(fit(a, s, n, phi).matrix)


def align(G):
    def f(e):
        e = e / np.linalg.norm(e); g = e @ G
        return float(e @ g / max(np.linalg.norm(g), 1e-12))
    rng = np.random.default_rng(0)
    starts = rng.normal(size=(4000, len(G)))
    best = min(starts, key=f)
    for x0 in [best, *starts[:20]]:
        r = minimize(f, x0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10})
        best = r.x if f(r.x) < f(best) else best
    return f(best)


ACTS = {t: np.load(X / f"franka_demos_{t}.npz")["act"] for t in "AR"}
ACTS = {t: a[np.linalg.norm(a, axis=1) > 1e-3] for t, a in ACTS.items()}


def rho(G, t):
    a = ACTS[t]; g = a @ G
    c = np.sum(a * g, 1) / np.linalg.norm(a, axis=1) / np.maximum(np.linalg.norm(g, axis=1), 1e-12)
    return round(float(np.percentile(c, 5)), 3)


def stats(G, ok=None):
    s = float(np.trace(G) / 6)
    out = {"gain": round(s, 3), "dir": round(float(np.linalg.norm(G - s * np.eye(6)) / np.linalg.norm(s * np.eye(6))), 3),
           "align": round(align(G), 3), "align_t": round(align(G[:3, :3]), 3), "align_r": round(align(G[3:, 3:]), 3),
           "rho_A": rho(G, "A"), "rho_R": rho(G, "R")}
    if ok is not None:
        out["success"] = ok   # {task: successes of n}
    return out


def wins(path):
    e = json.loads(Path(path).read_text())["episodes"]
    return [sum(x["success"] for x in e), len(e)]


def analytic(cmd_yaw, offset=0.245):
    """Documented UR10e interface inverted at the start pose (as xembody.analytic_map), as a 6x6 action matrix."""
    cmd = Rotation.from_euler("z", cmd_yaw).as_matrix()
    A = np.zeros((6, 6))
    for i in range(6):
        a = np.eye(6)[i]; dp, dr = a[:3] * UNITS[:3], a[3:] * UNITS[3:]
        dp0 = dp - np.cross(dr, START_R @ np.array([0.0, 0.0, offset]))
        A[i] = np.concatenate([cmd.T @ dp0 / UNITS[:3], cmd.T @ dr / UNITS[3:]])
    return A


def main():
    out = {}
    pinv = np.linalg.pinv
    # --- simulation, three calibration pairs (seed k maps, truth from the next seed)
    for k, t in ((11, 12), (12, 13), (13, 11)):
        F = {p: W(X / f"long/franka_long_seed{k}.jsonl", p) for p in ("world", "body")}
        U = {p: W(X / f"long/ur10e_long_seed{k}.jsonl", p) for p in ("world", "body")}
        Ft, Ut = W(X / f"long/franka_long_seed{t}.jsonl"), W(X / f"long/ur10e_long_seed{t}.jsonl")
        eff = lambda M: M @ Ut @ np.linalg.inv(Ft)   # noqa: E731
        ok, no = {"A": [50, 50], "R": [50, 50]}, {"A": [0, 50], "R": [0, 50]}   # replication_summary.txt, Table 3
        out[f"sim pair {k}"] = {
            "direct": stats(eff(np.eye(6)), no),
            "matched world/world": stats(eff(F["world"] @ pinv(U["world"])), ok),
            "matched body/body": stats(eff(F["body"] @ pinv(U["body"])), ok),
            "mismatched world/body": stats(eff(F["world"] @ pinv(U["body"])), no),
        }
        for n in (16, 32, 64, 128, 300, 1500):
            m = json.loads((X / f"budget/map_s{k}_n{n}.json").read_text())
            out[f"sim pair {k}"][f"budget n{n}"] = stats(eff(np.array(m["W_F"]) @ pinv(np.array(m["W_U"]))),
                                                         {t: wins(X / f"budget/bud_{t}_s{k}_n{n}.json") for t in "AR"})
    # --- simulation, undocumented frame rotation theta (one UR10e run per theta: map from one half, truth the other)
    Ft = W(X / "long/franka_long_seed12.jsonl"); F11 = W(X / "long/franka_long_seed11.jsonl")
    for th in (0, 15, 30, 60, 90):
        run = X / f"analytic/ur10e_theta{th}.jsonl"
        Ut, Uh = W(run, half=1), W(run, half=0)
        eff = lambda M: M @ Ut @ np.linalg.inv(Ft)   # noqa: E731
        w = lambda c: {t: wins(X / f"analytic/an_{t}_theta{th}_{c}.json") for t in "AR"}   # noqa: E731
        out[f"sim theta {th}"] = {"direct": stats(eff(np.eye(6)), w("direct")),
                                  "analytic": stats(eff(analytic(np.pi)), w("analytic")),
                                  "calibration": stats(eff(F11 @ pinv(Uh)), w("matched"))}
    # --- real UR10e: maps from the simulated Franka seed 1 and real run 1; truth Franka seed 2, real run 2
    Fs = {p: W(X / "franka_calib_seed1.jsonl", p) for p in ("world", "body")}
    Ft = W(X / "franka_calib_seed2.jsonl")
    for frame, pre, yaw in (("real base_link", "xr", 0.0), ("real hidden 180", "xr180", np.pi)):
        U1 = {p: W(X / f"real/calib/{pre}_calib_seed1_T.jsonl", p) for p in ("world", "body")}
        Ut = W(X / f"real/calib/{pre}_calib_seed2_T.jsonl")
        eff = lambda M: M @ Ut @ np.linalg.inv(Ft)   # noqa: E731
        r = lambda k: {"R": [k, 10]}   # noqa: E731   (trials_summary.txt, trials_frame180_summary.txt)
        out[frame] = {"direct": stats(eff(np.eye(6)), r(10 if yaw == 0 else 0)),
                      "analytic (true frame)": stats(eff(analytic(yaw)), r(10)),
                      "matched world/world": stats(eff(Fs["world"] @ pinv(U1["world"])), r(10)),
                      "mismatched world/body": stats(eff(Fs["world"] @ pinv(U1["body"])), r(0))}
    for k, v in out.items():
        print(k)
        for c, s in v.items():
            print(f"   {c:24s} gain {s['gain']:+6.2f}  dir {s['dir']:5.2f}  align {s['align']:+6.2f}  "
                  f"t {s['align_t']:+5.2f} r {s['align_r']:+5.2f}  rho_A {s['rho_A']:+5.2f}  rho_R {s['rho_R']:+5.2f}  success {s.get('success')}")
    here = Path(__file__).resolve().parent
    (here / "alignment.json").write_text(json.dumps(out, indent=1))
    (here / "figures").mkdir(exist_ok=True)
    figure(out, here / "figures")


def family(cell, cond):
    if cond == "direct":
        return "commands unchanged"
    if "mismatched" in cond:
        return "mismatched calibration"
    if "analytic" in cond:
        return "hand-written conversion"
    return "matched calibration"


def figure(out, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"matched calibration": "#2a78d6", "hand-written conversion": "#eb6834",
           "mismatched calibration": "#1baf7a", "commands unchanged": "#eda100"}
    mk = {"matched calibration": "o", "hand-written conversion": "D", "mismatched calibration": "v",
          "commands unchanged": "s"}
    OFF = {"matched calibration": 0.0, "hand-written conversion": 2.5, "mismatched calibration": -2.5,
           "commands unchanged": 2.5}   # vertical offset so that coincident cells stay visible
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, t, name in zip(axes, "AR", ("Task A (pick and place)", "Task R (reach a pose)")):
        seen = set()
        for cell, v in out.items():
            for cond, s in v.items():
                if t not in s.get("success", {}):
                    continue
                k, n = s["success"][t]; f = family(cell, cond)
                ax.scatter(min(s["align_t"], s["align_r"]), 100 * k / n + OFF[f], s=30 if n == 50 else 60,
                           marker=mk[f], color=col[f], edgecolor="white", linewidth=0.8, zorder=3,
                           label=None if f in seen else f)
                seen.add(f)
        ax.axvline(0, color="0.75", lw=0.8, zorder=1)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(r"alignment $\alpha$ (worst of translation, rotation)", fontsize=8)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-6, 106)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="0.92", lw=0.6, zorder=0)
    axes[0].set_ylabel("success (%)", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("pdf", "png"):
        fig.savefig(d / f"fig_alignment.{ext}", dpi=200)


if __name__ == "__main__":
    main()
