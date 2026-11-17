"""Figures for the symmetric-calibration paper, from the result files in ~/sim2real_runs/xembody."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

X = Path(__file__).resolve().parents[1] / "results/xembody"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)


def rate(path):
    e = json.load(open(path))["episodes"]
    return 100.0 * sum(x["success"] for x in e) / len(e)


def shift_figure():
    s4 = [0, 3, 6, 9]
    s6 = [0, 3, 6, 9, 12, 15]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    a = ax[0]
    a.plot(s4, [rate(X / f"env/tcpfull_A_shift{d}_franka.json") for d in s4], "o-", color="0.35", label="absolute obs, Franka")
    a.plot(s4, [rate(X / f"env/tcpfull_A_shift{d}_matched.json") for d in s4], "s--", color="tab:orange", label="absolute obs, UR10e matched")
    a.plot(s6, [rate(X / f"relative/rel_V1_relative_shift{d}_franka.json") for d in s6], "o-", color="tab:green", label="relative obs, Franka")
    a.plot(s6, [rate(X / f"relative/rel_V1_relative_shift{d}_matched.json") for d in s6], "s--", color="tab:blue", label="relative obs, UR10e matched")
    a.set_title("Task A (pick and place)", fontsize=9)
    a = ax[1]
    a.plot(s4, [rate(X / f"env/env_R_shift{d}_franka.json") for d in s4], "o-", color="0.35", label="yaw-only obs, Franka")
    a.plot(s4, [rate(X / f"env/env_R_shift{d}_matched.json") for d in s4], "s--", color="tab:red", label="yaw-only obs, UR10e matched")
    a.plot(s4, [rate(X / f"env/tcpfull_R_shift{d}_franka.json") for d in s4], "o-", color="tab:green", label="full tool orientation, Franka")
    a.plot(s4, [rate(X / f"env/tcpfull_R_shift{d}_matched.json") for d in s4], "s--", color="tab:blue", label="full tool orientation, UR10e matched")
    a.set_title("Task R (pose and yaw)", fontsize=9)
    for a in ax:
        a.set_xlabel("layout shift from the demonstration region [cm]", fontsize=8)
        a.set_ylim(-5, 105); a.grid(alpha=0.3); a.tick_params(labelsize=8)
        a.legend(fontsize=6.5, loc="lower left")
    ax[0].set_ylabel("success [%] (50 episodes)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_shift.pdf"); fig.savefig(OUT / "fig_shift.png", dpi=200)


# Real UR10e trials: fixed categorical slots per mode (validated palette, light surface), plus line style and an
# end marker for aborts so identity never rests on colour alone.
MODES = [("direct", "direct", "#2a78d6", ":"), ("matched", "matched calibration", "#eb6834", "-"),
         ("analytic", "analytic (frame known)", "#1baf7a", "--"), ("mismatched", "mismatched calibration", "#eda100", "-.")]


def real_figure():
    panels = [("trials", "r", "commands in base_link (the frame the policy expects)", {"analytic": "analytic"}, 25),
              ("trials_frame180", "f180_r", "commands in a hidden frame turned 180 deg", {"analytic": "analytic_exact"}, 12)]
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.0), sharex=True, sharey="row", gridspec_kw={"height_ratios": [1.4, 1]})
    for col, (folder, prefix, title, alias, tilt_limit) in enumerate(panels):
        top, bottom = ax[0, col], ax[1, col]
        for key, label, colour, style in MODES:
            mode = alias.get(key, key)
            files = sorted((X / "real" / folder).glob(f"{prefix}*_R_{mode}.json"))
            for k, f in enumerate(files):
                d = json.load(open(f))
                t = [0.1 * s["step"] for s in d["steps"]]
                err = [s["pos_err_mm"] for s in d["steps"]]
                tilt = [s["tilt_deg"] for s in d["steps"]]
                top.plot(t, err, style, color=colour, lw=1.0, alpha=0.85, label=f"{label} ({len(files)})" if k == 0 else None)
                bottom.plot(t, tilt, style, color=colour, lw=1.0, alpha=0.85)
                if d["outcome"] != "success":
                    top.plot(t[-1], err[-1], "x", color=colour, ms=5, mew=1.2)
                    bottom.plot(t[-1], tilt[-1], "x", color=colour, ms=5, mew=1.2)
        top.axhline(10, color="0.4", lw=0.8, ls=(0, (2, 2)))
        top.set_yscale("log"); top.set_ylim(0.3, 600)
        top.set_title(title, fontsize=8.5)
        bottom.axhline(tilt_limit, color="0.4", lw=0.8, ls=(0, (2, 2)))
        bottom.text(7.9, tilt_limit + 0.8, f"tilt abort ({tilt_limit} deg)", fontsize=6.5, color="0.3", ha="right")
        bottom.set_ylim(0, 28); bottom.set_xlim(0, 8); bottom.set_xlabel("time [s]", fontsize=8)
        for a in (top, bottom):
            a.grid(alpha=0.25, lw=0.5); a.tick_params(labelsize=7.5)
    ax[0, 0].text(0.15, 6.0, "1 cm success band", fontsize=6.5, color="0.3")
    ax[0, 0].set_ylabel("TCP distance to target [mm]", fontsize=8)
    ax[1, 0].set_ylabel("tool tilt [deg]", fontsize=8)
    h, l = ax[0, 1].get_legend_handles_labels()
    ax[1, 1].legend(h, l, fontsize=6.5, loc="upper right", framealpha=0.9, borderpad=0.3, labelspacing=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_real.pdf"); fig.savefig(OUT / "fig_real.png", dpi=200)


if __name__ == "__main__":
    shift_figure()
    real_figure()
    print("figures ->", OUT)
