#!/usr/bin/env python3
"""Closed-loop UR10e trials: counterbalanced run sheet and paired analysis (steps 15-24).

  plan     a CSV run sheet: every task x start pose x calibration condition, the condition order
           rotated per pose (cyclic Latin square), with empty result columns. Fill one row per
           trial as it is run; success criteria are written into the sheet header before any run.
  analyse  per task and condition: success with a Wilson 95% interval, median position /
           orientation error; matched vs each mismatched condition: exact McNemar on the paired
           binary outcomes (paired_tests.exact_mcnemar) and a Wilcoxon signed-rank test on the
           paired errors.

  python3 trials.py plan --poses 10 --out trials.csv
  python3 trials.py analyse trials.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_experiments.task_free_calibration.paired_tests import exact_mcnemar, wilson  # noqa: E402

CONDITIONS = ["body/body", "world/body", "body/world"]
TASKS = {
    "tolerant": "success = final object centre within 30 mm of the target, no collision",
    "critical": "success = insertion / aligned grasp completed, no collision, object in the required region",
}
COLUMNS = ["task", "pose", "order", "condition", "success", "position_error_mm", "orientation_error_deg",
           "collision", "time_s", "notes"]


def plan(args):
    rows = []
    for task in args.tasks:
        for pose in range(1, args.poses + 1):
            shift = (pose - 1) % len(args.conditions)
            for order, condition in enumerate(args.conditions[shift:] + args.conditions[:shift], 1):
                rows.append({"task": task, "pose": f"pose{pose:02d}", "order": order, "condition": condition})
    with args.out.open("w", newline="") as f:
        for task in args.tasks:
            f.write(f"# {task}: {TASKS.get(task, 'define success before running')}\n")
        f.write("# Fixed: robot, controller, TCP, workspace, policy. Only the calibration condition changes.\n")
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[TRIALS] {len(rows)} trials ({len(args.tasks)} tasks x {args.poses} poses x {len(args.conditions)} "
          f"conditions) -> {args.out}")


def truthy(value) -> bool | None:
    value = str(value).strip().lower()
    return True if value in ("1", "y", "yes", "true", "success") else False if value in ("0", "n", "no", "false",
                                                                                        "failure", "fail") else None


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def analyse(args):
    lines = [line for line in args.sheet.read_text().splitlines() if not line.startswith("#")]
    rows = [r for r in csv.DictReader(lines) if truthy(r.get("success")) is not None]
    from scipy.stats import wilcoxon

    matched = args.matched
    for task in sorted({r["task"] for r in rows}):
        trial = {(r["pose"], r["condition"]): r for r in rows if r["task"] == task}
        conditions = sorted({c for _, c in trial}, key=lambda c: (c != matched, c))
        print(f"\n== {task}")
        print(f"{'condition':12s} {'n':>3s} {'success':>8s} {'95% CI':>14s} {'pos err mm':>11s} {'ori err deg':>12s} {'collisions':>10s}")
        for condition in conditions:
            rs = [r for (p, c), r in trial.items() if c == condition]
            k = sum(truthy(r["success"]) for r in rs)
            lo, hi = wilson(k, len(rs))
            pos = [number(r["position_error_mm"]) for r in rs if number(r["position_error_mm"]) is not None]
            ori = [number(r["orientation_error_deg"]) for r in rs if number(r["orientation_error_deg"]) is not None]
            col = sum(truthy(r.get("collision")) or False for r in rs)
            print(f"{condition:12s} {len(rs):3d} {k / len(rs):8.0%} {f'[{lo:.0%}, {hi:.0%}]':>14s} "
                  f"{np.median(pos) if pos else float('nan'):11.1f} {np.median(ori) if ori else float('nan'):12.1f} {col:10d}")
        for condition in conditions:
            if condition == matched:
                continue
            poses = sorted({p for p, c in trial if c == condition} & {p for p, c in trial if c == matched})
            a = [truthy(trial[(p, matched)]["success"]) for p in poses]
            b = [truthy(trial[(p, condition)]["success"]) for p in poses]
            only_matched = sum(x and not y for x, y in zip(a, b))
            only_other = sum(y and not x for x, y in zip(a, b))
            line = (f"{matched} vs {condition}: {len(poses)} paired starts, discordant {only_matched}/{only_other}, "
                    f"exact McNemar p = {exact_mcnemar(only_matched, only_other):.4f}")
            for column in ("position_error_mm", "orientation_error_deg"):
                pairs = [(number(trial[(p, matched)][column]), number(trial[(p, condition)][column])) for p in poses]
                pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                if len(pairs) >= 5 and any(x != y for x, y in pairs):
                    line += f"; {column} Wilcoxon p = {wilcoxon([x for x, _ in pairs], [y for _, y in pairs]).pvalue:.4f}"
            print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--poses", type=int, default=10)
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--out", type=Path, default=Path("trials.csv"))
    a = sub.add_parser("analyse")
    a.add_argument("sheet", type=Path)
    a.add_argument("--matched", default="body/body")
    args = parser.parse_args()
    plan(args) if args.command == "plan" else analyse(args)


if __name__ == "__main__":
    main()
