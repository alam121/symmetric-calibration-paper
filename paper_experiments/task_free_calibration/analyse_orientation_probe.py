#!/usr/bin/env python3
"""Orientation probe: phi-matched vs phi-mismatched calibrators on tolerant vs critical tasks.

Each condition (orient_probe_{body_body,world_body,body_world}.generated.json, run by
run_orientation_probe.sh) evaluated ONE task from each of the same official CALVIN starts: the
first 50 starts whose first task is orientation-tolerant (push / slider / LED) and the first 50
whose first task is orientation-critical (rotate block). Outcomes are paired by start, so the
matched-vs-mismatched comparison is an exact McNemar test on discordant starts.

  python3 analyse_orientation_probe.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_experiments.task_free_calibration.paired_tests import exact_mcnemar, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "results/task_free_calibration/orientation_probe"
CONDITIONS = ["body_body", "world_body", "body_world"]


def load(condition: str) -> dict[int, dict]:
    path = next((ROOT / condition).glob("*/first_task_results.json"))
    return {row["official_index"]: row for row in json.loads(path.read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    runs = {c: load(c) for c in CONDITIONS}
    starts = sorted(set.intersection(*(set(r) for r in runs.values())))
    report = {"starts": len(starts), "families": {}}
    print(f"{len(starts)} starts evaluated under all three conditions\n")
    for family in ("tolerant", "critical"):
        ids = [i for i in starts if runs["body_body"][i]["family"] == family]
        print(f"== {family} ({len(ids)} paired starts)")
        print(f"{'condition':12s} {'success':>8s} {'95% CI':>14s}")
        fam = {"n": len(ids), "conditions": {}, "vs_matched": {}}
        for c in CONDITIONS:
            k = sum(runs[c][i]["success"] for i in ids)
            lo, hi = wilson(k, len(ids))
            fam["conditions"][c] = {"successes": k, "rate": k / len(ids), "ci": [lo, hi]}
            print(f"{c.replace('_', '/'):12s} {k / len(ids):8.0%} {f'[{lo:.0%}, {hi:.0%}]':>14s}")
        for c in CONDITIONS[1:]:
            only_matched = sum(runs["body_body"][i]["success"] and not runs[c][i]["success"] for i in ids)
            only_other = sum(runs[c][i]["success"] and not runs["body_body"][i]["success"] for i in ids)
            p = exact_mcnemar(only_matched, only_other)
            fam["vs_matched"][c] = {"only_matched": only_matched, "only_mismatched": only_other, "p": p}
            print(f"  body/body vs {c.replace('_', '/')}: discordant {only_matched} / {only_other}, exact McNemar p = {p:.2g}")
        tasks = sorted({runs["body_body"][i]["task"] for i in ids})
        fam["per_task"] = {}
        for t in tasks:
            tid = [i for i in ids if runs["body_body"][i]["task"] == t]
            fam["per_task"][t] = {c: f"{sum(runs[c][i]['success'] for i in tid)}/{len(tid)}" for c in CONDITIONS}
        print("  per task: " + "; ".join(f"{t} " + " ".join(v for v in fam["per_task"][t].values())
                                         for t in tasks))
        report["families"][family] = fam
        print()
    drop = {f: {c: report["families"][f]["conditions"]["body_body"]["rate"] - report["families"][f]["conditions"][c]["rate"]
                for c in CONDITIONS[1:]} for f in report["families"]}
    report["drop_vs_matched"] = drop
    print("Success drop from body/body: " + "; ".join(
        f"{f}: " + ", ".join(f"{c.replace('_', '/')} {d * 100:+.0f} pts" for c, d in v.items()) for f, v in drop.items()))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
