#!/usr/bin/env python3
"""Per-subtask outcomes by chain position, reconstructed from evaluation.log.

Why this is needed. Comparing the per-task rates in results.json across two arms
is confounded: CALVIN only attempts chain position k+1 when position k succeeded,
so a weaker arm accumulates fewer attempts and a biased task mix. Comparing a
degraded arm's task table with a healthy one therefore mixes "this task got
harder" with "this task was only reached in easier contexts".

Position 1 is free of that confound. Every sequence attempts it, and when two
arms share a --sequence-seed they draw the SAME position-1 task per sequence, so
the comparison is paired per sequence.

evaluation.log emits one CALVIN_VIDEO_DEBUG line per attempt, in order, so
sequences can be segmented: a sequence ends at the first failure or after five
successes. The reconstruction is checked against chain_lengths in
per_sequence_results.json and refuses to report if they disagree.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "results/task_free_calibration"
LINE = re.compile(r"subtask=(\S+)\s+success=(True|False)")


def attempts(rel: str) -> list[tuple[str, bool]]:
    log = R / rel / "evaluation.log"
    out = []
    for m in LINE.finditer(log.read_text(errors="replace")):
        out.append((m.group(1), m.group(2) == "True"))
    return out


def sequences(rel: str) -> list[list[tuple[str, bool]]]:
    """Segment the flat attempt list, then verify against the recorded chains."""
    seqs, cur = [], []
    for task, ok in attempts(rel):
        cur.append((task, ok))
        if not ok or len(cur) == 5:
            seqs.append(cur)
            cur = []
    if cur:
        seqs.append(cur)
    recorded = json.loads((R / rel / "per_sequence_results.json").read_text())["chain_lengths"]
    derived = [sum(1 for _, ok in s if ok) for s in seqs]
    if derived != recorded:
        raise ValueError(f"{rel}: reconstruction disagrees with chain_lengths "
                         f"({len(derived)} vs {len(recorded)} sequences); refusing to report")
    return seqs


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("comparison")
    ap.add_argument("--min-attempts", type=int, default=4)
    args = ap.parse_args()

    a, b = sequences(args.reference), sequences(args.comparison)
    print(f"reference  {len(a)} sequences  avg chain "
          f"{sum(sum(1 for _, ok in s if ok) for s in a)/len(a):.3f}")
    print(f"comparison {len(b)} sequences  avg chain "
          f"{sum(sum(1 for _, ok in s if ok) for s in b)/len(b):.3f}")

    # position 1, paired per sequence
    pa = [s[0] for s in a]
    pb = [s[0] for s in b]
    same = sum(1 for x, y in zip(pa, pb) if x[0] == y[0])
    print(f"\nposition-1 task identical in {same}/{min(len(pa), len(pb))} sequences")
    bb = sum(1 for x, y in zip(pa, pb) if x[1] and not y[1])
    cc = sum(1 for x, y in zip(pa, pb) if y[1] and not x[1])
    print(f"position-1 success: reference {sum(1 for _, ok in pa if ok)}/{len(pa)}  "
          f"comparison {sum(1 for _, ok in pb if ok)}/{len(pb)}  "
          f"b={bb} c={cc} p={mcnemar(bb, cc):.4f}")

    # per task, position 1 only, paired
    per: dict[str, list[int]] = {}
    for (ta, oka), (tb, okb) in zip(pa, pb):
        if ta != tb:
            continue
        r = per.setdefault(ta, [0, 0, 0])
        r[0] += 1
        r[1] += int(oka)
        r[2] += int(okb)
    print(f"\n{'task (position 1 only)':30s} {'n':>3s} {'ref':>6s} {'cmp':>6s} {'delta':>7s}")
    print("-" * 58)
    for t, (n, sa, sb) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        if n < args.min_attempts:
            continue
        print(f"{t:30s} {n:3d} {100*sa/n:5.0f}% {100*sb/n:5.0f}% {100*(sb-sa)/n:+6.0f}pp")
    tail = [(t, v) for t, v in per.items() if v[0] < args.min_attempts]
    if tail:
        n = sum(v[0] for _, v in tail)
        sa = sum(v[1] for _, v in tail)
        sb = sum(v[2] for _, v in tail)
        print(f"{'(tasks with n<%d, pooled)' % args.min_attempts:30s} {n:3d} "
              f"{100*sa/n:5.0f}% {100*sb/n:5.0f}% {100*(sb-sa)/n:+6.0f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
