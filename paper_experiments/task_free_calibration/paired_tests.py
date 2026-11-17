#!/usr/bin/env python3
"""Paired McNemar tests between evaluation arms that share a sequence seed.

CALVIN's long-horizon protocol gives each sequence the same start state and the
same task chain across arms when --sequence-seed matches, so arm-vs-arm
comparison should be paired, not treated as two independent proportions. With
n = 100 the unpaired Wilson intervals on 85% and 90% overlap heavily and settle
nothing; the paired test conditions on the sequences where the two arms differ
and is far more sensitive.

D1 is the fraction of sequences completing at least one subtask, i.e.
chain_length >= 1. The exact binomial on the discordant pairs is used rather
than the chi-square approximation, because the discordant count here is small
(single digits) and the approximation is unreliable in that regime.

COMPLETENESS GATE. per_sequence_results.json is written incrementally while an
evaluation runs, so a partially-written file looks like a valid shorter run. An
86-row file once produced a confident 57-vs-74 comparison that was pure
artefact. Every arm is therefore required to (a) have a manifest reporting a
finished run and (b) carry exactly the expected number of sequences; anything
else is refused rather than silently compared.
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "results/task_free_calibration"


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact binomial on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(rel: str, expect: int = 100) -> dict:
    d = R / rel
    ps, mf = d / "per_sequence_results.json", d / "manifest.json"
    if not ps.exists():
        raise FileNotFoundError(f"{rel}: no per_sequence_results.json")
    payload = json.loads(ps.read_text())
    chains = payload["chain_lengths"]
    if len(chains) != expect:
        raise ValueError(f"{rel}: {len(chains)} sequences, expected {expect} "
                         f"-- incomplete or still running, refusing to compare")
    status = None
    if mf.exists():
        status = json.loads(mf.read_text()).get("status")
    return {"name": rel.split("/")[-1], "seed": payload["sequence_seed"],
            "chains": chains, "status": status}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="result dirs relative to results/task_free_calibration")
    ap.add_argument("--expect", type=int, default=100)
    args = ap.parse_args()

    loaded = []
    for a in args.arms:
        try:
            loaded.append(load(a, args.expect))
        except (FileNotFoundError, ValueError) as e:
            print(f"SKIP {e}")
    if not loaded:
        return 1

    print(f"{'arm':34s} {'seed':>6s} {'D1':>7s} {'95% CI':>16s} {'avg chain':>10s}")
    print("-" * 80)
    for a in loaded:
        ch = a["chains"]
        k = sum(1 for x in ch if x >= 1)
        lo, hi = wilson(k, len(ch))
        print(f"{a['name'][:34]:34s} {a['seed']:>6d} {100*k/len(ch):6.1f}% "
              f"[{100*lo:5.1f},{100*hi:5.1f}] {sum(ch)/len(ch):10.3f}")

    rows = []
    for x, y in combinations(loaded, 2):
        if x["seed"] != y["seed"]:
            print(f"{x['name'][:20]} vs {y['name'][:20]}: seeds differ, not paired")
            continue
        bx = [1 if v >= 1 else 0 for v in x["chains"]]
        by = [1 if v >= 1 else 0 for v in y["chains"]]
        b = sum(1 for i, j in zip(bx, by) if i == 1 and j == 0)
        c = sum(1 for i, j in zip(bx, by) if i == 0 and j == 1)
        rows.append((f"{x['name'][:21]} vs {y['name'][:21]}", b, c, exact_mcnemar(b, c)))

    # Holm-Bonferroni over the whole family. Running every pair of arms and then
    # reading the smallest p as a result is the classic way to manufacture a
    # finding: with ten comparisons at alpha=0.05 roughly one nominal hit is
    # expected under a complete null. The adjusted column is the one to quote.
    order = sorted(range(len(rows)), key=lambda i: rows[i][3])
    m = len(rows)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * rows[i][3]))
        adjusted[i] = running

    print(f"\n{'pair':46s} {'b':>3s} {'c':>3s} {'disc':>5s} {'p':>8s} {'p_holm':>8s}")
    print("-" * 82)
    for i, (label, b, c, p) in enumerate(rows):
        mark = "  *" if adjusted[i] < 0.05 else ("  (ns after correction)" if p < 0.05 else "")
        print(f"{label:46s} {b:3d} {c:3d} {b+c:5d} {p:8.4f} {adjusted[i]:8.4f}{mark}")
    print(f"\nb = first arm succeeded, second failed.  c = the reverse.")
    print(f"p_holm = Holm-Bonferroni adjusted over all {m} pairs in this call. "
          f"Quote p_holm, not p,\nunless the comparison was pre-registered as the single "
          f"primary test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
