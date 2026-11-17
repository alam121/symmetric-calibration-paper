#!/usr/bin/env python3
"""Predict, before calibrating, whether calibration can help.

Task-free calibration repairs an interface mismatch and cannot repair a semantic
mismatch. A practitioner therefore needs to know which they have *before*
spending effort. This computes a decision statistic from the same task-free
interaction shards the calibrator is fitted from -- no demonstrations, language,
rewards, success labels, or policy gradients.

Statistic: the per-channel appearance shift between source and target
observations, standardised by their pooled spread and averaged over cameras.
Small shift means the target is visually in-distribution for the policy, so any
failure is attributable to the interface and calibration should pay off. Large
shift means the policy is being asked about a world it has not seen, where no
interface correction can supply the missing semantics.

The statistic is deliberately cheap: it is already a by-product of fitting the
calibrator, so the diagnostic costs nothing beyond what calibration collects.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

# Deliberately dependency-free: the statistic is arithmetic over a handful of
# per-channel moments, and keeping it importable from any interpreter lets the
# paper's table generator reuse it rather than reimplement it.

# Separates every interface-only case measured here (max 0.075) from the
# semantic case (0.411) with roughly 2x margin on either side. Calibrated on a
# small number of cases; treat as indicative, not a tuned operating point.
THRESHOLD = 0.15


def visual_shift(calibrator: dict) -> float | None:
    """Standardised source-to-target appearance shift, averaged over cameras."""
    source = calibrator.get("source_image_stats")
    target = calibrator.get("target_image_stats")
    if not source or not target:
        return None
    shifts = []
    for camera in source:
        if camera not in target:
            continue
        channels = zip(source[camera]["mean"], source[camera]["std"],
                       target[camera]["mean"], target[camera]["std"])
        per_channel = [
            abs(s_mean - t_mean) / max((s_std + t_std) / 2.0, 1e-6)
            for s_mean, s_std, t_mean, t_std in channels
        ]
        if per_channel:
            shifts.append(sum(per_channel) / len(per_channel))
    return sum(shifts) / len(shifts) if shifts else None


def verdict(shift: float) -> str:
    return ("interface mismatch -- calibration should help"
            if shift < THRESHOLD else
            "semantic mismatch -- calibration cannot help; collect demonstrations")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrator", help="single calibrator JSON to score")
    parser.add_argument("--scan", metavar="GLOB",
                        help="score every calibrator matching a glob")
    args = parser.parse_args()

    paths = [args.calibrator] if args.calibrator else sorted(glob.glob(args.scan or ""))
    if not paths:
        parser.error("pass --calibrator or --scan")

    seen: dict[float, str] = {}
    for path in paths:
        shift = visual_shift(json.loads(open(path).read()))
        if shift is None:
            print(f"{os.path.basename(path):34s} no image statistics")
            continue
        duplicate = " (duplicate statistic)" if round(shift, 6) in seen else ""
        seen.setdefault(round(shift, 6), path)
        print(f"{os.path.basename(path):34s} shift={shift:.4f}  {verdict(shift)}{duplicate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
