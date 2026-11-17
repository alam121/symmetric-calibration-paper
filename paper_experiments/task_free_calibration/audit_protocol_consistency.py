#!/usr/bin/env python3
"""Check that every arm of a condition faces the same hidden interface.

An arm is only a valid comparison for another if both deploy the identical
distortion. This is easy to violate by hand: protocols are generated per
experiment, the distortion is spread across two environment variables
(CALVIN_DEPLOYMENT_ACTION_MATRIX and CALVIN_DEPLOYMENT_GRIPPER_POLARITY), and
nothing ties an oracle arm to the fitted arm it is supposed to bound.

That failure already happened. oracle_sign_scale deployed gripper polarity -1
while the hidden-interface and fitted arms of the same condition deployed +1, so
the oracle faced sign+scale PLUS a gripper inversion. Each arm was internally
consistent -- the oracle's own calibrator correctly inverted the gripper -- which
is why nothing looked wrong in isolation, and the published consequence was an
"upper bound" (85.0%) sitting below the method it was meant to bound (93.0%).

Two checks:

  1. Arms grouped by deployed action matrix must agree on deployed gripper
     polarity. Disagreement means the group is not one condition.
  2. Each arm's calibrator polarity, sign(source_slope * target_slope), must
     match its own deployed polarity. A mismatch means the repair inverts a
     gripper the distortion did not, or fails to invert one it did.

Exit status is nonzero if anything is flagged, so this can gate a commit.

A method that has been re-run under a corrected configuration can carry
"superseded": true. The arm is then excluded from the consistency grouping but
still listed, because the protocol remains the record of what was actually
executed and should not be deleted to make an audit pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parents[2] / "data/task_free_calibration"


def calibrator_polarity(path: str) -> int | None:
    p = Path(path)
    if not p.is_absolute():
        p = DATA / p.name
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return int(np.sign(d["source"]["gripper_slope"] * d["target"]["gripper_slope"]))
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def collect(directory: Path) -> list[dict]:
    arms = []
    for proto in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(proto.read_text())
        except json.JSONDecodeError:
            continue
        for name, method in (payload.get("methods") or {}).items():
            client = method.get("client_environment") or {}
            env = method.get("environment") or {}
            matrix = client.get("CALVIN_DEPLOYMENT_ACTION_MATRIX")
            if not matrix:
                continue                      # clean arm, no interface change
            cal = (env.get("CALVIN_TASK_FREE_CALIBRATION_PATH")
                   or client.get("CALVIN_CLIENT_CALIBRATION_PATH") or "")
            arms.append({
                "superseded": bool(method.get("superseded")),
                "protocol": proto.name,
                "method": name,
                "matrix": json.dumps(json.loads(matrix)),   # canonical form
                "deployed": client.get("CALVIN_DEPLOYMENT_GRIPPER_POLARITY"),
                "calibrator": Path(cal).name if cal else None,
                "polarity": calibrator_polarity(cal) if cal else None,
            })
    return arms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--directory", type=Path, default=HERE)
    ap.add_argument("--quiet", action="store_true", help="only report problems")
    args = ap.parse_args()

    arms = collect(args.directory)
    groups: dict[str, list[dict]] = {}
    for a in arms:
        groups.setdefault(a["matrix"], []).append(a)
    retired = [a for a in arms if a["superseded"]]

    problems = 0
    for index, (_, group) in enumerate(sorted(groups.items())):
        live = [a for a in group if not a["superseded"]]
        deployed = {a["deployed"] for a in live}
        split = len(deployed) > 1
        bad = [a for a in live
               if a["polarity"] is not None and a["deployed"] is not None
               and a["polarity"] != int(a["deployed"])]
        if not split and not bad:
            if not args.quiet:
                note = f" (+{len(group)-len(live)} superseded)" if len(live) != len(group) else ""
                print(f"condition {index}: {len(live):3d} arms, deployed gripper "
                      f"{deployed.pop() if deployed else '?'}, consistent{note}")
            continue
        problems += 1
        print(f"\ncondition {index}: PROBLEM")
        if split:
            print(f"  arms disagree on deployed gripper polarity: {sorted(deployed)}")
        for a in group:
            flag = "  <-- calibrator polarity != deployed" if a in bad else ""
            flag += "  [superseded]" if a["superseded"] else ""
            print(f"    {a['protocol']:46s} {a['method']:28s} "
                  f"deployed={str(a['deployed']):>4s} cal_polarity={str(a['polarity']):>4s} "
                  f"{a['calibrator'] or '(none)'}{flag}")

    print(f"\n{len(arms)} distorted arms across {len(groups)} conditions; "
          f"{len(retired)} superseded; {problems} condition(s) flagged")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
