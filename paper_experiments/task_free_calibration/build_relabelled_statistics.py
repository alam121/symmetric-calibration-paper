#!/usr/bin/env python3
"""Compute dataset statistics for CALVIN actions expressed in LIBERO convention.

Why this is needed at all: normalisation statistics are part of an action
convention, not a detail beside it. The calibrated arm relabels CALVIN's actions
into LIBERO's action space, which is a wider space -- 37.9% of relabelled actions
fall outside CALVIN's own q01/q99. Normalising them with CALVIN's statistics
under BOUNDS_Q99 would saturate more than a third of the training targets at
+-1, and the arm would fail for reasons unrelated to the hypothesis.

So the relabelled arm gets its own statistics, computed here over exactly the
relabelled distribution. Deployment stays consistent because the server
unnormalises with the statistics stored in the checkpoint.

Proprio is copied through unchanged: the relabelling touches actions only.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "paper_experiments/task_free_calibration"))

from verify_relabel_roundtrip import load, relabel  # noqa: E402

CALVIN = Path("/home/alam/vla_thesis/calvin/dataset/task_D_D/training")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibrator", type=Path,
                    default=REPO / "data/task_free_calibration/calibrator_seed101.json")
    ap.add_argument("--reference", type=Path,
                    default=Path("/home/alam/vla_thesis/final_results/calvin_native_oft_runs"
                                 "/calvin_task_d_native_oft_lora32--35000_chkpt"
                                 "/dataset_statistics.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all episodes")
    args = ap.parse_args()

    cal = load(args.calibrator)
    ref = json.loads(args.reference.read_text())
    key = next(iter(ref))

    files = sorted(glob.glob(str(CALVIN / "episode_*.npz")))
    if args.limit:
        files = files[:args.limit]
    print(f"reading {len(files)} transitions", flush=True)

    acts = np.stack([np.load(f)["rel_actions"] for f in files]).astype(np.float64)
    lib = np.asarray(relabel(cal, acts), dtype=np.float64)

    stats = json.loads(json.dumps(ref))  # deep copy; keep proprio and counts
    a = stats[key]["action"]
    a["mean"] = lib.mean(0).tolist()
    a["std"] = lib.std(0).tolist()
    a["max"] = lib.max(0).tolist()
    a["min"] = lib.min(0).tolist()
    a["q01"] = np.percentile(lib, 1, axis=0).tolist()
    a["q99"] = np.percentile(lib, 99, axis=0).tolist()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {args.out}")

    old_q01 = np.asarray(ref[key]["action"]["q01"])
    old_q99 = np.asarray(ref[key]["action"]["q99"])
    new_q01 = np.asarray(a["q01"])
    new_q99 = np.asarray(a["q99"])
    print("\n  CALVIN-convention q01/q99 (old):")
    print("   ", np.array2string(old_q01[:6], precision=3, floatmode="fixed"))
    print("   ", np.array2string(old_q99[:6], precision=3, floatmode="fixed"))
    print("  LIBERO-convention q01/q99 (new):")
    print("   ", np.array2string(new_q01[:6], precision=3, floatmode="fixed"))
    print("   ", np.array2string(new_q99[:6], precision=3, floatmode="fixed"))

    sat_old = ((lib[:, :6] < old_q01[:6]) | (lib[:, :6] > old_q99[:6]))
    sat_new = ((lib[:, :6] < new_q01[:6]) | (lib[:, :6] > new_q99[:6]))
    print(f"\n  saturation of relabelled actions under OLD stats: "
          f"{100*sat_old.any(1).mean():.2f}% of actions")
    print(f"  saturation of relabelled actions under NEW stats: "
          f"{100*sat_new.any(1).mean():.2f}% of actions   <- must be ~2% (q01/q99 by construction)")
    ok = sat_new.any(1).mean() < 0.15
    print(f"\n{'OK' if ok else 'FAILED'} - statistics {'match' if ok else 'do NOT match'} "
          f"the relabelled distribution")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
