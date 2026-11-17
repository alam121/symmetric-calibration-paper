"""Run one row of run_sheet_R.csv on the real UR10e and record the outcome in the sheet.

  /usr/bin/python3 run_trial.py RUN_NUMBER        (ROS env for domain 6 must be set; see SESSION_PLAN.md)
  /usr/bin/python3 run_trial.py RUN_NUMBER --note "what the operator saw"   (adds a note, no motion)
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
HW = Path(__file__).resolve().parents[3] / "paper_experiments/ur10e_hardware"
SHEET = R / "run_sheet_R.csv"
START = ["-1.397", "-0.745", "1.486", "4.001", "-1.593", "3.301"]


def read():
    lines = SHEET.read_text().splitlines(keepends=True)
    return lines[0], list(csv.reader(lines[1:]))


def write(header_line, rows):
    with SHEET.open("w", newline="") as f:
        f.write(header_line)
        csv.writer(f).writerows(rows)


n = int(sys.argv[1])
header_line, rows = read()
cols = rows[0]
i = next(k for k, r in enumerate(rows) if r and r[0] == str(n))
row = dict(zip(cols, rows[i]))

if len(sys.argv) > 3 and sys.argv[2] == "--note":
    rows[i][cols.index("operator_notes")] = sys.argv[3]
    write(header_line, rows)
    print(f"run {n}: note saved")
    sys.exit(0)

if row["outcome"]:
    sys.exit(f"run {n} already has an outcome ({row['outcome']}); not re-running")
trial = f"r{n:02d}_{row['target']}"
cmd = ["/usr/bin/python3", "xembody_real.py", "--policy", str(R / "policy_R.npz"), "--mode", row["mode"],
       "--franka_calib", str(Path(__file__).resolve().parents[3] / "results/xembody/franka_calib_seed1.jsonl"),
       "--real_calib", str(R / "calib/xr_calib_seed1_T.jsonl"),
       "--target", row["x"], row["y"], row["z"], row["yaw_deg"], "--start_joints", *START,
       "--trial", trial, "--out", str(R / "trials")]
print(f"run {n}: target {row['target']} ({row['x']}, {row['y']}, {row['z']}, yaw {row['yaw_deg']}), mode {row['mode']}",
      flush=True)
proc = subprocess.run(cmd, cwd=HW, capture_output=True, text=True)
for line in (proc.stdout + proc.stderr).splitlines():
    if "[XREAL]" in line or "Error" in line or "Traceback" in line:
        print(line, flush=True)
out = R / "trials" / f"{trial}_R_{row['mode']}.json"
if not out.exists():
    sys.exit(f"run {n}: no result file (exit {proc.returncode})")
d = json.loads(out.read_text())
last = d["steps"][-1] if d["steps"] else {}
rows[i][cols.index("outcome")] = d["outcome"]
rows[i][cols.index("final_pos_err_mm")] = f"{last.get('pos_err_mm', float('nan')):.1f}"
rows[i][cols.index("final_yaw_err_deg")] = f"{last.get('yaw_err_deg', float('nan')):.1f}"
write(header_line, rows)
tilt = max((s["tilt_deg"] for s in d["steps"]), default=float("nan"))
print(f"run {n}: {d['outcome']}; {len(d['steps'])} ticks; max tilt {tilt:.1f} deg; "
      f"IK refusals {sum(not s['sent'] for s in d['steps'])}", flush=True)
