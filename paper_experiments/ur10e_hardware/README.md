# UR10e hardware validation of interface calibration

Hardware version of the paper's two mechanisms: matched **excitation** (beta) and matched
**representation** (rotation convention phi). Same UR10e, end-effector, controller, TCP,
workspace and policy throughout; only the calibration condition changes.

| Step | Tool |
| --- | --- |
| 2-9 collect calibration runs | `ur10e_calibration_run.py` |
| 10-14 fit, compare, go / no-go | `fit_compare.py` |
| 15-24 trial sheet and paired statistics | `trials.py` |

Definitions are the paper's: `SmoothExcitation` (collect_calibration.py), `fit_local_action_model`
and the `euler_xyz` (world) / `euler_xyz_body` (body) conventions (calibrator.py), `blocks` /
`per_block` (analyse_gain_vs_direction.py), `wilson` / `exact_mcnemar` (paired_tests.py).

## Action and limits

`a_t = [dx, dy, dz, drx, dry, drz]`, clipped to +-0.65 as in the paper; 1 unit = 20 mm base-frame
translation / 5 deg world-frame rotation (`R_target = exp(a_rot * unit) R_now`), so a step is at
most 13 mm / 3.25 deg, hard-capped at 20 mm / 5 deg. The TCP stays inside a box around the start
position (default +-10/10/8 cm) and within 15 deg of the start orientation (the excitation
reflects off both; a target outside is never sent). IK is solved on the robot's own
`/robot_description`, checked against TF at start (2 mm), with a per-step joint cap (0.15 rad);
a start pose near a singularity (Jacobian condition > 200) is refused. Every step: one short
joint trajectory (`--step_time`, default 0.8 s), then wait until the joints are still.
Keep the e-stop in hand. `--dry_run` sends nothing.

**Same action magnitude for every beta (step 9).** The paper's excitation `m = beta m + (1 - beta) n`
has stationary std `sigma sqrt((1 - beta) / (1 + beta))`: 0.28 at beta 0, 0.08 at 0.85, 0.045 at
0.95. `--match_magnitude 0.15` (default) rescales the noise so every beta has the same stationary
std; `--match_magnitude 0` reproduces the paper's raw setting. Both are logged in the run's meta.

## Run order

```bash
source /opt/ros/humble/setup.bash          # with the UR driver and robot_state_publisher running
HOME_J="-1.352599 -0.582120 1.307726 2.136123 -1.784902 -1.059150"   # or your calibration start
R="--start_joints $HOME_J --steps 600 --out runs/"
python3 ur10e_calibration_run.py --run_id beta085_run1 --beta 0.85 --seed 1 $R --dry_run   # check first
python3 ur10e_calibration_run.py --run_id beta085_run1 --beta 0.85 --seed 1 $R
python3 ur10e_calibration_run.py --run_id beta085_run2 --beta 0.85 --seed 2 $R
python3 ur10e_calibration_run.py --run_id beta085_run3 --beta 0.85 --seed 3 $R
python3 ur10e_calibration_run.py --run_id beta000_run1 --beta 0.0  --seed 4 $R
python3 ur10e_calibration_run.py --run_id beta095_run1 --beta 0.95 --seed 5 $R
python3 fit_compare.py runs/ --json runs/report.json
```

`fit_compare.py` fits every run under both conventions from the same recorded poses (step 12: no
re-collection), then scores the composed map `W_A pinv(W_B)` against the identity (both sides are
the same robot) per block: repeatability (matched beta and phi: the hardware noise floor), beta
mismatch, and phi body/body, world/world, world/body, body/world. It ends with the step-14 verdict:
**GO** only if a mismatched phi breaks rotation (E_rot > 3x matched) while translation is
unchanged. On NO-GO, stop before any closed-loop trial.

## Closed-loop trials

```bash
python3 trials.py plan --poses 10 --out trials.csv     # 2 tasks x 10 poses x 3 conditions, order rotated per pose
# run, filling success / errors / collisions per row; success criteria are in the sheet header
python3 trials.py analyse trials.csv                  # Wilson CIs, exact McNemar, paired Wilcoxon
```

## Pipeline check on the PC (kinematic UR10e stand-in)

Before the robot, the whole pipeline was run against the bridge's kinematic UR10e mirror (ideal
tracking) with robot_state_publisher, 5 runs x 150 steps from the lab home (Jacobian condition 7):
every move executed, realised = commanded; matched world/world E = 0.000, body/body E_rot
0.06-0.13 (world-frame commands seen in the body frame over +-15 deg of orientation); world/body
and body/world E_rot 1.49-1.51 with rotation gain -0.10 to -0.13 and translation exact -- the
paper's phi pattern. A beta mismatch has no effect on an ideal arm (the paper's B(beta) comes
from real tracking dynamics), so that part needs the robot. `trials.py analyse` was checked on
synthetic outcomes only.

## Sim-to-real transfer through the action interface (no policy fine-tuning)

A state-based policy (object pose + robot state; `manipulatorsdatepalm/isaac/sim2real/policy.py`)
succeeds in Isaac on two tasks built to be reproduced on the table (`isaac/sim2real/spec.py`):
**A** (orientation tolerant) bring the Delto grasp point to a marked point; **B** (orientation
critical) pinch a 100 x 30 x 80 mm block with F1 + F3, lift, and seat it in a receptacle
(opening 108 x 38 mm, 15 mm walls, yawed 30 deg). Both sides implement one relative Cartesian
action spec natively: 6-D in [-1, 1] at 10 Hz, streamed, 1 cm / 0.05 rad per unit, base-frame
translation and `R = exp(dr) R_now` on tool0 -- Isaac Lab relative differential IK in the sim,
IK + one 0.1 s trajectory point per tick on the UR10e (`ur10e_interface.py`).

| | Sim (Isaac) | Robot |
| --- | --- | --- |
| task episodes | `tasks_sim.py --task A/B` (sim: A 5/5, B 12/12) | `sim2real_deploy.py --task A/B --mode ...` |
| task-free calibration, beta 0.85, streamed | `tasks_sim.py --calibrate RUN --seed N` | `ur10e_calibration_run.py --stream_hz 10 --unit_translation 0.01 --unit_rotation_deg 2.8648 --start_joints <sim start>` |

Deployment modes: `direct` (a), `matched` (a W_sim pinv(W_real), body/body by default) and
`mismatched` (W_sim^world pinv(W_real^body)). Measured so far: the sim interface realises ~63% of
each streamed step (R^2 0.99, run-to-run E_rel 0.04-0.06); an ideal kinematic UR10e stand-in
realises ~67%, so there matched ~ direct, while the mismatched map inverts ry / rz (-0.92, -0.90):
task A succeeds in all three modes, task B completes under direct and matched and never gets past
aligning the wrist under mismatched. Whether direct transfer fails on the real UR10e depends on
its own interface gap -- that is what the real calibration measures.

Robot run order: build the props to spec.py (or pass measured sizes), measure the receptacle
(x y z yaw) and marker in base_link, start at the sim's start joints
(-0.4228 -1.7953 2.1764 4.3534 -1.5905 -2.0051), collect 2-3 streamed calibration runs, then per
start pose and task run the three modes in the counterbalanced order of
`trials.py plan --conditions direct matched mismatched`, judging success on the table.
