# Task-free calibration run status

Last updated: **2026-09-02 Asia/Riyadh**.

## Implemented and validated

- Frozen protocol with an explicit no-language, no-task-ID, no-reward,
  no-success-label, and no-expert-action supervision boundary.
- Smooth bounded random-excitation collector for CALVIN target transitions.
- Random-excitation collector for the LIBERO source controller.
- Per-shard leakage and excitation audit.
- Affine local-dynamics interface fingerprint for source and target controllers.
- Source-to-target action-effect transduction through target inverse dynamics.
- Task-free target-to-source proprioception alignment with subtask-relative TCP
  coordinates and absolute gripper alignment.
- Optional per-camera RGB moment alignment from unlabeled frames.
- Deployment-server integration with independent action, proprioception, and
  vision calibration switches.
- Disjoint CALVIN development sequence seed `1729`; official final seed `0`
  remains untouched. A 100-sequence audit found `0` exact sequence overlaps.
- Unit and regression tests pass.

## Results obtained

The task-free datasets each contain `20 × 64 = 1,280` transitions and pass the
leakage audit. The affine dynamics models achieved held-out motion
`R² = 0.876` (LIBERO) and `R² = 0.916` (CALVIN).

Full LIBERO-to-CALVIN transfer on 20 disjoint development sequences showed that
interface fitting alone cannot repair the simultaneous visual-semantic shift:

| Condition | Average chain | Depth-1 |
| --- | ---: | ---: |
| Direct frozen transfer | 0.05 | 5% |
| Action-only calibration | 0.05 | 5% |
| Proprioception-only | 0.00 | 0% |
| Joint action + proprioception | 0.00 | 0% |
| Full RGB/state/action moments | 0.00 | 0% |

A controlled experiment held the capable CALVIN-native VLA, observations,
tasks, and evaluation sequences fixed while changing only the actuator
interface. The hidden interface swaps axes, changes signs/scales, and reverses
gripper polarity. Calibration used `1,280` random transitions and no task
signal.

| Controlled condition, seed 0 pilot | Average chain | Depth-1 | Depth-2 |
| --- | ---: | ---: | ---: |
| Clean original interface | **2.35** | **75%** | **60%** |
| Hidden interface, uncalibrated | 0.05 | 5% | 0% |
| Hidden interface, task-free calibrated | **0.40** | **35%** | **5%** |

The `+30` percentage-point depth-1 gain passes the first diagnostic gate. The
calibrated policy succeeded on five task types. Relative to the hidden-interface
failure, it recovered `42.9%` of the depth-1 gap to the clean ceiling but only
`15.2%` of the average-chain gap. This remains preliminary until additional
shifts, calibration seeds, and larger matched evaluations finish.

### 100-sequence confirmation

The same calibration and hidden interface were subsequently evaluated on 100
matched development sequences:

| Condition | Average chain | Depth-1 | Depth-2 | Depth-3 |
| --- | ---: | ---: | ---: | ---: |
| Hidden interface, uncalibrated | 0.02 | 2% | 0% | 0% |
| Hidden interface, task-free calibrated | **0.61** | **43%** | **17%** | **1%** |

The absolute gains are `+41` percentage points at depth 1 and `+0.59` average
chain length. The calibrated policy succeeded on seven task types, compared
with one for the uncalibrated policy. Wilson 95% intervals for depth 1 are
`33.7–52.8%` and `0.6–7.0%`, respectively. An aggregate one-sided Fisher exact
test gives `p = 1.41e-13`; this is not a substitute for a paired test because
the current result files retain aggregate rather than per-sequence outcomes.
The predefined development gate passed.

Artifacts are under `results/task_free_calibration/closed_loop/` and
`results/task_free_calibration/controlled_interface/`.

## Falsification rule

Offline transition reconstruction is not evidence that the method works. The
first method passes only if it improves matched closed-loop CALVIN performance.
The 100-sequence development gate additionally requires a depth-1 gain of at
least 10 percentage points and non-zero success on at least two task types.
