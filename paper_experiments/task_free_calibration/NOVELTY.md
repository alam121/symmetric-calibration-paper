# Novelty audit and falsifiable claim

## Candidate contribution

**Post-hoc task-free interface calibration for an already-trained frozen VLA.**
The method treats a source VLA as a controller that cannot be retrained. It
collects generic, reward-free source and target transitions, estimates how each
control interface turns commands into physical TCP/gripper motion, aligns the
target proprioception distribution to the source policy input, and composes the
two interfaces at deployment. It does not use target task demonstrations,
language annotations, rewards, success labels, or VLA gradients.

The intended claim is deliberately narrower than zero-shot cross-embodiment
pretraining:

> A previously trained action-predicting VLA can be wrapped after training and
> recover closed-loop control under a new state/action interface using only
> short task-free calibration interaction.

This claim is valid only if the frozen calibrated policy improves matched
closed-loop success over direct transfer. Offline transition reconstruction is
not sufficient evidence.

The experiment is split deliberately. Controlled actuator-interface shifts
test the causal claim while holding task semantics and visual representation
fixed. LIBERO-to-CALVIN is retained as a harder boundary case: its first sweep
shows that task-free interface calibration cannot supply visual/task semantics
that the frozen VLA never learned. This distinction prevents an interface
method from being overclaimed as arbitrary zero-shot domain adaptation.

## First positive pilot

On 20 disjoint development sequences, a hidden axis/sign/scale/gripper shift
reduced the frozen CALVIN-native policy to `5%` depth-1 and `0.05` average chain
length. A wrapper fitted from `1,280` random, task-free transitions recovered
`35%` depth-1, `5%` depth-2, and `0.40` average chain length. This is a
`+30` percentage-point depth-1 improvement with the VLA checkpoint frozen.
The matched clean-interface ceiling was `75%` depth-1, `60%` depth-2, and
`2.35` average chain length. Calibration therefore recovered `42.9%` of the
lost depth-1 performance but only `15.2%` of the lost chain-length performance,
indicating residual error accumulation. It is evidence to continue, not a
publication-scale estimate.

The subsequent matched 100-sequence confirmation strengthened the result:
uncalibrated deployment achieved `2%` depth-1 and `0.02` average chain length,
whereas task-free calibration achieved `43%` depth-1, `17%` depth-2, `1%`
depth-3, and `0.61` average chain length. This is a `+41` percentage-point
depth-1 gain across seven successful task types. Wilson 95% depth-1 intervals
do not overlap (`0.6–7.0%` versus `33.7–52.8%`). The method has therefore
passed the development confirmation, while multi-seed, multi-shift, and
official-sequence evaluation remain required for a paper-scale claim.

## Closest work and distinction

| Work | What it already covers | Required distinction for this paper |
| --- | --- | --- |
| [Cross-Embodiment Skill Transfer via Latent Space Alignment](https://arxiv.org/abs/2406.01968) | Reward-free target alignment with cycle and latent-dynamics losses | Its source encoders, decoders, latent policy, and latent space are jointly designed/trained. Ours must wrap an arbitrary already-trained VLA post hoc without access to its original training objective. |
| [SPACE](https://arxiv.org/abs/2606.24049) | Cartesian state delta as a universal action and robot-specific command execution | SPACE trains the policy to emit its canonical state-delta representation. Ours must translate the native outputs of a pre-existing action VLA that was never trained for canonical delta control. |
| [X-VLA](https://openreview.net/pdf?id=kt51kZH4aG) | Cross-embodiment pretraining with embodiment-specific soft prompts | Ours is post-training calibration and cannot modify or repeat VLA pretraining. |
| [UniAct](https://openreview.net/pdf/2fef4a16699aaead92146b71023d7e47f7a28b2a.pdf) | Universal latent action codebook and embodiment decoders | Ours cannot require a universal action representation learned during source-policy training. |
| [LAP](https://openreview.net/pdf/1f7ea964389a751aebe5a1c5959ee4282446535d.pdf) | Language-action pretraining for zero-shot embodiment transfer | Ours starts from a conventional action-predicting checkpoint and adds only a deployment-time interface. |
| [LoRA-SP](https://arxiv.org/abs/2603.07404) | Adaptive parameter-efficient target fine-tuning | Ours takes no VLA gradient and uses no target task/action labels. |

The local-dynamics mapper by itself is not a sufficient novelty because action
adapters and dynamics alignment already exist. The defensible novelty is their
**post-hoc, black-box composition around a conventional frozen VLA under a
strict task-free supervision boundary**, together with controlled evidence that
separates state, action, camera, visual-domain, and task mismatch.

## Required evidence

1. Direct and calibrated policies use the same frozen VLA checkpoint and
   official evaluation sequences.
2. Calibration shards pass an automated leakage audit proving that they contain
   no task, language, reward, success, or expert-action fields.
3. Action-only, proprio-only, and joint calibration are evaluated separately.
4. Calibration duration is swept at 1, 5, 10, and 30 minutes.
5. Three independent calibration seeds are evaluated on 1,000 CALVIN sequences.
6. At least one second source checkpoint or deployment setting confirms that
   the result is not a LIBERO/CALVIN-specific coordinate trick.
7. For a strong journal claim, demonstrate the same post-hoc procedure on a
   physical robot without collecting target task demonstrations.

## Go/no-go thresholds

| Stage | Minimum evidence to continue |
| --- | --- |
| First 20-sequence diagnostic | Calibrated depth-1 and average chain length both strictly exceed matched direct transfer |
| 100-sequence development result | At least `10` percentage-point absolute depth-1 gain and success on more than one task family |
| Paper-scale result | At least `50%` depth-1 or `1.0` average chain length, three seeds, and materially lower target supervision than target-native LoRA |
| Reliability claim | Pre-registered gate: mean depth-1 at least `80%`, mean chain length at least `2.5`, and no collapsed seed |

If the first affine state/action compiler fails, the next permitted iterations
remain task-free: nonlinear inverse dynamics, camera-frame delta alignment,
history-conditioned system identification, and unpaired visual feature
alignment. Target expert trajectories are not permitted in the proposed method.
