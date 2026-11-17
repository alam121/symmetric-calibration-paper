# Task A (pick cylinder -> tray), V1 relative policy (Franka-trained, frozen), UR10e mountfix asset, 2026-11-04
50 episodes each, seed 2000. Calibration task R seed 11, 1500 steps; UR fit R2 0.978-0.983.
| condition | base_link | hidden 180 |
|---|---|---|
| expert through matched map (grasp gate) | 50/50 | 50/50 |
| direct | 50/50 | 0/50 |
| analytic (frame known) | 50/50 | 50/50 |
| matched world/world | 50/50 | 50/50 |
| matched body/body | 50/50 | 49/50 (ep 20 knocked cylinder over) |
| mismatched world/body | 0/50 | 0/50 |
| mismatched body/world | 0/50 | 0/50 |
Map diag translation ~0.59-0.68 (signs flipped in x,y for hidden180), rotation 0.60-0.63; mismatch flips the sign of rx,rz.
Every failure except hidden180 body/body ep 20 leaves the cylinder untouched at its start (no grasp, 400-step timeout, no tilt).
