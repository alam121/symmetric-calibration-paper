"""The relative Cartesian action interface both the sim and the UR10e implement (sim-to-real study).

One nominal specification, implemented natively on each side: what differs between the sim's
and the robot's realised motion for the same command is the interface gap that task-free
calibration (W_sim, W_real) measures and a_real = a_sim W_sim pinv(W_real) compensates.

  a_t = [dx, dy, dz, drx, dry, drz] in [-1, 1], at RATE_HZ, streamed (no waiting to settle)
  translation: UNIT_TRANSLATION m per unit, robot base frame
  rotation:    UNIT_ROTATION rad per unit, rotation vector applied as R_target = exp(dr) R_now
  commanded frame: tool0
"""

RATE_HZ = 10.0
UNIT_TRANSLATION = 0.01
UNIT_ROTATION = 0.05
CLIP = 1.0
TOOL_FRAME = "tool0"
GRASP_DEPTH = 0.225  # tool0 -> Delto grasp point along tool z (the date-harvest value)

# Shared task geometry (base frame, metres). Mirror these on the table, or pass the measured values.
BLOCK_SIZE = (0.10, 0.03, 0.08)        # long, narrow, height (tall enough that the Delto fingertips,
                                       # ~35 mm below the grasp point, clear the table; 30 mm wide leaves
                                       # ~9 mm between the block and each open fingertip cap)
GRASP_ABOVE_CENTRE = 0.025             # grasp point this far above the block centre: the open F2 cap
                                       # touched the table at 0.01 and the receptacle wall at 0.02
RECEPTACLE_CLEARANCE = 0.004           # per side, on width and length
RECEPTACLE_WALL = (0.01, 0.015)        # wall thickness, height (15 mm still stops a misaligned block;
                                       # the open F2, ~59 mm out along the block, must pass over it)
