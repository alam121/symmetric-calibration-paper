"""Export a trained xembody policy (.pt) to plain numpy (.npz) for the robot side (no torch there).

  ~/lerobot_venv/bin/python isaac/sim2real/xembody_export.py POLICY.pt OUT.npz
"""

import sys

import numpy as np
import torch

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
rep = ck.get("rep") or ("planar" if ck.get("planar") else "tcp_full" if ck.get("tcp_full") else "full")
arrays = {k.replace(".", "_"): v.numpy() for k, v in ck["state"].items()}
np.savez(sys.argv[2], mu=np.asarray(ck["mu"]), sd=np.asarray(ck["sd"]), rep=np.array(rep), **arrays)
print(f"{sys.argv[1]} -> {sys.argv[2]} (rep {rep}, layers {sorted(arrays)})")
