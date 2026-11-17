"""Behaviour cloning on the Franka demos (xembody.py --mode collect): observation -> 6-D action + gripper.

State-based MLP; trained on the Franka only and frozen. Successful demo episodes only; 10% of the
episodes held out for validation.

  ~/miniconda3/envs/env_isaaclab/bin/python isaac/sim2real/xembody_train.py ~/sim2real_runs/xembody/franka_demos_B.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


def planar(obs):
    """Full observation (25: TCP pos, TCP rot6, object pos, object rot6, target pos, target sin/cos, gripper, ticks)
    -> the planar one (17): positions plus only the yaw (sin, cos) of the TCP and the object. Top-down tasks need no
    tilt, and tilt is where the hands differ: the Franka's jaws hold the box upright, the Delto's pinch lets it
    hang ~11 deg -- an input the Franka-trained policy never saw."""
    obs = np.asarray(obs, dtype=np.float32)
    one = obs.ndim == 1
    o = obs[None] if one else obs
    tyaw = np.arctan2(o[:, 4], o[:, 3])      # rot6 = [first column, second column]: x axis = (r00, r10, r20)
    oyaw = np.arctan2(o[:, 13], o[:, 12])
    out = np.concatenate([o[:, 0:3], np.sin(tyaw)[:, None], np.cos(tyaw)[:, None], o[:, 9:12],
                          np.sin(oyaw)[:, None], np.cos(oyaw)[:, None], o[:, 18:25]], axis=1)
    return out[0] if one else out


def tcp_full(obs):
    """Planar for the object (yaw only: the hands hold it differently) but the TCP's full orientation, so the
    policy can see and correct tool tilt -- which the Franka never drifts into but the mapped UR10e can."""
    obs = np.asarray(obs, dtype=np.float32)
    one = obs.ndim == 1
    o = obs[None] if one else obs
    oyaw = np.arctan2(o[:, 13], o[:, 12])
    out = np.concatenate([o[:, 0:9], o[:, 9:12], np.sin(oyaw)[:, None], np.cos(oyaw)[:, None], o[:, 18:25]], axis=1)
    return out[0] if one else out


def relative(obs):
    """Task-relative (18): object - TCP, target - object, TCP rot6, object yaw, target yaw, gripper, ticks.
    No absolute workspace coordinates: translating the whole scene leaves it unchanged."""
    obs = np.asarray(obs, dtype=np.float32)
    one = obs.ndim == 1
    o = obs[None] if one else obs
    oyaw = np.arctan2(o[:, 13], o[:, 12])
    out = np.concatenate([o[:, 9:12] - o[:, 0:3], o[:, 18:21] - o[:, 9:12], o[:, 3:9],
                          np.sin(oyaw)[:, None], np.cos(oyaw)[:, None], o[:, 21:25]], axis=1)
    return out[0] if one else out


def proprio(obs):
    """Robot state only (11): TCP position, TCP rot6, gripper command, ticks since it changed. Object and target
    must come from the image."""
    obs = np.asarray(obs, dtype=np.float32)
    one = obs.ndim == 1
    o = obs[None] if one else obs
    out = np.concatenate([o[:, 0:9], o[:, 23:25]], axis=1)
    return out[0] if one else out


REPS = {"planar": None, "tcp_full": None, "relative": None, "full": None, "proprio": None}   # filled below


def translate_xy(raw, rng, max_shift):
    """Consistent planar translation of TCP, object and target (actions are relative: unchanged)."""
    d = rng.uniform(-max_shift, max_shift, (len(raw), 2)).astype(np.float32)
    out = raw.copy()
    for i in (0, 9, 18):
        out[:, i:i + 2] += d
    return out


class ImageNet(nn.Module):
    """64x64 RGB wrist image (CNN) + proprioception -> 6-D action + gripper logit."""

    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 32, 5, stride=2), nn.ReLU(), nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=2), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 6 * 6, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 7))

    def forward(self, x, img):
        out = self.head(torch.cat([self.cnn(img), x], dim=1))
        return torch.cat([torch.tanh(out[:, :6]), out[:, 6:]], dim=1)


class Net(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 7))

    def forward(self, x):
        out = self.body(x)
        return torch.cat([torch.tanh(out[:, :6]), out[:, 6:]], dim=1)   # actions in [-1, 1], gripper logit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("demos", type=Path)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sd_floor", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--obs_noise", type=float, default=0.1, help="std of input noise in normalised units while training")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--full_obs", action="store_true", help="use the 25-D observation with full 3-D orientations")
    p.add_argument("--tcp_full", action="store_true", help="full TCP orientation, object yaw only (21-D)")
    p.add_argument("--image", action="store_true", help="train the image policy (needs demos collected with --camera)")
    p.add_argument("--mask_bottom", type=int, default=0,
                   help="black out the bottom N image rows (where the hand appears) in training and deployment")
    p.add_argument("--rep", choices=("planar", "tcp_full", "relative", "full", "proprio"), default=None,
                   help="observation representation (overrides --full_obs / --tcp_full)")
    p.add_argument("--init", type=Path, default=None,
                   help="fine-tune from this checkpoint (its representation, size and normalisation are kept)")
    p.add_argument("--label_sign", default=None,
                   help="per-component sign for the 6 action labels, e.g. -1,-1,1,-1,-1,1 (demos recorded in the task "
                   "frame, labels wanted in the UR10e's command frame)")
    p.add_argument("--max_episodes", type=int, default=None, help="use only the first N successful episodes")
    p.add_argument("--aug_shift", type=float, default=0.0, help="consistent xy translation augmentation, +- this [m]")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    d = np.load(a.demos)
    ok = np.flatnonzero(d["success"])
    if a.max_episodes is not None:
        ok = ok[:a.max_episodes]
    keep = np.isin(d["episode"], ok)
    episodes = rng.permutation(ok)
    val_eps = episodes[: max(1, len(ok) // 10)] if len(ok) > 1 else episodes[:0]
    val = keep & np.isin(d["episode"], val_eps)
    train = keep & ~val
    init = torch.load(a.init, map_location="cpu", weights_only=False) if a.init else None
    if init is not None:
        a.rep, a.hidden = init.get("rep") or ("planar" if init.get("planar") else "tcp_full"), init.get("hidden", 256)
    rep = a.rep or ("full" if a.full_obs else "tcp_full" if a.tcp_full else "planar")
    prep = REPS[rep]
    raw = d["obs"].astype(np.float32)
    obs = torch.tensor(prep(translate_xy(raw, rng, a.aug_shift) if a.aug_shift else raw), dtype=torch.float32)
    act, grip = (torch.tensor(d[k], dtype=torch.float32) for k in ("act", "grip"))
    if a.label_sign:
        act = act * torch.tensor([float(v) for v in a.label_sign.split(",")])
    # Floor the scale: features that are near-constant on the Franka (task A's tool orientation, std ~1e-3)
    # otherwise turn a sub-degree tilt on another robot into a 10-sigma input.
    mu, sd = obs[train].mean(0), torch.clamp(obs[train].std(0), min=a.sd_floor)
    if init is not None:
        mu, sd = init["mu"], init["sd"]
    x = (obs - mu) / sd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = None
    if a.image:
        imgs = torch.tensor(d["img"]).permute(0, 3, 1, 2).contiguous()   # uint8, N x 3 x 64 x 64
        if a.mask_bottom:
            imgs[:, :, -a.mask_bottom:, :] = 0
        net = ImageNet(obs.shape[1], a.hidden)
    else:
        net = Net(obs.shape[1], a.hidden)
    if init is not None:
        net.load_state_dict(init["state"])
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    xt, at, gt = x[train].to(dev), act[train].to(dev), grip[train].to(dev)
    train_idx = np.flatnonzero(train)
    xv, av, gv = x[val].to(dev), act[val].to(dev), grip[val].to(dev)
    bce = nn.BCEWithLogitsLoss()
    print(f"[TRAIN] {a.demos.name}: {len(ok)}/{len(d['success'])} successful episodes, {int(train.sum())} train / "
          f"{int(val.sum())} val steps, obs {obs.shape[1]}", flush=True)
    for step in range(a.steps + 1):
        i = torch.randint(0, len(xt), (a.batch,), device=dev)
        if a.aug_shift:   # a fresh translation per sample, per step
            j = train_idx[i.cpu().numpy()]
            xb = ((torch.tensor(prep(translate_xy(raw[j], rng, a.aug_shift))) - mu) / sd).to(dev)
        else:
            xb = xt[i]
        if imgs is not None:
            j = torch.as_tensor(np.flatnonzero(train))[i.cpu()]
            ib = imgs[j].to(dev).float() / 255.0
            ib = nn.functional.pad(ib, (4, 4, 4, 4), mode="replicate")   # random shift +-4 px
            dx, dy = np.random.randint(0, 9, 2)
            ib = ib[:, :, dy:dy + 64, dx:dx + 64]
            out = net(xb, ib)
        else:
            out = net(xb + a.obs_noise * torch.randn_like(xb))
        loss = ((out[:, :6] - at[i]) ** 2).mean() + 0.1 * bce(out[:, 6], gt[i])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 5000 == 0:
            if len(xv) == 0:
                print(f"[TRAIN] step {step}: train loss {loss.item():.4f} (no validation episodes)", flush=True)
                continue
            with torch.no_grad():
                if imgs is not None:
                    jv = torch.as_tensor(np.flatnonzero(val))
                    ov = torch.cat([net(xv[k:k + 2048], imgs[jv[k:k + 2048]].to(dev).float() / 255.0) for k in range(0, len(jv), 2048)])
                else:
                    ov = net(xv)
                v_mse = ((ov[:, :6] - av) ** 2).mean().item()
                v_acc = ((ov[:, 6] > 0).float() == gv).float().mean().item()
            print(f"[TRAIN] step {step}: train loss {loss.item():.4f}, val action MSE {v_mse:.4f}, val gripper acc {v_acc:.3f}", flush=True)
    out = a.out or a.demos.with_name(a.demos.stem.replace("demos", "policy") + ".pt")
    torch.save({"state": net.cpu().state_dict(), "in_dim": obs.shape[1], "hidden": a.hidden, "rep": rep, "aug_shift": a.aug_shift, "image": a.image, "mask_bottom": a.mask_bottom,
                "planar": rep == "planar", "tcp_full": rep == "tcp_full", "mu": mu, "sd": sd,
                "demos": str(a.demos), "val_episodes": val_eps.tolist()}, out)
    print(f"[TRAIN] -> {out}", flush=True)


REPS.update(planar=planar, tcp_full=tcp_full, relative=relative, proprio=proprio, full=lambda o: np.asarray(o, dtype=np.float32))


if __name__ == "__main__":
    sys.exit(main())
