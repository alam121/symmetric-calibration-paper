"""Observation adapters for the frozen xembody image policy (paper 3, first experiment).

The source camera is the policy's training camera (xembody.py --camera: fixed relative to the TCP). A target camera
is the same camera mounted elsewhere on the hand. Three ways to give the frozen policy what it expects:

  plane      warp the target image into the source view through the table plane (known mounts + intrinsics)
  reproject  per-pixel depth: back-project the target image, move it into the source camera, z-buffered splat
             (exact for every visible surface; disoccluded pixels stay empty)
  latent     a small network between the frozen encoder and the frozen head: A(f(I_t)) ~ f(I_s), trained on
             paired task-free observations, with a feature or a behaviour (frozen-head action) loss
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

W = H = 64
FOCAL, APERTURE = 12.0, 20.955
FX = FOCAL / APERTURE * W
K = np.array([[FX, 0.0, W / 2], [0.0, FX, H / 2], [0.0, 0.0, 1.0]])


def source_mount(tcp_offset):
    """Source camera in the end-effector frame: 10 cm back along the tool axis, 14 cm along tool x, aimed 3 cm past
    the pinch point. Returns (position, rotation) with ROS optical axes (z forward, y down in the image)."""
    pos = np.array([0.14, 0.0, tcp_offset - 0.10])
    z = np.array([0.0, 0.0, tcp_offset + 0.03]) - pos
    z /= np.linalg.norm(z)
    x = np.array([0.0, 1.0, 0.0])
    return pos, np.column_stack([x, np.cross(z, x), z])


def target_mount(tcp_offset, shift):
    """The source mount moved by shift = (dx, dy, dz [m], rx, ry, rz [deg]), all in the source camera's own frame."""
    pos, rot = source_mount(tcp_offset)
    d = np.asarray(shift, float)
    return pos + rot @ d[:3], rot @ Rotation.from_euler("xyz", np.radians(d[3:])).as_matrix()


def quat_wxyz(rot):
    q = Rotation.from_matrix(rot).as_quat()
    return (q[3], q[0], q[1], q[2])


def camera_in_world(ee_p, ee_r, mount):
    pos, rot = mount
    return ee_p + ee_r @ pos, ee_r @ rot


def plane_homography(src_cam, tgt_cam, plane_z):
    """H mapping SOURCE pixels to TARGET pixels for points on the plane z = plane_z (world): use with
    cv2.warpPerspective(target_img, H, (W, H), flags=WARP_INVERSE_MAP) to render the target in the source view."""
    (ps, rs), (pt, rt) = src_cam, tgt_cam
    r_ts = rt.T @ rs                       # source-camera coords -> target-camera coords
    t_ts = rt.T @ (ps - pt)
    n_w = np.array([0.0, 0.0, 1.0])
    n_s = rs.T @ n_w                       # plane normal in source coords
    d_s = float(n_w @ ps - plane_z)        # distance of the source camera above the plane (along n)
    # plane in source coords: n_s . X = -d_s  ->  X_t = (R + t n^T / (-d_s)) X_s
    hom = K @ (r_ts + np.outer(t_ts, n_s) / (-d_s)) @ np.linalg.inv(K)
    return hom / hom[2, 2]


def plane_warp(target_rgb, src_cam, tgt_cam, plane_z):
    import cv2
    cv2.setNumThreads(1)   # inside the sim process: OpenCV's own thread pool on 128 cores stalled it ~20x
    hom = plane_homography(src_cam, tgt_cam, plane_z)
    return cv2.warpPerspective(target_rgb, hom, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def reproject(target_rgb, target_depth, src_cam, tgt_cam, supersample=3):
    """Move every visible target surface into the source camera: the target image and depth are supersampled,
    back-projected, transformed and splatted with a z-buffer (nearest surface wins). Pixels the target camera
    never saw stay black (missing information, not a reprojection error)."""
    import cv2
    cv2.setNumThreads(1)
    (ps, rs), (pt, rt) = src_cam, tgt_cam
    n = supersample
    rgb = cv2.resize(np.asarray(target_rgb), (W * n, H * n), interpolation=cv2.INTER_NEAREST)
    z = cv2.resize(np.asarray(target_depth, np.float32).reshape(H, W), (W * n, H * n), interpolation=cv2.INTER_NEAREST)
    v, u = np.mgrid[0:H * n, 0:W * n]
    ok = np.isfinite(z) & (z > 0.005) & (z < 5.0)
    pix = np.stack([(u[ok] + 0.5) / n, (v[ok] + 0.5) / n, np.ones(ok.sum())])
    x_t = (np.linalg.inv(K) @ pix) * z[ok]
    x_s = rs.T @ (rt @ x_t + (pt - ps)[:, None])
    front = x_s[2] > 0.005
    uv = K @ (x_s[:, front] / x_s[2, front])
    us, vs = np.floor(uv[0]).astype(int), np.floor(uv[1]).astype(int)
    inside = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
    depth, cols = x_s[2, front][inside], rgb[ok][front][inside]
    order = np.argsort(-depth)                    # far first, so the nearest surface is written last
    out = np.zeros((H, W, 3), np.uint8)
    out[vs[inside][order], us[inside][order]] = cols[order]
    return out


def latent_adapter(hidden=256):
    """Residual MLP on the 256-D encoder feature, initialised to the identity."""
    import torch
    from torch import nn

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(hidden, 512), nn.ReLU(), nn.Linear(512, hidden))
            nn.init.zeros_(self.net[2].weight)
            nn.init.zeros_(self.net[2].bias)

        def forward(self, f):
            return torch.relu(f + self.net(f))   # features are post-ReLU

    return Adapter()
