"""Camera calibration + rectification map, mirroring ETTCM/rectify.hpp.

calib.txt (one whitespace-separated list):
    fx fy cx cy k1 k2 p1 p2 k3 width height

The official code:
  1. builds K and the distortion vector D,
  2. K_new = cv::getOptimalNewCameraMatrix(K, D, size, alpha=0, size)
  3. map  = cv::initInverseRectificationMap(K, D, I, K_new, size)
     -> map[y, x] = undistorted (rectified) position of raw pixel (x, y).
All the maths afterwards uses K_new (fx, fy, cx, cy of the *rectified* camera).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    # map[y, x] = (u, v): rectified coordinates of raw pixel (x, y); float32 (H, W, 2)
    map: np.ndarray


def read_calibration(path):
    # Read all numbers regardless of line layout (VL's calib.txt has 9 values on
    # the first line and "width height" on the second), like the C++ `fin >>`.
    with open(path) as f:
        vals = np.array(f.read().split(), dtype=np.float64)
    fx, fy, cx, cy = vals[:4]
    dist = vals[4:9]
    width, height = int(vals[9]), int(vals[10])
    return fx, fy, cx, cy, dist, width, height


def make_camera(path=None, alpha=0.0, *, fx=None, fy=None, cx=None, cy=None,
                dist=None, width=None, height=None):
    """Load calib.txt (or take explicit intrinsics) and build the rectification map."""
    if path is not None:
        fx, fy, cx, cy, dist, width, height = read_calibration(path)
    dist = np.zeros(5) if dist is None else np.asarray(dist, float)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)

    xs, ys = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    if not np.any(dist):
        # no distortion: identity map, rectified camera = original camera
        K_new = K
        m = np.stack([xs, ys], axis=-1).astype(np.float32)
    else:
        import cv2  # pip install opencv-python

        K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (width, height), alpha,
                                                 (width, height))
        pts = np.stack([xs.ravel(), ys.ravel()], axis=-1).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, K, dist, P=K_new)  # same as the inverse map
        m = und.reshape(height, width, 2).astype(np.float32)
    return Camera(width, height, float(K_new[0, 0]), float(K_new[1, 1]),
                  float(K_new[0, 2]), float(K_new[1, 2]), m)


def rectify_events(events, cam):
    """Attach rectified coordinates and drop events that fall outside the image.

    Mirrors global_divergence.hpp: an event is kept only if round(map(x, y)) is
    inside the image. Returns (events_kept, xr, yr) with xr, yr float32.
    """
    uv = cam.map[events["y"], events["x"]]
    xr, yr = uv[:, 0], uv[:, 1]
    xi, yi = np.round(xr), np.round(yr)
    keep = (xi >= 0) & (xi < cam.width) & (yi >= 0) & (yi < cam.height)
    return events[keep], xr[keep].astype(np.float32), yr[keep].astype(np.float32)
