"""MVSEC support: calibration, event streaming, ground truth and metrics (Tables 3-5).

Files (one set per sequence, e.g. indoor_flying1), as downloaded from the MVSEC site:
    <seq>_data.hdf5          events  davis/left/events (N, 4) = x, y, t [s], p (+-1)
                             frames  davis/left/image_raw_ts (M,) [s]
    <seq>_gt.hdf5            depth   davis/left/depth_image_rect (K, 260, 346) [m]
                                     davis/left/depth_image_rect_ts (K,) [s]
    <seq>_gt_flow_dist.npz   flow    timestamps (K,), x_flow_dist, y_flow_dist (K, 260, 346)
    <seq>_odom.npz           camera  timestamps (K,), lin_vel, ang_vel (K, 3): the velocity
                                     MVSEC used to generate the GT flow (Table 3 GT)
    <scene>_calib.zip        calib   camchain-imucam-<scene>.yaml, <scene>_left_x_map.txt,
                                     <scene>_left_y_map.txt   (scene = indoor_flying / outdoor_day)

Coordinates. MVSEC's GT flow (mvsec/tools/gt_flow/compute_flow.py) is computed on the
*rectified* depth image with the rectified projection matrix P, despite the "_dist" in
its name. So the estimator runs in that same rectified camera: fx, fy, cx, cy = P, and
each raw pixel is moved to its rectified position with the calib x/y maps. Estimated
flow (px/s, rectified pixels) is then directly comparable with the GT flow.

Evaluation protocol (EV-FlowNet [51], as used by the papers in Tables 4-5):
  * time windows: the whole indoor_flying sequences and outdoor_day1 from 222.4 s to
    240.4 s (EV-FlowNet, Sec. 5). Total ~266 s, matching the paper's "the evaluated
    sequences span approximately 265sec".
  * intervals between grayscale frames [t_k, t_k+dt]; GT displacement over the interval
    by propagating the GT flow (estimate_corresponding_gt_flow, ported from EV-FlowNet)
  * only pixels with events and with non-zero, finite GT flow; outdoor: rows < 190 only
    (the car's hood is not in the GT)
  * ETTCM evaluates "on the respective pixel locations" of its per-event estimates:
    each event's flow m_i = dt_frame * v_i is compared with the GT at its pixel.
"""
import io
import os
import zipfile

import numpy as np

from .calib import Camera

HEIGHT, WIDTH = 260, 346

# outdoor_day1 window in seconds from the start of the recording (EV-FlowNet);
# indoor sequences are used whole.
OUTDOOR_DAY1_WINDOW = (222.4, 240.4)
OUTDOOR_MAX_ROW = 190   # EV-FlowNet: only the top 190 rows for the car sequences

# Paper numbers (Nunes et al., ICCV 2023).
PAPER_TABLE4 = {  # (AEE px/frame, Out %) per sequence
    "driving":     {"indoor_flying1": (1.40, 7.50), "indoor_flying2": (2.42, 26.32),
                    "indoor_flying3": (2.14, 26.05), "outdoor_day1": (0.39, 0.12)},
    "translation": {"indoor_flying1": (0.63, 0.41), "indoor_flying2": (0.94, 2.83),
                    "indoor_flying3": (0.79, 1.65), "outdoor_day1": (0.55, 0.97)},
    "six_dof":     {"indoor_flying1": (0.88, 1.97), "indoor_flying2": (1.60, 12.02),
                    "indoor_flying3": (1.38, 9.99), "outdoor_day1": (0.81, 2.26)},
}
PAPER_TABLE5 = {  # (AREE %, AAE deg)
    "driving":     {"indoor_flying3": (104.05, 60.36), "outdoor_day1": (50.25, 13.31)},
    "translation": {"indoor_flying3": (44.46, 11.38), "outdoor_day1": (72.67, 21.77)},
    "six_dof":     {"indoor_flying3": (95.23, 21.35), "outdoor_day1": (122.04, 31.68)},
}
PAPER_TABLE3 = {  # divergence AREE %
    "driving":     {"indoor_flying1": 153.30, "indoor_flying2": 84.80, "outdoor_day1": 47.27},
    "translation": {"indoor_flying1": 19.82, "indoor_flying2": 28.23, "outdoor_day1": 45.78},
    "six_dof":     {"indoor_flying1": 53.65, "indoor_flying2": 56.12, "outdoor_day1": 56.66},
}


def scene_of(seq):
    return "outdoor_day" if seq.startswith("outdoor_day") else "indoor_flying"


# ----------------------------------------------------------------------------- files
def find_file(root, name):
    """Find `name` anywhere under root (the user's folders may be nested)."""
    for d, _, files in os.walk(root):
        if name in files:
            return os.path.join(d, name)
    raise FileNotFoundError(f"{name} not found under {root}")


def _h5py():
    import h5py  # noqa: PLC0415  (imported lazily so the VL code does not need it)
    return h5py


# ----------------------------------------------------------------------- calibration
def load_calib(zip_path, side="left"):
    """Rectified camera from an MVSEC <scene>_calib.zip.

    fx, fy, cx, cy come from the rectified projection matrix P (cam0 = left); the
    map gives, for every raw pixel, its rectified position (MVSEC x/y map files).
    """
    import yaml  # noqa: PLC0415

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        yml = [n for n in names if os.path.basename(n).startswith("camchain") and n.endswith(".yaml")]
        if not yml:
            raise ValueError(f"no camchain yaml in {zip_path}: {names}")
        cal = yaml.safe_load(z.read(yml[0]))
        cam = cal["cam0" if side == "left" else "cam1"]
        P = np.array(cam["projection_matrix"], dtype=np.float64)
        width, height = cam["resolution"]

        def load_map(axis):
            hit = [n for n in names if n.endswith(f"{side}_{axis}_map.txt")]
            if not hit:
                raise ValueError(f"no {side}_{axis}_map.txt in {zip_path}: {names}")
            return np.loadtxt(io.BytesIO(z.read(hit[0])))

        mx, my = load_map("x"), load_map("y")
    assert mx.shape == (height, width), (mx.shape, height, width)
    m = np.stack([mx, my], axis=-1).astype(np.float32)
    return Camera(int(width), int(height), float(P[0, 0]), float(P[1, 1]),
                  float(P[0, 2]), float(P[1, 2]), m), cam


def rectify(x, y, cam):
    """Rectified positions of raw pixels; keep those that land inside the image."""
    uv = cam.map[y, x]
    xr, yr = uv[:, 0], uv[:, 1]
    xi, yi = np.round(xr), np.round(yr)
    keep = (xi >= 0) & (xi < cam.width) & (yi >= 0) & (yi < cam.height)
    return keep, xr, yr


# ------------------------------------------------------------------------- events
class EventFile:
    """Streams davis/left/events from <seq>_data.hdf5 without loading it all."""

    def __init__(self, path):
        self.f = _h5py().File(path, "r")
        self.ev = self.f["davis"]["left"]["events"]
        self.n = int(self.ev.shape[0])
        self.gray_ts = np.asarray(self.f["davis"]["left"]["image_raw_ts"][:], dtype=np.float64)

    def time(self, i):
        return float(self.ev[i, 2])

    def index_of_time(self, t):
        """First event index with timestamp >= t (binary search on the file)."""
        lo, hi = 0, self.n
        while lo < hi:
            mid = (lo + hi) // 2
            if self.time(mid) < t:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def read(self, i, j):
        a = np.asarray(self.ev[i:j], dtype=np.float64)
        return (a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2],
                a[:, 3] > 0)

    def close(self):
        self.f.close()


# --------------------------------------------------------------------- GT flow npz
def npz_array(path, key):
    """Memory-map one array of an .npz (np.savez, uncompressed); else load it.

    The GT-flow files are 2-7 GB; memory-mapping avoids loading them into RAM.
    """
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(key + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            return np.load(path)[key]
        with open(path, "rb") as f:
            f.seek(info.header_offset)
            hdr = f.read(30)
            n_name = int.from_bytes(hdr[26:28], "little")
            n_extra = int.from_bytes(hdr[28:30], "little")
            f.seek(info.header_offset + 30 + n_name + n_extra)
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            else:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape,
                     order="F" if fortran else "C")


class GTFlow:
    """GT flow of one sequence. Frame k = displacement between timestamps k and k+1
    (EV-FlowNet convention), in rectified pixels. Arrays are memory-mapped."""

    def __init__(self, path, seq=None):
        self.seq = seq
        self.ts_all = np.asarray(npz_array(path, "timestamps"), dtype=np.float64)
        self.u_all = npz_array(path, "x_flow_dist")
        self.v_all = npz_array(path, "y_flow_dist")

    def window(self, t_rec0=None):
        """(t_start, t_end) of the evaluated part, absolute seconds.

        t_rec0: start of the recording (first event), needed for outdoor_day1."""
        ts = self.ts_all
        if self.seq == "outdoor_day1":
            if t_rec0 is None:
                raise ValueError("outdoor_day1 needs the recording start time")
            return t_rec0 + OUTDOOR_DAY1_WINDOW[0], t_rec0 + OUTDOOR_DAY1_WINDOW[1]
        return ts[0], ts[-1]

    def displacement(self, t0, t1):
        """GT pixel displacement (H, W) x 2 over [t0, t1] (EV-FlowNet)."""
        return estimate_corresponding_gt_flow(self.u_all, self.v_all, self.ts_all, t0, t1)


def _prop_flow(x_flow, y_flow, xi, yi, xm, ym, scale=1.0):
    import cv2  # noqa: PLC0415

    fx = cv2.remap(np.ascontiguousarray(x_flow, dtype=np.float32), xi, yi, cv2.INTER_NEAREST)
    fy = cv2.remap(np.ascontiguousarray(y_flow, dtype=np.float32), xi, yi, cv2.INTER_NEAREST)
    xm[fx == 0] = False
    ym[fy == 0] = False
    xi += fx * scale
    yi += fy * scale


def estimate_corresponding_gt_flow(x_flow_in, y_flow_in, gt_ts, t0, t1):
    """Ported from EV-FlowNet (src/eval_utils.py), unchanged in behaviour."""
    it = np.searchsorted(gt_ts, t0, side="right") - 1
    gt_dt = gt_ts[it + 1] - gt_ts[it]
    x_flow = np.asarray(x_flow_in[it], dtype=np.float64)
    y_flow = np.asarray(y_flow_in[it], dtype=np.float64)
    dt = t1 - t0
    if gt_dt > dt:
        return x_flow * dt / gt_dt, y_flow * dt / gt_dt

    xi, yi = np.meshgrid(np.arange(x_flow.shape[1]), np.arange(x_flow.shape[0]))
    xi = xi.astype(np.float32)
    yi = yi.astype(np.float32)
    x0, y0 = xi.copy(), yi.copy()
    xm = np.ones(xi.shape, dtype=bool)
    ym = np.ones(yi.shape, dtype=bool)

    _prop_flow(x_flow, y_flow, xi, yi, xm, ym, (gt_ts[it + 1] - t0) / gt_dt)
    it += 1
    while gt_ts[it + 1] < t1:
        _prop_flow(x_flow_in[it], y_flow_in[it], xi, yi, xm, ym)
        it += 1
    final_dt = t1 - gt_ts[it]
    _prop_flow(x_flow_in[it], y_flow_in[it], xi, yi, xm, ym,
               final_dt / (gt_ts[it + 1] - gt_ts[it]))
    xs = (xi - x0).astype(np.float64)
    ys = (yi - y0).astype(np.float64)
    xs[~xm] = 0
    ys[~ym] = 0
    return xs, ys


# ------------------------------------------------------------------ flow metrics
class FlowMetrics:
    """Accumulates AEE / Out / AREE / AAE (Eqs. 13-15) over (pred, gt) pairs.

    Two averages are kept: over all samples (the paper's 1/N sum), and the mean of
    per-frame averages (EV-FlowNet style).
    """

    def __init__(self):
        self.s = np.zeros(5)          # n, sum EE, n(EE > 3), sum rel EE, sum AE
        self.frames = []              # per-interval (n, AEE, Out, AREE, AAE)

    @staticmethod
    def _terms(pred, gt):
        d = pred - gt
        ee = np.hypot(d[:, 0], d[:, 1])
        ng = np.hypot(gt[:, 0], gt[:, 1])
        npd = np.hypot(pred[:, 0], pred[:, 1])
        cos = (pred[:, 0] * gt[:, 0] + pred[:, 1] * gt[:, 1]) / np.maximum(npd * ng, 1e-12)
        ae = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        return ee, ee / ng, ae

    def add(self, pred, gt):
        if len(gt) == 0:
            return
        ee, rel, ae = self._terms(pred, gt)
        n = len(ee)
        self.s += [n, ee.sum(), (ee > 3.0).sum(), rel.sum(), ae.sum()]
        self.frames.append((n, ee.mean(), 100 * (ee > 3.0).mean(), 100 * rel.mean(), ae.mean()))

    def result(self):
        n = max(self.s[0], 1)
        f = np.array(self.frames) if self.frames else np.zeros((1, 5))
        return {
            "n": int(self.s[0]), "frames": len(self.frames),
            "AEE": self.s[1] / n, "Out": 100 * self.s[2] / n,
            "AREE": 100 * self.s[3] / n, "AAE": self.s[4] / n,
            "AEE_frame": f[:, 1].mean(), "Out_frame": f[:, 2].mean(),
            "AREE_frame": f[:, 3].mean(), "AAE_frame": f[:, 4].mean(),
        }


def gt_mask(gu, gv, rows):
    ok = np.isfinite(gu) & np.isfinite(gv) & (np.hypot(gu, gv) > 0)
    if rows is not None:
        ok[rows:, :] = False
    return ok


# ------------------------------------------------------- divergence GT (Table 3)
class DivergenceGT:
    """Per-pixel GT divergence mu = V_z / Z (paper Sec. 4.1: "GT TTCMs by applying
    Eq. (1) given the GT depth maps and camera velocity").

    Z comes from the GT depth map (depth_image_rect). The camera velocity V_z can come from
    three sources (VEL_SOURCES):
      "odom_npz"  MVSEC's published camera velocity, <seq>_odom.npz "lin_vel" (the velocity
                  MVSEC used to generate its GT flow). This is the Table 3 ground truth.
      "poses"     the same velocity rebuilt from the odometry poses in <seq>_gt.hdf5, exactly
                  as MVSEC's compute_flow.py does it (pose k paired with depth frame k,
                  V = translation of inv(H_{k-1}) H_k / dt, then a +-10-sample mean computed
                  in place). Equal to "odom_npz" (ratio 1.000); used when odom.npz is missing.
      "flow"      recovered from the GT flow of the same frame by least squares:
                    flow_x / (dt_k Pfx) = (x V_z - V_x) / Z + (x y, -(1 + x^2), y) . Omega
                    flow_y / (dt_k Pfy) = (y V_z - V_y) / Z + (1 + y^2, -x y, -x) . Omega
                  Only a data check: MVSEC's flow files imply 55-65 % of the odometry's
                  forward speed (an inconsistency inside MVSEC).
    """

    VEL_SOURCES = ("odom_npz", "poses", "flow")

    def __init__(self, gt_h5_path, flow, cam, odom_npz_path=None):
        h5 = _h5py().File(gt_h5_path, "r")
        self.h5 = h5
        g = h5["davis"]["left"]
        self.depth = g["depth_image_rect"]
        self.dts = np.asarray(g["depth_image_rect_ts"][:], dtype=np.float64)
        self.flow = flow
        self.cam = cam
        H, W = self.depth.shape[1], self.depth.shape[2]
        u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        self.xn = (u - cam.cx) / cam.fx
        self.yn = (v - cam.cy) / cam.fy
        self._cache = {}
        self._npz = None
        self._poses = None
        if odom_npz_path is not None:
            d = np.load(odom_npz_path)
            self._npz = (np.asarray(d["timestamps"], dtype=np.float64),
                         np.asarray(d["lin_vel"], dtype=np.float64))
        if "odometry" in g:
            ots = np.asarray(g["odometry_ts"][:], dtype=np.float64)
            Hs = np.asarray(g["odometry"][:], dtype=np.float64)
            n = min(len(self.dts), len(Hs))
            V = np.zeros((n, 3))
            for f in range(1, n):
                H01 = np.linalg.inv(Hs[f - 1]) @ Hs[f]
                V[f] = H01[:3, 3] / (ots[f] - ots[f - 1])
            for f in range(n):                       # in place, like MVSEC's smoothed_Vs = Vs
                V[f] = V[max(f - 10, 0):min(f + 11, n)].mean(axis=0)
            self._poses = V
        self.default_source = "odom_npz" if self._npz is not None else "poses"

    def frame_of(self, t):
        return np.clip(np.searchsorted(self.dts, t, side="right") - 1, 0, len(self.dts) - 1)

    def velocity(self, k):
        """(V, Omega) of depth frame k, from the GT flow frame with the same timestamp."""
        ts = self.flow.ts_all
        j = int(np.argmin(np.abs(ts - self.dts[k])))
        if j == 0 or abs(ts[j] - self.dts[k]) > 1e-3:
            return None
        dt = ts[j] - ts[j - 1]
        Z = np.asarray(self.depth[k], dtype=np.float64)
        fu = np.asarray(self.flow.u_all[j], dtype=np.float64)
        fv = np.asarray(self.flow.v_all[j], dtype=np.float64)
        ok = np.isfinite(Z) & (Z > 0) & ((fu != 0) | (fv != 0))
        ok[::2, :] = False                         # every other row is plenty
        if ok.sum() < 50 or dt <= 0:
            return None
        x, y, iz = self.xn[ok], self.yn[ok], 1.0 / Z[ok]
        z0 = np.zeros_like(x)
        Ax = np.stack([-iz, z0, x * iz, x * y, -(1 + x * x), y], axis=1)
        Ay = np.stack([z0, -iz, y * iz, 1 + y * y, -x * y, -x], axis=1)
        A = np.concatenate([Ax, Ay])
        b = np.concatenate([fu[ok] / (dt * self.cam.fx), fv[ok] / (dt * self.cam.fy)])
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        return sol[:3], sol[3:]

    def vz(self, k, source):
        """Forward camera speed V_z [m/s] at depth frame k (None if unavailable)."""
        if source == "odom_npz":
            if self._npz is None:
                return None
            ts, lv = self._npz
            j = int(np.argmin(np.abs(ts - self.dts[k])))
            return lv[j][2] if abs(ts[j] - self.dts[k]) <= 0.025 else None
        if source == "poses":
            return None if self._poses is None or k >= len(self._poses) else self._poses[k][2]
        vel = self.velocity(k)
        return None if vel is None else vel[0][2]

    def mu_map(self, k, source=None):
        """(H, W) GT divergence of depth frame k (nan where depth is invalid)."""
        source = source or self.default_source
        key = (k, source)
        if key in self._cache:
            return self._cache[key]
        vz = self.vz(k, source)
        Z = np.asarray(self.depth[k], dtype=np.float64)
        if vz is None:
            mu = np.full(Z.shape, np.nan)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                mu = np.where(np.isfinite(Z) & (Z > 0), vz / Z, np.nan)
        if len(self._cache) > 32:
            self._cache.clear()
        self._cache[key] = mu
        return mu

    def close(self):
        self.h5.close()
