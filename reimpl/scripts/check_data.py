"""Check that the datasets are in place and readable, before running the tables.

    python scripts/check_data.py                       # ../datasets/VL and ../datasets/MVSEC
    python scripts/check_data.py --vl ../datasets/VL --mvsec ''     # VL only

Quick: it only reads headers, timestamps and a few values (VL events are read in full,
~100 MB). Prints what it finds and, at the end, a list of anything missing or unreadable.
"""
import argparse
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ettcm import mvsec as M  # noqa: E402

SEQS = ["indoor_flying1", "indoor_flying2", "indoor_flying3", "outdoor_day1"]
VL_SEQS = ["2D-1", "2D-2", "2D-3", "2D-4", "2D-5", "2D-6", "2D-7", "3D"]
PROBLEMS = []


def problem(what, e):
    PROBLEMS.append((what, e))
    print(f"  PROBLEM  {what}: {e!r}")


def h5_tree(path):
    import h5py

    with h5py.File(path, "r") as f:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"    {name:45s} {str(obj.shape):22s} {obj.dtype}")
        f.visititems(show)


def rate(ts):
    d = np.diff(ts)
    return f"{len(ts)} stamps, {ts[0]:.3f} .. {ts[-1]:.3f} s ({ts[-1] - ts[0]:.1f} s), ~{1 / np.median(d):.1f} Hz"


def check_mvsec(root):
    print("\n######## MVSEC (Tables 3-5):", root)
    print("== files under", root)
    for d, _, files in os.walk(root):
        for f in sorted(files):
            p = os.path.join(d, f)
            print(f"  {os.path.relpath(p, root):55s} {os.path.getsize(p) / 1e9:7.2f} GB")

    for scene in ["indoor_flying", "outdoor_day"]:
        print(f"\n== calibration {scene}")
        try:
            path = M.find_file(root, f"{scene}_calib.zip")
        except FileNotFoundError as e:
            problem("MVSEC calibration", e)
            continue
        with zipfile.ZipFile(path) as z:
            print("  zip contents:", z.namelist())
        cam, raw = M.load_calib(path)
        print(f"  resolution {cam.width}x{cam.height}; rectified fx={cam.fx:.3f} fy={cam.fy:.3f} "
              f"cx={cam.cx:.3f} cy={cam.cy:.3f}")
        print(f"  distortion {raw.get('distortion_model')} {raw.get('distortion_coeffs')}")
        print(f"  map raw(0,0) -> {cam.map[0, 0]}, raw(173,130) -> {cam.map[130, 173]}")

    for seq in SEQS:
        print(f"\n== {seq}")
        try:
            data = M.find_file(root, f"{seq}_data.hdf5")
            print("  data:", data)
            h5_tree(data)
            ef = M.EventFile(data)
            x, y, t, p = ef.read(0, 5)
            print(f"  first events: x={x} y={y} p={p.astype(int)}")
            print(f"  events: {ef.n}, t {ef.time(0):.3f} .. {ef.time(ef.n - 1):.3f} s")
            print("  gray frames:", rate(ef.gray_ts))
            ef.close()
        except Exception as e:  # noqa: BLE001
            problem(f"MVSEC {seq} data", e)
        try:
            fl = M.find_file(root, f"{seq}_gt_flow_dist.npz")
            with zipfile.ZipFile(fl) as z:
                for i in z.infolist():
                    kind = "stored" if i.compress_type == zipfile.ZIP_STORED else "COMPRESSED"
                    print(f"  npz member {i.filename:22s} {i.file_size / 1e9:6.2f} GB  {kind}")
            g = M.GTFlow(fl, seq)
            print(f"  gt flow: {g.u_all.shape} {g.u_all.dtype}; all {rate(g.ts_all)}")
            t_rec0 = M.EventFile(M.find_file(root, f"{seq}_data.hdf5")).time(0)
            a, b = g.window(t_rec0)
            print(f"  evaluated window: {a - t_rec0:7.1f} .. {b - t_rec0:7.1f} s after start "
                  f"= {b - a:.1f} s")
            k = len(g.ts_all) // 2
            u = np.asarray(g.u_all[k])
            print(f"  frame {k}: non-zero flow at {(u != 0).mean() * 100:.1f} % of pixels, "
                  f"|u| median {np.median(np.abs(u[u != 0])) if (u != 0).any() else 0:.3f} px")
        except Exception as e:  # noqa: BLE001
            problem(f"MVSEC {seq} gt flow", e)
        try:
            gt = M.find_file(root, f"{seq}_gt.hdf5")
            print("  gt:", gt)
            h5_tree(gt)
        except Exception as e:  # noqa: BLE001
            problem(f"MVSEC {seq} gt (Table 3 only)", e)
        try:
            od = np.load(M.find_file(root, f"{seq}_odom.npz"))
            print(f"  odom.npz: {sorted(od.files)}, lin_vel {od['lin_vel'].shape}; "
                  f"{rate(np.asarray(od['timestamps'], dtype=np.float64))}")
        except Exception as e:  # noqa: BLE001
            problem(f"MVSEC {seq} odom.npz (Table 3 only)", e)


def check_vl(root):
    print("\n######## VL (Table 2):", root)
    from ettcm.calib import read_calibration
    from ettcm.evaluate import read_vl_sequence

    try:
        fx, fy, cx, cy, dist, w, h = read_calibration(os.path.join(root, "calib.txt"))
        print(f"  calib.txt: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} dist={[float(v) for v in dist]} {w}x{h}")
    except Exception as e:  # noqa: BLE001
        problem("VL calib.txt", e)
    for seq in VL_SEQS:
        d = os.path.join(root, seq)
        try:
            ev, w, h, ts, gt = read_vl_sequence(d)
            t = ev["t"]
            print(f"  {seq:5s} {len(ev):9d} events, {w}x{h}, {1e-6 * (int(t[-1]) - int(t[0])):5.1f} s, "
                  f"{len(gt)} GT divergence values ({gt.min():.3f} .. {gt.max():.3f} 1/s)")
        except Exception as e:  # noqa: BLE001
            problem(f"VL {seq}", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vl", default="../datasets/VL", help="VL folder ('' to skip)")
    ap.add_argument("--mvsec", default="../datasets/MVSEC", help="MVSEC folder ('' to skip)")
    args = ap.parse_args()
    for name, root, fn in (("VL", args.vl, check_vl), ("MVSEC", args.mvsec, check_mvsec)):
        if not root:
            continue
        if not os.path.isdir(root):
            problem(name, f"folder {root} not found")
            continue
        fn(root)
    print("\n######## summary")
    if PROBLEMS:
        for what, e in PROBLEMS:
            print(f"  PROBLEM  {what}: {e}")
    else:
        print("  all files found and readable")


if __name__ == "__main__":
    main()
