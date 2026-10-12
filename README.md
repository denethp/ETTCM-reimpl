# ETTCM-reimpl

An unofficial Python + Numba re-implementation of **"Time-to-Contact Map by Joint Estimation of
Up-to-Scale Inverse Depth and Global Motion using a Single Event Camera"**
(U. M. Nunes, L. U. Perrinet, S.-H. Ieng, ICCV 2023), with scripts that reproduce every table of
the paper on the same datasets.

- Paper: [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Nunes_Time-to-Contact_Map_by_Joint_Estimation_of_Up-to-Scale_Inverse_Depth_and_ICCV_2023_paper.html)
- Official C++ code: [neuromorphic-paris/ETTCM](https://github.com/neuromorphic-paris/ETTCM)
  (CC BY-NC-SA 4.0, not included here). It only covers Table 2; everything for MVSEC
  (Tables 3–5, figures) is written from scratch in this repo.

## Contents

1. [The method in short](#1-the-method-in-short)
2. [Where we stand](#2-where-we-stand)
3. [Installation](#3-installation)
4. [Datasets: download and folder structure](#4-datasets-download-and-folder-structure)
5. [Reproducing the tables](#5-reproducing-the-tables)
6. [Results](#6-results)
7. [Table 3: the open problem](#7-table-3-the-open-problem)
8. [Problems we hit and how we fixed them](#8-problems-we-hit-and-how-we-fixed-them)
9. [What is left to do](#9-what-is-left-to-do)
10. [Code layout](#10-code-layout)
11. [Details needed to match the official code](#11-details-needed-to-match-the-official-code)

---

## 1. The method in short

An event camera reports, per pixel and asynchronously, "brightness went up/down at time t".
The method processes these events **one at a time** and keeps two things up to date:

- the camera's **global motion** θ (how the whole camera moves: linear velocity ν, and for some
  models angular velocity ω), and
- an **inverse depth** ρ = 1/Z for every event (how far the thing that caused it is, up to scale).

From them it gets, for every event, the **optical flow** v = Bθ (Eq. 9), the **divergence**
μ = ρ·ν_z and the **time-to-contact** TTC = 1/μ (Eq. 1). Keeping the latest TTC per pixel gives
the **TTC map** (TTCM). Each event takes about 1–2 µs, i.e. real time.

Per event: look at the (2s+1)² neighbourhood (shifted back along the predicted flow), initialise
ρ from the neighbours (Eq. 11), run at most 2 Gauss–Newton steps on (θ, log ρ) with priors
(Eq. 8, Sec. 3.6), keep |ν| ≤ 1, then output flow / divergence / TTC.

Motion models: **Scaling** (ν_z), **Translation** (ν), **Driving** (ν_z, ω_y, with |ν_z| = 1),
**6-DOF** (ν, ω).

## 2. Where we stand

| Paper item | Dataset | Status |
|---|---|---|
| Table 2: global divergence REE + runtime | VL | **Reproduced.** Every average within 0.1 of the paper (Scaling/Translation within 0.01); ~1–2 µs/event |
| Table 4: optical flow AEE / Out | MVSEC | **Reproduced.** Driving and 6-DOF within 0.05 px indoors; Translation 0.03–0.25 px worse; outdoor Driving 0.59 vs 0.39 |
| Table 5: optical flow AREE / AAE | MVSEC | **Mostly reproduced.** AAE (direction) within ~2° everywhere; AREE close for Translation and 6-DOF; Driving's AREE much higher |
| Table 3: divergence (TTC) AREE | MVSEC | **Partly reproduced.** Translation matches when near-zero ground truth is handled; outdoor is in the right range for all models; 6-DOF and Driving indoors do not match. See [Section 7](#7-table-3-the-open-problem) |
| Fig. 5/6: flow and TTC maps | MVSEC | Script ready (`figures.py`), not yet compared with the paper's figures |
| Per-event agreement with the official C++ | VL | **Identical up to rounding** (~1e-7 relative, against a double-precision build), all models |

In one sentence: **the estimator is a faithful copy of the authors' method (it matches their code
event by event), Tables 2, 4 and 5 come out close to the paper, and Table 3 can only be partly
reproduced because the paper does not say exactly how it was scored.**

## 3. Installation

Python 3.10 or newer. Numba is what makes it fast (without it everything still runs, ~300×
slower).

```bash
git clone https://github.com/denethp/ETTCM-reimpl.git
cd ETTCM-reimpl/reimpl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # numpy, numba, opencv-python, h5py, pyyaml, matplotlib, pytest

# quick self-checks (a few seconds each)
python tests/test_io.py                # PASS test_roundtrip
python tests/test_models.py            # PASS x4: motion models vs brute-force 3D projection
python tests/test_chunks.py            # PASS: processing in chunks == one run
```

All commands below are run from inside `reimpl/` with the virtual environment active.
To keep a copy of what a script prints, append `| tee ../some_file.txt`.

## 4. Datasets: download and folder structure

The datasets are not in the repo (`datasets/` is git-ignored). Put them next to `reimpl/`:

```
ETTCM-reimpl/
├── reimpl/                                  code
└── datasets/
    ├── VL/                                  Table 2   (~110 MB)
    │   ├── calib.txt
    │   ├── 2D-1/  events.es  divergence.txt  timestamps.txt
    │   ├── 2D-2/  ...
    │   ├── ...    (2D-3 … 2D-7)
    │   └── 3D/    events.es  divergence.txt  timestamps.txt
    └── MVSEC/                               Tables 3–5   (~52 GB)
        ├── indoor_flying_calib.zip
        ├── outdoor_day_calib.zip
        ├── data/
        │   ├── indoor/
        │   │   ├── indoor_flying1_data.hdf5     1.7 GB
        │   │   ├── indoor_flying1_gt.hdf5       3.4 GB
        │   │   ├── indoor_flying2_data.hdf5     2.5 GB
        │   │   ├── indoor_flying2_gt.hdf5       4.3 GB
        │   │   ├── indoor_flying3_data.hdf5     2.6 GB
        │   │   └── indoor_flying3_gt.hdf5       4.5 GB
        │   └── outdoor/
        │       ├── outdoor_day1_data.hdf5      12.1 GB
        │       └── outdoor_day1_gt.hdf5         6.3 GB
        └── GT/
            ├── indoor/
            │   ├── indoor_flying1_gt_flow_dist.npz   2.0 GB
            │   ├── indoor_flying1_odom.npz
            │   ├── indoor_flying2_gt_flow_dist.npz   2.4 GB
            │   ├── indoor_flying2_odom.npz
            │   ├── indoor_flying3_gt_flow_dist.npz   2.7 GB
            │   └── indoor_flying3_odom.npz
            └── outdoor/
                ├── outdoor_day1_gt_flow_dist.npz     7.4 GB
                └── outdoor_day1_odom.npz
```

The MVSEC scripts search `--data` recursively by file name, so the sub-folders under `MVSEC/` can
be arranged differently; only the file names matter.

### Ventral Landing (VL), for Table 2

- Download `VL.zip` (the copy the official code uses): https://figshare.com/ndownloader/files/44987794
- `unzip VL.zip -d datasets/` should give `datasets/VL/calib.txt` and the 8 sequence folders.
- Original dataset (McLeod et al.): https://github.com/s-mcleod/ventral-landing-event-dataset

### MVSEC, for Tables 3, 4, 5

Download page: https://daniilidis-group.github.io/mvsec/download/. Only these four sequences are
used: `indoor_flying1`, `indoor_flying2`, `indoor_flying3`, `outdoor_day1`.

| File | What it holds | Needed for | Where |
|---|---|---|---|
| `<seq>_data.hdf5` | events, grayscale frame times | Tables 3, 4, 5 | MVSEC "HDF5" Google Drive folder (sub-folders `indoor_flying`, `outdoor_day`): https://drive.google.com/open?id=1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV |
| `<seq>_gt.hdf5` | depth maps, odometry poses | Table 3 | same HDF5 folder |
| `<seq>_gt_flow_dist.npz` | ground-truth optical flow (EV-FlowNet) | Tables 3, 4, 5 | MVSEC "ground truth optical flow" Google Drive folder: https://drive.google.com/drive/folders/1XS0AQTuCwUaWOmtjyJWRHkbXjj_igJLp |
| `<seq>_odom.npz` | camera velocity used to make the GT flow | Table 3 | same ground-truth flow folder, next to the `.npz` flow files |
| `indoor_flying_calib.zip` | calibration + rectification maps | all | https://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying_calib.zip |
| `outdoor_day_calib.zip` | calibration + rectification maps | all | https://visiondata.cis.upenn.edu/mvsec/outdoor_day/outdoor_day_calib.zip |

Keep the calibration zips zipped. Check that everything is found and readable:

```bash
python scripts/check_data.py          # both datasets; ends with a list of anything missing
```

## 5. Reproducing the tables

### Table 2 (VL): global divergence REE and runtime

```bash
# all 3 models x s in {2, 3}, 8 sequences each; prints ours next to the paper per sequence
python scripts/table2_vl.py --data ../datasets/VL --all --csv ../results_python_vl.csv

# a single row
python scripts/table2_vl.py --data ../datasets/VL --model translation -s 2
```

Settings (official `table_2_results.sh`): s ∈ {2, 3}, w_thresh = 0.01, l_θ = 1000, l_ρ = 1,
2 Gauss–Newton iterations. Models: `scaling`, `translation`, `six_dof`.

### Tables 4 and 5 (MVSEC): optical flow

```bash
# all 4 sequences x 3 models; prints AEE, Out, AREE, AAE next to the paper's
python scripts/tables45_mvsec.py --data ../datasets/MVSEC --seq all --model all --csv ../results_python_mvsec.csv

# one sequence / one model, or a quick 5-second try
python scripts/tables45_mvsec.py --data ../datasets/MVSEC --seq indoor_flying1 --model translation
python scripts/tables45_mvsec.py --data ../datasets/MVSEC --seq indoor_flying1 --model translation --max-seconds 5
```

Table 4 uses all four sequences, Table 5 only `indoor_flying3` and `outdoor_day1` (the script
prints all four metrics for every sequence). Settings (paper Table 1, Sec. 4.1): s = 3,
w_thresh = 0.01, l_θ = 100, l_ρ = 1. Models: `driving`, `translation`, `six_dof`.

### Table 3 (MVSEC): divergence

```bash
# the paper's columns (indoor_flying1, indoor_flying2, outdoor_day1) x 3 models,
# under several scoring protocols, plus data checks
python scripts/table3_mvsec.py --data ../datasets/MVSEC

# indoor_flying3 (see Section 7 for why), or a single model
python scripts/table3_mvsec.py --data ../datasets/MVSEC --seq indoor_flying3
python scripts/table3_mvsec.py --data ../datasets/MVSEC --seq indoor_flying1 --model translation
```

### Figures 5 and 6 (qualitative maps)

```bash
python scripts/figures.py --data ../datasets/MVSEC --seq indoor_flying1 --model translation \
    --t 20 --out ../figures/if1_t20.png
python scripts/figures.py --dataset vl --data ../datasets/VL --seq 2D-1 --model translation \
    -s 2 --l-theta 1000 --t 8 --out ../figures/vl_2d1.png
```

Panels: accumulated events, GT flow, estimated flow, GT TTC map, estimated TTC map
(TTC colours 1 s = red … 10 s = violet, as in the paper).

### How MVSEC is evaluated

The paper gives few details, so the standard MVSEC protocol (EV-FlowNet [51]) is used:

- **Camera.** MVSEC's GT flow is computed on the *rectified* image with the rectified projection
  matrix P (despite the `_dist` in its name). The estimator runs in that rectified camera (fx, fy,
  cx, cy from P, raw pixels moved with the calibration x/y maps), so estimate and GT are in the
  same pixels.
- **Time windows.** The whole `indoor_flying1–3` and `outdoor_day1` from 222.4 s to 240.4 s:
  69.9 + 84.5 + 93.7 + 18 = **266 s**, matching the paper's "approximately 265 sec".
  For `outdoor_day1` the estimator starts at the beginning of the recording (warm start) and only
  the window is scored (`--cold` starts at 222.4 s instead).
- **Flow (Tables 4, 5).** Intervals between consecutive grayscale frames; GT displacement over
  each interval with EV-FlowNet's `estimate_corresponding_gt_flow`; only pixels with non-zero GT;
  `outdoor_day1` rows < 190 only (the car's hood). Each event with an estimate gives
  m_i = Δt_frame · v_i, compared with the GT at its pixel ("we only evaluate on the respective
  pixel locations"); AEE, Out (> 3 px), AREE, AAE are averaged over these events (Eqs. 13–15).
- **Divergence (Table 3).** Per event μ̂_i = ρ_i ν_z vs μ_gt = V_z / Z at its pixel: Z from the
  rectified depth map of the latest depth frame, V_z from MVSEC's camera velocity
  (`<seq>_odom.npz`, `lin_vel`). This is the paper's "GT TTCMs by applying Eq. (1) given the GT
  depth maps and camera velocity".

## 6. Results

Detailed per-sequence numbers, including the official C++ run, are in [`RESULTS.md`](RESULTS.md).

### Table 2 (VL), average REE % over the 8 sequences

| Model | s | Paper | Official C++ (our run) | **Ours** | Ours µs/event |
|---|---|---:|---:|---:|---:|
| Scaling | 2 | 8.29 | 8.30 | **8.29** | 0.81 |
| Translation | 2 | 5.64 | 5.64 | **5.64** | 1.02 |
| 6-DOF | 2 | 10.11 | 10.10 | **10.13** | 1.49 |
| Scaling | 3 | 15.08 | 15.09 | **15.08** | 1.26 |
| Translation | 3 | 9.31 | 9.31 | **9.31** | 1.69 |
| 6-DOF | 3 | 6.42 | 6.39 | **6.34** | 2.05 |

### Table 4 (MVSEC), AEE (px/frame) / Out (%)

| Model | | indoor_flying1 | indoor_flying2 | indoor_flying3 | outdoor_day1 |
|---|---|---|---|---|---|
| Driving | Paper | 1.40 / 7.50 | 2.42 / 26.32 | 2.14 / 26.05 | 0.39 / 0.12 |
| | **Ours** | 1.35 / 6.93 | 2.47 / 27.36 | 2.16 / 26.78 | 0.59 / 1.25 |
| Translation | Paper | 0.63 / 0.41 | 0.94 / 2.83 | 0.79 / 1.65 | 0.55 / 0.97 |
| | **Ours** | 0.66 / 0.87 | 1.19 / 6.48 | 0.94 / 3.69 | 0.67 / 1.59 |
| 6-DOF | Paper | 0.88 / 1.97 | 1.60 / 12.02 | 1.38 / 9.99 | 0.81 / 2.26 |
| | **Ours** | 0.84 / 1.79 | 1.60 / 10.77 | 1.38 / 8.66 | 0.76 / 1.91 |

### Table 5 (MVSEC), AREE (%) / AAE (°)

| Model | | indoor_flying3 | outdoor_day1 |
|---|---|---|---|
| Driving | Paper | 104.05 / 60.36 | 50.25 / 13.31 |
| | **Ours** | 362.85 / 60.75 | 87.27 / 15.11 |
| Translation | Paper | 44.46 / 11.38 | 72.67 / 21.77 |
| | **Ours** | 50.90 / 12.12 | 92.73 / 21.95 |
| 6-DOF | Paper | 95.23 / 21.35 | 122.04 / 31.68 |
| | **Ours** | 84.35 / 21.41 | 120.29 / 22.52 |

### Table 3 (MVSEC), divergence AREE (%)

| Model | | indoor_flying1 | indoor_flying2 | outdoor_day1 |
|---|---|---:|---:|---:|
| Driving | Paper | 153.30 | 84.80 | 47.27 |
| | Ours, all events (plain mean) | 315.52 | 212.16 | 70.96 |
| | Ours, median | 120.33 | 103.35 | 36.2 |
| Translation | Paper | 19.82 | 28.23 | 45.78 |
| | Ours, all events (plain mean) | 66.77 | 81.88 | 54.31 |
| | Ours, median | **23.82** | **28.69** | **44.0** |
| | Ours, only \|μ_gt\| ≥ 0.2 /s | **18.24** | **29.44** | – |
| 6-DOF | Paper | 53.65 | 56.12 | 56.66 |
| | Ours, all events (plain mean) | 113.15 | 120.27 | 66.09 |
| | Ours, median | 83.24 | 95.73 | 53.7 |

Numbers come from the runs before the code clean-up (identical code paths; the outdoor run then
started 0.4 s after the recording start instead of at it, and medians were taken from a sample of
events, so re-runs can differ in the last digit).

## 7. Table 3: the open problem

### What does not match

- **Translation** (the paper's best model) matches on every indoor sequence, but only when
  events where the true divergence is almost zero are handled: by a median (23.8 / 28.7 vs
  19.8 / 28.2) or by ignoring events with |μ_gt| < 0.2 /s (18.2 / 29.4). The plain mean of
  Eq. 14 is 2–3× higher (67 / 82).
- **Outdoor** (`outdoor_day1`) is in the right range for all three models (71 / 54 / 66 vs
  47 / 46 / 57), with the correct sign of motion for 97–100 % of events.
- **6-DOF indoors** stays at ~75–96 % under every scoring we tried, while the paper reports 54–56 %.
- **Driving indoors** is 212–316 % with a plain mean and ~100–125 % with a median or threshold
  (paper 85–153 %). It is the right range only under those rules.

### Why the plain mean blows up

AREE divides by the true value: |μ̂ − μ_gt| / |μ_gt|. The drone in `indoor_flying` often barely
moves forward (|V_z| < 0.05 m/s in 13–20 % of frames; in indoor_flying3, < 0.2 m/s in 42 %), so
μ_gt ≈ 0 there, and
any small error becomes thousands of percent. A few such events dominate the mean. The paper does
not say how it treated them.

### What we checked and tried

| Hypothesis | Test | Result |
|---|---|---|
| Wrong GT velocity | Built V_z three ways: MVSEC's `odom.npz`, rebuilt from the poses exactly as MVSEC's `compute_flow.py`, and recovered from the GT flow | `odom.npz` = poses (ratio 1.000). **The GT flow implies only 55–65 % of that speed**: an inconsistency inside MVSEC. Using the flow-implied speed made our error 2× worse, so the odometry speed (what the paper describes) is used |
| Wrong calibration / camera | GT flow in `gt.hdf5` vs the `.npz` files; rectified vs raw pixels; Tables 4–5 | Identical files (max diff 0.0000 px); Table 4 matches to 0.05 px, so the camera model and pixel mapping are right |
| Scoring of near-zero GT | Thresholds \|μ_gt\| ≥ 0.01…0.2 /s, frames with \|V_z\| ≥ 0.05…0.2 m/s, median, errors capped at 100 %, error relative to the estimate | Translation matches with median or \|μ_gt\| ≥ 0.2; no single rule matches all nine numbers |
| Paper scored the TTC *map*, not events | Latest estimate per pixel (≤ 0.05 s old) compared with each 20 Hz depth frame, mean over pixels and over frames | Worse than per event (e.g. indoor_flying3: 397 / 167 / 214) |
| Cold start outdoors | Start the estimator at 222.4 s vs at the recording start | Cold start flips Driving's sign; warm start fixes it (71 %), now the default |
| Wrong sequence in Table 3 | The paper's text says Driving "performs poorly on indoor flying3", but Table 3 shows indoor_flying2. Ran indoor_flying3 | Behaves like indoor_flying2 (median 106 / 24.5 / 87 vs paper's "if2" 84.8 / 28.2 / 56.1), so a label swap does not explain the gap |
| Same scoring issue in Table 5 | Driving's flow AEE matches (2.16 vs 2.14) but its AREE does not (363 vs 104) on the same estimates | Points to the same unstated handling of near-zero GT in the paper's relative errors |

indoor_flying3 under the different protocols (`table3_mvsec.py --seq indoor_flying3`):

| Protocol | Driving | Translation | 6-DOF |
|---|---:|---:|---:|
| all events (plain mean) | 389 | 154 | 209 |
| median | 106 | 24.5 | 87 |
| \|μ_gt\| ≥ 0.2 /s (52 % of events kept) | 106 | 25.1 | 83 |
| errors capped at 100 % | 91 | 37 | 75 |
| TTC map at depth frames | 397 | 167 | 214 |

### Likely causes

1. **Unstated handling of near-zero ground truth** in the relative errors (Tables 3 and 5): a
   threshold, a median or a cap. This explains Translation and most of Driving.
2. **The 6-DOF Jacobian in the official code.** In `ParameterizedSixDOF` the Gauss–Newton Jacobian
   does not match the model's own B matrix (the log-depth column is not multiplied by −Δt, two
   rotation columns differ). We copy it as is because the VL numbers (Table 2) were produced with
   it, and with it 6-DOF underestimates the forward motion indoors (median estimate/GT 0.49 on
   indoor_flying1, 0.07 on indoor_flying2). Since no scoring rule fixes 6-DOF, the cause is in
   the estimate, and this Jacobian is the main suspect. The MVSEC numbers in the paper may have
   been produced with a different version.
3. **How `outdoor_day1` was run** (start point, settings): our outdoor Driving is weaker on all
   three tables (Table 4: 0.59 vs 0.39; Table 5: 87 vs 50; Table 3: 71 vs 47).

### Questions sent to the authors

1. Which 6-DOF Jacobian produced Tables 3–5?
2. How were pixels/events with near-zero GT handled in the relative errors (Tables 3 and 5)?
3. Table 3's text mentions indoor_flying3 but the table shows indoor_flying2: which one?
4. How was `outdoor_day1` run (from the recording start or from 222.4 s, same settings)?
5. Can you share the MVSEC evaluation script?

## 8. Problems we hit and how we fixed them

| Problem | Cause | Fix |
|---|---|---|
| 6-DOF numbers on VL did not match the paper | The paper's 6-DOF is not Eq. 9 written out, but the official re-parameterised `ParameterizedSixDOF` model (unknowns m0 = ρν_x + ω_y, m1 = ρν_y − ω_x), which the paper does not describe | Implemented that model and its (mismatched) Jacobian exactly as in the C++ |
| 6-DOF Jacobian columns differed from the C++ | Columns for m0, m1 must use fx, fy (not ρ·fx, ρ·fy) | Fixed; per-event agreement ~1e-7 |
| Small differences everywhere | The official build uses an approximate `exp()` (Schraudolph, up to 6 % off) for the residual weights | Implemented the same fast exp (on by default) |
| Signs looked flipped against the C++ | The C++ uses B_code = −B_paper | Our code uses the paper's sign; outputs identical |
| Events near the image border | The C++ window centre is `uint16`; negative values wrap and give an empty window | Mirrored (those events get no estimate) |
| VL scores off by a hair | The official scoring reads past the end of the GT file and scores the last point twice | Mirrored in `evaluate.py` |
| Comparison with the C++ looked wrong at first | Compared on truncated (10 %) sequences, and the double-precision reference build was missing `-DETTCM_FAST_EXP` | Compared full sequences and rebuilt the reference correctly |
| Pure-Python runs took ~10 hours | No Numba | All hot loops are Numba-JIT; runs take minutes |
| MVSEC files are 2–12 GB each | Do not fit in RAM | Events streamed from HDF5 in chunks (`EttcmState` carries the estimator between chunks, identical to one run); GT flow memory-mapped from the `.npz` |
| MVSEC flow did not line up with GT | MVSEC's GT flow lives in rectified pixels with the projection matrix P, despite the `_dist` file name | Estimator runs in the rectified camera from the calibration zips |
| 463 s evaluated instead of the paper's ~265 s | Used another paper's ("Secrets") frame ranges | Switched to EV-FlowNet's windows: 266 s |
| Table 3 error was ~2× too large | GT velocity recovered from the GT flow, which implies only 55–65 % of the real forward speed | Use MVSEC's own camera velocity (`odom.npz`) |
| Outdoor Driving had the wrong sign | Estimator started cold at 222.4 s while the car was already moving | Warm start from the recording start (now the default) |

## 9. What is left to do

| Item | What it involves | Size |
|---|---|---|
| Re-run everything with the cleaned code | The four commands in Section 5 | ~1–2 h of laptop time, unattended |
| Table 3 / Table 5 scoring | Waiting on the authors' answers; then implement their rule in `table3_mvsec.py` / `tables45_mvsec.py` | 1–2 h once we know the rule |
| 6-DOF on MVSEC | Add a switch for a corrected (self-consistent) 6-DOF Jacobian, re-run Tables 3–5 for 6-DOF, check Table 2 is unaffected or still close | About half a day |
| Outdoor Driving gap | Try start points and settings for `outdoor_day1` (or use the authors' answer) | A few hours |
| Translation slightly worse on Table 4 (0.03–0.25 px) | Small sweep of s, l_θ around the paper's values | A few hours |
| Figures 5 and 6 | Run `figures.py` at the paper's times and compare by eye | 1–2 h |
| Figure 1 (runtime vs accuracy) | Already have µs/event and REE; plot against the paper's other methods | 1 h |

Without the authors' input, the reproduction is essentially complete: what remains is either
waiting for them or optional checks.

## 10. Code layout

```
reimpl/
  ettcm/
    estimator.py    per-event estimator (_run: Scaling, Translation, Driving; _run_param: 6-DOF);
                    EttcmState to process long streams in chunks
    models.py       B matrix of Eq. 9 for each motion model, official 6-DOF parameterisation
    decay.py        adaptive global decay: activity a(t) of Eq. 11 + global-divergence decay
    linalg.py       small Cholesky solve (like Eigen's LLT)
    io_es.py        .es event files (VL): reader and writer
    calib.py        VL calib.txt + rectification map (like the official rectify.hpp)
    evaluate.py     REE (Eq. 12) exactly as the official benchmark computes it, VL loader
    mvsec.py        MVSEC: calibration, event streaming, GT flow, EV-FlowNet GT propagation,
                    flow metrics, GT divergence
    _jit.py         Numba JIT (falls back to plain Python; ETTCM_NO_JIT=1 forces plain Python)
  scripts/
    check_data.py      checks that the VL and MVSEC files are in place and readable
    table2_vl.py       Table 2 (VL)
    table3_mvsec.py    Table 3 (MVSEC divergence; all scoring protocols + data checks)
    tables45_mvsec.py  Tables 4 and 5 (MVSEC optical flow)
    figures.py         Figs. 5/6: events, flow and TTC maps (MVSEC or VL)
  tests/            test_io.py, test_models.py, test_chunks.py
  requirements.txt
RESULTS.md          detailed results (per sequence, official C++ comparison)
sources/links.txt   paper, code, datasets and background links
```

## 11. Details needed to match the official code

All of these are mirrored in `reimpl/`:

- **Sign convention.** The C++ uses B_code = −B_paper, so θ_code = −θ_paper and its divergence
  is −ρ·θ_z. This code uses the paper's sign (divergence = +ρ·ν_z); outputs are identical.
- **6-DOF** is the official `ParameterizedSixDOF` (see Section 8), with its Gauss–Newton Jacobian
  copied as is (the λ column is not multiplied by −Δt; columns for m0, m1 use fx, fy).
- **Driving** imposes |ν_z| = 1 exactly (not just ≤ 1), as the official model does.
- **Fast exp** for the residual weight (Schraudolph), on by default (`fast_exp=True`).
- **Window shift.** Centre at x_i − trunc(s·v̂), v̂ = predicted flow direction with ρ = 1, in raw
  pixel coordinates; neighbour positions come from the rectification map.
- **Image border.** A window centre left of / above the image wraps around (uint16) and is empty.
- **Eq. 10.** Neighbour j uses its own position but the *current* event's ρ_i.
- **λ prior** pulls towards the neighbourhood-initialised log ρ, not the previous event's.
- **w_thresh** is applied in Gauss–Newton only; the ρ initialisation uses every neighbour.
- **Iterations.** The first always runs; stop when |Δγ| < 1e-4 or after 2.
- **Scoring quirk (VL).** The last GT point is scored twice.
- **Precision.** The C++ runs in float32, this code in float64: per event they agree to ~1e-3
  (~1e-7 against a double build). 6-DOF is the most sensitive, so compare full-sequence REE,
  not the last digit.

## Citation

```bibtex
@inproceedings{nunes2023ttcm,
  title     = {Time-to-Contact Map by Joint Estimation of Up-to-Scale Inverse Depth
               and Global Motion using a Single Event Camera},
  author    = {Nunes, Urbano Miguel and Perrinet, Laurent Udo and Ieng, Sio-Hoi},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {23653--23663},
  year      = {2023}
}
```
