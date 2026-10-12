# Results: Table 2 (Ventral Landing dataset)

Metric: average relative error of the global divergence, REE in % (Eq. 12). Lower is better.
Dataset: all 8 VL sequences (`VL.zip`, the same data as the paper). Hyper-parameters as in the
official scripts: w_thresh = 0.01, l_theta = 1000, l_rho = 1, 2 Gauss-Newton iterations.

## Summary (averages over the 8 sequences)

| Model | s | Paper | Official code (our run) | **Our Python** | Python µs/event |
|---|---|---:|---:|---:|---:|
| Scaling | 2 | 8.29 | 8.30 | **8.29** | 0.81 |
| Translation | 2 | 5.64 | 5.64 | **5.64** | 1.02 |
| 6-DOF | 2 | 10.11 | 10.10 | **10.13** | 1.49 |
| Scaling | 3 | 15.08 | 15.09 | **15.08** | 1.26 |
| Translation | 3 | 9.31 | 9.31 | **9.31** | 1.69 |
| 6-DOF | 3 | 6.42 | 6.39 | **6.34** | 2.05 |

**Our Python implementation reproduces the paper's Table 2.**

- **Scaling and Translation:** every average matches the paper to within 0.01, and every single
  sequence to within 0.04.
- **6-DOF:** averages within 0.08 of the paper (10.13 vs 10.11, 6.34 vs 6.42); single sequences
  within 0.6. 6-DOF is very sensitive to rounding: our code runs in float64, the official code in
  float32, and the official code itself lands up to 0.35 away from the paper's numbers.
- **Speed:** Python + Numba on a laptop (Lenovo LOQ), about 0.8–2 µs per event, i.e. real time.
  The paper reports about 1.15 µs/event for its C++ (Translation, s = 2) on different hardware.

"Our Python" was run on the laptop with:

```bash
python scripts/table2_vl.py --data ../datasets/VL --all --csv ../results_python_vl.csv
```

"Official code" is the authors' C++ (github.com/neuromorphic-paris/ETTCM), built from source
(Eigen 3.4, OpenCV 4.x, g++ 13, Release, `ETTCM_FAST_EXP` on) and run on the same data.

## Our Python vs the official code, event by event

Before the full runs, our estimator (`reimpl/ettcm/estimator.py`) was compared with the official
C++ on every one of the first 30,000 events of sequence 3D. To take float32 rounding out of the
comparison, the official code was recompiled in double precision for this test. Median relative
difference of the outputs:

| Model | global divergence | inverse depth ρ | ν_z |
|---|---:|---:|---:|
| Scaling | 1.7e-7 | 9.1e-9 | 2.7e-7 |
| Translation | 2.8e-7 | 2.8e-8 | 4.6e-7 |
| 6-DOF (paper's parameterised model) | 3.1e-7 | 1.6e-7 | 4.1e-7 |

So the Python code computes the same thing as the official code, up to rounding.

## Per-sequence numbers

### s = 2

| Model | Source | 2D-1 | 2D-2 | 2D-3 | 2D-4 | 2D-5 | 2D-6 | 2D-7 | 3D | Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Scaling | Paper | 7.19 | 8.79 | 9.13 | 12.18 | 4.98 | 7.79 | 4.92 | 11.34 | 8.29 |
| | Official code | 7.20 | 8.80 | 9.12 | 12.19 | 4.99 | 7.80 | 4.92 | 11.36 | 8.30 |
| | **Our Python** | 7.20 | 8.80 | 9.13 | 12.19 | 4.98 | 7.77 | 4.92 | 11.36 | **8.29** |
| Translation | Paper | 5.03 | 7.02 | 6.17 | 5.25 | 4.61 | 4.34 | 6.18 | 6.54 | 5.64 |
| | Official code | 5.04 | 7.02 | 6.17 | 5.25 | 4.58 | 4.34 | 6.18 | 6.55 | 5.64 |
| | **Our Python** | 5.03 | 7.04 | 6.18 | 5.24 | 4.57 | 4.34 | 6.18 | 6.54 | **5.64** |
| 6-DOF | Paper | 6.77 | 13.73 | 14.97 | 7.50 | 6.91 | 12.15 | 12.77 | 6.10 | 10.11 |
| | Official code | 6.83 | 13.71 | 15.28 | 7.45 | 6.56 | 12.29 | 12.79 | 5.86 | 10.10 |
| | **Our Python** | 6.84 | 13.77 | 15.37 | 7.49 | 6.94 | 12.27 | 12.86 | 5.52 | **10.13** |

### s = 3

| Model | Source | 2D-1 | 2D-2 | 2D-3 | 2D-4 | 2D-5 | 2D-6 | 2D-7 | 3D | Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Scaling | Paper | 14.99 | 15.19 | 11.69 | 21.69 | 11.19 | 15.44 | 11.01 | 19.43 | 15.08 |
| | Official code | 14.99 | 15.20 | 11.69 | 21.70 | 11.20 | 15.44 | 11.02 | 19.43 | 15.09 |
| | **Our Python** | 14.99 | 15.20 | 11.69 | 21.69 | 11.20 | 15.44 | 11.02 | 19.43 | **15.08** |
| Translation | Paper | 11.15 | 7.01 | 7.98 | 13.80 | 6.37 | 11.17 | 4.55 | 12.46 | 9.31 |
| | Official code | 11.14 | 7.01 | 8.00 | 13.81 | 6.38 | 11.16 | 4.55 | 12.46 | 9.31 |
| | **Our Python** | 11.15 | 7.02 | 7.99 | 13.80 | 6.38 | 11.16 | 4.55 | 12.46 | **9.31** |
| 6-DOF | Paper | 6.51 | 6.81 | 7.25 | 6.77 | 6.06 | 5.59 | 6.18 | 6.19 | 6.42 |
| | Official code | 6.27 | 7.07 | 7.28 | 6.77 | 6.06 | 5.43 | 6.16 | 6.10 | 6.39 |
| | **Our Python** | 6.43 | 7.24 | 7.28 | 6.63 | 6.06 | 5.06 | 5.87 | 6.12 | **6.34** |

## Notes on the official code

Details that must be copied to reproduce the numbers (all mirrored in `reimpl/`):

- **Fast exp.** The residual weight uses an approximate `exp()` (up to ~6 % off), on by default.
- **6-DOF model.** The paper's 6-DOF rows come from the official `ParameterizedSixDOF` model
  (unknowns m0 = ρν_x + ω_y, m1 = ρν_y − ω_x), which the paper does not describe. Our `six_dof`
  model is that same re-parameterised model.
- **6-DOF derivatives.** In `ParameterizedSixDOF`, the Jacobian column for log inverse depth is
  not multiplied by −Δt, and the columns for two rotation unknowns don't match the model's own
  B matrix. Kept as is here, because the paper's numbers were produced with them.
- **Image border.** A search window centred left of / above the image wraps around (uint16) and is
  empty, so those events get no estimate.
- **Scoring.** The last ground-truth point is scored twice (end-of-file read quirk).
- **Sign convention.** The code uses B_code = −B_paper, θ_code = −θ_paper. Our code uses the
  paper's signs; outputs (divergence, TTC, depth, flow) are identical.

---

# MVSEC: Tables 3, 4 and 5 (optical flow and divergence)

The official code does not include these experiments; everything below is our own pipeline
(`reimpl/ettcm/mvsec.py`, `reimpl/scripts/tables45_mvsec.py` for Tables 4–5, `table3_mvsec.py` for
Table 3). Settings from the
paper: s = 3, w_thresh = 0.01, l_θ = 100, l_ρ = 1. Sequences indoor_flying1–3 (whole) and
outdoor_day1 (222.4–240.4 s, EV-FlowNet's test split): **265.96 s evaluated**, matching the paper's
"approximately 265 sec". Flow intervals: consecutive grayscale frames (dt = 1). Each event's flow
m_i = Δt_frame · v_i is compared with the MVSEC GT flow at its pixel; outdoor_day1 rows < 190.
For outdoor_day1 the estimator runs from the start of the recording (warm start, the default) and
only the window is evaluated; starting cold at 222.4 s can leave the Driving model on the wrong sign
of ν_z.

### Table 4: optical flow, AEE (px/frame) / Out (%)

| Model | Source | indoor_flying1 | indoor_flying2 | indoor_flying3 | outdoor_day1 |
|---|---|---|---|---|---|
| Driving | Paper | 1.40 / 7.50 | 2.42 / 26.32 | 2.14 / 26.05 | 0.39 / 0.12 |
| | **Ours** | **1.35** / 6.93 | **2.47** / 27.36 | **2.16** / 26.78 | **0.59** / 1.25 |
| Translation | Paper | 0.63 / 0.41 | 0.94 / 2.83 | 0.79 / 1.65 | 0.55 / 0.97 |
| | **Ours** | **0.66** / 0.87 | **1.19** / 6.48 | **0.94** / 3.69 | **0.67** / 1.59 |
| 6-DOF | Paper | 0.88 / 1.97 | 1.60 / 12.02 | 1.38 / 9.99 | 0.81 / 2.26 |
| | **Ours** | **0.84** / 1.79 | **1.60** / 10.77 | **1.38** / 8.66 | **0.76** / 1.91 |

### Table 5: optical flow, AREE (%) / AAE (deg)

| Model | Source | indoor_flying3 | outdoor_day1 |
|---|---|---|---|
| Driving | Paper | 104.05 / 60.36 | 50.25 / 13.31 |
| | **Ours** | 362.85 / **60.75** | 87.27 / **15.11** |
| Translation | Paper | 44.46 / 11.38 | 72.67 / 21.77 |
| | **Ours** | 50.90 / **12.12** | 92.73 / **21.95** |
| 6-DOF | Paper | 95.23 / 21.35 | 122.04 / 31.68 |
| | **Ours** | 84.35 / **21.41** | 120.29 / **22.52** |

### Table 3: divergence (inverse TTC), AREE (%)

GT: μ = V_z / Z per event, Z from MVSEC's rectified depth map, V_z from MVSEC's published camera
velocity (`<seq>_odom.npz`, `lin_vel`).

| Model | Source | indoor_flying1 | indoor_flying2 | outdoor_day1 |
|---|---|---:|---:|---:|
| Driving | Paper | 153.30 | 84.80 | 47.27 |
| | Ours, all events (Eq. 14) | 315.52 | 212.16 | 70.96 |
| | Ours, median | 120.33 | 103.35 | 36.2 |
| Translation | Paper | 19.82 | 28.23 | 45.78 |
| | Ours, all events (Eq. 14) | 66.77 | 81.88 | 54.31 |
| | Ours, median | **23.82** | **28.69** | **44.0** |
| | Ours, only \|μ_gt\| ≥ 0.2 /s | **18.24** | **29.44** | – |
| 6-DOF | Paper | 53.65 | 56.12 | 56.66 |
| | Ours, all events (Eq. 14) | 113.15 | 120.27 | 66.09 |
| | Ours, median | 83.24 | 95.73 | 53.7 |

indoor_flying3 is not in the paper's Table 3, but its discussion says Driving "performs poorly on
the indoor flying3 sequence", so we ran it too (`table3_mvsec.py --seq indoor_flying3`):

| Protocol | Driving | Translation | 6-DOF |
|---|---:|---:|---:|
| all events (Eq. 14) | 388.91 | 153.85 | 209.21 |
| median | 105.59 | 24.52 | 87.49 |
| errors capped at 100 % | 90.88 | 37.28 | 74.80 |
| \|μ_gt\| ≥ 0.05 /s (84 % of events kept) | 137.84 | 39.32 | 88.06 |
| \|μ_gt\| ≥ 0.1 /s (73 % kept) | 124.56 | 32.31 | 83.96 |
| \|μ_gt\| ≥ 0.2 /s (52 % kept) | 106.39 | 25.11 | 82.56 |
| frames \|V_z\| ≥ 0.1 m/s (84 % kept) | 141.58 | 35.75 | 91.94 |
| frames \|V_z\| ≥ 0.2 m/s (71 % kept) | 125.63 | 30.51 | 86.24 |
| TTC map at the depth frames (≤ 0.05 s old) | 397.21 | 166.97 | 213.69 |
| all events, GT velocity from the GT flow | 760.74 | 313.86 | 350.27 |

It behaves like indoor_flying2: a mislabelled column would not explain the gap.

**Summary**

- **Optical flow (Table 4) is reproduced.** Driving and 6-DOF match the paper on the indoor
  sequences to within 0.05 px/frame (e.g. 6-DOF indoor_flying2 1.60 vs 1.60, indoor_flying3 1.38 vs
  1.38). Translation is slightly worse than the paper (0.03–0.25 px/frame), outdoor Driving more so
  (0.59 vs 0.39).
- **Flow direction (AAE, Table 5) is reproduced** to within ~2° everywhere except 6-DOF on
  outdoor_day1, where ours is better (22.5° vs 31.7°). AREE is close for Translation and 6-DOF on
  indoor_flying3 (50.9 vs 44.5, 84.4 vs 95.2); Driving's AREE is much higher.
- **Divergence (Table 3) is reproduced for the Translation model (the paper's best) and
  approximately outdoors, not for 6-DOF and Driving indoors.**
  - Outdoors all three models land within 9–24 points of the paper (71 / 54 / 66 vs 47 / 46 / 57),
    with the right sign (97–100 % of events) and, for Driving, the right size (median est/GT 1.01).
  - Indoors, the paper's Translation numbers (19.8 / 28.2) are matched by our median (23.8 / 28.7)
    or by excluding events with near-zero GT divergence, |μ_gt| < 0.2 /s (18.2 / 29.4). Our plain
    mean is 2–3× higher: the drone often barely moves forward (|V_z| < 0.05 m/s in 13–20 % of
    frames), where any error becomes a huge relative error. The paper does not say how it treated
    these; no single protocol we tried (`table3_mvsec.py`: thresholds, medians, caps, TTC-space
    errors, map-based evaluation at the depth frames) matches all nine numbers at once.
  - Indoor 6-DOF underestimates the forward motion (median est/GT 0.49 and 0.07 on indoor_flying1/2),
    and Driving (which can only move forward and yaw) cannot explain the drone's motion; no
    evaluation protocol brings 6-DOF to the paper's values, and Driving only into its range.
  - Likely causes and the questions for the authors are in the README, Section 7.

**Ground truth and data checks** (`table3_mvsec.py`):

- The GT flow inside `<seq>_gt.hdf5` is identical to `<seq>_gt_flow_dist.npz`.
- Our camera velocity from the odometry poses equals MVSEC's published `_odom.npz` (ratio 1.000).
- **MVSEC's GT flow files imply only 55–65 % of that forward speed** (least-squares fit of the
  velocity to the GT flow and depth). Using the flow-implied velocity as GT made our divergence look
  2× too large; the odometry velocity is what the paper describes ("GT depth maps and camera
  velocity") and what we use. This is an inconsistency inside MVSEC, worth knowing for any
  divergence/TTC evaluation on it.

Reproduce (from `reimpl/`):

```bash
python scripts/tables45_mvsec.py --data ../datasets/MVSEC --seq all --model all --csv ../results_python_mvsec.csv   # Tables 4, 5
python scripts/table3_mvsec.py --data ../datasets/MVSEC                                                      # Table 3 + data checks
python scripts/table3_mvsec.py --data ../datasets/MVSEC --seq indoor_flying3
```
