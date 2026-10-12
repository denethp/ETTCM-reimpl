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
python scripts/run_vl.py --data ../datasets/VL --all --csv ../results_python_vl.csv
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
(`reimpl/ettcm/mvsec.py`, `reimpl/scripts/run_mvsec.py`). Settings from the paper: s = 3,
w_thresh = 0.01, l_θ = 100, l_ρ = 1. Sequences indoor_flying1–3 (whole) and outdoor_day1
(222.4–240.4 s, EV-FlowNet's test split): **265.96 s evaluated**, matching the paper's
"approximately 265 sec". Flow intervals: consecutive grayscale frames (dt = 1). Each event's
flow m_i = Δt_frame · v_i is compared with the MVSEC GT flow at its pixel; outdoor_day1 rows < 190.

### Table 4: optical flow, AEE (px/frame) / Out (%)

| Model | Source | indoor_flying1 | indoor_flying2 | indoor_flying3 | outdoor_day1 |
|---|---|---|---|---|---|
| Driving | Paper | 1.40 / 7.50 | 2.42 / 26.32 | 2.14 / 26.05 | 0.39 / 0.12 |
| | **Ours** | **1.35** / 6.93 | **2.47** / 27.36 | **2.16** / 26.78 | **0.62** / 1.32 |
| Translation | Paper | 0.63 / 0.41 | 0.94 / 2.83 | 0.79 / 1.65 | 0.55 / 0.97 |
| | **Ours** | **0.66** / 0.87 | **1.19** / 6.48 | **0.94** / 3.69 | **0.68** / 1.58 |
| 6-DOF | Paper | 0.88 / 1.97 | 1.60 / 12.02 | 1.38 / 9.99 | 0.81 / 2.26 |
| | **Ours** | **0.84** / 1.79 | **1.60** / 10.77 | **1.38** / 8.66 | **0.77** / 2.38 |

### Table 5: optical flow, AREE (%) / AAE (deg)

| Model | Source | indoor_flying3 | outdoor_day1 |
|---|---|---|---|
| Driving | Paper | 104.05 / 60.36 | 50.25 / 13.31 |
| | **Ours** | 362.85 / **60.75** | 98.57 / **15.67** |
| Translation | Paper | 44.46 / 11.38 | 72.67 / 21.77 |
| | **Ours** | 50.90 / **12.12** | 96.23 / **20.38** |
| 6-DOF | Paper | 95.23 / 21.35 | 122.04 / 31.68 |
| | **Ours** | 84.35 / **21.41** | 121.23 / **22.89** |

### Table 3: divergence (inverse TTC), AREE (%)

| Model | Source | indoor_flying1 | indoor_flying2 | outdoor_day1 |
|---|---|---|---|---|
| Driving | Paper | 153.30 | 84.80 | 47.27 |
| | Ours, mean (Eq. 14) | 473.29 | 250.33 | 85.51 |
| | Ours, median | 194.67 | 128.17 | 36.97 |
| Translation | Paper | 19.82 | 28.23 | 45.78 |
| | Ours, mean (Eq. 14) | 66.77 | 81.88 | 60.24 |
| | Ours, median | 23.82 | 28.62 | 39.67 |
| 6-DOF | Paper | 53.65 | 56.12 | 56.66 |
| | Ours, mean (Eq. 14) | 116.84 | 120.01 | 67.17 |
| | Ours, median | 81.02 | 95.74 | 44.48 |

**Summary**

- **Optical flow (Table 4) is reproduced.** Driving and 6-DOF match the paper on the indoor
  sequences to within 0.05 px/frame (e.g. 6-DOF indoor_flying2 1.60 vs 1.60, indoor_flying3 1.38 vs
  1.38). Translation is slightly worse than the paper (0.03–0.25 px/frame), outdoor Driving more so
  (0.62 vs 0.39).
- **Flow direction (AAE, Table 5) is reproduced** to within ~2.5° everywhere except 6-DOF on
  outdoor_day1, where ours is better (22.9° vs 31.7°). AREE is close for Translation and 6-DOF on
  indoor_flying3 (50.9 vs 44.5, 84.4 vs 95.2); Driving's AREE is much higher (relative errors explode where the GT flow is
  almost zero, and the paper does not say how it handled those).
- **Divergence (Table 3)**: our per-event estimates have the right sign (91–100 % of events for
  Translation and 6-DOF) and, for Translation, the right size (median estimate / GT 1.10, 1.12,
  0.78). Our *mean* relative error (Eq. 14) is 2–3× the paper's; our *median* relative error is
  close to the paper's numbers for Translation (23.8 / 28.6 / 39.7 vs 19.8 / 28.2 / 45.8). The
  mean is dominated by events where the true forward speed is close to zero. The paper does not
  describe its exact Table 3 evaluation (which events, any exclusions), so this table is only
  approximately reproduced.

**How the GT divergence was built.** μ_gt = V_z / Z per event: Z from MVSEC's rectified depth
map, V_z from MVSEC's camera velocity exactly as MVSEC's own tool computes it (odometry poses,
±10-sample in-place smoothing). Recovering V from the GT flow files instead gives about half the
forward speed of the odometry, which made our divergence look 2× too large; the odometry-based
GT is what the paper describes ("GT depth maps and camera velocity"). `run_mvsec.py` prints the
results for all four GT velocity variants.

Reproduce (on the laptop, ~20 min for everything):

```bash
python scripts/run_mvsec.py --data ../datasets/MVSEC --seq all --model all --table3 --csv ../results_python_mvsec.csv
```
