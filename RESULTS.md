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
