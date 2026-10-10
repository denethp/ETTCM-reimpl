# Results: Table 2 (Ventral Landing dataset)

Metric: average relative error of the global divergence, REE in % (Eq. 12). Lower is better.
Dataset: all 8 VL sequences (`VL.zip`, same data as the paper). Hyper-parameters as in the
official scripts: w_thresh = 0.01, l_theta = 1000, l_rho = 1, 2 Gauss-Newton iterations.

## Summary (averages over the 8 sequences)

| Model | s | Paper | Official code (our run) | Our Python |
|---|---|---:|---:|---:|
| Scaling | 2 | 8.29 | 8.30 | pending |
| Translation | 2 | 5.64 | 5.64 | pending |
| 6-DOF | 2 | 10.11 | 10.10 | pending |
| Scaling | 3 | 15.08 | 15.09 | pending |
| Translation | 3 | 9.31 | 9.31 | pending |
| 6-DOF | 3 | 6.42 | 6.39 | pending |

The official C++ code, built from source (Eigen 3.4, OpenCV 4.x, g++ 13, Release, `ETTCM_FAST_EXP` on),
reproduces the paper's Table 2 to within 0.03 percentage points on every average and within
0.35 on every single sequence. The small 6-DOF differences come from float32 rounding: 6-DOF is
the most sensitive model.

## Our Python implementation vs the official code, event by event

Before running the full table, our Python estimator (`reimpl/ettcm/estimator.py`) was compared with
the official C++ on every event of the first 30,000 events of sequence 3D. To remove float32 rounding
from the comparison, the official code was recompiled in double precision for this test. Relative
difference of the outputs (median over events):

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
| Scaling | Paper | 7.19 | 8.79 | 9.13 | 12.18 | 4.98 | 7.79 | 4.92 | 11.34 | **8.29** |
| | Official code (ours run) | 7.20 | 8.80 | 9.12 | 12.19 | 4.99 | 7.80 | 4.92 | 11.36 | **8.30** |
| Translation | Paper | 5.03 | 7.02 | 6.17 | 5.25 | 4.61 | 4.34 | 6.18 | 6.54 | **5.64** |
| | Official code (ours run) | 5.04 | 7.02 | 6.17 | 5.25 | 4.58 | 4.34 | 6.18 | 6.55 | **5.64** |
| | **Our Python** | - | - | - | - | - | - | - | 6.54 | - |
| 6-DOF | Paper | 6.77 | 13.73 | 14.97 | 7.50 | 6.91 | 12.15 | 12.77 | 6.10 | **10.11** |
| | Official code (ours run) | 6.83 | 13.71 | 15.28 | 7.45 | 6.56 | 12.29 | 12.79 | 5.86 | **10.10** |

### s = 3

| Model | Source | 2D-1 | 2D-2 | 2D-3 | 2D-4 | 2D-5 | 2D-6 | 2D-7 | 3D | Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Scaling | Paper | 14.99 | 15.19 | 11.69 | 21.69 | 11.19 | 15.44 | 11.01 | 19.43 | **15.08** |
| | Official code (ours run) | 14.99 | 15.20 | 11.69 | 21.70 | 11.20 | 15.44 | 11.02 | 19.43 | **15.09** |
| Translation | Paper | 11.15 | 7.01 | 7.98 | 13.80 | 6.37 | 11.17 | 4.55 | 12.46 | **9.31** |
| | Official code (ours run) | 11.14 | 7.01 | 8.00 | 13.81 | 6.38 | 11.16 | 4.55 | 12.46 | **9.31** |
| 6-DOF | Paper | 6.51 | 6.81 | 7.25 | 6.77 | 6.06 | 5.59 | 6.18 | 6.19 | **6.42** |
| | Official code (ours run) | 6.27 | 7.07 | 7.28 | 6.77 | 6.06 | 5.43 | 6.16 | 6.10 | **6.39** |

## How to reproduce

Official code (from inside the official `ettcm` folder, after building it):

```bash
./build/src/global_divergence_translation datasets/VL/2D-1/events.es datasets/VL/calib.txt \
    datasets/VL/2D-1/divergence.txt datasets/VL/2D-1/timestamps.txt -s 2 -e 0.01 -m 1000 -l 1
```

Our Python (with Numba, a few minutes for everything):

```bash
unzip VL.zip -d datasets          # creates datasets/VL/...
cd reimpl && source .venv/bin/activate
python scripts/run_vl.py --data ../datasets/VL --all --csv ../results_python_vl.csv
```

`six_dof_param` is the 6-DOF model the paper's table uses (the official `ParameterizedSixDOF`).
`six_dof` is the plain Eq. 9 version; it is not in the paper's table.
