# ETTCM re-implementation (Python + NumPy + Numba)

Own implementation of Nunes, Perrinet, Ieng, *Time-to-Contact Map by Joint Estimation of
Up-to-Scale Inverse Depth and Global Motion using a Single Event Camera*, ICCV 2023.

## Layout

| File | Status | What it is |
|---|---|---|
| `ettcm/io_es.py` | done | `.es` (sepia Event Stream v2, dvs) reader + writer |
| `ettcm/calib.py` | done | `calib.txt` + rectification map, like `rectify.hpp` |
| `ettcm/decay.py` | done | adaptive global decay → a(t) of Eq. 11 + divergence decay |
| `ettcm/linalg.py` | done | small Cholesky solve (Eigen LLT equivalent) |
| `ettcm/synthetic.py` | done | approaching textured plane, idealised event camera, exact GT |
| `ettcm/evaluate.py` | done | REE (Eq. 12) at GT timestamps, VL loader |
| `ettcm/models.py` | **TODO (M2)** | B matrix of Eq. 9. Scaling done as the worked example |
| `ettcm/estimator.py` | **TODO (M1–M3)** | per-event estimator, 5 TODO blocks |
| `scripts/run_synthetic.py` | done | M1–M3 test on synthetic data, saves a plot |
| `scripts/run_vl.py` | done | M4: Table 2 on VL |
| `tests/` | done | `test_models.py` checks your B against brute-force 3D projection |
| `cpp/harness.cpp` | done | per-event output of the *official* core, for side-by-side debugging |

## Setup

```bash
cd reimpl
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_io.py          # PASS
python tests/test_models.py      # Scaling PASS, others TODO until M2
```

If `numba` will not install (very new Python), everything still runs without it, only
~500× slower. Use a venv with Python 3.12 for Numba, or debug in pure Python on purpose:
`ETTCM_NO_JIT=1 python scripts/run_synthetic.py ...` (gives readable tracebacks).

## Milestones

**M1 – Scaling model on synthetic data.** Fill TODO 1–5 in `estimator.py`.
`python scripts/run_synthetic.py --model scaling --duration 0.4`
Target: divergence converges in ~0.15 s, then REE ~1–3 %.

**M2 – Translation and 6-DOF.** Fill `b_matrix` in `models.py` until
`tests/test_models.py` passes, then
`python scripts/run_synthetic.py --model translation --nu 0.3,0,1` and `--model six_dof`.

**M3 – Match the official C++ per event.** Build `cpp/harness.cpp` (below), save the
same synthetic data with `--save-es syn.es`, run both, diff the divergence columns.
Expect agreement to ~1e-4 (float32 vs float64); the official 6-DOF uses a
re-parameterised model (`ParameterizedSixDOF`), so only Scaling/Translation match exactly.

**M4 – VL / Table 2.** `python scripts/run_vl.py --data <ettcm>/datasets/VL --all`
and compare with the official binaries + the paper's numbers.

**M5 – TTCM and MVSEC.** Per-pixel TTC = 1/(rho·nu_z) (already in `run_ettcm` output),
then MVSEC loader + optical flow metrics (AEE/Out/AREE/AAE, Eqs. 13–15), l_theta = 100.
Not in the official repo, so this part is fully yours.

## Building the C++ harness (after the official code is built)

From inside the official `ettcm` folder:

```bash
g++ -O3 -std=c++17 -march=native -DETTCM_FAST_EXP \
    -Iinclude -Ibuild/_deps/pontella-src/source -Ibuild/_deps/sepia-src/source \
    $(pkg-config --cflags eigen3 opencv4) \
    ../reimpl/cpp/harness.cpp -o harness $(pkg-config --libs opencv4) -lpthread
./harness translation syn.es cpp_out.txt 2 100 59.5 44.5 120 90
#          model      input  output    s  f   cx   cy  W   H   (synthetic defaults)
```

## Things that bite (learned while checking this kit against the C++)

* **Sign convention.** Code uses B_code = −B_paper, so θ_code = −θ_paper and
  divergence_code = −ρ·θ_z. This kit uses the paper's sign: divergence = +ρ·ν_z.
* **Window shift.** The neighbourhood is centred at x_i − trunc(s·v̂), v̂ = predicted
  flow direction with ρ = 1; it is in *raw* pixel coordinates, positions come from the
  rectification map.
* **Eq. 10.** Neighbour j uses its own position p_j but the *current* event's ρ_i.
* **The λ prior** pulls towards the neighbourhood-initialised log ρ, not the previous event's.
* **w_thresh** is applied in Gauss–Newton only; the ρ initialisation uses every
  neighbour with a timestamp.
* **Several events per edge bias the flow up.** With a low contrast threshold, each edge
  fires many events per pixel; the current event (first crossing) gets matched to the
  neighbour's latest event (last crossing), Δt looks too short, flow and divergence come
  out ~20–50 % too high. That is why the synthetic default is C = 0.6 with sharp edges.
  This is a property of the method (the official C++ shows the same), not a bug.
* **First iteration runs always**; stop when |Δγ| < 1e-4 or after 2 iterations.
