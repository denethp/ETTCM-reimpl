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
| `ettcm/models.py` | done | B matrix of Eq. 9 (Scaling, Translation, 6-DOF) + the official re-parameterised 6-DOF |
| `ettcm/estimator.py` | done | per-event estimator (`_run`) + official re-parameterised 6-DOF (`_run_param`) |
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
python tests/test_models.py      # Scaling, Translation, 6-DOF PASS
```

If `numba` will not install (very new Python), everything still runs without it, only
~500× slower. Use a venv with Python 3.12 for Numba, or debug in pure Python on purpose:
`ETTCM_NO_JIT=1 python scripts/run_synthetic.py ...` (gives readable tracebacks).

## Milestones

**M1 – Scaling model on synthetic data.** (done)
`python scripts/run_synthetic.py --model scaling --duration 0.4`
Target: divergence converges in ~0.15 s, then REE ~1–3 %.

**M2 – Translation and 6-DOF.** (done) `tests/test_models.py` passes;
`python scripts/run_synthetic.py --model translation --nu 0.3,0,1` and `--model six_dof`.

**M3 – Match the official C++ per event.** (done: Scaling, Translation and the
paper's 6-DOF agree with the official C++ compiled in double precision to ~1e-7
relative per event on real VL data, fast exp on.) Build `cpp/harness.cpp` (below), save the
same synthetic data with `--save-es syn.es`, run both, diff the divergence columns.
The official 6-DOF is the re-parameterised `six_dof_param` model (see below).

**M4 – VL / Table 2.** (done, see `../RESULTS.md`) Unzip VL.zip into `../datasets/`, then
`python scripts/run_vl.py --data ../datasets/VL --all --csv ../results_python_vl.csv`.
Each line prints our REE next to the paper's. Results: see `../RESULTS.md`.

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

Found while matching the official code on the real VL data (all mirrored in this kit):

* **Fast exp.** The official build uses an approximate `exp()` (Schraudolph, up to
  ~6 % off) for the residual weight, ON by default. `run_ettcm(..., fast_exp=True)`
  reproduces it bit for bit; `--exact-exp` in `run_vl.py` turns it off.
* **The paper's 6-DOF is `ParameterizedSixDOF`**, not plain Eq. 9: unknowns
  m0 = ρν_x + ω_y, m1 = ρν_y − ω_x. Use model `six_dof_param`. Two C++ details are
  kept as is: its λ-Jacobian column is not multiplied by −Δt, and its Jacobian for
  m0, m1 uses fx, fy (not ρ·fx, ρ·fy).
* **Image border.** The C++ window centre is a uint16, so a centre left of / above the
  image wraps around and gives an empty window (event gets ρ = 0).
* **Scoring quirk.** The official evaluation scores the last GT point twice (reading
  past the end of the file leaves the last value in place). `evaluate.py` does the same.
* **float32 vs float64.** The official code runs in float32, this kit in float64. Per
  event the results agree to ~1e-3 with the float build and ~1e-7 with a double build;
  6-DOF is the most sensitive, so tiny differences can grow over a long sequence.
  Compare full-sequence REE, not the last digit.
* **calib.txt** is 9 numbers on line 1 and `width height` on line 2.
