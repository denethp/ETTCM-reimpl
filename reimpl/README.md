# ETTCM re-implementation (Python + NumPy + Numba)

Own implementation of Nunes, Perrinet, Ieng, *Time-to-Contact Map by Joint Estimation of
Up-to-Scale Inverse Depth and Global Motion using a Single Event Camera*, ICCV 2023.
It reproduces the paper's Table 2 on the Ventral Landing (VL) dataset; see `../RESULTS.md`.

## Layout

| File | What it is |
|---|---|
| `ettcm/io_es.py` | `.es` (sepia Event Stream v2, dvs) reader + writer |
| `ettcm/calib.py` | `calib.txt` + rectification map, like the official `rectify.hpp` |
| `ettcm/decay.py` | adaptive global decay: activity a(t) of Eq. 11 + divergence decay |
| `ettcm/linalg.py` | small Cholesky solve (Eigen LLT equivalent) |
| `ettcm/models.py` | B matrix of Eq. 9: Scaling, Translation, and the official re-parameterised 6-DOF |
| `ettcm/estimator.py` | per-event estimator: `_run` (Scaling, Translation), `_run_param` (6-DOF) |
| `ettcm/evaluate.py` | REE (Eq. 12) as the official benchmark computes it, VL loader |
| `ettcm/_jit.py` | optional Numba JIT (falls back to plain Python) |
| `scripts/run_vl.py` | Table 2: runs the models on VL and prints our REE next to the paper's |
| `tests/test_io.py` | `.es` write/read round trip |
| `tests/test_models.py` | B matrices checked against brute-force 3D projection |

## Setup

```bash
cd reimpl
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_io.py          # PASS
python tests/test_models.py      # PASS x3
```

Without Numba everything still runs, only ~300× slower. Set `ETTCM_NO_JIT=1` to force
plain Python (readable tracebacks when debugging).

## Reproducing Table 2

Unzip `VL.zip` (https://figshare.com/ndownloader/files/44987794) into `../datasets/`, then:

```bash
python scripts/run_vl.py --data ../datasets/VL --model translation -s 2      # one row
python scripts/run_vl.py --data ../datasets/VL --all --csv ../results_python_vl.csv
```

Models: `scaling`, `translation`, `six_dof`. Settings are those of the official script
`table_2_results.sh`: window s ∈ {2, 3}, w_thresh = 0.01, l_theta = 1000, l_rho = 1,
2 Gauss–Newton iterations.

## How the estimator works (per event)

1. Predict the flow direction from the current motion, shift the (2s+1)² window back along it.
2. Initialise ρ from the neighbours, recent ones weighted more (Eq. 11).
3. Gauss–Newton on (motion, log ρ), at most 2 iterations, with priors (Eq. 8, Sec. 3.6).
4. Per-event flow (Eq. 9); store the event in the context.
5. Global divergence = decayed mean of ρ·ν_z (what Table 2 evaluates).

## Details needed to match the official code (all mirrored here)

* **Sign convention.** The C++ uses B_code = −B_paper, so θ_code = −θ_paper and
  divergence_code = −ρ·θ_z. This code uses the paper's sign: divergence = +ρ·ν_z.
  Outputs (divergence, TTC, depth, flow) are identical.
* **The paper's 6-DOF is the official `ParameterizedSixDOF`**, not Eq. 9 written out
  directly: unknowns m0 = ρν_x + ω_y, m1 = ρν_y − ω_x (code sign). The paper does not
  describe this. Its Gauss–Newton Jacobian is copied as is from the C++ (the λ column is
  not multiplied by −Δt; columns for m0, m1 use fx, fy), because the paper's numbers
  were produced with it.
* **Fast exp.** The official build uses an approximate `exp()` (Schraudolph, up to ~6 %
  off) for the residual weight, ON by default. `run_ettcm(..., fast_exp=True)` (default)
  reproduces it bit for bit.
* **Window shift.** Centred at x_i − trunc(s·v̂), v̂ = predicted flow direction with ρ = 1,
  in *raw* pixel coordinates; positions come from the rectification map.
* **Image border.** The C++ window centre is a uint16, so a centre left of / above the
  image wraps around and gives an empty window (the event gets ρ = 0).
* **Eq. 10.** Neighbour j uses its own position p_j but the *current* event's ρ_i.
* **The λ prior** pulls towards the neighbourhood-initialised log ρ, not the previous event's.
* **w_thresh** is applied in Gauss–Newton only; the ρ initialisation uses every
  neighbour with a timestamp.
* **First iteration always runs**; stop when |Δγ| < 1e-4 or after 2 iterations.
* **Scoring quirk.** The official evaluation scores the last GT point twice (reading past
  the end of the file leaves the last value in place). `evaluate.py` does the same.
* **float32 vs float64.** The official code runs in float32, this code in float64. Per
  event the results agree to ~1e-3 with the float build and ~1e-7 with a double build.
  6-DOF is the most sensitive, so tiny differences can grow over a long sequence:
  compare full-sequence REE, not the last digit.
* **calib.txt** is 9 numbers on line 1 and `width height` on line 2.
