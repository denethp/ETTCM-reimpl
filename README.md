# ETTCM-reimpl

Unofficial Python + Numba re-implementation of **"Time-to-Contact Map by Joint Estimation of Up-to-Scale Inverse Depth and Global Motion using a Single Event Camera"** (U. M. Nunes, L. U. Perrinet, S.-H. Ieng, ICCV 2023).

The method processes events one at a time and jointly estimates the camera's global motion and a per-event, up-to-scale inverse depth. From that it gets per-event optical flow, a global divergence estimate and a per-pixel time-to-contact (TTC) map, at around 1 µs per event in the official C++.

- Paper: [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Nunes_Time-to-Contact_Map_by_Joint_Estimation_of_Up-to-Scale_Inverse_Depth_and_ICCV_2023_paper.html)
- Official C++ code: [neuromorphic-paris/ETTCM](https://github.com/neuromorphic-paris/ETTCM) (CC BY-NC-SA 4.0, not included in this repo)

## Status

**Table 2 of the paper (Ventral Landing dataset) is reproduced** with this Python implementation:
every average within 0.1 of the paper, at about 1 µs per event. The code matches the official
C++ event by event (to ~1e-7 against a double-precision build). Details in [`RESULTS.md`](RESULTS.md).

## Repository layout

```
reimpl/
  ettcm/
    io_es.py        .es event file (sepia Event Stream v2) reader/writer
    calib.py        calibration + rectification map
    decay.py        adaptive global decay, event activity a(t)
    linalg.py       small Cholesky solve
    models.py       motion models: Scaling, Translation, 6-DOF (B matrix, Eq. 9)
    estimator.py    per-event estimator
    evaluate.py     REE metric (Eq. 12), VL loader
    _jit.py         optional Numba JIT
  scripts/run_vl.py Table 2 on VL, prints our numbers next to the paper's
  tests/            test_io.py, test_models.py
RESULTS.md          Table 2: paper vs official code vs this implementation
sources/links.txt   paper, code, dataset and background links
```

## Quick start

```bash
cd reimpl
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_io.py
python tests/test_models.py
unzip -n ../VL.zip -d ../datasets       # VL dataset, see below
python scripts/run_vl.py --data ../datasets/VL --all --csv ../results_python_vl.csv
```

Implementation notes (including details needed to match the official code) are in [`reimpl/README.md`](reimpl/README.md).

## Datasets (not stored in this repo)

- **Ventral Landing (VL)**, for Table 2: https://figshare.com/ndownloader/files/44987794
- **MVSEC**, for Tables 3–5: https://daniilidis-group.github.io/mvsec/

## Target numbers (paper, Table 2, average REE %)

| Model       | s = 2 | s = 3 |
|-------------|-------|-------|
| Scaling     | 8.29  | 15.08 |
| Translation | 5.64  | 9.31  |
| 6-DOF       | 10.11 | 6.42  |

Baseline ECMD: 8.92 %. Runtime for Translation, s = 2: about 1.15 µs/event.

## Roadmap

- [x] Build the official code and reproduce Table 2 on VL
- [x] Own implementation: Scaling, Translation and 6-DOF models
- [x] Match the official C++ per event
- [x] Reproduce Table 2 on VL with the own implementation
- [ ] TTC maps and MVSEC optical flow (Tables 3–5; not in the official code)

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
