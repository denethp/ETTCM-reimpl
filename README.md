# ETTCM-reimpl

Unofficial Python + Numba re-implementation of **"Time-to-Contact Map by Joint Estimation of Up-to-Scale Inverse Depth and Global Motion using a Single Event Camera"** (U. M. Nunes, L. U. Perrinet, S.-H. Ieng, ICCV 2023).

The method processes events one at a time and jointly estimates the camera's global motion and a per-event, up-to-scale inverse depth. From that it gets per-event optical flow, a global divergence estimate and a per-pixel time-to-contact (TTC) map, at around 1 µs per event in the official C++.

- Paper: [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Nunes_Time-to-Contact_Map_by_Joint_Estimation_of_Up-to-Scale_Inverse_Depth_and_ICCV_2023_paper.html)
- Official C++ code: [neuromorphic-paris/ETTCM](https://github.com/neuromorphic-paris/ETTCM) (CC BY-NC-SA 4.0, not included in this repo)

## Goals

1. Build and run the official C++ code and reproduce Table 2 (Ventral Landing dataset).
2. Write my own implementation in Python + NumPy (+ Numba for speed) that matches the official code per event.
3. Go beyond the official repo: TTC maps and MVSEC optical-flow evaluation.

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
    synthetic.py    approaching textured plane with exact ground truth
    evaluate.py     REE metric (Eq. 12), VL loader
  scripts/          run_synthetic.py, run_vl.py
  tests/            test_io.py, test_models.py
  cpp/harness.cpp   per-event output of the official C++ core, for comparison
sources/links.txt   paper, code, dataset and background links
```

## Quick start

```bash
cd reimpl
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_io.py
python tests/test_models.py
python scripts/run_synthetic.py --model scaling --duration 0.4
```

Setup details, milestone commands and implementation notes are in [`reimpl/README.md`](reimpl/README.md).

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

- [x] Resources, notes and starter kit
- [ ] Build the official code and run Table 2 on VL
- [ ] M1: Scaling model on synthetic data
- [ ] M2: Translation and 6-DOF models
- [ ] M3: match the official C++ per event
- [ ] M4: reproduce Table 2 on VL
- [ ] M5: TTC maps and MVSEC optical flow

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
