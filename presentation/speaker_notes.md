# Speaker notes: ETTCM reproduction (20 pages: title, 17 slides, thank-you, 1 backup; ~20 min)

Slide numbers are the ones shown bottom right. The title page also shows the outline; the current
section is named at the top right of every slide.

## 1. The problem (~2 min)

**1. Time-to-contact.** TTC = seconds until the camera reaches a point if nothing changes.
τ = Z / ν_z: depth divided by approach speed. Key point: only the *ratio* is needed, so one camera
without metric scale is enough. Used when you must react fast: landing, braking, dodging.

**2. Event cameras.** Each pixel independently reports "brightness went up/down at time t".
Microsecond resolution, no blur, high dynamic range, low power. Catch: there are no frames, so
normal vision algorithms do not apply; methods must work on the stream of events.

## 2. Why this paper is different (~3 min)

**3. The task and prior methods.** Top: old approach = one global TTC for the whole view, assuming
a flat wall facing the camera; this paper = a TTC *map*, one value per pixel, updated with every
single event, on a CPU (the example map is the paper's Fig. 5, shown on slide 13). Table: all
earlier methods give a single global value and assume a planar scene; ECMD is the strongest (8.9 %
error) but needs a GPU and ~51 µs per event. ETTCM: per-pixel, 5.6 % error, ~1 µs per event on a
CPU, about 40× faster than ECMD. Values marked ≈ were read off the paper's figure.

**4. Fig. 1 re-made with our code.** Left: all methods. Right: zoom; our six configurations
(hollow) land next to the paper's (filled): same error, similar runtime.

**5. Three key ideas + use cases.** (1) Estimate one global camera motion and a per-event depth
together. (2) Incremental: tiny neighbourhood, at most 2 optimisation steps per event. (3) Event
collapse fix (explained in methodology). Use cases (bottom row, section 3): drone landing (VL
benchmark), obstacle avoidance (indoor drone, MVSEC), cars (a dedicated Driving model, MVSEC
outdoor), embedded robots (CPU, and the supplementary shows a further 9× speed-up by sampling events).

## 4. Methodology (~4 min)

**6. Equations.** Every event's image velocity is B_i θ: depth-dependent translation part plus
rotation part. Divergence μ = ρ ν_z, TTC = 1/μ. Depth and speed are only known up to a common
scale, the method fixes |ν| ≤ 1, and the scale cancels in TTC. Four motion models: Scaling,
Translation, Driving (forward speed + yaw, |ν_z| = 1), 6-DOF.

**7. Pipeline (walk through the 5 boxes).** New event → look at the small window of recent
events, shifted back along where the edge came from → initialise this event's depth from the
neighbours, weighted by how recent they are → at most 2 Gauss–Newton steps on motion + log-depth,
with priors so the global motion changes smoothly → output flow, divergence, TTC. Motion carries
over to the next event. ~1–2 µs per event, enough for 400k events/s.

**8. Dispersion minimisation.** Classic approach warps all events to one reference time; with
a scaling motion the optimiser can "win" by shrinking everything to a point (event collapse).
Here each neighbour is only moved to the *current* event's time: small time gaps, and a diverging
motion makes residuals grow, so collapse is penalised automatically.

## 5. Experiments (~7 min)

**9. Re-implementation, datasets, metrics.** Python + Numba, matches the official C++ event by
event (~1e-7). Official code only has Table 2; the whole MVSEC evaluation is ours. Settings from
paper Table 1 (top right). Datasets: VL = drone landing, GT divergence; MVSEC = drone indoors + car
outdoors, GT depth, flow and velocity, 266 s scored (paper says ~265 s); ECD slider_depth for the
depth figure. Metrics: relative error of divergence; flow end-point error, outliers > 3 px,
relative error, angle error.

**10. Tables 2 and 6.** Left, Table 2: green = matches, every average within 0.08, real-time like
the paper. Right, supplementary Table 6 / Fig. 8: process only 1 in r² events at a reduced
resolution. At r = 3 you process 11× fewer events and lose almost nothing. Our bars sit on the
paper's ticks: 23 of 24 averages within 0.08.

**11. Tables 4 and 5 (optical flow).** Table 4: Driving and 6-DOF indoors match to 0.05 px.
Translation slightly worse; we swept the parameters and the best is 1.05 vs the paper's 0.94, so it
is not a tuning issue. Table 5 (last two columns): angle error (direction) matches everywhere.
Relative error for Driving is far off although its AEE matched: same estimates, different numbers
→ the difference is in the scoring.

**12. Table 3 (the hard one).** Plain mean is 2–3× the paper. Reason: the drone often hovers,
true divergence ≈ 0, dividing by ~0 explodes. With a median, Translation (the paper's best model)
matches on all three sequences. The paper never states its rule.

**13–14. Figures.** Paper on top / left, ours below / right. Same scene, same colours; the
estimated maps follow the ground truth. (Fig. 6: the paper used a different colour scale and dense
GT; the frame times are not given, we picked moments of forward flight.)

## 6. What we did and how much (~3 min)

**15. What we built + problems.** Left: the checklist; everything in the paper and supplementary
was run. Right: highlight two problems: the paper's 6-DOF is an undocumented model inside the C++,
and we discovered that MVSEC's own ground-truth flow implies only 55–65 % of the speed in its
odometry.

**16. How much.** By paper item ≈ 85 % (8.5 of 10 items). Number by number 65–77 % depending on
tolerance (Table 3 counted with the paper's literal formula). The method itself: 100 %.

**17. Why the rest differs + conclusions.** Left: three evaluation details not in the paper:
near-zero GT scoring, which 6-DOF Jacobian (the C++ disagrees with the paper's own supplementary
Eq. 18; we tested both), how the outdoor sequence was run. Questions sent to the authors. Right:
conclusions and the lesson: without the evaluation script a paper cannot be fully reproduced, even
with the method code public.

**18. Thank you / questions.**

## Likely questions

- *Why Python, not the C++?* To understand every step; then verified it equals the C++ event by event.
- *Is 6-DOF wrong in your code?* It equals the official code exactly (VL Table 2 matches). On MVSEC
  neither the official nor the corrected Jacobian gives the paper's numbers (backup slide).
- *Why not just use the median everywhere?* The paper's Eq. 14 is a mean; we report both and say so.
- *Runtime differs from the paper?* Different hardware and language; both ~1–2 µs/event, real time.
- *Which sequence/time for Fig. 6?* Not stated in the paper; we chose strong forward motion.
