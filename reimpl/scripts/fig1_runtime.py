"""Fig. 1 of the paper: runtime per event vs. accuracy on the Ventral Landing (VL) benchmark.

    python scripts/fig1_runtime.py --csv ../results_python_vl.csv --out ../figures/fig1_runtime.png

Two panels. Left: the paper's Fig. 1 (all methods, log runtime axis) with the proposed method's
corner boxed. Right: that corner zoomed in, with the paper's six configurations (Table 2: model x
neighbourhood size s, filled markers) next to ours (hollow markers, same colour).

Our points come from table2_vl.py's CSV (model, s, seq, REE %, us/event, events; if a
(model, s, seq) appears more than once, the last row is used), averaged over the 8 sequences.
The other methods' points are not given as numbers in the paper (only ECMD, in Table 2); they
were read off the paper's Fig. 1 (log x axis), checked against ECMD's exact values
(read 51 us / 8.96 % vs Table 2's 50.90 / 8.92), so they are good to a few percent.
"""
import argparse
import csv
import os

# (name, us/event, REE %, hardware) as in the paper's Fig. 1
PAPER_POINTS = [
    ("PF [36]", 1.92, 40.6, "CPU"),
    ("E-RAFT [17]", 2.25, 42.2, "GPU"),
    ("CT [43]", 314.0, 58.2, "CPU"),
    ("SOFAS [45]", 486.0, 34.8, "CPU"),
    ("ECMD [28]", 50.90, 8.92, "GPU"),      # exact, Table 2
]
# Paper Table 2 averages: (model, s) -> (us/event, REE %)
PAPER_OURS = {
    ("scaling", 2): (0.78, 8.29), ("translation", 2): (1.15, 5.64), ("six_dof", 2): (1.46, 10.11),
    ("scaling", 3): (1.29, 15.08), ("translation", 3): (1.74, 9.31), ("six_dof", 3): (2.36, 6.42),
}
NAMES = {"scaling": "Scaling", "translation": "Translation", "six_dof": "6-DOF"}
COLORS = {"scaling": "#7E57C2", "translation": "#17A398", "six_dof": "#E4572E"}
HW = {"CPU": "#0072BD", "GPU": "#D95319"}


def load(path):
    last = {}
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0] == "model":
                continue
            model, s, seq, ree, us = row[0], int(row[1]), row[2], float(row[3]), float(row[4])
            last[(model, s, seq)] = (ree, us)
    groups = {}
    for (model, s, _), v in last.items():
        groups.setdefault((model, s), []).append(v)
    return {k: (sum(u for _, u in v) / len(v), sum(r for r, _ in v) / len(v), len(v))
            for k, v in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../results_python_vl.csv")
    ap.add_argument("--out", default="../figures/fig1_runtime.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch, Rectangle

    ours = load(args.csv)
    print(f"{'model':12s} {'s':>2s} {'REE %':>7s} {'us/event':>9s}   paper REE / us")
    for k in sorted(ours):
        us, ree, _ = ours[k]
        pu, pr = PAPER_OURS[k]
        print(f"{k[0]:12s} {k[1]:2d} {ree:7.2f} {us:9.2f}   {pr:6.2f} / {pu:.2f}")

    fig, (ax, az) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})

    # ---------------- left: the paper's Fig. 1
    offset = {"PF [36]": (-10, -13), "CT [43]": (-12, -14), "SOFAS [45]": (-30, -14)}
    for name, us, ree, hw in PAPER_POINTS:
        ax.scatter(us, ree, color=HW[hw], s=45, zorder=3)
        ax.annotate(name, (us, ree), textcoords="offset points", xytext=offset.get(name, (6, 4)),
                    fontsize=9)
    for (m, s), (us, ree) in PAPER_OURS.items():
        ax.scatter(us, ree, color=COLORS[m], s=18, zorder=4)
    ax.annotate("Proposed\n(6 configurations)", (1.6, 16), textcoords="offset points",
                xytext=(18, 18), fontsize=9, weight="bold",
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8))
    box = (0.62, 3.0, 2.7 - 0.62, 17.5 - 3.0)       # x0, y0, w, h of the zoomed corner
    ax.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, ls="--", lw=1.0, ec="0.35"))
    ax.scatter([], [], color=HW["CPU"], label="CPU")
    ax.scatter([], [], color=HW["GPU"], label="GPU")
    ax.legend(fontsize=8, loc="upper center", ncol=2, frameon=False)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 1000)
    ax.set_ylim(0, 62)
    ax.set_xlabel("Runtime per event (µs, log scale)")
    ax.set_ylabel("REE (%)  (lower is better)")
    ax.set_title("Paper's Fig. 1: all methods", fontsize=10)

    # ---------------- right: zoom on the proposed method, paper vs ours
    label_off = {("scaling", 2): (-8, 9), ("translation", 2): (10, -4), ("six_dof", 2): (10, -4),
                 ("scaling", 3): (12, -4), ("translation", 3): (10, -4), ("six_dof", 3): (-25, 13)}
    for k, (pu, pr) in PAPER_OURS.items():
        m, s = k
        c = COLORS[m]
        if k in ours:
            ou, orr, _ = ours[k]
            az.plot([pu, ou], [pr, orr], color=c, lw=1.0, alpha=0.6, zorder=2)
            az.scatter(ou, orr, s=150, facecolor="none", edgecolor=c, linewidth=1.8, zorder=3)
        az.scatter(pu, pr, s=50, color=c, zorder=4)
        right = max(pu, ours[k][0]) if k in ours else pu
        az.annotate(f"{NAMES[m]}, s={s}", (right, pr), textcoords="offset points",
                    xytext=label_off[k], fontsize=9, color=c, va="center",
                    ha="right" if label_off[k][0] < 0 else "left")
    az.scatter([], [], s=50, color="0.3", label="paper (C++)")
    az.scatter([], [], s=150, facecolor="none", edgecolor="0.3", linewidth=1.8,
               label="ours (Python + Numba)")
    az.legend(fontsize=8, loc="upper right", frameon=False)
    az.set_xlim(box[0], box[0] + box[2])
    az.set_ylim(box[1], box[1] + box[3])
    az.set_xlabel("Runtime per event (µs)")
    az.set_title("Zoom: proposed method, paper vs ours", fontsize=10)
    az.grid(alpha=0.25)
    for sp in az.spines.values():
        sp.set_linestyle("--")
        sp.set_color("0.35")

    # connectors from the box to the zoom panel
    for yb in (box[1], box[1] + box[3]):
        fig.add_artist(ConnectionPatch(xyA=(box[0] + box[2], yb), coordsA=ax.transData,
                                       xyB=(box[0], yb), coordsB=az.transData,
                                       color="0.55", lw=0.8, ls=":"))

    fig.suptitle("Runtime vs. accuracy on the VL benchmark (average over 8 sequences)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print("saved", args.out)


if __name__ == "__main__":
    main()
