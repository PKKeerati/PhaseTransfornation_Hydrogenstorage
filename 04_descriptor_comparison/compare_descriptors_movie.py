#!/usr/bin/env python3
"""Side-by-side MP4: the same trajectory classified with three local descriptors (CN, SOAP, MACE).

Usage:  python compare_descriptors_movie.py [--window 5] [--rc-mgh 2.6] [--soap-dcut 600]
                                            [--mace-dcut 6.5] [--rc-mgmg 4.1] [--fps 24] [--stride 1]
                                            [--mace-npz ../03_mace_descriptors/mace_A3g4.npz]
Inputs: ../01_structure_phase_transition/soap.2.md.064.diff.extxyz (positions, SOAP distances) and
the MACE npz (a .partial.npz checkpoint also works: only its first `done` frames are used).

Same rule for every descriptor, Mg only, after a centred moving average over --window frames:
  CN    Mg-H coordination number (H within --rc-mgh): hcp if CN <= 1, rutile if CN >= 4, else other
  SOAP  stored SOAP distances (01): nearer reference, other if both > --soap-dcut (600, as in 01)
  MACE  A3 gen-4 invariant-descriptor distances (03): nearer reference, other if both > --mace-dcut
        (6.5 = half the hcp-rutile reference distance 12.9)
Nucleus: largest connected cluster of rutile-labelled Mg in the Mg-Mg graph (edges < --rc-mgmg).

Frame: three atom panels (view along z), one per descriptor; below, the rutile fraction and the
largest rutile cluster vs step for all three, with a cursor.
Writes in this folder: compare_descriptors.mp4 (or _partial.mp4), compare_descriptors.csv,
compare_descriptors.png (fractions and clusters, static).
Needs: numpy, scipy, matplotlib, imageio-ffmpeg -> conda env PEM.
"""
import argparse
import csv
import os
import sys

import numpy as np
import scipy.sparse as sps
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "01_structure_phase_transition"))
sys.dont_write_bytecode = True     # keep 01 free of __pycache__
from plot_phase_movie import read_extxyz, moving_average, classify, COL  # noqa: E402

CLASSES = ["hcp", "rutile", "other"]
DESC = ["CN", "SOAP", "MACE"]
LS = {"CN": "-", "SOAP": ":", "MACE": "--"}
SHADE = {"CN": "#222222", "SOAP": "#AA3377", "MACE": "#228833"}   # Tol bright: black, purple, green


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--extxyz", default=os.path.join(ROOT, "01_structure_phase_transition", "soap.2.md.064.diff.extxyz"))
    p.add_argument("--mace-npz", default=os.path.join(ROOT, "03_mace_descriptors", "mace_A3g4.npz"))
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--rc-mgh", type=float, default=2.6)
    p.add_argument("--soap-dcut", type=float, default=600.0)
    p.add_argument("--mace-dcut", type=float, default=6.5)
    p.add_argument("--rc-mgmg", type=float, default=4.1)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--dpi", type=int, default=110)
    return p.parse_args()


def largest_cluster(p, L, rc):
    if len(p) == 0:
        return 0
    d = p[:, None] - p[None]
    d -= L * np.round(d / L)
    adj = ((d ** 2).sum(-1) < rc ** 2) & ~np.eye(len(p), dtype=bool)
    _, comp = connected_components(sps.csr_matrix(adj), directed=False)
    return np.bincount(comp).max()


def main():
    a = parse_args()
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        print("imageio-ffmpeg not found; relying on ffmpeg in PATH")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    md = np.load(a.mace_npz)
    partial = "done" in md
    species, pos, dist, cells, ts = read_extxyz(a.extxyz)
    F = int(md["done"]) if partial else len(pos)
    if not partial and not np.array_equal(md["timestep"], ts):
        sys.exit(f"{a.mace_npz} does not cover every frame (run compute_mace_descriptors.py without --stride)")
    pos, dist, cells, ts = pos[:F], dist[:F], cells[:F], ts[:F]
    n = len(species)
    mg, hy = species == "Mg", species == "H"
    L = np.einsum("fii->fi", cells)
    pos = pos - L[:, None] * np.floor(pos / L[:, None])
    print(f"{F} frames{' (partial MACE checkpoint)' if partial else ''}, steps {ts[0]}..{ts[-1]}")

    cn = np.zeros((F, mg.sum()))
    for k in range(F):
        d = pos[k, mg][:, None] - pos[k, hy][None]
        d -= L[k] * np.round(d / L[k])
        cn[k] = ((d ** 2).sum(-1) < a.rc_mgh ** 2).sum(1)
    cn = moving_average(cn, a.window)
    cls = {"CN": np.where(cn >= 4, 1, np.where(cn <= 1, 0, 2)),
           "SOAP": classify(moving_average(dist[:, mg, 0], a.window), moving_average(dist[:, mg, 1], a.window),
                            a.soap_dcut),
           "MACE": classify(moving_average(md["d_hcp"][:F], a.window), moving_average(md["d_rutile"][:F], a.window),
                            a.mace_dcut)}
    frac = {k: np.stack([(c == i).mean(1) for i in range(3)], 1) for k, c in cls.items()}
    nmax = {k: np.array([largest_cluster(pos[f, mg][c[f] == 1], L[f], a.rc_mgmg) for f in range(F)])
            for k, c in cls.items()}

    late = slice(int(0.8 * F), F)
    for k in DESC:
        f = frac[k][late].mean(0)
        line = f"{k:4s} last 20 %: hcp {f[0]:.3f} rutile {f[1]:.3f} other {f[2]:.3f}; largest rutile cluster " \
               f"{nmax[k][late].mean():.0f};"
        for th in (10, 50, 100):
            hit = np.nonzero(nmax[k] >= th)[0]
            line += f" >= {th}: step {ts[hit[0]] if len(hit) else '-'};"
        print(line)
    for i, x in enumerate(DESC):
        for y in DESC[i + 1:]:
            print(f"per-atom agreement {x}-{y}: all {(cls[x] == cls[y]).mean():.3f}, "
                  f"last 20 % {(cls[x][late] == cls[y][late]).mean():.3f}")

    stem = os.path.join(HERE, "compare_descriptors" + ("_partial" if partial else ""))
    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "timestep"] + [f"{k}_{c}" for k in DESC for c in CLASSES]
                   + [f"{k}_n_max_rutile_cluster" for k in DESC])
        for f in range(F):
            w.writerow([f, ts[f]] + [int((cls[k][f] == i).sum()) for k in DESC for i in range(3)]
                       + [int(nmax[k][f]) for k in DESC])

    tk = ts / 1e3

    def curves(axr, axn):
        for k in DESC:
            axr.plot(tk, frac[k][:, 1], color=SHADE[k], ls=LS[k], lw=1.5, label=k)
            axn.plot(tk, nmax[k], color=SHADE[k], ls=LS[k], lw=1.5, label=k)
        axr.set_ylim(0, 1); axr.set_ylabel("rutile fraction of Mg")
        axn.set_ylim(0, mg.sum() * 0.75); axn.set_ylabel("largest rutile cluster (Mg)")
        for ax in (axr, axn):
            ax.set_xlim(tk[0], tk[-1]); ax.set_xlabel("MD step (×10$^3$)")
            ax.axhline(0.5 if ax is axr else mg.sum() / 2, color="#000000", lw=0.6, alpha=0.4)
        axr.legend(frameon=False, fontsize=9, loc="lower right")

    fig, (axr, axn) = plt.subplots(1, 2, figsize=(11, 3.8))
    curves(axr, axn)
    fig.suptitle(f"Rutile labels from three descriptors (same rule, window {a.window} frames)", fontsize=11,
                 x=0.06, ha="left")
    fig.tight_layout(); fig.savefig(f"{stem}.png", dpi=200); plt.close(fig)

    rad = np.where(mg, 0.65, 0.30)
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(4, 2, height_ratios=[2, 2, 2, 1.3], hspace=0.5, wspace=0.25)
    axs = [fig.add_subplot(gs[i, :]) for i in range(3)]
    axr, axn = fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])
    rule = {"CN": f"H count within {a.rc_mgh:g} Å (≤ 1 / ≥ 4)",
            "SOAP": f"SOAP, cutoff {a.soap_dcut:g}",
            "MACE": f"MACE A3 gen-4, cutoff {a.mace_dcut:g}"}
    titles = []
    for ax, k in zip(axs, DESC):
        ax.set_xlim(-1.5, L[:, 0].max() + 1.5); ax.set_ylim(-1.5, L[:, 1].max() + 1.5); ax.set_aspect("equal")
        ax.set_ylabel("y (Å)")
        titles.append(ax.set_title("", fontsize=10, loc="left"))
    axs[-1].set_xlabel("x (Å)")
    handles = [Line2D([], [], marker="o", ls="", ms=9, mfc=COL[c], mec="#333333", label=f"Mg, {c}") for c in CLASSES]
    handles.append(Line2D([], [], marker="o", ls="", ms=5, mfc=COL["H"], mec="#333333", label="H"))
    fig.legend(handles=handles, frameon=False, fontsize=9.5, ncol=4, loc="upper right", bbox_to_anchor=(0.98, 0.995))
    curves(axr, axn)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.05)
    fig.canvas.draw()
    curs = [ax.axvline(tk[0], color="#000000", lw=1) for ax in (axr, axn)]
    boxes = [ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1.2, ec="#000000")) for ax in axs]
    scs = [ax.scatter([], [], edgecolors="#333333", linewidths=0.3) for ax in axs]
    head = fig.text(0.07, 0.995, "", fontsize=13, va="top")
    fig.text(0.07, 0.968, f"Same trajectory, same rule (centred {a.window}-frame average → nearer reference, "
             f"far from both → other); view along z, full depth", fontsize=9, va="top", color="#555555")
    bb = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    size = (2 * rad * bb.width * 72 / np.diff(axs[0].get_xlim())[0]) ** 2

    out = f"{stem}.mp4"
    writer = FFMpegWriter(fps=a.fps, bitrate=8000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, out, dpi=a.dpi):
        for j, f in enumerate(range(0, F, a.stride)):
            o = np.argsort(pos[f, :, 2])                                  # painter's order back-to-front
            for ax, sc, b, tt, k in zip(axs, scs, boxes, titles, DESC):
                col = np.empty(n, object)
                col[mg] = [COL[CLASSES[c]] for c in cls[k][f]]
                col[hy] = COL["H"]
                sc.set_offsets(pos[f, o, :2]); sc.set_facecolors(col[o].tolist()); sc.set_sizes(size[o])
                b.set_width(L[f, 0]); b.set_height(L[f, 1])
                fr = frac[k][f]
                tt.set_text(f"{k} ({rule[k]}):  hcp {fr[0]:.2f}  rutile {fr[1]:.2f}  other {fr[2]:.2f}  "
                            f"·  largest rutile cluster {nmax[k][f]}")
            for cu in curs:
                cu.set_xdata([tk[f], tk[f]])
            head.set_text(f"MgH  ·  step {ts[f]:>7d}  ·  CN vs SOAP vs MACE")
            writer.grab_frame()
            if j % 100 == 0:
                print(f"  frame {f}/{F}")
    plt.close(fig)
    for o in (out, f"{stem}.png", f"{stem}.csv"):
        print("wrote", o)


if __name__ == "__main__":
    main()
