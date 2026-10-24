#!/usr/bin/env python3
"""MP4 of the Mg/MgH2 separation seen through the MACE A3 gen-4 descriptor and per-atom energy.

Usage:  python plot_mace_movie.py [NPZ] [--window 5] [--dcut 6.5] [--rc-mgmg 4.1]
                                  [--fps 24] [--stride 1] [--dpi 120]
  NPZ  default: mace_A3g4.npz next to this script; a checkpoint mace_A3g4.partial.npz also works
       (frames 0 .. done-1, for a quick look while compute_mace_descriptors.py is running).

Classification (Mg only): centred moving average of d_hcp, d_rutile over --window frames, then
  hcp if d_hcp < dcut and d_hcp < d_rutile; rutile if d_rutile < dcut and d_rutile < d_hcp; else other.
  Default dcut 6.5 = half the distance between the hcp and rutile reference descriptors (12.9).
Nucleus: largest connected cluster of rutile-labelled Mg in the Mg-Mg graph (edges < --rc-mgmg).

Frame layout: (top) atoms, view along z, Mg coloured by MACE class, H small;
(middle) the same atoms coloured by per-atom energy (Mg and H on separate colour scales);
(bottom) class fractions, energy per atom and largest rutile cluster vs step, with a cursor.
Writes mace_A3g4.mp4 (or mace_A3g4_partial.mp4) next to NPZ.
Needs: numpy, scipy, matplotlib, imageio-ffmpeg -> conda env PEM.
"""
import argparse
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
LABEL = {"hcp": "Mg, hcp-like", "rutile": "Mg, rutile-like", "other": "Mg, other"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("npz", nargs="?", default=os.path.join(HERE, "mace_A3g4.npz"))
    p.add_argument("--extxyz", default=os.path.join(ROOT, "01_structure_phase_transition", "soap.2.md.064.diff.extxyz"))
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--dcut", type=float, default=6.5)
    p.add_argument("--rc-mgmg", type=float, default=4.1)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--dpi", type=int, default=120)
    return p.parse_args()


def main():
    a = parse_args()
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        print("imageio-ffmpeg not found; relying on ffmpeg in PATH")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    r = np.load(a.npz)
    partial = "done" in r
    F = int(r["done"]) if partial else len(r["frames"])
    fr = np.arange(F) if partial else r["frames"]
    species, pos, _, cells, ts = read_extxyz(a.extxyz)
    pos, cells, ts = pos[fr], cells[fr], ts[fr]
    n = len(species)
    mg, hy = species == "Mg", species == "H"
    L = np.einsum("fii->fi", cells)
    pos = pos - L[:, None] * np.floor(pos / L[:, None])
    print(f"{F} frames{' (partial checkpoint)' if partial else ''}, steps {ts[0]}..{ts[-1]}")

    dh = moving_average(r["d_hcp"][:F], a.window)
    dr = moving_average(r["d_rutile"][:F], a.window)
    cls = classify(dh, dr, a.dcut)
    frac = np.stack([(cls == c).mean(1) for c in range(3)], 1)
    ne = r["node_energy"][:F]
    E = r["energy"][:F] / n
    nmax = np.zeros(F, int)
    for k in range(F):
        rut = np.where(cls[k] == 1)[0]
        if len(rut):
            p = pos[k, mg][rut]
            d = p[:, None] - p[None]
            d -= L[k] * np.round(d / L[k])
            adj = ((d ** 2).sum(-1) < a.rc_mgmg ** 2) & ~np.eye(len(rut), dtype=bool)
            _, comp = connected_components(sps.csr_matrix(adj), directed=False)
            nmax[k] = np.bincount(comp).max()
    late = slice(int(0.8 * F), F)
    print(f"last 20 %: hcp {frac[late, 0].mean():.3f} rutile {frac[late, 1].mean():.3f} "
          f"other {frac[late, 2].mean():.3f}; largest rutile cluster {nmax[late].mean():.0f} Mg")
    for th in (10, 50, 100):
        hit = np.nonzero(nmax >= th)[0]
        print(f"MACE-rutile cluster first reaches {th:3d} Mg at step {ts[hit[0]] if len(hit) else '-'}")

    tk = ts / 1e3
    rad = np.where(mg, 0.65, 0.30)
    emg = np.percentile(ne[:, mg], [2, 98]); eh = np.percentile(ne[:, hy], [2, 98])
    cm_mg, cm_h = plt.get_cmap("viridis"), plt.get_cmap("magma")

    fig = plt.figure(figsize=(12, 9.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.2, 2.2, 1], hspace=0.45, wspace=0.35)
    axa, axe = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])
    axf, axE, axn = (fig.add_subplot(gs[2, i]) for i in range(3))
    for ax in (axa, axe):
        ax.set_xlim(-1.5, L[:, 0].max() + 1.5); ax.set_ylim(-1.5, L[:, 1].max() + 1.5); ax.set_aspect("equal")
        ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
    handles = [Line2D([], [], marker="o", ls="", ms=9, mfc=COL[c], mec="#333333", label=LABEL[c]) for c in CLASSES]
    handles.append(Line2D([], [], marker="o", ls="", ms=5, mfc=COL["H"], mec="#333333", label="H"))
    axa.legend(handles=handles, frameon=False, fontsize=9.5, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    axe.set_title("per-atom energy (MACE node energy): Mg viridis, H magma; dark = low", fontsize=9.5, loc="left")
    for ax in (axf, axE, axn):
        ax.set_xlim(tk[0], tk[-1]); ax.set_xlabel("MD step (×10$^3$)")
    for c, nm in enumerate(CLASSES):
        axf.plot(tk, frac[:, c], color=COL[nm], lw=1.5)
    axf.set_ylim(0, 1); axf.set_ylabel("fraction of Mg")
    axE.plot(tk, (E - E[late].mean()) * 1e3, color="#222222", lw=1.2)
    axE.set_ylabel("E/atom − late mean (meV)")
    axn.plot(tk, nmax, color=COL["rutile"], lw=1.5)
    axn.set_ylim(0, mg.sum() * 0.6); axn.set_ylabel("largest rutile cluster (Mg)")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.06)
    fig.canvas.draw()
    curs = [ax.axvline(tk[0], color="#000000", lw=1) for ax in (axf, axE, axn)]
    boxes = [ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1.2, ec="#000000")) for ax in (axa, axe)]
    sca = axa.scatter([], [], edgecolors="#333333", linewidths=0.3)
    sce = axe.scatter([], [], edgecolors="#333333", linewidths=0.3)
    head = fig.text(0.07, 0.99, "", fontsize=13, va="top")
    fig.text(0.07, 0.962, f"MACE A3 gen-4 invariant descriptor, distance to hcp / rutile reference "
             f"(window {a.window} frames, cutoff {a.dcut:g}); view along z, full depth",
             fontsize=9, va="top", color="#555555")
    bb = axa.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    size = (2 * rad * bb.width * 72 / np.diff(axa.get_xlim())[0]) ** 2

    out = os.path.join(os.path.dirname(os.path.abspath(a.npz)), "mace_A3g4" + ("_partial" if partial else "") + ".mp4")
    writer = FFMpegWriter(fps=a.fps, bitrate=6000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, out, dpi=a.dpi):
        for j, k in enumerate(range(0, F, a.stride)):
            o = np.argsort(pos[k, :, 2])                                  # painter's order back-to-front
            col = np.empty(n, object)
            col[mg] = [COL[CLASSES[c]] for c in cls[k]]
            col[hy] = COL["H"]
            ecol = np.zeros((n, 4))
            ecol[mg] = cm_mg(np.clip((ne[k, mg] - emg[0]) / (emg[1] - emg[0]), 0, 1))
            ecol[hy] = cm_h(np.clip((ne[k, hy] - eh[0]) / (eh[1] - eh[0]), 0, 1))
            for sc, c in ((sca, col[o].tolist()), (sce, ecol[o])):
                sc.set_offsets(pos[k, o, :2]); sc.set_facecolors(c); sc.set_sizes(size[o])
            for b in boxes:
                b.set_width(L[k, 0]); b.set_height(L[k, 1])
            for cu in curs:
                cu.set_xdata([tk[k], tk[k]])
            head.set_text(f"MgH  ·  step {ts[k]:>7d}  ·  hcp {frac[k,0]:.2f}  rutile {frac[k,1]:.2f}  "
                          f"other {frac[k,2]:.2f}  ·  largest rutile cluster {nmax[k]}")
            writer.grab_frame()
            if j % 100 == 0:
                print(f"  frame {k}/{F}")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
