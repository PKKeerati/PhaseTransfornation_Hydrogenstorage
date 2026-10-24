#!/usr/bin/env python3
"""MP4 of the Mg (hcp) / MgH2 (rutile) phase transformation from a SOAP-tagged MD extxyz.

Usage:  python plot_phase_movie.py [EXTXYZ] [--window 5] [--dcut 600] [--cn-cut 2.3]
                                   [--slab 0] [--no-h] [--fps 24] [--stride 1]
  EXTXYZ  extxyz with per-atom dist_hcp / dist_rutile (SOAP distances to the hcp-Mg and
          rutile-MgH2 references; finite for Mg, NaN for H).
          Default: soap.2.md.064.diff.extxyz next to this script.

Classification (Mg only, existing distances, no SOAP recomputation):
  1. time smoothing: centred moving average of dist_hcp and dist_rutile per atom over
     --window saved frames (window shrinks at the trajectory ends);
  2. hcp     if d_hcp    < dcut and d_hcp    < d_rutile
     rutile  if d_rutile < dcut and d_rutile < d_hcp
     other   otherwise (far from both references: disordered / interface).
  The absolute cutoff is what separates "other"; the old relative score
  (d_hcp - d_rut)/(d_hcp + d_rut) forced every atom into one of the two phases.
Cross-check: Mg-H coordination number (CN, cutoff --cn-cut) per Mg; rutile Mg should have
CN ~ 6 and hcp Mg CN ~ 0. The agreement is printed and written to the CSV.

Writes (next to EXTXYZ, prefix = file stem):
  <stem>.phase.mp4            atoms (orthographic view along z) + phase fractions vs time
  <stem>.phase_fraction.png   static phase-fraction figure
  <stem>.phase_snapshots.png  four snapshots (first, 1/3, 2/3, last)
  <stem>.phase_classes.csv    per-frame counts (SOAP classes, raw score, CN classes)
Needs: numpy, matplotlib, imageio-ffmpeg (bundled ffmpeg)  -> conda env PEM.
"""
import argparse
import csv
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D

# ---- palette: Paul Tol 'bright' from repo colour.md ----
COL = {"hcp": "#4477AA",      # blue  : hcp Mg
       "rutile": "#EE6677",   # red   : rutile MgH2 (the focus)
       "other": "#BBBBBB",    # grey  : other / unassigned bucket
       "H": "#CCBB44"}        # yellow: H
LABEL = {"hcp": "Mg in hcp Mg", "rutile": "Mg in rutile MgH$_2$", "other": "Mg, other"}
CLASSES = ["hcp", "rutile", "other"]
R_MG, R_H = 0.65, 0.30        # drawn radii (Å), visual only

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("extxyz", nargs="?", default=os.path.join(HERE, "soap.2.md.064.diff.extxyz"))
    p.add_argument("--window", type=int, default=5, help="time-smoothing window, saved frames (odd)")
    p.add_argument("--dcut", type=float, default=600.0, help="absolute SOAP-distance cutoff")
    p.add_argument("--cn-cut", type=float, default=2.3, help="Mg-H cutoff for CN (Å)")
    p.add_argument("--slab", type=float, default=0.0,
                   help="if > 0, draw only atoms within this thickness (Å) around the cell centre in z")
    p.add_argument("--no-h", action="store_true", help="do not draw H atoms")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--stride", type=int, default=1, help="use every n-th frame in the movie")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def read_extxyz(fn):
    """Return species (n,), positions (F,n,3), dist (F,n,2), cells (F,3,3), timesteps (F,)."""
    lines = open(fn).read().splitlines()
    pos, dist, cells, ts = [], [], [], []
    i = 0
    while i < len(lines):
        n = int(lines[i]); h = lines[i + 1]
        props = re.search(r"Properties=(\S+)", h).group(1).split(":")
        names = props[0::3]
        assert names[:4] == ["species", "pos", "dist_hcp", "dist_rutile"], props
        cells.append(np.array(re.search(r'Lattice="([^"]+)"', h).group(1).split(), float).reshape(3, 3))
        ts.append(int(re.search(r"timestep=(\d+)", h).group(1)))
        rows = [l.split() for l in lines[i + 2:i + 2 + n]]
        if not pos:
            species = np.array([r[0] for r in rows])
        a = np.array([r[1:6] for r in rows], float)
        pos.append(a[:, :3]); dist.append(a[:, 3:5])
        i += n + 2
    return species, np.array(pos), np.array(dist), np.array(cells), np.array(ts)


def moving_average(x, w):
    """Centred moving average along axis 0; the window shrinks at both ends."""
    if w <= 1:
        return x.copy()
    h = w // 2
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    F = x.shape[0]
    lo = np.clip(np.arange(F) - h, 0, F); hi = np.clip(np.arange(F) + h + 1, 0, F)
    return (c[hi] - c[lo]) / (hi - lo).reshape((-1,) + (1,) * (x.ndim - 1))


def classify(dh, dr, dcut):
    """0 = hcp, 1 = rutile, 2 = other."""
    cls = np.full(dh.shape, 2, int)
    cls[(dh < dcut) & (dh < dr)] = 0
    cls[(dr < dcut) & (dr < dh)] = 1
    return cls


def mg_h_cn(pos, cells, mg, hy, rc):
    """Mg-H coordination number per Mg per frame (orthorhombic minimum image)."""
    cn = np.zeros((len(pos), mg.sum()), int)
    for k in range(len(pos)):
        L = np.diag(cells[k])
        d = pos[k, mg][:, None] - pos[k, hy][None]
        d -= L * np.round(d / L)
        cn[k] = ((d ** 2).sum(-1) < rc ** 2).sum(1)
    return cn


def main():
    a = parse_args()
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        print("imageio-ffmpeg not found; relying on ffmpeg in PATH")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    stem = os.path.splitext(a.extxyz)[0]
    print(f"reading {a.extxyz}")
    species, pos, dist, cells, ts = read_extxyz(a.extxyz)
    F, n = pos.shape[:2]
    if np.abs(cells - np.einsum("fii->fi", cells)[:, :, None] * np.eye(3)).max() > 1e-8:
        sys.exit("non-orthorhombic cell: projection/wrapping here assumes an orthorhombic box")
    mg, hy = species == "Mg", species == "H"
    print(f"{F} frames, {n} atoms (Mg {mg.sum()}, H {hy.sum()}), timestep {ts[0]}..{ts[-1]}")

    # wrap into the cell
    L = np.einsum("fii->fi", cells)[:, None, :]
    pos = pos - L * np.floor(pos / L)

    # ---- classification ----
    dh_raw, dr_raw = dist[:, mg, 0], dist[:, mg, 1]
    dh, dr = moving_average(dh_raw, a.window), moving_average(dr_raw, a.window)
    cls = classify(dh, dr, a.dcut)
    cls_raw = classify(dh_raw, dr_raw, a.dcut)
    frac = np.stack([(cls == c).mean(1) for c in range(3)], 1)          # (F,3)
    score_rut = ((dh_raw - dr_raw) / (dh_raw + dr_raw) > 0).mean(1)      # old relative score

    sw_raw = (np.diff(cls_raw, axis=0) != 0).mean()
    sw = (np.diff(cls, axis=0) != 0).mean()
    print(f"class changes per Mg per saved frame: raw {sw_raw:.3f} -> smoothed (w={a.window}) {sw:.3f}")

    cn = mg_h_cn(pos, cells, mg, hy, a.cn_cut)
    cn_cls = np.where(cn >= 4, 1, np.where(cn <= 1, 0, 2))
    agree = (cn_cls == cls).mean()
    late = slice(int(0.8 * F), F)
    print(f"SOAP class vs CN class (hcp: CN<=1, rutile: CN>=4, other: 2-3) agreement: "
          f"all frames {agree:.3f}, last 20% {(cn_cls[late] == cls[late]).mean():.3f}")
    for c, nm in enumerate(CLASSES):
        m = cls[late] == c
        if m.any():
            print(f"  last 20%  {nm:6s}: {m.mean():.3f} of Mg, mean CN {cn[late][m].mean():.2f}")
    print(f"fractions frame 0  : hcp {frac[0,0]:.3f} rutile {frac[0,1]:.3f} other {frac[0,2]:.3f}")
    print(f"fractions last 20% : hcp {frac[late,0].mean():.3f} rutile {frac[late,1].mean():.3f} "
          f"other {frac[late,2].mean():.3f}")

    with open(f"{stem}.phase_classes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "timestep", "n_hcp", "n_rutile", "n_other", "frac_rutile_relscore",
                    "n_cn_le1", "n_cn_ge4", "n_cn_2to3"])
        for k in range(F):
            w.writerow([k, ts[k], *np.bincount(cls[k], minlength=3), f"{score_rut[k]:.4f}",
                        (cn_cls[k] == 0).sum(), (cn_cls[k] == 1).sum(), (cn_cls[k] == 2).sum()])

    tk = ts / 1e3
    handles = [Line2D([], [], marker="o", ls="", ms=9, mfc=COL[c], mec="#333333", label=LABEL[c])
               for c in CLASSES]
    if not a.no_h:
        handles.append(Line2D([], [], marker="o", ls="", ms=5, mfc=COL["H"], mec="#333333", label="H"))

    def fraction_axes(ax):
        for c, nm in enumerate(CLASSES):
            ax.plot(tk, frac[:, c], color=COL[nm], lw=1.8, label=LABEL[nm])
        ax.plot(tk, score_rut, color=COL["rutile"], lw=1.0, ls=":",
                label="rutile, rel. score > 0\n(no cutoff)")
        ax.set_xlim(tk[0], tk[-1]); ax.set_ylim(0, 1)
        ax.set_xlabel("MD step (×10$^3$)"); ax.set_ylabel("Fraction of Mg")

    # static fraction figure
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    fraction_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title(f"SOAP phase fractions (window {a.window} frames, cutoff {a.dcut:g})", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{stem}.phase_fraction.png", dpi=200); plt.close(fig)

    # ---- atom drawing (orthographic view along z, painter's order back-to-front) ----
    xmax = float(cells[:, 0, 0].max()) + 1.5
    ymax = float(cells[:, 1, 1].max()) + 1.5

    def frame_artists(k):
        p = pos[k]
        keep = np.ones(n, bool) if not a.no_h else mg.copy()
        if a.slab > 0:
            keep &= np.abs(p[:, 2] - cells[k, 2, 2] / 2) < a.slab / 2
        col = np.empty(n, object); rad = np.where(mg, R_MG, R_H)
        col[mg] = [COL[CLASSES[c]] for c in cls[k]]
        col[hy] = COL["H"]
        idx = np.where(keep)[0]
        idx = idx[np.argsort(p[idx, 2])]
        return p[idx, :2], col[idx].tolist(), rad[idx]

    def size_pts2(ax, rad):
        fig = ax.figure
        bb = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        pts_per_A = bb.width * 72 / (ax.get_xlim()[1] - ax.get_xlim()[0])
        return (2 * rad * pts_per_A) ** 2

    def atom_axes(ax):
        ax.set_xlim(-1.5, xmax); ax.set_ylim(-1.5, ymax); ax.set_aspect("equal")
        ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
        ax.spines[["top", "right"]].set_visible(False)

    view = "view along z" + (f", slab {a.slab:g} Å" if a.slab > 0 else ", full depth")

    # snapshots
    snaps = [0, F // 3, 2 * F // 3, F - 1]
    fig, axs = plt.subplots(len(snaps), 1, figsize=(9, 2.9 * len(snaps)))
    for ax, k in zip(axs, snaps):
        atom_axes(ax)
        xy, c, r = frame_artists(k)
        ax.add_patch(plt.Rectangle((0, 0), cells[k, 0, 0], cells[k, 1, 1], fill=False, lw=1, ec="#000000"))
        fig.canvas.draw()
        ax.scatter(xy[:, 0], xy[:, 1], s=size_pts2(ax, r), c=c, edgecolors="#333333", linewidths=0.3)
        ax.set_title(f"step {ts[k]}   hcp {frac[k,0]:.2f}  rutile {frac[k,1]:.2f}  other {frac[k,2]:.2f}",
                     fontsize=10, loc="left")
    axs[0].legend(handles=handles, frameon=False, fontsize=9, ncol=len(handles),
                  loc="lower left", bbox_to_anchor=(0, 1.12))
    fig.tight_layout(); fig.savefig(f"{stem}.phase_snapshots.png", dpi=150); plt.close(fig)

    # movie
    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.6, 1], hspace=0.35)
    axa, axf = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    atom_axes(axa)
    fraction_axes(axf)
    axf.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    axa.legend(handles=handles, frameon=False, fontsize=10, ncol=len(handles),
               loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.subplots_adjust(left=0.06, right=0.82, top=0.88, bottom=0.08)
    fig.canvas.draw()
    cursor = axf.axvline(tk[0], color="#000000", lw=1)
    box = axa.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1.2, ec="#000000"))
    sc = axa.scatter([], [], edgecolors="#333333", linewidths=0.3)
    title = fig.text(0.06, 0.985, "", fontsize=13, va="top")
    fig.text(0.06, 0.95, f"Mg classified by SOAP distance (time window {a.window} frames, "
             f"cutoff {a.dcut:g}); {view}", fontsize=9, va="top", color="#555555")

    out = f"{stem}.phase.mp4"
    frames = range(0, F, a.stride)
    writer = FFMpegWriter(fps=a.fps, bitrate=6000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, out, dpi=a.dpi):
        for j, k in enumerate(frames):
            xy, c, r = frame_artists(k)
            sc.set_offsets(xy); sc.set_facecolors(c); sc.set_sizes(size_pts2(axa, r))
            box.set_width(cells[k, 0, 0]); box.set_height(cells[k, 1, 1])
            cursor.set_xdata([tk[k], tk[k]])
            title.set_text(f"MgH  ·  step {ts[k]:>7d}  ·  hcp {frac[k,0]:.2f}   "
                           f"rutile {frac[k,1]:.2f}   other {frac[k,2]:.2f}")
            writer.grab_frame()
            if j % 100 == 0:
                print(f"  frame {k}/{F}")
    plt.close(fig)
    for o in (out, f"{stem}.phase_fraction.png", f"{stem}.phase_snapshots.png", f"{stem}.phase_classes.csv"):
        print("wrote", o)


if __name__ == "__main__":
    main()
