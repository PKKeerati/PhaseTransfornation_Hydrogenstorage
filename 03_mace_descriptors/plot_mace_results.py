#!/usr/bin/env python3
"""Figures from mace_A3g4.npz: energy vs MD step, per-atom energy map, descriptor-space scatter.

Usage:  python plot_mace_results.py [NPZ] [--window 5]
  NPZ  default: mace_A3g4.npz next to this script (from compute_mace_descriptors.py)

Writes (next to NPZ):
  mace_A3g4_energy.png       E per atom vs step, with the rutile fraction from 02 (glc_cn.csv) and the
                             mean per-atom energy of hcp-like / rutile-like Mg (Mg-H CN <= 1 / >= 4)
  mace_A3g4_node_energy.png  last frame, view along z: Mg and H coloured by per-atom energy
  mace_A3g4_descriptor.png   Mg in (d_hcp, d_rutile) plane for the first and last frame, coloured by CN
Needs: numpy, matplotlib -> conda env PEM.
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "01_structure_phase_transition"))
sys.dont_write_bytecode = True     # keep 01 free of __pycache__
from plot_phase_movie import read_extxyz, COL  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("npz", nargs="?", default=os.path.join(HERE, "mace_A3g4.npz"))
    p.add_argument("--extxyz", default=os.path.join(ROOT, "01_structure_phase_transition", "soap.2.md.064.diff.extxyz"))
    p.add_argument("--rc-mgh", type=float, default=2.6, help="Mg-H cutoff for CN (A)")
    return p.parse_args()


def main():
    a = parse_args()
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    r = np.load(a.npz)
    species, pos, _, cells, ts = read_extxyz(a.extxyz)
    fr = r["frames"]
    pos, cells, ts = pos[fr], cells[fr], ts[fr]
    mg, hy = species == "Mg", species == "H"
    n = len(species)
    L = np.einsum("fii->fi", cells)
    cn = np.zeros((len(fr), mg.sum()), int)
    for k in range(len(fr)):
        d = pos[k, mg][:, None] - pos[k, hy][None]
        d -= L[k] * np.round(d / L[k])
        cn[k] = ((d ** 2).sum(-1) < a.rc_mgh ** 2).sum(1)
    ne_mg = r["node_energy"][:, mg]
    e_hcp = np.array([ne_mg[k][cn[k] <= 1].mean() for k in range(len(fr))])
    e_rut = np.array([ne_mg[k][cn[k] >= 4].mean() for k in range(len(fr))])
    tk = ts / 1e3
    stem = os.path.join(os.path.dirname(os.path.abspath(a.npz)), "mace_A3g4")
    E = r["energy"] / n
    print(f"E per atom: frame 0 {E[0]:.4f}, last {E[-1]:.4f} eV (change {(E[-1] - E[0]) * 1e3:.1f} meV/atom, "
          f"{E[-1] * n - E[0] * n:.1f} eV total)")
    late = slice(int(0.8 * len(E)), None)
    print(f"last 20 %: E per atom {E[late].mean():.4f} ± {E[late].std() * 1e3:.2f} meV; "
          f"Mg node energy hcp-like {np.nanmean(e_hcp[late]):.3f}, rutile-like {np.nanmean(e_rut[late]):.3f} eV")
    for thr in (0.5, 0.9):
        drop = E[0] - E[late].mean()
        hit = np.nonzero(E[0] - E >= thr * drop)[0]
        print(f"energy has fallen {thr:.0%} of the way at step {ts[hit[0]]}")

    # energy vs step
    fig, axs = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True)
    ax = axs[0]
    ax.plot(tk, E * 1e3 - E[late].mean() * 1e3, color="#222222", lw=1.3)
    ax.set_ylabel("E per atom − late mean\n(meV/atom)")
    ax.set_title(f"MACE A3 gen-4 energy along the trajectory ({os.path.basename(str(r['model']))})",
                 fontsize=11, loc="left")
    glc = os.path.join(ROOT, "02_graph_laplacian_clustering", "glc_cn.csv")
    if os.path.exists(glc):
        rows = list(csv.DictReader(open(glc)))
        ax2 = ax.twinx()
        ax2.plot([int(x["timestep"]) / 1e3 for x in rows], [int(x["n_max_soap_rutile_cluster"]) for x in rows],
                 color=COL["rutile"], lw=1.0, ls="--")
        ax2.set_ylabel("largest crystalline\nrutile cluster (Mg)", color=COL["rutile"])
        ax2.spines["top"].set_visible(False)
    ax = axs[1]
    ax.plot(tk, e_hcp, color=COL["hcp"], lw=1.3, label="Mg with CN ≤ 1 (hcp-like)")
    ax.plot(tk, e_rut, color=COL["rutile"], lw=1.3, label="Mg with CN ≥ 4 (rutile-like)")
    ax.set_xlim(tk[0], tk[-1])
    ax.set_xlabel("MD step (×10$^3$)"); ax.set_ylabel("mean Mg per-atom\nenergy (eV)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{stem}_energy.png", dpi=200); plt.close(fig)

    # per-atom energy map, last frame
    k = len(fr) - 1
    p = pos[k] - L[k] * np.floor(pos[k] / L[k])
    o = np.argsort(p[:, 2])
    fig, axs = plt.subplots(2, 1, figsize=(9, 6.6))
    for ax, sel, nm in zip(axs, (mg, hy), ("Mg", "H")):
        idx = o[sel[o]]
        v = r["node_energy"][k, idx]
        sc = ax.scatter(p[idx, 0], p[idx, 1], c=v, cmap="viridis", s=40 if nm == "Mg" else 14,
                        vmin=np.percentile(v, 2), vmax=np.percentile(v, 98), edgecolors="#333333", linewidths=0.3)
        fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, label="eV")
        ax.add_patch(plt.Rectangle((0, 0), L[k, 0], L[k, 1], fill=False, lw=1, ec="#000000"))
        ax.set_aspect("equal"); ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
        ax.set_title(f"step {ts[k]}: per-atom energy of {nm} (MACE node energy)", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(f"{stem}_node_energy.png", dpi=150); plt.close(fig)

    # descriptor space
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    for ax, k in zip(axs, (0, len(fr) - 1)):
        sc = ax.scatter(r["d_hcp"][k], r["d_rutile"][k], c=cn[k], cmap="viridis", vmin=0, vmax=6, s=10)
        ax.set_xlabel("distance to hcp reference"); ax.set_title(f"step {ts[k]}", fontsize=10, loc="left")
    axs[0].set_ylabel("distance to rutile reference")
    fig.colorbar(sc, ax=axs, fraction=0.03, pad=0.01, label="Mg–H CN")
    fig.suptitle("Mg in MACE descriptor space (invariant node features, 2 × 512)", fontsize=11, x=0.06, ha="left")
    fig.savefig(f"{stem}_descriptor.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    for suf in ("_energy.png", "_node_energy.png", "_descriptor.png"):
        print("wrote", stem + suf)


if __name__ == "__main__":
    main()
