#!/usr/bin/env python3
"""What causes the second MACE energy drop (steps ~245k-300k)? Per-frame order parameters of the hydride.

Usage:  python analyse_energy_drop.py [NPZ] [--rc-mgh 2.6] [--rc-chain 3.25] [--window 9]
  NPZ  default: mace_A3g4.npz next to this script

Hydride Mg = Mg with >= 5 H within --rc-mgh. Per frame:
  energy split   sum of MACE node energies over hydride (hydride Mg + all H) and over hcp-like Mg (CN <= 1)
  defects        Mg with >= 7 H, H with >= 4 Mg (ideal rutile: Mg 6 H, H 3 Mg)
  chain order    rutile c-axis = Mg-Mg chains at 3.0 A. Bonds < --rc-chain between hydride Mg give unit vectors v;
                 nematic tensor Q = <3/2 v v^T - 1/2 I>; S = largest eigenvalue (0 random, 1 all parallel),
                 director = its eigenvector (sign-free, so |components|)
  mobility       mean squared displacement between saved frames (1000 MD steps), hydride Mg and H
  SOAP           mean dist_rutile of hydride Mg (01 column)
Writes energy_drop.png and energy_drop.csv next to NPZ.
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
from plot_phase_movie import read_extxyz, moving_average, COL  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("npz", nargs="?", default=os.path.join(HERE, "mace_A3g4.npz"))
    p.add_argument("--extxyz", default=os.path.join(ROOT, "01_structure_phase_transition", "soap.2.md.064.diff.extxyz"))
    p.add_argument("--rc-mgh", type=float, default=2.6)
    p.add_argument("--rc-chain", type=float, default=3.25)
    p.add_argument("--window", type=int, default=9, help="moving average for the plotted curves (frames)")
    return p.parse_args()


def mic(d, L):
    return d - L * np.round(d / L)


def main():
    a = parse_args()
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    r = np.load(a.npz)
    species, pos, dist, cells, ts = read_extxyz(a.extxyz)
    fr = r["frames"]
    pos, dist, cells, ts = pos[fr], dist[fr], cells[fr], ts[fr]
    mg, hy = species == "Mg", species == "H"
    L = np.einsum("fii->fi", cells)
    F = len(fr)
    keys = ["E_hydride", "E_hcp", "n_mg7", "n_h4", "S", "dir_x", "dir_y", "dir_z", "msd_mg", "msd_h", "soap_rutile"]
    o = {k: np.full(F, np.nan) for k in keys}
    iu = np.triu(np.ones((mg.sum(), mg.sum()), bool), 1)
    for k in range(F):
        pm, ph = pos[k, mg], pos[k, hy]
        c = (mic(pm[:, None] - ph[None], L[k]) ** 2).sum(-1) < a.rc_mgh ** 2
        cmg, ch = c.sum(1), c.sum(0)
        hyd = cmg >= 5
        ne = r["node_energy"][k]
        o["E_hydride"][k] = ne[mg][hyd].sum() + ne[hy].sum()
        o["E_hcp"][k] = ne[mg][cmg <= 1].sum()
        o["n_mg7"][k], o["n_h4"][k] = (cmg >= 7).sum(), (ch >= 4).sum()
        dm = mic(pm[:, None] - pm[None], L[k])
        i, j = np.nonzero(((dm ** 2).sum(-1) < a.rc_chain ** 2) & hyd[:, None] & hyd[None] & iu)
        v = dm[i, j] / np.linalg.norm(dm[i, j], axis=1)[:, None]
        w, U = np.linalg.eigh(1.5 * (v[:, :, None] * v[:, None]).mean(0) - 0.5 * np.eye(3))
        o["S"][k] = w[-1]
        o["dir_x"][k], o["dir_y"][k], o["dir_z"][k] = np.abs(U[:, -1])
        o["soap_rutile"][k] = dist[k][mg][hyd, 1].mean()
        if k:
            dd = (mic(pos[k] - pos[k - 1], L[k]) ** 2).sum(1)
            o["msd_mg"][k], o["msd_h"][k] = dd[mg][hyd].mean(), dd[hy].mean()
    for k in ("msd_mg", "msd_h"):
        o[k][0] = o[k][1]          # no displacement for frame 0; keeps the moving average finite

    late = slice(int(0.8 * F), F)
    before = (ts >= 200000) & (ts < 245000)
    after = (ts >= 300000) & (ts < 345000)
    print("quantity            200-245k     300-345k    last 20 %")
    for k in keys:
        print(f"{k:18s} {np.nanmean(o[k][before]):10.3f}  {np.nanmean(o[k][after]):10.3f}  {np.nanmean(o[k][late]):10.3f}")
    sm = {k: moving_average(o[k][:, None], a.window)[:, 0] for k in ("S", "msd_h", "msd_mg")}
    for k, th, up in (("S", 0.10, True), ("msd_h", 1.5, False)):
        hit = np.nonzero((sm[k] > th) if up else (sm[k] < th))[0]
        hit = hit[ts[hit] > 100000]
        print(f"smoothed {k} first {'>' if up else '<'} {th} at step {ts[hit[0]] if len(hit) else '-'}")

    stem = os.path.join(os.path.dirname(os.path.abspath(a.npz)), "energy_drop")
    with open(stem + ".csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["timestep", "energy_eV"] + keys)
        for k in range(F):
            wr.writerow([ts[k], f"{r['energy'][k]:.3f}"] + [f"{o[x][k]:.4f}" for x in keys])

    tk = ts / 1e3
    sl = lambda x: moving_average(np.asarray(x)[:, None], a.window)[:, 0]  # noqa: E731
    fig, axs = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    ax = axs[0]
    ax.plot(tk, o["E_hydride"] - o["E_hydride"][late].mean(), color=COL["rutile"], lw=0.6, alpha=0.4)
    ax.plot(tk, sl(o["E_hydride"] - o["E_hydride"][late].mean()), color=COL["rutile"], lw=1.6, label="hydride (Mg CN ≥ 5 + all H)")
    ax.plot(tk, o["E_hcp"] - o["E_hcp"][late].mean(), color=COL["hcp"], lw=0.6, alpha=0.4)
    ax.plot(tk, sl(o["E_hcp"] - o["E_hcp"][late].mean()), color=COL["hcp"], lw=1.6, label="hcp-like Mg (CN ≤ 1)")
    ax.set_ylabel("Σ per-atom energy\n− late mean (eV)"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Second energy drop: the hydride orders and stops moving", fontsize=11, loc="left")
    ax = axs[1]
    ax.plot(tk, o["S"], color="#222222", lw=0.6, alpha=0.4); ax.plot(tk, sl(o["S"]), color="#222222", lw=1.6)
    ax.set_ylabel("chain order S\n(nematic, 0–1)")
    ax = axs[2]
    for k, c in (("dir_x", "#4477AA"), ("dir_y", "#228833"), ("dir_z", "#AA3377")):
        ax.plot(tk, sl(o[k]), color=c, lw=1.4, label=k[-1])
    ax.set_ylim(0, 1); ax.set_ylabel("|director|\ncomponents"); ax.legend(frameon=False, fontsize=9, ncol=3)
    ax = axs[3]
    ax.plot(tk, sl(o["msd_h"]), color=COL["H"], lw=1.6, label="H")
    ax.plot(tk, sl(o["msd_mg"]), color=COL["rutile"], lw=1.6, label="hydride Mg")
    ax.set_ylabel("MSD per 1000\nsteps (Å²)"); ax.legend(frameon=False, fontsize=9)
    ax = axs[4]
    ax.plot(tk, sl(o["n_mg7"]), color=COL["rutile"], lw=1.6, label="Mg with ≥ 7 H")
    ax.plot(tk, sl(o["n_h4"]), color=COL["H"], lw=1.6, label="H with ≥ 4 Mg")
    ax.set_ylabel("over-coordinated\natoms"); ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("MD step (×10$^3$)"); ax.set_xlim(tk[0], tk[-1])
    for ax in axs:
        ax.axvspan(245, 300, color="#DDDDDD", alpha=0.5, lw=0)
    fig.tight_layout(); fig.savefig(stem + ".png", dpi=200); plt.close(fig)
    print("wrote", stem + ".png", stem + ".csv")


if __name__ == "__main__":
    main()
