#!/usr/bin/env python3
"""Graph-Laplacian (spectral) clustering of the Mg (hcp) / MgH2 (rutile) phase separation.

Usage:  python graph_laplacian_clustering.py [EXTXYZ] [--feature cn|soap|mace] [--k 2] [--window 5]
                                             [--rc-mgmg 4.1] [--rc-mgh 2.6] [--sigma-cn 1.0]
                                             [--sigma-soap 0.5]
  EXTXYZ  SOAP-tagged MD trajectory (default: ../01_structure_phase_transition/soap.2.md.064.diff.extxyz).

Method, per saved frame (Mg atoms only are nodes):
  1. Graph G: edge i-j if the Mg-Mg minimum-image distance < --rc-mgmg (first minimum of the
     Mg-Mg RDF, 4.1 A).
  2. Node feature f_i, centred time average over --window frames:
       cn   : Mg-H coordination number (H within --rc-mgh, first minimum of the Mg-H RDF, 2.6 A)
       soap : (log d_hcp, log d_rutile) from the stored SOAP distances
       mace : (d_hcp, d_rutile) distances of the MACE A3 gen-4 invariant descriptor to perfect hcp /
              rutile references (../03_mace_descriptors/mace_A3g4.npz)
  3. Edge weight w_ij = exp(-|f_i - f_j|^2 / (2 sigma^2)), fixed sigma (--sigma-cn in H atoms,
     --sigma-soap in log units, --sigma-mace in descriptor units), so weights are comparable
     between frames.
  4. Normalised Laplacian L = I - D^-1/2 W D^-1/2; eigenvectors of the k smallest eigenvalues,
     rows normalised (Ng-Jordan-Weiss), k-means (10 restarts) -> k clusters.
  5. Clusters named by mean Mg-H CN: lowest = hcp, highest = rutile (k > 2: the rest = interface).
     The default k = 2 is what the spectrum shows: a single small eigenvalue lambda_2 separated
     from lambda_3 by a gap of ~10x once the system has separated.
  6. Interface = boundary nodes of the cut: Mg with at least one G-neighbour in the other cluster.
     Reported classes are hcp / rutile / interface (cluster label, with boundary nodes moved to
     interface).
  7. Nucleus = connected components in G of (a) the spectral rutile cluster (boundary included;
     an H-rich domain) and (b) Mg labelled rutile by the 01 SOAP threshold (crystalline rutile
     order); n_max(t) = the largest one.
     lambda_2(t) (algebraic connectivity of the weighted graph) is a scalar order parameter:
     ~ 0 when the box has split into two weakly coupled domains.
Spectral clustering always returns k clusters, also for the disordered start, so the labels mean
something only once the eigengap opens (see the spectrum figure).

Writes (in this folder, prefix glc_<feature>):
  .csv               per frame: lambda_1..6, hcp/rutile/interface counts, size of the rutile
                     cluster (two-phase count), n_max, number of rutile clusters,
                     SOAP- and CN-threshold counts for comparison
  _spectrum.png      lambda_2..6 vs MD step
  _fractions.png     phase fractions vs step (spectral vs SOAP threshold vs CN threshold)
  _nucleus.png       largest H-rich (spectral) and crystalline-rutile (SOAP) cluster vs step
  _final.png         last frame: Fiedler vector, spectral clusters, SOAP classes (view along z)
  .mp4               atoms coloured by spectral class (view along z) + fractions and lambda_2,3
                     vs step (skip with --no-movie; needs imageio-ffmpeg)
Needs: numpy, scipy, matplotlib -> conda env PEM.
"""
import argparse
import csv
import os
import sys

import numpy as np
import scipy.sparse as sps
from scipy.cluster.vq import kmeans2
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "01_structure_phase_transition")
sys.path.insert(0, SRC)
sys.dont_write_bytecode = True     # keep 01 free of __pycache__
from plot_phase_movie import read_extxyz, moving_average, classify, COL  # noqa: E402

CLASSES = ["hcp", "rutile", "interface"]
COL = dict(COL, interface=COL["other"])
LABEL = {"hcp": "hcp Mg", "rutile": "rutile MgH$_2$", "interface": "interface / other"}
N_EIG = 6


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("extxyz", nargs="?", default=os.path.join(SRC, "soap.2.md.064.diff.extxyz"))
    p.add_argument("--feature", choices=["cn", "soap", "mace"], default="cn")
    p.add_argument("--k", type=int, default=2, help="number of clusters")
    p.add_argument("--window", type=int, default=5, help="time-averaging window, saved frames")
    p.add_argument("--rc-mgmg", type=float, default=4.1, help="Mg-Mg edge cutoff (A)")
    p.add_argument("--rc-mgh", type=float, default=2.6, help="Mg-H cutoff for CN (A)")
    p.add_argument("--sigma-cn", type=float, default=1.0, help="weight scale for CN (H atoms)")
    p.add_argument("--sigma-soap", type=float, default=0.5, help="weight scale for log SOAP distance")
    p.add_argument("--sigma-mace", type=float, default=2.0,
                   help="weight scale for MACE descriptor distance (hcp-rutile reference gap 12.9)")
    p.add_argument("--mace-npz", default=os.path.join(HERE, "..", "03_mace_descriptors", "mace_A3g4.npz"))
    p.add_argument("--dcut", type=float, default=600.0, help="SOAP cutoff of the 01 baseline")
    p.add_argument("--no-movie", action="store_true", help="skip the MP4")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--stride", type=int, default=1, help="use every n-th frame in the movie")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def mic_dist(pa, pb, L):
    """Minimum-image distance matrix for an orthorhombic box with edge lengths L."""
    d = pa[:, None] - pb[None]
    d -= L * np.round(d / L)
    return np.sqrt((d ** 2).sum(-1))


def spectral(adj, f, sigma, k, seed=0):
    """Weighted graph from adjacency + node features -> (eigenvalues, eigenvectors, labels)."""
    i, j = np.nonzero(adj)
    w = np.exp(-((f[i] - f[j]) ** 2).sum(1) / (2 * sigma ** 2))
    n = len(f)
    W = sps.csr_matrix((w, (i, j)), shape=(n, n))
    deg = np.asarray(W.sum(1)).ravel()
    Dm = sps.diags(1 / np.sqrt(deg))
    L = sps.eye(n) - Dm @ W @ Dm
    vals, vecs = eigsh(L.tocsc(), k=N_EIG, sigma=-1e-3, which="LM")   # shift-invert near 0
    o = np.argsort(vals)
    vals, vecs = vals[o], vecs[:, o]
    U = vecs[:, :k] / np.linalg.norm(vecs[:, :k], axis=1, keepdims=True)
    best = None
    for r in range(10):
        cen, lab = kmeans2(U, k, minit="++", seed=seed + r)
        inertia = ((U - cen[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    fiedler = Dm @ vecs[:, 1]                       # random-walk Fiedler vector
    return vals, fiedler, best[1]


def name_clusters(lab, cn, k):
    """Map k-means labels to 0 = hcp (lowest mean CN), 1 = rutile (highest), 2 = interface."""
    order = np.argsort([cn[lab == c].mean() if (lab == c).any() else np.inf for c in range(k)])
    out = np.full(lab.shape, 2, int)
    out[lab == order[0]] = 0
    out[lab == order[-1]] = 1
    return out


def main():
    a = parse_args()
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    print(f"reading {a.extxyz}")
    species, pos, dist, cells, ts = read_extxyz(a.extxyz)
    F = len(pos)
    mg, hy = species == "Mg", species == "H"
    nmg = mg.sum()
    Ls = np.einsum("fii->fi", cells)

    # time-averaged node features
    cn = np.array([(mic_dist(pos[k, mg], pos[k, hy], Ls[k]) < a.rc_mgh).sum(1) for k in range(F)], float)
    cn_s = moving_average(cn, a.window)
    dh, dr = moving_average(dist[:, mg, 0], a.window), moving_average(dist[:, mg, 1], a.window)
    soap_cls = classify(dh, dr, a.dcut)                                   # 01 baseline
    cn_cls = np.where(cn_s >= 4, 1, np.where(cn_s <= 1, 0, 2))           # CN threshold baseline
    if a.feature == "cn":
        feat, sigma = cn_s[..., None], a.sigma_cn
    elif a.feature == "soap":
        feat, sigma = np.stack([np.log(dh), np.log(dr)], -1), a.sigma_soap
    else:
        md = np.load(a.mace_npz)
        if not np.array_equal(md["timestep"], ts):
            sys.exit(f"{a.mace_npz} does not cover every frame of {a.extxyz} (run without --stride)")
        feat = np.stack([moving_average(md["d_hcp"], a.window), moving_average(md["d_rutile"], a.window)], -1)
        sigma = a.sigma_mace

    lam = np.zeros((F, N_EIG))
    cls = np.zeros((F, nmg), int)        # 0 hcp, 1 rutile, 2 interface (boundary nodes)
    ph = np.zeros((F, nmg), int)         # cluster label before boundary nodes are moved
    nmax = np.zeros((F, 2), int)          # largest cluster: spectral rutile, SOAP rutile
    ncl = np.zeros((F, 2), int)           # number of clusters: spectral rutile, SOAP rutile
    fied_last = None
    for k in range(F):
        r = mic_dist(pos[k, mg], pos[k, mg], Ls[k])
        adj = (r < a.rc_mgmg) & (r > 0)
        vals, fied, lab = spectral(adj, feat[k], sigma, a.k)
        lam[k] = vals
        ph[k] = name_clusters(lab, cn_s[k], a.k)
        i, j = np.nonzero(adj)
        edge = np.zeros(nmg, bool)
        edge[i[ph[k][i] != ph[k][j]]] = True
        cls[k] = np.where(edge, 2, ph[k])
        for c, rut in enumerate((ph[k] == 1, soap_cls[k] == 1)):
            if rut.any():
                nc, comp = connected_components(sps.csr_matrix(adj[np.ix_(rut, rut)]), directed=False)
                nmax[k, c], ncl[k, c] = np.bincount(comp).max(), nc
        if k == F - 1:
            fied_last = fied
        if k % 100 == 0:
            print(f"  frame {k}/{F}  lambda_2 {vals[1]:.4f}  counts {np.bincount(cls[k], minlength=3)}")

    frac = np.stack([(cls == c).mean(1) for c in range(3)], 1)
    late = slice(int(0.8 * F), F)
    for n in (10, 50, 100, 200):
        hit = np.nonzero(nmax[:, 1] >= n)[0]
        print(f"SOAP-rutile cluster first reaches {n:3d} Mg at step {ts[hit[0]] if len(hit) else '-'}")
    print(f"last 20 % eigengap lambda_3/lambda_2 {np.median(lam[late, 2] / lam[late, 1]):.1f}, "
          f"lambda_4/lambda_3 {np.median(lam[late, 3] / lam[late, 2]):.1f}")
    print(f"last 20 % two-phase : hcp {(ph[late] == 0).mean():.3f} rutile {(ph[late] == 1).mean():.3f}")
    print(f"last 20 % spectral  : hcp {frac[late,0].mean():.3f} rutile {frac[late,1].mean():.3f} "
          f"interface {frac[late,2].mean():.3f}")
    for nm, ref in (("SOAP threshold", soap_cls), ("CN threshold", cn_cls)):
        f = [(ref[late] == c).mean() for c in range(3)]
        print(f"last 20 % {nm:15s}: hcp {f[0]:.3f} rutile {f[1]:.3f} other {f[2]:.3f}   "
              f"per-atom agreement with spectral {(ref[late] == cls[late]).mean():.3f}")
    for c, nm in enumerate(CLASSES):
        m = cls[late] == c
        print(f"  last 20 %  {nm:9s}: mean CN {cn_s[late][m].mean():.2f}")

    stem = os.path.join(HERE, f"glc_{a.feature}")
    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "timestep"] + [f"lambda_{i+1}" for i in range(N_EIG)]
                   + ["n_hcp", "n_rutile", "n_interface", "n_rutile_two_phase", "n_max_rutile_cluster", "n_rutile_clusters",
                      "n_max_soap_rutile_cluster", "n_soap_rutile_clusters",
                      "soap_hcp", "soap_rutile", "soap_other", "cn_hcp", "cn_rutile", "cn_other"])
        for k in range(F):
            w.writerow([k, ts[k]] + [f"{v:.6f}" for v in lam[k]]
                       + list(np.bincount(cls[k], minlength=3)) + [(ph[k] == 1).sum(), nmax[k, 0], ncl[k, 0], nmax[k, 1], ncl[k, 1]]
                       + list(np.bincount(soap_cls[k], minlength=3)) + list(np.bincount(cn_cls[k], minlength=3)))

    tk = ts / 1e3
    title = f"feature {a.feature}, k = {a.k}, Mg–Mg edges < {a.rc_mgmg:g} Å"

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    cm = plt.get_cmap("viridis")
    for i in range(1, N_EIG):
        ax.plot(tk, lam[:, i], lw=1.4, color=cm((i - 1) / (N_EIG - 1)), label=f"$\\lambda_{i+1}$")
    ax.set_yscale("log"); ax.set_xlim(tk[0], tk[-1])
    ax.set_xlabel("MD step (×10$^3$)"); ax.set_ylabel("eigenvalue of $L_{sym}$")
    ax.set_title(f"Laplacian spectrum ({title})", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(); fig.savefig(f"{stem}_spectrum.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    for c, nm in enumerate(CLASSES):
        ax.plot(tk, frac[:, c], color=COL[nm], lw=1.8, label=f"{LABEL[nm]} (spectral)")
        ax.plot(tk, (soap_cls == c).mean(1), color=COL[nm], lw=1.0, ls=":",
                label=f"{LABEL[nm]} (SOAP < {a.dcut:g})")
        ax.plot(tk, (cn_cls == c).mean(1), color=COL[nm], lw=1.0, ls="--",
                label=f"{LABEL[nm]} (CN threshold)")
    ax.axhline(0.5, color="#000000", lw=0.6, alpha=0.4)
    ax.set_xlim(tk[0], tk[-1]); ax.set_ylim(0, 1)
    ax.set_xlabel("MD step (×10$^3$)"); ax.set_ylabel("Fraction of Mg")
    ax.set_title(f"Phase fractions ({title})", fontsize=11)
    ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(); fig.savefig(f"{stem}_fractions.png", dpi=200); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    zoom = ts <= 150_000
    for ax, m in zip(axs, (slice(None), zoom)):
        ax.plot(tk[m], nmax[m, 0], color=COL["rutile"], lw=1.8, label="H-rich domain (spectral cluster)")
        ax.plot(tk[m], nmax[m, 1], color=COL["rutile"], lw=1.2, ls="--",
                label=f"crystalline rutile (SOAP < {a.dcut:g})")
        ax.axhline(nmg / 2, color="#000000", lw=0.6, alpha=0.4)
        ax.set_xlim(tk[m][0], tk[m][-1]); ax.set_ylim(0, nmg * 0.75)
        ax.set_xlabel("MD step (×10$^3$)")
    axs[0].set_ylabel("Mg in largest rutile cluster")
    axs[0].legend(frameon=False, fontsize=9, loc="lower right")
    axs[0].set_title(f"Largest connected rutile cluster in the Mg–Mg graph ({title})", fontsize=10, loc="left")
    axs[1].set_title("first 150 000 steps", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(f"{stem}_nucleus.png", dpi=200); plt.close(fig)

    k = F - 1
    p = pos[k, mg] - Ls[k] * np.floor(pos[k, mg] / Ls[k])
    o = np.argsort(p[:, 2])
    fig, axs = plt.subplots(3, 1, figsize=(9, 9.5))
    panels = [("Fiedler vector (random-walk normalised)", fied_last, None),
              ("spectral clusters", None, cls[k]), (f"SOAP threshold (01 baseline)", None, soap_cls[k])]
    for ax, (ttl, val, lab) in zip(axs, panels):
        if val is not None:
            v = val * np.sign(np.corrcoef(val, cn_s[k])[0, 1])
            sc = ax.scatter(p[o, 0], p[o, 1], c=v[o], cmap="coolwarm", s=40, edgecolors="#333333", linewidths=0.3)
            fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
        else:
            ax.scatter(p[o, 0], p[o, 1], c=[COL[CLASSES[c]] for c in lab[o]], s=40,
                       edgecolors="#333333", linewidths=0.3)
            ax.scatter([], [], c=COL["hcp"], label="hcp"); ax.scatter([], [], c=COL["rutile"], label="rutile")
            ax.scatter([], [], c=COL["other"], label="interface / other")
            ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
        ax.add_patch(plt.Rectangle((0, 0), Ls[k, 0], Ls[k, 1], fill=False, lw=1, ec="#000000"))
        ax.set_aspect("equal"); ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
        ax.set_title(f"step {ts[k]}: {ttl}", fontsize=10, loc="left", pad=18 if val is None else 6)
    fig.tight_layout(); fig.savefig(f"{stem}_final.png", dpi=150); plt.close(fig)
    for suf in (".csv", "_spectrum.png", "_fractions.png", "_nucleus.png", "_final.png"):
        print("wrote", stem + suf)
    if not a.no_movie:
        movie(a, f"{stem}.mp4", title, species, pos, cells, ts, cls, frac, lam)


def movie(a, out, title, species, pos, cells, ts, cls, frac, lam):
    """MP4: atoms (orthographic view along z, Mg by spectral class, H small) + fractions and lambda_2,3."""
    from matplotlib.animation import FFMpegWriter
    from matplotlib.lines import Line2D
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        print("imageio-ffmpeg not found; relying on ffmpeg in PATH")
    F, n = pos.shape[:2]
    mg, hy = species == "Mg", species == "H"
    L = np.einsum("fii->fi", cells)
    pos = pos - L[:, None] * np.floor(pos / L[:, None])
    rad = np.where(mg, 0.65, 0.30)                                       # drawn radii (A), as in 01
    tk = ts / 1e3

    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.6, 1], hspace=0.4, wspace=0.3)
    axa = fig.add_subplot(gs[0, :])
    axf, axl = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    axa.set_xlim(-1.5, L[:, 0].max() + 1.5); axa.set_ylim(-1.5, L[:, 1].max() + 1.5); axa.set_aspect("equal")
    axa.set_xlabel("x (Å)"); axa.set_ylabel("y (Å)")
    handles = [Line2D([], [], marker="o", ls="", ms=9, mfc=COL[c], mec="#333333", label=LABEL[c]) for c in CLASSES]
    handles.append(Line2D([], [], marker="o", ls="", ms=5, mfc=COL["H"], mec="#333333", label="H"))
    axa.legend(handles=handles, frameon=False, fontsize=10, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    for c, nm in enumerate(CLASSES):
        axf.plot(tk, frac[:, c], color=COL[nm], lw=1.6)
    axf.set_xlim(tk[0], tk[-1]); axf.set_ylim(0, 1)
    axf.set_xlabel("MD step (×10$^3$)"); axf.set_ylabel("Fraction of Mg")
    axl.plot(tk, lam[:, 1], color="#222222", lw=1.4, label="$\\lambda_2$")
    axl.plot(tk, lam[:, 2], color="#888888", lw=1.0, label="$\\lambda_3$")
    axl.set_yscale("log"); axl.set_xlim(tk[0], tk[-1])
    axl.set_xlabel("MD step (×10$^3$)"); axl.set_ylabel("eigenvalue")
    axl.legend(frameon=False, fontsize=9, loc="upper right")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.08)
    fig.canvas.draw()
    curs = [ax.axvline(tk[0], color="#000000", lw=1) for ax in (axf, axl)]
    box = axa.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1.2, ec="#000000"))
    sc = axa.scatter([], [], edgecolors="#333333", linewidths=0.3)
    head = fig.text(0.06, 0.985, "", fontsize=13, va="top")
    fig.text(0.06, 0.95, f"Mg classified by graph-Laplacian clustering ({title}); view along z, full depth",
             fontsize=9, va="top", color="#555555")
    bb = axa.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    size = (2 * rad * bb.width * 72 / np.diff(axa.get_xlim())[0]) ** 2
    writer = FFMpegWriter(fps=a.fps, bitrate=6000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, out, dpi=a.dpi):
        for j, k in enumerate(range(0, F, a.stride)):
            col = np.empty(n, object)
            col[mg] = [COL[CLASSES[c]] for c in cls[k]]
            col[hy] = COL["H"]
            o = np.argsort(pos[k, :, 2])                                  # painter's order back-to-front
            sc.set_offsets(pos[k, o, :2]); sc.set_facecolors(col[o].tolist()); sc.set_sizes(size[o])
            box.set_width(L[k, 0]); box.set_height(L[k, 1])
            for cu in curs:
                cu.set_xdata([tk[k], tk[k]])
            head.set_text(f"MgH  ·  step {ts[k]:>7d}  ·  hcp {frac[k,0]:.2f}   rutile {frac[k,1]:.2f}   "
                          f"interface {frac[k,2]:.2f}  ·  λ3/λ2 {lam[k,2] / lam[k,1]:.1f}")
            writer.grab_frame()
            if j % 100 == 0:
                print(f"  movie frame {k}/{F}")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
