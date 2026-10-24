#!/usr/bin/env python3
"""Per-atom MACE descriptors and energies for the Mg/MgH2 trajectory (fine-tuned A3 gen-4 model).

Usage:  python compute_mace_descriptors.py [EXTXYZ] [--model ../models/A3g4_ft_MACE_mh_stagetwo.model]
                                          [--head Default] [--npc 16] [--threads 12] [--stride 1]
  EXTXYZ  default: ../01_structure_phase_transition/soap.2.md.064.diff.extxyz

For every saved frame, one forward pass of the MACE model (no forces, CPU, float32) gives
  - node_feats: the invariant (l = 0) part of both interaction layers, 2 x 512 = 1024 numbers per atom
    (the same per-atom representation the model reads its energy from);
  - node_energy: the per-atom energy contribution (eV) and the total energy E.
Reference descriptors: mean Mg descriptor of perfect hcp Mg (a 3.21, c 5.21 A) and rutile MgH2
(a 4.50, c 3.01 A, H x 0.304), 0 K geometry, from the same model.

Stored (the full 672 x 1024 x 1024 array would be ~2.8 GB, so it is reduced):
  d_hcp, d_rutile   (F, n_Mg) Euclidean distance of each Mg descriptor to the hcp / rutile reference,
                    the MACE analogue of the 01 SOAP columns
  pca               (F, n, npc) projection of every atom (Mg and H) on the first npc principal
                    components, fitted on Mg and H descriptors of 12 frames spread over the run
  node_energy       (F, n) eV;  energy (F,) eV;  volume (F,) A^3
  + species, timestep, PCA mean/components/explained variance, reference vectors
-> mace_A3g4.npz (in this folder). Checkpoints every 25 frames (mace_A3g4.partial.npz) and resumes.
Needs: mace-torch 0.3.15, torch (CPU), ase -> conda env `mace`.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
_jit_load = torch.jit.load
# the model was saved on GPU; e3nn's compiled sub-modules ignore map_location unless forced
torch.jit.load = lambda f, *a, **k: _jit_load(f, map_location="cpu")

from ase import Atoms  # noqa: E402
from ase.spacegroup import crystal  # noqa: E402
from e3nn import o3  # noqa: E402
from mace.calculators import MACECalculator  # noqa: E402
from mace.modules.utils import extract_invariant  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "01_structure_phase_transition"))
sys.dont_write_bytecode = True     # keep 01 free of __pycache__
from plot_phase_movie import read_extxyz  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("extxyz", nargs="?",
                   default=os.path.join(ROOT, "01_structure_phase_transition", "soap.2.md.064.diff.extxyz"))
    p.add_argument("--model", default=os.path.join(ROOT, "models", "A3g4_ft_MACE_mh_stagetwo.model"))
    p.add_argument("--head", default="Default")
    p.add_argument("--npc", type=int, default=16, help="number of principal components kept")
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--stride", type=int, default=1)
    return p.parse_args()


class Mace:
    def __init__(self, path, head):
        self.calc = MACECalculator(model_paths=path, device="cpu", default_dtype="float32", head=head)
        self.model = self.calc.models[0]
        irr = o3.Irreps(str(self.model.products[0].linear.irreps_out))
        self.lmax, self.nf = irr.lmax, irr.dim // (irr.lmax + 1) ** 2
        self.nlayer = int(self.model.num_interactions)

    def __call__(self, atoms):
        """-> invariant descriptors (n, nlayer*nf), node energies (n,), total energy."""
        b = self.calc._atoms_to_batch(atoms)
        with torch.no_grad():
            out = self.model(b.to_dict(), compute_force=False, compute_stress=False, training=False)
        D = extract_invariant(out["node_feats"], num_layers=self.nlayer, num_features=self.nf, l_max=self.lmax)
        return D.numpy().astype(np.float32), out["node_energy"].numpy(), float(out["energy"])


def references(m):
    hcp = crystal("Mg", [(1 / 3, 2 / 3, 0.25)], spacegroup=194, cellpar=[3.21, 3.21, 5.21, 90, 90, 120]) * (4, 4, 3)
    rut = crystal(["Mg", "H"], [(0, 0, 0), (0.304, 0.304, 0)], spacegroup=136,
                  cellpar=[4.50, 4.50, 3.01, 90, 90, 90]) * (3, 3, 4)
    ref = {}
    for nm, at in (("hcp", hcp), ("rutile", rut)):
        D, _, e = m(at)
        mg = np.array(at.get_chemical_symbols()) == "Mg"
        ref[nm] = D[mg].mean(0)
        print(f"reference {nm}: {len(at)} atoms, E/atom {e / len(at):.4f} eV, "
              f"Mg descriptor spread {np.linalg.norm(D[mg] - ref[nm], axis=1).max():.2e}")
    return ref


def main():
    a = parse_args()
    torch.set_num_threads(a.threads)
    species, pos, _, cells, ts = read_extxyz(a.extxyz)
    frames = np.arange(0, len(pos), a.stride)
    F, n = len(frames), len(species)
    mg = species == "Mg"
    m = Mace(a.model, a.head)
    print(f"{F} frames x {n} atoms; descriptor {m.nlayer} x {m.nf}; model {os.path.basename(a.model)}")
    ref = references(m)

    def atoms(k):
        at = Atoms(species, positions=pos[k], cell=cells[k], pbc=True)
        at.wrap()
        return at

    # PCA basis from 12 frames spread over the run
    sample = np.unique(np.linspace(0, len(pos) - 1, 12).astype(int))
    X = np.concatenate([m(atoms(k))[0] for k in sample])
    mean = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    comps, evr = Vt[:a.npc], (S ** 2 / (S ** 2).sum())[:a.npc]
    print(f"PCA on {len(X)} atom descriptors: first {a.npc} PCs explain {evr.sum():.3f} "
          f"(PC1 {evr[0]:.3f}, PC2 {evr[1]:.3f})")
    del X

    out = os.path.join(HERE, "mace_A3g4.npz")
    part = os.path.join(HERE, "mace_A3g4.partial.npz")
    res = dict(d_hcp=np.full((F, mg.sum()), np.nan, np.float32), d_rutile=np.full((F, mg.sum()), np.nan, np.float32),
               pca=np.full((F, n, a.npc), np.nan, np.float32), node_energy=np.full((F, n), np.nan, np.float32),
               energy=np.full(F, np.nan), volume=np.array([np.linalg.det(cells[k]) for k in frames]))
    start = 0
    if os.path.exists(part):
        old = np.load(part)
        start = int(old["done"])
        for key in res:
            res[key][:start] = old[key][:start]
        print(f"resuming at frame {start}")
    t0 = time.time()
    for j in range(start, F):
        D, ne, e = m(atoms(frames[j]))
        res["d_hcp"][j] = np.linalg.norm(D[mg] - ref["hcp"], axis=1)
        res["d_rutile"][j] = np.linalg.norm(D[mg] - ref["rutile"], axis=1)
        res["pca"][j] = (D - mean) @ comps.T
        res["node_energy"][j], res["energy"][j] = ne, e
        if (j + 1) % 25 == 0 or j == F - 1:
            np.savez(part, done=j + 1, **res)
            el = time.time() - t0
            print(f"  frame {j + 1}/{F}  E {e:.2f} eV  {el / (j + 1 - start):.1f} s/frame", flush=True)
    np.savez_compressed(out, species=species, timestep=ts[frames], frames=frames, model=os.path.basename(a.model),
                        head=a.head, pca_mean=mean, pca_components=comps, pca_explained=evr,
                        ref_hcp=ref["hcp"], ref_rutile=ref["rutile"], **res)
    os.remove(part)
    print("wrote", out)


if __name__ == "__main__":
    main()
