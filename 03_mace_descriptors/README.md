# 03 — MACE descriptors and per-atom energies

The fine-tuned MACE model (A3, gen-4; `../models/A3g4_ft_MACE_mh_stagetwo.model`) is a **message-passing graph neural
network (GNN)**. For every atom, its last layers hold a learned vector (**node embedding**) from which the atom's energy is
read. Here that vector is used as a local descriptor in place of SOAP (**a learned feature map instead of a hand-made
one**), and the per-atom energies are used as a second node feature.

## Method

- One forward pass per frame (672 frames, CPU, float32, no forces). Keep the invariant (l = 0) part of both
  interaction layers: 2 × 512 = **1024 numbers per atom**.
- Reference vectors: the mean Mg descriptor of perfect hcp Mg and perfect rutile MgH₂ (0 K, same model).
  The two references are 12.9 apart.
- `d_hcp`, `d_rutile` = Euclidean distance of every Mg to each reference — the MACE analogue of the SOAP columns in 01.
- PCA: 16 components, fitted on 12 frames; they explain 99.6 % of the variance (PC1 91.7 %).
- Classes for the movie: centred 5-frame average, then the nearer reference; "other" if both distances > `--dcut`
  (default 6.5 = half the reference gap).

## Results

| quantity | value |
|---|---|
| E/atom, step 0 → end | −2.325 → −2.433 eV (−108 meV/atom, −111 eV total) |
| energy drop 50 % / 90 % complete | step 48 000 / 267 000 |
| Mg per-atom energy, last 20 %: CN ≤ 1 / CN ≥ 4 | −1.372 / −1.668 eV |
| MACE classes, last 20 %: hcp / rutile / other | 0.427 / 0.454 / 0.119 |
| largest MACE-rutile cluster, last 20 % | 232 Mg |

- **The energy falls in two stages** (`mace_A3g4_energy.png`):
  1. **Steps 0–60k:** the crystalline-rutile cluster grows from 0 to ~150 Mg.
  2. **Steps 245k–300k:** a second, sharp drop of ~40 meV/atom. The box shape has already been constant since about
     step 200k (≈ 50 × 16.5 × 16.5 Å), and the cluster size has also stopped changing. The per-atom energies of both
     hcp-like and rutile-like Mg drop here; hcp-like Mg also has an earlier drop near step 150k.

  So the energy picks up an **ordering event inside domains that already exist**, which the cluster counts miss.
  See the next section.
- **Descriptor space** (`mace_A3g4_descriptor.png`): at step 0 the Mg spread continuously from hcp-like to rutile-like.
  By the end they form two separate groups, with a sparse bridge of intermediate (interface) Mg.
- **No nucleation event.** As with CN, the MACE-rutile cluster is already ≥ 100 Mg at step 0 with dcut 6.5: MACE sees
  the local MgH₆ octahedra that already exist in the random start.

### What causes the second drop (steps 245k–300k)

`analyse_energy_drop.py` → `energy_drop.png`, `energy_drop.csv`. Hydride = Mg with ≥ 5 H, plus all H.

| quantity | 200–245k | 300–345k |
|---|---|---|
| Σ per-atom energy, hydride / hcp-like Mg (eV) | −2083.8 / −310.8 | −2126.0 / −309.3 |
| chain order S of the rutile c-axis (Mg–Mg chains at 3.0 Å) | 0.04 | 0.19 |
| director (\|x\|, \|y\|, \|z\|) | wanders, (0.29, 0.45, 0.72) on average | locked at (0.29, 0.90, 0.31) |
| MSD per 1000 steps, hydride Mg / H (Å²) | 1.14 / 2.68 | 0.39 / 0.89 |
| Mg with ≥ 7 H / H with ≥ 4 Mg (ideal: 6 H / 3 Mg) | 74 / 156 | 45 / 111 |
| SOAP `dist_rutile` of hydride Mg | 777 | 675 |

**The drop happens entirely in the hydride (−42 eV); the hcp Mg does not change.** In the same window:

- **The rutile chains align.** Before, their directions are random and the averaged direction wanders between y and z.
  After, one direction is chosen, close to y, and stays fixed for the rest of the run.
  In math terms: **a nematic order parameter (the largest eigenvalue of the orientation tensor) switches from ≈ 0 to a
  finite value — orientational symmetry breaking**.
- **Atoms move about 3× less.**
- **Over-coordinated defects heal.**

The smoothed curves cross their midpoints together: S > 0.1 at step 267k, H MSD < 1.5 Å² at step 269k.

**Interpretation.** Up to ~245k the hydride slab is locally MgH₆/rutile-like (CN and SOAP already say so). But it is a
**mobile, polycrystalline (orientationally disordered) solid**: small rutile grains with different orientations that
keep rearranging, with many grain-boundary defects (Mg with 7 H, H with 4 Mg). At ~250–300k the grains **coarsen into
one orientation** (grain coalescence / recrystallisation). Grain boundaries disappear, the over-coordinated sites heal,
diffusion freezes, and the energy falls by ~40 meV per atom of the whole cell (~80 meV per hydride atom).

**This is a second, separate ordering step.** The sequence is: composition first (H-rich domain at step 0), then local
rutile order (SOAP cluster, 10–60k), then long-range orientation (250–300k). Only the last step has an
induction-like wait (~200k steps of nothing, then a fast change) — the closest thing to a nucleation event in this
run. Here the "nucleus" would be a grain of the winning orientation (**the order parameter is S or the director, not
a cluster size**).

**Caveats.**
- S = 0.19 is far from 1. The 3.25 Å bond cutoff also catches some non-chain Mg–Mg pairs (3.45 Å at 0 K, broadened
  by temperature), and the slab is strained in a box that does not fit rutile. So S tracks the change but does not
  measure the fraction of aligned crystal.
- The MSD includes vibrations and any MC moves, so it is a relative mobility measure, not a diffusion coefficient.
- One trajectory: the orientation chosen (≈ y) and the timing are single samples.

### The cutoff decides the "nucleation time" (warning)

| dcut | rutile fraction, step 0 | largest cluster, step 0 | reaches 10 / 50 / 100 Mg at step |
|---|---|---|---|
| 4.5 | 0.02 | 3 | 8k / 14k / 66k |
| 5.0 | 0.06 | 6 | 4k / 8k / 16k |
| 5.5 | 0.10 | 17 | 0 / 6k / 12k |
| 6.5 (default) | 0.20 | 100 | 0 / 0 / 0 |
| SOAP (01, dcut 600) | 0 | 0 | 12k / 27k / 40k |

The cutoff can be tuned to make a "nucleus" appear at any time from step 0 to ~60k, so no single nucleation time can be
read off this trajectory. The cause is overlapping distributions: CN ≥ 4 Mg at step 0 have median `d_rutile` 7.0 (10–90 %:
4.9–8.7); at the end, 4.2 (3.1–6.6). **The descriptor is continuous and there is no gap to place a threshold in** — a
topic for Nan (a density-based or Gaussian-mixture split, or a committor-based check).

## Files

| file | what |
|---|---|
| `compute_mace_descriptors.py` | descriptors + energies → `mace_A3g4.npz` (env `mace`; ~8 s/frame, ~6 GB RAM) |
| `mace_A3g4.npz` | 46 MB; `d_hcp`, `d_rutile` (672 × 512), `pca` (672 × 1024 × 16), `node_energy`, `energy`, `volume`, references |
| `plot_mace_results.py` | → `mace_A3g4_energy.png`, `mace_A3g4_node_energy.png`, `mace_A3g4_descriptor.png` |
| `analyse_energy_drop.py` | → `energy_drop.png`, `energy_drop.csv`: hydride energy, chain order S, director, MSD, defects vs step |
| `plot_mace_movie.py` | → `mace_A3g4.mp4` (20 MB): MACE classes, per-atom energy map, fractions / energy / cluster vs step |

## Run

```bash
conda activate mace && python compute_mace_descriptors.py          # ~1.5 h on 12 CPU threads
conda activate PEM  && python plot_mace_results.py && python plot_mace_movie.py [--dcut 6.5]
```

The model was saved on a GPU. `compute_mace_descriptors.py` therefore patches `torch.jit.load` to use
`map_location="cpu"`, so it runs on a CPU-only machine.
