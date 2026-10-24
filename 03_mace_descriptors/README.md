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
  What changes structurally is not identified yet — this is open.
- **Descriptor space** (`mace_A3g4_descriptor.png`): at step 0 the Mg spread continuously from hcp-like to rutile-like.
  By the end they form two separate groups, with a sparse bridge of intermediate (interface) Mg.
- **No nucleation event.** As with CN, the MACE-rutile cluster is already ≥ 100 Mg at step 0 with dcut 6.5: MACE sees
  the local MgH₆ octahedra that already exist in the random start.

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
| `plot_mace_movie.py` | → `mace_A3g4.mp4` (20 MB): MACE classes, per-atom energy map, fractions / energy / cluster vs step |

## Run

```bash
conda activate mace && python compute_mace_descriptors.py          # ~1.5 h on 12 CPU threads
conda activate PEM  && python plot_mace_results.py && python plot_mace_movie.py [--dcut 6.5]
```

The model was saved on a GPU. `compute_mace_descriptors.py` therefore patches `torch.jit.load` to use
`map_location="cpu"`, so it runs on a CPU-only machine.
