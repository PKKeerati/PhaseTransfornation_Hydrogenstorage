# 04 — Same trajectory, three descriptors (CN, SOAP, MACE)

`compare_descriptors.mp4` shows the trajectory three times side by side, with Mg coloured by each descriptor's
hcp / rutile / other label. Below the atoms: the rutile fraction and the largest rutile cluster vs step for all three.

**One rule for every descriptor** (Mg only): take a centred 5-frame moving average, assign the nearer phase, and use
"other" if the atom is far from both.

| descriptor | what it measures | rule |
|---|---|---|
| CN | Mg–H coordination number within 2.6 Å (**node degree in the bipartite Mg–H graph**) — composition | hcp ≤ 1, rutile ≥ 4 |
| SOAP | distance to the hcp / rutile reference (01) — crystalline order | other if both > 600 |
| MACE | distance to the hcp / rutile reference in the A3 gen-4 embedding (03) — learned | other if both > 6.5 |

The nucleus is the largest connected cluster of rutile-labelled Mg in the Mg–Mg graph (edges < 4.1 Å).

## Results

| | CN | SOAP | MACE |
|---|---|---|---|
| last 20 %: hcp / rutile / other | 0.425 / 0.488 / 0.087 | 0.347 / 0.349 / 0.305 | 0.427 / 0.454 / 0.119 |
| largest rutile cluster, last 20 % | 250 | 178 | 232 |
| cluster reaches 10 / 50 / 100 Mg at step | 0 / 0 / 0 | 12k / 27k / 40k | 0 / 0 / 0 |

Per-atom agreement over all frames (last 20 % in brackets): CN–MACE 0.949 (0.959), SOAP–MACE 0.772 (0.801),
CN–SOAP 0.741 (0.780).

- **MACE ≈ CN.** With this cutoff, the learned descriptor mostly encodes the local H count. It labels the interface
  bands slightly more often as "other" (0.12 vs 0.09).
- **SOAP is the only descriptor that shows growth from zero**, because it needs crystalline order, not only H nearby.
  It calls 30 % of Mg "other" at the end: mostly the H-rich, disordered band at x ≈ 0–8 Å and Mg at the interfaces.
- **The descriptors disagree mainly about the disordered start.** Which one "sees nucleation" depends on what is
  called the new phase: an MgH₆ octahedron (CN, MACE) or a piece of rutile lattice (SOAP). For MACE this choice is
  set by the cutoff (see `../03_mace_descriptors/README.md`, cutoff table).
- Spectral clustering with MACE features (`../02_graph_laplacian_clustering/glc_mace_*`): eigengap λ₃/λ₂ = 17.2 (CN 11.7).
  Two-phase split 0.437 / 0.563; agreement 0.86 with SOAP and 0.83 with the CN threshold.

## Files and run

`compare_descriptors_movie.py` (env PEM) → `compare_descriptors.mp4` (27 MB), `compare_descriptors.png` (static
curves), `compare_descriptors.csv` (fractions and cluster sizes per frame for each descriptor).

```bash
python compare_descriptors_movie.py [--mace-dcut 6.5] [--soap-dcut 600] [--stride 1]
```
