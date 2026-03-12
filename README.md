# Geometric Phase Transition Enables Extreme Hippocampal Memory Capacity

Code for reproducing all analyses and figures. Three tiers of analysis test the hypothesis that topological organization of hippocampal spatial codes enables extreme memory capacity in food-caching birds.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the raw data

Download the Aronov et al. (2021) electrophysiology dataset from Dryad:

> Payne, H. L., Lynch, G. F., & Aronov, D. (2021). *Neural representations of space in the hippocampus of a food-caching bird* [Dataset]. Dryad. <https://doi.org/10.5061/dryad.pg4f4qrp7>

Unzip `payne_et_al_2021_data.zip` into a local directory. The archive contains two MAT files (`RESULTS_T.mat` for titmouse, `RESULTS_Z.mat` for zebra finch).

### 3. Build the analysis dataset

Run `build_dataset.py` pointing to the directory containing the unzipped MAT files:

```bash
python build_dataset.py /path/to/unzipped/data/
```

This classifies each unit as excitatory or inhibitory via hierarchical clustering on waveform features and exports:
- `aronov_dataset.csv` — 993 units with scalar columns (for inspection)
- `aronov_dataset.pkl` — full dataset including rate map arrays (used by all analysis scripts)

Move (or symlink) the `.pkl` file into a `data/` directory at the repository root:

```bash
mkdir -p data
mv /path/to/unzipped/data/aronov_dataset.pkl data/
```

All analysis scripts expect `data/aronov_dataset.pkl` by default.

---

## Repository structure

```
├── build_dataset.py              # Raw MAT → annotated dataset
├── data/                         # aronov_dataset.pkl (user-supplied)
├── output/                       # All CSV / figure outputs (auto-created)
├── tier 1/                       # Species comparison analyses + figures
├── tier 2/                       # E/I circuit analyses + figures
└── tier 3/                       # Capacity modeling + figures
```

---

## Tier 1 — Species comparison: geometric stability

Tests whether food-caching chickadees exhibit higher geometric stability than non-caching zebra finches, using split-half representational geometry.

### Analysis scripts

**`tier 1/tier1.py`** — Core Tier 1 analysis. Computes Shesha (split-half RDM stability) for every session in both species. Includes negative controls (circular shift, map shuffle), neuron-count-matched downsampling, permutation tests, jackknife robustness checks, and two alternative benchmarks (test-retest PV correlation, CCA stability) with concordance testing.

```bash
python "tier 1/tier1.py"
```

**`tier 1/tier1_valiant_shesha.py`** — Extended hypothesis tests combining Shesha with revised Valiant stability metrics (place field size consistency, population overlap CV, split-half allocation reliability), Mantel tests (neural RDM vs physical distance), A-P gradient analysis within chickadee subdivisions, within-session drift, and a double-dissociation analysis of geometric vs allocation stability.

```bash
python "tier 1/tier1_valiant_shesha.py"
```

**`tier 1/tier1_benchmarks.py`** — Alternative stability benchmarks. Runs Procrustes-aligned split-half error and CCA stability alongside Shesha on the same sessions, with a concordance panel showing all three metrics agree directionally.

```bash
python "tier 1/tier1_benchmarks.py"
```

### Figure scripts

- `tier 1/fig_1.py` — Main figure 1: physically-sorted RDMs + Mantel statistics
- `tier 1/fig_sorted_rdms_supp.py` — Appendix: cluster-sorted RDMs + topogeographic correspondence
- `tier 1/fig_controls_2x2.py` — Appendix: methodological controls (neuron-matched downsampling, spatial permutation controls, PV correlation, CCA stability)
- `tier 1/fig_valiant_2x2.py` — Appendix: Valiant SMA and double dissociation (place field size CV, allocation reliability, SHESHA vs allocation scatter, A-P gradient)
- `tier 1/fig_raw_maps.py` — Appendix: raw spatial firing rate maps for representative chickadee and finch place cells
- `tier 1/fig_species_coding.py` — Appendix: species-specific spatial coding signatures (information, regularity, coverage, stability, firing rates)
- `tier 1/fig_temporal_drift.py` — Appendix: extended temporal drift simulations (stand-alone, no data required)

---

## Tier 2 — E/I circuit contributions to geometric stability

Tests whether geometric stability arises from excitatory recurrence alone or from synergistic E/I coordination.

### Analysis scripts

**`tier 2/tier2_ei_stability.py`** — E vs I contributions to geometric stability. Computes Shesha and Mantel tests separately for excitatory and inhibitory populations, spatial information by cell type, within-session stability by cell type, knockout tests (E-only vs I-only vs full population), E-I spatial correlation analysis, and subspace angle measurements between E and I representations.

```bash
python "tier 2/tier2_ei_stability.py"
```

**`tier 2/tier2_enhanced.py`** — Comprehensive within-chickadee E/I analysis. Includes partial information decomposition (PID), residual analysis after removing shared variance, bootstrap and Bayesian credible intervals, E-I coordination index, spatial frequency analysis, temporal dynamics, negative controls (anti-correlated, random pairing, scaled noise), and dimensionality analysis.

```bash
python "tier 2/tier2_enhanced.py"
```

### Figure scripts

- `tier 2/fig_2.py` — Main figure 2: Bayesian posterior for Delta_EI, E vs I Shesha scatter, principal subspace angles, dimensionality comparison
- `tier 2/fig_ei_single_cell.py` — Appendix: E/I single-cell properties and population synergy (spatial information, temporal stability, Mantel r, and knockout test by cell type)

---

## Tier 3 — Memory capacity and topological phase transitions

Computational modeling proving that topological preservation maximizes memory capacity, with empirical validation from redundancy measurements.

### Analysis scripts

**`tier 3/tier3_capacity_redundancy.py`** — Core capacity model + empirical redundancy. Simulates population codes at three topology levels (crystal/mist/noise) under a noisy nearest-neighbor decoder across memory loads. Tests that topological codes degrade gracefully while random codes collapse. Includes empirical redundancy ratios from the Aronov data comparing chickadee vs finch.

```bash
python "tier 3/tier3_capacity_redundancy.py"
```

**`tier 3/tier3_parameter_sweep.py`** — Large-scale parameter sweep. Varies population size, sparsity, number of trials, and topology strength across 10,000+ configurations. Includes multiple decoder comparisons (NN, linear, SVM, Bayesian), noise robustness analysis (additive, multiplicative, correlated), critical noise threshold detection, and information-theoretic capacity bounds. Supports batch execution for long-running sweeps.

```bash
python "tier 3/tier3_parameter_sweep.py"
```

### Figure scripts

- `tier 3/fig_3.py` — Main figure 3: decoding error vs memory load, parameter sweep topology advantage, empirical redundancy
- `tier 3/fig_parameter_sweep.py` — Appendix: 5x5 heatmap of topology advantage across the full parameter sweep
