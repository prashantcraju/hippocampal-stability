#!/usr/bin/env python3
"""
fig_parameter_sweep.py — Camera-ready 5×5 heatmap figure for the
Tier 3 10,000-configuration parameter sweep.

Each of the 25 panels corresponds to one sparsity level (ρ = 0.01–0.25).
Within each panel: x = population size N, y = number of trials T,
colour = topology advantage (ΔError = random − crystal NN error).

Usage
-----
    python fig_parameter_sweep.py
    python fig_parameter_sweep.py  path/to/tier3_parameter_sweep_complete.csv
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from pathlib import Path

DEFAULT_CSV = 'output/tier3_sweep/tier3_parameter_sweep_complete.csv'

# ── Colour palette ──
CMAP = 'YlOrRd_r'
BG_MISSING = '#e8e8e8'

# ── Typography ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


def load_sweep(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'success'].copy()
    if 'crystal_advantage' not in df.columns:
        df['crystal_advantage'] = df['random_nn_error'] - df['crystal_nn_error']
    return df


def build_cube(df, metric='crystal_advantage'):
    """Reshape long-form dataframe into a 3-D array (sparsity × trials × neurons)."""
    sparsities = sorted(df['sparsity'].unique())
    neurons = sorted(df['n_neurons'].unique())
    trials = sorted(df['n_trials'].unique())

    cube = np.full((len(sparsities), len(trials), len(neurons)), np.nan)

    sp_map = {v: i for i, v in enumerate(sparsities)}
    tr_map = {v: i for i, v in enumerate(trials)}
    nn_map = {v: i for i, v in enumerate(neurons)}

    for _, row in df.iterrows():
        si = sp_map[row['sparsity']]
        ti = tr_map[row['n_trials']]
        ni = nn_map[row['n_neurons']]
        cube[si, ti, ni] = row[metric]

    return cube, sparsities, neurons, trials


def plot_sweep(cube, sparsities, neurons, trials,
               out_path='tier3_parameter_sweep_heatmaps.png'):

    nrows, ncols = 5, 5
    fig = plt.figure(figsize=(17.5, 15.5))

    gs = gridspec.GridSpec(
        nrows, ncols + 1,
        width_ratios=[1] * ncols + [0.045],
        hspace=0.40, wspace=0.22,
        left=0.065, right=0.915, top=0.915, bottom=0.055,
    )

    finite = cube[np.isfinite(cube)]
    vmin, vmax = np.percentile(finite, [1, 99])
    norm = Normalize(vmin=vmin, vmax=vmax)

    n_N = len(neurons)
    n_T = len(trials)

    xtick_pos = np.linspace(0, n_N - 1, 5, dtype=int)
    ytick_pos = np.linspace(0, n_T - 1, 5, dtype=int)

    im = None
    for idx, sp in enumerate(sparsities):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])

        ax.set_facecolor(BG_MISSING)
        im = ax.imshow(
            cube[idx], aspect='auto', origin='lower',
            cmap=CMAP, norm=norm, interpolation='nearest',
        )

        med = np.nanmedian(cube[idx])
        ax.set_title(f'ρ = {sp:.2f}  (med {med:.3f})',
                     fontsize=9, fontweight='bold', pad=4)

        if idx == 0:
            ax.text(0.5, 0.5, 'near-zero\nadvantage',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#8B0000', ec='white', alpha=0.7))

        if c == 0:
            ax.set_yticks(ytick_pos)
            ax.set_yticklabels([trials[i] for i in ytick_pos])
            ax.set_ylabel('Trials (T)', fontsize=8)
        else:
            ax.set_yticks([])

        if r == nrows - 1:
            ax.set_xticks(xtick_pos)
            ax.set_xticklabels([neurons[i] for i in xtick_pos],
                                rotation=45, ha='right')
            ax.set_xlabel('Population size (N)', fontsize=8)
        else:
            ax.set_xticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color('#555555')

    # empirical chickadee sparsity highlight -- after loop closes
    emp_idx = sparsities.index(0.15)    
    ax_emp = fig.axes[emp_idx]
    for spine in ax_emp.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_color('#C62D50')
    ax_emp.set_title(f'ρ = 0.15  (med {np.nanmedian(cube[emp_idx]):.3f})  ← empirical',
                     fontsize=9, fontweight='bold', pad=4, color='#C62D50')

    # ── Shared colourbar ──
    cax = fig.add_subplot(gs[:, ncols])
    cb = fig.colorbar(im, cax=cax, extend='both')
    cb.set_label('Topology advantage  (Δ Error)',
                 fontsize=11, labelpad=10)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.4)

    fig.suptitle(
        'Parameter Sweep: Topology Advantage Across 10,000 Configurations\n'
        r'$\Delta\mathrm{Error} = \mathrm{Error}_{\mathrm{random}} '
        r'- \mathrm{Error}_{\mathrm{crystal}}$'
        '   (higher → topology helps more)',
        fontsize=13, fontweight='bold', y=0.975,
    )

    fig.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig('tier3_parameter_sweep_heatmaps.pdf', dpi=300, facecolor='white', bbox_inches='tight')

    print(f"Saved: {out_path}")
    plt.show()


def main(csv_path=None):
    csv_path = csv_path or DEFAULT_CSV
    path = Path(csv_path)
    if not path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    df = load_sweep(path)
    print(f"Loaded {len(df):,} successful configurations from {csv_path}")

    cube, sparsities, neurons, trials = build_cube(df)
    print(f"  Sparsity levels : {len(sparsities)}  "
          f"({sparsities[0]:.2f} – {sparsities[-1]:.2f})")
    print(f"  Population sizes: {len(neurons)}  "
          f"({neurons[0]} – {neurons[-1]})")
    print(f"  Trial counts    : {len(trials)}  "
          f"({trials[0]} – {trials[-1]})")

    plot_sweep(cube, sparsities, neurons, trials)


if __name__ == '__main__':
    csv = 'output/tier3_sweep/tier3_parameter_sweep_complete.csv'
    main(csv)
