#!/usr/bin/env python3
"""
fig_species_coding.py — Species-Specific Spatial Coding Signatures

Appendix figure showing that networks operating in the "extreme
capacity" regime (caching titmouse) exhibit distinct statistical
signatures compared to baseline networks (zebra finch).

Panels (2 x 3):
  A — Spatial information distribution (bits/spike), violin + strip
  B — Firing regularity (CV2), violin + strip
  C — Spatial coverage (fraction of environment visited), violin + strip
  D — Map stability (split-half xcorr_map), violin + strip
  E — Firing rate distribution (Hz), violin + strip
  F — Fraction spatially selective by species x cell type (E / I)

Data: loads aronov_dataset.pkl  (or .csv fallback).

Usage:
    python fig_species_coding.py [path/to/aronov_dataset.pkl]
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu
from pathlib import Path
import sys, os, warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'output/species_coding'

# ── Colours — colorblind-safe, matching paper palette ────────────────
C_CHICK   = '#C62D50'   # titmouse / chickadee red
C_FINCH   = '#4A90C4'   # zebra finch blue
SPECIES_PAL = {'titmouse': C_CHICK, 'zebra_finch': C_FINCH}
SPECIES_LABELS = {'titmouse': 'Chickadee', 'zebra_finch': 'Zebra finch'}


# ── Data loading ─────────────────────────────────────────────────────
def load_data(path=None):
    """Try pkl first, then csv.  Returns cleaned DataFrame."""
    candidates = [
        path,
        'data/aronov_dataset.pkl',
        'data/aronov_dataset.csv',
    ]
    df = None
    for p in candidates:
        if p is None:
            continue
        p = Path(p)
        if not p.exists():
            continue
        if p.suffix == '.pkl':
            df = pd.read_pickle(p)
        else:
            df = pd.read_csv(p)
        print(f"Loaded {len(df)} units from {p}")
        break

    if df is None:
        raise FileNotFoundError(
            "Could not find aronov_dataset.pkl or .csv.  "
            "Pass the path as the first argument.")

    required = ['species', 'info']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


# ── Helpers ──────────────────────────────────────────────────────────
def _mw_str(a, b, alternative='two-sided'):
    """Format a Mann-Whitney test result."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return ''
    _, p = mannwhitneyu(a, b, alternative=alternative)
    if p < 0.001:
        return f'$p = {p:.1e}$'
    return f'$p = {p:.3f}$'


def _violin(ax, data_t, data_z, ylabel, title, log=False):
    """
    Draw a split violin + jittered strip for two species.
    Handles NaN / empty gracefully.
    """
    data_t = np.asarray(data_t, dtype=float)
    data_z = np.asarray(data_z, dtype=float)
    data_t = data_t[np.isfinite(data_t)]
    data_z = data_z[np.isfinite(data_z)]

    if log:
        data_t = data_t[data_t > 0]
        data_z = data_z[data_z > 0]

    datasets = [data_t, data_z]
    positions = [0, 1]
    colors = [C_CHICK, C_FINCH]
    labels = ['Chickadee', 'Zebra finch']

    for i, (d, pos, c, lab) in enumerate(
            zip(datasets, positions, colors, labels)):
        if len(d) == 0:
            continue
        vp = ax.violinplot(d, positions=[pos], widths=0.7,
                           showmeans=False, showmedians=False,
                           showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(c)
            body.set_edgecolor(c)
            body.set_alpha(0.30)

        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.vlines(pos, q1, q3, color=c, lw=2.5, zorder=4)
        ax.scatter([pos], [med], color=c, s=35, zorder=5,
                   edgecolors='white', linewidths=0.6)

        rng = np.random.RandomState(320 + i)
        jitter = rng.uniform(-0.15, 0.15, len(d))
        ax.scatter(pos + jitter, d, color=c, s=8, alpha=0.25,
                   edgecolors='none', zorder=3)

    p_str = _mw_str(data_t, data_z)
    if p_str:
        ax.text(0.5, 0.96, p_str, transform=ax.transAxes,
                ha='center', va='top', fontsize=9, color='#444444',
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec='#cccccc', alpha=0.9))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', loc='left', pad=8)
    if log:
        ax.set_yscale('log')


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def make_figure(df):
    # ── rcParams ─────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.sans-serif':   ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size':         10,
        'axes.labelsize':    11,
        'axes.titlesize':    11.5,
        'axes.linewidth':    0.8,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'xtick.labelsize':   9.5,
        'ytick.labelsize':   9,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'legend.fontsize':   8.5,
        'legend.frameon':    False,
        'figure.dpi':        150,
        'savefig.dpi':       300,
        'savefig.facecolor': 'white',
    })

    t = df[df['species'] == 'titmouse']
    z = df[df['species'] == 'zebra_finch']

    fig = plt.figure(figsize=(16, 9.5))
    gs  = gridspec.GridSpec(2, 3, hspace=0.48, wspace=0.36,
                            left=0.065, right=0.96, top=0.90, bottom=0.07)

    # ── A: Spatial information ───────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    _violin(ax_a,
            t['info'].dropna().values,
            z['info'].dropna().values,
            'Spatial information (bits / spike)',
            'A.  Spatial information',
            log=True)

    # ── B: CV2 ───────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    if 'cv2' in df.columns:
        _violin(ax_b,
                t['cv2'].dropna().values,
                z['cv2'].dropna().values,
                'Local CV (CV$_2$)',
                'B.  Firing regularity (CV$_2$)')
    else:
        ax_b.text(0.5, 0.5, 'cv2 not available',
                  transform=ax_b.transAxes, ha='center', color='grey')
        ax_b.set_title('B.  Firing regularity (CV$_2$)',
                       fontweight='bold', loc='left')

    # ── C: Coverage ──────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    if 'coverage' in df.columns:
        _violin(ax_c,
                t['coverage'].dropna().values,
                z['coverage'].dropna().values,
                'Spatial coverage (frac.)',
                'C.  Spatial coverage')
    else:
        ax_c.text(0.5, 0.5, 'coverage not available',
                  transform=ax_c.transAxes, ha='center', color='grey')
        ax_c.set_title('C.  Spatial coverage',
                       fontweight='bold', loc='left')

    # ── D: Map stability ─────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    if 'xcorr_map' in df.columns:
        _violin(ax_d,
                t['xcorr_map'].dropna().values,
                z['xcorr_map'].dropna().values,
                'Map stability (split-half $r$)',
                'D.  Within-session map stability')
    else:
        ax_d.text(0.5, 0.5, 'xcorr_map not available',
                  transform=ax_d.transAxes, ha='center', color='grey')
        ax_d.set_title('D.  Within-session map stability',
                       fontweight='bold', loc='left')

    # ── E: Firing rate ───────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    if 'rate' in df.columns:
        _violin(ax_e,
                t['rate'].dropna().values,
                z['rate'].dropna().values,
                'Firing rate (Hz)',
                'E.  Firing rate',
                log=True)
    else:
        ax_e.text(0.5, 0.5, 'rate not available',
                  transform=ax_e.transAxes, ha='center', color='grey')
        ax_e.set_title('E.  Firing rate',
                       fontweight='bold', loc='left')

    # ── F: Fraction spatially selective ──────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    if 'spatially_selective' in df.columns and 'cell_type' in df.columns:
        groups = []
        xlabels = []
        colors  = []
        x_pos   = []

        for xi, (sp, sp_label, sp_color) in enumerate([
                ('titmouse',    'Chickadee', C_CHICK),
                ('zebra_finch', 'Zebra finch', C_FINCH)]):
            for ci, ct in enumerate(['E', 'I']):
                sub = df[(df['species'] == sp) & (df['cell_type'] == ct)]
                n_sel = sub['spatially_selective'].sum()
                n_tot = len(sub)
                frac  = n_sel / n_tot if n_tot > 0 else 0

                pos = xi * 2.5 + ci
                ax_f.bar(pos, frac, width=0.8, color=sp_color,
                         alpha=0.85 if ct == 'E' else 0.50,
                         edgecolor='white', linewidth=0.8)
                ax_f.text(pos, frac + 0.015,
                          f'{frac:.0%}\n({n_sel}/{n_tot})',
                          ha='center', va='bottom', fontsize=8)
                x_pos.append(pos)
                xlabels.append(f'{sp_label}\n{ct}')

        ax_f.set_xticks(x_pos)
        ax_f.set_xticklabels(xlabels, fontsize=8.5)
        ax_f.set_ylabel('Fraction spatially selective')
        ax_f.set_ylim(0, 1.0)
    else:
        ax_f.text(0.5, 0.5, 'cell_type / spatially_selective\nnot available',
                  transform=ax_f.transAxes, ha='center', color='grey')

    ax_f.set_title('F.  Spatial selectivity by species & type',
                    fontweight='bold', loc='left', pad=8)

    # ── Suptitle ─────────────────────────────────────────────────────
    n_t = len(t)
    n_z = len(z)
    fig.suptitle(
        'Species-Specific Spatial Coding Signatures',
        fontsize=14, fontweight='bold', y=0.97)
    fig.text(0.5, 0.935,
             f'Chickadee: $n = {n_t}$ units   |   '
             f'Zebra finch: $n = {n_z}$ units   |   '
             'Mann-Whitney $U$ (two-sided)',
             ha='center', fontsize=9.5, color='#555555')

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUTPUT_DIR,
                                 f'species_coding.{ext}'),
                    dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Saved to {OUTPUT_DIR}/species_coding.{{png,pdf}}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_data(data_path)
    make_figure(df)
    print("\nDone.")
