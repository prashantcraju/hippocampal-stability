#!/usr/bin/env python3
"""
fig_1.py — Main figure: physically-sorted RDMs + Mantel stats.

Panels (Laid out horizontally 1x4)
------
  A  Chickadee RDM under physical sort (best session)
  B  Finch RDM under physical sort (best session)
  C  Mantel r across all sessions — box + strip
  D  Mantel scatter: physical distance vs neural cosine distance 
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, mannwhitneyu
from pathlib import Path

# -- I/O --------------------------------------------------------------------
OUTPUT_DIR   = 'output/plots'
DEFAULT_DATA = 'data/aronov_dataset.pkl'
MANTEL_CSV   = 'output/tier1_valiant/tier1_mantel_sessions.csv'
SESSION_CSV  = 'output/tier1_valiant/tier1_session_results.csv'

# -- Config -----------------------------------------------------------------
MAX_BINS    = 300
GRID_SIZE   = 40
N_BINS      = GRID_SIZE ** 2
MIN_NEURONS = 5
RNG_SEED    = 320

# -- Aesthetics -------------------------------------------------------------
C_CHICK  = '#C62D50'
C_FINCH  = '#4A90C4'
CMAP_RDM = 'magma'

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.linewidth'    : 0.8,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'axes.labelsize'    : 10,
    'figure.dpi'        : 150,
    'savefig.dpi'       : 300,
})

# =========================================================================
# DATA LOADING & HELPERS
# =========================================================================

def load_precomputed(mantel_csv=MANTEL_CSV, session_csv=SESSION_CSV):
    def _try_read(path):
        if Path(path).exists():
            df = pd.read_csv(path)
            print(f"  Loaded {len(df)} rows from {path}")
            return df
        print(f"  Warning: {path} not found -- will recompute from raw data.")
        return pd.DataFrame(columns=['session', 'species', 'mantel_r'])
    return {'mantel': _try_read(mantel_csv), 'session': _try_read(session_csv)}

def load_sessions(path, min_neurons=MIN_NEURONS):
    df = pd.read_pickle(path)
    df_exc = df[df['cell_type'] == 'E'].copy()

    def _extract(df_sub):
        sessions = {}
        for sess in df_sub['session'].unique():
            sdf = df_sub[df_sub['session'] == sess]
            maps = []
            for _, row in sdf.iterrows():
                m = row['map']
                if isinstance(m, np.ndarray) and m.size == N_BINS:
                    maps.append(m.flatten())
            if len(maps) >= min_neurons:
                M = np.nan_to_num(np.vstack(maps), nan=0.0)
                sessions[sess] = {'M': M, 'n_neurons': len(maps)}
        return sessions

    s_t = _extract(df_exc[df_exc['species'] == 'titmouse'])
    s_z = _extract(df_exc[df_exc['species'] == 'zebra_finch'])
    print(f"  Loaded {len(s_t)} chickadee, {len(s_z)} finch sessions (>={min_neurons} neurons)")
    return s_t, s_z

def active_bins(M):
    n = M.shape[0]
    return np.where(np.sum(M > 0, axis=0) >= max(2, n // 3))[0]

def cosine_rdm(M_sub):
    X = M_sub.T
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    cos_sim = np.clip(Xn @ Xn.T, -1.0, 1.0)
    rdm = 1.0 - cos_sim
    np.fill_diagonal(rdm, 0.0)
    return rdm

def mantel(rdm, active_idx):
    rows = (active_idx // GRID_SIZE).astype(float)
    cols = (active_idx % GRID_SIZE).astype(float)
    coords = np.column_stack([rows, cols])
    phys_flat = pdist(coords, metric='euclidean')
    n = len(active_idx)
    iu = np.triu_indices(n, k=1)
    neur_flat = rdm[iu]
    r, p = spearmanr(phys_flat, neur_flat)
    return float(r), float(p), phys_flat, neur_flat

def subsample(active_idx, max_bins, rng):
    if len(active_idx) <= max_bins:
        return active_idx
    chosen = np.sort(rng.choice(len(active_idx), size=max_bins, replace=False))
    return active_idx[chosen]

def process_session(sess, data, max_bins, rng):
    idx = active_bins(data['M'])
    if len(idx) < 30:
        return None
    idx = subsample(idx, max_bins, rng)
    M_sub = data['M'][:, idx]
    rdm = cosine_rdm(M_sub)
    r_m, p_m, phys_flat, neur_flat = mantel(rdm, idx)
    return {
        'active_idx' : idx, 'rdm_phys' : rdm,
        'mantel_r' : r_m, 'mantel_p' : p_m,
        'phys_flat' : phys_flat, 'neur_flat' : neur_flat,
        'n_neurons' : data['n_neurons'], 'n_active' : len(idx),
        'session' : sess,
    }

def best_session(sessions, max_bins, rng, precomputed_mantel=None):
    if precomputed_mantel is not None and len(precomputed_mantel) > 0:
        canon = dict(zip(precomputed_mantel['session'], precomputed_mantel['mantel_r']))
    else:
        canon = {}
    best, best_r = None, -np.inf
    for sess, data in sessions.items():
        res = process_session(sess, data, max_bins, rng)
        if res is None: continue
        rank_r = canon.get(sess, res['mantel_r'])
        if rank_r > best_r:
            best_r = rank_r
            best = res
            if sess in canon:
                best['mantel_r'] = canon[sess]
    return best

def all_mantel_r(sessions, max_bins, rng, precomputed_mantel=None):
    if precomputed_mantel is not None and len(precomputed_mantel) > 0:
        canon = dict(zip(precomputed_mantel['session'], precomputed_mantel['mantel_r']))
    else:
        canon = {}
    rows = []
    for sess, data in sessions.items():
        res = process_session(sess, data, max_bins, rng)
        if res is not None:
            rows.append({
                'session' : sess, 'mantel_r' : canon.get(sess, res['mantel_r']),
                'n_neurons': res['n_neurons'],
            })
    return rows

# =========================================================================
# PLOTTING
# =========================================================================

def _rdm_scale(rdm, lo=2, hi=98):
    v = rdm.ravel()
    v = v[v > 1e-6]
    return (np.percentile(v, lo), np.percentile(v, hi)) if len(v) else (0, 1)

def _rdm_panel(ax, rdm, title, color, vmin, vmax, r_val, n_neurons, n_bins):
    im = ax.imshow(rdm, cmap=CMAP_RDM, vmin=vmin, vmax=vmax,
                   aspect='equal', interpolation='nearest', origin='upper')
    ax.set_xlabel('Spatial Bin (Arena Order)', fontsize=10)
    ax.set_ylabel('Spatial Bin (Arena Order)', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(color)
        sp.set_linewidth(2.0)

    text_str = f"Mantel $r = {r_val:.3f}$\n{n_neurons} neurons, {n_bins} bins"
    ax.text(0.04, 0.96, text_str, transform=ax.transAxes,
            ha='left', va='top', fontsize=9, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5, ec='none'))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label('Cosine Distance', fontsize=9)
    cb.outline.set_linewidth(0.8)
    cb.outline.set_edgecolor('#555555')
    cb.ax.tick_params(labelsize=8)

    return im

def _boxstrip(ax, vals_a, vals_b, col_a, col_b, ylabel, title, rng_j):
    bp = ax.boxplot([vals_a, vals_b], positions=[1, 2], widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1.2, color='#555555'),
                    capprops=dict(linewidth=1.2, color='#555555'))

    bp['boxes'][0].set(facecolor=col_a, alpha=0.6, edgecolor='none')
    bp['boxes'][1].set(facecolor=col_b, alpha=0.6, edgecolor='none')

    for vals, xpos, col in [(vals_a, 1, col_a), (vals_b, 2, col_b)]:
        jitter = rng_j.uniform(-0.12, 0.12, len(vals))
        ax.scatter(xpos + jitter, vals, color=col, s=35, alpha=0.7,
                   zorder=5, edgecolors='white', linewidths=0.6)

    ax.axhline(0, color='#AAAAAA', ls='--', lw=1, zorder=0)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Chickadee\n(all sessions)', 'Finch\n(all sessions)'], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    if len(vals_a) >= 3 and len(vals_b) >= 3:
        _, p = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
        ymax = max(max(vals_a, default=0), max(vals_b, default=0))
        y_bar = ymax + 0.05
        ax.plot([1, 2], [y_bar, y_bar], color='#333333', lw=1.2)
        ax.text(1.5, y_bar + 0.01, f'$p = {p:.4f}$', ha='center', va='bottom', fontsize=9)
        ax.set_ylim(bottom=-0.1, top=y_bar + 0.12)

def make_main_figure(chick, finch, chick_stats, finch_stats):
    fig = plt.figure(figsize=(20, 5.0), facecolor='white')

    outer = gridspec.GridSpec(1, 2, width_ratios=[2.1, 1.9], wspace=0.25,
                              left=0.03, right=0.98, top=0.88, bottom=0.15)

    left_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.35)
    ax_a = fig.add_subplot(left_gs[0])
    ax_b = fig.add_subplot(left_gs[1])

    right_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                                width_ratios=[0.8, 1.2], wspace=0.35)
    ax_c = fig.add_subplot(right_gs[0])
    ax_d = fig.add_subplot(right_gs[1])

    # -- Nature-style panel labels ------------------------------------------
    # A and B use tighter x-offset because make_axes_locatable shifts the bbox
    for ax, label, x_off in [
        (ax_a, 'b', -0.08),
        (ax_b, 'c', -0.08),
        (ax_c, 'd', -0.18),
        (ax_d, 'e', -0.12),
    ]:
        ax.text(x_off, 1.08, label,
                transform=ax.transAxes,
                fontsize=14, fontweight='bold',
                va='top', ha='left', clip_on=False)

    chick_vmin, chick_vmax = _rdm_scale(chick['rdm_phys'])
    finch_vmin, finch_vmax = _rdm_scale(finch['rdm_phys'])

    # -- A: Chickadee RDM ---------------------------------------------------
    _rdm_panel(ax_a, chick['rdm_phys'], "A. Chickadee (Physical Sort)",
               C_CHICK, chick_vmin, chick_vmax,
               chick['mantel_r'], chick['n_neurons'], chick['n_active'])

    # -- B: Finch RDM -------------------------------------------------------
    _rdm_panel(ax_b, finch['rdm_phys'], "B. Finch (Physical Sort)",
               C_FINCH, finch_vmin, finch_vmax,
               finch['mantel_r'], finch['n_neurons'], finch['n_active'])

    # -- C: Mantel Boxplot --------------------------------------------------
    c_mr = [s['mantel_r'] for s in chick_stats]
    f_mr = [s['mantel_r'] for s in finch_stats]
    rng_j = np.random.RandomState(320)
    _boxstrip(ax_c, c_mr, f_mr, C_CHICK, C_FINCH,
              'Mantel $r$\n(physical vs. neural distance)', 'C. Distance Correlation', rng_j)

    # -- D: Mantel Scatter --------------------------------------------------
    rng_sc = np.random.RandomState(320)
    for res, col, lbl in [
        (chick, C_CHICK, f'Chickadee ($r={chick["mantel_r"]:.3f}$)'),
        (finch, C_FINCH, f'Finch ($r={finch["mantel_r"]:.3f}$)')
    ]:
        ph = res['phys_flat']
        ne = res['neur_flat']
        n_sc = min(2000, len(ph))
        idx = rng_sc.choice(len(ph), n_sc, replace=False)

        ax_d.scatter(ph[idx], ne[idx], c=col, alpha=0.15, s=8,
                     edgecolors='none', rasterized=True)

        bins_e = np.linspace(ph.min(), ph.max(), 14)
        bx, by = [], []
        for i in range(len(bins_e) - 1):
            m = (ph >= bins_e[i]) & (ph < bins_e[i + 1])
            if m.sum() > 10:
                bx.append(0.5 * (bins_e[i] + bins_e[i + 1]))
                by.append(np.mean(ne[m]))

        ax_d.plot(bx, by, color='white', lw=4.5, zorder=9)
        ax_d.plot(bx, by, color=col, lw=2.5, zorder=10, label=lbl)

    ax_d.set_xlabel('Physical Distance (grid bins)', fontsize=10)
    ax_d.set_ylabel('Neural Cosine Distance', fontsize=10)
    ax_d.legend(fontsize=9, frameon=False, loc='upper left', handlelength=1.5)

    return fig

# =========================================================================
# MAIN
# =========================================================================

def main(data_path=None):
    data_path = data_path or DEFAULT_DATA
    if not Path(data_path).exists():
        print(f"Error: data file not found at {data_path}")
        sys.exit(1)

    print(f"Loading data from {data_path} ...")
    sessions_t, sessions_z = load_sessions(data_path)

    print(f"\nLoading pre-computed stats ...")
    precomp = load_precomputed()
    mdf = precomp['mantel']
    if 'species' in mdf.columns:
        mantel_t = mdf[mdf['species'].str.lower().str.contains('chick', na=False)]
        mantel_z = mdf[mdf['species'].str.lower().str.contains('finch|zebra', na=False, regex=True)]
    else:
        mantel_t = mdf
        mantel_z = mdf

    rng = np.random.RandomState(RNG_SEED)

    print("\nSelecting best session per species ...")
    chick_best = best_session(sessions_t, MAX_BINS, rng, precomputed_mantel=mantel_t)
    finch_best = best_session(sessions_z, MAX_BINS, rng, precomputed_mantel=mantel_z)

    if chick_best is None or finch_best is None:
        print("Error: could not find a valid session for one or both species.")
        sys.exit(1)

    print(f"  Chickadee: {chick_best['session']} ({chick_best['n_neurons']} neurons, Mantel r={chick_best['mantel_r']:.3f})")
    print(f"  Finch:     {finch_best['session']} ({finch_best['n_neurons']} neurons, Mantel r={finch_best['mantel_r']:.3f})")

    print("\nComputing per-session Mantel r ...")
    rng2 = np.random.RandomState(RNG_SEED)
    c_stats = all_mantel_r(sessions_t, MAX_BINS, rng2, precomputed_mantel=mantel_t)
    rng2 = np.random.RandomState(RNG_SEED)
    f_stats = all_mantel_r(sessions_z, MAX_BINS, rng2, precomputed_mantel=mantel_z)
    print(f"  {len(c_stats)} chickadee, {len(f_stats)} finch sessions")

    print("\nBuilding main figure ...")
    fig = make_main_figure(chick_best, finch_best, c_stats, f_stats)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, 'fig1_sorted_rdms_main_beautiful.png')
    out1 = os.path.join(OUTPUT_DIR, 'fig1_sorted_rdms_main_beautiful.pdf')
    fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(out1, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"\nSaved: {out1}")
    plt.show()

if __name__ == '__main__':
    main()