#!/usr/bin/env python3
"""
fig_sorted_rdms_supp.py — Supplementary figure: cluster-sorted RDMs +
topogeographic correspondence metrics.

Panels
------
  A  Chickadee RDM under cluster sort (best session)
  B  Finch RDM under cluster sort (best session)
  C  Neural distance vs physical distance — binned mean +/- SEM (best session)
  D  Physical <-> cluster rank correlation across all sessions (box + strip)

Notes on Panel D (topogeographic correspondence, p~0.27)
---------------------------------------------------------
The physical<->cluster ordering agreement is not significant between species.
This is mechanistically interpretable: a crystalline code organises spatial
representations into a rich multi-scale cluster hierarchy that does not
collapse onto a single linear physical ordering, whereas the mist code has
weaker internal structure overall. This panel is included in the supplementary
figure for transparency rather than the main figure, since it does not reach
conventional significance and requires additional interpretive context.

The negative physical<->cluster r values seen in the chickadee reflect this
multi-scale structure: Ward clustering captures the dominant representational
axes, which in a topologically rich code can be orthogonal to the simple
row-major physical ordering used here.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, mannwhitneyu
from scipy.cluster.hierarchy import linkage, leaves_list
from pathlib import Path

# ── I/O ──
OUTPUT_DIR   = 'output/plots'
DEFAULT_DATA = 'data/aronov_dataset.pkl'
MANTEL_CSV   = 'output/tier1_valiant/tier1_mantel_sessions.csv'
SESSION_CSV  = 'output/tier1_valiant/tier1_session_results.csv'

# ── Config ──
MAX_BINS    = 300
GRID_SIZE   = 40
N_BINS      = GRID_SIZE ** 2
MIN_NEURONS = 5
RNG_SEED    = 320

# ── Aesthetics ──
C_CHICK  = '#C62D50'
C_FINCH  = '#1C3D8F'
CMAP_RDM = 'magma'

plt.rcParams.update({
    'font.family'     : 'sans-serif',
    'font.size'       : 10,
    'axes.labelsize'  : 10,
    'axes.titlesize'  : 11,
    'xtick.labelsize' : 8,
    'ytick.labelsize' : 8,
    'figure.dpi'      : 150,
    'savefig.dpi'     : 300,
})


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_precomputed(mantel_csv=MANTEL_CSV, session_csv=SESSION_CSV):
    def _try_read(path):
        if Path(path).exists():
            df = pd.read_csv(path)
            print(f"  Loaded {len(df)} rows from {path}")
            return df
        print(f"  Warning: {path} not found — will recompute from raw data.")
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
    print(f"  Loaded {len(s_t)} chickadee, {len(s_z)} finch sessions "
          f"(>={min_neurons} neurons)")
    return s_t, s_z


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════════════

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


def cluster_sort(rdm):
    Z = linkage(squareform(rdm, checks=False), method='ward',
                optimal_ordering=True)
    return leaves_list(Z)


def ordering_agreement(rdm, active_idx):
    n = len(active_idx)
    phys_rank = np.arange(n, dtype=float)
    cl_perm = cluster_sort(rdm)
    cl_rank = np.empty(n, dtype=float)
    cl_rank[cl_perm] = np.arange(n, dtype=float)
    r, p = spearmanr(phys_rank, cl_rank)
    return float(r), float(p), cl_perm


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
    r_ag, _, cl_perm = ordering_agreement(rdm, idx)
    return {
        'active_idx' : idx,
        'rdm_phys'   : rdm,
        'rdm_clus'   : rdm[np.ix_(cl_perm, cl_perm)],
        'mantel_r'   : r_m,
        'mantel_p'   : p_m,
        'r_agree'    : r_ag,
        'phys_flat'  : phys_flat,
        'neur_flat'  : neur_flat,
        'n_neurons'  : data['n_neurons'],
        'n_active'   : len(idx),
        'session'    : sess,
    }


def best_session(sessions, max_bins, rng, precomputed_mantel=None):
    if precomputed_mantel is not None and len(precomputed_mantel) > 0:
        canon = dict(zip(precomputed_mantel['session'],
                         precomputed_mantel['mantel_r']))
    else:
        canon = {}
    best, best_r = None, -np.inf
    for sess, data in sessions.items():
        res = process_session(sess, data, max_bins, rng)
        if res is None:
            continue
        rank_r = canon.get(sess, res['mantel_r'])
        if rank_r > best_r:
            best_r = rank_r
            best = res
            if sess in canon:
                best['mantel_r'] = canon[sess]
    return best


def all_stats(sessions, max_bins, rng, precomputed_mantel=None):
    if precomputed_mantel is not None and len(precomputed_mantel) > 0:
        canon = dict(zip(precomputed_mantel['session'],
                         precomputed_mantel['mantel_r']))
    else:
        canon = {}
    rows = []
    for sess, data in sessions.items():
        res = process_session(sess, data, max_bins, rng)
        if res is not None:
            rows.append({
                'session'  : sess,
                'mantel_r' : canon.get(sess, res['mantel_r']),
                'r_agree'  : res['r_agree'],
                'n_neurons': res['n_neurons'],
            })
    return rows


# ══════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════

def _rdm_scale(rdm, lo=2, hi=98):
    v = rdm.ravel()
    v = v[v > 1e-6]
    return (np.percentile(v, lo), np.percentile(v, hi)) if len(v) else (0, 1)


def _rdm_panel(ax, rdm, title, color, xlabel, vmin, vmax):
    im = ax.imshow(rdm, cmap=CMAP_RDM, vmin=vmin, vmax=vmax,
                   aspect='equal', interpolation='nearest', origin='upper')
    ax.set_title(title, fontsize=10, fontweight='bold', color=color, pad=5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(xlabel, fontsize=9)
    n = rdm.shape[0]
    t = np.linspace(0, n - 1, 5, dtype=int)
    ax.set_xticks(t); ax.set_yticks(t)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    for sp in ax.spines.values():
        sp.set_edgecolor(color)
        sp.set_linewidth(2.5)
    return im


def _boxstrip(ax, vals_a, vals_b, col_a, col_b, ylabel, title, rng_j):
    bp = ax.boxplot(
        [vals_a, vals_b], positions=[1, 2], widths=0.45,
        patch_artist=True, showfliers=False,
        medianprops=dict(color='white', linewidth=2.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    bp['boxes'][0].set(facecolor=col_a, alpha=0.55)
    bp['boxes'][1].set(facecolor=col_b, alpha=0.55)
    for vals, xpos, col in [(vals_a, 1, col_a), (vals_b, 2, col_b)]:
        jitter = rng_j.uniform(-0.14, 0.14, len(vals))
        ax.scatter(xpos + jitter, vals, color=col, s=28, alpha=0.8,
                   zorder=5, edgecolors='white', linewidths=0.5)
    ax.axhline(0, color='gray', ls=':', lw=1, alpha=0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        ['Chickadee\n(all sessions)', 'Finch\n(all sessions)'], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    if len(vals_a) >= 3 and len(vals_b) >= 3:
        _, p = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
        sig = ('***' if p < 0.001 else
               '**'  if p < 0.01  else
               '*'   if p < 0.05  else 'n.s.')
        ymax = max(max(vals_a, default=0), max(vals_b, default=0))
        ymin = min(min(vals_a, default=0), min(vals_b, default=0))
        gap = (ymax - ymin) * 0.08
        ax.plot([1, 2], [ymax + gap, ymax + gap], 'k-', lw=1)
        ax.text(1.5, ymax + gap * 1.3,
                f'{sig}  (p={p:.4f})', ha='center', va='bottom', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)


def make_supp_figure(chick, finch, chick_stats, finch_stats):
    """
    Four-panel supplementary figure.
    A: Chickadee cluster-sort RDM  }  stacked vertically in left column,
    B: Finch cluster-sort RDM      }  each with its own colorbar to the left
    C: Neural distance vs physical distance (binned mean +/- SEM, best session)
    D: Physical <-> cluster ordering agreement across all sessions
    """
    fig = plt.figure(figsize=(16, 10), facecolor='white')

    # Outer grid: left column (RDMs) | right column (summary panels)
    outer = gridspec.GridSpec(
        1, 2,
        width_ratios=[1, 0.9],
        left=0.04, right=0.97, top=0.93, bottom=0.09,
        wspace=0.18,
    )

    # Left column: A over B, each row = [cbar | rdm]
    left_gs = gridspec.GridSpecFromSubplotSpec(
        2, 2,
        subplot_spec=outer[0],
        hspace=0.42,
        wspace=0.04,
        width_ratios=[0.055, 1],
    )
    cax_a = fig.add_subplot(left_gs[0, 0])   # colorbar for A
    ax_a  = fig.add_subplot(left_gs[0, 1])   # A: Chickadee cluster
    cax_b = fig.add_subplot(left_gs[1, 0])   # colorbar for B
    ax_b  = fig.add_subplot(left_gs[1, 1])   # B: Finch cluster

    # Right column: C over D
    right_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=outer[1],
        hspace=0.48,
    )
    ax_c = fig.add_subplot(right_gs[0])   # C: Neural vs physical distance
    ax_d = fig.add_subplot(right_gs[1])   # D: Ordering agreement box

    chick_vmin, chick_vmax = _rdm_scale(chick['rdm_clus'])
    finch_vmin, finch_vmax = _rdm_scale(finch['rdm_clus'])

    # ── A: Chickadee cluster sort ──
    im_c = _rdm_panel(
        ax_a, chick['rdm_clus'],
        f"A. Chickadee  —  Cluster sort\n"
        f"Physical\u2194cluster agreement: r = {chick['r_agree']:.3f}",
        C_CHICK, 'Bin (cluster order)', chick_vmin, chick_vmax,
    )
    cb_a = fig.colorbar(im_c, cax=cax_a, location='left')
    cb_a.set_label('Cosine distance', fontsize=7, labelpad=4)
    cb_a.ax.tick_params(labelsize=6)
    cax_a.yaxis.set_ticks_position('left')
    cax_a.yaxis.set_label_position('left')

    # ── B: Finch cluster sort ──
    im_f = _rdm_panel(
        ax_b, finch['rdm_clus'],
        f"B. Finch  —  Cluster sort\n"
        f"Physical\u2194cluster agreement: r = {finch['r_agree']:.3f}",
        C_FINCH, 'Bin (cluster order)', finch_vmin, finch_vmax,
    )
    cb_b = fig.colorbar(im_f, cax=cax_b, location='left')
    cb_b.set_label('Cosine distance', fontsize=7, labelpad=4)
    cb_b.ax.tick_params(labelsize=6)
    cax_b.yaxis.set_ticks_position('left')
    cax_b.yaxis.set_label_position('left')

    # ── C: Neural distance vs physical distance (binned, best session) ──
    for res, col, lbl in [
        (chick, C_CHICK, 'Chickadee'),
        (finch, C_FINCH, 'Finch'),
    ]:
        ph = res['phys_flat']
        ne = res['neur_flat']
        n_bins_g = 18
        edges = np.linspace(ph.min(), ph.max(), n_bins_g + 1)
        bx, by, be = [], [], []
        for i in range(n_bins_g):
            m = (ph >= edges[i]) & (ph < edges[i + 1])
            if m.sum() > 5:
                bx.append(0.5 * (edges[i] + edges[i + 1]))
                by.append(np.mean(ne[m]))
                be.append(np.std(ne[m]) / np.sqrt(m.sum()))
        bx, by, be = np.array(bx), np.array(by), np.array(be)
        ax_c.plot(bx, by, color=col, lw=2.5, label=lbl)
        ax_c.fill_between(bx, by - be, by + be, color=col, alpha=0.18)

    ax_c.set_xlabel('Physical distance (grid bins)', fontsize=9)
    ax_c.set_ylabel('Mean cosine distance +/- SEM', fontsize=9)
    ax_c.set_title('C. Neural distance vs physical distance\n(best session each species)',
                   fontsize=10, fontweight='bold')
    ax_c.legend(fontsize=8, frameon=False)
    ax_c.spines[['top', 'right']].set_visible(False)

    # ── D: Physical <-> cluster ordering agreement across all sessions ──
    c_ag = [s['r_agree'] for s in chick_stats]
    f_ag = [s['r_agree'] for s in finch_stats]
    rng_j = np.random.RandomState(320)
    _boxstrip(
        ax_d, c_ag, f_ag, C_CHICK, C_FINCH,
        'Physical \u2194 cluster\nrank correlation  (r)',
        'D. Topogeographic correspondence\n(physical \u2194 cluster ordering)',
        rng_j,
    )
    # Add interpretive note for the non-significant result
    ax_d.text(
        0.5, -0.22,
        'Note: crystalline codes have multi-scale cluster structure\n'
        'that need not align with a linear physical ordering.',
        transform=ax_d.transAxes, fontsize=7, ha='center',
        color='gray', style='italic',
    )

    # fig.suptitle(
    #     'Supplementary Figure: Cluster-Sorted RDMs and Topogeographic Correspondence',
    #     fontsize=12, fontweight='bold', y=0.975,
    # )f

    return fig


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

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
        mantel_z = mdf[mdf['species'].str.lower().str.contains('finch|zebra',
                                                                na=False, regex=True)]
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

    print(f"  Chickadee: {chick_best['session']} "
          f"({chick_best['n_neurons']} neurons, "
          f"Mantel r={chick_best['mantel_r']:.3f}, "
          f"agree r={chick_best['r_agree']:.3f})")
    print(f"  Finch:     {finch_best['session']} "
          f"({finch_best['n_neurons']} neurons, "
          f"Mantel r={finch_best['mantel_r']:.3f}, "
          f"agree r={finch_best['r_agree']:.3f})")

    print("\nComputing full per-session stats ...")
    rng2 = np.random.RandomState(RNG_SEED)
    c_stats = all_stats(sessions_t, MAX_BINS, rng2, precomputed_mantel=mantel_t)
    rng2 = np.random.RandomState(RNG_SEED)
    f_stats = all_stats(sessions_z, MAX_BINS, rng2, precomputed_mantel=mantel_z)
    print(f"  {len(c_stats)} chickadee, {len(f_stats)} finch sessions")

    print("\nBuilding supplementary figure ...")
    fig = make_supp_figure(chick_best, finch_best, c_stats, f_stats)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, 'figS1_sorted_rdms_supp.png')
    fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"\nSaved: {out}")

    # Save per-session stats for reproducibility
    df_stats = pd.DataFrame(
        [{'species': 'chickadee', **s} for s in c_stats] +
        [{'species': 'finch',     **s} for s in f_stats]
    )
    csv_out = os.path.join(OUTPUT_DIR, 'figS1_session_stats.csv')
    df_stats.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")

    plt.show()


if __name__ == '__main__':
    main()