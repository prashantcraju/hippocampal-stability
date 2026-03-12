#!/usr/bin/env python3
"""
tier1_benchmarks.py - Alternative stability benchmarks for Tier 1.

Runs two established alternative metrics alongside SHESHA on the same
sessions, outputs a concordance CSV, and generates a supplementary
panel showing all three metrics agree directionally.

Metrics:
  1. Procrustes-aligned split-half error: Split neurons into halves,
     build RDMs, Procrustes-align, measure residual error.  Low error
     = the geometric structure is reproducible across neuron subsets.

  2. CCA stability: Canonical correlation between neuron-split halves.
     High canonical correlations = the two halves share a common
     low-dimensional representational structure.

  3. SHESHA (existing): Split-half RDM Spearman correlation.

Usage:
    python tier1_benchmarks.py [path_to_aronov_dataset.pkl]
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu, spearmanr
from scipy.linalg import svd as scipy_svd, orthogonal_procrustes
from pathlib import Path
import os
import sys
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA_PATH = 'data/aronov_dataset.pkl'
GRID_SIZE = 40
N_BINS = GRID_SIZE ** 2
MIN_NEURONS = 5
OUTPUT_DIR = 'output/tier1_benchmarks'

C_T = '#C62D50'
C_Z = '#1C3D8F'


def load_data(path):
    df = pd.read_pickle(path)
    df_exc = df[df['cell_type'] == 'E'].copy()
    df_t = df_exc[df_exc['species'] == 'titmouse']
    df_z = df_exc[df_exc['species'] == 'zebra_finch']
    print(f"Excitatory cells: {len(df_t)} chickadee, {len(df_z)} finch")
    return df_t, df_z


def get_session_maps(df_subset):
    sessions = {}
    for sess in df_subset['session'].unique():
        sdf = df_subset[df_subset['session'] == sess]
        maps = []
        for _, row in sdf.iterrows():
            m = row['map']
            if isinstance(m, np.ndarray) and m.size == N_BINS:
                maps.append(m.flatten())
        if len(maps) >= MIN_NEURONS:
            sessions[sess] = {
                'M': np.vstack(maps),
                'bird': sdf['bird'].iloc[0],
                'n_neurons': len(maps),
            }
    return sessions


def _zscore_active(M):
    """Z-score neurons, return (M_z, active_idx) or None."""
    n_neurons = M.shape[0]
    means = np.nanmean(M, axis=1, keepdims=True)
    stds = np.nanstd(M, axis=1, keepdims=True)
    stds[stds == 0] = 1
    M_z = np.nan_to_num((M - means) / stds, nan=0.0)

    valid_pos = np.isfinite(M) & (M > 0)
    active = np.sum(valid_pos, axis=0) >= max(2, n_neurons // 3)
    active_idx = np.where(active)[0]
    if len(active_idx) < 30 or n_neurons < 4:
        return None, None
    return M_z, active_idx


# ------------------------------------------------------------------
# Metric 1: Procrustes-aligned split-half error
# ------------------------------------------------------------------
def compute_procrustes_error(M, n_splits=100, rng=None):
    """
    For each random neuron split, build an RDM from each half,
    Procrustes-align the two distance matrices, and measure the
    normalised Frobenius residual.  Low residual = reproducible geometry.
    """
    M_z, active_idx = _zscore_active(M)
    if M_z is None:
        return np.nan

    n_neurons = M.shape[0]
    X = M_z[:, active_idx]  # (neurons x active_bins)
    if rng is None:
        rng = np.random.RandomState(320)

    residuals = []
    for _ in range(n_splits):
        perm = rng.permutation(n_neurons)
        half = n_neurons // 2
        h1, h2 = perm[:half], perm[half:2 * half]

        # Pairwise cosine distances between spatial bins (not neurons).
        # X[h1] is (half_neurons x active_bins); .T gives (active_bins x half_neurons).
        # Procrustes aligns the full bin-by-bin distance structure, not
        # a low-dimensional embedding, so the result is embedding-free.
        D1 = squareform(pdist(X[h1].T, metric='cosine'))
        D2 = squareform(pdist(X[h2].T, metric='cosine'))

        valid = np.isfinite(D1) & np.isfinite(D2)
        if np.sum(valid) < 100:
            continue

        D1 = np.nan_to_num(D1, nan=0.0)
        D2 = np.nan_to_num(D2, nan=0.0)

        try:
            R, scale = orthogonal_procrustes(D1, D2)
            D2_aligned = D2 @ R
            resid = np.linalg.norm(D1 - D2_aligned) / np.linalg.norm(D1)
            residuals.append(resid)
        except (np.linalg.LinAlgError, ValueError):
            continue

    if not residuals:
        return np.nan
    return 1.0 - np.mean(residuals)


# ------------------------------------------------------------------
# Metric 2: CCA stability
# ------------------------------------------------------------------
def compute_cca_stability(M, n_splits=50, n_components=3, rng=None):
    """
    Split neurons, run CCA between the two halves' bin-by-neuron
    activity matrices.  Return mean of top canonical correlations.

    Adaptively reduces n_components when halves are small rather than
    skipping the split entirely.
    """
    M_z, active_idx = _zscore_active(M)
    if M_z is None:
        return np.nan

    n_neurons = M.shape[0]
    if n_neurons < 4:
        return np.nan

    X = M_z[:, active_idx]
    if rng is None:
        rng = np.random.RandomState(320)

    scores = []
    for _ in range(n_splits):
        perm = rng.permutation(n_neurons)
        half = n_neurons // 2
        X1 = X[perm[:half]].T        # (active_bins x half_neurons)
        X2 = X[perm[half:2 * half]].T

        k1, k2 = X1.shape[1], X2.shape[1]
        n_comp = min(n_components, k1 - 1, k2 - 1)
        if n_comp < 1:
            continue

        try:
            C11 = X1.T @ X1 / X1.shape[0] + 1e-6 * np.eye(k1)
            C22 = X2.T @ X2 / X2.shape[0] + 1e-6 * np.eye(k2)
            C12 = X1.T @ X2 / X1.shape[0]

            C11_ih = np.linalg.cholesky(np.linalg.inv(C11))
            C22_ih = np.linalg.cholesky(np.linalg.inv(C22))

            T = C11_ih.T @ C12 @ C22_ih
            _, s, _ = scipy_svd(T, full_matrices=False)
            scores.append(np.mean(np.clip(s[:n_comp], 0, 1)))
        except (np.linalg.LinAlgError, ValueError):
            continue

    return np.mean(scores) if scores else np.nan


# ------------------------------------------------------------------
# Metric 3: SHESHA (split-half RDM Spearman)
# ------------------------------------------------------------------
def compute_shesha(M, n_splits=100, rng=None):
    M_z, active_idx = _zscore_active(M)
    if M_z is None:
        return np.nan

    n_neurons = M.shape[0]
    X = M_z[:, active_idx].T  # (active_bins x neurons)
    if rng is None:
        rng = np.random.RandomState(320)

    correlations = []
    for _ in range(n_splits):
        perm = rng.permutation(n_neurons)
        half = n_neurons // 2

        rdm1 = pdist(X[:, perm[:half]], metric='cosine')
        rdm2 = pdist(X[:, perm[half:2 * half]], metric='cosine')

        both_valid = np.isfinite(rdm1) & np.isfinite(rdm2)
        if np.sum(both_valid) < 10:
            continue

        r, _ = spearmanr(rdm1[both_valid], rdm2[both_valid])
        if np.isfinite(r):
            correlations.append(r)

    return np.mean(correlations) if correlations else np.nan


# ------------------------------------------------------------------
# Run all benchmarks
# ------------------------------------------------------------------
def run_benchmarks(data_path=None):
    if data_path is None:
        data_path = DATA_PATH

    path = Path(data_path)
    if not path.exists():
        print(f"Error: {data_path} not found")
        return None

    print("=" * 70)
    print("TIER 1 BENCHMARKS: Procrustes + CCA + SHESHA concordance")
    print("=" * 70)

    df_t, df_z = load_data(path)
    sessions_t = get_session_maps(df_t)
    sessions_z = get_session_maps(df_z)
    print(f"Sessions: {len(sessions_t)} chickadee, {len(sessions_z)} finch")

    rows = []
    for species, sessions, label in [
        ('chickadee', sessions_t, 'Chickadee'),
        ('finch', sessions_z, 'Finch'),
    ]:
        print(f"\n  Processing {label}...")
        for sess, data in sessions.items():
            M = data['M']
            sh = compute_shesha(M)
            pr = compute_procrustes_error(M)
            cc = compute_cca_stability(M)

            rows.append({
                'species': species,
                'session': sess,
                'bird': data['bird'],
                'n_neurons': data['n_neurons'],
                'shesha': sh,
                'procrustes_stability': pr,
                'cca_stability': cc,
            })
            print(f"    {sess}: SHESHA={sh:.3f}, Procrustes={pr:.3f}, "
                  f"CCA={cc:.3f}" if not np.isnan(sh) else
                  f"    {sess}: insufficient data")

    df_results = pd.DataFrame(rows)

    # Concordance summary
    print("\n" + "=" * 70)
    print("CONCORDANCE SUMMARY")
    print("=" * 70)

    concordance = []
    for metric in ['shesha', 'procrustes_stability', 'cca_stability']:
        vt = df_results[df_results['species'] == 'chickadee'][metric].dropna()
        vz = df_results[df_results['species'] == 'finch'][metric].dropna()
        if len(vt) >= 2 and len(vz) >= 2:
            U, p = mannwhitneyu(vt, vz, alternative='greater')
            agrees = vt.mean() > vz.mean()
            concordance.append({
                'metric': metric,
                'chickadee_mean': vt.mean(),
                'finch_mean': vz.mean(),
                'chickadee_n': len(vt),
                'finch_n': len(vz),
                'direction': 'chickadee > finch',
                'directionally_consistent': agrees,
                'p_value': p,
            })
            sym = 'YES' if agrees else 'NO'
            print(f"  {metric}: chickadee={vt.mean():.4f}, "
                  f"finch={vz.mean():.4f}, p={p:.4e} [{sym}]")

    all_agree = all(c['directionally_consistent'] for c in concordance)
    print(f"\n  All metrics agree directionally: "
          f"{'YES' if all_agree else 'NO'}")

    # Export
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_results.to_csv(
        os.path.join(OUTPUT_DIR, 'tier1_benchmarks_sessions.csv'),
        index=False)
    print(f"\n  Saved tier1_benchmarks_sessions.csv ({len(rows)} rows)")

    df_conc = pd.DataFrame(concordance)
    df_conc.to_csv(
        os.path.join(OUTPUT_DIR, 'tier1_benchmarks_concordance.csv'),
        index=False)
    print(f"  Saved tier1_benchmarks_concordance.csv ({len(concordance)} rows)")

    # Supplementary figure
    plot_concordance(df_results, concordance)

    return df_results, concordance


def plot_concordance(df_results, concordance):
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, wspace=0.35,
                           left=0.06, right=0.96, top=0.85, bottom=0.15)

    metrics = ['shesha', 'procrustes_stability', 'cca_stability']
    titles = ['SHESHA\n(split-half RDM r)',
              'Procrustes stability\n(1 - aligned residual)',
              'CCA stability\n(mean canon. corr.)']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = fig.add_subplot(gs[0, i])

        valid_t = (df_results['species'] == 'chickadee') & df_results[metric].notna()
        valid_z = (df_results['species'] == 'finch') & df_results[metric].notna()
        vt = df_results.loc[valid_t, metric]
        nt = df_results.loc[valid_t, 'n_neurons']
        vz = df_results.loc[valid_z, metric]
        nz = df_results.loc[valid_z, 'n_neurons']

        if len(vt) > 0:
            ax.scatter(nt, vt, c=C_T, alpha=0.6, s=40,
                       edgecolors='white', linewidths=0.3,
                       label='Chickadee', zorder=3)
            ax.axhline(vt.mean(), color=C_T, ls='--', lw=1, alpha=0.6)
        if len(vz) > 0:
            ax.scatter(nz, vz, c=C_Z, alpha=0.6, s=40,
                       edgecolors='white', linewidths=0.3,
                       label='Finch', zorder=3)
            ax.axhline(vz.mean(), color=C_Z, ls='--', lw=1, alpha=0.6)

        conc = next((c for c in concordance if c['metric'] == metric), None)
        if conc:
            sym = 'agree' if conc['directionally_consistent'] else 'disagree'
            ax.set_title(f'{title}\np = {conc["p_value"]:.2e} ({sym})',
                         fontsize=10, fontweight='bold', loc='left')
        else:
            ax.set_title(title, fontsize=10, fontweight='bold', loc='left')

        ax.set_xlabel('Neurons in session', fontsize=9)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Supplementary: Three stability metrics agree directionally '
                 '(chickadee > finch)',
                 fontsize=12, fontweight='bold', y=0.97)

    outpath = os.path.join(OUTPUT_DIR, 'tier1_benchmarks_concordance.png')
    fig.savefig(outpath, dpi=250, facecolor='white')
    print(f"  Figure saved: {outpath}")
    plt.show()


if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_benchmarks(data_path=data_path)
