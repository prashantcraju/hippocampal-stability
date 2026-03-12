#!/usr/bin/env python3
"""
tier1_valiant_shesha.py - Revised Tier 1 hypothesis tests.

Key revision from v1:
  The original Valiant-stability metric (CV of N_active across bins)
  actually measures SPATIAL TUNING SHARPNESS, not allocation stability.
  Chickadees have higher CV *because* they have better place fields,
  not because their allocation is less stable.

  Revised Valiant metrics:
  (a) Place field size consistency: For each neuron with a place field,
      measure the field area. Low CV = each neuron is allocated a
      consistent-sized territory (SMA prediction).
  (b) Population overlap: For identified place fields, how many neurons
      are co-active? Low CV = each location recruits a consistent
      ensemble size (SMA prediction).
  (c) Split-half N_active reliability: Split neurons into halves,
      count N_active per bin in each half. High correlation = the
      allocation is reproducible, not noisy.

Usage:
    python tier1_valiant_shesha.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu, spearmanr, sem as sp_sem
from scipy.ndimage import label as ndimage_label, gaussian_filter
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA_PATH = 'data/aronov_dataset.pkl'
GRID_SIZE = 40
N_BINS = GRID_SIZE ** 2
MIN_NEURONS = 5

OUTPUT_DIR = 'output/tier1_valiant'

C_T = '#C62D50'
C_Z = '#1C3D8F'
C_T2 = '#E8909F'
C_Z2 = '#7B9BD4'


def load_data(path):
    df = pd.read_pickle(path)
    df_exc = df[df['cell_type'] == 'E'].copy()
    df_t = df_exc[df_exc['species'] == 'titmouse']
    df_z = df_exc[df_exc['species'] == 'zebra_finch']
    print(f"Excitatory cells: {len(df_t)} chickadee, {len(df_z)} finch")
    return df, df_t, df_z


def get_session_maps(df_subset, min_neurons=MIN_NEURONS):
    sessions = {}
    for sess in df_subset['session'].unique():
        sdf = df_subset[df_subset['session'] == sess]
        maps = []
        for _, row in sdf.iterrows():
            m = row['map']
            if isinstance(m, np.ndarray) and m.size == N_BINS:
                maps.append(m.flatten())
        if len(maps) >= min_neurons:
            sessions[sess] = {
                'M': np.vstack(maps),
                'bird': sdf['bird'].iloc[0],
                'n_neurons': len(maps),
                'subdivision': sdf['subdivision'].iloc[0]
                    if 'subdivision' in sdf.columns else None,
            }
    return sessions


# ======================================================================
# MASKED DISTANCE FUNCTIONS (pairwise deletion for NaN bins)
# ======================================================================
def masked_cosine_pdist(X, min_valid=2):
    """
    Pairwise cosine distance with NaN masking (pairwise deletion).

    For each pair of rows, only dimensions where both rows are finite
    contribute.  Returns a condensed distance vector (upper triangle).
    """
    n, d = X.shape
    valid = np.isfinite(X).astype(np.float64)
    X_0 = np.where(np.isfinite(X), X, 0.0)

    dots = X_0 @ X_0.T
    X_sq = X_0 ** 2
    norm_sq_left = X_sq @ valid.T
    norm_sq_right = valid @ X_sq.T
    n_valid = valid @ valid.T

    denom = np.sqrt(norm_sq_left) * np.sqrt(norm_sq_right)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_sim = np.where(denom > 0, dots / denom, 0.0)

    cos_dist = 1.0 - cos_sim
    cos_dist[n_valid < min_valid] = np.nan
    np.fill_diagonal(cos_dist, 0.0)

    idx = np.triu_indices(n, k=1)
    return cos_dist[idx]


def masked_euclidean_pdist(X, min_valid=2):
    """
    Pairwise RMSD with NaN masking (pairwise deletion).

    Computes sqrt(mean_valid((x_i - x_j)^2)) for each pair, averaging
    only over mutually valid dimensions.
    """
    n, d = X.shape
    valid = np.isfinite(X).astype(np.float64)
    X_0 = np.where(np.isfinite(X), X, 0.0)

    n_valid = valid @ valid.T
    X_sq = X_0 ** 2
    term1 = X_sq @ valid.T
    term2 = valid @ X_sq.T
    dots = X_0 @ X_0.T

    sq_dist = term1 + term2 - 2 * dots
    sq_dist = np.maximum(sq_dist, 0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        dist = np.sqrt(sq_dist / np.maximum(n_valid, 1))

    dist[n_valid < min_valid] = np.nan
    np.fill_diagonal(dist, 0.0)

    idx = np.triu_indices(n, k=1)
    return dist[idx]


# ======================================================================
# SHESHA (split-half RDM stability with pairwise NaN deletion)
# ======================================================================
def compute_shesha(M, n_splits=100, rng=None):
    """
    Split-half RDM correlation across neuron halves.

    Splits the neuron population into random halves, builds a pairwise
    cosine-distance RDM from each half, and correlates (Spearman) the
    two RDMs.  Uses masked cosine distance so that unvisited spatial
    bins (NaN) are excluded per pair rather than zero-filled.
    """
    n_neurons, n_bins = M.shape

    means = np.nanmean(M, axis=1, keepdims=True)
    stds = np.nanstd(M, axis=1, keepdims=True)
    stds[stds == 0] = 1
    M_z = (M - means) / stds

    valid_positive = np.isfinite(M) & (M > 0)
    active = np.sum(valid_positive, axis=0) >= max(2, n_neurons // 3)
    active_idx = np.where(active)[0]
    if len(active_idx) < 30 or n_neurons < 4:
        return np.nan, np.nan

    X = M_z[:, active_idx].T  # (active_bins x neurons)

    if rng is None:
        rng = np.random.RandomState(320)

    correlations = []
    for _ in range(n_splits):
        perm = rng.permutation(n_neurons)
        half = n_neurons // 2
        h1 = perm[:half]
        h2 = perm[half:2 * half]

        rdm1 = masked_cosine_pdist(X[:, h1])
        rdm2 = masked_cosine_pdist(X[:, h2])

        both_valid = np.isfinite(rdm1) & np.isfinite(rdm2)
        if np.sum(both_valid) < 10:
            continue

        r, _ = spearmanr(rdm1[both_valid], rdm2[both_valid])
        if np.isfinite(r):
            correlations.append(r)

    if not correlations:
        return np.nan, np.nan

    return np.mean(correlations), np.nan


# ======================================================================
# ALTERNATIVE STABILITY BENCHMARKS
# ======================================================================
def compute_test_retest_pv_correlation(M, n_splits=100, rng=None):
    """
    Test-retest population vector correlation.

    Splits neurons into random halves, computes the population vector
    (mean firing rate per spatial bin) for each half, and correlates
    (Spearman) the two PV profiles.  High r = stable spatial code.
    """
    n_neurons, n_bins = M.shape
    if n_neurons < 4:
        return np.nan

    M_filled = np.nan_to_num(M, nan=0.0)
    if rng is None:
        rng = np.random.RandomState(320)

    correlations = []
    for _ in range(n_splits):
        perm = rng.permutation(n_neurons)
        half = n_neurons // 2
        h1, h2 = perm[:half], perm[half:2 * half]

        pv1 = np.mean(M_filled[h1], axis=0)
        pv2 = np.mean(M_filled[h2], axis=0)

        active = (pv1 > 0) | (pv2 > 0)
        if np.sum(active) < 30:
            continue

        r, _ = spearmanr(pv1[active], pv2[active])
        if np.isfinite(r):
            correlations.append(r)

    if not correlations:
        return np.nan
    return np.mean(correlations)


def compute_cca_stability(M, n_splits=50, n_components=3, rng=None):
    """
    Canonical Correlation Analysis stability across neuron splits.

    Splits neurons into halves, runs CCA between the two halves'
    activity matrices, and returns the mean of the top canonical
    correlations.  High CCA = the two halves share a common low-
    dimensional structure.
    """
    n_neurons, n_bins = M.shape
    if n_neurons < 4:
        return np.nan

    means = np.nanmean(M, axis=1, keepdims=True)
    stds = np.nanstd(M, axis=1, keepdims=True)
    stds[stds == 0] = 1
    M_z = np.nan_to_num((M - means) / stds, nan=0.0)

    valid_positive = np.isfinite(M) & (M > 0)
    active = np.sum(valid_positive, axis=0) >= max(2, n_neurons // 3)
    active_idx = np.where(active)[0]
    if len(active_idx) < 30:
        return np.nan

    X = M_z[:, active_idx]
    if rng is None:
        rng = np.random.RandomState(320)

    from scipy.linalg import svd as scipy_svd

    cca_scores = []
    for _ in range(n_splits):
        perm = rng.permutation(n_neurons)
        half = n_neurons // 2
        h1, h2 = perm[:half], perm[half:2 * half]

        X1 = X[h1].T  # (active_bins x half_neurons)
        X2 = X[h2].T

        k1, k2 = X1.shape[1], X2.shape[1]
        n_comp = min(n_components, k1 - 1, k2 - 1)
        if n_comp < 1:
            continue

        try:
            C11 = X1.T @ X1 / X1.shape[0] + 1e-6 * np.eye(k1)
            C22 = X2.T @ X2 / X2.shape[0] + 1e-6 * np.eye(k2)
            C12 = X1.T @ X2 / X1.shape[0]

            C11_inv_half = np.linalg.cholesky(np.linalg.inv(C11))
            C22_inv_half = np.linalg.cholesky(np.linalg.inv(C22))

            T = C11_inv_half.T @ C12 @ C22_inv_half
            _, s, _ = scipy_svd(T, full_matrices=False)
            canon_corrs = np.clip(s[:n_comp], 0, 1)
            cca_scores.append(np.mean(canon_corrs))
        except (np.linalg.LinAlgError, ValueError):
            continue

    if not cca_scores:
        return np.nan
    return np.mean(cca_scores)


# ======================================================================
# REVISED VALIANT METRICS
# ======================================================================
def compute_place_field_properties(M, threshold_frac=0.3):
    """
    For each neuron, identify place fields and measure their properties.

    A place field is a contiguous region where firing rate exceeds
    threshold_frac * peak_rate.

    Returns per-neuron: n_fields, mean_field_size, peak_rate
    """
    M = np.nan_to_num(M, nan=0.0)
    n_neurons = M.shape[0]
    results = []

    for i in range(n_neurons):
        rate_map = M[i].reshape(GRID_SIZE, GRID_SIZE)

        # Smooth slightly to avoid fragmentation
        rate_smooth = gaussian_filter(rate_map, sigma=1.0)
        peak = np.max(rate_smooth)

        if peak < 0.5:  # sub-threshold neuron
            results.append({
                'n_fields': 0,
                'field_sizes': [],
                'peak_rate': peak,
                'total_field_area': 0,
            })
            continue

        thresh = threshold_frac * peak
        binary = (rate_smooth >= thresh).astype(int)
        labeled, n_fields = ndimage_label(binary)

        # Filter tiny fields (< 4 bins = noise)
        field_sizes = []
        for f_id in range(1, n_fields + 1):
            size = np.sum(labeled == f_id)
            if size >= 4:
                field_sizes.append(size)

        results.append({
            'n_fields': len(field_sizes),
            'field_sizes': field_sizes,
            'peak_rate': peak,
            'total_field_area': sum(field_sizes),
        })

    return results


def compute_valiant_metrics_revised(M):
    """
    Three revised Valiant-stability metrics:

    1. Field size CV: For neurons with place fields, how consistent are
       field sizes? Low CV = consistent allocation territory per neuron.

    2. Population overlap CV: For each spatial bin above threshold, how
       many neurons are co-active? For PLACE FIELD BINS only. Low CV =
       each location recruits a consistent-sized ensemble.

    3. Split-half N_active correlation: Split neurons randomly, count
       N_active per bin in each half, correlate. High r = allocation
       pattern is reproducible.
    """
    n_neurons, n_bins = M.shape
    pf = compute_place_field_properties(M)

    # --- Metric 1: Field size consistency ---
    all_field_sizes = []
    for p in pf:
        all_field_sizes.extend(p['field_sizes'])

    field_size_cv = np.nan
    if len(all_field_sizes) >= 5:
        fs = np.array(all_field_sizes)
        field_size_cv = np.std(fs) / np.mean(fs) if np.mean(fs) > 0 else np.nan

    # --- Metric 2: Per-field population overlap ---
    # For each bin that's inside some neuron's place field, count
    # how many neurons are active there (above their individual threshold)
    M_filled = np.nan_to_num(M, nan=0.0)
    field_masks = np.zeros((n_neurons, n_bins), dtype=bool)
    for i in range(n_neurons):
        rate_map = M_filled[i].reshape(GRID_SIZE, GRID_SIZE)
        rate_smooth = gaussian_filter(rate_map, sigma=1.0)
        peak = np.max(rate_smooth)
        if peak >= 0.5:
            field_masks[i] = (rate_smooth >= 0.3 * peak).flatten()

    # Bins that are in at least one neuron's field
    any_field = np.any(field_masks, axis=0)
    field_bins = np.where(any_field)[0]

    if len(field_bins) > 0:
        n_active_per_field_bin = np.sum(field_masks[:, field_bins], axis=0)
        pop_overlap_cv = (np.std(n_active_per_field_bin) /
                          np.mean(n_active_per_field_bin)
                          if np.mean(n_active_per_field_bin) > 0 else np.nan)
        pop_overlap_mean = np.mean(n_active_per_field_bin)
    else:
        pop_overlap_cv = np.nan
        pop_overlap_mean = np.nan

    # --- Metric 3: Split-half N_active correlation ---
    rng = np.random.RandomState(320)
    split_corrs = []
    for _ in range(50):
        perm = rng.permutation(n_neurons)
        half = n_neurons // 2
        h1 = perm[:half]
        h2 = perm[half:2*half]

        # N_active per bin for each half (using field threshold per neuron)
        n1 = np.sum(field_masks[h1], axis=0)
        n2 = np.sum(field_masks[h2], axis=0)

        # Only consider bins where at least one neuron is in a field
        active_any = (n1 + n2) > 0
        if np.sum(active_any) > 20:
            r, _ = spearmanr(n1[active_any], n2[active_any])
            if np.isfinite(r):
                split_corrs.append(r)

    split_half_r = np.mean(split_corrs) if split_corrs else np.nan

    # --- Per-neuron field counts ---
    n_fields_per_neuron = [p['n_fields'] for p in pf if p['n_fields'] > 0]
    neurons_with_fields = len(n_fields_per_neuron)
    frac_with_fields = neurons_with_fields / n_neurons

    return {
        'field_size_cv': field_size_cv,
        'pop_overlap_cv': pop_overlap_cv,
        'pop_overlap_mean': pop_overlap_mean,
        'split_half_r': split_half_r,
        'n_fields_per_neuron': n_fields_per_neuron,
        'frac_with_fields': frac_with_fields,
        'n_fields_total': sum(p['n_fields'] for p in pf),
        'all_field_sizes': all_field_sizes,
    }


# ======================================================================
# MANTEL TEST (pairwise NaN deletion, 1000 permutations)
# ======================================================================
def compute_mantel_test(M, n_perms=1000, rng=None):
    """
    Mantel test: correlation between neural RDM and physical distance.

    Uses masked euclidean distance (pairwise deletion) so unvisited
    spatial bins (NaN) are excluded per pair rather than zero-filled.
    """
    if rng is None:
        rng = np.random.RandomState(320)

    n_neurons, n_bins = M.shape
    means = np.nanmean(M, axis=1, keepdims=True)
    stds = np.nanstd(M, axis=1, keepdims=True)
    stds[stds == 0] = 1
    M_z = (M - means) / stds

    valid_positive = np.isfinite(M) & (M > 0)
    active = np.sum(valid_positive, axis=0) >= max(2, n_neurons // 3)
    active_idx = np.where(active)[0]
    if len(active_idx) < 50:
        return np.nan, np.nan, []

    # Physical distance (no NaN issue -- grid coordinates)
    rows = active_idx // GRID_SIZE
    cols = active_idx % GRID_SIZE
    phys_vec = pdist(np.column_stack([rows, cols]), metric='euclidean')

    X = M_z[:, active_idx].T  # (bins x neurons)
    neural_vec = masked_euclidean_pdist(X)

    valid = np.isfinite(phys_vec) & np.isfinite(neural_vec)
    if np.sum(valid) < 100:
        return np.nan, np.nan, []

    r_obs, _ = spearmanr(phys_vec[valid], neural_vec[valid])

    null_rs = []
    n = len(active_idx)
    for _ in range(n_perms):
        perm = rng.permutation(n)
        X_perm = X[perm, :]
        neural_perm = masked_euclidean_pdist(X_perm)
        valid_perm = np.isfinite(phys_vec) & np.isfinite(neural_perm)
        if np.sum(valid_perm) < 100:
            continue
        r_perm, _ = spearmanr(phys_vec[valid_perm], neural_perm[valid_perm])
        null_rs.append(r_perm)

    null_rs = np.array(null_rs) if null_rs else np.array([])
    p_value = np.mean(null_rs >= r_obs) if len(null_rs) > 0 else np.nan
    return r_obs, p_value, null_rs


# ======================================================================
# AP GRADIENT
# ======================================================================
def compute_ap_gradient(df_t, sessions_t):
    results = {}
    for sess_name, sess_data in sessions_t.items():
        subdiv = sess_data['subdivision']
        if pd.isna(subdiv):
            continue
        shesha_mean, _ = compute_shesha(sess_data['M'])
        if np.isnan(shesha_mean):
            continue
        if subdiv not in results:
            results[subdiv] = []
        results[subdiv].append({
            'session': sess_name, 'shesha': shesha_mean,
            'n_neurons': sess_data['n_neurons'],
            'bird': sess_data['bird'],
        })
    return results


# ======================================================================
# WITHIN-SESSION DRIFT
# ======================================================================
def compute_drift(df):
    results = {}
    for species in ['titmouse', 'zebra_finch']:
        sub = df[(df['species'] == species) & (df['cell_type'] == 'E')]
        xcorr = sub['xcorr_map'].dropna()
        if len(xcorr) == 0:
            continue
        results[species] = {
            'xcorr': xcorr.values,
            'n': len(xcorr),
            # Collect per-cell metadata for CSV export
            'cell_rows': sub[['session', 'bird', 'xcorr_map']].dropna(
                subset=['xcorr_map']).to_dict('records'),
        }
    return results


# ======================================================================
# MAIN
# ======================================================================
def run_analyses(df, df_t, df_z):
    print("\n" + "="*70)
    print("TIER 1 v2: VALIANT SMA / SHESHA HYPOTHESIS TESTS")
    print("="*70)

    sessions_t = get_session_maps(df_t)
    sessions_z = get_session_maps(df_z)
    print(f"\nSessions: {len(sessions_t)} chickadee, {len(sessions_z)} finch")

    # -- 1. SHESHA --
    print("\n" + "-"*60)
    print("1. SHESHA (no bin subsampling, 100 splits)")
    print("-"*60)

    shesha_t, shesha_z = [], []
    for sess, data in sessions_t.items():
        s, _ = compute_shesha(data['M'])
        if not np.isnan(s):
            shesha_t.append({'session': sess, 'shesha': s,
                             'n': data['n_neurons'],
                             'bird': data['bird']})
    for sess, data in sessions_z.items():
        s, _ = compute_shesha(data['M'])
        if not np.isnan(s):
            shesha_z.append({'session': sess, 'shesha': s,
                             'n': data['n_neurons'],
                             'bird': data['bird']})

    s_t = [x['shesha'] for x in shesha_t]
    s_z = [x['shesha'] for x in shesha_z]
    if s_t and s_z:
        U, p = mannwhitneyu(s_t, s_z, alternative='greater')
        print(f"  Chickadee: mean={np.mean(s_t):.4f}, median={np.median(s_t):.4f} (n={len(s_t)})")
        print(f"  Finch:     mean={np.mean(s_z):.4f}, median={np.median(s_z):.4f} (n={len(s_z)})")
        print(f"  MW (chickadee > finch): p={p:.4e}")

    # -- 1b. Alternative Stability Benchmarks (concordance check) --
    print("\n" + "-"*60)
    print("1b. Alternative Stability Benchmarks")
    print("-"*60)

    pv_t, pv_z = [], []
    cca_t, cca_z = [], []
    for sess, data in sessions_t.items():
        pv = compute_test_retest_pv_correlation(data['M'])
        cc = compute_cca_stability(data['M'])
        if not np.isnan(pv):
            pv_t.append({'session': sess, 'pv_corr': pv,
                          'n': data['n_neurons'], 'bird': data['bird']})
        if not np.isnan(cc):
            cca_t.append({'session': sess, 'cca': cc,
                           'n': data['n_neurons'], 'bird': data['bird']})
    for sess, data in sessions_z.items():
        pv = compute_test_retest_pv_correlation(data['M'])
        cc = compute_cca_stability(data['M'])
        if not np.isnan(pv):
            pv_z.append({'session': sess, 'pv_corr': pv,
                          'n': data['n_neurons'], 'bird': data['bird']})
        if not np.isnan(cc):
            cca_z.append({'session': sess, 'cca': cc,
                           'n': data['n_neurons'], 'bird': data['bird']})

    pv_vals_t = [x['pv_corr'] for x in pv_t]
    pv_vals_z = [x['pv_corr'] for x in pv_z]
    cca_vals_t = [x['cca'] for x in cca_t]
    cca_vals_z = [x['cca'] for x in cca_z]

    if pv_vals_t and pv_vals_z:
        U, p = mannwhitneyu(pv_vals_t, pv_vals_z, alternative='greater')
        print(f"  Test-retest PV correlation:")
        print(f"    Chickadee: mean={np.mean(pv_vals_t):.4f} (n={len(pv_vals_t)})")
        print(f"    Finch:     mean={np.mean(pv_vals_z):.4f} (n={len(pv_vals_z)})")
        print(f"    MW (chickadee > finch): p={p:.4e}")

    if cca_vals_t and cca_vals_z:
        U, p = mannwhitneyu(cca_vals_t, cca_vals_z, alternative='greater')
        print(f"  CCA stability (top-3 canonical correlations):")
        print(f"    Chickadee: mean={np.mean(cca_vals_t):.4f} (n={len(cca_vals_t)})")
        print(f"    Finch:     mean={np.mean(cca_vals_z):.4f} (n={len(cca_vals_z)})")
        print(f"    MW (chickadee > finch): p={p:.4e}")

    # Concordance check: do all three metrics agree directionally?
    concordance = []
    if pv_vals_t and pv_vals_z:
        concordance.append(('PV_corr', np.mean(pv_vals_t) > np.mean(pv_vals_z)))
    if cca_vals_t and cca_vals_z:
        concordance.append(('CCA', np.mean(cca_vals_t) > np.mean(cca_vals_z)))
    if s_t and s_z:
        concordance.append(('SHESHA', np.mean(s_t) > np.mean(s_z)))
    all_agree = all(c[1] for c in concordance)
    print(f"\n  Concordance check (chickadee > finch):")
    for name, agrees in concordance:
        print(f"    {name}: {'YES' if agrees else 'NO'}")
    print(f"    All metrics agree: {'YES' if all_agree else 'NO'}")

    # -- 2. Revised Valiant Metrics --
    print("\n" + "-"*60)
    print("2. Revised Valiant-Stability Metrics")
    print("-"*60)

    val_t, val_z = [], []
    for sess, data in sessions_t.items():
        print(f"  Processing chickadee session {sess}...", end='\r')
        v = compute_valiant_metrics_revised(data['M'])
        v['session'] = sess
        v['n_neurons'] = data['n_neurons']
        v['bird'] = data['bird']
        val_t.append(v)
    print()
    for sess, data in sessions_z.items():
        print(f"  Processing finch session {sess}...", end='\r')
        v = compute_valiant_metrics_revised(data['M'])
        v['session'] = sess
        v['n_neurons'] = data['n_neurons']
        v['bird'] = data['bird']
        val_z.append(v)
    print()

    # 2a: Field size consistency
    fs_cv_t = [v['field_size_cv'] for v in val_t if not np.isnan(v['field_size_cv'])]
    fs_cv_z = [v['field_size_cv'] for v in val_z if not np.isnan(v['field_size_cv'])]
    if fs_cv_t and fs_cv_z:
        U, p = mannwhitneyu(fs_cv_t, fs_cv_z, alternative='two-sided')
        print(f"\n  2a. Place field size CV:")
        print(f"    Chickadee: mean={np.mean(fs_cv_t):.4f} (n={len(fs_cv_t)})")
        print(f"    Finch:     mean={np.mean(fs_cv_z):.4f} (n={len(fs_cv_z)})")
        print(f"    MW: p={p:.4e}")

    # 2b: Population overlap
    po_cv_t = [v['pop_overlap_cv'] for v in val_t if not np.isnan(v['pop_overlap_cv'])]
    po_cv_z = [v['pop_overlap_cv'] for v in val_z if not np.isnan(v['pop_overlap_cv'])]
    if po_cv_t and po_cv_z:
        U, p = mannwhitneyu(po_cv_t, po_cv_z, alternative='less')
        print(f"\n  2b. Population overlap CV (N co-active in field bins):")
        print(f"    Chickadee: mean={np.mean(po_cv_t):.4f} (n={len(po_cv_t)})")
        print(f"    Finch:     mean={np.mean(po_cv_z):.4f} (n={len(po_cv_z)})")
        print(f"    MW (chickadee < finch): p={p:.4e}")

    # 2c: Split-half allocation reliability
    sh_t = [v['split_half_r'] for v in val_t if not np.isnan(v['split_half_r'])]
    sh_z = [v['split_half_r'] for v in val_z if not np.isnan(v['split_half_r'])]
    if sh_t and sh_z:
        U, p = mannwhitneyu(sh_t, sh_z, alternative='less')
        print(f"\n  2c. Split-half N_active correlation:")
        print(f"    Chickadee: mean={np.mean(sh_t):.4f} (n={len(sh_t)})")
        print(f"    Finch:     mean={np.mean(sh_z):.4f} (n={len(sh_z)})")
        print(f"    MW (chickadee < finch, one-tailed): p={p:.4e}")

    # Field statistics
    print(f"\n  Field statistics:")
    for label, vals in [('Chickadee', val_t), ('Finch', val_z)]:
        frac = np.mean([v['frac_with_fields'] for v in vals])
        n_fields = np.mean([np.mean(v['n_fields_per_neuron'])
                            if v['n_fields_per_neuron'] else 0 for v in vals])
        print(f"    {label}: {frac*100:.1f}% neurons with fields, "
              f"mean {n_fields:.1f} fields per place cell")

    # -- 3. Mantel Test --
    print("\n" + "-"*60)
    print("3. Mantel Test (1000 permutations, no bin subsampling)")
    print("-"*60)

    mantel_t, mantel_z = [], []
    for sess, data in sessions_t.items():
        r, p_val, _ = compute_mantel_test(data['M'], n_perms=1000)
        if not np.isnan(r):
            mantel_t.append({'session': sess, 'r': r, 'p': p_val,
                             'bird': data['bird'],
                             'n_neurons': data['n_neurons']})
    for sess, data in sessions_z.items():
        r, p_val, _ = compute_mantel_test(data['M'], n_perms=1000)
        if not np.isnan(r):
            mantel_z.append({'session': sess, 'r': r, 'p': p_val,
                             'bird': data['bird'],
                             'n_neurons': data['n_neurons']})

    mt = [x['r'] for x in mantel_t]
    mz = [x['r'] for x in mantel_z]
    if mt and mz:
        U, p = mannwhitneyu(mt, mz, alternative='greater')
        sig_t = sum(1 for x in mantel_t if x['p'] < 0.05)
        sig_z = sum(1 for x in mantel_z if x['p'] < 0.05)
        print(f"  Chickadee: mean r={np.mean(mt):.4f}, "
              f"{sig_t}/{len(mt)} significant")
        print(f"  Finch:     mean r={np.mean(mz):.4f}, "
              f"{sig_z}/{len(mz)} significant")
        print(f"  MW: p={p:.4e}")

    # -- 4. AP Gradient --
    print("\n" + "-"*60)
    print("4. A-P Gradient (chickadee only)")
    print("-"*60)
    ap = compute_ap_gradient(df_t, sessions_t)
    for subdiv in sorted(ap.keys()):
        vals = [x['shesha'] for x in ap[subdiv]]
        print(f"  {subdiv}: SHESHA={np.mean(vals):.4f}, n={len(vals)}")
    subdivs = sorted(ap.keys())
    if len(subdivs) >= 2:
        for i in range(len(subdivs)):
            for j in range(i+1, len(subdivs)):
                v1 = [x['shesha'] for x in ap[subdivs[i]]]
                v2 = [x['shesha'] for x in ap[subdivs[j]]]
                if len(v1) >= 3 and len(v2) >= 3:
                    U, p = mannwhitneyu(v1, v2, alternative='two-sided')
                    print(f"    {subdivs[i]} vs {subdivs[j]}: p={p:.4f}")

    # -- 5. Drift --
    print("\n" + "-"*60)
    print("5. Within-session drift")
    print("-"*60)
    drift = compute_drift(df)
    for sp, data in drift.items():
        label = 'Chickadee' if sp == 'titmouse' else 'Finch'
        print(f"  {label}: median xcorr={np.median(data['xcorr']):.4f} (n={data['n']})")
    if 'titmouse' in drift and 'zebra_finch' in drift:
        U, p = mannwhitneyu(drift['titmouse']['xcorr'],
                             drift['zebra_finch']['xcorr'], alternative='greater')
        print(f"  MW: p={p:.4e}")

    # -- Double Dissociation (revised) --
    print("\n" + "-"*60)
    print("DOUBLE DISSOCIATION: SHESHA x Split-Half Allocation")
    print("-"*60)

    dd_t, dd_z = [], []
    shesha_dict_t = {x['session']: x['shesha'] for x in shesha_t}
    shesha_dict_z = {x['session']: x['shesha'] for x in shesha_z}

    for v in val_t:
        sess = v['session']
        if sess in shesha_dict_t and not np.isnan(v['split_half_r']):
            dd_t.append({'shesha': shesha_dict_t[sess],
                          'split_half_r': v['split_half_r'],
                          'pop_overlap_cv': v['pop_overlap_cv'],
                          'session': sess,
                          'bird': v['bird']})
    for v in val_z:
        sess = v['session']
        if sess in shesha_dict_z and not np.isnan(v['split_half_r']):
            dd_z.append({'shesha': shesha_dict_z[sess],
                          'split_half_r': v['split_half_r'],
                          'pop_overlap_cv': v['pop_overlap_cv'],
                          'session': sess,
                          'bird': v['bird']})

    if dd_t:
        print(f"\n  Chickadee (n={len(dd_t)}):")
        print(f"    SHESHA: {np.mean([d['shesha'] for d in dd_t]):.4f}")
        print(f"    Split-half alloc r: {np.mean([d['split_half_r'] for d in dd_t]):.4f}")
        if len(dd_t) > 5:
            r, p = spearmanr([d['shesha'] for d in dd_t],
                              [d['split_half_r'] for d in dd_t])
            print(f"    SHESHA ~ alloc r: rho={r:.3f}, p={p:.3f}")
    if dd_z:
        print(f"  Finch (n={len(dd_z)}):")
        print(f"    SHESHA: {np.mean([d['shesha'] for d in dd_z]):.4f}")
        print(f"    Split-half alloc r: {np.mean([d['split_half_r'] for d in dd_z]):.4f}")

    # -- Export CSVs --
    print("\n" + "-"*60)
    print("Exporting results to CSV...")
    print("-"*60)
    export_all_plot_data(
        shesha_t=shesha_t, shesha_z=shesha_z,
        val_t=val_t, val_z=val_z,
        mantel_t=mantel_t, mantel_z=mantel_z,
        ap=ap, drift=drift,
        dd_t=dd_t, dd_z=dd_z,
        s_t=s_t, s_z=s_z,
        sh_t=sh_t, sh_z=sh_z,
        fs_cv_t=fs_cv_t, fs_cv_z=fs_cv_z,
        po_cv_t=po_cv_t, po_cv_z=po_cv_z,
        mt=mt, mz=mz,
        pv_t=pv_t, pv_z=pv_z,
        cca_t=cca_t, cca_z=cca_z,
    )

    return {
        'shesha_t': shesha_t, 'shesha_z': shesha_z,
        'val_t': val_t, 'val_z': val_z,
        'mantel_t': mantel_t, 'mantel_z': mantel_z,
        'ap': ap, 'drift': drift,
        'dd_t': dd_t, 'dd_z': dd_z,
        'pv_t': pv_t, 'pv_z': pv_z,
        'cca_t': cca_t, 'cca_z': cca_z,
    }


def export_all_plot_data(*, shesha_t, shesha_z, val_t, val_z,
                          mantel_t, mantel_z, ap, drift,
                          dd_t, dd_z, s_t, s_z, sh_t, sh_z,
                          fs_cv_t, fs_cv_z, po_cv_t, po_cv_z,
                          mt, mz, pv_t=None, pv_z=None,
                          cca_t=None, cca_z=None):
    """
    Export ALL data underlying every plotted panel to CSV files.

    Files produced:
      tier1_summary.csv              - aggregated test results (original)
      tier1_session_results.csv      - per-session master table (original, expanded)
      tier1_shesha_scatter.csv       - panel A: per-session SHESHA + n_neurons
      tier1_mantel_sessions.csv      - panel C: per-session Mantel r + p
      tier1_ap_gradient.csv          - panel D: per-session subdivision SHESHA
      tier1_drift_cells.csv          - panel E: per-cell xcorr_map values
      tier1_double_dissociation.csv  - panel F: matched SHESHA + split_half_r
      tier1_field_sizes.csv          - supporting: all individual field sizes
    """

    # ----------------------------------------------------------------
    # 1. tier1_summary.csv  (original aggregated table, preserved)
    # ----------------------------------------------------------------
    summary = []
    if s_t and s_z:
        summary.append({'test': 'shesha', 'chickadee_mean': np.mean(s_t),
                         'finch_mean': np.mean(s_z),
                         'p_value': mannwhitneyu(s_t, s_z, alternative='greater')[1],
                         'direction': 'chickadee > finch'})
    if sh_t and sh_z:
        summary.append({'test': 'split_half_alloc_r',
                         'chickadee_mean': np.mean(sh_t),
                         'finch_mean': np.mean(sh_z),
                         'p_value': mannwhitneyu(sh_t, sh_z, alternative='less')[1],
                         'direction': 'chickadee < finch (one-tailed)'})
    if fs_cv_t and fs_cv_z:
        summary.append({'test': 'field_size_cv',
                         'chickadee_mean': np.mean(fs_cv_t),
                         'finch_mean': np.mean(fs_cv_z),
                         'p_value': mannwhitneyu(fs_cv_t, fs_cv_z)[1],
                         'direction': 'two-sided'})
    if po_cv_t and po_cv_z:
        summary.append({'test': 'pop_overlap_cv',
                         'chickadee_mean': np.mean(po_cv_t),
                         'finch_mean': np.mean(po_cv_z),
                         'p_value': mannwhitneyu(po_cv_t, po_cv_z, alternative='less')[1],
                         'direction': 'chickadee < finch'})
    if mt and mz:
        summary.append({'test': 'mantel_r',
                         'chickadee_mean': np.mean(mt),
                         'finch_mean': np.mean(mz),
                         'p_value': mannwhitneyu(mt, mz, alternative='greater')[1],
                         'direction': 'chickadee > finch'})
    if 'titmouse' in drift and 'zebra_finch' in drift:
        summary.append({'test': 'xcorr_map_drift',
                         'chickadee_mean': np.mean(drift['titmouse']['xcorr']),
                         'finch_mean': np.mean(drift['zebra_finch']['xcorr']),
                         'p_value': mannwhitneyu(
                             drift['titmouse']['xcorr'],
                             drift['zebra_finch']['xcorr'],
                             alternative='greater')[1],
                         'direction': 'chickadee > finch'})
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR, 'tier1_summary.csv'), index=False)
    print(f"  Saved tier1_summary.csv ({len(summary)} rows)")

    # ----------------------------------------------------------------
    # 2. tier1_session_results.csv  (per-session master, expanded)
    # ----------------------------------------------------------------
    shesha_lookup = {x['session']: x for x in shesha_t + shesha_z}
    mantel_lookup = {x['session']: x for x in mantel_t + mantel_z}

    sess_rows = []
    for v in val_t + val_z:
        sess = v['session']
        species = 'chickadee' if any(v['session'] == x['session'] for x in val_t) else 'finch'
        row = {
            'session': sess,
            'species': species,
            'bird': v.get('bird', ''),
            'n_neurons': v['n_neurons'],
            'field_size_cv': v['field_size_cv'],
            'pop_overlap_cv': v['pop_overlap_cv'],
            'pop_overlap_mean': v['pop_overlap_mean'],
            'split_half_r': v['split_half_r'],
            'frac_with_fields': v['frac_with_fields'],
            'n_fields_total': v['n_fields_total'],
        }
        if sess in shesha_lookup:
            row['shesha'] = shesha_lookup[sess]['shesha']
        if sess in mantel_lookup:
            row['mantel_r'] = mantel_lookup[sess]['r']
            row['mantel_p'] = mantel_lookup[sess]['p']
        sess_rows.append(row)

    pd.DataFrame(sess_rows).to_csv(os.path.join(OUTPUT_DIR, 'tier1_session_results.csv'), index=False)
    print(f"  Saved tier1_session_results.csv ({len(sess_rows)} rows)")

    # ----------------------------------------------------------------
    # 3. tier1_shesha_scatter.csv  (panel A scatter data)
    #    One row per session: SHESHA, n_neurons, species
    # ----------------------------------------------------------------
    scatter_rows = []
    for x in shesha_t:
        scatter_rows.append({
            'species': 'chickadee', 'session': x['session'],
            'bird': x.get('bird', ''),
            'n_neurons': x['n'], 'shesha': x['shesha'],
        })
    for x in shesha_z:
        scatter_rows.append({
            'species': 'finch', 'session': x['session'],
            'bird': x.get('bird', ''),
            'n_neurons': x['n'], 'shesha': x['shesha'],
        })
    pd.DataFrame(scatter_rows).to_csv(os.path.join(OUTPUT_DIR, 'tier1_shesha_scatter.csv'), index=False)
    print(f"  Saved tier1_shesha_scatter.csv ({len(scatter_rows)} rows)")

    # ----------------------------------------------------------------
    # 4. tier1_mantel_sessions.csv  (panel C histogram data)
    #    One row per session: Mantel r, p, significance flag
    # ----------------------------------------------------------------
    mantel_rows = []
    for x in mantel_t:
        mantel_rows.append({
            'species': 'chickadee', 'session': x['session'],
            'bird': x.get('bird', ''),
            'n_neurons': x.get('n_neurons', ''),
            'mantel_r': x['r'], 'mantel_p': x['p'],
            'significant_005': x['p'] < 0.05,
        })
    for x in mantel_z:
        mantel_rows.append({
            'species': 'finch', 'session': x['session'],
            'bird': x.get('bird', ''),
            'n_neurons': x.get('n_neurons', ''),
            'mantel_r': x['r'], 'mantel_p': x['p'],
            'significant_005': x['p'] < 0.05,
        })
    pd.DataFrame(mantel_rows).to_csv(os.path.join(OUTPUT_DIR, 'tier1_mantel_sessions.csv'), index=False)
    print(f"  Saved tier1_mantel_sessions.csv ({len(mantel_rows)} rows)")

    # ----------------------------------------------------------------
    # 5. tier1_ap_gradient.csv  (panel D bar chart data)
    #    One row per session within each subdivision
    # ----------------------------------------------------------------
    ap_rows = []
    for subdiv, entries in ap.items():
        for e in entries:
            ap_rows.append({
                'subdivision': subdiv,
                'session': e['session'],
                'bird': e.get('bird', ''),
                'n_neurons': e.get('n_neurons', ''),
                'shesha': e['shesha'],
            })
    if ap_rows:
        pd.DataFrame(ap_rows).to_csv(os.path.join(OUTPUT_DIR, 'tier1_ap_gradient.csv'), index=False)
        print(f"  Saved tier1_ap_gradient.csv ({len(ap_rows)} rows)")
    else:
        print("  Skipped tier1_ap_gradient.csv (no subdivision data)")

    # ----------------------------------------------------------------
    # 6. tier1_drift_cells.csv  (panel E histogram data)
    #    One row per neuron: xcorr_map, species, session, bird
    # ----------------------------------------------------------------
    drift_rows = []
    for species, label in [('titmouse', 'chickadee'), ('zebra_finch', 'finch')]:
        if species in drift:
            for cr in drift[species]['cell_rows']:
                drift_rows.append({
                    'species': label,
                    'session': cr.get('session', ''),
                    'bird': cr.get('bird', ''),
                    'xcorr_map': cr.get('xcorr_map', np.nan),
                })
    pd.DataFrame(drift_rows).to_csv(os.path.join(OUTPUT_DIR, 'tier1_drift_cells.csv'), index=False)
    print(f"  Saved tier1_drift_cells.csv ({len(drift_rows)} rows)")

    # ----------------------------------------------------------------
    # 7. tier1_double_dissociation.csv  (panel F scatter data)
    #    One row per session: SHESHA + split_half_r + pop_overlap_cv
    # ----------------------------------------------------------------
    dd_rows = []
    for d in dd_t:
        dd_rows.append({
            'species': 'chickadee', 'session': d['session'],
            'bird': d.get('bird', ''),
            'shesha': d['shesha'],
            'split_half_r': d['split_half_r'],
            'pop_overlap_cv': d['pop_overlap_cv'],
        })
    for d in dd_z:
        dd_rows.append({
            'species': 'finch', 'session': d['session'],
            'bird': d.get('bird', ''),
            'shesha': d['shesha'],
            'split_half_r': d['split_half_r'],
            'pop_overlap_cv': d['pop_overlap_cv'],
        })
    pd.DataFrame(dd_rows).to_csv(os.path.join(OUTPUT_DIR, 'tier1_double_dissociation.csv'), index=False)
    print(f"  Saved tier1_double_dissociation.csv ({len(dd_rows)} rows)")

    # ----------------------------------------------------------------
    # 8. tier1_alternative_metrics.csv  (PV correlation + CCA stability)
    # ----------------------------------------------------------------
    alt_rows = []
    if pv_t:
        for x in pv_t:
            alt_rows.append({
                'species': 'chickadee', 'session': x['session'],
                'bird': x.get('bird', ''), 'n_neurons': x['n'],
                'pv_corr': x['pv_corr'],
            })
    if pv_z:
        for x in pv_z:
            alt_rows.append({
                'species': 'finch', 'session': x['session'],
                'bird': x.get('bird', ''), 'n_neurons': x['n'],
                'pv_corr': x['pv_corr'],
            })
    # Merge CCA into same rows by session
    cca_lookup = {}
    if cca_t:
        for x in cca_t:
            cca_lookup[x['session']] = x['cca']
    if cca_z:
        for x in cca_z:
            cca_lookup[x['session']] = x['cca']
    for row in alt_rows:
        row['cca_stability'] = cca_lookup.get(row['session'], np.nan)

    # Also merge SHESHA for concordance table
    shesha_lookup = {x['session']: x['shesha'] for x in shesha_t + shesha_z}
    for row in alt_rows:
        row['shesha'] = shesha_lookup.get(row['session'], np.nan)

    if alt_rows:
        pd.DataFrame(alt_rows).to_csv(
            os.path.join(OUTPUT_DIR, 'tier1_alternative_metrics.csv'),
            index=False)
        print(f"  Saved tier1_alternative_metrics.csv ({len(alt_rows)} rows)")

    # Concordance summary
    concordance_rows = []
    pv_vals_t = [x['pv_corr'] for x in (pv_t or [])]
    pv_vals_z = [x['pv_corr'] for x in (pv_z or [])]
    cca_vals_t = [x['cca'] for x in (cca_t or [])]
    cca_vals_z = [x['cca'] for x in (cca_z or [])]

    for name, vt, vz, alt_dir in [
        ('SHESHA', s_t, s_z, 'greater'),
        ('PV_correlation', pv_vals_t, pv_vals_z, 'greater'),
        ('CCA_stability', cca_vals_t, cca_vals_z, 'greater'),
    ]:
        if vt and vz:
            U, p = mannwhitneyu(vt, vz, alternative=alt_dir)
            concordance_rows.append({
                'metric': name,
                'chickadee_mean': np.mean(vt),
                'finch_mean': np.mean(vz),
                'chickadee_n': len(vt),
                'finch_n': len(vz),
                'direction': 'chickadee > finch',
                'directionally_consistent': np.mean(vt) > np.mean(vz),
                'p_value': p,
            })
    if concordance_rows:
        pd.DataFrame(concordance_rows).to_csv(
            os.path.join(OUTPUT_DIR, 'tier1_concordance.csv'), index=False)
        print(f"  Saved tier1_concordance.csv ({len(concordance_rows)} rows)")

    # ----------------------------------------------------------------
    # 9. tier1_field_sizes.csv  (supporting data for valiant metrics)
    #    One row per detected place field: session, species, field_size
    # ----------------------------------------------------------------
    fs_rows = []
    for v in val_t:
        for sz in v['all_field_sizes']:
            fs_rows.append({
                'species': 'chickadee', 'session': v['session'],
                'bird': v.get('bird', ''),
                'field_size_bins': sz,
            })
    for v in val_z:
        for sz in v['all_field_sizes']:
            fs_rows.append({
                'species': 'finch', 'session': v['session'],
                'bird': v.get('bird', ''),
                'field_size_bins': sz,
            })
    pd.DataFrame(fs_rows).to_csv(os.path.join(OUTPUT_DIR, 'tier1_field_sizes.csv'), index=False)
    print(f"  Saved tier1_field_sizes.csv ({len(fs_rows)} rows)")


# ======================================================================
# PLOTTING
# ======================================================================
def plot_results(results):
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35,
                           left=0.06, right=0.96, top=0.91, bottom=0.07)

    # -- A: SHESHA --
    ax_a = fig.add_subplot(gs[0, 0])
    s_t = [x['shesha'] for x in results['shesha_t']]
    s_z = [x['shesha'] for x in results['shesha_z']]
    n_t = [x['n'] for x in results['shesha_t']]
    n_z = [x['n'] for x in results['shesha_z']]

    ax_a.scatter(n_t, s_t, c=C_T, alpha=0.6, s=45, edgecolors='white',
                  linewidths=0.3, label='Chickadee', zorder=3)
    ax_a.scatter(n_z, s_z, c=C_Z, alpha=0.6, s=45, edgecolors='white',
                  linewidths=0.3, label='Finch', zorder=3)
    ax_a.axhline(np.mean(s_t), color=C_T, ls='--', lw=1, alpha=0.6)
    ax_a.axhline(np.mean(s_z), color=C_Z, ls='--', lw=1, alpha=0.6)
    U, p = mannwhitneyu(s_t, s_z, alternative='greater')
    ax_a.set_title(f'A. SHESHA (split-half RDM)\n'
                    f'mean {np.mean(s_t):.3f} vs {np.mean(s_z):.3f}, p = {p:.2e}',
                    fontsize=11, fontweight='bold', loc='left')
    ax_a.set_xlabel('Neurons in session', fontsize=10)
    ax_a.set_ylabel('SHESHA (Spearman r)', fontsize=10)
    ax_a.legend(fontsize=8, frameon=False)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # -- B: Split-half allocation reliability --
    ax_b = fig.add_subplot(gs[0, 1])
    sh_t = [v['split_half_r'] for v in results['val_t'] if not np.isnan(v['split_half_r'])]
    sh_z = [v['split_half_r'] for v in results['val_z'] if not np.isnan(v['split_half_r'])]

    bp = ax_b.boxplot([sh_t, sh_z], positions=[1, 2], widths=0.5,
                       patch_artist=True, showfliers=True,
                       flierprops=dict(markersize=3))
    bp['boxes'][0].set_facecolor(C_T2)
    bp['boxes'][1].set_facecolor(C_Z2)
    for line in bp['medians']:
        line.set_color('black'); line.set_linewidth(1.5)

    ax_b.set_xticks([1, 2])
    ax_b.set_xticklabels(['Chickadee', 'Finch'])
    if sh_t and sh_z:
        U, p = mannwhitneyu(sh_t, sh_z, alternative='less')
        ax_b.set_title(f'B. Split-half allocation reliability\n'
                        f'mean {np.mean(sh_t):.3f} vs {np.mean(sh_z):.3f}, '
                        f'p = {p:.2e} (chick < finch)',
                        fontsize=11, fontweight='bold', loc='left')
    ax_b.set_ylabel('Spearman r (N_active correlation)', fontsize=10)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # -- C: Mantel test --
    ax_c = fig.add_subplot(gs[0, 2])
    mt = [x['r'] for x in results['mantel_t']]
    mz = [x['r'] for x in results['mantel_z']]
    bins_m = np.linspace(
        min(min(mt, default=0), min(mz, default=0)) - 0.05,
        max(max(mt, default=0), max(mz, default=0)) + 0.05, 25)
    ax_c.hist(mt, bins=bins_m, density=True, alpha=0.55, color=C_T,
               edgecolor='none', label=f'Chickadee (n={len(mt)})')
    ax_c.hist(mz, bins=bins_m, density=True, alpha=0.55, color=C_Z,
               edgecolor='none', label=f'Finch (n={len(mz)})')
    ax_c.axvline(np.mean(mt), color=C_T, ls='--', lw=1.5)
    ax_c.axvline(np.mean(mz), color=C_Z, ls='--', lw=1.5)
    sig_t = sum(1 for x in results['mantel_t'] if x['p'] < 0.05)
    sig_z = sum(1 for x in results['mantel_z'] if x['p'] < 0.05)
    U, p = mannwhitneyu(mt, mz, alternative='greater')
    ax_c.set_title(f'C. Mantel test (RDM ~ physical distance)\n'
                    f'sig: {sig_t}/{len(mt)} vs {sig_z}/{len(mz)}, p = {p:.2e}',
                    fontsize=11, fontweight='bold', loc='left')
    ax_c.set_xlabel('Mantel r', fontsize=10)
    ax_c.set_ylabel('Density', fontsize=10)
    ax_c.legend(fontsize=8, frameon=False)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # -- D: AP gradient --
    ax_d = fig.add_subplot(gs[1, 0])
    ap = results['ap']
    if ap:
        subdivs = sorted(ap.keys())
        means_d = [np.mean([x['shesha'] for x in ap[s]]) for s in subdivs]
        sems_d = [sp_sem([x['shesha'] for x in ap[s]])
                   if len(ap[s]) > 1 else 0 for s in subdivs]
        ns_d = [len(ap[s]) for s in subdivs]
        colors_d = [C_T if s == 'DMm' else C_T2 for s in subdivs]
        ax_d.bar(range(len(subdivs)), means_d, yerr=sems_d,
                  color=colors_d, edgecolor='white', lw=1.5, capsize=4, width=0.5)
        ax_d.set_xticks(range(len(subdivs)))
        ax_d.set_xticklabels([f'{s}\n(n={n})' for s, n in zip(subdivs, ns_d)])
    ax_d.set_title('D. A-P gradient (chickadee)\nSHESHA by subdivision',
                     fontsize=11, fontweight='bold', loc='left')
    ax_d.set_ylabel('SHESHA', fontsize=10)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    # -- E: Drift --
    ax_e = fig.add_subplot(gs[1, 1])
    drift = results['drift']
    if 'titmouse' in drift and 'zebra_finch' in drift:
        bins_d = np.linspace(-0.2, 1.0, 40)
        ax_e.hist(drift['titmouse']['xcorr'], bins=bins_d, density=True,
                   alpha=0.55, color=C_T, edgecolor='none', label='Chickadee')
        ax_e.hist(drift['zebra_finch']['xcorr'], bins=bins_d, density=True,
                   alpha=0.55, color=C_Z, edgecolor='none', label='Finch')
        ax_e.axvline(np.median(drift['titmouse']['xcorr']), color=C_T, ls='--', lw=1.5)
        ax_e.axvline(np.median(drift['zebra_finch']['xcorr']), color=C_Z, ls='--', lw=1.5)
        U, p = mannwhitneyu(drift['titmouse']['xcorr'],
                             drift['zebra_finch']['xcorr'], alternative='greater')
        ax_e.set_title(f'E. Within-session stability\n'
                        f'median {np.median(drift["titmouse"]["xcorr"]):.3f} vs '
                        f'{np.median(drift["zebra_finch"]["xcorr"]):.3f}, p = {p:.1e}',
                        fontsize=11, fontweight='bold', loc='left')
    ax_e.set_xlabel('First/second half map correlation', fontsize=10)
    ax_e.set_ylabel('Density', fontsize=10)
    ax_e.legend(fontsize=8, frameon=False)
    ax_e.spines['top'].set_visible(False)
    ax_e.spines['right'].set_visible(False)

    # -- F: Double dissociation (revised) --
    ax_f = fig.add_subplot(gs[1, 2])
    dd_t = results['dd_t']
    dd_z = results['dd_z']
    if dd_t and dd_z:
        ax_f.scatter([d['shesha'] for d in dd_t],
                      [d['split_half_r'] for d in dd_t],
                      c=C_T, alpha=0.6, s=50, edgecolors='white',
                      linewidths=0.3, label='Chickadee', zorder=3)
        ax_f.scatter([d['shesha'] for d in dd_z],
                      [d['split_half_r'] for d in dd_z],
                      c=C_Z, alpha=0.6, s=50, edgecolors='white',
                      linewidths=0.3, label='Finch', zorder=3)

        # Correlation line for chickadee
        if len(dd_t) > 5:
            r, p = spearmanr([d['shesha'] for d in dd_t],
                              [d['split_half_r'] for d in dd_t])
            ax_f.set_title(f'F. SHESHA vs allocation reliability\n'
                            f'within-chickadee rho={r:.2f}, p={p:.3f}',
                            fontsize=11, fontweight='bold', loc='left')
        else:
            ax_f.set_title('F. SHESHA vs allocation reliability',
                            fontsize=11, fontweight='bold', loc='left')

        ax_f.set_xlabel('SHESHA (geometric stability)', fontsize=10)
        ax_f.set_ylabel('Split-half allocation r', fontsize=10)
        ax_f.legend(fontsize=8, frameon=False)
    ax_f.spines['top'].set_visible(False)
    ax_f.spines['right'].set_visible(False)

    fig.suptitle(
        'Tier 1 v2: Valiant SMA / SHESHA Hypothesis Tests\n'
        'Chickadee (food-cacher) vs Zebra Finch (non-cacher)',
        fontsize=14, fontweight='bold', y=0.97)

    outpath = os.path.join(OUTPUT_DIR, 'tier1_valiant_shesha.png')
    fig.savefig(outpath, dpi=250, facecolor='white')
    print(f"\nFigure saved: {outpath}")
    plt.show()


def main():
    if not Path(DATA_PATH).exists():
        print(f"Error: {DATA_PATH} not found")
        return
    df, df_t, df_z = load_data(DATA_PATH)
    results = run_analyses(df, df_t, df_z)
    plot_results(results)


if __name__ == '__main__':
    main()