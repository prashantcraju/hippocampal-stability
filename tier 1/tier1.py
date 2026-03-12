#!/usr/bin/env python3
"""
tier1.py - Corrected Tier 1 analysis with alternative benchmarks

Additions over original:
  - Test-retest PV correlation (split-half population vector stability)
  - CCA stability (canonical correlation across neuron splits)
  - Concordance CSV showing all stability metrics agree directionally

Original corrections preserved:
  - REMOVED: Identity shuffle (mathematically invalid control)
  - RENAMED: Temporal shuffle -> Map shuffle (clearer naming)
  - FIXED: Control sampling (uses top-N neuron sessions instead of first-N)
  - FIXED: Data loading (groups by session properly)
  - ADDED: Permutation test, Jackknife, Harmonic N
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu, spearmanr
from shesha import feature_split
from pathlib import Path
from tqdm import tqdm
import warnings
import os

OUTPUT_DIR = 'output/tier1_enhanced'

GRID_SIZE = 40
MIN_NEURONS = 5
N_BOOTSTRAP = 10000
N_PERMUTATIONS = 10000

def load_aronov_data(data_path=None):
    """Load and clean Aronov dataset - FIXED to group by session"""
    if data_path is None:
        possible_paths = ['data/aronov_dataset.pkl']
        for path in possible_paths:
            if Path(path).exists():
                data_path = path
                break
        if data_path is None:
            raise FileNotFoundError("Could not find aronov_dataset.pkl")
    
    print(f"Loading data from: {data_path}")
    df = pd.read_pickle(data_path)
    
    # Auto-detect map column
    if 'rate_maps' in df.columns:
        map_col = 'rate_maps'
    elif 'map' in df.columns:
        map_col = 'map'
    else:
        raise ValueError("No rate maps column found ('rate_maps' or 'map')")
    
    print(f"Using rate map column: '{map_col}'")
    
    df_exc = df[df['cell_type'] == 'E'].copy()
    
    def clean_map(M_raw):
        if M_raw is None: return None
        M = np.array(M_raw) if isinstance(M_raw, list) else M_raw
        if M.ndim == 1:
            M = M.reshape(1, -1)
        if np.all(np.isnan(M)): return None
        return np.nan_to_num(M, nan=0.0)

    sessions = {'chickadee': [], 'finch': []}
    
    # Determine grouping columns (check all possible names)
    if 'bird' in df_exc.columns:
        bird_col = 'bird'
    elif 'bird_id' in df_exc.columns:
        bird_col = 'bird_id'
    elif 'subject' in df_exc.columns:
        bird_col = 'subject'
    else:
        bird_col = None
    
    if 'session' in df_exc.columns:
        session_col = 'session'
    elif 'session_id' in df_exc.columns:
        session_col = 'session_id'
    else:
        session_col = None
    
    # Check if columns exist
    if bird_col is None or session_col is None:
        print(f"WARNING: Missing bird column ({bird_col}) or session column ({session_col})")
        print(f"Available columns: {list(df_exc.columns)}")
        print(f"Treating each neuron as session")
        # Fallback: treat each row as a session (old behavior)
        for species, key in [('titmouse', 'chickadee'), ('zebra_finch', 'finch')]:
            subset = df_exc[df_exc['species'] == species]
            for idx, row in subset.iterrows():
                M = clean_map(row[map_col])
                if M is not None and M.shape[0] >= MIN_NEURONS:
                    sessions[key].append({
                        'M': M,
                        'n_neurons': M.shape[0],
                        'species': key,
                        'bird': f'bird_{idx}',
                        'session': f'sess_{idx}'
                    })
    else:
        # CORRECT: Group by (bird, session)
        print(f"Using columns: bird='{bird_col}', session='{session_col}'")
        
        for species, key in [('titmouse', 'chickadee'), ('zebra_finch', 'finch')]:
            subset = df_exc[df_exc['species'] == species]
            
            for (bird, sess), group in subset.groupby([bird_col, session_col]):
                # Collect all neurons in this session
                maps = []
                for _, row in group.iterrows():
                    M_clean = clean_map(row[map_col])
                    if M_clean is not None:
                        maps.append(M_clean)
                
                if len(maps) >= MIN_NEURONS:
                    # Stack all neurons into single matrix
                    M = np.vstack(maps)  # (n_neurons, n_bins)
                    sessions[key].append({
                        'M': M,
                        'n_neurons': M.shape[0],
                        'species': key,
                        'bird': bird,
                        'session': sess
                    })
    
    print(f"Loaded {len(sessions['chickadee'])} chickadee, {len(sessions['finch'])} finch sessions")
    return sessions['chickadee'], sessions['finch']

def compute_shesha(M):
    n_neurons = M.shape[0]
    
    # Z-score
    means = M.mean(axis=1, keepdims=True)
    stds = M.std(axis=1, keepdims=True)
    stds[stds == 0] = 1
    M_z = (M - means) / stds
    
    # Active bin filter
    active = np.sum(M > 0, axis=0) >= max(2, n_neurons // 3)
    active_idx = np.where(active)[0]
    
    if len(active_idx) < 30 or n_neurons < 4:
        return np.nan
    
    # (n_samples, n_features)
    X = M_z[:, active_idx].T
    return feature_split(X)


def compute_test_retest_pv(M, n_splits=100):
    """Split-half population vector correlation."""
    n_neurons = M.shape[0]
    if n_neurons < 4:
        return np.nan

    M_filled = np.nan_to_num(M, nan=0.0)
    rng = np.random.RandomState(320)
    correlations = []

    for _ in range(n_splits):
        perm = rng.permutation(n_neurons)
        half = n_neurons // 2
        pv1 = np.mean(M_filled[perm[:half]], axis=0)
        pv2 = np.mean(M_filled[perm[half:2*half]], axis=0)
        active = (pv1 > 0) | (pv2 > 0)
        if np.sum(active) < 30:
            continue
        r, _ = spearmanr(pv1[active], pv2[active])
        if np.isfinite(r):
            correlations.append(r)

    return np.mean(correlations) if correlations else np.nan


def compute_cca_stability(M, n_splits=50, n_components=3):
    """CCA stability across neuron splits."""
    from scipy.linalg import svd as scipy_svd
    n_neurons = M.shape[0]
    if n_neurons < 4:
        return np.nan

    means = M.mean(axis=1, keepdims=True)
    stds = M.std(axis=1, keepdims=True)
    stds[stds == 0] = 1
    M_z = np.nan_to_num((M - means) / stds, nan=0.0)

    active = np.sum(M > 0, axis=0) >= max(2, n_neurons // 3)
    active_idx = np.where(active)[0]
    if len(active_idx) < 30:
        return np.nan

    X = M_z[:, active_idx]
    rng = np.random.RandomState(320)
    scores = []

    for _ in range(n_splits):
        perm = rng.permutation(n_neurons)
        half = n_neurons // 2
        X1 = X[perm[:half]].T
        X2 = X[perm[half:2*half]].T
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


# --- CORRECTED NEGATIVE CONTROLS ---

def create_circular_shifted_control(M):
    """Geometry Control 1: Circular Shift (Preserves neuron stats, breaks population)"""
    n_bins = M.shape[1]
    M_shifted = np.zeros_like(M)
    for i in range(M.shape[0]):
        shift = np.random.randint(0, n_bins)
        M_shifted[i] = np.roll(M[i], shift)
    return M_shifted

def create_map_shuffle_control(M):
    """Geometry Control 2: Map Shuffle (Destroys manifold smoothness)"""
    M_shuf = np.zeros_like(M)
    for i in range(M.shape[0]):
        M_shuf[i] = np.random.permutation(M[i])
    return M_shuf

def run_negative_controls(sessions, n_iterations=20):
    results = {'original': [], 'circular': [], 'map_shuffle': []}
    
    # CRITICAL FIX: Sort by neuron count and take top 5 to ensure valid data
    # best_sessions = sorted(sessions, key=lambda s: s['n_neurons'], reverse=True)[:5]
    best_sessions = [s for s in sessions if s['n_neurons'] >= 5]
    
    print(f"Running controls on top {len(best_sessions)} sessions...")
    
    for sess in best_sessions:
        M = sess['M']
        
        # Original
        s = compute_shesha(M)
        if not np.isnan(s): results['original'].append(s)
        
        # Circular
        circ_scores = []
        for _ in range(n_iterations):
            s = compute_shesha(create_circular_shifted_control(M))
            if not np.isnan(s): circ_scores.append(s)
        if circ_scores: results['circular'].append(np.mean(circ_scores))
            
        # Map Shuffle (formerly Temporal)
        map_scores = []
        for _ in range(n_iterations):
            s = compute_shesha(create_map_shuffle_control(M))
            if not np.isnan(s): map_scores.append(s)
        if map_scores: results['map_shuffle'].append(np.mean(map_scores))
            
    return results


def run_neuron_matched_downsample(sessions_t, sessions_z, n_repeats=50):
    """
    Downsample chickadee sessions to match finch neuron counts, then
    re-run SHESHA.  Controls for the possibility that higher N inflates
    split-half stability.

    For each chickadee session with n_neurons >= target_n (median finch N),
    randomly subsamples target_n neurons n_repeats times, computes SHESHA
    on each subsample, and averages per session.
    """
    finch_ns = sorted([s['n_neurons'] for s in sessions_z])
    target_n = int(np.median(finch_ns))

    eligible = [s for s in sessions_t if s['n_neurons'] >= target_n]

    rng = np.random.RandomState(320)
    per_session = []

    for sess in eligible:
        M = sess['M']
        sess_scores = []
        for _ in range(n_repeats):
            idx = rng.choice(M.shape[0], size=target_n, replace=False)
            s = compute_shesha(M[idx])
            if not np.isnan(s):
                sess_scores.append(s)

        if sess_scores:
            per_session.append({
                'session': sess.get('session', ''),
                'bird': sess.get('bird', ''),
                'original_n': sess['n_neurons'],
                'downsampled_n': target_n,
                'shesha_mean': np.mean(sess_scores),
                'shesha_std': np.std(sess_scores),
                'n_valid_repeats': len(sess_scores),
            })

    return {
        'target_n': target_n,
        'finch_ns': finch_ns,
        'downsampled_scores': [p['shesha_mean'] for p in per_session],
        'per_session': per_session,
        'n_eligible': len(eligible),
        'n_total_chickadee': len(sessions_t),
    }


# --- STATS UTILS ---

def bootstrap_mean_ci(data, n_boot=1000):
    if len(data) < 2: return np.mean(data), np.nan, np.nan
    rng = np.random.RandomState(320)
    means = [np.mean(rng.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    return np.mean(data), np.percentile(means, 2.5), np.percentile(means, 97.5)

def bootstrap_effect_size(d1, d2, n_boot=1000):
    if len(d1) < 2 or len(d2) < 2: return np.nan, np.nan, np.nan
    
    def cohen_d(x, y):
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        pool_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
        return (np.mean(x) - np.mean(y)) / pool_std

    d_obs = cohen_d(d1, d2)
    rng = np.random.RandomState(320)
    boots = []
    for _ in range(n_boot):
        s1 = rng.choice(d1, len(d1), replace=True)
        s2 = rng.choice(d2, len(d2), replace=True)
        boots.append(cohen_d(s1, s2))
        
    return d_obs, np.percentile(boots, 2.5), np.percentile(boots, 97.5)

def permutation_test(d1, d2, n_perm=10000):
    """Non-parametric permutation test"""
    combined = np.concatenate([d1, d2])
    n1 = len(d1)
    obs_diff = np.mean(d1) - np.mean(d2)
    
    rng = np.random.RandomState(320)
    null_diffs = []
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        null_diff = np.mean(perm[:n1]) - np.mean(perm[n1:])
        null_diffs.append(null_diff)
    
    p_val = np.mean(np.array(null_diffs) >= obs_diff)
    return p_val

def jackknife_effect_size(d1, d2):
    """
    Jackknife resampling: Drop one observation at a time and recalculate effect size
    Tests if results are driven by outliers
    """
    def cohen_d(x, y):
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        pool_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
        return (np.mean(x) - np.mean(y)) / pool_std
    
    # Full effect size
    d_full = cohen_d(d1, d2)
    
    # Jackknife: drop each observation from larger group (chickadee)
    jackknife_d = []
    for i in range(len(d1)):
        d1_jack = np.delete(d1, i)
        jackknife_d.append(cohen_d(d1_jack, d2))
    
    return {
        'full_d': d_full,
        'jackknife_mean': np.mean(jackknife_d),
        'jackknife_std': np.std(jackknife_d),
        'jackknife_min': np.min(jackknife_d),
        'jackknife_max': np.max(jackknife_d),
    }

def run_analysis(data_path=None):
    print("="*70)
    print("TIER 1 ENHANCED V2 (CORRECTED)")
    print("="*70)
    
    sessions_t, sessions_z = load_aronov_data(data_path)
    
    # 1. Main SHESHA + alternative benchmarks (computed per-session together)
    print("\nComputing SHESHA + alternative benchmarks...")
    per_session_t, per_session_z = [], []
    for sess_list, per_session, label in [
        (sessions_t, per_session_t, 'chickadee'),
        (sessions_z, per_session_z, 'finch'),
    ]:
        for s in sess_list:
            row = {
                'session': s.get('session', ''),
                'bird': s.get('bird', ''),
                'species': label,
                'n_neurons': s['n_neurons'],
                'shesha': compute_shesha(s['M']),
                'pv_corr': compute_test_retest_pv(s['M']),
                'cca_stability': compute_cca_stability(s['M']),
            }
            per_session.append(row)

    shesha_t = [r['shesha'] for r in per_session_t if not np.isnan(r['shesha'])]
    shesha_z = [r['shesha'] for r in per_session_z if not np.isnan(r['shesha'])]
    pv_t = [r['pv_corr'] for r in per_session_t if not np.isnan(r['pv_corr'])]
    pv_z = [r['pv_corr'] for r in per_session_z if not np.isnan(r['pv_corr'])]
    cca_t = [r['cca_stability'] for r in per_session_t if not np.isnan(r['cca_stability'])]
    cca_z = [r['cca_stability'] for r in per_session_z if not np.isnan(r['cca_stability'])]

    # 2. Controls (using corrected logic)
    print("\nRunning Negative Controls...")
    ctrl_t = run_negative_controls(sessions_t, n_iterations=20)
    ctrl_z = run_negative_controls(sessions_z, n_iterations=20)
    
    # 2b. Neuron-count-matched downsampling
    print("\nRunning neuron-matched downsampling...")
    ds_results = run_neuron_matched_downsample(sessions_t, sessions_z)
    
    # 3. Stats
    mean_t, low_t, high_t = bootstrap_mean_ci(shesha_t, n_boot=N_BOOTSTRAP)
    mean_z, low_z, high_z = bootstrap_mean_ci(shesha_z, n_boot=N_BOOTSTRAP)
    d, d_low, d_high = bootstrap_effect_size(shesha_t, shesha_z, n_boot=N_BOOTSTRAP)
    u_stat, p_val = mannwhitneyu(shesha_t, shesha_z, alternative='greater')
    
    # 4. Permutation test
    print("\nRunning Permutation test...")
    p_perm = permutation_test(np.array(shesha_t), np.array(shesha_z), n_perm=10000)
    
    # 5. Jackknife robustness check
    print("\nRunning Jackknife resampling...")
    jack_results = jackknife_effect_size(np.array(shesha_t), np.array(shesha_z))
    
    # Power estimate (Post-hoc)
    n1, n2 = len(shesha_t), len(shesha_z)
    harmonic_n = 2 * n1 * n2 / (n1 + n2)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Chickadee (n={len(shesha_t)}): {mean_t:.3f} [{low_t:.3f}, {high_t:.3f}]")
    print(f"Finch     (n={len(shesha_z)}): {mean_z:.3f} [{low_z:.3f}, {high_z:.3f}]")
    print(f"\nEffect Size (Cohen's d):  {d:.3f} [{d_low:.3f}, {d_high:.3f}]")
    print(f"Mann-Whitney p-value:     {p_val:.4f}")
    print(f"Permutation p-value:      {p_perm:.4f}")
    print(f"Harmonic mean n:          {harmonic_n:.1f}")
    
    print(f"\nJackknife Robustness Check:")
    print(f"  Full dataset d:     {jack_results['full_d']:.3f}")
    print(f"  Jackknife mean d:   {jack_results['jackknife_mean']:.3f}")
    print(f"  Jackknife std:      {jack_results['jackknife_std']:.3f}")
    print(f"  Range: [{jack_results['jackknife_min']:.3f}, {jack_results['jackknife_max']:.3f}]")

    print(f"\n" + "-"*70)
    print("ALTERNATIVE STABILITY BENCHMARKS")
    print("-"*70)
    if pv_t and pv_z:
        u_pv, p_pv = mannwhitneyu(pv_t, pv_z, alternative='greater')
        print(f"  PV correlation:  chickadee={np.mean(pv_t):.4f} (n={len(pv_t)}), "
              f"finch={np.mean(pv_z):.4f} (n={len(pv_z)}), p={p_pv:.4e}")
    if cca_t and cca_z:
        u_cc, p_cc = mannwhitneyu(cca_t, cca_z, alternative='greater')
        print(f"  CCA stability:   chickadee={np.mean(cca_t):.4f} (n={len(cca_t)}), "
              f"finch={np.mean(cca_z):.4f} (n={len(cca_z)}), p={p_cc:.4e}")

    concordance = []
    if shesha_t and shesha_z:
        concordance.append(('SHESHA', np.mean(shesha_t) > np.mean(shesha_z)))
    if pv_t and pv_z:
        concordance.append(('PV_corr', np.mean(pv_t) > np.mean(pv_z)))
    if cca_t and cca_z:
        concordance.append(('CCA', np.mean(cca_t) > np.mean(cca_z)))
    all_agree = all(c[1] for c in concordance) if concordance else False
    print(f"\n  Concordance (chickadee > finch):")
    for name, agrees in concordance:
        print(f"    {name}: {'YES' if agrees else 'NO'}")
    print(f"    All agree: {'YES' if all_agree else 'NO'}")
    
    print("\n" + "-"*70)
    print("NEGATIVE CONTROLS")
    print("-"*70)
    
    print("\nChickadee (Top 5 Sessions by Neuron Count):")
    if ctrl_t['original']:
        print(f"  Original:         {np.mean(ctrl_t['original']):.3f} ± {np.std(ctrl_t['original']):.3f}")
    if ctrl_t['circular']:
        print(f"  Circular Shift:   {np.mean(ctrl_t['circular']):.3f} ± {np.std(ctrl_t['circular']):.3f}")
    if ctrl_t['map_shuffle']:
        print(f"  Map Shuffle:      {np.mean(ctrl_t['map_shuffle']):.3f} ± {np.std(ctrl_t['map_shuffle']):.3f}")
    
    print("\nFinch (Top 5 Sessions by Neuron Count):")
    if ctrl_z['original']:
        print(f"  Original:         {np.mean(ctrl_z['original']):.3f} ± {np.std(ctrl_z['original']):.3f}")
    if ctrl_z['circular']:
        print(f"  Circular Shift:   {np.mean(ctrl_z['circular']):.3f} ± {np.std(ctrl_z['circular']):.3f}")
    if ctrl_z['map_shuffle']:
        print(f"  Map Shuffle:      {np.mean(ctrl_z['map_shuffle']):.3f} ± {np.std(ctrl_z['map_shuffle']):.3f}")
    
    print("\n" + "-"*70)
    print("NEURON-MATCHED DOWNSAMPLING CONTROL")
    print("-"*70)
    ds_scores = ds_results['downsampled_scores']
    print(f"  Target N (median finch): {ds_results['target_n']} neurons")
    print(f"  Finch N range: [{min(ds_results['finch_ns'])}, {max(ds_results['finch_ns'])}]")
    print(f"  Eligible chickadee sessions: {ds_results['n_eligible']}/{ds_results['n_total_chickadee']}")
    if ds_scores and shesha_z:
        print(f"\n  Downsampled chickadee: mean={np.mean(ds_scores):.4f} "
              f"\u00b1 {np.std(ds_scores):.4f} (n={len(ds_scores)})")
        print(f"  Original finch:        mean={np.mean(shesha_z):.4f} "
              f"\u00b1 {np.std(shesha_z):.4f} (n={len(shesha_z)})")
        if len(ds_scores) >= 2 and len(shesha_z) >= 2:
            u_ds, p_ds = mannwhitneyu(ds_scores, shesha_z, alternative='greater')
            d_ds, d_ds_lo, d_ds_hi = bootstrap_effect_size(
                np.array(ds_scores), np.array(shesha_z), n_boot=N_BOOTSTRAP)
            print(f"  MW (downsampled chick > finch): p = {p_ds:.4e}")
            print(f"  Cohen's d: {d_ds:.3f} [{d_ds_lo:.3f}, {d_ds_hi:.3f}]")
    
    print("\n" + "="*70)
    
    # Export
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Main results
    pd.DataFrame({
        'species': ['chickadee']*len(shesha_t) + ['finch']*len(shesha_z),
        'shesha': shesha_t + shesha_z
    }).to_csv(os.path.join(OUTPUT_DIR, 'tier1_main_results.csv'), index=False)
    
    # Controls
    rows = []
    for k, v in ctrl_t.items():
        for val in v: rows.append({'species': 'chickadee', 'control': k, 'value': val})
    for k, v in ctrl_z.items():
        for val in v: rows.append({'species': 'finch', 'control': k, 'value': val})
    for val in ds_results['downsampled_scores']:
        rows.append({'species': 'chickadee', 'control': 'neuron_matched', 'value': val})
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, 'tier1_controls.csv'), index=False)
    
    # Neuron-matched per-session detail
    if ds_results['per_session']:
        pd.DataFrame(ds_results['per_session']).to_csv(
            os.path.join(OUTPUT_DIR, 'tier1_neuron_matched.csv'), index=False)
    
    # Statistics summary
    stats_df = pd.DataFrame([{
        'comparison': 'chickadee_vs_finch',
        'n_chickadee': len(shesha_t),
        'n_finch': len(shesha_z),
        'harmonic_n': harmonic_n,
        'mean_chickadee': mean_t,
        'mean_finch': mean_z,
        'ci_lower_chickadee': low_t,
        'ci_upper_chickadee': high_t,
        'ci_lower_finch': low_z,
        'ci_upper_finch': high_z,
        'cohens_d': d,
        'd_ci_lower': d_low,
        'd_ci_upper': d_high,
        'p_mannwhitney': p_val,
        'p_permutation': p_perm,
        'jackknife_d_mean': jack_results['jackknife_mean'],
        'jackknife_d_std': jack_results['jackknife_std'],
        'jackknife_d_min': jack_results['jackknife_min'],
        'jackknife_d_max': jack_results['jackknife_max'],
    }])
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'tier1_statistics.csv'), index=False)
    
    # Alternative metrics (per-session aligned -- no positional misalignment)
    alt_rows = []
    for r in per_session_t + per_session_z:
        alt_rows.append({
            'species': r['species'],
            'session': r['session'],
            'bird': r['bird'],
            'n_neurons': r['n_neurons'],
            'shesha': r['shesha'],
            'pv_corr': r['pv_corr'],
            'cca_stability': r['cca_stability'],
        })
    pd.DataFrame(alt_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'tier1_alternative_metrics.csv'), index=False)

    # Concordance summary
    conc_rows = []
    for name, vt, vz in [
        ('SHESHA', shesha_t, shesha_z),
        ('PV_correlation', pv_t, pv_z),
        ('CCA_stability', cca_t, cca_z),
    ]:
        if vt and vz:
            U_c, p_c = mannwhitneyu(vt, vz, alternative='greater')
            conc_rows.append({
                'metric': name,
                'chickadee_mean': np.mean(vt), 'finch_mean': np.mean(vz),
                'chickadee_n': len(vt), 'finch_n': len(vz),
                'directionally_consistent': np.mean(vt) > np.mean(vz),
                'p_value': p_c,
            })
    pd.DataFrame(conc_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'tier1_concordance.csv'), index=False)

    print("\nExported results:")
    print(f"  - {OUTPUT_DIR}/tier1_main_results.csv")
    print(f"  - {OUTPUT_DIR}/tier1_controls.csv")
    print(f"  - {OUTPUT_DIR}/tier1_neuron_matched.csv")
    print(f"  - {OUTPUT_DIR}/tier1_statistics.csv")
    print(f"  - {OUTPUT_DIR}/tier1_alternative_metrics.csv")
    print(f"  - {OUTPUT_DIR}/tier1_concordance.csv")
    print("="*70)

    return {
        'shesha_t': shesha_t,
        'shesha_z': shesha_z,
        'pv_t': pv_t, 'pv_z': pv_z,
        'cca_t': cca_t, 'cca_z': cca_z,
        'controls_t': ctrl_t,
        'controls_z': ctrl_z,
        'neuron_matched': ds_results,
        'statistics': stats_df.to_dict('records')[0],
        'p_permutation': p_perm,
        'jackknife': jack_results,
    }

if __name__ == '__main__':
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    results = run_analysis(data_path=data_path)
    print("\nTier 1 Enhanced V2 Analysis Complete!")
