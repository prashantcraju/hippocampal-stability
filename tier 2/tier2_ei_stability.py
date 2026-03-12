#!/usr/bin/env python3
"""
tier2_ei_stability.py - Excitatory vs Inhibitory contributions to
geometric stability in the hippocampal spatial code.

Hypothesis: If the geometric attractor is maintained by excitatory
recurrence (standard continuous attractor model), then:
  - E neurons should carry high SHESHA (stable geometry)
  - I neurons should show lower SHESHA (not directly encoding space)
  - Removing I neurons should not destroy geometric structure
  - Removing E neurons should collapse it

Alternative: If E/I balance sculpts the manifold (inhibitory
stabilization model), then:
  - Both E and I populations should show spatial structure
  - I neurons should show ANTI-correlated spatial tuning
    (inhibiting nearby representations)
  - SHESHA might be high for both, but Mantel r might differ

Analyses:
  1. SHESHA by cell type (E vs I) within each species
  2. Mantel test by cell type
  3. Spatial information by cell type
  4. Within-session stability (xcorr_map) by cell type
  5. "Knockout" test: compute SHESHA using only E or only I neurons
     from the same sessions, compare to full population
  6. E-I spatial correlation: do E and I populations encode
     similar or complementary spatial information?

Usage:
    python tier2_ei_stability.py

Expects: aronov_dataset.pkl
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu, spearmanr, sem as sp_sem, wilcoxon
from shesha import feature_split

OUTPUT_DIR = 'output/tier2_ei_session'
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA_PATH = 'data/aronov_dataset.pkl'
GRID_SIZE = 40
N_BINS = GRID_SIZE ** 2
MIN_NEURONS_FULL = 5
MIN_NEURONS_TYPE = 3  # minimum per cell type for within-type analysis

C_T = '#C62D50'    # chickadee
C_Z = '#1C3D8F'    # finch
C_E = '#2E8B57'    # excitatory
C_I = '#FF8C00'    # inhibitory
C_TE = '#C62D50'   # chickadee excitatory
C_TI = '#E8909F'   # chickadee inhibitory
C_ZE = '#1C3D8F'   # finch excitatory
C_ZI = '#7B9BD4'   # finch inhibitory


def load_data(path):
    df = pd.read_pickle(path)
    print(f"Total units: {len(df)}")
    for sp in ['titmouse', 'zebra_finch']:
        sub = df[df['species'] == sp]
        n_e = (sub['cell_type'] == 'E').sum()
        n_i = (sub['cell_type'] == 'I').sum()
        label = 'Chickadee' if sp == 'titmouse' else 'Finch'
        print(f"  {label}: {n_e} E, {n_i} I ({n_i/(n_e+n_i)*100:.1f}% I)")
    return df


def get_session_maps_by_type(df, species, min_e=MIN_NEURONS_TYPE,
                               min_i=MIN_NEURONS_TYPE):
    """
    For each session, return separate E and I population matrices,
    plus the combined matrix.
    """
    sub = df[df['species'] == species]
    sessions = {}

    for sess in sub['session'].unique():
        sdf = sub[sub['session'] == sess]

        maps_e, maps_i, maps_all = [], [], []
        for _, row in sdf.iterrows():
            m = row['map']
            if not isinstance(m, np.ndarray) or m.size != N_BINS:
                continue
            flat = np.nan_to_num(m.flatten(), nan=0.0)
            maps_all.append(flat)
            if row['cell_type'] == 'E':
                maps_e.append(flat)
            else:
                maps_i.append(flat)

        if len(maps_all) < MIN_NEURONS_FULL:
            continue

        sessions[sess] = {
            'M_all': np.vstack(maps_all) if maps_all else None,
            'M_e': np.vstack(maps_e) if len(maps_e) >= min_e else None,
            'M_i': np.vstack(maps_i) if len(maps_i) >= min_i else None,
            'n_all': len(maps_all),
            'n_e': len(maps_e),
            'n_i': len(maps_i),
            'bird': sdf['bird'].iloc[0],
        }

    return sessions


def compute_shesha(M, n_splits=100, rng=None, min_neurons=4):
    """
    Split-half RDM correlation across neuron halves.
    Uses the shesha-geometry package (pip install shesha-geometry).
    feature_split expects (n_samples, n_features) = (bins, neurons).

    min_neurons=4 (strict, default): each half has >= 2 neurons, avoiding
      degenerate single-neuron RDMs.
    min_neurons=3 (relaxed): allows one half of 1 neuron; used only as a
      sensitivity check to match the subspace-angle sample size.
    """
    if M is None:
        return np.nan, np.nan

    n_neurons, n_bins = M.shape
    if n_neurons < min_neurons:
        return np.nan, np.nan

    # Z-score each neuron
    means = M.mean(axis=1, keepdims=True)
    stds = M.std(axis=1, keepdims=True)
    stds[stds == 0] = 1
    M_z = (M - means) / stds

    active = np.sum(M > 0, axis=0) >= max(2, n_neurons // 3)
    active_idx = np.where(active)[0]
    if len(active_idx) < 30:
        return np.nan, np.nan

    # Transpose: (active_bins x neurons) for shesha package
    X = M_z[:, active_idx].T

    score = feature_split(
        X,
        n_splits=n_splits,
        metric='cosine',
        seed=320,
        max_samples=None,
    )

    return score, np.nan


def compute_mantel(M, rng=None):
    """Quick Mantel r (no permutation test, just observed correlation)."""
    if M is None:
        return np.nan
    if rng is None:
        rng = np.random.RandomState(320)

    n_neurons, n_bins = M.shape
    means = M.mean(axis=1, keepdims=True)
    stds = M.std(axis=1, keepdims=True)
    stds[stds == 0] = 1
    M_z = (M - means) / stds

    active = np.sum(M > 0, axis=0) >= max(2, n_neurons // 3)
    active_idx = np.where(active)[0]
    if len(active_idx) < 50:
        return np.nan

    # Subsample for speed (this is just for observed r)
    if len(active_idx) > 300:
        active_idx = rng.choice(active_idx, 300, replace=False)
        active_idx.sort()

    rows = active_idx // GRID_SIZE
    cols = active_idx % GRID_SIZE
    phys = pdist(np.column_stack([rows, cols]), metric='euclidean')
    X = M_z[:, active_idx].T
    neural = pdist(X, metric='euclidean') / np.sqrt(n_neurons)

    valid = np.isfinite(phys) & np.isfinite(neural)
    if np.sum(valid) < 100:
        return np.nan
    r, _ = spearmanr(phys[valid], neural[valid])
    return r


def compute_principal_angles(M_e, M_i, n_components=3, min_neurons=3):
    """
    Compute principal angles between E and I principal subspaces.

    Uses SVD on each population to extract top-k principal components,
    then computes canonical angles between the two subspaces via SVD
    of the cross-product matrix.  Returns angles in degrees.

    Small angles = overlapping subspaces (shared spatial structure).
    Large angles = orthogonal dynamics (independent coding).

    min_neurons=3 (relaxed, default): matches global MIN_NEURONS_TYPE;
      SVD is valid whenever n_neurons >= n_components.
    min_neurons=4 (strict): matches compute_shesha's default threshold,
      used to produce a sample-matched comparison with the SHESHA analyses.
    """
    if M_e is None or M_i is None:
        return np.full(n_components, np.nan)

    n_e, n_bins = M_e.shape
    n_i = M_i.shape[0]
    if n_e < min_neurons or n_i < min_neurons:
        return np.full(n_components, np.nan)
    if min(n_e, n_bins) < n_components or min(n_i, n_bins) < n_components:
        return np.full(n_components, np.nan)

    M_e_z = np.nan_to_num(M_e, nan=0.0)
    M_i_z = np.nan_to_num(M_i, nan=0.0)
    M_e_z = M_e_z - M_e_z.mean(axis=0)
    M_i_z = M_i_z - M_i_z.mean(axis=0)

    try:
        U_e, _, _ = np.linalg.svd(M_e_z.T, full_matrices=False)
        U_i, _, _ = np.linalg.svd(M_i_z.T, full_matrices=False)

        k = min(n_components, U_e.shape[1], U_i.shape[1])
        Q_e = U_e[:, :k]
        Q_i = U_i[:, :k]

        _, sigma, _ = np.linalg.svd(Q_e.T @ Q_i)
        sigma = np.clip(sigma[:k], -1, 1)
        angles_deg = np.degrees(np.arccos(sigma))
        result = np.full(n_components, np.nan)
        result[:len(angles_deg)] = angles_deg
        return result
    except np.linalg.LinAlgError:
        return np.full(n_components, np.nan)


def compute_ei_spatial_correlation(M_e, M_i):
    """
    Correlation between E and I population spatial maps.

    For each spatial bin, compute the mean E rate and mean I rate.
    Then correlate across bins.

    High positive r: E and I co-activate (same spatial tuning)
    Negative r: E and I are anti-correlated (inhibitory surround)
    Near zero: independent spatial maps
    """
    if M_e is None or M_i is None:
        return np.nan

    # Mean rate per bin across neurons of each type
    mean_e = np.mean(M_e, axis=0)
    mean_i = np.mean(M_i, axis=0)

    # Only bins with some activity
    active = (mean_e > 0) | (mean_i > 0)
    if np.sum(active) < 30:
        return np.nan

    r, _ = spearmanr(mean_e[active], mean_i[active])
    return r


# ======================================================================
# MAIN
# ======================================================================
def run_analyses(df):
    print("\n" + "="*70)
    print("TIER 2: EXCITATORY vs INHIBITORY STABILITY")
    print("="*70)

    results = {}

    for species, label in [('titmouse', 'Chickadee'), ('zebra_finch', 'Finch')]:
        print(f"\n{'_'*50}")
        print(f"  {label}")
        print(f"{'_'*50}")

        sessions = get_session_maps_by_type(df, species)
        print(f"  Sessions with >= {MIN_NEURONS_FULL} total neurons: {len(sessions)}")

        n_with_e = sum(1 for s in sessions.values() if s['M_e'] is not None)
        n_with_i = sum(1 for s in sessions.values() if s['M_i'] is not None)
        print(f"  Sessions with >= {MIN_NEURONS_TYPE} E neurons: {n_with_e}")
        print(f"  Sessions with >= {MIN_NEURONS_TYPE} I neurons: {n_with_i}")

        # -- 1. SHESHA by cell type --
        # Primary (strict): min_neurons=4 so each half has >= 2 neurons.
        # Sensitivity (relaxed): min_neurons=3 to match subspace-angle sample size.
        print(f"\n  1. SHESHA by cell type:")
        shesha_all, shesha_e, shesha_i = [], [], []
        shesha_e_relaxed, shesha_i_relaxed = [], []
        shesha_session_rows = []

        for sess, data in sessions.items():
            s_all, _ = compute_shesha(data['M_all'])
            s_e,   _ = compute_shesha(data['M_e'],   min_neurons=4)
            s_i,   _ = compute_shesha(data['M_i'],   min_neurons=4)
            s_e_rx, _ = compute_shesha(data['M_e'],  min_neurons=3)
            s_i_rx, _ = compute_shesha(data['M_i'],  min_neurons=3)

            shesha_session_rows.append({
                'session': sess,
                'bird': data['bird'],
                'n_all': data['n_all'],
                'n_e': data['n_e'],
                'n_i': data['n_i'],
                'shesha_all': s_all,
                'shesha_e': s_e,
                'shesha_i': s_i,
                'shesha_e_relaxed': s_e_rx,
                'shesha_i_relaxed': s_i_rx,
            })

            if not np.isnan(s_all):
                shesha_all.append(s_all)
            if not np.isnan(s_e):
                shesha_e.append(s_e)
            if not np.isnan(s_i):
                shesha_i.append(s_i)
            if not np.isnan(s_e_rx):
                shesha_e_relaxed.append(s_e_rx)
            if not np.isnan(s_i_rx):
                shesha_i_relaxed.append(s_i_rx)

        print(f"    All:  mean={np.mean(shesha_all):.4f} (n={len(shesha_all)})")
        if shesha_e:
            print(f"    E (strict, n>=4):   mean={np.mean(shesha_e):.4f} (n={len(shesha_e)})")
        if shesha_e_relaxed:
            print(f"    E (relaxed, n>=3):  mean={np.mean(shesha_e_relaxed):.4f} (n={len(shesha_e_relaxed)})")
        if shesha_i:
            print(f"    I (strict, n>=4):   mean={np.mean(shesha_i):.4f} (n={len(shesha_i)})")
        if shesha_i_relaxed:
            print(f"    I (relaxed, n>=3):  mean={np.mean(shesha_i_relaxed):.4f} (n={len(shesha_i_relaxed)})")
        if shesha_e and shesha_i and len(shesha_e) >= 3 and len(shesha_i) >= 3:
            U, p = mannwhitneyu(shesha_e, shesha_i, alternative='two-sided')
            print(f"    E vs I (strict): p={p:.4e}")
        if shesha_e_relaxed and shesha_i_relaxed and len(shesha_e_relaxed) >= 3 and len(shesha_i_relaxed) >= 3:
            U, p = mannwhitneyu(shesha_e_relaxed, shesha_i_relaxed, alternative='two-sided')
            print(f"    E vs I (relaxed): p={p:.4e}")

        # -- 2. Mantel r by cell type --
        print(f"\n  2. Mantel r by cell type:")
        mantel_all, mantel_e, mantel_i = [], [], []
        mantel_session_rows = []

        for sess, data in sessions.items():
            m_all = compute_mantel(data['M_all'])
            m_e = compute_mantel(data['M_e'])
            m_i = compute_mantel(data['M_i'])

            mantel_session_rows.append({
                'session': sess,
                'bird': data['bird'],
                'n_all': data['n_all'],
                'n_e': data['n_e'],
                'n_i': data['n_i'],
                'mantel_all': m_all,
                'mantel_e': m_e,
                'mantel_i': m_i,
            })

            if not np.isnan(m_all):
                mantel_all.append(m_all)
            if not np.isnan(m_e):
                mantel_e.append(m_e)
            if not np.isnan(m_i):
                mantel_i.append(m_i)

        print(f"    All:  mean r={np.mean(mantel_all):.4f} (n={len(mantel_all)})")
        if mantel_e:
            print(f"    E:    mean r={np.mean(mantel_e):.4f} (n={len(mantel_e)})")
        if mantel_i:
            print(f"    I:    mean r={np.mean(mantel_i):.4f} (n={len(mantel_i)})")
        if mantel_e and mantel_i and len(mantel_e) >= 3 and len(mantel_i) >= 3:
            U, p = mannwhitneyu(mantel_e, mantel_i, alternative='two-sided')
            print(f"    E vs I: p={p:.4e}")

        # -- 3. Per-cell spatial information by type --
        print(f"\n  3. Spatial information by cell type:")
        sub = df[df['species'] == species]
        info_e = sub[sub['cell_type'] == 'E']['info'].dropna()
        info_i = sub[sub['cell_type'] == 'I']['info'].dropna()
        print(f"    E:  median={info_e.median():.4f} bits/spike (n={len(info_e)})")
        print(f"    I:  median={info_i.median():.4f} bits/spike (n={len(info_i)})")
        if len(info_e) >= 3 and len(info_i) >= 3:
            U, p = mannwhitneyu(info_e, info_i, alternative='two-sided')
            print(f"    E vs I: p={p:.4e}")

        # Fraction spatially selective
        sel_e = sub[(sub['cell_type'] == 'E') & (sub['spatially_selective'] == True)]
        sel_i = sub[(sub['cell_type'] == 'I') & (sub['spatially_selective'] == True)]
        tot_e = (sub['cell_type'] == 'E').sum()
        tot_i = (sub['cell_type'] == 'I').sum()
        print(f"    Spatially selective: E={len(sel_e)}/{tot_e} "
              f"({len(sel_e)/tot_e*100:.1f}%), "
              f"I={len(sel_i)}/{tot_i} ({len(sel_i)/tot_i*100:.1f}%)")

        # -- 4. Within-session stability by type --
        print(f"\n  4. Within-session stability (xcorr_map):")
        xcorr_e = sub[sub['cell_type'] == 'E']['xcorr_map'].dropna()
        xcorr_i = sub[sub['cell_type'] == 'I']['xcorr_map'].dropna()
        print(f"    E:  median={xcorr_e.median():.4f} (n={len(xcorr_e)})")
        print(f"    I:  median={xcorr_i.median():.4f} (n={len(xcorr_i)})")
        if len(xcorr_e) >= 3 and len(xcorr_i) >= 3:
            U, p = mannwhitneyu(xcorr_e, xcorr_i, alternative='two-sided')
            print(f"    E vs I: p={p:.4e}")

        # -- 5. E-I spatial correlation --
        print(f"\n  5. E-I spatial correlation:")
        ei_corrs = []
        ei_corr_rows = []
        for sess, data in sessions.items():
            r = compute_ei_spatial_correlation(data['M_e'], data['M_i'])
            ei_corr_rows.append({
                'session': sess,
                'bird': data['bird'],
                'n_e': data['n_e'],
                'n_i': data['n_i'],
                'ei_spatial_r': r,
            })
            if not np.isnan(r):
                ei_corrs.append(r)
        if ei_corrs:
            print(f"    Mean E-I spatial r: {np.mean(ei_corrs):.4f} "
                  f"(n={len(ei_corrs)} sessions)")
            print(f"    Range: [{min(ei_corrs):.3f}, {max(ei_corrs):.3f}]")

        # -- 5b. E-I subspace angle analysis --
        # Relaxed (min_neurons=3): matches global inclusion criterion, n=21.
        # Strict  (min_neurons=4): matches SHESHA threshold, n=12; sample-matched comparison.
        print(f"\n  5b. E-I principal subspace angles:")
        angle_rows = []
        all_angles, all_angles_strict = [], []
        for sess, data in sessions.items():
            angles        = compute_principal_angles(data['M_e'], data['M_i'], min_neurons=3)
            angles_strict = compute_principal_angles(data['M_e'], data['M_i'], min_neurons=4)
            angle_rows.append({
                'session': sess, 'bird': data['bird'],
                'n_e': data['n_e'], 'n_i': data['n_i'],
                # relaxed (n>=3)
                'angle_1': angles[0], 'angle_2': angles[1], 'angle_3': angles[2],
                'mean_angle': np.nanmean(angles),
                # strict (n>=4)
                'angle_1_strict': angles_strict[0],
                'angle_2_strict': angles_strict[1],
                'angle_3_strict': angles_strict[2],
                'mean_angle_strict': np.nanmean(angles_strict),
            })
            if not np.isnan(angles[0]):
                all_angles.append(np.nanmean(angles))
            if not np.isnan(angles_strict[0]):
                all_angles_strict.append(np.nanmean(angles_strict))

        if all_angles:
            print(f"    Relaxed (n>=3): mean angle={np.mean(all_angles):.1f}° "
                  f"(n={len(all_angles)} sessions)")
            near_orth = sum(1 for a in all_angles if a > 45)
            print(f"      Range: [{min(all_angles):.1f}°, {max(all_angles):.1f}°]  "
                  f"sessions > 45°: {near_orth}/{len(all_angles)}")
        if all_angles_strict:
            print(f"    Strict  (n>=4): mean angle={np.mean(all_angles_strict):.1f}° "
                  f"(n={len(all_angles_strict)} sessions, matched to SHESHA n)")
            near_orth_s = sum(1 for a in all_angles_strict if a > 45)
            print(f"      Range: [{min(all_angles_strict):.1f}°, {max(all_angles_strict):.1f}°]  "
                  f"sessions > 45°: {near_orth_s}/{len(all_angles_strict)}")
        if all_angles:
            print(f"    Interpretation: {'E and I occupy largely distinct subspaces' if np.mean(all_angles) > 30 else 'E and I share substantial subspace overlap'}")

        # -- 6. Knockout test --
        # Strict (min_neurons=4): primary analysis; each half >= 2 neurons.
        # Relaxed (min_neurons=3): sensitivity check matching subspace-angle n.
        print(f"\n  6. Knockout test (SHESHA with only E or only I):")
        ko_full, ko_e, ko_i = [], [], []
        ko_session_rows = []
        for sess, data in sessions.items():
            s_full,    _ = compute_shesha(data['M_all'],                 )
            s_e,       _ = compute_shesha(data['M_e'],   min_neurons=4   )
            s_i,       _ = compute_shesha(data['M_i'],   min_neurons=4   )
            s_e_rx,    _ = compute_shesha(data['M_e'],   min_neurons=3   )
            s_i_rx,    _ = compute_shesha(data['M_i'],   min_neurons=3   )

            ko_session_rows.append({
                'session': sess,
                'bird': data['bird'],
                'n_all': data['n_all'],
                'n_e': data['n_e'],
                'n_i': data['n_i'],
                # strict
                'shesha_full': s_full,
                'shesha_e_only': s_e,
                'shesha_i_only': s_i,
                'diff_full_minus_e': s_full - s_e if (not np.isnan(s_full) and not np.isnan(s_e)) else np.nan,
                'diff_full_minus_i': s_full - s_i if (not np.isnan(s_full) and not np.isnan(s_i)) else np.nan,
                # relaxed
                'shesha_e_only_relaxed': s_e_rx,
                'shesha_i_only_relaxed': s_i_rx,
                'diff_full_minus_e_relaxed': s_full - s_e_rx if (not np.isnan(s_full) and not np.isnan(s_e_rx)) else np.nan,
                'diff_full_minus_i_relaxed': s_full - s_i_rx if (not np.isnan(s_full) and not np.isnan(s_i_rx)) else np.nan,
            })

            if not np.isnan(s_full):
                ko_full.append(s_full)
                ko_e.append(s_e if not np.isnan(s_e) else np.nan)
                ko_i.append(s_i if not np.isnan(s_i) else np.nan)

        ko_e_valid = [x for x in ko_e if not np.isnan(x)]
        ko_i_valid = [x for x in ko_i if not np.isnan(x)]
        print(f"    Full population: mean SHESHA = {np.mean(ko_full):.4f}")
        if ko_e_valid:
            print(f"    E-only (strict, n>=4): mean SHESHA = {np.mean(ko_e_valid):.4f}")
        if ko_i_valid:
            print(f"    I-only (strict, n>=4): mean SHESHA = {np.mean(ko_i_valid):.4f}")

        def _run_paired_synergy_test(label, paired_key_e, paired_key_full='shesha_full'):
            """Helper: run Wilcoxon on Full vs E-only for a given E key."""
            pairs = [(r[paired_key_full], r[paired_key_e])
                     for r in ko_session_rows
                     if not np.isnan(r[paired_key_full]) and not np.isnan(r[paired_key_e])]
            if len(pairs) < 5:
                print(f"\n    Paired Full vs E-only ({label}): insufficient sessions "
                      f"(n={len(pairs)}, need >= 5)")
                return
            diffs = [f - e for f, e in pairs]
            n_gt = sum(1 for d in diffs if d > 0)
            print(f"\n    Paired Full vs E-only ({label}):")
            print(f"      n sessions with both valid: {len(pairs)}")
            print(f"      mean(Full - E-only) = {np.mean(diffs):.4f}")
            print(f"      sessions where Full > E-only: {n_gt}/{len(pairs)}")
            try:
                stat, p_w = wilcoxon([f for f, _ in pairs],
                                     [e for _, e in pairs],
                                     alternative='greater')
                print(f"      Wilcoxon signed-rank: W={stat:.1f}, p={p_w:.4e}")
            except ValueError as exc:
                print(f"      Wilcoxon test could not be computed: {exc}")

        _run_paired_synergy_test("strict, n>=4",   paired_key_e='shesha_e_only')
        _run_paired_synergy_test("relaxed, n>=3",  paired_key_e='shesha_e_only_relaxed')

        # Build per-cell data for panels C and D
        cell_rows = []
        for _, row in sub.iterrows():
            cell_rows.append({
                'session': row.get('session', np.nan),
                'bird': row.get('bird', np.nan),
                'cell_type': row.get('cell_type', np.nan),
                'info': row.get('info', np.nan),
                'xcorr_map': row.get('xcorr_map', np.nan),
                'spatially_selective': row.get('spatially_selective', np.nan),
            })

        results[species] = {
            'sessions': sessions,
            'shesha_all': shesha_all,
            'shesha_e': shesha_e,
            'shesha_i': shesha_i,
            'shesha_e_relaxed': shesha_e_relaxed,
            'shesha_i_relaxed': shesha_i_relaxed,
            'mantel_all': mantel_all,
            'mantel_e': mantel_e,
            'mantel_i': mantel_i,
            'ei_corrs': ei_corrs,
            'info_e': info_e.values,
            'info_i': info_i.values,
            'xcorr_e': xcorr_e.values,
            'xcorr_i': xcorr_i.values,
            'principal_angles': all_angles,
            'principal_angles_strict': all_angles_strict,
            'angle_rows': angle_rows,
            'shesha_session_rows': shesha_session_rows,
            'mantel_session_rows': mantel_session_rows,
            'ei_corr_rows': ei_corr_rows,
            'cell_rows': cell_rows,
            'ko_session_rows': ko_session_rows,
        }

    # -- Export CSVs --
    print("\n" + "-"*60)
    print("Exporting results...")
    export_all_plot_data(results)

    # -- Consolidated summary printed after all exports --
    print_final_summary(results)

    return results


def print_final_summary(results):
    """
    Print a single consolidated table of all new-test results so they
    are visible together in the run log without hunting through sections.
    """
    print("\n" + "="*70)
    print("TIER 2 CONSOLIDATED RESULTS SUMMARY")
    print("(strict = n>=4 neurons/type  |  relaxed = n>=3 neurons/type)")
    print("="*70)

    for species, label in [('titmouse', 'Chickadee'), ('zebra_finch', 'Finch')]:
        r = results[species]
        ksr = r.get('ko_session_rows', [])
        print(f"\n  {label}")
        print(f"  {'─'*60}")

        # --- SHESHA by cell type ---
        print(f"  SHESHA E-only (strict  n>=4): "
              f"mean={np.nanmean(r['shesha_e']):.4f}  n={len(r['shesha_e'])}")
        print(f"  SHESHA E-only (relaxed n>=3): "
              f"mean={np.nanmean(r['shesha_e_relaxed']):.4f}  n={len(r['shesha_e_relaxed'])}")
        print(f"  SHESHA I-only (strict  n>=4): "
              f"mean={np.nanmean(r['shesha_i']):.4f}  n={len(r['shesha_i'])}")
        print(f"  SHESHA I-only (relaxed n>=3): "
              f"mean={np.nanmean(r['shesha_i_relaxed']):.4f}  n={len(r['shesha_i_relaxed'])}")

        for e_vals, i_vals, tag in [
            (r['shesha_e'],         r['shesha_i'],         'strict '),
            (r['shesha_e_relaxed'], r['shesha_i_relaxed'], 'relaxed'),
        ]:
            if len(e_vals) >= 3 and len(i_vals) >= 3:
                _, p = mannwhitneyu(e_vals, i_vals, alternative='two-sided')
                print(f"  SHESHA E vs I ({tag}):  MWU p={p:.4e}")

        # --- Subspace angles ---
        ang_r = r['principal_angles']
        ang_s = r['principal_angles_strict']
        if ang_r:
            pct_r = 100 * sum(1 for a in ang_r if a > 45) / len(ang_r)
            print(f"  Subspace angles (relaxed n>=3): "
                  f"mean={np.mean(ang_r):.1f}°  n={len(ang_r)}  >45°: {pct_r:.0f}%")
        if ang_s:
            pct_s = 100 * sum(1 for a in ang_s if a > 45) / len(ang_s)
            print(f"  Subspace angles (strict  n>=4): "
                  f"mean={np.mean(ang_s):.1f}°  n={len(ang_s)}  >45°: {pct_s:.0f}%")

        # --- Paired synergy test ---
        for e_key, tag in [('shesha_e_only', 'strict '), ('shesha_e_only_relaxed', 'relaxed')]:
            pairs = [(row['shesha_full'], row[e_key])
                     for row in ksr
                     if not np.isnan(row['shesha_full']) and not np.isnan(row[e_key])]
            diffs = [f - e for f, e in pairs]
            n_gt = sum(1 for d in diffs if d > 0)
            if len(pairs) >= 5:
                try:
                    w, p_w = wilcoxon([f for f, _ in pairs], [e for _, e in pairs],
                                      alternative='greater')
                    print(f"  Paired Full>E-only ({tag}): "
                          f"n={len(pairs)}  mean_diff={np.mean(diffs):.4f}  "
                          f"{n_gt}/{len(pairs)} sessions  W={w:.1f}  p={p_w:.4e}")
                except ValueError:
                    print(f"  Paired Full>E-only ({tag}): "
                          f"n={len(pairs)}  Wilcoxon could not be computed")
            else:
                print(f"  Paired Full>E-only ({tag}): n={len(pairs)} (insufficient for test)")

    print("\n" + "="*70)


def export_all_plot_data(results):
    """
    Export ALL data underlying every plotted panel to CSV files.

    Files produced:
      tier2_ei_summary.csv            - aggregated means + p-values for ALL variants
                                        (strict/relaxed SHESHA, subspace angles, paired synergy)
      tier2_threshold_comparison.csv  - four cross-check rows (strict/relaxed × SHESHA/angles)
                                        addressing the n=12 vs n=21 discrepancy
      tier2_session_shesha.csv        - per-session SHESHA: all / E strict / I strict / E relaxed / I relaxed
      tier2_session_mantel.csv        - panel B: per-session Mantel r (all/E/I)
      tier2_cell_info_xcorr.csv       - panels C & D: per-cell info + xcorr_map
      tier2_session_ei_corr.csv       - panel E: per-session E-I spatial correlation
      tier2_session_subspace_angles.csv - per-session angles: relaxed (n>=3) + strict (n>=4) columns
      tier2_knockout_summary.csv      - panel F: knockout bar chart + paired Wilcoxon (both thresholds)
      tier2_knockout_paired.csv       - per-session full/E-only/I-only SHESHA (strict + relaxed)
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. tier2_ei_summary.csv
    #    Aggregated means + Mann-Whitney p for E vs I, all metric variants
    # ----------------------------------------------------------------
    summary_rows = []
    for species, label in [('titmouse', 'chickadee'), ('zebra_finch', 'finch')]:
        r = results[species]
        for metric, e_vals, i_vals, threshold in [
            ('shesha',         r['shesha_e'],         r['shesha_i'],         'strict_n>=4'),
            ('shesha',         r['shesha_e_relaxed'],  r['shesha_i_relaxed'],  'relaxed_n>=3'),
            ('mantel_r',       r['mantel_e'],          r['mantel_i'],          'default'),
        ]:
            if e_vals and i_vals:
                p = mannwhitneyu(e_vals, i_vals, alternative='two-sided')[1] \
                    if len(e_vals) >= 3 and len(i_vals) >= 3 else np.nan
                summary_rows.append({
                    'species': label, 'metric': metric, 'threshold': threshold,
                    'E_mean': np.mean(e_vals), 'E_sem': sp_sem(e_vals) if len(e_vals) > 1 else np.nan,
                    'E_n': len(e_vals),
                    'I_mean': np.mean(i_vals), 'I_sem': sp_sem(i_vals) if len(i_vals) > 1 else np.nan,
                    'I_n': len(i_vals),
                    'mwu_p_two_sided': p,
                })
        # Subspace angles — summary row per threshold
        for angles_list, threshold in [
            (r['principal_angles'],        'relaxed_n>=3'),
            (r['principal_angles_strict'], 'strict_n>=4'),
        ]:
            if angles_list:
                summary_rows.append({
                    'species': label, 'metric': 'subspace_mean_angle_deg',
                    'threshold': threshold,
                    'E_mean': np.mean(angles_list),
                    'E_sem': sp_sem(angles_list) if len(angles_list) > 1 else np.nan,
                    'E_n': len(angles_list),
                    'I_mean': np.nan, 'I_sem': np.nan, 'I_n': np.nan,
                    'mwu_p_two_sided': np.nan,
                })
        # Paired synergy test — summary row per threshold
        ksr = r.get('ko_session_rows', [])
        for e_key, threshold in [
            ('shesha_e_only',         'strict_n>=4'),
            ('shesha_e_only_relaxed', 'relaxed_n>=3'),
        ]:
            pairs = [(row['shesha_full'], row[e_key])
                     for row in ksr
                     if not np.isnan(row['shesha_full']) and not np.isnan(row[e_key])]
            diffs = [f - e for f, e in pairs]
            if len(pairs) >= 5:
                try:
                    w, p_w = wilcoxon([f for f, _ in pairs], [e for _, e in pairs],
                                      alternative='greater')
                except ValueError:
                    w, p_w = np.nan, np.nan
            else:
                w, p_w = np.nan, np.nan
            summary_rows.append({
                'species': label, 'metric': 'paired_full_vs_e_synergy',
                'threshold': threshold,
                'E_mean': np.mean([f for f, _ in pairs]) if pairs else np.nan,
                'E_sem': sp_sem([f for f, _ in pairs]) if len(pairs) > 1 else np.nan,
                'E_n': len(pairs),
                'I_mean': np.mean([e for _, e in pairs]) if pairs else np.nan,
                'I_sem': sp_sem([e for _, e in pairs]) if len(pairs) > 1 else np.nan,
                'I_n': len(pairs),
                'mwu_p_two_sided': np.nan,
                'mean_diff_full_minus_e': np.mean(diffs) if diffs else np.nan,
                'wilcoxon_W': w,
                'wilcoxon_p_one_sided': p_w,
            })
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, 'tier2_ei_summary.csv'), index=False)
    print("  Saved tier2_ei_summary.csv")

    # ----------------------------------------------------------------
    # 1b. tier2_threshold_comparison.csv
    #     One row per species × analysis × threshold — the four cross-checks
    #     that address the n=12 vs n=21 discrepancy in one place
    # ----------------------------------------------------------------
    tc_rows = []
    for species, label in [('titmouse', 'chickadee'), ('zebra_finch', 'finch')]:
        r = results[species]
        ksr = r.get('ko_session_rows', [])

        for analysis, threshold, vals_or_pairs, extra in [
            # SHESHA E-only
            ('shesha_e_only', 'strict_n>=4',
             [row['shesha_e_only'] for row in ksr if not np.isnan(row['shesha_e_only'])],
             {}),
            ('shesha_e_only', 'relaxed_n>=3',
             [row['shesha_e_only_relaxed'] for row in ksr if not np.isnan(row['shesha_e_only_relaxed'])],
             {}),
            # Subspace mean angle
            ('subspace_mean_angle_deg', 'relaxed_n>=3', r['principal_angles'], {}),
            ('subspace_mean_angle_deg', 'strict_n>=4',  r['principal_angles_strict'], {}),
        ]:
            n = len(vals_or_pairs)
            tc_rows.append({
                'species': label, 'analysis': analysis, 'threshold': threshold,
                'n_sessions': n,
                'mean': np.mean(vals_or_pairs) if n else np.nan,
                'sem':  sp_sem(vals_or_pairs)  if n > 1 else np.nan,
                'min':  np.min(vals_or_pairs)  if n else np.nan,
                'max':  np.max(vals_or_pairs)  if n else np.nan,
            })
        # Paired synergy Wilcoxon rows
        for e_key, threshold in [
            ('shesha_e_only',         'strict_n>=4'),
            ('shesha_e_only_relaxed', 'relaxed_n>=3'),
        ]:
            pairs = [(row['shesha_full'], row[e_key])
                     for row in ksr
                     if not np.isnan(row['shesha_full']) and not np.isnan(row[e_key])]
            diffs = [f - e for f, e in pairs]
            if len(pairs) >= 5:
                try:
                    w, p_w = wilcoxon([f for f, _ in pairs], [e for _, e in pairs],
                                      alternative='greater')
                except ValueError:
                    w, p_w = np.nan, np.nan
            else:
                w, p_w = np.nan, np.nan
            tc_rows.append({
                'species': label, 'analysis': 'paired_synergy_full_gt_e',
                'threshold': threshold,
                'n_sessions': len(pairs),
                'mean': np.mean(diffs) if diffs else np.nan,
                'sem':  sp_sem(diffs)  if len(diffs) > 1 else np.nan,
                'min':  np.min(diffs)  if diffs else np.nan,
                'max':  np.max(diffs)  if diffs else np.nan,
                'wilcoxon_W': w,
                'wilcoxon_p': p_w,
                'n_sessions_full_gt_e': sum(1 for d in diffs if d > 0),
            })
    pd.DataFrame(tc_rows).to_csv(os.path.join(OUTPUT_DIR, 'tier2_threshold_comparison.csv'), index=False)
    print("  Saved tier2_threshold_comparison.csv")

    # ----------------------------------------------------------------
    # 2. tier2_session_shesha.csv  (panels A and F)
    #    One row per session: SHESHA for full / E-only / I-only populations
    # ----------------------------------------------------------------
    shesha_rows = []
    for species, label in [('titmouse', 'chickadee'), ('zebra_finch', 'finch')]:
        for row in results[species]['shesha_session_rows']:
            shesha_rows.append({
                'species': label,
                'session': row['session'],
                'bird': row['bird'],
                'n_all': row['n_all'],
                'n_e': row['n_e'],
                'n_i': row['n_i'],
                'shesha_all': row['shesha_all'],
                'shesha_e_strict': row['shesha_e'],
                'shesha_i_strict': row['shesha_i'],
                'shesha_e_relaxed': row['shesha_e_relaxed'],
                'shesha_i_relaxed': row['shesha_i_relaxed'],
            })
    pd.DataFrame(shesha_rows).to_csv(os.path.join(OUTPUT_DIR, 'tier2_session_shesha.csv'), index=False)
    print("  Saved tier2_session_shesha.csv")

    # ----------------------------------------------------------------
    # 3. tier2_session_mantel.csv  (panel B)
    #    One row per session: Mantel r for full / E-only / I-only
    # ----------------------------------------------------------------
    mantel_rows = []
    for species, label in [('titmouse', 'chickadee'), ('zebra_finch', 'finch')]:
        for row in results[species]['mantel_session_rows']:
            mantel_rows.append({
                'species': label,
                'session': row['session'],
                'bird': row['bird'],
                'n_all': row['n_all'],
                'n_e': row['n_e'],
                'n_i': row['n_i'],
                'mantel_all': row['mantel_all'],
                'mantel_e': row['mantel_e'],
                'mantel_i': row['mantel_i'],
            })
    pd.DataFrame(mantel_rows).to_csv(os.path.join(OUTPUT_DIR, 'tier2_session_mantel.csv'), index=False)
    print("  Saved tier2_session_mantel.csv")

    # ----------------------------------------------------------------
    # 4. tier2_cell_info_xcorr.csv  (panels C and D)
    #    One row per neuron: spatial info, xcorr_map, cell_type
    # ----------------------------------------------------------------
    cell_rows = []
    for species, label in [('titmouse', 'chickadee'), ('zebra_finch', 'finch')]:
        for row in results[species]['cell_rows']:
            cell_rows.append({
                'species': label,
                'session': row['session'],
                'bird': row['bird'],
                'cell_type': row['cell_type'],
                'info': row['info'],
                'xcorr_map': row['xcorr_map'],
                'spatially_selective': row['spatially_selective'],
            })
    pd.DataFrame(cell_rows).to_csv(os.path.join(OUTPUT_DIR, 'tier2_cell_info_xcorr.csv'), index=False)
    print("  Saved tier2_cell_info_xcorr.csv")

    # ----------------------------------------------------------------
    # 5. tier2_session_ei_corr.csv  (panel E)
    #    One row per session: E-I spatial correlation
    # ----------------------------------------------------------------
    ei_rows = []
    for species, label in [('titmouse', 'chickadee'), ('zebra_finch', 'finch')]:
        for row in results[species]['ei_corr_rows']:
            ei_rows.append({
                'species': label,
                'session': row['session'],
                'bird': row['bird'],
                'n_e': row['n_e'],
                'n_i': row['n_i'],
                'ei_spatial_r': row['ei_spatial_r'],
            })
    pd.DataFrame(ei_rows).to_csv(os.path.join(OUTPUT_DIR, 'tier2_session_ei_corr.csv'), index=False)
    print("  Saved tier2_session_ei_corr.csv")

    # ----------------------------------------------------------------
    # 6. tier2_session_subspace_angles.csv  (E-I subspace analysis)
    #    One row per session: principal angles between E and I subspaces
    # ----------------------------------------------------------------
    angle_csv_rows = []
    for species, label in [('titmouse', 'chickadee'), ('zebra_finch', 'finch')]:
        for row in results[species].get('angle_rows', []):
            angle_csv_rows.append({
                'species': label,
                'session': row['session'],
                'bird': row['bird'],
                'n_e': row['n_e'],
                'n_i': row['n_i'],
                # relaxed threshold (n>=3): primary, n=21
                'angle_1_deg': row['angle_1'],
                'angle_2_deg': row['angle_2'],
                'angle_3_deg': row['angle_3'],
                'mean_angle_deg': row['mean_angle'],
                # strict threshold (n>=4): sample-matched to SHESHA, n=12
                'angle_1_deg_strict': row['angle_1_strict'],
                'angle_2_deg_strict': row['angle_2_strict'],
                'angle_3_deg_strict': row['angle_3_strict'],
                'mean_angle_deg_strict': row['mean_angle_strict'],
            })
    if angle_csv_rows:
        pd.DataFrame(angle_csv_rows).to_csv(
            os.path.join(OUTPUT_DIR, 'tier2_session_subspace_angles.csv'),
            index=False)
        print("  Saved tier2_session_subspace_angles.csv")

    # ----------------------------------------------------------------
    # 7a. tier2_knockout_summary.csv  (panel F bar chart)
    #     One row per species: mean + SEM for full / E-only / I-only
    # ----------------------------------------------------------------
    def _paired_wilcoxon(ko_session_rows, e_key):
        pairs = [(row['shesha_full'], row[e_key])
                 for row in ko_session_rows
                 if not np.isnan(row['shesha_full']) and not np.isnan(row[e_key])]
        diffs = [f - e for f, e in pairs]
        if len(pairs) >= 5:
            try:
                w, p = wilcoxon([f for f, _ in pairs], [e for _, e in pairs],
                                alternative='greater')
            except ValueError:
                w, p = np.nan, np.nan
        else:
            w, p = np.nan, np.nan
        return len(pairs), np.mean(diffs) if diffs else np.nan, w, p

    ko_rows = []
    for species, label in [('titmouse', 'chickadee'), ('zebra_finch', 'finch')]:
        r = results[species]
        ksr = r.get('ko_session_rows', [])
        n_s, md_s, w_s, p_s = _paired_wilcoxon(ksr, 'shesha_e_only')
        n_r, md_r, w_r, p_r = _paired_wilcoxon(ksr, 'shesha_e_only_relaxed')

# paired full means -- computed from the same subset used for the Wilcoxon
        paired_full_strict = [row['shesha_full'] for row in ksr
                              if not np.isnan(row['shesha_full'])
                              and not np.isnan(row['shesha_e_only'])]
        paired_full_relaxed = [row['shesha_full'] for row in ksr
                               if not np.isnan(row['shesha_full'])
                               and not np.isnan(row['shesha_e_only_relaxed'])]

        ko_rows.append({
            'species': label,
            'full_mean': np.mean(paired_full_strict) if paired_full_strict else np.nan,
            'full_sem': sp_sem(paired_full_strict) if len(paired_full_strict) > 1 else np.nan,
            'full_n': len(paired_full_strict),
            # strict
            'e_only_strict_mean': np.mean(r['shesha_e']) if r['shesha_e'] else np.nan,
            'e_only_strict_sem': sp_sem(r['shesha_e']) if len(r['shesha_e']) > 1 else np.nan,
            'e_only_strict_n': len(r['shesha_e']),
            'i_only_strict_mean': np.mean(r['shesha_i']) if r['shesha_i'] else np.nan,
            'i_only_strict_sem': sp_sem(r['shesha_i']) if len(r['shesha_i']) > 1 else np.nan,
            'i_only_strict_n': len(r['shesha_i']),
            'paired_n_strict': n_s,
            'paired_mean_diff_strict': md_s,
            'wilcoxon_W_strict': w_s,
            'wilcoxon_p_strict': p_s,
            # relaxed
            'full_mean_relaxed': np.mean(paired_full_relaxed) if paired_full_relaxed else np.nan,
            'full_sem_relaxed': sp_sem(paired_full_relaxed) if len(paired_full_relaxed) > 1 else np.nan,
            'full_n_relaxed': len(paired_full_relaxed),
            'e_only_relaxed_mean': np.mean(r['shesha_e_relaxed']) if r['shesha_e_relaxed'] else np.nan,
            'e_only_relaxed_sem': sp_sem(r['shesha_e_relaxed']) if len(r['shesha_e_relaxed']) > 1 else np.nan,
            'e_only_relaxed_n': len(r['shesha_e_relaxed']),
            'i_only_relaxed_mean': np.mean(r['shesha_i_relaxed']) if r['shesha_i_relaxed'] else np.nan,
            'i_only_relaxed_sem': sp_sem(r['shesha_i_relaxed']) if len(r['shesha_i_relaxed']) > 1 else np.nan,
            'i_only_relaxed_n': len(r['shesha_i_relaxed']),
            'paired_n_relaxed': n_r,
            'paired_mean_diff_relaxed': md_r,
            'wilcoxon_W_relaxed': w_r,
            'wilcoxon_p_relaxed': p_r,
        })
    pd.DataFrame(ko_rows).to_csv(os.path.join(OUTPUT_DIR, 'tier2_knockout_summary.csv'), index=False)
    print("  Saved tier2_knockout_summary.csv")

    # ----------------------------------------------------------------
    # 7b. tier2_knockout_paired.csv
    #     One row per session: full / E-only / I-only SHESHA + differences
    #     This is the primary evidence for the synergy (Full > E-only) claim
    # ----------------------------------------------------------------
    ko_paired_rows = []
    for species, label in [('titmouse', 'chickadee'), ('zebra_finch', 'finch')]:
        for row in results[species].get('ko_session_rows', []):
            ko_paired_rows.append({
                'species': label,
                'session': row['session'],
                'bird': row['bird'],
                'n_all': row['n_all'],
                'n_e': row['n_e'],
                'n_i': row['n_i'],
                'shesha_full': row['shesha_full'],
                # strict (n>=4)
                'shesha_e_only_strict': row['shesha_e_only'],
                'shesha_i_only_strict': row['shesha_i_only'],
                'diff_full_minus_e_strict': row['diff_full_minus_e'],
                'diff_full_minus_i_strict': row['diff_full_minus_i'],
                'full_gt_e_strict': (row['diff_full_minus_e'] > 0)
                                    if not np.isnan(row['diff_full_minus_e']) else np.nan,
                # relaxed (n>=3)
                'shesha_e_only_relaxed': row['shesha_e_only_relaxed'],
                'shesha_i_only_relaxed': row['shesha_i_only_relaxed'],
                'diff_full_minus_e_relaxed': row['diff_full_minus_e_relaxed'],
                'diff_full_minus_i_relaxed': row['diff_full_minus_i_relaxed'],
                'full_gt_e_relaxed': (row['diff_full_minus_e_relaxed'] > 0)
                                     if not np.isnan(row['diff_full_minus_e_relaxed']) else np.nan,
            })
    pd.DataFrame(ko_paired_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'tier2_knockout_paired.csv'), index=False)
    print("  Saved tier2_knockout_paired.csv")


# ======================================================================
# PLOTTING
# ======================================================================
def plot_results(results):
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35,
                           left=0.06, right=0.96, top=0.91, bottom=0.07)

    # -- A: SHESHA by cell type (both species) --
    ax_a = fig.add_subplot(gs[0, 0])
    data_plot = []
    labels_plot = []
    colors_plot = []

    for species, label in [('titmouse', 'Chick'), ('zebra_finch', 'Finch')]:
        r = results[species]
        ce = C_TE if species == 'titmouse' else C_ZE
        ci = C_TI if species == 'titmouse' else C_ZI

        if r['shesha_e']:
            data_plot.append(r['shesha_e'])
            labels_plot.append(f'{label} E')
            colors_plot.append(ce)
        if r['shesha_i']:
            data_plot.append(r['shesha_i'])
            labels_plot.append(f'{label} I')
            colors_plot.append(ci)

    if data_plot:
        bp = ax_a.boxplot(data_plot, positions=range(len(data_plot)),
                           widths=0.5, patch_artist=True, showfliers=True,
                           flierprops=dict(markersize=3))
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors_plot[i])
            box.set_alpha(0.7)
        for line in bp['medians']:
            line.set_color('black')
            line.set_linewidth(1.5)
        ax_a.set_xticks(range(len(labels_plot)))
        ax_a.set_xticklabels(labels_plot, fontsize=9)

    ax_a.set_title('A. SHESHA by cell type', fontsize=11,
                     fontweight='bold', loc='left')
    ax_a.set_ylabel('SHESHA (Spearman r)', fontsize=10)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # -- B: Mantel r by cell type --
    ax_b = fig.add_subplot(gs[0, 1])
    data_m = []
    labels_m = []
    colors_m = []

    for species, label in [('titmouse', 'Chick'), ('zebra_finch', 'Finch')]:
        r = results[species]
        ce = C_TE if species == 'titmouse' else C_ZE
        ci = C_TI if species == 'titmouse' else C_ZI

        if r['mantel_e']:
            data_m.append(r['mantel_e'])
            labels_m.append(f'{label} E')
            colors_m.append(ce)
        if r['mantel_i']:
            data_m.append(r['mantel_i'])
            labels_m.append(f'{label} I')
            colors_m.append(ci)

    if data_m:
        bp = ax_b.boxplot(data_m, positions=range(len(data_m)),
                           widths=0.5, patch_artist=True, showfliers=True,
                           flierprops=dict(markersize=3))
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors_m[i])
            box.set_alpha(0.7)
        for line in bp['medians']:
            line.set_color('black')
            line.set_linewidth(1.5)
        ax_b.set_xticks(range(len(labels_m)))
        ax_b.set_xticklabels(labels_m, fontsize=9)

    ax_b.set_title('B. Mantel r by cell type', fontsize=11,
                     fontweight='bold', loc='left')
    ax_b.set_ylabel('Mantel r (RDM ~ physical distance)', fontsize=10)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # -- C: Spatial information by type (both species) --
    ax_c = fig.add_subplot(gs[0, 2])
    for species, label in [('titmouse', 'Chickadee'), ('zebra_finch', 'Finch')]:
        r = results[species]
        ce = C_TE if species == 'titmouse' else C_ZE
        ci = C_TI if species == 'titmouse' else C_ZI

        bins_info = np.linspace(0, 1.5, 50)
        if len(r['info_e']) > 0:
            ax_c.hist(r['info_e'], bins=bins_info, density=True, alpha=0.4,
                       color=ce, edgecolor='none',
                       label=f'{label} E (n={len(r["info_e"])})')
        if len(r['info_i']) > 0:
            ax_c.hist(r['info_i'], bins=bins_info, density=True, alpha=0.4,
                       color=ci, edgecolor='none',
                       label=f'{label} I (n={len(r["info_i"])})')

    ax_c.set_title('C. Spatial information by type', fontsize=11,
                     fontweight='bold', loc='left')
    ax_c.set_xlabel('Bits per spike', fontsize=10)
    ax_c.set_ylabel('Density', fontsize=10)
    ax_c.legend(fontsize=7, frameon=False)
    ax_c.set_xlim(0, 1.5)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # -- D: Within-session stability by type --
    ax_d = fig.add_subplot(gs[1, 0])
    data_x = []
    labels_x = []
    colors_x = []

    for species, label in [('titmouse', 'Chick'), ('zebra_finch', 'Finch')]:
        r = results[species]
        ce = C_TE if species == 'titmouse' else C_ZE
        ci = C_TI if species == 'titmouse' else C_ZI

        if len(r['xcorr_e']) > 0:
            data_x.append(r['xcorr_e'])
            labels_x.append(f'{label} E')
            colors_x.append(ce)
        if len(r['xcorr_i']) > 0:
            data_x.append(r['xcorr_i'])
            labels_x.append(f'{label} I')
            colors_x.append(ci)

    if data_x:
        bp = ax_d.boxplot(data_x, positions=range(len(data_x)),
                           widths=0.5, patch_artist=True, showfliers=False,
                           flierprops=dict(markersize=2))
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors_x[i])
            box.set_alpha(0.7)
        for line in bp['medians']:
            line.set_color('black')
            line.set_linewidth(1.5)
        ax_d.set_xticks(range(len(labels_x)))
        ax_d.set_xticklabels(labels_x, fontsize=9)

    ax_d.set_title('D. Within-session stability by type', fontsize=11,
                     fontweight='bold', loc='left')
    ax_d.set_ylabel('xcorr_map (1st vs 2nd half)', fontsize=10)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    # -- E: E-I spatial correlation --
    ax_e = fig.add_subplot(gs[1, 1])
    for species, label in [('titmouse', 'Chickadee'), ('zebra_finch', 'Finch')]:
        r = results[species]
        color = C_T if species == 'titmouse' else C_Z
        if r['ei_corrs']:
            jitter = np.random.RandomState(320).normal(0, 0.08, len(r['ei_corrs']))
            x_pos = 0 if species == 'titmouse' else 1
            ax_e.scatter(x_pos + jitter, r['ei_corrs'], c=color,
                          alpha=0.5, s=40, edgecolors='white', linewidths=0.3)
            ax_e.plot([x_pos - 0.2, x_pos + 0.2],
                       [np.mean(r['ei_corrs'])] * 2,
                       color='black', lw=2)

    ax_e.axhline(0, color='gray', ls=':', lw=0.8)
    ax_e.set_xticks([0, 1])
    ax_e.set_xticklabels(['Chickadee', 'Finch'])
    ax_e.set_title('E. E-I spatial correlation\n(per session)',
                     fontsize=11, fontweight='bold', loc='left')
    ax_e.set_ylabel('Spearman r (E vs I mean maps)', fontsize=10)
    ax_e.spines['top'].set_visible(False)
    ax_e.spines['right'].set_visible(False)

    # -- F: Knockout test --
    ax_f = fig.add_subplot(gs[1, 2])
    ko_data = []
    for species, label in [('titmouse', 'Chickadee'), ('zebra_finch', 'Finch')]:
        r = results[species]
        ko_data.append({
            'species': label,
            'full': np.mean(r['shesha_all']) if r['shesha_all'] else 0,
            'e_only': np.mean(r['shesha_e']) if r['shesha_e'] else 0,
            'i_only': np.mean(r['shesha_i']) if r['shesha_i'] else 0,
            'full_sem': sp_sem(r['shesha_all']) if len(r['shesha_all']) > 1 else 0,
            'e_sem': sp_sem(r['shesha_e']) if len(r['shesha_e']) > 1 else 0,
            'i_sem': sp_sem(r['shesha_i']) if len(r['shesha_i']) > 1 else 0,
        })

    x = np.arange(len(ko_data))
    w = 0.22
    for i, kd in enumerate(ko_data):
        color_sp = C_T if 'Chick' in kd['species'] else C_Z
        ax_f.bar(i - w, kd['full'], w, yerr=kd['full_sem'],
                  color=color_sp, alpha=0.9, capsize=3, label='Full' if i == 0 else '')
        ax_f.bar(i, kd['e_only'], w, yerr=kd['e_sem'],
                  color=C_E, alpha=0.7, capsize=3, label='E-only' if i == 0 else '')
        ax_f.bar(i + w, kd['i_only'], w, yerr=kd['i_sem'],
                  color=C_I, alpha=0.7, capsize=3, label='I-only' if i == 0 else '')

    ax_f.set_xticks(x)
    ax_f.set_xticklabels([kd['species'] for kd in ko_data])
    ax_f.legend(fontsize=8, frameon=False)
    ax_f.set_title('F. Knockout test: SHESHA by population',
                     fontsize=11, fontweight='bold', loc='left')
    ax_f.set_ylabel('SHESHA', fontsize=10)
    ax_f.spines['top'].set_visible(False)
    ax_f.spines['right'].set_visible(False)

    fig.suptitle(
        'Tier 2: Excitatory vs Inhibitory Contributions to Geometric Stability',
        fontsize=14, fontweight='bold', y=0.97)

    outpath = os.path.join(OUTPUT_DIR, 'tier2_ei_stability.png')
    fig.savefig(outpath, dpi=250, facecolor='white')
    print(f"\nFigure saved: {outpath}")
    plt.show()


def main():
    if not Path(DATA_PATH).exists():
        print(f"Error: {DATA_PATH} not found")
        return
    df = load_data(DATA_PATH)
    results = run_analyses(df)
    plot_results(results)


if __name__ == '__main__':
    main()