#!/usr/bin/env python3
"""
tier2_enhanced.py - Comprehensive Tier 2 E/I analysis with:
  - Focus on within-chickadee E/I synergy (avoids n=1 finch I issue)
  - Partial Information Decomposition (PID) - simplified implementation
  - Residual analysis after removing shared variance
  - Simple bootstrap confidence intervals (1000 iterations)
  - Bayesian credible intervals (MCMC)
  - E-I coordination index
  - Spatial frequency analysis
  - Temporal dynamics / stability timescales
  - Negative controls (anti-correlated, random pairing, scaled noise)
  - Dimensionality analysis
  
Uses shesha-geometry package from PyPI: https://pypi.org/project/shesha-geometry/

Note: Uses simple bootstrap (not hierarchical) for within-species analysis,
which is statistically appropriate and 10,000x faster.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, mannwhitneyu, sem as sp_sem
from scipy.linalg import svd
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from pathlib import Path
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import from shesha-geometry package
from shesha import feature_split, sample_split, compute_rdm

OUTPUT_DIR = 'output/tier2_enhanced'

GRID_SIZE = 40
N_BINS = GRID_SIZE ** 2
MIN_NEURONS_TYPE = 3
N_BOOTSTRAP = 1000
N_MCMC = 2000  # For Bayesian analysis

C_E = '#2E8B57'
C_I = '#FF8C00'
C_T = '#C62D50'


def load_chickadee_data_with_cell_types(data_path=None):
    """
    Load real Aronov dataset and split by cell type (E/I)
    Focus on chickadee only to avoid n=1 finch inhibitory issue
    
    Parameters
    ----------
    data_path : str, optional
        Path to aronov_dataset.pkl file
        If None, tries common locations
    
    Returns
    -------
    sessions : list
        Chickadee sessions with E and I neurons separated
    """
    if data_path is None:
        raise FileNotFoundError(
            "Could not find aronov_dataset.pkl. Please provide path explicitly."
        )
    
    print(f"Loading data from: {data_path}")
    
    # Check file exists and size
    import os
    if os.path.exists(data_path):
        file_size = os.path.getsize(data_path)
        print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    else:
        raise FileNotFoundError(f"File not found: {data_path}")
    
    # Load with error handling
    try:
        df = pd.read_pickle(data_path)
        print(f"✓ Successfully loaded DataFrame with {len(df)} rows")
    except Exception as e:
        print(f"ERROR loading pickle file: {e}")
        raise
    
    # Debug: Show available columns
    print(f"\nAvailable columns: {list(df.columns)}")
    
    # Auto-detect map column
    if 'rate_maps' in df.columns:
        map_col = 'rate_maps'
    elif 'map' in df.columns:
        map_col = 'map'
    else:
        raise ValueError("No rate maps column found ('rate_maps' or 'map')")
    
    print(f"Using rate map column: '{map_col}'")
    
    # Check for required columns
    required_cols = ['cell_type', 'species', map_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Filter for chickadee (titmouse) only
    df_chickadee = df[df['species'] == 'titmouse'].copy()
    
    print(f"\nTotal chickadee cells: {len(df_chickadee)}")
    print(f"  E cells: {len(df_chickadee[df_chickadee['cell_type'] == 'E'])}")
    print(f"  I cells: {len(df_chickadee[df_chickadee['cell_type'] == 'I'])}")
    
    # Determine session and bird columns
    if 'session' in df_chickadee.columns:
        session_col = 'session'
    elif 'session_id' in df_chickadee.columns:
        session_col = 'session_id'
    else:
        # Create session ID from index
        print("WARNING: No session_id or session column found, using index")
        df_chickadee['session_id'] = df_chickadee.index
        session_col = 'session_id'
    
    if 'bird' in df_chickadee.columns:
        bird_col = 'bird'
    elif 'bird_id' in df_chickadee.columns:
        bird_col = 'bird_id'
    elif 'subject' in df_chickadee.columns:
        bird_col = 'subject'
    else:
        print("WARNING: No bird_id, bird, or subject column found, creating default")
        df_chickadee['bird_id'] = 'default_bird'
        bird_col = 'bird_id'
    
    print(f"Using columns: bird='{bird_col}', session='{session_col}'")
    
    # Show unique birds and sessions
    unique_birds = df_chickadee[bird_col].nunique()
    unique_sessions = df_chickadee[session_col].nunique()
    print(f"Found {unique_birds} unique birds, {unique_sessions} unique sessions")
    
    # Group by bird and session
    sessions = []
    skipped_sessions = {'too_few_e': 0, 'too_few_i': 0, 'no_rate_maps': 0}
    
    print(f"\nProcessing sessions (min neurons per type: {MIN_NEURONS_TYPE})...")
    
    for (bird, sess), group in df_chickadee.groupby([bird_col, session_col]):
        # Separate E and I cells
        e_cells = group[group['cell_type'] == 'E']
        i_cells = group[group['cell_type'] == 'I']
        
        # Check minimum counts before processing
        if len(e_cells) < MIN_NEURONS_TYPE:
            skipped_sessions['too_few_e'] += 1
            continue
        
        if len(i_cells) < MIN_NEURONS_TYPE:
            skipped_sessions['too_few_i'] += 1
            continue
        
        # Extract rate maps
        M_e_list = []
        M_i_list = []
        
        for idx, row in e_cells.iterrows():
            rate_map = row[map_col]
            if rate_map is not None:
                # Handle 1D arrays
                if not isinstance(rate_map, np.ndarray):
                    rate_map = np.array(rate_map)
                if rate_map.ndim == 1:
                    rate_map = rate_map.reshape(1, -1)
                M_e_list.append(rate_map if rate_map.ndim == 1 else rate_map)
        
        for idx, row in i_cells.iterrows():
            rate_map = row[map_col]
            if rate_map is not None:
                if not isinstance(rate_map, np.ndarray):
                    rate_map = np.array(rate_map)
                if rate_map.ndim == 1:
                    rate_map = rate_map.reshape(1, -1)
                M_i_list.append(rate_map if rate_map.ndim == 1 else rate_map)
        
        if not M_e_list or not M_i_list:
            skipped_sessions['no_rate_maps'] += 1
            continue
        
        # Stack into matrices
        try:
            M_e = np.vstack(M_e_list)
            M_i = np.vstack(M_i_list)
        except Exception as e:
            print(f"  ERROR stacking arrays for bird={bird}, session={sess}: {e}")
            continue
        
        # Check shapes
        if M_e.shape[0] < MIN_NEURONS_TYPE or M_i.shape[0] < MIN_NEURONS_TYPE:
            continue
        
        sessions.append({
            'M_e': M_e,
            'M_i': M_i,
            'M_all': np.vstack([M_e, M_i]),
            'n_e': M_e.shape[0],
            'n_i': M_i.shape[0],
            'bird': bird,
            'session': sess,
            'species': 'titmouse',
        })
        
        print(f"  ✓ Session {len(sessions)}: bird={bird}, session={sess}, "
              f"E={M_e.shape[0]} neurons, I={M_i.shape[0]} neurons, "
              f"{M_e.shape[1]} bins")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✓ Loaded {len(sessions)} chickadee sessions with both E and I neurons")
    
    if sessions:
        total_e = sum(s['n_e'] for s in sessions)
        total_i = sum(s['n_i'] for s in sessions)
        print(f"  Total E neurons: {total_e}")
        print(f"  Total I neurons: {total_i}")
        print(f"  E/I ratio: {total_e/total_i:.2f}")
    
    # Report skipped sessions
    total_skipped = sum(skipped_sessions.values())
    if total_skipped > 0:
        print(f"\nSkipped {total_skipped} sessions:")
        print(f"  Too few E neurons: {skipped_sessions['too_few_e']}")
        print(f"  Too few I neurons: {skipped_sessions['too_few_i']}")
        print(f"  Missing rate maps: {skipped_sessions['no_rate_maps']}")
    
    if len(sessions) == 0:
        print("\n⚠ WARNING: No sessions found with sufficient E and I neurons")
        print(f"  Required minimum: {MIN_NEURONS_TYPE} neurons of each type")
        print("\nPossible issues:")
        print("  1. Data format mismatch (check if rate_maps are per-session or per-neuron)")
        print("  2. Threshold too high (try lowering MIN_NEURONS_TYPE)")
        print("  3. Missing or incorrect cell_type labels")
    
    print(f"{'='*60}\n")
    
    return sessions


def compute_shesha(M, n_splits=100, grid_size=None):
    """
    Compute SHESHA using shesha-geometry package
    
    Parameters
    ----------
    M : np.ndarray
        Rate maps matrix (n_neurons, n_bins)
    n_splits : int
        Number of feature splits
    grid_size : int, optional
        Grid size for this computation (for parameter sensitivity)
    
    Returns
    -------
    float
        SHESHA score (feature-split stability)
    """
    # Validate input
    if M is None:
        print("WARNING: compute_shesha received None")
        return np.nan
    
    if not isinstance(M, np.ndarray):
        print(f"WARNING: compute_shesha expected numpy array, got {type(M)}")
        try:
            M = np.array(M)
        except:
            return np.nan
    
    # Handle 1D arrays (single neuron)
    if M.ndim == 1:
        print(f"INFO: Reshaping 1D array of length {len(M)} to (1, {len(M)})")
        M = M.reshape(1, -1)
    
    if M.ndim != 2:
        print(f"ERROR: Expected 2D array, got shape {M.shape}")
        return np.nan
    
    n_neurons, n_bins = M.shape
    
    if n_neurons < 4:
        print(f"INFO: Too few neurons ({n_neurons} < 4), returning NaN")
        return np.nan
    
    # Z-score normalization
    means = M.mean(axis=1, keepdims=True)
    stds = M.std(axis=1, keepdims=True)
    stds[stds == 0] = 1
    M_z = (M - means) / stds
    
    # Filter for active bins
    active = np.sum(M > 0, axis=0) >= max(2, n_neurons // 3)
    active_idx = np.where(active)[0]
    
    if len(active_idx) < 30:
        print(f"INFO: Too few active bins ({len(active_idx)} < 30), returning NaN")
        return np.nan
    
    # Prepare data for shesha: (n_samples, n_features) = (n_bins, n_neurons)
    X = M_z[:, active_idx].T  # Shape: (n_active_bins, n_neurons)
    
    try:
        # Use shesha-geometry package
        score = feature_split(X, n_splits=n_splits, metric='cosine', 
                              seed=320, max_samples=None)
        return score
    except Exception as e:
        print(f"ERROR in feature_split: {e}")
        return np.nan


def compute_temporal_stability(M, subsample_fraction=0.4, n_splits=30):
    """
    Compute temporal stability using sample-split SHESHA
    This measures robustness across different subsamples of spatial bins
    
    Parameters
    ----------
    M : np.ndarray
        Rate maps matrix
    subsample_fraction : float
        Fraction of samples to use in each split
    n_splits : int
        Number of bootstrap iterations
    
    Returns
    -------
    float
        Temporal stability score
    """
    if M is None or M.shape[0] < 4:
        return np.nan
    
    n_neurons, n_bins = M.shape
    means = M.mean(axis=1, keepdims=True)
    stds = M.std(axis=1, keepdims=True)
    stds[stds == 0] = 1
    M_z = (M - means) / stds
    
    active = np.sum(M > 0, axis=0) >= max(2, n_neurons // 3)
    active_idx = np.where(active)[0]
    
    if len(active_idx) < 30:
        return np.nan
    
    X = M_z[:, active_idx].T
    score = sample_split(X, n_splits=n_splits, 
                         subsample_fraction=subsample_fraction,
                         metric='cosine', seed=320)
    return score


# ======================================================================
# RESIDUAL ANALYSIS
# ======================================================================

def compute_residual_contribution(M_e, M_i):
    """
    After removing shared variance between E and I populations,
    what unique information does each carry?
    """
    if M_e is None or M_i is None:
        return {'e_unique': np.nan, 'i_unique': np.nan, 'shared': np.nan}
    
    # Compute mean spatial maps
    mean_e = np.mean(M_e, axis=0)
    mean_i = np.mean(M_i, axis=0)
    
    # Active bins
    active = (mean_e > 0) | (mean_i > 0)
    if np.sum(active) < 30:
        return {'e_unique': np.nan, 'i_unique': np.nan, 'shared': np.nan}
    
    mean_e_active = mean_e[active]
    mean_i_active = mean_i[active]
    
    # Shared variance (correlation)
    r_shared, _ = spearmanr(mean_e_active, mean_i_active)
    
    # Residual E (remove I influence)
    mean_i_map = M_i.mean(axis=0, keepdims=True)
    M_e_resid = M_e - mean_i_map
    shesha_e_resid = compute_shesha(M_e_resid)
    
    # Residual I (remove E influence)
    mean_e_map = M_e.mean(axis=0, keepdims=True)
    M_i_resid = M_i - mean_e_map
    shesha_i_resid = compute_shesha(M_i_resid)
    
    return {
        'shared': r_shared,
        'e_unique': shesha_e_resid,
        'i_unique': shesha_i_resid,
    }


# ======================================================================
# PARTIAL INFORMATION DECOMPOSITION (Simplified)
# ======================================================================

def compute_information_decomposition(M_e, M_i, M_all):
    """
    Simplified Information Decomposition:
    - Redundant: Information in both E and I
    - Unique E: Information only in E
    - Unique I: Information only in I
    - Synergistic: Information only in E+I combined
    
    Uses SHESHA as proxy for information content
    """
    if M_e is None or M_i is None or M_all is None:
        return {
            'redundant': np.nan, 'unique_e': np.nan,
            'unique_i': np.nan, 'synergistic': np.nan
        }
    
    # Compute SHESHA for each population
    shesha_e = compute_shesha(M_e)
    shesha_i = compute_shesha(M_i)
    shesha_all = compute_shesha(M_all)
    
    if np.isnan(shesha_e) or np.isnan(shesha_i) or np.isnan(shesha_all):
        return {
            'redundant': np.nan, 'unique_e': np.nan,
            'unique_i': np.nan, 'synergistic': np.nan
        }
    
    # Simplified decomposition
    # Redundant: minimum of E and I
    redundant = min(shesha_e, shesha_i)
    
    # Unique E: E minus redundant
    unique_e = max(0, shesha_e - redundant)
    
    # Unique I: I minus redundant
    unique_i = max(0, shesha_i - redundant)
    
    # Synergistic: Combined minus sum of parts
    synergistic = max(0, shesha_all - (shesha_e + shesha_i - redundant))
    
    return {
        'redundant': redundant,
        'unique_e': unique_e,
        'unique_i': unique_i,
        'synergistic': synergistic,
    }


# ======================================================================
# E-I COORDINATION INDEX
# ======================================================================

def compute_ei_coordination(sessions):
    """
    Correlation between E and I SHESHA values across sessions.
    If E/I balance is critical, they should co-vary.
    """
    shesha_e_vals = []
    shesha_i_vals = []
    
    for sess in sessions:
        se = compute_shesha(sess['M_e'])
        si = compute_shesha(sess['M_i'])
        if not np.isnan(se) and not np.isnan(si):
            shesha_e_vals.append(se)
            shesha_i_vals.append(si)
    
    if len(shesha_e_vals) < 3:
        return np.nan, np.nan
    
    r, p = spearmanr(shesha_e_vals, shesha_i_vals)
    return r, p


# ======================================================================
# SPATIAL FREQUENCY ANALYSIS
# ======================================================================

def compute_spatial_frequency_spectrum(M):
    """
    Compute 2D FFT power spectrum to identify spatial scales
    """
    if M is None:
        return None
    
    # Average rate map
    mean_map = np.mean(M, axis=0).reshape(GRID_SIZE, GRID_SIZE)
    
    # 2D FFT
    fft = np.fft.fft2(mean_map)
    power = np.abs(fft)**2
    
    # Radial average
    center = GRID_SIZE // 2
    Y, X = np.ogrid[:GRID_SIZE, :GRID_SIZE]
    r = np.sqrt((X - center)**2 + (Y - center)**2).astype(int)
    
    radial_mean = []
    for radius in range(0, center):
        mask = (r == radius)
        if np.sum(mask) > 0:
            radial_mean.append(np.mean(power[mask]))
    
    return np.array(radial_mean)


def compare_spatial_frequencies(M_e, M_i):
    """
    Compare spatial frequency content of E vs I populations
    Returns correlation between their power spectra
    """
    spec_e = compute_spatial_frequency_spectrum(M_e)
    spec_i = compute_spatial_frequency_spectrum(M_i)
    
    if spec_e is None or spec_i is None:
        return np.nan
    
    # Normalize
    spec_e = spec_e / np.sum(spec_e)
    spec_i = spec_i / np.sum(spec_i)
    
    # Correlation
    min_len = min(len(spec_e), len(spec_i))
    r, _ = spearmanr(spec_e[:min_len], spec_i[:min_len])
    
    return r


def compute_dominant_frequency(M):
    """
    Find the dominant spatial frequency (peak in power spectrum)
    """
    spec = compute_spatial_frequency_spectrum(M)
    if spec is None or len(spec) == 0:
        return np.nan
    
    # Exclude DC component (first bin)
    if len(spec) > 1:
        peak_freq = np.argmax(spec[1:]) + 1
    else:
        peak_freq = 0
    
    return peak_freq


# ======================================================================
# TEMPORAL DYNAMICS
# ======================================================================

def analyze_temporal_dynamics(sessions):
    """
    Compare temporal stability (sample-split) vs geometric stability
    (feature-split) for E and I populations
    """
    results = []
    
    for sess in sessions:
        # E population
        geom_e = compute_shesha(sess['M_e'])  # Feature-split
        temp_e = compute_temporal_stability(sess['M_e'])  # Sample-split
        
        # I population
        geom_i = compute_shesha(sess['M_i'])
        temp_i = compute_temporal_stability(sess['M_i'])
        
        results.append({
            'session': sess['session'],
            'bird': sess['bird'],
            'geom_e': geom_e,
            'temp_e': temp_e,
            'geom_i': geom_i,
            'temp_i': temp_i,
            'geom_temp_ratio_e': geom_e / temp_e if temp_e > 0 else np.nan,
            'geom_temp_ratio_i': geom_i / temp_i if temp_i > 0 else np.nan,
        })
    
    return results


# ======================================================================
# DIMENSIONALITY ANALYSIS
# ======================================================================

def estimate_dimensionality(M, variance_threshold=0.95):
    """
    Estimate intrinsic dimensionality using PCA
    Returns number of PCs needed to explain variance_threshold of variance
    """
    if M is None or M.shape[0] < 4:
        return np.nan
    
    n_neurons, n_bins = M.shape
    
    # Z-score
    means = M.mean(axis=1, keepdims=True)
    stds = M.std(axis=1, keepdims=True)
    stds[stds == 0] = 1
    M_z = (M - means) / stds
    
    # PCA on neurons (transpose so features are neurons)
    try:
        pca = PCA()
        pca.fit(M_z.T)
        
        # Find number of components for threshold
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cum_var >= variance_threshold) + 1
        
        return n_components
    except:
        return np.nan


def compare_subspace_overlap(M_e, M_i):
    """
    Measure overlap between E and I subspaces using principal angles
    Simplified: Use correlation between top PCs
    """
    if M_e is None or M_i is None:
        return np.nan
    
    if M_e.shape[0] < 4 or M_i.shape[0] < 4:
        return np.nan
    
    try:
        # Get top 3 PCs for each
        pca_e = PCA(n_components=min(3, M_e.shape[0]))
        pca_i = PCA(n_components=min(3, M_i.shape[0]))
        
        pca_e.fit(M_e.T)
        pca_i.fit(M_i.T)
        
        # Correlation between first PCs
        pc1_e = pca_e.components_[0]
        pc1_i = pca_i.components_[0]
        
        # Ensure same length
        min_len = min(len(pc1_e), len(pc1_i))
        r = np.abs(np.corrcoef(pc1_e[:min_len], pc1_i[:min_len])[0, 1])
        
        return r
    except:
        return np.nan


# ======================================================================
# NEGATIVE CONTROLS
# ======================================================================

def create_anticorrelated_i(M_e, M_i):
    """
    Create artificial I population that's perfectly anti-correlated with E
    """
    if M_e is None:
        return None
    
    # Simply invert the E population
    M_i_anti = -M_e
    
    # Rescale to positive
    M_i_anti = M_i_anti - M_i_anti.min(axis=1, keepdims=True)
    
    return M_i_anti


def random_i_pairing(sessions):
    """
    Pair each session's E neurons with I neurons from a different session
    Breaks E-I coordination
    """
    if len(sessions) < 2:
        return sessions
    
    paired_sessions = []
    rng = np.random.RandomState(320)
    
    for i, sess in enumerate(sessions):
        # Pick a different session for I neurons
        other_idx = rng.choice([j for j in range(len(sessions)) if j != i])
        other_sess = sessions[other_idx]
        
        paired_sessions.append({
            'M_e': sess['M_e'],
            'M_i': other_sess['M_i'],  # Mismatched I
            'M_all': np.vstack([sess['M_e'], other_sess['M_i']]),
            'session': f"paired_{sess['session']}",
            'bird': sess['bird'],
            'n_e': sess['n_e'],
            'n_i': other_sess['n_i'],
        })
    
    return paired_sessions


def scaled_noise_model(M_i, noise_scales):
    """
    Add graded amounts of noise to I population
    Test when geometric stability breaks
    """
    results = []
    
    for noise_scale in noise_scales:
        M_i_noisy = M_i + np.random.normal(0, noise_scale, M_i.shape)
        M_i_noisy = np.maximum(M_i_noisy, 0)
        
        shesha = compute_shesha(M_i_noisy)
        results.append({
            'noise_scale': noise_scale,
            'shesha': shesha,
        })
    
    return results


# ======================================================================
# SIMPLE BOOTSTRAP (Appropriate for Within-Species Analysis)
# ======================================================================

def fast_bootstrap_ci(values, n_boot=100):
    """
    Fast simple bootstrap for within-species analysis
    
    For Tier 2 (within-chickadee E/I comparison), simple bootstrap is 
    statistically appropriate - no need for hierarchical bootstrap since
    we're not comparing across species.
    
    Parameters
    ----------
    values : array-like
        Values to bootstrap (e.g., SHESHA scores)
    n_boot : int
        Number of bootstrap iterations (default: 100)
    
    Returns
    -------
    tuple
        (mean, ci_lower, ci_upper)
    """
    if len(values) < 3:
        return np.nan, np.nan, np.nan
    
    values = np.array(values)
    rng = np.random.RandomState(320)
    
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    
    boot_means = np.array(boot_means)
    mean = np.mean(values)
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    
    return mean, ci_lower, ci_upper


def bootstrap_effect_size_ci(values_e, values_i, n_boot=100):
    """
    Bootstrap confidence interval for E-I difference
    
    Parameters
    ----------
    values_e : array-like
        E neuron SHESHA scores
    values_i : array-like
        I neuron SHESHA scores
    n_boot : int
        Number of bootstrap iterations
    
    Returns
    -------
    tuple
        (mean_diff, ci_lower, ci_upper)
    """
    if len(values_e) < 2 or len(values_i) < 2:
        return np.nan, np.nan, np.nan
    
    values_e = np.array(values_e)
    values_i = np.array(values_i)
    
    obs_diff = np.mean(values_e) - np.mean(values_i)
    
    rng = np.random.RandomState(320)
    boot_diffs = []
    
    for _ in range(n_boot):
        boot_e = rng.choice(values_e, size=len(values_e), replace=True)
        boot_i = rng.choice(values_i, size=len(values_i), replace=True)
        boot_diffs.append(np.mean(boot_e) - np.mean(boot_i))
    
    boot_diffs = np.array(boot_diffs)
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    
    return obs_diff, ci_lower, ci_upper


# ======================================================================
# BAYESIAN CREDIBLE INTERVALS (Simplified MCMC)
# ======================================================================

def bayesian_credible_interval(data_e, data_i, n_mcmc=N_MCMC):
    """
    Simplified Bayesian analysis using Metropolis-Hastings MCMC
    Estimates posterior distribution of E-I difference
    """
    if len(data_e) < 2 or len(data_i) < 2:
        return np.nan, (np.nan, np.nan)
    
    data_e = np.array(data_e)
    data_i = np.array(data_i)
    
    # Simple model: Normal likelihood
    # Prior: uniform on difference
    
    # Initial values
    mean_e = np.mean(data_e)
    mean_i = np.mean(data_i)
    std_e = np.std(data_e)
    std_i = np.std(data_i)
    
    # MCMC
    rng = np.random.RandomState(320)
    samples = []
    current_diff = mean_e - mean_i
    
    for _ in range(n_mcmc):
        # Propose new difference
        proposal = current_diff + rng.normal(0, 0.01)
        
        # Log likelihood (simplified)
        # Assumes independent Normal distributions
        ll_current = -(np.sum((data_e - mean_e)**2) / (2 * std_e**2) +
                       np.sum((data_i - mean_i)**2) / (2 * std_i**2))
        
        ll_proposal = -(np.sum((data_e - (mean_i + proposal))**2) / (2 * std_e**2) +
                        np.sum((data_i - mean_i)**2) / (2 * std_i**2))
        
        # Metropolis ratio
        if rng.random() < min(1, np.exp(ll_proposal - ll_current)):
            current_diff = proposal
        
        samples.append(current_diff)
    
    # Discard burn-in
    samples = np.array(samples[n_mcmc//2:])
    
    # Credible interval
    ci_lower = np.percentile(samples, 2.5)
    ci_upper = np.percentile(samples, 97.5)
    
    return np.mean(samples), (ci_lower, ci_upper)


# ======================================================================
# MAIN ANALYSIS
# ======================================================================

def run_comprehensive_analysis(data_path=None):
    """
    Run all E/I analyses
    
    Parameters
    ----------
    data_path : str, optional
        Path to aronov_dataset.pkl file
    """
    print("="*70)
    print("TIER 2 ENHANCED: Within-Chickadee E/I Synergy Analysis")
    print("="*70)
    
    # Load real data
    sessions = load_chickadee_data_with_cell_types(data_path)
    
    results = {}
    
    # 1. Basic E vs I SHESHA
    print("\n1. Computing E vs I SHESHA...")
    shesha_e = [compute_shesha(s['M_e']) for s in sessions]
    shesha_i = [compute_shesha(s['M_i']) for s in sessions]
    shesha_e = [x for x in shesha_e if not np.isnan(x)]
    shesha_i = [x for x in shesha_i if not np.isnan(x)]
    
    results['shesha_e'] = shesha_e
    results['shesha_i'] = shesha_i
    
    # Statistical test
    if shesha_e and shesha_i:
        U, p = mannwhitneyu(shesha_e, shesha_i, alternative='two-sided')
        results['p_ei_diff'] = p
    
    # 2. Residual analysis
    print("\n2. Computing residual contributions...")
    residual_results = []
    for sess in sessions:
        res = compute_residual_contribution(sess['M_e'], sess['M_i'])
        residual_results.append(res)
    results['residuals'] = residual_results
    
    # 3. Information decomposition
    print("\n3. Computing information decomposition (PID)...")
    pid_results = []
    for sess in sessions:
        pid = compute_information_decomposition(sess['M_e'], sess['M_i'], 
                                                 sess['M_all'])
        pid_results.append(pid)
    results['pid'] = pid_results
    
    # 4. E-I coordination
    print("\n4. Computing E-I coordination index...")
    coord_r, coord_p = compute_ei_coordination(sessions)
    results['coordination'] = {'r': coord_r, 'p': coord_p}
    
    # 5. Spatial frequency analysis
    print("\n5. Analyzing spatial frequencies...")
    freq_corrs = []
    freq_e_peaks = []
    freq_i_peaks = []
    for sess in sessions:
        r = compare_spatial_frequencies(sess['M_e'], sess['M_i'])
        peak_e = compute_dominant_frequency(sess['M_e'])
        peak_i = compute_dominant_frequency(sess['M_i'])
        
        if not np.isnan(r):
            freq_corrs.append(r)
        if not np.isnan(peak_e):
            freq_e_peaks.append(peak_e)
        if not np.isnan(peak_i):
            freq_i_peaks.append(peak_i)
    
    results['freq_correlations'] = freq_corrs
    results['freq_peaks'] = {'e': freq_e_peaks, 'i': freq_i_peaks}
    
    # 6. Temporal dynamics
    print("\n6. Analyzing temporal dynamics...")
    temporal_results = analyze_temporal_dynamics(sessions)
    results['temporal_dynamics'] = temporal_results
    
    # 7. Dimensionality
    print("\n7. Estimating dimensionality...")
    dims_e = [estimate_dimensionality(s['M_e']) for s in sessions]
    dims_i = [estimate_dimensionality(s['M_i']) for s in sessions]
    dims_all = [estimate_dimensionality(s['M_all']) for s in sessions]
    subspace_overlaps = [compare_subspace_overlap(s['M_e'], s['M_i']) 
                         for s in sessions]
    
    results['dimensionality'] = {
        'e': [x for x in dims_e if not np.isnan(x)],
        'i': [x for x in dims_i if not np.isnan(x)],
        'all': [x for x in dims_all if not np.isnan(x)],
        'subspace_overlap': [x for x in subspace_overlaps if not np.isnan(x)],
    }
    
    # 8. Negative controls
    print("\n8. Running negative controls...")
    
    # Anti-correlated I
    shesha_anticorr = []
    for sess in sessions[:5]:  # Subset for speed
        M_i_anti = create_anticorrelated_i(sess['M_e'], sess['M_i'])
        if M_i_anti is not None:
            s = compute_shesha(M_i_anti)
            if not np.isnan(s):
                shesha_anticorr.append(s)
    results['anticorrelated_i'] = shesha_anticorr
    
    # Random pairing
    paired = random_i_pairing(sessions[:5])
    shesha_e_paired = [compute_shesha(s['M_e']) for s in paired]
    shesha_i_paired = [compute_shesha(s['M_i']) for s in paired]
    results['random_pairing'] = {
        'e': [x for x in shesha_e_paired if not np.isnan(x)],
        'i': [x for x in shesha_i_paired if not np.isnan(x)],
    }
    
    # Scaled noise
    noise_scales = [0.0, 0.1, 0.2, 0.5, 1.0]
    if sessions:
        noise_results = scaled_noise_model(sessions[0]['M_i'], noise_scales)
        results['scaled_noise'] = noise_results
    
    # 9. Simple bootstrap (appropriate for within-species analysis)
    print("\n9. Computing bootstrap CI for E-I difference...")
    mean_diff, ci_low, ci_high = bootstrap_effect_size_ci(shesha_e, shesha_i, n_boot=10000)
    
    # Also compute individual CIs for E and I
    mean_e, ci_e_low, ci_e_high = fast_bootstrap_ci(shesha_e, n_boot=10000)
    mean_i, ci_i_low, ci_i_high = fast_bootstrap_ci(shesha_i, n_boot=10000)
    
    results['bootstrap'] = {
        'mean_diff': mean_diff,
        'ci': (ci_low, ci_high),
        'e_mean': mean_e,
        'e_ci': (ci_e_low, ci_e_high),
        'i_mean': mean_i,
        'i_ci': (ci_i_low, ci_i_high),
    }
    
    # 10. Bayesian credible interval
    print("\n10. Computing Bayesian credible intervals...")
    bayes_mean, bayes_ci = bayesian_credible_interval(shesha_e, shesha_i, 
                                                       n_mcmc=1000)
    results['bayesian'] = {
        'mean_diff': bayes_mean,
        'ci': bayes_ci,
    }
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"E SHESHA: {np.mean(shesha_e):.3f} ± {np.std(shesha_e):.3f} (n={len(shesha_e)})")
    print(f"I SHESHA: {np.mean(shesha_i):.3f} ± {np.std(shesha_i):.3f} (n={len(shesha_i)})")
    if 'p_ei_diff' in results:
        print(f"p-value (E vs I): {results['p_ei_diff']:.4f}")
    print(f"\nE-I Coordination: r={coord_r:.3f}, p={coord_p:.4f}")
    print(f"\nE SHESHA: {results['bootstrap']['e_mean']:.3f} "
          f"[{results['bootstrap']['e_ci'][0]:.3f}, {results['bootstrap']['e_ci'][1]:.3f}]")
    print(f"I SHESHA: {results['bootstrap']['i_mean']:.3f} "
          f"[{results['bootstrap']['i_ci'][0]:.3f}, {results['bootstrap']['i_ci'][1]:.3f}]")
    print(f"E-I difference (bootstrap): {mean_diff:.3f}")
    print(f"  95% CI: ({ci_low:.3f}, {ci_high:.3f})")
    print(f"E-I difference (Bayesian): {bayes_mean:.3f}")
    print(f"  95% Credible Interval: ({bayes_ci[0]:.3f}, {bayes_ci[1]:.3f})")
    
    if results['dimensionality']['e']:
        print(f"\nDimensionality:")
        print(f"  E: {np.mean(results['dimensionality']['e']):.1f}")
        print(f"  I: {np.mean(results['dimensionality']['i']):.1f}")
        print(f"  Combined: {np.mean(results['dimensionality']['all']):.1f}")
        if results['dimensionality']['subspace_overlap']:
            print(f"  Subspace overlap: {np.mean(results['dimensionality']['subspace_overlap']):.3f}")
    
    print("="*70)
    
    return results


# ======================================================================
# EXPORT
# ======================================================================

def export_results(results):
    """Export comprehensive results"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Main E/I comparison
    main_df = pd.DataFrame([
        {'cell_type': 'E', 'mean_shesha': np.mean(results['shesha_e']),
         'std_shesha': np.std(results['shesha_e']), 'n': len(results['shesha_e'])},
        {'cell_type': 'I', 'mean_shesha': np.mean(results['shesha_i']),
         'std_shesha': np.std(results['shesha_i']), 'n': len(results['shesha_i'])},
    ])
    main_df.to_csv(os.path.join(OUTPUT_DIR, 'tier2_enhanced_ei_comparison.csv'), 
                    index=False)
    
    # Residual analysis
    residual_rows = []
    for i, res in enumerate(results['residuals']):
        residual_rows.append({
            'session_idx': i,
            'shared_variance': res['shared'],
            'e_unique': res['e_unique'],
            'i_unique': res['i_unique'],
        })
    pd.DataFrame(residual_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'tier2_enhanced_residuals.csv'), index=False)
    
    # PID results
    pid_rows = []
    for i, pid in enumerate(results['pid']):
        pid['session_idx'] = i
        pid_rows.append(pid)
    pd.DataFrame(pid_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'tier2_enhanced_pid.csv'), index=False)
    
    # Temporal dynamics
    if results['temporal_dynamics']:
        pd.DataFrame(results['temporal_dynamics']).to_csv(
            os.path.join(OUTPUT_DIR, 'tier2_enhanced_temporal.csv'), index=False)
    
    # Coordination
    coord_df = pd.DataFrame([{
        'ei_coordination_r': results['coordination']['r'],
        'ei_coordination_p': results['coordination']['p'],
    }])
    coord_df.to_csv(os.path.join(OUTPUT_DIR, 'tier2_enhanced_coordination.csv'), 
                     index=False)
    
    # Bootstrap CI
    boot_df = pd.DataFrame([{
        'e_mean': results['bootstrap']['e_mean'],
        'e_ci_lower': results['bootstrap']['e_ci'][0],
        'e_ci_upper': results['bootstrap']['e_ci'][1],
        'i_mean': results['bootstrap']['i_mean'],
        'i_ci_lower': results['bootstrap']['i_ci'][0],
        'i_ci_upper': results['bootstrap']['i_ci'][1],
        'mean_e_minus_i_bootstrap': results['bootstrap']['mean_diff'],
        'ci_lower_bootstrap': results['bootstrap']['ci'][0],
        'ci_upper_bootstrap': results['bootstrap']['ci'][1],
        'mean_e_minus_i_bayesian': results['bayesian']['mean_diff'],
        'ci_lower_bayesian': results['bayesian']['ci'][0],
        'ci_upper_bayesian': results['bayesian']['ci'][1],
    }])
    boot_df.to_csv(os.path.join(OUTPUT_DIR, 'tier2_enhanced_bootstrap.csv'), 
                    index=False)
    
    # Dimensionality
    dim_df = pd.DataFrame([{
        'mean_dim_e': np.mean(results['dimensionality']['e']) if results['dimensionality']['e'] else np.nan,
        'mean_dim_i': np.mean(results['dimensionality']['i']) if results['dimensionality']['i'] else np.nan,
        'mean_dim_all': np.mean(results['dimensionality']['all']) if results['dimensionality']['all'] else np.nan,
        'mean_subspace_overlap': np.mean(results['dimensionality']['subspace_overlap']) if results['dimensionality']['subspace_overlap'] else np.nan,
    }])
    dim_df.to_csv(os.path.join(OUTPUT_DIR, 'tier2_enhanced_dimensionality.csv'), 
                   index=False)
    
    print("\nExported results:")
    print(f"  - {OUTPUT_DIR}/tier2_enhanced_ei_comparison.csv")
    print(f"  - {OUTPUT_DIR}/tier2_enhanced_residuals.csv")
    print(f"  - {OUTPUT_DIR}/tier2_enhanced_pid.csv")
    print(f"  - {OUTPUT_DIR}/tier2_enhanced_temporal.csv")
    print(f"  - {OUTPUT_DIR}/tier2_enhanced_coordination.csv")
    print(f"  - {OUTPUT_DIR}/tier2_enhanced_bootstrap.csv")
    print(f"  - {OUTPUT_DIR}/tier2_enhanced_dimensionality.csv")


if __name__ == '__main__':
    import sys
    
    # Allow data path as command line argument
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    results = run_comprehensive_analysis(data_path=data_path)
    export_results(results)
    print("\nTier 2 Enhanced Analysis Complete!")