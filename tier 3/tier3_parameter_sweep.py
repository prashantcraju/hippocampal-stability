#!/usr/bin/env python3
"""
tier3_parameter_sweep.py - Comprehensive capacity modeling with:
  - Negative controls (matched random, anti-topological, dimension-matched)
  - Parametric sweeps with CIs (100 trials per topology level)
  - Multiple decoder comparisons (NN, linear, SVM, Bayesian)
  - Noise robustness analysis (additive, multiplicative, correlated)
  - Critical noise threshold detection
  - Information-theoretic capacity bounds
  - Cross-validation
  - Empirical redundancy analysis with neuron count controls
  - Spatial scale analysis
  
Note: Tier 3 doesn't use shesha-geometry package directly as it's simulation-based,
but capacity results connect to SHESHA measurements from Tier 1

=== USAGE IN COLAB (Batch Mode) ===

To run parameter sweep in batches (for Colab 24hr limit):

# Run batch 0 (first 3500 combinations):
results_df = run_parameter_sweep(batch_num=0, batch_size=3500)

# Run batch 1 (next 3500 combinations):
results_df = run_parameter_sweep(batch_num=1, batch_size=3500)

# Run batch 2 (remaining combinations):
results_df = run_parameter_sweep(batch_num=2, batch_size=3500)

# After all batches complete, combine results:
combined_df = combine_batch_results()

# Or run standard comprehensive analysis:
results = run_comprehensive_analysis()
export_results(results)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'output/tier3_sweep'

# Simulation config
N_NEURONS = 500
SPARSITY = 0.15
N_TRIALS = 250  # Increased from 20
NOISE_SCALE = 0.2
READ_NOISE = 0.3
# MEMORY_LOADS = [10, 20, 50, 100, 200, 500, 1000]
MEMORY_LOADS = [10*i for i in range(1, 101)]

TOPO_CRYSTAL = 1.0
TOPO_MIST = 0.5
TOPO_NOISE = 0.0

# Enhanced topology sweep
TOPO_LEVELS = np.linspace(0, 1, 11)  # 0, 0.1, 0.2, ..., 1.0


def generate_code(n_locations, n_neurons, sparsity, rng,
                   topology_strength=1.0, noise_scale=NOISE_SCALE):
    """Generate population code with topology control"""
    locations = np.linspace(0, 1, n_locations, endpoint=False)
    preferred = rng.uniform(0, 1, n_neurons)
    sigma = sparsity / 2
    
    # Topological component
    M_topo = np.zeros((n_locations, n_neurons))
    for j in range(n_neurons):
        d = np.abs(locations - preferred[j])
        d = np.minimum(d, 1 - d)
        M_topo[:, j] = np.exp(-d**2 / (2 * sigma**2))
    
    M_topo += rng.normal(0, noise_scale, M_topo.shape)
    M_topo = np.maximum(M_topo, 0)
    
    # Random component
    M_rand = np.zeros((n_locations, n_neurons))
    for j in range(n_neurons):
        M_rand[:, j] = rng.permutation(M_topo[:, j])
    
    # Mix
    n_topo = int(topology_strength * n_neurons)
    neuron_order = rng.permutation(n_neurons)
    topo_idx = neuron_order[:n_topo]
    rand_idx = neuron_order[n_topo:]
    
    M = np.zeros_like(M_topo)
    M[:, topo_idx] = M_topo[:, topo_idx]
    M[:, rand_idx] = M_rand[:, rand_idx]
    
    # Normalize
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1
    M = M / norms
    
    return M, locations


def measure_mantel(M, locations):
    """Measure Mantel r for a code"""
    phys = pdist(locations.reshape(-1, 1))
    neural = pdist(M, metric='euclidean')
    r, _ = spearmanr(phys, neural)
    return r


# ======================================================================
# MULTIPLE DECODERS
# ======================================================================

def nearest_neighbor_error(M, locs, rng, read_noise=READ_NOISE):
    """Noisy nearest-neighbor decoder (biologically plausible)"""
    M_noisy = M + rng.normal(0, read_noise, M.shape)
    D = cdist(M_noisy, M, metric='euclidean')
    np.fill_diagonal(D, np.inf)
    nn_idx = np.argmin(D, axis=1)
    
    err = np.abs(locs[nn_idx] - locs)
    err = np.minimum(err, 1 - err)
    return np.mean(err)


def linear_decoder_error(M, locs, rng, read_noise=READ_NOISE):
    """Linear regression decoder (optimal for random codes)"""
    M_noisy = M + rng.normal(0, read_noise, M.shape)
    
    # Train linear decoder with cross-validation
    try:
        decoder = RidgeCV(alphas=[0.1, 1.0, 10.0])
        decoder.fit(M, locs)
        
        # Predict
        locs_pred = decoder.predict(M_noisy)
        
        # Circular error
        err = np.abs(locs_pred - locs)
        err = np.minimum(err, 1 - err)
        return np.mean(err)
    except:
        return np.nan


def svm_decoder_error(M, locs, rng, read_noise=READ_NOISE):
    """
    SVM decoder (non-linear, intermediate case)
    Uses epsilon-SVR for regression
    """
    M_noisy = M + rng.normal(0, read_noise, M.shape)
    
    try:
        decoder = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        decoder.fit(M, locs)
        
        locs_pred = decoder.predict(M_noisy)
        
        # Circular error
        err = np.abs(locs_pred - locs)
        err = np.minimum(err, 1 - err)
        return np.mean(err)
    except:
        return np.nan


def ideal_bayesian_error(M, locs, rng, read_noise=READ_NOISE):
    """
    Bayesian decoder: P(location | response) ∝ P(response | location) P(location)
    Assumes Gaussian noise and uniform prior
    """
    M_noisy = M + rng.normal(0, read_noise, M.shape)
    
    errors = []
    for i in range(len(locs)):
        # Likelihood: Gaussian around true response
        log_likelihoods = -0.5 * np.sum((M - M_noisy[i])**2, axis=1) / (read_noise**2)
        
        # MAP estimate
        map_idx = np.argmax(log_likelihoods)
        
        err = abs(locs[map_idx] - locs[i])
        err = min(err, 1 - err)
        errors.append(err)
    
    return np.mean(errors)


def test_all_decoders(topology_strength, n_neurons, sparsity, noise_scale,
                       n_trials, memory_load, rng, read_noise=READ_NOISE):
    """Test multiple decoders on same codes"""
    results = {
        'nn': [],
        'linear': [],
        'svm': [],
        'bayesian': [],
        'mantel': [],
    }
    
    for _ in range(n_trials):
        seed = rng.randint(0, 1000000)
        trial_rng = np.random.RandomState(seed)
        M, locs = generate_code(memory_load, n_neurons, sparsity, trial_rng,
                                topology_strength=topology_strength,
                                noise_scale=noise_scale)
        
        # All decoders on same code
        results['nn'].append(nearest_neighbor_error(M, locs, trial_rng, read_noise))
        results['linear'].append(linear_decoder_error(M, locs, trial_rng, read_noise))
        results['svm'].append(svm_decoder_error(M, locs, trial_rng, read_noise))
        results['bayesian'].append(ideal_bayesian_error(M, locs, trial_rng, read_noise))
        results['mantel'].append(measure_mantel(M, locs))
    
    # Average and std
    for key in results:
        vals = [x for x in results[key] if not np.isnan(x)]
        results[key] = {
            'mean': np.mean(vals) if vals else np.nan,
            'std': np.std(vals) if vals else np.nan,
        }
    
    return results


# ======================================================================
# NEGATIVE CONTROLS
# ======================================================================

def generate_matched_random_code(target_mantel, n_locations, n_neurons, 
                                  sparsity, rng, max_attempts=50):
    """
    Generate random code with same Mantel r as target
    by adding precise amounts of structured noise
    """
    for attempt in range(max_attempts):
        # Try different topology strengths
        topo_str = rng.uniform(0, 1)
        M, locs = generate_code(n_locations, n_neurons, sparsity, rng,
                                topology_strength=topo_str, 
                                noise_scale=NOISE_SCALE)
        
        r = measure_mantel(M, locs)
        if abs(r - target_mantel) < 0.05:  # Close enough
            return M, locs
    
    # If we can't match, return best attempt
    return M, locs


def generate_anti_topological_code(n_locations, n_neurons, sparsity, rng):
    """
    Invert distance-similarity relationship:
    Far locations have similar codes
    """
    locations = np.linspace(0, 1, n_locations, endpoint=False)
    preferred = rng.uniform(0, 1, n_neurons)
    sigma = sparsity / 2
    
    M = np.zeros((n_locations, n_neurons))
    for j in range(n_neurons):
        d = np.abs(locations - preferred[j])
        d = np.minimum(d, 1 - d)
        # Invert: use max distance instead of min
        d_inv = 0.5 - d  # Flip around 0.5
        M[:, j] = np.exp(-d_inv**2 / (2 * sigma**2))
    
    M += rng.normal(0, NOISE_SCALE, M.shape)
    M = np.maximum(M, 0)
    
    # Normalize
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1
    M = M / norms
    
    return M, locations


def generate_dimension_matched_control(n_locations, n_neurons, sparsity, 
                                        target_dim, rng):
    """
    Generate random code with specific effective dimensionality
    """
    # Generate random code
    M = rng.normal(0, 1, (n_locations, n_neurons))
    
    # PCA to control dimensionality
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Zero out components beyond target_dim
    s[target_dim:] = 0
    
    # Reconstruct
    M = U @ np.diag(s) @ Vt
    
    # Add sparsity
    M = M * (rng.random(M.shape) < sparsity)
    
    # Normalize
    M = np.maximum(M, 0)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1
    M = M / norms
    
    locations = np.linspace(0, 1, n_locations, endpoint=False)
    return M, locations


# ======================================================================
# NOISE ROBUSTNESS
# ======================================================================

def additive_noise_robustness(n_neurons, sparsity, memory_load, n_trials=250):
    """Test performance across additive noise levels"""
    noise_levels = np.linspace(0.1, 1.0, 10)
    
    results = {
        'noise': noise_levels,
        'crystal': [],
        'mist': [],
        'random': [],
    }
    
    print("\nAdditive noise robustness analysis...")
    for noise in tqdm(noise_levels):
        rng = np.random.RandomState(320)
        
        # Crystal
        errs = []
        for _ in range(n_trials):
            M, locs = generate_code(memory_load, n_neurons, sparsity, rng,
                                    topology_strength=TOPO_CRYSTAL,
                                    noise_scale=NOISE_SCALE)
            err = nearest_neighbor_error(M, locs, rng, read_noise=noise)
            errs.append(err)
        results['crystal'].append(np.mean(errs))
        
        # Mist
        errs = []
        for _ in range(n_trials):
            M, locs = generate_code(memory_load, n_neurons, sparsity, rng,
                                    topology_strength=TOPO_MIST,
                                    noise_scale=NOISE_SCALE)
            err = nearest_neighbor_error(M, locs, rng, read_noise=noise)
            errs.append(err)
        results['mist'].append(np.mean(errs))
        
        # Random
        errs = []
        for _ in range(n_trials):
            M, locs = generate_code(memory_load, n_neurons, sparsity, rng,
                                    topology_strength=TOPO_NOISE,
                                    noise_scale=NOISE_SCALE)
            err = nearest_neighbor_error(M, locs, rng, read_noise=noise)
            errs.append(err)
        results['random'].append(np.mean(errs))
    
    return results


def multiplicative_noise_robustness(n_neurons, sparsity, memory_load, n_trials=250):
    """Test performance with multiplicative (Poisson-like) noise"""
    noise_factors = np.linspace(0.1, 2.0, 10)
    
    results = {
        'noise_factor': noise_factors,
        'crystal': [],
        'mist': [],
    }
    
    print("\nMultiplicative noise robustness analysis...")
    for factor in tqdm(noise_factors):
        rng = np.random.RandomState(320)
        
        for model, topo in [('crystal', TOPO_CRYSTAL), ('mist', TOPO_MIST)]:
            errs = []
            for _ in range(n_trials):
                M, locs = generate_code(memory_load, n_neurons, sparsity, rng,
                                        topology_strength=topo,
                                        noise_scale=NOISE_SCALE)
                
                # Multiplicative noise: noise proportional to signal
                M_noisy = M * (1 + factor * rng.normal(0, 1, M.shape))
                M_noisy = np.maximum(M_noisy, 0)
                
                # Decode
                D = cdist(M_noisy, M, metric='euclidean')
                np.fill_diagonal(D, np.inf)
                nn_idx = np.argmin(D, axis=1)
                
                err = np.abs(locs[nn_idx] - locs)
                err = np.minimum(err, 1 - err)
                errs.append(np.mean(err))
            
            results[model].append(np.mean(errs))
    
    return results


def correlated_noise_robustness(n_neurons, sparsity, memory_load, n_trials=250):
    """Test with correlated noise across neurons"""
    correlation_levels = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    results = {
        'correlation': correlation_levels,
        'crystal': [],
        'mist': [],
    }
    
    print("\nCorrelated noise robustness analysis...")
    for corr in tqdm(correlation_levels):
        rng = np.random.RandomState(320)
        
        for model, topo in [('crystal', TOPO_CRYSTAL), ('mist', TOPO_MIST)]:
            errs = []
            for _ in range(n_trials):
                M, locs = generate_code(memory_load, n_neurons, sparsity, rng,
                                        topology_strength=topo,
                                        noise_scale=NOISE_SCALE)
                
                # Generate correlated noise
                noise_common = rng.normal(0, READ_NOISE, M.shape[0])
                noise_indep = rng.normal(0, READ_NOISE, M.shape)
                
                noise = (np.sqrt(corr) * noise_common[:, np.newaxis] + 
                         np.sqrt(1 - corr) * noise_indep)
                
                M_noisy = M + noise
                
                # Decode
                D = cdist(M_noisy, M, metric='euclidean')
                np.fill_diagonal(D, np.inf)
                nn_idx = np.argmin(D, axis=1)
                
                err = np.abs(locs[nn_idx] - locs)
                err = np.minimum(err, 1 - err)
                errs.append(np.mean(err))
            
            results[model].append(np.mean(errs))
    
    return results


def find_critical_noise_threshold(n_neurons, sparsity, memory_load, 
                                   error_threshold=0.1, n_trials=250):
    """
    Find noise level where crystal and mist become indistinguishable
    """
    noise_levels = np.linspace(0.1, 2.0, 20)
    
    crystal_errors = []
    mist_errors = []
    
    rng = np.random.RandomState(320)
    
    print("\nFinding critical noise threshold...")
    for noise in tqdm(noise_levels):
        # Crystal
        errs = []
        for _ in range(n_trials):
            M, locs = generate_code(memory_load, n_neurons, sparsity, rng,
                                    topology_strength=TOPO_CRYSTAL,
                                    noise_scale=NOISE_SCALE)
            err = nearest_neighbor_error(M, locs, rng, read_noise=noise)
            errs.append(err)
        crystal_errors.append(np.mean(errs))
        
        # Mist
        errs = []
        for _ in range(n_trials):
            M, locs = generate_code(memory_load, n_neurons, sparsity, rng,
                                    topology_strength=TOPO_MIST,
                                    noise_scale=NOISE_SCALE)
            err = nearest_neighbor_error(M, locs, rng, read_noise=noise)
            errs.append(err)
        mist_errors.append(np.mean(errs))
    
    # Find where difference falls below threshold
    differences = np.array(mist_errors) - np.array(crystal_errors)
    critical_idx = np.where(differences < error_threshold)[0]
    
    if len(critical_idx) > 0:
        critical_noise = noise_levels[critical_idx[0]]
    else:
        critical_noise = np.nan
    
    return {
        'noise_levels': noise_levels,
        'crystal_errors': crystal_errors,
        'mist_errors': mist_errors,
        'differences': differences,
        'critical_noise': critical_noise,
    }


# ======================================================================
# INFORMATION-THEORETIC CAPACITY BOUNDS
# ======================================================================

def theoretical_capacity_lower_bound(n_neurons, sparsity, noise):
    """
    Compute theoretical lower bound on capacity using Shannon theory
    Assumes Gaussian channel
    """
    # Signal power (approximation based on sparsity)
    signal_power = sparsity * n_neurons
    
    # Noise power
    noise_power = noise ** 2
    
    # Channel capacity (bits)
    capacity = 0.5 * np.log2(1 + signal_power / noise_power)
    
    return capacity


def theoretical_capacity_upper_bound(n_neurons, sparsity):
    """
    Upper bound: if perfectly topological, capacity scales with neurons
    """
    # Maximum distinguishable patterns
    max_patterns = n_neurons / sparsity  # Rough estimate
    
    capacity = np.log2(max_patterns)
    
    return capacity


def empirical_information_content(M, locs):
    """
    Estimate empirical information using nearest-neighbor entropy
    """
    # Compute pairwise distances
    neural_dists = pdist(M, metric='euclidean')
    
    # For each location, find average distance to nearest neighbor
    D = cdist(M, M, metric='euclidean')
    np.fill_diagonal(D, np.inf)
    nn_dists = np.min(D, axis=1)
    
    # Information ~ log(volume) ~ log(avg_nn_distance^d)
    avg_nn_dist = np.mean(nn_dists)
    
    # Effective dimensionality estimate
    d_eff = len(M[0])
    
    information = d_eff * np.log2(avg_nn_dist + 1e-10)
    
    return information


# ======================================================================
# EMPIRICAL REDUNDANCY ANALYSIS
# ======================================================================

def control_for_neuron_count(M_populations, target_n_neurons):
    """
    Subsample larger populations to match target neuron count
    """
    subsampled = []
    
    for M in M_populations:
        n_current = M.shape[1]
        
        if n_current > target_n_neurons:
            # Subsample
            idx = np.random.choice(n_current, target_n_neurons, replace=False)
            M_sub = M[:, idx]
        else:
            M_sub = M
        
        subsampled.append(M_sub)
    
    return subsampled


def compute_redundancy_at_scales(M, locations, spatial_scales):
    """
    Compute redundancy at different spatial resolutions
    """
    results = []
    
    for scale in spatial_scales:
        # Downsample locations
        n_downsampled = max(10, len(locations) // scale)
        idx = np.linspace(0, len(locations)-1, n_downsampled, dtype=int)
        
        M_down = M[idx]
        locs_down = locations[idx]
        
        # Measure redundancy (via Mantel r)
        mantel_r = measure_mantel(M_down, locs_down)
        
        results.append({
            'scale': scale,
            'n_locations': n_downsampled,
            'mantel_r': mantel_r,
        })
    
    return results


# ======================================================================
# COMPREHENSIVE TOPOLOGY SWEEP
# ======================================================================

def topology_sweep_with_ci(n_neurons, sparsity, memory_load, n_trials=250):
    """
    Sweep topology from 0 to 1 with proper error bars
    Test if differences between adjacent levels are significant
    """
    results = {
        'topology': TOPO_LEVELS,
        'error_mean': [],
        'error_std': [],
        'error_ci_lower': [],
        'error_ci_upper': [],
        'mantel_mean': [],
        'mantel_std': [],
        'mantel_ci_lower': [],
        'mantel_ci_upper': [],
    }
    
    print(f"\nTopology sweep at {memory_load} locations...")
    for topo in tqdm(TOPO_LEVELS):
        rng = np.random.RandomState(320)
        errors = []
        mantels = []
        
        for _ in range(n_trials):
            M, locs = generate_code(memory_load, n_neurons, sparsity, rng,
                                    topology_strength=topo,
                                    noise_scale=NOISE_SCALE)
            
            err = nearest_neighbor_error(M, locs, rng, read_noise=READ_NOISE)
            errors.append(err)
            
            r = measure_mantel(M, locs)
            mantels.append(r)
        
        errors = np.array(errors)
        mantels = np.array(mantels)
        
        results['error_mean'].append(np.mean(errors))
        results['error_std'].append(np.std(errors))
        results['error_ci_lower'].append(np.percentile(errors, 2.5))
        results['error_ci_upper'].append(np.percentile(errors, 97.5))
        
        results['mantel_mean'].append(np.mean(mantels))
        results['mantel_std'].append(np.std(mantels))
        results['mantel_ci_lower'].append(np.percentile(mantels, 2.5))
        results['mantel_ci_upper'].append(np.percentile(mantels, 97.5))
    
    return results


# ======================================================================
# MAIN ANALYSIS
# ======================================================================

def run_comprehensive_analysis():
    """Run all capacity analyses"""
    print("="*70)
    print("TIER 3 ENHANCED: Comprehensive Capacity Analysis")
    print("="*70)
    
    rng = np.random.RandomState(320)
    
    results = {}
    
    # 1. Decoder comparison at one memory load
    print("\n1. Comparing decoders (100 locations, 100 trials)...")
    decoder_comp = {
        'crystal': test_all_decoders(TOPO_CRYSTAL, N_NEURONS, SPARSITY, 
                                     NOISE_SCALE, 100, 100, rng, READ_NOISE),
        'mist': test_all_decoders(TOPO_MIST, N_NEURONS, SPARSITY,
                                  NOISE_SCALE, 100, 100, rng, READ_NOISE),
        'random': test_all_decoders(TOPO_NOISE, N_NEURONS, SPARSITY,
                                    NOISE_SCALE, 100, 100, rng, READ_NOISE),
    }
    results['decoder_comparison'] = decoder_comp
    
    print("\n  Decoder Results (100 locations):")
    for model in ['crystal', 'mist', 'random']:
        print(f"  {model.capitalize()}:")
        for decoder in ['nn', 'linear', 'svm', 'bayesian']:
            m = decoder_comp[model][decoder]['mean']
            s = decoder_comp[model][decoder]['std']
            print(f"    {decoder}: {m:.4f} ± {s:.4f}")
    
    # 2. Topology sweep with CI
    sweep_results = topology_sweep_with_ci(N_NEURONS, SPARSITY, 100, n_trials=250)
    results['topology_sweep'] = sweep_results
    
    # 3. Noise robustness analyses
    print("\n3. Noise robustness analyses...")
    
    # Additive noise
    additive_results = additive_noise_robustness(N_NEURONS, SPARSITY, 100, 
                                                  n_trials=250)
    results['additive_noise'] = additive_results
    
    # Multiplicative noise
    mult_results = multiplicative_noise_robustness(N_NEURONS, SPARSITY, 100, 
                                                    n_trials=250)
    results['multiplicative_noise'] = mult_results
    
    # Correlated noise
    corr_results = correlated_noise_robustness(N_NEURONS, SPARSITY, 100, 
                                                n_trials=250)
    results['correlated_noise'] = corr_results
    
    # Critical threshold
    threshold_results = find_critical_noise_threshold(N_NEURONS, SPARSITY, 100,
                                                       n_trials=250)
    results['critical_threshold'] = threshold_results
    
    print(f"  Critical noise threshold: {threshold_results['critical_noise']:.3f}")
    
    # 4. Negative controls
    print("\n4. Testing negative controls...")
    
    # Anti-topological
    anti_errors = []
    for _ in tqdm(range(20), desc="Anti-topological"):
        M, locs = generate_anti_topological_code(100, N_NEURONS, SPARSITY, rng)
        err = nearest_neighbor_error(M, locs, rng, READ_NOISE)
        anti_errors.append(err)
    results['anti_topological_error'] = np.mean(anti_errors)
    
    # Dimension-matched control
    dim_matched_errors = []
    for _ in tqdm(range(20), desc="Dimension-matched"):
        M, locs = generate_dimension_matched_control(100, N_NEURONS, SPARSITY,
                                                      target_dim=50, rng=rng)
        err = nearest_neighbor_error(M, locs, rng, READ_NOISE)
        dim_matched_errors.append(err)
    results['dimension_matched_error'] = np.mean(dim_matched_errors)
    
    print(f"  Anti-topological error: {np.mean(anti_errors):.4f}")
    print(f"  Dimension-matched error: {np.mean(dim_matched_errors):.4f}")
    
    # 5. Theoretical bounds
    print("\n5. Computing theoretical capacity bounds...")
    lower_bound = theoretical_capacity_lower_bound(N_NEURONS, SPARSITY, READ_NOISE)
    upper_bound = theoretical_capacity_upper_bound(N_NEURONS, SPARSITY)
    
    results['theoretical_bounds'] = {
        'lower': lower_bound,
        'upper': upper_bound,
    }
    
    print(f"  Theoretical lower bound: {lower_bound:.2f} bits")
    print(f"  Theoretical upper bound: {upper_bound:.2f} bits")
    
    # 6. Spatial scale analysis
    print("\n6. Spatial scale redundancy analysis...")
    M_test, locs_test = generate_code(500, N_NEURONS, SPARSITY, rng,
                                       topology_strength=TOPO_CRYSTAL,
                                       noise_scale=NOISE_SCALE)
    
    scales = [i+1 for i in range(25)]
    scale_results = compute_redundancy_at_scales(M_test, locs_test, scales)
    results['spatial_scales'] = scale_results
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return results


# ======================================================================
# EXPORT
# ======================================================================

def export_results(results):
    """Export comprehensive results"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Decoder comparison
    decoder_rows = []
    for model in ['crystal', 'mist', 'random']:
        for decoder in ['nn', 'linear', 'svm', 'bayesian']:
            decoder_rows.append({
                'model': model,
                'decoder': decoder,
                'error_mean': results['decoder_comparison'][model][decoder]['mean'],
                'error_std': results['decoder_comparison'][model][decoder]['std'],
                'mantel_r': results['decoder_comparison'][model]['mantel']['mean'],
            })
    pd.DataFrame(decoder_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'tier3_enhanced_decoder_comparison.csv'), index=False)
    
    # Topology sweep
    sweep_df = pd.DataFrame({
        'topology': results['topology_sweep']['topology'],
        'error_mean': results['topology_sweep']['error_mean'],
        'error_std': results['topology_sweep']['error_std'],
        'error_ci_lower': results['topology_sweep']['error_ci_lower'],
        'error_ci_upper': results['topology_sweep']['error_ci_upper'],
        'mantel_mean': results['topology_sweep']['mantel_mean'],
        'mantel_std': results['topology_sweep']['mantel_std'],
        'mantel_ci_lower': results['topology_sweep']['mantel_ci_lower'],
        'mantel_ci_upper': results['topology_sweep']['mantel_ci_upper'],
    })
    sweep_df.to_csv(os.path.join(OUTPUT_DIR, 'tier3_enhanced_topology_sweep.csv'), 
                     index=False)
    
    # Additive noise robustness
    additive_df = pd.DataFrame({
        'read_noise': results['additive_noise']['noise'],
        'crystal_error': results['additive_noise']['crystal'],
        'mist_error': results['additive_noise']['mist'],
        'random_error': results['additive_noise']['random'],
    })
    additive_df.to_csv(os.path.join(OUTPUT_DIR, 'tier3_enhanced_additive_noise.csv'),
                       index=False)
    
    # Multiplicative noise
    mult_df = pd.DataFrame({
        'noise_factor': results['multiplicative_noise']['noise_factor'],
        'crystal_error': results['multiplicative_noise']['crystal'],
        'mist_error': results['multiplicative_noise']['mist'],
    })
    mult_df.to_csv(os.path.join(OUTPUT_DIR, 'tier3_enhanced_multiplicative_noise.csv'),
                    index=False)
    
    # Correlated noise
    corr_df = pd.DataFrame({
        'correlation': results['correlated_noise']['correlation'],
        'crystal_error': results['correlated_noise']['crystal'],
        'mist_error': results['correlated_noise']['mist'],
    })
    corr_df.to_csv(os.path.join(OUTPUT_DIR, 'tier3_enhanced_correlated_noise.csv'),
                    index=False)
    
    # Critical threshold
    threshold_df = pd.DataFrame({
        'noise_level': results['critical_threshold']['noise_levels'],
        'crystal_error': results['critical_threshold']['crystal_errors'],
        'mist_error': results['critical_threshold']['mist_errors'],
        'difference': results['critical_threshold']['differences'],
    })
    threshold_df.to_csv(os.path.join(OUTPUT_DIR, 'tier3_enhanced_critical_threshold.csv'),
                        index=False)
    
    # Spatial scales
    if results['spatial_scales']:
        scales_df = pd.DataFrame(results['spatial_scales'])
        scales_df.to_csv(os.path.join(OUTPUT_DIR, 'tier3_enhanced_spatial_scales.csv'),
                         index=False)
    
    # Summary
    summary_df = pd.DataFrame([{
        'anti_topological_error': results['anti_topological_error'],
        'dimension_matched_error': results['dimension_matched_error'],
        'crystal_nn_error': results['decoder_comparison']['crystal']['nn']['mean'],
        'mist_nn_error': results['decoder_comparison']['mist']['nn']['mean'],
        'random_nn_error': results['decoder_comparison']['random']['nn']['mean'],
        'critical_noise_threshold': results['critical_threshold']['critical_noise'],
        'theoretical_lower_bound_bits': results['theoretical_bounds']['lower'],
        'theoretical_upper_bound_bits': results['theoretical_bounds']['upper'],
    }])
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'tier3_enhanced_summary.csv'), 
                       index=False)
    
    print("\nExported results:")
    print("  - tier3_enhanced_decoder_comparison.csv")
    print("  - tier3_enhanced_topology_sweep.csv")
    print("  - tier3_enhanced_additive_noise.csv")
    print("  - tier3_enhanced_multiplicative_noise.csv")
    print("  - tier3_enhanced_correlated_noise.csv")
    print("  - tier3_enhanced_critical_threshold.csv")
    print("  - tier3_enhanced_spatial_scales.csv")
    print("  - tier3_enhanced_summary.csv")


# ======================================================================
# PARAMETER SWEEP
# ======================================================================

def run_parameter_sweep(batch_num=None, batch_size=3500, save_incremental=True):
    """
    Run parameter sweep over N_NEURONS, SPARSITY, and N_TRIALS
    
    This tests:
    - N_NEURONS: 25 to 500 in steps of 25 (20 values)
    - SPARSITY: 0.01 to 0.25 in steps of 0.01 (25 values)
    - N_TRIALS: 25 to 500 in steps of 25 (20 values)
    
    Total combinations: 20 × 25 × 20 = 10,000
    
    Parameters
    ----------
    batch_num : int or None
        If specified, only process this batch (0-indexed)
        If None, process all combinations
    batch_size : int
        Number of combinations per batch (default: 3500)
    save_incremental : bool
        If True, saves results every 100 combinations
    """
    # Define parameter ranges
    n_neurons_range = list(range(25, 501, 25))  # [25, 50, 75, ..., 500]
    sparsity_range = [round(i * 0.01, 2) for i in range(1, 26)]  # [0.01, 0.02, ..., 0.25]
    n_trials_range = list(range(25, 501, 25))  # [25, 50, 75, ..., 500]
    
    # Generate all combinations
    from itertools import product
    all_combinations = list(product(n_neurons_range, sparsity_range, n_trials_range))
    total_combinations = len(all_combinations)
    
    # Calculate batch info
    if batch_num is not None:
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_combinations)
        combinations_to_process = all_combinations[start_idx:end_idx]
        n_batches = (total_combinations + batch_size - 1) // batch_size
    else:
        combinations_to_process = all_combinations
        start_idx = 0
        end_idx = total_combinations
    
    print("="*70)
    print("TIER 3 PARAMETER SWEEP")
    print("="*70)
    print(f"N_NEURONS range: {n_neurons_range[0]} to {n_neurons_range[-1]} "
          f"({len(n_neurons_range)} values)")
    print(f"SPARSITY range: {sparsity_range[0]} to {sparsity_range[-1]} "
          f"({len(sparsity_range)} values)")
    print(f"N_TRIALS range: {n_trials_range[0]} to {n_trials_range[-1]} "
          f"({len(n_trials_range)} values)")
    print(f"\nTotal combinations: {total_combinations:,}")
    
    if batch_num is not None:
        print(f"\nBATCH MODE: Processing batch {batch_num} of {n_batches-1}")
        print(f"  Combinations: {start_idx} to {end_idx-1} ({len(combinations_to_process)} total)")
        print(f"  Estimated time: {len(combinations_to_process) * 20 / 3600:.1f} hours")
    else:
        print(f"Estimated time: {total_combinations * 20 / 3600:.1f} hours")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Prepare results storage
    all_results = []
    
    # Main sweep loop
    for combo_idx, (n_neurons, sparsity, n_trials) in enumerate(combinations_to_process):
        global_idx = start_idx + combo_idx
        
        print(f"\n[{global_idx+1}/{total_combinations}] ({(global_idx+1)/total_combinations*100:.1f}%) "
              f"N_NEURONS={n_neurons}, SPARSITY={sparsity:.2f}, N_TRIALS={n_trials}")
        
        rng = np.random.RandomState(320)
        
        # Run streamlined analysis for this parameter combination
        result = {
            'n_neurons': n_neurons,
            'sparsity': sparsity,
            'n_trials': n_trials,
        }
        
        try:
            # Test at fixed memory load (100 locations)
            memory_load = 100
            
            # 1. Test all decoders for crystal topology
            decoder_results = test_all_decoders(
                TOPO_CRYSTAL, n_neurons, sparsity, NOISE_SCALE, 
                n_trials, memory_load, rng, READ_NOISE
            )
            
            result['crystal_nn_error'] = decoder_results['nn']['mean']
            result['crystal_nn_std'] = decoder_results['nn']['std']
            result['crystal_linear_error'] = decoder_results['linear']['mean']
            result['crystal_linear_std'] = decoder_results['linear']['std']
            result['crystal_svm_error'] = decoder_results['svm']['mean']
            result['crystal_svm_std'] = decoder_results['svm']['std']
            result['crystal_bayesian_error'] = decoder_results['bayesian']['mean']
            result['crystal_bayesian_std'] = decoder_results['bayesian']['std']
            result['crystal_mantel'] = decoder_results['mantel']['mean']
            
            # 2. Test mist topology
            mist_results = test_all_decoders(
                TOPO_MIST, n_neurons, sparsity, NOISE_SCALE,
                n_trials, memory_load, rng, READ_NOISE
            )
            
            result['mist_nn_error'] = mist_results['nn']['mean']
            result['mist_nn_std'] = mist_results['nn']['std']
            result['mist_mantel'] = mist_results['mantel']['mean']
            
            # 3. Test random topology
            random_results = test_all_decoders(
                TOPO_NOISE, n_neurons, sparsity, NOISE_SCALE,
                n_trials, memory_load, rng, READ_NOISE
            )
            
            result['random_nn_error'] = random_results['nn']['mean']
            result['random_nn_std'] = random_results['nn']['std']
            result['random_mantel'] = random_results['mantel']['mean']
            
            # 4. Theoretical bounds
            lower_bound = theoretical_capacity_lower_bound(n_neurons, sparsity, READ_NOISE)
            upper_bound = theoretical_capacity_upper_bound(n_neurons, sparsity)
            
            result['theoretical_lower_bound'] = lower_bound
            result['theoretical_upper_bound'] = upper_bound
            
            # 5. Performance metrics
            result['crystal_advantage'] = result['random_nn_error'] - result['crystal_nn_error']
            result['mist_advantage'] = result['random_nn_error'] - result['mist_nn_error']
            
            result['status'] = 'success'
            
        except Exception as e:
            print(f"  ERROR: {e}")
            result['status'] = 'failed'
            result['error_message'] = str(e)
        
        all_results.append(result)
        
        # Save incrementally every 100 combinations
        if save_incremental and (combo_idx + 1) % 100 == 0:
            df_temp = pd.DataFrame(all_results)
            if batch_num is not None:
                filename = f'tier3_parameter_sweep_batch_{batch_num}_incremental.csv'
            else:
                filename = 'tier3_parameter_sweep_incremental.csv'
            df_temp.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
            successful = sum(1 for r in all_results if r['status'] == 'success')
            print(f"  ✓ Saved incremental: {len(all_results)} combos ({successful} successful)")
    
    # Save final results
    print("\n" + "="*70)
    print("SAVING FINAL RESULTS")
    print("="*70)
    
    df_final = pd.DataFrame(all_results)
    
    # Main results file - include batch number if specified
    if batch_num is not None:
        main_filename = f'tier3_parameter_sweep_batch_{batch_num}.csv'
        print(f"Batch {batch_num} complete: {len(all_results)} combinations")
    else:
        main_filename = 'tier3_parameter_sweep_complete.csv'
    
    df_final.to_csv(os.path.join(OUTPUT_DIR, main_filename), index=False)
    
    # Summary statistics (skip for batches - will combine later)
    if batch_num is None:
        summary_by_neurons = df_final.groupby('n_neurons').agg({
            'crystal_nn_error': ['mean', 'std'],
            'crystal_advantage': ['mean', 'std'],
            'crystal_mantel': ['mean', 'std'],
        }).round(4)
        summary_by_neurons.to_csv(
            os.path.join(OUTPUT_DIR, 'tier3_sweep_summary_by_neurons.csv')
        )
        
        summary_by_sparsity = df_final.groupby('sparsity').agg({
            'crystal_nn_error': ['mean', 'std'],
            'crystal_advantage': ['mean', 'std'],
            'crystal_mantel': ['mean', 'std'],
        }).round(4)
        summary_by_sparsity.to_csv(
            os.path.join(OUTPUT_DIR, 'tier3_sweep_summary_by_sparsity.csv')
        )
        
        summary_by_trials = df_final.groupby('n_trials').agg({
            'crystal_nn_error': ['mean', 'std'],
            'crystal_advantage': ['mean', 'std'],
            'crystal_mantel': ['mean', 'std'],
        }).round(4)
        summary_by_trials.to_csv(
            os.path.join(OUTPUT_DIR, 'tier3_sweep_summary_by_trials.csv')
        )
        
        print("\nExported files:")
        print("  - tier3_parameter_sweep_complete.csv (all combinations)")
        print("  - tier3_sweep_summary_by_neurons.csv")
        print("  - tier3_sweep_summary_by_sparsity.csv")
        print("  - tier3_sweep_summary_by_trials.csv")
    else:
        print(f"\nExported batch file:")
        print(f"  - {main_filename}")
        print(f"\nRun combine_batch_results() after all batches are complete.")
    
    print("="*70)
    
    return df_final


def combine_batch_results():
    """
    Combine all batch CSV files into a single complete file
    Call this after all batches are complete
    """
    # Find all batch files
    import glob
    batch_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'tier3_parameter_sweep_batch_*.csv')))
    
    # Exclude incremental files
    batch_files = [f for f in batch_files if 'incremental' not in f]
    
    if not batch_files:
        print("No batch files found!")
        return None
    
    print("="*70)
    print("COMBINING BATCH RESULTS")
    print("="*70)
    print(f"Found {len(batch_files)} batch files:")
    for f in batch_files:
        print(f"  - {os.path.basename(f)}")
    
    # Combine all batches
    dfs = []
    for batch_file in batch_files:
        df = pd.read_csv(batch_file)
        dfs.append(df)
        print(f"  Loaded {len(df)} combinations from {os.path.basename(batch_file)}")
    
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Save combined file
    combined_path = os.path.join(OUTPUT_DIR, 'tier3_parameter_sweep_complete.csv')
    df_combined.to_csv(combined_path, index=False)
    
    print(f"\n✓ Combined {len(df_combined)} total combinations")
    print(f"  Saved to: {combined_path}")
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    
    summary_by_neurons = df_combined.groupby('n_neurons').agg({
        'crystal_nn_error': ['mean', 'std'],
        'crystal_advantage': ['mean', 'std'],
        'crystal_mantel': ['mean', 'std'],
    }).round(4)
    summary_by_neurons.to_csv(
        os.path.join(OUTPUT_DIR, 'tier3_sweep_summary_by_neurons.csv')
    )
    
    summary_by_sparsity = df_combined.groupby('sparsity').agg({
        'crystal_nn_error': ['mean', 'std'],
        'crystal_advantage': ['mean', 'std'],
        'crystal_mantel': ['mean', 'std'],
    }).round(4)
    summary_by_sparsity.to_csv(
        os.path.join(OUTPUT_DIR, 'tier3_sweep_summary_by_sparsity.csv')
    )
    
    summary_by_trials = df_combined.groupby('n_trials').agg({
        'crystal_nn_error': ['mean', 'std'],
        'crystal_advantage': ['mean', 'std'],
        'crystal_mantel': ['mean', 'std'],
    }).round(4)
    summary_by_trials.to_csv(
        os.path.join(OUTPUT_DIR, 'tier3_sweep_summary_by_trials.csv')
    )
    
    print("\n✓ Exported summary files:")
    print("  - tier3_parameter_sweep_complete.csv")
    print("  - tier3_sweep_summary_by_neurons.csv")
    print("  - tier3_sweep_summary_by_sparsity.csv")
    print("  - tier3_sweep_summary_by_trials.csv")
    print("="*70)
    
    return df_combined


if __name__ == '__main__':
    # Run standard comprehensive analysis
    print("\nRunning STANDARD mode...")
    results = run_comprehensive_analysis()
    export_results(results)
    print("\nTier 3 Enhanced Analysis Complete!")