#!/usr/bin/env python3
"""
tier3_capacity_redundancy.py - Functional consequences of topological
fidelity: simulation proof + empirical redundancy.

PART 1 (Panels A-F): Computational model proving topological
preservation maximizes memory capacity.

  Decoder: Noisy nearest-neighbor. The brain identifies "where am I?"
  by matching its current (noisy) population vector against stored
  templates. Topology helps because nearby locations produce similar
  templates, so noise shifts the readout to a spatial neighbor rather
  than a random wrong answer.

  This is superior to linear decoding, which actually favors random
  codes (high dimensionality gives more orthogonal bases for the
  decoder, masking the benefit of topological structure).

  Two orthogonal parameters:
    noise_scale (0.2): Functional brain-internal noise. Held constant.
    topology_strength: Fraction of topological neurons. Variable.

  Crystal (chickadee): topology=1.0 (all neurons spatially organized)
  Mist (finch):        topology=0.5 (half, matching empirical 2:1 ratio)
  Noise (null):        topology=0.0

PART 2 (Panel G): Empirical redundancy ratio from Aronov data.

Usage:
    python tier3_capacity_redundancy.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'output/tier3_capacity'

# ── Simulation config ──
N_NEURONS = 500
SPARSITY = 0.15
N_TRIALS = 20
NOISE_SCALE = 0.2     # Functional (brain-internal) noise
READ_NOISE = 0.3      # Noise added at readout for noisy-NN test
MEMORY_LOADS = [10, 20, 50, 100, 200, 500, 1000]

TOPO_CRYSTAL = 1.0
TOPO_MIST = 0.5
TOPO_NOISE = 0.0

# ── Empirical data config ──
DATA_PATH = 'data/aronov_dataset.pkl'

GRID_SIZE = 40
N_BINS = GRID_SIZE ** 2
MIN_NEURONS = 5

# ── Colors ──
C_CRYSTAL = '#C62D50'
C_MIST = '#1C3D8F'
C_NOISE = '#888888'


# ══════════════════════════════════════════════════════════════════════
# PART 1: CAPACITY MODEL
# ══════════════════════════════════════════════════════════════════════

def generate_code(n_locations, n_neurons, sparsity, rng,
                   topology_strength=1.0, noise_scale=NOISE_SCALE):
    """
    Generate a neural population code with independent control over
    spatial topology and biological noise.
    """
    locations = np.linspace(0, 1, n_locations, endpoint=False)

    preferred = rng.uniform(0, 1, n_neurons)
    sigma = sparsity / 2

    M_topo = np.zeros((n_locations, n_neurons))
    for j in range(n_neurons):
        d = np.abs(locations - preferred[j])
        d = np.minimum(d, 1 - d)
        M_topo[:, j] = np.exp(-d**2 / (2 * sigma**2))

    M_topo += rng.normal(0, noise_scale, M_topo.shape)
    M_topo = np.maximum(M_topo, 0)

    M_rand = np.zeros((n_locations, n_neurons))
    for j in range(n_neurons):
        M_rand[:, j] = rng.permutation(M_topo[:, j])

    n_topo = int(topology_strength * n_neurons)
    neuron_order = rng.permutation(n_neurons)
    topo_idx = neuron_order[:n_topo]
    rand_idx = neuron_order[n_topo:]

    M = np.zeros_like(M_topo)
    M[:, topo_idx] = M_topo[:, topo_idx]
    M[:, rand_idx] = M_rand[:, rand_idx]

    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1
    M = M / norms

    return M, locations


def measure_mantel(n_locations, n_neurons, sparsity, topology_strength,
                    noise_scale):
    """Average Mantel r over n_avg random seeds."""
    rs = []
    seeds = [320, 1991, 9, 7258, 7, 2222, 724, 3, 12, 108, 18, 11, 1754, 411, 103]
    for seed in seeds:
        rng = np.random.RandomState(seed)
        M, locs = generate_code(n_locations, n_neurons, sparsity, rng,
                                 topology_strength=topology_strength,
                                 noise_scale=noise_scale)
        phys = pdist(locs.reshape(-1, 1))
        neural = pdist(M, metric='euclidean')
        r, _ = spearmanr(phys, neural)
        rs.append(r)
    return np.mean(rs)


# ── Noisy nearest-neighbor capacity test ──

def noisy_nn_error(M, locs, rng, read_noise=READ_NOISE):
    """
    Noisy nearest-neighbor decoding error.

    Simulates the brain reading its own noisy activity: for each
    location, add readout noise to its population vector, then find
    the nearest template among all stored locations. Error is the
    circular spatial distance between the true and decoded locations.

    Topology helps because nearby locations have similar templates,
    so noise-induced misidentification tends toward spatial neighbors
    rather than random wrong answers.
    """
    M_noisy = M + rng.normal(0, read_noise, M.shape)
    D = cdist(M_noisy, M, metric='euclidean')
    np.fill_diagonal(D, np.inf)
    nn_idx = np.argmin(D, axis=1)
    err = np.abs(locs[nn_idx] - locs)
    err = np.minimum(err, 1 - err)
    return np.mean(err) / 0.25  # normalize by chance level


def test_capacity(topology_strength, n_neurons, sparsity, noise_scale,
                   n_trials, memory_loads, rng, read_noise=READ_NOISE):
    """Test noisy-NN decoding at each memory load."""
    errors = []
    mantels = []

    for n_mem in memory_loads:
        trial_errors = []
        trial_mantels = []

        for t in range(n_trials):
            seed = rng.randint(0, 1000000)
            trial_rng = np.random.RandomState(seed)
            M, locs = generate_code(n_mem, n_neurons, sparsity, trial_rng,
                                     topology_strength=topology_strength,
                                     noise_scale=noise_scale)

            # Noisy NN error
            err = noisy_nn_error(M, locs, trial_rng, read_noise)
            trial_errors.append(err)

            # Mantel r
            if n_mem >= 10:
                phys = pdist(locs.reshape(-1, 1))
                neural = pdist(M, metric='euclidean')
                valid = np.isfinite(phys) & np.isfinite(neural)
                if np.sum(valid) > 10:
                    r, _ = spearmanr(phys[valid], neural[valid])
                    trial_mantels.append(r)

        errors.append(np.mean(trial_errors))
        mantels.append(np.mean(trial_mantels) if trial_mantels else 0.0)

    return np.array(errors), np.array(mantels)


def find_critical_capacity(errors, memory_loads, threshold):
    """Memory load at which error exceeds threshold."""
    for i, (err, load) in enumerate(zip(errors, memory_loads)):
        if err > threshold:
            if i > 0:
                prev_err = errors[i-1]
                prev_load = memory_loads[i-1]
                frac = (threshold - prev_err) / (err - prev_err)
                return prev_load + frac * (load - prev_load)
            return load
    return memory_loads[-1]


def topology_sensitivity(noise_scale, n_neurons, sparsity, n_trials, rng,
                          n_mem=100, read_noise=READ_NOISE,
                          topology_levels=None):
    """Sweep topology at fixed noise and memory load."""
    if topology_levels is None:
        topology_levels = np.linspace(0, 1, 11)

    errors = []
    mantels = []

    for topo in topology_levels:
        trial_errs = []
        trial_mants = []

        for _ in range(n_trials):
            trial_rng = np.random.RandomState(rng.randint(0, 1000000))
            M, locs = generate_code(n_mem, n_neurons, sparsity, trial_rng,
                                     topology_strength=topo,
                                     noise_scale=noise_scale)

            err = noisy_nn_error(M, locs, trial_rng, read_noise)
            trial_errs.append(err)

            phys = pdist(locs.reshape(-1, 1))
            neural = pdist(M, metric='euclidean')
            valid = np.isfinite(phys) & np.isfinite(neural)
            if np.sum(valid) > 10:
                r, _ = spearmanr(phys[valid], neural[valid])
                trial_mants.append(r)

        errors.append(np.mean(trial_errs))
        mantels.append(np.mean(trial_mants))

    return topology_levels, np.array(errors), np.array(mantels)


# ══════════════════════════════════════════════════════════════════════
# PART 2: EMPIRICAL REDUNDANCY
# ══════════════════════════════════════════════════════════════════════

def load_aronov_data(path):
    df = pd.read_pickle(path)
    df_exc = df[df['cell_type'] == 'E'].copy()
    df_t = df_exc[df_exc['species'] == 'titmouse']
    df_z = df_exc[df_exc['species'] == 'zebra_finch']
    return df_t, df_z


def get_session_maps(df_subset, min_neurons=MIN_NEURONS):
    sessions = {}
    for sess in df_subset['session'].unique():
        sdf = df_subset[df_subset['session'] == sess]
        maps, infos = [], []
        for _, row in sdf.iterrows():
            m = row['map']
            if isinstance(m, np.ndarray) and m.size == N_BINS:
                maps.append(np.nan_to_num(m.flatten(), nan=0.0))
                infos.append(row.get('info', 0))
        if len(maps) >= min_neurons:
            sessions[sess] = {
                'M': np.vstack(maps),
                'n_neurons': len(maps),
                'single_cell_info': np.array(infos),
            }
    return sessions


def compute_population_info(M, n_cv=5):
    n_neurons, n_bins = M.shape
    means = M.mean(axis=1, keepdims=True)
    stds = M.std(axis=1, keepdims=True)
    stds[stds == 0] = 1
    M_z = (M - means) / stds

    active = np.sum(M > 0, axis=0) >= max(2, n_neurons // 3)
    active_idx = np.where(active)[0]
    if len(active_idx) < 30:
        return np.nan, np.nan

    rows = active_idx // GRID_SIZE
    cols = active_idx % GRID_SIZE
    X = M_z[:, active_idx].T

    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    try:
        s_row = cross_val_score(ridge, X, rows,
                                  cv=min(n_cv, len(active_idx)),
                                  scoring='r2')
        s_col = cross_val_score(ridge, X, cols,
                                  cv=min(n_cv, len(active_idx)),
                                  scoring='r2')
        r2 = (max(np.mean(s_row), 0) + max(np.mean(s_col), 0)) / 2
    except Exception:
        return np.nan, np.nan

    if 0 < r2 < 1:
        bits = -0.5 * np.log2(1 - r2)
    else:
        bits = 0
    return r2, bits


def compute_redundancy(M, single_cell_info):
    _, pop_bits = compute_population_info(M)
    if np.isnan(pop_bits) or pop_bits <= 0:
        return np.nan
    sum_individual = np.sum(single_cell_info[single_cell_info > 0])
    return sum_individual / pop_bits if pop_bits > 0 else np.nan


def bootstrap_ipop_ci(M, n_boot=1000, rng=None):
    """
    Bootstrap CI for population information (Ipop) in bits.

    Resamples neurons with replacement, recomputes Ipop each time,
    and returns the point estimate with 95% CI.
    """
    if rng is None:
        rng = np.random.RandomState(320)

    _, ipop_obs = compute_population_info(M)
    if np.isnan(ipop_obs):
        return np.nan, np.nan, np.nan

    n_neurons = M.shape[0]
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.choice(n_neurons, n_neurons, replace=True)
        _, bits = compute_population_info(M[idx])
        if not np.isnan(bits):
            boot_vals.append(bits)

    if len(boot_vals) < 10:
        return ipop_obs, np.nan, np.nan

    return ipop_obs, np.percentile(boot_vals, 2.5), np.percentile(boot_vals, 97.5)


def run_redundancy_analysis():
    I_POP_FLOOR = 0.05  # bits — sessions below this excluded from filtered results

    path = Path(DATA_PATH)
    if not path.exists():
        print(f"\n  [SKIP] Data not found: {DATA_PATH}")
        return None, None, None, None

    df_t, df_z = load_aronov_data(path)
    print(f"\n  Excitatory cells: {len(df_t)} chickadee, {len(df_z)} finch")

    sessions_t = get_session_maps(df_t)
    sessions_z = get_session_maps(df_z)
    print(f"  Sessions (>={MIN_NEURONS} neurons): "
          f"{len(sessions_t)} chickadee, {len(sessions_z)} finch")

    red_t, red_z = [], []
    red_t_filtered, red_z_filtered = [], []
    ipop_rows = []
    excluded_t, excluded_z = 0, 0

    for sess, data in sessions_t.items():
        r = compute_redundancy(data['M'], data['single_cell_info'])
        ipop, ci_lo, ci_hi = bootstrap_ipop_ci(data['M'], n_boot=500)
        if not np.isnan(ipop):
            ipop_rows.append({
                'species': 'chickadee', 'session': sess,
                'n_neurons': data['n_neurons'],
                'Ipop_bits': ipop, 'ci_lower': ci_lo, 'ci_upper': ci_hi,
                'redundancy': r if not np.isnan(r) else np.nan,
            })
        if not np.isnan(r):
            red_t.append(r)
            _, pop_bits = compute_population_info(data['M'])
            if not np.isnan(pop_bits) and pop_bits >= I_POP_FLOOR:
                red_t_filtered.append(r)
            else:
                excluded_t += 1
    for sess, data in sessions_z.items():
        r = compute_redundancy(data['M'], data['single_cell_info'])
        ipop, ci_lo, ci_hi = bootstrap_ipop_ci(data['M'], n_boot=500)
        if not np.isnan(ipop):
            ipop_rows.append({
                'species': 'finch', 'session': sess,
                'n_neurons': data['n_neurons'],
                'Ipop_bits': ipop, 'ci_lower': ci_lo, 'ci_upper': ci_hi,
                'redundancy': r if not np.isnan(r) else np.nan,
            })
        if not np.isnan(r):
            red_z.append(r)
            _, pop_bits = compute_population_info(data['M'])
            if not np.isnan(pop_bits) and pop_bits >= I_POP_FLOOR:
                red_z_filtered.append(r)
            else:
                excluded_z += 1

    if red_t:
        print(f"  Chickadee redundancy: mean={np.mean(red_t):.1f} "
              f"(n={len(red_t)})")
    if red_z:
        print(f"  Finch redundancy:     mean={np.mean(red_z):.1f} "
              f"(n={len(red_z)})")
    if red_t and red_z:
        U, p = mannwhitneyu(red_t, red_z, alternative='two-sided')
        print(f"  Mann-Whitney: p={p:.4e}")

    # Absolute Ipop reporting
    ipop_t = [r for r in ipop_rows if r['species'] == 'chickadee']
    ipop_z = [r for r in ipop_rows if r['species'] == 'finch']
    if ipop_t:
        vals = [r['Ipop_bits'] for r in ipop_t]
        print(f"\n  Ipop (absolute, bits):")
        print(f"    Chickadee: mean={np.mean(vals):.3f} bits "
              f"(n={len(vals)} sessions)")
        for r in ipop_t:
            print(f"      {r['session']}: {r['Ipop_bits']:.3f} "
                  f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]")
    if ipop_z:
        vals = [r['Ipop_bits'] for r in ipop_z]
        print(f"    Finch:     mean={np.mean(vals):.3f} bits "
              f"(n={len(vals)} sessions)")
        for r in ipop_z:
            print(f"      {r['session']}: {r['Ipop_bits']:.3f} "
                  f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]")

    print(f"\n  I_pop floor = {I_POP_FLOOR} bits:")
    print(f"    Excluded: {excluded_t} chickadee, {excluded_z} finch sessions")
    if red_t_filtered:
        print(f"    Chickadee (filtered): mean={np.mean(red_t_filtered):.1f} "
              f"(n={len(red_t_filtered)})")
    if red_z_filtered:
        print(f"    Finch (filtered):     mean={np.mean(red_z_filtered):.1f} "
              f"(n={len(red_z_filtered)})")
    if red_t_filtered and red_z_filtered:
        U_f, p_f = mannwhitneyu(red_t_filtered, red_z_filtered,
                                alternative='two-sided')
        print(f"    Mann-Whitney (filtered): p={p_f:.4e}")

    # Export per-session Ipop with bootstrap CIs
    if ipop_rows:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        pd.DataFrame(ipop_rows).to_csv(
            os.path.join(OUTPUT_DIR, 'tier3_ipop_absolute.csv'), index=False)
        print(f"  Saved tier3_ipop_absolute.csv ({len(ipop_rows)} rows)")

    return red_t, red_z, red_t_filtered, red_z_filtered


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def run_all():
    rng = np.random.RandomState(320)

    print("=" * 60)
    print("TIER 3: CAPACITY + REDUNDANCY")
    print(f"N={N_NEURONS}, noise={NOISE_SCALE}, read_noise={READ_NOISE}")
    print("=" * 60)

    # ── Verify Mantel r values ──
    r_crystal = measure_mantel(100, N_NEURONS, SPARSITY,
                                TOPO_CRYSTAL, NOISE_SCALE)
    r_mist = measure_mantel(100, N_NEURONS, SPARSITY,
                              TOPO_MIST, NOISE_SCALE)
    r_noise = measure_mantel(100, N_NEURONS, SPARSITY,
                               TOPO_NOISE, NOISE_SCALE)

    print(f"\n  Crystal: topology={TOPO_CRYSTAL} -> r={r_crystal:.3f}")
    print(f"  Mist:    topology={TOPO_MIST} -> r={r_mist:.3f}")
    print(f"  Noise:   topology={TOPO_NOISE} -> r={r_noise:.3f}")

    # ── Capacity curves ──
    print(f"\nTesting Crystal...")
    crys_err, crys_mant = test_capacity(
        TOPO_CRYSTAL, N_NEURONS, SPARSITY, NOISE_SCALE,
        N_TRIALS, MEMORY_LOADS, rng)

    print(f"Testing Mist...")
    mist_err, mist_mant = test_capacity(
        TOPO_MIST, N_NEURONS, SPARSITY, NOISE_SCALE,
        N_TRIALS, MEMORY_LOADS, rng)

    print(f"Testing Noise...")
    noise_err, noise_mant = test_capacity(
        TOPO_NOISE, N_NEURONS, SPARSITY, NOISE_SCALE,
        N_TRIALS, MEMORY_LOADS, rng)

    # Use error threshold = midpoint between Crystal and Noise at 100 locs
    idx_100 = MEMORY_LOADS.index(100) if 100 in MEMORY_LOADS else 3
    err_thresh = (crys_err[idx_100] + noise_err[idx_100]) / 2
    print(f"\n  Error threshold (auto): {err_thresh:.4f}")

    crys_cap = find_critical_capacity(crys_err, MEMORY_LOADS, err_thresh)
    mist_cap = find_critical_capacity(mist_err, MEMORY_LOADS, err_thresh)
    noise_cap = find_critical_capacity(noise_err, MEMORY_LOADS, err_thresh)

    print(f"\n  Critical capacity (error > {err_thresh:.3f}):")
    print(f"    Crystal: {crys_cap:.0f}")
    print(f"    Mist:    {mist_cap:.0f}")
    print(f"    Noise:   {noise_cap:.0f}")

    for i, load in enumerate(MEMORY_LOADS):
        print(f"  N={load:5d}: Crystal={crys_err[i]:.4f}  "
              f"Mist={mist_err[i]:.4f}  Noise={noise_err[i]:.4f}")

    # ── Topology sensitivity ──
    print("\nTopology sensitivity...")
    topo_levels, topo_errors, topo_mantels = topology_sensitivity(
        NOISE_SCALE, N_NEURONS, SPARSITY, N_TRIALS, rng, n_mem=100)

    for tl, te, tm in zip(topo_levels, topo_errors, topo_mantels):
        print(f"  topo={tl:.1f} -> err={te:.4f}, r={tm:.3f}")

    # ── Redundancy ──
    print("\n" + "-" * 60)
    print("EMPIRICAL REDUNDANCY")
    print("-" * 60)
    red_t, red_z, red_t_filt, red_z_filt = run_redundancy_analysis()

    # ────────────────────────────────────────────────────────────────
    # PLOTTING: 2 rows x 4 cols
    # ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(2, 4, hspace=0.42, wspace=0.35,
                           left=0.05, right=0.97, top=0.89, bottom=0.08)

    # ── A: Capacity curves (noisy NN) ──
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(MEMORY_LOADS, crys_err, 'o-', color=C_CRYSTAL, lw=2,
               markersize=6, label='Crystal (topo=1.0)')
    ax_a.plot(MEMORY_LOADS, mist_err, 's-', color=C_MIST, lw=2,
               markersize=6, label='Mist (topo=0.5)')
    ax_a.plot(MEMORY_LOADS, noise_err, '^-', color=C_NOISE, lw=1.5,
               markersize=5, label='Noise (topo=0.0)')
    ax_a.set_xscale('log')
    ax_a.set_xlabel('Number of stored locations', fontsize=10)
    ax_a.set_ylabel('Mean spatial error (noisy NN)', fontsize=10)
    ax_a.set_title('A. Noisy nearest-neighbor decoding',
                     fontsize=11, fontweight='bold', loc='left')
    ax_a.legend(fontsize=7, frameon=False)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # ── B: Mantel r at each load ──
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(MEMORY_LOADS, crys_mant, 'o-', color=C_CRYSTAL, lw=2,
               markersize=6, label='Crystal')
    ax_b.plot(MEMORY_LOADS, mist_mant, 's-', color=C_MIST, lw=2,
               markersize=6, label='Mist')
    ax_b.plot(MEMORY_LOADS, noise_mant, '^-', color=C_NOISE, lw=1.5,
               markersize=5, label='Noise')
    ax_b.set_xscale('log')
    ax_b.set_xlabel('Number of stored locations', fontsize=10)
    ax_b.set_ylabel('Mantel r (topology fidelity)', fontsize=10)
    ax_b.set_title('B. Topology preservation vs load',
                     fontsize=11, fontweight='bold', loc='left')
    ax_b.legend(fontsize=7, frameon=False)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # ── C: Topology sensitivity (Mantel r vs error) ──
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.plot(topo_mantels, topo_errors, 'o-', color='black', lw=2,
               markersize=6)
    ax_c.axvline(r_crystal, color=C_CRYSTAL, ls='--', lw=1.5, alpha=0.6)
    ax_c.axvline(r_mist, color=C_MIST, ls='--', lw=1.5, alpha=0.6)
    ylims = ax_c.get_ylim()
    y_lab = ylims[1] - 0.05 * (ylims[1] - ylims[0])
    ax_c.text(r_crystal, y_lab, 'Crystal', fontsize=8,
               color=C_CRYSTAL, ha='center')
    ax_c.text(r_mist, y_lab, 'Mist', fontsize=8,
               color=C_MIST, ha='center')
    ax_c.set_xlabel('Mantel r (topology fidelity)', fontsize=10)
    ax_c.set_ylabel('Mean spatial error (100 locs)', fontsize=10)
    ax_c.set_title('C. Topology predicts accuracy',
                     fontsize=11, fontweight='bold', loc='left')
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # ── D: Error vs topology strength ──
    ax_d = fig.add_subplot(gs[0, 3])
    ax_d.plot(topo_levels, topo_errors, 'o-', color='black', lw=2,
               markersize=6)
    ax_d.axvline(TOPO_CRYSTAL, color=C_CRYSTAL, ls='--', lw=1.5, alpha=0.6,
                  label='Crystal')
    ax_d.axvline(TOPO_MIST, color=C_MIST, ls='--', lw=1.5, alpha=0.6,
                  label='Mist')
    ax_d.set_xlabel('Topology strength', fontsize=10)
    ax_d.set_ylabel('Mean spatial error (100 locs)', fontsize=10)
    ax_d.set_title('D. Topology controls accuracy',
                     fontsize=11, fontweight='bold', loc='left')
    ax_d.legend(fontsize=7, frameon=False, loc='upper right')
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    # ── E: Crystal RDM ──
    ax_e = fig.add_subplot(gs[1, 0])
    trial_rng = np.random.RandomState(0)
    M_vis, _ = generate_code(50, 30, 0.15, trial_rng,
                               topology_strength=TOPO_CRYSTAL,
                               noise_scale=NOISE_SCALE)
    neural_d = squareform(pdist(M_vis, metric='euclidean'))
    ax_e.imshow(neural_d, cmap='viridis', aspect='auto')
    ax_e.set_title('E. Crystal RDM (topo=1.0)',
                     fontsize=11, fontweight='bold', loc='left')
    ax_e.set_xlabel('Location', fontsize=10)
    ax_e.set_ylabel('Location', fontsize=10)

    # ── F: Mist RDM ──
    ax_f = fig.add_subplot(gs[1, 1])
    trial_rng = np.random.RandomState(0)
    M_vis2, _ = generate_code(50, 30, 0.15, trial_rng,
                                topology_strength=TOPO_MIST,
                                noise_scale=NOISE_SCALE)
    neural_d2 = squareform(pdist(M_vis2, metric='euclidean'))
    ax_f.imshow(neural_d2, cmap='viridis', aspect='auto')
    ax_f.set_title('F. Mist RDM (topo=0.5)',
                     fontsize=11, fontweight='bold', loc='left')
    ax_f.set_xlabel('Location', fontsize=10)
    ax_f.set_ylabel('Location', fontsize=10)

    # ── G: Empirical redundancy ──
    ax_g = fig.add_subplot(gs[1, 2])
    if red_t is not None and red_z is not None and red_t and red_z:
        bp = ax_g.boxplot([red_t, red_z], positions=[1, 2], widths=0.5,
                           patch_artist=True, showfliers=True,
                           flierprops=dict(markersize=3))
        bp['boxes'][0].set_facecolor(C_CRYSTAL)
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor(C_MIST)
        bp['boxes'][1].set_alpha(0.6)
        for line in bp['medians']:
            line.set_color('black')
            line.set_linewidth(1.5)
        ax_g.set_xticks([1, 2])
        ax_g.set_xticklabels(['Chickadee', 'Finch'])
        ax_g.axhline(1, color='gray', ls=':', lw=1, label='Independence')

        U, p = mannwhitneyu(red_t, red_z, alternative='two-sided')
        ax_g.set_title(f'G. Empirical redundancy\n'
                        f'{np.mean(red_t):.0f}x vs {np.mean(red_z):.0f}x, '
                        f'p={p:.2e}',
                        fontsize=11, fontweight='bold', loc='left')
        ax_g.set_ylabel('sum(individual) / population bits', fontsize=10)
        ax_g.legend(fontsize=7, frameon=False, loc='upper right')
    else:
        ax_g.text(0.5, 0.5, 'Data not available',
                   ha='center', va='center', fontsize=12,
                   transform=ax_g.transAxes, color='gray')
        ax_g.set_title('G. Empirical redundancy',
                        fontsize=11, fontweight='bold', loc='left')
    ax_g.spines['top'].set_visible(False)
    ax_g.spines['right'].set_visible(False)

    # ── H: Summary ──
    ax_h = fig.add_subplot(gs[1, 3])
    ax_h.axis('off')
    lines = [
        'Simulation',
        f'  {N_NEURONS} neurons, noise={NOISE_SCALE}',
        f'  read noise={READ_NOISE}',
        '',
        'Mantel r',
        f'  Crystal: {r_crystal:.3f}',
        f'  Mist:    {r_mist:.3f}',
        f'  Noise:   {r_noise:.3f}',
        '',
        'Error at 100 locations',
        f'  Crystal: {crys_err[MEMORY_LOADS.index(100)]:.4f}',
        f'  Mist:    {mist_err[MEMORY_LOADS.index(100)]:.4f}',
        f'  Noise:   {noise_err[MEMORY_LOADS.index(100)]:.4f}',
        '',
        'Error at 500 locations',
        f'  Crystal: {crys_err[MEMORY_LOADS.index(500)]:.4f}',
        f'  Mist:    {mist_err[MEMORY_LOADS.index(500)]:.4f}',
        f'  Noise:   {noise_err[MEMORY_LOADS.index(500)]:.4f}',
    ]
    if red_t is not None and red_z is not None:
        lines += [
            '',
            'Empirical redundancy',
            f'  Chickadee: {np.mean(red_t):.0f}x (n={len(red_t)})',
            f'  Finch:     {np.mean(red_z):.0f}x (n={len(red_z)})',
        ]
        if red_t_filt and red_z_filt:
            lines += [
                f'  (I_pop \u2265 0.05 floor)',
                f'  Chickadee: {np.mean(red_t_filt):.0f}x (n={len(red_t_filt)})',
                f'  Finch:     {np.mean(red_z_filt):.0f}x (n={len(red_z_filt)})',
            ]
    ax_h.text(0.05, 0.95, '\n'.join(lines),
               transform=ax_h.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#f0f0f0',
                          alpha=0.8))
    ax_h.set_title('H. Summary', fontsize=11,
                     fontweight='bold', loc='left')

    fig.suptitle(
        'Functional Consequences: Topology Determines Noise-Robust '
        'Memory Capacity\n'
        f'{N_NEURONS} neurons, noise={NOISE_SCALE}, '
        f'read noise={READ_NOISE}  |  '
        'Crystal/Mist ratio matches empirical 2:1',
        fontsize=13, fontweight='bold', y=0.97)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, 'tier3_capacity_redundancy.png')
    fig.savefig(outpath, dpi=250, facecolor='white')
    print(f"\nFigure saved: {outpath}")

    # ── Export ──
    rows = []
    for i, load in enumerate(MEMORY_LOADS):
        rows.append({'model': 'crystal', 'topology': TOPO_CRYSTAL,
                      'memory_load': load,
                      'error': crys_err[i], 'mantel_r': crys_mant[i]})
        rows.append({'model': 'mist', 'topology': TOPO_MIST,
                      'memory_load': load,
                      'error': mist_err[i], 'mantel_r': mist_mant[i]})
        rows.append({'model': 'noise', 'topology': TOPO_NOISE,
                      'memory_load': load,
                      'error': noise_err[i], 'mantel_r': noise_mant[i]})
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, 'tier3_capacity_results.csv'), index=False)
    print("Saved tier3_capacity_results.csv")

    if red_t is not None and red_z is not None:
        red_rows = ([{'species': 'chickadee', 'redundancy': r}
                     for r in red_t] +
                    [{'species': 'finch', 'redundancy': r}
                     for r in red_z])
        pd.DataFrame(red_rows).to_csv(
            os.path.join(OUTPUT_DIR, 'tier3_redundancy_results.csv'), index=False)
        print("Saved tier3_redundancy_results.csv")

        if red_t_filt or red_z_filt:
            filt_rows = ([{'species': 'chickadee', 'redundancy': r}
                          for r in (red_t_filt or [])] +
                         [{'species': 'finch', 'redundancy': r}
                          for r in (red_z_filt or [])])
            pd.DataFrame(filt_rows).to_csv(
                os.path.join(OUTPUT_DIR, 'tier3_redundancy_filtered.csv'), index=False)
            print("Saved tier3_redundancy_filtered.csv")

    plt.show()


if __name__ == '__main__':
    run_all()