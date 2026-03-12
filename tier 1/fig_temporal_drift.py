#!/usr/bin/env python3
"""
fig_temporal_drift.py — Extended Temporal Drift Analysis

Stress-tests the core finding that topological (SHESHA) geometric
stability dissociates from temporal representational drift.  Uses the
same population-code simulation engine as tier3 to model extended
epochs of neural activity where the code drifts via cumulative noise.

Model
-----
At each epoch the population code matrix M is perturbed by additive
Gaussian noise (representing synaptic turnover, slow gain changes,
etc.).  Temporal drift is the cosine distance between the current
population vector and the epoch-0 template, averaged over locations.
SHESHA (split-half RDM correlation) is computed on the drifted code
at every epoch to track geometric stability.

If the geometry is topologically protected (crystal), SHESHA should
remain high even as raw drift accumulates — the core dissociation.

Panels (2 rows x 3 cols)
-------
A — Temporal drift trajectories across 200 epochs, one line per
    noise level, for the Crystal code (tau = 1.0).  SE bands over seeds.
B — Same layout for Mist code (tau = 0.5).
C — Same layout for Noise / random code (tau = 0.0).

D — SHESHA (geometric stability) vs epoch under the same noise
    levels for Crystal.  Demonstrates persistence of geometry
    despite rising drift.
E — Same for Mist.
F — Direct overlay: final-epoch SHESHA vs cumulative drift at that
    epoch, for all three codes and all noise levels.  The
    dissociation appears as a vertical spread (high SHESHA despite
    high drift) for Crystal vs. collapse for Noise.

Stand-alone: pure simulation, no data files required.

Usage:
    python fig_temporal_drift.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter1d
import warnings, os
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'output/temporal_drift'

# ── Simulation parameters ────────────────────────────────────────────
N_NEURONS       = 400
N_LOCATIONS     = 100
SPARSITY        = 0.15
NOISE_SCALE     = 0.20

N_EPOCHS        = 200
N_SEEDS         = 10

DRIFT_NOISE_LEVELS = [0.005, 0.01, 0.02, 0.04, 0.08]

TOPO_CRYSTAL    = 1.0
TOPO_MIST       = 0.5
TOPO_NOISE      = 0.0

# ── Colours — colorblind-safe, high-contrast ─────────────────────────
C_CRYSTAL = '#C62D50'   # chickadee red
C_MIST    = '#4A90C4'   # finch blue
C_NOISE   = '#9E9E9E'   # null grey

# Noise-level palette: 5 perceptually-ordered, colorblind-distinguishable
NOISE_COLORS = ['#FDBE85', '#FD8D3C', '#E6550D', '#A63603', '#7F2704']


# ── Code generation (same as tier3) ─────────────────────────────────
def generate_code(n_locations, n_neurons, sparsity, rng,
                  topology_strength=1.0, noise_scale=NOISE_SCALE):
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

    M_rand = np.zeros_like(M_topo)
    for j in range(n_neurons):
        M_rand[:, j] = rng.permutation(M_topo[:, j])

    n_topo = int(topology_strength * n_neurons)
    order  = rng.permutation(n_neurons)
    topo_idx, rand_idx = order[:n_topo], order[n_topo:]

    M = np.zeros_like(M_topo)
    M[:, topo_idx] = M_topo[:, topo_idx]
    M[:, rand_idx] = M_rand[:, rand_idx]

    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return M / norms, locations


# ── Metrics ──────────────────────────────────────────────────────────
def temporal_drift(M_current, M_template):
    """Mean cosine distance between current and template PVs."""
    dot   = np.sum(M_current * M_template, axis=1)
    n1    = np.linalg.norm(M_current, axis=1)
    n2    = np.linalg.norm(M_template, axis=1)
    denom = n1 * n2
    denom[denom == 0] = 1.0
    return np.mean(1 - dot / denom)


def shesha_proxy(M):
    """
    Fast split-half RDM correlation (Spearman) — proxy for the full
    SHESHA score.
    """
    n_neurons = M.shape[1]
    if n_neurons < 4:
        return np.nan

    M_z = M - M.mean(axis=0, keepdims=True)
    std = M.std(axis=0, keepdims=True)
    std[std == 0] = 1
    M_z = M_z / std

    rng_split = np.random.RandomState(320)
    perm = rng_split.permutation(n_neurons)
    half = n_neurons // 2
    rdm1 = pdist(M_z[:, perm[:half]], metric='euclidean')
    rdm2 = pdist(M_z[:, perm[half:2*half]], metric='euclidean')

    valid = np.isfinite(rdm1) & np.isfinite(rdm2)
    if np.sum(valid) < 20:
        return np.nan
    r, _ = spearmanr(rdm1[valid], rdm2[valid])
    return r


# ── Simulation engine ────────────────────────────────────────────────
def simulate_drift(topology_strength, drift_noise, seed):
    rng = np.random.RandomState(seed)
    M0, locs = generate_code(N_LOCATIONS, N_NEURONS, SPARSITY, rng,
                              topology_strength=topology_strength)
    M_template = M0.copy()
    M_current  = M0.copy()

    drifts  = np.zeros(N_EPOCHS)
    sheshas = np.zeros(N_EPOCHS)

    for ep in range(N_EPOCHS):
        perturbation = rng.normal(0, drift_noise, M_current.shape)
        M_current = M_current + perturbation
        M_current = np.maximum(M_current, 0)
        norms = np.linalg.norm(M_current, axis=1, keepdims=True)
        norms[norms == 0] = 1
        M_current = M_current / norms

        drifts[ep]  = temporal_drift(M_current, M_template)
        sheshas[ep] = shesha_proxy(M_current)

    return drifts, sheshas


def run_all_simulations():
    """Sweep topology x noise x seed.  Returns raw per-seed arrays."""
    configs = [
        ('Crystal', TOPO_CRYSTAL),
        ('Mist',    TOPO_MIST),
        ('Noise',   TOPO_NOISE),
    ]

    results = {}
    total = len(configs) * len(DRIFT_NOISE_LEVELS) * N_SEEDS
    done  = 0
    print(f"Running {total} drift trajectories "
          f"({N_EPOCHS} epochs each) ...")

    for label, topo in configs:
        results[label] = {}
        for noise in DRIFT_NOISE_LEVELS:
            drift_arr  = np.zeros((N_SEEDS, N_EPOCHS))
            shesha_arr = np.zeros((N_SEEDS, N_EPOCHS))
            for s in range(N_SEEDS):
                d, sh = simulate_drift(topo, noise, seed=320 + s)
                drift_arr[s]  = d
                shesha_arr[s] = sh
                done += 1
            results[label][noise] = {
                'drift_all':  drift_arr,
                'shesha_all': shesha_arr,
            }
        pct = 100 * done / total
        print(f"  {done}/{total}  ({pct:.0f}%)")

    return results


# ══════════════════════════════════════════════════════════════════════
# PLOTTING — camera-ready, Nature / PNAS house style
# ══════════════════════════════════════════════════════════════════════

def _noise_palette(n):
    """Return n colours.  Uses hand-picked list for <=5, else YlOrRd."""
    if n <= len(NOISE_COLORS):
        return NOISE_COLORS[:n]
    cmap = plt.cm.YlOrRd
    return [cmap(0.25 + 0.65 * i / max(n - 1, 1)) for i in range(n)]


def _se_band(arr, sigma_smooth=3):
    """Smoothed mean +/- SE."""
    mean = gaussian_filter1d(np.mean(arr, axis=0), sigma=sigma_smooth)
    se   = gaussian_filter1d(np.std(arr, axis=0) / np.sqrt(arr.shape[0]),
                             sigma=sigma_smooth)
    return mean, mean - se, mean + se


def _plot_lines(ax, results_for_model, y_key, y_label, panel_letter,
                subtitle):
    """Generic line + SE-band panel for drift or SHESHA."""
    epochs = np.arange(N_EPOCHS)
    noise_keys = sorted(results_for_model.keys())
    palette = _noise_palette(len(noise_keys))

    for li, noise in enumerate(noise_keys):
        arr = results_for_model[noise][y_key]
        mean, lo, hi = _se_band(arr)
        ax.plot(epochs, mean, color=palette[li], lw=2.2,
                label=f'$\\sigma_d = {noise}$')
        ax.fill_between(epochs, lo, hi, color=palette[li], alpha=0.20)

    ax.set_xlabel('Simulation epoch ($t$)')
    ax.set_ylabel(y_label)
    ax.set_title(f'{panel_letter}.  {subtitle}', fontweight='bold',
                 loc='left', pad=8)


def make_figure(results):
    # ── Global rcParams ──────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.sans-serif':   ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size':         10,
        'axes.labelsize':    11,
        'axes.titlesize':    11.5,
        'axes.linewidth':    0.8,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'legend.fontsize':   8.5,
        'legend.frameon':    False,
        'figure.dpi':        150,
        'savefig.dpi':       300,
        'savefig.facecolor': 'white',
    })

    model_keys   = list(results.keys())        # Crystal, Mist, Noise
    model_colors = [C_CRYSTAL, C_MIST, C_NOISE]
    model_labels = {
        'Crystal': r'Crystal ($\tau\!=\!1.0$, chickadee)',
        'Mist':    r'Mist ($\tau\!=\!0.5$, finch)',
        'Noise':   r'Noise ($\tau\!=\!0.0$, null)',
    }
    markers = ['o', 's', '^']

    fig = plt.figure(figsize=(17, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.44, wspace=0.34,
                            left=0.07, right=0.96, top=0.89, bottom=0.08)

    # ── Row 1: Temporal drift vs epoch ───────────────────────────────
    drift_axes = []
    for idx, key in enumerate(model_keys):
        ax = fig.add_subplot(gs[0, idx])
        drift_axes.append(ax)
        _plot_lines(ax, results[key], 'drift_all',
                    'Temporal drift (cosine dist.)',
                    chr(ord('A') + idx),
                    f'Drift \u2014 {model_labels[key]}')
        if idx == 0:
            ax.legend(title='Drift noise', loc='upper left')

    # Shared y-limits across drift panels
    drift_ymax = max(ax.get_ylim()[1] for ax in drift_axes)
    for ax in drift_axes:
        ax.set_ylim(bottom=0, top=drift_ymax * 1.02)

    # ── Row 2 cols 0-1: SHESHA (geometry) vs epoch ───────────────────
    shesha_axes = []
    for idx, key in enumerate(model_keys[:2]):
        ax = fig.add_subplot(gs[1, idx])
        shesha_axes.append(ax)
        _plot_lines(ax, results[key], 'shesha_all',
                    'SHESHA (Spearman $r$)',
                    chr(ord('D') + idx),
                    f'Geometric stability \u2014 {model_labels[key]}')
        if idx == 0:
            ax.legend(title='Drift noise', loc='lower left')

    # Shared y-limits across SHESHA panels
    shesha_ymin = min(ax.get_ylim()[0] for ax in shesha_axes)
    shesha_ymax = max(ax.get_ylim()[1] for ax in shesha_axes)
    for ax in shesha_axes:
        ax.set_ylim(shesha_ymin, shesha_ymax * 1.01)

    # ── Panel F: Dissociation scatter ────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])

    for mi, (key, mc, mk) in enumerate(
            zip(model_keys, model_colors, markers)):
        res = results[key]
        noise_keys = sorted(res.keys())

        # Per-seed points (pooled across noise levels)
        all_drift  = np.concatenate(
            [res[n]['drift_all'][:, -1] for n in noise_keys])
        all_shesha = np.concatenate(
            [res[n]['shesha_all'][:, -1] for n in noise_keys])
        ax_f.scatter(all_drift, all_shesha, color=mc, marker=mk,
                     s=20, alpha=0.30, edgecolors='none')

        # Mean trend line (one point per noise level)
        md = [np.mean(res[n]['drift_all'][:, -1]) for n in noise_keys]
        ms = [np.mean(res[n]['shesha_all'][:, -1]) for n in noise_keys]
        ax_f.plot(md, ms, color=mc, lw=2.5, marker=mk, ms=8,
                  mec='white', mew=0.7, zorder=5,
                  label=model_labels[key])

    ax_f.set_xlabel('Temporal drift (cosine dist., final epoch)')
    ax_f.set_ylabel('SHESHA (Spearman $r$, final epoch)')
    ax_f.set_title('F.  Dissociation: geometry vs. drift',
                    fontweight='bold', loc='left', pad=8)
    ax_f.legend(loc='lower left')

    ax_f.annotate(
        'Topological codes preserve\ngeometry despite large drift',
        xy=(0.97, 0.97), xycoords='axes fraction',
        ha='right', va='top', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.35', fc='#FFF8F0',
                  ec=C_CRYSTAL, lw=0.8, alpha=0.92))

    # ── Suptitle ─────────────────────────────────────────────────────
    fig.suptitle('Extended Temporal Drift Analysis',
                 fontsize=14, fontweight='bold', y=0.97)
    fig.text(0.5, 0.935,
             f'$N = {N_NEURONS}$ neurons, '
             f'${N_LOCATIONS}$ locations, '
             f'${N_EPOCHS}$ epochs, '
             f'${N_SEEDS}$ seeds / condition   |   '
             f'error bands = SEM',
             ha='center', fontsize=9.5, color='#555555')

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUTPUT_DIR, f'temporal_drift.{ext}'),
                    dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Saved to {OUTPUT_DIR}/temporal_drift.{{png,pdf}}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = run_all_simulations()
    make_figure(results)
    print("\nDone.")
