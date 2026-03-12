#!/usr/bin/env python3
"""
fig_controls_2x2.py — Supplementary Figure: Methodological Controls

Panels (2x2 Layout):
  Top-Left (A):     Neuron-Matched Downsampling
  Top-Right (B):    Spatial Permutation Controls (Shuffles)
  Bottom-Left (C):  Linear Metric: PV Correlation
  Bottom-Right (D): Geometric Metric: CCA Stability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu
from pathlib import Path

# ── I/O Paths ──
DATA_DIR   = Path('output/tier1_enhanced')
OUTPUT_DIR = Path('output/controls')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Aesthetics ──
C_CHICK  = '#C62D50'
C_FINCH  = '#4A90C4'
C_CTRL   = '#9E9E9E'  # Grey for controls/downsampled

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.linewidth'    : 0.8,
    'xtick.labelsize'   : 10,
    'ytick.labelsize'   : 10,
    'axes.labelsize'    : 11,
    'figure.dpi'        : 150,
    'savefig.dpi'       : 300,
})

# ── Helpers ──
def _boxstrip(ax, vals_a, vals_b, col_a, col_b, labels, ylabel, title, rng_j, alt_test='two-sided'):
    """Clean, modern boxplot with jittered scatter overlay"""
    vals_a = np.array(vals_a)[~np.isnan(vals_a)]
    vals_b = np.array(vals_b)[~np.isnan(vals_b)]
    
    bp = ax.boxplot([vals_a, vals_b], positions=[1, 2], widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.8),
                    whiskerprops=dict(linewidth=1.2, color='#555555'),
                    capprops=dict(linewidth=1.2, color='#555555'))
    
    bp['boxes'][0].set(facecolor=col_a, alpha=0.6, edgecolor='none')
    bp['boxes'][1].set(facecolor=col_b, alpha=0.6, edgecolor='none')
    
    for vals, xpos, col in [(vals_a, 1, col_a), (vals_b, 2, col_b)]:
        jitter = rng_j.uniform(-0.15, 0.15, len(vals))
        ax.scatter(xpos + jitter, vals, color=col, s=45, alpha=0.7,
                   zorder=5, edgecolors='white', linewidths=0.6)
                   
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', loc='left', pad=12)
    
    if len(vals_a) >= 3 and len(vals_b) >= 3:
        _, p = mannwhitneyu(vals_a, vals_b, alternative=alt_test)
        ymax = max(max(vals_a, default=0), max(vals_b, default=0))
        ymin = min(min(vals_a, default=0), min(vals_b, default=0))
        y_range = ymax - ymin
        y_bar = ymax + (y_range * 0.08)
        
        ax.plot([1, 2], [y_bar, y_bar], color='#333333', lw=1.5)
        
        # Format p-value nicely
        p_str = f'$p = {p:.3f}$' if p >= 0.001 else f'$p < 0.001$'
        ax.text(1.5, y_bar + (y_range * 0.02), p_str, ha='center', va='bottom', fontsize=10)
        ax.set_ylim(bottom=ymin - (y_range * 0.05), top=y_bar + (y_range * 0.15))

def main():
    # 1. Load Data
    main_df = pd.read_csv(DATA_DIR / 'tier1_main_results.csv')
    ctrl_df = pd.read_csv(DATA_DIR / 'tier1_controls.csv')
    alt_df  = pd.read_csv(DATA_DIR / 'tier1_alternative_metrics.csv')
    
    rng = np.random.RandomState(42)

    # 2. Setup 2x2 Figure
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    gs = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.35, 
                           left=0.08, right=0.96, top=0.92, bottom=0.08)
    
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ── A: Neuron-Matched Downsampling ──
    # Compare Chickadee Downsampled vs Finch Original
    chick_ds = ctrl_df[(ctrl_df['species'] == 'chickadee') & (ctrl_df['control'] == 'neuron_matched')]['value'].values
    finch_orig = main_df[main_df['species'] == 'finch']['shesha'].values
    
    _boxstrip(ax_a, chick_ds, finch_orig, C_CTRL, C_FINCH, 
              ['Chickadee\n(Downsampled)', 'Finch\n(Original)'], 
              'Geometric Rigidity (Shesha)', 
              'A. Neuron-Matched Control', rng, alt_test='greater')
    ax_a.axhline(0, color='#AAAAAA', ls='--', lw=1, zorder=0)

    # ── B: Spatial Permutation Controls ──
    # Bar chart showing Original -> Circular -> Map Shuffle
    c_orig = ctrl_df[(ctrl_df['species'] == 'chickadee') & (ctrl_df['control'] == 'original')]['value'].values
    c_circ = ctrl_df[(ctrl_df['species'] == 'chickadee') & (ctrl_df['control'] == 'circular')]['value'].values
    c_shuf = ctrl_df[(ctrl_df['species'] == 'chickadee') & (ctrl_df['control'] == 'map_shuffle')]['value'].values
    
    f_orig = ctrl_df[(ctrl_df['species'] == 'finch') & (ctrl_df['control'] == 'original')]['value'].values
    f_circ = ctrl_df[(ctrl_df['species'] == 'finch') & (ctrl_df['control'] == 'circular')]['value'].values
    f_shuf = ctrl_df[(ctrl_df['species'] == 'finch') & (ctrl_df['control'] == 'map_shuffle')]['value'].values

    # Means and SEMs
    c_means = [np.mean(c_orig), np.mean(c_circ), np.mean(c_shuf)]
    c_sems  = [np.std(c_orig)/np.sqrt(len(c_orig)), np.std(c_circ)/np.sqrt(len(c_circ)), np.std(c_shuf)/np.sqrt(len(c_shuf))]
    
    f_means = [np.mean(f_orig), np.mean(f_circ), np.mean(f_shuf)]
    f_sems  = [np.std(f_orig)/np.sqrt(len(f_orig)), np.std(f_circ)/np.sqrt(len(f_circ)), np.std(f_shuf)/np.sqrt(len(f_shuf))]

    x_pos = np.arange(3)
    width = 0.35
    
    ax_b.bar(x_pos - width/2, c_means, yerr=c_sems, width=width, color=C_CHICK, alpha=0.8, 
             label='Chickadee', capsize=4, edgecolor='white', linewidth=1.2)
    ax_b.bar(x_pos + width/2, f_means, yerr=f_sems, width=width, color=C_FINCH, alpha=0.8, 
             label='Finch', capsize=4, edgecolor='white', linewidth=1.2)
    
    ax_b.axhline(0, color='#AAAAAA', ls='--', lw=1, zorder=0)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(['Original\nData', 'Circular\nShift', 'Map\nShuffle'], fontsize=11)
    ax_b.set_ylabel('Geometric Rigidity (Shesha)', fontsize=11)
    ax_b.set_title('B. Spatial Permutation Controls', fontsize=13, fontweight='bold', loc='left', pad=12)
    ax_b.legend(fontsize=10, frameon=False)

    # ── C: PV Correlation (Linear Metric) ──
    # Note: alt_test='two-sided' because Finch is actually numerically higher here
    chick_pv = alt_df[alt_df['species'] == 'chickadee']['pv_corr'].values
    finch_pv = alt_df[alt_df['species'] == 'finch']['pv_corr'].values
    
    _boxstrip(ax_c, chick_pv, finch_pv, C_CHICK, C_FINCH, 
              ['Chickadee', 'Finch'], 
              'PV Correlation ($r$)', 
              'C. Linear Spatial Metric', rng, alt_test='two-sided')
    ax_c.axhline(0, color='#AAAAAA', ls='--', lw=1, zorder=0)

    # ── D: CCA Stability (Alternative Geometric Metric) ──
    chick_cca = alt_df[alt_df['species'] == 'chickadee']['cca_stability'].values
    finch_cca = alt_df[alt_df['species'] == 'finch']['cca_stability'].values
    
    _boxstrip(ax_d, chick_cca, finch_cca, C_CHICK, C_FINCH, 
              ['Chickadee', 'Finch'], 
              'Mean Canonical Correlation', 
              'D. Alternative Geometric Metric (CCA)', rng, alt_test='greater')
    ax_d.axhline(0, color='#AAAAAA', ls='--', lw=1, zorder=0)

    # ── Save ──
    out_pdf = OUTPUT_DIR / 'figS3_rigorous_controls.pdf'
    out_png = OUTPUT_DIR / 'figS3_rigorous_controls.png'
    fig.savefig(out_pdf, dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(out_png, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Saved: {out_pdf}")
    plt.show()

if __name__ == '__main__':
    main()