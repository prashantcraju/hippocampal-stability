#!/usr/bin/env python3
"""
fig_valiant_2x2.py — Supplementary Figure: Valiant SMA and Double Dissociation

Panels (2x2 Layout):
  Top-Left (A):     Place Field Size Heterogeneity (CV)
  Top-Right (B):    Split-Half Allocation Reliability
  Bottom-Left (C):  Double Dissociation Scatter (Shesha vs Allocation)
  Bottom-Right (D): A-P Gradient of Geometric Stability (Chickadee only)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu, spearmanr, sem
from pathlib import Path

# ── I/O Paths ──
DATA_DIR   = Path('output/tier1_valiant')
OUTPUT_DIR = Path('output/valiant')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Aesthetics ──
C_CHICK = '#C62D50'
C_FINCH = '#4A90C4'

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
def _boxstrip(ax, vals_a, vals_b, col_a, col_b, ylabel, title, rng_j, alt_test='two-sided'):
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
    ax.set_xticklabels(['Chickadee', 'Finch'], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', loc='left', pad=12)
    
    if len(vals_a) >= 3 and len(vals_b) >= 3:
        _, p = mannwhitneyu(vals_a, vals_b, alternative=alt_test)
        ymax = max(max(vals_a, default=0), max(vals_b, default=0))
        ymin = min(min(vals_a, default=0), min(vals_b, default=0))
        y_range = ymax - ymin
        y_bar = ymax + (y_range * 0.08)
        
        ax.plot([1, 2], [y_bar, y_bar], color='#333333', lw=1.5)
        ax.text(1.5, y_bar + (y_range * 0.02), f'$p = {p:.3f}$', ha='center', va='bottom', fontsize=10)
        ax.set_ylim(bottom=ymin - (y_range * 0.05), top=y_bar + (y_range * 0.15))

def main():
    # 1. Load Data
    sess_df = pd.read_csv(DATA_DIR / 'tier1_session_results.csv')
    chick_df = sess_df[sess_df['species'] == 'chickadee']
    finch_df = sess_df[sess_df['species'] == 'finch']
    
    ap_path = DATA_DIR / 'tier1_ap_gradient.csv'
    ap_df = pd.read_csv(ap_path) if ap_path.exists() else None

    # 2. Setup 2x2 Figure
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    gs = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.35, 
                           left=0.08, right=0.96, top=0.92, bottom=0.08)
    
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    
    rng = np.random.RandomState(42)

    # ── A: Place Field Size Heterogeneity (CV) ──
    _boxstrip(ax_a, chick_df['field_size_cv'], finch_df['field_size_cv'], 
              C_CHICK, C_FINCH, 'Coefficient of Variation (CV)', 
              'A. Place Field Size Heterogeneity', rng, alt_test='two-sided')

    # ── B: Split-half Allocation Reliability ──
    _boxstrip(ax_b, chick_df['split_half_r'], finch_df['split_half_r'], 
              C_CHICK, C_FINCH, 'Allocation Reliability (Split-half $r$)', 
              'B. Discrete Memory Allocation', rng, alt_test='less')
    ax_b.axhline(0, color='#AAAAAA', ls='--', lw=1, zorder=0)

    # ── C: Double Dissociation Scatter ──
    c_valid = chick_df.dropna(subset=['shesha', 'split_half_r'])
    f_valid = finch_df.dropna(subset=['shesha', 'split_half_r'])
    
    ax_c.scatter(c_valid['split_half_r'], c_valid['shesha'], color=C_CHICK, 
                 alpha=0.7, s=55, edgecolors='white', linewidths=0.8, label='Chickadee')
    ax_c.scatter(f_valid['split_half_r'], f_valid['shesha'], color=C_FINCH, 
                 alpha=0.7, s=55, edgecolors='white', linewidths=0.8, label='Finch')
    
    for df_sub, col in [(c_valid, C_CHICK), (f_valid, C_FINCH)]:
        if len(df_sub) > 2:
            m, b = np.polyfit(df_sub['split_half_r'], df_sub['shesha'], 1)
            x_line = np.linspace(df_sub['split_half_r'].min() - 0.05, df_sub['split_half_r'].max() + 0.05, 100)
            ax_c.plot(x_line, m * x_line + b, color=col, lw=2.5, ls='--', alpha=0.8)

    ax_c.axvline(0, color='#AAAAAA', ls='--', lw=1.2, zorder=0)
    ax_c.set_xlabel('Discrete Allocation Reliability (Split-half $r$)', fontsize=11)
    ax_c.set_ylabel('Geometric Rigidity (Shesha)', fontsize=11)
    ax_c.set_title('C. The Double Dissociation', fontsize=13, fontweight='bold', loc='left', pad=12)
    ax_c.legend(fontsize=10, frameon=False, loc='upper right')

    # ── D: A-P Gradient (Chickadee Only) ──
    if ap_df is not None and len(ap_df) > 0:
        subdivs = sorted(ap_df['subdivision'].unique())
        means = [ap_df[ap_df['subdivision'] == s]['shesha'].mean() for s in subdivs]
        sems = [ap_df[ap_df['subdivision'] == s]['shesha'].sem() for s in subdivs]
        ns = [len(ap_df[ap_df['subdivision'] == s]) for s in subdivs]
        
        x_pos = np.arange(len(subdivs))
        ax_d.bar(x_pos, means, yerr=sems, color=C_CHICK, alpha=0.75, width=0.55,
                 capsize=5, edgecolor='white', linewidth=1.5, 
                 error_kw=dict(elinewidth=1.5, ecolor='#555555'))
        
        ax_d.set_xticks(x_pos)
        ax_d.set_xticklabels([f'{s}\n($n={n}$)' for s, n in zip(subdivs, ns)], fontsize=11)
        ax_d.set_ylabel('Geometric Rigidity (Shesha)', fontsize=11)
        ax_d.set_title('D. A-P Anatomical Gradient\n(Chickadee)', fontsize=13, fontweight='bold', loc='left', pad=12)
    else:
        ax_d.text(0.5, 0.5, 'A-P Data Not Found', ha='center', va='center')
        ax_d.set_axis_off()

    # ── Save ──
    out_pdf = OUTPUT_DIR / 'figS2_valiant_dissociation_2x2.pdf'
    out_png = OUTPUT_DIR / 'figS2_valiant_dissociation_2x2.png'
    fig.savefig(out_pdf, dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(out_png, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Saved: {out_pdf}")
    plt.show()

if __name__ == '__main__':
    main()