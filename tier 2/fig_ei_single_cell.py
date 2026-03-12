#!/usr/bin/env python3
"""
fig_ei_single_cell.py — Supplementary Figure 4: E-I Single-Cell Properties 
and Population Synergy.

Panels (2x2 Layout):
  Top-Left (A):     Spatial Information by Cell Type (Single cells)
  Top-Right (B):    Temporal Stability by Cell Type (Single cells)
  Bottom-Left (C):  Topogeographic Structure (Mantel r) by Cell Type (Sessions)
  Bottom-Right (D): The Knockout Test (Geometric Stability Bar Chart)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu, sem
from pathlib import Path

# ── I/O Paths ──
DATA_DIR   = Path('output/tier2_ei_session')
OUTPUT_DIR = Path('output/ei_single_cell')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Aesthetics ──
C_CHICK  = '#C62D50'
C_FINCH  = '#4A90C4'
C_E      = '#2E8B57'   # Excitatory Green
C_I      = '#FF8C00'   # Inhibitory Orange

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

def _grouped_boxplot(ax, data_dict, ylabel, title, sig_heights, show_scatter=False):
    """
    Plots a grouped boxplot for E vs I cells across species.
    sig_heights: list of two y-values for the significance bars [chick_y, finch_y]
    """
    positions = [1, 1.6, 3.2, 3.8]
    keys = ['chick_E', 'chick_I', 'finch_E', 'finch_I']
    colors = [C_E, C_I, C_E, C_I]
    
    plot_data = [np.array(data_dict[k])[~np.isnan(data_dict[k])] for k in keys]
    
    bp = ax.boxplot(plot_data, positions=positions, widths=0.4,
                    patch_artist=True, showfliers=not show_scatter,
                    flierprops=dict(marker='o', markersize=2, alpha=0.15, markeredgecolor='none', markerfacecolor='#555555'),
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1.2, color='#555555'),
                    capprops=dict(linewidth=1.2, color='#555555'))
    
    for i, patch in enumerate(bp['boxes']):
        patch.set(facecolor=colors[i], alpha=0.7, edgecolor='none')
        
    if show_scatter:
        rng = np.random.RandomState(42)
        for i, vals in enumerate(plot_data):
            jitter = rng.uniform(-0.1, 0.1, len(vals))
            ax.scatter(positions[i] + jitter, vals, color=colors[i], s=25, alpha=0.7,
                       zorder=5, edgecolors='white', linewidths=0.5)

    ax.set_xticks([1.3, 3.5])
    ax.set_xticklabels(['Chickadee', 'Finch'], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', loc='left', pad=12)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=C_E, lw=4, alpha=0.7),
                    Line2D([0], [0], color=C_I, lw=4, alpha=0.7)]
    ax.legend(custom_lines, ['Excitatory (E)', 'Inhibitory (I)'], frameon=False, loc='upper right', fontsize=9)
    
    # Explicit Significance Bars
    for i, (pair, x_pos) in enumerate([((plot_data[0], plot_data[1]), 1.3), ((plot_data[2], plot_data[3]), 3.5)]):
        if len(pair[0]) > 3 and len(pair[1]) > 3:
            _, p = mannwhitneyu(pair[0], pair[1], alternative='two-sided')
            y_bar = sig_heights[i]
            
            ax.plot([x_pos - 0.3, x_pos + 0.3], [y_bar, y_bar], color='#333333', lw=1.2)
            
            p_str = f'$p < 0.001$' if p < 0.001 else f'$p = {p:.3f}$'
            if p > 0.05: p_str = 'n.s.'
            
            # Subtle pad above the bar
            ax.text(x_pos, y_bar + (y_bar * 0.02), p_str, ha='center', va='bottom', fontsize=10)

def main():
    # 1. Load Data
    cell_df = pd.read_csv(DATA_DIR / 'tier2_cell_info_xcorr.csv')
    mantel_df = pd.read_csv(DATA_DIR / 'tier2_session_mantel.csv')
    ko_df = pd.read_csv(DATA_DIR / 'tier2_knockout_paired.csv')

    # 2. Setup Figure
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    gs = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.35, 
                           left=0.08, right=0.96, top=0.92, bottom=0.08)
    
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ── A: Spatial Information ──
    info_data = {
        'chick_E': cell_df[(cell_df['species'] == 'chickadee') & (cell_df['cell_type'] == 'E')]['info'].values,
        'chick_I': cell_df[(cell_df['species'] == 'chickadee') & (cell_df['cell_type'] == 'I')]['info'].values,
        'finch_E': cell_df[(cell_df['species'] == 'finch') & (cell_df['cell_type'] == 'E')]['info'].values,
        'finch_I': cell_df[(cell_df['species'] == 'finch') & (cell_df['cell_type'] == 'I')]['info'].values,
    }
    # Hardcoded bar heights and expanded Y-limit so it clears the legend perfectly
    _grouped_boxplot(ax_a, info_data, 'Spatial Information (bits/spike)', 'A. Spatial Information per Cell', 
                     sig_heights=[1.15, 0.35], show_scatter=False)
    ax_a.set_ylim(0, 1.45) 

    # ── B: Temporal Stability ──
    xcorr_data = {
        'chick_E': cell_df[(cell_df['species'] == 'chickadee') & (cell_df['cell_type'] == 'E')]['xcorr_map'].values,
        'chick_I': cell_df[(cell_df['species'] == 'chickadee') & (cell_df['cell_type'] == 'I')]['xcorr_map'].values,
        'finch_E': cell_df[(cell_df['species'] == 'finch') & (cell_df['cell_type'] == 'E')]['xcorr_map'].values,
        'finch_I': cell_df[(cell_df['species'] == 'finch') & (cell_df['cell_type'] == 'I')]['xcorr_map'].values,
    }
    _grouped_boxplot(ax_b, xcorr_data, 'Within-Session Reliability ($r$)', 'B. Temporal Stability per Cell', 
                     sig_heights=[1.1, 1.1], show_scatter=False)
    ax_b.set_ylim(-0.4, 1.4) 

    # ── C: Mantel R (Session Level) ──
    mantel_data = {
        'chick_E': mantel_df[mantel_df['species'] == 'chickadee']['mantel_e'].values,
        'chick_I': mantel_df[mantel_df['species'] == 'chickadee']['mantel_i'].values,
        'finch_E': mantel_df[mantel_df['species'] == 'finch']['mantel_e'].values,
        'finch_I': mantel_df[mantel_df['species'] == 'finch']['mantel_i'].values,
    }
    _grouped_boxplot(ax_c, mantel_data, 'Mantel $r$ (physical vs neural)', 'C. Topogeographic Structure', 
                     sig_heights=[0.75, 0.50], show_scatter=True)
    ax_c.set_ylim(0.0, 0.9)

    # ── D: The Knockout Test (Bar Chart) ──
    x_centers = [1, 4]
    offsets = [-0.7, 0, 0.7]
    bar_width = 0.6
    
    for i, species in enumerate(['chickadee', 'finch']):
        sub = ko_df[ko_df['species'] == species]
        
        full = sub['shesha_full'].dropna()
        e_only = sub['shesha_e_only_strict'].dropna()
        i_only = sub['shesha_i_only_strict'].dropna()
        
        means = [full.mean(), e_only.mean(), i_only.mean()]
        sems = [full.sem(), e_only.sem(), i_only.sem()]
        
        base_color = C_CHICK if species == 'chickadee' else C_FINCH
        colors = [base_color, C_E, C_I]
        labels = ['Full\n(E+I)', 'E-Only', 'I-Only'] if i == 0 else [None, None, None]
        
        for j in range(3):
            ax_d.bar(x_centers[i] + offsets[j], means[j], yerr=sems[j], width=bar_width, 
                     color=colors[j], alpha=0.8, edgecolor='white', linewidth=1.2, 
                     capsize=4, label=labels[j])

    ax_d.set_xticks(x_centers)
    ax_d.set_xticklabels(['Chickadee', 'Finch'], fontsize=11)
    ax_d.set_ylabel('Geometric Rigidity (Shesha)', fontsize=11)
    ax_d.set_title('D. The Circuit Knockout Test', fontsize=13, fontweight='bold', loc='left', pad=12)
    ax_d.legend(fontsize=9, frameon=False, loc='upper right')
    ax_d.set_ylim(0, 0.35) # Hard limit so it doesn't crowd the top

    # ── Save ──
    out_pdf = OUTPUT_DIR / 'figS4_ei_single_cell.pdf'
    out_png = OUTPUT_DIR / 'figS4_ei_single_cell.png'
    fig.savefig(out_pdf, dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(out_png, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Saved: {out_pdf}")
    plt.show()

if __name__ == '__main__':
    main()