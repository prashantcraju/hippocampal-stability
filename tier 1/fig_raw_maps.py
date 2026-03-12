#!/usr/bin/env python3
"""
fig_raw_maps.py — Supplementary Figure 6: Raw Spatial Firing Rate Maps

Generates a 4x4 grid of representative place cell heatmaps:
  Top 2 rows (8 cells): Chickadee (Crystal - sharp, multi-field)
  Bottom 2 rows (8 cells): Finch (Mist - diffuse, unstructured)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter
from pathlib import Path

# ── I/O Paths ──

DATA_PATH  = Path('data/aronov_dataset.pkl')
OUTPUT_DIR = Path('output/raw_maps')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Aesthetics ──
C_CHICK  = '#C62D50'
C_FINCH  = '#4A90C4'
GRID_SIZE = 40

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'figure.dpi'        : 150,
    'savefig.dpi'       : 300,
})

def clean_map(m_raw):
    """Safely extracts and reshapes the rate map."""
    if m_raw is None: return None
    m = np.array(m_raw, dtype=float)
    if m.size != GRID_SIZE * GRID_SIZE: return None
    return m.reshape((GRID_SIZE, GRID_SIZE))

def get_representative_cells(df, species, n_cells=8):
    """
    Selects 8 highly informative cells to serve as representative examples.
    We sample evenly from the top 30 most spatially informative cells to get 
    a clean, biologically representative cross-section of 'good' cells.
    """
    sub = df[(df['species'] == species) & (df['cell_type'] == 'E')].copy()
    sub = sub.dropna(subset=['info', 'map'])
    
    # Sort by spatial information
    sub = sub.sort_values('info', ascending=False)
    
    # Take evenly spaced samples from the top 30 cells
    top_pool = sub.head(30)
    idx = np.linspace(0, len(top_pool)-1, n_cells, dtype=int)
    return top_pool.iloc[idx]

def plot_rate_map(ax, rate_map, info, max_rate):
    """Plots a single smoothed spatial firing rate map."""
    # Identify unvisited bins (NaNs)
    unvisited = np.isnan(rate_map)
    
    # Fill NaNs with 0 for smoothing, then apply Gaussian filter
    filled = np.nan_to_num(rate_map, nan=0.0)
    smoothed = gaussian_filter(filled, sigma=1.2)
    
    # Re-apply NaN mask so unvisited bins can be colored distinctly
    smoothed[unvisited] = np.nan
    
    # Use Viridis (neuroscience standard for rate maps), with grey for unvisited
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='#E5E5E5')
    
    im = ax.imshow(smoothed, cmap=cmap, origin='lower', 
                   vmin=0, vmax=np.nanmax(smoothed))
    
    # Clean axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#333333')
        spine.set_linewidth(1.5)
        
    # Add text overlay for Max Rate and Spatial Info
    # PathEffects add a black outline to the white text so it is readable over any color
    txt = f"{max_rate:.1f} Hz\n{info:.2f} bits"
    ax.text(0.95, 0.95, txt, transform=ax.transAxes, 
            ha='right', va='top', color='white', fontsize=9, fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])

def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}")
        return
        
    df = pd.read_pickle(DATA_PATH)
    
    chick_cells = get_representative_cells(df, 'titmouse', n_cells=8)
    finch_cells = get_representative_cells(df, 'zebra_finch', n_cells=8)

    # ── Setup Nested Figure ──
    fig = plt.figure(figsize=(10, 11), facecolor='white')
    
    # Outer Grid: 2 Rows (Top=Chickadee, Bottom=Finch)
    outer = gridspec.GridSpec(2, 1, hspace=0.3, top=0.92, bottom=0.05, left=0.05, right=0.95)
    
    # Inner Grids: 2x4 for each species
    gs_chick = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
    gs_finch = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
    
    # Add Section Titles
    ax_chick_title = fig.add_subplot(outer[0])
    ax_chick_title.axis('off')
    ax_chick_title.set_title('A. Chickadee Place Cells (Crystalline Map)', 
                             fontsize=14, fontweight='bold', color=C_CHICK, loc='left', pad=25)
                             
    ax_finch_title = fig.add_subplot(outer[1])
    ax_finch_title.axis('off')
    ax_finch_title.set_title('B. Finch Place Cells (Mist Map)', 
                             fontsize=14, fontweight='bold', color=C_FINCH, loc='left', pad=25)

    # ── Plot Chickadee Cells ──
    for i, (_, row) in enumerate(chick_cells.iterrows()):
        ax = fig.add_subplot(gs_chick[i // 4, i % 4])
        rmap = clean_map(row['map'])
        if rmap is not None:
            max_rate = np.nanmax(rmap)
            plot_rate_map(ax, rmap, row['info'], max_rate)

    # ── Plot Finch Cells ──
    for i, (_, row) in enumerate(finch_cells.iterrows()):
        ax = fig.add_subplot(gs_finch[i // 4, i % 4])
        rmap = clean_map(row['map'])
        if rmap is not None:
            max_rate = np.nanmax(rmap)
            plot_rate_map(ax, rmap, row['info'], max_rate)

    # ── Save ──
    out_pdf = OUTPUT_DIR / 'figS6_raw_rate_maps.pdf'
    out_png = OUTPUT_DIR / 'figS6_raw_rate_maps.png'
    fig.savefig(out_pdf, dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(out_png, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Saved: {out_pdf}")
    plt.show()

if __name__ == '__main__':
    main()