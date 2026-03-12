#!/usr/bin/env python3
"""
Figure 3: Topological structure determines memory capacity.

Panel A : Decoding error vs memory load M for crystal/mist/noise.
Panel B : Parameter sweep topology advantage (Marginalized Line Plot).
Panel C : Empirical representational redundancy for chickadee vs finch.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# -- paths ------------------------------------------------------------------
DATA_DIR  = Path('output/tier3_capacity')
SWEEP_CSV = Path('output/tier3_sweep/tier3_parameter_sweep_complete.csv')
OUT_DIR   = Path('output/plots')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- colours ----------------------------------------------------------------
C_CRYSTAL  = '#C62D50'   # chickadee red
C_MIST     = '#4A90C4'   # blue
C_NOISE    = '#9E9E9E'   # grey
C_CHICK    = '#C62D50'
C_FINCH    = '#4A90C4'
C_THRESH   = '#333333'

FONT = 'DejaVu Sans'
plt.rcParams.update({
    'font.family'       : FONT,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.linewidth'    : 0.8,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'axes.labelsize'    : 10,
})

# -- load CSVs --------------------------------------------------------------
cap_df   = pd.read_csv(DATA_DIR / 'tier3_capacity_results.csv')
red_df   = pd.read_csv(DATA_DIR / 'tier3_redundancy_results.csv')
redf_df  = pd.read_csv(DATA_DIR / 'tier3_redundancy_filtered.csv')

# Load and process parameter sweep
sweep_df = pd.read_csv(SWEEP_CSV)
sweep_df = sweep_df[sweep_df['status'] == 'success'].copy()
if 'crystal_advantage' not in sweep_df.columns:
    sweep_df['crystal_advantage'] = sweep_df['random_nn_error'] - sweep_df['crystal_nn_error']

# topology sweep -- hardcoded from tier3 results txt
topo_tau = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                     0.6, 0.7, 0.8, 0.9, 1.0])
topo_err = np.array([0.2484, 0.2251, 0.1962, 0.1714, 0.1478, 0.1277,
                     0.1159, 0.1003, 0.0888, 0.0775, 0.0661])

# error threshold and critical transition
ERR_THRESH = 0.1601
TAU_CRIT   = 0.35

# -- figure layout ----------------------------------------------------------
fig = plt.figure(figsize=(18.0, 5.0))
fig.patch.set_facecolor('white')

outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30,
                          left=0.05, right=0.98, top=0.88, bottom=0.14,
                          width_ratios=[1.15, 1, 1])

ax_a = fig.add_subplot(outer[0])
ax_b = fig.add_subplot(outer[1])
ax_c = fig.add_subplot(outer[2])

# ============================================================================
# PANEL A: Decoding error vs memory load
# ============================================================================
for model, color, label, ls in [
    ('crystal', C_CRYSTAL, 'Crystal ($\\tau=1.0$, chickadee)', '-'),
    ('mist',    C_MIST,    'Mist ($\\tau=0.5$, finch)',        '--'),
    ('noise',   C_NOISE,   'Noise ($\\tau=0.0$)',              ':'),
]:
    sub = cap_df[cap_df['model'] == model].sort_values('memory_load')
    ax_a.plot(sub['memory_load'], sub['error'],
              color=color, lw=2.2, ls=ls, label=label, zorder=4)
    ax_a.scatter(sub['memory_load'].iloc[-1], sub['error'].iloc[-1],
                 color=color, s=28, zorder=5)

ax_a.axhline(ERR_THRESH, color=C_THRESH, lw=1.0, ls='-.', zorder=3,
             label=f'Error threshold ({ERR_THRESH:.3f})')

ax_a.axvspan(8, 12, color=C_MIST, alpha=0.10, zorder=1)
ax_a.text(11, ERR_THRESH + 0.006, r'$\tau_{\rm crit} \approx 0.35$',
          ha='left', va='bottom', fontsize=8.5, color=C_THRESH,
          bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))

ax_a.annotate('', xy=(1000, 0.065), xytext=(10, 0.065),
              arrowprops=dict(arrowstyle='<->', color=C_CRYSTAL, lw=1.1, shrinkA=0, shrinkB=0))
ax_a.text(100, 0.055, '>100-fold capacity advantage',
          ha='center', va='top', fontsize=8, color=C_CRYSTAL,
          bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))

ax_a.set_xscale('log')
ax_a.set_xlabel('Memory load $M$ (stored locations)', fontsize=10)
ax_a.set_ylabel('Mean spatial decoding error', fontsize=10)
ax_a.legend(fontsize=7.5, frameon=False, loc='upper right', bbox_to_anchor=(1, 1.04))
ax_a.set_xlim(8, 1200)
ax_a.set_ylim(0.04, 0.30)

# -- inset -------------------------------------------------------------------
ax_ins = ax_a.inset_axes([0.64, 0.43, 0.28, 0.26])
ax_ins.plot(topo_tau, topo_err, color='#555555', lw=1.8, zorder=4)
ax_ins.scatter(topo_tau, topo_err, color='#555555', s=18, zorder=5)
ax_ins.axhline(ERR_THRESH, color=C_THRESH, lw=0.9, ls='-.', zorder=3)
ax_ins.axvline(TAU_CRIT, color=C_THRESH, lw=0.9, ls=':', zorder=3)

ax_ins.scatter([1.0], [0.0661], color=C_CRYSTAL, s=32, zorder=6)
ax_ins.scatter([0.5], [0.1277], color=C_MIST,    s=32, zorder=6)
ax_ins.text(1.12, 0.075, 'Crystal', ha='right', fontsize=7, color=C_CRYSTAL)
ax_ins.text(0.52, 0.140, 'Mist',    ha='left',  fontsize=7, color=C_MIST)

ax_ins.set_xlabel(r'Topology $\tau$', fontsize=7.5)
ax_ins.set_ylabel('Error', fontsize=7.5, labelpad=-2)
ax_ins.set_title('$M=100$ fixed', fontsize=7.5)
ax_ins.tick_params(labelsize=7)
ax_ins.spines['top'].set_visible(False)
ax_ins.spines['right'].set_visible(False)

# ============================================================================
# PANEL B: Parameter Sweep (Marginalized Line Plot)
# ============================================================================
grouped = sweep_df.groupby('sparsity')['crystal_advantage']
med = grouped.median()
p10 = grouped.apply(lambda x: np.percentile(x, 10))
p90 = grouped.apply(lambda x: np.percentile(x, 90))

ax_b.plot(med.index, med.values, color=C_CRYSTAL, linewidth=2.5, label='Median Advantage')
ax_b.fill_between(med.index, p10.values, p90.values, color=C_CRYSTAL, alpha=0.15,
                  label='10th - 90th Percentile\n(Variance from N and T)')

empirical_rho = 0.15
ax_b.axvline(empirical_rho, color='#888888', linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)
ax_b.text(empirical_rho + 0.005, med.min() + 0.01, r'$\leftarrow$ Empirical $\rho$',
          color='#555555', fontsize=8.5, va='bottom')

ax_b.set_xlabel(r'Sparsity ($\rho$)', fontsize=10)
ax_b.set_ylabel(r'Topology Advantage ($\Delta$Error)', fontsize=10)
ax_b.legend(fontsize=7.5, frameon=False, loc='lower right', bbox_to_anchor=(0.95, 0.15))
ax_b.grid(True, linestyle='--', alpha=0.3)

# ============================================================================
# PANEL C: Empirical redundancy
# ============================================================================
chick_all  = red_df[red_df['species'] == 'chickadee']['redundancy'].values
finch_all  = red_df[red_df['species'] == 'finch']['redundancy'].values
chick_filt = redf_df[redf_df['species'] == 'chickadee']['redundancy'].values
finch_filt = redf_df[redf_df['species'] == 'finch']['redundancy'].values

x_cu, x_fu = 0.0, 0.6
x_cf, x_ff = 1.4, 2.0
jitter = 0.07
np.random.seed(320)

def plot_group(ax, x, vals, color, alpha_pt=0.65, s=28):
    jit = np.random.uniform(-jitter, jitter, len(vals))
    ax.scatter(x + jit, vals, color=color, alpha=alpha_pt, s=s,
               edgecolors='white', linewidths=0.4, zorder=4)
    mean_val = np.mean(vals)
    ax.plot([x - 0.15, x + 0.15], [mean_val, mean_val],
            color=color, lw=2.5, zorder=5)
    return mean_val

m_cu = plot_group(ax_c, x_cu, chick_all,  C_CHICK)
m_fu = plot_group(ax_c, x_fu, finch_all,  C_FINCH)
m_cf = plot_group(ax_c, x_cf, chick_filt, C_CHICK)
m_ff = plot_group(ax_c, x_ff, finch_filt, C_FINCH)

ax_c.set_yscale('log')
ax_c.set_ylim(0.5, max(chick_all.max(), chick_filt.max()) * 2.5)
ax_c.set_xlim(-0.35, 2.35)
y_max = ax_c.get_ylim()[1]

ax_c.axvline(1.0, color='#DDDDDD', lw=1.2, ls='--', zorder=1)
ax_c.text(1.0, y_max * 0.6, 'noise-floor\nfilter applied', ha='center', va='top',
          fontsize=7.5, color='#888888',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#BBBBBB', linewidth=0.8, alpha=0.9))

for x, m, c in [(x_cu, m_cu, C_CHICK), (x_fu, m_fu, C_FINCH),
                (x_cf, m_cf, C_CHICK), (x_ff, m_ff, C_FINCH)]:
    ax_c.text(x, m * 1.25, f'{m:.0f}x', ha='center', va='bottom',
              fontsize=8, color=c, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.15', fc='white', zorder=6, ec=c, linewidth=0.8, alpha=1))

ax_c.text(0.3, 0.65, '$p=0.041$', ha='center', fontsize=8, color='#444444',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#BBBBBB', linewidth=0.8, alpha=0.9))
ax_c.text(1.7, 0.65, '$p=0.057$', ha='center', fontsize=8, color='#888888',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#BBBBBB', linewidth=0.8, alpha=0.9))

ax_c.set_xticks([x_cu, x_fu, x_cf, x_ff])
ax_c.set_xticklabels(['Chickadee\n(unfiltered)', 'Finch\n(unfiltered)',
                      'Chickadee\n(filtered)', 'Finch\n(filtered)'], fontsize=8.5)

for tick, color in zip(ax_c.get_xticklabels(), [C_CHICK, C_FINCH, C_CHICK, C_FINCH]):
    tick.set_color(color)

ax_c.set_ylabel('Representational redundancy\n' r'($\Sigma I_k / I_{\rm pop}$)', fontsize=10)

# ============================================================================
# NATURE-STYLE PANEL LABELS (bold lowercase, top-left outside each axis)
# ============================================================================
for ax, label in [(ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c')]:
    ax.text(
        -0.10, 1.08,          # x slightly left of axis, y slightly above
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        va='top',
        ha='right',
        fontfamily='sans-serif',
    )

# -- save -------------------------------------------------------------------
out_path = OUT_DIR / 'fig3_capacity_redundancy_tier3.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
out_path = OUT_DIR / 'fig3_capacity_redundancy_tier3.pdf'
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path}")
plt.show()
plt.close()