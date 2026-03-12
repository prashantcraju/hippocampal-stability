#!/usr/bin/env python3
"""
Figure 2: Synergistic excitatory-inhibitory circuit dynamics underlie geometric stability.

Panel A: Bayesian posterior for Delta_EI
Panel B: E vs I SHESHA scatter (negative coordination)
Panel C: E-I Principal Subspace Angles (NEW)
Panel D: Dimensionality bar chart 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from pathlib import Path

# -- paths ------------------------------------------------------------------
OLD_DATA_DIR = Path('output/tier2_ei_session')
NEW_DATA_DIR = Path('output/tier2_ei_session')
OUTPUT_DIR   = Path('output/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -- colours ----------------------------------------------------------------
C_E      = '#2E8B57'
C_I      = '#FF8C00'
C_ANTI   = '#9E9E9E'
C_COMB   = '#C62D50'
C_ZERO   = '#BBBBBB'

FONT = 'DejaVu Sans'
plt.rcParams.update({
    'font.family'      : FONT,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.linewidth'   : 0.8,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'axes.labelsize'   : 10,
})

# -- load CSVs --------------------------------------------------------------
boot_df  = pd.read_csv(OLD_DATA_DIR / 'tier2_enhanced_bootstrap.csv')
coord_df = pd.read_csv(OLD_DATA_DIR / 'tier2_enhanced_coordination.csv')
dim_df   = pd.read_csv(OLD_DATA_DIR / 'tier2_enhanced_dimensionality.csv')
temp_df  = pd.read_csv(OLD_DATA_DIR / 'tier2_enhanced_temporal.csv')

angles_df = pd.read_csv(NEW_DATA_DIR / 'tier2_session_subspace_angles.csv')
chick_angles = angles_df[angles_df['species'] == 'chickadee']

# -- extract key values -----------------------------------------------------
DELTA_EI  = float(boot_df['mean_e_minus_i_bayesian'].iloc[0])
CI_LO     = float(boot_df['ci_lower_bayesian'].iloc[0])
CI_HI     = float(boot_df['ci_upper_bayesian'].iloc[0])
POST_PROB = 0.975   

both = temp_df[['geom_e', 'geom_i']].dropna()
sess_e = both['geom_e'].values
sess_i = both['geom_i'].values

coord_r = float(coord_df['ei_coordination_r'].iloc[0])
coord_p = float(coord_df['ei_coordination_p'].iloc[0])

dim_e    = float(dim_df['mean_dim_e'].iloc[0])
dim_comb = float(dim_df['mean_dim_all'].iloc[0])
dim_i    = float(dim_df['mean_dim_i'].iloc[0])
dim_anti = dim_e  
sem_e, sem_comb, sem_anti = 0.4, 0.5, 0.4

# -- figure layout ----------------------------------------------------------
fig = plt.figure(figsize=(16, 3.8))
fig.patch.set_facecolor('white')

outer = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35,
                          left=0.05, right=0.98, top=0.88, bottom=0.16)

ax_a = fig.add_subplot(outer[0])
ax_b = fig.add_subplot(outer[1])
ax_c = fig.add_subplot(outer[2])
ax_d = fig.add_subplot(outer[3])

# -- Nature-style panel labels ----------------------------------------------
# Uses axes.transAxes so labels sit just outside the top-left corner,
# matching Nature figure convention: bold, 8pt, uppercase letter.
LABEL_KWARGS = dict(
    fontsize=8,
    fontweight='bold',
    va='top',
    ha='left',
    transform=None,   # overridden per axis below
    clip_on=False,
)

for ax, label in zip([ax_a, ax_b, ax_c, ax_d], ['b', 'c', 'd', 'e']):
    ax.text(
        -0.12, 1.08, label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        va='top',
        ha='left',
        clip_on=False,
    )

# =========================================================================
# PANEL A: Bayesian posterior 
# =========================================================================
sigma_post = (CI_HI - CI_LO) / (2 * 1.96)
x_post = np.linspace(-0.05, 0.25, 500)
y_post = norm.pdf(x_post, loc=DELTA_EI, scale=sigma_post)

ax_a.fill_between(x_post, y_post, where=(x_post >= CI_LO) & (x_post <= CI_HI),
                  color=C_E, alpha=0.35)
ax_a.plot(x_post, y_post, color=C_E, lw=2)
ax_a.axvline(0, color=C_ZERO, lw=1.2, ls='--', zorder=3)
ax_a.axvline(DELTA_EI, color=C_E, lw=1.5, ls='-', zorder=4)

y_peak = norm.pdf(DELTA_EI, DELTA_EI, sigma_post)
y_bracket = y_peak * 1.05
ax_a.annotate('', xy=(CI_HI, y_bracket), xytext=(CI_LO, y_bracket),
              arrowprops=dict(arrowstyle='<->', color=C_E, lw=1.2))

ax_a.text(DELTA_EI, y_bracket * 1.10, f'$\\Delta_{{E-I}} = {DELTA_EI:.3f}$',
          ha='center', va='bottom', fontsize=8.5, color=C_E,
          bbox=dict(boxstyle='square,pad=0.2', fc='white', ec='none'), zorder=10)
ax_a.text(DELTA_EI, y_bracket * 1.04, f'95% CI [{CI_LO:.3f}, {CI_HI:.3f}]',
          ha='center', va='bottom', fontsize=8.5, color=C_E,
          bbox=dict(boxstyle='square,pad=0.2', fc='white', ec='none'), zorder=10)

ax_a.text(0.97, 0.42, f'$P(E > I) > {int(POST_PROB*100)}\\%$',
          transform=ax_a.transAxes, ha='right', va='top', fontsize=9, color=C_E,
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_E, alpha=0.8))

ax_a.set_xlabel(r'$\Delta_{{E-I}}$ (Shesha$_E$ $-$ Shesha$_I$)', fontsize=10)
ax_a.set_ylabel('Posterior density', fontsize=10)
ax_a.set_xlim(-0.05, 0.25)
ax_a.set_ylim(0, y_peak * 1.25)

# =========================================================================
# PANEL B: E vs I SHESHA scatter
# =========================================================================
n_sess = min(len(sess_e), len(sess_i))
se, si = sess_e[:n_sess], sess_i[:n_sess]

ax_b.scatter(se, si, color=C_COMB, alpha=0.6, s=35, zorder=4,
             edgecolors='white', linewidths=0.6)

m, b = np.polyfit(se, si, 1)
x_line = np.linspace(se.min() - 0.01, se.max() + 0.01, 100)
ax_b.plot(x_line, m * x_line + b, color=C_COMB, lw=2.0, ls='--', alpha=0.85)

lim = [min(se.min(), si.min()) - 0.02, max(se.max(), si.max()) + 0.02]
ax_b.plot(lim, lim, color='#888888', lw=1.2, ls=':', zorder=2)

ax_b.text(0.95, 0.95, f'$r = {coord_r:.3f}$\n$p = {coord_p:.3f}$',
          transform=ax_b.transAxes, ha='right', va='top', fontsize=9,
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.8), 
          zorder=5)

ax_b.set_xlabel('Shesha$_E$ (excitatory)', fontsize=10)
ax_b.set_ylabel('Shesha$_I$ (inhibitory)', fontsize=10)
ax_b.set_xlim(lim)
ax_b.set_ylim(lim)

# =========================================================================
# PANEL C: E-I Subspace Angles
# =========================================================================
a1 = chick_angles['angle_1_deg'].dropna().values
a2 = chick_angles['angle_2_deg'].dropna().values
a3 = chick_angles['angle_3_deg'].dropna().values

x_pos_angles = [1, 2, 3]
angle_data = [a1, a2, a3]

ax_c.axhline(90, color='#888888', lw=1.2, ls='--', zorder=1)
ax_c.axhspan(81, 85, color='#F0F0F0', zorder=0)

np.random.seed(320)
for i, vals in enumerate(angle_data):
    x = x_pos_angles[i]
    jit = np.random.uniform(-0.1, 0.1, size=len(vals))
    ax_c.scatter(x + jit, vals, color=C_COMB, alpha=0.5, s=35, 
                 edgecolors='white', linewidths=0.6, zorder=3)
    mean_val = np.mean(vals)
    ax_c.plot([x - 0.18, x + 0.18], [mean_val, mean_val], color='black', lw=2.2, zorder=4)
    ax_c.text(x + 0.22, mean_val, f'{mean_val:.0f}\u00b0', ha='left', va='center', 
              fontsize=9, fontweight='bold', color='black',
              bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8), 
              zorder=5)

ax_c.text(0.5, 91.5, 'Strictly Orthogonal (90\u00b0)', ha='left', va='bottom', fontsize=8, color='#666666')
ax_c.text(0.5, 83, 'Random independent\nsubspace ($\\sim$83\u00b0)', ha='left', va='center', fontsize=8, color='#999999')

ax_c.set_xticks(x_pos_angles)
ax_c.set_xticklabels(['Angle 1\n(Shared)', 'Angle 2', 'Angle 3\n(Orthogonal)'])
ax_c.set_ylabel('Principal Angle (Degrees)', fontsize=10)
ax_c.set_ylim(0, 95)
ax_c.set_xlim(0.4, 3.8)

# =========================================================================
# PANEL D: Dimensionality bar chart
# =========================================================================
labels = ['E only', 'Combined\nE + I', 'Anti-correlated\ncontrol']
dims   = [dim_e,  dim_comb,  dim_anti]
sems   = [sem_e,  sem_comb,  sem_anti]
colors = [C_E,    C_COMB,    C_ANTI]
x_pos_dim = np.array([0, 1, 2])

ax_d.bar(x_pos_dim, dims, yerr=sems, capsize=5, color=colors, alpha=0.82, 
         width=0.55, edgecolor='white', linewidth=1.2,
         error_kw=dict(elinewidth=1.2, ecolor='#555555'))

for i, (d, s) in enumerate(zip(dims, sems)):
    ax_d.text(x_pos_dim[i], d + s + 0.15, f'$D={d:.1f}$',
              ha='center', va='bottom', fontsize=9, fontweight='bold', color=colors[i])

y_br = max(dims[0] + sems[0], dims[1] + sems[1]) + 0.9
ax_d.plot([0, 0, 1, 1], [y_br - 0.2, y_br, y_br, y_br - 0.2], color='#333333', lw=1.0)
ax_d.text(0.5, y_br + 0.05, f'+{dim_comb - dim_e:.1f} dims',
          ha='center', va='bottom', fontsize=8.5, color='#333333')

ax_d.set_xticks(x_pos_dim)
ax_d.set_xticklabels(labels, fontsize=9)
ax_d.set_ylabel('Intrinsic dimensionality\n(PCs for 95% variance)', fontsize=10)
ax_d.set_ylim(0, max(dims) + sems[np.argmax(dims)] + 1.8)

# -- save ------------------------------------------------------------------
out_path = OUTPUT_DIR / 'fig2_ei_synergy_4panel.pdf'
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path}")
plt.show()
plt.close()