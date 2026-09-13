"""
figure_generation.py — Reproduce Figures 2 and 3 from the paper.

Reads the CSV files in data/ and writes PNGs to figures/.

Usage
-----
  python analysis/figure_generation.py

Run from the repository root. Requires the data/ CSVs to exist; if they
don't, run the simulations first:

  python simulation/casper_sim.py --csv data/raw_ate_trials.csv
  python simulation/casper_robustness.py --csv data/recall_sweep_trials.csv
"""

from pathlib import Path
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / 'data'
FIGURES = ROOT / 'figures'
FIGURES.mkdir(exist_ok=True)

COLORS = {
    'raw':       '#cc3333',
    'geometric': '#e07b29',
    'semantic':  '#1a6fa8',
}
LABELS = {
    'raw':       'Raw LiDAR (no filter)',
    'geometric': 'Geometric Filter (≥0.5 m)',
    'semantic':  'Oracle Semantic Filter',
}

# ---------------------------------------------------------------------------
# Figure 2 — ATE boxplot (Table IIb)
# ---------------------------------------------------------------------------
def figure2():
    csv_path = DATA / 'raw_ate_trials.csv'
    if not csv_path.exists():
        sys.exit(f'ERROR: {csv_path} not found. Run casper_sim.py first.')

    data = {}   # (level, condition) → list of ATE values
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['level'], row['condition'])
            data.setdefault(key, []).append(float(row['ate_m']))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x_pos   = {'low': 1.0, 'high': 2.0}
    x_off   = {'raw': -0.22, 'geometric': 0.0, 'semantic': 0.22}
    width   = 0.18

    for level, xc in x_pos.items():
        for cond, xo in x_off.items():
            vals = data.get((level, cond), [])
            if not vals:
                continue
            ax.boxplot(vals,
                       positions=[xc + xo],
                       widths=width,
                       patch_artist=True,
                       showfliers=True,
                       medianprops=dict(color='black', linewidth=1.5),
                       flierprops=dict(marker='o', markersize=3,
                                       markerfacecolor=COLORS[cond], alpha=0.5),
                       boxprops=dict(facecolor=COLORS[cond], alpha=0.75,
                                     linewidth=1.2),
                       whiskerprops=dict(linewidth=1.2),
                       capprops=dict(linewidth=1.2))

    # Significance brackets (high dynamic)
    def sig_bar(x1, x2, y, text, dy=0.02):
        ax.plot([x1, x1, x2, x2], [y, y+dy, y+dy, y], lw=1.0, color='black')
        ax.text((x1+x2)/2, y+dy+0.005, text,
                ha='center', va='bottom', fontsize=7.5)

    high_vals = {c: data.get(('high', c), []) for c in COLORS}
    y_base = max(max(v) for v in high_vals.values() if v) + 0.03
    sig_bar(2.0+x_off['geometric'], 2.0+x_off['raw'],  y_base+0.04, '** p=0.001')
    sig_bar(2.0+x_off['semantic'], 2.0+x_off['geometric'], y_base, '*** p<0.001')
    sig_bar(2.0+x_off['semantic'],  2.0+x_off['raw'],  y_base+0.09, '*** p<0.001')

    ax.set_xticks([1.0, 2.0])
    ax.set_xticklabels(['Low-Dynamic\nCondition', 'High-Dynamic\nCondition'],
                       fontsize=10)
    ax.set_ylabel('Absolute Trajectory Error (m)', fontsize=10)
    ax.set_xlim(0.5, 2.7)

    patches = [mpatches.Patch(facecolor=COLORS[c], alpha=0.75, label=LABELS[c])
               for c in ('raw', 'geometric', 'semantic')]
    ax.legend(handles=patches, fontsize=8.5, loc='upper left', framealpha=0.9)
    ax.set_title('Figure 2: ATE Across 20 Simulation Trials Per Condition\n'
                 '(Brackets: one-tailed Welch\'s t-test, n = 20; high-dynamic only)',
                 fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    out = FIGURES / 'figure2_boxplot.png'
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Figure 2 written → {out}')

# ---------------------------------------------------------------------------
# Figure 3 — Recall sweep curve
# ---------------------------------------------------------------------------
def figure3():
    csv_path = DATA / 'recall_sweep_trials.csv'
    if not csv_path.exists():
        sys.exit(f'ERROR: {csv_path} not found. Run casper_robustness.py first.')

    sweep = {}   # recall → list of ATE values
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = float(row['recall'])
            sweep.setdefault(r, []).append(float(row['ate_m']))

    recalls  = sorted(sweep)
    means    = [np.mean(sweep[r]) for r in recalls]
    stds     = [np.std(sweep[r], ddof=1) for r in recalls]

    GEO_ATE, GEO_STD = 1.363, 0.023
    RAW_ATE, RAW_STD = 1.384, 0.015
    n = len(next(iter(sweep.values())))

    # Welch p-values vs geometric
    p_vals = []
    for r in recalls:
        arr = np.array(sweep[r])
        t_s = (GEO_ATE - arr.mean()) / np.sqrt(arr.std(ddof=1)**2/n + GEO_STD**2/n)
        df  = (arr.std(ddof=1)**2/n + GEO_STD**2/n)**2 / (
              (arr.std(ddof=1)**2/n)**2/(n-1) + (GEO_STD**2/n)**2/(n-1))
        p_vals.append(1 - scipy_stats.t.cdf(t_s, df))

    r_pct = [r * 100 for r in recalls]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    ax.fill_between(r_pct,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.18, color='#1a6fa8')
    ax.plot(r_pct, means, 'o-', color='#1a6fa8', linewidth=2.0, markersize=4.5,
            label=f'Oracle Semantic Filter (varying recall, n = {n})')

    ax.axhline(GEO_ATE, color='#e07b29', linewidth=1.8, linestyle='--',
               label=f'Geometric Filter ({GEO_ATE:.3f} m)')
    ax.fill_between([0, 100], GEO_ATE-GEO_STD, GEO_ATE+GEO_STD,
                    alpha=0.12, color='#e07b29')
    ax.axhline(RAW_ATE, color='#cc3333', linewidth=1.8, linestyle=':',
               label=f'Raw LiDAR ({RAW_ATE:.3f} m)')

    ax.axvspan(70, 90, alpha=0.09, color='green',
               label='Published RandLA-Net range (out-of-domain LiDAR, 70–90%)')
    ax.axvline(70, color='green', linewidth=1.0, linestyle='-.', alpha=0.7)
    ax.axvline(90, color='green', linewidth=1.0, linestyle='-.', alpha=0.7)

    # Significance markers
    for r, m, p in zip(r_pct, means, p_vals):
        marker = ('***' if p < 0.001 else '**' if p < 0.01 else
                  '*'   if p < 0.05  else '')
        if marker:
            ax.annotate(marker, (r, m - 0.003), ha='center', va='top',
                        fontsize=7.5, color='#1a6fa8', fontweight='bold')

    # Breakeven
    above = [m > GEO_ATE for m in means]
    if any(above):
        cr = r_pct[next(i for i, a in enumerate(above) if a)]
        ax.axvline(cr, color='gray', linewidth=1.0, linestyle='--', alpha=0.5)
        ax.annotate(f'Breakeven ≈{cr:.0f}%',
                    xy=(cr, GEO_ATE),
                    xytext=(cr+6, GEO_ATE+0.013), fontsize=7.5, color='gray',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    ax.set_xlabel('Tier-1 Class Recall (%)\n(false-negative noise injection only)', fontsize=10)
    ax.set_ylabel('Mean ATE (m)', fontsize=10)
    ax.set_title(f'Figure 3: ATE vs Segmentation Recall — High-Dynamic Condition\n'
                 f'(n = {n} per level; ±1 SD band; *** p<0.001, ** p<0.01, * p<0.05 vs geometric)',
                 fontsize=9)
    ax.set_xlim(-2, 102)
    ax.set_xticks(range(0, 101, 10))
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    out = FIGURES / 'figure3_recall_curve.png'
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Figure 3 written → {out}')

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    figure2()
    figure3()
