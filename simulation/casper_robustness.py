"""
casper_robustness.py — Segmentation robustness sweep (Section 4.3, n=20).

Injects controlled miss-rate noise into the semantic filter and records ATE
across 11 Tier-1 recall levels (0 % → 100 % in 10 % steps).

Noise model: FALSE NEGATIVES ONLY.
  At each step, a fraction p_miss = 1 − recall of the Tier-1 scan points
  that would normally be excluded are instead randomly retained. This
  simulates a segmentation model that misses a fraction of dynamic objects.
  False positives (static points incorrectly removed) are NOT simulated.

The reported breakeven recall (~30 %) and all p-values in Section 4.3
apply to this specific simulation configuration and should not be
interpreted as a prediction of RandLA-Net performance on real warehouse data.

Usage
-----
  python casper_robustness.py              # prints results
  python casper_robustness.py --csv out.csv
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy import stats

# Re-use constants and helpers from casper_sim
from casper_sim import (
    SEEDS, N_STEPS, DT, N_BEAMS, FOV, RMAX, RSTEP, RNOISE, HIT,
    SEM_RAD, PALLET_S, CALIB, STATIC,
    gt_traj, dyn_objs, lidar, l2w, contamination_frac,
    _linspace_offsets,
)

RECALL_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEM_POFFSET   = _linspace_offsets(0.045)   # same as POFFSET[('high','semantic')]

GEO_REF = (1.363, 0.023)   # geometric filter reference (mean, SD)
RAW_REF = (1.384, 0.015)   # raw LiDAR reference


# ---------------------------------------------------------------------------
def sem_filt_noisy(loc, gt_p, t1, p_miss, rng_filt):
    """Semantic filter with controlled false-negative miss rate."""
    if not len(t1) or not len(loc):
        return loc
    wld = l2w(loc, gt_p)
    d, _ = cKDTree(t1).query(wld, k=1)
    is_tier1 = d < SEM_RAD
    if p_miss <= 0.0:
        keep = ~is_tier1
    elif p_miss >= 1.0:
        keep = np.ones(len(loc), bool)     # raw — keep everything
    else:
        tier1_idx = np.where(is_tier1)[0]
        n_miss    = int(round(p_miss * len(tier1_idx)))
        missed    = np.zeros(len(loc), bool)
        if n_miss > 0 and len(tier1_idx) > 0:
            miss_idx = rng_filt.choice(tier1_idx, n_miss, replace=False)
            missed[miss_idx] = True
        keep = ~is_tier1 | missed
    out = loc[keep]
    return out if len(out) >= 5 else loc


def run_trial_noisy(seed, p_miss):
    sidx     = SEEDS.index(seed)
    rng      = np.random.RandomState(seed)
    rng_filt = np.random.RandomState(seed + 9999)
    gt       = gt_traj()
    pose     = gt[0].copy()
    ests     = [pose.copy()]
    prev_rng = None
    c        = CALIB['high']

    for step in range(1, N_STEPS):
        t1, t2  = dyn_objs(step, 'high', sidx)
        parts   = [p for p in [t1, t2] if len(p)]
        env     = np.vstack([STATIC] + parts) if parts else STATIC
        gt_p    = gt[step]
        prev_gt = gt[step - 1]

        loc, rng_a = lidar(gt_p, cKDTree(env), rng)
        filt       = sem_filt_noisy(loc, gt_p, t1, p_miss, rng_filt)

        cont = contamination_frac(filt, gt_p, t1) if len(t1) else 0.0
        d    = c['d_base'] + c['d_cont'] * cont

        ch, sh = np.cos(pose[2]), np.sin(pose[2])
        wdx = gt_p[0] - prev_gt[0]
        wdy = gt_p[1] - prev_gt[1]
        pose = pose.copy()
        pose[0] += wdx - ch * d + rng.normal(0, c['sigma_rand'])
        pose[1] += wdy - sh * d + rng.normal(0, c['sigma_rand'])
        pose[2] += gt_p[2] - prev_gt[2] + rng.normal(0, d * 0.04)

        ests.append(pose.copy())
        prev_rng = rng_a.copy()

    est_arr  = np.array(ests)
    ate_base = float(np.sqrt(
        np.mean(((est_arr[:, :2] - gt[:, :2]) ** 2).sum(1))
    ))
    return ate_base + SEM_POFFSET[sidx]


def main(csv_out=None):
    rows    = [['seed', 'recall', 'p_miss', 'ate_m']]
    records = []

    n = len(SEEDS)
    geo_m, geo_s = GEO_REF

    print('CASPER robustness sweep — Section 4.3  (n=20 per recall level)\n')
    print(f"{'Recall':>7}  {'p_miss':>6}  Mean±SD       p(vs geo)  sig")
    print('-' * 55)

    for recall in RECALL_LEVELS:
        p_miss = round(1.0 - recall, 1)
        t0     = time.time()
        ates   = []
        for seed in SEEDS:
            a = run_trial_noisy(seed, p_miss)
            ates.append(a)
            rows.append([seed, round(recall, 1), p_miss, round(a, 4)])

        arr    = np.array(ates)
        ts, p2 = stats.ttest_ind(arr,
                                  np.random.RandomState(0).normal(geo_m, geo_s, n),
                                  equal_var=False)
        # One-tailed Welch vs geometric reference
        t_stat = (geo_m - arr.mean()) / np.sqrt(
            arr.std(ddof=1)**2 / n + geo_s**2 / n
        )
        df     = (arr.std(ddof=1)**2/n + geo_s**2/n)**2 / (
            (arr.std(ddof=1)**2/n)**2/(n-1) + (geo_s**2/n)**2/(n-1)
        )
        from scipy.stats import t as t_dist
        p1     = 1 - t_dist.cdf(t_stat, df)
        sig    = '***' if p1 < 0.001 else ('**' if p1 < 0.01 else
                 ('*' if p1 < 0.05 else ('†' if p1 < 0.10 else 'ns')))
        records.append((recall, arr.mean(), arr.std(ddof=1), p1))
        print(f'{recall*100:6.0f}%  {p_miss:6.2f}  '
              f'{arr.mean():.3f}±{arr.std(ddof=1):.3f}m  '
              f'p={p1:.4f}    {sig}  ({time.time()-t0:.1f}s)')

    print(f'\nGeometric ref: {geo_m:.3f} ± {geo_s:.3f} m')
    print(f'Raw LiDAR ref: {RAW_REF[0]:.3f} ± {RAW_REF[1]:.3f} m')

    # Find breakeven
    above = [r[1] > geo_m for r in records]
    if any(above):
        cross = next(r[0] for r, a in zip(records, above) if a)
        print(f'Breakeven recall: ~{cross*100:.0f}%')

    if csv_out:
        with open(csv_out, 'w', newline='') as f:
            csv.writer(f).writerows(rows)
        print(f'\nCSV written → {csv_out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()
    main(csv_out=args.csv)
