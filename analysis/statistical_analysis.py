"""Reproduce all statistical results reported in the paper."""
import numpy as np
from scipy import stats

N = 20
RESULTS = {
    ('high','raw'):      (1.384, 0.015),
    ('high','geometric'):(1.363, 0.023),
    ('high','semantic'): (1.309, 0.045),
    ('low', 'raw'):      (0.681, 0.021),
    ('low', 'geometric'):(0.682, 0.013),
    ('low', 'semantic'): (0.679, 0.003),
}

def hedges_g(m1, s1, m2, s2, n=N):
    pooled = np.sqrt(((n-1)*s1**2 + (n-1)*s2**2)/(2*n-2))
    return (m2-m1)/pooled * (1 - 3/(4*(2*n-2)-1))

def welch(m1, s1, m2, s2, n=N):
    diff = m2 - m1
    se = np.sqrt(s1**2/n + s2**2/n)
    df = (s1**2/n + s2**2/n)**2 / ((s1**2/n)**2/(n-1) + (s2**2/n)**2/(n-1))
    t  = diff/se
    p1 = 1 - stats.t.cdf(t, df)
    tc = stats.t.ppf(0.975, df)
    return diff, df, t, p1, diff-tc*se, diff+tc*se

print("HIGH-DYNAMIC PAIRWISE COMPARISONS (n=20 each)")
print("="*60)
for a,b,lbl in [('semantic','raw','Semantic filter  vs  Raw LiDAR'),
                 ('geometric','raw','Geometric filter  vs  Raw LiDAR'),
                 ('semantic','geometric','Semantic filter  vs  Geometric filter')]:
    ma,sa = RESULTS[('high',a)]; mb,sb = RESULTS[('high',b)]
    diff,df,t,p,lo,hi = welch(ma,sa,mb,sb)
    g = hedges_g(ma,sa,mb,sb)
    print(f"\n{lbl}")
    print(f"  ATE reduction : {diff:.3f} m  ({diff/mb*100:.1f}%)")
    print(f"  Hedges' g     : {g:.2f}")
    print(f"  95% CI on Δ   : ({lo:.3f}, {hi:.3f}) m")
    print(f"  t = {t:.2f}, df = {df:.1f}, p (one-tail) = {p:.4f}")
