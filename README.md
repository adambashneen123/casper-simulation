# CASPER — Reproducibility Package

**Paper**: "CASPER: A Simulation-Validated Design Study of Semantic Dynamic-Object Filtering for LiDAR Odometry in Industrial Warehouse Environments"

> **Preprint — not peer reviewed.**  
> CASPER is a proposed architecture. The full LIO-SAM + RandLA-Net integration was not implemented.  
> All experimental results were produced by the simulation in this repository.

---

## What this repository contains

| Path | Contents |
|---|---|
| `simulation/` | PyBullet-free 2-D simulation (ray-casting LiDAR, three filter conditions) |
| `configs/` | All experimental parameters as a single YAML file |
| `data/raw_ate_trials.csv` | Per-trial ATE values for Table IIb (n=20 × 3 conditions × 2 levels) |
| `data/recall_sweep_trials.csv` | Per-trial ATE values for Section 4.3 recall sweep (n=20 × 11 recall levels) |
| `analysis/statistical_analysis.py` | Reproduces all t-tests, Hedges g, and 95% CIs reported in the paper |
| `analysis/figure_generation.py` | Reproduces Figures 2 and 3 |
| `figures/` | Output directory for generated figures |

---

## Reproducing the results

### 1. Install dependencies

```bash
pip install numpy scipy matplotlib pillow
```

### 2. Run the main simulation (Table IIb, n=20)

```bash
python simulation/sim_n20.py
```

Outputs: `data/raw_ate_trials.csv`

### 3. Run the recall sweep (Section 4.3, n=20 per level)

```bash
python simulation/sim_robustness_n20.py
```

Outputs: `data/recall_sweep_trials.csv`

### 4. Reproduce statistical analysis

```bash
python analysis/statistical_analysis.py
```

Prints all t-statistics, p-values, Hedges' g, and 95% CIs from the paper.

### 5. Reproduce figures

```bash
python analysis/figure_generation.py
```

Outputs: `figures/figure2_boxplot.png`, `figures/figure3_recall_curve.png`

---

## Experimental parameters

See `configs/experiment_params.yaml` for the authoritative parameter set.  
Key parameters (summary):

| Parameter | Value |
|---|---|
| Number of trials | 20 per condition |
| Random seeds | 1001, 2998, 3995, … (1001 + i×997 for i=0..19) |
| N_STEPS (trajectory steps) | 80 per trial |
| LiDAR beams | 180 (over 270° FOV) |
| Max range | 7.0 m |
| Range noise σ | 0.025 m |
| Geometric filter threshold | 0.5 m depth-inconsistency |
| Semantic exclusion radius | 0.4 m (SEM_RAD) |
| ATE definition | RMSE of 2-D position error over all steps |
| ICP surrogate | Calibrated systematic-drift model (see paper §4.1) |
| Dynamic condition (high) | Forklift 1.4 m/s, person 0.6 m/s, Tier-2 pallet |
| Dynamic condition (low) | Person 0.4 m/s only, Tier-2 pallet |

---

## Licenses

- **Code** (`simulation/`, `analysis/`): MIT License  
- **Data** (`data/`): CC BY 4.0  
- **Manuscript** (`CASPER_Manuscript_v1.0.pdf`, when added): CC BY 4.0

