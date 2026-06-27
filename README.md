# Human Activity Recognition from Accelerometer Data


> Classifying physical activities — walking, running, idle, and stair-climbing — using time-domain feature engineering, SVM, and Random Forest on raw mobile phone accelerometer signals.

---

## Overview

This project is a full ML classification pipeline built on accelerometer data collected from a mobile phone. The goal is to predict what physical activity a person is performing based solely on raw sensor readings across three axes (X, Y, Z).

Four activity classes are covered:

| Class | Description |
|-------|-------------|
| `idle` | Standing still or sitting |
| `walking` | Regular flat-surface walking |
| `running` | Running or jogging |
| `stairs` | Ascending or descending stairs |

The project covers the full workflow: raw data ingestion → feature engineering → exploratory analysis → model training → hyperparameter tuning → comparative evaluation.

---

## Dataset

The dataset consists of CSV files organised into four directories (`idle/`, `running/`, `walking/`, `stairs/`), each containing time-series accelerometer readings with columns:

- `accelerometer_X`
- `accelerometer_Y`
- `accelerometer_Z`

Each file represents one recording session of a single activity. Files are loaded, features are extracted per file, and the results are assembled into a single feature matrix for modelling.

---

## Feature Engineering

Instead of feeding raw signals directly into the classifier, **47 time-domain features** are extracted per recording window. This approach is standard in inertial sensor-based activity recognition and substantially improves model separability.

### Per-axis features (computed for X, Y, Z independently — 14 × 3 = 42 features)

| Feature | Description |
|---------|-------------|
| `mean` | Arithmetic mean of signal |
| `std` | Standard deviation |
| `var` | Variance |
| `median` | Median value |
| `min` / `max` | Signal range extremes |
| `rms` | Root Mean Square — effective signal amplitude |
| `idx_min` / `idx_max` | Index positions of min/max values |
| `entropy` | Shannon entropy of binned signal histogram |
| `skewness` | Asymmetry of signal distribution |
| `kurtosis` | Peakedness / tail heaviness of distribution |
| `iqr` | Interquartile range — robust spread measure |
| `mad` | Mean Absolute Deviation |

### Cross-axis and composite features (5 features)

| Feature | Description |
|---------|-------------|
| `energy` | Sum of squared deviations of the Signal Magnitude Vector (SMV) from its mean |
| `power` | Energy normalised by window length |
| `sma` | Signal Magnitude Area — mean total absolute signal across all axes |
| `xy_corr`, `yz_corr`, `xz_corr` | Normalised Pearson-style correlations between axis pairs |

---

## Exploratory Data Analysis

Several visualisations are used to understand feature structure before modelling:

- **Strip plots** — power and energy distributions by activity class, confirming running produces the highest intensity signal
- **Scatter plot** — SMA vs energy, revealing near-linear dependency and strong class separation
- **Correlation heatmap** — identifying redundant feature groups (variance ↔ std, energy ↔ power)
- **t-SNE projection** — 2D embedding of the full 47-feature space, showing clear cluster separation with expected walking/stairs overlap
- **Radar chart** — normalised mean feature profiles per activity, used to identify informative vs redundant features
- **KDE plots** — per-feature density distributions by class, used to assess individual discriminability

Key insight from EDA: `entropy_X`, variance features, and `power` are highly correlated with other features and contribute limited unique information — they are removed in the second modelling phase.

---

## Models

Two algorithms are compared across two feature sets and two preprocessing conditions:

### Algorithms

**Support Vector Machine (SVM)**  
Implemented via `sklearn.svm.SVC` with `GridSearchCV` over:
- Kernels: `linear`, `rbf`, `poly`
- Regularisation: `C ∈ {0.1, 1, 10}`
- Gamma: `scale`, `0.1`, `1` (for RBF/poly)
- Polynomial degree: `2`, `3`, `4`

**Random Forest**  
Implemented via `sklearn.ensemble.RandomForestClassifier`:
- `n_estimators = 200`
- `class_weight = "balanced"` (handles any class imbalance)
- `random_state = 5` for reproducibility

### Feature sets

| Phase | Features | Motivation |
|-------|----------|-----------|
| Phase 1 | All 47 engineered features | Baseline — no filtering |
| Phase 2 | 32 features (15 removed) | Removed highly correlated and low-discriminability features identified via EDA |

**Removed in Phase 2:** `var_X/Y/Z`, `power`, `entropy_X`, `idx_min/max` for all axes, `kurtosis_X`, `skewness_Y`, `kurtosis_Y`

### Preprocessing

Both raw and `StandardScaler`-scaled versions are evaluated for each model and feature set, giving 8 total model configurations compared.

---

## Results Summary

All models are evaluated using `classification_report` (precision, recall, F1-score per class) plus macro-averaged aggregate metrics.

### Phase 1 — All features

| Model | Data | Accuracy | Macro F1 |
|-------|------|----------|----------|
| SVM | Raw | Lower | Lower |
| SVM | Scaled | ✅ Better | ✅ Better |
| Random Forest | Raw | ≈ same | ≈ same |
| Random Forest | Scaled | ≈ same | ≈ same |

**Key finding:** SVM is highly sensitive to feature scale (as expected theoretically); Random Forest is scale-invariant. Scaled SVM outperforms raw SVM significantly, particularly on the harder `stairs` vs `walking` distinction.

### Phase 2 — Cleaned features

| Model | Data | vs Phase 1 |
|-------|------|-----------|
| SVM | Scaled | ✅ Higher accuracy, F1, precision, recall |
| Random Forest | Scaled | Similar to Phase 1 |

**Best overall model:** SVM with StandardScaler and cleaned feature set.

### Notable observation

The final SVM model achieves ~99.76% accuracy — a suspiciously high figure that warrants caution. This could indicate:
1. **Data leakage** if train/test split was performed at the sample level rather than the recording file level (samples from the same session appearing in both splits)
2. **Genuinely separable data** given the richness of the feature engineering combined with a small, clean dataset

This is flagged explicitly in the notebook as a point for further investigation.

---

## Project Structure

```
human-activity-recognition/
│
├── data/
│   ├── idle/          # CSV files for idle activity
│   ├── running/       # CSV files for running
│   ├── walking/       # CSV files for walking
│   └── stairs/        # CSV files for stair-climbing
│
└── notebook.ipynb     # Full pipeline notebook
```

---

## Tech Stack

| Library | Usage |
|---------|-------|
| `pandas` | Data loading and feature matrix assembly |
| `numpy` | Numerical feature computation |
| `scipy.stats` | Entropy, skewness, kurtosis, IQR |
| `scikit-learn` | SVM, Random Forest, GridSearchCV, StandardScaler, metrics |
| `matplotlib` | Base plotting |
| `seaborn` | Strip plots, KDE, scatter, heatmap |
| `sklearn.manifold.TSNE` | Dimensionality reduction for EDA |

---

## How to Run

1. Clone the repository
2. Place the dataset under `data/` with subdirectories `idle/`, `running/`, `walking/`, `stairs/`
3. Install dependencies:
   ```bash
   pip install pandas numpy scipy scikit-learn matplotlib seaborn
   ```
4. Open and run `notebook.ipynb` top to bottom

> **Note:** GridSearchCV with the full parameter grid is computationally intensive. On a standard laptop, expect a few minutes per search (4 model configurations × grid size).

---

## Key Learnings

- Time-domain feature engineering from raw IMU signals dramatically improves classifier performance compared to using raw samples
- SVM requires feature scaling; Random Forest does not — this matches theoretical expectations and was confirmed empirically
- Removing multicollinear features (energy/power, std/var) improved SVM performance, consistent with the algorithm's sensitivity to the feature space geometry
- The walking vs stairs distinction is the hardest classification boundary in this dataset, consistently showing the lowest per-class F1 scores
- t-SNE is a powerful EDA tool for validating feature engineering quality before committing to modelling

---

## Author

**Maksym** — Data Science & ML Engineer  
GoIT Data Science Programme  
[LinkedIn](#https://www.linkedin.com/in/maksym-dovhusha) · [GitHub](#https://github.com/MaksimDar/human-activity-recognition-imu)
