# MARSIM: ML vs MANA — GPS Spoofing Detection

Comparing classical machine learning classifiers against the rule-based **[MANA framework](https://github.com/fkie-cad/mana)** for detecting GPS spoofing attacks on the **[MARSIM dataset](https://doi.org/10.3390/jmse11050928)**.

## Key Findings

- **4 ML classifiers** (Random Forest, XGBoost, LightGBM, SVM) are trained on statistical features extracted from NMEA-0183 data and compared against the published **MANA** rule-based baseline.
- The core analysis focuses on **Scenario A3** (advanced simulator attacker with gradual position drift), where MANA's detection performance degrades at low drift speeds while ML classifiers maintain robust detection.
- Per-parameter heatmaps and detection-cliff curves reveal precisely where and why MANA fails.

## Dataset

The **MARSIM dataset** (Spravil et al., 2023) contains **43,320 pcap files** of simulated maritime GPS data:

| Scenario | Attacker | Parameters | Files |
|----------|----------|------------|-------|
| A1 | Replay (AR) | `distance_to_ship` × `time_difference` | 14,440 |
| A2 | Meaconing (AM) | `distance_to_ship` × `delay` | 14,440 |
| A3 | Simulator (AS) | `shift_angle` × `shift_speed` | 14,440 |

Each scenario is 50/50 balanced (spoofed vs. unspoofed) with 19×19 parameter combinations × 20 repetitions.

> **Note:** The dataset is not included in this repository due to size (~560MB compressed, ~43K files). Download it from [Zenodo (DOI: 10.5281/zenodo.8202936)](https://doi.org/10.5281/zenodo.8202936).

## Pipeline

```
                    ┌────────────────┐
  43,320 pcap files │  MARSIM Dataset│
                    └───────┬────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼                           ▼
   ┌──────────────────┐        ┌──────────────────┐
   │ extract_features │        │ run_mana_baseline │
   │     .py          │        │     .py           │
   └────────┬─────────┘        └────────┬──────────┘
            │                           │
            ▼                           ▼
   marsim_features.csv        mana_predictions.csv
            │                           │
            └─────────┬─────────────────┘
                      ▼
           ┌──────────────────┐
           │ train_classifiers│
           │     .py          │
           └────────┬─────────┘
                    ▼
             results/ (CSV + PNG)
```

### Scripts

| Script | Purpose |
|--------|---------|
| `extract_features.py` | Parse pcap files → extract 138 statistical features per file → `marsim_features.csv` |
| `run_mana_baseline.py` | Run MANA's 6 detection methods on all pcap files → `mana_predictions.csv` |
| `train_classifiers.py` | Train ML models, evaluate all 5 classifiers, generate results tables and figures |

## Methodology

### Feature Extraction

138 statistical features (mean, std, min, max, median) computed from NMEA sentence fields including position deltas, SOG discrepancies, COG deviations, altitude, HDOP, satellite C/N₀, clock drift, and inter-receiver distances.

### Preprocessing

1. **Train/test split by index** (indices 0–15 train, 16–19 test) — prevents data leakage
2. Drop zero/near-zero variance features
3. Median imputation for NaN values
4. Log1p transform for highly skewed features (|skew| > 5)
5. Winsorization at 1st/99th percentile
6. Correlation removal (r > 0.95)
7. StandardScaler (for SVM only; tree models use unscaled data)

### Classifiers

| Classifier | Key Hyperparameters |
|------------|-------------------|
| Random Forest | 500 trees, balanced class weights |
| XGBoost | 500 rounds, max_depth=6, lr=0.1 |
| LightGBM | 500 rounds, max_depth=6, lr=0.1 |
| SVM (RBF) | C=10, subsampled to 10K for training |
| MANA (baseline) | 6 rule-based methods with default thresholds |

### Evaluation

- Overall: Accuracy, Precision, Recall, F1, AUC-ROC
- Per-scenario (A1, A2, A3) breakdown
- Per-parameter heatmaps (19×19 grid per scenario)
- Detection-cliff curves (F1 vs. parameter value)
- McNemar's test for statistical significance

## Usage

### Prerequisites

```bash
# Create conda environment
conda env create -f environment.yml
conda activate marsim_env
conda install scipy statsmodels

# Install MANA
cd mana
python setup.py install
cd ..
```

### Running the Pipeline

```bash
# Step 1: Extract features from pcap files (~10 min)
python extract_features.py

# Step 2: Run MANA baseline on all pcap files (~30-90 min)
python run_mana_baseline.py

# Step 3: Train ML classifiers and evaluate (~5-10 min)
python train_classifiers.py          # full pipeline
python train_classifiers.py --skip-svm  # skip SVM (faster)
```

### Output

All results are saved to `results/`:

**Tables (CSV):** `overall_results.csv`, `per_scenario_results.csv`, `a3_heatmap_data.csv`, `feature_importance.csv`, `confusion_matrices.csv`, `mcnemar_results.csv`, and more.

**Figures (PNG, 300 DPI):** ROC curves, confusion matrices, feature importance bar chart, per-parameter heatmaps, and the key **A3 shift_speed detection-cliff curve**.

## MANA Baseline

[MANA](https://github.com/fkie-cad/mana) (Spravil et al., 2023) is an open-source rule-based GPS spoofing detection framework. We use 6 of its 8 detection methods:

| Method | Abbreviation | Detection Principle |
|--------|-------------|-------------------|
| MultipleReceiversMethod | PDM | Inter-receiver distance anomaly |
| PhysicalSpeedLimitMethod | PCC_sog | SOG exceeds 30 kn |
| PhysicalRateOfTurnLimitMethod | PCC_rot | Rate of turn exceeds 7.5°/s |
| PhysicalHeightLimitMethod | PCC_height | Altitude outside [-1m, +1m] |
| TimeDriftMethod | CDM | Clock drift exceeds threshold |
| CarrierToNoiseDensityMethod | CNM | C/N₀ outside [20, 50] dB-Hz |

Two methods are excluded (`OrbitPositionsMethod`, `PhysicalEnvironmentLimitMethod`) due to missing data files.

## References

- Spravil, J., Hemminghaus, C., von Rechenberg, M., & Padilla, E. (2023). *Detecting Maritime GPS Spoofing Attacks Based on NMEA Sentence Integrity Monitoring.* Journal of Marine Science and Engineering, 11(5), 928. [DOI: 10.3390/jmse11050928](https://doi.org/10.3390/jmse11050928)
- MARSIM Dataset: [Zenodo (DOI: 10.5281/zenodo.8202936)](https://doi.org/10.5281/zenodo.8202936)
- MANA Framework: [github.com/fkie-cad/mana](https://github.com/fkie-cad/mana)

## License

This project uses the MARSIM dataset and MANA framework under their respective licenses. See individual repositories for details.
