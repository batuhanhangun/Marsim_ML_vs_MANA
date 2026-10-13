# Classical ML vs. rule-based GPS spoofing detection on MARSIM

Code for the paper *Classical machine learning versus rule-based GPS spoofing detection at the NMEA level: benchmark, generalization limits, and a hybrid architecture*.

Four classifiers (Random Forest, SVM, XGBoost, LightGBM) are compared with the rule-based [MANA](https://github.com/fkie-cad/mana) framework on the [MARSIM](https://doi.org/10.5281/zenodo.8202802) dataset (43,320 labelled NMEA recordings).

## Data

The extracted features and the MANA predictions used in the paper are archived at [10.5281/zenodo.22133265](https://doi.org/10.5281/zenodo.22133265). Place `marsim_features.csv` and `mana_predictions.csv` in `data/`; their hashes are checked against `data/SHA256SUMS` on load.

To re-extract features or run MANA yourself, download MARSIM and point `pcap_dir` at the folder containing `dataset.json` (default `data/marsim_pcaps`), for example with a `config.json` in the repository root:

```json
{"pcap_dir": "/path/to/MARSIM/dataset"}
```

## Installation

```
pip install -r requirements.txt
pip install scapy                                   # feature extraction only
pip install git+https://github.com/fkie-cad/mana    # MANA baseline only
```

## Usage

```
python run_all.py                  # every experiment, figure and table
python run_all.py --no-tune        # skip the hyperparameter search
python run_all.py --no-windows     # skip the pcap-dependent latency experiment
```

Single steps:

```
python -m marsim_ml.run --config baseline     # one experiment (see table below)
python -m marsim_ml.extract --all-windows     # features for the latency experiment
python -m marsim_ml.window_experiment
python -m marsim_ml.plots
python -m marsim_ml.tables
```

Results are written to `results/<config>/`, figures to `figures/`, LaTeX tables to `tables/`.

## Experiments

| config | test set |
|---|---|
| `baseline` | repetition indices 16 to 19 |
| `alt_preprocessing` | same, alternative correlation pruning, no winsorization |
| `heldout_speed_low`, `heldout_speed_mid` | A3 shift speeds {0, 4, 8, 12} or {28, ..., 40} unseen in training |
| `heldout_speed_interleaved`, `heldout_angle_interleaved`, `heldout_distance_A2` | alternate parameter values unseen in training |
| `heldout_scenario_A1`, `_A2`, `_A3` | one attack scenario unseen in training |
| `repetition_cv` | five folds over repetition indices |
| `tuned` | repetition split, randomized hyperparameter search |

The latency experiment (`window_experiment`) uses windows starting at attack onset, 60 s into each recording.

## Layout

```
marsim_ml/       extraction, splits, preprocessing, models, evaluation, plots, tables
configs/         MANA method configuration
data/            input CSVs and their SHA-256 hashes
run_all.py       end-to-end driver
```
