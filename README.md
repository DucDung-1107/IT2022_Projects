# Project folders — Quick overview

This document describes the project folder structure and the purpose of the main directories.

## Overview

The project contains code for experiments, datasets, models, and documentation (reports and slides).

## Metrics (`metrics/`)

Standardised evaluation metrics used across all classification experiments.

| File | Description |
|------|-------------|
| `classification_metrics.py` | Directional classification metrics: `evaluate_directional_metrics()` returns accuracy, precision, recall, and F1-score (macro-averaged) for binary directional labels (`up`/`down`). |

Additional evaluation functions are available inside each project's utility module (see **Utils** below), which compute a fuller suite including:
- **Accuracy** — overall correct classification rate.
- **F1 (macro)** — harmonic mean of precision & recall, averaged per class.
- **F2 (macro)** — F-beta with β=2, weighting recall higher than precision.
- **Precision & Recall (macro)** — per-class averages.
- **AUC-ROC (macro, OvR)** — area under the ROC curve, one-vs-rest macro average.

## Utils (`utils/`)

Reusable helper modules shared across notebooks and methodology scripts.

| File | Description |
|------|-------------|
| `lstm_denoiser_utils.py` | Stock LSTM denoiser helpers: `seed_everything()`, `resample_weekly()`, `make_sequences()`, `make_rolling_sequences()` for time-series windowing. |
| `ecg_denoiser_utils.py` | ECG denoiser helpers (MIT-BIH dataset): <br>• **Constants** — `FS=360`, `BEAT_LEN=180`, `LABEL_MAP`, `CLASSES_ORDER`, `TRAIN_PATIENTS`, `TEST_PATIENTS`.<br>• **Data loading** — `load_record()`, `build_per_patient_beats()`, `build_dataset()`, `beats_to_arrays()`.<br>• **Preprocessing** — `remove_baseline()` (median filter), `bandpass_filter()`, `preprocess_signal()`.<br>• **Feature extraction** — `compute_rr_bpm_zscore()`, `feat_morph()`, `feat_wavelet()`, `extract_handcrafted()`.<br>• **Sequence building** — `build_rr_sequences()` for per-patient RR-interval windows.<br>• **Normalisation** — `zscore_waveform()` (per-beat), `standardize_split()` (RR-sequence).<br>• **SMOTE** — `apply_smote_multi()` for multi-input (waveform + features + sequence) oversampling.<br>• **Evaluation** — `evaluate_predictions()` (accuracy, F1, F2, AUC-ROC), `print_metrics()`. |

## Folder structure

- **assets/**: place PDF reports and slide decks (PPTX, PDF).
- **metrics/**: evaluation metric implementations for classification tasks.
- **utils/**: shared utility functions (data loading, preprocessing, model helpers) reused across experiments.
- **src/**: main source directory containing:
  - **src/data/**: raw data files (CSV, annotation text files, etc.).
  - **src/methodology/**: implementations of methods and denoisers used in experiments.
  - **src/model/**: backbone model code used for inference and related utilities.
  - **src/notebook/**: Jupyter Notebooks (.ipynb) for running experiments and reproducing results.
  - **src/preprocess/**: data preprocessing scripts (cleaning, normalization, signal extraction).
  - **src/eda/**: scripts and notebooks for exploratory data analysis and visualization.

  ## Naming conventions and subfolders

  Follow these conventions to keep the project consistent:

  - **Folder layout**: For each of the following folders create two subfolders: `ecg/` and `stock/`.
    - `src/eda/ecg/`, `src/eda/stock/`
    - `src/methodology/ecg/`, `src/methodology/stock/`
    - `src/model/ecg/`, `src/model/stock/`
    - `src/notebook/ecg/`, `src/notebook/stock/`
    - `src/preprocess/ecg/`, `src/preprocess/stock/`

  - **File naming examples**:
    - Method implementations: `method.py` (replace with descriptive names like `denoiser_wavelet.py`, `method_bayesian.py`).
    - Notebooks: use `experiment_<data>_<method>.ipynb`, e.g. `experiment_ecg_wavelet.ipynb` or `experiment_stock_lstm.ipynb`.
    - Models: `model_<backbone>.py` or `backbone_<name>.py`, e.g. `model_unet.py`.
    - Preprocessing scripts: `preprocess_<step>.py`, e.g. `preprocess_bandpass.py`.
    - EDA scripts/notebooks: `eda_<data>_<view>.py` or `eda_<data>_<view>.ipynb`, e.g. `eda_ecg_overview.ipynb`.

  Keeping this structure makes it easier to find code and reproduce experiments.

## Quick start

- Open notebooks in `src/notebook/` with Jupyter Notebook or JupyterLab to run experiments.
- If a `requirements.txt` or `environment.yml` exists, create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

- Put final report and slides into `assets/` (PDF, PPTX, etc.).

## Suggested next steps

- Add a `requirements.txt` if one does not exist to make experiments reproducible.
- Review notebooks in `src/notebook/` and add run instructions if needed.

---

If you want the README expanded (badges, example runs, CI instructions, or an experiments template), tell me what to include.