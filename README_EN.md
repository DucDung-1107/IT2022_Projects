# IT2022_Projects — ECG Denoising & Stock Trend Classification

This repository contains code for two related classification tasks:

1. **ECG arrhythmia classification (3 classes)**: Normal / Atrial / Other
   - Multiple **denoising strategies** are tested, while keeping the overall data pipeline and classifier comparable.
2. **Stock trend classification (binary up/down)**
   - Weekly labeling + denoising on the price series + feature engineering + ML/DL models.

---

## ECG: pipeline (3-class)

For each ECG record (MIT-BIH formatted as `*.csv` + `*annotations.txt`), the standard pipeline is:

1. **Preprocess signal**
   - baseline removal (median filtering)
   - bandpass filtering
2. **Beat segmentation**
   - extract a fixed window around each annotated beat (BEAT_LEN=180)
3. **Feature extraction**
   - handcrafted beat morphology + wavelet features
   - RR-derived features (RR, BPM, RR_z)
4. **Model**
   - Multi-Input BiLSTM + Attention (waveform branch + RR-sequence branch + handcrafted features)

### Denoising methods implemented/tested

These methods are used as an extra “front-end” to suppress noise:

- **Median filter denoiser** (short-window impulse suppression)
  - module: `src/model/ecg/median_denoiser.py`
  - experiment: `src/notebook/ecg/experiment_ecg_median.ipynb`

- **Adaptive Kalman filter denoiser** (price/latent smoothing)
  - module: `src/model/stock/*` (stock pipeline)
  - experiment: `src/notebook/stock/experiment_stock_kalman.ipynb`

- **ARIMA/SARIMA denoising + hybrid classifier**
  - methodology: `src/methodology/ecg/arima_denoiser.py`

- **Spectral gating denoiser** (STFT-based soft attenuation)
  - methodology: `src/methodology/ecg/spectral_denoiser.py`

- **LSTM Denoising Autoencoder (DAE) + BiLSTM classifier**
  - methodology: `src/methodology/ecg/lstm_denoiser.py`

> Note: some denoisers are implemented as methodology pipelines, while others provide utility functions that notebooks call directly.

---

## Stock: pipeline (binary up/down)

For weekly OHLCV series (e.g., `FPT raw.csv`), the pipeline is:

1. **Resample** to weekly frequency (W-FRI)
2. **Label** using the next-week return threshold (e.g., SIGNAL_THR=2.0)
3. **Denoise** the close price using **Adaptive Kalman Filter** (walk-forward safe)
4. **Feature engineering** on the denoised output (`kf_price`, residuals, and Kalman state scalars)
5. **Models**
   - ensemble (LightGBM + RandomForest)
   - LSTM variants including a DualStream + Cross-Attention architecture

---

## Utilities & metrics

- **ECG utilities**: `src/utils/ecg_denoiser_utils.py`
  - load MIT-BIH CSV + annotations
  - preprocessing (baseline removal, bandpass)
  - beat segmentation & features
  - RR sequence building, normalization, SMOTE
  - evaluation helpers

- **Metrics**: `src/metrics/classification_metrics.py`
  - provides directional (up/down) metric helpers

---

## Quick start

### 1) Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Data locations

- ECG: `src/data/ecg_data/` (MIT-BIH style CSV + `*annotations.txt`)
- Stock: `src/data/stock/` (e.g., `FPT raw.csv`)

### 3) Run the main notebooks

- ECG median denoiser experiment:
  - `src/notebook/ecg/experiment_ecg_median.ipynb`
- Stock Adaptive Kalman + LSTM DualStream experiment:
  - `src/notebook/stock/experiment_stock_kalman.ipynb`

---

## Outputs

Notebooks typically save figures/CSVs under:
- `outputs/` (project-relative), or
- Kaggle-style `/kaggle/working` paths (if you run on Kaggle).

