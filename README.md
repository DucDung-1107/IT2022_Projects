# IT2022_Projects — ECG Denoising + Arrhythmia Classification & Stock Trend Classification


Tổng quan dự án: mã nguồn cho (1) **phân loại nhịp tim ECG** với nhiều phương pháp khử nhiễu (median / Kalman / ARIMA / spectral gating / LSTM DAE) và (2) **phân loại xu hướng cổ phiếu theo tuần** bằng các mô hình học máy sâu (ensemble + LSTM + Cross-Attention) dựa trên chuỗi thời gian và đặc trưng denoise.

## Overview

- **Experiments / Notebooks**: chạy và so sánh các ablation/pipeline khác nhau.
- **src/model/**: các backbone model (ví dụ ECG multi-input BiLSTM + attention, stock backbone + mô-đun denoising).
- **src/utils/**: utilities chung để đọc dữ liệu, trích đặc trưng, normalize, và tính metrics.
- **src/methodology/**: các pipeline/method denoising (median, Kalman, ARIMA, spectral gating, ...).

## Project folders — Quick overview

- **src/notebook/**: Jupyter Notebooks (.ipynb) dùng để chạy thí nghiệm.
- **src/data/**: dữ liệu thô (CSV + annotation txt cho ECG; CSV cho stock).
- **src/utils/**: hàm chung dùng bởi notebooks/methodology.

## Metrics (`metrics/`)

Standardised evaluation metrics dùng xuyên suốt các bài toán phân loại.

| File | Description |
|------|-------------|
| `classification_metrics.py` | Directional metrics: `evaluate_directional_metrics()` trả về accuracy/precision/recall/F1 (macro) cho nhãn nhị phân (`up`/`down`). |

Ngoài ra, trong `src/utils/ecg_denoiser_utils.py` có thêm:
- Accuracy
- F1 (macro)
- F2 (macro)
- Precision/Recall (macro)
- AUC-ROC (macro, OvR)

## Utils (`src/utils/`)

| File | Description |
|------|-------------|
| `lstm_denoiser_utils.py` | Stock: `seed_everything()`, `resample_weekly()`, `make_sequences()`, `make_rolling_sequences()` |
| `ecg_denoiser_utils.py` | ECG: constants (FS=360, BEAT_LEN=180, ...) + load/preprocess/features/sequence + SMOTE + evaluation |

## ECG pipeline (mức ý tưởng)

Trong ECG, mỗi ablation thường giữ nguyên phần dữ liệu nhịp tim và classifier, chỉ thay đổi “stream waveform” theo phương pháp khử nhiễu.

- **Chia dữ liệu theo patient** (train/test): dùng MIT-BIH CSV + `*annotations.txt`.
- **Preprocess**: remove baseline (median filter) + bandpass.
- **Segment beat**: lấy cửa sổ quanh R-peak (BEAT_LEN=180).
- **Features**: RR/BPM/RR_z + handcrafted waveform features.
- **Model**: Multi-Input BiLSTM + Attention (waveform + RR-sequence + handcrafted features) hoặc các ablation/denoising variants.

Ví dụ denoising methods nằm trong:
- `src/model/ecg/` (denoiser + model backbone)
- `src/methodology/ecg/` (pipeline phương pháp)
- `src/notebook/ecg/` (thí nghiệm)

## Stock pipeline (mức ý tưởng)

Trong Stock trend classification (binary `up`/`down`) pipeline gồm:
- **Resample theo tuần** (W-FRI)
- **Label** từ weekly return với ngưỡng % (ví dụ SIGNAL_THR=2.0)
- **Denoise** chuỗi close theo Adaptive Kalman Filter (walk-forward an toàn)
- **Feature engineering** trên chuỗi đã denoise (kf_price)
- **Model**: ensemble (LightGBM + RandomForest) và/hoặc LSTM DualStream + Cross-Attention

## Naming conventions & subfolders

- Mỗi thư mục theo hướng dữ liệu (ecg/stock) có cặp đối xứng:
  - `src/eda/ecg/` , `src/eda/stock/`
  - `src/methodology/ecg/` , `src/methodology/stock/`
  - `src/model/ecg/` , `src/model/stock/`
  - `src/notebook/ecg/` , `src/notebook/stock/`
  - `src/preprocess/ecg/` , `src/preprocess/stock/`

## Quick start

### 1) Cài môi trường

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Chuẩn bị dữ liệu

- ECG: nằm sẵn tại `src/data/ecg_data/` (MIT-BIH CSV + `*annotations.txt`).
- Stock: nằm sẵn tại `src/data/stock/` (ví dụ `FPT raw.csv`).

### 3) Chạy notebook chính

- ECG median denoiser: `src/notebook/ecg/experiment_ecg_median.ipynb`
- Stock Adaptive Kalman + LSTM DualStream: `src/notebook/stock/experiment_stock_kalman.ipynb`

Thông thường output (hình/CSV summary) sẽ được lưu trong thư mục `outputs/` hoặc trong `/kaggle/working` tuỳ notebook.

- Final report & slides: đặt vào `assets/` (PDF, PPTX, ...).

## Suggested next steps

- Bổ sung/chuẩn hoá hướng dẫn chạy cho tất cả notebooks (nên có một mục “Run config” cho từng notebook).
- Xem thêm các notebook trong `src/notebook/ecg/` và `src/notebook/stock/` để mở rộng ablation.

