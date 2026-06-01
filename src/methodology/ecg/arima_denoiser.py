"""
ARIMA / SARIMA Denoising Methodology for ECG Arrhythmia Classification.

Extracted from `ex_ecg_arima_denoiser.ipynb`:
    - Phase 1: Pure ARIMA/SARIMA likelihood-based classifier (RR-sequence)
    - Phase 2: ARIMA-denoised waveform → Multi-Input BiLSTM Classifier (Hybrid)

This module provides the high-level pipeline for loading ECG data, fitting ARIMA
denoisers, and evaluating both phases.

Usage:
    from src.methodology.ecg.arima_denoiser import run_arima_denoiser_pipeline

    results = run_arima_denoiser_pipeline(extract_dir="/path/to/mitbih-database")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks
from tensorflow.keras.utils import to_categorical

from ...model.ecg.arima_denoiser import (
    ARIMA_ORDER_BEAT,
    SARIMA_ORDER_RR,
    SARIMA_SEASONAL,
    arima_denoise_batch,
    fit_class_sarima,
    predict_pure_sarima,
)
from ...model.ecg.lstm_denoiser import AttentionLayer, build_multi_input_classifier
from ...utils.ecg_denoiser_utils import (
    CLASSES_ORDER,
    N_CLASSES,
    apply_smote_multi,
    beats_to_arrays,
    build_dataset,
    build_rr_sequences,
    evaluate_predictions,
    print_metrics,
    seed_everything,
    standardize_split,
    zscore_waveform,
)


@dataclass
class ARIMAConfig:
    """Configuration for ARIMA denoiser methodology."""

    seq_len: int = 10
    lstm_units: int = 64
    lstm_dropout: float = 0.3
    lstm_epochs: int = 40
    lstm_batch: int = 128
    lstm_lr: float = 1e-3
    seed: int = 42
    validation_split: float = 0.15

    arima_order_beat: tuple[int, int, int] = ARIMA_ORDER_BEAT
    sarima_order_rr: tuple[int, int, int] = SARIMA_ORDER_RR
    sarima_seasonal: tuple[int, int, int, int] = SARIMA_SEASONAL

    palette: dict = field(
        default_factory=lambda: {
            "P1_PureARIMA": "#6A1B9A",
            "P2_ARIMA_LSTM": "#1565C0",
        }
    )


def run_arima_denoiser_pipeline(
    extract_dir: str,
    cfg: ARIMAConfig | None = None,
    out_dir: str = "/kaggle/working",
) -> dict:
    """
    Full ARIMA/SARIMA denoiser pipeline for ECG arrhythmia classification.

    Parameters
    ----------
    extract_dir : str
        Path to the MIT-BIH database (containing .csv + annotations.txt files).
    cfg : ARIMAConfig or None
        Configuration object. Uses defaults if None.
    out_dir : str
        Directory to save outputs (models, CSV summary, figures).

    Returns
    -------
    dict
        Dictionary with phases, summary DataFrame, and all results.
    """
    if cfg is None:
        cfg = ARIMAConfig()

    seed_everything(cfg.seed)
    tf.random.set_seed(cfg.seed)
    os.makedirs(out_dir, exist_ok=True)

    BEAT_LEN = 180  # WB+WA = 90+90

    # =========================================================================
    # 1. DATA LOADING
    # =========================================================================
    print("\n" + "=" * 72)
    print("  LOADING DATA — Patient-wise split, 3-class (Normal/Atrial/Other)")
    print("=" * 72)

    train_beats, test_beats = build_dataset(extract_dir)
    print(f"\n  Total train beats : {len(train_beats)}")
    print(f"  Total test  beats : {len(test_beats)}")

    X_w_tr, X_f_tr, X_rr_tr, y_tr_raw, pid_tr = beats_to_arrays(train_beats)
    X_w_te, X_f_te, X_rr_te, y_te_raw, pid_te = beats_to_arrays(test_beats)

    le = LabelEncoder()
    le.fit(CLASSES_ORDER)
    y_tr_enc = le.transform(y_tr_raw)
    y_te_enc = le.transform(y_te_raw)

    print(f"\n  Classes     : {list(le.classes_)}")
    print(f"  Train dist  : {dict(zip(*np.unique(y_tr_raw, return_counts=True)))}")
    print(f"  Test  dist  : {dict(zip(*np.unique(y_te_raw, return_counts=True)))}")

    # =========================================================================
    # 2. RR SEQUENCE BUILDING
    # =========================================================================
    print("\n  Building RR sequences (10-beat windows per patient) ...")
    X_seq_tr = build_rr_sequences(X_rr_tr, pid_tr, cfg.seq_len)
    X_seq_te = build_rr_sequences(X_rr_te, pid_te, cfg.seq_len)
    print(f"  Sequence shape — train: {X_seq_tr.shape}, test: {X_seq_te.shape}")

    # =========================================================================
    # 3. NORMALIZATION
    # =========================================================================
    X_w_tr_z = zscore_waveform(X_w_tr)
    X_w_te_z = zscore_waveform(X_w_te)

    sc_f = StandardScaler()
    X_f_tr_s = sc_f.fit_transform(X_f_tr)
    X_f_te_s = sc_f.transform(X_f_te)

    _, X_seq_tr_s, X_seq_te_s = standardize_split(X_seq_tr, X_seq_te)

    # =========================================================================
    # 4. PHASE 1 — PURE ARIMA/SARIMA CLASSIFIER (Likelihood-based)
    # =========================================================================
    print("\n" + "=" * 72)
    print("  PHASE 1 — PURE SARIMA likelihood classifier (RR-sequence only)")
    print("=" * 72)

    # Fit 1 SARIMA per class (using raw RR sequences)
    print("\n  Fitting one SARIMA per class on RR-sequence ...")
    class_models: dict[int, object] = {}
    for c, cname in enumerate(CLASSES_ORDER):
        mask = y_tr_enc == c
        if mask.sum() < 20:
            print(f"    [skip] Class {cname}: too few samples ({mask.sum()})")
            class_models[c] = None
            continue
        seqs_c = X_seq_tr[mask]
        print(f"    Fitting class {cname} ({mask.sum()} sequences) ...")
        class_models[c] = fit_class_sarima(
            seqs_c,
            order=cfg.sarima_order_rr,
            seasonal_order=cfg.sarima_seasonal,
        )

    print("\n  Predicting on test set with pure SARIMA ...")
    yp_p1, yprob_p1 = predict_pure_sarima(X_seq_te, class_models, N_CLASSES)

    m_p1 = evaluate_predictions(y_te_enc, yp_p1, yprob_p1, N_CLASSES)
    m_p1.update(
        {
            "phase": "P1_PureARIMA",
            "y_true": y_te_enc,
            "y_pred": yp_p1,
            "y_proba": yprob_p1,
            "history": None,
        }
    )
    print_metrics("P1_PureARIMA", m_p1)

    # =========================================================================
    # 5. PHASE 2 — ARIMA-DENOISED → BiLSTM HYBRID
    # =========================================================================
    print("\n" + "=" * 72)
    print("  PHASE 2 — ARIMA-denoised → Multi-Input BiLSTM Classifier (Hybrid)")
    print("=" * 72)

    print("\n  Applying ARIMA denoising on train set (this may take a while) ...")
    X_w_tr_arima = arima_denoise_batch(X_w_tr_z, order=cfg.arima_order_beat)
    print("\n  Applying ARIMA denoising on test set ...")
    X_w_te_arima = arima_denoise_batch(X_w_te_z, order=cfg.arima_order_beat)

    # SMOTE on ARIMA-denoised data
    print("\n  Applying SMOTE on ARIMA-denoised data ...")
    X_w_tr_sm, X_f_tr_sm, X_seq_tr_sm, y_tr_sm = apply_smote_multi(
        X_w_tr_arima, X_f_tr_s, X_seq_tr_s, y_tr_enc, cfg.seed
    )
    print(f"  Train after SMOTE: {len(y_tr_sm)}")
    print(f"  Class dist: {dict(zip(*np.unique(y_tr_sm, return_counts=True)))}")

    # Build and train classifier
    tf.keras.backend.clear_session()
    classifier_p2 = build_multi_input_classifier(
        beat_len=BEAT_LEN,
        n_feat=X_f_tr_sm.shape[1],
        seq_len=cfg.seq_len,
        seq_chan=3,
        n_classes=N_CLASSES,
        lstm_units=cfg.lstm_units,
        lstm_dropout=cfg.lstm_dropout,
        lr=cfg.lstm_lr,
    )

    cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y_tr_sm)
    class_weight_dict = {i: w for i, w in enumerate(cw)}

    cb2 = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    print(f"\n  Training Phase 2 ({cfg.lstm_epochs} epochs max) ...")
    hist_p2 = classifier_p2.fit(
        [X_w_tr_sm, X_seq_tr_sm, X_f_tr_sm],
        to_categorical(y_tr_sm, N_CLASSES),
        validation_split=cfg.validation_split,
        epochs=cfg.lstm_epochs,
        batch_size=cfg.lstm_batch,
        callbacks=cb2,
        class_weight=class_weight_dict,
        verbose=2,
    )

    yprob_p2 = classifier_p2.predict(
        [X_w_te_arima, X_seq_te_s, X_f_te_s],
        batch_size=cfg.lstm_batch,
        verbose=0,
    )
    yp_p2 = yprob_p2.argmax(axis=1)

    m_p2 = evaluate_predictions(y_te_enc, yp_p2, yprob_p2, N_CLASSES)
    m_p2.update(
        {
            "phase": "P2_ARIMA_LSTM",
            "y_true": y_te_enc,
            "y_pred": yp_p2,
            "y_proba": yprob_p2,
            "history": hist_p2.history,
        }
    )
    print_metrics("P2_ARIMA_LSTM", m_p2)

    classifier_p2.save(f"{out_dir}/p2_arima_lstm_classifier.keras")

    # =========================================================================
    # 6. SUMMARY
    # =========================================================================
    import pandas as pd

    all_phases = [m_p1, m_p2]
    rows = [
        {
            "phase": m["phase"],
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "f2": round(m["f2"], 4),
            "auc_roc": round(m["auc_roc"], 4),
        }
        for m in all_phases
    ]
    summary_df = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("  CROSS-PHASE SUMMARY")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(f"{out_dir}/phase_summary_arima.csv", index=False)

    best = max(all_phases, key=lambda m: m["f1"])
    print(f"\n  BEST PHASE : {best['phase']}")
    print(f"  F1 macro   = {best['f1']:.4f}")
    print(f"  ΔF1 (P2−P1) = {m_p2['f1'] - m_p1['f1']:+.4f}")

    return {
        "phases": all_phases,
        "summary": summary_df,
        "best": best,
        "classifier_p2": classifier_p2,
        "label_encoder": le,
        "y_te_enc": y_te_enc,
    }