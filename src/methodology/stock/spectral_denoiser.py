"""
Spectral Gating Feature Module for Stock Trend Classification.

This module implements the spectral feature extraction component used in the
FPT weekly trend classification experiment.

Repository path:

    src/methodology/stock/spectral_denoiser.py

The module is intentionally a feature-extraction / denoising component, not a
standalone classifier. It is designed to be combined with the existing stock
backbones:

    - Tree Ensemble
    - BiLSTM + Attention
    - Dual-stream BiLSTM + Cross-Attention

The extracted features are causal: feature values at time t are computed only
from historical close prices before time t, matching the baseline shift(1)
convention in the stock notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import stft, istft


EPS = 1e-12


@dataclass(frozen=True)
class StockSpectralGatingConfig:
    """
    Configuration for causal stock spectral features.

    Parameters
    ----------
    lookback:
        Number of past weekly observations used to compute spectral features.
        A single setting is used by default. No W8/W13 split is used here.
    nperseg:
        STFT segment length for the historical window.
    noverlap:
        STFT overlap.
    noise_percentile:
        Percentile used to estimate frequency-wise noise floor.
    gate_strength:
        Controls attenuation strength.
    mask_floor:
        Minimum mask value to avoid overly aggressive suppression.
    """
    lookback: int = 26
    nperseg: int = 16
    noverlap: int = 12
    noise_percentile: float = 25.0
    gate_strength: float = 1.15
    mask_floor: float = 0.05


DEFAULT_SPECTRAL_STREAM_COLUMNS = [
    "spectral_ret",
    "spectral_trend_grad",
    "spectral_resid_abs",
    "spectral_entropy",
]


def resolve_fpt_data_path(
    candidates: list[str | Path] | None = None,
    filename_hint: str = "FPT raw.csv",
) -> str:
    """
    Resolve FPT raw CSV path robustly on Kaggle or local runtime.

    This prevents FileNotFoundError when the Kaggle dataset is mounted under a
    different folder name.
    """
    default_candidates = [
        "/kaggle/input/datasets/cdnghnam/fptckhon/FPT raw.csv",
        "/kaggle/input/datasets/kkcom28/dataset/FPT raw.csv",
        "/kaggle/input/fpt-raw/FPT raw.csv",
        "/kaggle/input/fpt/FPT raw.csv",
        "/mnt/data/FPT raw.csv",
        "FPT raw.csv",
    ]

    candidates = candidates or default_candidates

    for p in candidates:
        path = Path(p)
        if path.exists():
            return str(path)

    scan_roots = [Path("/kaggle/input"), Path("/mnt/data"), Path(".")]
    matches: list[Path] = []

    for root in scan_roots:
        if root.exists():
            matches.extend(root.rglob(filename_hint))

    if not matches:
        for root in scan_roots:
            if root.exists():
                for p in root.rglob("*.csv"):
                    name = p.name.lower()
                    if "fpt" in name and "raw" in name:
                        matches.append(p)

    if matches:
        matches = sorted(matches, key=lambda x: len(str(x)))
        return str(matches[0])

    raise FileNotFoundError(
        "Could not find FPT raw.csv. Attach the file to Kaggle input or pass an explicit path."
    )


def spectral_entropy(power: np.ndarray) -> float:
    """Normalised spectral entropy of a nonnegative power spectrum."""
    power = np.asarray(power, dtype=float)
    p = power / (np.sum(power) + EPS)
    return float(-np.sum(p * np.log(p + EPS)) / np.log(len(p) + EPS))


def band_energy_ratio(freqs: np.ndarray, power: np.ndarray, lo: float, hi: float) -> float:
    """Energy ratio within a frequency band."""
    mask = (freqs >= lo) & (freqs < hi)
    return float(np.sum(power[mask]) / (np.sum(power) + EPS))


def stft_soft_gate_1d(
    x: np.ndarray,
    cfg: StockSpectralGatingConfig | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Apply soft spectral gating to one historical price window.

    The window is normally instance-normalised before this function is called,
    but the function also works on any 1D numeric sequence.
    """
    cfg = cfg or StockSpectralGatingConfig()
    x = np.asarray(x, dtype=float).reshape(-1)

    if len(x) < 4:
        return x.copy(), {"mask_mean": 1.0, "suppressed_ratio": 0.0}

    nper = min(int(cfg.nperseg), len(x))
    nover = min(int(cfg.noverlap), max(0, nper - 1))

    _, _, zxx = stft(
        x,
        nperseg=nper,
        noverlap=nover,
        boundary="zeros",
        padded=True,
    )

    magnitude = np.abs(zxx)
    noise_floor = np.percentile(magnitude, cfg.noise_percentile, axis=1, keepdims=True)

    mask = 1.0 - cfg.gate_strength * noise_floor / (magnitude + EPS)
    mask = np.clip(mask, cfg.mask_floor, 1.0)

    _, rec = istft(
        zxx * mask,
        nperseg=nper,
        noverlap=nover,
        input_onesided=True,
        boundary=True,
    )

    if len(rec) < len(x):
        rec = np.pad(rec, (0, len(x) - len(rec)), mode="edge")
    rec = rec[:len(x)]

    stats = {
        "mask_mean": float(np.mean(mask)),
        "suppressed_ratio": float(np.mean(mask < 0.5)),
    }
    return np.asarray(rec, dtype=float), stats


def spectral_features_from_history(
    close_history: np.ndarray,
    cfg: StockSpectralGatingConfig | None = None,
) -> dict[str, float]:
    """
    Extract spectral features from one historical close-price window.

    The input should contain only observations available before the prediction
    time. For week t, pass close prices from earlier weeks only.
    """
    cfg = cfg or StockSpectralGatingConfig()
    close_history = np.asarray(close_history, dtype=float).reshape(-1)

    if len(close_history) < 4:
        last = float(close_history[-1]) if len(close_history) else 0.0
        return {
            "spectral_denoised": last,
            "spectral_resid": 0.0,
            "spectral_resid_abs": 0.0,
            "spectral_trend_grad": 0.0,
            "spectral_ret": 0.0,
            "spectral_entropy": 0.0,
            "spectral_dom_freq": 0.0,
            "spectral_low_ratio": 0.0,
            "spectral_mid_ratio": 0.0,
            "spectral_high_ratio": 0.0,
            "spectral_mask_mean": 1.0,
            "spectral_suppressed_ratio": 0.0,
        }

    mean = close_history.mean()
    std = close_history.std() + EPS
    normalized = (close_history - mean) / std

    den_norm, gate_stats = stft_soft_gate_1d(normalized, cfg)
    denoised = den_norm * std + mean

    residual = close_history - denoised
    trend_grad = np.gradient(denoised)[-1] if len(denoised) >= 2 else 0.0

    if len(denoised) >= 2:
        spectral_ret = (denoised[-1] - denoised[-2]) / (abs(denoised[-2]) + EPS) * 100
    else:
        spectral_ret = 0.0

    log_close = np.log(np.maximum(close_history, EPS))
    returns = np.diff(log_close)

    if len(returns) < 2:
        returns = np.array([0.0, 0.0])

    spectrum = np.fft.rfft(returns - returns.mean())
    freqs = np.fft.rfftfreq(len(returns), d=1.0)
    power = np.abs(spectrum) ** 2

    if len(power) <= 1 or np.sum(power) <= EPS:
        dom_freq = 0.0
    else:
        dom_idx = 1 + np.argmax(power[1:])
        dom_freq = float(freqs[dom_idx])

    return {
        "spectral_denoised": float(denoised[-1]),
        "spectral_resid": float(residual[-1]),
        "spectral_resid_abs": float(abs(residual[-1])),
        "spectral_trend_grad": float(trend_grad),
        "spectral_ret": float(spectral_ret),
        "spectral_entropy": spectral_entropy(power),
        "spectral_dom_freq": dom_freq,
        "spectral_low_ratio": band_energy_ratio(freqs, power, 0.00, 0.10),
        "spectral_mid_ratio": band_energy_ratio(freqs, power, 0.10, 0.25),
        "spectral_high_ratio": band_energy_ratio(freqs, power, 0.25, 0.51),
        "spectral_mask_mean": float(gate_stats["mask_mean"]),
        "spectral_suppressed_ratio": float(gate_stats["suppressed_ratio"]),
    }


def build_causal_spectral_features(
    close: np.ndarray | pd.Series,
    cfg: StockSpectralGatingConfig | None = None,
) -> pd.DataFrame:
    """
    Build causal spectral features for an entire weekly close-price series.

    For each row t:
        features[t] = f(close[max(0, t-lookback):t])

    The current close value close[t] is excluded to match the baseline feature
    convention and prevent look-ahead leakage.
    """
    cfg = cfg or StockSpectralGatingConfig()
    close_arr = np.asarray(close, dtype=float).reshape(-1)

    rows = []
    for t in range(len(close_arr)):
        start = max(0, t - cfg.lookback)
        history = close_arr[start:t]
        rows.append(spectral_features_from_history(history, cfg))

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().fillna(0.0)
    return df.astype(np.float32)


def append_causal_spectral_features(
    weekly_df: pd.DataFrame,
    close_col: str = "close",
    cfg: StockSpectralGatingConfig | None = None,
) -> pd.DataFrame:
    """
    Return a copy of weekly_df with causal spectral features appended.
    """
    if close_col not in weekly_df.columns:
        raise ValueError(f"Column '{close_col}' not found in weekly_df.")

    spec = build_causal_spectral_features(weekly_df[close_col].values, cfg=cfg)
    out = weekly_df.reset_index(drop=True).copy()
    return pd.concat([out, spec], axis=1)


def get_spectral_stream_columns(
    df: pd.DataFrame,
    requested: list[str] | None = None,
) -> list[str]:
    """
    Select compact spectral columns for the BiLSTM stream-B input.
    """
    requested = requested or DEFAULT_SPECTRAL_STREAM_COLUMNS
    return [c for c in requested if c in df.columns]


def build_spectral_report_table(
    results: pd.DataFrame,
    phase_col: str = "phase",
) -> pd.DataFrame:
    """
    Format stock spectral result rows for the paper.

    Expected input columns:
        phase, accuracy, precision, recall, f1

    Output columns:
        Method, Acc., Prec., Rec., F1
    """
    required = [phase_col, "accuracy", "precision", "recall", "f1"]
    missing = [c for c in required if c not in results.columns]
    if missing:
        raise ValueError(f"Missing result columns: {missing}")

    out = results[required].copy()
    out = out.rename(
        columns={
            phase_col: "Method",
            "accuracy": "Acc.",
            "precision": "Prec.",
            "recall": "Rec.",
            "f1": "F1",
        }
    )

    for c in ["Acc.", "Prec.", "Rec.", "F1"]:
        out[c] = out[c].astype(float).round(4)

    return out
