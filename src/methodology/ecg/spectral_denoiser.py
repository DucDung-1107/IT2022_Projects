"""
Spectral Gating Denoiser for ECG signals.

This module contains the spectral denoising utilities used in the MIT-BIH ECG
experiments. It is designed to fit the repository structure:

    src/methodology/ecg/spectral_denoiser.py

The module supports two experimental settings used in the report:

    1. Beat-level spectral gating:
        Median + bandpass filtering
        -> R-peak segmentation
        -> STFT spectral gating on each 180-sample beat

    2. Record-level spectral gating:
        Median + bandpass filtering
        -> STFT spectral gating on the full ECG record
        -> R-peak segmentation

The functions in this file do not define the downstream classifier. They only
perform denoising / feature extraction so that the same MultiInput BiLSTM
Attention backbone can be reused across baseline and denoising methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import stft, istft


EPS = 1e-12


@dataclass(frozen=True)
class ECGSpectralGatingConfig:
    """
    Configuration for STFT-based spectral gating on ECG signals.

    Parameters
    ----------
    fs:
        Sampling frequency of the ECG signal. MIT-BIH uses 360 Hz.
    record_nperseg:
        STFT segment length for record-level denoising.
    record_noverlap:
        STFT overlap for record-level denoising.
    beat_nperseg:
        STFT segment length for beat-level denoising.
    beat_noverlap:
        STFT overlap for beat-level denoising.
    noise_percentile:
        Percentile used to estimate the frequency-wise noise floor.
    gate_strength:
        Controls the aggressiveness of the soft attenuation mask.
    mask_floor:
        Minimum allowed mask value. A nonzero floor prevents excessive
        morphology distortion.
    """
    fs: int = 360
    record_nperseg: int = 256
    record_noverlap: int = 192
    beat_nperseg: int = 64
    beat_noverlap: int = 48
    noise_percentile: float = 25.0
    gate_strength: float = 1.15
    mask_floor: float = 0.05


def _as_1d_float(x: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Input signal must contain at least one sample.")
    return arr


def _fix_reconstruction_length(x_rec: np.ndarray, n: int) -> np.ndarray:
    """Pad or trim iSTFT output to match the original signal length."""
    if len(x_rec) < n:
        x_rec = np.pad(x_rec, (0, n - len(x_rec)), mode="edge")
    return x_rec[:n]


def stft_soft_spectral_gate(
    x: np.ndarray | list[float],
    *,
    fs: int = 360,
    nperseg: int = 256,
    noverlap: int = 192,
    noise_percentile: float = 25.0,
    gate_strength: float = 1.15,
    mask_floor: float = 0.05,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float], np.ndarray]:
    """
    Apply soft STFT spectral gating to a 1D ECG signal.

    The method computes an STFT, estimates a frequency-wise noise floor using a
    low percentile of magnitudes, applies a soft attenuation mask, and reconstructs
    the denoised signal using inverse STFT.

    Parameters
    ----------
    x:
        Input ECG signal or ECG beat.
    fs:
        Sampling frequency.
    nperseg:
        STFT segment length.
    noverlap:
        STFT overlap.
    noise_percentile:
        Percentile used to estimate the noise floor per frequency bin.
    gate_strength:
        Strength of noise attenuation.
    mask_floor:
        Minimum allowed value of the soft mask.
    return_mask:
        If True, also return mask statistics and the full mask matrix.

    Returns
    -------
    denoised:
        Denoised signal with the same length as the input.
    stats:
        Returned only if `return_mask=True`.
    mask:
        Returned only if `return_mask=True`.
    """
    x = _as_1d_float(x)
    n = len(x)

    nper = min(int(nperseg), n)
    nover = min(int(noverlap), max(0, nper - 1))

    _, _, zxx = stft(
        x,
        fs=fs,
        nperseg=nper,
        noverlap=nover,
        boundary="zeros",
        padded=True,
    )

    magnitude = np.abs(zxx)
    noise_floor = np.percentile(magnitude, noise_percentile, axis=1, keepdims=True)

    mask = 1.0 - gate_strength * noise_floor / (magnitude + EPS)
    mask = np.clip(mask, mask_floor, 1.0)

    _, x_rec = istft(
        zxx * mask,
        fs=fs,
        nperseg=nper,
        noverlap=nover,
        input_onesided=True,
        boundary=True,
    )

    denoised = _fix_reconstruction_length(np.asarray(x_rec, dtype=float), n)

    if not return_mask:
        return denoised

    stats = {
        "mask_mean": float(np.mean(mask)),
        "mask_median": float(np.median(mask)),
        "suppressed_ratio": float(np.mean(mask < 0.5)),
        "noise_floor_mean": float(np.mean(noise_floor)),
    }
    return denoised, stats, mask


def denoise_ecg_record(
    signal: np.ndarray | list[float],
    cfg: ECGSpectralGatingConfig | None = None,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    """
    Record-level spectral gating.

    This is the recommended ECG spectral setting in the report:
        preprocessed ECG record -> spectral gating -> segmentation -> classifier.
    """
    cfg = cfg or ECGSpectralGatingConfig()

    if return_stats:
        den, stats, _ = stft_soft_spectral_gate(
            signal,
            fs=cfg.fs,
            nperseg=cfg.record_nperseg,
            noverlap=cfg.record_noverlap,
            noise_percentile=cfg.noise_percentile,
            gate_strength=cfg.gate_strength,
            mask_floor=cfg.mask_floor,
            return_mask=True,
        )
        return den, stats

    return stft_soft_spectral_gate(
        signal,
        fs=cfg.fs,
        nperseg=cfg.record_nperseg,
        noverlap=cfg.record_noverlap,
        noise_percentile=cfg.noise_percentile,
        gate_strength=cfg.gate_strength,
        mask_floor=cfg.mask_floor,
        return_mask=False,
    )


def denoise_ecg_beat(
    beat: np.ndarray | list[float],
    cfg: ECGSpectralGatingConfig | None = None,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    """
    Beat-level spectral gating.

    This setting applies spectral gating to each segmented heartbeat independently.
    It is useful as an ablation but may distort morphology when the beat segment
    is very short.
    """
    cfg = cfg or ECGSpectralGatingConfig()

    if return_stats:
        den, stats, _ = stft_soft_spectral_gate(
            beat,
            fs=cfg.fs,
            nperseg=cfg.beat_nperseg,
            noverlap=cfg.beat_noverlap,
            noise_percentile=cfg.noise_percentile,
            gate_strength=cfg.gate_strength,
            mask_floor=cfg.mask_floor,
            return_mask=True,
        )
        return den, stats

    return stft_soft_spectral_gate(
        beat,
        fs=cfg.fs,
        nperseg=cfg.beat_nperseg,
        noverlap=cfg.beat_noverlap,
        noise_percentile=cfg.noise_percentile,
        gate_strength=cfg.gate_strength,
        mask_floor=cfg.mask_floor,
        return_mask=False,
    )


def denoise_ecg_beats(
    beats: np.ndarray,
    cfg: ECGSpectralGatingConfig | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Apply beat-level spectral gating to a batch of ECG beats.

    Parameters
    ----------
    beats:
        Array of shape (n_beats, beat_len) or (n_beats, beat_len, 1).

    Returns
    -------
    denoised_beats:
        Array with the same shape as the input.
    """
    cfg = cfg or ECGSpectralGatingConfig()
    arr = np.asarray(beats, dtype=float)
    original_shape = arr.shape

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr_2d = arr[..., 0]
        add_channel = True
    elif arr.ndim == 2:
        arr_2d = arr
        add_channel = False
    else:
        raise ValueError("beats must have shape (n_beats, beat_len) or (n_beats, beat_len, 1).")

    den = np.zeros_like(arr_2d, dtype=np.float32)

    iterator: Iterable[int]
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(len(arr_2d)), desc="Beat-level spectral gating")
        except Exception:
            iterator = range(len(arr_2d))
    else:
        iterator = range(len(arr_2d))

    for i in iterator:
        den[i] = denoise_ecg_beat(arr_2d[i], cfg=cfg)

    if add_channel:
        den = den[..., None]

    return den.reshape(original_shape).astype(np.float32)


def build_record_level_denoised_map(
    records: dict[int | str, np.ndarray],
    cfg: ECGSpectralGatingConfig | None = None,
) -> dict[int | str, np.ndarray]:
    """
    Denoise multiple ECG records at record level.

    Parameters
    ----------
    records:
        Dictionary mapping record_id -> preprocessed ECG signal.

    Returns
    -------
    dict
        record_id -> record-level spectrally denoised signal.
    """
    cfg = cfg or ECGSpectralGatingConfig()
    return {rid: denoise_ecg_record(sig, cfg=cfg) for rid, sig in records.items()}


def spectral_residual(x: np.ndarray, denoised: np.ndarray) -> np.ndarray:
    """Return residual/noise proxy: original - denoised."""
    return np.asarray(x, dtype=float) - np.asarray(denoised, dtype=float)


def ecg_spectral_feature_summary(x: np.ndarray, denoised: np.ndarray | None = None) -> dict[str, float]:
    """
    Optional compact feature summary for analysis or debugging.

    This is not required by the ECG backbone in the report, but it is useful for
    inspecting how much the spectral gate changes the signal.
    """
    x = _as_1d_float(x)
    if denoised is None:
        denoised = denoise_ecg_beat(x)
    denoised = _as_1d_float(denoised)
    resid = spectral_residual(x, denoised)

    def energy(v: np.ndarray) -> float:
        return float(np.mean(np.square(v)))

    return {
        "raw_energy": energy(x),
        "denoised_energy": energy(denoised),
        "residual_energy": energy(resid),
        "residual_abs_mean": float(np.mean(np.abs(resid))),
        "residual_std": float(np.std(resid)),
        "energy_ratio_residual_raw": energy(resid) / (energy(x) + EPS),
    }


def build_ecg_spectral_report_table(
    results: pd.DataFrame,
    method_col: str = "phase",
) -> pd.DataFrame:
    """
    Format spectral ECG result rows for report tables.

    Expects columns such as:
        phase, accuracy, precision_macro, recall_macro, f1_macro

    Returns columns:
        Method, Acc., Prec., Rec., F1
    """
    rename_map = {
        "accuracy": "Acc.",
        "precision_macro": "Prec.",
        "recall_macro": "Rec.",
        "f1_macro": "F1",
        "precision": "Prec.",
        "recall": "Rec.",
        "f1": "F1",
    }

    df = results.copy()
    cols = [method_col] + [c for c in rename_map if c in df.columns]
    out = df[cols].rename(columns={method_col: "Method", **rename_map})

    for c in ["Acc.", "Prec.", "Rec.", "F1"]:
        if c in out.columns:
            out[c] = out[c].astype(float).round(4)

    return out
