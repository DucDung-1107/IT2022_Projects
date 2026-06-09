"""
Median Filter Denoiser for ECG beat classification.

This module belongs in:
    src/model/ecg/median_denoiser.py

Purpose
-------
The median filter is a non-linear preprocessing filter used to suppress short,
impulse-like noise while preserving sharp ECG morphology better than a simple
mean filter. For this project it is intended as a lightweight denoising baseline
that can be inserted before the existing Multi-Input BiLSTM classifier.

Typical use in the ECG pipeline
-------------------------------
    from src.model.ecg.median_denoiser import median_denoise_batch

    # X_w_tr_z and X_w_te_z have shape (n_beats, beat_len, 1)
    X_w_tr_med = median_denoise_batch(X_w_tr_z, kernel_size=5)
    X_w_te_med = median_denoise_batch(X_w_te_z, kernel_size=5)

Then train the same classifier using X_w_tr_med / X_w_te_med. Keep RR features
and handcrafted features unchanged unless you deliberately rebuild features from
median-filtered beats for another ablation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import medfilt
from tqdm import tqdm


@dataclass(frozen=True)
class ECGMedianDenoiserConfig:
    """
    Hyper-parameters for ECG median filtering.

    Parameters
    ----------
    kernel_size:
        Odd filter length. For 180-sample MIT-BIH beats, 3 or 5 is usually a
        safe starting point. Larger values may smooth QRS morphology too much.
    preserve_shape:
        If True, output is reshaped to match the exact input shape.
    """

    kernel_size: int = 5
    preserve_shape: bool = True


def _validate_kernel_size(kernel_size: int) -> int:
    """Return a positive odd integer kernel size accepted by scipy.medfilt."""
    k = int(kernel_size)
    if k < 1:
        raise ValueError("kernel_size must be >= 1.")
    if k % 2 == 0:
        k += 1
    return k


def _as_1d_float(x: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Input ECG signal/beat must contain at least one sample.")
    return arr


def median_denoise_beat(
    beat: np.ndarray | list[float],
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Apply a short-window median filter to one ECG beat.

    Parameters
    ----------
    beat:
        One ECG beat, normally length 180 for this project.
    kernel_size:
        Odd median window length. Use 3 or 5 for morphology-preserving tests.

    Returns
    -------
    np.ndarray
        Median-filtered beat as float32 with the same length as the input.
    """
    x = _as_1d_float(beat)
    k = _validate_kernel_size(kernel_size)

    # scipy.medfilt requires kernel <= signal length in practice. If the beat is
    # very short, shrink the kernel to the largest valid odd value.
    if k > len(x):
        k = len(x) if len(x) % 2 == 1 else len(x) - 1
        k = max(k, 1)

    return medfilt(x, kernel_size=k).astype(np.float32)


def median_denoise_record(
    signal: np.ndarray | list[float],
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Apply median filtering to a full ECG record.

    Use this when testing record-level preprocessing before R-peak segmentation.
    For the current segmented-beat experiments, `median_denoise_batch` is usually
    easier because it can be applied directly to X_w arrays.
    """
    return median_denoise_beat(signal, kernel_size=kernel_size)


def median_denoise_batch(
    W: np.ndarray,
    kernel_size: int = 5,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Apply median filtering to a batch of ECG beats.

    Parameters
    ----------
    W:
        Beat tensor with shape (n_beats, beat_len) or (n_beats, beat_len, 1).
    kernel_size:
        Median window length. Even values are automatically converted to the
        next odd value.
    show_progress:
        Display tqdm progress bar when True.

    Returns
    -------
    np.ndarray
        Filtered beat tensor with the same shape as W.
    """
    arr = np.asarray(W, dtype=np.float32)
    original_shape = arr.shape

    if arr.ndim == 2:
        beats_2d = arr
        add_channel = False
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        beats_2d = arr[:, :, 0]
        add_channel = True
    else:
        raise ValueError("W must have shape (n_beats, beat_len) or (n_beats, beat_len, 1).")

    out = np.empty_like(beats_2d, dtype=np.float32)
    iterator = range(len(beats_2d))
    if show_progress:
        iterator = tqdm(iterator, desc="Median filtering ECG beats")

    for i in iterator:
        out[i] = median_denoise_beat(beats_2d[i], kernel_size=kernel_size)

    if add_channel:
        out = out[..., None]

    return out.reshape(original_shape).astype(np.float32)


def median_residual(original: np.ndarray, denoised: np.ndarray) -> np.ndarray:
    """Return the removed component: original - median_filtered."""
    return np.asarray(original, dtype=float) - np.asarray(denoised, dtype=float)


def median_feature_summary(
    beat: np.ndarray | list[float],
    denoised: np.ndarray | None = None,
    kernel_size: int = 5,
) -> dict[str, float]:
    """
    Small diagnostic summary for reporting / ablation analysis.

    This does not replace the existing handcrafted ECG features; it only gives
    useful numbers for comparing raw vs median-filtered beats.
    """
    x = _as_1d_float(beat)
    y = median_denoise_beat(x, kernel_size=kernel_size) if denoised is None else _as_1d_float(denoised)
    r = median_residual(x, y)
    return {
        "median_kernel": float(_validate_kernel_size(kernel_size)),
        "raw_std": float(np.std(x)),
        "denoised_std": float(np.std(y)),
        "residual_std": float(np.std(r)),
        "residual_abs_mean": float(np.mean(np.abs(r))),
        "max_abs_change": float(np.max(np.abs(r))),
    }


class ECGMedianDenoiser:
    """
    Lightweight class wrapper so the median filter can be used like a denoiser.

    The filter has no trainable parameters, so fit() simply returns self.
    """

    def __init__(self, cfg: ECGMedianDenoiserConfig | None = None) -> None:
        self.cfg = cfg or ECGMedianDenoiserConfig()

    def fit(self, W: np.ndarray | None = None, verbose: int = 0) -> "ECGMedianDenoiser":
        return self

    def transform(self, W: np.ndarray, show_progress: bool = False) -> np.ndarray:
        return median_denoise_batch(
            W,
            kernel_size=self.cfg.kernel_size,
            show_progress=show_progress,
        )

    def fit_transform(self, W: np.ndarray, show_progress: bool = False) -> np.ndarray:
        return self.fit(W).transform(W, show_progress=show_progress)


__all__ = [
    "ECGMedianDenoiserConfig",
    "ECGMedianDenoiser",
    "median_denoise_beat",
    "median_denoise_record",
    "median_denoise_batch",
    "median_residual",
    "median_feature_summary",
]
