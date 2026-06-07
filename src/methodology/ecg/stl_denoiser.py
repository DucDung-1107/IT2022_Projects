"""
STL record-level denoising methodology for ECG classification.

This module contains the STL denoising logic used by
`src/notebook/ecg/experiment_ecg_stl.ipynb`. STL is applied to a long ECG
record before R-peak segmentation, because a 180-sample beat at 360 Hz only
covers 0.5 seconds and does not provide enough repeated cycles for a meaningful
seasonal decomposition.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy import signal as sps


@dataclass(frozen=True)
class ECGSTLConfig:
    fs: int = 360
    period: int = 360
    seasonal_window: int = 13
    inner_iter: int = 2
    outer_iter: int = 1
    downsample_factor: int = 3
    chunk_seconds: int = 45
    overlap_seconds: int = 5
    cache_dir: str = "/kaggle/working/stl_cache"


class STLDecomposer:
    """Minimal STL implementation using local linear LOESS smoothing."""

    def __init__(
        self,
        period: int,
        seasonal_window: int = 13,
        trend_window: int | None = None,
        lowpass_window: int | None = None,
        inner_iter: int = 2,
        outer_iter: int = 1,
        degree: int = 1,
    ) -> None:
        self.period = int(period)
        self.seasonal_window = self._odd(max(3, int(seasonal_window)))
        if trend_window is None:
            tw = int(np.ceil(1.5 * self.period / (1 - 1.5 / self.seasonal_window)))
            self.trend_window = self._odd(max(3, tw))
        else:
            self.trend_window = self._odd(max(3, int(trend_window)))
        lw = self.period if lowpass_window is None else int(lowpass_window)
        self.lowpass_window = self._odd(max(3, lw))
        self.inner_iter = max(1, int(inner_iter))
        self.outer_iter = max(0, int(outer_iter))
        self.degree = 1 if degree != 0 else 0

    @staticmethod
    def _odd(v: int) -> int:
        return v if v % 2 == 1 else v + 1

    @staticmethod
    def _tricube(u: np.ndarray) -> np.ndarray:
        au = np.abs(u)
        w = np.zeros_like(au)
        mask = au < 1
        w[mask] = (1 - au[mask] ** 3) ** 3
        return w

    @staticmethod
    def _bisquare(u: np.ndarray) -> np.ndarray:
        au = np.abs(u)
        w = np.zeros_like(au)
        mask = au < 1
        w[mask] = (1 - au[mask] ** 2) ** 2
        return w

    def _loess(self, y: np.ndarray, window: int, robust_weights: np.ndarray | None = None) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        n = len(y)
        x = np.arange(n, dtype=float)
        fit = np.empty(n, dtype=float)
        half = window // 2
        if robust_weights is None:
            robust_weights = np.ones(n, dtype=float)

        for i in range(n):
            left = max(0, i - half)
            right = min(n - 1, i + half)
            if right - left + 1 < window:
                left = max(0, right - window + 1)
                right = min(n - 1, left + window - 1)

            idx = np.arange(left, right + 1)
            xi = x[idx]
            yi = y[idx]
            d = np.max(np.abs(xi - x[i]))
            if d == 0:
                fit[i] = y[i]
                continue

            w = self._tricube((xi - x[i]) / d) * robust_weights[idx]
            if np.all(w <= 1e-12):
                fit[i] = y[i]
                continue

            if self.degree == 0:
                fit[i] = np.sum(w * yi) / np.sum(w)
            else:
                X = np.column_stack([np.ones_like(xi), xi - x[i]])
                WX = X * w[:, None]
                beta, *_ = np.linalg.lstsq(WX.T @ X, WX.T @ yi, rcond=None)
                fit[i] = beta[0]

        return fit

    def _moving_average(self, y: np.ndarray, window: int) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        out = np.empty(len(y), dtype=float)
        half = window // 2
        for i in range(len(y)):
            left = max(0, i - half)
            right = min(len(y), i + half + 1)
            out[i] = y[left:right].mean()
        return out

    def _smooth_cycle(self, detrended: np.ndarray, robust_weights: np.ndarray) -> np.ndarray:
        seasonal = np.zeros(len(detrended), dtype=float)
        for phase in range(self.period):
            idx = np.arange(phase, len(detrended), self.period)
            if len(idx) > 0:
                seasonal[idx] = self._loess(detrended[idx], self.seasonal_window, robust_weights[idx])
        return seasonal

    def fit_transform(self, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = np.asarray(ts, dtype=float).reshape(-1)
        n = len(y)
        if n < 2 * self.period:
            trend = np.convolve(y, np.ones(3) / 3, mode="same")
            seasonal = np.zeros_like(y)
            return seasonal.astype(np.float32), trend.astype(np.float32), (y - trend).astype(np.float32)

        trend = self._loess(y, self.trend_window)
        seasonal = np.zeros(n, dtype=float)
        robust = np.ones(n, dtype=float)

        for _ in range(self.outer_iter + 1):
            for _ in range(self.inner_iter):
                raw = self._smooth_cycle(y - trend, robust)
                lowpass = self._moving_average(raw, self.period)
                lowpass = self._moving_average(lowpass, self.period)
                lowpass = self._moving_average(lowpass, 3)
                lowpass = self._loess(lowpass, self.lowpass_window)
                seasonal = raw - lowpass
                trend = self._loess(y - seasonal, self.trend_window, robust)
            residual = y - seasonal - trend
            mad = np.median(np.abs(residual))
            robust = np.ones(n) if mad <= 1e-12 else self._bisquare(residual / (6.0 * mad))

        residual = y - seasonal - trend
        return seasonal.astype(np.float32), trend.astype(np.float32), residual.astype(np.float32)


def stl_denoise_record(signal: np.ndarray, cfg: ECGSTLConfig = ECGSTLConfig()) -> np.ndarray:
    """Denoise one ECG record with downsampled, chunked STL."""
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 2 * cfg.period:
        return signal.astype(np.float32)

    effective_period = max(3, cfg.period // cfg.downsample_factor)
    model = STLDecomposer(
        period=effective_period,
        seasonal_window=cfg.seasonal_window,
        inner_iter=cfg.inner_iter,
        outer_iter=cfg.outer_iter,
    )

    down = sps.decimate(signal, q=cfg.downsample_factor, ftype="fir", zero_phase=True)
    fs_down = cfg.fs / cfg.downsample_factor
    chunk_len = max(int(fs_down * cfg.chunk_seconds), 2 * model.period + 50)
    overlap = int(fs_down * cfg.overlap_seconds)
    step = max(1, chunk_len - overlap)

    out = np.zeros(len(down), dtype=float)
    wsum = np.zeros(len(down), dtype=float)
    for left in range(0, len(down), step):
        right = min(len(down), left + chunk_len)
        chunk = down[left:right]
        try:
            _, _, residual = model.fit_transform(chunk)
        except Exception:
            residual = chunk

        win = np.hanning(len(residual)) if len(residual) > 2 else np.ones(len(residual))
        if np.allclose(win.sum(), 0):
            win = np.ones(len(residual))
        out[left:right] += residual * win
        wsum[left:right] += win
        if right >= len(down):
            break

    denoised_down = out / np.clip(wsum, 1e-8, None)
    x_down = np.linspace(0, len(signal) - 1, num=len(denoised_down))
    denoised = np.interp(np.arange(len(signal)), x_down, denoised_down)
    return denoised.astype(np.float32)


def get_or_create_stl_record(record_id: int | str, signal: np.ndarray, cfg: ECGSTLConfig = ECGSTLConfig()) -> np.ndarray:
    """Load cached STL-denoised record if present; otherwise compute and cache it."""
    os.makedirs(cfg.cache_dir, exist_ok=True)
    path = os.path.join(cfg.cache_dir, f"record_{record_id}_stl_ds{cfg.downsample_factor}_p{cfg.period}.npy")
    if os.path.exists(path):
        cached = np.load(path)
        if len(cached) == len(signal):
            return cached.astype(np.float32)

    denoised = stl_denoise_record(signal, cfg)
    np.save(path, denoised.astype(np.float32))
    return denoised
