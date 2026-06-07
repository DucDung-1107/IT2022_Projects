"""
STL trend extraction methodology for weekly stock classification.

This module contains the STL feature builder used by
`src/notebook/stock/experiment_stock_stl.ipynb`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StockSTLConfig:
    period: int = 8
    seasonal_window: int = 7
    inner_iter: int = 5
    outer_iter: int = 20
    degree: int = 1


class STLDecomposer:
    """Minimal STL implementation for weekly close-price decomposition."""

    def __init__(
        self,
        period: int = 8,
        seasonal_window: int = 7,
        trend_window: int | None = None,
        lowpass_window: int | None = None,
        inner_iter: int = 5,
        outer_iter: int = 20,
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

    def fit_transform(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = np.asarray(series, dtype=float).reshape(-1)
        if len(y) < 2 * self.period:
            trend = pd.Series(y).rolling(3, min_periods=1).mean().to_numpy()
            seasonal = np.zeros_like(y)
            return seasonal.astype(np.float32), trend.astype(np.float32), (y - trend).astype(np.float32)

        trend = self._loess(y, self.trend_window)
        seasonal = np.zeros(len(y), dtype=float)
        robust = np.ones(len(y), dtype=float)
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
            robust = np.ones(len(y)) if mad <= 1e-12 else self._bisquare(residual / (6.0 * mad))

        residual = y - seasonal - trend
        return seasonal.astype(np.float32), trend.astype(np.float32), residual.astype(np.float32)


def build_stl_features(close: np.ndarray, cfg: StockSTLConfig = StockSTLConfig()) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build STL auxiliary features from a weekly close-price series."""
    model = STLDecomposer(
        period=cfg.period,
        seasonal_window=cfg.seasonal_window,
        inner_iter=cfg.inner_iter,
        outer_iter=cfg.outer_iter,
        degree=cfg.degree,
    )
    seasonal, trend, residual = model.fit_transform(close)
    features = pd.DataFrame(
        {
            "stl_trend": trend,
            "stl_seasonal": seasonal,
            "stl_resid": residual,
            "stl_resid_abs": np.abs(residual),
            "stl_detrended": np.asarray(close, dtype=float) - trend,
            "stl_trend_grad": np.gradient(trend),
            "stl_ret": np.diff(trend, prepend=trend[0]) / (np.abs(trend) + 1e-9) * 100,
        }
    )
    return features, trend, residual
