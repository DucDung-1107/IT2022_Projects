"""
Causal Median Filter features for FPT weekly stock classification.

This module belongs in:
    src/model/stock/median_denoiser.py

For stock data, the median filter must be used causally by default: the filtered
value at week t is computed only from current and previous close prices. This
avoids leaking future prices into walk-forward validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-9


@dataclass(frozen=True)
class StockMedianDenoiserConfig:
    """
    Hyper-parameters for stock median filtering.

    Parameters
    ----------
    window:
        Rolling median window. For weekly close prices, 3 or 5 is a reasonable
        starting point.
    causal:
        If True, only current/past values are used. Keep True for prediction.
    min_periods:
        Minimum observations required for rolling median. None means use 1.
    """

    window: int = 5
    causal: bool = True
    min_periods: int | None = None


def _validate_window(window: int) -> int:
    w = int(window)
    if w < 1:
        raise ValueError("window must be >= 1.")
    if w % 2 == 0:
        w += 1
    return w


def causal_rolling_median(close_arr: np.ndarray | list[float], window: int = 5) -> np.ndarray:
    """Median filter using only current and previous observations."""
    x = np.asarray(close_arr, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return np.array([], dtype=np.float64)
    w = _validate_window(window)
    return (
        pd.Series(x)
        .rolling(window=w, min_periods=1)
        .median()
        .ffill()
        .bfill()
        .to_numpy(dtype=np.float64)
    )


def centered_rolling_median(close_arr: np.ndarray | list[float], window: int = 5) -> np.ndarray:
    """
    Centered median filter for offline visualization only.

    Do not use this for walk-forward prediction because it uses future values.
    """
    x = np.asarray(close_arr, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return np.array([], dtype=np.float64)
    w = _validate_window(window)
    return (
        pd.Series(x)
        .rolling(window=w, min_periods=1, center=True)
        .median()
        .ffill()
        .bfill()
        .to_numpy(dtype=np.float64)
    )


class StockMedianDenoiser:
    """
    Median-filter denoiser with the same fit/transform style as other stock models.

    The model has no trainable parameters. `fit()` exists for interface
    consistency with LSTM/Kalman-style denoisers.
    """

    def __init__(self, cfg: StockMedianDenoiserConfig | None = None) -> None:
        self.cfg = cfg or StockMedianDenoiserConfig()
        self.window = _validate_window(self.cfg.window)
        self._fitted = False

    def fit(self, close_arr: np.ndarray, verbose: int = 0) -> "StockMedianDenoiser":
        self._fitted = True
        return self

    def fine_tune(self, close_arr: np.ndarray, epochs: int = 0, verbose: int = 0) -> "StockMedianDenoiser":
        return self.fit(close_arr, verbose=verbose)

    def transform(self, close_arr: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(close_arr, dtype=np.float64).reshape(-1)
        if len(x) == 0:
            empty = np.array([], dtype=np.float64)
            return pd.DataFrame(), empty, empty, empty.reshape(-1, 1)

        if self.cfg.causal:
            fitted = causal_rolling_median(x, window=self.window)
        else:
            fitted = centered_rolling_median(x, window=self.window)

        resid = x - fitted
        resid_abs = np.abs(resid)
        trend = np.gradient(fitted) if len(fitted) >= 2 else np.zeros_like(fitted)
        curvature = np.gradient(trend) if len(trend) >= 2 else np.zeros_like(fitted)
        fitted_ret = np.diff(fitted, prepend=fitted[0]) / (np.abs(fitted) + EPS) * 100
        raw_ret = np.diff(x, prepend=x[0]) / (np.abs(x) + EPS) * 100
        detrended_ret = resid / (np.abs(x) + EPS) * 100
        roll_std = (
            pd.Series(x)
            .rolling(window=max(2, self.window), min_periods=2)
            .std()
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        noise_ratio = resid_abs / (roll_std + EPS)

        feat = pd.DataFrame(
            {
                "median_fitted": fitted,
                "median_resid": resid,
                "median_resid_abs": resid_abs,
                "median_trend": trend,
                "median_curvature": curvature,
                "median_fitted_ret": fitted_ret,
                "median_raw_ret": raw_ret,
                "median_detrended_ret": detrended_ret,
                "median_noise_ratio": noise_ratio,
                "median_window": np.full(len(x), float(self.window)),
            }
        )

        # Fourth return keeps compatibility with Kalman-style downstream code that
        # expects a latent/dummy array.
        dummy_z = np.zeros((len(x), 1), dtype=np.float64)
        return feat, fitted, resid, dummy_z

    def get_feature_names(self) -> list[str]:
        return [
            "median_fitted",
            "median_resid",
            "median_resid_abs",
            "median_trend",
            "median_curvature",
            "median_fitted_ret",
            "median_raw_ret",
            "median_detrended_ret",
            "median_noise_ratio",
            "median_window",
        ]


def build_median_denoiser_features(
    close_arr: np.ndarray,
    cfg: StockMedianDenoiserConfig | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, StockMedianDenoiser]:
    """Convenience helper matching the stock LSTM helper style."""
    denoiser = StockMedianDenoiser(cfg)
    denoiser.fit(close_arr, verbose=0)
    df_feat, fitted, resid, _ = denoiser.transform(close_arr)
    return df_feat, fitted, resid, denoiser


__all__ = [
    "StockMedianDenoiserConfig",
    "StockMedianDenoiser",
    "causal_rolling_median",
    "centered_rolling_median",
    "build_median_denoiser_features",
]
