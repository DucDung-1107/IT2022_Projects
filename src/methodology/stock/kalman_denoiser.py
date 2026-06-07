"""
Adaptive Kalman Filter denoiser methodology for weekly stock classification.

This module contains the Kalman filter denoiser used by
`src/notebook/stock/kalman-stock-fpt.ipynb`.

State vector  : x = [price, velocity, acceleration]
Transition    : constant-acceleration kinematic model
Observation   : z(t) = price(t)  +  measurement noise R
Noise adapt   : Sage-Husa exponential moving-average estimator
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass(frozen=True)
class KalmanDenoiserConfig:
    """Hyper-parameters for the Adaptive Kalman Filter.

    Attributes
    ----------
    q_scale : float
        Initial process-noise scale multiplied by the variance of first
        differences of the training close-price series.
    r_scale : float
        Initial measurement-noise scale (same base variance as q_scale).
    adaptive : bool
        When True, Q and R are updated online via Sage-Husa EMA at every
        time step after the first observation.
    finetune : bool
        When True, ``fine_tune()`` re-estimates Q / R on the expanded
        training window at each walk-forward fold.
    """

    q_scale: float = 1e-4
    r_scale: float = 1e-2
    adaptive: bool = True
    finetune: bool = True


# =============================================================================
# ADAPTIVE KALMAN FILTER
# =============================================================================

class AdaptiveKalmanFilter:
    """Adaptive Kalman Filter on weekly close prices.

    State vector: x = [price, velocity, acceleration]  (3-dim)

    Transition model (constant-acceleration kinematic):
        price(t)        = price(t-1) + velocity(t-1) + 0.5 * acceleration(t-1)
        velocity(t)     = velocity(t-1) + acceleration(t-1)
        acceleration(t) = acceleration(t-1)

    Observation model:
        z(t) = price(t) + measurement noise R

    Adaptive noise estimation (Sage-Husa):
        Q (process noise) and R (measurement noise) are initialised from
        ``q_scale`` / ``r_scale``, then updated online via an exponential
        moving average of the innovation sequence.

    Public interface
    ----------------
    fit(close_arr)            – estimate Q / R from a training window
    transform(close_arr)      – return (DataFrame, kf_price, kf_residual, dummy_z)
    get_feature_names()       – list of 10 column names
    fine_tune(close_arr)      – re-estimate Q / R (same as fit)

    ``transform()`` return signature
    ---------------------------------
    df          : DataFrame with 10 Kalman feature columns
    kf_price    : 1-D ndarray of filtered prices        (= d_fit in pipeline)
    kf_residual : 1-D ndarray of measurement residuals  (= d_resid in pipeline)
    dummy_z     : zeros array shaped (N, 1)             (preserves 4-tuple
                  downstream interface; replaces legacy latent z_seq)
    """

    # ── State-space matrices ──────────────────────────────────────────────────
    # F: transition  (constant-acceleration kinematic model)
    _F = np.array(
        [[1.0, 1.0, 0.5],
         [0.0, 1.0, 1.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # H: observation (price only)
    _H = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)  # (1, 3)

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, cfg: KalmanDenoiserConfig = KalmanDenoiserConfig()) -> None:
        self._cfg     = cfg
        self._Q       = None   # process noise covariance  (3, 3)
        self._R       = None   # measurement noise variance (scalar)
        self._fitted  = False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _init_noise(self, close_arr: np.ndarray) -> None:
        """Initialise Q and R from close-price first-difference variance."""
        arr       = np.asarray(close_arr, dtype=np.float64)
        price_var = float(np.var(np.diff(arr))) if len(arr) > 1 else 1.0
        self._R   = max(price_var * self._cfg.r_scale, 1e-8)
        q_base    = price_var * self._cfg.q_scale
        # Diagonal Q — velocity / acceleration get smaller noise than price
        self._Q   = np.diag([q_base, q_base * 0.1, q_base * 0.01])

    def _run_filter(
        self, close_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward Kalman pass over *close_arr* (in log space).

        Returns
        -------
        x_hist : (N, 3)  filtered state  [log-price, velocity, acceleration]
        P_hist : (N,)    trace(P_t) — total state uncertainty
        K_hist : (N,)    scalar Kalman gain for the price dimension
        innov  : (N,)    innovation  z - H @ x_pred
        """
        arr  = np.log(np.asarray(close_arr, dtype=np.float64))  # log space
        N    = len(arr)
        F, H = self._F, self._H
        Q    = self._Q.copy()
        R    = self._R

        # Initial state: log-price = first obs, velocity = acceleration = 0
        x = np.array([arr[0], 0.0, 0.0], dtype=np.float64)
        P = np.eye(3, dtype=np.float64) * R  # initial covariance

        x_hist = np.zeros((N, 3), dtype=np.float64)
        P_hist = np.zeros(N,      dtype=np.float64)
        K_hist = np.zeros(N,      dtype=np.float64)
        innov  = np.zeros(N,      dtype=np.float64)

        # Adaptive noise: exponential forgetting factor
        alpha = 0.05  # weight given to each new innovation sample

        for t in range(N):
            # ── Predict ──────────────────────────────────────────────────────
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # ── Update ───────────────────────────────────────────────────────
            z   = arr[t]
            inn = z - float(H @ x_pred)        # scalar innovation
            S   = float(H @ P_pred @ H.T) + R  # innovation covariance
            K   = (P_pred @ H.T) / S            # (3, 1)
            x   = x_pred + (K @ [[inn]]).ravel()
            P   = (np.eye(3) - K @ H) @ P_pred

            # ── Adaptive Q / R update (Sage-Husa) ────────────────────────────
            if self._cfg.adaptive and t > 0:
                R  = (1 - alpha) * R + alpha * (inn ** 2 - S + R)
                R  = max(R, 1e-8)
                Q  = (1 - alpha) * Q + alpha * (
                    np.outer(K.ravel() * inn, K.ravel() * inn)
                )
                # Ensure Q stays symmetric positive semi-definite
                Q  = (Q + Q.T) * 0.5
                Q  = np.where(Q < 0, 0.0, Q)

            x_hist[t] = x
            P_hist[t] = float(np.trace(P))
            K_hist[t] = float(K[0, 0])         # gain for price dimension
            innov[t]  = inn

        return x_hist, P_hist, K_hist, innov

    # ── Public interface ──────────────────────────────────────────────────────

    def fit(self, close_arr: np.ndarray, verbose: int = 0) -> "AdaptiveKalmanFilter":
        """Estimate Q and R from *close_arr* (training window).

        ``verbose`` is accepted for API compatibility but is not used.
        """
        self._init_noise(close_arr)
        self._fitted = True
        return self

    def fine_tune(
        self, close_arr: np.ndarray, epochs: int = 10, verbose: int = 0
    ) -> "AdaptiveKalmanFilter":
        """Re-estimate noise from an expanded training window (same as fit)."""
        return self.fit(close_arr, verbose=verbose)

    def transform(
        self, close_arr: np.ndarray
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Run forward Kalman filter over *close_arr*.

        Returns
        -------
        df          : DataFrame — 10 Kalman feature columns
        kf_price    : 1-D ndarray of filtered prices (exp-transformed)
        kf_residual : 1-D ndarray of measurement residuals (raw - kf_price)
        dummy_z     : zeros (N, 1) — preserves 4-tuple downstream interface
        """
        if not self._fitted:
            self.fit(close_arr)

        arr = np.asarray(close_arr, dtype=np.float64)
        N   = len(arr)

        x_hist, P_hist, K_hist, innov = self._run_filter(arr)

        kf_price        = np.exp(x_hist[:, 0])  # convert back from log space
        kf_velocity     = x_hist[:, 1]
        kf_acceleration = x_hist[:, 2]
        kf_residual     = arr - kf_price
        kf_residual_abs = np.abs(kf_residual)
        kf_return       = (
            np.diff(kf_price, prepend=kf_price[0])
            / (np.abs(kf_price) + 1e-9) * 100
        )

        # noise_ratio: |residual| / rolling std of raw price
        roll_std    = (
            pd.Series(arr).rolling(13, min_periods=2)
            .std().fillna(1.0).values
        )
        noise_ratio = kf_residual_abs / (roll_std + 1e-9)

        # signal_to_noise: kf_price std / noise std (rolling window)
        kf_std          = (
            pd.Series(kf_price).rolling(13, min_periods=2)
            .std().fillna(1e-9).values
        )
        signal_to_noise = kf_std / (roll_std + 1e-9)

        # uncertainty_ratio: normalised trace(P) relative to price level
        uncertainty_ratio = P_hist / (np.abs(kf_price) ** 2 + 1e-9)

        feat = {
            "kf_price"          : kf_price,
            "kf_return"         : kf_return,
            "kf_velocity"       : kf_velocity,
            "kf_acceleration"   : kf_acceleration,
            "kf_residual"       : kf_residual,
            "kf_residual_abs"   : kf_residual_abs,
            "noise_ratio"       : noise_ratio,
            "signal_to_noise"   : signal_to_noise,
            "uncertainty_ratio" : uncertainty_ratio,
            "kalman_gain"       : K_hist,
        }

        dummy_z = np.zeros((N, 1), dtype=np.float64)
        return pd.DataFrame(feat), kf_price, kf_residual, dummy_z

    def get_feature_names(self) -> list[str]:
        """Return the ordered list of feature column names produced by transform()."""
        return [
            "kf_price", "kf_return", "kf_velocity", "kf_acceleration",
            "kf_residual", "kf_residual_abs",
            "noise_ratio", "signal_to_noise", "uncertainty_ratio", "kalman_gain",
        ]


# =============================================================================
# STANDALONE BUILDER
# =============================================================================

def build_kalman_features_walkforward(close_arr, n_init, n_folds, fold_size,
                                      finetune=KalmanDenoiserConfig().finetune,
                                      verbose=0):
    """Walk-forward safe Kalman feature extraction.

    Walk-forward protocol
    ---------------------
    - fit()       called once on close_arr[:n_init]   (training set only)
    - transform() called on expanding window per fold  (no future leakage)
    - fine_tune() re-estimates Q/R on expanded train   (if finetune=True)

    Each fold contributes rows [tr_end : val_end] to the output arrays.
    Rows [0 : n_init] are filled from the initial fit window.

    Returns
    -------
    df_all       : DataFrame, shape (N, n_feat)
                   Kalman feature columns aligned with close_arr.
    kf_price_all : ndarray, shape (N,)
                   Walk-forward denoised close prices.
    kf_resid_all : ndarray, shape (N,)
                   Walk-forward measurement residuals (raw - kf_price).
    kf_obj       : AdaptiveKalmanFilter
                   Fitted filter object (used for final full-sequence plot).
    """
    N      = len(close_arr)
    kf_obj = AdaptiveKalmanFilter()

    # ── Initial fit on training window ────────────────────────────────────────
    print(f"  → Fitting AdaptiveKalmanFilter on first {n_init} weeks ...",
          flush=True)
    kf_obj.fit(close_arr[:n_init], verbose=verbose)

    df0, kfp0, kfr0, _ = kf_obj.transform(close_arr[:n_init])
    n_feat = df0.shape[1]

    # Pre-allocate output arrays
    all_feat     = np.zeros((N, n_feat), dtype=np.float32)
    kf_price_all = np.zeros(N,           dtype=np.float64)
    kf_resid_all = np.zeros(N,           dtype=np.float64)

    # Fill init window
    all_feat[:n_init]     = df0.values.astype(np.float32)
    kf_price_all[:n_init] = kfp0
    kf_resid_all[:n_init] = kfr0

    # ── Walk-forward folds ────────────────────────────────────────────────────
    for fold in range(n_folds):
        tr_end  = n_init + fold * fold_size
        val_end = min(tr_end + fold_size, N)
        if tr_end >= N:
            break

        # Optional: re-estimate noise parameters on expanded training set
        if finetune and fold > 0:
            print(f"  → Fine-tuning Kalman filter at fold {fold} "
                  f"(train={tr_end}) ...", flush=True)
            kf_obj.fine_tune(close_arr[:tr_end], verbose=verbose)

        # Transform expanding window; extract only the new validation slice
        df_ext, kfp_ext, kfr_ext, _ = kf_obj.transform(close_arr[:val_end])
        all_feat[tr_end:val_end]     = df_ext.values[tr_end:val_end].astype(np.float32)
        kf_price_all[tr_end:val_end] = kfp_ext[tr_end:val_end]
        kf_resid_all[tr_end:val_end] = kfr_ext[tr_end:val_end]

    df_all = pd.DataFrame(all_feat, columns=df0.columns)
    print(f"  Kalman features: {df_all.shape[1]}  "
          f"columns: {kf_obj.get_feature_names()}", flush=True)

    return df_all, kf_price_all, kf_resid_all, kf_obj
