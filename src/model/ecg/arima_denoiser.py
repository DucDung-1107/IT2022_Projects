"""
ARIMA / SARIMA Denoiser for ECG — implemented FROM SCRATCH.

No statsmodels dependency. Implements:
    - ARIMA(p,d,q): OLS-based AR + differencing + iterative MA residual estimation
    - SARIMA(P,D,Q,s): seasonal ARIMA wrapper
    - Per-beat ARIMA denoising (fitted values = denoised signal)
    - Pure ARIMA/SARIMA likelihood classifier

References:
    - Box, Jenkins, Reinsel (Time Series Analysis, 4th ed.)
    - Hamilton (Time Series Analysis, 1994)

Usage:
    denoised = arima_denoise_beat(beat_array, p=2, d=1, q=2)
    fitted = arima_fit(series, p=1, d=0, q=1)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =============================================================================
#  DIFFERENCING (Integration — I of ARIMA)
# =============================================================================
def _diff(x: np.ndarray, d: int) -> np.ndarray:
    """Apply d-th order differencing. Returns shorter array by d elements."""
    y = x.copy()
    for _ in range(d):
        y = np.diff(y)
    return y


def _undiff(diff_series: np.ndarray, original_first_d: np.ndarray, d: int) -> np.ndarray:
    """
    Reconstruct original series from d-th order differenced series.
    original_first_d : first d values of the original series.
    """
    y = diff_series.copy().astype(float)
    for order in range(d - 1, -1, -1):
        n = len(y)
        cum = np.empty(n + order)
        cum[:order] = original_first_d[:order]
        if order == 0:
            cum[order] = original_first_d[0]
            for i in range(1, len(cum)):
                cum[i] = cum[i - 1] + y[i - 1]
        else:
            cum[order] = original_first_d[order]
            for i in range(order + 1, len(cum)):
                cum[i] = cum[i - 1] + y[i - 1 - order]
        y = cum if order == 0 else cum[order:]
    return y


# =============================================================================
#  LEAST SQUARES SOLVER
# =============================================================================
def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ordinary Least Squares: β = (X^T X)^{-1} X^T y."""
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        XtX += 1e-8 * np.eye(XtX.shape[0])  # ridge regularisation for rank-deficiency
    return np.linalg.solve(XtX, X.T @ y)


# =============================================================================
#  AR(p) — AUTOREGRESSIVE MODEL FROM SCRATCH
# =============================================================================
def _ar_fit(y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit AR(p) via OLS.
    
    Returns
    -------
    phi : np.ndarray, shape (p,)
        AR coefficients (phi_1 ... phi_p).
    residuals : np.ndarray, shape (n-p,)
        Residuals (e_t) for t = p..n-1.
    """
    n = len(y)
    if p <= 0 or n <= p:
        return np.zeros(p), np.zeros(n)
    # Build design matrix: X_t = [y_{t-1}, y_{t-2}, ..., y_{t-p}]
    X = np.column_stack([y[p - 1 - i : n - 1 - i] for i in range(p)])
    y_target = y[p:]
    phi = _ols(X, y_target)
    residuals = y_target - X @ phi
    return phi, residuals


def _ar_predict(y: np.ndarray, phi: np.ndarray, p: int) -> np.ndarray:
    """Compute fitted values of AR(p) for the entire series."""
    n = len(y)
    fitted = np.full(n, np.nan)
    if p <= 0:
        return fitted
    X = np.column_stack([y[p - 1 - i : n - 1 - i] for i in range(p)])
    fitted[p:] = X @ phi
    return fitted


def _ar_loglik(residuals: np.ndarray) -> float:
    """Gaussian log-likelihood from residuals."""
    m = len(residuals)
    if m < 2:
        return -float("inf")
    sigma2 = np.var(residuals, ddof=0) + 1e-12
    return -0.5 * m * (np.log(2 * np.pi) + np.log(sigma2)) - 0.5 * np.sum(residuals**2) / sigma2


# =============================================================================
#  MA(q) — MOVING AVERAGE MODEL FROM SCRATCH (Innovations Algorithm)
# =============================================================================
def _ma_fit(
    y: np.ndarray, q: int, max_iter: int = 500, tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit MA(q) via iterative conditional least squares (Box-Jenkins style).
    
    The MA model: y_t = mu + e_t + theta_1 * e_{t-1} + ... + theta_q * e_{t-q}
    We estimate theta by iteratively computing residuals and re-estimating.

    Returns
    -------
    theta : np.ndarray, shape (q,)
        MA coefficients (theta_1 ... theta_q).
    residuals : np.ndarray, shape (n,)
        Estimated residuals (innovations) e_t.
    """
    n = len(y)
    if q <= 0 or n < q + 2:
        return np.zeros(q), np.zeros(n)

    mu = np.mean(y)
    y_dm = y - mu  # demeaned

    # Initialise theta = 0, residuals = y_dem
    theta = np.zeros(q)
    residuals = y_dm.copy()

    for iteration in range(max_iter):
        old_theta = theta.copy()
        # Compute residuals given current theta: e_t = y_dm_t - sum(theta_i * e_{t-i})
        e = np.zeros(n)
        for t in range(n):
            e[t] = y_dm[t]
            for i in range(1, min(q, t) + 1):
                e[t] -= theta[i - 1] * e[t - i]

        # Estimate theta: regress y_dm_t on [e_{t-1}, ..., e_{t-q}]
        X = np.column_stack(
            [e[q - 1 - i : n - 1 - i] for i in range(q)]
        )
        y_target = y_dm[q:]
        if X.shape[0] < q + 1:
            break
        theta = _ols(X, y_target)

        residuals = e
        if np.allclose(theta, old_theta, atol=tol):
            break

    return theta, residuals


def _ma_predict(y: np.ndarray, theta: np.ndarray, q: int) -> np.ndarray:
    """Compute fitted values of MA(q): y_fitted_t = mu + sum(theta_i * e_{t-i})."""
    n = len(y)
    fitted = np.full(n, np.nan)
    mu = np.mean(y)
    if q <= 0:
        return fitted
    
    # Compute residuals first
    _, e = _ma_fit(y, q, max_iter=10)  # quick pass
    
    fitted[:] = mu
    for t in range(n):
        for i in range(1, min(q, t) + 1):
            fitted[t] += theta[i - 1] * e[t - i]
    return fitted


def _ma_loglik(residuals: np.ndarray) -> float:
    """Gaussian log-likelihood from MA residuals (innovations)."""
    return _ar_loglik(residuals)


# =============================================================================
#  FULL ARIMA(p,d,q) FROM SCRATCH
# =============================================================================
def arima_fit(
    y: np.ndarray,
    p: int,
    d: int,
    q: int,
    include_mean: bool = True,
) -> dict:
    """
    Fit ARIMA(p,d,q) from scratch.

    Steps:
        1. Apply d-th order differencing to make series stationary.
        2. Estimate AR(p) coefficients via OLS (if p > 0).
        3. Estimate MA(q) coefficients via iterative LS (if q > 0).
        4. Combine fitted values = AR_fitted + MA_fitted + (mean if included).
        5. Undifference to reconstruct original scale.

    Parameters
    ----------
    y : np.ndarray
        Input time series (1D).
    p : int
        AR order.
    d : int
        Integration order (differencing).
    q : int
        MA order.
    include_mean : bool
        If True, include constant term.

    Returns
    -------
    dict with keys:
        'phi' : AR coefficients
        'theta' : MA coefficients
        'mu' : mean of differenced series
        'residuals' : model residuals (innovations)
        'fitted' : fitted values on original scale
        'llf' : log-likelihood
        'sigma2' : residual variance
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    # 1. Differencing
    if d > 0:
        y_diff = _diff(y, d)
        y_first_d = y[:d].copy()
    else:
        y_diff = y.copy()
        y_first_d = np.array([])

    nd = len(y_diff)

    # 2. Demean for stationarity
    mu = float(np.mean(y_diff))
    y_dm = y_diff - mu

    # 3. AR(p) estimation
    phi = np.zeros(p)
    ar_residuals = y_dm.copy()
    if p > 0 and nd > p:
        phi, ar_residuals = _ar_fit(y_dm, p)

    # 4. Remove AR component → series for MA estimation
    if p > 0:
        ar_fitted_part = _ar_predict(y_dm, phi, p)
        y_for_ma = y_dm.copy()
        y_for_ma[p:] -= ar_fitted_part[p:]
    else:
        y_for_ma = y_dm.copy()
        ar_fitted_part = np.full(nd, np.nan)

    # 5. MA(q) estimation on residuals
    theta = np.zeros(q)
    residuals = y_for_ma.copy()
    if q > 0 and nd > q:
        theta, residuals = _ma_fit(y_for_ma, q)

    # 6. Reconstruct fitted values on differenced scale
    fitted_diff = np.full(nd, mu)
    if p > 0:
        fitted_diff[p:] += ar_fitted_part[p:]
    if q > 0:
        # MA fitted = mu + theta_1 * e_{t-1} + ...
        for t in range(1, nd):
            for i in range(1, min(q, t) + 1):
                fitted_diff[t] += theta[i - 1] * residuals[t - i]

    # 7. Undifference
    if d > 0:
        fitted = _undiff(fitted_diff, y_first_d, d)
    else:
        fitted = fitted_diff.copy()

    # Pad leading NaN from differencing
    fitted = np.where(np.isnan(fitted), y, fitted)

    # 8. Compute stats
    valid_resid = y - fitted
    valid_resid = np.where(np.isnan(valid_resid), 0, valid_resid)
    sigma2 = np.var(valid_resid, ddof=0) + 1e-12
    llf = _ar_loglik(valid_resid)

    return {
        "phi": phi,
        "theta": theta,
        "mu": mu,
        "residuals": valid_resid,
        "fitted": fitted,
        "llf": float(llf),
        "sigma2": float(sigma2),
    }


def arima_predict_fitted(result: dict, y_orig: np.ndarray) -> np.ndarray:
    """Return fitted values (denoised signal) from ARIMA fit result."""
    return result["fitted"]


def arima_loglik(result: dict, new_seq: np.ndarray) -> float:
    """
    Compute log-likelihood of new data under a fitted ARIMA model.

    Uses the model parameters (phi, theta, mu, sigma2) to compute
    prediction errors on new_seq.

    Parameters
    ----------
    result : dict
        Output from arima_fit().
    new_seq : np.ndarray
        New time series to evaluate.

    Returns
    -------
    float
        Log-likelihood.
    """
    y = np.asarray(new_seq, dtype=float).ravel()
    n = len(y)
    phi = result["phi"]
    theta = result["theta"]
    mu = result["mu"]
    sigma2 = result["sigma2"]
    p = len(phi)
    q = len(theta)
    d = 0  # new_seq assumed already differenced if needed

    # Prediction errors (one-step-ahead)
    e = np.zeros(n)
    for t in range(n):
        pred = mu
        for i in range(1, min(p, t) + 1):
            pred += phi[i - 1] * y[t - i]
        for i in range(1, min(q, t) + 1):
            pred += theta[i - 1] * e[t - i]
        e[t] = y[t] - pred

    # Compute log-likelihood
    m = n
    ll = -0.5 * m * np.log(2 * np.pi) - 0.5 * m * np.log(sigma2) - 0.5 * np.sum(e**2) / sigma2
    return float(ll)


# =============================================================================
#  SARIMA(P,D,Q,s) FROM SCRATCH — via seasonal differencing + ARIMA
# =============================================================================
def _seasonal_diff(y: np.ndarray, D: int, s: int) -> np.ndarray:
    """Apply D-th order seasonal differencing with period s."""
    yd = y.copy()
    for _ in range(D):
        yd = yd[s:] - yd[:-s]
    # Pad to keep same length? No — seasonal diff shortens by D*s
    # We keep the shortened version for modelling
    return yd


def _seasonal_undiff(
    y_diff: np.ndarray, y_original: np.ndarray, D: int, s: int
) -> np.ndarray:
    """
    Reconstruct series after seasonal differencing D times with period s.
    Simple approach: inverse by cumulative addition.
    """
    result = y_original.copy()
    if D == 0:
        return result
    # We know the first D*s values of original to start reconstruction
    result[:D*s] = y_original[:D*s]
    for t in range(D*s, len(result)):
        if D == 1:
            result[t] = result[t - s] + y_diff[t - D*s]
        else:
            # Higher D: treat recursively
            result[t] = 2 * result[t - s] - result[t - 2*s] + y_diff[t - D*s]
    return result


def sarima_fit(
    y: np.ndarray,
    P: int = 1,
    D: int = 0,
    Q: int = 1,
    s: int = 5,
    include_mean: bool = True,
) -> dict:
    """
    Fit SARIMA(P,D,Q,s) from scratch using seasonal differencing + ARIMA on seasonal lags.

    Simplified approach:
        1. Apply D seasonal differences of period s.
        2. Fit ARIMA(P, 0, Q) on the seasonally differenced series
           (using only seasonal lags: t-s, t-2s, ...)
        3. Undifference to reconstruct.

    Parameters
    ----------
    y : np.ndarray
        Input series.
    P : int
        Seasonal AR order.
    D : int
        Seasonal differencing order.
    Q : int
        Seasonal MA order.
    s : int
        Seasonal period (e.g. 5 for beat-to-beat RR regularity).
    include_mean : bool
        Include constant term.

    Returns
    -------
    dict with same keys as arima_fit().
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    # 1. Seasonal differencing
    if D > 0:
        y_sdiff = _seasonal_diff(y, D, s)
        y_sfirst = y[:D*s].copy()
    else:
        y_sdiff = y.copy()
        y_sfirst = np.array([])

    nd = len(y_sdiff)

    # 2. Demean
    mu = float(np.mean(y_sdiff))
    y_dm = y_sdiff - mu

    # 3. Build seasonal lag design matrix for AR: X_t = [y_{t-s}, y_{t-2s}, ..., y_{t-P*s}]
    phi_s = np.zeros(P)
    resid_s = y_dm.copy()
    if P > 0 and nd > P * s:
        X_s = np.column_stack([y_dm[P * s - (i + 1) * s : nd - (i + 1) * s] for i in range(P)])
        y_target_s = y_dm[P * s:]
        phi_s = _ols(X_s, y_target_s)
        resid_s = np.full(nd, 0.0)
        resid_s[:P * s] = y_dm[:P * s]
        resid_s[P * s:] = y_target_s - X_s @ phi_s

    # 4. MA component on seasonal residuals (treat as seasonal MA: theta_s * e_{t-s})
    theta_s = np.zeros(Q)
    residuals = resid_s.copy()
    if Q > 0 and nd > (P + Q) * s:
        # Iterative LS
        for _ in range(100):
            e = np.zeros(nd)
            for t in range(nd):
                e[t] = resid_s[t]
                for i in range(1, min(Q, t // s) + 1):
                    e[t] -= theta_s[i - 1] * e[t - i * s]
            X_sma = np.column_stack(
                [e[Q * s - (i + 1) * s : nd - (i + 1) * s] for i in range(Q)]
            )
            if X_sma.shape[0] < Q + 1:
                break
            theta_s = _ols(X_sma, resid_s[Q * s:])
            residuals = e

    # 5. Reconstruct fitted on seasonally differenced scale
    fitted_sdiff = np.full(nd, mu)
    if P > 0:
        X_s_fit = np.column_stack([y_dm[P * s - (i + 1) * s : nd - (i + 1) * s] for i in range(P)])
        fitted_sdiff[P * s:] += X_s_fit @ phi_s
    if Q > 0:
        for t in range(s, nd):
            for i in range(1, min(Q, t // s) + 1):
                fitted_sdiff[t] += theta_s[i - 1] * residuals[t - i * s]

    # 6. Undifference seasonally
    if D > 0:
        fitted = _seasonal_undiff(fitted_sdiff, y, D, s)
    else:
        fitted = fitted_sdiff.copy()

    fitted = np.where(np.isnan(fitted), y, fitted)
    valid_resid = y - fitted
    valid_resid = np.where(np.isnan(valid_resid), 0, valid_resid)
    sigma2 = np.var(valid_resid, ddof=0) + 1e-12
    llf = _ar_loglik(valid_resid)

    return {
        "phi": phi_s,
        "theta": theta_s,
        "mu": mu,
        "residuals": valid_resid,
        "fitted": fitted,
        "llf": float(llf),
        "sigma2": float(sigma2),
        "seasonal_period": s,
    }


# =============================================================================
#  DENOISING — ARIMA Per-Beat
# =============================================================================
ARIMA_ORDER_BEAT = (2, 1, 2)
SARIMA_ORDER_RR = (1, 0, 1)
SARIMA_SEASONAL = (1, 0, 1, 5)


def arima_denoise_beat(
    beat: np.ndarray, order: tuple[int, int, int] = ARIMA_ORDER_BEAT
) -> np.ndarray:
    """
    Fit ARIMA(p,d,q) per beat and return fitted (denoised) values.

    The fitted values represent the autocorrelated component of the signal;
    residuals are treated as noise.

    Parameters
    ----------
    beat : np.ndarray
        1D array of shape (beat_len,) — e.g. 180 samples.
    order : tuple[int, int, int]
        ARIMA order (p, d, q).

    Returns
    -------
    np.ndarray
        Denoised beat, same length as input.
    """
    p, d, q = order
    try:
        result = arima_fit(beat, p=p, d=d, q=q)
        denoised = result["fitted"]
        return denoised.astype(np.float32)
    except Exception:
        return beat.astype(np.float32)


def arima_denoise_batch(
    W: np.ndarray,
    order: tuple[int, int, int] = ARIMA_ORDER_BEAT,
    max_samples: int | None = None,
) -> np.ndarray:
    """
    Apply ARIMA denoising to a batch of beats.

    Parameters
    ----------
    W : np.ndarray
        Shape (N, BEAT_LEN, 1) — batch of waveforms.
    order : tuple[int, int, int]
        ARIMA order (p, d, q).
    max_samples : int or None
        If set, only denoise the first `max_samples` beats (rest unchanged).

    Returns
    -------
    np.ndarray
        Denoised batch, same shape as input.
    """
    N = len(W) if max_samples is None else min(max_samples, len(W))
    out = np.zeros_like(W)
    for i in tqdm(range(N), desc="   ARIMA-denoising (from scratch)"):
        out[i, :, 0] = arima_denoise_beat(W[i, :, 0], order=order)
    if max_samples is not None and max_samples < len(W):
        out[N:] = W[N:]
    return out


# =============================================================================
#  SARIMA LIKELIHOOD CLASSIFIER (Phase 1 — Pure ARIMA/SARIMA)
# =============================================================================
def fit_class_sarima(
    rr_sequences_class: np.ndarray,
    max_train_samples: int = 2000,
    order: tuple[int, int, int] = SARIMA_ORDER_RR,
    seasonal_order: tuple[int, int, int, int] = SARIMA_SEASONAL,
) -> dict | None:
    """
    Fit 1 SARIMA model on pooled RR sequences of a single class.

    Uses the from-scratch sarima_fit() — no statsmodels dependency.

    Parameters
    ----------
    rr_sequences_class : np.ndarray
        Shape (n_seq, SEQ_LEN, 3) — only channel 0 (RR raw) is used.
    max_train_samples : int
        Cap on number of samples to fit (for speed).
    order : tuple[int, int, int]
        Non-seasonal order (used as fallback ARIMA).
    seasonal_order : tuple[int, int, int, int]
        Seasonal order (P, D, Q, s).

    Returns
    -------
    dict or None
        Fitted model result dict, or None on failure.
    """
    series = rr_sequences_class[:, :, 0].flatten()
    if len(series) > max_train_samples:
        series = series[:max_train_samples]

    P, D, Q, s = seasonal_order
    try:
        if D > 0 or Q > 0 or P > 0:
            result = sarima_fit(series, P=P, D=D, Q=Q, s=s)
            return result
        else:
            p, d, q = order
            return arima_fit(series, p=p, d=d, q=q)
    except Exception as e:
        print(f"    [SARIMA fit fail] {e}")
        try:
            p, d, q = order
            return arima_fit(series, p=p, d=d, q=q)
        except Exception as e2:
            print(f"    [ARIMA fallback fail] {e2}")
            return None


def sarima_loglik_for_seq(result: dict | None, seq: np.ndarray) -> float:
    """
    Compute log-likelihood of an RR sequence under a fitted SARIMA/ARIMA model.

    Parameters
    ----------
    result : dict or None
        Output from sarima_fit() or arima_fit().
    seq : np.ndarray
        1D array of length SEQ_LEN (RR values).

    Returns
    -------
    float
        Log-likelihood (or -inf on failure).
    """
    if result is None:
        return -float("inf")
    try:
        return arima_loglik(result, seq)
    except Exception:
        return -float("inf")


def predict_pure_sarima(
    X_seq: np.ndarray,
    models: dict[int, dict | None],
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict class labels using likelihood under per-class SARIMA models.

    Parameters
    ----------
    X_seq : np.ndarray
        Shape (N, SEQ_LEN, 3) — RR sequences.
    models : dict[int, dict | None]
        Per-class fitted model dicts.
    n_classes : int
        Number of classes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        preds: (N,) class predictions.
        probs: (N, n_classes) softmax probabilities from log-likelihoods.
    """
    N = len(X_seq)
    scores = np.full((N, n_classes), -float("inf"))

    for i in tqdm(range(N), desc="   SARIMA predicting (from scratch)"):
        seq = X_seq[i, :, 0]  # RR raw channel
        for c in range(n_classes):
            scores[i, c] = sarima_loglik_for_seq(models[c], seq)

    # Convert log-likelihood to softmax probabilities
    scores_max = np.nanmax(scores, axis=1, keepdims=True)
    safe_scores = np.where(np.isfinite(scores), scores, scores_max - 50)
    exps = np.exp(safe_scores - scores_max)
    probs = exps / (exps.sum(axis=1, keepdims=True) + 1e-12)
    preds = scores.argmax(axis=1)
    return preds, probs


__all__ = [
    "ARIMA_ORDER_BEAT",
    "SARIMA_ORDER_RR",
    "SARIMA_SEASONAL",
    "arima_fit",
    "arima_predict_fitted",
    "arima_loglik",
    "sarima_fit",
    "arima_denoise_beat",
    "arima_denoise_batch",
    "fit_class_sarima",
    "sarima_loglik_for_seq",
    "predict_pure_sarima",
]