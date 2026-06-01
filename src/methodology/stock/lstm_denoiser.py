from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import Input, Model, layers, optimizers, regularizers

from ....utils.lstm_denoiser_utils import make_rolling_sequences


@dataclass(frozen=True)
class P4MethodConfig:
    signal_threshold_pct: float = 2.0
    classes: tuple[str, str] = ("up", "down")

    seq_len: int = 12
    tcn_seq_len: int = 20
    seed: int = 42

    tcn_filters: int = 32
    tcn_dropout: float = 0.10
    tcn_epochs: int = 80
    tcn_batch: int = 32
    tcn_lr: float = 1e-3

    lstm_units_a: int = 48
    lstm_units_b: int = 16
    lstm_dropout: float = 0.30
    lstm_lr: float = 5e-4


def load_daily(
    path: str,
    high_col: str,
    low_col: str,
    vol_col: str,
    close_col: str,
    date_col: str = "Date",
) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df[df.columns[1:]] = df[df.columns[1:]].ffill().bfill()
    return pd.DataFrame(
        {
            "date": df[date_col],
            "close": df[close_col].astype(float),
            "high": df[high_col].astype(float),
            "low": df[low_col].astype(float),
            "volume": df[vol_col].astype(float),
        }
    )


def label_weekly(df_w: pd.DataFrame, signal_thr: float) -> pd.DataFrame:
    df = df_w.copy()
    df["weekly_ret"] = df["close"].pct_change() * 100
    df["next_ret"] = df["weekly_ret"].shift(-1)
    df["next_close"] = df["close"].shift(-1)
    df["label"] = np.where(
        df["next_ret"] > signal_thr,
        "up",
        np.where(df["next_ret"] < -signal_thr, "down", "sideways"),
    )
    return df[df["label"] != "sideways"].dropna(subset=["next_ret", "label"]).reset_index(drop=True)


def build_weekly_features(df_w: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    f = df_w.copy().reset_index(drop=True)
    c = f["close"].shift(1)
    h = f["high"].shift(1)
    l = f["low"].shift(1)
    v = f["volume"].shift(1)
    r = f["weekly_ret"].shift(1)

    for w in [1, 2, 3, 4, 8, 13, 26]:
        f[f"ret_{w}w"] = c.pct_change(w) * 100

    for w in [4, 8, 13, 26, 52]:
        ma = c.rolling(w, min_periods=2).mean()
        f[f"mom_{w}w"] = (c / ma.replace(0, np.nan) - 1) * 100
        f[f"ma_std_{w}w"] = c.rolling(w, min_periods=2).std()

    ema4 = c.ewm(span=4, adjust=False).mean()
    ema13 = c.ewm(span=13, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    f["ema_cross_4_13"] = (ema4 - ema13) / (ema13.abs() + 1e-9) * 100
    f["ema_cross_13_26"] = (ema13 - ema26) / (ema26.abs() + 1e-9) * 100

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26e = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26e
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd"] = macd / (c.abs() + 1e-9) * 100
    f["macd_sig"] = sig / (c.abs() + 1e-9) * 100
    f["macd_hist"] = (macd - sig) / (c.abs() + 1e-9) * 100

    for w in [6, 14]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(w, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(w, min_periods=1).mean()
        f[f"rsi_{w}w"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-3))

    for w in [8, 20]:
        ma = c.rolling(w, min_periods=2).mean()
        std = c.rolling(w, min_periods=2).std().fillna(1e-9)
        ub, lb = ma + 2 * std, ma - 2 * std
        f[f"bb_pos_{w}w"] = (c - lb) / (ub - lb + 1e-9)
        f[f"bb_width_{w}w"] = (ub - lb) / (ma.abs() + 1e-9) * 100

    for w in [4, 8, 13]:
        f[f"vol_{w}w"] = r.rolling(w, min_periods=2).std()

    f["hl_range_pct"] = (h - l) / (c.abs() + 1e-9) * 100
    f["body_pct"] = (f["close"].shift(1) - f["open"].shift(1)) / (c.abs() + 1e-9) * 100

    for w in [4, 8, 13]:
        vm = v.rolling(w, min_periods=1).mean()
        f[f"vol_ratio_{w}w"] = v / (vm + 1e-9)

    f["month_sin"] = np.sin(2 * np.pi * f["date"].dt.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * f["date"].dt.month / 12)

    f = f.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

    exclude = [
        "date",
        "close",
        "high",
        "low",
        "volume",
        "open",
        "mid",
        "weekly_ret",
        "next_ret",
        "next_close",
        "label",
    ]
    x = f.drop(columns=[c2 for c2 in exclude if c2 in f.columns], errors="ignore")
    y = f["label"]
    return x, y


def build_tcn_denoiser(seq_len: int, cfg: P4MethodConfig) -> Model:
    inp = Input(shape=(seq_len, 1))
    x = inp
    for d in [1, 2, 4, 8]:
        x = layers.Conv1D(cfg.tcn_filters, 3, padding="causal", dilation_rate=d, activation="relu")(x)
        x = layers.Dropout(cfg.tcn_dropout)(x)
    x = layers.Conv1D(1, 1, padding="same")(x)
    out = layers.Lambda(lambda t: t[:, -1, 0])(x)
    model = Model(inp, out)
    model.compile(optimizer=optimizers.Adam(cfg.tcn_lr), loss="mse")
    return model


def build_tcn_features(close_series: np.ndarray, cfg: P4MethodConfig) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    close_arr = np.asarray(close_series, dtype=np.float32)
    close_seq = make_rolling_sequences(close_arr, cfg.tcn_seq_len)

    model = build_tcn_denoiser(cfg.tcn_seq_len, cfg)
    model.fit(
        close_seq,
        close_arr,
        epochs=cfg.tcn_epochs,
        batch_size=cfg.tcn_batch,
        verbose=0,
    )

    fitted = model.predict(close_seq, verbose=0).reshape(-1)
    resid = close_arr - fitted

    trend = np.gradient(fitted)
    curvature = np.gradient(trend)
    noise_ratio = np.abs(resid) / (np.abs(close_arr) + 1e-9)
    fitted_ret = np.diff(fitted, prepend=fitted[0]) / (np.abs(fitted) + 1e-9) * 100

    df_tcn = pd.DataFrame(
        {
            "tcn_trend": trend,
            "tcn_curvature": curvature,
            "tcn_noise_ratio": noise_ratio,
            "tcn_fitted_ret": fitted_ret,
        }
    )
    return df_tcn, fitted, resid


def build_p4_dualstream_model(seq_len: int, n_feat_a: int, n_feat_b: int, cfg: P4MethodConfig) -> Model:
    inp_a = Input(shape=(seq_len, n_feat_a), name="stream_a")
    inp_b = Input(shape=(seq_len, n_feat_b), name="stream_b")

    xa = layers.Bidirectional(
        layers.LSTM(
            cfg.lstm_units_a,
            return_sequences=True,
            dropout=cfg.lstm_dropout,
            recurrent_dropout=0.1,
            kernel_regularizer=regularizers.l2(1e-3),
        )
    )(inp_a)
    xa = layers.LayerNormalization()(xa)

    xb = layers.Bidirectional(
        layers.LSTM(
            cfg.lstm_units_b,
            return_sequences=True,
            dropout=cfg.lstm_dropout,
            recurrent_dropout=0.1,
            kernel_regularizer=regularizers.l2(1e-3),
        )
    )(inp_b)
    xb = layers.LayerNormalization()(xb)

    attn = layers.Attention()([xa, xb])
    gate = layers.Dense(xa.shape[-1], activation="sigmoid")(attn)
    fused = layers.Multiply()([xa, gate])
    fused = layers.GlobalAveragePooling1D()(fused)

    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(fused)
    x = layers.Dropout(cfg.lstm_dropout)(x)
    out = layers.Dense(len(cfg.classes), activation="softmax")(x)

    model = Model([inp_a, inp_b], out)
    model.compile(
        optimizer=optimizers.Adam(cfg.lstm_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prepare_dualstream_inputs(
    x_base: pd.DataFrame,
    y: pd.Series,
    df_tcn: pd.DataFrame,
    stream_b_cols: list[str],
    classes: tuple[str, str]-> tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder, StandardScaler, StandardScaler]):
    x_a = x_base.values.astype(np.float32)
    x_b = df_tcn[stream_b_cols].values.astype(np.float32)

    sc_a = StandardScaler().fit(x_a)
    sc_b = StandardScaler().fit(x_b)
    x_a_sc = sc_a.transform(x_a).astype(np.float32)
    x_b_sc = sc_b.transform(x_b).astype(np.float32)

    le = LabelEncoder()
    le.classes_ = np.array(list(classes))
    y_enc = le.transform(y.values)
    return x_a_sc, x_b_sc, y_enc, le, sc_a, sc_b


__all__ = [
    "P4MethodConfig",
    "load_daily",
    "label_weekly",
    "build_weekly_features",
    "build_tcn_denoiser",
    "build_tcn_features",
    "build_p4_dualstream_model",
    "prepare_dualstream_inputs",
]
