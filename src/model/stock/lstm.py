from __future__ import annotations

import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model, callbacks, layers, optimizers

from .utils.stock_utils import make_rolling_sequences, seed_everything


class LSTMDenoiserConfig:
    def __init__(
        self,
        window: int = 26,
        units: int = 32,
        latent_dim: int = 16,
        epochs: int = 60,
        batch_size: int = 32,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        self.window = window
        self.units = units
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed


class LSTMDenoiser:
    def __init__(self, cfg: LSTMDenoiserConfig | None = None):
        self.cfg = cfg or LSTMDenoiserConfig()
        self.window = self.cfg.window
        self.units = self.cfg.units
        self.latent_dim = self.cfg.latent_dim
        self.epochs = self.cfg.epochs
        self.batch_size = self.cfg.batch_size
        self.lr = self.cfg.lr
        self._ae = None
        self._enc = None
        seed_everything(self.cfg.seed)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        tf.random.set_seed(self.cfg.seed)

    def _build(self):
        tf.keras.backend.clear_session()
        w = self.window

        inp = Input(shape=(w, 1), name="dn_input")
        x = layers.Bidirectional(
            layers.LSTM(self.units, return_sequences=True, dropout=0.10, name="enc_bilstm"),
            name="enc_bi",
        )(inp)
        x = layers.LayerNormalization(name="enc_norm")(x)
        z = layers.LSTM(self.latent_dim, return_sequences=False, dropout=0.10, name="enc_bottleneck")(x)

        d = layers.RepeatVector(w, name="repeat")(z)
        d = layers.LSTM(self.units, return_sequences=True, dropout=0.10, name="dec_lstm")(d)
        d = layers.LayerNormalization(name="dec_norm")(d)
        out = layers.TimeDistributed(layers.Dense(1), name="dec_out")(d)

        ae = Model(inp, out, name="lstm_autoencoder")
        enc = Model(inp, z, name="lstm_encoder")
        ae.compile(optimizer=optimizers.Adam(learning_rate=self.lr), loss="mse")
        self._ae = ae
        self._enc = enc

    @staticmethod
    def _instance_norm(seg: np.ndarray):
        mu = seg.mean()
        sig = seg.std() + 1e-8
        return (seg - mu) / sig, mu, sig

    def _build_windows(self, close_arr: np.ndarray):
        n = len(close_arr)
        w = self.window
        x = np.zeros((n, w, 1), dtype=np.float32)
        mus = np.zeros(n, dtype=np.float64)
        sigs = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - w + 1)
            seg = close_arr[start : i + 1].astype(np.float32)
            if len(seg) < w:
                pad = np.full(w - len(seg), seg[0], dtype=np.float32)
                seg = np.concatenate([pad, seg])
            seg_n, mu, sg = self._instance_norm(seg)
            x[i, :, 0] = seg_n
            mus[i] = mu
            sigs[i] = sg
        return x, mus, sigs

    def fit(self, close_arr: np.ndarray, verbose: int = 0):
        if self._ae is None:
            self._build()
        x, _, _ = self._build_windows(close_arr)
        if len(x) < max(8, self.batch_size):
            return self
        val_split = float(np.clip(20 / len(x), 0.05, 0.20))
        self._ae.fit(
            x,
            x,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=val_split,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, min_delta=1e-5),
                callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0),
            ],
            shuffle=True,
            verbose=verbose,
        )
        return self

    def transform(self, close_arr: np.ndarray):
        if self._ae is None:
            raise RuntimeError("LSTMDenoiser must be fitted before transform().")
        x, mus, sigs = self._build_windows(close_arr)
        recon_norm = self._ae.predict(x, batch_size=128, verbose=0)
        latents = self._enc.predict(x, batch_size=128, verbose=0)
        fitted = recon_norm[:, -1, 0].astype(np.float64) * sigs + mus
        resid = close_arr.astype(np.float64) - fitted
        resid_abs = np.abs(resid)
        trend = np.gradient(fitted)
        curvature = np.gradient(trend)
        detrended_ret = resid / (np.abs(close_arr) + 1e-9) * 100
        roll_std = pd.Series(close_arr).rolling(self.window // 2, min_periods=2).std().fillna(1.0).values
        noise_ratio = resid_abs / (roll_std + 1e-9)
        fitted_ret = np.diff(fitted, prepend=fitted[0]) / (np.abs(fitted) + 1e-9) * 100
        feat = {
            "lstm_fitted": fitted,
            "lstm_resid": resid,
            "lstm_resid_abs": resid_abs,
            "lstm_trend": trend,
            "lstm_curvature": curvature,
            "lstm_detrended_ret": detrended_ret,
            "lstm_noise_ratio": noise_ratio,
            "lstm_fitted_ret": fitted_ret,
        }
        for k in range(self.latent_dim):
            feat[f"lstm_latent_{k}"] = latents[:, k].astype(np.float64)
        return pd.DataFrame(feat), fitted, resid


def build_lstm_denoiser_features(close_arr: np.ndarray, cfg: LSTMDenoiserConfig):
    denoiser = LSTMDenoiser(cfg)
    denoiser.fit(close_arr, verbose=0)
    df_feat, fitted, resid = denoiser.transform(close_arr)
    return df_feat, fitted, resid, denoiser
