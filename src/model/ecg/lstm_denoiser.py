"""
LSTM Denoising Autoencoder + Multi-Input BiLSTM Attention Classifier for ECG.

Extracted from `ex_ecg_lstm_denoiser.ipynb`:
    - Phase 1: Filtered ECG → Multi-Input BiLSTM+Attention (Baseline)
    - Phase 2: LSTM Denoising Autoencoder → Denoised → BiLSTM Classifier

Usage:
    dae = build_lstm_dae(beat_len=180)
    cls = build_multi_input_classifier(beat_len=180, n_feat=50, seq_len=10, seq_chan=3, n_classes=3)
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Input, Model, layers, optimizers, regularizers


# =============================================================================
# Attention Layer (Bahdanau additive attention)
# =============================================================================
class AttentionLayer(layers.Layer):
    """Bahdanau additive attention mechanism."""

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        D = input_shape[-1]
        self.W = self.add_weight(
            shape=(D, self.units), initializer="glorot_uniform", name="W"
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", name="b"
        )
        self.u = self.add_weight(
            shape=(self.units, 1), initializer="glorot_uniform", name="u"
        )

    def call(self, x):
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.squeeze(tf.tensordot(uit, self.u, axes=1), -1)
        a = tf.expand_dims(tf.nn.softmax(ait, axis=1), -1)
        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


# =============================================================================
# Model 1: LSTM Denoising Autoencoder
# =============================================================================
def build_lstm_dae(
    beat_len: int,
    lstm_units_enc: tuple[int, int] = (48, 24),
    lstm_units_dec: tuple[int, int] = (24, 48),
    lr: float = 1e-3,
) -> Model:
    """
    LSTM Denoising Autoencoder.

    Architecture:
        Input (beat_len, 1)
        → BiLSTM(48, return_sequences=True)
        → BiLSTM(24, return_sequences=False)
        → LayerNorm → RepeatVector(beat_len)
        → BiLSTM(24, return_sequences=True)
        → BiLSTM(48, return_sequences=True)
        → TimeDistributed(Dense(1, linear))

    Parameters
    ----------
    beat_len : int
        Number of samples per beat (e.g. 180).
    lstm_units_enc : tuple[int, int]
        Units for encoder BiLSTM layers.
    lstm_units_dec : tuple[int, int]
        Units for decoder BiLSTM layers.
    lr : float
        Adam learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled LSTM DAE (loss='mse').
    """
    inp = Input(shape=(beat_len, 1), name="noisy_ecg")

    # Encoder
    x = layers.Bidirectional(
        layers.LSTM(lstm_units_enc[0], return_sequences=True)
    )(inp)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units_enc[1], return_sequences=False)
    )(x)
    x = layers.LayerNormalization()(x)

    # Bottleneck
    x = layers.RepeatVector(beat_len)(x)

    # Decoder
    x = layers.Bidirectional(
        layers.LSTM(lstm_units_dec[0], return_sequences=True)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units_dec[1], return_sequences=True)
    )(x)
    out = layers.TimeDistributed(
        layers.Dense(1, activation="linear"), name="clean_ecg"
    )(x)

    ae = Model(inp, out, name="LSTM_Denoising_Autoencoder")
    ae.compile(optimizer=optimizers.Adam(lr), loss="mse")
    return ae


# =============================================================================
# Model 2: Multi-Input BiLSTM + Attention Classifier
# =============================================================================
def build_multi_input_classifier(
    beat_len: int,
    n_feat: int,
    seq_len: int,
    seq_chan: int,
    n_classes: int,
    lstm_units: int = 64,
    lstm_dropout: float = 0.3,
    lr: float = 1e-3,
) -> Model:
    """
    Multi-branch classifier for ECG arrhythmia classification.

    Branches:
        A. Waveform (beat_len, 1) → BiLSTM stack → Attention
        B. RR-sequence (seq_len, seq_chan) → BiLSTM → last hidden
        C. Handcrafted features (n_feat,) → Dense(64) → Dense(32)

    Fusion → Dense(64) → Dropout → Dense(32) → Softmax(n_classes)

    Parameters
    ----------
    beat_len : int
        Number of samples per beat waveform.
    n_feat : int
        Number of handcrafted features.
    seq_len : int
        Length of RR-interval sequence.
    seq_chan : int
        Number of channels in RR sequence (typically 3: RR, BPM, RR_z).
    n_classes : int
        Number of output classes.
    lstm_units : int
        Number of LSTM units for waveform branch.
    lstm_dropout : float
        Dropout rate for LSTM and Dense layers.
    lr : float
        Adam learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled multi-input classifier.
    """
    # ── Branch A: waveform morphology ────────────────────────────────────────
    inp_w = Input(shape=(beat_len, 1), name="wave_input")
    a = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=lstm_dropout,
            kernel_regularizer=regularizers.l2(1e-4),
        ),
        name="bilstm_wave_1",
    )(inp_w)
    a = layers.LayerNormalization()(a)
    a = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=lstm_dropout,
        ),
        name="bilstm_wave_2",
    )(a)
    a = layers.LayerNormalization()(a)
    a = AttentionLayer(units=lstm_units, name="attention_wave")(a)

    # ── Branch B: RR / BPM / RR_z sequence ───────────────────────────────────
    inp_seq = Input(shape=(seq_len, seq_chan), name="rr_seq_input")
    b = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True, dropout=lstm_dropout),
        name="bilstm_rr_1",
    )(inp_seq)
    b = layers.Bidirectional(
        layers.LSTM(16, return_sequences=False, dropout=lstm_dropout),
        name="bilstm_rr_2",
    )(b)

    # ── Branch C: handcrafted features ───────────────────────────────────────
    inp_f = Input(shape=(n_feat,), name="feat_input")
    c = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(inp_f)
    c = layers.Dropout(lstm_dropout)(c)
    c = layers.Dense(32, activation="relu")(c)

    # ── Fusion ───────────────────────────────────────────────────────────────
    z = layers.Concatenate()([a, b, c])
    z = layers.Dense(64, activation="relu")(z)
    z = layers.Dropout(lstm_dropout)(z)
    z = layers.Dense(32, activation="relu")(z)
    out = layers.Dense(n_classes, activation="softmax", name="output")(z)

    model = Model(
        inputs=[inp_w, inp_seq, inp_f],
        outputs=out,
        name="MultiInput_BiLSTM_Attention",
    )
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


__all__ = [
    "AttentionLayer",
    "build_lstm_dae",
    "build_multi_input_classifier",
]