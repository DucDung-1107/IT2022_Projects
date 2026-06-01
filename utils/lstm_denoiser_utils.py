from __future__ import annotations

import random

import numpy as np
import pandas as pd


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resample_weekly(df_daily: pd.DataFrame, rule: str = "W-FRI") -> pd.DataFrame:
    df_d = df_daily.copy()
    df_d["date"] = pd.to_datetime(df_d["date"])
    df_d = df_d.set_index("date").sort_index()
    df_d["volume"] = df_d["volume"] / 1e6

    wk = (
        df_d.resample(rule)
        .agg(
            close=("close", "last"),
            high=("high", "max"),
            low=("low", "min"),
            volume=("volume", "sum"),
            open=("close", "first"),
        )
        .dropna()
        .reset_index()
    )
    wk["mid"] = (wk["high"] + wk["low"]) / 2
    return wk


def make_sequences(x_arr: np.ndarray, seq_len: int) -> np.ndarray:
    x_arr = np.asarray(x_arr, dtype=np.float32)
    n, f = x_arr.shape
    seqs = np.zeros((n, seq_len, f), dtype=np.float32)
    for i in range(n):
        start = max(0, i - seq_len + 1)
        seg = x_arr[start : i + 1]
        if len(seg) < seq_len:
            pad = np.tile(seg[0:1], (seq_len - len(seg), 1))
            seg = np.concatenate([pad, seg], axis=0)
        seqs[i] = seg
    return seqs


def make_rolling_sequences(series: np.ndarray, seq_len: int) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float32)
    n = len(arr)
    seqs = np.zeros((n, seq_len, 1), dtype=np.float32)
    for i in range(n):
        start = max(0, i - seq_len + 1)
        seg = arr[start : i + 1]
        if len(seg) < seq_len:
            pad = np.repeat(seg[0:1], seq_len - len(seg))
            seg = np.concatenate([pad, seg])
        seqs[i, :, 0] = seg
    return seqs
