from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import pywt
from scipy import signal as sps
from scipy.signal import medfilt
from scipy.stats import skew, kurtosis
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    label_binarize,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

FS = 360
WB, WA = 90, 90
BEAT_LEN = WB + WA  # 180

LABEL_MAP = {
    "N": "Normal", "L": "Normal", "R": "Normal", "e": "Normal", "j": "Normal",
    "A": "Atrial", "a": "Atrial", "J": "Atrial", "S": "Atrial",
    "V": "Other", "E": "Other", "F": "Other",
    "/": "Other", "f": "Other", "Q": "Other",
}
CLASSES_ORDER = ["Normal", "Atrial", "Other"]
N_CLASSES = 3

TRAIN_PATIENTS = [
    100, 101, 103, 105, 106, 108, 109, 111, 112, 115,
    116, 117, 118, 119, 121, 122, 123, 124, 200, 201,
    202, 203, 205, 207, 208, 209, 215,
]
TEST_PATIENTS = [220, 221, 222, 223, 228, 230, 231, 232, 233]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_record(rid: int, extract_dir: str) -> tuple[np.ndarray, list[tuple[int, str]], int]:
    csv_path = os.path.join(extract_dir, f"{rid}.csv")
    ann_path = os.path.join(extract_dir, f"{rid}annotations.txt")

    sig_df = pd.read_csv(csv_path)
    sig_df.columns = [c.strip().strip("'\"") for c in sig_df.columns]
    col_map = {c.upper(): c for c in sig_df.columns}

    mlii_col = col_map.get("MLII")
    if mlii_col is None:
        num_cols = [
            c for c in sig_df.columns
            if sig_df[c].dtype.kind in "iuf" and "sample" not in c.lower()
        ]
        mlii_col = num_cols[0] if num_cols else None

    mlii = sig_df[mlii_col].interpolate("linear").ffill().bfill().values.astype(float)

    beats_ann: list[tuple[int, str]] = []
    with open(ann_path, "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                sample_pos = int(parts[1])
                beat_type = parts[2]
                beats_ann.append((sample_pos, beat_type))
            except ValueError:
                continue

    return mlii, beats_ann, len(mlii)


# =============================================================================
# PREPROCESSING
# =============================================================================
def remove_baseline(sig: np.ndarray) -> np.ndarray:
    baseline = medfilt(sig, 71)
    baseline = medfilt(baseline, 215)
    return sig - baseline


def bandpass_filter(
    x: np.ndarray, lo: float = 0.5, hi: float = 40.0, fs: int = FS, order: int = 4
) -> np.ndarray:
    nyq = fs / 2
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, np.asarray(x, dtype=float))


def preprocess_signal(sig: np.ndarray) -> np.ndarray:
    sig = remove_baseline(sig.astype(float))
    sig = bandpass_filter(sig)
    return sig


# =============================================================================
# RR / BPM / RR_z FEATURES
# =============================================================================
def compute_rr_bpm_zscore(
    r_positions: np.ndarray, fs: int = FS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(r_positions)
    if n < 2:
        return np.zeros(n), np.zeros(n), np.zeros(n)

    rr_samples = np.diff(r_positions)
    rr_seconds = rr_samples / fs
    RR = np.concatenate([[rr_seconds[0]], rr_seconds])
    BPM = 60.0 / np.clip(RR, 0.2, 3.0)
    mu, sd = RR.mean(), RR.std() + 1e-8
    RR_z = (RR - mu) / sd
    return RR, BPM, RR_z


# =============================================================================
# MORPHOLOGICAL + WAVELET FEATURES
# =============================================================================
def feat_morph(b: np.ndarray) -> list[float]:
    b = np.asarray(b, dtype=float)
    return [
        float(b.mean()),
        float(b.std()),
        float(b.min()),
        float(b.max()),
        float(b.argmax()) / len(b),
        float(b.argmin()) / len(b),
        float(skew(b)),
        float(kurtosis(b)),
        float(np.trapz(np.abs(b))),
        float(np.sum(np.diff(b) ** 2)),
        float(np.max(np.abs(np.diff(b)))) if len(b) > 1 else 0.0,
        float(np.sum(b > 0)) / len(b),
        float(np.percentile(b, 25)),
        float(np.percentile(b, 75)),
    ]


def feat_wavelet(b: np.ndarray, wavelet: str = "db1", level: int = 3) -> list[float]:
    coeffs = pywt.wavedec(np.asarray(b, dtype=float), wavelet, level=level)
    feats: list[float] = []
    for c in coeffs:
        feats += [float(np.sum(c**2)), float(np.abs(c).max()), float(c.mean())]
    return feats


def extract_handcrafted(
    beat: np.ndarray, rr: float, bpm: float, rr_z: float
) -> list[float]:
    return feat_morph(beat) + feat_wavelet(beat) + [rr, bpm, rr_z]


def sanitize(X: np.ndarray) -> np.ndarray:
    X = np.array(X, dtype=float)
    X = np.where(np.isinf(X), np.nan, X)
    df = pd.DataFrame(X)
    return df.interpolate("linear").ffill().bfill().fillna(0).values


# =============================================================================
# DATASET BUILDING
# =============================================================================
def build_per_patient_beats(
    rid: int, extract_dir: str
) -> list[dict]:
    try:
        mlii, beats_ann, sig_len = load_record(rid, extract_dir)
    except (FileNotFoundError, Exception) as e:
        print(f"  [WARN] record {rid}: {e}")
        return []

    sig_clean = preprocess_signal(mlii)

    r_positions: list[int] = []
    labels: list[str] = []
    for sample_pos, beat_type in beats_ann:
        if beat_type not in LABEL_MAP:
            continue
        if sample_pos - WB < 0 or sample_pos + WA >= sig_len:
            continue
        r_positions.append(sample_pos)
        labels.append(LABEL_MAP[beat_type])

    r_positions_arr = np.array(r_positions)
    if len(r_positions_arr) < 2:
        return []

    RR, BPM, RR_z = compute_rr_bpm_zscore(r_positions_arr, fs=FS)

    out: list[dict] = []
    for i, (s, lab) in enumerate(zip(r_positions_arr, labels)):
        beat = sig_clean[s - WB : s + WA]
        feat = extract_handcrafted(beat, RR[i], BPM[i], RR_z[i])
        out.append(
            {
                "beat": beat.astype(np.float32),
                "rr": float(RR[i]),
                "bpm": float(BPM[i]),
                "rr_z": float(RR_z[i]),
                "feat": feat,
                "label": lab,
                "patient": rid,
            }
        )
    return out


def build_dataset(
    extract_dir: str,
) -> tuple[list[dict], list[dict]]:
    train_beats: list[dict] = []
    test_beats: list[dict] = []

    print("\n  Loading TRAIN patients ...")
    for rid in tqdm(TRAIN_PATIENTS):
        train_beats += build_per_patient_beats(rid, extract_dir)

    print("\n  Loading TEST patients ...")
    for rid in tqdm(TEST_PATIENTS):
        test_beats += build_per_patient_beats(rid, extract_dir)

    return train_beats, test_beats


def beats_to_arrays(
    beats_list: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W = np.array([b["beat"] for b in beats_list], dtype=np.float32)[..., None]
    F = np.array([b["feat"] for b in beats_list], dtype=np.float32)
    RR = np.array(
        [[b["rr"], b["bpm"], b["rr_z"]] for b in beats_list], dtype=np.float32
    )
    y = np.array([b["label"] for b in beats_list])
    pid = np.array([b["patient"] for b in beats_list])
    return W, F, RR, y, pid


# =============================================================================
# RR SEQUENCE BUILDING
# =============================================================================
def build_rr_sequences(
    X_rr: np.ndarray, pid: np.ndarray, seq_len: int = 10
) -> np.ndarray:
    N = len(X_rr)
    seqs = np.zeros((N, seq_len, 3), dtype=np.float32)
    for i in range(N):
        cur_pid = pid[i]
        start = max(0, i - seq_len + 1)
        candidate_idx = [j for j in range(start, i + 1) if pid[j] == cur_pid]
        seq = X_rr[candidate_idx]
        if len(seq) < seq_len:
            pad = np.repeat(seq[:1], seq_len - len(seq), axis=0)
            seq = np.vstack([pad, seq])
        seqs[i] = seq[-seq_len:]
    return seqs


# =============================================================================
# NORMALIZATION
# =============================================================================
def zscore_waveform(W: np.ndarray) -> np.ndarray:
    mu = W.mean(axis=1, keepdims=True)
    sd = W.std(axis=1, keepdims=True) + 1e-8
    return (W - mu) / sd


def standardize_split(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    sc = StandardScaler()
    N, T, C = X_train.shape
    X_tr_s = sc.fit_transform(X_train.reshape(-1, C)).reshape(N, T, C)
    X_te_s = sc.transform(X_test.reshape(-1, C)).reshape(len(X_test), T, C)
    return sc, X_tr_s, X_te_s


# =============================================================================
# SMOTE (multi-input compatible)
# =============================================================================
def apply_smote_multi(
    X_w: np.ndarray, X_f: np.ndarray, X_seq: np.ndarray, y: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from imblearn.over_sampling import SMOTE

    Nw, T_w, _ = X_w.shape
    N_, T_s, C = X_seq.shape
    X_comb = np.hstack(
        [X_w.reshape(Nw, T_w), X_f, X_seq.reshape(N_, T_s * C)]
    )
    try:
        counts = np.bincount(y)
        min_c = counts[counts > 0].min()
        k_nn = max(1, min(5, min_c - 1))
        X_sm, y_sm = SMOTE(random_state=seed, k_neighbors=k_nn).fit_resample(
            X_comb, y
        )
    except Exception as e:
        print(f"  [SMOTE warn] {e} — skipping SMOTE")
        return X_w, X_f, X_seq, y

    X_w_sm = X_sm[:, :T_w].reshape(-1, T_w, 1)
    X_f_sm = X_sm[:, T_w : T_w + X_f.shape[1]]
    X_seq_sm = X_sm[:, T_w + X_f.shape[1] :].reshape(-1, T_s, C)
    return X_w_sm, X_f_sm, X_seq_sm, y_sm


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, n_cls: int
) -> dict:
    m: dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f2": fbeta_score(
            y_true, y_pred, average="macro", beta=2, zero_division=0
        ),
    }
    try:
        yt_bin = label_binarize(y_true, classes=np.arange(n_cls))
        valid = yt_bin.sum(axis=0) > 0
        m["auc_roc"] = (
            roc_auc_score(
                yt_bin[:, valid], y_proba[:, valid], multi_class="ovr", average="macro"
            )
            if valid.sum() > 1
            else float("nan")
        )
    except Exception:
        m["auc_roc"] = float("nan")
    return m


def print_metrics(name: str, m: dict) -> None:
    print(
        f"  [{name:18s}]  acc={m['accuracy']:.4f}  "
        f"F1={m['f1']:.4f}  F2={m['f2']:.4f}  AUC={m['auc_roc']:.4f}"
    )


__all__ = [
    "FS", "WB", "WA", "BEAT_LEN",
    "LABEL_MAP", "CLASSES_ORDER", "N_CLASSES",
    "TRAIN_PATIENTS", "TEST_PATIENTS",
    "seed_everything",
    "load_record", "remove_baseline", "bandpass_filter", "preprocess_signal",
    "compute_rr_bpm_zscore",
    "feat_morph", "feat_wavelet", "extract_handcrafted", "sanitize",
    "build_per_patient_beats", "build_dataset", "beats_to_arrays",
    "build_rr_sequences",
    "zscore_waveform", "standardize_split",
    "apply_smote_multi",
    "evaluate_predictions", "print_metrics",
]