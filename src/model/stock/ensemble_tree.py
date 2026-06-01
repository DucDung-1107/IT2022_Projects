from __future__ import annotations

import gc
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

from .metrics.classification_metrics import evaluate_directional_metrics
from .utils.stock_utils import make_sequences


@dataclass(frozen=True)
class TreeEnsembleConfig:
    seed: int = 42
    n_estimators_lgb: int = 500
    max_depth_lgb: int = 4
    num_leaves_lgb: int = 15
    learning_rate_lgb: float = 0.02
    subsample_lgb: float = 0.8
    colsample_lgb: float = 0.8
    min_child_samples_lgb: int = 5
    reg_alpha_lgb: float = 0.1
    reg_lambda_lgb: float = 1.0
    n_estimators_rf: int = 300
    max_depth_rf: int = 6
    min_samples_split_rf: int = 6
    min_samples_leaf_rf: int = 2
    batch_size: int = 16


def walk_forward_ensemble(
    x_all: np.ndarray,
    y_all_enc: np.ndarray,
    label_list: list[str],
    n_init: int,
    n_folds: int,
    fold_size: int,
    cfg: TreeEnsembleConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame | None]:
    n = len(y_all_enc)
    n_classes = len(label_list)
    all_true, all_pred, all_proba = [], [], []

    sc = StandardScaler().fit(x_all[:n_init])
    x_sc_all = sc.transform(x_all).astype(np.float32)

    lgb_model = None
    rf_model = None

    for fold in range(n_folds):
        tr_end = n_init + fold * fold_size
        va_start = tr_end
        va_end = min(tr_end + fold_size, n) if fold < n_folds - 1 else n
        if va_start >= n:
            break

        x_tr = x_sc_all[:tr_end]
        y_tr = y_all_enc[:tr_end]
        x_va = x_sc_all[va_start:va_end]
        y_va = y_all_enc[va_start:va_end]

        counts = np.bincount(y_tr, minlength=n_classes)
        class_w = {i: len(y_tr) / (n_classes * max(c, 1)) for i, c in enumerate(counts)}
        sample_w = np.array([class_w[y] for y in y_tr])

        lgb_model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=cfg.n_estimators_lgb,
            max_depth=cfg.max_depth_lgb,
            num_leaves=cfg.num_leaves_lgb,
            learning_rate=cfg.learning_rate_lgb,
            subsample=cfg.subsample_lgb,
            colsample_bytree=cfg.colsample_lgb,
            min_child_samples=cfg.min_child_samples_lgb,
            reg_alpha=cfg.reg_alpha_lgb,
            reg_lambda=cfg.reg_lambda_lgb,
            random_state=cfg.seed,
            n_jobs=-1,
            device="cpu",
            verbose=-1,
        )
        lgb_model.fit(x_tr, y_tr, sample_weight=sample_w)

        rf_model = RandomForestClassifier(
            n_estimators=cfg.n_estimators_rf,
            max_depth=cfg.max_depth_rf,
            min_samples_split=cfg.min_samples_split_rf,
            min_samples_leaf=cfg.min_samples_leaf_rf,
            max_features="sqrt",
            class_weight=class_w,
            random_state=cfg.seed,
            n_jobs=-1,
        )
        rf_model.fit(x_tr, y_tr)

        proba = 0.60 * lgb_model.predict_proba(x_va) + 0.40 * rf_model.predict_proba(x_va)
        pred = proba.argmax(axis=1)

        all_true.extend(y_va.tolist())
        all_pred.extend(pred.tolist())
        all_proba.extend(proba.tolist())

    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)
    proba_arr = np.asarray(all_proba)

    imp_df = None
    if lgb_model is not None and rf_model is not None:
        imp_df = pd.DataFrame(
            {
                "feature": [f"f{i}" for i in range(x_all.shape[1])],
                "importance": 0.60 * lgb_model.feature_importances_ + 0.40 * rf_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

    return y_true, y_pred, proba_arr, imp_df
