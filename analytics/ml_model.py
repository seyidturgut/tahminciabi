"""LightGBM per-number sınıflandırıcı.

Her sayı (1-90) için binary classifier:
"Bir sonraki çekilişte bu sayı çıkacak mı?"

Feature mühendisliği:
- gap (en son çıkıştan beri kaç çekiliş)
- son 10/30/100/300 çekilişteki frekans
- son çıktığı çekilişteki pozisyon (1-6)
- sayının ortalaması (1-90 üzerinden absolute)
- haftanın günü, ayın haftası (sin/cos encoding)

Train: rolling/walk-forward CV. Cache: ~/.tahminci_cache/lgbm.pkl (7 gün).
LightGBM kurulu değilse fallback: sklearn GradientBoostingClassifier.
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ._common import detect_params

CACHE_DIR = Path.home() / ".tahminci_cache"
CACHE_TTL_SEC = 7 * 24 * 3600


def _cache_file_for(total: int) -> Path:
    """Oyuna özel cache dosyası — total değerine göre."""
    return CACHE_DIR / f"lgbm_total{total}_v2.pkl"

try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
    from sklearn.ensemble import GradientBoostingClassifier


def _build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Her (çekiliş_t, sayı_n) çiftini bir gözleme dönüştürür."""
    cols, total = detect_params(df)
    df = df.sort_values("tarih").reset_index(drop=True)
    n_draws = len(df)
    draws = df[cols].values
    presence = np.zeros((n_draws, total), dtype=np.int8)
    for t in range(n_draws):
        for v in draws[t]:
            presence[t, int(v) - 1] = 1

    dates = pd.to_datetime(df["tarih"])
    dow = dates.dt.dayofweek.values
    week = ((dates.dt.day - 1) // 7).values
    dow_sin, dow_cos = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    week_sin, week_cos = np.sin(2 * np.pi * week / 5), np.cos(2 * np.pi * week / 5)

    features = []
    targets = []
    warmup = 50
    for t in range(warmup, n_draws):
        for num_idx in range(total):
            # Gap
            past = presence[:t, num_idx]
            last_seen = np.where(past == 1)[0]
            gap = (t - last_seen[-1]) if len(last_seen) > 0 else t

            # Window frequencies
            f10 = past[-10:].sum()
            f30 = past[-30:].sum()
            f100 = past[-100:].sum()
            f300 = past[-300:].sum() if t >= 300 else past.sum() * 300 / max(t, 1)

            # Last position (1-6) — eğer son çıktıysa
            last_pos = 0
            if len(last_seen) > 0:
                last_t = last_seen[-1]
                row = draws[last_t]
                pos = int(np.where(row == (num_idx + 1))[0][0]) + 1 if (num_idx + 1) in row else 0
                last_pos = pos

            features.append([
                num_idx + 1,  # sayı kimliği
                gap,
                f10, f30, f100, f300,
                last_pos,
                dow_sin[t], dow_cos[t],
                week_sin[t], week_cos[t],
            ])
            targets.append(presence[t, num_idx])

    X = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=int)
    return X, y


def _load_cache(total: int):
    cache_file = _cache_file_for(total)
    if not cache_file.exists():
        return None
    age = time.time() - cache_file.stat().st_mtime
    if age > CACHE_TTL_SEC:
        return None
    try:
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_cache(obj, total: int):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = _cache_file_for(total)
    with open(cache_file, "wb") as f:
        pickle.dump(obj, f)


def train_model(df: pd.DataFrame, force: bool = False, verbose: bool = False):
    """Modeli eğitir veya cache'ten yükler."""
    _, total = detect_params(df)
    if not force:
        cached = _load_cache(total)
        if cached is not None:
            return cached

    X, y = _build_features(df)

    if HAS_LGBM:
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            reg_lambda=1.0,
            verbose=-1,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3
        )
    model.fit(X, y)
    if verbose:
        try:
            from sklearn.metrics import roc_auc_score, log_loss
            preds = model.predict_proba(X)[:, 1]
            print(f"Train AUC={roc_auc_score(y, preds):.4f}, "
                  f"LogLoss={log_loss(y, preds):.4f}")
        except Exception:
            pass

    bundle = {"model": model, "trained_at": time.time(), "n_samples": len(y),
              "total": total}
    _save_cache(bundle, total)
    return bundle


def predict_next_draw_probabilities(df: pd.DataFrame, force_train: bool = False) -> np.ndarray:
    """Her sayı için bir sonraki çekilişte çıkma olasılığı (total elemanlı)."""
    cols, total = detect_params(df)
    df = df.sort_values("tarih").reset_index(drop=True)
    bundle = train_model(df, force=force_train)
    model = bundle["model"]

    n_draws = len(df)
    presence = np.zeros((n_draws, total), dtype=np.int8)
    draws = df[cols].values
    for t in range(n_draws):
        for v in draws[t]:
            presence[t, int(v) - 1] = 1

    last_date = pd.to_datetime(df["tarih"]).iloc[-1]
    next_date = last_date + pd.Timedelta(days=2)
    dow = next_date.dayofweek
    week = (next_date.day - 1) // 7
    dow_sin, dow_cos = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    week_sin, week_cos = np.sin(2 * np.pi * week / 5), np.cos(2 * np.pi * week / 5)

    features = []
    for num_idx in range(total):
        past = presence[:, num_idx]
        last_seen = np.where(past == 1)[0]
        gap = (n_draws - last_seen[-1]) if len(last_seen) > 0 else n_draws
        f10 = past[-10:].sum()
        f30 = past[-30:].sum()
        f100 = past[-100:].sum()
        f300 = past[-300:].sum() if n_draws >= 300 else past.sum() * 300 / max(n_draws, 1)
        last_pos = 0
        if len(last_seen) > 0:
            last_t = last_seen[-1]
            row = draws[last_t]
            if (num_idx + 1) in row:
                last_pos = int(np.where(row == (num_idx + 1))[0][0]) + 1
        features.append([
            num_idx + 1, gap, f10, f30, f100, f300, last_pos,
            dow_sin, dow_cos, week_sin, week_cos,
        ])
    X_next = np.asarray(features, dtype=float)
    probs = model.predict_proba(X_next)[:, 1]
    if probs.sum() <= 0:
        return np.full(total, 1.0 / total)
    return probs / probs.sum()


def cv_report(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """Walk-forward cross-validation: son n_splits çekilişte test."""
    from sklearn.metrics import roc_auc_score, log_loss

    X, y = _build_features(df)
    fold_size = len(X) // (n_splits + 1)
    aucs, losses = [], []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = fold_size * (i + 2)
        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[train_end:test_end], y[train_end:test_end]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        if HAS_LGBM:
            m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, verbose=-1)
        else:
            m = GradientBoostingClassifier(n_estimators=80, learning_rate=0.05, max_depth=3)
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te, p))
        losses.append(log_loss(y_te, p))
    return {
        "n_folds": len(aucs),
        "mean_auc": float(np.mean(aucs)) if aucs else 0.0,
        "mean_logloss": float(np.mean(losses)) if losses else 0.0,
    }


if __name__ == "__main__":
    from data_manager import load_data
    df = load_data()
    print("CV report:", cv_report(df))
    p = predict_next_draw_probabilities(df, force_train=True)
    print(f"ML sum={p.sum():.4f}, top10={np.argsort(p)[-10:][::-1] + 1}")
