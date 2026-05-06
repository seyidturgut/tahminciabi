"""Rasgelelik istatistiksel testleri ve sapma sömürüsü.

- Chi-square goodness-of-fit (H0: uniform 6/90)
- Kolmogorov–Smirnov (frekans CDF vs uniform)
- Per-number z-score (gözlemlenen frekans − beklenen / σ)

Çıktı sapma vektörü (90,): istatistiksel anlamlı yüksek z-score'lar bir sonraki
çekilişte aynı şekilde devam edebilir (regresyon-ortalamaya doğru) — bu yüzden
"sapan sayıları sömürmek" kontekstine bağlıdır. Bu modül ham anomali skoru
verir; engine ağırlıklandırırken yorumlar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._common import detect_params


def _counts(df: pd.DataFrame) -> tuple[np.ndarray, int, list[str]]:
    cols, total = detect_params(df)
    counts = np.zeros(total)
    for col in cols:
        for v in df[col].values:
            counts[int(v) - 1] += 1
    return counts, total, cols


def chi_square(df: pd.DataFrame) -> dict:
    """Tüm geçmişin uniform'a uygunluğu."""
    counts, total, _ = _counts(df)
    expected = counts.sum() / total
    chi2 = ((counts - expected) ** 2 / expected).sum()
    p = 1 - stats.chi2.cdf(chi2, df=total - 1)
    return {
        "chi2": float(chi2),
        "df": total - 1,
        "p_value": float(p),
        "rejects_uniform_at_05": bool(p < 0.05),
    }


def ks_test(df: pd.DataFrame) -> dict:
    """Frekans dağılımının uniform CDF'e KS uzaklığı."""
    counts, total, _ = _counts(df)
    obs_cdf = np.cumsum(counts) / counts.sum()
    expected_cdf = np.arange(1, total + 1) / total
    d = float(np.max(np.abs(obs_cdf - expected_cdf)))
    n = int(counts.sum())
    p = 2 * np.exp(-2 * (d ** 2) * n)
    return {"D": d, "p_value": float(min(1.0, max(0.0, p)))}


def number_z_scores(df: pd.DataFrame) -> np.ndarray:
    """Her sayının uniform beklentiden z-skoru."""
    counts, total, _ = _counts(df)
    n_obs = counts.sum()
    p_uniform = 1.0 / total
    expected = n_obs * p_uniform
    var = n_obs * p_uniform * (1 - p_uniform)
    sd = np.sqrt(var) if var > 0 else 1.0
    return (counts - expected) / sd


def deviation_scores(df: pd.DataFrame) -> np.ndarray:
    """Sapan sayıları sömürmek için olasılık vektörü (total elemanlı)."""
    _, total, _ = _counts(df)
    z = number_z_scores(df)
    abs_z = np.abs(z)
    if abs_z.max() == 0:
        return np.full(total, 1.0 / total)
    weights = 1.0 + abs_z ** 1.8
    return weights / weights.sum()


def runs_test_odd_even(df: pd.DataFrame) -> dict:
    """Çekiliş başına tek/çift sayı sayılarının runs testi."""
    cols, _ = detect_params(df)
    odds_per_draw = []
    for _, row in df[cols].iterrows():
        odds_per_draw.append(sum(int(v) % 2 for v in row.values))
    odds_per_draw = np.array(odds_per_draw)
    median = np.median(odds_per_draw)
    above = odds_per_draw > median
    runs = 1 + np.sum(above[1:] != above[:-1])
    n1 = int(above.sum())
    n2 = len(above) - n1
    if n1 == 0 or n2 == 0:
        return {"runs": int(runs), "p_value": 1.0}
    expected = 1 + 2 * n1 * n2 / (n1 + n2)
    var = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    sd = np.sqrt(var) if var > 0 else 1.0
    z = (runs - expected) / sd
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"runs": int(runs), "z": float(z), "p_value": float(p)}


def summary(df: pd.DataFrame) -> dict:
    return {
        "chi_square": chi_square(df),
        "ks": ks_test(df),
        "runs_odd_even": runs_test_odd_even(df),
    }


if __name__ == "__main__":
    from data_manager import load_data
    df = load_data()
    import json
    print(json.dumps(summary(df), indent=2, ensure_ascii=False))
    print("\nDeviation top10:", np.argsort(deviation_scores(df))[-10:][::-1] + 1)
