"""Bayesian frekans tahmini (Dirichlet–Categorical).

Prior: Dirichlet(α=1) — uniform.
Posterior: α_i' = α_i + Σ ağırlıklı_gözlem_i.
Recency weighting: son çekilişler exponential decay ile daha ağırlıklı.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMBER_COLS = [f"sayi_{i+1}" for i in range(6)]
TOTAL = 90


def posterior_probabilities(
    df: pd.DataFrame,
    prior_alpha: float = 1.0,
    half_life: int = 200,
) -> np.ndarray:
    """
    Recency-weighted posterior beklenen değer.

    Args:
        df: Çekiliş geçmişi (kronolojik sırada).
        prior_alpha: Dirichlet uniform prior parametresi.
        half_life: Bu kadar çekiliş öncesindeki gözlem yarı ağırlığa düşer.

    Returns:
        (90,) — her sayı için E[θ_i] posterior beklenen olasılığı (toplam 1).
    """
    n = len(df)
    if n == 0:
        return np.full(TOTAL, 1.0 / TOTAL)

    # Recency weights: en yeni çekiliş ağırlığı 1, half_life kadar önceki 0.5.
    decay = np.log(2) / max(half_life, 1)
    ages = np.arange(n)[::-1]  # 0=en yeni, n-1=en eski
    weights = np.exp(-decay * ages)

    counts = np.full(TOTAL, prior_alpha, dtype=float)
    draws = df[NUMBER_COLS].values
    for w, draw in zip(weights, draws):
        for num in draw:
            counts[num - 1] += w

    return counts / counts.sum()


def credible_interval(
    df: pd.DataFrame,
    prior_alpha: float = 1.0,
    half_life: int = 200,
    level: float = 0.95,
) -> np.ndarray:
    """Dirichlet marjinal Beta yaklaşımı ile %95 güven aralığı (90,2)."""
    from scipy.stats import beta

    n = len(df)
    decay = np.log(2) / max(half_life, 1)
    ages = np.arange(n)[::-1]
    weights = np.exp(-decay * ages)
    total_w = weights.sum() * 6  # her çekiliş 6 gözlem

    counts = np.full(TOTAL, prior_alpha, dtype=float)
    draws = df[NUMBER_COLS].values
    for w, draw in zip(weights, draws):
        for num in draw:
            counts[num - 1] += w

    alpha_total = counts.sum()
    lo = (1 - level) / 2
    hi = 1 - lo
    intervals = np.zeros((TOTAL, 2))
    for i in range(TOTAL):
        a = counts[i]
        b = alpha_total - a
        intervals[i, 0] = beta.ppf(lo, a, b)
        intervals[i, 1] = beta.ppf(hi, a, b)
    return intervals


if __name__ == "__main__":
    from data_manager import load_data
    df = load_data()
    p = posterior_probabilities(df)
    print(f"Bayesian sum={p.sum():.4f}, top10={np.argsort(p)[-10:][::-1] + 1}")
