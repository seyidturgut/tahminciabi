"""Bayesian frekans tahmini (Dirichlet–Categorical).

Prior: Dirichlet(α=1) — uniform.
Posterior: α_i' = α_i + Σ ağırlıklı_gözlem_i.
Recency weighting: son çekilişler exponential decay ile daha ağırlıklı.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import detect_params


def posterior_probabilities(
    df: pd.DataFrame,
    prior_alpha: float = 1.0,
    half_life: int = 200,
) -> np.ndarray:
    """Recency-weighted posterior beklenen değer (total elemanlı vektör)."""
    cols, total = detect_params(df)
    n = len(df)
    if n == 0:
        return np.full(total, 1.0 / total)

    decay = np.log(2) / max(half_life, 1)
    ages = np.arange(n)[::-1]
    weights = np.exp(-decay * ages)

    counts = np.full(total, prior_alpha, dtype=float)
    draws = df[cols].values
    for w, draw in zip(weights, draws):
        for num in draw:
            counts[int(num) - 1] += w

    return counts / counts.sum()


def credible_interval(
    df: pd.DataFrame,
    prior_alpha: float = 1.0,
    half_life: int = 200,
    level: float = 0.95,
) -> np.ndarray:
    """Dirichlet marjinal Beta yaklaşımı ile %95 güven aralığı (total, 2)."""
    from scipy.stats import beta

    cols, total = detect_params(df)
    n = len(df)
    decay = np.log(2) / max(half_life, 1)
    ages = np.arange(n)[::-1]
    weights = np.exp(-decay * ages)

    counts = np.full(total, prior_alpha, dtype=float)
    draws = df[cols].values
    for w, draw in zip(weights, draws):
        for num in draw:
            counts[int(num) - 1] += w

    alpha_total = counts.sum()
    lo = (1 - level) / 2
    hi = 1 - lo
    intervals = np.zeros((total, 2))
    for i in range(total):
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
