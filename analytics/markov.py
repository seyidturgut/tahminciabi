"""Sayı geçiş Markov zinciri.

P(sayı_t = j | önceki çekilişte i çıktı) — 90×90 geçiş matrisi.
Her çekiliş 6 sayı içerdiği için bir önceki çekilişten her sayı, sonraki
çekilişin 6 sayısının her biri için bir geçiş örneği üretir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMBER_COLS = [f"sayi_{i+1}" for i in range(6)]
TOTAL = 90


def build_transition_matrix(df: pd.DataFrame, smoothing: float = 1.0) -> np.ndarray:
    """
    Laplace smoothing ile geçiş matrisi.

    Returns:
        T (90,90) — T[i,j] = P(sayı j sonraki çekilişte | i bu çekilişte)
    """
    counts = np.full((TOTAL, TOTAL), smoothing, dtype=float)
    draws = df[NUMBER_COLS].values  # (n_draws, 6)
    for prev, curr in zip(draws[:-1], draws[1:]):
        for i in prev:
            for j in curr:
                counts[i - 1, j - 1] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums


def next_draw_probabilities(df: pd.DataFrame, smoothing: float = 1.0) -> np.ndarray:
    """
    Bir sonraki çekilişte her sayının çıkma Markov olasılığı (90 elemanlı vektör).

    Son çekilişin 6 sayısının her biri için T[i,:] sırasını topla, normalleştir.
    """
    if len(df) < 2:
        return np.full(TOTAL, 1.0 / TOTAL)
    T = build_transition_matrix(df, smoothing=smoothing)
    last = df.iloc[-1][NUMBER_COLS].values
    scores = np.zeros(TOTAL)
    for i in last:
        scores += T[i - 1]
    scores /= scores.sum()
    return scores


if __name__ == "__main__":
    from data_manager import load_data
    df = load_data()
    p = next_draw_probabilities(df)
    print(f"Markov sum={p.sum():.4f}, top10={np.argsort(p)[-10:][::-1] + 1}")
