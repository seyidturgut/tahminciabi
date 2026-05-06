"""4 modelin ağırlıklı birleşik olasılık skoru.

Markov 0.20 + Bayesian 0.25 + Sapma 0.20 + ML 0.35.

Her bir alt-model 90 elemanlı normalize olasılık vektörü döner; final skor
da 90 elemanlı normalize olasılık vektörüdür.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import bayesian, markov, ml_model, randomness_tests

DEFAULT_WEIGHTS = {
    "markov": 0.20,
    "bayesian": 0.25,
    "deviation": 0.20,
    "ml": 0.35,
}


@dataclass
class ModelOutput:
    markov: np.ndarray
    bayesian: np.ndarray
    deviation: np.ndarray
    ml: np.ndarray
    final: np.ndarray
    randomness_summary: dict
    weights: dict


def _normalize(v: np.ndarray) -> np.ndarray:
    s = v.sum()
    if s <= 0:
        return np.full_like(v, 1.0 / len(v))
    return v / s


def compute(df: pd.DataFrame, weights: Optional[dict] = None,
            force_train_ml: bool = False) -> ModelOutput:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    m = _normalize(markov.next_draw_probabilities(df))
    b = _normalize(bayesian.posterior_probabilities(df))
    d = _normalize(randomness_tests.deviation_scores(df))
    ml = _normalize(ml_model.predict_next_draw_probabilities(df, force_train=force_train_ml))

    final = (
        w["markov"] * m
        + w["bayesian"] * b
        + w["deviation"] * d
        + w["ml"] * ml
    )
    final = _normalize(final)

    return ModelOutput(
        markov=m,
        bayesian=b,
        deviation=d,
        ml=ml,
        final=final,
        randomness_summary=randomness_tests.summary(df),
        weights=w,
    )


if __name__ == "__main__":
    from data_manager import load_data
    df = load_data()
    out = compute(df, force_train_ml=True)
    print(f"Final sum={out.final.sum():.4f}")
    print(f"Top 15 sayı: {np.argsort(out.final)[-15:][::-1] + 1}")
    print(f"Bottom 15 sayı: {np.argsort(out.final)[:15] + 1}")
    print(f"Chi-square p={out.randomness_summary['chi_square']['p_value']:.4f}")
