"""Popülerlik / Avoid-the-Crowd ağırlıklandırması.

Loto kazanma şansını ARTIRMAZ — ama kazanırsan ödülü kaç kişiyle
paylaşacağını AZALTIR (Expected Value).

Çoğu oyuncu doğum tarihi seçer:
    1-12 (ay) en popüler
    13-31 (gün) çok popüler
    32-90 az tercih edilir, jackpot tek kazanan olma şansı yüksek

Resmi Milli Piyango "kazanan sayı dağılımı" verileri:
    - 1-31 aralığı toplam oynanan kuponun ~%60-70'inde yer alır
    - 32+ aralığı sadece "rastgele kupon doldurma" yapan ~%30'da görünür

Bu modül, olasılık vektörüne **post-multiplier** uygular.
"""
from __future__ import annotations

import numpy as np


def crowd_uncrowdedness_weights(total: int, boost: float = 1.4) -> np.ndarray:
    """
    Avoid-the-Crowd ağırlık vektörü (total elemanlı).

    1-31: ağırlık 1.0 (cezalanmaz, sadece bonus almaz)
    32+:  ağırlık `boost` (popüler olmayan = paylaşılmayan jackpot)

    Args:
        total: oyun aralığı (örn. 90).
        boost: 32+ sayılarına uygulanan çarpan. 1.0 = etkisiz, 1.4 = orta,
            1.8+ = agresif (filtreleri aşırı zorlayabilir).

    Returns:
        (total,) — element-wise prob vektörüyle çarpılır.
    """
    w = np.ones(total)
    for i in range(31, total):  # index 31 = sayı 32
        w[i] = boost
    return w


def apply_avoid_crowd(probs: np.ndarray, boost: float = 1.4) -> np.ndarray:
    """Olasılık vektörüne avoid-crowd boost uygular ve normalize eder.

    Args:
        probs: orijinal olasılık vektörü (total elemanlı, toplam ~1).
        boost: 32+ sayılarına uygulanan çarpan (1.0 = pasif).

    Returns:
        Normalleştirilmiş yeni olasılık vektörü.
    """
    if boost <= 1.0:
        return probs
    total = len(probs)
    w = crowd_uncrowdedness_weights(total, boost)
    boosted = probs * w
    s = boosted.sum()
    return boosted / s if s > 0 else probs


def crowd_concentration_score(picks: list[int]) -> float:
    """
    0-1 arası: kuponun ne kadar "kalabalık" sayılarına bağımlı olduğu.

    1.0 = hepsi 1-31 arasında (en kalabalık, paylaşma riski yüksek)
    0.0 = hepsi 32+ (en az kalabalık, en yüksek beklenen ödül)
    """
    if not picks:
        return 0.0
    crowd = sum(1 for p in picks if p <= 31)
    return crowd / len(picks)
