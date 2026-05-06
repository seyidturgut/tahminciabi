"""Sayısal Loto kupon üretici.

Profesör Modu: 4 modelin birleşik olasılık vektörü ile ağırlıklı sampling +
matematiksel filtreler (Gauss toplam, ardışık limit, asal denge, ondalık dağılım,
pozisyonel sınırlar, yüksek-skorlu sayı zorunluluğu).

Klasik modlar (Sıcak / Soğuk / Süper Hibrit) korunur; bunlar olasılık motorunu
kullanmaz.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from analytics.probability_engine import ModelOutput, compute as compute_probabilities
from math_engine import MathEngine

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
          53, 59, 61, 67, 71, 73, 79, 83, 89}


class Predictor:
    def __init__(self, df: pd.DataFrame, total_numbers: int = 90, draw_size: int = 6):
        self.df = df
        self.total_numbers = total_numbers
        self.draw_size = draw_size
        self.engine = MathEngine(df, total_numbers, draw_size)

        self.freq_df = self.engine.calculate_frequencies()
        self.gaps_df = self.engine.calculate_gaps()
        self.mean_sum, self.std_sum = self.engine.get_sum_distribution_stats()
        self.pos_stats = self.engine.analyze_positions()

        self._probabilities: Optional[ModelOutput] = None

    def get_probabilities(self, force_train: bool = False) -> ModelOutput:
        if self._probabilities is None or force_train:
            self._probabilities = compute_probabilities(self.df, force_train_ml=force_train)
        return self._probabilities

    # --- Filtreler -------------------------------------------------------

    def _is_valid_combination(self, combo, top20: Optional[set] = None) -> bool:
        combo = sorted(combo)

        # 1. Toplam Gauss aralığında
        s = sum(combo)
        if not (self.mean_sum - 1.5 * self.std_sum <= s <= self.mean_sum + 1.5 * self.std_sum):
            return False

        # 2. Maks 3 ardışık
        max_consec = 1
        cur = 1
        for i in range(1, len(combo)):
            if combo[i] == combo[i - 1] + 1:
                cur += 1
                max_consec = max(max_consec, cur)
            else:
                cur = 1
        if max_consec > 3:
            return False

        # 3. Tek/çift dengesi
        odds = sum(1 for x in combo if x % 2 != 0)
        if odds == 0 or odds == self.draw_size:
            return False

        # 4. Asal denge
        prime_count = sum(1 for x in combo if x in PRIMES)
        if prime_count == 0 or prime_count > 3:
            return False

        # 5. Ondalık dağılım
        decade_counts: dict[int, int] = {}
        for x in combo:
            decade_counts[x // 10] = decade_counts.get(x // 10, 0) + 1
        if any(c > 3 for c in decade_counts.values()):
            return False

        # 6. Pozisyonel sınırlar
        for i, num in enumerate(combo):
            ps = self.pos_stats[i]
            lo = max(1, ps["mean"] - 2.5 * ps["std"])
            hi = min(self.total_numbers, ps["mean"] + 2.5 * ps["std"])
            if not (lo <= num <= hi):
                return False

        # 7. Profesör Modu: en az 2 yüksek-skorlu sayı
        if top20 is not None:
            if sum(1 for x in combo if x in top20) < 2:
                return False

        return True

    # --- Stratejiler -----------------------------------------------------

    def _weights_for_strategy(self, strategy: str) -> tuple[np.ndarray, Optional[set]]:
        freq_w = self.freq_df.sort_values("sayi")["frekans"].values + 1.0

        if strategy == "Sıcak Sayılar":
            w = freq_w / freq_w.sum()
            return w, None
        if strategy == "Soğuk Sayılar":
            inv = 1.0 / freq_w
            return inv / inv.sum(), None
        if strategy == "Süper Hibrit":
            w = np.full(self.total_numbers, 1.0 / self.total_numbers)
            return w, None
        if strategy == "Profesör Modu":
            out = self.get_probabilities()
            top20 = set(int(x) for x in np.argsort(out.final)[-20:] + 1)
            return out.final.copy(), top20

        # Bilinmeyen strateji → uniform
        return np.full(self.total_numbers, 1.0 / self.total_numbers), None

    # --- Üretim ----------------------------------------------------------

    def _ticket_confidence(self, combo: tuple[int, ...]) -> float:
        """0-1 arası güven skoru: log-olasılığın uniform baseline'a oranı."""
        out = self.get_probabilities()
        log_p = sum(math.log(max(out.final[n - 1], 1e-9)) for n in combo)
        baseline = self.draw_size * math.log(1.0 / self.total_numbers)
        # Pozitif: olasılık > uniform. Sigmoid ile 0-1'e sıkıştır.
        delta = log_p - baseline
        return float(1.0 / (1.0 + math.exp(-delta)))

    def generate_tickets(
        self,
        num_tickets: int = 5,
        strategy: str = "Profesör Modu",
        mc_iterations: int = 80000,
    ) -> tuple[list[tuple[int, ...]], int, list[float]]:
        weights, top20 = self._weights_for_strategy(strategy)
        all_numbers = np.arange(1, self.total_numbers + 1)
        cold_numbers = self.gaps_df.head(15)["sayi"].values

        valid: list[tuple[int, ...]] = []
        attempts = 0
        rng = np.random.default_rng()

        while len(valid) < num_tickets and attempts < mc_iterations:
            attempts += 1
            try:
                combo = rng.choice(all_numbers, size=self.draw_size, replace=False, p=weights)
            except ValueError:
                # Ağırlık vektörü 1'e tam toplanmazsa (float precision)
                weights = weights / weights.sum()
                combo = rng.choice(all_numbers, size=self.draw_size, replace=False, p=weights)
            combo = list(int(x) for x in combo)

            if strategy == "Süper Hibrit":
                forced = int(rng.choice(cold_numbers))
                if forced not in combo:
                    combo[0] = forced

            combo_t = tuple(sorted(combo))
            if combo_t in valid:
                continue
            if not self._is_valid_combination(combo_t, top20=top20):
                continue
            valid.append(combo_t)

        confidences = [self._ticket_confidence(c) if strategy == "Profesör Modu" else 0.0
                       for c in valid]
        return valid, attempts, confidences
