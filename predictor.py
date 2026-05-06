"""Sayısal Loto kupon üretici.

Profesör Modu: 4 modelin birleşik olasılık vektörü ile ağırlıklı sampling +
matematiksel filtreler (Gauss toplam, ardışık limit, asal denge, ondalık dağılım,
pozisyonel sınırlar, yüksek-skorlu sayı zorunluluğu).

Sistemli Oyun: top-N olasılıklı sayılardan tüm C(N,6) kombinasyonu — havuzdaki
N sayıdan 3+ doğru çıkarsa kolonlardan en az biri garantili 3+ tutar.

Klasik modlar (Sıcak / Soğuk / Süper Hibrit) korunur.
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from analytics.probability_engine import ModelOutput, compute as compute_probabilities
from math_engine import MathEngine
from games import get_game, primes_for_game, number_columns

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
          53, 59, 61, 67, 71, 73, 79, 83, 89}


class Predictor:
    def __init__(self, df: pd.DataFrame, total_numbers: int = 90, draw_size: int = 6,
                 game: Optional[dict] = None):
        """
        Args:
            df: çekiliş geçmişi DataFrame.
            total_numbers/draw_size: backward-compat parametreler.
            game: oyun konfigi (games.py). Verilirse total/picks/bonuses/primes
                game'den okunur, picks ile draw_size override edilir.
        """
        if game is None:
            game = get_game("sayisal_loto")
        self.game = game
        self.df = df
        self.total_numbers = game.get("total", total_numbers)
        # picks: kupon üretiminde her kuponun ana sayı kapasitesi
        self.picks = game.get("picks", draw_size)
        # drawn: her çekilişte küreden gelen sayı sayısı (frekans/gap için)
        self.drawn = game.get("drawn", self.picks)
        # geriye uyumluluk için draw_size = picks
        self.draw_size = self.picks
        self.bonuses = game.get("bonuses", [])
        self.primes = primes_for_game(game)

        cols = number_columns(game)
        self.engine = MathEngine(df, total_numbers=self.total_numbers,
                                 draw_size=len(cols), number_cols=cols)

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

        # 2. Ardışık limit — picks'e göre ölçeklenir
        max_consec = 1
        cur = 1
        for i in range(1, len(combo)):
            if combo[i] == combo[i - 1] + 1:
                cur += 1
                max_consec = max(max_consec, cur)
            else:
                cur = 1
        consec_limit = max(3, self.picks // 2)
        if max_consec > consec_limit:
            return False

        # 3. Tek/çift dengesi
        odds = sum(1 for x in combo if x % 2 != 0)
        if odds == 0 or odds == self.draw_size:
            return False

        # 4. Asal denge
        prime_count = sum(1 for x in combo if x in self.primes)
        max_primes = max(2, self.picks // 2)
        if prime_count == 0 or prime_count > max_primes:
            return False

        # 5. Ondalık dağılım
        decade_counts: dict[int, int] = {}
        for x in combo:
            decade_counts[x // 10] = decade_counts.get(x // 10, 0) + 1
        if any(c > 3 for c in decade_counts.values()):
            return False

        # 6. Pozisyonel sınırlar — sadece picks == drawn olduğunda anlamlı
        # (On Numara'da 22 çekilen sayı için pozisyon istatistiği var ama
        #  10 picks üretiyoruz; bu kontrolü atla)
        if self.picks == self.drawn and len(combo) <= len(self.pos_stats):
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

    def _build_smart_pool(
        self, probs: np.ndarray, pool_size: int, randomize: bool
    ) -> list[int]:
        """
        Akıllı havuz: yüksek olasılıklı sayılardan, gerçek çekiliş dağılımına
        benzer şekilde seçer (dekat dağılımı + tek/çift dengesi).

        Greedy seçim: her adımda kalan adaylardan, dağılım kısıtlarını ihlal
        etmeyen en yüksek olasılıklı sayıyı seçer. Randomize modda olasılık
        ağırlıklı sampling, deterministik modda argmax.
        """
        n_total = self.total_numbers
        candidate_k = min(n_total, max(pool_size * 3, pool_size + 15))
        candidates = list(np.argsort(probs)[-candidate_k:][::-1])  # high → low

        rng = np.random.default_rng() if randomize else None

        # Hedef dağılımlar (6 sayılık çekiliş için tipik):
        #   - dekat başına max ceil(pool_size/3)
        #   - tek/çift en az pool_size//3 her birinden
        max_per_decade = max(2, (pool_size + 2) // 3)
        min_odd = max(1, pool_size // 3)
        min_even = max(1, pool_size // 3)

        pool: list[int] = []
        decade_counts: dict[int, int] = {}
        odd_count = 0
        even_count = 0

        def violates(num: int) -> bool:
            d = num // 10
            if decade_counts.get(d, 0) >= max_per_decade:
                return True
            return False

        remaining = pool_size
        while remaining > 0 and candidates:
            slots_left = remaining
            need_odd = max(0, min_odd - odd_count) >= slots_left
            need_even = max(0, min_even - even_count) >= slots_left

            valid = []
            for idx in candidates:
                num = int(idx) + 1
                if violates(num):
                    continue
                if need_odd and num % 2 == 0:
                    continue
                if need_even and num % 2 != 0:
                    continue
                valid.append(idx)

            if not valid:
                # Kısıt çok sıkıysa gevşet
                valid = [idx for idx in candidates if not violates(int(idx) + 1)] or list(candidates)

            if randomize and len(valid) > 1:
                w = probs[valid].copy()
                w = w / w.sum()
                pick = int(rng.choice(valid, p=w))
            else:
                pick = int(valid[0])

            num = pick + 1
            pool.append(num)
            decade_counts[num // 10] = decade_counts.get(num // 10, 0) + 1
            if num % 2 == 0:
                even_count += 1
            else:
                odd_count += 1
            candidates.remove(pick)
            remaining -= 1

        return sorted(pool)

    def generate_system_tickets(
        self, pool_size: int = 7, randomize_pool: bool = False
    ) -> tuple[list[dict], list[int]]:
        """
        Sistemli oyun: top-N olasılıklı sayılardan tüm 6'lı kombinasyonları
        üretir. Filtreler uygulanmaz — sistemin matematiksel garantisini
        bozmamak için.

        Garanti: havuzdaki N sayıdan en az 3'ü çekilişte çıkarsa, üretilen
        kolonlardan EN AZ BİRİ 3+ tutar.

        Args:
            pool_size: havuz boyutu (7-10 arası mantıklı).
            randomize_pool: False ise her zaman aynı top-N (deterministik).
                True ise top-(N+8) içinden olasılık-ağırlıklı rastgele örnekleme
                yapar; her çağrıda farklı pool gelir ama yine "akıllı" sayılar.

        Returns:
            (tickets, pool) — pool: havuza alınan sayılar (sıralı)
        """
        if not 6 <= pool_size <= 20:
            raise ValueError("pool_size 6-20 arasında olmalı")

        out = self.get_probabilities()
        pool = self._build_smart_pool(out.final, pool_size, randomize_pool)

        # Joker ve Süper Star: pool dışı en yüksek 2 olasılık (tek seferlik, tüm sisteme)
        masked = out.final.copy()
        for n in pool:
            masked[n - 1] = -1
        sorted_remaining = np.argsort(masked)[::-1]
        sys_joker = int(sorted_remaining[0]) + 1
        sys_ss = int(sorted_remaining[1]) + 1

        tickets = []
        for combo in combinations(pool, 6):
            conf = self._ticket_confidence(combo)
            tickets.append({
                "main": tuple(combo),
                "joker": sys_joker,
                "superstar": sys_ss,
                "confidence": conf,
            })
        return tickets, pool

    def generate_multi_mini_system(
        self, num_systems: int = 5, pool_size_each: int = 7
    ) -> tuple[list[dict], list[list[int]]]:
        """
        Çoklu Mini-Sistem: N adet bağımsız küçük havuz üretir, her birinin
        tüm C(pool_size_each, 6) kombinasyonunu döner.

        Avantaj: tek büyük sistem (örn. Sistem 10 = 210 kolon) yerine 5 farklı
        Sistem 7 (5×7 = 35 kolon) — havuzlar bağımsız olduğu için "all-or-nothing"
        riski dağılır.

        Args:
            num_systems: bağımsız havuz sayısı.
            pool_size_each: her havuzun boyutu (6-10 mantıklı).

        Returns:
            (tickets, pools) — tickets her biri pool_index içerir; pools listesi
            her havuzun sıralı sayılarını verir.
        """
        if not 2 <= num_systems <= 10:
            raise ValueError("num_systems 2-10 arasında olmalı")
        if not 6 <= pool_size_each <= 10:
            raise ValueError("pool_size_each 6-10 arasında olmalı")

        out = self.get_probabilities()
        pools: list[list[int]] = []
        used: set[int] = set()  # havuzlar arası örtüşmeyi azalt

        for _ in range(num_systems):
            probs = out.final.copy()
            for n in used:
                probs[n - 1] *= 0.3  # daha önce kullanılanların ağırlığını düşür
            pool = self._build_smart_pool(probs, pool_size_each, randomize=True)
            pools.append(pool)
            used.update(pool)

        # Tüm pool'ların dışından joker + ss seç
        all_pool_nums = set().union(*[set(p) for p in pools])
        masked = out.final.copy()
        for n in all_pool_nums:
            masked[n - 1] = -1
        sorted_remaining = np.argsort(masked)[::-1]
        sys_joker = int(sorted_remaining[0]) + 1
        sys_ss = int(sorted_remaining[1]) + 1

        tickets = []
        for idx, pool in enumerate(pools):
            for combo in combinations(pool, 6):
                conf = self._ticket_confidence(combo)
                tickets.append({
                    "main": tuple(combo),
                    "joker": sys_joker,
                    "superstar": sys_ss,
                    "confidence": conf,
                    "pool_index": idx,
                })
        return tickets, pools

    def _pick_joker_and_superstar(
        self, main6: tuple[int, ...], weights: np.ndarray, rng: np.random.Generator
    ) -> tuple[int, int]:
        """
        Ana 6'da olmayan sayılardan ağırlıklı olarak Joker ve Süper Star seçer.
        Joker ve Süper Star resmi çekilişte ayrı toplar — bizim üretimimizde de
        ana 6 ile çakışmamasına özen gösteriyoruz.
        """
        excluded = set(main6)
        mask = np.array([(i + 1) not in excluded for i in range(self.total_numbers)])
        avail_w = weights * mask
        if avail_w.sum() <= 0:
            avail_w = mask.astype(float)
        avail_w = avail_w / avail_w.sum()
        joker = int(rng.choice(np.arange(1, self.total_numbers + 1), p=avail_w))

        excluded.add(joker)
        mask = np.array([(i + 1) not in excluded for i in range(self.total_numbers)])
        avail_w = weights * mask
        if avail_w.sum() <= 0:
            avail_w = mask.astype(float)
        avail_w = avail_w / avail_w.sum()
        superstar = int(rng.choice(np.arange(1, self.total_numbers + 1), p=avail_w))

        return joker, superstar

    def generate_tickets(
        self,
        num_tickets: int = 5,
        strategy: str = "Profesör Modu",
        mc_iterations: int = 80000,
    ) -> tuple[list[dict], int]:
        """
        Her kupon için 6 ana sayı + Joker + Süper Star + güven skoru üretir.

        Returns:
            (tickets, attempts) — tickets her biri:
            {"main": (n,n,n,n,n,n), "joker": int, "superstar": int, "confidence": float}
        """
        weights, top20 = self._weights_for_strategy(strategy)
        all_numbers = np.arange(1, self.total_numbers + 1)
        cold_numbers = self.gaps_df.head(15)["sayi"].values

        valid: list[dict] = []
        seen_main: set[tuple[int, ...]] = set()
        attempts = 0
        rng = np.random.default_rng()

        while len(valid) < num_tickets and attempts < mc_iterations:
            attempts += 1
            try:
                combo = rng.choice(all_numbers, size=self.draw_size, replace=False, p=weights)
            except ValueError:
                weights = weights / weights.sum()
                combo = rng.choice(all_numbers, size=self.draw_size, replace=False, p=weights)
            combo = list(int(x) for x in combo)

            if strategy == "Süper Hibrit":
                forced = int(rng.choice(cold_numbers))
                if forced not in combo:
                    combo[0] = forced

            combo_t = tuple(sorted(combo))
            if combo_t in seen_main:
                continue
            if not self._is_valid_combination(combo_t, top20=top20):
                continue

            joker, superstar = self._pick_joker_and_superstar(combo_t, weights, rng)
            conf = self._ticket_confidence(combo_t) if strategy == "Profesör Modu" else 0.0

            valid.append({
                "main": combo_t,
                "joker": joker,
                "superstar": superstar,
                "confidence": conf,
            })
            seen_main.add(combo_t)

        return valid, attempts
