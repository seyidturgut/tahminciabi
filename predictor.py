import numpy as np
import random
from math_engine import MathEngine

class Predictor:
    def __init__(self, df, total_numbers=90, draw_size=6):
        self.engine = MathEngine(df, total_numbers, draw_size)
        self.total_numbers = total_numbers
        self.draw_size = draw_size
        
        # Ön hesaplamaları yap
        self.freq_df = self.engine.calculate_frequencies()
        self.gaps_df = self.engine.calculate_gaps()
        self.mean_sum, self.std_sum = self.engine.get_sum_distribution_stats()
        self.pos_stats = self.engine.analyze_positions()
        
        # Asal sayılar kümesi (1-90 arası)
        self.primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89}
        
    def _is_valid_combination(self, combo):
        """
        Kombinasyonun Derin İstatistiksel filtrelere (Super Analysis) uyup uymadığını kontrol eder.
        """
        combo = sorted(combo)
        
        # KURAL 1: Toplam Gauss aralığında olmalı (Ortalama ± 1.5 Standart Sapma)
        combo_sum = sum(combo)
        min_sum = self.mean_sum - (self.std_sum * 1.5)
        max_sum = self.mean_sum + (self.std_sum * 1.5)
        if not (min_sum <= combo_sum <= max_sum):
            return False
            
        # KURAL 2: 3'ten fazla ardışık sayı olmamalı
        consecutive_count = 1
        max_consecutive = 1
        for i in range(1, len(combo)):
            if combo[i] == combo[i-1] + 1:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 1
        if max_consecutive > 3:
            return False
            
        # KURAL 3: Tek / Çift Dengesi (Çok uçuk olanları ele: 6 Tek veya 6 Çift)
        odds = sum(1 for x in combo if x % 2 != 0)
        evens = self.draw_size - odds
        if odds == 0 or evens == 0:
            return False
            
        # KURAL 4 (SÜPER ANALİZ): Asal Sayı Dengesi (1 ile 3 arası asal sayı olmalı)
        prime_count = sum(1 for x in combo if x in self.primes)
        if prime_count == 0 or prime_count > 3:
            return False
            
        # KURAL 5 (SÜPER ANALİZ): Ondalık Dağılımı (Bir 10'luk dilimden max 3 sayı çıkabilir)
        decades = [x // 10 for x in combo]
        decade_counts = {x: decades.count(x) for x in set(decades)}
        if any(count > 3 for count in decade_counts.values()):
            return False
            
        # KURAL 6 (SÜPER ANALİZ): Pozisyonel Sınırlar
        # Her bir top, tarihsel olarak çıktığı Min ve Max değerlerin dışına (veya çok sapmaya) çıkmamalı.
        # Esneklik payı bırakarak (Std dev'in 2 katı) aşırı uçları törpülüyoruz.
        for i, num in enumerate(combo):
            stats = self.pos_stats[i]
            pos_min = max(1, stats['mean'] - (stats['std'] * 2.5))
            pos_max = min(90, stats['mean'] + (stats['std'] * 2.5))
            
            if not (pos_min <= num <= pos_max):
                return False
                
        return True

    def generate_tickets(self, num_tickets=5, strategy="Süper Hibrit", mc_iterations=50000):
        """
        Süper Analiz filtreleriyle Monte Carlo simülasyonu kullanarak kupon üretir.
        """
        valid_tickets = []
        attempts = 0
        
        # Frekans ağırlıkları
        freq_weights = self.freq_df.sort_values(by='sayi')['frekans'].values + 1
        
        # Süper Hibrit Strateji için: En sıcak ve en soğuk sayıları bul
        hot_numbers = self.freq_df.head(15)['sayi'].values
        cold_numbers = self.gaps_df.head(15)['sayi'].values
        
        if strategy == "Sıcak Sayılar":
            weights = freq_weights / freq_weights.sum()
        elif strategy == "Soğuk Sayılar":
            inv_weights = 1 / freq_weights
            weights = inv_weights / inv_weights.sum()
        else:
            # Süper Hibrit veya Dengeli
            weights = [1/self.total_numbers] * self.total_numbers
            
        all_numbers = np.arange(1, self.total_numbers + 1)
        
        while len(valid_tickets) < num_tickets and attempts < mc_iterations:
            attempts += 1
            
            # Rastgele 6 sayı çek
            combo = np.random.choice(all_numbers, size=self.draw_size, replace=False, p=weights)
            combo = list(int(x) for x in combo)
            
            # Eğer Süper Hibrit Stratejisi ise, kolonun içine mecburen 1 soğuk sayı enjekte et
            if strategy == "Dengeli" or strategy == "Süper Hibrit":
                # Bir tane garanti soğuk sayı ekle (çok uzun süredir çıkmayan)
                forced_cold = int(np.random.choice(cold_numbers))
                if forced_cold not in combo:
                    combo[0] = forced_cold # Rastgele bir sayıyı değiştir
                    
            combo = tuple(sorted(combo))
            
            # Çoklu ve derin filtreden geçir
            if self._is_valid_combination(combo) and combo not in valid_tickets:
                valid_tickets.append(combo)
                
        return valid_tickets, attempts
