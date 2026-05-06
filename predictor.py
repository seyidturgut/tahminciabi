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
        self.mean_sum, self.std_sum = self.engine.get_sum_distribution_stats()
        
    def _is_valid_combination(self, combo):
        """
        Kombinasyonun mantıksal ve istatistiksel filtrelere uyup uymadığını kontrol eder.
        """
        combo = sorted(combo)
        
        # Kural 1: Toplam Gauss aralığında olmalı (Ortalama ± 1 Standart Sapma)
        # 6/90 için genelde ortalama 273 civarıdır.
        # Çan eğrisinin %68'lik göbeğini hedefleriz.
        combo_sum = sum(combo)
        min_sum = self.mean_sum - (self.std_sum * 1.5)
        max_sum = self.mean_sum + (self.std_sum * 1.5)
        
        if not (min_sum <= combo_sum <= max_sum):
            return False
            
        # Kural 2: 3'ten fazla ardışık sayı olmamalı (Örn: 12, 13, 14, 15 mantıksızdır)
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
            
        # Kural 3: Tek / Çift Dengesi (Sadece çok uçuk olanları ele - 6 Tek veya 6 Çift)
        odds = sum(1 for x in combo if x % 2 != 0)
        evens = self.draw_size - odds
        if odds == 0 or evens == 0:
            return False
            
        return True

    def generate_tickets(self, num_tickets=5, strategy="Dengeli", mc_iterations=10000):
        """
        Belirtilen stratejiye göre Monte Carlo simülasyonu ile kupon üretir.
        """
        valid_tickets = []
        attempts = 0
        
        # Ağırlıklandırma
        # Frekansları al
        freq_weights = self.freq_df.sort_values(by='sayi')['frekans'].values
        # 0 frekans olanlara ufak bir şans ver (Laplace smoothing)
        freq_weights = freq_weights + 1 
        
        if strategy == "Sıcak Sayılar":
            # Çok çıkanların olasılığını artır
            weights = freq_weights / freq_weights.sum()
        elif strategy == "Soğuk Sayılar":
            # Az çıkanların olasılığını artır (Ters orantı)
            inv_weights = 1 / freq_weights
            weights = inv_weights / inv_weights.sum()
        else:
            # "Dengeli": Eşit ağırlık (tamamen rastgele ama filtrelere takılan)
            weights = [1/self.total_numbers] * self.total_numbers
            
        all_numbers = np.arange(1, self.total_numbers + 1)
        
        while len(valid_tickets) < num_tickets and attempts < mc_iterations:
            attempts += 1
            
            # Seçilen ağırlıklara göre rastgele 6 sayı çek
            combo = np.random.choice(all_numbers, size=self.draw_size, replace=False, p=weights)
            combo = tuple(sorted(int(x) for x in combo))
            
            if self._is_valid_combination(combo) and combo not in valid_tickets:
                valid_tickets.append(combo)
                
        return valid_tickets, attempts
