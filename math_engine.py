import pandas as pd
import numpy as np

class MathEngine:
    def __init__(self, df, total_numbers=90, draw_size=6):
        self.df = df
        self.total_numbers = total_numbers
        self.draw_size = draw_size
        self.number_cols = [f'sayi_{i+1}' for i in range(draw_size)]
        
        # Sadece sayıların olduğu DataFrame
        self.numbers_df = self.df[self.number_cols]
        
    def calculate_frequencies(self):
        """
        Her sayının kaç kez çıktığını ve yüzdesini hesaplar.
        """
        all_numbers = self.numbers_df.values.flatten()
        counts = pd.Series(all_numbers).value_counts().sort_index()
        
        # Hiç çıkmayan sayıları da ekle (0 count)
        for i in range(1, self.total_numbers + 1):
            if i not in counts:
                counts[i] = 0
                
        freq_df = pd.DataFrame({'sayi': counts.index, 'frekans': counts.values})
        freq_df['yuzde'] = (freq_df['frekans'] / len(self.df)) * 100
        return freq_df.sort_values(by='frekans', ascending=False).reset_index(drop=True)
    
    def calculate_gaps(self):
        """
        Her sayı için 'gecikme' (gap) süresini hesaplar.
        Yani bir sayı son çekilişten bu yana kaç çekiliştir çıkmıyor.
        """
        gaps = {}
        for i in range(1, self.total_numbers + 1):
            # Sayının çıktığı çekilişlerin indeksleri (en son çekiliş index'i max olandır)
            mask = (self.numbers_df == i).any(axis=1)
            if mask.any():
                last_seen_idx = mask[mask].index[-1]
                # Toplam çekiliş sayısından son görülme indeksini çıkarıyoruz
                gap = (len(self.df) - 1) - last_seen_idx
            else:
                gap = len(self.df) # Hiç çıkmamışsa tüm çekilişler kadar gecikmiş
            gaps[i] = gap
            
        gaps_df = pd.DataFrame(list(gaps.items()), columns=['sayi', 'gecikme'])
        return gaps_df.sort_values(by='gecikme', ascending=False).reset_index(drop=True)
        
    def get_sum_distribution_stats(self):
        """
        Geçmişteki 6'lı kombinasyonların toplamlarının ortalama ve standart sapmasını verir.
        Bu, Gauss dağılımı (Çan eğrisi) fitresi için gereklidir.
        """
        sums = self.numbers_df.sum(axis=1)
        mean_sum = sums.mean()
        std_sum = sums.std()
        return mean_sum, std_sum
        
    def analyze_odd_even(self):
        """
        Tek / Çift oranlarının geçmişteki dağılımını hesaplar.
        """
        def count_odd(row):
            return sum(1 for x in row if x % 2 != 0)
            
        odd_counts = self.numbers_df.apply(count_odd, axis=1)
        distribution = odd_counts.value_counts().sort_index()
        
        # Format: {Tek Sayı Adedi: Yüzde}
        dist_dict = {}
        for i in range(self.draw_size + 1):
            count = distribution.get(i, 0)
            dist_dict[f"{i} Tek / {self.draw_size - i} Çift"] = (count / len(self.df)) * 100
            
        return dist_dict
