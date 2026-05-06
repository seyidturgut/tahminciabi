import pandas as pd
import numpy as np

class MathEngine:
    def __init__(self, df, total_numbers=90, draw_size=6, number_cols=None):
        """
        Args:
            df: çekiliş geçmişi DataFrame.
            total_numbers: oyun aralığı (örn. 90, 34, 80).
            draw_size: ANA sayı kolonu sayısı — frekans/gap analizine giren.
                Sayısal Loto: 6 (çekilen), Şans Topu: 5, On Numara: 22 (çekilen).
            number_cols: kolon adları override; None ise sayi_1..sayi_{draw_size}.
        """
        self.df = df
        self.total_numbers = total_numbers
        self.draw_size = draw_size
        self.number_cols = number_cols or [f'sayi_{i+1}' for i in range(draw_size)]

        # Sadece sayıların olduğu DataFrame
        self.numbers_df = self.df[self.number_cols]
        
    def calculate_frequencies(self):
        """Her sayının kaç kez çıktığını ve yüzdesini hesaplar."""
        all_numbers = self.numbers_df.values.flatten()
        counts = pd.Series(all_numbers).value_counts().sort_index()
        
        for i in range(1, self.total_numbers + 1):
            if i not in counts:
                counts[i] = 0
                
        freq_df = pd.DataFrame({'sayi': counts.index, 'frekans': counts.values})
        freq_df['yuzde'] = (freq_df['frekans'] / len(self.df)) * 100
        return freq_df.sort_values(by='frekans', ascending=False).reset_index(drop=True)
    
    def calculate_gaps(self):
        """Her sayı için 'gecikme' (gap) süresini hesaplar."""
        gaps = {}
        for i in range(1, self.total_numbers + 1):
            mask = (self.numbers_df == i).any(axis=1)
            if mask.any():
                last_seen_idx = mask[mask].index[-1]
                gap = (len(self.df) - 1) - last_seen_idx
            else:
                gap = len(self.df)
            gaps[i] = gap
            
        gaps_df = pd.DataFrame(list(gaps.items()), columns=['sayi', 'gecikme'])
        return gaps_df.sort_values(by='gecikme', ascending=False).reset_index(drop=True)
        
    def get_sum_distribution_stats(self):
        """Kombinasyonların toplamlarının ortalama ve standart sapmasını verir."""
        sums = self.numbers_df.sum(axis=1)
        mean_sum = sums.mean()
        std_sum = sums.std()
        return mean_sum, std_sum
        
    def analyze_positions(self):
        """
        Geçmiş çekilişlerdeki 1. Top, 2. Top vb. konumların 
        min, max ve ortalama (mean) değerlerini hesaplar.
        """
        stats = {}
        for i, col in enumerate(self.number_cols):
            stats[i] = {
                'min': self.numbers_df[col].min(),
                'max': self.numbers_df[col].max(),
                'mean': self.numbers_df[col].mean(),
                'std': self.numbers_df[col].std()
            }
        return stats
