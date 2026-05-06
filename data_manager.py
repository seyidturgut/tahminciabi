import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

DATA_FILE = "gecmis_cekilisler.csv"
TOTAL_NUMBERS = 90
DRAW_SIZE = 6

def generate_mock_data(num_draws=780):
    """
    Son 5 yıl için (haftada 3 çekilişten yakl. 780 çekiliş) 
    rastgele ama gerçeğe uygun Çılgın Sayısal Loto (6/90) verisi üretir.
    """
    print(f"{num_draws} adet geçmiş çekiliş verisi simüle ediliyor...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_draws * (7/3)) # Roughly 3 times a week
    
    dates = pd.date_range(start=start_date, end=end_date, periods=num_draws).round('D')
    
    draws = []
    for i in range(num_draws):
        # 1'den 90'a kadar 6 farklı rastgele sayı seç
        numbers = sorted(np.random.choice(range(1, TOTAL_NUMBERS + 1), DRAW_SIZE, replace=False))
        draws.append(numbers)
        
    df = pd.DataFrame(draws, columns=[f'sayi_{i+1}' for i in range(DRAW_SIZE)])
    df['tarih'] = dates
    
    # Sütun sırasını düzenle
    cols = ['tarih'] + [f'sayi_{i+1}' for i in range(DRAW_SIZE)]
    df = df[cols]
    
    # CSV'ye kaydet
    df.to_csv(DATA_FILE, index=False)
    print(f"Veriler '{DATA_FILE}' dosyasına kaydedildi.")
    return df

def load_data():
    """
    CSV dosyasından verileri okur. Yoksa yeni üretir.
    """
    if not os.path.exists(DATA_FILE):
        return generate_mock_data()
    
    df = pd.read_csv(DATA_FILE)
    df['tarih'] = pd.to_datetime(df['tarih'])
    return df

if __name__ == "__main__":
    df = load_data()
    print("Örnek Veri:")
    print(df.tail())
