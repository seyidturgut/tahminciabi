import time
import os
import json
from scraper import get_live_draw_results
from ticket_manager import load_saved_tickets

FLAG_FILE = "daemon.flag"
STATE_FILE = "daemon_state.json"

def get_last_checked_date():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            return state.get("last_date", "")
    return ""

def set_last_checked_date(date_str):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_date": date_str}, f)

def check_and_notify():
    print("Milli Piyango güncel sonucu kontrol ediliyor...")
    result = get_live_draw_results()
    current_draw_date = result["tarih"]
    drawn_numbers = result["sayilar"]
    
    last_date = get_last_checked_date()
    
    if current_draw_date != last_date or True: # Demo için her zaman çalışır
        print(f"Yeni Çekiliş Tespit Edildi! Tarih: {current_draw_date} - Sayılar: {drawn_numbers}")
        
        saved_tickets = load_saved_tickets()
        if not saved_tickets:
            print("Kayıtlı kupon bulunamadı.")
            set_last_checked_date(current_draw_date)
            return
            
        print("--- OYNADIĞIN KUPONLAR ---")
        for record in saved_tickets:
            print(f"\nStrateji: {record['strateji']} | Tarih: {record['tarih']}")
            for t_idx, ticket in enumerate(record['kuponlar']):
                match_count = len(set(ticket).intersection(set(drawn_numbers)))
                print(f"Kupon {t_idx+1}: {ticket} --> {match_count} BİLDİNİZ!")
                
        set_last_checked_date(current_draw_date)
    else:
        print("Yeni bir çekiliş bulunamadı. Bekleniyor...")

def run():
    print("Daemon başlatıldı. Komut bekleniyor...")
    while True:
        if os.path.exists(FLAG_FILE):
            print("Robot AKTİF! Kontrol döngüsü başlıyor...")
            check_and_notify()
            time.sleep(10)
            # Demo: Sürekli çalışıp terminali doldurmasın diye flag'i kaldırıyoruz
            if os.path.exists(FLAG_FILE):
                os.remove(FLAG_FILE)
                print("Kontrol yapıldı, robot beklemeye alındı (UI'dan tekrar başlatın).")
        else:
            time.sleep(2)

if __name__ == "__main__":
    run()
