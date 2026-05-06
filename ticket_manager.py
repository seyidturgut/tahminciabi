import json
import os
from datetime import datetime

TICKETS_FILE = "oynanan_kuponlar.json"

def load_saved_tickets():
    """Kayıtlı kuponları JSON dosyasından okur."""
    if not os.path.exists(TICKETS_FILE):
        return []
    try:
        with open(TICKETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Kuponlar okunamadı: {e}")
        return []

def save_tickets(tickets_list, strategy_name, valid_from_draw_no=None):
    """
    Yeni kuponları JSON dosyasına ekler.

    valid_from_draw_no: bu kuponların geçerli olduğu en küçük çekiliş numarası.
    Otomatik takip sadece bu numaradan büyük çekilişlerde değerlendirir.
    """
    existing_data = load_saved_tickets()

    new_record = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "tarih": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "strateji": strategy_name,
        "kuponlar": tickets_list,
        "valid_from_draw_no": valid_from_draw_no or 0,
    }

    existing_data.append(new_record)

    with open(TICKETS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    return new_record["id"]

def delete_all_tickets():
    """Tüm kayıtlı kuponları siler."""
    if os.path.exists(TICKETS_FILE):
        os.remove(TICKETS_FILE)
