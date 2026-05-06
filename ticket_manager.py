"""Kayıtlı kuponları yönetir — oyun başına filter destekli."""
import json
import os
from datetime import datetime
from typing import Optional

TICKETS_FILE = "oynanan_kuponlar.json"
DEFAULT_GAME = "sayisal_loto"


def _read_all() -> list[dict]:
    if not os.path.exists(TICKETS_FILE):
        return []
    try:
        with open(TICKETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Kuponlar okunamadı: {e}")
        return []


def _record_game(rec: dict) -> str:
    """Eski kayıtların game alanı yoktu — default sayisal_loto."""
    return rec.get("game", DEFAULT_GAME)


def load_saved_tickets(game: Optional[str] = None) -> list[dict]:
    """Kayıtlı kuponları döner. game verilirse o oyunla filtrelenir.

    Backward compat: eski kayıtlar (game alanı yok) sayisal_loto sayılır.
    """
    records = _read_all()
    if game is None:
        return records
    return [r for r in records if _record_game(r) == game]


def save_tickets(tickets_list, strategy_name, valid_from_draw_no=None,
                 game: str = DEFAULT_GAME) -> str:
    """Yeni kupon kaydı oluşturur.

    Args:
        valid_from_draw_no: bu kuponların geçerli olduğu en küçük çekiliş no.
            Otomatik takip sadece bu numaradan büyük çekilişlerde değerlendirir.
        game: oyun anahtarı (games.py içindeki key).
    """
    existing = _read_all()
    new_record = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "tarih": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "game": game,
        "strateji": strategy_name,
        "kuponlar": tickets_list,
        "valid_from_draw_no": valid_from_draw_no or 0,
    }
    existing.append(new_record)
    with open(TICKETS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=4)
    return new_record["id"]


def delete_all_tickets(game: Optional[str] = None) -> int:
    """Kayıtları siler. game verilirse sadece o oyununkileri.

    Returns: silinen kayıt sayısı.
    """
    records = _read_all()
    if game is None:
        n = len(records)
        if os.path.exists(TICKETS_FILE):
            os.remove(TICKETS_FILE)
        return n

    keep = [r for r in records if _record_game(r) != game]
    deleted = len(records) - len(keep)
    if keep:
        with open(TICKETS_FILE, "w", encoding="utf-8") as f:
            json.dump(keep, f, ensure_ascii=False, indent=4)
    elif os.path.exists(TICKETS_FILE):
        os.remove(TICKETS_FILE)
    return deleted
