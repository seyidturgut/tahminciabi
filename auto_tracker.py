"""Otomatik sonuç takibi ve kupon değerlendirmesi.

Kullanıcı bir kupon kaydettikten sonra, app her açıldığında bu modül:
1) Son çekiliş numarasını siteden alır,
2) Daha önce kontrol edilmemiş çekilişleri tek tek çeker,
3) Kayıtlı her kuponu değerlendirir (sadece o çekiliş için geçerli olanları),
4) Sonuçları otomatik takip dosyasına yazar.

Streamlit Cloud filesystem ephemeral — bu dosyalar app restart olduğunda
sıfırlanır; bu, oynanan_kuponlar.json ile aynı kabul gören bir kısıtlamadır.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from analytics.expected_value import DEFAULT_PRIZES_TL
from scraper import ScrapeFailedError, fetch_draw, fetch_latest_draw_number
from ticket_manager import load_saved_tickets

TRACKING_FILE = "otomatik_takip.json"
SETTINGS_FILE = "otomasyon_ayarlari.json"


# ---------- Settings ----------

def load_settings() -> dict:
    if not os.path.exists(SETTINGS_FILE):
        return {"enabled": False, "last_checked_draw_no": 0, "prizes": DEFAULT_PRIZES_TL}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Backfill defaults
        data.setdefault("enabled", False)
        data.setdefault("last_checked_draw_no", 0)
        data.setdefault("prizes", DEFAULT_PRIZES_TL)
        return data
    except Exception:
        return {"enabled": False, "last_checked_draw_no": 0, "prizes": DEFAULT_PRIZES_TL}


def save_settings(settings: dict) -> None:
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


# ---------- Tracking history ----------

def load_tracking() -> list:
    if not os.path.exists(TRACKING_FILE):
        return []
    try:
        with open(TRACKING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_tracking(records: list) -> None:
    with open(TRACKING_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def reset_tracking() -> None:
    if os.path.exists(TRACKING_FILE):
        os.remove(TRACKING_FILE)


# ---------- Evaluation ----------

def _evaluate_ticket(ticket, drawn: list, joker: Optional[int],
                     superstar: Optional[int], prizes: dict) -> dict:
    if isinstance(ticket, dict):
        main = list(ticket.get("main", []))
        t_joker = ticket.get("joker")
        t_ss = ticket.get("superstar")
    else:
        main = list(ticket)
        t_joker = None
        t_ss = None

    main_hits = len(set(main).intersection(drawn))
    joker_hit = t_joker is not None and joker is not None and t_joker == joker
    ss_hit = t_ss is not None and superstar is not None and t_ss == superstar

    prize = prizes.get(main_hits, 0) if main_hits >= 3 else 0
    return {
        "main": main,
        "joker": t_joker,
        "superstar": t_ss,
        "main_hits": main_hits,
        "joker_hit": joker_hit,
        "superstar_hit": ss_hit,
        "estimated_prize_tl": prize,
    }


def check_now(prizes: Optional[dict] = None, force_initial: bool = False) -> dict:
    """
    Yeni çekilişleri kontrol et, kayıtlı kuponları değerlendir, takibe ekle.

    Returns:
        {"checked": int, "new_records": int, "winnings": int, "error": str|None}
    """
    settings = load_settings()
    prizes = prizes or settings.get("prizes", DEFAULT_PRIZES_TL)

    if not settings.get("enabled", False):
        return {"checked": 0, "new_records": 0, "winnings": 0,
                "error": "Otomatik takip kapalı"}

    try:
        latest_remote = fetch_latest_draw_number()
    except ScrapeFailedError as e:
        return {"checked": 0, "new_records": 0, "winnings": 0, "error": str(e)}

    last_checked = settings.get("last_checked_draw_no", 0)

    # İlk kez açıldığında geriye dönük kontrol etme — sadece bu noktayı işaretle.
    if last_checked == 0 and not force_initial:
        settings["last_checked_draw_no"] = latest_remote
        save_settings(settings)
        return {"checked": 0, "new_records": 0, "winnings": 0, "error": None}

    if latest_remote <= last_checked:
        return {"checked": 0, "new_records": 0, "winnings": 0, "error": None}

    saved_records = load_saved_tickets()
    tracking = load_tracking()
    new_count = 0
    total_winnings = 0

    for n in range(last_checked + 1, latest_remote + 1):
        try:
            d = fetch_draw(n)
        except ScrapeFailedError:
            # Henüz publish olmayan çekilişi sessizce atla.
            continue

        drawn = list(d["sayilar"])
        joker = d.get("joker")
        ss = d.get("superstar")
        date_str = d["tarih"].strftime("%d-%m-%Y") if hasattr(d["tarih"], "strftime") else str(d["tarih"])

        ticket_evals = []
        draw_winnings = 0
        for record in saved_records:
            valid_from = record.get("valid_from_draw_no", 0)
            if n < valid_from:
                continue
            for idx, ticket in enumerate(record.get("kuponlar", [])):
                ev = _evaluate_ticket(ticket, drawn, joker, ss, prizes)
                # Sadece kazandıran kuponları kaydet (3+, joker veya superstar)
                if ev["main_hits"] >= 3 or ev["joker_hit"] or ev["superstar_hit"]:
                    ticket_evals.append({
                        "record_id": record.get("id"),
                        "record_tarih": record.get("tarih"),
                        "ticket_index": idx,
                        **ev,
                    })
                    draw_winnings += ev["estimated_prize_tl"]

        tracking.append({
            "cekilis_no": n,
            "tarih": date_str,
            "drawn": drawn,
            "joker": joker,
            "superstar": ss,
            "kazanan_kuponlar": ticket_evals,
            "toplam_tahmini_kazanc_tl": draw_winnings,
        })
        new_count += 1
        total_winnings += draw_winnings

    settings["last_checked_draw_no"] = latest_remote
    save_settings(settings)
    save_tracking(tracking)
    return {"checked": latest_remote - last_checked, "new_records": new_count,
            "winnings": total_winnings, "error": None}


def get_unread_count() -> int:
    """Son görüntülemeden bu yana eklenen yeni takip kayıtlarının sayısı."""
    settings = load_settings()
    tracking = load_tracking()
    last_seen = settings.get("last_seen_tracking_index", 0)
    return max(0, len(tracking) - last_seen)


def mark_all_seen() -> None:
    settings = load_settings()
    settings["last_seen_tracking_index"] = len(load_tracking())
    save_settings(settings)


def set_enabled(enabled: bool) -> None:
    settings = load_settings()
    settings["enabled"] = enabled
    if enabled and settings.get("last_checked_draw_no", 0) == 0:
        # İlk aktivasyonda mevcut son çekilişten sonrakilere bak — geriye dönük yapma.
        try:
            settings["last_checked_draw_no"] = fetch_latest_draw_number()
        except ScrapeFailedError:
            pass
    save_settings(settings)
