"""Otomatik sonuç takibi ve kupon değerlendirmesi — oyun başına izole."""
from __future__ import annotations

import json
import os
from typing import Optional

from games import get_game, DEFAULT_GAME_KEY
from scraper import ScrapeFailedError, fetch_draw, fetch_latest_draw_number
from ticket_manager import load_saved_tickets


def _tracking_file(game_key: str) -> str:
    return f"otomatik_takip_{game_key}.json"


def _settings_file(game_key: str) -> str:
    return f"otomasyon_ayarlari_{game_key}.json"


# ---------- Settings ----------

def load_settings(game_key: Optional[str] = None) -> dict:
    gk = game_key or DEFAULT_GAME_KEY
    g = get_game(gk)
    path = _settings_file(gk)
    default = {
        "enabled": False,
        "last_checked_draw_no": 0,
        "prizes": g["prizes_tl"],
    }
    # Migrate legacy file (Sayısal Loto only) on first read
    legacy = "otomasyon_ayarlari.json"
    if gk == DEFAULT_GAME_KEY and not os.path.exists(path) and os.path.exists(legacy):
        try:
            os.rename(legacy, path)
        except OSError:
            pass
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in default.items():
            data.setdefault(k, v)
        # prizes tuple keys serialize → string; uniform tuple-string normalize edilmiyor,
        # JSON için string keys de OK çünkü game config'i fallback olarak kullanırız
        return data
    except Exception:
        return default


def save_settings(settings: dict, game_key: Optional[str] = None) -> None:
    gk = game_key or DEFAULT_GAME_KEY
    # tuple key'li prize tabloları JSON'a yazılırken string'e çevrilmeli
    serializable = dict(settings)
    if isinstance(serializable.get("prizes"), dict):
        serializable["prizes"] = {str(k): v for k, v in serializable["prizes"].items()}
    with open(_settings_file(gk), "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


# ---------- Tracking history ----------

def load_tracking(game_key: Optional[str] = None) -> list:
    gk = game_key or DEFAULT_GAME_KEY
    path = _tracking_file(gk)
    legacy = "otomatik_takip.json"
    if gk == DEFAULT_GAME_KEY and not os.path.exists(path) and os.path.exists(legacy):
        try:
            os.rename(legacy, path)
        except OSError:
            pass
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_tracking(records: list, game_key: Optional[str] = None) -> None:
    gk = game_key or DEFAULT_GAME_KEY
    with open(_tracking_file(gk), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def reset_tracking(game_key: Optional[str] = None) -> None:
    gk = game_key or DEFAULT_GAME_KEY
    path = _tracking_file(gk)
    if os.path.exists(path):
        os.remove(path)


# ---------- Evaluation ----------

def _calc_prize(game: dict, main_hits: int, bonus_hits: dict) -> int:
    prizes = game["prizes_tl"]
    if game["key"] == "sans_topu":
        st_hit = 1 if bonus_hits.get("sans_topu") else 0
        return int(prizes.get((main_hits, st_hit), 0))
    if game["key"] == "on_numara":
        return int(prizes.get(main_hits, 0))
    # sayisal_loto
    main = prizes.get(main_hits, 0) if main_hits >= 3 else 0
    bonus = 0
    if bonus_hits.get("joker"):
        bonus += 50
    if bonus_hits.get("superstar"):
        bonus += 100
    return int(main + bonus)


def _is_winning(game: dict, main_hits: int, bonus_hits: dict) -> bool:
    if game["key"] == "on_numara":
        return main_hits >= 6 or main_hits == 0
    if game["key"] == "sans_topu":
        st_hit = bonus_hits.get("sans_topu", False)
        # 3+ ana her zaman ödüllü; 0/1/2/3/4/5 ana + Şans Topu da ödüllü
        return main_hits >= 3 or st_hit
    # sayisal_loto: 3+ ana ya da joker/ss
    return main_hits >= 3 or bonus_hits.get("joker") or bonus_hits.get("superstar")


def _evaluate_ticket(ticket, drawn: list, draw_bonuses: dict, game: dict) -> dict:
    if isinstance(ticket, dict):
        main = list(ticket.get("main", []))
        t_joker = ticket.get("joker")
        t_ss = ticket.get("superstar")
        t_st = ticket.get("sans_topu")
    else:
        main = list(ticket)
        t_joker = t_ss = t_st = None

    main_hits = len(set(main).intersection(drawn))
    joker_hit = t_joker is not None and draw_bonuses.get("joker") is not None and t_joker == draw_bonuses["joker"]
    ss_hit = t_ss is not None and draw_bonuses.get("superstar") is not None and t_ss == draw_bonuses["superstar"]
    st_hit = t_st is not None and draw_bonuses.get("sans_topu") is not None and t_st == draw_bonuses["sans_topu"]
    bonus_hits = {"joker": joker_hit, "superstar": ss_hit, "sans_topu": st_hit}
    prize = _calc_prize(game, main_hits, bonus_hits)
    return {
        "main": main,
        "joker": t_joker,
        "superstar": t_ss,
        "sans_topu": t_st,
        "main_hits": main_hits,
        "joker_hit": joker_hit,
        "superstar_hit": ss_hit,
        "sans_topu_hit": st_hit,
        "estimated_prize_tl": prize,
    }


def check_now(game_key: Optional[str] = None, force_initial: bool = False) -> dict:
    """Yeni çekilişleri kontrol et, kayıtlı kuponları değerlendir."""
    gk = game_key or DEFAULT_GAME_KEY
    g = get_game(gk)
    settings = load_settings(gk)

    if not settings.get("enabled", False):
        return {"checked": 0, "new_records": 0, "winnings": 0,
                "error": "Otomatik takip kapalı"}

    try:
        latest_remote = fetch_latest_draw_number(g)
    except ScrapeFailedError as e:
        return {"checked": 0, "new_records": 0, "winnings": 0, "error": str(e)}

    last_checked = settings.get("last_checked_draw_no", 0)

    if last_checked == 0 and not force_initial:
        settings["last_checked_draw_no"] = latest_remote
        save_settings(settings, gk)
        return {"checked": 0, "new_records": 0, "winnings": 0, "error": None}

    if latest_remote <= last_checked:
        return {"checked": 0, "new_records": 0, "winnings": 0, "error": None}

    saved_records = load_saved_tickets(game=gk)
    tracking = load_tracking(gk)
    new_count = 0
    total_winnings = 0

    for n in range(last_checked + 1, latest_remote + 1):
        try:
            d = fetch_draw(n, game=g)
        except ScrapeFailedError:
            continue

        drawn = list(d["sayilar"])
        draw_bonuses = {b["key"]: d.get(b["key"]) for b in g["bonuses"]}
        date_str = d["tarih"].strftime("%d-%m-%Y") if hasattr(d["tarih"], "strftime") else str(d["tarih"])

        ticket_evals = []
        draw_winnings = 0
        for record in saved_records:
            valid_from = record.get("valid_from_draw_no", 0)
            if n < valid_from:
                continue
            for idx, ticket in enumerate(record.get("kuponlar", [])):
                ev = _evaluate_ticket(ticket, drawn, draw_bonuses, g)
                bonus_hits = {
                    "joker": ev["joker_hit"], "superstar": ev["superstar_hit"],
                    "sans_topu": ev["sans_topu_hit"],
                }
                if _is_winning(g, ev["main_hits"], bonus_hits):
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
            **draw_bonuses,
            "kazanan_kuponlar": ticket_evals,
            "toplam_tahmini_kazanc_tl": draw_winnings,
        })
        new_count += 1
        total_winnings += draw_winnings

    settings["last_checked_draw_no"] = latest_remote
    save_settings(settings, gk)
    save_tracking(tracking, gk)
    return {"checked": latest_remote - last_checked, "new_records": new_count,
            "winnings": total_winnings, "error": None}


def get_unread_count(game_key: Optional[str] = None) -> int:
    gk = game_key or DEFAULT_GAME_KEY
    settings = load_settings(gk)
    tracking = load_tracking(gk)
    last_seen = settings.get("last_seen_tracking_index", 0)
    return max(0, len(tracking) - last_seen)


def mark_all_seen(game_key: Optional[str] = None) -> None:
    gk = game_key or DEFAULT_GAME_KEY
    settings = load_settings(gk)
    settings["last_seen_tracking_index"] = len(load_tracking(gk))
    save_settings(settings, gk)


def set_enabled(enabled: bool, game_key: Optional[str] = None) -> None:
    gk = game_key or DEFAULT_GAME_KEY
    g = get_game(gk)
    settings = load_settings(gk)
    settings["enabled"] = enabled
    if enabled and settings.get("last_checked_draw_no", 0) == 0:
        try:
            settings["last_checked_draw_no"] = fetch_latest_draw_number(g)
        except ScrapeFailedError:
            pass
    save_settings(settings, gk)
