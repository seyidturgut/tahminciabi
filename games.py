"""Tahminci AI — Oyun konfigürasyonları.

Her oyun için: pool aralığı, picks/drawn, bonus toplar, scrape parametreleri,
çekiliş günleri, kupon ücreti ve ödül tablosu (kullanıcı override edebilir).

Bonus topu yapısı:
    {"key": str, "label": str, "total": int}

prizes_tl yapısı oyuna göre değişir:
    Sayısal Loto: {match_count: tl}                         (3, 4, 5, 6 anahtarları)
    Şans Topu:    {(main, bonus): tl}                       ((5,1), (5,0), (4,1), ...)
    On Numara:    {match_count: tl}                         (10, 9, 8, 7, 6, 0 anahtarları)
"""
from __future__ import annotations

from typing import Optional


GAMES: dict[str, dict] = {
    "sayisal_loto": {
        "key": "sayisal_loto",
        "name": "Sayısal Loto",
        "emoji": "🔮",
        "total": 90,
        "picks": 6,
        "drawn": 6,
        "bonuses": [
            {"key": "joker", "label": "Joker", "total": 90},
            {"key": "superstar", "label": "Süper Star", "total": 90},
        ],
        "scrape_id": 9,
        "scrape_slug": "SAYISAL-LOTO-SISAL",
        "csv_path": "gecmis_sayisal.csv",
        "csv_legacy_path": "gecmis_cekilisler.csv",  # mevcut dosya adı
        "draw_days": [0, 2, 5],   # Pzt, Çar, Cmt
        "ticket_cost_tl": 25,
        "prizes_tl": {6: 100_000_000, 5: 150_000, 4: 800, 3: 30},
        "joker_bonus_tl": 50,
        "ss_bonus_tl": 100,
        "strategies": ["multi_mini", "system", "professor", "super_hybrid",
                       "hot", "cold"],
    },
    "sans_topu": {
        "key": "sans_topu",
        "name": "Şans Topu",
        "emoji": "🎯",
        "total": 34,
        "picks": 5,
        "drawn": 5,
        "bonuses": [
            {"key": "sans_topu", "label": "Şans Topu", "total": 14},
        ],
        "scrape_id": 10,
        "scrape_slug": "SANS-TOPU-SISAL",
        "csv_path": "gecmis_sans_topu.csv",
        "draw_days": [2, 6],  # Çar, Paz
        "ticket_cost_tl": 7,  # tahmini, kullanıcı override edebilir
        # (main_hits, bonus_hits) → TL
        "prizes_tl": {
            (5, 1): 5_000_000, (5, 0): 50_000,
            (4, 1): 5_000, (4, 0): 200,
            (3, 1): 100, (3, 0): 25,
            (2, 1): 15, (1, 1): 5,
        },
        "strategies": ["multi_mini", "system", "professor", "super_hybrid",
                       "hot", "cold"],
    },
    "on_numara": {
        "key": "on_numara",
        "name": "On Numara",
        "emoji": "🔟",
        "total": 80,
        "picks": 10,
        "drawn": 22,
        "bonuses": [],
        "scrape_id": 11,
        "scrape_slug": "ON-NUMARA-SISAL",
        "csv_path": "gecmis_on_numara.csv",
        "draw_days": [0, 4],  # Pzt, Cum
        "ticket_cost_tl": 4,  # tahmini
        # match_count → TL (0 doğru ödüllü)
        "prizes_tl": {10: 1_000_000, 9: 30_000, 8: 2_000, 7: 200, 6: 30, 0: 50},
        "strategies": ["professor", "super_hybrid", "hot", "cold"],  # sistem yok
    },
}


DEFAULT_GAME_KEY = "sayisal_loto"


def get_game(key: Optional[str] = None) -> dict:
    """Oyun config sözlüğünü döner. None ise default."""
    return GAMES[key or DEFAULT_GAME_KEY]


def csv_columns(game: dict) -> list[str]:
    """Oyunun CSV başlıkları."""
    cols = ["cekilis_no", "tarih"]
    if game["picks"] == game["drawn"]:
        # Sayısal Loto, Şans Topu: oyuncu picks = çekilen
        cols += [f"sayi_{i+1}" for i in range(game["picks"])]
    else:
        # On Numara: 22 çekilen (drawn) sayı saklanır
        cols += [f"cekilen_{i+1}" for i in range(game["drawn"])]
    for bonus in game["bonuses"]:
        cols.append(bonus["key"])
    return cols


def number_columns(game: dict) -> list[str]:
    """Frekans/gap analizi için kullanılan ana sayı kolonları."""
    if game["picks"] == game["drawn"]:
        return [f"sayi_{i+1}" for i in range(game["picks"])]
    return [f"cekilen_{i+1}" for i in range(game["drawn"])]


PRIMES_FULL = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
    53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
}


def primes_for_game(game: dict) -> set[int]:
    """Oyun aralığındaki asal sayılar."""
    return {p for p in PRIMES_FULL if p <= game["total"]}
