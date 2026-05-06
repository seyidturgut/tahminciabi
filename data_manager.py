"""
Çekiliş veritabanı yöneticisi — Sayısal Loto, Şans Topu, On Numara.

Her oyun kendi CSV'sinde tutulur (games.py'deki csv_path). Mock üretim yoktur —
eksik veri scraper ile çekilir, çekilemiyorsa ScrapeFailedError yukarı geçer.

Public API:
    load_data(game=None, progress_callback=None) -> pd.DataFrame
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from games import get_game
from scraper import (
    ScrapeFailedError,
    fetch_full_history,
    fetch_latest_draw_number,
    fetch_draw,
)

# Sayısal Loto default (geriye uyumluluk)
DATA_FILE = "gecmis_cekilisler.csv"
TOTAL_NUMBERS = 90
DRAW_SIZE = 6


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["tarih"] = pd.to_datetime(df["tarih"])
    if "cekilis_no" in df.columns:
        df["cekilis_no"] = df["cekilis_no"].astype(int)
        df = df.sort_values("cekilis_no").reset_index(drop=True)
    else:
        df = df.sort_values("tarih").reset_index(drop=True)
    return df


def _is_legacy_sayisal(df: pd.DataFrame) -> bool:
    """Eski mock format: cekilis_no, joker veya superstar yoksa legacy."""
    required = {"cekilis_no", "joker", "superstar"}
    return not required.issubset(set(df.columns))


def _draw_to_record(d: dict, game: dict) -> dict:
    rec = {"cekilis_no": d["cekilis_no"], "tarih": d["tarih"]}
    if game["picks"] == game["drawn"]:
        prefix, count = "sayi", game["picks"]
    else:
        prefix, count = "cekilen", game["drawn"]
    for i in range(count):
        rec[f"{prefix}_{i+1}"] = d["sayilar"][i] if i < len(d["sayilar"]) else None
    for b in game["bonuses"]:
        rec[b["key"]] = d.get(b["key"])
    return rec


def load_data(game: Optional[dict] = None, progress_callback=None) -> pd.DataFrame:
    """
    Veriyi yükler:
    - CSV varsa: incremental update (eksik son çekilişleri ekler).
    - Yoksa veya legacy formatta ise sıfırdan tüm tarihçeyi scrape eder.
    """
    g = game or get_game()
    csv_path = g["csv_path"]

    if os.path.exists(csv_path):
        df = _read_csv(csv_path)
        # Legacy mock kontrolü sadece Sayısal Loto için anlamlı
        if g["key"] == "sayisal_loto" and _is_legacy_sayisal(df):
            df = fetch_full_history(progress_callback=progress_callback, game=g)
            df.to_csv(csv_path, index=False)
            return df

        # Incremental update — kaynak siteye ulaşılamazsa mevcut CSV ile devam
        try:
            latest_remote = fetch_latest_draw_number(g)
        except ScrapeFailedError:
            return df
        latest_local = int(df["cekilis_no"].max())
        if latest_remote > latest_local:
            new_rows = []
            for n in range(latest_local + 1, latest_remote + 1):
                try:
                    d = fetch_draw(n, game=g)
                except ScrapeFailedError:
                    continue
                new_rows.append(_draw_to_record(d, g))
            if new_rows:
                df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                df = df.sort_values("cekilis_no").reset_index(drop=True)
                df["tarih"] = pd.to_datetime(df["tarih"])
                df.to_csv(csv_path, index=False)
        return df

    # CSV yok — tüm tarihçeyi çek
    df = fetch_full_history(progress_callback=progress_callback, game=g)
    df.to_csv(csv_path, index=False)
    return df


__all__ = ["load_data", "DATA_FILE", "TOTAL_NUMBERS", "DRAW_SIZE", "ScrapeFailedError"]


if __name__ == "__main__":
    for key in ["sayisal_loto", "sans_topu", "on_numara"]:
        g = get_game(key)
        try:
            df = load_data(g, progress_callback=lambda d, t: None)
            print(f"{g['emoji']} {g['name']}: {len(df)} çekiliş, son #{int(df['cekilis_no'].max())}")
            print(df.tail(2).to_string())
        except ScrapeFailedError as e:
            print(f"{g['emoji']} {g['name']}: {e}")
        print("---")
