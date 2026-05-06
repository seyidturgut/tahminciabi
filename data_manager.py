"""
Sayısal Loto gerçek çekiliş veritabanı yöneticisi.

CSV'ye yazılır. Mock üretim yoktur — eksik veri scraper ile çekilir,
çekilemiyorsa ScrapeFailedError yukarı geçer.
"""
from __future__ import annotations

import os

import pandas as pd

from scraper import (
    ScrapeFailedError,
    fetch_full_history,
    fetch_latest_draw_number,
    fetch_draw,
)

DATA_FILE = "gecmis_cekilisler.csv"
TOTAL_NUMBERS = 90
DRAW_SIZE = 6


def _read_csv(path: str = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["tarih"] = pd.to_datetime(df["tarih"])
    if "cekilis_no" in df.columns:
        df["cekilis_no"] = df["cekilis_no"].astype(int)
        df = df.sort_values("cekilis_no").reset_index(drop=True)
    else:
        df = df.sort_values("tarih").reset_index(drop=True)
    return df


def _is_legacy_csv(df: pd.DataFrame) -> bool:
    """Eski format: cekilis_no, joker veya superstar kolonu yoksa legacy."""
    required = {"cekilis_no", "joker", "superstar"}
    return not required.issubset(set(df.columns))


def load_data(progress_callback=None) -> pd.DataFrame:
    """
    Veri yükler:
    - CSV varsa ve `cekilis_no` kolonu içeriyorsa, eksik son çekilişleri ekler.
    - Yoksa veya legacy formatta ise sıfırdan tüm tarihçeyi scrape eder.
    """
    if os.path.exists(DATA_FILE):
        df = _read_csv()
        if _is_legacy_csv(df):
            # Eski mock CSV — sıfırdan inşa et
            df = fetch_full_history(progress_callback=progress_callback)
            df.to_csv(DATA_FILE, index=False)
            return df

        latest_remote = fetch_latest_draw_number()
        latest_local = int(df["cekilis_no"].max())
        if latest_remote > latest_local:
            new_rows = []
            for n in range(latest_local + 1, latest_remote + 1):
                d = fetch_draw(n)
                new_rows.append({
                    "cekilis_no": d["cekilis_no"],
                    "tarih": d["tarih"],
                    **{f"sayi_{i+1}": d["sayilar"][i] for i in range(DRAW_SIZE)},
                    "joker": d.get("joker"),
                    "superstar": d.get("superstar"),
                })
            if new_rows:
                df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                df = df.sort_values("cekilis_no").reset_index(drop=True)
                df["tarih"] = pd.to_datetime(df["tarih"])
                df.to_csv(DATA_FILE, index=False)
        return df

    # CSV yok — tüm tarihçeyi çek
    df = fetch_full_history(progress_callback=progress_callback)
    df.to_csv(DATA_FILE, index=False)
    return df


__all__ = ["load_data", "DATA_FILE", "TOTAL_NUMBERS", "DRAW_SIZE", "ScrapeFailedError"]


if __name__ == "__main__":
    df = load_data(progress_callback=lambda d, t: print(f"{d}/{t}"))
    print(f"\n{len(df)} çekiliş yüklendi.")
    print(df.tail())
