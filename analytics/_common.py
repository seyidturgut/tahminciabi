"""Analytics modülleri için ortak yardımcılar.

DataFrame'in kolonlarından oyun parametrelerini auto-detect eder, böylece
her modül `game` parametresi alma zorunda kalmaz.
"""
from __future__ import annotations

import pandas as pd


def detect_params(df: pd.DataFrame) -> tuple[list[str], int]:
    """
    DataFrame kolonlarından (cols, total) tahmin eder.

    Returns:
        cols: ana sayı kolonları (sayi_* veya cekilen_*)
        total: oyun aralığı (90, 34, 80)
    """
    sayi_cols = sorted([c for c in df.columns if c.startswith("sayi_")],
                       key=lambda c: int(c.split("_")[1]))
    cekilen_cols = sorted([c for c in df.columns if c.startswith("cekilen_")],
                          key=lambda c: int(c.split("_")[1]))
    cols = sayi_cols or cekilen_cols
    if not cols:
        # geriye uyum: hiçbiri yoksa eski Sayısal Loto varsay
        cols = [f"sayi_{i+1}" for i in range(6)]

    if len(df) == 0:
        # default Sayısal Loto
        return cols, 90

    max_val = int(df[cols].max().max())
    # Standart aralıklar — yukarı yuvarla
    if max_val <= 34:
        total = 34
    elif max_val <= 80:
        total = 80
    else:
        total = 90
    return cols, total
