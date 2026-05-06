"""Lototurkiye'den son N çekilişin gerçek ödüllerini çekip medyan hesaplar.

Bir kerelik araç — games.py prize_tl default'larını gerçekçi yapmak için.
"""
from __future__ import annotations

import re
import statistics
import sys
from typing import Optional

import requests

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "tr"}


def parse_tl(s: str) -> Optional[float]:
    s = s.strip()
    m = re.search(r"([\d.]+),(\d{1,2})", s)
    if m:
        return float(m.group(1).replace(".", "") + "." + m.group(2))
    m = re.search(r"([\d.]+)", s)
    if m:
        return float(m.group(1).replace(".", ""))
    return None


def fetch_html(slug: str, game_id: int, no: int) -> Optional[str]:
    url = f"https://www.lototurkiye.com/{slug}/CEKILIS-SONUCLARI/{game_id}/{no}/01/01/2020/X"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200 and len(r.text) > 5000:
            return r.text
    except Exception:
        pass
    return None


def latest_no(slug: str, game_id: int) -> Optional[int]:
    url = f"https://www.lototurkiye.com/{slug}/ANA-SAYFA/{game_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        m = re.search(r"\d{1,4}\.\s*Çekiliş\s*\[(\d{1,4})\]", r.text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def parse_sayisal(html: str) -> dict:
    """Sayısal Loto: 6, 5, 4, 3 bilen + devir."""
    out: dict = {}
    # Devir / 6 bilen — örnek: "6 Bilen 1 kişi X TL" veya "6 Bilen çıkmayınca X TL devretti"
    # 5,4,3:  "X Bilen Y kişi Z TL"
    pat = re.compile(r"(\d+)\s*Bilen\s+(\d+)\s*kişi\s+(\d{1,3}(?:\.\d{3})*,\d{2})\s*TL")
    for m in pat.finditer(html):
        k = int(m.group(1))
        if k in {3, 4, 5, 6}:
            out[k] = parse_tl(m.group(3))
    # Devir varsa 6 bilen için onu kullan
    devir = re.search(r"6\s*Bilen[^.]{0,80}?(\d{1,3}(?:\.\d{3})*,\d{2})\s*lira[^.]{0,40}?devretti", html, re.IGNORECASE)
    if devir and 6 not in out:
        out[6] = parse_tl(devir.group(1))
    return out


def parse_sans_topu(html: str) -> dict:
    """Şans Topu: (main, st) tuple keyli."""
    out: dict = {}
    pat = re.compile(r"(\d+)\+(\d+)\s*Bilen\s+(\d+)\s*kişi\s+(\d{1,3}(?:\.\d{3})*,\d{2})\s*TL")
    for m in pat.finditer(html):
        a, b = int(m.group(1)), int(m.group(2))
        out[(a, b)] = parse_tl(m.group(4))
    pat2 = re.compile(r"(?<!\+)(\d)\s*Bilen\s+(\d+)\s*kişi\s+(\d{1,3}(?:\.\d{3})*,\d{2})\s*TL")
    for m in pat2.finditer(html):
        k = int(m.group(1))
        if k in {3, 4, 5}:
            out[(k, 0)] = parse_tl(m.group(3))
    # 5+1 devir
    devir = re.search(r"5\+1[^.]{0,80}?(\d{1,3}(?:\.\d{3})*,\d{2})\s*lira[^.]{0,40}?devretti", html, re.IGNORECASE)
    if devir and (5, 1) not in out:
        out[(5, 1)] = parse_tl(devir.group(1))
    return out


def parse_on_numara(html: str) -> dict:
    """On Numara: 10, 9, 8, 7, 6, 0 (Hiç Bilmeyen)."""
    out: dict = {}
    pat = re.compile(r"(\d+)\s*Bilen\s+(\d+)\s*kişi\s+(\d{1,3}(?:\.\d{3})*,\d{2})\s*TL")
    for m in pat.finditer(html):
        k = int(m.group(1))
        if k in {6, 7, 8, 9, 10}:
            out[k] = parse_tl(m.group(3))
    pat0 = re.search(r"Hiç\s*Bilmeyen\s+(\d+)\s*kişi\s+(\d{1,3}(?:\.\d{3})*,\d{2})\s*TL", html)
    if pat0:
        out[0] = parse_tl(pat0.group(2))
    devir10 = re.search(r"10\s*Bilen[^.]{0,80}?(\d{1,3}(?:\.\d{3})*,\d{2})\s*lira[^.]{0,40}?devretti", html, re.IGNORECASE)
    if devir10 and 10 not in out:
        out[10] = parse_tl(devir10.group(1))
    return out


def aggregate(slug: str, game_id: int, parser, last_n: int = 10) -> dict:
    latest = latest_no(slug, game_id)
    if latest is None:
        print(f"  ⚠️ Son çekiliş alınamadı: {slug}", file=sys.stderr)
        return {}
    print(f"  Son çekiliş: #{latest}")
    samples: dict = {}
    fetched = 0
    for no in range(latest, latest - last_n - 5, -1):  # küçük buffer
        if fetched >= last_n:
            break
        html = fetch_html(slug, game_id, no)
        if not html:
            continue
        prizes = parser(html)
        if not prizes:
            continue
        fetched += 1
        for k, v in prizes.items():
            samples.setdefault(k, []).append(v)
        print(f"    #{no}: {len(prizes)} kademe")

    medians = {}
    for k, vals in samples.items():
        if vals:
            medians[k] = int(round(statistics.median(vals)))
    return medians


if __name__ == "__main__":
    print("🔮 Sayısal Loto:")
    print(f"  prizes_tl = {aggregate('SAYISAL-LOTO-SISAL', 9, parse_sayisal, 10)}")
    print()
    print("🎯 Şans Topu:")
    st = aggregate("SANS-TOPU-SISAL", 10, parse_sans_topu, 10)
    print(f"  prizes_tl = {st}")
    print()
    print("🔟 On Numara:")
    print(f"  prizes_tl = {aggregate('ON-NUMARA-SISAL', 11, parse_on_numara, 10)}")
