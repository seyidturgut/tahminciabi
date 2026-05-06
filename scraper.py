"""
Çekiliş verisi kazıyıcı — Sayısal Loto, Şans Topu, On Numara için.

Kaynak: lototurkiye.com (statik HTML, b{N}.gif top imajları).
Milli Piyango Online (MPO) Cloudflare WAF nedeniyle programatik erişime kapalıdır.

Public API (game parametresi opsiyonel — default Sayısal Loto):
    fetch_full_history(start=1, end=None, game=None) -> pd.DataFrame
    fetch_latest_draw(game=None) -> dict
    fetch_latest_draw_number(game=None) -> int
    fetch_draw(no, game=None) -> dict

Hata durumunda ScrapeFailedError yükselir — fallback yoktur.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import requests

from games import get_game

BASE = "https://www.lototurkiye.com"
# {slug}/CEKILIS-SONUCLARI/{id}/{no}/01/01/2020/X
DRAW_URL_TPL = BASE + "/{slug}/CEKILIS-SONUCLARI/{id}/{no}/01/01/2020/X"
HOME_URL_TPL = BASE + "/{slug}/ANA-SAYFA/{id}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
}

MONTHS = {
    "OCAK": 1, "SUBAT": 2, "ŞUBAT": 2, "MART": 3,
    "NISAN": 4, "NİSAN": 4, "MAYIS": 5,
    "HAZIRAN": 6, "HAZİRAN": 6, "TEMMUZ": 7,
    "AGUSTOS": 8, "AĞUSTOS": 8, "EYLUL": 9, "EYLÜL": 9,
    "EKIM": 10, "EKİM": 10, "KASIM": 11, "ARALIK": 12,
}
DATE_RE = re.compile(
    r"(\d{1,2})\s+(OCAK|SUBAT|ŞUBAT|MART|NISAN|NİSAN|MAYIS|HAZIRAN|HAZİRAN|"
    r"TEMMUZ|AGUSTOS|AĞUSTOS|EYLUL|EYLÜL|EKIM|EKİM|KASIM|ARALIK)\s+(\d{4})",
    re.IGNORECASE,
)
DRAW_NO_RE = re.compile(r"\d{1,4}\.\s*Çekiliş\s*\[(\d{1,4})\]", re.IGNORECASE)


class ScrapeFailedError(RuntimeError):
    pass


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _request_with_retry(session: requests.Session, url: str, retries: int = 3) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=25)
            if r.status_code == 200 and len(r.text) > 5000:
                return r.text
            last_err = RuntimeError(f"HTTP {r.status_code}, len={len(r.text)}")
        except requests.RequestException as e:
            last_err = e
        time.sleep(1.5 ** attempt)
    raise ScrapeFailedError(f"GET {url} başarısız: {last_err}")


def _draw_url(game: dict, no: int) -> str:
    return DRAW_URL_TPL.format(slug=game["scrape_slug"], id=game["scrape_id"], no=no)


def _home_url(game: dict) -> str:
    return HOME_URL_TPL.format(slug=game["scrape_slug"], id=game["scrape_id"])


def _parse_draw_html(html: str, draw_no: int, game: dict) -> Optional[dict]:
    """HTML'den oyuna göre (tarih, ana sayılar, bonuslar) çıkarır.

    Sayısal Loto / Şans Topu: ana toplar bplus.gif separator'undan ÖNCE,
    bonuslar SONRA. (Sayısal Loto: 6 ana + joker + ss; Şans Topu: 5 ana + ŞT.)

    On Numara: bplus yok, 22 ana top.
    """
    expected_main = game["picks"] if game["picks"] == game["drawn"] else game["drawn"]
    bonus_specs = game["bonuses"]
    total = game["total"]

    # Tarih önce — kayıp ise direkt None
    dm = DATE_RE.search(html)
    if not dm:
        return None
    day = int(dm.group(1))
    month = MONTHS[dm.group(2).upper()]
    year = int(dm.group(3))
    if not (2000 <= year <= 2100):
        return None
    tarih = pd.Timestamp(year=year, month=month, day=day)

    if bonus_specs:
        # bplus.gif separator'ı kullan
        parts = html.split("/bplus.gif")
        if len(parts) < 2:
            return None
        before = parts[0]
        after = parts[1]

        main_balls = re.findall(r"/b(\d+)\.gif", before)
        if len(main_balls) < expected_main:
            return None
        main_nums = sorted({int(x) for x in main_balls[-expected_main:]})
        if len(main_nums) != expected_main or not all(1 <= n <= total for n in main_nums):
            return None

        # bplus sonrası: bonus topları sırasıyla
        next_section = after.split("/bplus.gif")[0]
        bonus_balls = re.findall(r"/b(\d+)\.gif", next_section)
        bonus_values: dict[str, Optional[int]] = {}
        for i, spec in enumerate(bonus_specs):
            if i < len(bonus_balls):
                v = int(bonus_balls[i])
                if 1 <= v <= spec["total"]:
                    bonus_values[spec["key"]] = v
                else:
                    bonus_values[spec["key"]] = None
            else:
                bonus_values[spec["key"]] = None
    else:
        # On Numara: bplus yok, 22 ana
        all_balls = re.findall(r"/b(\d+)\.gif", html)
        # ana sayfa bilgilerinde top resmleri kullanılmıyorsa son N tanesi
        valid = [int(x) for x in all_balls if 1 <= int(x) <= total]
        if len(valid) < expected_main:
            return None
        main_nums = sorted(set(valid[-expected_main:]))
        if len(main_nums) != expected_main:
            return None
        bonus_values = {}

    return {
        "cekilis_no": draw_no,
        "tarih": tarih,
        "sayilar": main_nums,
        **bonus_values,
    }


def fetch_latest_draw_number(game: Optional[dict] = None) -> int:
    """Ana sayfadan en son çekiliş numarasını alır."""
    g = game or get_game()
    s = _make_session()
    html = _request_with_retry(s, _home_url(g))
    m = DRAW_NO_RE.search(html)
    if not m:
        raise ScrapeFailedError(f"{g['name']}: ana sayfada çekiliş numarası bulunamadı.")
    return int(m.group(1))


def fetch_draw(no: int, session: Optional[requests.Session] = None,
               game: Optional[dict] = None) -> dict:
    """Tek bir çekilişi indirir."""
    g = game or get_game()
    s = session or _make_session()
    html = _request_with_retry(s, _draw_url(g, no))
    parsed = _parse_draw_html(html, no, g)
    if parsed is None:
        raise ScrapeFailedError(f"{g['name']} #{no} parse edilemedi.")
    return parsed


def fetch_latest_draw(game: Optional[dict] = None) -> dict:
    g = game or get_game()
    no = fetch_latest_draw_number(g)
    return fetch_draw(no, game=g)


def fetch_full_history(
    start: int = 1,
    end: Optional[int] = None,
    workers: int = 10,
    progress_callback=None,
    game: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Belirtilen aralıktaki tüm çekilişleri concurrent indirir.

    DataFrame kolonları oyun konfigine göre:
        Sayısal Loto: cekilis_no, tarih, sayi_1..sayi_6, joker, superstar
        Şans Topu:    cekilis_no, tarih, sayi_1..sayi_5, sans_topu
        On Numara:    cekilis_no, tarih, cekilen_1..cekilen_22
    """
    g = game or get_game()
    if end is None:
        end = fetch_latest_draw_number(g)
    if end < start:
        raise ScrapeFailedError(f"Geçersiz aralık: start={start} end={end}")

    session = _make_session()
    rows: list[dict] = []
    failed: list[int] = []
    total = end - start + 1
    done = 0

    def task(n: int):
        try:
            return fetch_draw(n, session=session, game=g)
        except ScrapeFailedError:
            return n

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(task, n): n for n in range(start, end + 1)}
        for f in as_completed(futures):
            res = f.result()
            done += 1
            if isinstance(res, dict):
                rows.append(res)
            else:
                failed.append(res)
            if progress_callback and done % 10 == 0:
                progress_callback(done, total)

    if progress_callback:
        progress_callback(done, total)

    if len(rows) < total * 0.95:
        raise ScrapeFailedError(
            f"{g['name']}: çok fazla başarısız çekiliş: {len(failed)}/{total}. "
            f"İlk hatalar: {failed[:5]}"
        )

    rows.sort(key=lambda r: r["cekilis_no"])

    # Kolon prefix'i: picks==drawn ise sayi_, değilse cekilen_
    if g["picks"] == g["drawn"]:
        num_prefix = "sayi"
        num_count = g["picks"]
    else:
        num_prefix = "cekilen"
        num_count = g["drawn"]

    bonus_keys = [b["key"] for b in g["bonuses"]]

    records = []
    for r in rows:
        rec = {"cekilis_no": r["cekilis_no"], "tarih": r["tarih"]}
        for i in range(num_count):
            rec[f"{num_prefix}_{i+1}"] = r["sayilar"][i] if i < len(r["sayilar"]) else None
        for k in bonus_keys:
            rec[k] = r.get(k)
        records.append(rec)
    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    for key in ["sayisal_loto", "sans_topu", "on_numara"]:
        g = get_game(key)
        try:
            no = fetch_latest_draw_number(g)
            d = fetch_latest_draw(g)
            print(f"{g['emoji']} {g['name']} #{no}: {d}")
        except ScrapeFailedError as e:
            print(f"{g['emoji']} {g['name']}: {e}")
