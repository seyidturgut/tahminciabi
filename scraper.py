"""
Sayısal Loto (6/90) gerçek çekiliş verisi kazıyıcı.

Birincil kaynak: lototurkiye.com (statik HTML, b{N}.gif top imajları).
Milli Piyango Online (MPO) Cloudflare WAF nedeniyle programatik erişime kapalıdır.

Tek public API:
    fetch_full_history(start=1, end=None) -> pd.DataFrame
    fetch_latest_draw() -> dict
    fetch_latest_draw_number() -> int

Hata durumunda ScrapeFailedError yükselir — fallback yoktur.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.lototurkiye.com"
DRAW_URL = BASE + "/SAYISAL-LOTO-SISAL/CEKILIS-SONUCLARI/9/{no}/01/01/2020/X"
HOME_URL = BASE + "/SAYISAL-LOTO-SISAL/ANA-SAYFA/9"

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


def _parse_draw_html(html: str, draw_no: int) -> Optional[dict]:
    """HTML'den (tarih, 6 ana sayı) çıkarır. Bulamazsa None."""
    before_plus = html.split("/bplus.gif")[0]
    balls = re.findall(r"/b(\d+)\.gif", before_plus)
    if len(balls) < 6:
        return None
    main6 = sorted({int(x) for x in balls[:6]})
    if len(main6) != 6 or not all(1 <= n <= 90 for n in main6):
        return None
    dm = DATE_RE.search(html)
    if not dm:
        return None
    day = int(dm.group(1))
    month = MONTHS[dm.group(2).upper()]
    year = int(dm.group(3))
    if not (2000 <= year <= 2100):
        return None
    return {
        "cekilis_no": draw_no,
        "tarih": pd.Timestamp(year=year, month=month, day=day),
        "sayilar": main6,
    }


def fetch_latest_draw_number() -> int:
    """Ana sayfadan en son çekiliş numarasını alır."""
    s = _make_session()
    html = _request_with_retry(s, HOME_URL)
    m = DRAW_NO_RE.search(html)
    if not m:
        raise ScrapeFailedError("Ana sayfada çekiliş numarası bulunamadı.")
    return int(m.group(1))


def fetch_draw(no: int, session: Optional[requests.Session] = None) -> dict:
    """Tek bir çekilişi indirir."""
    s = session or _make_session()
    html = _request_with_retry(s, DRAW_URL.format(no=no))
    parsed = _parse_draw_html(html, no)
    if parsed is None:
        raise ScrapeFailedError(f"Çekiliş #{no} parse edilemedi.")
    return parsed


def fetch_latest_draw() -> dict:
    no = fetch_latest_draw_number()
    return fetch_draw(no)


def fetch_full_history(
    start: int = 1,
    end: Optional[int] = None,
    workers: int = 10,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Belirtilen aralıktaki tüm çekilişleri concurrent olarak indirir.

    DataFrame kolonları: cekilis_no, tarih, sayi_1..sayi_6
    """
    if end is None:
        end = fetch_latest_draw_number()
    if end < start:
        raise ScrapeFailedError(f"Geçersiz aralık: start={start} end={end}")

    session = _make_session()
    rows: list[dict] = []
    failed: list[int] = []
    total = end - start + 1
    done = 0

    def task(n: int):
        try:
            return fetch_draw(n, session=session)
        except ScrapeFailedError:
            return n  # int = failed marker

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
            f"Çok fazla başarısız çekiliş: {len(failed)}/{total}. "
            f"İlk hatalar: {failed[:5]}"
        )

    rows.sort(key=lambda r: r["cekilis_no"])
    df = pd.DataFrame([
        {
            "cekilis_no": r["cekilis_no"],
            "tarih": r["tarih"],
            **{f"sayi_{i+1}": r["sayilar"][i] for i in range(6)},
        }
        for r in rows
    ])
    return df


if __name__ == "__main__":
    print("Son çekiliş numarası:", fetch_latest_draw_number())
    latest = fetch_latest_draw()
    print(f"Son çekiliş: {latest}")
