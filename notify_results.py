"""GitHub Actions cron'da çalışır:

1. Son Sayısal Loto çekilişini lototurkiye'den çeker
2. oynanan_kuponlar.json'daki kuponları (varsa) değerlendirir
3. Telegram'a sonuç + tahmini kazanç mesajı yollar

Çevre değişkenleri:
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
"""
from __future__ import annotations

import json
import os
import sys

# DEFAULT_PRIZES_TL'i import etmek için sys.path düzenle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics.expected_value import DEFAULT_PRIZES_TL
from notifier import send_telegram
from scraper import fetch_latest_draw, ScrapeFailedError

TICKETS_FILE = "oynanan_kuponlar.json"


def fmt_tl(v: float) -> str:
    return f"{v:,.0f} TL".replace(",", ".")


def evaluate_tickets(records, draw_no, drawn, joker, superstar, prizes):
    wins = []
    for record in records:
        valid_from = record.get("valid_from_draw_no", 0)
        if draw_no < valid_from:
            continue
        for ticket in record.get("kuponlar", []):
            if isinstance(ticket, dict):
                main = list(ticket.get("main", []))
                t_joker = ticket.get("joker")
                t_ss = ticket.get("superstar")
            else:
                main = list(ticket)
                t_joker = None
                t_ss = None
            hits = len(set(main).intersection(drawn))
            joker_hit = t_joker is not None and joker is not None and t_joker == joker
            ss_hit = t_ss is not None and superstar is not None and t_ss == superstar
            if hits >= 3 or joker_hit or ss_hit:
                prize = prizes.get(hits, 0) if hits >= 3 else 0
                wins.append({
                    "main": main, "joker": t_joker, "superstar": t_ss,
                    "hits": hits, "joker_hit": joker_hit, "ss_hit": ss_hit,
                    "prize": prize,
                })
    return wins


def build_message(draw, wins, has_tickets: bool) -> str:
    no = draw["cekilis_no"]
    date = draw["tarih"].strftime("%d-%m-%Y") if hasattr(draw["tarih"], "strftime") else str(draw["tarih"])
    drawn = draw["sayilar"]
    joker = draw.get("joker")
    ss = draw.get("superstar")

    lines = [
        f"🎲 <b>Sayısal Loto Çekiliş #{no}</b>",
        f"📅 {date}",
        "",
        f"<b>🟢 Çekilen Sayılar:</b> {' · '.join(f'{n:02d}' for n in drawn)}",
    ]
    if joker:
        lines.append(f"<b>🟠 Joker:</b> {joker:02d}")
    if ss:
        lines.append(f"<b>🟣 Süper Star:</b> {ss:02d}")

    if not has_tickets:
        lines.append("")
        lines.append("📝 Kayıtlı kuponun yok. Kupon kaydetmek için Tahminci uygulamasına git.")
        return "\n".join(lines)

    if not wins:
        lines.append("")
        lines.append("😔 Kayıtlı kuponlardan hiçbiri tutmadı (3+ ana, joker veya süper star).")
        return "\n".join(lines)

    total = sum(w["prize"] for w in wins)
    lines.append("")
    lines.append(f"🎉 <b>{len(wins)} TUTAN KUPON!</b>")
    if total > 0:
        lines.append(f"💰 Tahmini toplam kazanç: <b>~{fmt_tl(total)}</b>")
    lines.append("")
    for w in wins[:20]:  # Telegram mesaj uzunluk limitine takılmamak için ilk 20
        main_str = " ".join(f"{n:02d}" for n in w["main"])
        bits = [main_str]
        if w["joker"]:
            bits.append(f"J:{w['joker']:02d}")
        if w["superstar"]:
            bits.append(f"S:{w['superstar']:02d}")
        line = "• " + " · ".join(bits) + " → "
        badges = []
        if w["hits"] >= 3:
            badges.append(f"<b>{w['hits']} ANA</b>")
        if w["joker_hit"]:
            badges.append("+JOKER")
        if w["ss_hit"]:
            badges.append("+SS")
        line += " ".join(badges)
        if w["prize"]:
            line += f" (~{fmt_tl(w['prize'])})"
        lines.append(line)
    if len(wins) > 20:
        lines.append(f"… ve {len(wins) - 20} kupon daha")
    return "\n".join(lines)


def main():
    try:
        draw = fetch_latest_draw()
    except ScrapeFailedError as e:
        send_telegram(f"❌ <b>Tahminci hata:</b> Sonuç çekilemedi.\n<pre>{e}</pre>")
        sys.exit(1)

    records = []
    has_tickets = False
    if os.path.exists(TICKETS_FILE):
        try:
            with open(TICKETS_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
                has_tickets = bool(records)
        except Exception:
            records = []

    wins = evaluate_tickets(
        records,
        draw["cekilis_no"],
        list(draw["sayilar"]),
        draw.get("joker"),
        draw.get("superstar"),
        DEFAULT_PRIZES_TL,
    )

    msg = build_message(draw, wins, has_tickets)
    send_telegram(msg)
    print("Sent:", len(msg), "chars")


if __name__ == "__main__":
    main()
