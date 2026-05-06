"""Telegram bildirim gönderici.

Bot token ve chat ID environment variable'lardan alınır:
    TELEGRAM_BOT_TOKEN — @BotFather'dan alınan token
    TELEGRAM_CHAT_ID   — bot ile sohbet ettiğindeki chat_id

GitHub Actions cron'undan veya manuel olarak çağrılır.
"""
from __future__ import annotations

import os
from typing import Optional

import requests


def send_telegram(
    message: str,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> bool:
    """HTML formatlı mesajı belirtilen chat'e yollar."""
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID environment variable'ları gerekli."
        )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=15,
    )
    r.raise_for_status()
    return bool(r.json().get("ok", False))


def get_my_chat_id(token: str) -> list[dict]:
    """Bot ile sohbet eden chat'leri listeler. /start atıldıktan sonra çalıştır."""
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    chats = []
    for upd in data.get("result", []):
        msg = upd.get("message") or upd.get("channel_post")
        if not msg:
            continue
        chat = msg.get("chat", {})
        if chat:
            chats.append({
                "id": chat.get("id"),
                "type": chat.get("type"),
                "title": chat.get("title") or chat.get("first_name") or "?",
                "username": chat.get("username"),
            })
    # Unique
    seen = set()
    uniq = []
    for c in chats:
        if c["id"] not in seen:
            seen.add(c["id"])
            uniq.append(c)
    return uniq


if __name__ == "__main__":
    # Manuel test
    import sys
    msg = sys.argv[1] if len(sys.argv) > 1 else "🔔 Tahminci AI test mesajı."
    send_telegram(msg)
    print("OK")
