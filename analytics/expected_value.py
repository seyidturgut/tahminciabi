"""Sayısal Loto bütçe ve beklenen kazanç hesabı.

6/90 hipergeometrik olasılık + Türkiye Sayısal Loto güncel ortalama ikramiye
yapısı kullanılarak gerçekçi beklenen değer (EV) ve risk dağılımı hesaplanır.

İkramiye değerleri çekilişten çekilişe büyük varyasyon gösterir; bu modül
kullanıcı tarafından override edilebilen makul ortalamalar sunar.
"""
from __future__ import annotations

from math import comb
from typing import Optional

# Türkiye Sayısal Loto 2025-2026 ortalama ikramiye değerleri (TL).
# Kaynak: Milli Piyango son çekiliş raporları (medyan, jackpot devirsiz).
# Bunlar TAHMİNİDİR; çekilişe göre değişir, kullanıcı override edebilir.
DEFAULT_PRIZES_TL = {
    6: 100_000_000,   # 6 bilen — devirli ortalama jackpot (sık 50M-1B arası)
    5: 150_000,        # 5 bilen — ortalama paylı ödül
    4: 800,            # 4 bilen
    3: 30,             # 3 bilen
}

DEFAULT_COST_PER_TICKET_TL = 20  # 1 kolon ücreti (2026 itibarıyla)


def hypergeometric_prob(k: int, total: int = 90, drawn: int = 6, picked: int = 6) -> float:
    """Hipergeometrik dağılım: k tutturma olasılığı."""
    if k < 0 or k > picked or k > drawn:
        return 0.0
    return comb(picked, k) * comb(total - picked, drawn - k) / comb(total, drawn)


def hit_distribution() -> dict[int, float]:
    """Bir kupon için her tutturma seviyesinin tek kupondaki olasılığı."""
    return {k: hypergeometric_prob(k) for k in range(7)}


def expected_return_per_ticket(prizes: Optional[dict] = None) -> float:
    prizes = prizes or DEFAULT_PRIZES_TL
    return sum(hypergeometric_prob(k) * v for k, v in prizes.items())


def at_least_n_hits_prob(n: int, num_tickets: int = 1) -> float:
    """En az 1 kuponun >= n sayı tutturma olasılığı (bağımsız kuponlar)."""
    p_lt_n_per_ticket = sum(hypergeometric_prob(k) for k in range(n))
    return 1.0 - p_lt_n_per_ticket ** num_tickets


def budget_summary(
    budget_tl: float = 100,
    cost_per_ticket: float = DEFAULT_COST_PER_TICKET_TL,
    prizes: Optional[dict] = None,
) -> dict:
    """
    Verilen bütçeyle ne kadar kupon oynanabileceği, beklenen geri dönüş,
    net beklenen zarar ve tutturma olasılıklarını hesaplar.
    """
    prizes = prizes or DEFAULT_PRIZES_TL
    num_tickets = max(1, int(budget_tl // cost_per_ticket))
    actual_cost = num_tickets * cost_per_ticket

    ev_per = expected_return_per_ticket(prizes)
    total_ev = num_tickets * ev_per

    p3 = at_least_n_hits_prob(3, num_tickets)
    p4 = at_least_n_hits_prob(4, num_tickets)
    p5 = at_least_n_hits_prob(5, num_tickets)
    p6 = at_least_n_hits_prob(6, num_tickets)

    # Jackpot olmadan beklenen değer (yan ödüller, "gerçekleşebilir" senaryo)
    side_prizes = {k: v for k, v in prizes.items() if k < 6}
    ev_per_side = expected_return_per_ticket(side_prizes)
    total_ev_side = num_tickets * ev_per_side

    return {
        "num_tickets": num_tickets,
        "actual_cost_tl": actual_cost,
        "expected_return_tl": total_ev,
        "expected_return_no_jackpot_tl": total_ev_side,
        "expected_loss_tl": actual_cost - total_ev,
        "rtp_pct": (total_ev / actual_cost) * 100 if actual_cost > 0 else 0,
        "p_3_plus_pct": p3 * 100,
        "p_4_plus_pct": p4 * 100,
        "p_5_plus_pct": p5 * 100,
        "p_jackpot_one_in": int(round(1 / p6)) if p6 > 0 else None,
        "p_3_plus_one_in": int(round(1 / p3)) if p3 > 0 else None,
    }


if __name__ == "__main__":
    import json
    print("=== Bir kupon ===")
    print("Beklenen dönüş:", expected_return_per_ticket(), "TL")
    print("Tek kupon dağılımı:")
    for k, p in hit_distribution().items():
        print(f"  {k} tutturma: %{p*100:.4f}  (1 / {round(1/p) if p>0 else '∞'})")

    for budget in [100, 200, 500, 1000]:
        print(f"\n=== {budget} TL bütçe ===")
        print(json.dumps(budget_summary(budget), indent=2, ensure_ascii=False))
