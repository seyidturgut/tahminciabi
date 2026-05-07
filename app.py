import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from data_manager import load_data
from scraper import ScrapeFailedError, fetch_latest_draw
from math_engine import MathEngine
from predictor import Predictor
from ticket_manager import save_tickets, load_saved_tickets, delete_all_tickets
from analytics.expected_value import DEFAULT_PRIZES_TL, DEFAULT_COST_PER_TICKET_TL
from games import GAMES, get_game
import auto_tracker

APP_VERSION = "v2.7.4"
APP_BUILD_DATE = "2026-05-07"

st.set_page_config(page_title="Tahminci | Sayısal Loto AI", layout="wide", page_icon="🔮")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .stDeployButton {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .ticket-box {
        background: linear-gradient(145deg, #1e1e1e, #2a2a2a);
        border-radius: 12px; padding: 20px; margin: 15px 0;
        display: flex; justify-content: space-around; align-items: center;
        font-size: 26px; font-weight: 800; color: #00E676;
        border: 1px solid #333;
        box-shadow: 0 4px 15px rgba(0, 230, 118, 0.1);
        transition: transform 0.2s ease;
    }
    .ticket-box:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0, 230, 118, 0.2); }
    .ball {
        width: 50px; height: 50px;
        display: flex; align-items: center; justify-content: center;
        background-color: #252525; border-radius: 50%;
        border: 2px solid #00E676;
        text-shadow: 0 0 10px rgba(0, 230, 118, 0.5);
    }
    .ball-match {
        width: 50px; height: 50px;
        display: flex; align-items: center; justify-content: center;
        background-color: #00E676; color: #121212 !important;
        border-radius: 50%; border: 2px solid #00E676;
        box-shadow: 0 0 15px #00E676;
    }
    .ball-joker {
        width: 50px; height: 50px;
        display: flex; align-items: center; justify-content: center;
        background-color: #1a1c23; border-radius: 50%;
        border: 2px solid #FF9800; color: #FF9800;
        text-shadow: 0 0 10px rgba(255, 152, 0, 0.5);
    }
    .ball-superstar {
        width: 50px; height: 50px;
        display: flex; align-items: center; justify-content: center;
        background-color: #1a1c23; border-radius: 50%;
        border: 2px solid #BB86FC; color: #BB86FC;
        text-shadow: 0 0 10px rgba(187, 134, 252, 0.5);
    }
    .bonus-label {
        font-size: 10px; color: #888; align-self: center;
        font-weight: 600; letter-spacing: 0.5px;
        margin: 0 10px 0 20px;
    }
    .ticket-divider {
        width: 1px; height: 40px; background: #333; margin: 0 5px;
        align-self: center;
    }
    div[data-testid="metric-container"] {
        background-color: #1a1c23; border: 1px solid #2e303e;
        padding: 15px; border-radius: 10px;
    }
    .disclaimer {
        background: #1a1c23; border-left: 3px solid #ffaa00;
        padding: 10px 15px; border-radius: 4px;
        font-size: 12px; color: #888;
    }
</style>
""", unsafe_allow_html=True)

# --- LOGIN ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>🔮 Tahminci AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888;'>Sayısal Loto için profesör seviyesi istatistiksel asistan.</p>", unsafe_allow_html=True)
        password = st.text_input("Şifre", type="password", label_visibility="collapsed", placeholder="Şifreniz...")
        if st.button("Sisteme Giriş Yap", width="stretch", type="primary"):
            if password == "Beyincik**94":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Hatalı Şifre!")
        st.markdown(
            "<p class='disclaimer'>⚠️ Bu uygulama eğitim ve istatistiksel analiz amaçlıdır. "
            "Sayısal Loto matematiksel olarak rastgele bir sistemdir; hiçbir model "
            "uzun vadede kazanma olasılığını materyal olarak yükseltmez.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align:center;color:#444;font-size:11px;margin-top:20px;'>"
            f"<code style='color:#00E676;'>{APP_VERSION}</code> · build {APP_BUILD_DATE}</p>",
            unsafe_allow_html=True,
        )
    st.stop()

# --- DATA ---
# NOT: Streamlit'te '_' önekli parametreler cache key'e DAHİL EDİLMEZ.
# Bu yüzden cache busting için underscore-suz isim kullanmak ZORUNLU.

# --- GAME SELECTOR ---
if "selected_game" not in st.session_state:
    st.session_state.selected_game = "sayisal_loto"

game_key = st.session_state.selected_game
GAME = get_game(game_key)


@st.cache_data(show_spinner=False, ttl=3600)
def get_data_for_game(game_key: str, version: str):
    return load_data(get_game(game_key))


@st.cache_resource(show_spinner=False)
def get_engine_and_predictor(version: str, game_key: str):
    g = get_game(game_key)
    df = get_data_for_game(game_key, version)
    from games import number_columns
    cols = number_columns(g)
    return (
        MathEngine(df, total_numbers=g["total"], draw_size=len(cols), number_cols=cols),
        Predictor(df, game=g),
    )


try:
    with st.spinner(f"{GAME['name']} veritabanı yükleniyor / güncelleniyor..."):
        df = get_data_for_game(game_key, APP_VERSION)
except ScrapeFailedError as e:
    st.error(
        f"❌ {GAME['name']} verisi çekilemedi. İnternet bağlantınızı veya kaynak siteyi kontrol edin.\n\n"
        f"Hata: {e}"
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Veri yükleme hatası: {e}")
    st.stop()

engine, predictor = get_engine_and_predictor(APP_VERSION, game_key)


# --- AUTO TRACKER: Her oturumda 1 kez sessizce kontrol et ---
@st.cache_data(ttl=3600, show_spinner=False)
def auto_tracker_check(_version: str, gk: str):
    """1 saatte 1 kez otomatik takip kontrolü (cache_data ile rate-limit)."""
    return auto_tracker.check_now(game_key=gk)


_auto_tracker_settings = auto_tracker.load_settings(game_key=game_key)
if _auto_tracker_settings.get("enabled"):
    try:
        auto_tracker_check(APP_VERSION, game_key)
    except Exception:
        pass  # Sessiz başarısızlık — kullanıcı arayüzü etkilenmesin

# --- HEADER ---
title_col, ver_col = st.columns([5, 1])
with title_col:
    st.title(f"{GAME['emoji']} Tahminci: {GAME['name']} AI Asistanı")
    st.markdown("Markov + Bayesian + Rasgelelik Testleri + LightGBM ile çoklu-model olasılık analizi.")
with ver_col:
    st.markdown(
        f"<div style='text-align:right;color:#666;font-size:12px;margin-top:25px;'>"
        f"<code style='background:#1a1c23;padding:3px 8px;border-radius:4px;color:#00E676;'>"
        f"{APP_VERSION}</code><br><span style='font-size:10px;'>{APP_BUILD_DATE}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

# Oyun seçici
new_game = st.radio(
    "Oyun seç",
    list(GAMES.keys()),
    format_func=lambda k: f"{GAMES[k]['emoji']} {GAMES[k]['name']}",
    horizontal=True,
    label_visibility="collapsed",
    index=list(GAMES.keys()).index(game_key),
)
if new_game != game_key:
    st.session_state.selected_game = new_game
    st.session_state.current_tickets = None
    st.session_state.current_pool = None
    st.cache_resource.clear()
    st.rerun()

st.divider()

# --- SIDEBAR ---
with st.sidebar:
    st.header("📊 Veri Merkezi")
    c1, c2 = st.columns(2)
    c1.metric("Toplam Çekiliş", len(df))
    c2.metric("Son Tarih", df.iloc[-1]["tarih"].strftime("%d-%m-%Y"))
    st.caption(f"Son çekiliş no: **{int(df.iloc[-1]['cekilis_no'])}**" if "cekilis_no" in df.columns else "")

    st.divider()
    st.header("⚙️ Motor Ayarları")
    _ALL_STRATEGIES = [
        ("multi_mini", "🎲 Çoklu Mini-Sistem (Önerilen)"),
        ("wheel", "🎡 Akıllı Wheel Sistemi"),
        ("system", "🎯 Sistemli Oyun (Garanti)"),
        ("professor", "🎓 Profesör Modu"),
        ("super_hybrid", "Süper Hibrit"),
        ("hot", "Sıcak Sayılar"),
        ("cold", "Soğuk Sayılar"),
    ]
    available = [(k, label) for k, label in _ALL_STRATEGIES if k in GAME["strategies"]]
    strategy = st.radio(
        "Strateji",
        [label for _, label in available],
        help=(
            "🎲 Çoklu Mini-Sistem: N adet bağımsız küçük havuz — riski dağıtır.\n"
            "🎡 Akıllı Wheel: covering design — full wheel'in 1/5'i kupon ile "
            "matematiksel garantili kapsama.\n"
            "🎯 Sistemli Oyun: tek havuzlu full wheel — pahalı, jackpot garanti.\n"
            "🎓 Profesör Modu: 4 model ile bağımsız kuponlar."
        ),
    )
    strategy_clean = (strategy.replace("🎓 ", "").replace("🎯 ", "")
                      .replace("🎲 ", "").replace("🎡 ", ""))

    PICKS = GAME["picks"]
    DRAWN = GAME["drawn"]
    TOTAL = GAME["total"]
    COST = GAME["ticket_cost_tl"]
    NON_PICK = TOTAL - DRAWN  # 90-6=84, 34-5=29

    if strategy == "🎲 Çoklu Mini-Sistem (Önerilen)":
        from math import comb
        num_systems = st.slider("Mini-Sistem Sayısı", 2, 10, 5,
                                 help="Bağımsız küçük havuz sayısı")
        pool_size_each = st.select_slider(
            "Her Havuz Boyutu",
            options=[PICKS, PICKS+1, PICKS+2],
            value=PICKS+1,
        )
        kolon_each = comb(pool_size_each, PICKS)
        toplam_kolon = num_systems * kolon_each
        toplam_cost = toplam_kolon * COST

        c_t = comb(TOTAL, pool_size_each)
        p_lt3 = sum(comb(DRAWN, k) * comb(NON_PICK, pool_size_each - k) / c_t for k in range(3))
        p_per = 1 - p_lt3
        p_at_least_one = (1 - p_lt3 ** num_systems) * 100

        st.caption(
            f"**{num_systems} × Sistem {pool_size_each}**: "
            f"{toplam_kolon} kolon × {COST} TL = **{toplam_cost:,} TL**. "
            f"En az 1 havuzda 3+ tutturma şansı: **%{p_at_least_one:.1f}** "
            f"(tek Sistem {pool_size_each} → %{p_per*100:.1f})"
            .replace(",", ".")
        )
        st.info(
            f"💡 Bağımsız havuzlar = riski dağıtır. Aynı bütçeye tek büyük sistem "
            f"yerine birden fazla küçük sistem **3-5x daha sık** 'en az 1 tutar'."
        )
        num_tickets = toplam_kolon
        pool_size = None
        randomize_pool = True
    elif strategy == "🎡 Akıllı Wheel Sistemi":
        from math import comb
        from analytics.wheel_systems import estimated_block_count
        wheel_pool = st.select_slider(
            "Havuz Boyutu",
            options=list(range(PICKS+1, min(16, TOTAL+1))),
            value=min(10, TOTAL),
        )
        wheel_t = st.select_slider(
            "Garanti Seviyesi (havuzda kaç çıkarsa garanti tutar)",
            options=list(range(3, PICKS+1)),
            value=4,
            help="3 = ucuz/zayıf garanti, 4 = dengeli, 5+ = pahalı/güçlü",
        )
        full_wheel = comb(wheel_pool, PICKS)
        est = estimated_block_count(wheel_pool, PICKS, wheel_t)
        if est:
            est_cost = est * COST
            full_cost = full_wheel * COST
            savings = int(100 * (1 - est / full_wheel))
            st.caption(
                f"**Wheel({wheel_pool}, {PICKS}, {wheel_t})**: ~{est} kolon × {COST} TL = "
                f"**~{est_cost:,} TL**. Full wheel: {full_wheel} kolon ({full_cost:,} TL) — "
                f"**%{savings} tasarruf**.\n\n"
                f"📐 Garanti: havuzdaki **{wheel_t}+ sayı** çekilişte çıkarsa, "
                f"kuponlardan en az 1'i **{wheel_t} doğru** tutar."
                .replace(",", ".")
            )
        else:
            st.caption(f"Wheel({wheel_pool}, {PICKS}, {wheel_t}) — üretirken hesaplanacak.")
        randomize_pool = st.toggle("🎲 Her üretimde farklı havuz", value=True,
                                   key=f"wheel_randomize_{game_key}")
        num_tickets = est or full_wheel
        pool_size = wheel_pool
    elif strategy == "🎯 Sistemli Oyun (Garanti)":
        # Oyuna göre uygun pool size seçenekleri
        sys_options = [PICKS, PICKS+1, PICKS+2, PICKS+3, PICKS+4, PICKS+5,
                       PICKS+6, PICKS+9, PICKS+14]
        sys_options = sorted(set(s for s in sys_options if s <= TOTAL))
        pool_size = st.select_slider(
            "Sistem Boyutu (havuzdaki sayı)",
            options=sys_options,
            value=sys_options[2] if len(sys_options) > 2 else sys_options[0],
        )
        randomize_pool = st.toggle(
            "🎲 Her üretimde farklı havuz",
            value=True,
            help=("Açık: top olasılıklı sayıdan ağırlıklı rastgele seçim. "
                  "Kapalı: her zaman aynı top-N (deterministik)."),
        )
        from math import comb
        sys_kolon = comb(pool_size, PICKS)
        sys_cost = sys_kolon * COST
        c_total = comb(TOTAL, pool_size)
        p_lt3 = sum(comb(DRAWN, k) * comb(NON_PICK, pool_size - k) / c_total for k in range(3))
        p_3plus = (1 - p_lt3) * 100
        st.caption(
            f"**Sistem {pool_size}**: {sys_kolon} kolon × {COST} TL = "
            f"**{sys_cost:,} TL**. **Garanti: havuzda 3+ çıkarsa en az 1 kupon tutar.** "
            f"Bunun olma ihtimali **%{p_3plus:.1f}**."
            .replace(",", ".")
        )
        with st.expander("📊 Sistem Boyutları ve Olasılıklar", expanded=False):
            prob_data = []
            for size in sys_options:
                c_t = comb(TOTAL, size)
                p_lt = sum(comb(DRAWN, k) * comb(NON_PICK, size - k) / c_t for k in range(3))
                p_plus = (1 - p_lt) * 100
                kol = comb(size, PICKS)
                cost = kol * COST
                prob_data.append({
                    "Sistem": f"S{size}",
                    "Kolon": f"{kol:,}",
                    "Maliyet": f"{cost:,} TL",
                    "3+ İhtimali": f"%{p_plus:.2f}"
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(prob_data), use_container_width=True)
            st.markdown(
                "💡 **Not:** Havuzdaki 3+ sayının çıkma şansı matematiksel olarak düşüktür. "
                "Bu oyun stratejisidir — yüksek kazanç hedefi değil, garantili kazanç hedefidir."
            )
        num_tickets = sys_kolon
    else:
        pool_size = None
        randomize_pool = False
        num_tickets = st.slider("Üretilecek Kolon Sayısı", 1, 10, 5)

    # --- AVOID-THE-CROWD ---
    if GAME["total"] >= 32:
        st.divider()
        crowd_on = st.toggle(
            "🎯 Avoid-the-Crowd (paylaşma riskini azalt)",
            value=False,
            help=("Çoğu oyuncu doğum tarihi seçer (1-31). Bu mod **32+** sayılara "
                  "ağırlık verir. Kazanma şansını DEĞİŞTİRMEZ ama jackpot'u kaç "
                  "kişiyle paylaşacağını AZALTIR (Expected Value optimizasyonu)."),
            key=f"avoid_crowd_{game_key}",
        )
        if crowd_on:
            _CROWD_LEVELS = {"Hafif": 1.2, "Orta": 1.4, "Agresif": 1.8}
            crowd_label = st.selectbox(
                "Boost şiddeti",
                options=list(_CROWD_LEVELS.keys()),
                index=1,
                key=f"crowd_strength_{game_key}",
            )
            predictor.set_avoid_crowd(_CROWD_LEVELS[crowd_label])
        else:
            predictor.set_avoid_crowd(1.0)

    st.divider()
    if st.button("🌐 Son Çekilişi Senkronize Et", width="stretch"):
        try:
            with st.spinner("Son çekiliş çekiliyor..."):
                st.cache_data.clear()
                st.cache_resource.clear()
            st.rerun()
        except ScrapeFailedError as e:
            st.error(f"Senkronizasyon başarısız: {e}")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Tahmin Üretici",
    "🎟️ Kupon & Sonuç Kontrolü",
    "📈 Derin İstatistikler",
    "🤖 Otomasyon",
])


JOKER_BONUS_TL = 50
SS_BONUS_TL = 100


def calc_prize_tl(game: dict, match_count: int, bonus_hits: dict,
                  include_bonus: bool = False) -> int:
    """Oyuna göre **tek bir kupon** için ödül hesabı.

    Args:
        match_count: ana sayılarda doğru sayısı.
        bonus_hits: {"joker": True, "superstar": False, "sans_topu": True}
        include_bonus: Sayısal Loto için Joker/SS bonusunu dahil et.
            Sistem oyununda tüm kuponlar aynı Joker+SS paylaşır — toplam
            kazançta bonusu sadece BİR kez saymak için per-ticket False
            kullanılır, sistem geneli ayrıca eklenir.
    """
    prizes = game.get("prizes_tl", {})
    key = game["key"]

    if key == "sans_topu":
        st_hit = 1 if bonus_hits.get("sans_topu") else 0
        return int(prizes.get((match_count, st_hit), 0))

    if key == "on_numara":
        return int(prizes.get(match_count, 0))

    # Sayısal Loto: ana ödül + (opsiyonel) Joker/SS bonusu
    main = prizes.get(match_count, 0) if match_count >= 3 else 0
    if not include_bonus:
        return int(main)
    bonus = 0
    if bonus_hits.get("joker"):
        bonus += JOKER_BONUS_TL
    if bonus_hits.get("superstar"):
        bonus += SS_BONUS_TL
    return int(main + bonus)


def fmt_tl(v: int) -> str:
    if v >= 1_000_000:
        return f"₺{v/1_000_000:.1f}M".replace(".0M", "M")
    if v >= 1_000:
        return f"₺{v/1_000:.1f}K".replace(".0K", "K")
    return f"₺{v}"


def prize_badge_html(game: dict, match_count: int, bonus_hits: dict, drawn_known: bool) -> str:
    if not drawn_known:
        return ""
    prize = calc_prize_tl(game, match_count, bonus_hits)
    if prize > 0:
        color = "#00E676" if prize >= 800 else "#FFD54F"
        return (f"<span style='margin-left:12px;color:{color};font-weight:700;font-size:15px;'>"
                f"Kazanç: {fmt_tl(prize)}</span>")
    return ("<span style='margin-left:12px;color:#666;font-size:13px;'>"
            "Kazanç: ₺0</span>")


_BONUS_STYLE = {
    "joker": ("JOKER", "ball-joker"),
    "superstar": ("SÜPER STAR", "ball-superstar"),
    "sans_topu": ("ŞANS TOPU", "ball-joker"),
}


def render_ticket(numbers, joker=None, superstar=None, sans_topu=None,
                  drawn_numbers=None, drawn_joker=None, drawn_superstar=None,
                  drawn_sans_topu=None, badge_html=""):
    parts = []
    for n in numbers:
        cls = "ball-match" if drawn_numbers and n in drawn_numbers else "ball"
        parts.append(f"<div class='{cls}'>{n:02d}</div>")

    bonus_data = [
        ("joker", joker, drawn_joker),
        ("superstar", superstar, drawn_superstar),
        ("sans_topu", sans_topu, drawn_sans_topu),
    ]
    first_bonus = True
    for key, val, drawn_val in bonus_data:
        if val is None:
            continue
        if first_bonus:
            parts.append("<div class='ticket-divider'></div>")
            first_bonus = False
        label, base_cls = _BONUS_STYLE[key]
        parts.append(f"<span class='bonus-label'>{label}</span>")
        cls = "ball-match" if drawn_val is not None and val == drawn_val else base_cls
        parts.append(f"<div class='{cls}'>{val:02d}</div>")

    st.markdown(f"<div class='ticket-box'>{''.join(parts)}{badge_html}</div>",
                unsafe_allow_html=True)


# --- Notification: yeni otomatik takip sonuçları ---
_unread = auto_tracker.get_unread_count(game_key=game_key) if _auto_tracker_settings.get("enabled") else 0
if _unread > 0:
    st.info(
        f"🔔 **{_unread} yeni çekilişin sonucu** otomatik kontrol edildi. "
        f"📊 *Otomasyon* sekmesinde detayları görebilirsin."
    )

# TAB 1: GENERATOR
with tab1:
    st.subheader(f"🧠 Olasılık Motoru (Strateji: {strategy})")
    st.caption(
        "4 model ağırlıklı birleştirilir → Markov 0.20 + Bayesian 0.25 + Sapma 0.20 + ML 0.35. "
        "Mathematical filters: Gauss toplam, ardışık limit, asal denge, ondalık dağılım, "
        "pozisyonel sınırlar, yüksek-skorlu sayı zorunluluğu."
    )

    if "current_tickets" not in st.session_state:
        st.session_state.current_tickets = []
        st.session_state.current_pool = None

    if st.button("🚀 KUPON ÜRET", width="stretch", type="primary"):
        if strategy == "🎲 Çoklu Mini-Sistem (Önerilen)":
            with st.spinner(f"{num_systems} bağımsız mini-sistem hazırlanıyor..."):
                tickets, pools = predictor.generate_multi_mini_system(
                    num_systems=num_systems, pool_size_each=pool_size_each
                )
            st.session_state.current_tickets = tickets
            st.session_state.current_pool = pools  # list of lists
            st.session_state.current_multi = True
            st.success(
                f"✅ {num_systems} × Sistem {pool_size_each} üretildi — "
                f"**{len(tickets)} kolon** ({num_systems} bağımsız havuz)."
            )
        elif strategy == "🎡 Akıllı Wheel Sistemi":
            with st.spinner(f"Wheel({pool_size}, {PICKS}, {wheel_t}) hesaplanıyor..."):
                tickets, pool, info = predictor.generate_wheel_system(
                    pool_size=pool_size, guarantee_t=wheel_t,
                    randomize_pool=randomize_pool,
                )
            st.session_state.current_tickets = tickets
            st.session_state.current_pool = pool
            st.session_state.current_multi = False
            st.success(
                f"✅ Wheel({pool_size}, {PICKS}, {wheel_t}) — **{len(tickets)} kolon** "
                f"(full wheel'den %{info['savings_pct']} az). Havuz: {pool}"
            )
        elif strategy == "🎯 Sistemli Oyun (Garanti)":
            with st.spinner(f"Sistem {pool_size} hazırlanıyor..."):
                tickets, pool = predictor.generate_system_tickets(
                    pool_size=pool_size, randomize_pool=randomize_pool
                )
            st.session_state.current_tickets = tickets
            st.session_state.current_pool = pool
            st.session_state.current_multi = False
            st.success(
                f"✅ Sistem {pool_size} üretildi — **{len(tickets)} kolon**. "
                f"Havuz: {pool}"
            )
        else:
            with st.spinner("Olasılık motoru çalışıyor, filtreler uygulanıyor..."):
                tickets, attempts = predictor.generate_tickets(
                    num_tickets=num_tickets, strategy=strategy_clean
                )
            st.session_state.current_tickets = tickets
            st.session_state.current_pool = None
            st.session_state.current_multi = False
            st.success(f"✅ {len(tickets)} kupon üretildi ({attempts} varyasyon denendi).")

    if st.session_state.current_tickets:
        is_multi = st.session_state.get("current_multi", False)
        if is_multi and st.session_state.current_pool:
            pools = st.session_state.current_pool
            for idx, pool in enumerate(pools):
                pool_balls = "".join(
                    f"<div class='ball-superstar' style='border-color:#FFD54F;color:#FFD54F;'>{n:02d}</div>"
                    for n in pool
                )
                st.markdown(
                    f"<div style='background:#1a1c23;border:1px solid #333;border-radius:10px;"
                    f"padding:12px;margin:8px 0;'>"
                    f"<div style='color:#FFD54F;font-size:13px;font-weight:600;margin-bottom:8px;'>"
                    f"🎲 MİNİ-SİSTEM #{idx+1} ({len(pool)} sayı)</div>"
                    f"<div style='display:flex;gap:6px;flex-wrap:wrap;'>{pool_balls}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        elif st.session_state.current_pool:
            pool = st.session_state.current_pool
            pool_balls = "".join(
                f"<div class='ball-superstar' style='border-color:#FFD54F;color:#FFD54F;'>{n:02d}</div>"
                for n in pool
            )
            st.markdown(
                f"<div style='background:#1a1c23;border:1px solid #333;border-radius:10px;"
                f"padding:15px;margin:10px 0;'>"
                f"<div style='color:#FFD54F;font-size:14px;font-weight:600;margin-bottom:10px;'>"
                f"🎯 SİSTEM HAVUZU ({len(pool)} sayı)</div>"
                f"<div style='display:flex;gap:8px;flex-wrap:wrap;'>{pool_balls}</div>"
                f"<div style='color:#888;font-size:12px;margin-top:10px;'>"
                f"Bu sayılardan en az 3'ü çekilirse <b>en az bir kolon kesin 3 tuttur</b>. "
                f"Daha fazla doğru çıkarsa daha çok ve daha yüksek dereceli tutturma garantisi.</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        bonus_label = " + ".join(b["label"] for b in GAME["bonuses"]) if GAME["bonuses"] else ""
        ticket_header = f"{GAME['picks']} Ana"
        if bonus_label:
            ticket_header += f" + {bonus_label}"
        st.markdown(f"### 🎲 Üretilen Kolonlar ({ticket_header})")
        for ticket in st.session_state.current_tickets:
            conf = ticket.get("confidence", 0.0)
            badge = ""
            if conf > 0:
                color = "#00E676" if conf >= 0.55 else ("#ffd54f" if conf >= 0.5 else "#ff7043")
                badge = (f"<div style='color:{color};font-size:14px;align-self:center;"
                         f"margin-left:15px;'>Güven: %{conf*100:.1f}</div>")
            render_ticket(
                list(ticket["main"]),
                joker=ticket.get("joker"),
                superstar=ticket.get("superstar"),
                sans_topu=ticket.get("sans_topu"),
                badge_html=badge,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 KUPONLARI SİSTEME KAYDET", width="stretch"):
            next_draw = int(df["cekilis_no"].max()) + 1
            save_tickets(st.session_state.current_tickets, strategy_clean,
                         valid_from_draw_no=next_draw, game=game_key)
            st.toast(f"Kuponlar kaydedildi! Çekiliş #{next_draw}'dan itibaren otomatik takip edilir.")
            st.session_state.current_tickets = []
            st.rerun()

    if st.session_state.current_tickets:
        st.divider()
        st.markdown(f"### 💰 Bu Kupon Tutarsa Ne Kazanırsın? ({GAME['name']})")
        st.caption(
            f"{GAME['name']} ödül yapısına göre tahmini değerler. Çekilişe göre değişir; "
            "düzenleyebilirsin."
        )

        def fmt_tl_local(v: int) -> str:
            return f"{v:,.0f} TL".replace(",", ".")

        n_tickets = len(st.session_state.current_tickets)
        cost_per = GAME["ticket_cost_tl"]
        total_cost = n_tickets * cost_per

        if game_key == "sayisal_loto":
            with st.expander("⚙️ İkramiye Değerlerini Düzenle", expanded=False):
                pc1, pc2, pc3, pc4 = st.columns(4)
                prize6 = pc1.number_input("6 bilen (Jackpot)", min_value=1_000_000,
                                          max_value=10_000_000_000,
                                          value=GAME["prizes_tl"][6], step=10_000_000, key="p6")
                prize5 = pc2.number_input("5 bilen", min_value=1_000,
                                          max_value=5_000_000,
                                          value=GAME["prizes_tl"][5], step=10_000, key="p5")
                prize4 = pc3.number_input("4 bilen", min_value=10,
                                          max_value=100_000,
                                          value=GAME["prizes_tl"][4], step=50, key="p4")
                prize3 = pc4.number_input("3 bilen", min_value=1,
                                          max_value=1_000,
                                          value=GAME["prizes_tl"][3], step=5, key="p3")
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("3 Bilirsen", fmt_tl_local(prize3))
            pc2.metric("4 Bilirsen", fmt_tl_local(prize4))
            pc3.metric("5 Bilirsen", fmt_tl_local(prize5))
            pc4.metric("6 Bilirsen 🎰", fmt_tl_local(prize6))

        elif game_key == "sans_topu":
            prizes = GAME["prizes_tl"]
            with st.expander("⚙️ İkramiye Değerlerini Düzenle", expanded=False):
                st.caption("Anahtar: (ana, şans topu) — örn. (5,1) = 5 ana + Şans Topu")
                cols = st.columns(4)
                custom = {}
                for i, (k, v) in enumerate(sorted(prizes.items(), reverse=True)):
                    main_h, st_h = k
                    label = f"{main_h}+1" if st_h else f"{main_h}"
                    with cols[i % 4]:
                        custom[k] = st.number_input(
                            f"{label} bilen",
                            min_value=0, max_value=100_000_000,
                            value=int(v),
                            step=max(1, int(v // 50)),
                            key=f"st_p_{main_h}_{st_h}",
                        )
            cols = st.columns(4)
            top_keys = [(5, 1), (5, 0), (4, 1), (3, 1)]
            labels = ["5+1 🎰", "5 ana", "4+1 ⭐", "3+1"]
            for i, (key, lab) in enumerate(zip(top_keys, labels)):
                cols[i].metric(lab, fmt_tl_local(custom.get(key, prizes.get(key, 0))))

        elif game_key == "on_numara":
            prizes = GAME["prizes_tl"]
            with st.expander("⚙️ İkramiye Değerlerini Düzenle", expanded=False):
                st.caption("On Numara: 1-5 doğru ödüllü değildir.")
                cols = st.columns(3)
                custom = {}
                for i, (k, v) in enumerate(sorted(prizes.items(), reverse=True)):
                    label = f"{k} doğru" if k > 0 else "0 doğru"
                    with cols[i % 3]:
                        custom[k] = st.number_input(
                            f"{label}",
                            min_value=0, max_value=100_000_000,
                            value=int(v),
                            step=max(1, int(v // 50)),
                            key=f"on_p_{k}",
                        )
            cols = st.columns(5)
            top_keys = [10, 9, 8, 7, 0]
            labels = ["10 🎰", "9 doğru", "8 doğru", "7 doğru", "0 doğru ⭐"]
            for i, (key, lab) in enumerate(zip(top_keys, labels)):
                cols[i].metric(lab, fmt_tl_local(custom.get(key, prizes.get(key, 0))))

        st.info(
            f"💸 Toplam Maliyet: **{n_tickets} kolon × {cost_per} TL = "
            f"{fmt_tl_local(total_cost)}**"
        )

# TAB 2: TICKET CHECKER
with tab2:
    st.subheader(f"🎟️ {GAME['name']} — Kayıtlı Kuponlar ve Sonuç Sorgulama")
    saved = load_saved_tickets(game=game_key)
    if not saved:
        st.info(f"Sistemde {GAME['name']} için henüz oynanmış bir kuponunuz bulunmuyor.")
    else:
        with st.expander("🔍 Sonuç Kontrol Paneli", expanded=True):
            draw_input = st.text_input(
                f"{GAME['drawn']} Çekilen Sayıyı Girin (virgülle ayır)",
                placeholder=f"Örn: {', '.join(str(i) for i in range(1, GAME['drawn']+1))}",
            )
            drawn_bonus_inputs: dict[str, int] = {}
            if GAME["bonuses"]:
                bonus_cols = st.columns(len(GAME["bonuses"]))
                for i, bspec in enumerate(GAME["bonuses"]):
                    with bonus_cols[i]:
                        v = st.number_input(
                            f"{bspec['label']} (1-{bspec['total']})",
                            min_value=0, max_value=bspec["total"], value=0, step=1,
                            help="0 = girilmedi",
                            key=f"bonus_input_{bspec['key']}",
                        )
                        drawn_bonus_inputs[bspec["key"]] = v if v > 0 else None
            drawn_numbers = []
            if draw_input:
                try:
                    drawn_numbers = [int(x.strip()) for x in draw_input.split(",")]
                    if len(drawn_numbers) != GAME["drawn"]:
                        st.error(f"Lütfen tam {GAME['drawn']} adet sayı girin!")
                        drawn_numbers = []
                except ValueError:
                    st.error("Lütfen geçerli sayılar girin!")
            # Geriye uyumluluk için legacy isimler
            drawn_joker = drawn_bonus_inputs.get("joker")
            drawn_ss = drawn_bonus_inputs.get("superstar")

        st.divider()
        col1, col2 = st.columns([4, 1])
        col1.markdown("### 📋 Oynanan Kuponlar")
        with col2:
            if st.button("🗑️ Tümünü Sil", width="stretch"):
                delete_all_tickets(game=game_key)
                st.rerun()

        for record in reversed(saved):
            st.caption(f"📅 {record['tarih']} | 🧠 Strateji: {record['strateji']}")
            for ticket in record["kuponlar"]:
                if isinstance(ticket, dict):
                    main = list(ticket["main"]) if "main" in ticket else list(ticket.get("kuponlar", []))
                    t_joker = ticket.get("joker")
                    t_ss = ticket.get("superstar")
                    t_st = ticket.get("sans_topu")
                else:
                    main = list(ticket)
                    t_joker = t_ss = t_st = None

                match_count = len(set(main).intersection(drawn_numbers)) if drawn_numbers else 0
                joker_hit = t_joker is not None and drawn_joker is not None and t_joker == drawn_joker
                ss_hit = t_ss is not None and drawn_ss is not None and t_ss == drawn_ss
                drawn_st = drawn_bonus_inputs.get("sans_topu")
                st_hit = t_st is not None and drawn_st is not None and t_st == drawn_st
                bonus_hits = {"joker": joker_hit, "superstar": ss_hit, "sans_topu": st_hit}
                badge = ""
                if drawn_numbers:
                    parts_b = []
                    win_threshold = 3 if game_key != "on_numara" else 6
                    if match_count >= win_threshold or (game_key == "on_numara" and match_count == 0):
                        parts_b.append(f"<span style='color:#111;background:#00E676;padding:3px 10px;"
                                       f"border-radius:6px;font-size:14px;'>{match_count} ANA 🏆</span>")
                    elif match_count > 0:
                        parts_b.append(f"<span style='color:#ff4444;font-size:14px;'>{match_count} ana</span>")
                    if joker_hit:
                        parts_b.append("<span style='color:#FF9800;font-size:14px;'>+JOKER ⭐</span>")
                    if ss_hit:
                        parts_b.append("<span style='color:#BB86FC;font-size:14px;'>+SÜPER STAR ⭐</span>")
                    if st_hit:
                        parts_b.append("<span style='color:#FF9800;font-size:14px;'>+ŞANS TOPU ⭐</span>")
                    parts_b.append(prize_badge_html(GAME, match_count, bonus_hits, True))
                    if parts_b:
                        badge = (f"<div style='align-self:center;margin-left:15px;display:flex;"
                                 f"gap:8px;align-items:center;'>{''.join(parts_b)}</div>")
                render_ticket(main, joker=t_joker, superstar=t_ss, sans_topu=t_st,
                              drawn_numbers=drawn_numbers,
                              drawn_joker=drawn_joker, drawn_superstar=drawn_ss,
                              drawn_sans_topu=drawn_st,
                              badge_html=badge)
            st.markdown("<br>", unsafe_allow_html=True)

# TAB 3: STATISTICS
with tab3:
    st.subheader("📈 Derin Analitik Merkez")
    freq_df = engine.calculate_frequencies()
    gaps_df = engine.calculate_gaps()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🔥 En Sıcak Sayılar")
        st.dataframe(freq_df.head(10).style.format({"yuzde": "{:.2f}%"}),
                     width="stretch")
    with c2:
        st.markdown("#### 🧊 En Soğuk Sayılar")
        st.dataframe(freq_df.tail(10).style.format({"yuzde": "{:.2f}%"}),
                     width="stretch")
    with c3:
        st.markdown("#### ⏳ Gecikme (Gap) Analizi")
        st.dataframe(gaps_df.head(10), width="stretch")

    st.divider()
    mean_sum, std_sum = engine.get_sum_distribution_stats()
    st.info(
        f"Kupon toplamı güven aralığı (μ ± 1.5σ): "
        f"**{mean_sum - 1.5*std_sum:.0f} – {mean_sum + 1.5*std_sum:.0f}**. "
        f"Bu aralığın dışı elenir."
    )

    st.divider()
    st.markdown("### 🎓 Profesör Katmanı Çıktıları")
    with st.spinner("Olasılık motoru çıktıları hesaplanıyor..."):
        prob_out = predictor.get_probabilities()

    rs = prob_out.randomness_summary
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("χ² p-değeri", f"{rs['chi_square']['p_value']:.4f}",
               help="H0: uniform 6/90. p<0.05 → uniform reddedilir.")
    cc2.metric("KS D", f"{rs['ks']['D']:.4f}",
               help="Frekans CDF'in uniform CDF'e maksimum uzaklığı.")
    cc3.metric("Tek/Çift Runs p", f"{rs['runs_odd_even']['p_value']:.4f}",
               help="Çekilişlerin tek/çift dağılımının ardışık bağımsızlığı.")

    if rs["chi_square"]["rejects_uniform_at_05"]:
        st.warning("⚠️ Veri **uniform dağılımdan istatistiksel olarak sapıyor** (p<0.05). "
                   "Sapan sayıların sömürülmesi anlamlı olabilir.")
    else:
        st.success("✅ Veri uniform dağılımla tutarlı. Tahminler matematik filtreler ve "
                   "ML pattern'lerine dayanmalıdır.")

    st.markdown("#### 📊 Bir Sonraki Çekiliş için Olasılık Skorları")
    score_df = pd.DataFrame({
        "sayi": np.arange(1, GAME["total"] + 1),
        "Markov": prob_out.markov,
        "Bayesian": prob_out.bayesian,
        "Sapma": prob_out.deviation,
        "ML (LightGBM)": prob_out.ml,
        "Final": prob_out.final,
    }).sort_values("Final", ascending=False)

    fig = px.bar(score_df.head(20), x="sayi", y="Final",
                 title="Top 20 sayı — Final birleşik olasılık",
                 labels={"sayi": "Sayı", "Final": "Olasılık"})
    fig.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                      font_color="#FAFAFA", xaxis=dict(tickmode="linear"))
    st.plotly_chart(fig, width="stretch")

    with st.expander("Tüm Sayılar — Model Bazında Olasılıklar"):
        st.dataframe(score_df.style.format({
            "Markov": "{:.4f}", "Bayesian": "{:.4f}", "Sapma": "{:.4f}",
            "ML (LightGBM)": "{:.4f}", "Final": "{:.4f}",
        }), width="stretch")

    with st.expander("Geçmiş Veritabanı (Son 100 Çekiliş)"):
        st.dataframe(df.sort_values("tarih", ascending=False).head(100),
                     width="stretch")

# TAB 4: AUTOMATION
with tab4:
    st.subheader("🤖 Otomatik Sonuç Takibi")
    st.markdown(
        "Açtığında: app her yüklendiğinde son çekilişler otomatik kontrol edilir, "
        "kayıtlı kuponların değerlendirilir ve aşağıdaki geçmişe işlenir. Sen app'i "
        "açtığında yeni sonuçlarını hazır bulursun."
    )

    settings = auto_tracker.load_settings(game_key=game_key)
    is_enabled = settings.get("enabled", False)

    cc1, cc2 = st.columns([2, 1])
    with cc1:
        new_state = st.toggle(
            "🟢 Otomatik Takip" if is_enabled else "⚪ Otomatik Takip",
            value=is_enabled,
            help=("Açıkken: app her yüklendiğinde (en fazla saatte 1) son "
                  "çekilişler kontrol edilir, kuponlar otomatik değerlendirilir."),
            key=f"auto_track_toggle_{game_key}",
        )
        if new_state != is_enabled:
            auto_tracker.set_enabled(new_state, game_key=game_key)
            st.cache_data.clear()
            st.rerun()
    with cc2:
        if st.button("🔍 Şimdi Kontrol Et", width="stretch",
                     disabled=not is_enabled, key=f"auto_check_{game_key}"):
            with st.spinner("Yeni çekilişler kontrol ediliyor..."):
                st.cache_data.clear()
                result = auto_tracker.check_now(game_key=game_key)
            if result.get("error"):
                st.error(f"Kontrol başarısız: {result['error']}")
            elif result["new_records"] == 0:
                st.toast("Yeni çekiliş yok.")
            else:
                st.success(f"{result['new_records']} yeni çekiliş işlendi. "
                           f"Toplam tahmini kazanç: {result['winnings']:,} TL")
                st.rerun()

    if is_enabled:
        st.success(
            f"✅ Otomatik takip **AÇIK**. Son kontrol edilen çekiliş: "
            f"#{settings.get('last_checked_draw_no', '—')}"
        )
    else:
        st.warning("⚠️ Otomatik takip **KAPALI**. Açtığında app her yüklendiğinde son sonuçları kontrol eder.")

    st.divider()
    st.markdown("### 📊 Otomatik Takip Geçmişi")

    tracking = auto_tracker.load_tracking(game_key=game_key)
    if not tracking:
        st.info(
            "Henüz otomatik takip sonucu yok. Otomatik takibi açtıktan sonra "
            "kupon kaydet → çekiliş yapıldıktan sonra app'i tekrar aç → "
            "sonuçların burada görünür."
        )
    else:
        ckl1, ckl2, ckl3 = st.columns(3)
        total_wins = sum(t.get("toplam_tahmini_kazanc_tl", 0) for t in tracking)
        total_winning_tickets = sum(len(t.get("kazanan_kuponlar", [])) for t in tracking)
        ckl1.metric("İşlenen Çekiliş", len(tracking))
        ckl2.metric("Tutturan Kupon Sayısı", total_winning_tickets)
        ckl3.metric("Toplam Tahmini Kazanç",
                    f"{total_wins:,.0f} TL".replace(",", "."))

        if st.button("✅ Hepsini Okundu İşaretle", width="stretch",
                     key=f"mark_seen_{game_key}"):
            auto_tracker.mark_all_seen(game_key=game_key)
            st.rerun()

        for record in reversed(tracking[-30:]):  # Son 30 çekiliş
            n = record["cekilis_no"]
            tarih = record["tarih"]
            wins = record.get("kazanan_kuponlar", [])
            tot = record.get("toplam_tahmini_kazanc_tl", 0)

            header = f"🎲 Çekiliş #{n} — {tarih}"
            if tot > 0:
                header += f"  ·  💰 {tot:,.0f} TL".replace(",", ".")

            with st.expander(header, expanded=(tot > 0)):
                drawn = record["drawn"]
                d_joker = record.get("joker")
                d_ss = record.get("superstar")
                d_st = record.get("sans_topu")
                st.caption("Çekilen Sayılar:")
                render_ticket(drawn, joker=d_joker, superstar=d_ss, sans_topu=d_st,
                              drawn_numbers=drawn,
                              drawn_joker=d_joker, drawn_superstar=d_ss,
                              drawn_sans_topu=d_st)

                if not wins:
                    st.write("Bu çekilişte kayıtlı kuponlardan hiçbiri ödüllü tutturma yapamadı.")
                else:
                    st.markdown("**Tutturan Kuponlar:**")
                    for w in wins:
                        prize = w.get("estimated_prize_tl", 0)
                        parts = []
                        if w["main_hits"] >= 3 or (w["main_hits"] == 0 and game_key == "on_numara"):
                            parts.append(f"<span style='color:#111;background:#00E676;"
                                         f"padding:2px 8px;border-radius:5px;font-size:13px;'>"
                                         f"{w['main_hits']} ana 🏆</span>")
                        if w.get("joker_hit"):
                            parts.append("<span style='color:#FF9800;font-size:13px;'>+JOKER</span>")
                        if w.get("superstar_hit"):
                            parts.append("<span style='color:#BB86FC;font-size:13px;'>+SÜPER STAR</span>")
                        if w.get("sans_topu_hit"):
                            parts.append("<span style='color:#FF9800;font-size:13px;'>+ŞANS TOPU</span>")
                        if prize > 0:
                            parts.append(f"<span style='color:#00E676;font-weight:600;"
                                         f"font-size:14px;'>≈ {prize:,.0f} TL</span>".replace(",", "."))
                        badge = (f"<div style='align-self:center;margin-left:15px;display:flex;"
                                 f"gap:8px;flex-wrap:wrap;'>{''.join(parts)}</div>")
                        render_ticket(w["main"],
                                      joker=w.get("joker"),
                                      superstar=w.get("superstar"),
                                      sans_topu=w.get("sans_topu"),
                                      drawn_numbers=drawn,
                                      drawn_joker=d_joker, drawn_superstar=d_ss,
                                      drawn_sans_topu=d_st,
                                      badge_html=badge)
                        st.caption(f"📅 Kaydedildi: {w.get('record_tarih', '?')}")

    st.divider()
    st.markdown(f"### 🌐 {GAME['name']} — Manuel Anlık Sonuç (Hızlı Bakış)")
    if st.button("Güncel Sonucu Getir", width="stretch", type="primary"):
        try:
            with st.spinner(f"{GAME['name']} son çekiliş kaynaktan çekiliyor..."):
                latest = fetch_latest_draw(game=GAME)
            date_str = latest["tarih"].strftime("%d-%m-%Y")
            drawn = latest["sayilar"]
            d_joker = latest.get("joker")
            d_ss = latest.get("superstar")
            d_st = latest.get("sans_topu")
            st.success(f"✅ Çekiliş #{latest['cekilis_no']} — {date_str}")

            st.markdown("### 🎲 Çekiliş Sonucu")
            render_ticket(drawn, joker=d_joker, superstar=d_ss, sans_topu=d_st,
                          drawn_numbers=drawn,
                          drawn_joker=d_joker, drawn_superstar=d_ss,
                          drawn_sans_topu=d_st)

            if saved:
                # Önce tüm kuponları değerlendir, toplam kazancı hesapla.
                # Sayısal Loto Sistem oyununda 28 kupon aynı Joker+SS'yi paylaşır;
                # bonus ödülleri kayıt (record) başına BİR KEZ sayılır.
                evaluated = []
                total_winnings = 0
                winning_count = 0
                bonus_credit_per_record: dict = {}  # record_id → (joker_done, ss_done)
                for record in reversed(saved):
                    rec_id = record.get("id", id(record))
                    bonus_credit_per_record.setdefault(rec_id, {"joker": False, "ss": False})
                    for ticket in record["kuponlar"]:
                        if isinstance(ticket, dict):
                            main = list(ticket["main"]) if "main" in ticket else list(ticket.get("kuponlar", []))
                            t_joker = ticket.get("joker")
                            t_ss = ticket.get("superstar")
                            t_st = ticket.get("sans_topu")
                        else:
                            main = list(ticket)
                            t_joker = t_ss = t_st = None
                        m = len(set(main).intersection(drawn))
                        joker_hit = t_joker is not None and t_joker == d_joker
                        ss_hit = t_ss is not None and t_ss == d_ss
                        st_hit = t_st is not None and d_st is not None and t_st == d_st
                        bonus_hits = {"joker": joker_hit, "superstar": ss_hit, "sans_topu": st_hit}
                        # Per-ticket badge: sadece ana ödül (bonusu sistem-genelinde topluyoruz)
                        prize_main_only = calc_prize_tl(GAME, m, bonus_hits, include_bonus=False)
                        if prize_main_only > 0:
                            winning_count += 1
                            total_winnings += prize_main_only
                        # Sistem-geneli Joker/SS bonusu: kayıt başına bir kez
                        if game_key == "sayisal_loto":
                            credit = bonus_credit_per_record[rec_id]
                            if joker_hit and not credit["joker"]:
                                total_winnings += JOKER_BONUS_TL
                                credit["joker"] = True
                            if ss_hit and not credit["ss"]:
                                total_winnings += SS_BONUS_TL
                                credit["ss"] = True
                        evaluated.append({
                            "main": main, "t_joker": t_joker, "t_ss": t_ss, "t_st": t_st,
                            "m": m, "joker_hit": joker_hit, "ss_hit": ss_hit,
                            "st_hit": st_hit, "bonus_hits": bonus_hits, "prize": prize_main_only,
                        })

                # BÜYÜK TOPLAM KAZANÇ BANNER'I
                total_str = f"{total_winnings:,.0f}".replace(",", ".")
                banner_color = "#00E676" if total_winnings > 0 else "#666"
                glow = f"box-shadow:0 0 30px {banner_color}55;" if total_winnings > 0 else ""
                st.markdown(
                    f"<div style='background:linear-gradient(135deg,#0e1f17,#1a2e23);"
                    f"border:2px solid {banner_color};border-radius:16px;padding:30px 25px;"
                    f"margin:20px 0;text-align:center;{glow}'>"
                    f"<div style='color:#888;font-size:14px;font-weight:600;letter-spacing:1.5px;"
                    f"text-transform:uppercase;margin-bottom:8px;'>"
                    f"💰 Toplam Tahmini Kazancın</div>"
                    f"<div style='color:{banner_color};font-size:54px;font-weight:900;"
                    f"line-height:1.1;text-shadow:0 0 20px {banner_color}88;'>"
                    f"₺ {total_str}</div>"
                    f"<div style='color:#aaa;font-size:13px;margin-top:10px;'>"
                    f"{winning_count} tutan kupon / {len(evaluated)} toplam kupon"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

                st.markdown("### 📊 Sizin Kuponlarınızdaki Durum")
                for ev in evaluated:
                    parts_b = []
                    win_threshold = 3 if game_key != "on_numara" else 6
                    m = ev["m"]
                    if m >= win_threshold or (game_key == "on_numara" and m == 0):
                        parts_b.append(f"<span style='color:#111;background:#00E676;padding:3px 10px;"
                                       f"border-radius:6px;font-size:14px;'>{m} ANA 🏆</span>")
                    elif m > 0:
                        parts_b.append(f"<span style='color:#ff4444;font-size:14px;'>{m} ana</span>")
                    if ev["joker_hit"]:
                        parts_b.append("<span style='color:#FF9800;font-size:14px;'>+JOKER ⭐</span>")
                    if ev["ss_hit"]:
                        parts_b.append("<span style='color:#BB86FC;font-size:14px;'>+SÜPER STAR ⭐</span>")
                    if ev["st_hit"]:
                        parts_b.append("<span style='color:#FF9800;font-size:14px;'>+ŞANS TOPU ⭐</span>")
                    parts_b.append(prize_badge_html(GAME, m, ev["bonus_hits"], True))
                    badge = (f"<div style='align-self:center;margin-left:15px;display:flex;"
                             f"gap:8px;align-items:center;'>{''.join(parts_b)}</div>")
                    render_ticket(ev["main"], joker=ev["t_joker"], superstar=ev["t_ss"],
                                  sans_topu=ev["t_st"],
                                  drawn_numbers=drawn,
                                  drawn_joker=d_joker, drawn_superstar=d_ss,
                                  drawn_sans_topu=d_st,
                                  badge_html=badge)
            else:
                st.info("Sistemde karşılaştırılacak kayıtlı kuponunuz bulunmuyor.")
        except ScrapeFailedError as e:
            st.error(f"❌ Sonuç çekilemedi: {e}")
