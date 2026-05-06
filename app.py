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
import auto_tracker

APP_VERSION = "v2.4.0"
APP_BUILD_DATE = "2026-05-06"

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
# Cache anahtarlarına APP_VERSION ekleyerek her deploy'da otomatik invalidation.
@st.cache_data(show_spinner=False)
def get_data(_version: str = APP_VERSION):
    return load_data()

@st.cache_resource(show_spinner=False)
def get_engine_and_predictor(_df_hash: int, _version: str = APP_VERSION):
    df = get_data(_version)
    return MathEngine(df), Predictor(df)


try:
    with st.spinner("Gerçek çekiliş veritabanı yükleniyor / güncelleniyor..."):
        df = get_data(APP_VERSION)
except ScrapeFailedError as e:
    st.error(
        "❌ Sayısal Loto verisi çekilemedi. İnternet bağlantınızı veya kaynak siteyi kontrol edin.\n\n"
        f"Hata: {e}"
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Veri yükleme hatası: {e}")
    st.stop()

engine, predictor = get_engine_and_predictor(len(df), APP_VERSION)


# --- AUTO TRACKER: Her oturumda 1 kez sessizce kontrol et ---
@st.cache_data(ttl=3600, show_spinner=False)
def auto_tracker_check(_version: str):
    """1 saatte 1 kez otomatik takip kontrolü (cache_data ile rate-limit)."""
    return auto_tracker.check_now()


_auto_tracker_settings = auto_tracker.load_settings()
if _auto_tracker_settings.get("enabled"):
    try:
        auto_tracker_check(APP_VERSION)
    except Exception:
        pass  # Sessiz başarısızlık — kullanıcı arayüzü etkilenmesin

# --- HEADER ---
title_col, ver_col = st.columns([5, 1])
with title_col:
    st.title("🔮 Tahminci: Sayısal Loto AI Asistanı")
    st.markdown("Markov + Bayesian + Rasgelelik Testleri + LightGBM ile çoklu-model olasılık analizi.")
with ver_col:
    st.markdown(
        f"<div style='text-align:right;color:#666;font-size:12px;margin-top:25px;'>"
        f"<code style='background:#1a1c23;padding:3px 8px;border-radius:4px;color:#00E676;'>"
        f"{APP_VERSION}</code><br><span style='font-size:10px;'>{APP_BUILD_DATE}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
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
    strategy = st.radio(
        "Strateji",
        ["🎓 Profesör Modu", "Süper Hibrit", "Sıcak Sayılar", "Soğuk Sayılar"],
        help=(
            "Profesör Modu: Markov zincirleri, Bayesian posterior, rasgelelik testleri ve "
            "LightGBM ML ensemble'ın ağırlıklı birleşimi. Üretilen kuponlar yüksek olasılıklı "
            "ve istatistiksel filtrelerden geçer; her kupon için güven skoru hesaplanır."
        ),
    )
    strategy_clean = strategy.replace("🎓 ", "")
    num_tickets = st.slider("Üretilecek Kolon Sayısı", 1, 10, 5)

    st.divider()
    if st.button("🔄 ML Modelini Yeniden Eğit", width="stretch"):
        with st.spinner("LightGBM yeniden eğitiliyor..."):
            predictor.get_probabilities(force_train=True)
        st.success("Model güncellendi!")

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


def render_ticket(numbers, joker=None, superstar=None,
                  drawn_numbers=None, drawn_joker=None, drawn_superstar=None,
                  badge_html=""):
    parts = []
    for n in numbers:
        cls = "ball-match" if drawn_numbers and n in drawn_numbers else "ball"
        parts.append(f"<div class='{cls}'>{n:02d}</div>")

    if joker is not None:
        parts.append("<div class='ticket-divider'></div>")
        parts.append("<span class='bonus-label'>JOKER</span>")
        joker_cls = "ball-match" if drawn_joker is not None and joker == drawn_joker else "ball-joker"
        parts.append(f"<div class='{joker_cls}'>{joker:02d}</div>")

    if superstar is not None:
        parts.append("<span class='bonus-label'>SÜPER STAR</span>")
        ss_cls = ("ball-match" if drawn_superstar is not None and superstar == drawn_superstar
                  else "ball-superstar")
        parts.append(f"<div class='{ss_cls}'>{superstar:02d}</div>")

    st.markdown(f"<div class='ticket-box'>{''.join(parts)}{badge_html}</div>",
                unsafe_allow_html=True)


# --- Notification: yeni otomatik takip sonuçları ---
_unread = auto_tracker.get_unread_count() if _auto_tracker_settings.get("enabled") else 0
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

    if st.button("🚀 KUPON ÜRET", width="stretch", type="primary"):
        with st.spinner("Olasılık motoru çalışıyor, filtreler uygulanıyor..."):
            tickets, attempts = predictor.generate_tickets(
                num_tickets=num_tickets, strategy=strategy_clean
            )
        st.session_state.current_tickets = tickets
        st.success(f"✅ {len(tickets)} kupon üretildi ({attempts} varyasyon denendi).")

    if st.session_state.current_tickets:
        st.markdown("### 🎲 Üretilen Kolonlar (6 Ana + Joker + Süper Star)")
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
                badge_html=badge,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 KUPONLARI SİSTEME KAYDET", width="stretch"):
            next_draw = int(df["cekilis_no"].max()) + 1
            save_tickets(st.session_state.current_tickets, strategy_clean,
                         valid_from_draw_no=next_draw)
            st.toast(f"Kuponlar kaydedildi! Çekiliş #{next_draw}'dan itibaren otomatik takip edilir.")
            st.session_state.current_tickets = []
            st.rerun()

    if st.session_state.current_tickets:
        st.divider()
        st.markdown("### 💰 Bu Kupon Tutarsa Ne Kazanırsın?")
        st.caption(
            "Sayısal Loto resmi ikramiye yapısına göre, bir kuponun tutturduğu sayıya göre "
            "alabileceği ödüller. Jackpot çekilişten çekilişe değişir; aşağıdaki değerleri "
            "güncelleyebilirsin."
        )

        with st.expander("⚙️ İkramiye Değerlerini Düzenle", expanded=False):
            pc1, pc2, pc3, pc4 = st.columns(4)
            prize6 = pc1.number_input("6 bilen (Jackpot)", min_value=1_000_000,
                                      max_value=10_000_000_000,
                                      value=DEFAULT_PRIZES_TL[6], step=10_000_000,
                                      key="p6")
            prize5 = pc2.number_input("5 bilen", min_value=1_000,
                                      max_value=5_000_000,
                                      value=DEFAULT_PRIZES_TL[5], step=10_000,
                                      key="p5")
            prize4 = pc3.number_input("4 bilen", min_value=10,
                                      max_value=100_000,
                                      value=DEFAULT_PRIZES_TL[4], step=50,
                                      key="p4")
            prize3 = pc4.number_input("3 bilen", min_value=1,
                                      max_value=1_000,
                                      value=DEFAULT_PRIZES_TL[3], step=5,
                                      key="p3")

        n_tickets = len(st.session_state.current_tickets)
        total_cost = n_tickets * DEFAULT_COST_PER_TICKET_TL

        def fmt_tl(v: int) -> str:
            return f"{v:,.0f} TL".replace(",", ".")

        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("3 Bilirsen", fmt_tl(prize3),
                   help="Bir kupondan 3 sayı tutarsa kazanılan tahmini ödül")
        pc2.metric("4 Bilirsen", fmt_tl(prize4))
        pc3.metric("5 Bilirsen", fmt_tl(prize5))
        pc4.metric("6 Bilirsen 🎰", fmt_tl(prize6))

        st.info(
            f"💸 Toplam Maliyet: **{n_tickets} kolon × {DEFAULT_COST_PER_TICKET_TL} TL "
            f"= {fmt_tl(n_tickets * DEFAULT_COST_PER_TICKET_TL)}** "
            f"(Joker / Süper Star eklersen ekstra ücret yansır)"
        )

# TAB 2: TICKET CHECKER
with tab2:
    st.subheader("🎟️ Kayıtlı Kuponlar ve Sonuç Sorgulama")
    saved = load_saved_tickets()
    if not saved:
        st.info("Sistemde henüz oynanmış bir kuponunuz bulunmuyor.")
    else:
        with st.expander("🔍 Sonuç Kontrol Paneli", expanded=True):
            draw_input = st.text_input(
                "6 Ana Sayıyı Girin (Örn: 5, 12, 34, 56, 78, 89)",
                placeholder="Sayıları virgülle ayırarak giriniz...",
            )
            jc1, jc2 = st.columns(2)
            joker_input = jc1.number_input("Joker (1-90)", min_value=0, max_value=90,
                                           value=0, step=1,
                                           help="0 = girilmedi")
            ss_input = jc2.number_input("Süper Star (1-90)", min_value=0, max_value=90,
                                        value=0, step=1,
                                        help="0 = girilmedi")
            drawn_numbers = []
            if draw_input:
                try:
                    drawn_numbers = [int(x.strip()) for x in draw_input.split(",")]
                    if len(drawn_numbers) != 6:
                        st.error("Lütfen tam 6 adet sayı girin!")
                        drawn_numbers = []
                except ValueError:
                    st.error("Lütfen geçerli sayılar girin!")
            drawn_joker = joker_input if joker_input > 0 else None
            drawn_ss = ss_input if ss_input > 0 else None

        st.divider()
        col1, col2 = st.columns([4, 1])
        col1.markdown("### 📋 Oynanan Kuponlar")
        with col2:
            if st.button("🗑️ Tümünü Sil", width="stretch"):
                delete_all_tickets()
                st.rerun()

        for record in reversed(saved):
            st.caption(f"📅 {record['tarih']} | 🧠 Strateji: {record['strateji']}")
            for ticket in record["kuponlar"]:
                # Backward compat: legacy = list[int], new = dict
                if isinstance(ticket, dict):
                    main = list(ticket["main"]) if "main" in ticket else list(ticket.get("kuponlar", []))
                    t_joker = ticket.get("joker")
                    t_ss = ticket.get("superstar")
                else:
                    main = list(ticket)
                    t_joker = None
                    t_ss = None

                match_count = len(set(main).intersection(drawn_numbers)) if drawn_numbers else 0
                joker_hit = t_joker is not None and drawn_joker is not None and t_joker == drawn_joker
                ss_hit = t_ss is not None and drawn_ss is not None and t_ss == drawn_ss
                badge = ""
                if drawn_numbers:
                    parts_b = []
                    if match_count >= 3:
                        parts_b.append(f"<span style='color:#111;background:#00E676;padding:3px 10px;"
                                       f"border-radius:6px;font-size:14px;'>{match_count} ANA 🏆</span>")
                    elif match_count > 0:
                        parts_b.append(f"<span style='color:#ff4444;font-size:14px;'>{match_count} ana</span>")
                    if joker_hit:
                        parts_b.append("<span style='color:#FF9800;font-size:14px;'>+JOKER ⭐</span>")
                    if ss_hit:
                        parts_b.append("<span style='color:#BB86FC;font-size:14px;'>+SÜPER STAR ⭐</span>")
                    if parts_b:
                        badge = (f"<div style='align-self:center;margin-left:15px;display:flex;"
                                 f"gap:8px;'>{''.join(parts_b)}</div>")
                render_ticket(main, joker=t_joker, superstar=t_ss,
                              drawn_numbers=drawn_numbers,
                              drawn_joker=drawn_joker, drawn_superstar=drawn_ss,
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
        "sayi": np.arange(1, 91),
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

    settings = auto_tracker.load_settings()
    is_enabled = settings.get("enabled", False)

    cc1, cc2 = st.columns([2, 1])
    with cc1:
        new_state = st.toggle(
            "🟢 Otomatik Takip" if is_enabled else "⚪ Otomatik Takip",
            value=is_enabled,
            help=("Açıkken: app her yüklendiğinde (en fazla saatte 1) son "
                  "çekilişler kontrol edilir, kuponlar otomatik değerlendirilir. "
                  "Kapalıyken: hiçbir otomatik işlem yapılmaz."),
        )
        if new_state != is_enabled:
            auto_tracker.set_enabled(new_state)
            st.cache_data.clear()
            st.rerun()
    with cc2:
        if st.button("🔍 Şimdi Kontrol Et", width="stretch",
                     disabled=not is_enabled):
            with st.spinner("Yeni çekilişler kontrol ediliyor..."):
                st.cache_data.clear()
                result = auto_tracker.check_now()
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

    tracking = auto_tracker.load_tracking()
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

        if st.button("✅ Hepsini Okundu İşaretle", width="stretch"):
            auto_tracker.mark_all_seen()
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
                st.caption("Çekilen Sayılar:")
                render_ticket(drawn, joker=d_joker, superstar=d_ss,
                              drawn_numbers=drawn,
                              drawn_joker=d_joker, drawn_superstar=d_ss)

                if not wins:
                    st.write("Bu çekilişte kayıtlı kuponlardan hiçbiri 3+ tutturamadı, "
                             "Joker veya Süper Star yakalayamadı.")
                else:
                    st.markdown("**Tutturan Kuponlar:**")
                    for w in wins:
                        prize = w.get("estimated_prize_tl", 0)
                        parts = []
                        if w["main_hits"] >= 3:
                            parts.append(f"<span style='color:#111;background:#00E676;"
                                         f"padding:2px 8px;border-radius:5px;font-size:13px;'>"
                                         f"{w['main_hits']} ana 🏆</span>")
                        if w.get("joker_hit"):
                            parts.append("<span style='color:#FF9800;font-size:13px;'>+JOKER</span>")
                        if w.get("superstar_hit"):
                            parts.append("<span style='color:#BB86FC;font-size:13px;'>+SÜPER STAR</span>")
                        if prize > 0:
                            parts.append(f"<span style='color:#00E676;font-weight:600;"
                                         f"font-size:14px;'>≈ {prize:,.0f} TL</span>".replace(",", "."))
                        badge = (f"<div style='align-self:center;margin-left:15px;display:flex;"
                                 f"gap:8px;flex-wrap:wrap;'>{''.join(parts)}</div>")
                        render_ticket(w["main"],
                                      joker=w.get("joker"),
                                      superstar=w.get("superstar"),
                                      drawn_numbers=drawn,
                                      drawn_joker=d_joker, drawn_superstar=d_ss,
                                      badge_html=badge)
                        st.caption(f"📅 Kaydedildi: {w.get('record_tarih', '?')}")

    st.divider()
    st.markdown("### 🌐 Manuel Anlık Sonuç (Hızlı Bakış)")
    if st.button("Güncel Sonucu Getir", width="stretch", type="primary"):
        try:
            with st.spinner("Son çekiliş kaynaktan çekiliyor..."):
                latest = fetch_latest_draw()
            date_str = latest["tarih"].strftime("%d-%m-%Y")
            drawn = latest["sayilar"]
            d_joker = latest.get("joker")
            d_ss = latest.get("superstar")
            st.success(f"✅ Çekiliş #{latest['cekilis_no']} — {date_str}")

            st.markdown("### 🎲 Çekiliş Sonucu")
            render_ticket(drawn, joker=d_joker, superstar=d_ss,
                          drawn_numbers=drawn,
                          drawn_joker=d_joker, drawn_superstar=d_ss)

            saved = load_saved_tickets()
            if saved:
                st.markdown("### 📊 Sizin Kuponlarınızdaki Durum")
                for record in reversed(saved):
                    for ticket in record["kuponlar"]:
                        if isinstance(ticket, dict):
                            main = list(ticket["main"]) if "main" in ticket else list(ticket.get("kuponlar", []))
                            t_joker = ticket.get("joker")
                            t_ss = ticket.get("superstar")
                        else:
                            main = list(ticket)
                            t_joker = None
                            t_ss = None
                        m = len(set(main).intersection(drawn))
                        joker_hit = t_joker is not None and t_joker == d_joker
                        ss_hit = t_ss is not None and t_ss == d_ss
                        parts_b = []
                        if m >= 3:
                            parts_b.append(f"<span style='color:#111;background:#00E676;padding:3px 10px;"
                                           f"border-radius:6px;font-size:14px;'>{m} ANA 🏆</span>")
                        elif m > 0:
                            parts_b.append(f"<span style='color:#ff4444;font-size:14px;'>{m} ana</span>")
                        if joker_hit:
                            parts_b.append("<span style='color:#FF9800;font-size:14px;'>+JOKER ⭐</span>")
                        if ss_hit:
                            parts_b.append("<span style='color:#BB86FC;font-size:14px;'>+SÜPER STAR ⭐</span>")
                        badge = ""
                        if parts_b:
                            badge = (f"<div style='align-self:center;margin-left:15px;display:flex;"
                                     f"gap:8px;'>{''.join(parts_b)}</div>")
                        render_ticket(main, joker=t_joker, superstar=t_ss,
                                      drawn_numbers=drawn,
                                      drawn_joker=d_joker, drawn_superstar=d_ss,
                                      badge_html=badge)
            else:
                st.info("Sistemde karşılaştırılacak kayıtlı kuponunuz bulunmuyor.")
        except ScrapeFailedError as e:
            st.error(f"❌ Sonuç çekilemedi: {e}")
