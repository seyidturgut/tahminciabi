import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from data_manager import load_data
from scraper import ScrapeFailedError, fetch_latest_draw
from math_engine import MathEngine
from predictor import Predictor
from ticket_manager import save_tickets, load_saved_tickets, delete_all_tickets
from analytics.expected_value import budget_summary, DEFAULT_PRIZES_TL, DEFAULT_COST_PER_TICKET_TL

APP_VERSION = "v2.1.0"
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
@st.cache_data(show_spinner=False)
def get_data():
    return load_data()

@st.cache_resource(show_spinner=False)
def get_engine_and_predictor(_df_hash):
    df = get_data()
    return MathEngine(df), Predictor(df)


try:
    with st.spinner("Gerçek çekiliş veritabanı yükleniyor / güncelleniyor..."):
        df = get_data()
except ScrapeFailedError as e:
    st.error(
        "❌ Sayısal Loto verisi çekilemedi. İnternet bağlantınızı veya kaynak siteyi kontrol edin.\n\n"
        f"Hata: {e}"
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Veri yükleme hatası: {e}")
    st.stop()

engine, predictor = get_engine_and_predictor(len(df))

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


def render_ticket(numbers, drawn_numbers=None, badge_html=""):
    parts = []
    for n in numbers:
        cls = "ball-match" if drawn_numbers and n in drawn_numbers else "ball"
        parts.append(f"<div class='{cls}'>{n:02d}</div>")
    st.markdown(f"<div class='ticket-box'>{''.join(parts)}{badge_html}</div>",
                unsafe_allow_html=True)


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
        st.session_state.current_confidences = []

    if st.button("🚀 KUPON ÜRET", width="stretch", type="primary"):
        with st.spinner("Olasılık motoru çalışıyor, filtreler uygulanıyor..."):
            tickets, attempts, confidences = predictor.generate_tickets(
                num_tickets=num_tickets, strategy=strategy_clean
            )
        st.session_state.current_tickets = tickets
        st.session_state.current_confidences = confidences
        st.success(f"✅ {len(tickets)} kupon üretildi ({attempts} varyasyon denendi).")

    if st.session_state.current_tickets:
        st.markdown("### 🎲 Üretilen Kolonlar")
        for ticket, conf in zip(st.session_state.current_tickets,
                                st.session_state.current_confidences):
            badge = ""
            if conf > 0:
                color = "#00E676" if conf >= 0.55 else ("#ffd54f" if conf >= 0.5 else "#ff7043")
                badge = (f"<div style='color:{color};font-size:14px;align-self:center;"
                         f"margin-left:15px;'>Güven: %{conf*100:.1f}</div>")
            render_ticket(list(ticket), badge_html=badge)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 KUPONLARI SİSTEME KAYDET", width="stretch"):
            save_tickets(st.session_state.current_tickets, strategy_clean)
            st.toast("Kuponlar kaydedildi!")
            st.session_state.current_tickets = []
            st.session_state.current_confidences = []
            st.rerun()

    st.divider()
    st.markdown("### 💰 Gerçekçi Kazanç Hesabı")
    st.caption(
        "Hipergeometrik 6/90 olasılığı + Türkiye Sayısal Loto güncel ortalama "
        "ikramiye değerleriyle hesaplanır. Jackpot tutarları çekilişe göre değişir; "
        "aşağıdaki değerleri kendi tahminlerinle güncelleyebilirsin."
    )

    bc1, bc2 = st.columns(2)
    budget = bc1.number_input("Bütçe (TL)", min_value=20, max_value=100000,
                              value=100, step=20)
    cost = bc2.number_input("1 Kolon Ücreti (TL)", min_value=1, max_value=200,
                            value=DEFAULT_COST_PER_TICKET_TL, step=1)

    with st.expander("⚙️ İkramiye Tablosu (özelleştirilebilir)", expanded=False):
        pc1, pc2, pc3, pc4 = st.columns(4)
        prize6 = pc1.number_input("6 bilen (TL)", min_value=1_000_000,
                                  max_value=10_000_000_000,
                                  value=DEFAULT_PRIZES_TL[6], step=10_000_000)
        prize5 = pc2.number_input("5 bilen (TL)", min_value=1_000,
                                  max_value=5_000_000,
                                  value=DEFAULT_PRIZES_TL[5], step=10_000)
        prize4 = pc3.number_input("4 bilen (TL)", min_value=10,
                                  max_value=100_000,
                                  value=DEFAULT_PRIZES_TL[4], step=50)
        prize3 = pc4.number_input("3 bilen (TL)", min_value=1,
                                  max_value=1_000,
                                  value=DEFAULT_PRIZES_TL[3], step=5)

    custom_prizes = {6: prize6, 5: prize5, 4: prize4, 3: prize3}
    summary = budget_summary(budget_tl=budget, cost_per_ticket=cost,
                             prizes=custom_prizes)

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Oynanacak Kolon", summary["num_tickets"],
               help=f"{budget} TL ÷ {cost} TL = {summary['num_tickets']} kolon")
    sc2.metric("Beklenen Dönüş",
               f"{summary['expected_return_tl']:.2f} TL",
               help="Tüm tutturma seviyelerinin olasılık-ağırlıklı ortalaması")
    sc3.metric("Beklenen Net Zarar",
               f"-{summary['expected_loss_tl']:.2f} TL",
               help="Maliyet − beklenen dönüş. Uzun vade ortalaması.")
    sc4.metric("RTP (Geri Dönüş %)",
               f"%{summary['rtp_pct']:.2f}",
               help="100 TL'nin uzun vadede ortalama ne kadarı geri döner")

    st.markdown("#### 🎯 Tutturma Olasılıkları (bu bütçe için)")
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("3+ Tutturma",
               f"%{summary['p_3_plus_pct']:.2f}",
               help=f"Yaklaşık 1/{summary['p_3_plus_one_in']} bütçeyle bir kez")
    pc2.metric("4+ Tutturma",
               f"%{summary['p_4_plus_pct']:.4f}")
    pc3.metric("5+ Tutturma",
               f"%{summary['p_5_plus_pct']:.5f}")
    pc4.metric("Jackpot (6)",
               f"1 / {summary['p_jackpot_one_in']:,}".replace(",", "."))

    if summary["expected_loss_tl"] > 0:
        st.warning(
            f"⚠️ Bu bütçe için **uzun vade ortalama beklenen zarar "
            f"{summary['expected_loss_tl']:.0f} TL**. Sayısal Loto matematiksel olarak "
            f"negatif beklenen değerli bir oyundur — sadece eğlence amaçlı oynayın. "
            f"Tek seferde dalgalanma çok büyüktür: %{summary['p_3_plus_pct']:.1f} "
            f"olasılıkla kâra geçebilir, ama %{100 - summary['p_3_plus_pct']:.1f} "
            f"olasılıkla bütçeyi kaybedersiniz."
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
                "Çekiliş Sonucunu Girin (Örn: 5, 12, 34, 56, 78, 89)",
                placeholder="Sayıları virgülle ayırarak giriniz...",
            )
            drawn_numbers = []
            if draw_input:
                try:
                    drawn_numbers = [int(x.strip()) for x in draw_input.split(",")]
                    if len(drawn_numbers) != 6:
                        st.error("Lütfen tam 6 adet sayı girin!")
                        drawn_numbers = []
                except ValueError:
                    st.error("Lütfen geçerli sayılar girin!")

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
                match_count = len(set(ticket).intersection(drawn_numbers)) if drawn_numbers else 0
                badge = ""
                if drawn_numbers:
                    if match_count >= 3:
                        badge = (f"<div style='color:#111;background:#00E676;padding:5px 15px;"
                                 f"border-radius:8px;font-size:16px;align-self:center;"
                                 f"margin-left:15px;'>{match_count} BİLDİNİZ! 🏆</div>")
                    else:
                        badge = (f"<div style='color:#ff4444;font-size:16px;align-self:center;"
                                 f"margin-left:15px;'>{match_count} Bildiniz</div>")
                render_ticket(list(ticket), drawn_numbers=drawn_numbers, badge_html=badge)
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
    st.subheader("🤖 Canlı Sonuç Asistanı")
    st.markdown("Tek tıkla en güncel Sayısal Loto sonucunu çekip kayıtlı kuponlarınızla karşılaştırır.")

    if st.button("🌐 Güncel Sonucu Getir", width="stretch", type="primary"):
        try:
            with st.spinner("Son çekiliş kaynaktan çekiliyor..."):
                latest = fetch_latest_draw()
            date_str = latest["tarih"].strftime("%d-%m-%Y")
            drawn = latest["sayilar"]
            st.success(f"✅ Çekiliş #{latest['cekilis_no']} — {date_str}")

            st.markdown("### 🎲 Çekiliş Sonucu")
            render_ticket(drawn, drawn_numbers=drawn)

            saved = load_saved_tickets()
            if saved:
                st.markdown("### 📊 Sizin Kuponlarınızdaki Durum")
                for record in reversed(saved):
                    for ticket in record["kuponlar"]:
                        m = len(set(ticket).intersection(drawn))
                        if m >= 3:
                            badge = (f"<div style='color:#111;background:#00E676;padding:5px 15px;"
                                     f"border-radius:8px;font-size:16px;align-self:center;"
                                     f"margin-left:15px;'>{m} BİLDİNİZ! 🏆</div>")
                        else:
                            badge = (f"<div style='color:#ff4444;font-size:16px;align-self:center;"
                                     f"margin-left:15px;'>{m} Bildiniz</div>")
                        render_ticket(list(ticket), drawn_numbers=drawn, badge_html=badge)
            else:
                st.info("Sistemde karşılaştırılacak kayıtlı kuponunuz bulunmuyor.")
        except ScrapeFailedError as e:
            st.error(f"❌ Sonuç çekilemedi: {e}")
