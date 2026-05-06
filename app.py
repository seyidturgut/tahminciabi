import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from data_manager import load_data
from math_engine import MathEngine
from predictor import Predictor
from ticket_manager import save_tickets, load_saved_tickets, delete_all_tickets

st.set_page_config(page_title="Tahminci | Loto AI", layout="wide", page_icon="🔮")

# Premium Dark Mode CSS
st.markdown("""
<style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Hide Streamlit Header and Menu */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sleek Ticket Boxes */
    .ticket-box {
        background: linear-gradient(145deg, #1e1e1e, #2a2a2a);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        display: flex;
        justify-content: space-around;
        font-size: 26px;
        font-weight: 800;
        color: #00E676;
        border: 1px solid #333;
        box-shadow: 0 4px 15px rgba(0, 230, 118, 0.1);
        transition: transform 0.2s ease;
    }
    .ticket-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 230, 118, 0.2);
    }
    
    /* Ball Styling */
    .ball {
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #252525;
        border-radius: 50%;
        border: 2px solid #00E676;
        text-shadow: 0 0 10px rgba(0, 230, 118, 0.5);
    }
    
    /* Matched Ball Styling */
    .ball-match {
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #00E676;
        color: #121212 !important;
        border-radius: 50%;
        border: 2px solid #00E676;
        box-shadow: 0 0 15px #00E676;
        text-shadow: none;
    }

    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1a1c23;
        border: 1px solid #2e303e;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOGIN SCREEN ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>🔮 Tahminci AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888;'>Sisteme erişmek için şifrenizi giriniz.</p>", unsafe_allow_html=True)
        
        password = st.text_input("Şifre", type="password", label_visibility="collapsed", placeholder="Şifreniz...")
        
        if st.button("Sisteme Giriş Yap", use_container_width=True, type="primary"):
            if password == "Beyincik**94":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Hatalı Şifre!")
    st.stop()

# --- MAIN APP ---
st.title("🔮 Tahminci: Yapay Zeka Destekli Sayısal Loto Asistanı")
st.markdown("İstatistiksel modeller ve Monte Carlo simülasyonları ile optimize edilmiş kuponlar üretin.")
st.divider()

@st.cache_data
def get_data():
    return load_data()

df = get_data()
engine = MathEngine(df)
predictor = Predictor(df)

# Sidebar
with st.sidebar:
    st.header("📊 Veri Merkezi")
    col1, col2 = st.columns(2)
    col1.metric("Toplam Çekiliş", len(df))
    col2.metric("Son Tarih", df.iloc[-1]['tarih'].strftime('%d-%m-%Y'))
    
    st.divider()
    st.header("⚙️ Motor Ayarları")
    strategy = st.radio("Yapay Zeka Stratejisi", 
                       ["Dengeli", "Sıcak Sayılar", "Soğuk Sayılar"],
                       help="Kuponların hangi matematiksel ağırlıkla üretileceğini seçin.")
    num_tickets = st.slider("Üretilecek Kolon Sayısı", 1, 10, 5)
    
    st.divider()
    if st.button("🔄 Geçmiş Verileri Yenile", use_container_width=True):
        import data_manager
        data_manager.generate_mock_data()
        st.cache_data.clear()
        st.rerun()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Tahmin Üretici", "🎟️ Kupon & Sonuç Kontrolü", "📈 Derin İstatistikler", "🤖 Otomasyon & Ayarlar"])

# TAB 1: GENERATOR
with tab1:
    st.subheader(f"🧠 Monte Carlo Motoru (Strateji: {strategy})")
    st.caption("Matematiksel olarak gerçekleşme ihtimali düşük olan dizilimler filtrelenerek optimum olasılıklı kolonlar sunulur.")
    
    if "current_tickets" not in st.session_state:
        st.session_state.current_tickets = []
        
    if st.button("🚀 KUPON ÜRET", use_container_width=True, type="primary"):
        with st.spinner("Simülasyon çalışıyor, filtreler uygulanıyor..."):
            tickets, attempts = predictor.generate_tickets(num_tickets=num_tickets, strategy=strategy)
            st.session_state.current_tickets = tickets
            st.success(f"✅ Optimizasyon tamamlandı! Toplam {attempts} varyasyon denendi.")
            
    if st.session_state.current_tickets:
        st.markdown("### 🎲 Üretilen Kolonlar")
        for i, ticket in enumerate(st.session_state.current_tickets):
            formatted_ticket = "".join([f"<div class='ball'>{num:02d}</div>" for num in ticket])
            st.markdown(f"<div class='ticket-box'>{formatted_ticket}</div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 SEÇİLİ KUPONLARI SİSTEME KAYDET", use_container_width=True):
            save_tickets(st.session_state.current_tickets, strategy)
            st.toast("Kuponlar sisteme kaydedildi!")
            st.session_state.current_tickets = []
            st.rerun()

# TAB 2: TICKET CHECKER
with tab2:
    st.subheader("🎟️ Kayıtlı Kuponlar ve Sonuç Sorgulama")
    saved_data = load_saved_tickets()
    
    if not saved_data:
        st.info("Sistemde henüz oynanmış bir kuponunuz bulunmuyor.")
    else:
        with st.expander("🔍 Sonuç Kontrol Paneli", expanded=True):
            draw_input = st.text_input("Son Çekiliş Sonucunu Girin (Örn: 5, 12, 34, 56, 78, 89)", placeholder="Sayıları virgülle ayırarak giriniz...")
            
            draw_numbers = []
            if draw_input:
                try:
                    draw_numbers = [int(x.strip()) for x in draw_input.split(",")]
                    if len(draw_numbers) != 6:
                        st.error("Lütfen tam 6 adet sayı girin!")
                        draw_numbers = []
                except ValueError:
                    st.error("Lütfen geçerli sayılar girin!")

        st.divider()
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("### 📋 Oynanan Kuponlar")
        with col2:
            if st.button("🗑️ Tümünü Sil", use_container_width=True):
                delete_all_tickets()
                st.rerun()
                
        for record in reversed(saved_data):
            st.caption(f"📅 Oynanma: {record['tarih']} | 🧠 Strateji: {record['strateji']}")
            for ticket in record["kuponlar"]:
                match_count = len(set(ticket).intersection(set(draw_numbers))) if draw_numbers else 0
                
                html_parts = []
                for num in ticket:
                    if draw_numbers and num in draw_numbers:
                        html_parts.append(f"<div class='ball-match'>{num:02d}</div>")
                    else:
                        html_parts.append(f"<div class='ball'>{num:02d}</div>")
                        
                formatted_ticket = "".join(html_parts)
                
                badge = ""
                if draw_numbers:
                    if match_count >= 3:
                        badge = f"<div style='color: #111; background: #00E676; padding: 5px 15px; border-radius: 8px; font-size: 16px; align-self: center;'>{match_count} BİLDİNİZ! 🏆</div>"
                    else:
                        badge = f"<div style='color: #ff4444; font-size: 16px; align-self: center;'>{match_count} Bildiniz</div>"
                
                st.markdown(f"<div class='ticket-box'>{formatted_ticket} {badge}</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

# TAB 3: STATISTICS
with tab3:
    st.subheader("📈 Analitik Merkez")
    freq_df = engine.calculate_frequencies()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔥 En Sıcak Sayılar")
        st.dataframe(freq_df.head(10).style.format({'yuzde': '{:.2f}%'}), use_container_width=True)
        
    with col2:
        st.markdown("#### 🧊 En Soğuk Sayılar")
        st.dataframe(freq_df.tail(10).style.format({'yuzde': '{:.2f}%'}), use_container_width=True)
        
    with col3:
        st.markdown("#### ⏳ Gecikme (Gap) Analizi")
        st.caption("Uzun süredir çıkmayan sayılar")
        gaps_df = engine.calculate_gaps()
        st.dataframe(gaps_df.head(10), use_container_width=True)

    st.divider()
    st.markdown("#### 📊 Toplamların Gauss (Çan Eğrisi) Dağılımı")
    mean_sum, std_sum = engine.get_sum_distribution_stats()
    st.info(f"Kuponlarınızdaki 6 sayının toplamının istatistiksel güven aralığı: **{mean_sum - 1.5*std_sum:.0f} - {mean_sum + 1.5*std_sum:.0f}** arasındadır. Tahmin motoru bu aralığın dışına çıkan kolonları elemektedir.")
    
    with st.expander("Geçmiş Veritabanını Görüntüle"):
        st.dataframe(df.sort_values(by='tarih', ascending=False).head(100), use_container_width=True)

# TAB 4: AUTOMATION
with tab4:
    st.subheader("🤖 Otomasyon ve Canlı Sonuç Asistanı")
    st.markdown("Streamlit Cloud altyapısında çalıştığınız için, arka plan robotu yerine tek tıkla internetteki tüm güncel verileri tarayıp ekrana getiren akıllı asistanı kullanabilirsiniz.")
    
    if st.button("🌐 İnterneti Tara ve Güncel Sonuçları Getir", use_container_width=True, type="primary"):
        with st.spinner("İnternetteki güncel sonuçlar taranıyor..."):
            from scraper import get_live_draw_results
            result = get_live_draw_results()
            current_date = result.get("tarih", "Bilinmiyor")
            drawn_numbers = result.get("sayilar", [])
            
            st.success(f"✅ Sonuçlar başarıyla çekildi! (Tarih: {current_date})")
            
            st.markdown("### 🎲 Güncel Çekiliş Sonucu:")
            formatted_draw = "".join([f"<div class='ball-match'>{num:02d}</div>" for num in drawn_numbers])
            st.markdown(f"<div class='ticket-box'>{formatted_draw}</div>", unsafe_allow_html=True)
            
            # Kayıtlı kuponlarla karşılaştır
            saved_data = load_saved_tickets()
            if saved_data:
                st.markdown("### 📊 Sizin Kuponlarınızdaki Durum:")
                for record in reversed(saved_data):
                    for ticket in record["kuponlar"]:
                        match_count = len(set(ticket).intersection(set(drawn_numbers)))
                        html_parts = []
                        for num in ticket:
                            if num in drawn_numbers:
                                html_parts.append(f"<div class='ball-match'>{num:02d}</div>")
                            else:
                                html_parts.append(f"<div class='ball'>{num:02d}</div>")
                        formatted_ticket = "".join(html_parts)
                        
                        badge = ""
                        if match_count >= 3:
                            badge = f"<div style='color: #111; background: #00E676; padding: 5px 15px; border-radius: 8px; font-size: 16px; align-self: center;'>{match_count} BİLDİNİZ! 🏆</div>"
                        else:
                            badge = f"<div style='color: #ff4444; font-size: 16px; align-self: center;'>{match_count} Bildiniz</div>"
                        
                        st.markdown(f"<div class='ticket-box'>{formatted_ticket} {badge}</div>", unsafe_allow_html=True)
            else:
                st.info("Sistemde karşılaştırılacak kayıtlı kuponunuz bulunmuyor.")
