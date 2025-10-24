import streamlit as st
import pandas as pd
import numpy as np
import locale


def setup_page():
    st.set_page_config(
        page_title="Prediksi Kualitas Internet dengan Eksogen",
        page_icon="📶",
    )
    locale.setlocale(locale.LC_TIME, 'id_ID')

def display_header():
    st.title('📶 Prediksi Kualitas Internet (QoS) - SARIMAX dengan Variabel Eksogen')

def display_data_preview(df):
    st.markdown("### 👀 Preview Data dengan Variabel Eksogen:")
    st.dataframe(df, width="stretch")

def display_dataset_info(df):
    st.markdown("### 📊 Informasi Dataset")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("**📈 Metrik QoS Tersedia:**")
        metrik_qos = [col for col in df.columns if col in ['upload', 'download', 'latency', 'packet_loss', 'jitter']]
        for metrik in metrik_qos:
            st.write(f"• {metrik.title()}")
    with col_info2:
        st.markdown("**🎯 Variabel Eksogen Tersedia:**")
        eksogen_vars = [col for col in df.columns if col in ['hari_encoded', 'jam', 'sit_person']]
        for var in eksogen_vars:
            if var == 'hari_encoded': st.write("• Hari")
            elif var == 'jam': st.write("• Jam")
            elif var == 'sit_person': st.write("• Sit Person")
    st.divider()

def get_prediction_parameters(df):
    st.markdown("### ⚙️ Konfigurasi Prediksi")
    kolom_kecepatan = [col for col in df.columns if col in ['upload', 'download']]
    metrik_terpilih = 'download' if 'download' in kolom_kecepatan else (kolom_kecepatan[0] if kolom_kecepatan else None)
    hari_prediksi = st.number_input("🗓️ Berapa hari ke depan ingin diprediksi?", min_value=1, max_value=30, value=2)
    jam_prediksi_terpilih = [9, 13, 17] # Default prediction times
    mulai_prediksi = st.button("🚀 Mulai Prediksi SARIMAX", type="primary", width="stretch")
    if not mulai_prediksi:
        st.info("👆 Klik tombol di atas untuk memulai proses prediksi.")
        st.stop()
    return metrik_terpilih, hari_prediksi, jam_prediksi_terpilih

def display_prediction_results(metrik, prediksi, akurasi):
    nilai_tertinggi = np.max(prediksi)
    nilai_terendah = np.min(prediksi)
    rata_rata_prediksi = np.mean(prediksi)

    col_insight1, col_insight2, col_insight3, col_insight4 = st.columns(4)
    with col_insight1:
        st.metric("📈 Nilai Tertinggi", f"{nilai_tertinggi:.2f}")
    with col_insight2:
        st.metric("📉 Nilai Terendah", f"{nilai_terendah:.2f}")
    with col_insight3:
        st.metric("📊 Rata-rata", f"{rata_rata_prediksi:.2f}")
    with col_insight4:
        st.metric("🎯 Akurasi (MAPE)", f"{akurasi:.1f}%")

def display_full_prediction_table(dataframe_forcast, semua_prediksi, semua_eksogen, metrik_terpilih):
    st.markdown("### 📋 Tabel Lengkap Prediksi dengan Eksogen")
    tabel_prediksi = pd.DataFrame(index=dataframe_forcast)
    tabel_prediksi['📅 Tanggal'] = tabel_prediksi.index.date
    tabel_prediksi['🕐 Jam'] = tabel_prediksi.index.strftime('%H:%M')
    tabel_prediksi['📆 Hari'] = tabel_prediksi.index.strftime('%A')

    if metrik_terpilih in semua_eksogen and not semua_eksogen[metrik_terpilih].empty:
        eksogen_tampil = semua_eksogen[metrik_terpilih]
        if 'jam' in eksogen_tampil.columns:
            tabel_prediksi['🕐 Jam'] = eksogen_tampil['jam'].round(1)

    for metrik, prediksi in semua_prediksi.items():
        if metrik == 'upload': nama_tampilan = '⬆️ Upload (Mbps)'
        elif metrik == 'download': nama_tampilan = '⬇️ Download (Mbps)'
        elif metrik == 'latency': nama_tampilan = '⏱️ Latency (ms)'
        elif metrik == 'packet_loss': nama_tampilan = '📦 Packet Loss (%)'
        elif metrik == 'jitter': nama_tampilan = '📶 Jitter (ms)'
        else: nama_tampilan = metrik.title()
        tabel_prediksi[nama_tampilan] = np.round(prediksi, 2)

    tabel_prediksi.reset_index(drop=True, inplace=True)
    st.dataframe(tabel_prediksi, width="stretch")

def display_model_summary(skor_akurasi):
    st.markdown("### 🎯 Ringkasan Model SARIMAX dengan Eksogen")
    col_akurasi, _ = st.columns(2)
    with col_akurasi:
        st.markdown("#### 📊 Tingkat Akurasi Model")
        tabel_akurasi = pd.DataFrame([
            {
                '📊 Metrik': metrik.title(),
                '🎯 MAPE (%)': f"{akurasi:.1f}%",
                '📈 Kualitas': (
                    "Excellent" if akurasi < 10 else
                    "Very Good" if akurasi < 20 else
                    "Good" if akurasi < 35 else
                    "Fair" if akurasi < 50 else "Poor"
                ),
            }
            for metrik, akurasi in skor_akurasi.items()
        ])
        st.dataframe(tabel_akurasi, hide_index=True, width="stretch")

    rata_rata_akurasi = np.mean(list(skor_akurasi.values()))
    if rata_rata_akurasi < 20:
        st.success(f"🎉 **Model Sangat Akurat!** SARIMAX dengan variabel eksogen berhasil mencapai akurasi tinggi dengan rata-rata error {rata_rata_akurasi:.1f}%. Prediksi dapat diandalkan untuk perencanaan.")
    elif rata_rata_akurasi < 35:
        st.success(f"✅ **Model Cukup Reliable!** Dengan error rata-rata {rata_rata_akurasi:.1f}%, model menunjukkan performa baik dan dapat digunakan untuk estimasi.")
    elif rata_rata_akurasi < 50:
        st.warning(f"⚠️ **Model Memadai** dengan error {rata_rata_akurasi:.1f}%. Gunakan dengan hati-hati dan pertimbangkan untuk menambah data historis.")
    else:
        st.warning(f"⚠️ **Model Perlu Improvement** (error {rata_rata_akurasi:.1f}%). Pertimbangkan untuk menambah variabel eksogen atau memperbanyak data historis.")
