import streamlit as st
import pandas as pd
import numpy as np
import locale

def setup_page():
    st.set_page_config(
        page_title="Prediksi Kualitas Internet (QoS) - SARIMAX",
        page_icon="📶"
    )
    try:
        locale.setlocale(locale.LC_TIME, 'id_ID.UTF-8')
    except Exception:
        try:
            locale.setlocale(locale.LC_TIME, 'Indonesian_Indonesia.1252')
        except Exception:
            pass

def display_header():
    st.title("Prediksi Kualitas Internet (QoS) - SARIMAX")

def display_data_preview(df):
    st.subheader("Preview Data")
    st.dataframe(df, width='content')

def display_dataset_info(df):
    st.subheader("Informasi Dataset")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("**Metrik QoS:**")
        metrik_qos = [col for col in ['throughput', 'latency', 'jitter', 'packet_loss', 'upload', 'download'] if col in df.columns]
        for metrik in metrik_qos:
            if metrik == 'throughput':
                st.write("• Throughput (Mbps)")
            elif metrik == 'latency':
                st.write("• Latency (ms)")
            elif metrik == 'jitter':
                st.write("• Jitter (ms)")
            elif metrik == 'packet_loss':
                st.write("• Packet Loss (%)")
            else:
                st.write(f"• {metrik.title()}")
    with col_info2:
        st.markdown("**Variabel Eksogen:**")
        eksogen_vars = [col for col in ['hari_encoded', 'jam', 'orang'] if col in df.columns]
        for var in eksogen_vars:
            if var == 'hari_encoded': st.write("• Hari (Pola Mingguan)")
            elif var == 'jam': st.write("• Jam (Pola Harian: 09:00, 12:00, 15:00)")
            elif var == 'orang': st.write("• Jumlah Pengguna (Orang)")
    st.divider()

def get_prediction_parameters(df):
    st.subheader("Konfigurasi Prediksi")
    hari_prediksi = st.number_input("Berapa hari ke depan ingin diprediksi?", min_value=1, max_value=30, value=2)
    jam_prediksi_terpilih = [9, 12, 15]

    mulai_prediksi = st.button("Mulai Prediksi SARIMAX", type="primary", width='content')
    if not mulai_prediksi:
        st.stop()
    return hari_prediksi, jam_prediksi_terpilih

def _satuan(metrik):
    if metrik in ('throughput', 'upload', 'download'):
        return 'Mbps'
    if metrik in ('latency', 'jitter'):
        return 'ms'
    if metrik == 'packet_loss':
        return '%'
    return ''

def display_prediction_results(metrik, prediksi, akurasi, diagnostik=None):
    nilai_tertinggi = np.max(prediksi)
    nilai_terendah = np.min(prediksi)
    rata_rata_prediksi = np.mean(prediksi)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nilai Tertinggi", f"{nilai_tertinggi:.2f}")
    with col2:
        st.metric("Nilai Terendah", f"{nilai_terendah:.2f}")
    with col3:
        st.metric("Rata-rata", f"{rata_rata_prediksi:.2f}")
    with col4:
        st.metric("Error (%)", f"{akurasi:.1f}%")

    if diagnostik:
        satuan = _satuan(metrik)
        mae = diagnostik.get('mae')
        rmse = diagnostik.get('rmse')
        if mae is not None and rmse is not None:
            col_mae, col_rmse = st.columns(2)
            col_mae.metric("MAE (test)", f"{mae:.2f} {satuan}".strip())
            col_rmse.metric("RMSE (test)", f"{rmse:.2f} {satuan}".strip())

def display_full_prediction_table(dataframe_forcast, semua_prediksi, semua_eksogen, metrik_utama=None):
    st.subheader("Tabel Hasil Prediksi")
    tabel_prediksi = pd.DataFrame(index=dataframe_forcast)
    tabel_prediksi['Tanggal'] = tabel_prediksi.index.date
    tabel_prediksi['Jam'] = tabel_prediksi.index.strftime('%H:%M')
    tabel_prediksi['Hari'] = tabel_prediksi.index.strftime('%A')

    kunci_eksogen = metrik_utama if metrik_utama in semua_eksogen else (list(semua_eksogen.keys())[0] if semua_eksogen else None)
    if kunci_eksogen and not semua_eksogen[kunci_eksogen].empty:
        eksogen_tampil = semua_eksogen[kunci_eksogen]
        if 'log_person' in eksogen_tampil.columns:
            tabel_prediksi['Estimasi Orang'] = np.expm1(eksogen_tampil['log_person']).round().astype(int)

    for metrik, prediksi in semua_prediksi.items():
        if metrik == 'throughput': nama_tampilan = 'Throughput (Mbps)'
        elif metrik == 'upload': nama_tampilan = 'Upload (Mbps)'
        elif metrik == 'download': nama_tampilan = 'Download (Mbps)'
        elif metrik == 'latency': nama_tampilan = 'Latency (ms)'
        elif metrik == 'packet_loss': nama_tampilan = 'Packet Loss (%)'
        elif metrik == 'jitter': nama_tampilan = 'Jitter (ms)'
        else: nama_tampilan = metrik.title()
        tabel_prediksi[nama_tampilan] = np.round(prediksi, 2)

    tabel_prediksi.reset_index(drop=True, inplace=True)
    st.dataframe(tabel_prediksi, width='content')

def display_model_summary(skor_akurasi, semua_diagnostik=None):
    """Ringkasan akurasi per model."""
    st.subheader("Ringkasan Akurasi Model")
    diag = semua_diagnostik or {}
    baris = []
    for metrik, akurasi in skor_akurasi.items():
        d = diag.get(metrik) or {}
        baris.append({
            'Metrik': (
                'Throughput' if metrik == 'throughput' else
                'Latency' if metrik == 'latency' else
                'Packet Loss' if metrik == 'packet_loss' else
                'Jitter' if metrik == 'jitter' else
                metrik.title()
            ),
            'Error (%)': f"{akurasi:.1f}%",
            'MAE': f"{d['mae']:.2f} {_satuan(metrik)}".strip() if d.get('mae') is not None else '-',
            'RMSE': f"{d['rmse']:.2f} {_satuan(metrik)}".strip() if d.get('rmse') is not None else '-',
            'Kategori': (
                "Sangat Baik (<10%)" if akurasi < 10 else
                "Baik (10-20%)" if akurasi < 20 else
                "Cukup (20-35%)" if akurasi < 35 else
                "Memadai (35-50%)" if akurasi < 50 else "Kurang (>50%)"
            ),
        })
    st.dataframe(pd.DataFrame(baris), hide_index=True, width='content')

