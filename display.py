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
    hari_prediksi = st.number_input("Berapa hari ke depan ingin diprediksi?", min_value=1, max_value=30, value=3)
    jam_prediksi_terpilih = [9, 12, 15]

    if "sudah_prediksi" not in st.session_state:
        st.session_state["sudah_prediksi"] = False

    if st.button("Mulai Prediksi SARIMAX", type="primary", width='content'):
        st.session_state["sudah_prediksi"] = True

    if not st.session_state["sudah_prediksi"]:
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
    if diagnostik:
        satuan = _satuan(metrik)
        mae = diagnostik.get('mae')
        rmse = diagnostik.get('rmse')
        if mae is not None and rmse is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Error (%)", f"{akurasi:.1f}%")
            with col2:
                st.metric("MAE (test)", f"{mae:.2f} {satuan}".strip())
            with col3:
                st.metric("RMSE (test)", f"{rmse:.2f} {satuan}".strip())
            return

    st.metric("Error (%)", f"{akurasi:.1f}%")

def display_full_prediction_table(dataframe_forcast, semua_prediksi, semua_eksogen, metrik_utama=None):
    st.subheader("Tabel Hasil Prediksi")
    hari_map = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
    tabel_prediksi = pd.DataFrame(index=dataframe_forcast)
    tabel_prediksi['Tanggal'] = [d.strftime('%Y-%m-%d') for d in tabel_prediksi.index]
    tabel_prediksi['Jam'] = tabel_prediksi.index.strftime('%H:%M')
    tabel_prediksi['Hari'] = [hari_map.get(d.dayofweek, d.strftime('%A')) for d in tabel_prediksi.index]

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

def display_comparison_table(dataframe_forcast, semua_prediksi, df_actual):
    """Menampilkan tabel perbandingan Prediksi vs Data Aktual per kolom (bukan expander/collapse)."""
    st.subheader("Tabel Perbandingan Prediksi vs Data Aktual")
    hari_map = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
    tabel_komp = pd.DataFrame(index=dataframe_forcast)
    tabel_komp['Tanggal'] = [d.strftime('%Y-%m-%d') for d in tabel_komp.index]
    tabel_komp['Jam'] = tabel_komp.index.strftime('%H:%M')
    tabel_komp['Hari'] = [hari_map.get(d.dayofweek, d.strftime('%A')) for d in tabel_komp.index]

    for metrik in semua_prediksi.keys():
        lbl = (
            'Throughput (Mbps)' if metrik == 'throughput' else
            'Latency (ms)' if metrik == 'latency' else
            'Jitter (ms)' if metrik == 'jitter' else
            'Packet Loss (%)' if metrik == 'packet_loss' else metrik.title()
        )
        pred_val = np.round(semua_prediksi[metrik], 2)
        tabel_komp[f'{lbl} (Pred)'] = pred_val
        if metrik in df_actual.columns:
            act_series = df_actual[metrik].reindex(dataframe_forcast)
            act_val = np.round(act_series.values, 2)
            tabel_komp[f'{lbl} (Akt)'] = act_val
            tabel_komp[f'Selisih {lbl}'] = np.round(pred_val - act_val, 2)

    tabel_komp.reset_index(drop=True, inplace=True)
    st.dataframe(tabel_komp, width='content')

def display_model_summary(skor_akurasi, semua_diagnostik=None, tipe_evaluasi="aktual"):
    """Ringkasan akurasi per model."""
    if tipe_evaluasi == "aktual":
        st.subheader("Ringkasan Akurasi Prediksi (Evaluasi terhadap Data Aktual)")
    else:
        st.subheader("Ringkasan Akurasi Model (Evaluasi Data Uji)")

    diag = semua_diagnostik or {}
    col_mae = "MAE (vs Aktual)" if tipe_evaluasi == "aktual" else "MAE (test)"
    col_rmse = "RMSE (vs Aktual)" if tipe_evaluasi == "aktual" else "RMSE (test)"
    col_err = "Error Prediksi (%)" if tipe_evaluasi == "aktual" else "Error (%)"

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
            col_err: f"{akurasi:.1f}%",
            col_mae: f"{d['mae']:.2f} {_satuan(metrik)}".strip() if d.get('mae') is not None else '-',
            col_rmse: f"{d['rmse']:.2f} {_satuan(metrik)}".strip() if d.get('rmse') is not None else '-',
            'Kategori': (
                "Sangat Baik (<10%)" if akurasi < 10 else
                "Baik (10-20%)" if akurasi < 20 else
                "Cukup (20-35%)" if akurasi < 35 else
                "Memadai (35-50%)" if akurasi < 50 else "Kurang (>50%)"
            ),
        })
    st.dataframe(pd.DataFrame(baris), hide_index=True, width='content')
