import os
import streamlit as st
import numpy as np
import traceback
from statsmodels.stats.diagnostic import acorr_ljungbox

# Import components
import importlib
import model
import display
importlib.reload(model)
importlib.reload(display)
from pre_processing import load_data, clean_data
from model import SarimaxEksogenPrediktor
from visualization import plot_prediction, plot_acf_pacf, get_metric_display_name
from display import (
    setup_page,
    display_header,
    display_data_preview,
    display_dataset_info,
    get_prediction_parameters,
    display_prediction_results,
    display_full_prediction_table,
    display_comparison_table,
    display_model_summary
)


@st.cache_data(show_spinner=False)
def train_and_predict_metric(df, metrik, hari_prediksi, jam_prediksi_tuple, _cache_version="5.0"):
    """Fungsi training & peramalan SARIMAX dengan caching Streamlit untuk optimasi performa."""
    jam_prediksi = list(jam_prediksi_tuple)
    predictor = SarimaxEksogenPrediktor(jam_prediksi)
    predictor.set_selected_metric(metrik)

    waktu_terakhir = df.index[-1]
    dataframe_forcast = predictor.dataframe_prediksi(waktu_terakhir, hari_prediksi)
    result = predictor.prediksi_dengan_eksogen(df, metrik, dataframe_forcast)
    return dataframe_forcast, result


def main():
    setup_page()
    display_header()

    uploaded_file = st.file_uploader("Unggah file dataset kualitas internet (CSV atau XLSX)", type=["csv", "xlsx"])

    if uploaded_file is None:
        st.session_state["sudah_prediksi"] = False
        return

    # Reset status prediksi bila pengguna mengunggah file training baru
    file_id = getattr(uploaded_file, "name", "default")
    if st.session_state.get("current_train_file") != file_id:
        st.session_state["current_train_file"] = file_id
        st.session_state["sudah_prediksi"] = False

    try:
        with st.spinner("Memproses data..."):
            raw_df = load_data(uploaded_file)
            df = clean_data(raw_df)

        if df.empty or df.index.empty:
            st.error("Data kosong setelah diproses. Pastikan format file dan kolom waktu sesuai.")
            return

        display_data_preview(df)
        display_dataset_info(df)

        hari_prediksi, jam_prediksi = get_prediction_parameters(df)

        urutan_prioritas = ['throughput', 'latency', 'jitter', 'packet_loss', 'upload', 'download']
        kolom_tersedia = [col for col in urutan_prioritas if col in df.columns]

        if not kolom_tersedia:
            st.warning("Tidak ada metrik QoS yang ditemukan dalam data.")
            st.stop()

        metrik_utama = 'throughput' if 'throughput' in kolom_tersedia else kolom_tersedia[0]
        jam_prediksi_tuple = tuple(jam_prediksi)

        semua_prediksi = {}
        skor_akurasi = {}
        semua_eksogen = {}
        semua_diagnostik = {}
        dataframe_forcast = None

        st.subheader("Hasil Prediksi SARIMAX")
        progress_bar = st.progress(0)

        for i, metrik in enumerate(kolom_tersedia):
            nama_tampil = get_metric_display_name(metrik)

            with st.spinner(f"Melatih model SARIMAX ({nama_tampil})..."):
                df_fc, result = train_and_predict_metric(df, metrik, hari_prediksi, jam_prediksi_tuple)
                dataframe_forcast = df_fc

                if result is None or result[0] is None:
                    st.error(f"Gagal memprediksi {nama_tampil}.")
                    continue

                prediksi, akurasi, eksogen_data, data_test, _, prediksi_test, diagnostik = result
                semua_prediksi[metrik] = prediksi
                skor_akurasi[metrik] = akurasi
                semua_eksogen[metrik] = eksogen_data
                semua_diagnostik[metrik] = diagnostik

            progress_bar.progress((i + 1) / len(kolom_tersedia))

            st.markdown(f"#### {nama_tampil}")

            # Data Latih (Train) tail dan Data Testing (Kuning)
            n_test = len(data_test) if data_test is not None else 0
            data_train = df[metrik].iloc[:-n_test] if n_test > 0 else df[metrik]
            train_plot = data_train.tail(60)

            plot_prediction(metrik, akurasi, train_plot, dataframe_forcast, prediksi, data_test=data_test)
            display_prediction_results(metrik, prediksi, akurasi, diagnostik)

            if diagnostik is not None and data_test is not None and len(data_test) > 0:
                with st.expander(f"Detail Analisis Residu ({nama_tampil})"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Order ARIMA", str(diagnostik.get('order', '-')))
                    col2.metric("Seasonal Order", str(diagnostik.get('seasonal_order', '-')))
                    col3.metric("AIC", f"{diagnostik.get('aic', 0.0):.2f}" if diagnostik.get('aic') is not None else "N/A")

                    w = diagnostik.get('bobot_shrinkage', 0.0)
                    in_sample_resid = diagnostik.get('in_sample_resid')
                    if w > 0:
                        ac1 = diagnostik.get('autocorr_lag1')
                        ac1_teks = f"{ac1:.3f}" if isinstance(ac1, (int, float)) and np.isfinite(ac1) else "N/A"
                    if in_sample_resid is not None and len(in_sample_resid) > 10:
                        plot_acf_pacf(metrik, np.array(in_sample_resid).flatten())
                        try:
                            lags_ljung = min(10, len(in_sample_resid) // 5)
                            if lags_ljung > 0:
                                hasil_ljung_box = acorr_ljungbox(in_sample_resid, lags=[lags_ljung], return_df=True)
                                p_value = hasil_ljung_box['lb_pvalue'].iloc[0]
                                if p_value > 0.05:
                                    st.write(f"Uji Ljung-Box in-sample (p-value: {p_value:.4f}): Residu in-sample tidak menunjukkan autokorelasi signifikan — model telah menangkap pola secara memadai.")
                                else:
                                    st.write(f"Uji Ljung-Box in-sample (p-value: {p_value:.4f}): Terdapat autokorelasi sisa pada residu — struktur belum sepenuhnya ditangkap model.")
                        except Exception:
                            pass

        progress_bar.empty()

        # Tampilkan Tabel Hasil Prediksi Utama
        if dataframe_forcast is not None and semua_prediksi:
            display_full_prediction_table(dataframe_forcast, semua_prediksi, semua_eksogen, metrik_utama)

            # Bagian Opsi Bandingkan dengan Data Aktual
            st.divider()
            st.subheader("Bandingkan dengan Data Aktual (Pengujian Model)")
            st.markdown("Unggah file data aktual lapangan pada periode prediksi untuk memvalidasi akurasi model.")

            uploaded_actual = st.file_uploader(
                "Unggah file data aktual (CSV atau XLSX)",
                type=["csv", "xlsx"],
                key="uploader_data_aktual"
            )

            df_act = None
            if uploaded_actual is not None:
                try:
                    df_act = clean_data(load_data(uploaded_actual))
                except Exception as e:
                    st.error(f"Gagal memproses file data aktual: {e}")

            if df_act is not None and not df_act.empty:
                # Tampilkan tabel perbandingan per kolom langsung (BUKAN dalam expander/collapse)
                display_comparison_table(dataframe_forcast, semua_prediksi, df_act)

                # Hitung dan tampilkan Ringkasan Akurasi Prediksi (Evaluasi terhadap Data Aktual)
                skor_eval_aktual = {}
                diag_eval_aktual = {}
                p_eval = SarimaxEksogenPrediktor()

                for metrik in semua_prediksi.keys():
                    if metrik in df_act.columns:
                        y_act_series = df_act[metrik].reindex(dataframe_forcast).dropna()
                        if len(y_act_series) == len(semua_prediksi[metrik]):
                            y_act_arr = y_act_series.values.astype(float)
                            pred_arr = np.asarray(semua_prediksi[metrik], dtype=float)
                            err_act = p_eval.hitung_error_evaluasi(y_act_arr, pred_arr, metrik)
                            mae_act = float(np.mean(np.abs(y_act_arr - pred_arr)))
                            rmse_act = float(np.sqrt(np.mean((y_act_arr - pred_arr) ** 2)))
                            skor_eval_aktual[metrik] = err_act
                            diag_eval_aktual[metrik] = {'mae': mae_act, 'rmse': rmse_act}

                if skor_eval_aktual:
                    display_model_summary(skor_eval_aktual, diag_eval_aktual, tipe_evaluasi="aktual")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat menjalankan aplikasi: {str(e)}")
        with st.expander("Detail Error (Debug)", expanded=False):
            st.code(traceback.format_exc(), language="python")


if __name__ == "__main__":
    main()
