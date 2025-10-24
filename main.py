import streamlit as st
import numpy as np
import locale
import traceback
from statsmodels.stats.diagnostic import acorr_ljungbox

# Import components
from pre_processing import load_data, clean_data
from model import SarimaxEksogenPrediktor
from visualization import plot_prediction, plot_acf_pacf
from display import (
    setup_page,
    display_header,
    display_data_preview,
    display_dataset_info,
    get_prediction_parameters,
    display_prediction_results,
    display_full_prediction_table,
    display_model_summary
)

def main():
    # Setup
    setup_page()
   
    display_header()

    # File Uploader
    uploaded_file = st.file_uploader("Pilih file data internet Anda", type=["csv", "xlsx"])

    if uploaded_file is None:
        st.info("ℹ️ Silakan unggah file CSV atau XLSX untuk memulai prediksi.")
        return

    try:
        # --- Data Loading and Processing ---
        with st.spinner("📊 Memuat dan memproses data..."):
            raw_df = load_data(uploaded_file)
            df = clean_data(raw_df)
        st.success(f"✅ Data berhasil dimuat dan diproses: {len(df)} baris data")

        if df.empty or df.index.empty:
            st.error("❌ Data kosong setelah diproses. Mohon periksa format file Anda.")
            return

        # --- UI: Display Data Info and Get Parameters ---
        display_data_preview(df)
        display_dataset_info(df)
        
        kolom_terpilih, hari_prediksi, jam_prediksi = get_prediction_parameters(df)
        if not kolom_terpilih:
            st.warning("Tidak ada metrik kecepatan (upload/download) yang ditemukan dalam data.")
            st.stop()

        # --- Prediction ---
        predictor = SarimaxEksogenPrediktor(jam_prediksi)
        predictor.set_selected_metric(kolom_terpilih)

        kolom_prediksi = ['latency', 'packet_loss', 'jitter', kolom_terpilih]
        kolom_tersedia = [col for col in kolom_prediksi if col in df.columns]

        semua_prediksi = {}
        skor_akurasi = {}
        semua_eksogen = {}

        waktu_terakhir = df.index[-1]
        dataframe_forcast = predictor.dataframe_prediksi(waktu_terakhir, hari_prediksi)

        st.markdown("### 📈 Hasil Prediksi SARIMAX")
        progress_bar = st.progress(0)

        for i, metrik in enumerate(kolom_tersedia):
            st.markdown(f"#### 🔄 Memproses {metrik.title()}")
            with st.spinner(f"Membuat model SARIMAX untuk {metrik}..."):
                result = predictor.prediksi_dengan_eksogen(df, metrik, dataframe_forcast)
                if result is None:
                    st.error(f"Gagal memprediksi {metrik}.")
                    continue
                
                prediksi, akurasi, eksogen_data, data_test, _, prediksi_test = result
                semua_prediksi[metrik] = prediksi
                skor_akurasi[metrik] = akurasi
                semua_eksogen[metrik] = eksogen_data

            progress_bar.progress((i + 1) / len(kolom_tersedia))

            # --- UI: Display Results for each metric ---
            st.markdown(f"#### 📊 Hasil Prediksi {metrik.title()}")
            plot_prediction(metrik, akurasi, df[metrik].tail(72), dataframe_forcast, prediksi)
            display_prediction_results(metrik, prediksi, akurasi)

            # --- UI: ACF/PACF Analysis ---
            st.markdown(f"#### 🔍 Analisa ACF/PACF {metrik.title()}")
            with st.expander("Lihat Detail Analisa"):
                # Hitung residu
                residu_test = (data_test.values - prediksi_test.values).flatten()
                plot_acf_pacf(metrik, residu_test)

                # Uji Ljung-Box
                try:
                    lags_ljung = min(10, len(residu_test) // 5)
                    if lags_ljung <= 0:
                        st.info("ℹ️ Data terlalu sedikit untuk uji Ljung-Box.")
                        return

                    hasil_ljung_box = acorr_ljungbox(residu_test, lags=[lags_ljung], return_df=True)
                    p_value = hasil_ljung_box['lb_pvalue'].iloc[0]

                    if p_value > 0.05:
                        st.success(
                            f"✅ Uji Ljung-Box (p-value: {p_value:.4f}): "
                            "Residu tidak menunjukkan autokorelasi signifikan. Model sudah cukup baik."
                        )
                    else:
                        st.warning(
                            f"⚠️ Uji Ljung-Box (p-value: {p_value:.4f}): "
                            "Residu masih menunjukkan autokorelasi signifikan. Model mungkin perlu perbaikan."
                        )

                except Exception as e:
                    st.info(f"ℹ️ Tidak dapat melakukan uji Ljung-Box: {e}")


        progress_bar.progress(1.0)
        st.success("✅ Semua prediksi telah selesai!")

        # --- UI: Final Summary ---
        display_full_prediction_table(dataframe_forcast, semua_prediksi, semua_eksogen, kolom_terpilih)
        display_model_summary(skor_akurasi)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan pada aplikasi: {str(e)}")
        st.error(f"Detail error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
