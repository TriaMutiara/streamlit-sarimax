# 📊 Proyek Forecasting SARIMAX dengan Exogenous Variables

Proyek ini bertujuan untuk melakukan **forecasting time series** menggunakan model **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors)**.  
SARIMAX dipilih karena dapat memodelkan **pola musiman (seasonality)** sekaligus memperhitungkan pengaruh **variabel eksternal (exogenous variables)**.

### 1. Upload Data
- User upload file Excel (.xlsx) berisi data kualitas internet.
- Data dibaca dengan pandas → pd.read_excel.

### 2. Preprocessing Data
- Menghapus kolom yang tidak relevan untuk prediksi.
- Memastikan kolom waktu (Timestamp) dipakai sebagai index (DateTimeIndex).
- Drop missing values supaya model lebih stabil.

### 3. Pilih Target 
- User pilih target (misalnya upload) dari dropdown Streamlit.
- Target inilah yang digunakan untuk throughput.
- Variabel lain otomatis jadi variabel eksogen (faktor eksternal).

### 4. Analisis Korelasi
- Hitung korelasi Pearson antar variabel (upload, download, latency, jitter, packet_loss).
- Dibuat heatmap dengan seaborn → supaya terlihat variabel mana yang paling berpengaruh.

### 5. Split Data
- Data dibagi 80% training, 20% testing.
- train_exog = variabel input eksternal untuk train.
- test_exog = variabel input eksternal untuk test.

### 6. Optimasi Parameter SARIMAX
- Ada fungsi optimize_sarimax → coba beberapa kombinasi parameter p, d, q.
- Pilih model dengan AIC terkecil (lebih baik).
- Model SARIMAX dilatih pakai training data.

### 7. Prediksi
- Model prediksi nilai target pada periode test.
- Hasil prediksi dibandingkan dengan data asli.
- Hitung MAPE sebagai skor akurasi.

### 8. Prediksi Masa Depan
- Buat data future exogenous variables (pakai pola jam & hari).
- Model prediksi ke depan (default 4 step).
- Output jadi forecast kualitas internet di masa depan.

## 9. Visualisasi
- Grafik interaktif (Plotly):
- Data asli
- Prediksi
- Forecast ke depan
- Tabel hasil prediksi masa depan.

### 10. Insight
- Streamlit tampilkan hasil:
- Variabel target
- Skor akurasi
- Tabel prediksi masa depan

### Jadi alurnya:
📂 Upload data → 🧹 Preprocessing → 🎯 Pilih target → 📊 Analisis korelasi → 🔧 Latih SARIMAX → 📈 Prediksi & Forecast → 📝 Insight
