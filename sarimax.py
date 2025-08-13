import pandas as pd ## untuk manipulasi data
import matplotlib.pyplot as plt ## untuk visualisasi data
from matplotlib.dates import DateFormatter ## untuk format tanggal pada plot
from statsmodels.tsa.statespace.sarimax import SARIMAX ## model SARIMAX
import streamlit as st ## web interface
from itertools import product ## untuk kombinasi parameter SARIMAX

# Judul aplikasi Streamlit
st.title('Prediksi QoS dengan SARIMAX')

# Pengunggah file CSV
unggahFile = st.file_uploader("Unggah file Excel Anda", type=["xlsx"])

if unggahFile is not None:
    # Muat dataset dari file Excel
    df = pd.read_excel(unggahFile)

    # Gabungkan 'Tanggal' dan 'Jam' menjadi satu kolom datetime
    df['datetime'] = pd.to_datetime(df['Tanggal'].astype(str).str.split(' ').str[0] + ' ' + df['Jam'].astype(str), format='%Y-%m-%d %H.%M')
    df.set_index('datetime', inplace=True)
    df.drop(['Tanggal', 'Jam'], axis=1, inplace=True) # Hapus kolom asli
    
    # Cari kolom upload dan download yang tersedia
    upload_columns = [col for col in df.columns if 'upload' in col.lower()]
    download_columns = [col for col in df.columns if 'download' in col.lower()]
    available_columns = upload_columns + download_columns
    
    # Dropdown untuk memilih kolom upload atau download
    selected_column = st.selectbox("Pilih kolom untuk di-forecast:", available_columns)
    
    # Input untuk jumlah langkah peramalan
    langkah_peramalan = st.number_input("Jumlah langkah peramalan (hari):", min_value=1, max_value=30, value=1)

    # Tombol untuk memproses data
    if st.button("Proses"):
        # Kolom yang akan di-forecast: semua kolom QoS
        qos_columns = ['latency', 'packet_loss', 'jitter', selected_column]
        qos_columns = [col for col in qos_columns if col in df.columns]

        # Tampilkan data lengkap
        st.write("Pratinjau Data Lengkap:")
        st.dataframe(df)

        # Bagi data menjadi set pelatihan dan pengujian
        train_size = int(len(df) * 0.75)
        train_df, test_df = df[:train_size], df[train_size:]

        # Fungsi untuk mencari parameter SARIMAX terbaik
        def optimalkan_sarimax(endog, exog=None, max_p=2, max_d=1, max_q=2):
            best_aic = float('inf')
            best_order = None
            for p, d, q in product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
                try:
                    model = SARIMAX(endog, exog=exog, order=(p, d, q), seasonal_order=(0, 0, 0, 0))
                    result = model.fit(disp=False)
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p, d, q)
                except:
                    continue
            # Jika tidak ada model yang berhasil, gunakan order default
            if best_order is None:
                best_order = (1, 1, 1)
                logging.warning("No optimal order found, using default (1,1,1)")
            return best_order

        # Inisialisasi hasil peramalan
        hasil_peramalan = []

        # Buat indeks peramalan untuk N hari ke depan dengan jam yang sesuai
        last_date = df.index[-1]
        unique_hours = sorted(df.index.hour.unique())
        
        # Logika perbaikan: Buat forecast untuk N hari ke depan, satu forecast per hari
        forecast_dates = []
        
        # Gunakan jam yang berbeda untuk setiap hari
        for i in range(langkah_peramalan):
            day_offset = i + 1  # Mulai dari hari besok
            
            # Pilih jam secara bergantian dari unique_hours
            hour_index = i % len(unique_hours)  # Gunakan modulo untuk memilih jam bergantian
            selected_hour = unique_hours[hour_index]
            
            # Buat datetime untuk forecast
            next_date = last_date + pd.Timedelta(days=day_offset)
            forecast_datetime = next_date.replace(hour=selected_hour, minute=0, second=0, microsecond=0)
            forecast_dates.append(forecast_datetime)
            
        
        indeks_peramalan = pd.DatetimeIndex(forecast_dates)

        # Proses setiap kolom QoS sebagai variabel endogen
        for target_column in qos_columns:
            st.subheader(f"Peramalan untuk {target_column}")

            # Siapkan variabel endogen dan eksogen
            endog_pelatihan = train_df[target_column]
            kolom_eksogen = [col for col in df.columns if col != target_column]
            
            exog_pelatihan = train_df[kolom_eksogen] if kolom_eksogen else None
            exog_pengujian = test_df[kolom_eksogen] if kolom_eksogen else None
            exog_peramalan = train_df[kolom_eksogen].mean() if kolom_eksogen else None
            if exog_peramalan is not None:
                # Buat DataFrame dengan nilai rata-rata untuk setiap langkah peramalan
                exog_peramalan = pd.DataFrame([exog_peramalan] * langkah_peramalan,
                columns=kolom_eksogen, index=indeks_peramalan)
            
            urutan_terbaik = optimalkan_sarimax(endog_pelatihan, exog_pelatihan)
            st.write(f"Parameter terbaik untuk {target_column}: p={urutan_terbaik[0]}, d={urutan_terbaik[1]}, q={urutan_terbaik[2]}")

            # Sesuaikan model SARIMAX
            model = SARIMAX(endog_pelatihan, exog=exog_pelatihan, order=urutan_terbaik, seasonal_order=(0, 0, 0, 0))
            model_disesuaikan = model.fit(disp=False)

            # Ramalkan nilai masa depan
            peramalan = model_disesuaikan.forecast(steps=langkah_peramalan, exog=exog_peramalan)
            
            # Simpan hasil peramalan
            df_peramalan = pd.DataFrame({target_column: peramalan.values}, index=indeks_peramalan)
            hasil_peramalan.append(df_peramalan)

            # Pembuatan visualisasi data
            plt.figure(figsize=(12, 6))
            plt.plot(train_df.index, train_df[target_column], label='Data Pelatihan', color='blue', linestyle='--')
            plt.plot(test_df.index, test_df[target_column], label='Data Pengujian', color='gray', linestyle='--')
            plt.plot(train_df.index, model_disesuaikan.fittedvalues, label='Nilai Disesuaikan', color='orange')
            plt.plot(indeks_peramalan, peramalan.values, label='Peramalan', color='green', marker='o')
            
            # Format sumbu x untuk menampilkan tanggal
            plt.gca().xaxis.set_major_formatter(DateFormatter('%d-%m-%Y'))
            plt.gcf().autofmt_xdate()
            plt.title(f'Prediksi SARIMAX untuk {target_column}')
            plt.xlabel('Tanggal')
            plt.ylabel(target_column)
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

        # Gabungkan semua peramalan menjadi satu tabel
        if hasil_peramalan:
            peramalan_akhir = pd.concat(hasil_peramalan, axis=1)
            peramalan_akhir.index.name = 'Tanggal'
            peramalan_akhir.reset_index(inplace=True)
            peramalan_akhir.insert(0, 'No', range(1, len(peramalan_akhir) + 1))

            # Tampilkan tabel peramalan akhir
            st.write(f"Hasil Peramalan QoS {langkah_peramalan} Hari ke Depan:")
            st.dataframe(peramalan_akhir, hide_index=True)

            # Tombol unduh/download hasil peramalan gabungan
            csv = peramalan_akhir.to_csv(index=False)
            st.download_button("Unduh Hasil Prediksi QoS", data=csv,
            file_name="qos_forecast.csv", mime="text/csv")

else:
    st.write("Silakan unggah file Excel untuk memulai prediksi.")

