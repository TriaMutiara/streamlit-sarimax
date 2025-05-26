import pandas as pd ## untuk manipulasi data
import matplotlib.pyplot as plt ## untuk visualisasi data
from matplotlib.dates import DateFormatter, DayLocator ## untuk format tanggal pada plot
from statsmodels.tsa.statespace.sarimax import SARIMAX ## model SARIMAX
import streamlit as st ## web interface
from itertools import product ## untuk kombinasi parameter SARIMAX

# Judul aplikasi Streamlit
st.title('Prediksi QoS dengan SARIMAX (75% Pelatihan, 25% Pengujian)')

# Pengunggah file CSV
unggahFileCsv = st.file_uploader("Unggah file CSV Anda", type=["csv"])

if unggahFileCsv is not None:
    # Muat dataset/ file csv dari pengambilan data
    df = pd.read_csv(unggahFileCsv)

    # Ubah kolom 'Tanggal' menjadi datetime dan tetapkan sebagai indeks
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df.set_index('Tanggal', inplace=True)

    # Tampilkan data lengkap
    st.write("Pratinjau Data Lengkap:")
    st.dataframe(df)

    # Bagi data menjadi set pelatihan dan pengujian (75% pelatihan, 25% pengujian)
    train_size = int(len(df) * 0.75)
    train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

    # Fungsi untuk mencari parameter SARIMAX terbaik
    def optimalkan_sarimax(endog, exog=None, max_p=2, max_d=1, max_q=2):
        best_aic = float('inf')
        best_order = None
        for p, d, q in product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
            try:
                model = SARIMAX(endog, exog=exog, order=(p, d, q),
                              seasonal_order=(0, 0, 0, 0))
                result = model.fit(disp=False)
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, d, q)
            except:
                continue
        return best_order

    # Pengaturan peramalan
    langkah_peramalan = 4  # Ramalkan 4 hari ke depan
    hasil_peramalan = []

    # Proses setiap kolom sebagai variabel endogen
    for kolom_sasaran in df.columns:
        st.subheader(f"Peramalan untuk {kolom_sasaran}")
        
        # Siapkan variabel endogen dan eksogen
        endog_pelatihan = train_df[kolom_sasaran]
        kolom_eksogen = [kol for kol in df.columns if kol != kolom_sasaran]
        
        if kolom_eksogen:
            exog_pelatihan = train_df[kolom_eksogen]
            exog_pengujian = test_df[kolom_eksogen]
            # Gunakan nilai eksogen terakhir yang diketahui untuk peramalan
            exog_peramalan = pd.concat([df[kolom_eksogen].iloc[-1:]] * langkah_peramalan)
        else:
            exog_pelatihan = None
            exog_pengujian = None
            exog_peramalan = None

        # Cari parameter terbaik
        urutan_terbaik = optimalkan_sarimax(endog_pelatihan, exog_pelatihan)
        st.write(f"Parameter terbaik untuk {kolom_sasaran}: p={urutan_terbaik[0]}, d={urutan_terbaik[1]}, q={urutan_terbaik[2]}")

        # Sesuaikan model SARIMAX
        model = SARIMAX(endog_pelatihan, exog=exog_pelatihan, order=urutan_terbaik,
                       seasonal_order=(0, 0, 0, 0))
        model_disesuaikan = model.fit(disp=False)

        # Ramalkan nilai masa depan
        peramalan = model_disesuaikan.forecast(steps=langkah_peramalan, exog=exog_peramalan)
        indeks_peramalan = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), 
                                       periods=langkah_peramalan)

        # Simpan hasil peramalan
        df_peramalan = pd.DataFrame({kolom_sasaran: peramalan.values}, index=indeks_peramalan)
        hasil_peramalan.append(df_peramalan)

        # Pembuatan visualisasi data
        plt.figure(figsize=(10, 4))
        # Plot data pelatihan
        plt.plot(train_df.index, train_df[kolom_sasaran], label='Data Pelatihan', 
                color='blue', linestyle='--')
        # Plot data pengujian
        plt.plot(test_df.index, test_df[kolom_sasaran], label='Data Pengujian', 
                color='gray', linestyle='--')
        # Plot fitted values
        plt.plot(train_df.index, model_disesuaikan.fittedvalues, label='Nilai Disesuaikan', 
                color='orange')
        # Plot peramalan
        plt.plot(indeks_peramalan, peramalan.values, label='Peramalan', 
                color='green')
        
        # Section menampilkan tanggal pada sumbu x
        plt.gca().xaxis.set_major_locator(DayLocator(interval=2))
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate() ## memiringkan tanggal agar tidak tumpang tindih
        plt.title(f'Prediksi SARIMAX untuk {kolom_sasaran}')
        plt.xlabel('Tanggal')
        plt.ylabel(kolom_sasaran)
        plt.legend()
        st.pyplot(plt) ## menampilkan plot di Streamlit

    # Gabungkan semua peramalan menjadi satu tabel
    peramalan_akhir = pd.concat(hasil_peramalan, axis=1)
    peramalan_akhir.index.name = 'Tanggal'
    peramalan_akhir.reset_index(inplace=True)
    peramalan_akhir.insert(0, 'No', range(1, len(peramalan_akhir) + 1))

    # Tampilkan tabel peramalan akhir
    st.write("Hasil Peramalan QoS 4 Hari ke Depan:")
    st.dataframe(peramalan_akhir, hide_index=True) ## menampilkan table

    # Tombol unduh/download hasil peramalan
    csv = peramalan_akhir.to_csv(index=False)
    st.download_button("Unduh Hasil Prediksi QoS", data=csv, 
                      file_name="qos_forecast.csv", mime="text/csv")

else:
    st.write("Silakan unggah file CSV untuk memulai prediksi.")
