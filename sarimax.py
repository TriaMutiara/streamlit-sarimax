import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kualitas Internet",
    page_icon="📶",
)

st.title('📶 Prediksi Kualitas Internet (QoS) - Sarimax')
uploaded_file = st.file_uploader("Pilih file data internet Anda", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Load data
        with st.spinner("📊 Memuat dan memproses data..."):
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8')
            else:
                df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Data berhasil dimuat: {len(df)} baris data")
        
        def bersihkan_data(df): 
            # Buat salinan data
            data_bersih = df.copy()
            
            # Hapus baris kosong
            data_bersih = data_bersih.dropna(how='all')
            
            # Bersihkan nama kolom
            data_bersih.columns = data_bersih.columns.str.strip()
            
            # Gabungkan tanggal dan jam menjadi satu kolom waktu
            if 'Tanggal' in data_bersih.columns and 'Jam' in data_bersih.columns:
                waktu_gabungan = data_bersih['Tanggal'].astype(str) + ' ' + data_bersih['Jam'].astype(str)
                
                # Coba berbagai format tanggal
                for format_tanggal in ['%d/%m/%Y %H.%M', '%Y-%m-%d %H.%M', '%d-%m-%Y %H.%M']:
                    try:
                        data_bersih['waktu'] = pd.to_datetime(waktu_gabungan, format=format_tanggal, errors='raise')
                        break
                    except:
                        continue
                else:
                    data_bersih['waktu'] = pd.to_datetime(waktu_gabungan, format='%d/%m/%Y %H.%M', errors='coerce')
                
                # Hapus data dengan waktu tidak valid
                data_bersih = data_bersih.dropna(subset=['waktu'])
                data_bersih.set_index('waktu', inplace=True)
                
                # Hapus kolom yang tidak diperlukan
                data_bersih = data_bersih.drop(['Tanggal', 'Jam', 'Hari'], axis=1, errors='ignore')
            
            # Bersihkan data numerik
            kolom_angka = ['upload', 'download', 'latency', 'packet_loss', 'jitter']
            for kolom in kolom_angka:
                if kolom in data_bersih.columns:
                    # Ubah koma menjadi titik untuk desimal
                    if data_bersih[kolom].dtype == 'object':
                        data_bersih[kolom] = data_bersih[kolom].astype(str).str.replace(',', '.')
                    # Konversi ke angka
                    data_bersih[kolom] = pd.to_numeric(data_bersih[kolom], errors='coerce')
            
            # Hapus baris yang semua kolom pentingnya kosong
            kolom_tersedia = [col for col in kolom_angka if col in data_bersih.columns]
            data_bersih = data_bersih.dropna(subset=kolom_tersedia, how='all')
            
            # Isi nilai kosong dengan interpolasi
            data_bersih = data_bersih.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
            
            return data_bersih
        
        # Proses data
        df = bersihkan_data(df)
        
        # Tampilkan preview data
        st.markdown("### 👀 Preview Data Anda:")
        st.dataframe(df, use_container_width=True)
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pilih metrik yang akan diprediksi
            kolom_kecepatan = [col for col in df.columns if col in ['upload', 'download']]
            if kolom_kecepatan:
                metrik_terpilih = st.selectbox(
                    "🚀 Pilih metrik kecepatan yang ingin diprediksi:",
                    kolom_kecepatan,
                    help="Upload: kecepatan mengirim data, Download: kecepatan menerima data"
                )
            else:
                st.error("❌ Data upload/download tidak ditemukan!")
                st.stop()
        
        with col2:
            hari_prediksi = st.number_input(
                "📅 Berapa hari ke depan ingin diprediksi?",
                min_value=1, 
                max_value=14, 
                value=2,
                help="Semakin banyak hari, semakin tidak akurat prediksinya"
            )

        class AutoEksogenPrediktor:
            def __init__(self):
                self.scaler_data = {}
                self.model_data = {}
                self.jam_prediksi = [9, 13, 17]
                self.eksogen_scalers = {}
                self.variabel_eksogen_terpilih = {}
                self.selected_metric = None
                self.excluded = None

            def cek_stasioneritas(self, series, nama):
                result = adfuller(series.dropna())
                if result[1] < 0.05:
                    st.success(f"✅ Data **{nama}** Sudah Stasioner")
                    return 0          # d = 0
                else:
                    st.warning("⚠️ Non-stasioner → otomatis differencing 1x")
                    return 1          # d = 1
        
            def set_selected_metric(self, metrik):
                self.selected_metric = metrik
                if metrik == 'upload':
                    self.excluded = 'download'
                elif metrik == 'download':
                    self.excluded = 'upload'
                else:
                    self.excluded = None
            
            def normalisasi_data(self, data, nama_kolom):
                scaler = MinMaxScaler(feature_range=(0.01, 0.99))
                data_ternormalisasi = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
                self.scaler_data[nama_kolom] = scaler
                return pd.Series(data_ternormalisasi, index=data.index)
            
            def kembalikan_skala_asli(self, data, nama_kolom):
                if nama_kolom in self.scaler_data:
                    return self.scaler_data[nama_kolom].inverse_transform(data.reshape(-1, 1)).flatten()
                return data
            
            def analisis_korelasi(self, df, target_kolom):
                """Otomatis menganalisis korelasi dan memilih variabel eksogen terbaik"""
                semua_variabel = {}
                target_data = df[target_kolom]
                
                st.info(f"🔍 Menganalisis korelasi untuk {target_kolom}...")
                
                # 1. Variabel waktu dasar
                variabel_waktu = pd.DataFrame(index=df.index)
                variabel_waktu['jam'] = df.index.hour
                variabel_waktu['hari_minggu'] = df.index.dayofweek
                variabel_waktu['akhir_pekan'] = (df.index.dayofweek >= 5).astype(int)
                
                # Pola trigonometri untuk siklus
                variabel_waktu['siklus_harian_x'] = np.sin(2 * np.pi * df.index.hour / 24)
                variabel_waktu['siklus_harian_y'] = np.cos(2 * np.pi * df.index.hour / 24)
                variabel_waktu['siklus_mingguan_x'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
                variabel_waktu['siklus_mingguan_y'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

                # 3. Metrik lain sebagai eksogen
                base_list = ['upload', 'download', 'latency', 'packet_loss', 'jitter']
                metrik_lain = [col for col in df.columns if col != target_kolom and col in base_list and col != self.excluded]
                variabel_metrik = pd.DataFrame(index=df.index)
                
                for metrik in metrik_lain:
                    # Nilai asli
                    variabel_metrik[f'nilai_asli_{metrik}'] = df[metrik]
                    
                    # Moving averages (jika data cukup)
                    if len(df) > 5:
                        variabel_metrik[f'periode_pendek_{metrik}'] = df[metrik].rolling(window=3, min_periods=1).mean()
                        variabel_metrik[f'periode_menengah_{metrik}'] = df[metrik].rolling(window=5, min_periods=1).mean()
                    
                    # Lag values
                    variabel_metrik[f'nilai_sebelumnya_{metrik}'] = df[metrik].shift(1)
                    if len(df) > 10:
                        variabel_metrik[f'2_nilai_{metrik}_sebelumnya'] = df[metrik].shift(2)
                    
                    # Volatilitas (standar deviasi rolling)
                    if len(df) > 5:
                        variabel_metrik[f'volatilitas_{metrik}'] = df[metrik].rolling(window=5, min_periods=1).std().fillna(0)
                
                # 4. Variabel target sendiri
                variabel_target = pd.DataFrame(index=df.index)
                if len(df) > 5:
                    variabel_target[f'periode_pendek_{target_kolom}'] = target_data.rolling(window=3, min_periods=1).mean()
                    variabel_target[f'periode_menengah_{target_kolom}'] = target_data.rolling(window=7, min_periods=1).mean()
                    variabel_target[f'volatilitas_{target_kolom}'] = target_data.rolling(window=3, min_periods=1).std().fillna(0)
                
                # Gabungkan semua variabel
                semua_variabel_df = pd.concat([variabel_waktu, variabel_metrik, variabel_target], axis=1)
                semua_variabel_df = semua_variabel_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

                # Hitung korelasi untuk setiap variabel
                korelasi_hasil = {}
                variabel_terpilih = []
                korelasi_data = []
                
                for kolom in semua_variabel_df.columns:
                    try:
                        # Hitung korelasi Pearson
                        korelasi, nilai_probabilitas = pearsonr(target_data, semua_variabel_df[kolom])
                        korelasi_absolut = abs(korelasi)
                        
                        korelasi_hasil[kolom] = {
                            'korelasi': korelasi,
                            'korelasi_absolut': korelasi_absolut,
                            'nilai_probabilitas': nilai_probabilitas,
                            'signifikan': nilai_probabilitas < 0.05
                        }
                        
                        # Simpan data untuk tabel
                        korelasi_data.append({
                            'Variabel': kolom,
                            'Korelasi': f"{korelasi:.3f}",
                            'Kekuatan': korelasi_absolut,
                            'Signifikan': "✅" if nilai_probabilitas < 0.05 else "❌"
                        })
                        
                        # variabel dengan korelasi kuat dan signifikan
                        if korelasi_absolut > 0.1 and nilai_probabilitas < 0.1:  # Threshold yang lebih luas
                            variabel_terpilih.append(kolom)
                            
                    except Exception as e:
                        continue
                
                # Urutkan berdasarkan kekuatan korelasi
                korelasi_data.sort(key=lambda x: x['Kekuatan'], reverse=True)
                
                # Batasi jumlah variabel untuk menghindari overfitting
                max_variabel = min(len(variabel_terpilih), max(3, len(df) // 10))
                
                # Pilih variabel terbaik berdasarkan korelasi
                variabel_terbaik = sorted(variabel_terpilih, 
                                        key=lambda x: korelasi_hasil[x]['korelasi_absolut'], 
                                        reverse=True)[:max_variabel]
                
                # Pastikan setidaknya ada beberapa variabel waktu dasar
                variabel_waktu_penting = ['jam', 'siklus_harian_x', 'siklus_harian_y']
                for var in variabel_waktu_penting:
                    if var in semua_variabel_df.columns and var not in variabel_terbaik:
                        variabel_terbaik.append(var)
                
                self.variabel_eksogen_terpilih[target_kolom] = variabel_terbaik
                df_korelasi = pd.DataFrame(korelasi_data)
                st.write(df_korelasi)
                
                
                # Normalisasi variabel eksogen
                eksogen_final = pd.DataFrame(index=df.index)
                scaler_dict = {}
                
                for var in variabel_terbaik:
                    if var in semua_variabel_df.columns:
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        data_scaled = scaler.fit_transform(semua_variabel_df[var].values.reshape(-1, 1)).flatten()
                        eksogen_final[var] = data_scaled
                        scaler_dict[var] = scaler
                
                self.eksogen_scalers[target_kolom] = scaler_dict
                
                return eksogen_final
            
            def buat_eksogen_forcasting(self, df, target_kolom, jadwal_prediksi):
                variabel_terpilih = self.variabel_eksogen_terpilih.get(target_kolom, [])
                scaler_dict = self.eksogen_scalers.get(target_kolom, {})
                
                eksogen_forcast = pd.DataFrame(index=jadwal_prediksi)
                
                for var in variabel_terpilih:
                    if var.startswith('jam'):
                        if var == 'jam':
                            nilai = jadwal_prediksi.hour
                        else:
                            # Jam sibuk
                            jam_sibuk = int(var.split('_')[-1]) - 1 if '_' in var else 9
                            if jam_sibuk < len(self.jam_prediksi):
                                jam_target = self.jam_prediksi[jam_sibuk]
                                nilai = (jadwal_prediksi.hour == jam_target).astype(int)
                            else:
                                nilai = np.zeros(len(jadwal_prediksi))
                    
                    elif var.startswith('hari'):
                        if var == 'hari_minggu':
                            nilai = jadwal_prediksi.dayofweek
                        else:
                            nilai = np.zeros(len(jadwal_prediksi))
                    
                    elif var == 'akhir_pekan':
                        nilai = (jadwal_prediksi.dayofweek >= 5).astype(int)
                    
                    elif var.endswith(('_x', '_y')):
                        freq_map = {
                            'jam': (jadwal_prediksi.hour, 24),
                            'hari': (jadwal_prediksi.dayofweek, 7)
                        }

                        # cari key yg ada di nama variabel
                        key = next((k for k in freq_map if k in var), None)

                        if key:
                            val, period = freq_map[key]
                            func = np.sin if var.endswith('_x') else np.cos
                            nilai = func(2 * np.pi * val / period)
                        else:
                            nilai = np.zeros(len(jadwal_prediksi))

                    
                    elif '_' in var:
                        # Variabel dari metrik lain atau target
                        base_metrik = var.split('_')[0]
                        if base_metrik in df.columns:
                            if 'periode' in var or 'nilai' in var or 'volatilitas' in var or 'asli' in var:
                                nilai_rata = df[base_metrik].tail(min(24, len(df))).mean()
                                nilai = np.full(len(jadwal_prediksi), nilai_rata)
                            else:
                                nilai = np.zeros(len(jadwal_prediksi))
                        else:
                            nilai = np.zeros(len(jadwal_prediksi))
                    
                    else:
                        nilai = np.zeros(len(jadwal_prediksi))
                    
                    # Normalisasi menggunakan scaler yang sudah disimpan
                    if var in scaler_dict:
                        try:
                            if isinstance(nilai, (int, float)):
                                nilai = np.full(len(jadwal_prediksi), nilai)
                            nilai_scaled = scaler_dict[var].transform(nilai.reshape(-1, 1)).flatten()
                            eksogen_forcast[var] = nilai_scaled
                        except:
                            eksogen_forcast[var] = np.zeros(len(jadwal_prediksi))
                    else:
                        eksogen_forcast[var] = nilai if hasattr(nilai, '__len__') else np.full(len(jadwal_prediksi), nilai)
                
                return eksogen_forcast
            
            def cari_pengaturan_terbaik(self, data, nama_kolom):
                akurasi_terbaik = float('inf')
                d = self.cek_stasioneritas(data, nama_kolom) 
                pengaturan_terbaik = (1, d, 1)
                musiman_terbaik = (0, 0, 0, 0)
                
                # Pengaturan parameter berdasarkan karakteristik data
                if 'jitter' in nama_kolom.lower():
                    pilihan_parameter = [(2, 1, 2), (3, 1, 1), (2, 1, 3)]
                elif 'packet_loss' in nama_kolom.lower():
                    pilihan_parameter = [(1, 1, 1), (2, 1, 1), (1, 1, 2)]
                else:
                    pilihan_parameter = [(1, 1, 1), (2, 1, 2), (1, 1, 2)]
                
                pilihan_musiman = [(0, 0, 0, 0)]
                if len(data) > 30:
                    pilihan_musiman.extend([(1, 0, 1, 24), (1, 1, 0, 24)])
                
                for (p, d, q) in pilihan_parameter:
                    for (P, D, Q, s) in pilihan_musiman:
                        try:
                            if s > 0 and len(data) < 2 * s:
                                continue
                            
                            model = SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, s))
                            model_terlatih = model.fit(disp=False, maxiter=200)
                            
                            if model_terlatih.aic < akurasi_terbaik:
                                akurasi_terbaik = model_terlatih.aic
                                pengaturan_terbaik = (p, d, q)
                                musiman_terbaik = (P, D, Q, s)
                        except:
                            continue
                
                return pengaturan_terbaik, musiman_terbaik
            
            def buat_jadwal_prediksi(self, waktu_terakhir, jumlah_hari):
                jadwal_prediksi = []
                
                for hari in range(1, jumlah_hari + 1):
                    tanggal_berikutnya = waktu_terakhir + pd.Timedelta(days=hari)
                    
                    for jam in self.jam_prediksi:
                        waktu_prediksi = tanggal_berikutnya.replace(
                            hour=jam, minute=0, second=0, microsecond=0
                        )
                        jadwal_prediksi.append(waktu_prediksi)
                
                return pd.DatetimeIndex(jadwal_prediksi)
            
            def prediksi_dengan_auto_eksogen(self, df, nama_kolom, jumlah_prediksi, jadwal_prediksi):
                try:
                    # Penanganan khusus untuk packet loss
                    if 'packet_loss' in nama_kolom.lower():
                        rasio_nol = (df[nama_kolom] == 0).sum() / len(df[nama_kolom])
                        if rasio_nol > 0.8:
                            st.info(f"📊 {nama_kolom}: Sebagian besar data bernilai 0 ({rasio_nol:.1%})")
                            data_bukan_nol = df[nama_kolom][df[nama_kolom] > 0]
                            if len(data_bukan_nol) > 5:
                                rata_rata = data_bukan_nol.mean()
                                prediksi = np.random.poisson(rata_rata * 0.1, jumlah_prediksi)
                                prediksi = np.clip(prediksi, 0, 2)
                            else:
                                prediksi = np.zeros(jumlah_prediksi)
                            return prediksi, 0.0
                    
                    # Analisis otomatis dan buat variabel eksogen
                    eksogen = self.analisis_korelasi(df, nama_kolom)
                    eksogen_forcast = self.buat_eksogen_forcasting(df, nama_kolom, jadwal_prediksi)
                    
                    # Normalisasi data target
                    data_normal = self.normalisasi_data(df[nama_kolom], nama_kolom)
                    pengaturan_terbaik, musiman_terbaik = self.cari_pengaturan_terbaik(data_normal, nama_kolom)
                    
                    # Split data
                    ukuran_latih = int(len(data_normal) * 0.85)
                    data_latih = data_normal[:ukuran_latih]
                    data_test = data_normal[ukuran_latih:]
                    eksogen_latih = eksogen[:ukuran_latih]
                    eksogen_test = eksogen[ukuran_latih:]
                    
                    # Train model dengan eksogen
                    if len(eksogen.columns) > 0:
                        model = SARIMAX(
                            data_latih, 
                            exog=eksogen_latih,
                            order=pengaturan_terbaik, 
                            seasonal_order=musiman_terbaik
                        )
                    else:
                        model = SARIMAX(
                            data_latih, 
                            order=pengaturan_terbaik, 
                            seasonal_order=musiman_terbaik
                        )
                    
                    model_terlatih = model.fit(disp=False, maxiter=300)
                    
                    # Hitung akurasi
                    tingkat_akurasi = 0.0
                    if len(data_test) > 0:
                        if len(eksogen.columns) > 0:
                            prediksi_test = model_terlatih.forecast(len(data_test), exog=eksogen_test)
                        else:
                            prediksi_test = model_terlatih.forecast(len(data_test))
                        
                        prediksi_asli = self.kembalikan_skala_asli(prediksi_test.values, nama_kolom)
                        test_asli = self.kembalikan_skala_asli(data_test.values, nama_kolom)
                        tingkat_akurasi = self.hitung_akurasi(test_asli, prediksi_asli)
                    
                    # Prediksi masa depan
                    if len(eksogen.columns) > 0 and len(eksogen_forcast.columns) > 0:
                        prediksi_masa_depan = model_terlatih.forecast(jumlah_prediksi, exog=eksogen_forcast)
                    else:
                        prediksi_masa_depan = model_terlatih.forecast(jumlah_prediksi)
                    
                    hasil_prediksi = self.kembalikan_skala_asli(prediksi_masa_depan.values, nama_kolom)
                    
                    # Validasi hasil
                    if 'packet_loss' in nama_kolom.lower():
                        hasil_prediksi = np.clip(hasil_prediksi, 0, 10)
                    elif 'jitter' in nama_kolom.lower():
                        hasil_prediksi = np.maximum(hasil_prediksi, 0)
                    elif 'latency' in nama_kolom.lower():
                        hasil_prediksi = np.maximum(hasil_prediksi, 1)
                    elif nama_kolom in ['upload', 'download']:
                        hasil_prediksi = np.maximum(hasil_prediksi, 0.1)
                    
                    return hasil_prediksi, tingkat_akurasi
                    
                except Exception as e:
                    st.error(f"❌ Error dalam prediksi {nama_kolom}: {str(e)}")
                    # Fallback ke prediksi sederhana
                    rata_rata = df[nama_kolom].tail(24).mean()
                    std_dev = df[nama_kolom].tail(24).std()
                    prediksi_fallback = np.random.normal(rata_rata, std_dev * 0.1, jumlah_prediksi)
                    return np.maximum(prediksi_fallback, 0), 50.0
            
            def hitung_akurasi(self, nilai_asli, nilai_prediksi):
                nilai_asli = np.array(nilai_asli)
                nilai_prediksi = np.array(nilai_prediksi)
                epsilon = 1e-8
                
                mask = np.abs(nilai_asli) > epsilon
                
                if np.sum(mask) == 0:
                    return 0.0
                
                mape = np.mean(np.abs(nilai_asli[mask] - nilai_prediksi[mask]) / 
                              (np.abs(nilai_asli[mask]) + np.abs(nilai_prediksi[mask]) + epsilon)) * 200
                
                return min(mape, 999.0)
        
        
        if st.button("🚀 Mulai Prediksi dengan", type="primary"):
            predictor = AutoEksogenPrediktor()
            predictor.set_selected_metric(metrik_terpilih)
            
            # Tentukan metrik yang akan diprediksi
            semua_metrik = ['latency', 'packet_loss', 'jitter', metrik_terpilih]
            metrik_tersedia = [col for col in semua_metrik if col in df.columns]
            
            # Siapkan container untuk hasil
            semua_prediksi = {}
            skor_akurasi = {}
            
            # Validasi data
            if df.empty or df.index.empty:
                st.error("❌ Data kosong setelah diproses. Mohon periksa format file Anda.")
                st.stop()
            
            # Buat jadwal prediksi
            waktu_terakhir = df.index[-1]
            jadwal_prediksi = predictor.buat_jadwal_prediksi(waktu_terakhir, hari_prediksi)
            jumlah_titik_prediksi = len(jadwal_prediksi)
            
            st.markdown("### 📈 Hasil Prediksi dengan")
            
            # Proses setiap metrik
            for metrik in metrik_tersedia:
                st.markdown(f"#### 🔄 Memproses {metrik.title()}")
                
                with st.spinner(f"Menganalisis korelasi dan membuat prediksi untuk {metrik}..."):
                    prediksi, akurasi = predictor.prediksi_dengan_auto_eksogen(
                        df, metrik, jumlah_titik_prediksi, jadwal_prediksi
                    )
                
                semua_prediksi[metrik] = prediksi
                skor_akurasi[metrik] = akurasi
                
                # Tampilkan grafik
                st.markdown(f"#### 📊 Hasil Prediksi {metrik.title()}")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot data historis (50 titik terakhir untuk clarity)
                data_terkini = df[metrik].tail(50)
                ax.plot(data_terkini.index, data_terkini.values,
                       label='Data Historis', color='#1f77b4', alpha=0.8, linewidth=2)
                
                # Plot prediksi
                ax.plot(jadwal_prediksi, prediksi,
                       label='Prediksi', color='#ff7f0e',
                       marker='o', markersize=8, linewidth=3, alpha=0.9)
                
                # Highlight jam puncak
                for waktu in jadwal_prediksi:
                    if waktu.hour in [9, 13, 17]:
                        ax.axvline(waktu, color='red', alpha=0.2, linestyle='--')
                
                ax.set_title(f'{metrik.title()} (MAPE: {akurasi:.1f}%)', 
                            fontweight='bold', fontsize=14)
                ax.set_ylabel(f'{metrik.title()}', fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Tampilkan insight
                nilai_tertinggi = np.max(prediksi)
                nilai_terendah = np.min(prediksi)
                rata_rata_prediksi = np.mean(prediksi)
                
                col_insight1, col_insight2, col_insight3 = st.columns(3)
                with col_insight1:
                    st.metric("📈 Nilai Tertinggi", f"{nilai_tertinggi:.2f}")
                with col_insight2:
                    st.metric("📉 Nilai Terendah", f"{nilai_terendah:.2f}")
                with col_insight3:
                    st.metric("📊 Rata-rata", f"{rata_rata_prediksi:.2f}")
                
                st.divider()
            
            # Buat tabel ringkasan semua prediksi
            st.markdown("### 📋 Tabel Lengkap Prediksi")
            
            tabel_prediksi = pd.DataFrame(index=jadwal_prediksi)
            tabel_prediksi['📅 Tanggal'] = tabel_prediksi.index.date
            tabel_prediksi['🕐 Jam'] = tabel_prediksi.index.strftime('%H:%M')
            
            # Tambahkan kolom prediksi
            for metrik, prediksi in semua_prediksi.items():
                if metrik == 'upload':
                    nama_tampilan = '⬆️ Upload (Mbps)'
                elif metrik == 'download':
                    nama_tampilan = '⬇️ Download (Mbps)'
                elif metrik == 'latency':
                    nama_tampilan = '⏱️ Latency (ms)'
                elif metrik == 'packet_loss':
                    nama_tampilan = '📦 Packet Loss (%)'
                elif metrik == 'jitter':
                    nama_tampilan = '📶 Jitter (ms)'
                else:
                    nama_tampilan = metrik.title()
                
                tabel_prediksi[nama_tampilan] = np.round(prediksi, 2)
            
            # Reset index untuk tampilan yang lebih bersih
            tabel_prediksi.reset_index(drop=True, inplace=True)
            tabel_prediksi.insert(0, '🔢 No', range(1, len(tabel_prediksi) + 1))
            
            st.dataframe(tabel_prediksi, use_container_width=True)
            
            # Ringkasan akurasi dan variabel eksogen
            st.markdown("### 🎯 Ringkasan Model")
            
            col_akurasi, col_variabel = st.columns(2)
            
            with col_akurasi:
                st.markdown("#### 📊 Tingkat Akurasi")
                tabel_akurasi = pd.DataFrame([
                    {
                        '📊 Metrik': metrik.title(),
                        '🎯 Error (MAPE)': f"{akurasi:.1f}%",
                        '📈 Status': "Excellent" if akurasi < 20 else "Good" if akurasi < 40 else "Fair"
                    }
                    for metrik, akurasi in skor_akurasi.items()
                ])
                st.dataframe(tabel_akurasi, hide_index=True, use_container_width=True)
            
            # Interpretasi hasil
            rata_rata_akurasi = np.mean(list(skor_akurasi.values()))
            
            st.markdown("### 🎭 Interpretasi Hasil")
            
            if rata_rata_akurasi < 25:
                st.success(f"🎉 **Prediksi Sangat Akurat!** Sistem berhasil mengidentifikasi pola dengan rata-rata error hanya {rata_rata_akurasi:.1f}%")
            elif rata_rata_akurasi < 45:
                st.success(f"✅ **Prediksi Cukup Reliable!** Model menunjukkan performa baik dengan rata-rata error {rata_rata_akurasi:.1f}%")
            else:
                st.warning(f"⚠️**Model menunjukkan performa cukup** Rata-rata error {rata_rata_akurasi:.1f}%.")          
            
            
            # Download hasil
            data_csv = tabel_prediksi.to_csv(index=False, sep=';', decimal=',')
            st.download_button(
                label="📥 Download Hasil Prediksi (CSV)",
                data=data_csv,
                file_name=f"prediksi_auto_eksogen_{hari_prediksi}hari.csv",
                mime="text/csv",
                help="Download tabel prediksi dengan dalam format CSV"
            )
            

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {str(e)}")
        import traceback
        st.error(f"Detail error: {traceback.format_exc()}")

