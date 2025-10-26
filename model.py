import pandas as pd
import numpy as np
import math
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

class SarimaxEksogenPrediktor:
    def __init__(self, jam_prediksi_custom=None):
        self.scaler_data = {}
        self.scaler_eksogen = {}
        self.jam_prediksi = jam_prediksi_custom if jam_prediksi_custom else [9, 13, 17]
        self.selected_metric = None

    def set_selected_metric(self, column):
        self.selected_metric = column

    def handle_median_outlier(self, series):
        # Transformasi log untuk mengurangi skewness
        log_series = np.log1p(series)
        
        # Deteksi outlier pada data yang ditransformasi
        q1, q3 = log_series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        # Identifikasi outlier
        outlier_mask = (log_series < lower) | (log_series > upper)
        n_outliers = outlier_mask.sum()
        
        if n_outliers > 0:
            # Ganti outlier dengan median per jam
            median_per_jam = series.groupby(series.index.hour).transform('median')
            series_cleaned = series.copy()
            series_cleaned[outlier_mask] = median_per_jam[outlier_mask]
            st.info(f"Outlier upload: {n_outliers} nilai diganti dengan median per jam")
            return series_cleaned
        
        return series

    def handler_clip_outlier(self, series):
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        series_cleaned = series.clip(lower=lower, upper=upper)
        n_outliers = ((series < lower) | (series > upper)).sum()
        if n_outliers > 0:
            st.info(f"Outlier handling: {n_outliers} nilai ekstrem di-clip (range: {lower:.1f} - {upper:.1f})")
        return series_cleaned
    
    def cek_stasioneritas(self, series, nama):
        result = adfuller(series.dropna())
        if result[1] < 0.05:
            st.success(f"Data **{nama}** Sudah Stasioner (p-value: {result[1]:.4f})")
            return 0
        else:
            st.warning(f"**{nama}** Non-stasioner (p-value: {result[1]:.4f}) → otomatis differencing 1x")
            return 1

    def normalisasi_data(self, data, nama_kolom):
        scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        data_ternormalisasi = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
        self.scaler_data[nama_kolom] = scaler
        return pd.Series(data_ternormalisasi, index=data.index)

    def normalisasi_eksogen(self, data_eksogen):
        data_eksogen_bersih = data_eksogen.copy()
        data_eksogen_bersih = data_eksogen_bersih.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
        data_eksogen_bersih = data_eksogen_bersih.replace([np.inf, -np.inf], np.nan)
        data_eksogen_bersih = data_eksogen_bersih.fillna(data_eksogen_bersih.mean())
        if data_eksogen_bersih.isnull().any().any():
            data_eksogen_bersih = data_eksogen_bersih.fillna(0)
        scaler = MinMaxScaler(feature_range=(0, 1))
        kolom_eksogen = data_eksogen_bersih.columns
        data_ternormalisasi = scaler.fit_transform(data_eksogen_bersih)
        self.scaler_eksogen = scaler
        return pd.DataFrame(data_ternormalisasi, index=data_eksogen_bersih.index, columns=kolom_eksogen)

    def kembalikan_skala_asli(self, data, nama_kolom):
        if nama_kolom in self.scaler_data:
            return self.scaler_data[nama_kolom].inverse_transform(data.reshape(-1, 1)).flatten()
        return data

    def siapkan_variabel_eksogen(self, df):
        eksogen = pd.DataFrame(index=df.index)
        if 'hari_encoded' in df.columns:
            eksogen['hari'] = df['hari_encoded']
        if 'jam' in df.columns:
            eksogen['jam'] = df['jam']
        if 'sit_person' in df.columns:
            eksogen['sit_person'] = pd.to_numeric(df['sit_person'], errors='coerce').round().fillna(0).astype(int)
        if 'jam' in eksogen.columns:
            eksogen['jam_sibuk'] = eksogen['jam'].apply(lambda x: 1 if x in [9.0, 13.0, 17.0] else 0)
        if 'sit_person' in eksogen.columns:
            eksogen['total_person'] = eksogen['sit_person']
        eksogen = eksogen.replace([np.inf, -np.inf], np.nan)
        eksogen = eksogen.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
        if eksogen.isnull().any().any():
            eksogen = eksogen.fillna(0)
        return eksogen

    def buat_eksogen_prediksi(self, waktu_prediksi, df_historis):
        eksogen_pred = pd.DataFrame(index=waktu_prediksi)
        eksogen_hist = self.siapkan_variabel_eksogen(df_historis)
        if 'hari' in eksogen_hist.columns:
            eksogen_pred['hari'] = waktu_prediksi.dayofweek + 1
        if 'jam' in eksogen_hist.columns:
            eksogen_pred['jam'] = waktu_prediksi.hour.astype(float)
        for kolom in ['sit_person']:
            if kolom in eksogen_hist.columns:
                nilai_prediksi = []
                for waktu in waktu_prediksi:
                    jam_sama = eksogen_hist[eksogen_hist.index.hour == waktu.hour]
                    if len(jam_sama) > 0:
                        rata_rata = jam_sama[kolom].mean()
                    else:
                        rata_rata = eksogen_hist[kolom].mean()
                    nilai_prediksi.append(rata_rata)
                eksogen_pred[kolom] = pd.Series(nilai_prediksi, index=waktu_prediksi).round().astype(int)
        if 'jam' in eksogen_pred.columns:
            eksogen_pred['jam_sibuk'] = eksogen_pred['jam'].apply(lambda x: 1 if x in [9.0, 13.0, 17.0] else 0)
        if 'sit_person' in eksogen_pred.columns:
            eksogen_pred['total_person'] = eksogen_pred['sit_person']
        eksogen_pred = eksogen_pred.replace([np.inf, -np.inf], np.nan)
        eksogen_pred = eksogen_pred.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
        if eksogen_pred.isnull().any().any():
            eksogen_pred = eksogen_pred.fillna(0)
        return eksogen_pred

    def cari_parameter_optimal(self, data, eksogen, nama_kolom):
        best_aic = math.inf
        dif = self.cek_stasioneritas(data, nama_kolom)
        best_order = []
        best_seasonal = []
        
        obs_per_day = 3 ## 9 13 17
        obs_per_week = 7 * obs_per_day
        candidate_orders = [ (0, dif, 1),(1, dif, 0),(1, dif, 1), (1, dif, 2),(2, dif, 1),(2, dif, 2)]
        candidate_seasonals = [(1, dif, 0, obs_per_day),(1, dif, 1, obs_per_day),(1, dif, 0, obs_per_week),(1, dif, 1, obs_per_week)]
        
        for order in candidate_orders:
            for seasonal in candidate_seasonals:
                p, d, q = order
                P, D, Q, s = seasonal
                try:
                    model = SARIMAX(
                        data,
                        exog=eksogen,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, s)
                    )
                    result = model.fit(disp=False, maxiter=500)
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = order
                        best_seasonal = seasonal
                except Exception as e:
                    continue
        st.info(f"Parameter optimal untuk {nama_kolom}: order={best_order}, seasonal_order={best_seasonal}, AIC={best_aic:.2f}")
        return best_order, best_seasonal

    def prediksi_dengan_eksogen(self, df, nama_kolom, waktu_prediksi):
        try:
            df_cleaned = df.copy()
            if 'upload' in nama_kolom.lower():
                df_cleaned[nama_kolom] = self.handle_median_outlier(df[nama_kolom])
            elif 'packet_loss' not in nama_kolom.lower():
                df_cleaned[nama_kolom] = self.handler_clip_outlier(df[nama_kolom])
            eksogen_historis = self.siapkan_variabel_eksogen(df_cleaned)
            data_normal = self.normalisasi_data(df_cleaned[nama_kolom], nama_kolom)
            eksogen_normal = self.normalisasi_eksogen(eksogen_historis)
            ukuran_train = int(len(data_normal) * 0.8)
            data_train, data_test = data_normal[:ukuran_train], data_normal[ukuran_train:]
            eksogen_train, eksogen_test = eksogen_normal[:ukuran_train], eksogen_normal[ukuran_train:]
            parameter_optimal, musiman_optimal = self.cari_parameter_optimal(data_train, eksogen_train, nama_kolom)
            model = SARIMAX(
                data_train,
                exog=eksogen_train,
                order=parameter_optimal,
                seasonal_order=musiman_optimal
            )
            model_fit = model.fit(disp=False, maxiter=500)
            akurasi = 0.0

            if len(data_test) > 0:
                prediksi_test = model_fit.forecast(len(data_test), exog=eksogen_test)
                prediksi_asli = self.kembalikan_skala_asli(prediksi_test.values, nama_kolom)
                test_asli = self.kembalikan_skala_asli(data_test.values, nama_kolom)
                akurasi = self.hitung_mape(test_asli, prediksi_asli, nama_kolom)
            eksogen_prediksi = self.buat_eksogen_prediksi(waktu_prediksi, df_cleaned)

            if eksogen_prediksi.isnull().any().any() or np.isinf(eksogen_prediksi.values).any():
                st.warning(f"Data eksogen prediksi untuk {nama_kolom} mengandung nilai tidak valid")
                eksogen_prediksi = eksogen_prediksi.replace([np.inf, -np.inf], np.nan)
                eksogen_prediksi = eksogen_prediksi.fillna(eksogen_prediksi.mean())
                if eksogen_prediksi.isnull().any().any():
                    eksogen_prediksi = eksogen_prediksi.fillna(0)
            try:
                eksogen_pred_normal = pd.DataFrame(
                    self.scaler_eksogen.transform(eksogen_prediksi),
                    index=eksogen_prediksi.index,
                    columns=eksogen_prediksi.columns
                )
            except Exception as e:
                st.error(f"Error dalam normalisasi eksogen prediksi: {str(e)}")
            prediksi_masa_depan = model_fit.forecast(len(waktu_prediksi), exog=eksogen_pred_normal)
            hasil_prediksi = self.kembalikan_skala_asli(prediksi_masa_depan.values, nama_kolom)
            
            # Validasi berbasis distribusi historis
            if nama_kolom in ['upload', 'download'] and nama_kolom in df.columns:
                series_hist = pd.to_numeric(df[nama_kolom], errors='coerce').dropna()
                if len(series_hist) > 0:
                    # Batas berdasarkan percentil 5-95
                    lb = np.percentile(series_hist, 5)
                    ub = np.percentile(series_hist, 95)
                    hasil_prediksi = np.clip(hasil_prediksi, lb, ub)
                    st.info(f"Prediksi {nama_kolom} di-clip antara {lb:.2f} dan {ub:.2f}")
            
            return hasil_prediksi, akurasi, eksogen_prediksi, data_test, eksogen_test, prediksi_test
        except Exception as e:
            st.error(f"Error dalam prediksi {nama_kolom} dengan eksogen: {str(e)}")
            return None, 0, None, None, None, None, None

    def _validasi_hasil_prediksi(self, prediksi, nama_kolom, lower_bound=None):
        if 'packet_loss' in nama_kolom.lower():
            prediksi = np.clip(prediksi, 0, 10)
        elif 'jitter' in nama_kolom.lower():
            prediksi = np.maximum(prediksi, 0)
        elif 'latency' in nama_kolom.lower():
            prediksi = np.maximum(prediksi, 1)
        elif nama_kolom in ['upload', 'download']:
            if lower_bound is None:
                lower_bound = 0.1
            prediksi = np.maximum(prediksi, lower_bound)
        return prediksi

    def hitung_mape(self, actual, predicted, nama_column=None):
        actual, predicted = np.array(actual), np.array(predicted)
        
        if 'upload' in (nama_column or '').lower() or 'download' in (nama_column or '').lower():
            smape = 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted) + 1e-8))
            return min(smape, 100.0)
        
        # Metode lama untuk column lainnya
        if nama_column and ('latency' in nama_column.lower() or 'jitter' in nama_column.lower()):
            q1, q3 = np.percentile(actual, [25, 75])
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (actual >= lower_bound) & (actual <= upper_bound)
            if np.sum(mask) >= len(actual) * 0.5:
                actual_filtered, predicted_filtered = actual[mask], predicted[mask]
                total_actual = np.sum(np.abs(actual_filtered))
                if total_actual > 0:
                    wmape = np.sum(np.abs(actual_filtered - predicted_filtered)) / total_actual * 100
                    return min(wmape, 100.0)
            mae = np.mean(np.abs(actual - predicted))
            mean_actual = np.mean(actual)
            return min((mae / mean_actual) * 100, 100.0) if mean_actual > 0 else 100.0
        
        persen_nol = np.sum(actual == 0) / len(actual)
        if persen_nol > 0.8:
            mae = np.mean(np.abs(actual - predicted))
            if mae < 0.1: return 5.0
            elif mae < 0.5: return 15.0
            elif mae < 1.0: return 25.0
            else: return 50.0
        
        if 'upload' in (nama_column or '').lower() or 'download' in (nama_column or '').lower():
            median_actual = np.median(actual)
            medae = np.median(np.abs(actual - predicted))
            return min((medae / median_actual) * 100, 100.0) if median_actual > 0 else 100.0
        
        mask = np.abs(actual) > np.percentile(np.abs(actual), 5)
        if np.sum(mask) == 0: return 100.0
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        return min(mape, 100.0)

    def dataframe_prediksi(self, waktu_terakhir, jumlah_hari):
        dataframe_forcast = []
        for hari in range(1, jumlah_hari + 1):
            tanggal_berikutnya = waktu_terakhir + pd.Timedelta(days=hari)
            for jam in self.jam_prediksi:
                waktu_prediksi = tanggal_berikutnya.replace(hour=jam, minute=0, second=0, microsecond=0)
                dataframe_forcast.append(waktu_prediksi)
        return pd.DatetimeIndex(dataframe_forcast)
