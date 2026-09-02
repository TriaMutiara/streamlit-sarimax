import pandas as pd
import numpy as np
import math
import warnings
import logging
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


class SarimaxEksogenPrediktor:
    """Prediktor SARIMAX + eksogen (QoS internet).

    CATATAN METODOLOGI - INTERVAL NON-EQUIDISTANT (untuk sidang skripsi):
    Pengukuran hanya pada sesi jam 09:00/12:00/15:00, sehingga jeda fisik antar
    observasi TIDAK seragam (3 jam antar sesi siang, 18 jam melewati malam).
    Statespace SARIMAX statsmodels mengasumsikan langkah waktu reguler, jadi deret
    diperlakukan sebagai sekuens ordinal sesi t=1..T, BUKAN *time*-based likelihood.
    Konsekuensi:
      1. Periode musiman s dinyatakan dalam SATUAN SESI per hari (bukan jam), dihitung
         dari data: s = N_obs / N_hari (dataset contoh 66/22 = 3) -> satu siklus harian
         penuh.
      2. Differencing musiman dan order musiman beroperasi pada indeks sesi.
      3. `freq` DatetimeIndex dibiarkan None; identitas jam sesi dibawa variabel eksogen
         dummy (jam_09/jam_12/jam_15), bukan oleh interval waktu model.
    """
    def __init__(self, jam_prediksi_custom=None):
        self.scaler_eksogen = None
        self.exog_cont_cols = ['log_person']
        self.jam_prediksi = jam_prediksi_custom if jam_prediksi_custom else [9, 12, 15]
        self.selected_metric = None
        self.bobot_shrinkage = 0.0
        self.autocorr_lag1 = None
        self.sesi_per_hari = None

    def set_selected_metric(self, column):
        self.selected_metric = column

    def siapkan_variabel_eksogen(self, df):
        exog = pd.DataFrame(index=df.index)

        # 1. Fitur beban pengguna (kontinu): satu regressor (log_person)
        if 'orang' in df.columns:
            orang = pd.to_numeric(df['orang'], errors='coerce')
        elif 'sit_person' in df.columns:
            orang = pd.to_numeric(df['sit_person'], errors='coerce')
        else:
            orang = pd.Series(10.0, index=df.index)

        # Interpolasi linear HANYA pada variabel kontinu
        orang_interp = orang.interpolate(method='linear').ffill().bfill().fillna(10.0)
        exog['log_person'] = np.log1p(np.maximum(0.0, orang_interp.values))

        # 2. Fitur dummy sesi jam (09:00, 12:00, 15:00) dihitung eksak biner dari index / kolom
        if hasattr(df.index, 'hour'):
            jam = df.index.hour
        elif 'jam' in df.columns:
            jam = pd.to_numeric(df['jam'], errors='coerce').fillna(9).astype(int)
        else:
            jam = pd.Series(9, index=df.index)

        exog['jam_09'] = (jam == 9).astype(float)
        exog['jam_12'] = (jam == 12).astype(float)
        exog['jam_15'] = (jam == 15).astype(float)

        # 3. Fitur hari dalam seminggu (Senin=1 .. Minggu=7)
        if hasattr(df.index, 'dayofweek'):
            hari = df.index.dayofweek + 1
        elif 'hari_encoded' in df.columns:
            hari = pd.to_numeric(df['hari_encoded'], errors='coerce').fillna(1).astype(float)
        else:
            hari = pd.Series(1.0, index=df.index)
        exog['hari'] = hari.astype(float)

        # Kolom kontinu untuk standard scaling; dummy sesi & hari tetap biner tanpa standardisasi
        self.exog_cont_cols = [c for c in ['log_person'] if c in exog.columns]
        return exog

    def buat_eksogen_prediksi(self, waktu_prediksi, df_historis):
        exog_pred = pd.DataFrame(index=waktu_prediksi)
        exog_hist = self.siapkan_variabel_eksogen(df_historis)

        jam_pred = np.asarray(waktu_prediksi.hour)
        # Weekend = Sabtu(5)/Minggu(6); pola beban weekday != weekend
        tipe_hari_pred = (np.asarray(waktu_prediksi.dayofweek) >= 5).astype(int)

        if 'log_person' in exog_hist.columns and len(exog_hist) > 0:
            lp = exog_hist['log_person'].astype(float)
            mean_global = float(lp.mean())
            jam_hist = np.asarray(exog_hist.index.hour)
            tipe_hist = (np.asarray(exog_hist.index.dayofweek) >= 5).astype(int)

            # Rata-rata rolling beban pada sesi-sesi terakhir -> anchor level terkini
            n_rolling = int(min(max(len(self.jam_prediksi) * 2, 3), len(lp)))
            level_terkini = float(lp.rolling(n_rolling, min_periods=1).mean().iloc[-1])

            dev_jam = lp.groupby(jam_hist).mean() - mean_global
            n_jam = lp.groupby(jam_hist).count()
            dev_tipe = lp.groupby(tipe_hist).mean() - mean_global
            n_tipe = lp.groupby(tipe_hist).count()

            k_pseudo = 2.0

            def efek(dev, cnt, kunci):
                if kunci not in dev.index:
                    return 0.0
                c = float(cnt.loc[kunci])
                return float(dev.loc[kunci]) * (c / (c + k_pseudo))

            estimasi = np.asarray(
                [level_terkini + efek(dev_jam, n_jam, h) + efek(dev_tipe, n_tipe, t)
                 for h, t in zip(jam_pred, tipe_hari_pred)],
                dtype=float,
            )
            exog_pred['log_person'] = np.maximum(0.0, estimasi)
        else:
            exog_pred['log_person'] = 0.0

        exog_pred['jam_09'] = (jam_pred == 9).astype(float)
        exog_pred['jam_12'] = (jam_pred == 12).astype(float)
        exog_pred['jam_15'] = (jam_pred == 15).astype(float)
        exog_pred['hari'] = (np.asarray(waktu_prediksi.dayofweek) + 1).astype(float)

        exog_pred = exog_pred[exog_hist.columns]
        return exog_pred

    def hitung_error_evaluasi(self, actual, predicted, nama_column=None):
        """Metrik error baku per karakteristik metrik QoS:
        - Packet Loss (zero-inflated): NMAE berbasis rentang data.
        - Throughput / Upload / Download: sMAPE (simetris & robust terhadap variasi skala).
        - Latency, Jitter & default: WMAPE (Weighted MAPE berbasis total volume aktual).
        """
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        nama = (nama_column or '').lower()

        mae = float(np.mean(np.abs(actual - predicted)))

        # 1. Packet Loss (zero-inflated): NMAE berbasis rentang data
        if 'packet_loss' in nama:
            rentang = float(np.max(actual) - np.min(actual))
            return min(mae / rentang * 100.0, 100.0) if rentang > 1e-8 else (0.0 if mae < 1e-4 else 100.0)

        # 2. Throughput: sMAPE
        if any(k in nama for k in ['throughput', 'upload', 'download']):
            smape = 100.0 * np.mean(2.0 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted) + 1e-8))
            return min(float(smape), 100.0)

        # 3. Latency, Jitter, dan default: WMAPE
        total_actual = float(np.sum(np.abs(actual)))
        if total_actual > 1e-8:
            return min(float(np.sum(np.abs(actual - predicted)) / total_actual * 100.0), 100.0)
        return 0.0 if mae < 1e-4 else 100.0

    # Alias backward-compatible
    hitung_mape = hitung_error_evaluasi

    def cek_stasioneritas(self, series, nama=None):
        try:
            result = adfuller(series.dropna())
            return 0 if result[1] < 0.05 else 1
        except Exception:
            return 1

    def cari_parameter_optimal(self, data, eksogen):
        best_aic = math.inf
        dif = self.cek_stasioneritas(data)
        y_arr = np.asarray(data, dtype=float)
        x_arr = np.asarray(eksogen, dtype=float)

        # Periode musiman: jumlah sesi per hari dari data (non-equidistant -> satuan SESI)
        s = len(self.jam_prediksi) if self.jam_prediksi else 3
        if hasattr(data, 'index') and isinstance(data.index, pd.DatetimeIndex) and len(data) > 0:
            n_hari = max(1, data.index.normalize().nunique())
            s = max(1, int(round(len(data) / n_hari)))
        self.sesi_per_hari = s

        # Kandidat orde differencing musiman (D)
        candidate_seasonals = [(1, 0, 0, s), (0, 0, 1, s), (1, 0, 1, s), (0, 0, 0, 0)]
        if s > 1 and len(y_arr) > 2 * s:
            try:
                diff_seas = y_arr[s:] - y_arr[:-s]
                if adfuller(diff_seas)[1] < 0.05:
                    candidate_seasonals.insert(0, (1, 1, 1, s))
            except Exception:
                pass

        # Grid order SARIMAX
        candidate_orders = [
            (1, dif, 1), (1, dif, 0), (0, dif, 1), (2, dif, 1), (0, 0, 0),
            (2, dif, 0), (0, dif, 2), (2, dif, 2), (1, dif, 2)
        ]

        best_order = (0, dif, 0)
        best_seasonal = (0, 0, 0, s)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for order in candidate_orders:
                for seasonal in candidate_seasonals:
                    try:
                        result = SARIMAX(
                            y_arr,
                            exog=x_arr,
                            order=order,
                            seasonal_order=seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        ).fit(disp=False, maxiter=100)
                        if np.isfinite(result.aic) and result.aic < best_aic:
                            best_aic = result.aic
                            best_order = order
                            best_seasonal = seasonal
                    except Exception:
                        continue

        return best_order, best_seasonal

    def kembalikan_skala(self, pred_raw, use_log, y_raw, nama_kolom):
        pred = np.asarray(pred_raw, dtype=float)
        if use_log:
            max_allowed = np.log1p(float(np.max(y_raw)) * 2.5)
            pred = np.expm1(np.clip(pred, 0, max_allowed))
        if nama_kolom == 'packet_loss':
            pred = np.clip(pred, 0, 100)
        else:
            pred = np.maximum(0, pred)
        return pred

    def _level_sesi(self, y_hist, idx_hist, idx_target):
        """Median historis per jam-of-day (09/12/15) sebagai target shrinkage.

        Data sesi non-equidistant: tiap jam punya level berbeda. Median global skalar
        meratakan profil ini -> forecast ketiga sesi identik. Median per jam mempertahankan
        struktur sesi-of-day. Fallback ke median global bila jam belum pernah terlihat.
        """
        ser = pd.Series(np.asarray(y_hist, dtype=float), index=pd.DatetimeIndex(idx_hist))
        if ser.empty:
            return 0.0
        med_global = float(ser.median())
        med_by_hour = ser.groupby(ser.index.hour).median()
        out = np.asarray([
            float(med_by_hour.loc[t.hour]) if t.hour in med_by_hour.index else med_global
            for t in pd.DatetimeIndex(idx_target)
        ], dtype=float)
        return out

    def _pilih_bobot_shrinkage(self, y_fit, y_raw, exog, matriks_eksogen, n_train,
                                order, seasonal, nama_kolom, use_log):
        # Gate: hanya shrink bila lag-1 autocorrelation < 0.25 (noise-dominated)
        s_train = y_raw.iloc[:n_train].dropna()
        self.autocorr_lag1 = None
        if len(s_train) > 5:
            autocorr_1 = float(s_train.autocorr(lag=1))
            self.autocorr_lag1 = autocorr_1
            if np.isnan(autocorr_1) or autocorr_1 >= 0.25:
                return 0.0

        h = max(3, min(12, int(round(n_train * 0.2))))
        batas = sorted({int(f) for f in np.linspace(int(n_train * 0.5), n_train - h, 3)
                        if int(f) >= max(15, h)})
        if not batas:
            return 0.0

        bobot = np.arange(0.0, 1.01, 0.1)
        skor = np.zeros(len(bobot))
        terpakai = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for e in batas:
                v_end = e + h
                try:
                    fit = SARIMAX(
                        np.asarray(y_fit.iloc[:e], dtype=float),
                        exog=matriks_eksogen(exog.iloc[:e]),
                        order=order, seasonal_order=seasonal,
                        enforce_stationarity=False, enforce_invertibility=False
                    ).fit(disp=False, maxiter=100)
                    f_raw = np.asarray(fit.forecast(v_end - e, exog=matriks_eksogen(exog.iloc[e:v_end])))
                except Exception:
                    continue

                f_skala = self.kembalikan_skala(f_raw, use_log, y_raw, nama_kolom)
                level = self._level_sesi(y_raw.iloc[:e], y_raw.index[:e], y_raw.index[e:v_end])
                aktual = np.asarray(y_raw.iloc[e:v_end], dtype=float)
                for i, w in enumerate(bobot):
                    skor[i] += self.hitung_error_evaluasi(aktual, (1 - w) * f_skala + w * level, nama_kolom)
                terpakai += 1

        if terpakai == 0:
            return 0.0

        w_cv = float(bobot[int(np.argmin(skor))])
        # Target kini level per sesi-jam (profil 09/12/15 berulang), bukan garis datar,
        # sehingga CAP dinaikkan ke 0.9. CAP lama 0.5 melarang profil sesi terbentuk
        # padahal SARIMAX murni tidak menangkap sinyal nyata (autocorr lag-1 < 0.15).
        return min(w_cv, 0.9)

    def prediksi_dengan_eksogen(self, df, nama_kolom, waktu_prediksi):
        try:
            y_raw = df[nama_kolom].copy()
            n_train = max(1, min(int(len(y_raw) * 0.8), len(y_raw) - 1))
            if nama_kolom in ['latency', 'jitter']:
                seg = y_raw.iloc[:n_train]
                q1, q3 = seg.quantile([0.25, 0.75])
                upper_limit = max(q3 + 3.0 * (q3 - q1), float(seg.quantile(0.995)))
                y_train_clean = y_raw.copy()
                y_train_clean.iloc[:n_train] = y_train_clean.iloc[:n_train].clip(upper=upper_limit)
            elif any(k in nama_kolom.lower() for k in ['throughput', 'upload', 'download']):
                log_y = np.log1p(np.maximum(0, y_raw.iloc[:n_train]))
                q1, q3 = log_y.quantile([0.25, 0.75])
                iqr = q3 - q1
                upper = max(np.expm1(q3 + 3.0 * iqr), float(np.expm1(log_y.quantile(0.995))))
                lower = min(np.expm1(max(0.0, q1 - 3.0 * iqr)), float(np.expm1(log_y.quantile(0.005))))
                y_train_clean = y_raw.copy()
                y_train_clean.iloc[:n_train] = y_train_clean.iloc[:n_train].clip(lower=lower, upper=upper)
            else:
                y_train_clean = y_raw.copy()

            # Transformasi log1p untuk metrik berskala positif skewed (Latency, Jitter)
            use_log = nama_kolom in ['latency', 'jitter']
            y_fit = np.log1p(np.maximum(0, y_train_clean)) if use_log else y_train_clean.copy()

            exog = self.siapkan_variabel_eksogen(df)
            train_y, test_y = y_fit.iloc[:n_train], y_raw.iloc[n_train:]
            train_exog, test_exog = exog.iloc[:n_train], exog.iloc[n_train:]

            # Eksogen kontinu discaling dari segmen latih saja
            sc_x = StandardScaler()
            cont = self.exog_cont_cols
            sc_x.fit(train_exog[cont])
            self.scaler_eksogen = sc_x

            def matriks_eksogen(df_e):
                arr = df_e.astype(float).copy()
                arr[cont] = sc_x.transform(arr[cont])
                return arr.values

            train_exog_s = matriks_eksogen(train_exog)

            # Pencarian parameter optimal otomatis berbasis AIC
            best_order, best_seasonal = self.cari_parameter_optimal(train_y, train_exog_s)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    np.asarray(train_y, dtype=float),
                    exog=train_exog_s,
                    order=best_order,
                    seasonal_order=best_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                model_fit = model.fit(disp=False, maxiter=200)

            # Bobot kombinasi SARIMAX + level median
            w_shrink = self._pilih_bobot_shrinkage(
                y_fit, y_raw, exog, matriks_eksogen, n_train,
                best_order, best_seasonal, nama_kolom, use_log)
            self.bobot_shrinkage = w_shrink
            level_train = float(np.median(y_raw.iloc[:n_train]))
            in_sample_resid = model_fit.resid

            # Evaluasi Uji (Test Validation, kronologis 80/20)
            pred_test_raw = model_fit.forecast(len(test_y), exog=matriks_eksogen(test_exog))
            pred_test = self.kembalikan_skala(pred_test_raw, use_log, y_raw, nama_kolom)
            if w_shrink > 0:
                level_test = self._level_sesi(y_raw.iloc[:n_train], y_raw.index[:n_train], y_raw.index[n_train:])
                pred_test = (1 - w_shrink) * pred_test + w_shrink * level_test
            akurasi = self.hitung_error_evaluasi(test_y.values, pred_test, nama_kolom)
            resid_test = np.asarray(test_y.values, dtype=float) - np.asarray(pred_test, dtype=float)
            mae_test = float(np.mean(np.abs(resid_test)))
            rmse_test = float(np.sqrt(np.mean(resid_test ** 2)))

            # Refit parameter terbaik pada 100% data (train+test) sebelum peramalan masa depan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_full = SARIMAX(
                    np.asarray(y_fit, dtype=float),
                    exog=matriks_eksogen(exog),
                    order=best_order,
                    seasonal_order=best_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                model_full_fit = model_full.filter(model_fit.params)

            # Peramalan Masa Depan (Future Forecasting)
            exog_pred = self.buat_eksogen_prediksi(waktu_prediksi, df)
            pred_future_raw = model_full_fit.forecast(len(waktu_prediksi), exog=matriks_eksogen(exog_pred))
            pred_future = self.kembalikan_skala(pred_future_raw, use_log, y_raw, nama_kolom)
            if w_shrink > 0:
                level_full = self._level_sesi(y_raw, y_raw.index, waktu_prediksi)
                pred_future = (1 - w_shrink) * pred_future + w_shrink * level_full

            # Validasi batas bawah dan atas distribusi data historis
            series_hist = pd.to_numeric(df[nama_kolom], errors='coerce').dropna()
            if len(series_hist) > 0:
                lb = max(0.0, float(np.percentile(series_hist, 1)))
                ub = float(np.percentile(series_hist, 99))
                if any(k in nama_kolom.lower() for k in ['throughput', 'upload', 'download']):
                    pred_future = np.clip(pred_future, lb, ub)
                elif nama_kolom in ['latency', 'jitter']:
                    pred_future = np.clip(pred_future, lb, ub * 1.5)

            diagnostik = {
                'order': best_order,
                'seasonal_order': best_seasonal,
                'bobot_shrinkage': w_shrink,
                'in_sample_resid': in_sample_resid,
                'aic': float(model_fit.aic) if np.isfinite(model_fit.aic) else None,
                'mae': mae_test,
                'rmse': rmse_test,
                'autocorr_lag1': self.autocorr_lag1,
                'sesi_per_hari': self.sesi_per_hari,
                'level_median': level_train if w_shrink > 0 else None,
            }

            return pred_future, akurasi, exog_pred, test_y, test_exog, pred_test, diagnostik
        except Exception as e:
            logger.error(f"Error dalam prediksi {nama_kolom}: {str(e)}", exc_info=True)
            return None, 0.0, None, None, None, None, None

    def dataframe_prediksi(self, waktu_terakhir, jumlah_hari):
        dataframe_forcast = []
        for hari in range(1, jumlah_hari + 1):
            tanggal_berikutnya = waktu_terakhir + pd.Timedelta(days=hari)
            for jam in self.jam_prediksi:
                waktu_prediksi = tanggal_berikutnya.replace(hour=jam, minute=0, second=0, microsecond=0)
                dataframe_forcast.append(waktu_prediksi)
        return pd.DatetimeIndex(dataframe_forcast)
