import pandas as pd
import numpy as np

def load_data(uploaded_file):
    file_name = uploaded_file.name if hasattr(uploaded_file, 'name') else str(uploaded_file)
    if file_name.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
            if df.shape[1] <= 1:
                if hasattr(uploaded_file, 'seek'):
                    uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
            return df
        except Exception:
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8')
    else:
        return pd.read_excel(uploaded_file)

def clean_data(df):
    data_bersih = df.copy()
    data_bersih = data_bersih.dropna(how='all')
    data_bersih.columns = data_bersih.columns.astype(str).str.strip().str.lower().str.replace(' ', '_')

    if 'tanggal' in data_bersih.columns and 'jam' in data_bersih.columns:
        waktu_gabungan = data_bersih['tanggal'].astype(str) + ' ' + data_bersih['jam'].astype(str)
        date_formats = [
            '%d/%m/%Y %H:%M', '%d/%m/%Y %H.%M',
            '%d-%m-%Y %H:%M', '%d-%m-%Y %H.%M',
            '%Y-%m-%d %H:%M', '%Y-%m-%d %H.%M',
            '%Y/%m/%d %H:%M', '%Y/%m/%d %H.%M',
            '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y'
        ]
        
        parsed_waktu = None
        for fmt in date_formats:
            try:
                parsed_waktu = pd.to_datetime(waktu_gabungan, format=fmt, errors='raise')
                break
            except (ValueError, TypeError):
                continue
        
        if parsed_waktu is None:
            parsed_waktu = pd.to_datetime(waktu_gabungan, dayfirst=True, errors='coerce')

        data_bersih['waktu'] = parsed_waktu
        data_bersih = data_bersih.dropna(subset=['waktu'])
        data_bersih.set_index('waktu', inplace=True)
        data_bersih = data_bersih.drop(['tanggal', 'jam', 'hari'], axis=1, errors='ignore')

    # Pembersihan kolom numerik
    kolom_angka = ['throughput', 'latency', 'jitter', 'packet_loss', 'orang', 'upload', 'download']
    for kolom in kolom_angka:
        if kolom in data_bersih.columns:
            if data_bersih[kolom].dtype == 'object':
                data_bersih[kolom] = data_bersih[kolom].astype(str).str.replace(',', '.')
            data_bersih[kolom] = pd.to_numeric(data_bersih[kolom], errors='coerce')

    # Filter baris di mana seluruh metrik QoS utama bernilai NaN
    kolom_penting = [col for col in ['throughput', 'latency', 'packet_loss', 'jitter', 'upload', 'download'] if col in data_bersih.columns]
    if kolom_penting:
        data_bersih = data_bersih.dropna(subset=kolom_penting, how='all')

    data_bersih = data_bersih.interpolate(method='linear').ffill().bfill()

    # Session Resampling: Jika data memiliki multiple pengukuran per jam (burst 5-menitan),
    # agregasikan rata-rata per sesi jam untuk konsistensi time series reguler.
    if isinstance(data_bersih.index, pd.DatetimeIndex):
        # Cek apakah terdapat multiple data points dalam jam yang sama
        jam_counts = data_bersih.index.floor('h').value_counts()
        if len(jam_counts) > 0 and jam_counts.max() > 1:
            data_bersih['session_time'] = data_bersih.index.floor('h')
            numeric_cols = data_bersih.select_dtypes(include=[np.number]).columns
            agg_dict = {c: 'mean' for c in numeric_cols if c != 'session_time'}
            data_bersih = data_bersih.groupby('session_time').agg(agg_dict)
            data_bersih.index.name = 'waktu'

    # Variabel eksogen berbasis waktu dari DatetimeIndex
    if hasattr(data_bersih.index, 'hour'):
        data_bersih['jam'] = data_bersih.index.hour.astype(float)
    if hasattr(data_bersih.index, 'dayofweek'):
        data_bersih['hari_encoded'] = (data_bersih.index.dayofweek + 1).astype(float)

    return data_bersih


