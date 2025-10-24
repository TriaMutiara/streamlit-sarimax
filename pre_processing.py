import pandas as pd

def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8')
    else:
        return pd.read_excel(uploaded_file)

def clean_data(df):
    data_bersih = df.copy()
    data_bersih = data_bersih.dropna(how='all')
    data_bersih.columns = data_bersih.columns.str.strip()
    kolom_eksogen_mapping = {}

    if 'Tanggal' in data_bersih.columns and 'Jam' in data_bersih.columns:
        waktu_gabungan = data_bersih['Tanggal'].astype(str) + ' ' + data_bersih['Jam'].astype(str)
        # Try different date formats
        for format_tanggal in ['%d/%m/%Y %H.%M', '%Y-%m-%d %H.%M', '%d-%m-%Y %H.%M']:
            try:
                data_bersih['waktu'] = pd.to_datetime(waktu_gabungan, format=format_tanggal, errors='raise')
                break
            except ValueError:
                continue
        else: # Fallback
            data_bersih['waktu'] = pd.to_datetime(waktu_gabungan, errors='coerce')

        data_bersih = data_bersih.dropna(subset=['waktu'])
        data_bersih.set_index('waktu', inplace=True)
        data_bersih = data_bersih.drop(['Tanggal', 'Jam', 'Hari'], axis=1, errors='ignore')

    # Add back exogenous columns
    for nama_baru, nilai in kolom_eksogen_mapping.items():
        data_bersih[nama_baru] = nilai.values
    data_bersih = data_bersih.drop(['sit person'], axis=1, errors='ignore')

    # Clean numeric columns
    kolom_angka = ['upload', 'download', 'latency', 'packet_loss', 'jitter', 'sit_person']
    for kolom in kolom_angka:
        if kolom in data_bersih.columns:
            if data_bersih[kolom].dtype == 'object':
                data_bersih[kolom] = data_bersih[kolom].astype(str).str.replace(',', '.')
            data_bersih[kolom] = pd.to_numeric(data_bersih[kolom], errors='coerce')

    # Special handling for 'jam' column if it exists separately
    if 'jam' in data_bersih.columns:
        if data_bersih['jam'].dtype == 'object':
            data_bersih['jam'] = data_bersih['jam'].astype(str).str.replace(',', '.')
            data_bersih['jam'] = pd.to_numeric(data_bersih['jam'], errors='coerce')

    # Drop rows where all important QoS metrics are missing
    kolom_penting = [col for col in ['upload', 'download', 'latency', 'packet_loss', 'jitter'] if col in data_bersih.columns]
    data_bersih = data_bersih.dropna(subset=kolom_penting, how='all')

    # Interpolate and fill missing values
    data_bersih = data_bersih.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')

    # Create time-based exogenous variables from the index
    if hasattr(data_bersih.index, 'hour'):
        data_bersih['jam'] = data_bersih.index.hour.astype(float)
    if hasattr(data_bersih.index, 'dayofweek'):
        # Monday=0, Sunday=6 -> convert to 1-7
        data_bersih['hari_encoded'] = (data_bersih.index.dayofweek + 1).astype(float)

    return data_bersih
