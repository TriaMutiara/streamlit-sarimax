# Arima-QoS
Penelitian QoS dengan Model ARIMA

## Cara Instalasi dan Menjalankan Aplikasi

### Langkah 1: Instalasi Dependensi
Instal semua paket Python yang diperlukan dengan menjalankan:
```bash
pip install -r requirements.txt
```
Perintah ini akan menginstal:
- pandas (untuk manipulasi data)
- matplotlib (untuk visualisasi data)
- statsmodels (untuk model SARIMAX)
- streamlit (untuk antarmuka web)
- openpyxl (untuk membaca file Excel)

### Langkah 2: Menjalankan Aplikasi
Setelah semua dependensi terinstal, jalankan aplikasi dengan perintah:
```bash
streamlit run main.py
```

### Langkah 3: Menggunakan Aplikasi
1. Aplikasi akan otomatis terbuka di browser web Anda (biasanya di http://localhost:8501)
2. Unggah file Excel yang berisi data QoS dengan format:
   - Kolom 'Tanggal' (format YYYY-MM-DD)
   - Kolom 'Jam' (format HH.MM)
   - Kolom metrik QoS (latency, packet_loss, jitter, upload, download)
3. Pilih kolom yang ingin diprediksi (upload atau download)
4. Klik tombol "Proses" untuk memulai prediksi
5. Lihat hasil prediksi dan visualisasi data
6. Unduh hasil prediksi dalam format CSV jika diperlukan

### Format Data Excel
Pastikan file Excel Anda memiliki struktur sebagai berikut:
- Tanggal: Tanggal pengukuran (contoh: 2023-01-01)
- Jam: Waktu pengukuran (contoh: 08.00)
- latency: Nilai latency dalam ms
- packet_loss: Nilai packet loss dalam %
- jitter: Nilai jitter dalam ms
- upload: Nilai upload speed dalam Mbps
- download: Nilai download speed dalam Mbps

### Contoh File
Anda dapat menggunakan file `tetst-drive ISP.xlsx` sebagai contoh data untuk mencoba aplikasi.


## Penjelasan Plot:
### Data Pelatihan (Garis biru putus-putus):
- Menampilkan data historis yang digunakan untuk melatih model SARIMAX
- Mewakili periode sebelum pembagian dataset

### Data Pengujian (Garis abu-abu putus-putus):
- Menampilkan data aktual dari periode pengujian
- Digunakan untuk memvalidasi akurasi pengujian model

### Nilai Disesuaikan (Garis oranye):
- Nilai hasil fitting model pada data pelatihan
- Menunjukkan seberapa baik model menangkap pola dalam data pelatihan

### Peramalan (Titik-titik hijau):
- Prediksi untuk periode masa depan
- Ditampilkan sebagai titik dengan marker 'o'
- Periode ini tidak termasuk dalam data asli
