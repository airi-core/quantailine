# Analisa Prediksi Waktu Trading dengan Tabel Square of 9
*SanClass Trading Labs dari hasil riset melalui berbagai sumber*

## Pendahuluan

Tabel Square of 9 adalah salah satu alat analisis teknikal yang dikembangkan oleh W.D. Gann, seorang trader legendaris yang terkenal dengan metode prediksi berbasis geometri dan waktu. Meskipun sering digunakan untuk analisis harga, tabel ini juga sangat efektif untuk menganalisis waktu (timing) dalam trading. Dalam dokumen ini, akan dijelaskan langkah demi langkah cara menggunakan Square of 9 untuk memprediksi waktu yang tepat untuk melakukan buy dan sell.

## Dasar-Dasar Tabel Square of 9

Tabel Square of 9 adalah susunan angka spiral yang dimulai dari angka 1 di tengah dan bergerak keluar dalam pola spiral searah jarum jam. Berikut adalah contoh dari tabel Square of 9:

```
--SanClass Trading  Labs--
65 64 63 62 61 60 59 58 57
66 37 36 35 34 33 32 31 56
67 38 17 16 15 14 13 30 55
68 39 18  5  4  3 12 29 54
69 40 19  6  1  2 11 28 53
70 41 20  7  8  9 10 27 52
71 42 21 22 23 24 25 26 51
72 43 44 45 46 47 48 49 50
73 74 75 76 77 78 79 80 81

--SanClass  Trading  Labs--
57 58 59 60 61 62 63 64 65
56 31 32 33 34 35 36 37 66
55 30 13 14 15 16 17 38 67
54 29 12  3  4  5 18 39 68
53 28 11  2  1  6 19 40 69
52 27 10  9  8  7 20 41 70
51 26 25 24 23 22 21 42 71
50 49 48 47 46 45 44 43 72
81 80 79 78 77 76 75 74 73


--SanClass Trading Labs--
81 80 79 78 77 76 75 74 73
50 49 48 47 46 45 44 43 72
51 26 25 24 23 22 21 42 71
52 27 10  9  8  7 20 41 70
53 28 11  2  1  6 19 40 69
54 29 12  3  4  5 18 39 68
55 30 13 14 15 16 17 38 67
56 31 32 33 34 35 36 37 66
57 58 59 60 61 62 63 64 65
```

Dalam tabel ini:
- Angka 1 berada di tengah
- Angka terus bertambah dalam pola spiral searah jarum jam
- Setiap lapisan membentuk "kuadran" atau "lingkaran" di sekitar pusat

## Prinsip Dasar Analisis Waktu dengan Square of 9

Dalam analisis waktu trading, prinsip dasarnya adalah:

1. **Sudut Kardinal**: 0°, 90°, 180°, 270° (atau 45°, 135°, 225°, 315° untuk sudut tambahan)
2. **Angka Kardinal**: 1, 2, 3, 4, 5, 7, 8, 9 (berbasis pada pembagian lingkaran)
3. **Waktu sebagai Derajat**: Dalam Square of 9, waktu direpresentasikan dalam derajat, dengan 24 jam setara dengan 360°

## Langkah-Langkah Analisis Waktu dengan Square of 9

### Langkah 1: Menentukan Titik Awal (Referensi)

Kita akan menggunakan contoh waktu low yang diberikan: **13:00 15/05/2025** dengan harga **3207.71**.

Langkah pertama adalah menjadikan waktu ini sebagai titik referensi untuk prediksi waktu selanjutnya.

### Langkah 2: Konversi Waktu ke Nilai Derajat

Konversi waktu ke derajat menggunakan formula:

```
Derajat = (Jam + Menit/60) × 15
```

Untuk jam 13:00:
```
Derajat = (13 + 0/60) × 15 = 195°
```

### Langkah 3: Menentukan Posisi pada Square of 9

Untuk menemukan posisi waktu referensi pada Square of 9, kita perlu mengkonversi derajat ke angka pada tabel.

Pertama, kita cari nilai yang sesuai dengan 195° pada tabel. Pada Square of 9, angka yang berada pada sudut 180° adalah 3, dan yang berada pada sudut 225° adalah 5. Maka, 195° berada di antara keduanya, yaitu sekitar angka 4.

### Langkah 4: Identifikasi Hubungan Kardinal dan Waktu Potensial

Dari titik referensi (13:00 dengan posisi ~195°), kita mencari waktu potensial untuk buy dan sell dengan mengidentifikasi hubungan sudut kardinal:

- **+90° (285°)**: Potensi reversal pertama (sekitar jam 19:00)
- **+180° (375° atau 15°)**: Potensi reversal kuat (sekitar jam 01:00 keesokan hari)
- **+270° (465° atau 105°)**: Potensi reversal (sekitar jam 07:00 keesokan hari)
- **+360° (555° atau 195°)**: Siklus penuh kembali (sekitar jam 13:00 keesokan hari)

### Langkah 5: Analisis Temporal dengan Fibonacci dan Gann

Selain menggunakan sudut kardinal, kita juga bisa menerapkan penghitungan Fibonacci dan prinsip Gann untuk waktu:

#### Formula Fibonacci untuk Waktu:
```
Waktu_Prediksi = Waktu_Referensi + (n × Faktor_Fibonacci)
```

Dimana Faktor_Fibonacci bisa berupa 0.618, 1.0, 1.618, 2.618, 4.236

Contoh perhitungan:
- Waktu reversal berpotensi = 13:00 + (1 × 1.618 jam) = 13:00 + ~1:37 = 14:37
- Waktu reversal berikutnya = 13:00 + (1 × 2.618 jam) = 13:00 + ~2:37 = 15:37

#### Formula Gann untuk Waktu:
```
Waktu_Prediksi = Waktu_Referensi + (n × 45 menit)
```

Dimana n adalah bilangan bulat 1, 2, 3, dst.

Contoh perhitungan:
- Waktu reversal pertama = 13:00 + (1 × 45 menit) = 13:45
- Waktu reversal kedua = 13:00 + (2 × 45 menit) = 14:30
- Waktu reversal ketiga = 13:00 + (3 × 45 menit) = 15:15

## Step-by-Step Analisis Lanjutan

### Langkah 6: Menentukan Siklus Harian

Untuk analisis siklus harian, kita bisa membagi 24 jam menjadi 8 bagian (setiap 3 jam):

1. **Pembagian Waktu Gann**:
   - 00:00 - 03:00: Kuadran 1
   - 03:00 - 06:00: Kuadran 2
   - 06:00 - 09:00: Kuadran 3
   - 09:00 - 12:00: Kuadran 4
   - 12:00 - 15:00: Kuadran 5 (waktu referensi kita ada di sini)
   - 15:00 - 18:00: Kuadran 6
   - 18:00 - 21:00: Kuadran 7
   - 21:00 - 24:00: Kuadran 8

2. **Analisis Level Energi Waktu**:
   
   Setiap kuadran memiliki "energi" yang berbeda berdasarkan teori Gann. Dengan titik referensi pada Kuadran 5, kita dapat mengidentifikasi kuadran yang berpotensi memiliki energi berlawanan:
   
   - Kuadran 1 (00:00-03:00): Energi berlawanan dengan Kuadran 5, berpotensi menjadi titik reversal
   - Kuadran 3 (06:00-09:00): Energi mendukung Kuadran 7, berpotensi menjadi titik kontinuasi
   - Kuadran 7 (18:00-21:00): Energi berlawanan dengan Kuadran 3, berpotensi menjadi titik reversal

### Langkah 7: Penerapan Rumus Square Roots untuk Waktu

Gann juga menggunakan akar kuadrat untuk menentukan titik-titik waktu penting:

```
Waktu_Prediksi = Waktu_Referensi ± √N × (Faktor_Waktu)
```

Dimana:
- N = 1, 2, 3, 4, dst.
- Faktor_Waktu biasanya adalah 60 menit (1 jam)

Contoh:
- Waktu potensial 1 = 13:00 + √1 × 60 menit = 13:00 + 1 × 60 = 14:00
- Waktu potensial 2 = 13:00 + √2 × 60 menit = 13:00 + 1.414 × 60 = 13:00 + ~85 menit = 14:25
- Waktu potensial 3 = 13:00 + √3 × 60 menit = 13:00 + 1.732 × 60 = 13:00 + ~104 menit = 14:44
- Waktu potensial 4 = 13:00 + √4 × 60 menit = 13:00 + 2 × 60 = 15:00

### Langkah 8: Metode Square of 9 untuk Perubahan Waktu

Untuk menghitung proyeksi waktu perubahan menggunakan metode Square of 9 secara langsung:

1. Temukan nilai waktu referensi pada tabel (13:00 = 195° = ~angka 4 pada tabel)
2. Identifikasi angka yang berada pada hubungan geometris penting:
   - Angka pada sudut 90° dari referensi (pergeseran +90° = 285°)
   - Angka pada sudut 180° dari referensi (pergeseran +180° = 15°)
   - Angka pada hubungan 45° (pergeseran +45° = 240°)

3. **Formula Konversi**:
   ```
   Jam = Derajat ÷ 15
   ```

   Contoh untuk 285°:
   ```
   Jam = 285 ÷ 15 = 19 jam (19:00)
   ```

## Contoh Analisis End-to-End untuk 13:00 15/05/2025

Berdasarkan waktu referensi 13:00 dengan harga 3207.71 (low), kita akan melakukan analisis lengkap:

### 1. Penentuan Titik Waktu Kritis:

- **Sudut 90°**: 13:00 + 6 jam = 19:00 (Potensi reversal - waktu untuk BUY)
- **Sudut 180°**: 13:00 + 12 jam = 01:00 16/05 (Potensi reversal kuat - waktu untuk SELL)
- **Sudut 270°**: 13:00 + 18 jam = 07:00 16/05 (Potensi reversal - waktu untuk BUY)
- **Sudut 360°**: 13:00 + 24 jam = 13:00 16/05 (Siklus penuh - konfirmasi tren)

### 2. Waktu Fibonacci:

- **0.618 × 24 jam**: 13:00 + 14.83 jam = 03:50 16/05 (Potensi koreksi)
- **1.618 × 24 jam**: 13:00 + 38.83 jam = 03:50 17/05 (Potensi reversal)

### 3. Proyeksi Waktu Gann:

- **45 menit**: 13:00 + 45m = 13:45 (Konfirmasi arah)
- **90 menit**: 13:00 + 90m = 14:30 (Potensi koreksi minor)
- **135 menit**: 13:00 + 135m = 15:15 (Potensi pembalikan gerakan)
- **180 menit**: 13:00 + 180m = 16:00 (Titik review tren)

### 4. Waktu Square Root:

- **√1 × 60m**: 13:00 + 60m = 14:00 (Konfirmasi gerakan)
- **√2 × 60m**: 13:00 + 85m = 14:25 (Potensi perubahan momentum)
- **√3 × 60m**: 13:00 + 104m = 14:44 (Potensi titik pivot)
- **√4 × 60m**: 13:00 + 120m = 15:00 (Konfirmasi tren jangka pendek)

### 5. Tabel Prediksi Waktu Penting:

| Waktu      | Tipe Sinyal             | Aksi Potensial | Keterangan                       |
|------------|-------------------------|----------------|----------------------------------|
| 13:45      | Konfirmasi Arah         | Monitor        | Konfirmasi arah setelah low      |
| 14:00      | Konfirmasi Gerakan      | Monitor        | Titik konfirmasi pertama         |
| 14:25      | Perubahan Momentum      | Monitor/Entry  | Potensi perubahan momentum       |
| 14:30      | Koreksi Minor           | Monitor/Entry  | Koreksi kecil, peluang entry     |
| 14:44      | Titik Pivot             | Entry/Exit     | Titik pembalikan potensial       |
| 15:00      | Konfirmasi Tren         | Entry/Exit     | Konfirmasi tren jangka pendek    |
| 15:15      | Pembalikan Gerakan      | Entry/Exit     | Potensi pembalikan arah utama    |
| 16:00      | Review Tren             | Exit/Evaluasi  | Evaluasi posisi dan tren         |
| 19:00      | Reversal Potensial      | BUY            | Sudut 90° - reversal potensial   |
| 01:00 16/05| Reversal Kuat           | SELL           | Sudut 180° - reversal kuat       |
| 03:50 16/05| Koreksi Fibonacci       | Entry/Exit     | Titik koreksi Fibonacci          |
| 07:00 16/05| Reversal Potensial      | BUY            | Sudut 270° - reversal potensial  |
| 13:00 16/05| Siklus Penuh            | Evaluasi       | Konfirmasi tren utama            |

## Petunjuk Praktis Analisa Waktu Square of 9

1. **Identifikasi Titik Referensi Penting**: 
   - Gunakan high, low, atau close harian sebagai titik referensi
   - Waktu pembukaan/penutupan sesi juga bisa dijadikan titik referensi

2. **Prioritaskan Sudut Kardinal**:
   - Sudut 90°, 180°, 270°, dan 360° adalah titik waktu paling kritis
   - Perhatikan juga sudut 45°, 135°, 225°, dan 315° sebagai konfirmasi tambahan

3. **Konfirmasi dengan Beberapa Metode**:
   - Gunakan minimal 2-3 metode analisis waktu untuk konfirmasi
   - Perhatikan ketika beberapa metode menunjuk ke titik waktu yang sama/berdekatan

4. **Hubungkan dengan Analisis Harga**:
   - Waktu reversal potensial harus dikonfirmasi dengan pergerakan harga
   - Perhatikan pada titik waktu kritis apakah terjadi perubahan momentum

5. **Buat Tabel Prediksi**:
   - Selalu siapkan tabel prediksi waktu untuk 24-48 jam ke depan
   - Tandai titik-titik waktu kritis beserta aksi potensial

## Kesimpulan

Analisis waktu menggunakan tabel Square of 9 memberikan perspektif unik dalam menentukan timing untuk entry dan exit dalam trading. Dengan menggabungkan prinsip geometri, astronomi, dan matematika yang digunakan oleh W.D. Gann, trader dapat mengidentifikasi titik-titik waktu kritis yang berpotensi menghasilkan perubahan arah pasar.

Untuk contoh kasus 13:00 15/05/2025 dengan harga 3207.71 (low), kita telah mengidentifikasi beberapa waktu kunci untuk diperhatikan, terutama 19:00 pada hari yang sama sebagai potensi titik BUY dan 01:00 16/05 sebagai potensi titik SELL.

Ingat bahwa analisis waktu harus selalu digunakan bersamaan dengan analisis harga dan volume untuk mendapatkan hasil yang optimal. Tabel Square of 9 bukanlah alat prediksi yang sempurna, tetapi menjadi alat tambahan yang kuat untuk mengidentifikasi titik-titik waktu dimana peluang perubahan arah pasar lebih tinggi.

*SanClass Trading Labs dari hasil riset melalui berbagai sumber*
