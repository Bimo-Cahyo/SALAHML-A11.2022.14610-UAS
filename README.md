

```markdown
# Prediksi Harga Emas Menggunakan Teknik Machine Learning

## Identitas Proyek

- **Nama**: Bimo Cahyo Widyanto
- **NIM**: A11.2022.14610
- **Kelompok**: A11.4412
- **Mata Kuliah**: Pembelajaran Mesin

## Ringkasan Proyek

Proyek ini bertujuan untuk memprediksi harga emas berdasarkan data historis dari berbagai sumber seperti ETF Emas, Indeks S&P 500, Dow Jones, dan lainnya. Harga emas dipengaruhi oleh banyak faktor dan fluktuasi pasar yang cepat, sehingga menentukan harga emas yang akurat merupakan tantangan yang memerlukan analisis data yang komprehensif.

## Tujuan Proyek

- Membangun model machine learning yang dapat memprediksi harga penutupan emas berdasarkan data historis yang tersedia.

## Dataset

Dataset yang digunakan adalah `FINAL_USO.csv`, yang berisi informasi historis tentang harga emas dan berbagai indikator pasar terkait. Dataset ini dimuat dan diperiksa sebagai langkah awal dalam analisis data.

### Contoh Memuat Data

```python
import pandas as pd

# Memuat dataset dari file CSV
df = pd.read_csv('FINAL_USO.csv')

# Menampilkan beberapa baris pertama dari dataset
print("Beberapa baris pertama dari dataset:")
print(df.head())
```

## Exploratory Data Analysis (EDA)

Langkah pertama dalam proyek ini adalah melakukan EDA untuk memahami data dan menemukan pola atau hubungan yang mungkin ada. Beberapa langkah EDA yang dilakukan termasuk:

- Memeriksa struktur dan informasi dataset
- Visualisasi data untuk melihat distribusi dan tren
- Menganalisis korelasi antar fitur

### Contoh Visualisasi

```python
import matplotlib.pyplot as plt

# Visualisasi distribusi harga emas
df['Gold_Price'].hist()
plt.title('Distribusi Harga Emas')
plt.xlabel('Harga Emas')
plt.ylabel('Frekuensi')
plt.show()
```

## Pemrosesan Fitur

Proses fitur melibatkan pemilihan dan transformasi fitur untuk meningkatkan performa model. Langkah-langkah yang dilakukan meliputi:

- Pemilihan fitur yang relevan
- Penanganan nilai hilang dengan metode imputasi
- Normalisasi atau standarisasi fitur
- Pembuatan fitur baru berdasarkan kombinasi fitur yang ada
- Pembagian dataset menjadi set pelatihan dan set pengujian

### Contoh Penanganan Nilai Hilang

```python
# Menangani nilai hilang dengan imputasi
df.fillna(method='ffill', inplace=True)
```

## Model Machine Learning

Beberapa model machine learning diuji dalam proyek ini untuk memprediksi harga emas, termasuk:

- Linear Regression
- Random Forest
- Support Vector Machine

Setiap model dievaluasi berdasarkan metrik performa tertentu untuk menentukan model yang paling efektif.

## Kesimpulan

Proyek ini berhasil membangun model machine learning yang dapat memprediksi harga emas dengan tingkat akurasi yang memadai. Langkah-langkah selanjutnya meliputi peningkatan model dan analisis lebih lanjut untuk menangani volatilitas pasar yang lebih kompleks.

## Referensi

- Sumber data: `FINAL_USO.csv`
- Teknik dan alat: Python, Pandas, Matplotlib, Scikit-learn


```
