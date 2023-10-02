# Laporan Proyek Machine Learning - Daffa Albari

## Domain Proyek

Domain proyek ini akan membahas mengenai permasalahan dalam bidang kesehatan yang dibuat untuk mengetahui prediksi status diabetes seseorang.
![image](https://github.com/daaffalbari/diabetic-prediction/assets/73302268/24909b6b-ca98-4a66-9e3f-9b32aa02645b)

Diabetes merupakan suatu penyakit tidak menular yang cukup serius di mana pankreas tidak dapat memproduksi insulin secara maksimal [1]. Diabetes dapat menyerang siapa saja tanpa mengenal usia baik lansia, orang dewasa, maupun anak-anak yang ditandai dengan meningkatnya kadar gula (glukosa) dalam tubuh manusia. 

Diabetes dapat disebabkan oleh banyak faktor seperti tekanan darah tinggi, kadar gula, berat badan, umur, pola hidup, dan juga gender seseorang [2]. Faktor-faktor tersebut merupakan variabel yang digunakana dalam penugasan ini untuk membuat model machine learning yang dapat memprediksi penyakit diabetes.

Machine learning merupakan bagian dari kecerdasan buatan yang mampu mempelajari data dengan sendirinya. Machine learning adalah suatu model statistika untuk memprediksi data dengan menggunakan komputer. Beberapa algoritma yang digunakan adalah KNN, Random Forest, dan Decision Tree.

Berdasarkan data dan latar belakang di atas, maka di dalam proyek ini akan dibuat sebuah model *machine learning* untuk melakukan analisis prediksi diabetes berdasarkan medical history seseorang. Dengan adanya model *machine learning* yang telah dibangun, diharapkan dapat membantu dalam memprediksi apakah bisa terkena diabetes dari faktor-faktor yang disebutkan di atas.

# Business Understanding

## Problem Statements

Berdasarkan latar belakang yang telah dijelaskan di atas, maka diperoleh rumusan masalah yang akan diselesaikan pada proyek ini, yaitu:
1. Bagaimana cara melakukan tahap persiapan data sebelum digunakan untuk membuat model *machine learning*?
2. Bagaimana cara membuat model *machine learning* untuk melakukan prediksi Diabetes?

## Goals

Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka didapatkan tujuan dari proyek ini, yaitu:
1. Melakukan tahap persiapan data (*data preparation*) sehingga data dapat digunakan pada model *machine learning* dengan baik.
2. Membuat model *machine learning* untuk melakukan analisis prediksi diabetes dengan tingkat *error* yang cukup rendah.

## Solution Statements

Berdasarkan penjelasan di atas, terdapat beberapa solusi yang dapat dilakukan untuk dapat mencapai tujuan dari proyek ini, yaitu:
1. Tahap persiapan data (*data preparation*) dapat   dengan beberapa teknik, sebagai berikut:
   - Melakukan pembagian data menjadi 2, yaitu data latih (*training data*) dan data uji (*testing data*) dengan perbandingan rasio sebesar 90 : 10 yang akan digunakan ketika membangun model *machine learning*.
   - Melakukan standarisasi nilai pada data fitur numerik untuk mencegah terjadinya penyimpangan nilai data yang cukup besar.
2. Tahap pembuatan model *machine learning* akan digunakan 3 model dengan algoritma *machine learning* yang berbeda. Algoritma yang akan digunakan adalah K-Nearest Neighbor Algorithm, Random Forest Algorithm, dan Decision Tree Algorithm. Dari ketiga model tersebut akan dilakukan evaluasi performa dan kinerja masing-masing algoritma dan akan dipilih satu algoritma yang memberikan hasil prediksi yang terbaik.
   - **Algoritma K-Nearest Neighbor**  
     Sesuai dengan namanya, yaitu "sejumlah k-tetangga terdekat" adalah algoritma *machine learning* yang tergolong ke dalam *supervised learning* yang bekerja dengan cara mengelompokkan data berdasarkan kemiripan antar data baru dengan sejumlah data (k) yang terdekat. [[3]](https://geospasialis.com/k-nearest-neighbor 'Mengenal K-Nearest Neighbor: Algoritma Populer untuk Machine Learning') Cara kerja algoritma K-Nearest Neighbor, sebagai berikut: [[3]](https://geospasialis.com/k-nearest-neighbor 'Mengenal K-Nearest Neighbor: Algoritma Populer untuk Machine Learning')
     - Tentukan jumlah tetangga terdekat (`k`) yang akan dipertimbangkan sebagai dasar klasifikasi.
     - Hitung jarak antara data baru terhadap semua titik data dalam *dataset* (tetangga terdekat).
     - Urutkan jarak pada dari kecil ke besar, lalu diambil titik data dengan jarak terkecil dari sejumlah `k` titik.
     - Hitung jumlah titik pada `k` setiap kelas atau kategori.
     - Masukkan data baru ke kelas dengan jumlah `k` terbanyak.
     
     <br>
     <img src="https://user-images.githubusercontent.com/64983961/188507827-0f729ab6-61a5-4dbc-9be2-afa424f6c294.png" alt="Ilustrasi Algoritma K-Nearest Neighbor" title="Ilustrasi Algoritma K-Nearest Neighbor">
     
     Perhitungan jarak ke tetangga terdekat dapat dilakukan dengan menggunakan metrik sebagai berikut:
     - *Euclidean distance*
       $$d(x,y)=\sqrt{\sum_{i=1}^n (x_i-y_i)^2}$$
     - *Manhattan distance*
       $$d(x,y)=\sum_{i=1}^n |x_i-y_i|$$
     - *Hamming distance*
       $$d(x,y)=\frac{1}{n}\sum_{n=1}^{n=n} |x_i-y_i|$$
     - *Minkowski distance*
       $$d(x,y)=\left(\sum_{i=1}^n |x_i-y_i|^p\right)^\frac{1}{p}$$
     
     Adapun kelebihan dari algoritma K-Nearest Neighbor, yaitu: [[3]](https://geospasialis.com/k-nearest-neighbor 'Mengenal K-Nearest Neighbor: Algoritma Populer untuk Machine Learning')
     - Sangat sederhana dan mudah untuk dipahami
     - Sangat mudah dalam penerapannya
     - Dapat digunakan dalam kasus klasifikasi maupun regresi
     - Dapat digunakan dalam jumlah kelas yang berbeda-beda
     - Tidak memerlukan proses trainig dan pembangunan model, karena data baru secara langsung akan dikelaskan
     - Mudah jika ingin untuk melakukan penambahan data
     - Parameter yang dibutuhkan hanya sedikit, yaitu jumlah k-tetangga (`n_neighbors`), dan metode perhitungan metrik jaraknya (`metric`) [[4]](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html 'sklearn.neighbors.KNeighborsRegressor')
     - Hasil pemodelan tidak linear, sehingga cocok untuk klasifikasi data yang batasannya tidak linear.
     
     Adapun kelemahan dari algoritma K-Nearest Neighbor, yaitu: [[3]](https://geospasialis.com/k-nearest-neighbor 'Mengenal K-Nearest Neighbor: Algoritma Populer untuk Machine Learning')
     - Perlu untuk menentukan nilai `k` yang tepat
     - *Computation cost* yang cukup tinggi
     - Waktu pemrosesan akan berlangsung lama jika *dataset* yang digunakan sangat besar
     - Kurang bagus untuk diterapkan pada *high dimensional data*
     - Sangat sensitif pada data yang memiliki banyak *noise* (*noisy data*), data yang hilang (*missing data*), dan data dengan nilai yang ekstrem serta kemunculannya yang jarang (*outliers*).
     
   - **Algoritma Random Forest**  
     Metode Random Forest merupakan jenis algoritma *supervised learning* dan termasuk ke dalam metode Decision Tree yang menggunakan kombinasi dari masing-masing model tree dan akan digabungkan menjadi sebuah model dalam membuat hasil prediksi akhir. Algoritma Random Forest menggunakan teknik *bagging* (*bootstrap aggregating*), di mana beberapa model akan dilatih dengan cara *random sampling with replacement*. [[5]](https://machinelearning.mipa.ugm.ac.id/2018/07/28/random-forest 'Random Forest')
     
     <img src="https://user-images.githubusercontent.com/64983961/188504775-b7e4aa9b-f1cd-41ef-8a70-a977db8f3d60.png" alt="Ilustrasi Algoritma Random Forest" title="Ilustrasi Algoritma Random Forest">
     
     Setelah dilakukan pelatihan, prediksi untuk sampel yang tidak terlihat ($x'$) dapat dibuat dengan menghitung rata-rata prediksi dari semua pohon setiap individu model pada $x'$. [[6]](https://en.wikipedia.org/wiki/Random_forest#Bagging 'Random Forest - Bagging')
     $$\hat{f}=\frac{1}{B}\sum_{b=1}^{B} f_b(x^{'})$$
     
   - **Algoritma Decision Tree**  
    Algoritma decision tree adalah metode dalam ilmu komputer dan statistik yang digunakan dalam pengambilan keputusan dan pembelajaran mesin. Algoritma ini digunakan untuk mengatasi masalah klasifikasi dan regresi. Ini bekerja dengan memecah data menjadi bagian-bagian yang lebih kecil dan lebih mudah dikelola, membentuk struktur pohon keputusan dengan cabang-cabang dan simpul-simpul. Setiap simpul dalam pohon ini mewakili keputusan atau pengujian pada atribut tertentu, dan setiap cabang menggambarkan kemungkinan hasil dari pengujian tersebut.
     
     ![image](https://github.com/daaffalbari/diabetic-prediction/assets/73302268/28fd01f2-d95d-42be-bb1b-94cb816f0f64)


## Data Understanding

<img src="https://github.com/daaffalbari/diabetic-prediction/assets/73302268/eea9b2b1-a7ad-45d0-8d68-30320e348755" alt="Electric Power Consumption Kaggle Dataset" title="Electric Power Consumption Kaggle Dataset" width="100%">

Data yang digunakan dalam proyek ini adalah *dataset* yang diambil dari Kaggle Dataset [Diabetes Prediction](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset') dengan kategori *dataset*, yaitu *Health* dan *Classification*. Dalam *dataset* tersebut terdapat sebuah *file* atau berkas dengan nama `diabetes_prediction_dataset.csv` yang berekstensi (*file format*) `.csv` atau [comma-separated values](https://en.wikipedia.org/wiki/Comma-separated_values 'Comma-separated values') berukuran 751 KB.


Kemudian dilakukan proses *Exploratory Data Analysis* (EDA) sebagai investigasi awal untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data dengan menggunakan teknik statistik dan representasi grafis atau visualisasi.

1. **Deskripsi Variabel**  
   Berikut adalah informasi mengenai variabel-variabel yang terdapat pada *dataset* *Electric Power Consumption* adalah sebagai berikut,
   
| gender | age  | hypertension | heart_disease | smoking_history | bmi   | HbA1c_level | blood_glucose_level | diabetes |
|--------|------|--------------|---------------|-----------------|-------|-------------|---------------------|----------|
| female | 80.0 | 0            | 1             | never           | 25.19 | 6.6         | 140                 | 0        |
| female | 54.0 | 0            | 0             | No Info         | 27.32 | 6.6         | 80                  | 0        |
| male   | 28.0 | 1            | 1             | never           | 27.32 | 5.7         | 158                 | 0        |
   
   Dari gambar di atas dapat dilihat bahwa terdapat 52.416 baris data dan 10 kolom atribut atau fitur. Di antaranya adalah enam (6) atribut/variabel dengan tipe data `float64 non-null` dan lima (5) atribut/variabel dengan tipe data `int64 non-null` yang merupakan hasil penguraian dari variabel `Datetime` yang sebelumnya memiliki tipe data `datetime64[ns]`. Berikut adalah keterangan untuk masing-masing variabel,
   - `Gender` : mengacu pada jenis kelamin biologis individu, yang dapat memengaruhi kerentanan mereka terhadap diabetes.
   - `Age`    : Age is an important factor as diabetes is more commonly diagnosed in older adults.Age ranges from 0-80 in our dataset.
   - `Hypertension`   : Hypertension is a medical condition in which the blood pressure in the arteries is persistently elevated. It has values a 0 or 1 where 0 indicates they don’t have hypertension and for 1 it means they have hypertension.
   - `heart_disease` : Heart disease is another medical condition that is associated with an increased risk of developing diabetes. It has values a 0 or 1 where 0 indicates they don’t have heart disease and for 1 it means they have heart disease.
   - `smoking_history` : Smoking history is also considered a risk factor for diabetes and can exacerbate the complications associated with diabetes.In our dataset we have 5 categories i.e not current,former,No Info,current,never and ever
   - `bmi` : BMI (Body Mass Index) is a measure of body fat based on weight and height. Higher BMI values are linked to a higher risk of diabetes. The range of BMI in the dataset is from 10.16 to 71.55. BMI less than 18.5 is underweight, 18.5-24.9 is normal, 25-29.9 is overweight, and 30 or more is obese.
   - `HbA1c_level`   : HbA1c (Hemoglobin A1c) level is a measure of a person's average blood sugar level over the past 2-3 months. Higher levels indicate a greater risk of developing diabetes. Mostly more than 6.5% of HbA1c Level indicates diabetes.
   - `blood_glucose_leve`  : Blood glucose level refers to the amount of glucose in the bloodstream at a given time. High blood glucose levels are a key indicator of diabetes.
   - `diabetes`    : Diabetes is the target variable being predicted, with values of 1 indicating the presence of diabetes and 0 indicating the absence of diabetes.
   
2. **Deskripsi Statistik**

   Untuk mendapatkan deskripsi dari data yang digunakan, kita menggunakan fungsi `describe()` yang sudah disediakan python. Hasil dari fungsi tersebut bisa dilihat pada tabel di bawah ini:

   |       | age           | hypertension | heart_disease | bmi           | HbA1c_level   | blood_glucose_level | diabetes      |
   |-------|---------------|--------------|---------------|---------------|---------------|---------------------|---------------|
   | count | 100000.000000 | 100000.00000 | 100000.000000 | 100000.000000 | 100000.000000 | 100000.000000       | 100000.000000 |
   | mean  | 41.885856     | 0.07485      | 0.039420      | 27.320767     | 5.527507      | 138.058060          | 0.085000      |
   | std   | 22.516840     | 0.26315      | 0.194593      | 6.636783      | 1.070672      | 40.708136           | 0.278883      |
   | min   | 0.080000      | 0.00000      | 0.000000      | 10.010000     | 3.500000      | 80.000000           | 0.000000      |
   | 25%   | 24.000000     | 0.00000      | 0.000000      | 23.630000     | 4.800000      | 100.000000          | 0.000000      |
   | 50%   | 43.000000     | 0.00000      | 0.000000      | 27.320000     | 5.800000      | 140.000000          | 0.000000      |
   | 75%   | 60.000000     | 0.00000      | 0.000000      | 29.580000     | 6.200000      | 159.000000          | 0.000000      |
   | max   | 80.000000     | 1.00000      | 1.000000      | 95.690000     | 9.000000      | 300.000000          | 1.000000      |


   - Count (Jumlah Data): Ini adalah jumlah entri non-null dalam kolom. Ini menggambarkan seberapa banyak data yang tersedia untuk kolom tersebut.
   - Mean (Rata-rata): Ini adalah nilai rata-rata dari seluruh data dalam kolom. Rata-rata adalah jumlah semua nilai dibagi dengan jumlah entri.
   - Std (Deviasi Standar): Ini adalah ukuran sebaran data. Ini mengukur seberapa jauh setiap nilai dari rata-rata. Standar deviasi yang lebih tinggi menunjukkan bahwa data lebih tersebar.
   - Min (Nilai Minimum): Ini adalah nilai terkecil dalam kolom, menunjukkan batas bawah data.
   - 25% (Kuartil Pertama): Ini adalah nilai yang membagi 25% data terendah dari 75% data tertinggi. Kuartil pertama juga dikenal sebagai kuartil bawah atau kuartil ke-1.
   - 50% (Median atau Kuartil Kedua): Ini adalah nilai yang membagi data menjadi dua setengah. Setengah data berada di atas nilai ini dan setengahnya lagi berada di bawahnya. Median juga dikenal sebagai kuartil tengah atau kuartil ke-2.
   - 75% (Kuartil Ketiga): Ini adalah nilai yang membagi 75% data terendah dari 25% data tertinggi. Kuartil ketiga juga dikenal sebagai kuartil atas atau kuartil ke-3.
   - Max (Nilai Maksimum): Ini adalah nilai terbesar dalam kolom, menunjukkan batas atas data.
   
4. **Menangani Missing Value**  

   <img src="https://github.com/daaffalbari/diabetic-prediction/assets/73302268/a4f99c90-d167-4a1c-addc-30742d8c80ea" alt="Menangani Missing Value" title="Menangani Missing Value">
   
   Berdasarkan gambar tersebut, tidak terdapat *missing value*.
   
5. **Menangani Outliers**  
   *Outliers* merupakan sampel data yang nilainya berada sangat jauh dari cakupan umum data utama yang dapat merusak hasil analisis data. Berikut adalah visualisasi *boxplot* untuk melakukan pengecekan keberadaan *outliers*.

   <img src="https://github.com/daaffalbari/diabetic-prediction/assets/73302268/1f0550a1-96e5-4db1-8402-47e57742831e" alt="Menangani Outliers - Sebelum" title="Menangani Outliers - Sebelum">
     
   Berdasarkan gambar tersebut, terdapat *outliers* pada fitur `HbA1c`, `blood_glucose_level`, dan `bmi`. Sehingga dilakukan proses pembersihan *outliers* dengan metode Z-Score.
   
   $$Z_score=(X_mean) / std.devation

   ![image](https://github.com/daaffalbari/diabetic-prediction/assets/73302268/43aead66-8488-4654-8013-2e78d2a84615)

   Untuk penjelasannya bisa dilihat pada gambar di atas.
   
6. **Univariate Analysis**  
   Melakukan proses analisis data *univariate* pada fitur-fitur numerik. Proses analisis ini menggunakan bantuan visualisasi histogram untuk masing-masing fitur numerik.

   <img src="https://github.com/daaffalbari/diabetic-prediction/assets/73302268/6a17eee6-41fb-4e69-869a-260a0df13536" alt="Univariate Analysis" title="Univariate Analysis">
   
   
7. **Analisis Korelasi Antar Fitur**  
   Melakukan pengecekan korelasi antar fitur numerik dengan menggunakan visualisasi diagram *heatmap* *correlation matrix*.
   
   <img src="https://github.com/daaffalbari/diabetic-prediction/assets/73302268/ad2a9a45-c918-493d-81b9-7a64021d43e3" alt="Correlation Matrix with Heatmap" title="Correlation Matrix with Heatmap">
   
   Dapat dilihat pada diagram *heatmap* di atas memiliki *range* atau rentang angka dari 1.0 hingga 0.4 dengan keterangan sebagai berikut,
   - Jika semakin mendekati 1, maka korelasi antar fitur numerik semakin kuat bernilai positif.
   - Jika semakin mendekati 0, maka korelasi antar fitur numerik semakin rendah.
   - Jika semakin mendekati -1, maka korelasi antar fitur numerik semakin kuat bernilai negatif.
   
   Jika korelasi bernilai positif, berarti nilai kedua fitur numerik cenderung meningkat bersama-sama.  
   
   Jika korelasi bernilai negatif, berarti nilai salah satu fitur numerik cenderung meningkat ketika nilai fitur numerik yang lain menurun.

## Data Preparation

Pada tahap persiapan data atau *data preparation* dilakukan berdasarkan penjelasan yang sudah dipaparkan pada bagian [Solution Statements](#solution-statements "Solution Statements"). Tahap ini penting dilakukan untuk mempersiapkan data sehingga dapat digunakan untuk melatih model *machine learning* dengan baik. Berikut adalah dua tahapan data preparation yang dilakukan, yaitu,

1. **Split Data**  
   Pembagian data dilakukan untuk memisahkan data keseluruhan menjadi dua (2) bagian, yaitu data latih (*training data*) dan data uji (*testing data*) dengan perbandingan rasio sebesar 90 : 10 menggunakan `train_test_split`. Alasan membagi kedalam jumlah 90:10 dikarenakan jumlah dataset yang digunakan jumlahnya sudah banyak. Sehingga membagi ke dalam 90 : 10 tidak menjadi masalah.

    
   Kemudian diperoleh hasil pembagian data masing-masing, yaitu sebagai berikut,

    - Total of sample in whole dataset: 92542
    - Total of sample in train dataset: 83287
    - Total of sample in test dataset: 9255

1. **Standarisasi pada Fitur Numerik**  
   Standarisasi fitur numerik menggunakan `StandardScaler` untuk mencegah terjadinya penyimpangan nilai data yang cukup besar. Proses standarisasi tersebut dilakukan dengan mengurangkan nilai rata-rata, lalu membaginya dengan standar deviasi atau simpangan baku untuk menggeser distribusi. Proses standarisasi akan menghasilkan distribusi dengan nilai rata-rata menjadi 0, dan nilai standar deviasi menjadi 1.


|   | gender    | age       | hypertension | heart_disease | smoking_history | bmi       | HbA1c_level | blood_glucose_level |
|---|-----------|-----------|--------------|---------------|-----------------|-----------|-------------|---------------------|
| 0 | -1.185701 | -0.054301 | -0.278626    | -0.198654     | 1.920119        | 0.072790  | -0.970700   | 0.645537            |
| 1 | -1.185701 | 1.233948  | -0.278626    | -0.198654     | 1.920119        | 0.679367  | 1.130653    | -1.247323           |
| 2 | -1.185701 | 1.189525  | -0.278626    | -0.198654     | -0.913236       | 0.919368  | 0.230074    | -0.972995           |
| 3 | -1.185701 | 1.233948  | -0.278626    | -0.198654     | -0.204898       | -0.482828 | 0.730396    | 0.124314            |
| 4 | -1.185701 | 0.212234  | -0.278626    | -0.198654     | -0.204898       | -0.038991 | 0.530267    | 0.124314            |

  Salah satu ciri-ciri dari data yang disudah distandarisasi adalah rata-rata di setiap kolom atau fitur akan mendekati nol. Untuk melihat hal tersebut, digunakan fungsi `describe` yang sudah disediakan oleh python, sehingga hasil yang didapat adalah sebagai berikut,

|       | gender     | age        | hypertension | heart_disease | smoking_history | bmi        | HbA1c_level | blood_glucose_level |
|-------|------------|------------|--------------|---------------|-----------------|------------|-------------|---------------------|
| count | 83287.0000 | 83287.0000 | 83287.0000   | 83287.0000    | 83287.0000      | 83287.0000 | 83287.0000  | 83287.0000          |
| mean  | -0.0000    | -0.0000    | 0.0000       | 0.0000        | -0.0000         | 0.0000     | 0.0000      | 0.0000              |
| std   | 1.0000     | 1.0000     | 1.0000       | 1.0000        | 1.0000          | 1.0000     | 1.0000      | 1.0000              |
| min   | -1.1857    | -1.8276    | -0.2786      | -0.1987       | -0.9132         | -2.7727    | -1.9713     | -1.5217             |
| 25%   | -1.1857    | -0.8095    | -0.2786      | -0.1987       | -0.9132         | -0.6078    | -0.6705     | -0.9730             |
| 50%   | 0.8434     | 0.0345     | -0.2786      | -0.1987       | -0.2049         | 0.0728     | 0.3301      | 0.1243              |
| 75%   | 0.8434     | 0.7897     | -0.2786      | -0.1987       | 0.5034          | 0.4147     | 0.7304      | 0.6181              |
| max   | 0.8434     | 1.7226     | 3.5890       | 5.0339        | 2.6285          | 3.4098     | 2.7317      | 3.4162              |


## Model Development

Setelah dilakukannya tahap *data preparation*, selanjutnya adalah melakukan tahap persiapan model terlebih dahulu sebelum mengembangkan model menggunakan algoritma yang telah ditentukan.

Tahap persiapan *dataframe* untuk analisis model menggunakan parameter `index`, yaitu train_mse dan test_mse, serta parameter `columns` yang merupakan algoritma yang akan digunakan untuk melakukan prediksi, yaitu algoritma K-Nearest Neighbor (KNN), Random Forest, dan Decesion Tree.

Kemudian terapkan ketiga algoritma ke dalam model tersebut.

1. **K-Nearest Neighbor (KNN) Algorithm**  
   Pada algoritma K-Nearest Neighbor digunakan parameter `n_neighbors` dengan nilai k = 10 tetangga dan `metric` bawaan, yaitu Euclidean. Kita menggunakan fungsi yang sudah disediakan oleh Scikit Learning, sehingga pemanggilan algoritma bisa dilakukan dengan mudah.
   
   Kemudian akan dilakukan analisis prediksi *error* menggunakan *Mean Squared Error* (MSE) pada data latih (*training data*) dan data uji (*testing data*)
   
2. **Random Forest Algorithm**  
   Pada algoritma K-Nearest Neighbor digunakan parameter `n_estimator` dengan jumlah 50 *trees* (pohon), `max_depth` dengan nilai kedalaman atau panjang pohon 16, `random_state` dengan nilai 55, dan `n_jobs` yang bernilai -1 (pekerjaan dilakukan secara paralel).
   
   Kemudian akan dilakukan analisis prediksi *error* menggunakan *Mean Squared Error* (MSE) pada data latih (*training data*) dan data uji (*testing data*)
   
3. **Decision Tree Algorithm**  
   Pada algoritma Decision Tree digunakan parameter `criterion` dengan `entropy`, dan `max_depth` dengan nilai 9.
   
   Kemudian akan dilakukan analisis prediksi *error* menggunakan *Mean Squared Error* (MSE) pada data latih (*training data*) dan data uji (*testing data*)

Ketiga model yang telah dibangun di atas, akan dilakukan pengujian kinerja untuk masing-masing model yang menggunakan algoritma K-Nearest Neighbor, algoritma Random Forest, dan algoritma Decision Tree. Dari ketiga model tersebut akan diperoleh satu (1) model dengan hasil prediksi yang paling baik dan tingkat *error* yang paling rendah.

## Evaluation

Pada tahap evaluasi model, akan dilakukan pengujian untuk melihat algoritma mana yang memberikan hasil prediksi paling baik dan dengan tingkat *error* yang paling rendah. Sebelumnya, akan dilakukan proses standarisasi atau *scaling* pada fitur numerik data uji (*testing data*) agar nilai rata-rata (*mean*) bernilai 0, dan varians bernilai 1.


Kemudian evaluasi dari ketiga model, yaitu algoritma K-Nearest Neighbor, Random Forest, dan Adaptive Boosting (AdaBoost) untuk masing-masing data latih (*training data*) dan data uji (*testing data*) dengan melihat tingkat *error*-nya menggunakan *Mean Squared Error* (MSE),

$$MSE=\frac{1}{N}\sum_{i=1}^{N} (y_i-y\\_pred_i)^2$$

di mana, nilai $N$ adalah jumlah *dataset*, nilai $y_i$ merupakan nilai sebenarnya, dan $y\\_pred$ yaitu nilai prediksinya.

Penggunaan metode metrik *Mean Squared Error* (MSE) memiliki kelebihan, yaitu cukup sederhana dan mudah dipahami dalam melakukan perhitungan. Meskipun begitu, terdapat kelemahan pada metrik ini, yaitu hasil akurasi prediksi yang kecil karena tidak dapat membandingan hasil peramalan tersebut dengan kenyataannya. 


Untuk mempermudahnya, ditampilkan dalam bentuk tabel seperti yang ada di bawah ini.

|           | DecisionTree | KNN      | Random Forest |
|-----------|--------------|----------|---------------|
| Accuracy  | 0.967801     | 0.958509 | 0.967585      |
| Precision | 0.974393     | 0.950201 | 0.974227      |
| Recall    | 0.760309     | 0.696673 | 0.758669      |
| F1-score  | 0.831994     | 0.767409 | 0.830522      |
| mse       | 0.032199     | 0.041491 | 0.032415      |


Dan untuk hasil mse dari data train dan test adalah sebagai berikut:

|              | train    | test     |
|--------------|----------|----------|
| KNN          | 0.000036 | 0.000041 |
| RandomForest | 0.000022 | 0.000032 |
| DecisionTree | 0.000028 | 0.000032 |

Dari data tabel tersebut dapat divisualisasikan pada grafik batang berikut.

<img src="https://github.com/daaffalbari/diabetic-prediction/assets/73302268/7d36cd68-689e-43ef-af66-ad7dd7275dfb" alt="Evaluation Graph" title="Evaluation Graph">

Dari visualisasi diagram di atas dapat disimpulkan bahwa,
1. Model RandomForest dengan Tingkat Error Terendah:
Model yang menggunakan algoritma RandomForest memberikan tingkat kesalahan yang paling rendah di antara ketiga model.
Pada data pelatihan (train error), kesalahan model RandomForest adalah 0.000022, sedangkan pada data pengujian (test error), kesalahan adalah 0.000032.

2. Model K-Nearest Neighbor (KNN) dengan Tingkat Error Menengah:
Model yang menggunakan algoritma K-Nearest Neighbor (KNN) memiliki tingkat kesalahan yang berada di tengah-tengah antara model RandomForest dan DecisionTree.
Tingkat kesalahan KNN adalah 0.000036 pada data pelatihan dan 0.000041 pada data pengujian.
Model DecisionTree dengan Tingkat Error Tertinggi:

3. Model yang menggunakan algoritma DecisionTree memiliki tingkat kesalahan yang paling tinggi di antara ketiga model.
Tingkat kesalahan DecisionTree adalah 0.000028 pada data pelatihan dan 0.000032 pada data pengujian.


Kesimpulannya, dari ketiga model yang telah diuji, model RandomForest memberikan tingkat kesalahan yang paling rendah pada kedua set data pelatihan dan pengujian. Oleh karena itu, dalam konteks ini, model RandomForest dianggap sebagai model yang paling baik dalam memprediksi data berdasarkan tingkat kesalahan yang lebih rendah.

---

## Referensi

[1] Y. Safitri and I. K. A. Nurhayati, ―Pengaruh Pemberian Sari Pati Bengkuang (Pachyrhizus Erosus) terhadap Kadar Glukosa Darah pada Penderita Diabetes Mellitus Tipe II Usia 40-50 Tahun di Kelurahan Bangkinang Wilayah Kerja Puskesmas Bangkinang Kota Tahun 2018,‖ J. Ners, vol. 3, no. 1, pp. 69–81, 2019

[2] F. Nasution, A. Andilala, and A. A. Siregar, ―Faktor Risiko Kejadian Diabetes Mellitus,‖ J. Ilmu Kesehat., vol. 9, no. 2, pp. 94–102, 2021

[3] R. R. Santoso, ―Implementasi Metode Machine Learning menggunakan Algoritma Evolving Artificial Neural Network pada Kasus Prediksi Diagnosis Diabetes.‖ Universitas Pendidikan Indonesia, 2020.

[4] scikit-learn, "sklearn.neighbors.KNeighborsRegressor", Retrieved from: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

[5] A. Yanuar, "Random Forest", *Universitas Gadjah Mada Menara Ilmu Machine Learning*, 2018, Retrieved from: https://machinelearning.mipa.ugm.ac.id/2018/07/28/random-forest

[6] "Random Forest", Retrieved from: https://en.wikipedia.org/wiki/Random_forest#Bagging

[7] "Decision Tree", Retrieved from: https://en.wikipedia.org/wiki/DecisionTree#Training

[8] S. R. P. Nur Hidayatika, and S. N. W.P, "USULAN PENGGUNAAN METODE FORECASTING UNTUK PERMINTAAN KOPI ROBUSTA PADA PT. XYZ," *Industrial Engineering Online Journal*, vol. 4, no. 3, 2016, Retrieved from: https://ejournal3.undip.ac.id/index.php/ieoj/article/view/9002

[9] A. Salam and A. E. Hibaoui, "Comparison of Machine Learning Algorithms for the Power Consumption Prediction : - Case Study of Tetouan city –," *2018 6th International Renewable and Sustainable Energy Conference (IRSEC)*, 2018, pp. 1-5, doi: 10.1109/IRSEC.2018.8703007.
