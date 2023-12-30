# Machine Learning
### Nama  : Doni Arafat
### Nim   : 211351049
### Kelas : Malam B

## Domain Proyek

Web App yang saya kembangkan ini boleh digunakan oleh siapapun yang membutuhkan. Hasil dari web app ini digunakan untuk mengidentifikasi apakah anda mengidap kanker paru-paru dengan menganalisa semua simptom-simptom yang sedang anda alami. Setelah mendapatkan hasil sebaiknya anda secepatnya pergi pada seorang profesional untuk diagnosa lebih lanjut dan melakukan pengobatan.

## Business Understanding

Bisa mengidentifikasi kanker paru-paru lebih cepat jikalau anda tidak memiliki waktu untuk bertemu dengan seorang profesional.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Semakin buruknya polusi udara di Indonesia menjadikannya rawan bagi orang-orang terkena kanker paru-paru. 

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mencari cara agar orang-orang bisa menggunakan suatu aplikasi untuk mengidentifikasi apakah mereka mengidap kanker paru-paru atau tidak dengan cara menginputkan simtom simtom yang dia alami.

## Data Understanding
Datasets yang saya gunakan berasal dari Kaggle, dari user Ms. Nancy Al Aswad. Dataset ini bernama Lung Cancer dengan 16 jumlah kolom dan 309 baris data sebelum dilakukannya data cleaning.

[Lung Cancer](https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer).

### Variabel-variabel pada Lung Cancer Dataset adalah sebagai berikut:
- GENDER : adalah kelamin pasien.
- AGE : adalah umur pasien.
- SMOKING : adalah status merokok pasian.
- YELLOW_FINGERS : adalah kondisi jari pasien saat itu.
- ANXIETY : merupakan perasaan cemas pasien.
- PEER_PRESSURE : adalah status ketertekanan pasien.
- CHRONIC DISEASE : adalah status apakah pasien mengidap penyakit kornis.
- FATIGUE : apakah pasien selalu merasa lelah/kekurangan energy.
- ALLERGY : adalah status apakah pasien mengidap suatu allergy.
- WHEEZING : adalah status apakah pasien mengidap mengi.
- ALCOHOL CONSUMING : merupakan status apakah pasien mengkonsumsi alkohol.
- COUGHING : adalah status apakah pasien sering batuk-batuk.
- SHORTNESS OF BREATH : adalah status apakah pasien kesulitan untuk bernafas.
- SWALLOWING DIFFICULTY : adalah status apakah pasien kesulitan untuk menelan.
- CHEST PAIN : adalah status apakah pasien merasakan sakit di dada.
- LUNG_CANCER : adalah status apakah pasien mengidap kanker paru-paru atau tidak.

## Data Preparation
Untuk teknik yang saya gunakan disini adalah EDA dan ada proses menghapus data duplicate serta membuat kolom baru jikalau terdapat kolom dengan relasi yang cukup tinggi(diatas 50%). Baik, langkah pertama adalah mengimpor file token kaggle agar bisa mengambil datasets yang telah dipilih,
``` bash
from google.colab import files
files.upload()
```
Lalu membuat folder untuk menampung file yang tadi dimasukkan dan mengunduh datasets yang kita inginkan,
``` bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle

!kaggle datasets download -d nancyalaswad90/lung-cancer
```
Setelah mengunduh datasetsnya, kita akan meng-unzipnya agar kita bisa menggunakan file csvnya,
``` bash
!unzip lung-cancer.zip -d lung-cancer
!ls lung-cancer
```
Langkah selanjutnya adalah mengimpor semua komponen yang akan digunakan selama proses EDA,
``` bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
Mari gunakan datasets yang tadi telah diekstrak dengan menggunakan read_csv dari pandas,
``` bash
df=pd.read_csv('lung-cancer/survey lung cancer.csv')
df.head()
```
Setelah itu, mari kita lihat apakah terdapat nilai berulang pada datasets ini,
``` bash
df.duplicated().sum()
```
Wah, terdapat 33 data duplicate di dalamnya, mari hapus data-data tersebut karena ianya akan mempengaruhi hasil model kita nanti,
``` bash
df=df.drop_duplicates()
```
Selanjutnya kita akan periksa apakah di dalamnya terdapat nilai null/kosong,
``` bash
df.isnull().sum()
```
Tampak aman dan tidak ada nilai null, lanjut dengan melihat tipe data dari masing-masing kolom,
``` bash
df.info()
```
Terlihat di sini terdapat dua tipe data object, mari kita ubah menjadi nilai integer serta untuk kolom lain yang sudah bernilai integer namun terdapat angka (1 = No, 2 = Yes) akan kita ubah agar nilai integernya hanya ada (0 = No, 1 = Yes),
``` bash
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
df['GENDER']=le.fit_transform(df['GENDER'])
df['LUNG_CANCER']=le.fit_transform(df['LUNG_CANCER'])
df['SMOKING']=le.fit_transform(df['SMOKING'])
df['YELLOW_FINGERS']=le.fit_transform(df['YELLOW_FINGERS'])
df['ANXIETY']=le.fit_transform(df['ANXIETY'])
df['PEER_PRESSURE']=le.fit_transform(df['PEER_PRESSURE'])
df['CHRONIC DISEASE']=le.fit_transform(df['CHRONIC DISEASE'])
df['FATIGUE ']=le.fit_transform(df['FATIGUE '])
df['ALLERGY ']=le.fit_transform(df['ALLERGY '])
df['WHEEZING']=le.fit_transform(df['WHEEZING'])
df['ALCOHOL CONSUMING']=le.fit_transform(df['ALCOHOL CONSUMING'])
df['COUGHING']=le.fit_transform(df['COUGHING'])
df['SHORTNESS OF BREATH']=le.fit_transform(df['SHORTNESS OF BREATH'])
df['SWALLOWING DIFFICULTY']=le.fit_transform(df['SWALLOWING DIFFICULTY'])
df['CHEST PAIN']=le.fit_transform(df['CHEST PAIN'])
df['LUNG_CANCER']=le.fit_transform(df['LUNG_CANCER'])
```
Mari kita lihat hasilnya,
``` bash
df.head()
```
Sudah sesuai, semua tipe datanya telah menjadi integer, selanjutnya mari kita banding-bandingkan data kolomnya,
```
sns.countplot(x='LUNG_CANCER', data=df,)
plt.title('Target Distribution');
```
![download](https://github.com/DonnyyA/lung-cancer-prediction/assets/149292708/65d07e40-9e3e-41b3-b776-3dbb393ad836) <br>
Data antara yang mengidap dan tidak mengidap kanker tidak seimbang, nanti akan kita selesaikan, untuk sekarang kita lanjut data visualizing,
``` bash
def plot(col, df=df):
    return df.groupby(col)['LUNG_CANCER'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(8,5))
```
Fungsi untuk plotting sudah dibuat, ini digunakan untuk memudahkan kita dalam melakukan visualisasi,
``` bash
plot('GENDER')
```
![download](https://github.com/DonnyyA/lung-cancer-prediction/assets/149292708/1ff487a1-a91f-4aba-8dea-bd5a1c319ded)
``` bash
plot('AGE')
```
![download](https://github.com/DonnyyA/lung-cancer-prediction/assets/149292708/6afad4e9-9b94-46e7-a4c4-7919d47f6dd1)
``` bash
plot('SMOKING')
```
![download](https://github.com/DonnyyA/lung-cancer-prediction/assets/149292708/407dd033-606e-48c6-831a-6755fb84d9c3)
``` bash
plot('YELLOW_FINGERS')
```
![download](https://github.com/DonnyyA/lung-cancer-prediction/assets/149292708/5a6277dc-4057-47ca-adf1-e41ccde8d760)
``` bash
plot('ANXIETY')
```
![download](https://github.com/DonnyyA/lung-cancer-prediction/assets/149292708/8844380b-4b66-480b-8f3a-b446369a25f5)
``` bash
plot('PEER_PRESSURE')
```
![download](https://github.com/DonnyyA/lung-cancer-prediction/assets/149292708/5c67de03-0439-448b-9826-3ca92717f878)
``` bash
plot('ALLERGY ')
```
![download](https://github.com/DonnyyA/lung-cancer-prediction/assets/149292708/0c73826a-67d4-41a2-83d0-1effc23f08d8)
``` bash
plot('WHEEZING')
```
![download](https://github.com/DonnyyA/lung-cancer-prediction/assets/149292708/0731e9f6-580d-4b4e-a9a9-d075826a363d)
``` bash
plot('SHORTNESS OF BREATH')
```
![download](https://github.com/DonnyyA/lung-cancer-prediction/assets/149292708/14791845-1483-4e21-92af-f092bb88b989)
``` bash
plot('SWALLOWING DIFFICULTY')
```
![download](https://github.com/DonnyyA/lung-cancer-prediction/assets/149292708/c8e875c7-8a9d-4d7d-8c0e-222fb4fb506c) <br>
Seperti yang terlihat diatas, korelasi antara pengidap kanker paru-paru dengan gender, age, sking, dan shortness of breath sangatlah kecil, maka sebaiknya kita hilangkan saja,
``` bash
df_new=df.drop(columns=['GENDER','AGE', 'SMOKING', 'SHORTNESS OF BREATH'])
df_new
```
selanjutnya kita akan melihat korelasi antar kolom,
``` bash
cn=df_new.corr()
cmap=sns.diverging_palette(260,-10,s=50, l=75, n=6,
as_cmap=True)
plt.subplots(figsize=(18,18))
sns.heatmap(cn,cmap=cmap,annot=True, square=True)
plt.show()
```
Karena korelasi antara kolom Yellow_Finger dan kolom Anxiety sangat tinggi (diatas 50%) maka akan kita gabungkan dan membuat satu kolom baru yang mengandung gabungan antara dua kolom,
``` bash
df_new['ANXYELFIN']=df_new['ANXIETY']*df_new['YELLOW_FINGERS']
df_new
```
Untuk data preparationnya sudah selesai ya, selanjutnya adalah melakukan proses Modeling.
## Modeling
Algorithma modeling yang saya gunakan disini adalah Random Forest Classifier (RFC), karena web app ini hanya akan mengklasifikasi sesuatu (antara mengidap kanker paru-paru atau tidak) maka ini merupakan algorithma yang cocok sekali, mari buat variabel fitur dan target lalu mengatasi data imbalance yang tadi,
``` bash
X = df_new.drop('LUNG_CANCER', axis = 1)
y = df_new['LUNG_CANCER']
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=42)
X, y = adasyn.fit_resample(X, y)
len(X)
```
Selanjutnya adalah membuat train_test split dengan variabel X_train, X_test, y_train, dan y_test, lalu menggunakan RandomForestClassifier,
``` bash
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=0)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
```
Model sudah jadi, sekarang mari kita uji dengan X_test, 
``` bash
y_rf_pred= rf_model.predict(X_test)
y_rf_pred
```

## Evaluation
Matriks evaluasi yang saya gunakan disini adalah f1, precision dan recall, saya gunakan matriks evaluasi ini karena ianya merupakan yang paling cocok untuk melihat berapa score presisi dan recallnya, karena kasus ini membutuhkan presisi yang tinggi untuk mendapatkan hasil yang sesuai, kode yang saya gunakan sebagai berikut,
``` bash
from sklearn.metrics import classification_report, accuracy_score, f1_score
rf_cr=classification_report(y_test, y_rf_pred)
print(rf_cr)
```
Dan hasil yang didapatkan adalah 98% presisi, recall dan f1.

## Deployment
(Lung Cancer Prediction)[https://lung-cancer-prediction-donny.streamlit.app/]
![image](https://github.com/DonnyyA/lung-cancer-prediction/assets/149292708/cb896a58-5725-42bf-a964-49743532ec1f)


