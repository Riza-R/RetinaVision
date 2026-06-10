# ============================================================
# SISTEM KLASIFIKASI PENYAKIT MATA BERBASIS CITRA FUNDUS
# ============================================================
#
# Penelitian:
# Hybrid ResNet-50 dan K-Nearest Neighbor (KNN)
# untuk klasifikasi penyakit retina.
#
# Framework:
# - Streamlit       : Antarmuka Web
# - TensorFlow      : Deep Learning
# - OpenCV          : Pengolahan Citra
# - Scikit-Learn    : Model Machine Learning
#
# Kelas yang Dideteksi:
# 1. Cataract
# 2. Diabetic Retinopathy
# 3. Glaucoma
# 4. Normal
#
# ============================================================


# ============================================================
# IMPORT LIBRARY
# ============================================================
#
# Library yang digunakan dalam aplikasi:
#
# Streamlit  : Membuat web aplikasi interaktif
# NumPy      : Operasi array dan numerik
# OpenCV     : Pengolahan citra digital
# Pickle     : Memuat model hasil training
# Pandas     : Pengolahan data tabel
# TensorFlow : Deep Learning Framework
# Matplotlib : Visualisasi grafik
#
# ============================================================

import streamlit as st
import numpy as np
import cv2
import pickle
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================
# KONFIGURASI HALAMAN STREAMLIT
# ============================================================
#
# page_title :
# Nama halaman yang muncul pada browser.
#
# layout :
# "wide" digunakan agar tampilan aplikasi
# lebih lebar dan nyaman digunakan.
#
# ============================================================

st.set_page_config(
    page_title="Klasifikasi Penyakit Mata",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================
#
# Bagian ini digunakan untuk mempercantik tampilan
# aplikasi Streamlit menggunakan CSS.
#
# Komponen yang dimodifikasi:
# - Background utama
# - Sidebar
# - Judul dan teks
# - Tombol
# - Upload file
# - Dataframe
# - Alert box
# - Metric card
#
# ============================================================

st.markdown("""
<style>

/* =====================================================
BACKGROUND UTAMA
===================================================== */

.stApp {
    background-color: #F4F8FB;
}


/* =====================================================
SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {
    background-color: #EAF1F8;
    border-right: 1px solid #D0DCE8;
}


/* =====================================================
WARNA TEKS DAN JUDUL
===================================================== */

h1 {
    color: #1E3A5F;
    font-weight: 700;
}

h2 {
    color: #24476B;
}

h3 {
    color: #2F5D8C;
}

p, li {
    color: #2E2E2E;
}


/* =====================================================
KARTU METRIK
===================================================== */

[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #DCE6F2;
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
}


/* =====================================================
TOMBOL
===================================================== */

.stButton>button {
    background-color: #1E3A5F;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
}

.stButton>button:hover {
    background-color: #24476B;
    color: white;
}


/* =====================================================
UPLOAD FILE
===================================================== */

[data-testid="stFileUploader"] {
    background-color: white;
    border-radius: 12px;
    padding: 10px;
    border: 1px solid #DCE6F2;
}


/* =====================================================
ALERT BOX
===================================================== */

.stAlert {
    border-radius: 12px;
}


/* =====================================================
DATAFRAME
===================================================== */

[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 12px;
    padding: 5px;
}


/* =====================================================
GARIS PEMBATAS (HR)
===================================================== */

hr {
    border: 1px solid #DCE6F2;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MEMUAT MODEL (LOAD MODELS)
# ============================================================
#
# Fungsi ini digunakan untuk memuat seluruh model
# yang diperlukan saat aplikasi berjalan.
#
# Model yang dimuat:
# 1. ResNet-50        → Feature Extractor
# 2. KNN             → Classifier
# 3. StandardScaler  → Normalisasi fitur
# 4. Label Encoder   → Konversi label
# 5. Fundus Checker  → Validasi citra retina
#
# @st.cache_resource digunakan agar model hanya
# dimuat satu kali sehingga aplikasi lebih cepat.
#
# ============================================================

@st.cache_resource
def load_models():

    # Membaca file model hasil training
    with open("models/hybrid_resnet50_knn_best.pkl", "rb") as f:
        obj = pickle.load(f)

    # Memuat model validasi fundus retina
    fundus_checker = load_model(
        "models/fundus_checker_best.h5"
    )

    # Mengembalikan seluruh model yang dibutuhkan
    return (
        obj["resnet_model"],
        obj["knn_model"],
        obj["scaler"],
        obj["label_encoder"],
        fundus_checker
    )


# ============================================================
# VALIDASI CITRA FUNDUS RETINA
# ============================================================
#
# Fungsi:
# Memastikan gambar yang diunggah pengguna
# benar-benar merupakan citra fundus retina.
#
# Alur:
# Input Gambar
#      ↓
# Resize 224x224
#      ↓
# Preprocessing MobileNetV2
#      ↓
# Fundus Checker
#      ↓
# Valid / Tidak Valid
#
# Output:
# True  = Citra fundus retina
# False = Bukan citra fundus retina
#
# ============================================================

def is_fundus(image, checker):

    # Menyesuaikan ukuran gambar
    img = cv2.resize(image, (224, 224))

    # Preprocessing sesuai model checker
    img = tf.keras.applications.mobilenet_v2.preprocess_input(
        img.astype("float32")
    )

    # Menambahkan dimensi batch
    img = np.expand_dims(img, axis=0)

    # Prediksi probabilitas
    prob = checker.predict(img)[0][0]

    # Probabilitas < 0.5 dianggap fundus retina
    return prob < 0.5


# ============================================================
# PREPROCESSING CITRA
# ============================================================
#
# Fungsi:
# Menyiapkan citra sebelum masuk ke model
# ResNet-50.
#
# Tahapan:
# 1. Resize gambar menjadi 224x224
# 2. Normalisasi menggunakan preprocess_input
# 3. Menambahkan dimensi batch
#
# Output:
# Shape = (1, 224, 224, 3)
#
# ============================================================

def preprocess_image(image):

    # Menyesuaikan ukuran gambar
    image = cv2.resize(image, (224, 224))

    # Normalisasi sesuai ResNet-50
    image = preprocess_input(
        image.astype("float32")
    )

    # Menambahkan dimensi batch
    return np.expand_dims(image, axis=0)

# ============================================================
# PIPELINE KLASIFIKASI
# ============================================================
#
# Fungsi:
# Melakukan proses klasifikasi citra retina menggunakan
# model Hybrid ResNet-50 + KNN.
#
# Alur Proses:
#
# Citra Retina
#      ↓
# ResNet-50
# (Feature Extraction)
#      ↓
# Feature Vector
#      ↓
# Standard Scaler
#      ↓
# K-Nearest Neighbor (KNN)
#      ↓
# Label Penyakit
#
# Output:
# - Cataract
# - Diabetic Retinopathy
# - Glaucoma
# - Normal
#
# ============================================================

def classify_image(resnet, knn, scaler, le, image):

    # --------------------------------------------------------
    # Ekstraksi fitur menggunakan ResNet-50
    # --------------------------------------------------------
    feat = resnet.predict(image)

    # --------------------------------------------------------
    # Normalisasi fitur menggunakan Standard Scaler
    # --------------------------------------------------------
    feat = scaler.transform(feat)

    # --------------------------------------------------------
    # Prediksi kelas menggunakan KNN
    # --------------------------------------------------------
    pred = knn.predict(feat)

    # --------------------------------------------------------
    # Mengubah label numerik menjadi nama kelas asli
    # --------------------------------------------------------
    return le.inverse_transform(pred)[0]


# ============================================================
# JUDUL APLIKASI
# ============================================================
#
# Menampilkan judul utama sistem dan deskripsi singkat
# mengenai metode yang digunakan.
#
# ============================================================

st.title(
    "Sistem Klasifikasi Penyakit Mata Berbasis Citra Fundus"
)

st.caption(
    "Implementasi metode Hybrid ResNet-50 dan "
    "K-Nearest Neighbor (KNN) untuk klasifikasi "
    "penyakit mata berbasis citra fundus retina."
)


# ============================================================
# SIDEBAR MENU
# ============================================================
#
# Sidebar digunakan sebagai navigasi utama aplikasi.
#
# Fitur yang tersedia:
# 1. Klasifikasi Citra Fundus
# 2. Perbandingan Performa Model
#
# ============================================================

st.sidebar.title("Menu Sistem")

menu = st.sidebar.radio(
    "Pilih Fitur",
    [
        "Klasifikasi Citra Fundus",
        "Perbandingan Performa Model"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Penelitian Skripsi\n\n"
    "Hybrid ResNet-50 + KNN\n"
    "untuk klasifikasi penyakit mata."
)


# ============================================================
# FEATURE 1
# KLASIFIKASI CITRA FUNDUS RETINA
# ============================================================

if menu == "Klasifikasi Citra Fundus":

    st.subheader("Klasifikasi Citra Fundus Retina")

    st.markdown("""
    Fitur ini digunakan untuk melakukan klasifikasi citra
    fundus retina ke dalam empat kategori penyakit retina,
    yaitu:

    - Cataract
    - Diabetic Retinopathy
    - Glaucoma
    - Normal
    """)

    st.markdown("---")

    # ========================================================
    # MEMUAT MODEL
    # ========================================================
    #
    # Model dimuat ketika pengguna membuka menu
    # klasifikasi citra.
    #
    # ========================================================

    resnet, knn, scaler, le, checker = load_models()

    # ========================================================
    # UPLOAD CITRA RETINA
    # ========================================================
    #
    # Pengguna dapat mengunggah gambar retina dengan format:
    # - JPG
    # - JPEG
    # - PNG
    #
    # ========================================================

    uploaded = st.file_uploader(
        "Unggah citra fundus retina",
        type=["jpg", "jpeg", "png"]
    )

    # ========================================================
    # PROSES GAMBAR
    # ========================================================

    if uploaded:

        # ----------------------------------------------------
        # Membaca file yang diunggah
        # ----------------------------------------------------
        file_bytes = np.asarray(
            bytearray(uploaded.read()),
            dtype=np.uint8
        )

        # ----------------------------------------------------
        # Decode file menjadi gambar OpenCV
        # ----------------------------------------------------
        img = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        # ----------------------------------------------------
        # Konversi warna BGR → RGB
        # OpenCV menggunakan BGR sedangkan Streamlit RGB
        # ----------------------------------------------------
        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Menampilkan gambar yang diunggah
        # ----------------------------------------------------
        st.image(
            img_rgb,
            caption="Citra Fundus Retina",
            width=350
        )

        st.markdown("---")

        # ====================================================
        # VALIDASI CITRA FUNDUS
        # ====================================================
        #
        # Memastikan gambar yang diunggah benar-benar
        # merupakan citra fundus retina.
        #
        # ====================================================

        if not is_fundus(img_rgb, checker):

            st.error(
                "Citra bukan citra fundus retina. "
                "Silakan unggah citra yang sesuai."
            )

        else:

            st.success(
                "Citra fundus retina berhasil divalidasi."
            )

            # ================================================
            # PREPROCESSING DAN KLASIFIKASI
            # ================================================
            #
            # Tahapan:
            # 1. Resize gambar
            # 2. Normalisasi citra
            # 3. Ekstraksi fitur ResNet-50
            # 4. Normalisasi fitur
            # 5. Prediksi menggunakan KNN
            #
            # ================================================

            x = preprocess_image(img_rgb)

            label = classify_image(
                resnet,
                knn,
                scaler,
                le,
                x
            )

            # ================================================
            # MENAMPILKAN HASIL KLASIFIKASI
            # ================================================

            st.markdown("### Hasil Klasifikasi")

            st.markdown(f"""
            <div style="
                background-color:#E8F5E9;
                padding:20px;
                border-radius:12px;
                border:1px solid #C8E6C9;
                text-align:center;
            ">
                <h2 style="color:#1B5E20;">
                    {label.upper()}
                </h2>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# FEATURE 2
# PERBANDINGAN PERFORMA MODEL
# ============================================================
#
# Menu ini digunakan untuk menampilkan hasil evaluasi
# dan perbandingan performa antara:
#
# 1. ResNet-50 Baseline
# 2. Hybrid ResNet-50 + KNN
#
# Data yang digunakan berasal dari hasil pengujian
# pada Bab IV penelitian.
#
# ============================================================

elif menu == "Perbandingan Performa Model":

    st.subheader("Perbandingan Performa Model")

    st.markdown("""
    Bagian ini menampilkan hasil perbandingan performa antara
    model **ResNet-50 (Baseline)** dan model
    **Hybrid ResNet-50 + K-Nearest Neighbor (KNN)**
    berdasarkan hasil pengujian pada Bab IV.
    """)

    st.markdown("---")

    # ========================================================
    # MEMBACA DATA HASIL PENGUJIAN
    # ========================================================
    #
    # File CSV berisi nilai accuracy dari setiap
    # skenario pengujian model.
    #
    # ========================================================

    df_acc = pd.read_csv(
        "data/accuracy_per_scenario.csv",
        sep=";"
    )

    # Membersihkan spasi pada nama kolom
    df_acc.columns = df_acc.columns.str.strip()

        # ========================================================
    # MENGHITUNG PENINGKATAN PERFORMA (IMPROVEMENT)
    # ========================================================
    #
    # Rumus:
    #
    # Improvement (%) =
    # ((Hybrid - Baseline) / Baseline) × 100
    #
    # Tujuan:
    # Mengukur seberapa besar peningkatan performa
    # model Hybrid dibandingkan model Baseline.
    #
    # ========================================================

    df_acc["Improvement (%)"] = (
        (
            df_acc["Hybrid"] - df_acc["Baseline"]
        )
        / df_acc["Baseline"]
    ) * 100


    # ========================================================
    # MENENTUKAN HASIL TERBAIK
    # ========================================================
    #
    # Bagian ini digunakan untuk mencari:
    #
    # 1. Accuracy tertinggi model Hybrid
    # 2. Skenario pengujian terbaik
    # 3. Nilai improvement tertinggi
    #
    # Informasi ini nantinya ditampilkan pada
    # ringkasan hasil pengujian.
    #
    # ========================================================

    # Mengambil baris dengan accuracy Hybrid tertinggi
    best_row = df_acc.loc[
        df_acc["Hybrid"].idxmax()
    ]

    # Accuracy terbaik model Hybrid
    best_acc = best_row["Hybrid"]

    # Nama skenario terbaik
    best_scenario = best_row["Skenario"]

    # Improvement tertinggi dari seluruh skenario
    best_improvement = df_acc[
        "Improvement (%)"
    ].max()


    # ========================================================
    # RINGKASAN HASIL PENGUJIAN (METRIC CARDS)
    # ========================================================
    #
    # Menampilkan informasi penting dalam bentuk
    # kartu statistik (metric cards).
    #
    # Informasi yang ditampilkan:
    # - Accuracy tertinggi
    # - Skenario terbaik
    # - Improvement tertinggi
    #
    # ========================================================

    st.markdown("### Ringkasan Hasil Pengujian")

    # Membuat 3 kolom sejajar
    col1, col2, col3 = st.columns(3)

    # Accuracy terbaik
    col1.metric(
        "Akurasi Tertinggi",
        f"{best_acc:.2f}%"
    )

    # Nama skenario terbaik
    col2.metric(
        "Skenario Terbaik",
        best_scenario
    )

    # Nilai peningkatan tertinggi
    col3.metric(
        "Peningkatan Tertinggi",
        f"{best_improvement:.2f}%"
    )

    st.markdown("---")


    # ========================================================
    # TABEL PERBANDINGAN AKURASI
    # ========================================================
    #
    # Menampilkan seluruh hasil pengujian dalam
    # bentuk tabel agar lebih mudah dianalisis.
    #
    # Kolom:
    # - Baseline
    # - Hybrid
    # - Improvement (%)
    #
    # ========================================================

    st.markdown("### Tabel Perbandingan Akurasi")

    # Mengatur format tampilan angka menjadi persen
    styled_df = df_acc.style.format({
        "Baseline": "{:.2f}%",
        "Hybrid": "{:.2f}%",
        "Improvement (%)": "{:.2f}%"
    })

    # Menampilkan dataframe ke halaman Streamlit
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

        # ========================================================
    # GRAFIK PERBANDINGAN AKURASI
    # ========================================================
    #
    # Tujuan:
    # Menampilkan perbandingan nilai accuracy antara
    # model Baseline dan model Hybrid pada setiap
    # skenario pengujian.
    #
    # Visualisasi:
    # - Bar Biru  : Baseline
    # - Bar Oranye: Hybrid
    #
    # Dengan grafik ini pengguna dapat melihat
    # perbedaan performa kedua model secara visual.
    #
    # ========================================================

    st.markdown("### Grafik Perbandingan Akurasi")

    # Membuat area grafik
    fig, ax = plt.subplots(figsize=(7, 4))

    # Membuat posisi sumbu X berdasarkan jumlah skenario
    x = np.arange(len(df_acc["Skenario"]))

    # Lebar masing-masing batang grafik
    width = 0.32

    # ====================================================
    # BAR CHART MODEL BASELINE
    # ====================================================

    bars1 = ax.bar(
        x - width / 2,
        df_acc["Baseline"],
        width,
        label="Baseline"
    )

    # ====================================================
    # BAR CHART MODEL HYBRID
    # ====================================================

    bars2 = ax.bar(
        x + width / 2,
        df_acc["Hybrid"],
        width,
        label="Hybrid"
    )

    # ====================================================
    # PENGATURAN LABEL DAN JUDUL GRAFIK
    # ====================================================

    ax.set_xlabel("Skenario")

    ax.set_ylabel("Accuracy (%)")

    ax.set_title(
        "Perbandingan Accuracy Model",
        fontsize=14,
        fontweight='bold'
    )

    # Menampilkan nama skenario pada sumbu X
    ax.set_xticks(x)

    ax.set_xticklabels(df_acc["Skenario"])

    # Menambahkan grid horizontal agar grafik lebih mudah dibaca
    ax.grid(
        axis='y',
        linestyle='--',
        alpha=0.5
    )

    # Menampilkan legenda
    ax.legend()

    # ====================================================
    # MENAMPILKAN NILAI DI ATAS BATANG GRAFIK
    # ====================================================
    #
    # Setiap batang akan diberi label angka
    # sehingga pengguna dapat langsung melihat
    # nilai accuracy tanpa harus memperkirakan
    # dari tinggi batang.
    #
    # ====================================================

    for bars in [bars1, bars2]:

        for bar in bars:

            # Mengambil tinggi batang
            height = bar.get_height()

            # Menampilkan teks nilai accuracy
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=9
            )

    # Menyesuaikan layout grafik agar tidak terpotong
    plt.tight_layout()

    # Menampilkan grafik ke halaman Streamlit
    st.pyplot(fig)

    st.markdown("---")

        # ========================================================
    # GRAFIK PENINGKATAN PERFORMA (IMPROVEMENT)
    # ========================================================
    #
    # Tujuan:
    # Menampilkan besarnya peningkatan performa model
    # Hybrid dibandingkan model Baseline pada setiap
    # skenario pengujian.
    #
    # Rumus Improvement:
    #
    # ((Hybrid - Baseline) / Baseline) × 100
    #
    # Visualisasi:
    # - Sumbu Y : Skenario Pengujian
    # - Sumbu X : Persentase Improvement
    #
    # Semakin panjang batang grafik,
    # semakin besar peningkatan performa
    # yang diperoleh model Hybrid.
    #
    # ========================================================

    st.markdown("### Grafik Peningkatan Performa")

    # Membuat area grafik
    fig2, ax2 = plt.subplots(figsize=(7, 3))

    # ====================================================
    # MEMBUAT HORIZONTAL BAR CHART
    # ====================================================
    #
    # barh() digunakan untuk membuat
    # grafik batang horizontal.
    #
    # ====================================================

    bars = ax2.barh(
        df_acc["Skenario"],
        df_acc["Improvement (%)"]
    )

    # ====================================================
    # PENGATURAN LABEL DAN JUDUL
    # ====================================================

    ax2.set_xlabel("Improvement (%)")

    ax2.set_title(
        "Peningkatan Performa Model Hybrid",
        fontsize=13,
        fontweight='bold'
    )

    # Menambahkan grid vertikal agar grafik lebih mudah dibaca
    ax2.grid(
        axis='x',
        linestyle='--',
        alpha=0.5
    )

    # ====================================================
    # MENAMPILKAN NILAI IMPROVEMENT PADA GRAFIK
    # ====================================================
    #
    # Nilai persentase improvement akan
    # ditampilkan di samping setiap batang.
    #
    # ====================================================

    for bar in bars:

        # Mengambil panjang batang
        width = bar.get_width()

        # Menampilkan nilai improvement
        ax2.text(
            width + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}%",
            va='center'
        )

    # Menyesuaikan layout grafik
    plt.tight_layout()

    # Menampilkan grafik ke halaman Streamlit
    st.pyplot(fig2)

    st.markdown("---")


    # ========================================================
    # INTERPRETASI HASIL
    # ========================================================
    #
    # Bagian ini memberikan penjelasan singkat
    # mengenai hasil pengujian yang diperoleh.
    #
    # Informasi yang ditampilkan:
    # 1. Perbandingan performa model
    # 2. Skenario terbaik
    # 3. Accuracy tertinggi yang diperoleh
    #
    # ========================================================

    st.markdown("### Interpretasi Hasil")

    st.info(
        f"""
        Model Hybrid ResNet-50 + KNN menunjukkan performa
        yang lebih baik dibandingkan model baseline
        ResNet-50 pada seluruh skenario pembagian dataset.

        Skenario terbaik diperoleh pada pembagian data
        {best_scenario} dengan accuracy sebesar
        {best_acc:.2f}%.
        """
    )

    st.markdown("---")


    # ========================================================
    # KESIMPULAN
    # ========================================================
    #
    # Bagian akhir aplikasi yang berisi
    # kesimpulan berdasarkan hasil eksperimen
    # yang telah dilakukan.
    #
    # Kesimpulan ini merangkum manfaat penggunaan
    # metode Hybrid ResNet-50 + KNN dibandingkan
    # pendekatan ResNet-50 end-to-end.
    #
    # ========================================================

    st.markdown("### Kesimpulan")

    st.success(
        """
        Berdasarkan hasil pengujian, metode Hybrid
        ResNet-50 dan K-Nearest Neighbor berhasil
        meningkatkan performa klasifikasi citra
        fundus retina dibandingkan metode ResNet-50
        end-to-end pada penelitian sebelumnya.
        """
    )


# ============================================================
# AKHIR PROGRAM
# ============================================================
#
# Ringkasan Alur Sistem:
#
# 1. User memilih menu pada sidebar
#
# 2. Jika memilih klasifikasi:
#    - Upload citra retina
#    - Validasi fundus retina
#    - Preprocessing citra
#    - Ekstraksi fitur ResNet-50
#    - Klasifikasi menggunakan KNN
#    - Menampilkan hasil prediksi
#
# 3. Jika memilih perbandingan model:
#    - Membaca data hasil pengujian
#    - Menghitung improvement
#    - Menampilkan tabel perbandingan
#    - Menampilkan grafik accuracy
#    - Menampilkan grafik improvement
#    - Menampilkan interpretasi hasil
#    - Menampilkan kesimpulan
#
# ============================================================