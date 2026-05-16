# =============================================================================
# app.py - Aplikasi Utama Streamlit untuk Klasifikasi Jenis Material
# Proyek: PKM Internal - Klasifikasi Material menggunakan Image Processing
# =============================================================================

# --- Import Library Standar ---
import streamlit as st          # Framework web interface
import numpy as np              # Komputasi array/matriks
from PIL import Image           # Membuka dan memproses gambar
import io                       # Operasi input/output byte
import time                     # Untuk timestamp dan delay
import datetime                 # Untuk format tanggal/waktu
import uuid                     # Untuk membuat ID unik

# --- Import Modul Lokal ---
from model import load_model, predict_material   # Fungsi model DL
from firebase_utils import init_firebase, save_detection_result  # Fungsi Firebase

# =============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Material Classifier - PKM Internal",  # Judul tab browser
    page_icon="🔬",                                   # Ikon tab browser
    layout="wide",                                    # Layout lebar penuh
    initial_sidebar_state="expanded"                  # Sidebar terbuka default
)

# =============================================================================
# CUSTOM CSS - Styling tambahan agar tampilan lebih profesional
# =============================================================================
st.markdown("""
    <style>
    /* Styling tombol utama */
    .stButton > button {
        background-color: #1a73e8;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        width: 100%;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #1558b0;
    }
    /* Styling kotak hasil prediksi */
    .result-box {
        background-color: #f0f4ff;
        border-left: 5px solid #1a73e8;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 16px;
    }
    /* Styling badge confidence */
    .confidence-high   { color: #1e8449; font-weight: bold; font-size: 1.3em; }
    .confidence-medium { color: #d4ac0d; font-weight: bold; font-size: 1.3em; }
    .confidence-low    { color: #c0392b; font-weight: bold; font-size: 1.3em; }
    /* Header judul */
    .main-title {
        font-size: 2.2em;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 1em;
        color: #555;
        margin-bottom: 24px;
    }
    </style>
""", unsafe_allow_html=True)  # Izinkan HTML mentah

# =============================================================================
# INISIALISASI - Load model dan Firebase sekali saat pertama kali dijalankan
# =============================================================================

# st.cache_resource memastikan model hanya di-load SEKALI, tidak setiap render ulang
@st.cache_resource
def get_model():
    """Fungsi wrapper untuk cache model agar tidak reload terus."""
    return load_model()

@st.cache_resource
def get_firebase():
    """Fungsi wrapper untuk cache koneksi Firebase."""
    return init_firebase()

# Panggil fungsi inisialisasi
model = get_model()        # Load model MobileNetV2
db    = get_firebase()     # Inisialisasi koneksi Firebase

# =============================================================================
# SIDEBAR - Informasi proyek dan panduan penggunaan
# =============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1200px-Tensorflow_logo.svg.png",
             width=80)                              # Logo TensorFlow sebagai dekorasi
    st.markdown("## 🔬 Material Classifier")
    st.markdown("**Proyek PKM Internal**")
    st.markdown("Dosen Peneliti | Prodi Silang")
    st.divider()

    st.markdown("### 📋 Panduan Penggunaan")
    st.markdown("""
    1. Upload gambar sampel material
    2. Pastikan gambar jelas & terang
    3. Klik **Proses Deteksi**
    4. Lihat hasil & confidence score
    5. Data otomatis tersimpan ke Firebase
    """)
    st.divider()

    st.markdown("### 🏷️ Kelas Material")
    # Daftar kelas material yang bisa dideteksi
    classes_info = {
        "⚙️ Logam":    "Besi, Baja, Aluminium, Tembaga",
        "🧴 Polimer":  "Plastik, Karet, Nilon, Resin",
        "🪨 Keramik":  "Porselen, Bata, Kaca, Semen",
        "🔩 Komposit": "CFRP, GFRP, Beton Bertulang",
    }
    for mat, desc in classes_info.items():
        st.markdown(f"**{mat}**: {desc}")

    st.divider()
    st.caption("Versi 1.0.0 | TensorFlow + Streamlit + Firebase")

# =============================================================================
# KONTEN UTAMA HALAMAN
# =============================================================================
st.markdown('<div class="main-title">🔬 Sistem Klasifikasi Jenis Material</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Image Processing & Deep Learning | PKM Internal Penelitian Dosen</div>', unsafe_allow_html=True)
st.divider()

# --- Layout dua kolom: Upload (kiri) | Hasil (kanan) ---
col_upload, col_result = st.columns([1, 1], gap="large")

# ==========================================
# KOLOM KIRI: Area Upload Gambar
# ==========================================
with col_upload:
    st.markdown("### 📁 Upload Gambar Sampel")
    st.markdown("_Format yang didukung: JPG, JPEG, PNG, BMP, TIFF_")

    # Widget file uploader dengan drag-and-drop bawaan Streamlit
    uploaded_file = st.file_uploader(
        label="Drag & drop gambar di sini, atau klik Browse",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],  # Ekstensi yang diizinkan
        help="Upload gambar sampel material yang sudah melalui preprocessing"
    )

    # Jika ada file yang di-upload, tampilkan gambarnya
    if uploaded_file is not None:
        # Baca file sebagai bytes lalu convert ke objek PIL Image
        image_bytes = uploaded_file.read()              # Baca byte data gambar
        image = Image.open(io.BytesIO(image_bytes))     # Buka sebagai PIL Image

        # Tampilkan gambar yang di-upload dengan caption
        st.image(
            image,
            caption=f"📷 {uploaded_file.name}",        # Tampilkan nama file
            use_container_width=True                    # Lebar penuh kolom
        )

        # Tampilkan informasi metadata gambar
        st.markdown("#### 📊 Info Gambar")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.metric("Ukuran Asli", f"{image.size[0]}×{image.size[1]} px")   # Width × Height
        with info_col2:
            st.metric("Mode Warna", image.mode)          # RGB, RGBA, L, dsb
        st.metric("Ukuran File", f"{len(image_bytes)/1024:.1f} KB")            # Konversi bytes ke KB

        st.divider()

        # ==========================================
        # TOMBOL PROSES DETEKSI
        # ==========================================
        detect_btn = st.button(
            "🚀 Proses Deteksi",
            help="Klik untuk memulai klasifikasi material",
            type="primary"
        )

    else:
        # Jika belum ada file, tampilkan pesan panduan
        st.info("⬆️ Silakan upload gambar sampel material terlebih dahulu.")
        detect_btn = False   # Nonaktifkan logika deteksi jika tidak ada file

# ==========================================
# KOLOM KANAN: Area Hasil Deteksi
# ==========================================
with col_result:
    st.markdown("### 📊 Hasil Deteksi")

    # Jalankan deteksi hanya jika tombol ditekan DAN ada file yang di-upload
    if detect_btn and uploaded_file is not None:

        # Tampilkan animasi spinner selama proses berjalan
        with st.spinner("🔄 Sedang memproses gambar... Harap tunggu."):
            time.sleep(0.5)   # Delay singkat untuk UX yang lebih natural

            # --- PANGGIL FUNGSI PREDIKSI dari model.py ---
            prediction_result = predict_material(model, image)
            # Hasil: dict berisi 'class', 'confidence', 'all_scores'

        # Ambil nilai hasil prediksi
        predicted_class  = prediction_result["class"]        # Nama kelas material
        confidence_score = prediction_result["confidence"]    # Skor kepercayaan (0.0 - 1.0)
        all_scores       = prediction_result["all_scores"]    # Skor semua kelas

        # Konversi confidence ke persentase
        confidence_pct   = confidence_score * 100   # Misal: 0.87 → 87.0

        # --- TENTUKAN WARNA BADGE BERDASARKAN CONFIDENCE LEVEL ---
        if confidence_pct >= 75:
            conf_class = "confidence-high"       # Hijau: tinggi (≥75%)
            conf_label = "Tinggi 🟢"
        elif confidence_pct >= 50:
            conf_class = "confidence-medium"     # Kuning: sedang (50–74%)
            conf_label = "Sedang 🟡"
        else:
            conf_class = "confidence-low"        # Merah: rendah (<50%)
            conf_label = "Rendah 🔴"

        # --- TAMPILKAN KOTAK HASIL UTAMA ---
        st.markdown(f"""
        <div class="result-box">
            <h4>✅ Deteksi Selesai!</h4>
            <p><b>Jenis Material:</b> <span style="font-size:1.4em; color:#1a73e8;">
                {predicted_class}
            </span></p>
            <p><b>Confidence Score:</b>
                <span class="{conf_class}">{confidence_pct:.2f}%</span>
                &nbsp;({conf_label})
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # --- TAMPILKAN SKOR SEMUA KELAS SEBAGAI PROGRESS BAR ---
        st.markdown("#### 📈 Distribusi Confidence Semua Kelas")
        for class_name, score in all_scores.items():
            # Tandai kelas dengan skor tertinggi
            label = f"**{class_name}** ✅" if class_name == predicted_class else class_name
            st.markdown(label)
            st.progress(float(score))                         # Progress bar 0.0 - 1.0
            st.caption(f"{score * 100:.2f}%")                 # Tampilkan persentase

        st.divider()

        # ==========================================
        # SIMPAN DATA KE FIREBASE
        # ==========================================
        timestamp_now = datetime.datetime.now()    # Ambil waktu saat ini

        # Buat dictionary data yang akan disimpan ke Firebase
        detection_data = {
            "id"              : str(uuid.uuid4()),                             # ID unik acak
            "nama_file"       : uploaded_file.name,                            # Nama file gambar
            "hasil_prediksi"  : predicted_class,                               # Kelas material
            "confidence_score": round(confidence_pct, 2),                      # Persentase (2 desimal)
            "waktu_pengujian" : timestamp_now.strftime("%Y-%m-%d %H:%M:%S"),   # Format tanggal waktu
            "timestamp_unix"  : int(timestamp_now.timestamp()),                # Unix timestamp (integer)
            "semua_skor"      : {k: round(float(v)*100, 2)                     # Semua skor dalam %
                                  for k, v in all_scores.items()}
        }

        # Panggil fungsi simpan Firebase dari firebase_utils.py
        with st.spinner("☁️ Menyimpan data ke Firebase..."):
            save_success = save_detection_result(db, detection_data)   # True/False

        # Tampilkan notifikasi hasil penyimpanan
        if save_success:
            st.success("✅ Data berhasil disimpan ke Firebase!")
        else:
            st.error("❌ Gagal menyimpan ke Firebase. Periksa konfigurasi.")

        # --- TAMPILKAN RINGKASAN DATA YANG DISIMPAN ---
        with st.expander("📦 Lihat Data yang Disimpan ke Firebase"):
            st.json(detection_data)   # Tampilkan dalam format JSON yang rapi

    elif not detect_btn:
        # Pesan default sebelum tombol ditekan
        st.info("💡 Upload gambar dan tekan **Proses Deteksi** untuk memulai.")
        st.markdown("""
        #### Tentang Sistem Ini
        Sistem ini menggunakan arsitektur **MobileNetV2** dengan transfer learning
        untuk mengklasifikasikan jenis material dari gambar mikroskopik atau
        gambar visual standar.

        **Kelas Material yang Didukung:**
        - ⚙️ **Logam** — Material metalik (besi, baja, dll)
        - 🧴 **Polimer** — Material plastik dan karet
        - 🪨 **Keramik** — Material berbasis tanah liat/mineral
        - 🔩 **Komposit** — Material gabungan (CFRP, beton, dll)
        """)

# =============================================================================
# SECTION RIWAYAT DETEKSI (di bawah kolom utama)
# =============================================================================
st.divider()
st.markdown("### 📜 Riwayat Deteksi (Sesi Ini)")

# Gunakan session_state untuk menyimpan riwayat selama sesi berlangsung
if "history" not in st.session_state:
    st.session_state.history = []   # Inisialisasi list kosong

# Tambahkan data ke riwayat sesi jika deteksi berhasil
if detect_btn and uploaded_file is not None and 'detection_data' in locals():
    st.session_state.history.append({
        "Waktu"     : detection_data["waktu_pengujian"],
        "File"      : detection_data["nama_file"],
        "Prediksi"  : detection_data["hasil_prediksi"],
        "Confidence": f"{detection_data['confidence_score']}%"
    })

# Tampilkan tabel riwayat jika ada data
if st.session_state.history:
    import pandas as pd   # Import pandas untuk tabel
    df_history = pd.DataFrame(st.session_state.history)   # Buat DataFrame
    st.dataframe(df_history, use_container_width=True)     # Tampilkan sebagai tabel
    st.caption(f"Total deteksi dalam sesi ini: {len(st.session_state.history)}")
else:
    st.caption("Belum ada riwayat deteksi dalam sesi ini.")
