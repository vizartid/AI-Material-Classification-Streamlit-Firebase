# 🔬 Material Classifier — Sistem Klasifikasi Jenis Material
### Proyek PKM Internal | Penelitian Dosen | Prodi Silang

---

## 📋 Deskripsi Proyek

Sistem klasifikasi jenis material berbasis **Deep Learning** dan **Image Processing** menggunakan:
- **MobileNetV2** sebagai backbone model (Transfer Learning dari ImageNet)
- **Streamlit** sebagai web interface interaktif
- **Firebase** sebagai backend penyimpanan data riwayat deteksi

**Kelas Material yang Dideteksi:**
| No | Kelas | Contoh Material |
|----|-------|-----------------|
| 1  | ⚙️ Logam | Besi, Baja, Aluminium, Tembaga |
| 2  | 🧴 Polimer | Plastik, Karet, Nilon, Resin |
| 3  | 🪨 Keramik | Porselen, Bata, Kaca, Semen |
| 4  | 🔩 Komposit | CFRP, GFRP, Beton Bertulang |

---

## 🗂️ Struktur Direktori Proyek

```
material-classifier/
│
├── app.py                      ← Aplikasi utama Streamlit (web interface)
├── model.py                    ← Model Deep Learning MobileNetV2
├── firebase_utils.py           ← Integrasi Firebase (Realtime DB / Firestore)
│
├── requirements.txt            ← Daftar library yang dibutuhkan
├── firebase_credentials.json   ← [BUAT SENDIRI] Credentials Firebase (jangan di-commit!)
├── .env                        ← [OPSIONAL] Environment variables
├── .gitignore                  ← File yang diabaikan Git
├── README.md                   ← Dokumentasi ini
│
├── material_model_weights.h5   ← [DIBUAT SAAT TRAINING] Bobot model terlatih
│
└── dataset/                    ← [OPSIONAL] Dataset untuk training
    ├── train/
    │   ├── Logam/
    │   ├── Polimer/
    │   ├── Keramik/
    │   └── Komposit/
    └── val/
        ├── Logam/
        ├── Polimer/
        ├── Keramik/
        └── Komposit/
```

---

## ⚙️ Panduan Instalasi

### Prasyarat
- Python 3.9 – 3.11 (TensorFlow 2.15 belum support Python 3.12)
- pip (Python Package Manager)
- Akun Google Firebase (gratis)

### Langkah 1: Clone / Download Proyek
```bash
git clone <url-repo> material-classifier
cd material-classifier
```

### Langkah 2: Buat Virtual Environment (Sangat Disarankan)
```bash
# Buat virtual environment bernama 'venv'
python -m venv venv

# Aktifkan virtual environment:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Langkah 3: Install Semua Library
```bash
pip install -r requirements.txt
```
> ⏳ Proses ini membutuhkan waktu 5–15 menit tergantung koneksi internet.
> TensorFlow berukuran ~500MB.

---

## 🔥 Konfigurasi Firebase

### Langkah 1: Buat Proyek Firebase
1. Buka [Firebase Console](https://console.firebase.google.com/)
2. Klik **"Add project"** → Beri nama proyek → Klik **Continue**
3. Nonaktifkan Google Analytics jika tidak diperlukan → Klik **Create project**

### Langkah 2: Aktifkan Database
**Opsi A — Realtime Database:**
1. Di sidebar kiri → **Build** → **Realtime Database**
2. Klik **Create Database** → Pilih lokasi → Klik **Next**
3. Pilih **"Start in test mode"** → Klik **Enable**
4. Salin URL database (format: `https://nama-proyek-default-rtdb.firebaseio.com/`)

**Opsi B — Cloud Firestore:**
1. Di sidebar kiri → **Build** → **Firestore Database**
2. Klik **Create database** → Pilih mode **Test** → Klik **Next**
3. Pilih lokasi server → Klik **Enable**

### Langkah 3: Download Service Account Key
1. **Project Settings** (ikon roda gigi) → Tab **Service Accounts**
2. Klik **"Generate new private key"** → Konfirmasi → File JSON akan terdownload
3. **Rename** file tersebut menjadi `firebase_credentials.json`
4. **Pindahkan** ke direktori proyek ini

### Langkah 4: Update Konfigurasi
Edit file `firebase_utils.py`, update variabel berikut:
```python
FIREBASE_CREDENTIALS_PATH = "firebase_credentials.json"   # Nama file credentials
FIREBASE_DATABASE_URL = "https://NAMA-PROYEK-default-rtdb.firebaseio.com/"  # URL DB Anda
FIREBASE_BACKEND = "realtime_db"   # Atau "firestore"
```

---

## 🚀 Menjalankan Aplikasi

```bash
# Pastikan virtual environment sudah aktif
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser pada: **http://localhost:8501**

---

## 🧪 Training Model (Jika Dataset Tersedia)

Jika Anda memiliki dataset gambar material sendiri:

```python
# Buat file train.py
from model import train_model

history = train_model(
    train_dir="dataset/train",  # Path folder training
    val_dir="dataset/val",      # Path folder validasi
    epochs=30,                  # Jumlah epoch
    batch_size=32               # Ukuran batch
)
print("Training selesai! Cek file material_model_weights.h5")
```

```bash
python train.py
```

---

## 📡 Struktur Data Firebase

### Realtime Database
```json
{
  "detection_history": {
    "-Nabc123xyz": {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "nama_file": "sampel_logam.jpg",
      "hasil_prediksi": "Logam",
      "confidence_score": 92.34,
      "waktu_pengujian": "2024-01-15 14:30:00",
      "timestamp_unix": 1705312200,
      "semua_skor": {
        "Logam": 92.34,
        "Polimer": 3.12,
        "Keramik": 2.45,
        "Komposit": 2.09
      }
    }
  }
}
```

---

## 🛠️ Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `ModuleNotFoundError: tensorflow` | Jalankan `pip install tensorflow==2.15.0` |
| `FileNotFoundError: firebase_credentials.json` | Download dan letakkan file credentials Firebase |
| `ValueError: FIREBASE_DATABASE_URL` | Update URL database di `firebase_utils.py` |
| Model tidak akurat | Lakukan training dengan dataset material yang sesuai |
| Port 8501 sudah dipakai | Jalankan dengan `streamlit run app.py --server.port 8502` |

---



## 📚 Referensi Teknis

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381) — Sandler et al., 2018
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Firebase Admin SDK Python](https://firebase.google.com/docs/admin/setup)
