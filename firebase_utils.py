# =============================================================================
# firebase_utils.py - Modul Integrasi Firebase untuk Penyimpanan Data
# Mendukung: Firebase Realtime Database dan Firestore
# =============================================================================

# --- Import Library ---
import os                  # Membaca environment variable
import logging             # Logging untuk debugging
import json                # Parsing JSON konfigurasi
from typing import Optional, Dict, Any   # Type hints untuk kode yang lebih jelas

# Firebase Admin SDK — perlu diinstall: pip install firebase-admin
import firebase_admin
from firebase_admin import credentials, db, firestore

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# KONFIGURASI FIREBASE
# =============================================================================

# Path ke file service account JSON yang diunduh dari Firebase Console
# Ganti dengan path yang sesuai di komputer Anda
FIREBASE_CREDENTIALS_PATH = "firebase_credentials.json"

# URL Realtime Database Firebase (format: https://PROJECT-ID-default-rtdb.firebaseio.com/)
# Ganti dengan URL database Anda dari Firebase Console
FIREBASE_DATABASE_URL = os.environ.get(
    "FIREBASE_DATABASE_URL",        # Baca dari environment variable dulu
    "https://YOUR-PROJECT-ID-default-rtdb.firebaseio.com/"  # Fallback default
)

# Nama koleksi di Firestore (jika menggunakan Firestore)
FIRESTORE_COLLECTION = "detection_history"

# Pilih backend Firebase yang digunakan:
# "realtime_db" untuk Firebase Realtime Database
# "firestore"   untuk Cloud Firestore
FIREBASE_BACKEND = os.environ.get("FIREBASE_BACKEND", "realtime_db")

# =============================================================================
# FUNGSI: Inisialisasi Koneksi Firebase
# =============================================================================

def init_firebase() -> Optional[Any]:
    """
    Menginisialisasi koneksi ke Firebase menggunakan Service Account credentials.

    Cara mendapatkan file credentials:
    1. Buka Firebase Console: https://console.firebase.google.com/
    2. Pilih proyek Anda
    3. Project Settings → Service Accounts
    4. Generate new private key → Download JSON
    5. Simpan file JSON ke direktori proyek ini
    6. Update FIREBASE_CREDENTIALS_PATH di atas

    Returns:
        Firebase DB object jika sukses, None jika gagal
    """

    logger.info("Menginisialisasi koneksi Firebase...")

    # Cek apakah Firebase sudah diinisialisasi sebelumnya
    # Hindari inisialisasi duplikat yang menyebabkan error
    if firebase_admin._apps:
        logger.info("Firebase sudah terinisialisasi sebelumnya. Menggunakan instance yang ada.")
        # Kembalikan referensi database yang sudah ada
        return _get_db_reference()

    # Cek keberadaan file credentials
    if not os.path.exists(FIREBASE_CREDENTIALS_PATH):
        logger.error(
            f"File credentials Firebase tidak ditemukan: '{FIREBASE_CREDENTIALS_PATH}'\n"
            "Solusi:\n"
            "  1. Download service account key dari Firebase Console\n"
            "  2. Simpan sebagai 'firebase_credentials.json' di direktori proyek\n"
            "  3. Update FIREBASE_CREDENTIALS_PATH jika nama file berbeda"
        )
        return None   # Kembalikan None jika gagal

    try:
        # Load credentials dari file JSON service account
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)

        # Inisialisasi Firebase App dengan URL database
        firebase_admin.initialize_app(
            cred,                                          # Credentials objek
            {"databaseURL": FIREBASE_DATABASE_URL}        # Konfigurasi database URL
        )

        logger.info(f"Firebase berhasil diinisialisasi! Backend: {FIREBASE_BACKEND}")
        return _get_db_reference()   # Kembalikan referensi database

    except Exception as e:
        # Tangkap semua error (file rusak, URL salah, network error, dll)
        logger.error(f"Gagal menginisialisasi Firebase: {e}")
        return None


def _get_db_reference() -> Any:
    """
    Helper privat: mengembalikan referensi database berdasarkan backend yang dipilih.

    Returns:
        Objek database reference (Realtime DB atau Firestore client)
    """
    if FIREBASE_BACKEND == "firestore":
        # Kembalikan Firestore client
        return firestore.client()
    else:
        # Kembalikan referensi root Realtime Database
        return db.reference("/")   # Referensi ke root node "/"


# =============================================================================
# FUNGSI: Simpan Data Hasil Deteksi ke Firebase Realtime Database
# =============================================================================

def _save_to_realtime_db(db_ref: Any, data: Dict) -> bool:
    """
    Menyimpan data deteksi ke Firebase Realtime Database.

    Struktur data di Realtime Database:
    /detection_history
        /{push_id_otomatis}
            id              : "uuid-string"
            nama_file       : "gambar.jpg"
            hasil_prediksi  : "Logam"
            confidence_score: 87.34
            waktu_pengujian : "2024-01-15 14:30:00"
            timestamp_unix  : 1705312200
            semua_skor      : {Logam: 87.34, Polimer: 5.12, ...}

    Args:
        db_ref: Referensi root Firebase Realtime Database
        data (dict): Data yang akan disimpan

    Returns:
        bool: True jika berhasil, False jika gagal
    """
    try:
        # Navigasi ke node "detection_history"
        # db.reference("/").child("detection_history") = path: /detection_history
        history_ref = db_ref.child("detection_history")

        # push() membuat node baru dengan ID unik otomatis (seperti UUID Firebase)
        # Ini mencegah overwrite data yang sudah ada
        new_record = history_ref.push(data)

        logger.info(f"Data berhasil disimpan ke Realtime DB. Key: {new_record.key}")
        return True

    except Exception as e:
        logger.error(f"Gagal menyimpan ke Realtime Database: {e}")
        return False


# =============================================================================
# FUNGSI: Simpan Data Hasil Deteksi ke Cloud Firestore
# =============================================================================

def _save_to_firestore(fs_client: Any, data: Dict) -> bool:
    """
    Menyimpan data deteksi ke Cloud Firestore.

    Struktur dokumen di Firestore:
    Collection: detection_history
        Document: {auto-id}
            id              : "uuid-string"
            nama_file       : "gambar.jpg"
            hasil_prediksi  : "Polimer"
            confidence_score: 92.15
            waktu_pengujian : "2024-01-15 14:30:00"
            timestamp_unix  : 1705312200
            semua_skor      : {map}

    Args:
        fs_client: Firestore client object
        data (dict): Data yang akan disimpan

    Returns:
        bool: True jika berhasil, False jika gagal
    """
    try:
        # Akses koleksi "detection_history"
        collection_ref = fs_client.collection(FIRESTORE_COLLECTION)

        # add() membuat dokumen baru dengan ID otomatis
        # Mirip dengan push() di Realtime Database
        timestamp, doc_ref = collection_ref.add(data)

        logger.info(f"Data berhasil disimpan ke Firestore. Doc ID: {doc_ref.id}")
        return True

    except Exception as e:
        logger.error(f"Gagal menyimpan ke Firestore: {e}")
        return False


# =============================================================================
# FUNGSI UTAMA: Simpan Hasil Deteksi (Gateway ke Realtime DB atau Firestore)
# =============================================================================

def save_detection_result(db_ref: Optional[Any], data: Dict) -> bool:
    """
    Fungsi utama untuk menyimpan hasil deteksi material ke Firebase.
    Secara otomatis merutekan ke Realtime Database atau Firestore
    berdasarkan konfigurasi FIREBASE_BACKEND.

    Args:
        db_ref: Referensi database Firebase (dari init_firebase())
        data (dict): Dictionary data hasil deteksi, berisi:
            - id              (str): ID unik UUID
            - nama_file       (str): Nama file gambar
            - hasil_prediksi  (str): Kelas material terprediksi
            - confidence_score(float): Persentase confidence
            - waktu_pengujian (str): Timestamp format string
            - timestamp_unix  (int): Unix timestamp
            - semua_skor      (dict): Skor per kelas dalam %

    Returns:
        bool: True jika berhasil disimpan, False jika gagal
    """

    # Validasi: cek apakah koneksi database valid
    if db_ref is None:
        logger.error("Koneksi Firebase tidak tersedia. Data tidak dapat disimpan.")
        return False

    # Validasi: cek kelengkapan field yang diperlukan
    required_fields = ["id", "nama_file", "hasil_prediksi", "confidence_score", "waktu_pengujian"]
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        logger.error(f"Data tidak lengkap. Field yang hilang: {missing_fields}")
        return False

    logger.info(f"Menyimpan data deteksi ke Firebase ({FIREBASE_BACKEND})...")
    logger.info(f"Data: {data}")

    # Routing berdasarkan backend yang dipilih
    if FIREBASE_BACKEND == "firestore":
        return _save_to_firestore(db_ref, data)   # Simpan ke Cloud Firestore
    else:
        return _save_to_realtime_db(db_ref, data)  # Simpan ke Realtime Database


# =============================================================================
# FUNGSI: Ambil Riwayat Deteksi dari Firebase
# =============================================================================

def get_detection_history(db_ref: Optional[Any], limit: int = 10) -> list:
    """
    Mengambil riwayat deteksi terbaru dari Firebase.
    Berguna untuk menampilkan histori di dashboard.

    Args:
        db_ref: Referensi database Firebase
        limit (int): Jumlah maksimum data yang diambil (default: 10)

    Returns:
        list: List of dict berisi riwayat deteksi, atau [] jika gagal
    """

    if db_ref is None:
        logger.error("Koneksi Firebase tidak tersedia.")
        return []

    try:
        if FIREBASE_BACKEND == "firestore":
            # Firestore: query dengan ordering dan limit
            docs = (db_ref
                    .collection(FIRESTORE_COLLECTION)
                    .order_by("timestamp_unix", direction=firestore.Query.DESCENDING)
                    .limit(limit)
                    .stream())   # Eksekusi query

            # Konversi dokumen Firestore ke list of dict
            history = [doc.to_dict() for doc in docs]

        else:
            # Realtime Database: ambil semua data dari node history
            snapshot = db_ref.child("detection_history").get()

            if snapshot is None:
                return []

            # Konversi dari dict nested ke list dan ambil N terakhir
            all_records = list(snapshot.values())                # Ambil semua value
            history = sorted(
                all_records,
                key=lambda x: x.get("timestamp_unix", 0),       # Urutkan by timestamp
                reverse=True                                     # Terbaru di depan
            )[:limit]                                            # Ambil N teratas

        logger.info(f"Berhasil mengambil {len(history)} riwayat deteksi.")
        return history

    except Exception as e:
        logger.error(f"Gagal mengambil riwayat dari Firebase: {e}")
        return []


# =============================================================================
# FUNGSI: Hapus Riwayat Deteksi (opsional, untuk keperluan reset)
# =============================================================================

def clear_detection_history(db_ref: Optional[Any]) -> bool:
    """
    Menghapus semua riwayat deteksi dari Firebase.
    HATI-HATI: Operasi ini tidak dapat dibatalkan!

    Args:
        db_ref: Referensi database Firebase

    Returns:
        bool: True jika berhasil, False jika gagal
    """

    if db_ref is None:
        return False

    try:
        if FIREBASE_BACKEND == "firestore":
            # Hapus semua dokumen di koleksi Firestore
            docs = db_ref.collection(FIRESTORE_COLLECTION).stream()
            for doc in docs:
                doc.reference.delete()   # Hapus satu per satu
        else:
            # Hapus node detection_history di Realtime Database
            db_ref.child("detection_history").delete()

        logger.info("Semua riwayat deteksi berhasil dihapus.")
        return True

    except Exception as e:
        logger.error(f"Gagal menghapus riwayat: {e}")
        return False


# =============================================================================
# BLOK UTAMA - Hanya berjalan jika file ini dieksekusi langsung (untuk testing)
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST KONEKSI FIREBASE")
    print("="*60)

    # Test inisialisasi Firebase
    database = init_firebase()

    if database:
        print("✅ Koneksi Firebase berhasil!")

        # Test simpan data dummy
        import datetime, uuid
        dummy_data = {
            "id"              : str(uuid.uuid4()),
            "nama_file"       : "test_sample.jpg",
            "hasil_prediksi"  : "Logam",
            "confidence_score": 91.23,
            "waktu_pengujian" : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_unix"  : int(datetime.datetime.now().timestamp()),
            "semua_skor"      : {"Logam": 91.23, "Polimer": 3.45, "Keramik": 2.11, "Komposit": 3.21}
        }

        success = save_detection_result(database, dummy_data)
        print(f"Test simpan data: {'✅ Berhasil' if success else '❌ Gagal'}")
    else:
        print("❌ Koneksi Firebase gagal. Periksa konfigurasi credentials.")

    print("="*60)
