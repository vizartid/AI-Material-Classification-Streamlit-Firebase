# =============================================================================
# model.py - Modul Model Deep Learning untuk Klasifikasi Material
# Arsitektur: MobileNetV2 (Transfer Learning) + Custom Classifier Head
# Framework: TensorFlow / Keras
# =============================================================================

# --- Import Library ---
import numpy as np                         # Komputasi numerik dan array
import tensorflow as tf                    # Framework Deep Learning utama
from tensorflow.keras import layers, Model # Blok pembangun model Keras
from tensorflow.keras.applications import MobileNetV2  # Arsitektur pre-trained
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Preprocessing standar
from PIL import Image                      # Pemrosesan gambar
import os                                  # Operasi sistem file
import logging                             # Logging untuk debugging

# Konfigurasi logging agar pesan debug tampil di terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# KONFIGURASI GLOBAL MODEL
# =============================================================================

# Ukuran input gambar standar MobileNetV2 (height, width, channels)
IMG_HEIGHT = 224    # Tinggi gambar dalam piksel
IMG_WIDTH  = 224    # Lebar gambar dalam piksel
IMG_CHANNELS = 3    # 3 channel warna: R, G, B

# Path file untuk menyimpan/memuat bobot model yang sudah dilatih
MODEL_WEIGHTS_PATH = "material_model_weights.h5"   # Format HDF5

# Definisi kelas material yang akan diklasifikasikan
# CATATAN: Urutan ini harus konsisten dengan label saat training!
CLASS_NAMES = [
    "Logam",      # Index 0: Material metalik (besi, baja, aluminium, tembaga)
    "Polimer",    # Index 1: Material plastik, karet, nilon, resin
    "Keramik",    # Index 2: Porselen, bata, kaca, semen
    "Komposit",   # Index 3: CFRP, GFRP, beton bertulang
]

# Jumlah kelas = panjang list CLASS_NAMES (4 dalam kasus ini)
NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# FUNGSI: Membangun Arsitektur Model MobileNetV2
# =============================================================================

def build_mobilenetv2_model(num_classes: int, freeze_base: bool = True) -> Model:
    """
    Membangun model klasifikasi material menggunakan MobileNetV2 sebagai
    backbone (feature extractor) dengan custom classification head di atasnya.

    Teknik yang digunakan: Transfer Learning
    - Base model (MobileNetV2) di-load dengan bobot ImageNet
    - Layer-layer base model di-freeze (tidak dilatih ulang)
    - Hanya custom head yang dilatih

    Args:
        num_classes (int): Jumlah kelas output yang diinginkan
        freeze_base (bool): Jika True, bekukan bobot base model (default: True)

    Returns:
        tf.keras.Model: Model yang sudah dikompilasi dan siap dilatih/digunakan
    """

    logger.info("Membangun arsitektur MobileNetV2...")

    # --------------------------------------------------
    # STEP 1: Load Base Model (MobileNetV2 + ImageNet weights)
    # --------------------------------------------------
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),  # Dimensi input gambar
        include_top=False,      # TIDAK include layer klasifikasi bawaan ImageNet
        weights="imagenet"      # Load pre-trained weights dari dataset ImageNet
    )

    # Bekukan semua layer base model agar bobotnya tidak berubah saat fine-tuning awal
    base_model.trainable = not freeze_base   # False = semua layer di-freeze
    logger.info(f"Base model trainable: {base_model.trainable}")

    # --------------------------------------------------
    # STEP 2: Bangun Custom Classification Head
    # --------------------------------------------------
    # Input layer — mendefinisikan bentuk tensor masukan
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="input_layer")

    # Lewatkan input ke base model untuk ekstraksi fitur
    # training=False → BatchNorm di base model berjalan dalam mode inference
    x = base_model(inputs, training=False)

    # Global Average Pooling: mengubah feature map (7×7×1280) → vektor (1280,)
    # Lebih ringan daripada Flatten dan mengurangi overfitting
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Dropout untuk regularisasi — matikan 30% neuron secara acak saat training
    # Mencegah model terlalu menghafal data training (overfitting)
    x = layers.Dropout(rate=0.3, name="dropout_regularizer")(x)

    # Dense layer pertama — belajar kombinasi fitur tingkat tinggi
    x = layers.Dense(
        units=128,           # 128 neuron
        activation="relu",   # ReLU: f(x) = max(0, x) — non-linearitas umum
        name="dense_features"
    )(x)

    # Batch Normalization — normalisasi output agar training lebih stabil
    x = layers.BatchNormalization(name="batch_norm")(x)

    # Dropout kedua sebelum layer output
    x = layers.Dropout(rate=0.2, name="dropout_final")(x)

    # Output layer — jumlah unit = jumlah kelas, aktivasi softmax
    # Softmax: mengubah nilai mentah menjadi distribusi probabilitas (sum = 1.0)
    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",   # Output probabilitas untuk setiap kelas
        name="output_classification"
    )(x)

    # --------------------------------------------------
    # STEP 3: Buat Model Lengkap dan Kompilasi
    # --------------------------------------------------
    model = Model(inputs=inputs, outputs=outputs, name="MaterialClassifier_MobileNetV2")

    # Kompilasi model dengan:
    # - optimizer Adam: algoritma gradient descent adaptif yang populer
    # - loss categorical_crossentropy: cocok untuk multi-class classification
    # - metrics accuracy: pantau akurasi selama training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # lr kecil untuk fine-tuning
        loss="categorical_crossentropy",     # Loss function multi-class
        metrics=["accuracy"]                 # Pantau akurasi
    )

    logger.info(f"Model berhasil dibangun. Total parameter: {model.count_params():,}")
    return model


# =============================================================================
# FUNGSI: Load Model (dengan bobot tersimpan atau buat baru)
# =============================================================================

def load_model() -> Model:
    """
    Load model yang siap digunakan untuk inferensi.

    Logika:
    1. Jika file bobot model SUDAH ADA → load bobot tersimpan (model terlatih)
    2. Jika BELUM ADA → buat model baru dengan bobot random (perlu dilatih dahulu)
       Pada demo/showcase, digunakan bobot random untuk simulasi fungsi sistem.

    Returns:
        tf.keras.Model: Model siap pakai untuk prediksi
    """

    logger.info("Memulai proses load model...")

    # Bangun arsitektur model terlebih dahulu
    model = build_mobilenetv2_model(num_classes=NUM_CLASSES, freeze_base=True)

    # Cek apakah file bobot model sudah tersedia
    if os.path.exists(MODEL_WEIGHTS_PATH):
        # --- Skenario A: Bobot model ditemukan → Load bobot ---
        logger.info(f"File bobot ditemukan: {MODEL_WEIGHTS_PATH}")
        try:
            model.load_weights(MODEL_WEIGHTS_PATH)   # Load bobot dari file .h5
            logger.info("Bobot model berhasil dimuat!")
        except Exception as e:
            logger.warning(f"Gagal memuat bobot: {e}. Menggunakan bobot ImageNet default.")
    else:
        # --- Skenario B: Tidak ada bobot tersimpan → Pakai bobot ImageNet ---
        # Ini terjadi pada demo pertama kali sebelum training dilakukan
        logger.warning(
            f"File '{MODEL_WEIGHTS_PATH}' tidak ditemukan. "
            "Menggunakan bobot pre-trained ImageNet saja. "
            "Untuk akurasi optimal, latih model dengan data material terlebih dahulu."
        )

    logger.info("Model siap digunakan untuk inferensi.")
    return model


# =============================================================================
# FUNGSI: Preprocessing Gambar untuk Input Model
# =============================================================================

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Melakukan preprocessing gambar agar sesuai dengan format input model.

    Tahapan preprocessing:
    1. Konversi ke mode RGB (hindari alpha channel atau grayscale)
    2. Resize ke dimensi standar model (224×224 piksel)
    3. Konversi ke array NumPy float32
    4. Normalisasi pixel value ke rentang [-1, 1] (standar MobileNetV2)
    5. Tambahkan dimensi batch (shape: (1, 224, 224, 3))

    Args:
        image (PIL.Image): Gambar input dari user

    Returns:
        np.ndarray: Tensor siap diumpankan ke model, shape (1, 224, 224, 3)
    """

    # Konversi ke RGB — pastikan 3 channel, bukan RGBA atau grayscale
    image_rgb = image.convert("RGB")

    # Resize gambar ke ukuran yang diharapkan model
    # Image.LANCZOS: algoritma resampling berkualitas tinggi
    image_resized = image_rgb.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)

    # Konversi PIL Image → NumPy array dengan tipe data float32
    # Shape setelah ini: (224, 224, 3)
    image_array = np.array(image_resized, dtype=np.float32)

    # Normalisasi menggunakan fungsi bawaan MobileNetV2
    # Mengubah rentang [0, 255] → [-1, 1]
    # Formula: pixel_normalized = (pixel / 127.5) - 1.0
    image_preprocessed = preprocess_input(image_array)

    # Tambahkan dimensi batch di axis 0
    # Shape berubah: (224, 224, 3) → (1, 224, 224, 3)
    # Model mengharapkan input dalam batch, minimal batch size = 1
    image_batch = np.expand_dims(image_preprocessed, axis=0)

    return image_batch


# =============================================================================
# FUNGSI: Prediksi/Klasifikasi Material
# =============================================================================

def predict_material(model: Model, image: Image.Image) -> dict:
    """
    Melakukan prediksi/klasifikasi jenis material dari gambar input.

    Proses:
    1. Preprocessing gambar
    2. Inferensi model (forward pass)
    3. Parsing output probabilitas
    4. Kembalikan hasil dalam format dictionary

    Args:
        model (tf.keras.Model): Model yang sudah di-load
        image (PIL.Image): Gambar input dari user

    Returns:
        dict: Dictionary berisi:
            - 'class'      (str): Nama kelas material dengan probabilitas tertinggi
            - 'confidence' (float): Skor kepercayaan kelas terprediksi (0.0 - 1.0)
            - 'all_scores' (dict): Probabilitas untuk setiap kelas {nama: skor}
    """

    logger.info(f"Memulai prediksi pada gambar: {image.size}")

    # --- STEP 1: Preprocessing ---
    input_tensor = preprocess_image(image)
    logger.info(f"Shape input tensor: {input_tensor.shape}")   # Seharusnya (1, 224, 224, 3)

    # --- STEP 2: Inferensi Model ---
    # model.predict() mengembalikan array probabilitas
    # Shape output: (1, NUM_CLASSES) → misal (1, 4) untuk 4 kelas
    predictions = model.predict(input_tensor, verbose=0)
    logger.info(f"Raw predictions: {predictions}")

    # Ambil baris pertama (batch size = 1, jadi hanya ada 1 baris)
    prob_array = predictions[0]   # Shape: (NUM_CLASSES,) → misal (4,)

    # --- STEP 3: Parsing Hasil ---
    # Cari index kelas dengan probabilitas tertinggi
    predicted_index = int(np.argmax(prob_array))    # Index kelas terprediksi

    # Ambil nama kelas dan skor confidence
    predicted_class  = CLASS_NAMES[predicted_index]    # Nama kelas material
    confidence_score = float(prob_array[predicted_index])  # Skor 0.0 - 1.0

    # Buat dictionary semua skor untuk semua kelas
    # Key: nama kelas, Value: probabilitas
    all_scores = {
        CLASS_NAMES[i]: float(prob_array[i])
        for i in range(NUM_CLASSES)
    }

    logger.info(f"Prediksi: {predicted_class} ({confidence_score*100:.2f}%)")

    # Kembalikan hasil lengkap
    return {
        "class"      : predicted_class,   # Nama kelas terprediksi
        "confidence" : confidence_score,  # Skor tertinggi (float 0-1)
        "all_scores" : all_scores         # Semua skor per kelas
    }


# =============================================================================
# FUNGSI: Training Model (Digunakan saat Dataset Tersedia)
# =============================================================================

def train_model(
    train_dir: str,        # Direktori dataset training
    val_dir: str,          # Direktori dataset validasi
    epochs: int = 20,      # Jumlah epoch training
    batch_size: int = 32   # Jumlah sampel per batch
) -> dict:
    """
    Fungsi untuk melatih model dengan dataset material yang tersedia.

    Struktur direktori dataset yang diperlukan:
    dataset/
    ├── train/
    │   ├── Logam/      ← Gambar logam untuk training
    │   ├── Polimer/    ← Gambar polimer untuk training
    │   ├── Keramik/    ← Gambar keramik untuk training
    │   └── Komposit/   ← Gambar komposit untuk training
    └── val/
        ├── Logam/      ← Gambar logam untuk validasi
        ├── Polimer/    ← Gambar polimer untuk validasi
        ├── Keramik/    ← Gambar keramik untuk validasi
        └── Komposit/   ← Gambar komposit untuk validasi

    Args:
        train_dir  (str): Path ke folder training
        val_dir    (str): Path ke folder validasi
        epochs     (int): Jumlah epoch (default: 20)
        batch_size (int): Ukuran batch (default: 32)

    Returns:
        dict: Riwayat training (loss dan akurasi per epoch)
    """

    logger.info(f"Memulai training dengan {epochs} epoch...")

    # --- Data Augmentation untuk Training ---
    # Augmentasi meningkatkan variasi data agar model lebih robust
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,  # Normalisasi MobileNetV2
        rotation_range=20,       # Rotasi acak ±20 derajat
        width_shift_range=0.1,   # Geser horizontal ±10%
        height_shift_range=0.1,  # Geser vertikal ±10%
        horizontal_flip=True,    # Balik horizontal acak
        zoom_range=0.1,          # Zoom acak ±10%
        fill_mode="nearest"      # Isi piksel kosong dengan nilai terdekat
    )

    # Validasi: hanya normalisasi, tidak ada augmentasi
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # Load gambar dari direktori secara otomatis
    train_generator = train_datagen.flow_from_directory(
        train_dir,                               # Direktori training
        target_size=(IMG_HEIGHT, IMG_WIDTH),     # Resize otomatis ke 224×224
        batch_size=batch_size,                   # Ukuran batch
        class_mode="categorical",                # One-hot encoding untuk multi-class
        shuffle=True                             # Acak urutan data tiap epoch
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,                                 # Direktori validasi
        target_size=(IMG_HEIGHT, IMG_WIDTH),     # Resize otomatis ke 224×224
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False                            # Jangan acak data validasi
    )

    # Bangun model baru untuk training
    model = build_mobilenetv2_model(num_classes=NUM_CLASSES, freeze_base=True)

    # Callbacks: fungsi yang dipanggil otomatis selama training
    callbacks = [
        # EarlyStopping: hentikan training jika validasi tidak meningkat 5 epoch
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",    # Pantau akurasi validasi
            patience=5,                # Toleransi 5 epoch tanpa peningkatan
            restore_best_weights=True  # Kembalikan bobot terbaik
        ),
        # ModelCheckpoint: simpan bobot terbaik secara otomatis
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_WEIGHTS_PATH,   # Simpan ke file .h5
            monitor="val_accuracy",        # Berdasarkan akurasi validasi
            save_best_only=True,           # Hanya simpan jika lebih baik
            save_weights_only=True         # Hanya simpan bobot, bukan arsitektur
        ),
        # ReduceLROnPlateau: kurangi learning rate jika training mandek
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",   # Pantau validation loss
            factor=0.5,           # Kurangi lr menjadi 50% dari nilai saat ini
            patience=3,           # Tunggu 3 epoch sebelum kurangi lr
            min_lr=1e-7           # Batas minimum learning rate
        )
    ]

    # --- Mulai Training ---
    history = model.fit(
        train_generator,                # Data training
        epochs=epochs,                  # Jumlah epoch
        validation_data=val_generator,  # Data validasi
        callbacks=callbacks,            # Callback list
        verbose=1                       # Tampilkan progress bar di terminal
    )

    logger.info("Training selesai! Bobot terbaik tersimpan.")
    return history.history   # Kembalikan dict riwayat {loss, accuracy, val_loss, val_accuracy}


# =============================================================================
# BLOK UTAMA - Hanya berjalan jika file ini dieksekusi langsung
# =============================================================================
if __name__ == "__main__":
    # Test build model dan tampilkan ringkasannya
    print("\n" + "="*60)
    print("TEST BUILD MODEL MATERIALCLASSIFIER")
    print("="*60)

    test_model = build_mobilenetv2_model(NUM_CLASSES)
    test_model.summary()   # Tampilkan ringkasan arsitektur layer

    print(f"\nJumlah kelas: {NUM_CLASSES}")
    print(f"Nama kelas  : {CLASS_NAMES}")
    print(f"Input shape : ({IMG_HEIGHT}, {IMG_WIDTH}, {IMG_CHANNELS})")
    print("="*60)
