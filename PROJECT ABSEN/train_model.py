import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

# === Konfigurasi ===
DATASET_PATH = "dataset/faces/"
MODEL_PATH = "models/face_recognition_model.h5"
ENCODINGS_PATH = "models/face_encodings.pkl"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10  # bisa kamu ubah jadi lebih tinggi kalau dataset besar

# === Buat folder models jika belum ada ===
os.makedirs("models", exist_ok=True)

# === Cek dataset ===
if not os.path.exists(DATASET_PATH):
    print("❌ Folder dataset tidak ditemukan:", DATASET_PATH)
    exit()

# === Load data dari folder ===
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_gen.class_indices)
print(f"📊 Jumlah kelas terdeteksi: {num_classes}")

# === Simpan label encoding ===
with open(ENCODINGS_PATH, 'wb') as f:
    pickle.dump(train_gen.class_indices, f)
print(f"💾 Label encoding disimpan ke: {ENCODINGS_PATH}")

# === Bangun model CNN (transfer learning) ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze layer dasar

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Training ===
print("\n🚀 Memulai pelatihan model wajah...")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

# === Simpan model ===
model.save(MODEL_PATH)
print(f"\n✅ Model selesai dilatih dan disimpan ke: {MODEL_PATH}")

# === Tampilkan akurasi ===
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"📈 Akurasi akhir - Train: {train_acc:.2f} | Val: {val_acc:.2f}")
