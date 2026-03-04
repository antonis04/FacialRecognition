import sys
import os
# Dodajemy główny katalog projektu do ścieżki Pythona, aby importy z src działały
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.model import create_multimodal_model
from src.features import FEATURE_NAMES

# 1. Wczytanie danych z pliku CSV
df = pd.read_csv('data/labels_with_features.csv')
X_img_paths = df['filename'].values          # ścieżki względne (np. "female/zdjecie.jpg")
y = df['label'].values                        # etykiety (0 – kobieta, 1 – mężczyzna)
X_features = df[FEATURE_NAMES].values.astype(np.float32)  # wektory cech

# 2. Podział na zbiór treningowy i walidacyjny (80% / 20%)
img_paths_train, img_paths_val, y_train, y_val, feat_train, feat_val = train_test_split(
    X_img_paths, y, X_features, test_size=0.2, random_state=42, stratify=y
)

batch_size = 16

# 3. Definicja struktury danych dla generatora (output_signature)
#    Oczekujemy dwóch wejść: (obraz, cechy) oraz jednej etykiety
output_signature = (
    (tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),   # batch obrazów
     tf.TensorSpec(shape=(None, len(FEATURE_NAMES)), dtype=tf.float32)),  # batch cech
    tf.TensorSpec(shape=(None,), dtype=tf.float32)                  # batch etykiet
)

def create_dataset(img_paths, features, labels, batch_size, shuffle=True):
    """
    Tworzy nieskończony dataset tf.data z podanych ścieżek, cech i etykiet.
    """
    def gen():
        while True:
            indices = np.arange(len(img_paths))
            if shuffle:
                np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_paths = img_paths[batch_indices]
                batch_features = features[batch_indices]
                batch_labels = labels[batch_indices]

                images = []
                for path in batch_paths:
                    # Uzupełniamy pełną ścieżkę do katalogu processed
                    full_path = os.path.join('data/processed', path)
                    img = load_img(full_path, target_size=(224, 224))
                    img = img_to_array(img) / 255.0
                    images.append(img)

                yield (np.array(images), np.array(batch_features)), np.array(batch_labels)

    dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# 4. Tworzymy zbiory danych
train_dataset = create_dataset(img_paths_train, feat_train, y_train, batch_size, shuffle=True)
val_dataset = create_dataset(img_paths_val, feat_val, y_val, batch_size, shuffle=False)

# 5. Tworzenie modelu
model = create_multimodal_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Liczba kroków na epokę – zaokrąglamy w górę, aby przetworzyć wszystkie próbki
steps_per_epoch = (len(img_paths_train) + batch_size - 1) // batch_size
validation_steps = (len(img_paths_val) + batch_size - 1) // batch_size

# 7. Trenowanie modelu (30 epok)
history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=30,
    validation_data=val_dataset,
    validation_steps=validation_steps
)

# 8. Wizualizacja procesu uczenia
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='trening')
plt.plot(history.history['val_loss'], label='walidacja')
plt.title('Strata w kolejnych epokach')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='trening')
plt.plot(history.history['val_accuracy'], label='walidacja')
plt.title('Dokładność w kolejnych epokach')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()

plt.tight_layout()
plt.show()

# 9. Ewaluacja na zbiorze walidacyjnym
print("\nOcena na zbiorze walidacyjnym:")
val_loss, val_acc = model.evaluate(val_dataset, steps=validation_steps)
print(f"Loss walidacyjny: {val_loss:.4f}")
print(f"Dokładność walidacyjna: {val_acc:.4f}")

# 10. Zapis wytrenowanego modelu
os.makedirs('models', exist_ok=True)
model.save('models/gender_model_multimodal.h5')
print("Model zapisany w 'models/gender_model_multimodal.h5'")