import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.model import create_multimodal_model
from src.features import FEATURE_NAMES

df = pd.read_csv('data/labels_with_features.csv')
X_img_paths = df['filename'].values
y = df['label'].values
X_features = df[FEATURE_NAMES].values.astype(np.float32)

img_paths_train, img_paths_val, y_train, y_val, feat_train, feat_val = train_test_split(
    X_img_paths, y, X_features, test_size=0.2, random_state=42, stratify=y
)

batch_size = 16

def generator(img_paths, features, labels):
    while True:
        indices = np.arange(len(img_paths))
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_paths = img_paths[batch_idx]
            batch_features = features[batch_idx]
            batch_labels = labels[batch_idx]

            images = []
            for path in batch_paths:
                full_path = os.path.join('data/processed', path)
                img = load_img(full_path, target_size=(224, 224))
                img = img_to_array(img) / 255.0
                images.append(img)

            # UWAGA: zwracamy krotkę ( (obraz, cechy), etykiety )
            yield (np.array(images), np.array(batch_features)), np.array(batch_labels)

output_signature = (
    (tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
     tf.TensorSpec(shape=(None, len(FEATURE_NAMES)), dtype=tf.float32)),
    tf.TensorSpec(shape=(None,), dtype=tf.float32)
)

train_dataset = tf.data.Dataset.from_generator(
    lambda: generator(img_paths_train, feat_train, y_train),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: generator(img_paths_val, feat_val, y_val),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)

steps_per_epoch = len(img_paths_train) // batch_size
validation_steps = len(img_paths_val) // batch_size

model = create_multimodal_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

callbacks_phase1 = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
    ModelCheckpoint('models/best_model_phase1.h5', monitor='val_accuracy', save_best_only=True)
]

history1 = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=30,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=callbacks_phase1
)

# fine-tuning
model.base_model.trainable = True
for layer in model.base_model.layers[:100]:
    layer.trainable = False
for layer in model.base_model.layers[100:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

callbacks_phase2 = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
    ModelCheckpoint('models/best_model_final.h5', monitor='val_accuracy', save_best_only=True)
]

history2 = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=callbacks_phase2
)

model.load_weights('models/best_model_final.h5')
val_loss, val_acc = model.evaluate(val_dataset, steps=validation_steps)
print(f"Wyniki: loss={val_loss:.4f}, acc={val_acc:.4f}")

model.save('models/gender_model_multimodal.h5')