import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import create_model

# Stałe
PROCESSED_DIR = "data/processed"
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
EPOCHS = 30
MODEL_SAVE_PATH = "models/gender_model.h5"

# Przygotowanie generatorów danych z augmentacją
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% danych do walidacji
)

# Generator treningowy
train_generator = train_datagen.flow_from_directory(
    PROCESSED_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=42
)

# Generator walidacyjny
validation_generator = train_datagen.flow_from_directory(
    PROCESSED_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True,
    seed=42
)

print(f"Znaleziono {train_generator.samples} obrazów treningowych i {validation_generator.samples} walidacyjnych.")

# Tworzenie modelu
model = create_model()

# Kompilacja
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max'),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

# Trening
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f"Model zapisany jako {MODEL_SAVE_PATH}")