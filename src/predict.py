import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

# Załaduj model przy pierwszym wywołaniu
_model = None
_detector = None


def load_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model("models/gender_model.h5")
    return _model


def load_detector():
    global _detector
    if _detector is None:
        _detector = MTCNN()
    return _detector


def predict_gender(image_path):
    """
    Zwraca 'Kobieta' lub 'Mężczyzna' dla zdjęcia pod ścieżką.
    """
    model = load_model()
    detector = load_detector()

    # Wczytaj obraz
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Wykryj twarz
    faces = detector.detect_faces(img)
    if len(faces) == 0:
        return "Nie wykryto twarzy"

    # Weź pierwszą twarz
    x, y, w, h = faces[0]['box']
    x, y = max(0, x), max(0, y)
    face = img[y:y + h, x:x + w]

    # Przygotuj do modelu
    face_resized = cv2.resize(face, (224, 224))
    face_array = np.expand_dims(face_resized, axis=0) / 255.0

    # Predykcja
    pred = model.predict(face_array)[0][0]
    return "Kobieta" if pred < 0.5 else "Mężczyzna"