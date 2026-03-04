import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
import tensorflow as tf
from src.features import get_default_feature_vector

MODEL_PATH = 'models/gender_model_multimodal.h5'

_model = None
_detector = None

def load_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def load_detector():
    global _detector
    if _detector is None:
        from mtcnn import MTCNN
        _detector = MTCNN()
    return _detector

def predict_gender(image_path, feature_vector=None):
    model = load_model()
    detector = load_detector()

    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)
    if len(faces) == 0:
        return "Nie wykryto twarzy", 0.0

    x, y, w, h = faces[0]['box']
    x, y = max(0, x), max(0, y)
    face = img[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (224, 224))
    face_array = np.expand_dims(face_resized, axis=0) / 255.0

    if feature_vector is None:
        feature_vector = get_default_feature_vector()
    if isinstance(feature_vector, list):
        feature_vector = np.array(feature_vector, dtype=np.float32)
    feature_vector = np.expand_dims(feature_vector, axis=0)

    pred = model.predict([face_array, feature_vector])[0][0]
    confidence = pred if pred >= 0.5 else 1 - pred
    gender = "Kobieta" if pred < 0.5 else "Mężczyzna"
    return gender, confidence