import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm
import csv
from src.features import get_feature_vector, FEATURE_NAMES

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TARGET_SIZE = (224, 224)

detector = MTCNN()

def detect_and_crop_face(image_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img)
    if len(results) == 0:
        return None
    x, y, w, h = results[0]['box']
    x, y = max(0, x), max(0, y)
    face = img[y:y+h, x:x+w]
    face_resized = cv2.resize(face, TARGET_SIZE)
    return face_resized

def process_category(category):
    input_dir = os.path.join(RAW_DIR, category)
    output_dir = os.path.join(PROCESSED_DIR, category)
    os.makedirs(output_dir, exist_ok=True)
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]
    for filename in tqdm(files, desc=f"Przetwarzanie {category}"):
        input_path = os.path.join(input_dir, filename)
        face = detect_and_crop_face(input_path)
        if face is not None:
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        else:
            print(f"Pominięto: {input_path}")

def create_csv_with_features():
    with open('data/labels_with_features.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['filename', 'label'] + FEATURE_NAMES
        writer.writerow(header)
        for category, label in [('female', 0), ('male', 1)]:
            feature_vec = get_feature_vector(category)
            dir_path = os.path.join(PROCESSED_DIR, category)
            for f in os.listdir(dir_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    row = [os.path.join(category, f), label] + feature_vec.tolist()
                    writer.writerow(row)
    print("Plik labels_with_features.csv utworzony.")

def create_csv():
    with open('data/labels.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        for category in ['female', 'male']:
            label = 0 if category == 'female' else 1
            dir_path = os.path.join(PROCESSED_DIR, category)
            for f in os.listdir(dir_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    writer.writerow([os.path.join(category, f), label])
    print("Plik labels.csv utworzony.")

if __name__ == "__main__":
    print("Rozpoczynam przetwarzanie zdjęć...")
    process_category("female")
    process_category("male")
    print("Gotowe!")
    create_csv()
    create_csv_with_features()