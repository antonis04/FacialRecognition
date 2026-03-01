import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm  # opcjonalnie, do paska postępu (pip install tqdm)
import csv

# Konfiguracja
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TARGET_SIZE = (224, 224)  # rozmiar, do którego przeskalujemy twarz

# Inicjalizacja detektora MTCNN
detector = MTCNN()

def detect_and_crop_face(image_path):
    """
    Wykrywa twarz na zdjęciu, zwraca wyciętą i przeskalowaną twarz lub None jeśli nie wykryto.
    """
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img)
    if len(results) == 0:
        print(f"Nie wykryto twarzy w: {image_path}")
        return None
    # Bierzemy pierwszą wykrytą twarz (zakładamy, że jest jedna)
    x, y, w, h = results[0]['box']
    # Zabezpieczenie przed ujemnymi współrzędnymi
    x, y = max(0, x), max(0, y)
    face = img[y:y+h, x:x+w]
    # Skalowanie do docelowego rozmiaru
    face_resized = cv2.resize(face, TARGET_SIZE)
    return face_resized

def process_category(category):
    """
    category: 'female' lub 'male'
    """
    input_dir = os.path.join(RAW_DIR, category)
    output_dir = os.path.join(PROCESSED_DIR, category)
    os.makedirs(output_dir, exist_ok=True)

    # Lista dozwolonych rozszerzeń
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # Pobierz wszystkie pliki w katalogu
    files = [f for f in os.listdir(input_dir) 
             if f.lower().endswith(valid_extensions)]

    for filename in tqdm(files, desc=f"Przetwarzanie {category}"):
        input_path = os.path.join(input_dir, filename)
        face = detect_and_crop_face(input_path)
        if face is not None:
            # Zapisz przetworzone zdjęcie
            output_path = os.path.join(output_dir, filename)
            # Konwersja z RGB do BGR przed zapisem przez OpenCV
            cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        else:
            print(f"Pominięto: {input_path}")

if __name__ == "__main__":
    print("Rozpoczynam przetwarzanie zdjęć...")
    process_category("female")
    process_category("male")
    print("Gotowe!")


    def create_csv():
        with open('data/labels.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'label'])
            for category in ['female', 'male']:
                label = 0 if category == 'female' else 1  # np. 0 - kobieta, 1 - mężczyzna
                dir_path = os.path.join(PROCESSED_DIR, category)
                for f in os.listdir(dir_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        writer.writerow([os.path.join(category, f), label])
        print("Plik labels.csv utworzony.")


    # Wywołaj po przetworzeniu:
    create_csv()