# Gender Recognition Bot

Projekt realizuje zadanie klasyfikacji płci na podstawie zdjęć twarzy. Wykorzystuje uczenie transferowe (MobileNetV2) oraz detekcję twarzy MTCNN. Interfejs użytkownika zbudowano w Streamlit.

## Funkcjonalności
- Przetwarzanie zdjęć (wykrywanie i kadrowanie twarzy)
- Trenowanie modelu na własnych danych
- Predykcja dla pojedynczego zdjęcia lub wielu
- Prosty interfejs webowy

## Wymagania
- Python 3.9
- Zależności w `requirements.txt` lub `environment.yml`

## Instalacja

1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/twojaz/gender_bot.git
   cd gender_bot
   
2. Instalacja:
       git clone https://github.com/twojaz/gender_bot.git
   cd gender_bot
   conda env create -f environment.yml
   conda activate gender_bot

3. Struktura projektu:
    gender_bot/
├── data/               # dane (raw, processed)
├── models/             # zapisane modele
├── src/                # kod źródłowy
├── app/                # interfejs użytkownika
├── requirements.txt    # zależności
└── README.md

4. Przygotowanie danych:
   - Umieść zdjęcia w `data/raw/male i data/raw/female`
   - Uruchom skrypt do przetwarzania danych:
     ```bash
     python src/data_preprocessing.py
     ```
     
5. Trenowanie modelu
    
    python src/train.py
    Po zakończeniu model zostanie zapisany w models/gender_model.h5.
    Predykcja z linii poleceń
    python src/predict.py --image ścieżka/do/zdjęcia.jpg 
    Uruchomienie interfejsu Streamlit
    streamlit run app/streamlit_app.py
    Po uruchomieniu otwórz przeglądarkę na http://localhost:8501.

## Autor
[GitHub](antonis04)