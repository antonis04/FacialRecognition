import streamlit as st
from src.predict import predict_gender
from PIL import Image
import cv2
import numpy as np
import tempfile

st.set_page_config(page_title="Rozpoznawanie płci", page_icon="👤")
st.title("👤 Rozpoznawanie płci na podstawie zdjęcia twarzy")

uploaded_file = st.file_uploader("Wybierz zdjęcie...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Wyświetl przesłane zdjęcie
    image = Image.open(uploaded_file)
    st.image(image, caption="Twoje zdjęcie", use_column_width=True)

    # Zapisz tymczasowo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Przycisk do analizy
    if st.button("Rozpoznaj płeć"):
        with st.spinner("Analizuję..."):
            result = predict_gender(tmp_path)
        st.success(f"**Wynik:** {result}")