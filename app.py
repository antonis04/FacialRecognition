import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
from src.predict import predict_gender

def predict(image):
    gender, confidence = predict_gender(image, feature_vector=None)
    return f"{gender} (pewność: {confidence:.2f})"

with gr.Blocks(title="Rozpoznawanie płci") as demo:
    gr.Markdown("# Rozpoznawanie płci na podstawie zdjęcia twarzy")
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Zdjęcie")
        output_text = gr.Textbox(label="Wynik")
    submit_btn = gr.Button("Rozpoznaj")
    submit_btn.click(fn=predict, inputs=image_input, outputs=output_text)

if __name__ == "__main__":
    demo.launch()