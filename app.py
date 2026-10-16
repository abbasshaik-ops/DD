
import os
import json
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image

MODEL_PATH = "crop_disease_model.keras"
LABELS_PATH = "class_names.json"

model = None
class_names = []

def load_resources():
    global model, class_names
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    if isinstance(class_names, dict):
        class_names = [class_names[str(i)] for i in range(len(class_names))]

def predict_disease(image):
    if image is None:
        return {}, "Please upload a leaf image."

    image = image.convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)

    predictions = model.predict(array, verbose=0)[0]
    indices = np.argsort(predictions)[::-1][:3]

    top_results = {
        str(class_names[int(i)]): float(predictions[int(i)])
        for i in indices
    }

    best = int(np.argmax(predictions))
    details = (
        f"**Predicted class:** {class_names[best]}\n\n"
        f"**Model confidence:** {predictions[best] * 100:.2f}%\n\n"
        "This is an AI prediction, not a confirmed diagnosis."
    )
    return top_results, details

with gr.Blocks(title="Crop Health & Disease Monitoring") as demo:
    gr.Markdown("# 🌿 Crop Health & Disease Monitoring")
    gr.Markdown("Upload a leaf image to predict its disease.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload leaf image")
        result = gr.Label(label="Top predictions", num_top_classes=3)

    details = gr.Markdown()
    button = gr.Button("Detect Disease", variant="primary")
    button.click(predict_disease, inputs=image_input, outputs=[result, details])

if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    load_resources()
    demo.launch()
else:
    raise FileNotFoundError(
        "Model files are missing. Upload crop_disease_model.keras "
        "and class_names.json to the repository."
    )
