import os

import numpy as np
import streamlit as st
from PIL import Image

# Import all our helper functions from utils.py
from utils import (
    IMAGE_EXTENSIONS,
    MODEL_CONFIG,
    get_preprocess_fn,
    load_my_labels,
    load_my_model,
    preprocess_image,
)

# --- SET UP THE PAGE ---
st.set_page_config(page_title="Fruit Classifier", layout="centered")
st.title("Fruit Vegetable Classifier")
st.write("Upload an image of a fruit or vegetable, and we shall find out what it is")


# Choose which model to use
# Initialize default model in session state
if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = "MobileNetV2"

_options = list(MODEL_CONFIG.keys())
_default_index = (
    _options.index(st.session_state["model_choice"])
    if st.session_state["model_choice"] in _options
    else 0
)

st.selectbox(
    "Choose a model",
    options=_options,
    index=_default_index,
    key="model_choice",
)

selected_model_name = st.session_state["model_choice"]
selected_model_cfg = MODEL_CONFIG[selected_model_name]

# --- Early validation: fail fast on misconfiguration ---
_supported_families = {"mobilenet_v2", "efficientnet_v2"}
if selected_model_cfg["family"] not in _supported_families:
    st.error(
        f"Unsupported model family '{selected_model_cfg['family']}'. Supported: {sorted(_supported_families)}."
    )
    st.stop()

model_path = selected_model_cfg["file"]
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

if not os.path.exists("labels.json"):
    st.error("Required file 'labels.json' not found in app directory.")
    st.stop()

model = load_my_model(model_path_in=model_path)
labels = load_my_labels()

# --- 4. THE UPLOAD WIDGET ---
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=IMAGE_EXTENSIONS,
)

if uploaded_file is not None:
    # 1. Display the uploaded image
    image = Image.open(uploaded_file)
    # For animated images (GIF/TIFF), use the first frame
    if getattr(image, "is_animated", False):
        image.seek(0)
    st.image(image, caption="You uploaded this image:", width="stretch")
    st.write("")

    # 2. Preprocess the image
    processed_image = preprocess_image(
        image,
        size=selected_model_cfg["size"],
        preprocess_fn=get_preprocess_fn(selected_model_cfg["family"]),
    )

    # 3. Make a prediction
    with st.spinner("Thinking..."):
        predictions = model.predict(processed_image)

    # 4. Get the result
    pred_index = np.argmax(predictions, axis=1)[0]
    pred_label = labels[pred_index]
    confidence = np.max(predictions) * 100

    # 5. Show the result
    st.success(f"**Prediction:** {pred_label} ({confidence:.2f}%)")
    st.caption(f"Model: {selected_model_name}")
