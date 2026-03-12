import numpy as np
import streamlit as st
from PIL import Image

from utils.cache import load_my_labels, load_my_model
from utils.config import IMAGE_EXTENSIONS, MODEL_CONFIG
from utils.grad_cam import generate_gradcam_overlay, make_gradcam_heatmap
from utils.preprocessing import get_preprocess_fn, preprocess_image

st.title("Grad-CAM Explorer")
st.write("See *why* a model is making its prediction.")

# --- MODEL SELECTION ---
if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = "MobileNetV2"

_options = list(MODEL_CONFIG.keys())
_default_index = _options.index(st.session_state["model_choice"])

st.selectbox(
    "Choose a model to visualize",
    options=_options,
    index=_default_index,
    key="model_choice",
)

selected_model_name = st.session_state["model_choice"]
selected_model_cfg = MODEL_CONFIG[selected_model_name]


# --- UPLOAD AND PREDICT ---
uploaded_file = st.file_uploader("Choose an image...", type=IMAGE_EXTENSIONS)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # load models (from cache)
    model = load_my_model(model_path_in=selected_model_cfg["file"])
    labels = load_my_labels()

    # Preprocess the image
    processed_image_array = preprocess_image(
        image,
        size=selected_model_cfg["size"],
        preprocess_fn=get_preprocess_fn(selected_model_cfg["family"]),
    )

    # Make prediction
    with st.spinner("Classifying and Building Heatmap..."):
        # Get prediction
        predictions = model.predict(processed_image_array)
        pred_index = np.argmax(predictions, axis=1)[0]
        pred_label = labels[pred_index]
        confidence = np.max(predictions) * 100

        # Generate Grad-CAM
        heatmap = make_gradcam_heatmap(
            processed_image_array, model, selected_model_cfg["last_conv_layer"]
        )

        # Create the overlay
        overlay_image = generate_gradcam_overlay(image, heatmap)

    st.success(f"**Prediction:** {pred_label} ({confidence:.2f}%)")
    st.caption(f"Model: {selected_model_name}")

    # --- DISPLAY RESULTS ---
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", width="stretch")
    with col2:
        st.image(overlay_image, caption="Grad-CAM Heatmap", width="stretch")
