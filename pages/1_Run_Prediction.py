from typing import Any, Callable, TypedDict
import streamlit as st
import keras
from PIL import Image
import numpy as np
import json
import os

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Fruit Classifier", layout="centered")
st.title("Fruit Vegetable Classifier")
st.write("Upload an image of a fruit or vegetable, and we shall find out what it is")


class ModelConfig(TypedDict):
    file: str
    size: tuple[int, int]
    family: str


# --- 2. LOAD THE MODEL AND LABELS ---

MODEL_CONFIG: dict[str, ModelConfig] = {
    "MobileNetV2": {
        "file": "mobilenet_model.keras",
        "size": (224, 224),
        "family": "mobilenet_v2",
    },
    "EfficientNetV2B0": {
        "file": "efficientnet_model.keras",
        "size": (224, 224),
        "family": "efficientnet_v2",
    },
}

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


@st.cache_resource
def load_my_model(model_path_in: str) -> Any:
    loaded_model = keras.models.load_model(model_path_in)
    assert loaded_model is not None
    return loaded_model


@st.cache_data
def load_my_labels():
    with open("labels.json", "r", encoding="utf-8") as f:
        # Load the dictionary and convert string keys back to integers
        labels_from_json = json.load(f)
        label_map = {int(k): v for k, v in labels_from_json.items()}
    return label_map


model = load_my_model(model_path_in=model_path)
labels = load_my_labels()


# --- 3. PREPROCESSING FUNCTION ---
def get_preprocess_fn(family: str) -> Callable[[Any], Any]:
    """Return the appropriate preprocess_input function for a given model family.

    Supported families: 'mobilenet_v2', 'efficientnet_v2'.
    Raises:
        ValueError: if the provided family is not supported.
    """
    if family == "mobilenet_v2":
        return keras.applications.mobilenet_v2.preprocess_input
    if family == "efficientnet_v2":
        return keras.applications.efficientnet_v2.preprocess_input
    raise ValueError(
        f"Unsupported model family '{family}'. Supported: 'mobilenet_v2', 'efficientnet_v2'."
    )


def preprocess_image(
    img_pil: Image.Image,
    size: tuple[int, int],
    preprocess_fn: Callable[[Any], Any],
) -> np.ndarray:
    # Ensure RGB
    img = img_pil.convert("RGB")

    # Resize the image
    img = img.resize(size)

    # Convert to numpy array
    img_array = np.array(img)

    # Add the "batch" dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Apply model-specific preprocessing
    img_array = preprocess_fn(img_array)

    return img_array


# --- 4. THE UPLOAD WIDGET ---
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif", "ico"],
)

if uploaded_file is not None:
    # 1. Display the uploaded image
    image = Image.open(uploaded_file)
    # For animated images (GIF/TIFF), use the first frame
    if getattr(image, "is_animated", False):
        image.seek(0)
    st.image(image, caption="You uploaded this image:", use_container_width=True)
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
