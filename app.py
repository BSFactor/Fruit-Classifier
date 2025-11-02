import streamlit as st
from tensorflow import keras  # type: ignore[import] # pylint: disable=E0611,W0611
from PIL import Image
import numpy as np
import json

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Fruit Classifier", layout="centered")
st.title("Fruit Vegetable Classifier")
st.write("Upload an image of a fruit or vegetable, and we shall find out what it is")

# --- 2. LOAD THE MODEL AND LABELS ---
MODEL_CONFIG = {
    "MobileNetV2": {
        "file": "mobilenet_model.keras",
        "size": (224, 224),
        "preprocess": keras.applications.mobilenet_v2.preprocess_input,
    },
    "EfficientNetV2B0": {
        "file": "efficientnet_model.keras",
        "size": (224, 224),
        "preprocess": keras.applications.efficientnet_v2.preprocess_input,
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


@st.cache_resource
def load_my_model(model_path: str):
    loaded_model = keras.models.load_model(model_path)
    return loaded_model


@st.cache_data
def load_my_labels():
    with open("labels.json", "r", encoding="utf-8") as f:
        # Load the dictionary and convert string keys back to integers
        labels_from_json = json.load(f)
        label_map = {int(k): v for k, v in labels_from_json.items()}
    return label_map


model = load_my_model(selected_model_cfg["file"])
labels = load_my_labels()


# --- 3. PREPROCESSING FUNCTION ---
def preprocess_image(img_pil, size, preprocess_fn):
    # Ensure RGB
    img = img_pil.convert("RGB")

    # Resize the image
    img = img.resize(size)

    # Convert to numpy array
    img_array = np.array(img)

    # Add the "batch" dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Apply model-specific preprocessing
    img_preprocessed = preprocess_fn(img_array)

    return img_preprocessed


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
        preprocess_fn=selected_model_cfg["preprocess"],
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
