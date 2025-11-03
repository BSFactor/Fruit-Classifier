import json
from typing import Any, Callable, TypedDict

import cv2
import keras
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.models import Model
from PIL import Image


# --- MODEL CONFIGURATION ---
class ModelConfig(TypedDict):
    file: str
    size: tuple[int, int]
    family: str
    last_conv_layer: str


# Import the specific preprocessing functions

MODEL_CONFIG: dict[str, ModelConfig] = {
    "MobileNetV2": {
        "file": "mobilenet_model.keras",
        "size": (224, 224),
        "family": "mobilenet_v2",
        "last_conv_layer": "out_relu",
    },
    "EfficientNetV2B0": {
        "file": "efficientnet_model.keras",
        "size": (224, 224),
        "family": "efficientnet_v2",
        "last_conv_layer": "top_conv",
    },
}

# --- CACHED HELPER FUNCTIONS ---


@st.cache_resource
def load_my_model(model_path_in: str) -> Any:
    loaded_model = keras.models.load_model(model_path_in)
    assert loaded_model is not None
    return loaded_model


@st.cache_data
def load_my_labels():
    with open("labels.json", "r", encoding="utf-8") as f:
        labels_from_json = json.load(f)
        label_map = {int(k): v for k, v in labels_from_json.items()}
    return label_map


# --- PREPROCESSING FUNCTIONS ---


def get_preprocess_fn(family: str) -> Callable[[Any], Any]:
    if family == "mobilenet_v2":
        return keras.applications.mobilenet_v2.preprocess_input
    if family == "efficientnet_v2":
        return keras.applications.efficientnet_v2.preprocess_input
    raise ValueError(f"Unsupported model family '{family}'.")


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


# --- GRAD-CAM LOGIC ---


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array])

        if isinstance(predictions, list):
            predictions_tensor = predictions[0]
        else:
            predictions_tensor = predictions

        if pred_index is None:
            pred_index = tf.argmax(predictions_tensor[0])

        class_channel = predictions_tensor[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# --- GRAD-CAM DISPLAY FUNCTION ---


def generate_gradcam_overlay(img_pil: Image.Image, heatmap: np.ndarray, alpha=0.4):
    """
    Overlays a heatmap on the original PIL image.
    This works IN-MEMORY, no files are saved.
    """
    # Convert original PIL image to NumPy array (RGB)
    img_rgb = np.array(img_pil.convert("RGB"))

    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))

    # Rescale heatmap to 0-255 and apply colormap
    heatmap_8bit = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)

    # Convert heatmap to RGB.
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend the original image with the heatmap
    superimposed_img = cv2.addWeighted(heatmap_rgb, alpha, img_rgb, 1 - alpha, 0)

    return superimposed_img
