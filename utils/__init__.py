from typing import Any, Callable

import keras  # type: ignore[import]
import nbformat
import numpy as np
import streamlit as st
from nbconvert import HTMLExporter
from PIL import Image

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


@st.cache_data(show_spinner=True)
def render_notebook_to_html(path: str, _mtime: float, theme: str) -> str:
    """Render a .ipynb to HTML. Cached by file modification time."""
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    exporter = HTMLExporter()
    exporter.exclude_input = False
    exporter.exclude_output = False
    # Align notebook theme with the Streamlit page theme; include theme in cache key above
    if hasattr(exporter, "theme"):
        exporter.theme = "dark" if theme == "dark" else "light"
    body, _resources = exporter.from_notebook_node(nb)
    return body
