# --- PREPROCESSING FUNCTIONS ---


from typing import Any, Callable

import keras  # type: ignore[import-untyped]
import numpy as np
from PIL import Image

__all__ = ["get_preprocess_fn", "preprocess_image"]


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
