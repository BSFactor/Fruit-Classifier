# --- MODEL CONFIGURATION ---
from typing import TypedDict


class ModelConfig(TypedDict):
    "model config shape"

    file: str
    size: tuple[int, int]
    family: str
    last_conv_layer: str


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
"model config, with keys being the model names and values conforming to [ModelConfig] schema"
IMAGE_EXTENSIONS: list[str] = [
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".tif",
    ".tiff",
    ".bmp",
    ".psd",
    ".ai",
    ".eps",
    ".raw",
    ".cr2",
    ".nef",
    ".dng",
]
"some common image extensions"
