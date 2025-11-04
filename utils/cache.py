import json
import typing

import keras  # type: ignore[import]
import streamlit as st

# --- CACHED HELPER FUNCTIONS ---


@st.cache_resource
def load_my_model(model_path_in: str) -> keras.Model:
    loaded_model = keras.models.load_model(model_path_in)
    assert loaded_model is not None
    return typing.cast(keras.Model, loaded_model)


@st.cache_data
def load_my_labels() -> dict[int, str]:
    with open("labels.json", "r", encoding="utf-8") as f:
        labels_from_json = json.load(f)
        label_map = {int(k): v for k, v in labels_from_json.items()}
    return label_map
