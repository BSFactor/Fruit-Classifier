import streamlit as st
from keras.utils import model_to_dot  # type: ignore[import-untyped]

from utils.cache import load_my_model
from utils.config import MODEL_CONFIG


for k, a in MODEL_CONFIG.items():
    r = load_my_model(a["file"])
    st.graphviz_chart(model_to_dot(r).to_string(), width="stretch", height=300)
