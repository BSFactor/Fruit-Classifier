import os

import streamlit as st
import streamlit.components.v1 as components

from utils import render_notebook_to_html

st.set_page_config(page_title="Project Notebooks", layout="wide")

# Source .ipynb notebooks
TRAIN_IPYNB = "notebooks/Fruit_Classification.ipynb"
EVAL_IPYNB = "notebooks/Fruit_Classification_Inference.ipynb"

st.title("Project Notebooks")
st.write(
    "Use the selector to switch view between the training and evaluation notebooks."
)


choice = st.radio(
    "Choose which notebook to display:",
    ("Training", "Evaluation"),
    horizontal=True,
)

if choice == "Training":
    notebook_path = TRAIN_IPYNB
    st.subheader("Training Notebook")
else:
    notebook_path = EVAL_IPYNB
    st.subheader("Evaluation Notebook")


if os.path.exists(notebook_path):
    mtime = os.path.getmtime(notebook_path)
    page_theme = st.context.theme.type or "light"
    html_data = render_notebook_to_html(notebook_path, mtime, page_theme)
    components.html(html_data, height=800, scrolling=True)
else:
    st.error(f"Could not find the notebook file at: {notebook_path}")
