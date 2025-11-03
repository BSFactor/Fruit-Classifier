import os

import nbformat
import streamlit as st
import streamlit.components.v1 as components
from nbconvert import HTMLExporter

st.set_page_config(page_title="Project Notebooks", layout="wide")
st.title("Project Notebooks")
st.write(
    "Use the selector to switch view between the training and evaluation notebooks."
)

# Source .ipynb notebooks
TRAIN_IPYNB = "notebooks/Fruit_Classification.ipynb"
EVAL_IPYNB = "notebooks/Fruit_Classification_Inference.ipynb"

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


@st.cache_data(show_spinner=False)
def render_notebook_to_html(path: str, _mtime: float) -> str:
    """Render a .ipynb to HTML. Cached by file modification time."""
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    exporter = HTMLExporter()
    exporter.exclude_input = False
    exporter.exclude_output = False
    body, _resources = exporter.from_notebook_node(nb)
    return body


if os.path.exists(notebook_path):
    mtime = os.path.getmtime(notebook_path)
    html_data = render_notebook_to_html(notebook_path, mtime)
    components.html(html_data, height=800, scrolling=True)
else:
    st.error(f"Could not find the notebook file at: {notebook_path}")
