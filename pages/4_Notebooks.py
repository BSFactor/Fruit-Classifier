import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Project Notebooks", layout="wide")
st.title("Project Notebooks")
st.write("Use the selector to switch view between the training and evaluation notebooks.")

TRAIN_NOTEBOOK_PATH = "notebooks/html/Fruit_Classification.html"
EVAL_NOTEBOOK_PATH = "notebooks/html/Fruit_Classification_Inference.html"

choice = st.radio(
    "Choose which notebook to display:",
    ("Training", "Evaluation"),
    horizontal=True,
)

if choice == "Training":
    notebook_path = TRAIN_NOTEBOOK_PATH
    st.subheader("Training Notebook")
else:
    notebook_path = EVAL_NOTEBOOK_PATH
    st.subheader("Evaluation Notebook")

if os.path.exists(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        html_data = f.read()
    
    st.components.v1.html(html_data, height=800, scrolling=True)
else:
    st.error(f"Could not find the notebook HTML file at: {notebook_path}")