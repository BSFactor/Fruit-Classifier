# app.py (This is your NEW main file)
import streamlit as st

st.set_page_config(page_title="Fruit Classifier - Home", layout="centered")

st.title("Fruit & Vegetable Classifier")

st.subheader("""
Welcome to the Fruit and Vegetable Classification project.

**Select a demo from the sidebar on the left** to get cooking.
""")

st.header("Project Features")
st.markdown("""
* **Run Prediction:** Upload an image and see predictions from both MobileNetV2 and EfficientNetV2-B0.
* **Grad-CAM:** See a heatmap of *why* the model made its prediction.
* **Notebooks:** View the original Colab notebooks used to train the models.
""")