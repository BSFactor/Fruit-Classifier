import streamlit as st

st.set_page_config(
    page_title="Fruit Classifier - Home",
    layout="centered",
    menu_items={
        "About": """
# group 45

this is an awesome project made by our group.

we have:

- guy 1
- guy 2
- [Lương Đức Anh](//ldlda.com)
    """
    },
)

# pylint: disable=W0105

"""
# Fruit & Vegetable Classifier

### Welcome to the Fruit and Vegetable Classification project

**Select a demo from the sidebar on the left** to get cooking.

# Project Features

- **Run Prediction:** Upload an image and see predictions from both
MobileNetV2 and EfficientNetV2-B0.
- **Grad-CAM:** See a heatmap of *why* the model made its prediction.
- **Real Time Demo:** See live predictions from the models.
- **Notebooks:** View the original Colab notebooks used to train the models.
"""
