# Fruit & Vegetable Classifier

This is a deep learning web application built with Keras and Streamlit.

## Features

* **Static Image Prediction:** Allows users to upload a single image (e.g., `.jpg`, `.png`, `.webp`). The app runs inference using either of the two available models (MobileNetV2 or EfficientNetV2-B0) and displays the top prediction with its confidence score.
* **Real-Time Classification:** Accesses the user's webcam for a live video feed. It performs inference on each frame, overlaying the top-ranked prediction in real-time.
* **Model Interpretability(Grad-CAM):** Implements Gradient-weighted Class Activation Mapping to visualize the model's decision-making process. Users can upload an image and see a heatmap overlaid on it, highlighting the specific pixel regions the model used to arrive at its classification.
* **Project Notebooks:** Displays the original Colab notebooks used for model training and evaluation.

## Installation
1.  **Clone the Repository**
    ```
    git clone [Repo URL]
    cd [Repo Folder]
    ```

2.  **Create and Activate a Virtual Environment**
    ```
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**
    This project uses Git LFS to manage large model files. Ensure Git LFS is installed (`git lfs install`).
    ```
    pip install -r requirements.txt
    git lfs pull
    ```
## Usage
The application is run using Streamlit. The main entry point is `Home.py`.
1.  Ensure virtual environment is activated.
2.  Run the command:
    ```
    streamlit run Home.py
    ```
3.  Your default web browser will automatically open to the application's URL.
4.  Use the navigation sidebar to select a feature.

## Full Project Report

For a detailed breakdown of the methodology, model training, evaluation, and outcomes, please see the full report.

[**Full Project Report (PDF)**](Group_45_Report.pdf)
---
