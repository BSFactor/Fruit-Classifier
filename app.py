import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import json

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Fruit Classifier", layout="centered")
st.title("Fruit Vegetable Classifier")
st.write("Upload an image of a fruit or vegetable, and we shall find out what it is")

# --- 2. LOAD THE MODEL AND LABELS ---
@st.cache_resource
def load_my_model():
    model = keras.models.load_model('mobilenet_model.keras')
    return model

@st.cache_data
def load_my_labels():
    with open('labels.json', 'r') as f:
        # Load the dictionary and convert string keys back to integers
        labels_from_json = json.load(f)
        labels = {int(k): v for k, v in labels_from_json.items()}
    return labels

model = load_my_model()
labels = load_my_labels()

# --- 3. PREPROCESSING FUNCTION ---
def preprocess_image(image):
    # Resize the image to 224x224
    img = image.resize((224, 224))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Add the "batch" dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply EfficientNetV2 preprocessing
    img_preprocessed = keras.applications.efficientnet_v2.preprocess_input(img_array)
    
    return img_preprocessed

# --- 4. THE UPLOAD WIDGET ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='You uploaded this image:', use_container_width=True)
    st.write("")
    
    # 2. Preprocess the image
    processed_image = preprocess_image(image)
    
    # 3. Make a prediction
    with st.spinner('Thinking...'):
        predictions = model.predict(processed_image)
    
    # 4. Get the result
    pred_index = np.argmax(predictions, axis=1)[0]
    pred_label = labels[pred_index]
    confidence = np.max(predictions) * 100
    
    # 5. Show the result
    st.success(f"**Prediction:** {pred_label} ({confidence:.2f}%)")