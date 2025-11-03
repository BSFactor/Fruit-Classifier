import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import numpy as np
import cv2
import av
from PIL import Image

from utils import (
    MODEL_CONFIG, 
    load_my_model, 
    load_my_labels, 
    get_preprocess_fn,
)

st.set_page_config(page_title="Real-Time Demo", layout="centered")
st.title("Real-Time Classifier")
st.write("Select a model and click 'START' to activate your webcam.")

# --- MODEL SELECTION ---
if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = "MobileNetV2"

_options = list(MODEL_CONFIG.keys())
_default_index = _options.index(st.session_state["model_choice"])

st.selectbox(
    "Choose a model",
    options=_options,
    index=_default_index,
    key="model_choice",
)

# --- VIDEO PROCESSING CLASS ---

class FruitClassifierProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        # Load the model and labels
        self.model_name = st.session_state.get("model_choice", "MobileNetV2")
        self.model_config = MODEL_CONFIG[self.model_name]
        self.model = load_my_model(self.model_config["file"])
        self.labels = load_my_labels()
        self.preprocess_fn = get_preprocess_fn(self.model_config["family"])
        self.size = self.model_config["size"]

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the frame from the stream into a NumPy array
        img = frame.to_ndarray(format="bgr24")

        # --- PREPROCESS THE FRAME ---
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_resized = cv2.resize(img_rgb, self.size)
        
        # Batch dimension
        img_array = np.expand_dims(img_resized, axis=0)
        
        # Apply model-specific preprocessing
        img_preprocessed = self.preprocess_fn(img_array)

        # --- PREDICT ---
        predictions = self.model.predict(img_preprocessed)[0]
        pred_index = np.argmax(predictions)
        pred_label = self.labels[pred_index]
        confidence = np.max(predictions)

        # --- DRAW ON FRAME ---
        # Draw the prediction text on the *original* BGR frame
        text = f"{pred_label} ({confidence:.2f})"
        
        cv2.putText(
            img,  # Draw on the original BGR frame
            text,
            (10, 30),  # Position
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # Font scale
            (0, 255, 0),  # Color (Green)
            2,  # Thickness
        )

        # Convert the modified BGR frame back to the stream format
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- RUN THE WEBRTC STREAMER ---
webrtc_streamer(
    key="realtime_classifier",
    video_processor_factory=FruitClassifierProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
)

st.caption(f"Using {st.session_state['model_choice']} model. Change model and restart webcam.")