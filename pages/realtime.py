import typing
from typing import Any, cast

import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorFactory, webrtc_streamer

from utils.config import MODEL_CONFIG
from utils.video_processing import FruitClassifierProcessor

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

# --- REALTIME SETTINGS ---
with st.expander("Real-time settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.session_state["realtime_poll_hz"] = fps = st.slider(
            "Polling rate (FPS)",
            min_value=1,
            max_value=30,
            value=int(st.session_state.get("realtime_poll_hz", 5)),
        )
    with col2:
        st.caption("Static ROI (relative)")
        st.session_state["roi_x"] = roi_x = st.slider(
            "ROI X", 0.0, 1.0, float(st.session_state.get("roi_x", 0.2)), 0.01
        )
        st.session_state["roi_y"] = roi_y = st.slider(
            "ROI Y", 0.0, 1.0, float(st.session_state.get("roi_y", 0.2)), 0.01
        )
        st.session_state["roi_w"] = roi_w = st.slider(
            "ROI W", 0.05, 1.0, float(st.session_state.get("roi_w", 0.6)), 0.01
        )
        st.session_state["roi_h"] = roi_h = st.slider(
            "ROI H", 0.05, 1.0, float(st.session_state.get("roi_h", 0.6)), 0.01
        )

# --- RUN THE WEBRTC STREAMER ---
ctx = webrtc_streamer(
    key="realtime_classifier",
    video_processor_factory=cast(VideoProcessorFactory, FruitClassifierProcessor),
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
    # async_processing=True,  # enable queued processing API
)

# Push settings into the processor (avoid using Streamlit from the processor thread)
if ctx and ctx.video_processor:
    vp = typing.cast(FruitClassifierProcessor, ctx.video_processor)
    vp.set_model(st.session_state["model_choice"])  # change model live if needed
    vp.set_fps(float(fps))
    vp.set_roi(
        float(roi_x),
        float(roi_y),
        float(roi_w),
        float(roi_h),
    )
    vp.set_pick_latest(True)

st.caption(
    f"Using {st.session_state['model_choice']} model. Change model and restart webcam."
)
