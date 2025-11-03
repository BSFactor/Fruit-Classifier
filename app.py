import streamlit as st

st.navigation(
    [
        st.Page("pages/home.py", title="Home", icon="🏠"),
        st.Page("pages/prediction.py", title="Run Prediction", icon="🖼️"),
        st.Page("pages/grad_cam.py", title="Grad-CAM", icon="🔥"),
        st.Page("pages/realtime.py", title="Real-Time Demo", icon="🎥"),
        st.Page("pages/view_notebooks.py", title="Notebooks", icon="📓"),
    ],
    position="sidebar",
).run()
