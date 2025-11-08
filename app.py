# pyright: reportUnusedExpression=false
# pylint: disable=pointless-statement
import streamlit as st
from streamlit.navigation.page import StreamlitPage
from streamlit.runtime.scriptrunner import get_script_run_ctx

pages = [
    st.Page("pages/home.py", title="Home", icon="🏠"),
    st.Page("pages/prediction.py", title="Run Prediction", icon="🖼️"),
    st.Page("pages/grad_cam.py", title="Grad-CAM", icon="🔥"),
    st.Page("pages/realtime.py", title="Real-Time Demo", icon="🎥"),
    st.Page("pages/view_notebooks.py", title="Notebooks", icon="📓"),
    st.Page("pages/help.py", title="Help", icon="🤓"),
    st.Page("pages/view_model.py", title="View models", icon="📊"),
]
pg = st.navigation(pages, position="top")
st.set_page_config(page_title="Fruit Classifier", page_icon=":apple:")


class malicious:
    def __init__(self, page: StreamlitPage):
        self.page = page

    @staticmethod
    def in_st():
        return get_script_run_ctx(True) is not None

    def run(self):
        "run the app and then quit"
        if self.in_st():
            self.page.run()
            st.stop()

    def __repr__(self):
        self.run()
        return super().__repr__()


# this runs the app magic style.
# malicious isnt anything st.write recognizes so it falls back to str(App), which runs the app.
# Cleaner when using st.help... idk why ts so ass
App = malicious(pg)
App

# or you can
# exec("pg.run()")
# lmao
