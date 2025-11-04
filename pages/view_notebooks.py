import os
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from utils.notebook_rendering import render_notebook_to_html, render_notebook_to_pdf

st.set_page_config(page_title="Project Notebooks", layout="wide")

# Source .ipynb notebooks
TRAIN_IPYNB = "notebooks/Fruit_Classification.ipynb"
EVAL_IPYNB = "notebooks/Fruit_Classification_Inference.ipynb"

st.title("Project Notebooks")
st.write(
    "Use the selector to switch view between the training and evaluation notebooks."
)


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

with st.sidebar:
    st.header("Download")


if os.path.exists(notebook_path):
    # Perf: measure render time
    t0 = time.perf_counter()
    mtime = os.path.getmtime(notebook_path)
    # Allow overriding the page theme for the notebook rendering
    theme_choice = st.radio(
        "Notebook theme",
        ("Match app", "Light", "Dark"),
        horizontal=True,
        help="Choose how the notebook is themed. 'Match app' follows Streamlit's theme; others override it.",
    )
    page_theme = st.context.theme.type or "light"
    nb_theme = (
        page_theme
        if theme_choice == "Match app"
        else ("light" if theme_choice == "Light" else "dark")
    )
    html_data = render_notebook_to_html(
        notebook_path, mtime, nb_theme
    )  # respects theme
    t_ms = (time.perf_counter() - t0) * 1000.0
    components.html(html_data, height=800, scrolling=True)
    st.caption(f"Rendered in {t_ms:.1f} ms")

    # Export/download options
    ipynb_bytes = Path(notebook_path).read_bytes()
    base_no_ext = os.path.splitext(os.path.basename(notebook_path))[0]

    @st.fragment
    def pdf_slot(key: str):
        stored = st.session_state.pdf_store.get(key)

        def _render_now():
            result = render_notebook_to_pdf(notebook_path, mtime, nb_theme)
            if result is None:
                st.session_state.pdf_store[key] = {"error": "No PDF exporter available"}
            else:
                pdf_bytes, final_theme = result
                st.session_state.pdf_store[key] = {
                    "bytes": pdf_bytes,
                    "final_theme": final_theme,
                }

        if stored and stored.get("bytes"):
            st.download_button(
                label="Download PDF",
                data=stored["bytes"],
                file_name=f"{base_no_ext}{'.' + stored.get('final_theme') if stored.get('final_theme') else ''}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            btn_label = (
                "Render PDF (unavailable)"
                if stored and stored.get("error")
                else "Render PDF"
            )
            st.button(btn_label, on_click=_render_now, use_container_width=True)

    with st.sidebar:
        st.download_button(
            label="Download .ipynb",
            data=ipynb_bytes,
            file_name=os.path.basename(notebook_path),
            mime="application/x-ipynb+json",
            use_container_width=True,
        )

        st.download_button(
            label="Download rendered HTML",
            data=html_data,
            file_name=f"{base_no_ext}.{nb_theme}.html",
            mime="text/html",
            use_container_width=True,
        )
        # Fragment slot for PDF render: button -> compute -> download
        if "pdf_store" not in st.session_state:
            st.session_state.pdf_store = {}

        pdf_key = f"{notebook_path}|{nb_theme}"

        pdf_slot(pdf_key)
else:
    st.error(f"Could not find the notebook file at: {notebook_path}")
    with st.sidebar:
        st.button("Unavailable", disabled=True)
