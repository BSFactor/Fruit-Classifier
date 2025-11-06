import os
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import streamlit.logger

from utils.notebook_rendering import (
    Strategy,
    apply_strategy,
    default_strategy,
    get_pdf_debug,
    render_notebook_to_html,
    render_notebook_to_pdf,
)

st.set_page_config(page_title="Project Notebooks", layout="wide")
st.title("Project Notebooks")
st.write(
    "Use the selector to switch view between the training and evaluation notebooks."
)

logger = streamlit.logger.get_logger(__name__)
# Source .ipynb notebooks
TRAIN_NOTEBOOK_PATH = "notebooks/Fruit_Classification.ipynb"
EVAL_NOTEBOOK_PATH = "notebooks/Fruit_Classification_Inference.ipynb"


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
        # Ensure store exists
        if "pdf_store" not in st.session_state:
            st.session_state.pdf_store = {}
        slot_state = st.session_state.pdf_store.get(key)

        def _render_now() -> None:
            # Read debug panel choices from session
            strat_bits = st.session_state.get("_pdf_strategy", int(default_strategy))
            try:
                strat_val: Strategy = Strategy(strat_bits)
            except (ValueError, TypeError):
                strat_val = default_strategy
            do_force: bool = st.session_state.get("_pdf_force", False)
            with apply_strategy(strat_val, do_force):  # more like
                if do_force:
                    render_notebook_to_pdf.clear()
                # include strategy bits in cache key
                result = render_notebook_to_pdf(notebook_path, mtime, int(strat_val))

            if result is None:
                st.session_state.pdf_store[key] = {
                    "error": "No PDF exporter available",
                    "debug": get_pdf_debug(),
                }
            else:
                pdf_bytes = result
                st.session_state.pdf_store[key] = {
                    "bytes": pdf_bytes,
                    "debug": get_pdf_debug(),
                }

        if slot_state and slot_state.get("bytes"):
            st.download_button(
                label="Download PDF",
                data=slot_state["bytes"],
                file_name=f"{base_no_ext}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            btn_label = (
                "Render PDF (unavailable)"
                if slot_state and slot_state.get("error")
                else "Render PDF"
            )
            st.button(btn_label, on_click=_render_now, use_container_width=True)

        # Debug/strategy panel
        with st.expander("PDF options"):
            _tex = st.checkbox("Enable TEX (LaTeX)", value=True, key="_tex")
            _web = st.checkbox("Enable WEBPDF (Chromium)", value=True, key="_web")
            _qt = st.checkbox("Enable QTPDF (Qt)", value=False, key="_qt")
            _force = st.checkbox(
                "Debug (cold render, log)", value=False, key="_pdf_force"
            )
            _strat_val = Strategy.NONE
            if _tex:
                _strat_val |= Strategy.TEX
            if _web:
                _strat_val |= Strategy.WEBPDF
            if _qt:
                _strat_val |= Strategy.QTPDF
            # store as int to avoid serialization issues
            st.session_state["_pdf_strategy"] = int(_strat_val)
        if _force:
            st.button(
                "Re-render PDF with current options",
                on_click=_render_now,
                use_container_width=True,
            )
            # Show attempts if available
            attempts = (slot_state or {}).get("debug") or []
            if attempts:
                with st.expander("Show PDF debug attempts"):
                    st.json(attempts)

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
