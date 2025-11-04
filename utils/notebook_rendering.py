from pathlib import Path
from typing import Optional

import nbconvert
import nbformat
import streamlit as st
import streamlit.logger
from nbconvert import HTMLExporter

__all__ = [
    "render_notebook_to_html",
    "render_notebook_to_pdf",
]

logger = streamlit.logger.get_logger(__name__)


def _build_export_resources(path: str) -> dict:
    """Create a shared resources dict for nbconvert exporters.

    - Provide a friendly title (no date) based on the filename.
    - Avoid adding any date metadata so templates that show dates render empty.
    """
    friendly_name = Path(path).stem.replace("_", " ").replace("-", " ").strip()
    resources: dict = {
        "metadata": {
            "name": friendly_name,
            # Explicitly set an empty date to avoid template-provided dates
            "date": "",
        }
    }
    return resources


def _export_pdf_via_tex(nb: nbformat.NotebookNode, resources: dict) -> Optional[bytes]:
    """Try exporting to PDF via LaTeX PDFExporter."""
    try:
        exporter: nbconvert.PDFExporter = nbconvert.PDFExporter()
        exporter.exclude_input = False
        exporter.exclude_output = False
        body, _ = exporter.from_notebook_node(nb, resources=resources)
        return body
    except Exception as e:  # pylint: disable=W0718
        logger.exception("PDFExporter failed: %s", e, exc_info=False)
        return None


def _export_pdf_via_webpdf(
    nb: nbformat.NotebookNode, resources: dict, theme: str
) -> Optional[bytes]:
    """Try exporting to PDF via WebPDFExporter (nbconvert[webpdf])."""
    try:
        exporter2: nbconvert.WebPDFExporter = nbconvert.WebPDFExporter()
        exporter2.exclude_input = False
        exporter2.exclude_output = False
        if hasattr(exporter2, "theme"):
            exporter2.theme = theme
        body, _ = exporter2.from_notebook_node(nb, resources=resources)
        return body
    except Exception as e:  # pylint: disable=W0718
        logger.exception("WebPDFExporter failed: %s", e, exc_info=False)
        return None


def _export_pdf_via_qtpdf(
    nb: nbformat.NotebookNode, resources: dict, theme: str
) -> Optional[bytes]:
    """Try exporting to PDF via QtPDFExporter (requires PyQt)."""
    try:
        exporter3: nbconvert.QtPDFExporter = nbconvert.QtPDFExporter()
        exporter3.exclude_input = False
        exporter3.exclude_output = False
        if hasattr(exporter3, "theme"):
            exporter3.theme = theme
        body, _ = exporter3.from_notebook_node(nb, resources=resources)
        return body
    except Exception as e:  # pylint: disable=W0718
        logger.exception("QtPDFExporter failed: %s", e, exc_info=False)
        return None


@st.cache_data(show_spinner=False)
def render_notebook_to_html(path: str, _mtime: float, theme: str = "light") -> str:
    """Render a .ipynb to HTML. Cached by file modification time."""
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    resources = _build_export_resources(path)

    exporter: HTMLExporter = HTMLExporter()
    exporter.exclude_input = False
    exporter.exclude_output = False
    # Align notebook theme with the Streamlit page theme; include theme in cache key above
    if hasattr(exporter, "theme"):
        exporter.theme = "dark" if str(theme).lower() == "dark" else "light"
    body, _resources = exporter.from_notebook_node(nb, resources=resources)
    return body


@st.cache_data(show_spinner=False)
def render_notebook_to_pdf(
    path: str,
    _mtime: float,
    theme: str = "light",
) -> Optional[tuple[bytes, str | None]]:  # pragma: no cover
    """Stub for rendering a .ipynb to PDF.

    Note: Implementing PDF export portably can require additional nbconvert/TeX
    dependencies (e.g., Pyppeteer/WeasyPrint/LaTeX). This returns a placeholder
    empty PDF header for now to let the UI wire up a future exporter.
    """
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    resources = _build_export_resources(path)

    # 1) Try LaTeX PDFExporter
    body = _export_pdf_via_tex(nb, resources)
    if body is not None:
        return body, None

    # 2) Try WebPDFExporter (uses headless browser)
    body = _export_pdf_via_webpdf(nb, resources, theme)
    if body is not None:
        return body, theme

    # 3) Try QtPDFExporter
    body = _export_pdf_via_qtpdf(nb, resources, theme)
    if body is not None:
        return body, theme

    return None
