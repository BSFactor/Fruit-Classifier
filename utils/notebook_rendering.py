import contextlib
import enum
import sys
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, ParamSpec, TypeVar, cast

import nbconvert
import nbformat
import streamlit as st
import streamlit.logger
from nbconvert import HTMLExporter

__all__ = [
    "render_notebook_to_html",
    "render_notebook_to_pdf",
    "Strategy",
    "force",
    "get_pdf_debug",
]

logger = streamlit.logger.get_logger(__name__)

_last_pdf_attempts: list[dict] = []  # process-global fallback


def _session_debug_list() -> Optional[list[dict]]:
    """Return a session-scoped list for debug attempts, or None if unavailable.

    Falls back to module-global when Streamlit session state is not available.
    """
    try:
        key = "_pdf_debug_attempts"
        if key not in st.session_state:
            st.session_state[key] = []
        return st.session_state[key]
    except Exception:  # pylint: disable=W0718 # pragma: no cover
        # guard if used outside Streamlit runtime
        return None


def _debug_attempt(strategy: str, status: str, note: str | None = None) -> None:
    entry = {"strategy": strategy, "status": status, "note": note}
    store = _session_debug_list()
    if store is not None:
        store.append(entry)
    else:
        _last_pdf_attempts.append(entry)


def get_pdf_debug() -> list[dict]:
    """Return a copy of the last PDF export attempts for client-side debug UI."""
    store = _session_debug_list()
    return deepcopy(store if store is not None else _last_pdf_attempts)


class Strategy(enum.IntFlag):
    NONE = 0
    TEX = enum.auto()
    WEBPDF = enum.auto()
    QTPDF = enum.auto()


_strategy_enabled: Strategy | None = None
default_strategy: Strategy = Strategy.TEX | Strategy.WEBPDF


# giving up is real
def strategy_influenced[**P, R](
    s: Strategy,
    name: str,
    start_log: Optional[Callable[[], None]] = None,
    end_log: Optional[Callable[[Any], None]] = None,  # this but Any will work.
) -> Callable[[Callable[P, R]], Callable[P, R | None]]:
    def trying():
        _debug_attempt(name, "try")

    start = start_log if start_log is not None else trying

    def log_result(r: R):
        _debug_attempt(name, "ok" if r is not None else "fail")

    end = cast(
        Callable[[R], None],
        end_log if end_log is not None else log_result,
    )

    def dec(f: Callable[P, R]) -> Callable[P, R | None]:
        @wraps(f)
        def fwraps(*a, **k) -> R | None:
            # Use default strategy when not forced, but only log debug when forced
            forced = _strategy_enabled is not None
            strat: Strategy = (
                _strategy_enabled if _strategy_enabled is not None else default_strategy
            )
            if not (strat & s):
                return None
            if forced:
                start()
            r = f(*a, **k)
            if forced:
                end(r)
            return r

        return fwraps

    return dec


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


@strategy_influenced(Strategy.TEX, "TEX")
def _export_pdf_via_tex(nb: nbformat.NotebookNode, resources: dict) -> Optional[bytes]:
    """Try exporting to PDF via LaTeX PDFExporter."""
    try:
        exporter: nbconvert.PDFExporter = nbconvert.PDFExporter()
        exporter.exclude_input = False
        exporter.exclude_output = False
        nbconvert.preprocessors.latex.LatexPreprocessor().preprocess(nb, resources)
        body, _ = exporter.from_notebook_node(nb, resources=resources)
        return body
    except Exception as e:  # pylint: disable=W0718
        logger.exception("PDFExporter failed: %s", e, exc_info=False)
        return None


@strategy_influenced(Strategy.WEBPDF, "WEBPDF")
def _export_pdf_via_webpdf(
    nb: nbformat.NotebookNode, resources: dict
) -> Optional[bytes]:
    """Try exporting to PDF via WebPDFExporter (nbconvert[webpdf])."""
    try:
        exporter2: nbconvert.WebPDFExporter = nbconvert.WebPDFExporter()
        exporter2.exclude_input = False
        exporter2.exclude_output = False

        # Theme isn't supported reliably for WebPDF; rely on default styling.
        body, _ = exporter2.from_notebook_node(nb, resources=resources)
        return body
    except Exception as e:  # pylint: disable=W0718
        logger.exception("WebPDFExporter failed: %s", e, exc_info=False)
        return None


@strategy_influenced(Strategy.QTPDF, "QTPDF")
def _export_pdf_via_qtpdf(
    nb: nbformat.NotebookNode, resources: dict
) -> Optional[bytes]:
    """Try exporting to PDF via QtPDFExporter (requires PyQt5)."""
    try:
        exporter3: nbconvert.QtPDFExporter = nbconvert.QtPDFExporter()
        exporter3.exclude_input = False
        exporter3.exclude_output = False
        # Theme isn't supported reliably for QtPDF; many setups render blank.
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
) -> Optional[bytes]:  # pragma: no cover
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
        return body

    # 2) Try WebPDFExporter (uses headless browser)
    body = _export_pdf_via_webpdf(nb, resources)
    if body is not None:
        return body

    # 3) Try QtPDFExporter
    body = _export_pdf_via_qtpdf(nb, resources)
    if body is not None:
        return body

    return None


@contextlib.contextmanager
def force(strat: Strategy):
    mod = sys.modules[__name__]
    prev = getattr(mod, "_strategy_enabled", None)
    try:
        setattr(mod, "_strategy_enabled", strat)
        # Reset debug attempts for this session when entering a forced context
        store = _session_debug_list()
        if store is not None:
            store.clear()
        else:
            _last_pdf_attempts.clear()
        yield
    finally:
        setattr(mod, "_strategy_enabled", prev)
