import contextlib
import enum
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional

import nbconvert
import nbformat
import streamlit as st
import streamlit.logger
from nbconvert import HTMLExporter

__all__ = [
    "render_notebook_to_html",
    "render_notebook_to_pdf",
    "Strategy",
    "default_strategy",
    "apply_strategy",
    "get_pdf_debug",
    "clear_pdf_debug",
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


def clear_pdf_debug() -> None:
    """Clear stored PDF debug attempts (session-scoped if available)."""
    store = _session_debug_list()
    if store is not None:
        store.clear()
    _last_pdf_attempts.clear()


class Strategy(enum.IntFlag):
    NONE = 0
    TEX = enum.auto()
    WEBPDF = enum.auto()
    QTPDF = enum.auto()


_strategy: Strategy | None = None
default_strategy: Strategy = Strategy.TEX | Strategy.WEBPDF
_logging_enabled: bool = False


# giving up is real
def strategy_influenced[**P, R](
    s: Strategy, name: str
) -> "Callable[[Callable[P, R]], StrategyInfluenced[P, R]]":
    return lambda f: StrategyInfluenced(s, name, f)


def _forced():
    return _logging_enabled


def _effective_strategy() -> Strategy:
    return _strategy if _strategy is not None else default_strategy


class MetaStrategyInfluenced(type):
    @property
    def logging(cls) -> bool:
        "class"
        return _forced()

    @property
    def effective_strategy(cls) -> Strategy:
        "class"
        return _effective_strategy()


# how they do this typa decorator class thing so good
class StrategyInfluenced[**P, R](metaclass=MetaStrategyInfluenced):
    def __init__(self, s: Strategy, name: str, f: Callable[P, R]) -> None:
        self._strat = s
        self.name = name

        def trying():
            _debug_attempt(self.name, "try")

        def log_result(r: R):
            _debug_attempt(self.name, "ok" if r is not None else "fail")

        def log_fail(e: Exception):
            logger.debug("%s failed: %s", self.name, e, stack_info=True)
            return _debug_attempt(self.name, "fail")

        self._start: Callable[[], None] = trying
        self._success: Callable[[R], None] = log_result
        self._fail: Callable[[Exception], None] = log_fail

        self._f = f

        # Snapshots to allow restoring defaults
        self._default_start = self._start
        self._default_success = self._success
        self._default_fail = self._fail

    @property
    def debug_start(self) -> Callable[[], None]:
        return self._start

    @debug_start.setter
    def debug_start(self, fn: Callable[[], None]) -> None:
        if fn is not None:
            self._start = fn

    @property
    def debug_end(self) -> Callable[[R], None]:
        return self._success

    @debug_end.setter
    def debug_end(self, fn: Callable[[R], None]) -> None:
        if fn is not None:
            self._success = fn

    @property
    def debug_fail(self) -> Callable[[Exception], None]:
        return self._fail

    @debug_fail.setter
    def debug_fail(self, fn: Callable[[Exception], None]) -> None:
        if fn is not None:
            self._fail = fn

    # how cant i just use debug_end
    def bind_end(self, r: Callable[[R], None]):
        self.debug_end = r
        return r

    def bind_start(self, r: Callable[[], None]):
        self.debug_start = r
        return r

    def bind_fail(self, r: Callable[[Exception], None]):
        self.debug_fail = r
        return r

    # Clear/unbind helpers
    def clear_start(self) -> None:
        self._start = self._default_start

    def clear_end(self) -> None:
        self._success = self._default_success

    def clear_fail(self) -> None:
        self._fail = self._default_fail

    @property
    def function(self) -> Callable[P, R]:
        return self._f

    @property
    def strategy(self) -> Strategy:
        return self._strat

    def __call__(self, *a: P.args, **k: P.kwargs) -> R | None:
        # Use default strategy when not forced, but only log debug when forced
        if not (StrategyInfluenced.effective_strategy & self.strategy):
            return None
        logging = StrategyInfluenced.logging
        if logging:
            self.debug_start()
        try:
            r = self.function(*a, **k)
        except Exception as e:  # pylint: disable=broad-except
            if logging:
                self.debug_fail(e)
            return None
        if logging:
            self.debug_end(r)
        return r

    # @property
    # def effective_strategy(self) -> Strategy:
    #     "instance"
    #     return _effective_strategy()

    # @property
    # def forced(self) -> bool:
    #     "instance"
    #     return _forced()

    # fuck mypy fuck pylint you made me do this instead of type(self)._thing
    # shadowing is crazy. or the linters stupid.


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
    from traitlets.config import Config

    c = Config()
    c.LatexPreprocessor.date = ""
    exporter: nbconvert.PDFExporter = nbconvert.PDFExporter(config=c)
    exporter.exclude_input = False
    exporter.exclude_output = False
    body, _ = exporter.from_notebook_node(nb, resources=resources)
    return body


@_export_pdf_via_tex.bind_fail
def _tex_fail(e: Exception):
    if isinstance(e, nbconvert.exporters.pdf.LatexFailed):
        note = None
        missing_re = re.compile(r"! LaTeX Error: File `([^']+)' not found\.")
        for line in str(e.output).splitlines():
            m = missing_re.search(line)
            if m:
                missing = m.group(1)
                note = f"install {missing}"
                break
    else:
        note = None
    logger.error("%s failed: %s", _export_pdf_via_tex.name, e)
    _debug_attempt(_export_pdf_via_tex.name, "fail", note)


@strategy_influenced(Strategy.WEBPDF, "WEBPDF")
def _export_pdf_via_webpdf(
    nb: nbformat.NotebookNode, resources: dict
) -> Optional[bytes]:
    """Try exporting to PDF via WebPDFExporter (nbconvert[webpdf])."""
    exporter2: nbconvert.WebPDFExporter = nbconvert.WebPDFExporter()
    exporter2.exclude_input = False
    exporter2.exclude_output = False
    # Allow auto-download of Chromium if playwright hasn't installed it yet.
    if hasattr(exporter2, "allow_chromium_download"):
        exporter2.allow_chromium_download = True  # type: ignore[attr-defined]

    # exporter2.allow_chromium_download = True

    # Theme isn't supported reliably for WebPDF; rely on default styling.
    body, _ = exporter2.from_notebook_node(nb, resources=resources)
    return body


@_export_pdf_via_webpdf.bind_fail
def _webpdf_fail(e: Exception):  # pragma: no cover
    # Typical failure reasons: chromium executable missing, playwright not installed.
    txt = str(e)
    note: str | None = None
    if "No suitable chromium executable" in txt:
        note = "playwright install chromium"
    elif "Playwright is not installed" in txt:
        note = "pip install nbconvert[webpdf]"
    logger.error("%s failed: %s", _export_pdf_via_webpdf.name, e)
    _debug_attempt(_export_pdf_via_webpdf.name, "fail", note)


@strategy_influenced(Strategy.QTPDF, "QTPDF")
def _export_pdf_via_qtpdf(
    nb: nbformat.NotebookNode, resources: dict
) -> Optional[bytes]:
    """Try exporting to PDF via QtPDFExporter (requires PyQt5)."""
    exporter3: nbconvert.QtPDFExporter = nbconvert.QtPDFExporter()
    exporter3.exclude_input = False
    exporter3.exclude_output = False
    # Theme isn't supported reliably for QtPDF; many setups render blank.
    body, _ = exporter3.from_notebook_node(nb, resources=resources)
    return body


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
    path: str, mtime: float, strategy_cache_key: int | Strategy = default_strategy
) -> Optional[bytes]:  # pragma: no cover
    """Render a .ipynb to PDF trying multiple exporters.

    The ``strategy_cache_key`` is unused at runtime but participates in the
    cache key so different strategy selections don't reuse old results.
    """
    # NOP reference so linters recognize the cache key parameter is intentional
    _ = strategy_cache_key, mtime
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
    # how do i know that ts dont ever run
    if body is not None:
        return body

    return None


@contextlib.contextmanager
def apply_strategy(strat: Strategy = default_strategy, logging: bool = False):
    mod = sys.modules[__name__]
    prev_strat = getattr(mod, "_strategy", None)
    prev_logging = getattr(mod, "_logging_enabled", False)

    try:
        setattr(mod, "_strategy", strat)
        if logging:
            setattr(mod, "_logging_enabled", True)
            # Reset debug attempts for this session when entering a logging context
            clear_pdf_debug()
        yield
    finally:
        setattr(mod, "_strategy", prev_strat)
        if logging:
            setattr(mod, "_logging_enabled", prev_logging)
