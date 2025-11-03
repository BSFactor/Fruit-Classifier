import logging
import threading
import time
from typing import Optional

import av
import cv2
import numpy as np
from keras import Model  # type: ignore[import]
from streamlit_webrtc import VideoProcessorBase

from utils import MODEL_CONFIG, get_preprocess_fn, load_my_labels, load_my_model

logger = logging.getLogger(__name__)


class FruitClassifierProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        # Initial defaults; the page will update these via setter methods.
        self.model_name = "MobileNetV2"
        self.model_config = MODEL_CONFIG[self.model_name]
        self.model: Model = load_my_model(self.model_config["file"])
        self.labels = load_my_labels()
        self.preprocess_fn = get_preprocess_fn(self.model_config["family"])
        self.size = self.model_config["size"]

        # Throttling/polling rate in seconds (default ~5 FPS). Can be overridden via session_state:
        # - st.session_state["realtime_poll_hz"] (e.g., 5 for 5 FPS)
        # - or st.session_state["realtime_poll_ms"] (e.g., 200 for 200 ms)
        self.poll_interval_s = 0.2

        # Maintain last prediction to reuse when skipping frames
        self._last_pred_label: Optional[str] = None
        self._last_pred_conf: Optional[float] = None
        self._last_processed_ts: float = 0.0

        # Static ROI ("hitbox") as relative coords (x, y, w, h) in [0,1].
        # Defaults to centered 60% region.
        self.roi_rel = (0.2, 0.2, 0.6, 0.6)

        # Whether to select the latest frame in the queue (lower latency) vs first (smoother)
        self.pick_latest = True

        # Background worker state
        self._lock = threading.Lock()
        self._funny_counter = 0
        # Keep the freshest raw frame as av.VideoFrame; convert only when needed
        self._latest_frame: Optional[av.VideoFrame] = None
        self._stop = False

        # Start background inference worker
        self._worker = threading.Thread(target=self._inference_loop, daemon=True)
        self._worker.start()

    # --- public setters called from the Streamlit page thread ---
    def set_model(self, name: str) -> None:
        if name not in MODEL_CONFIG:
            logger.warning("Unknown model %s; keeping %s", name, self.model_name)
            return
        if name != self.model_name:
            self.model_name = name
            self.model_config = MODEL_CONFIG[self.model_name]
            self.model = load_my_model(self.model_config["file"])
            self.preprocess_fn = get_preprocess_fn(self.model_config["family"])
            self.size = self.model_config["size"]

    def set_fps(self, fps: float) -> None:
        if fps and fps > 0:
            self.poll_interval_s = max(1.0 / float(fps), 0.001)
            # pacing change handled by worker; no redraw flag needed

    def set_roi(self, x: float, y: float, w: float, h: float) -> None:
        self.roi_rel = (
            float(np.clip(x, 0.0, 1.0)),
            float(np.clip(y, 0.0, 1.0)),
            float(np.clip(w, 0.0, 1.0)),
            float(np.clip(h, 0.0, 1.0)),
        )
    # ROI change takes effect on next frame overlay

    def set_pick_latest(self, val: bool) -> None:
        self.pick_latest = bool(val)
    # Affects which queued frame is chosen for latest update

    def _draw_overlay(self, img_bgr: np.ndarray) -> None:
        """Draws ROI rectangle and last prediction text on the frame in-place."""
        h, w = img_bgr.shape[:2]
        rx, ry, rw, rh = self.roi_rel
        x0 = int(rx * w)
        y0 = int(ry * h)
        x1 = int(min(x0 + rw * w, w - 1))
        y1 = int(min(y0 + rh * h, h - 1))

        # ROI rectangle
        cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (0, 255, 255), 2)

        # Prediction text
        if self._last_pred_label is not None and self._last_pred_conf is not None:
            text = f"{self._last_pred_label} ({self._last_pred_conf:.2f})"
            cv2.putText(
                img_bgr,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    def _update_latest_frame(self, frame: av.VideoFrame) -> None:
        """Store the most recent frame reference for the background worker to process.

        Avoids costly ndarray conversion on every recv; worker converts when needed.
        """
        # potentially expensive?
        with self._lock:
            self._latest_frame = frame
            self._funny_counter += 1

    def _run_inference_once(self, img_bgr: np.ndarray) -> None:
        """Run one inference on the given frame and cache results."""
        now = time.perf_counter()
        h, w = img_bgr.shape[:2]
        rx, ry, rw, rh = self.roi_rel
        x0 = int(rx * w)
        y0 = int(ry * h)
        x1 = int(min(x0 + rw * w, w))
        y1 = int(min(y0 + rh * h, h))
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(x0 + 1, min(x1, w))
        y1 = max(y0 + 1, min(y1, h))

        roi_bgr = img_bgr[y0:y1, x0:x1]
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi_resized = cv2.resize(roi_rgb, self.size)
        img_array = np.expand_dims(roi_resized, axis=0)
        img_preprocessed = self.preprocess_fn(img_array)

        # Suppress potential progress bar spam if supported; fallback otherwise
        try:
            predictions = self.model.predict(img_preprocessed, verbose=0)[0]  # type: ignore[call-arg]
        except TypeError:
            predictions = self.model.predict(img_preprocessed)[0]
        pred_index = int(np.argmax(predictions))
        pred_label = self.labels.get(pred_index, str(pred_index))
        confidence = float(np.max(predictions))

        self._last_pred_label = pred_label
        self._last_pred_conf = confidence
        self._last_processed_ts = time.time()
        print(
            f"damn: {((time.perf_counter() - now) * 1000):.4f} ms, funny counter = {self._funny_counter}"
        )

    def _inference_loop(self) -> None:
        """Background loop that processes the latest frame at the polling interval."""
        while not self._stop:
            now = time.time()
            if (now - self._last_processed_ts) < self.poll_interval_s:
                time.sleep(
                    max(
                        self.poll_interval_s - (now - self._last_processed_ts),
                        0.005,
                    )
                )
                continue
            with self._lock:
                latest = self._latest_frame
            if latest is None:
                time.sleep(0.005)
                continue
            try:
                img_bgr = latest.to_ndarray(format="bgr24")
                self._run_inference_once(img_bgr)
            except Exception as e:  # pragma: no cover
                logger.exception("Realtime inference failed: %s", e)
                time.sleep(0.05)

    def _overlay_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Draw overlay using cached result on a copy of the given frame."""
        img_bgr = frame.to_ndarray(format="bgr24")
        self._draw_overlay(img_bgr)
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    async def recv_queued(self, frames: list[av.VideoFrame]) -> list[av.VideoFrame]:  # type: ignore[override]
        """Process queued frames and return a list of frames.

        We only run the model on one selected frame per batch (first or latest)
        based on `self.pick_latest` and the polling interval, to reduce CPU.
        Other frames are returned unchanged for smooth streaming.
        """
        if not frames:
            return []

        # Always update worker with the freshest frame reference
        idx = len(frames) - 1 if self.pick_latest else 0
        self._update_latest_frame(frames[idx])

        # Draw overlay on every frame for stable visuals
        return [self._overlay_frame(f) for f in frames]

    def recv(
        self, frame: av.VideoFrame
    ) -> av.VideoFrame:  # Fallback if queued API not used
        # Update latest frame for the worker
        self._update_latest_frame(frame)  # do we need to do this every time
        # The UI overlay must be present every time
        return self._overlay_frame(frame)

    # Minimal transform for compatibility with VideoProcessorBase
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        # Keep consistent with recv: always overlay, and feed latest to worker
        self._update_latest_frame(frame)
        out = self._overlay_frame(frame)
        return out.to_ndarray(format="bgr24")

    def on_ended(self):
        """Stop background worker when stream ends."""
        self._stop = True
        try:
            if hasattr(self, "_worker") and self._worker.is_alive():
                self._worker.join(timeout=1.0)
        except Exception:  # pragma: no cover
            pass
