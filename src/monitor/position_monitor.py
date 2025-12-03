from __future__ import annotations

from dataclasses import dataclass
import logging
import pathlib
import time
from typing import List

import cv2
from ultralytics import YOLO

logger = logging.getLogger(__name__)
try:
    from ultralytics.utils import LOGGER

    LOGGER.setLevel(logging.ERROR)
except Exception:
    pass

try:
    import torch
except Exception:
    torch = None

from src.notifications.identified_notifier import IdentifiedNotifier
from src.notifications.token_registry import TokenRegistry
from src.db import Database
from src.utils.image_utils import encode_jpeg


@dataclass
class PostureResult:
    """Lightweight classification result wrapper."""

    label: str | None
    confidence: float
    raw_result: object | None = None


class PositionMonitor:
    """Monitor posture classification to detect risky positions."""

    def __init__(
        self,
        notifier: IdentifiedNotifier,
        registry: TokenRegistry,
        db: Database | None = None,
        *,
        model: YOLO | None = None,
        model_path: str = "clsModel.pt",
        model_conf: float = 0.25,
        model_iou: float = 0.5,
        face_down_margin: float = 20.0,  # legacy compatibility
        face_conf_min: float = 0.3,
        no_face_frames_threshold: int = 12,
        smoothing_window: int = 5,  # legacy compatibility
        side_offset_ratio: float = 0.3,  # legacy compatibility
        risk_threshold: float = 0.7,  # legacy compatibility
        orientation_margin_ratio: float = 0.5,  # legacy compatibility
        min_confidence: float = 0.35,
        stable_frames: int = 2,
        imgsz: int = 224,
        device: str | None = None,
        analysis_cooldown_secs: float = 5.0,
    ):
        """Configure monitor with notifier, registry and classification params."""
        self.notifier = notifier
        self.registry = registry
        self.db = db
        self._model = model
        self._model_path = model_path
        self._model_conf = float(model_conf)
        self._model_iou = float(model_iou)
        self._imgsz = int(imgsz)
        self._device = device or self._auto_device()
        self._min_confidence = float(min_confidence)
        self._stable_frames = max(1, int(stable_frames))
        self._analysis_cooldown = max(0.0, float(analysis_cooldown_secs))
        self.no_face_frames_threshold = int(no_face_frames_threshold)
        self._absent_threshold = max(1, self.no_face_frames_threshold)
        self._pose_window_initialized = False
        self._pose_window_name = "Pose"
        self._last_frame = None
        self._last_analysis_ts = 0.0

        self._current_label: str | None = None
        self._current_streak = 0
        self._absent_count = 0
        self._has_face_down_alert = False
        self._has_face_down_suspected = False

    def analyze_frame(self, frame, show: bool = False) -> None:
        """Classify a frame and trigger notifications according to posture."""
        self._last_frame = frame
        if frame is None:
            return

        now = time.monotonic()
        if (
            self._analysis_cooldown > 0
            and self._last_analysis_ts > 0
            and (now - self._last_analysis_ts) < self._analysis_cooldown
        ):
            return

        processed = self._preprocess_frame(frame)
        results = self._get_model()(
            processed,
            imgsz=self._imgsz,
            device=self._device,
            conf=self._model_conf,
            iou=self._model_iou,
            verbose=False,
        )
        self._last_analysis_ts = now

        raw_result = results[0] if results else None
        prediction = self._parse_prediction(raw_result)
        self._handle_prediction(prediction)

        if show:
            self._render_prediction(raw_result, frame, prediction)

    def _handle_prediction(self, prediction: PostureResult | None) -> None:
        """Apply simple state machine on top of classifier output."""
        if prediction is None or prediction.label is None:
            self._reset_streak()
            return

        label = prediction.label
        confidence = prediction.confidence
        if confidence < self._min_confidence:
            self._reset_streak()
            return

        if label != "absent":
            self._absent_count = 0

        if label == self._current_label:
            self._current_streak += 1
        else:
            self._current_label = label
            self._current_streak = 1

        if label == "absent":
            self._absent_count += 1
            if (
                self._absent_count >= self._absent_threshold
                and not self.face_down_suspected_sent
            ):
                self._emit_event(
                    title="Bebe ausente",
                    message="Sem deteccao do bebe",
                    level="Importante",
                    event_type="posture_absent",
                    confidence=confidence,
                    extra={"label": label},
                )
                self.face_down_suspected_sent = True
            self.face_down_sent = False
            return

        if label in ("supine", "sunpine"):
            self._clear_alerts()
            return

        if label == "prone":
            if self._current_streak >= self._stable_frames and not self.face_down_sent:
                self._emit_event(
                    title="Bebe de brucos",
                    message=f"Classificacao prone ({confidence:.2f})",
                    level="Urgente",
                    event_type="posture_prone",
                    confidence=confidence,
                    extra={"label": label},
                )
                self.face_down_sent = True
                self.face_down_suspected_sent = False
            return

        if label == "left":
            if (
                self._current_streak >= self._stable_frames
                and not self.face_down_suspected_sent
            ):
                self._emit_event(
                    title="Posicao suspeita",
                    message="Bebe de lado - monitorando possivel virada",
                    level="Importante",
                    event_type="posture_side",
                    confidence=confidence,
                    extra={"label": label},
                )
                self.face_down_suspected_sent = True
            self.face_down_sent = False
            return

        # Unknown but confident label; keep state reset.
        self._reset_streak()

    def _emit_event(
        self,
        *,
        title: str,
        message: str,
        level: str,
        event_type: str,
        confidence: float,
        extra: dict | None = None,
    ) -> None:
        """Send notification and persist the event if a database is present."""
        self._notify_all(title, message, level=level)
        self._record_event(
            event_type,
            confidence=max(0.0, min(1.0, confidence)),
            level=level,
            extra=extra,
        )

    def _parse_prediction(self, result) -> PostureResult | None:
        """Extract top-1 classification label and confidence."""
        if result is None:
            return None

        label = None
        confidence = 0.0
        probs = getattr(result, "probs", None)
        idx = None
        if probs is not None:
            idx_raw = getattr(probs, "top1", None)
            conf_raw = getattr(probs, "top1conf", None)
            if idx_raw is not None:
                try:
                    idx = int(idx_raw)
                except Exception:
                    pass
            if conf_raw is not None:
                try:
                    confidence = float(conf_raw)
                except Exception:
                    try:
                        confidence = float(conf_raw.item())
                    except Exception:
                        confidence = 0.0

        names = getattr(self._get_model(), "names", None)
        if idx is not None and names is not None:
            try:
                label = names[idx]
            except Exception:
                try:
                    label = names.get(idx)
                except Exception:
                    label = None

        normalized = self._normalize_label(label) if label is not None else None
        return PostureResult(label=normalized, confidence=confidence, raw_result=result)

    def _normalize_label(self, label: str) -> str:
        """Normalize label strings coming from the classifier."""
        normalized = label.strip().lower()
        if normalized == "sunpine":
            return "supine"
        return normalized

    def _preprocess_frame(self, frame):
        """Light histogram equalization to stabilize predictions."""
        if frame is None:
            return frame
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v_eq = cv2.equalizeHist(v)
            hsv_eq = cv2.merge((h, s, v_eq))
            return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
        except Exception:
            logger.debug("Frame preprocessing skipped", exc_info=True)
            return frame

    def _render_prediction(
        self,
        raw_result,
        frame,
        prediction: PostureResult | None,
    ) -> None:
        """Render classification overlay in a resizable window when requested."""
        if not self._pose_window_initialized:
            try:
                cv2.namedWindow(
                    self._pose_window_name,
                    cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_GUI_EXPANDED", 0),
                )
            except Exception:
                cv2.namedWindow(self._pose_window_name, cv2.WINDOW_NORMAL)
            self._pose_window_initialized = True

        render = frame.copy() if hasattr(frame, "copy") else frame
        if raw_result is not None:
            try:
                plotted = raw_result.plot()
                if plotted is not None:
                    render = plotted
            except Exception:
                logger.debug("Failed to render classifier output", exc_info=True)

        if prediction and prediction.label:
            try:
                text = f"{prediction.label} ({prediction.confidence:.2f})"
                cv2.putText(
                    render,
                    text,
                    (15, 30),
                    getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0),
                    1,
                    (0, 255, 255),
                    2,
                    getattr(cv2, "LINE_AA", 16),
                )
            except Exception:
                logger.debug("Failed to draw classification text", exc_info=True)

        if render is None:
            render = frame
        cv2.imshow(self._pose_window_name, render)
        cv2.waitKey(1)
        wnd_prop = getattr(cv2, "WND_PROP_VISIBLE", None)
        try:
            if wnd_prop is not None and hasattr(cv2, "getWindowProperty"):
                visible = cv2.getWindowProperty(self._pose_window_name, wnd_prop)
                if isinstance(visible, (int, float)) and visible < 1:
                    cv2.destroyAllWindows()
                    self._pose_window_initialized = False
        except Exception:
            logger.debug("Pose window property check failed", exc_info=True)

    def _notify_all(self, title: str, message: str, *, level: str = "info") -> None:
        """Send a notification to all tokens with level."""
        for token in self.registry.get_all():
            self.notifier.notify(token, title=title, message=message, level=level)

    def _record_event(
        self,
        event_type: str,
        *,
        confidence: float,
        level: str,
        extra: dict | None = None,
    ) -> None:
        """Persist event with optional frame snapshot."""
        if self.db is None:
            return
        payload = {"type": event_type, "confidence": confidence, "level": level}
        if extra:
            payload.update(extra)
        if self._last_frame is not None:
            try:
                payload["image_bytes"] = encode_jpeg(self._last_frame)
            except Exception:
                logger.exception(
                    "Failed to encode frame to JPEG for event %s", event_type
                )
        try:
            self.db.save_event(payload)
        except Exception:
            logger.exception("Failed to record event: %s", event_type)

    def _reset_streak(self) -> None:
        self._current_label = None
        self._current_streak = 0

    def _clear_alerts(self) -> None:
        self._reset_streak()
        self._absent_count = 0
        self.face_down_sent = False
        self.face_down_suspected_sent = False

    def _auto_device(self) -> str:
        """Pick GPU if available, otherwise CPU."""
        if torch is None:
            return "cpu"
        try:
            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _resolve_model_path(self) -> str:
        """Resolve model path relative to project root if needed."""
        path = pathlib.Path(self._model_path)
        if not path.is_file():
            candidate = pathlib.Path(__file__).resolve().parent.parent.parent / path
            if candidate.is_file():
                path = candidate
        return str(path)

    def _get_model(self) -> YOLO:
        """Load YOLO model on first use to avoid blocking constructor."""
        if self._model is None:
            self._model = YOLO(self._resolve_model_path())
        return self._model

    @property
    def model(self) -> YOLO:
        """Return the loaded classifier model."""
        return self._get_model()

    @model.setter
    def model(self, value: YOLO | None) -> None:
        self._model = value

    @property
    def face_down_sent(self) -> bool:
        """Return whether a prone alert has been dispatched."""
        return self._has_face_down_alert

    @face_down_sent.setter
    def face_down_sent(self, value: bool) -> None:
        self._has_face_down_alert = bool(value)

    @property
    def face_down_suspected_sent(self) -> bool:
        """Return whether a side/absent alert has been dispatched."""
        return self._has_face_down_suspected

    @face_down_suspected_sent.setter
    def face_down_suspected_sent(self, value: bool) -> None:
        self._has_face_down_suspected = bool(value)
