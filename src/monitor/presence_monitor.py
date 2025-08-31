import time
from typing import List

from src.notifications.identified_notifier import IdentifiedNotifier
from src.notifications.token_registry import TokenRegistry
from src.db import Database
from src.utils.image_utils import encode_jpeg


class PresenceMonitor:
    """Monitor camera frames and detections to notify events."""

    def __init__(
        self,
        notifier: IdentifiedNotifier,
        registry: TokenRegistry,
        db: Database,
        *,
        absence_timeout: int = 30,
    ):
        """Initialize dependencies and absence tracking state."""
        self.notifier = notifier
        self.registry = registry
        self.db = db
        self.absence_timeout = absence_timeout
        self.last_person_ts = time.time()
        self.absence_sent = False
        self.camera_sent = False
        self._last_frame = None  # armazena último frame útil para snapshot de eventos

    def check_camera(self, frame) -> None:
        """Notify if camera disconnected and persist events."""
        if frame is None:
            if not self.camera_sent:
                self._notify_all("Camera desconectada", "A camera parou de enviar frames", level="Importante")
                # Persist event on first disconnection edge
                try:
                    payload = {"type": "camera_disconnected", "confidence": 0.0, "level": "Importante"}
                    if self._last_frame is not None:
                        try:
                            payload["image_bytes"] = encode_jpeg(self._last_frame)
                        except Exception:
                            pass
                    self.db.save_event(payload)
                except Exception:
                    pass
                self.camera_sent = True
        else:
            # If we had signaled a disconnection previously, persist reconnection
            if self.camera_sent:
                try:
                    payload = {"type": "camera_connected", "confidence": 0.0, "level": "Info"}
                    try:
                        payload["image_bytes"] = encode_jpeg(frame)
                    except Exception:
                        pass
                    self.db.save_event(payload)
                except Exception:
                    pass
            self.camera_sent = False
            # Atualiza último frame útil
            self._last_frame = frame

    def handle_detections(self, results: List) -> None:
        """Track person absence and send notification."""
        now = time.time()
        detected = False
        for detection_result in results or []:
            if getattr(detection_result, "boxes", []):
                if len(detection_result.boxes) > 0:
                    detected = True
                    break
        if detected:
            self.last_person_ts = now
            self.absence_sent = False
        elif now - self.last_person_ts > self.absence_timeout and not self.absence_sent:
            self._notify_all("Ausência de humano", "Nenhuma pessoa detectada", level="Importante")
            payload = {"type": "absence", "confidence": 0.0, "level": "Importante"}
            if self._last_frame is not None:
                try:
                    payload["image_bytes"] = encode_jpeg(self._last_frame)
                except Exception:
                    pass
            self.db.save_event(payload)
            self.absence_sent = True

    def _notify_all(self, title: str, message: str, *, level: str = "info") -> None:
        """Send a notification to all registered tokens with level."""
        for t in self.registry.get_all():
            self.notifier.notify(t, title=title, message=message, level=level)
