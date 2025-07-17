import time
from typing import List

from src.notifications.identified_notifier import IdentifiedNotifier
from src.notifications.token_registry import TokenRegistry


class PresenceMonitor:
    """Monitor camera frames and detections to notify events."""

    def __init__(self, notifier: IdentifiedNotifier, registry: TokenRegistry, *, absence_timeout: int = 30):
        self.notifier = notifier
        self.registry = registry
        self.absence_timeout = absence_timeout
        self.last_person_ts = time.time()
        self.absence_sent = False
        self.camera_sent = False

    def check_camera(self, frame) -> None:
        """Notify if camera disconnected."""
        if frame is None:
            if not self.camera_sent:
                self._notify_all("Camera desconectada", "A camera parou de enviar frames")
                self.camera_sent = True
        else:
            self.camera_sent = False

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
            self._notify_all("Ausência de humano", "Nenhuma pessoa detectada")
            self.absence_sent = True

    def _notify_all(self, title: str, message: str) -> None:
        for t in self.registry.get_all():
            self.notifier.notify(t, title=title, message=message)
