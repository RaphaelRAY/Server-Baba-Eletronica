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
        camera_timeout: float = 0.0,
        camera_miss_threshold: int = 0,
    ):
        """Initialize dependencies and absence tracking state."""
        self.notifier = notifier
        self.registry = registry
        self.db = db
        self.absence_timeout = absence_timeout
        self.last_person_ts = time.time()
        self.absence_sent = False
        self.camera_sent = False
        self.camera_timeout = float(camera_timeout)
        self.camera_miss_threshold = int(camera_miss_threshold)
        self.last_frame_ts = time.time()  # atualizado ao receber frame válido
        self._last_frame = None  # armazena último frame útil para snapshot de eventos
        self._miss_count = 0  # contagem de leituras None consecutivas

    def check_camera(self, frame) -> None:
        """Notify if camera disconnected and persist events with debounce timeout."""
        now = time.time()
        if frame is None:
            # Incrementa contagem de falhas consecutivas
            self._miss_count += 1
            # Debounce por tempo: só sinaliza se passou do timeout (se configurado)
            if self.camera_timeout > 0 and (now - self.last_frame_ts) < self.camera_timeout:
                return
            # Debounce por contagem de misses: aguarda N leituras None consecutivas (se configurado)
            if self.camera_miss_threshold > 0 and self._miss_count < self.camera_miss_threshold:
                return
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
                self._notify_all("Camera reconectada", "A camera voltou a enviar frames", level="Info")
            self.camera_sent = False
            # Update last valid frame time and cache
            self.last_frame_ts = now
            self._last_frame = frame
            self._miss_count = 0

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
