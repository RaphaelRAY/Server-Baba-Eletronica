from typing import List

import cv2
from ultralytics import YOLO
import logging
try:
    from ultralytics.utils import LOGGER
    LOGGER.setLevel(logging.ERROR)
except Exception:
    pass

from src.notifications.identified_notifier import IdentifiedNotifier
from src.notifications.token_registry import TokenRegistry
from src.db import Database
from src.utils.image_utils import encode_jpeg


class PositionMonitor:
    """Monitor pose estimation to detect dangerous positions."""

    def __init__(
        self,
        notifier: IdentifiedNotifier,
        registry: TokenRegistry,
        db: Database | None = None,
        *,
        model: YOLO | None = None,
        face_down_margin: float = 20.0,
    ):
        """Store dependencies and pose parameters."""
        self.notifier = notifier
        self.registry = registry
        self.db = db
        self.model = model or YOLO("yolo11n.pt")
        self.face_down_margin = face_down_margin
        self.face_down_sent = False
        self._last_frame = None

    def analyze_frame(self, frame, show: bool = False) -> None:
        """Run pose model and optionally display keypoints."""
        # Cache frame for potential snapshot on event
        self._last_frame = frame
        results = self.model(frame)
        if show and results:
            cv2.imshow("Pose", results[0].plot())
            cv2.waitKey(1)
        self.handle_pose(results)

    def handle_pose(self, results: List) -> None:
        """Handle pose results and notify if face down."""
        for result in results or []:
            keypoints = getattr(result, "keypoints", None)
            if keypoints is None or not getattr(keypoints, "xy", []):
                continue
            nose_y = keypoints.xy[0][1]
            left_shoulder_y = keypoints.xy[5][1]
            right_shoulder_y = keypoints.xy[6][1]
            if nose_y > max(left_shoulder_y, right_shoulder_y) + self.face_down_margin:
                if not self.face_down_sent:
                    self._notify_all("Bebê de bruços", "Rosto voltado para baixo", level="Urgente")
                    # Persist event when first detected
                    if self.db is not None:
                        try:
                            payload = {"type": "face_down", "confidence": 1.0, "level": "Urgente"}
                            if self._last_frame is not None:
                                try:
                                    payload["image_bytes"] = encode_jpeg(self._last_frame)
                                except Exception:
                                    pass
                            self.db.save_event(payload)
                        except Exception:
                            pass
                    self.face_down_sent = True
                return
        self.face_down_sent = False

    def _notify_all(self, title: str, message: str, *, level: str = "info") -> None:
        """Send a notification to all tokens with level."""
        for token in self.registry.get_all():
            self.notifier.notify(token, title=title, message=message, level=level)
