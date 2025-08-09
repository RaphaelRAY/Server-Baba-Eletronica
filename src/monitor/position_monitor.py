from typing import List

from ultralytics import YOLO

from src.notifications.identified_notifier import IdentifiedNotifier
from src.notifications.token_registry import TokenRegistry


class PositionMonitor:
    """Monitor pose estimation to detect dangerous positions."""

    def __init__(
        self,
        notifier: IdentifiedNotifier,
        registry: TokenRegistry,
        *,
        model: YOLO | None = None,
        face_down_margin: float = 20.0,
    ):
        self.notifier = notifier
        self.registry = registry
        self.model = model or YOLO("yolo11n.pt")
        self.face_down_margin = face_down_margin
        self.face_down_sent = False

    def analyze_frame(self, frame) -> None:
        """Run pose model on a frame."""
        results = self.model(frame)
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
                    self._notify_all("Bebê de bruços", "Rosto voltado para baixo")
                    self.face_down_sent = True
                return
        self.face_down_sent = False

    def _notify_all(self, title: str, message: str) -> None:
        for token in self.registry.get_all():
            self.notifier.notify(token, title=title, message=message)
