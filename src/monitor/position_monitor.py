from typing import List

import cv2
from ultralytics import YOLO
import logging
logger = logging.getLogger(__name__)
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
        self.model = model or YOLO("yolo11n-pose.pt")
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
        #print(results[0].keypoints)  # Debug: show raw keypoints data

    def handle_pose(self, results: List) -> None:
        """Handle pose results and notify if face down."""
        for result in results or []:
            keypoints = getattr(result, "keypoints", None)
            if keypoints is None:
                continue

            # Extract and log keypoints safely (avoid truthiness on arrays/tensors)
            points = self._extract_xy_points(keypoints)
            if not points:
                # No usable keypoints structure found; skip
                continue
            if points:
                # Log a compact, rounded representation to avoid noisy output
                compact = [(round(float(x), 1), round(float(y), 1)) for x, y in points]
                logger.debug("Pose keypoints (%d): %s", len(compact), compact)

            # Use extracted points for face-down detection (indices 0,5,6: nose and shoulders)
            try:
                nose_y = points[0][1]
                left_shoulder_y = points[5][1]
                right_shoulder_y = points[6][1]
            except Exception:
                # Fallback to original access pattern if structure differs
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

    def _extract_xy_points(self, keypoints) -> list[tuple[float, float]]:
        """Convert various keypoints.xy shapes into a list of (x, y) floats.

        Handles common cases like [K,2] and [1,K,2]. Returns empty list on failure.
        """
        xy = getattr(keypoints, "xy", None)
        if xy is None:
            return []

        def to_float(v):
            try:
                return float(v)
            except Exception:
                try:
                    return float(v.item())
                except Exception:
                    return None

        # Try shape [K, 2]
        points: list[tuple[float, float]] = []
        try:
            for p in xy:
                if hasattr(p, "__len__") and len(p) >= 2 and not hasattr(p[0], "__len__"):
                    x, y = to_float(p[0]), to_float(p[1])
                    if x is not None and y is not None:
                        points.append((x, y))
            if points:
                return points
        except Exception:
            pass

        # Try shape [1, K, 2] (or [N, K, 2] -> first instance)
        try:
            inner = xy[0]
            for p in inner:
                if hasattr(p, "__len__") and len(p) >= 2:
                    x, y = to_float(p[0]), to_float(p[1])
                    if x is not None and y is not None:
                        points.append((x, y))
            return points
        except Exception:
            return []
