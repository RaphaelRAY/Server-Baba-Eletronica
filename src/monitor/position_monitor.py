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
        face_conf_min: float = 0.3,
        no_face_frames_threshold: int = 12,
    ):
        """Store dependencies and pose parameters."""
        self.notifier = notifier
        self.registry = registry
        self.db = db
        self.model = model or YOLO("yolo11n-pose.pt")
        self.face_down_margin = face_down_margin
        self.face_conf_min = float(face_conf_min)
        self.no_face_frames_threshold = int(no_face_frames_threshold)
        self.face_down_sent = False
        self.face_down_suspected_sent = False
        self._no_face_count = 0
        self._last_frame = None
        self._pose_window_initialized = False
        self._pose_window_name = "Pose"

    def analyze_frame(self, frame, show: bool = False) -> None:
        """Run pose model and optionally display keypoints."""
        # Cache frame for potential snapshot on event
        self._last_frame = frame
        results = self.model(frame)
        if show:
            if not self._pose_window_initialized:
                try:
                    cv2.namedWindow(
                        self._pose_window_name,
                        cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_GUI_EXPANDED", 0),
                    )
                except Exception:
                    cv2.namedWindow(self._pose_window_name, cv2.WINDOW_NORMAL)
                self._pose_window_initialized = True
            try:
                render = results[0].plot() if results else frame
            except Exception:
                render = frame
            if render is None:
                render = frame
            cv2.imshow(self._pose_window_name, render)
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
            # Confidence of face keypoints (if available)
            confs = self._extract_confidences(keypoints)
            def _conf(idx: int, default: float = 1.0) -> float:
                try:
                    c = confs[idx]
                    return float(c) if c is not None else default
                except Exception:
                    return default

            face_visible = (
                _conf(0, 0.0) >= self.face_conf_min  # nose
                or _conf(1, 0.0) >= self.face_conf_min  # left_eye
                or _conf(2, 0.0) >= self.face_conf_min  # right_eye
                or _conf(3, 0.0) >= self.face_conf_min  # left_ear
                or _conf(4, 0.0) >= self.face_conf_min  # right_ear
            )

            shoulders_y_max = max(left_shoulder_y, right_shoulder_y)
            strong_face_down = nose_y > shoulders_y_max + self.face_down_margin

            if strong_face_down:
                # reset no-face state
                self._no_face_count = 0
                self.face_down_suspected_sent = False
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

            # Weak/suspected face-down: face not visible for consecutive frames
            if not face_visible:
                self._no_face_count += 1
                if (
                    self._no_face_count >= self.no_face_frames_threshold
                    and not self.face_down_suspected_sent
                ):
                    # Raise a suspected event (lower confidence)
                    self._notify_all(
                        "Possível bebê de bruços",
                        "Rosto não visível e posição suspeita",
                        level="Importante",
                    )
                    if self.db is not None:
                        try:
                            payload = {
                                "type": "face_down_suspected",
                                "confidence": 0.5,
                                "level": "Importante",
                            }
                            if self._last_frame is not None:
                                try:
                                    payload["image_bytes"] = encode_jpeg(self._last_frame)
                                except Exception:
                                    pass
                            self.db.save_event(payload)
                        except Exception:
                            pass
                    self.face_down_suspected_sent = True
                # continue to next result if any
            else:
                # Face visible resets suspected counter
                self._no_face_count = 0
                self.face_down_suspected_sent = False
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

    def _extract_confidences(self, keypoints) -> list[float | None]:
        """Extract keypoint confidences as a flat list if available.

        Supports common shapes [K] or [1,K] or [N,K] -> first instance.
        Returns list with floats or None. Empty if not available.
        """
        conf = getattr(keypoints, "conf", None)
        if conf is None:
            conf = getattr(keypoints, "confidence", None)
        if conf is None:
            return []

        vals: list[float | None] = []
        try:
            # Try [K]
            for c in conf:
                if hasattr(c, "__len__"):
                    vals = []
                    raise Exception
                try:
                    vals.append(float(c))
                except Exception:
                    try:
                        vals.append(float(c.item()))
                    except Exception:
                        vals.append(None)
            if vals:
                return vals
        except Exception:
            pass

        try:
            # Try [1, K] (or [N,K] -> first)
            inner = conf[0]
            for c in inner:
                try:
                    vals.append(float(c))
                except Exception:
                    try:
                        vals.append(float(c.item()))
                    except Exception:
                        vals.append(None)
            return vals
        except Exception:
            return []
