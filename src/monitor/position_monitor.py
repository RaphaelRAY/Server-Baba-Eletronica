from typing import List, Iterable
from collections import deque
from dataclasses import dataclass
import math

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

LEFT_KP_INDICES = {5, 7, 9, 11, 13, 15}
RIGHT_KP_INDICES = {6, 8, 10, 12, 14, 16}
CENTER_KP_INDICES = {0, 1, 2, 3, 4}
TORSO_KP_INDICES = {5, 6, 11, 12}
LIMB_SEGMENTS = [
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
    (5, 6),  # shoulders
    (11, 12),  # hips
    (5, 11), (6, 12),  # torso connections
]
LEFT_COLOR = (255, 128, 64)  # BGR - blue-ish/orange-ish
RIGHT_COLOR = (64, 192, 255)  # BGR - orange/teal
CENTER_COLOR = (120, 255, 120)
NEUTRAL_COLOR = (200, 200, 200)
DRAW_CONF_THRESHOLD = 0.1
LYING_VERTICAL_THRESHOLD = 40.0


def check_prone_pose(
    keypoints: dict[str, tuple[float, float]],
    *,
    lying_threshold: float = LYING_VERTICAL_THRESHOLD,
    min_horizontal_gap: float = 15.0,
) -> tuple[str, str]:
    """Avalia orientação (bruços ou frente) e postura (deitado ou em pé)."""
    rs = keypoints["right_shoulder"]
    ls = keypoints["left_shoulder"]
    rh = keypoints["right_hip"]
    lh = keypoints["left_hip"]

    shoulders_gap = abs(rs[0] - ls[0])
    hips_gap = abs(rh[0] - lh[0])
    if (
        rs[0] > ls[0]
        and rh[0] > lh[0]
        and shoulders_gap >= min_horizontal_gap
        and hips_gap >= min_horizontal_gap
    ):
        facing = "De bruços (orientação invertida)"

    else:
        facing = "De costas para câmera"


    vertical_diff = abs(rs[1] - rh[1])
    if vertical_diff < lying_threshold:
        posture = "Deitado"
    else:
        posture = "Em pé"

    return facing, posture


@dataclass
class PoseMetrics:
    """Aggregated pose metrics for decision making."""

    face_visible: bool
    strong_face_down: bool
    is_face_down: bool
    is_side: bool
    orientation_inverted: bool
    is_lying_down: bool
    facing_label: str | None
    posture_label: str | None
    risk_score: float
    angle_degrees: float | None
    nose_y_avg: float | None
    shoulders_y: float | None
    head_y: float | None


class PositionMonitor:
    """Monitor pose estimation to detect dangerous positions."""

    def __init__(
        self,
        notifier: IdentifiedNotifier,
        registry: TokenRegistry,
        db: Database | None = None,
        *,
        model: YOLO | None = None,
        model_path: str = "yolo11x-pose.pt",
        model_conf: float = 0.25,
        model_iou: float = 0.5,
        face_down_margin: float = 20.0,
        face_conf_min: float = 0.3,
        no_face_frames_threshold: int = 12,
        smoothing_window: int = 5,
        side_offset_ratio: float = 0.3,
        risk_threshold: float = 0.7,
        orientation_margin_ratio: float = 0.5,
    ):
        """Store dependencies and pose parameters."""
        self.notifier = notifier
        self.registry = registry
        self.db = db
        self._model = model
        self._model_path = model_path
        self._model_conf = float(model_conf)
        self._model_iou = float(model_iou)
        self.face_down_margin = face_down_margin
        self.face_conf_min = float(face_conf_min)
        self.no_face_frames_threshold = int(no_face_frames_threshold)
        self._risk_threshold = float(risk_threshold)
        self._side_offset_ratio = float(side_offset_ratio)
        self._orientation_margin_ratio = max(0.0, float(orientation_margin_ratio))
        self._has_face_down_alert = False
        self._has_face_down_suspected = False
        self._no_face_count = 0
        self._side_pose_count = 0
        self._lateral_history: deque[float] = deque(maxlen=5)
        self._last_frame = None
        self._pose_window_initialized = False
        self._pose_window_name = "Pose"
        self._nose_y_history: deque[float] = deque(maxlen=max(1, int(smoothing_window)))

    def analyze_frame(self, frame, show: bool = False) -> None:
        """Run pose model and optionally display keypoints."""
        # Cache frame for potential snapshot on event
        self._last_frame = frame
        processed = self._preprocess_frame(frame)
        results = self._get_model().predict(
            processed,
            conf=self._model_conf,
            iou=self._model_iou,
            verbose=False,
        )
        if show:
            self._render_pose(results, frame)
        self.handle_pose(results)
        #print(results[0].keypoints)  # Debug: show raw keypoints data

    def handle_pose(self, results: List) -> None:
        """Handle pose results and notify if face down."""
        alert_triggered = False
        handled_pose = False
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

            confs = self._extract_confidences(keypoints)
            metrics = self._compute_pose_metrics(points, confs, keypoints)
            handled_pose = True
            alert_triggered = self._process_pose_state(metrics)
            if alert_triggered:
                return
        if not alert_triggered and handled_pose:
            self.face_down_sent = False

    def _preprocess_frame(self, frame):
        """Enhance frame contrast to aid detection while preserving original frame."""
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

    def _notify_all(self, title: str, message: str, *, level: str = "info") -> None:
        """Send a notification to all tokens with level."""
        for token in self.registry.get_all():
            self.notifier.notify(token, title=title, message=message, level=level)

    def _compute_pose_metrics(
        self,
        points: list[tuple[float, float]],
        confs: list[float | None],
        keypoints,
    ) -> PoseMetrics:
        """Aggregate geometric and confidence-based cues for pose assessment."""
        nose = self._safe_point(points, 0)
        left_eye = self._safe_point(points, 1)
        right_eye = self._safe_point(points, 2)
        left_ear = self._safe_point(points, 3)
        right_ear = self._safe_point(points, 4)
        left_shoulder = self._safe_point(points, 5)
        right_shoulder = self._safe_point(points, 6)
        left_hip = self._safe_point(points, 11)
        right_hip = self._safe_point(points, 12)

        shoulders_y = self._mean(p[1] for p in (left_shoulder, right_shoulder) if p)
        shoulder_center = None
        shoulder_span = None
        orientation_inverted = False
        side_aligned = False
        is_lying_down = False
        facing_label = None
        posture_label = None
        if left_shoulder and right_shoulder:
            shoulder_center = (
                (left_shoulder[0] + right_shoulder[0]) / 2,
                shoulders_y,
            )
            shoulder_span = abs(left_shoulder[0] - right_shoulder[0])
            if shoulder_span < 25:
                side_aligned = True

        if left_shoulder and right_shoulder and left_hip and right_hip:
            facing_label, posture_label = check_prone_pose(
                {
                    "right_shoulder": right_shoulder,
                    "left_shoulder": left_shoulder,
                    "right_hip": right_hip,
                    "left_hip": left_hip,
                }
            )
            orientation_inverted = facing_label == "De bruços"
            is_lying_down = posture_label == "Deitado"

        head_components = [
            candidate[1]
            for candidate in (nose, left_eye, right_eye, left_ear, right_ear)
            if candidate
        ]
        head_y = self._mean(head_components)

        nose_y = nose[1] if nose else None
        nose_y_avg = self._update_nose_history(nose_y)
        reference_y = head_y if head_y is not None else nose_y_avg

        strong_face_down = False
        if shoulders_y is not None and reference_y is not None:
            strong_face_down = reference_y > shoulders_y + self.face_down_margin
        fallback_face_down = False
        if not strong_face_down:
            fallback_face_down = self._fallback_face_down(keypoints)
            strong_face_down = fallback_face_down

        is_side = side_aligned
        if nose and shoulder_center and shoulder_span is not None and shoulder_span > 0:
            side_offset = abs(nose[0] - shoulder_center[0])
            is_side = is_side or (side_offset > shoulder_span * self._side_offset_ratio)

        if nose and shoulder_center:
            lateral_offset = nose[0] - shoulder_center[0]
            self._lateral_history.append(lateral_offset)
            if (
                len(self._lateral_history) >= self._lateral_history.maxlen
                and not orientation_inverted
            ):
                sign_changes = sum(
                    1
                    for i in range(1, len(self._lateral_history))
                    if self._lateral_history[i] * self._lateral_history[i - 1] < 0
                )
                if sign_changes >= 2:
                    is_side = False

        angle = self._compute_head_angle(nose, shoulder_center)

        face_visible = self._is_face_visible(confs)
        orientation_risky = False
        margin_delta = self.face_down_margin * self._orientation_margin_ratio
        if orientation_inverted:
            if not face_visible:
                orientation_risky = True
            elif shoulders_y is not None and reference_y is not None:
                orientation_risky = reference_y > shoulders_y + margin_delta
        if orientation_risky and not strong_face_down and shoulders_y is not None and reference_y is not None:
            strong_face_down = reference_y > shoulders_y + margin_delta

        face_conf_vals = [c for c in confs[:5] if c is not None]
        face_conf_avg = sum(face_conf_vals) / len(face_conf_vals) if face_conf_vals else 0.0

        distance_component = 0.0
        if shoulders_y is not None and reference_y is not None:
            distance_component = max(0.0, reference_y - shoulders_y)
        distance_component = min(1.0, distance_component / 30.0)

        conf_component = 1.0 - max(0.0, min(face_conf_avg, 1.0))
        angle_component = min(1.0, max(0.0, (angle or 0.0) / 120.0))

        risk_score = (
            0.15 * distance_component
            + 0.45 * conf_component
            + 0.4 * (1.0 if orientation_risky else angle_component)
        )
        if is_side and not orientation_inverted:
            risk_score += 0.1
        risk_score = min(1.0, max(0.0, risk_score))

        is_face_down = ((strong_face_down and not is_side) or risk_score >= self._risk_threshold)

        return PoseMetrics(
            face_visible=face_visible,
            strong_face_down=strong_face_down and not is_side,
            is_face_down=is_face_down,
            is_side=is_side,
            orientation_inverted=orientation_inverted,
            is_lying_down=is_lying_down,
            facing_label=facing_label,
            posture_label=posture_label,
            risk_score=risk_score,
            angle_degrees=angle,
            nose_y_avg=nose_y_avg,
            shoulders_y=shoulders_y,
            head_y=head_y,
        )

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

    def _get_model(self) -> YOLO:
        """Load YOLO model on first use to avoid blocking constructor."""
        if self._model is None:
            self._model = YOLO(self._model_path)
        return self._model

    def _render_pose(self, results, frame) -> None:
        """Render pose output in a resizable window when requested."""
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
        drew_overlay = False
        for result in results or []:
            keypoints = getattr(result, "keypoints", None)
            if keypoints is None:
                continue
            points = self._extract_xy_points(keypoints)
            if not points:
                continue
            confs = self._extract_confidences(keypoints)
            drew_overlay = self._draw_pose_overlay(render, points, confs) or drew_overlay
        if not drew_overlay:
            try:
                render = results[0].plot() if results else render
            except Exception:
                logger.exception("Failed to render pose frame")
                render = render if render is not None else frame
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

    def _draw_pose_overlay(
        self,
        image,
        points: list[tuple[float, float]],
        confs: list[float | None],
    ) -> bool:
        """Draw colored skeleton overlay differentiating left/right limbs."""
        if image is None:
            return False
        drawn = False

        def _valid(idx: int) -> bool:
            if idx >= len(points):
                return False
            conf = confs[idx] if idx < len(confs) else None
            if conf is not None and conf < DRAW_CONF_THRESHOLD:
                return False
            return True

        def _pt(idx: int) -> tuple[int, int]:
            x, y = points[idx]
            return int(round(float(x))), int(round(float(y)))

        for idx in range(len(points)):
            if not _valid(idx):
                continue
            color = self._keypoint_color(idx)
            try:
                cv2.circle(
                    image,
                    _pt(idx),
                    4,
                    color,
                    thickness=-1,
                    lineType=getattr(cv2, "LINE_AA", 16),
                )
                drawn = True
            except Exception:
                logger.debug("Failed to draw keypoint %s", idx, exc_info=True)

        for start, end in LIMB_SEGMENTS:
            if not (_valid(start) and _valid(end)):
                continue
            color = self._segment_color(start, end)
            try:
                cv2.line(
                    image,
                    _pt(start),
                    _pt(end),
                    color,
                    thickness=2,
                    lineType=getattr(cv2, "LINE_AA", 16),
                )
                drawn = True
            except Exception:
                logger.debug("Failed to draw limb %s-%s", start, end, exc_info=True)
        return drawn

    def _keypoint_color(self, idx: int) -> tuple[int, int, int]:
        if idx in LEFT_KP_INDICES:
            return LEFT_COLOR
        if idx in RIGHT_KP_INDICES:
            return RIGHT_COLOR
        if idx in CENTER_KP_INDICES:
            return CENTER_COLOR
        return NEUTRAL_COLOR

    def _segment_color(self, start: int, end: int) -> tuple[int, int, int]:
        if start in LEFT_KP_INDICES and end in LEFT_KP_INDICES:
            return LEFT_COLOR
        if start in RIGHT_KP_INDICES and end in RIGHT_KP_INDICES:
            return RIGHT_COLOR
        if {start, end} <= TORSO_KP_INDICES or (
            start in CENTER_KP_INDICES or end in CENTER_KP_INDICES
        ):
            return CENTER_COLOR
        return NEUTRAL_COLOR

    def _is_face_visible(self, confs: list[float | None]) -> bool:
        """Check if any facial keypoint confidence exceeds threshold."""
        visible_idxs = (0, 1, 2, 3, 4)  # nose, eyes, ears
        for idx in visible_idxs:
            if idx >= len(confs):
                continue
            conf = confs[idx]
            if conf is not None and conf >= self.face_conf_min:
                return True
        return False

    def _fallback_face_down(self, keypoints) -> bool:
        """Fallback for legacy keypoint structures when points parsing fails."""
        try:
            nose_y = keypoints.xy[0][1]
            left_shoulder_y = keypoints.xy[5][1]
            right_shoulder_y = keypoints.xy[6][1]
        except Exception:
            return False
        shoulders_y_max = max(left_shoulder_y, right_shoulder_y)
        return nose_y > shoulders_y_max + self.face_down_margin

    def _process_pose_state(self, metrics: PoseMetrics) -> bool:
        """Update state machine and trigger notifications/events as needed."""
        if metrics.is_face_down:
            self._no_face_count = 0
            self._side_pose_count = 0
            if metrics.strong_face_down:
                self.face_down_suspected_sent = False
                if not self.face_down_sent:
                    message = "Rosto voltado para baixo"
                    if metrics.risk_score >= self._risk_threshold and not metrics.strong_face_down:
                        message = f"Indice de risco alto ({metrics.risk_score:.2f})"
                    self._notify_all("Bebe de brucos", message, level="Urgente")
                    self._record_event(
                        "face_down",
                        confidence=1.0,
                        level="Urgente",
                        extra={
                            "risk": metrics.risk_score,
                            "angle_deg": metrics.angle_degrees,
                            "head_y": metrics.head_y,
                            "shoulders_y": metrics.shoulders_y,
                        },
                    )
                    self.face_down_sent = True
            else:
                self.face_down_sent = False
                if not self.face_down_suspected_sent:
                    message = f"Indice de risco alto ({metrics.risk_score:.2f})"
                    self._notify_all(
                        "Posicao suspeita",
                        message,
                        level="Importante",
                    )
                    self._record_event(
                        "face_down_suspected",
                        confidence=max(0.0, metrics.risk_score),
                        level="Importante",
                        extra={
                            "angle_deg": metrics.angle_degrees,
                            "head_y": metrics.head_y,
                            "shoulders_y": metrics.shoulders_y,
                        },
                    )
                    self.face_down_suspected_sent = True
            logger.debug(
                "Frame analyzed: face_visible=%s, face_down=%s, risk=%.2f, no_face_count=%d",
                metrics.face_visible,
                metrics.is_face_down,
                metrics.risk_score,
                self._no_face_count,
            )
            return True

        side_threshold = max(1, self.no_face_frames_threshold // 2)
        if metrics.is_side and not metrics.is_face_down:
            self._side_pose_count += 1
            if (
                self._side_pose_count >= side_threshold
                and not self.face_down_suspected_sent
            ):
                self._notify_all(
                    "Posição suspeita",
                    "Bebê de lado — monitorando possível virada",
                    level="Importante",
                )
                self._record_event(
                    "face_down_possible",
                    confidence=0.3,
                    level="Importante",
                    extra={"risk": metrics.risk_score},
                )
                self.face_down_suspected_sent = True
                self._side_pose_count = 0
        else:
            self._side_pose_count = 0

        if not metrics.face_visible:
            self._no_face_count += 1
            if (
                self._no_face_count >= self.no_face_frames_threshold
                and not self.face_down_suspected_sent
            ):
                self._notify_all(
                    "Possível bebê de bruços",
                    "Rosto não visível e posição suspeita",
                    level="Importante",
                )
                self._record_event(
                    "face_down_suspected",
                    confidence=0.5,
                    level="Importante",
                    extra={"risk": metrics.risk_score},
                )
                self.face_down_suspected_sent = True
        else:
            self._no_face_count = 0
            if not metrics.is_side:
                self.face_down_suspected_sent = False

        logger.debug(
            "Frame analyzed: face_visible=%s, face_down=%s, risk=%.2f, no_face_count=%d",
            metrics.face_visible,
            metrics.is_face_down,
            metrics.risk_score,
            self._no_face_count,
        )
        return False

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
                logger.exception("Failed to encode frame to JPEG for event %s", event_type)
        try:
            self.db.save_event(payload)
        except Exception:
            logger.exception("Failed to record event: %s", event_type)

    @property
    def model(self) -> YOLO:
        """Return the loaded pose model, loading lazily if needed."""
        return self._get_model()

    @model.setter
    def model(self, value: YOLO | None) -> None:
        self._model = value

    @property
    def face_down_sent(self) -> bool:
        """Return whether a face-down alert has been dispatched."""
        return self._has_face_down_alert

    @face_down_sent.setter
    def face_down_sent(self, value: bool) -> None:
        self._has_face_down_alert = bool(value)

    @property
    def face_down_suspected_sent(self) -> bool:
        """Return whether a suspected face-down alert has been dispatched."""
        return self._has_face_down_suspected

    @face_down_suspected_sent.setter
    def face_down_suspected_sent(self, value: bool) -> None:
        self._has_face_down_suspected = bool(value)

    @staticmethod
    def _safe_point(points: list[tuple[float, float]], idx: int):
        try:
            point = points[idx]
            if len(point) >= 2:
                return float(point[0]), float(point[1])
        except Exception:
            return None
        return None

    @staticmethod
    def _mean(values: Iterable[float]) -> float | None:
        seq = [float(v) for v in values if v is not None]
        if not seq:
            return None
        return sum(seq) / len(seq)

    def _update_nose_history(self, nose_y: float | None) -> float | None:
        if nose_y is None:
            return self._mean(self._nose_y_history)
        self._nose_y_history.append(float(nose_y))
        return self._mean(self._nose_y_history)

    @staticmethod
    def _compute_head_angle(nose, shoulder_center) -> float | None:
        if nose is None or shoulder_center is None:
            return None
        p1 = nose
        p2 = shoulder_center
        p3 = (shoulder_center[0], shoulder_center[1] - 100.0)
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)
        if mag1 == 0 or mag2 == 0:
            return None
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_theta))
