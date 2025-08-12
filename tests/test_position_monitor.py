import unittest
from unittest.mock import MagicMock

from src.monitor.position_monitor import PositionMonitor


class TestPositionMonitor(unittest.TestCase):
    def test_face_down_detection(self):
        notifier = MagicMock()
        registry = MagicMock()
        registry.get_all.return_value = ["tok"]
        monitor = PositionMonitor(notifier, registry, model=MagicMock(), face_down_margin=0)

        face_down = MagicMock()
        kps = MagicMock()
        # nose_y > shoulders_y triggers notification
        kps.xy = [
            (0, 50),  # nose
            (0, 0),   # placeholders for remaining keypoints
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 20),  # left_shoulder
            (0, 20),  # right_shoulder
        ]
        face_down.keypoints = kps

        safe = MagicMock()
        kps_safe = MagicMock()
        kps_safe.xy = [
            (0, 10),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 20),
            (0, 20),
        ]
        safe.keypoints = kps_safe

        monitor.handle_pose([face_down])
        notifier.notify.assert_called_once()

        monitor.handle_pose([safe])
        monitor.handle_pose([face_down])
        self.assertEqual(notifier.notify.call_count, 2)


if __name__ == "__main__":
    unittest.main()
