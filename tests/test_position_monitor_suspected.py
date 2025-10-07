import unittest
from unittest.mock import MagicMock

from src.monitor.position_monitor import PositionMonitor
from src.db import Database
from src.db.database import memory_events


class TestPositionMonitorSuspected(unittest.TestCase):
    def setUp(self):
        memory_events.clear()

    def test_suspected_face_down_when_face_not_visible(self):
        notifier = MagicMock()
        registry = MagicMock()
        registry.get_all.return_value = ["tok"]
        db = Database(server=Database.SERVER_MEMORY)

        # Configure low threshold to trigger quickly
        monitor = PositionMonitor(
            notifier,
            registry,
            db,
            model=MagicMock(),
            face_conf_min=0.8,
            no_face_frames_threshold=1,
            face_down_margin=20.0,
        )

        # Mock a pose result where face keypoints have low confidence (not visible)
        # but shoulders are present; nose position not strongly below shoulders
        res = MagicMock()
        kps = MagicMock()
        kps.xy = [
            (0, 25),  # nose (y close to shoulders, not strong face down)
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (-40, 20),  # left_shoulder
            (40, 20),  # right_shoulder
        ]
        kps.conf = [
            0.0,  # nose low conf
            0.0,
            0.0,
            0.0,
            0.0,
            0.9,
            0.9,
        ]
        res.keypoints = kps

        monitor.handle_pose([res])

        # Should notify suspected event
        notifier.notify.assert_called_once()
        evs = db.get_recent_events()
        self.assertTrue(any(e["type"] == "face_down_suspected" for e in evs))


if __name__ == "__main__":
    unittest.main()

