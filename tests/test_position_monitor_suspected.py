import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.monitor.position_monitor import PositionMonitor
from src.db import Database
from src.db.database import memory_events
import numpy as np


class TestPositionMonitorSuspected(unittest.TestCase):
    def setUp(self):
        memory_events.clear()

    def test_side_event_saved_to_db(self):
        notifier = MagicMock()
        registry = MagicMock()
        registry.get_all.return_value = ["tok"]
        db = Database(server=Database.SERVER_MEMORY)
        model = MagicMock()
        model.names = {0: "absent", 1: "left"}

        monitor = PositionMonitor(
            notifier,
            registry,
            db,
            model=model,
            stable_frames=1,
            min_confidence=0.1,
        )
        model.return_value = [
            SimpleNamespace(probs=SimpleNamespace(top1=1, top1conf=0.9))
        ]

        monitor.analyze_frame(np.zeros((4, 4, 3), dtype=np.uint8))

        events = db.get_recent_events()
        self.assertTrue(any(ev["type"] == "posture_side" for ev in events))
        notifier.notify.assert_called_once()


if __name__ == "__main__":
    unittest.main()
