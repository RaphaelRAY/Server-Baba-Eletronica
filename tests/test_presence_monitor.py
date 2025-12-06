import unittest
from unittest.mock import MagicMock, patch

from src.monitor.presence_monitor import PresenceMonitor
from src.db import Database
from src.db.database import memory_events


class TestPresenceMonitor(unittest.TestCase):
    def setUp(self):
        memory_events.clear()

    @patch('src.monitor.presence_monitor.time')
    def test_absence_and_camera_notifications(self, mock_time):
        notifier = MagicMock()
        registry = MagicMock()
        registry.get_all.return_value = ['tok']
        mock_time.time.side_effect = [0, 31, 31, 32, 32, 33]
        db = Database(server=Database.SERVER_MEMORY)

        monitor = PresenceMonitor(notifier, registry, db, absence_timeout=30)
        monitor.handle_detections([])
        self.assertTrue(any(e['type'] == 'absence' for e in db.get_recent_events()))
        monitor.check_camera('frame')
        monitor.handle_detections([])
        notifier.send_immediate.assert_called_once()

        monitor.check_camera(None)
        self.assertEqual(notifier.send_immediate.call_count, 2)


if __name__ == '__main__':
    unittest.main()
