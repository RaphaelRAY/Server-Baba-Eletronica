import unittest
from unittest.mock import MagicMock, patch

from src.monitor.presence_monitor import PresenceMonitor


class TestPresenceMonitor(unittest.TestCase):
    @patch('src.monitor.presence_monitor.time')
    def test_absence_and_camera_notifications(self, mock_time):
        notifier = MagicMock()
        registry = MagicMock()
        registry.get_all.return_value = ['tok']
        mock_time.time.side_effect = [0, 31, 31, 32]

        monitor = PresenceMonitor(notifier, registry, absence_timeout=30)
        monitor.handle_detections([])
        monitor.check_camera('frame')
        monitor.handle_detections([])
        notifier.notify.assert_called_once()

        monitor.check_camera(None)
        self.assertEqual(notifier.notify.call_count, 2)


if __name__ == '__main__':
    unittest.main()
