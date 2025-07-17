import unittest
from unittest.mock import patch

from src.notifications.status_notifier import StatusNotifier


class TestStatusNotifier(unittest.TestCase):
    @patch('src.notifications.status_notifier.time')
    @patch('src.notifications.status_notifier.Notifier')
    def test_no_person_cooldown(self, mock_notifier_cls, mock_time):
        mock_notifier = mock_notifier_cls.return_value
        mock_time.time.side_effect = [0, 10, 70]
        n = StatusNotifier('key', cooldown=60)
        n.notify_no_person('t')
        n.notify_no_person('t')
        n.notify_no_person('t')
        self.assertEqual(mock_notifier.send.call_count, 2)

    @patch('src.notifications.status_notifier.time')
    @patch('src.notifications.status_notifier.Notifier')
    def test_disconnect_cooldown(self, mock_notifier_cls, mock_time):
        mock_notifier = mock_notifier_cls.return_value
        mock_time.time.side_effect = [0, 30, 100]
        n = StatusNotifier('key', cooldown=60)
        n.notify_disconnect('t')
        n.notify_disconnect('t')
        n.notify_disconnect('t')
        self.assertEqual(mock_notifier.send.call_count, 2)


if __name__ == '__main__':
    unittest.main()
