import unittest
from unittest.mock import patch, MagicMock

from src.notifications.identified_notifier import IdentifiedNotifier


class TestIdentifiedNotifier(unittest.TestCase):
    @patch('src.notifications.identified_notifier.time')
    @patch('src.notifications.identified_notifier.Notifier')
    def test_notify_respects_threshold_and_cooldown(self, mock_notifier_cls, mock_time):
        mock_notifier = mock_notifier_cls.return_value
        mock_time.time.side_effect = [100, 130, 200]

        notifier = IdentifiedNotifier('key', threshold=0.5, cooldown=60)

        notifier.notify_if_identified(0.4, 'tkn')
        self.assertFalse(mock_notifier.send.called)

        notifier.notify_if_identified(0.6, 'tkn')
        mock_notifier.send.assert_called_once()

        notifier.notify_if_identified(0.7, 'tkn')
        mock_notifier.send.assert_called_once()

        notifier.notify_if_identified(0.8, 'tkn')
        self.assertEqual(mock_notifier.send.call_count, 2)


if __name__ == '__main__':
    unittest.main()
