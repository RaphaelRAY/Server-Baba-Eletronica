import unittest
from unittest.mock import patch

from src.notifications.notifier import Notifier


class TestNotifier(unittest.TestCase):
    @patch('src.notifications.notifier.messaging')
    def test_send_uses_firebase_admin(self, mock_messaging):
        notifier = Notifier()
        notifier.send('tok', 'A', 'B')
        mock_messaging.Notification.assert_called_once_with(title='A', body='B')
        mock_messaging.Message.assert_called_once_with(
            token='tok',
            notification=mock_messaging.Notification.return_value
        )
        mock_messaging.send.assert_called_once_with(
            mock_messaging.Message.return_value
        )


if __name__ == '__main__':
    unittest.main()
