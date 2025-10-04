import os
import unittest
from unittest.mock import patch

from src.notifications.notifier import Notifier


class TestNotifier(unittest.TestCase):
    @patch('src.notifications.notifier.messaging')
    def test_send_uses_firebase_admin(self, mock_messaging):
        notifier = Notifier()
        mock_messaging.send.return_value = 'mock-id'
        notifier.send('tok', 'A', 'B')
        mock_messaging.Notification.assert_called_once_with(title='A', body='B')
        mock_messaging.Message.assert_called_once_with(
            token='tok',
            notification=mock_messaging.Notification.return_value
        )
        mock_messaging.send.assert_called_once_with(
            mock_messaging.Message.return_value
        )

    @patch('src.notifications.notifier.messaging')
    def test_send_logs_detailed_response(self, mock_messaging):
        notifier = Notifier(log_details=True)
        mock_messaging.send.return_value = 'msg-id-123'

        with self.assertLogs(level='INFO') as log_ctx:
            notifier.send('tok', 'Title', 'Body')

        self.assertTrue(
            any('Sending FCM message to token tok' in record for record in log_ctx.output)
        )
        self.assertTrue(
            any('response id: msg-id-123' in record for record in log_ctx.output)
        )

    @patch('src.notifications.notifier.messaging')
    def test_log_details_flag_from_env(self, mock_messaging):
        mock_messaging.send.return_value = 'env-msg-id'

        with patch.dict(os.environ, {'FCM_LOG_DETAILS': 'true'}):
            notifier = Notifier()
            with patch('src.notifications.notifier.logging.info') as mock_log_info:
                notifier.send('tok', 'Env', 'Body')

        self.assertGreaterEqual(mock_log_info.call_count, 2)
        self.assertTrue(
            any(
                call.args == (
                    'FCM message delivered to token %s; response id: %s',
                    'tok',
                    'env-msg-id'
                )
                for call in mock_log_info.call_args_list
            )
        )

        with patch.dict(os.environ, {'FCM_LOG_DETAILS': 'false'}):
            notifier = Notifier()
            with patch('src.notifications.notifier.logging.info') as mock_log_info_disabled:
                notifier.send('tok', 'Env', 'Body')

        mock_log_info_disabled.assert_not_called()


if __name__ == '__main__':
    unittest.main()
