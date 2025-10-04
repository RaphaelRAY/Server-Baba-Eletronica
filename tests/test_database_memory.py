import os
import unittest
from unittest.mock import patch

from src.db.database import Database, memory_events


class TestDatabaseMemoryGlobal(unittest.TestCase):
    def setUp(self):
        memory_events.clear()

    def test_shared_events_across_instances(self):
        db1 = Database(server=Database.SERVER_MEMORY)
        db2 = Database(server=Database.SERVER_MEMORY)
        db1.save_event({'type': 'share', 'confidence': 0.1})
        self.assertEqual(len(db2.get_recent_events()), 1)
        db2.save_event({'type': 'other', 'confidence': 0.2})
        self.assertEqual(len(db1.get_recent_events()), 2)

    def test_suppressed_events_not_logged_by_default(self):
        with patch.dict(os.environ, {'EVENT_COOLDOWN_SECS': '60', 'EVENT_LOG_SUPPRESSED': ''}, clear=False):
            db = Database(server=Database.SERVER_MEMORY)
            db.save_event({'type': 'absence', 'confidence': 0.4})

            with patch('src.db.database.logger.info') as mock_log:
                db.save_event({'type': 'absence', 'confidence': 0.5})

        mock_log.assert_not_called()

    def test_suppressed_events_logged_when_enabled(self):
        with patch.dict(os.environ, {'EVENT_COOLDOWN_SECS': '60', 'EVENT_LOG_SUPPRESSED': 'true'}, clear=False):
            db = Database(server=Database.SERVER_MEMORY)
            db.save_event({'type': 'absence', 'confidence': 0.4})

            with patch('src.db.database.logger.info') as mock_log:
                db.save_event({'type': 'absence', 'confidence': 0.5})

        mock_log.assert_called_once()


if __name__ == '__main__':
    unittest.main()
