import unittest

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


if __name__ == '__main__':
    unittest.main()
