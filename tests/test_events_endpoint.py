import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import importlib

# provide fake external modules
sys.modules.setdefault('cv2', MagicMock())
sys.modules.setdefault('onvif', MagicMock())
sys.modules.setdefault('ultralytics', MagicMock())
firebase_admin_mock = MagicMock()
firebase_admin_mock.credentials = MagicMock()
sys.modules.setdefault('firebase_admin', firebase_admin_mock)
os.environ.setdefault('FIREBASE_CRED', 'path/key.json')

if importlib.util.find_spec('httpx') is None:
    raise unittest.SkipTest('httpx not installed')

from fastapi.testclient import TestClient

import src.main as main


class TestEventsEndpoint(unittest.TestCase):
    def test_get_events_default(self):
        events = [{"type": "a"}, {"type": "b"}]
        with patch.object(main.camera, 'start'), \
             patch.object(main.camera, 'stop'), \
             patch('src.main.Thread') as mock_thread, \
             patch.object(main, 'database') as mock_db:
            mock_thread.return_value.start.return_value = None
            mock_thread.return_value.join.return_value = None
            mock_db.get_recent_events.return_value = events
            client = TestClient(main.app)
            resp = client.get('/api/events')
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json(), events)
            mock_db.get_recent_events.assert_called_once_with(offset=0, limit=30)

    def test_get_events_paginated(self):
        events = [{"type": "a"}]
        with patch.object(main.camera, 'start'), \
             patch.object(main.camera, 'stop'), \
             patch('src.main.Thread') as mock_thread, \
             patch.object(main, 'database') as mock_db:
            mock_thread.return_value.start.return_value = None
            mock_thread.return_value.join.return_value = None
            mock_db.get_recent_events.return_value = events
            client = TestClient(main.app)
            resp = client.get('/api/events?offset=30&limit=30')
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json(), events)
            mock_db.get_recent_events.assert_called_once_with(offset=30, limit=30)


if __name__ == '__main__':
    unittest.main()
