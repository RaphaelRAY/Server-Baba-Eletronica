import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import importlib
import base64
import shutil
from pathlib import Path

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
    def test_root_status(self):
        with patch.object(main.camera, 'start'), \
             patch.object(main.camera, 'stop'), \
             patch('src.main.Thread') as mock_thread:
            mock_thread.return_value.start.return_value = None
            mock_thread.return_value.join.return_value = None
            client = TestClient(main.app)
            resp = client.get('/')
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json(), {'status': 'connected'})

    def test_get_events_all(self):
        events = [{"type": "x"}, {"type": "y"}, {"type": "z"}]
        with patch.object(main.camera, 'start'), \
             patch.object(main.camera, 'stop'), \
             patch('src.main.Thread') as mock_thread, \
             patch.object(main, 'database') as mock_db:
            mock_thread.return_value.start.return_value = None
            mock_thread.return_value.join.return_value = None
            mock_db.get_all_events.return_value = events
            client = TestClient(main.app)
            resp = client.get('/api/events')
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json(), events)
            mock_db.get_all_events.assert_called_once_with()

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
            resp = client.get('/api/events/0')
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
            resp = client.get('/api/events/30?limit=30')
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json(), events)
            mock_db.get_recent_events.assert_called_once_with(offset=30, limit=30)

    def test_get_events_noimg_includes_path(self):
        event = {"type": "a", "confidence": 0.9, "image_path": "data/events/img.jpg", "image_b64": "zzz"}
        with patch.object(main.camera, 'start'), \
             patch.object(main.camera, 'stop'), \
             patch('src.main.Thread') as mock_thread, \
             patch.object(main, 'database') as mock_db:
            mock_thread.return_value.start.return_value = None
            mock_thread.return_value.join.return_value = None
            mock_db.get_recent_events.return_value = [event]
            client = TestClient(main.app)
            resp = client.get('/api/events/0/noimg')
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json(), [{"type": "a", "confidence": 0.9, "image_path": "data/events/img.jpg"}])
            mock_db.get_recent_events.assert_called_once_with(offset=0, limit=30)

    def test_get_all_events_noimg_includes_path(self):
        event = {"type": "b", "confidence": 0.3, "image_path": "data/events/other.jpg", "image_b64": "zzz"}
        with patch.object(main.camera, 'start'), \
             patch.object(main.camera, 'stop'), \
             patch('src.main.Thread') as mock_thread, \
             patch.object(main, 'database') as mock_db:
            mock_thread.return_value.start.return_value = None
            mock_thread.return_value.join.return_value = None
            mock_db.get_all_events.return_value = [event]
            client = TestClient(main.app)
            resp = client.get('/api/events/noimg')
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json(), [{"type": "b", "confidence": 0.3, "image_path": "data/events/other.jpg"}])
            mock_db.get_all_events.assert_called_once_with()

    def test_get_event_image_returns_base64(self):
        tmp_dir = Path('tests/_tmp_events_api')
        tmp_dir.mkdir(parents=True, exist_ok=True)
        img_path = tmp_dir / 'event.jpg'
        img_bytes = b'fake-bytes'
        img_path.write_bytes(img_bytes)
        try:
            with patch.object(main.camera, 'start'), \
                 patch.object(main.camera, 'stop'), \
                 patch('src.main.Thread') as mock_thread, \
                 patch.object(main, 'database'):
                mock_thread.return_value.start.return_value = None
                mock_thread.return_value.join.return_value = None
                client = TestClient(main.app)
                resp = client.get('/api/events/image', params={'image_path': str(img_path)})
                expected_b64 = base64.b64encode(img_bytes).decode('ascii')
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.json(), {'image_b64': expected_b64})
                rel_path = img_path.relative_to(Path.cwd())
                resp_alias = client.get('/api/events/image', params={'imagem_path': str(rel_path)})
                self.assertEqual(resp_alias.status_code, 200)
                self.assertEqual(resp_alias.json(), {'image_b64': expected_b64})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
