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


class TestStreamEndpoint(unittest.TestCase):
    def test_stream_returns_503_when_no_frame(self):
        with patch.object(main.camera, 'start'), \
             patch.object(main.camera, 'stop'), \
             patch('src.main.Thread') as mock_thread, \
             patch.object(main.camera, 'get_frame', return_value=None):
            mock_thread.return_value.start.return_value = None
            mock_thread.return_value.join.return_value = None
            client = TestClient(main.app)
            resp = client.get('/api/stream')
            self.assertEqual(resp.status_code, 503)
            self.assertIn('Sem frame disponível', resp.text)


if __name__ == '__main__':
    unittest.main()
