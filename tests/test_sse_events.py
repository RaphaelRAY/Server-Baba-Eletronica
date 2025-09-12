import sys
import os
import unittest
import importlib
from unittest.mock import patch, MagicMock


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


class TestSSEEvents(unittest.TestCase):
    def test_sse_receives_event_on_save(self):
        with patch.object(main.camera, 'start'), \
             patch.object(main.camera, 'stop'), \
             patch('src.main.Thread') as mock_thread:
            mock_thread.return_value.start.return_value = None
            mock_thread.return_value.join.return_value = None
            with TestClient(main.app) as client:
                with client.stream('GET', '/api/events/sse') as resp:
                    self.assertEqual(resp.status_code, 200)
                    # Trigger a new event save which should broadcast to SSE
                    main.database.save_event({'type': 'test_event', 'confidence': 0.9, 'level': 'Info'})

                    # Consume a few lines until we find a data: payload
                    found = False
                    for line in resp.iter_lines():
                        if line.startswith('data: '):
                            self.assertIn('test_event', line)
                            # Ensure heavy fields were stripped
                            self.assertNotIn('image_b64', line)
                            found = True
                            break
                    self.assertTrue(found, 'Did not receive SSE data line')


if __name__ == '__main__':
    unittest.main()
