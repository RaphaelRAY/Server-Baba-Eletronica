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


class TestPoseSnapshotEndpoint(unittest.TestCase):
    def test_pose_snapshot_success(self):
        with patch.object(main.camera, 'start'), \
             patch.object(main.camera, 'stop'), \
             patch('src.main.Thread') as mock_thread, \
             patch.object(main.camera, 'get_frame') as mock_get_frame, \
             patch.object(main.position_monitor, 'model') as mock_model, \
             patch('src.main.encode_jpeg') as mock_encode:
            mock_thread.return_value.start.return_value = None
            mock_thread.return_value.join.return_value = None

            # Fake frame available
            mock_get_frame.return_value = 'frame'

            # Fake YOLO pose result
            res = MagicMock()
            res.plot.return_value = 'overlay'
            mock_model.return_value = [res]

            # Fake JPEG encoder
            mock_encode.return_value = b'JPEGDATA'

            client = TestClient(main.app)
            resp = client.get('/api/pose-snapshot')

            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.headers.get('content-type'), 'image/jpeg')
            self.assertEqual(resp.content, b'JPEGDATA')


if __name__ == '__main__':
    unittest.main()

