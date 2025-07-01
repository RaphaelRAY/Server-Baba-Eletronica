import sys
import unittest
from unittest.mock import patch, MagicMock
import importlib

# Provide fake external modules if not installed
sys.modules.setdefault('cv2', MagicMock())
sys.modules.setdefault('onvif', MagicMock())

from src.camera.handler import CameraHandler
from src.processing.video_processor import VideoProcessor

class TestCameraHandler(unittest.TestCase):
    @patch('src.camera.handler.cv2.VideoCapture')
    @patch('src.camera.handler.ONVIFCamera')
    def test_start_and_get_frame(self, mock_onvif, mock_videocap):
        media_service = MagicMock()
        media_service.GetProfiles.return_value = [MagicMock(token='token')]
        media_service.GetStreamUri.return_value = MagicMock(Uri='rtsp://cam/stream')
        mock_onvif.return_value.create_media_service.return_value = media_service

        fake_cap = MagicMock()
        fake_cap.read.side_effect = [(True, 'frame'), (False, None)]
        mock_videocap.return_value = fake_cap

        cam = CameraHandler('host', 80, 'user', 'pass')
        cam.start()
        cam._thread.join(timeout=0.1)
        frame = cam.get_frame()
        self.assertEqual(frame, 'frame')
        cam.stop()

class TestVideoProcessor(unittest.TestCase):
    def test_get_processed_frame(self):
        camera = MagicMock()
        camera.get_frame.return_value = 'frame'
        proc = VideoProcessor(camera)
        self.assertEqual(proc.get_processed_frame(), 'frame')

class TestDatabase(unittest.TestCase):
    def test_save_and_get_event(self):
        if importlib.util.find_spec('sqlalchemy') is None:
            self.skipTest('sqlalchemy not installed')
        from src.db.database import Database
        db = Database('sqlite:///:memory:')
        db.save_event({'type': 'test', 'confidence': 0.9})
        events = db.get_recent_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['type'], 'test')
        self.assertAlmostEqual(events[0]['confidence'], 0.9)

if __name__ == '__main__':
    unittest.main()
