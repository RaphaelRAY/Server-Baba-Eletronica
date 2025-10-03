import sys
import unittest
from unittest.mock import MagicMock, patch

sys.modules.setdefault('cv2', MagicMock())
sys.modules.setdefault('onvif', MagicMock())
sys.modules.setdefault('ultralytics', MagicMock())

from src.camera.handler import CameraHandler


class TestPTZControl(unittest.TestCase):
    def setUp(self):
        self.cam = CameraHandler('host', 80, 'user', 'pass')
        self.cam._camera = object()
        self.cam._ptz_service = MagicMock()
        self.cam._ptz_profile_token = 'Profile000'
        self.cam._ptz_pan_limit = 1.0
        self.cam._ptz_tilt_limit = 1.0
        self.cam._ptz_timeout = 1.0

    def test_control_ptz_throttles_commands(self):
        with patch('src.camera.handler.time.monotonic', side_effect=[1.0, 1.5, 1.6]) as _:
            with patch.object(self.cam, '_sleep_interruptible') as sleep_mock:
                sleep_mock.return_value = None
                self.cam.control_ptz(0.5, 0.0)

                self.cam._ptz_service.ContinuousMove.assert_called_once()
                payload = self.cam._ptz_service.ContinuousMove.call_args[0][0]
                self.assertEqual(payload['ProfileToken'], 'Profile000')
                self.assertIn('Velocity', payload)
                self.assertNotIn('Timeout', payload)

                sleep_mock.assert_called_once()
                move_duration = sleep_mock.call_args[0][0]
                expected_duration = min(self.cam._ptz_move_duration, self.cam._ptz_timeout)
                self.assertAlmostEqual(move_duration, expected_duration)

                self.cam._ptz_service.Stop.assert_called_once_with({'ProfileToken': 'Profile000'})

                self.cam.control_ptz(0.5, 0.0)
                self.assertEqual(self.cam._ptz_service.ContinuousMove.call_count, 1)

    def test_control_ptz_deadband(self):
        with patch('src.camera.handler.time.monotonic', return_value=10.0):
            self.cam.control_ptz(0.01, 0.01)
        self.cam._ptz_service.ContinuousMove.assert_not_called()


if __name__ == '__main__':
    unittest.main()
