import os
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
        with patch('src.camera.handler.time.monotonic', side_effect=[1.0, 1.1, 1.3, 1.6]):
            with patch.object(self.cam, '_sleep_interruptible') as sleep_mock:
                sleep_mock.return_value = None
                self.cam.control_ptz(0.5, 0.0)

                payload = self.cam._ptz_service.ContinuousMove.call_args[0][0]
                self.assertEqual(payload['ProfileToken'], 'Profile000')
                self.assertIn('Velocity', payload)
                self.assertNotIn('Timeout', payload)

                expected_duration = min(self.cam._ptz_move_duration, self.cam._ptz_timeout)

                self.cam.control_ptz(0.5, 0.0)

                self.assertEqual(self.cam._ptz_service.ContinuousMove.call_count, 2)
                self.assertEqual(self.cam._ptz_service.Stop.call_count, 2)
                self.cam._ptz_service.Stop.assert_called_with({'ProfileToken': 'Profile000'})

                calls = sleep_mock.call_args_list
                self.assertGreaterEqual(len(calls), 3)
                self.assertAlmostEqual(calls[0].args[0], expected_duration)
                throttle_wait = self.cam._ptz_command_interval - (1.3 - 1.1)
                self.assertAlmostEqual(calls[1].args[0], throttle_wait)
                self.assertAlmostEqual(calls[2].args[0], expected_duration)

    def test_control_ptz_deadband(self):
        with patch('src.camera.handler.time.monotonic', return_value=10.0):
            self.cam.control_ptz(0.01, 0.01)
        self.cam._ptz_service.ContinuousMove.assert_not_called()

    def test_ptz_timing_from_env(self):
        with patch.dict(os.environ, {
            'PTZ_COMMAND_INTERVAL_SECS': '1.2',
            'PTZ_MOVE_DURATION_SECS': '2.5',
        }):
            cam = CameraHandler('host', 80, 'user', 'pass')
        self.assertAlmostEqual(cam._ptz_command_interval, 1.2)
        self.assertAlmostEqual(cam._ptz_move_duration, 2.5)


if __name__ == '__main__':
    unittest.main()
