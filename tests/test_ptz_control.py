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
        self.addCleanup(self.cam.stop)

    def test_control_ptz_throttles_commands(self):
        scheduled: list[tuple[float, dict, float]] = []

        def fake_enqueue(execute_at, payload, duration):
            scheduled.append((execute_at, payload, duration))

        with patch('src.camera.handler.time.monotonic', side_effect=[1.0, 1.1]):
            with patch.object(self.cam, '_enqueue_ptz_command', side_effect=fake_enqueue):
                self.cam.control_ptz(0.5, 0.0)
                self.cam.control_ptz(0.5, 0.0)

        self.assertEqual(len(scheduled), 2)

        first_execute_at, first_payload, first_duration = scheduled[0]
        self.assertEqual(first_payload['ProfileToken'], 'Profile000')
        self.assertIn('Velocity', first_payload)
        self.assertAlmostEqual(
            first_duration, min(self.cam._ptz_move_duration, self.cam._ptz_timeout)
        )

        second_execute_at, second_payload, second_duration = scheduled[1]
        self.assertEqual(second_payload['Velocity'], first_payload['Velocity'])
        self.assertAlmostEqual(second_duration, first_duration)

        expected_second = first_execute_at + self.cam._ptz_command_interval
        self.assertAlmostEqual(second_execute_at, expected_second)
        self.assertAlmostEqual(self.cam._ptz_last_command_ts, expected_second)

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
