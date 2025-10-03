import sys
import types
from unittest import TestCase
from unittest.mock import patch


if "cv2" not in sys.modules:
    dummy_cv2 = types.SimpleNamespace(
        VideoCapture=lambda *args, **kwargs: None,
        CAP_FFMPEG=0,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_BUFFERSIZE=5,
        CAP_PROP_OPEN_TIMEOUT_MSEC=6,
        CAP_PROP_READ_TIMEOUT_MSEC=7,
    )
    sys.modules["cv2"] = dummy_cv2

from src.camera.handler import CameraHandler


class _DummyPTZService:
    def __init__(self):
        self.moves: list[dict] = []
        self.stop_payloads: list[dict] = []

    def ContinuousMove(self, payload):  # noqa: N802 - mantém interface ONVIF
        self.moves.append(payload)

    def Stop(self, payload):  # noqa: N802 - mantém interface ONVIF
        self.stop_payloads.append(payload)


class _DummyCamera:
    def __init__(self, ptz_service=None, media_profiles=None):
        self._ptz_service = ptz_service
        self._media_profiles = media_profiles or []

    def create_ptz_service(self):
        if isinstance(self._ptz_service, Exception):
            raise self._ptz_service
        return self._ptz_service

    def create_media_service(self):
        service = types.SimpleNamespace()
        service.GetProfiles = lambda: self._media_profiles
        return service


class CameraHandlerPTZTest(TestCase):
    def setUp(self):
        self.handler = CameraHandler("host", 80, "user", "pass")
        self.handler.ptz_enabled = True

    @patch("src.camera.handler.time.sleep", autospec=True)
    def test_control_ptz_limits_velocity_and_timeout(self, mock_sleep):
        ptz_service = _DummyPTZService()
        profile = types.SimpleNamespace(
            token="Profile000",
            PTZConfiguration=types.SimpleNamespace(
                token="PTZ000",
                DefaultPTZTimeout="PT10S",
            ),
        )
        self.handler._camera = _DummyCamera(ptz_service, [profile])
        self.handler._ptz_service = ptz_service
        self.handler._ptz_profile_token = "Profile000"
        self.handler._ptz_configuration_token = "PTZ000"
        self.handler._ptz_pan_limit = 1.0
        self.handler._ptz_tilt_limit = 1.0
        self.handler._ptz_timeout = 1.0

        self.handler.control_ptz(err_x=5.0, err_y=-5.0)

        self.assertEqual(len(ptz_service.moves), 1)
        move = ptz_service.moves[0]
        self.assertEqual(move["ProfileToken"], "Profile000")
        self.assertNotIn("Timeout", move)
        self.assertAlmostEqual(move["Velocity"]["PanTilt"]["x"], 1.0)
        self.assertAlmostEqual(move["Velocity"]["PanTilt"]["y"], -1.0)
        self.assertTrue(ptz_service.stop_payloads)
        self.assertEqual(
            ptz_service.stop_payloads[-1]["ProfileToken"], "Profile000"
        )
        mock_sleep.assert_called()

    def test_control_ptz_without_tokens_does_nothing(self):
        ptz_service = _DummyPTZService()
        failing_camera = _DummyCamera(ptz_service)
        failing_camera.create_media_service = lambda: (_ for _ in ()).throw(  # type: ignore
            RuntimeError("no media")
        )
        self.handler._camera = failing_camera
        self.handler._ptz_service = ptz_service
        self.handler._ptz_profile_token = None

        self.handler.control_ptz(err_x=1.0, err_y=1.0)

        self.assertEqual(ptz_service.moves, [])
        self.assertEqual(ptz_service.stop_payloads, [])
