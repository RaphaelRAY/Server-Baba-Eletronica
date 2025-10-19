import sys
import types
import unittest
from unittest.mock import MagicMock, patch

sys.modules.setdefault("cv2", MagicMock())
sys.modules.setdefault("onvif", MagicMock())
sys.modules.setdefault("ultralytics", MagicMock())

from src.camera.handler import CameraHandler
from src.camera.ptz_patrol import PTZPresetPatrol


class _DummyPTZService:
    def __init__(self):
        self.goto_payloads: list[dict] = []

    def GotoPreset(self, payload):
        self.goto_payloads.append(payload)


class _DummyCamera:
    def __init__(self, ptz_service=None, media_profiles=None):
        self._ptz_service = ptz_service or _DummyPTZService()
        self._media_profiles = media_profiles or []

    def create_ptz_service(self):
        if isinstance(self._ptz_service, Exception):
            raise self._ptz_service
        return self._ptz_service

    def create_media_service(self):
        service = types.SimpleNamespace()
        service.GetProfiles = lambda: self._media_profiles
        return service


class PTZPatrolTest(unittest.TestCase):
    def setUp(self):
        self.handler = CameraHandler("host", 80, "user", "pass")
        self.handler.ptz_enabled = True

    def test_goto_preset_uses_onvif_service(self):
        ptz_service = _DummyPTZService()
        profile = types.SimpleNamespace(
            token="Profile000",
            PTZConfiguration=types.SimpleNamespace(token="PTZ000"),
        )
        self.handler._camera = _DummyCamera(ptz_service, [profile])

        moved = self.handler.goto_preset(42)

        self.assertTrue(moved)
        self.assertEqual(len(ptz_service.goto_payloads), 1)
        payload = ptz_service.goto_payloads[0]
        self.assertEqual(payload["ProfileToken"], "Profile000")
        self.assertEqual(payload["PresetToken"], "42")

    def test_patrol_step_moves_and_calls_callbacks(self):
        patrol = PTZPresetPatrol(
            self.handler,
            start_preset=1,
            count_preset=2,
            preset_timeout=4.0,
        )
        callback = MagicMock()
        patrol.add_callback(callback)
        with patch.object(self.handler, "goto_preset", return_value=True) as goto_mock:
            with patch.object(self.handler, "_sleep_interruptible") as sleep_mock:
                visited = patrol.step()

        self.assertTrue(visited)
        goto_mock.assert_called_once_with(1)
        callback.assert_called_once()
        sleep_calls = [call.args[0] for call in sleep_mock.call_args_list]
        self.assertEqual(sleep_calls, [2.0, 2.0])
        self.assertEqual(patrol.last_preset, 1)

    def test_patrol_skips_presets_and_wraps(self):
        patrol = PTZPresetPatrol(
            self.handler,
            start_preset=1,
            count_preset=3,
            preset_timeout=1.0,
            skip_presets=[1],
        )
        with patch.object(self.handler, "goto_preset", return_value=True) as goto_mock:
            with patch.object(self.handler, "_sleep_interruptible"):
                first = patrol.step()
                second = patrol.step()

        self.assertFalse(first)
        self.assertTrue(second)
        self.assertEqual(goto_mock.call_args_list[0].args[0], 2)

    def test_patrol_start_creates_thread(self):
        patrol = PTZPresetPatrol(
            self.handler,
            start_preset=5,
            count_preset=2,
            preset_timeout=1.0,
        )
        with patch("src.camera.ptz_patrol.Thread") as thread_cls:
            thread_instance = MagicMock()
            thread_instance.is_alive.return_value = False
            thread_cls.return_value = thread_instance
            patrol.start()

        thread_cls.assert_called_once()
        thread_instance.start.assert_called_once()
        self.assertFalse(patrol.is_running())

    def test_count_zero_avoids_start(self):
        patrol = PTZPresetPatrol(self.handler)
        with patch("src.camera.ptz_patrol.Thread") as thread_cls:
            patrol.start()
        thread_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
