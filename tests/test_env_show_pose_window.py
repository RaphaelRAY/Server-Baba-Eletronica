import os
import sys
import unittest
import importlib
from unittest.mock import MagicMock


def _prepare_mocks():
    # Provide fake external modules to avoid import-time issues
    sys.modules.setdefault('cv2', MagicMock())
    sys.modules.setdefault('onvif', MagicMock())
    sys.modules.setdefault('ultralytics', MagicMock())
    firebase_admin_mock = MagicMock()
    firebase_admin_mock.credentials = MagicMock()
    sys.modules.setdefault('firebase_admin', firebase_admin_mock)
    os.environ.setdefault('FIREBASE_CRED', 'path/key.json')


class TestEnvShowPoseWindow(unittest.TestCase):
    def setUp(self):
        # Ensure a clean import of src.main each test
        sys.modules.pop('src.main', None)
        _prepare_mocks()
        # Clean env var unless a test sets it
        os.environ.pop('SHOW_POSE_WINDOW', None)

    def test_default_true_when_unset(self):
        import src.main as main
        self.assertTrue(main.SHOW_POSE_WINDOW)

    def test_false_when_env_set_false(self):
        os.environ['SHOW_POSE_WINDOW'] = 'false'
        import src.main as main
        self.assertFalse(main.SHOW_POSE_WINDOW)


if __name__ == '__main__':
    unittest.main()

