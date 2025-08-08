import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.modules.setdefault("firebase_admin", MagicMock())

from src.firebase_setup import FirebaseSetup


class TestFirebaseSetup(unittest.TestCase):
    @patch("src.firebase_setup.firebase_admin")
    @patch("src.firebase_setup.credentials")
    def test_initialize_called_env(self, mock_credentials, mock_firebase):
        mock_firebase._apps = []
        mock_firebase.get_app.side_effect = ValueError()
        with patch.dict(os.environ, {"FIREBASE_CRED": "path/key.json"}):
            FirebaseSetup().init_firebase()
            mock_credentials.Certificate.assert_called_once_with("path/key.json")
            mock_firebase.initialize_app.assert_called_once()

    @patch("src.firebase_setup.firebase_admin")
    @patch("src.firebase_setup.credentials")
    @patch("src.firebase_setup.FirebaseSetup._guess_default_path")
    def test_initialize_called_default(
        self, mock_guess, mock_credentials, mock_firebase
    ):
        mock_firebase._apps = []
        mock_firebase.get_app.side_effect = ValueError()
        default_path = os.path.join(os.path.expanduser("~"), "serviceAccountKey.json")
        mock_guess.return_value = default_path
        with patch.dict(os.environ, {}, clear=True):
            FirebaseSetup().init_firebase()
            mock_credentials.Certificate.assert_called_once_with(default_path)
            mock_firebase.initialize_app.assert_called_once()


if __name__ == "__main__":
    unittest.main()
