import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.modules.setdefault('firebase_admin', MagicMock())

from src.firebase_setup import init_firebase


class TestFirebaseSetup(unittest.TestCase):
    @patch('src.firebase_setup.firebase_admin')
    @patch('src.firebase_setup.credentials')
    @patch('src.firebase_setup.os.path.exists', return_value=True)
    def test_initialize_called_env(self, mock_exists, mock_credentials, mock_firebase):
        mock_firebase._apps = []
        with patch.dict(os.environ, {'FIREBASE_CRED': 'path/key.json'}):
            init_firebase()
            mock_credentials.Certificate.assert_called_once_with('path/key.json')
            mock_firebase.initialize_app.assert_called_once()

    @patch('src.firebase_setup.firebase_admin')
    @patch('src.firebase_setup.credentials')
    @patch('src.firebase_setup.os.path')
    def test_initialize_called_default(self, mock_path, mock_credentials, mock_firebase):
        mock_firebase._apps = []
        default_path = os.path.join(os.path.expanduser('~'), 'serviceAccountKey.json')
        mock_path.exists.return_value = True
        mock_path.dirname.return_value = os.path.expanduser('~')
        mock_path.join.return_value = default_path
        with patch.dict(os.environ, {}, clear=True):
            init_firebase()
            mock_credentials.Certificate.assert_called_once_with(default_path)
            mock_firebase.initialize_app.assert_called_once()


if __name__ == '__main__':
    unittest.main()
