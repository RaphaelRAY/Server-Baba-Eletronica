import sys
import os
import unittest
import importlib
from unittest.mock import patch, MagicMock

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


class TestDebugNotifyEndpoint(unittest.TestCase):
    def _thread_stub(self, mock_thread):
        mock_thread.return_value.start.return_value = None
        mock_thread.return_value.join.return_value = None

    def test_debug_page_served(self):
        with patch.object(main.camera, 'start'), \
             patch.object(main.camera, 'stop'), \
             patch('src.main.Thread') as mock_thread:
            self._thread_stub(mock_thread)
            client = TestClient(main.app)
            resp = client.get('/debug/notify')
            self.assertEqual(resp.status_code, 200)
            self.assertIn("Debug de Notifica", resp.text)

    def test_send_notification_single_token(self):
        with patch.object(main.camera, 'start'), \
             patch.object(main.camera, 'stop'), \
             patch('src.main.Thread') as mock_thread, \
             patch.object(main.notifier, 'send_immediate') as mock_send:
            self._thread_stub(mock_thread)
            client = TestClient(main.app)
            resp = client.post(
                '/api/debug/notify',
                json={"token": "abc", "title": "T", "message": "M", "level": "Urgente"},
            )
            self.assertEqual(resp.status_code, 200)
            mock_send.assert_called_once_with("abc", title="T", message="M", level="Urgente")
            self.assertEqual(resp.json()["sent"], 1)

    def test_send_notification_all_tokens(self):
        with patch.object(main.camera, 'start'), \
             patch.object(main.camera, 'stop'), \
             patch('src.main.Thread') as mock_thread, \
             patch.object(main, 'token_registry') as mock_registry, \
             patch.object(main.notifier, 'send_immediate') as mock_send:
            self._thread_stub(mock_thread)
            mock_registry.get_all.return_value = ["t1", "t2"]
            client = TestClient(main.app)
            resp = client.post('/api/debug/notify', json={"send_all": True})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["sent"], 2)
            mock_send.assert_any_call("t1", title="[Debug] Ping", message="Teste de debug", level="Info")
            mock_send.assert_any_call("t2", title="[Debug] Ping", message="Teste de debug", level="Info")

    def test_send_notification_missing_tokens_returns_400(self):
        with patch.object(main.camera, 'start'), \
             patch.object(main.camera, 'stop'), \
             patch('src.main.Thread') as mock_thread, \
             patch.object(main, 'token_registry') as mock_registry:
            self._thread_stub(mock_thread)
            mock_registry.get_all.return_value = []
            client = TestClient(main.app)
            resp = client.post('/api/debug/notify', json={"send_all": True})
            self.assertEqual(resp.status_code, 400)
            self.assertIn("Nenhum token", resp.json()["detail"])


if __name__ == '__main__':
    unittest.main()
