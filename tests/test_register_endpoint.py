import sys
import unittest
from unittest.mock import patch, MagicMock
import importlib

# provide fake external modules
sys.modules.setdefault('cv2', MagicMock())
sys.modules.setdefault('onvif', MagicMock())
sys.modules.setdefault('ultralytics', MagicMock())
sys.modules.setdefault('pyfcm', MagicMock())

if importlib.util.find_spec('httpx') is None:
    raise unittest.SkipTest('httpx not installed')

from fastapi.testclient import TestClient

import src.main as main


class TestRegisterEndpoint(unittest.TestCase):
    def test_register_token(self):
        with patch.object(main, 'token_registry') as mock_reg, \
             patch.object(main.camera, 'start'), \
             patch.object(main.camera, 'stop'), \
             patch('src.main.Thread') as mock_thread:
            mock_thread.return_value.start.return_value = None
            mock_thread.return_value.join.return_value = None
            client = TestClient(main.app)
            resp = client.post('/api/register-token', json={'token': 'abc'})
            self.assertEqual(resp.status_code, 200)
            mock_reg.add.assert_called_once_with('abc')


if __name__ == '__main__':
    unittest.main()
