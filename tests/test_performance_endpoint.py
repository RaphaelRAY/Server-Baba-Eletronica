import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import importlib

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


class TestPerformanceEndpoint(unittest.TestCase):
    def test_performance_returns_metrics(self):
        fps_stats = {
            "current": 12.5,
            "mean": 12.0,
            "min": 11.0,
            "max": 13.0,
            "samples": 4,
        }
        latency_stats = {"mean": 0.05, "min": 0.04, "max": 0.06, "count": 3}
        timing_stats = {
            "count": 2,
            "last": {
                "label": "prone",
                "confidence": 0.9,
                "process_s": 0.05,
                "alert_s": 0.01,
                "total_s": 0.06,
            },
            "averages": {"process_s": 0.05, "alert_s": 0.01, "total_s": 0.06},
        }

        with patch.object(main.camera, "get_fps_stats", return_value=fps_stats), \
             patch.object(main.camera, "get_latency_stats", return_value=latency_stats), \
             patch.object(main.camera, "get_last_latency", return_value=0.055), \
             patch.object(main.position_monitor, "get_timing_stats", return_value=timing_stats), \
             patch.object(main.camera, "start"), \
             patch.object(main.camera, "stop"), \
             patch("src.main.Thread") as mock_thread:
            mock_thread.return_value.start.return_value = None
            mock_thread.return_value.join.return_value = None

            client = TestClient(main.app)
            resp = client.get("/api/performance")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertAlmostEqual(data["fps"]["current"], 12.5)
            self.assertEqual(data["fps"]["samples"], 4)
            self.assertEqual(data["latency_ms"]["samples"], 3)
            self.assertAlmostEqual(data["latency_ms"]["last_ms"], 55.0)
            self.assertEqual(data["detection_to_alert_ms"]["count"], 2)
            self.assertAlmostEqual(data["detection_to_alert_ms"]["last"]["total_ms"], 60.0)


if __name__ == "__main__":
    unittest.main()
